import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from tqdm import tqdm

from torch.cuda.amp import GradScaler
from models.ecg_encoder import ECGEncoder, ECGDecoder, random_masking
from util.ukbb_dataset import UKBBLMDBDataset
import util.misc as misc
import sys
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

def get_args_parser():
    parser = argparse.ArgumentParser('ECG Pre-training with MAE', add_help=False)
    
    # Training parameters
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size per GPU')
    parser.add_argument('--epochs', default=100, type=int)
    
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='learning rate')
    parser.add_argument('--grad_clip', type=float, default=3.0, help='Gradient clip')
    
    # Model parameters
    parser.add_argument('--mask_ratio', type=float, default=0.5, help='Masking ratio (percentage of removed patches)')
    parser.add_argument('--w_phase', type=float, default=10.0, help='Weight for phase loss')
    parser.add_argument('--w_rec', type=float, default=100.0, help='Weight for reconstruction loss')
    
    # Dataset parameters
    parser.add_argument('--root_dir', default='', help='ukbb lmdb root path')
    parser.add_argument('--output_dir', default='', help='path where to save')
    parser.add_argument('--log_dir', default='', help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda:2', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    
    return parser

def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Trainer Wrapper ---
class MaskedECGTrainer(nn.Module):
    def __init__(self, encoder, decoder, mask_ratio=0.5):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mask_ratio = mask_ratio
        
    def forward(self, ecg_raw, gt_phase):
        # 1. Encoder Forward (获取长序列特征，不进行 ROI Align)
        # feats_long: (B, 768, 625), phase_pred: (B, 2, 625)
        feats_long, phase_pred = self.encoder(ecg_raw, force_indices="return_raw") 
        
        # 2. Phase Loss (监督节奏 - 强监督)
        loss_phase = F.mse_loss(phase_pred, gt_phase)
        
        # 3. Masking (仅针对 Semantic Feature)
        feats_masked, mask = random_masking(feats_long, self.mask_ratio)
        
        # 4. Decoder Forward (重建)
        ecg_rec = self.decoder(feats_masked) # (B, 12, 5000)
        
        # 5. Reconstruction Loss (监督形态 - 自监督)
        loss_rec = F.mse_loss(ecg_rec, ecg_raw)
        
        return loss_phase, loss_rec, ecg_rec, phase_pred

def detect_r_peaks_simple(ecg_signal_1d, fs=500, min_rr=0.25, std_scale=0.5):
    """
    纯 numpy 的简易 R 波检测：
    1) 阈值 = 均值 + std_scale * 标准差
    2) 找局部极大值且高于阈值
    3) 强制最小 RR 间隔去重
    """
    if len(ecg_signal_1d) < 3:
        return np.array([], dtype=int)
    
    thresh = ecg_signal_1d.mean() + std_scale * ecg_signal_1d.std()
    # 局部极大值候选
    candidates = np.where(
        (ecg_signal_1d[1:-1] > ecg_signal_1d[:-2]) &
        (ecg_signal_1d[1:-1] >= ecg_signal_1d[2:]) &
        (ecg_signal_1d[1:-1] > thresh)
    )[0] + 1  # 补回偏移
    
    min_distance = int(fs * min_rr)
    peaks = []
    for idx in candidates:
        if peaks and idx - peaks[-1] < min_distance:
            # 近距离冲突时保留幅值更大的峰
            if ecg_signal_1d[idx] > ecg_signal_1d[peaks[-1]]:
                peaks[-1] = idx
        else:
            peaks.append(idx)
    return np.array(peaks, dtype=int)


def process_single_ecg(ecg_signal_1d):
    # 1. 提取 R 波（简易实现，替代 neurokit2）
    try:
        r_indices = detect_r_peaks_simple(ecg_signal_1d, fs=500)
        
        # 2. 生成相位 (0 ~ 2pi)
        phase = np.zeros(len(ecg_signal_1d))
        if len(r_indices) > 1:
            for i in range(len(r_indices) - 1):
                start, end = r_indices[i], r_indices[i+1]
                phase[start:end] = np.linspace(0, 2*np.pi, end-start)
        
        # 3. 转换为 sin/cos 并下采样 (5000 -> 625)
        phase_down = phase[::8]
        sin_phase = np.sin(phase_down)
        cos_phase = np.cos(phase_down)
        
        return np.stack([sin_phase, cos_phase])
        
    except Exception:
        # 若检测失败，返回 2x625 全零
        return np.zeros((2, 625))

def train_one_epoch(model, data_loader, optimizer, scheduler, device, epoch, loss_scaler, args):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    
    logger = logging.getLogger("ecg_pretrain")
    
    # 训练状态累计
    running_stats = {"loss": 0.0, "rec": 0.0, "phase": 0.0, "count": 0}
    
    # 假定 Encoder 8x 下采样
    # 5000 / 8 = 625
    phase_len = 625 

    for data_iter_step, batch in enumerate(data_loader):
        # batch: (eids, ecg, cmr) -> 我们只需要 ecg
        _, ecg_raw, _ = batch
        
        # --- Online Phase Generation (CPU) ---
        # 使用 neurokit2 从真实 ECG (Lead II 通常比较好) 生成相位
        # ecg_raw: (B, 12, 5000)
        bs = ecg_raw.size(0)
        ecg_np = ecg_raw.numpy()
        
        gt_phase_list = []
        for i in range(bs):
            # 假设 Lead II 是第 2 个通道 (idx 1)，如果不可用可以用 Lead I (idx 0)
            # 这里简单取 idx 1
            signal_1d = ecg_np[i, 1] 
            phase_gt_np = process_single_ecg(signal_1d)
            gt_phase_list.append(phase_gt_np)
            
        gt_phase = torch.tensor(np.stack(gt_phase_list), dtype=torch.float32).to(device)
        
        ecg_raw = ecg_raw.to(device, non_blocking=True) # (B, 12, 5000)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            loss_phase, loss_rec, _, _ = model(ecg_raw, gt_phase)
            loss = args.w_phase * loss_phase + args.w_rec * loss_rec
            
        loss_value = loss.item()
        
        if not np.isfinite(loss_value):
            logger.error(f"Loss is {loss_value}, stopping training")
            sys.exit(1)
            
        loss_scaler.scale(loss).backward()
        if args.grad_clip is not None:
            loss_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        loss_scaler.step(optimizer)
        loss_scaler.update()
        
        # Log
        lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]["lr"]
        
        running_stats["loss"] += loss_value
        running_stats["rec"] += loss_rec.item()
        running_stats["phase"] += loss_phase.item()
        running_stats["count"] += 1
        
        metric_logger.update(loss=loss_value)
        metric_logger.update(rec_loss=loss_rec.item())
        metric_logger.update(phase_loss=loss_phase.item())
        metric_logger.update(lr=lr)
        
        if scheduler is not None:
             scheduler.step()

        if data_iter_step % print_freq == 0:
            avg_loss = running_stats["loss"] / running_stats["count"]
            avg_rec = running_stats["rec"] / running_stats["count"]
            avg_phase = running_stats["phase"] / running_stats["count"]
            
            logger.info(
                f"Epoch [{epoch}] [{data_iter_step}/{len(data_loader)}] "
                f"L_Total: {avg_loss:.4f} | Rec: {avg_rec:.4f} | Phase: {avg_phase:.4f} | LR: {lr:.2e}"
            )
            # Reset stats
            running_stats = {k: 0.0 for k in running_stats}
            running_stats["count"] = 0
            
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def validate(model, data_loader, device, epoch, args):
    model.eval()
    logger = logging.getLogger("ecg_pretrain")
    
    val_stats = {"loss": 0.0, "rec": 0.0, "phase": 0.0, "count": 0}
    phase_len = 625
    has_visualized = False
    
    for i, batch in enumerate(data_loader):
        _, ecg_raw, _ = batch
        
        # --- Online Phase Generation ---
        bs = ecg_raw.size(0)
        ecg_np = ecg_raw.numpy()
        gt_phase_list = []
        for b in range(bs):
            signal_1d = ecg_np[b, 1]
            phase_gt_np = process_single_ecg(signal_1d)
            gt_phase_list.append(phase_gt_np)
        gt_phase = torch.tensor(np.stack(gt_phase_list), dtype=torch.float32).to(device)
        
        ecg_raw = ecg_raw.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast():
            loss_phase, loss_rec, ecg_rec, _ = model(ecg_raw, gt_phase)
            loss = args.w_phase * loss_phase + args.w_rec * loss_rec
            
        val_stats["loss"] += loss.item() * bs
        val_stats["rec"] += loss_rec.item() * bs
        val_stats["phase"] += loss_phase.item() * bs
        val_stats["count"] += bs
        
        # 仅在每次验证的首批次做可视化，避免额外开销
        if not has_visualized:
            save_dir = os.path.join(args.output_dir, "val_viz", f"epoch{epoch:03d}")
            visualize_ecg_batch(ecg_raw, ecg_rec, save_dir, epoch, max_samples=8)
            has_visualized = True
        
    avg_loss = val_stats["loss"] / val_stats["count"]
    avg_rec = val_stats["rec"] / val_stats["count"]
    avg_phase = val_stats["phase"] / val_stats["count"]
    
    return {
        "val_loss": avg_loss,
        "val_rec_loss": avg_rec,
        "val_phase_loss": avg_phase
    }

def visualize_ecg_batch(gt_tensor, rec_tensor, save_dir, epoch, max_samples=8):
    """
    将重建结果与 GT 进行覆盖式对比可视化。
    gt_tensor / rec_tensor: (B, 12, 5000)
    """
    os.makedirs(save_dir, exist_ok=True)
    gt_np = gt_tensor.detach().cpu().numpy()
    rec_np = rec_tensor.detach().cpu().numpy()
    
    num_samples = min(gt_np.shape[0], max_samples)
    num_leads = gt_np.shape[1]
    
    for i in range(num_samples):
        fig, axes = plt.subplots(num_leads, 1, figsize=(12, 3*num_leads), sharex=True)
        if num_leads == 1:
            axes = [axes]
        for lead in range(num_leads):
            axes[lead].plot(gt_np[i, lead], color='red', linewidth=0.7, label='GT')
            axes[lead].plot(rec_np[i, lead], color='blue', linewidth=0.7, alpha=0.8, label='Fake')
            axes[lead].set_ylabel(f"Lead {lead+1}")
            if lead == 0:
                axes[lead].legend(loc='upper right', fontsize=8)
        axes[-1].set_xlabel("Time")
        fig.suptitle(f"Epoch {epoch} - Sample {i}")
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        save_path = os.path.join(save_dir, f"epoch{epoch:03d}_sample{i:02d}.png")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

def main(args):
    # --- Logging Setup ---
    logger = logging.getLogger("ecg_pretrain")
    if logger.handlers:
        logger.handlers.clear()
    logger.setLevel("INFO")
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.log_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(args.log_dir, f"{current_time}_train.log"), encoding="utf-8")
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Args: {args}")
    device = torch.device(args.device)
    seed_everything(args.seed)
    
    # --- Dataset ---
    dataset_train = UKBBLMDBDataset(root_dir=args.root_dir, split='train')
    dataset_val = UKBBLMDBDataset(root_dir=args.root_dir, split='val')
    logger.info(f'Train size: {len(dataset_train)}, Val size: {len(dataset_val)}')
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, num_workers=16, 
        pin_memory=True, shuffle=True, collate_fn=UKBBLMDBDataset.collate_fn
    )
    
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size, num_workers=8, 
        pin_memory=True, shuffle=False, collate_fn=UKBBLMDBDataset.collate_fn
    )
    
    # --- Model ---
    encoder = ECGEncoder(in_channels=12, out_dim=768, target_len=50) # Target len for downstream task, not used in pretrain phase/raw return
    decoder = ECGDecoder(in_dim=768, out_channels=12)
    model = MaskedECGTrainer(encoder, decoder, mask_ratio=args.mask_ratio)
    model.to(device)
    
    # --- Optimizer ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    loss_scaler = GradScaler()
    
    total_iters = args.epochs * len(data_loader_train)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iters, eta_min=1e-6)
    
    start_epoch = 1
    best_val = float("inf")
            
    # --- Training Loop ---
    logger.info(f"Start training for {args.epochs} epochs")
    
    for epoch in range(start_epoch, args.epochs + 1):
        train_stats = train_one_epoch(
            model, data_loader_train, optimizer, scheduler,
            device, epoch, loss_scaler, args
        )
        
        val_stats = validate(model, data_loader_val, device, epoch, args)
        
        val_loss = val_stats["val_rec_loss"] # Use reconstruction loss as metric
        logger.info(f"[Val Epoch {epoch}] Total: {val_stats['val_loss']:.6f}, Rec: {val_loss:.6f}, Phase: {val_stats['val_phase_loss']:.6f}")
        
        save_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': loss_scaler.state_dict(),
            'best_val': best_val,
        }
        
        # Save periodic
        torch.save(save_dict, os.path.join(args.output_dir, f"checkpoint-{epoch}.pth"))
        
        # Save Best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(save_dict, os.path.join(args.output_dir, "checkpoint-best.pth"))
            logger.info(f"New best model saved with loss {best_val:.6f}")
    
    logger.info("Pre-training Complete.")

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
