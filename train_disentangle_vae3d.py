import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path
import torch
import logging
from tqdm import tqdm

from torch.cuda.amp import GradScaler
from models import disentangle_vae3d
from engine_disentangle_vae3d import train_one_epoch, validate
from util.ukbb_dataset import UKBBLMDBDataset
from omegaconf import OmegaConf

import warnings
warnings.filterwarnings("ignore")

def get_args_parser():
    parser = argparse.ArgumentParser('MAR training with Diffusion Loss', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size per GPU')
    parser.add_argument('--epochs', default=10, type=int)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay')
    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR', help='learning rate')
    parser.add_argument('--grad_clip', type=float, default=3.0, help='Gradient clip')

    # Dataset parameters
    parser.add_argument('--root_dir', default='', help='ukbb lmdb root path')
    parser.add_argument('--output_dir', default='', help='path where to save')
    parser.add_argument('--log_dir', default='', help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda:1', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)

    # Config
    parser.add_argument('--cfg', default='config/disentangle_vae3d.yaml', help='config path')
    parser.add_argument('--disc_loss_scale', default=0.1, type=float, help='disc_loss_scale')
    parser.add_argument('--resume', default='', help='resume from checkpoint')

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

@torch.no_grad()
def extract_latents_and_template(model, data_loader, device, output_dir):
    """
    关键步骤：
    1. 提取所有数据的 Latent (Z_gt) 用于第二阶段训练 Flow Matching 的 Target。
    2. 计算 Latent 的均值 (Template) 用于第二阶段 Flow Matching 的 Source。
    """
    logger = logging.getLogger("vae")
    logger.info("Starting Latent Extraction and Template Calculation...")
    
    model.eval()
    all_latents = []
    
    # 确保保存目录存在
    save_dir = os.path.join(output_dir, "latents")
    os.makedirs(save_dir, exist_ok=True)
    
    for batch in tqdm(data_loader, desc="Extracting Latents"):
        # batch: (eids, ecg, cmr)
        _, _, samples = batch
        samples = samples.to(device, non_blocking=True)
        if samples.dim() == 4:
            samples = samples.unsqueeze(1).contiguous() # (B, 1, 50, 96, 96)
            
        inputs = model.get_input(samples)
        
        # 编码获取后验分布
        posterior = model.encode_3d(inputs)
        # 使用 mode() 获取均值作为真值 (减少随机性带来的噪声)，或者使用 sample()
        z = posterior.mode() # (B, 4, 50, 12, 12)
        
        # 转移到 CPU 节省显存
        all_latents.append(z.cpu())

    # 1. 保存完整 Latent 数据集 (Z_gt)
    # 维度: (N_total, 4, 50, 12, 12)
    full_latents = torch.cat(all_latents, dim=0)
    logger.info(f"Saving full latents with shape {full_latents.shape}...")
    torch.save(full_latents, os.path.join(save_dir, "all_latents.pt"))
    
    # 2. 计算并保存静态解剖模板 (P_template)
    # 对 Batch(0) 和 Time(2) 维度求平均
    # 维度变化: (N, 4, 50, 12, 12) -> (1, 4, 1, 12, 12)
    anatomy_template = torch.mean(full_latents, dim=[0, 2], keepdim=True)
    logger.info(f"Saving anatomy template with shape {anatomy_template.shape}...")
    torch.save(anatomy_template, os.path.join(save_dir, "anatomy_template.pt"))
    
    logger.info("Latent extraction and template generation finished.")

def main(args):
    # --- 日志设置 ---
    logger = logging.getLogger("vae")
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

    # --- 数据集 ---
    dataset_train = UKBBLMDBDataset(root_dir=args.root_dir, split='train')
    dataset_val = UKBBLMDBDataset(root_dir=args.root_dir, split='val')
    logger.info(f'Train size: {len(dataset_train)}, Val size: {len(dataset_val)}')

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, num_workers=16, 
        pin_memory=True, shuffle=True, collate_fn=UKBBLMDBDataset.collate_fn
    )
    # 提取 latent 用的 loader (不 shuffle)
    data_loader_extract = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size * 2, num_workers=16, 
        pin_memory=True, shuffle=False, collate_fn=UKBBLMDBDataset.collate_fn
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size, num_workers=8, 
        pin_memory=True, shuffle=False, collate_fn=UKBBLMDBDataset.collate_fn
    )

    # --- 模型构建 ---
    model_config = OmegaConf.load(args.cfg)
    model = disentangle_vae3d.__dict__['vae3d'](
        lossconfig=model_config.model.params.lossconfig,
        ddconfig=model_config.model.params.ddconfig,
        embed_dim=model_config.model.params.embed_dim,
    )
    model.to(device)

    # --- 优化器 ---
    # 分离 VAE 参数和判别器参数
    vae_params = list(model.encoder_3d.parameters()) + \
                 list(model.decoder.parameters()) + \
                 list(model.quant_conv.parameters()) + \
                 list(model.post_quant_conv.parameters())
    
    if model.loss.logvar.requires_grad:
        vae_params.append(model.loss.logvar)

    optimizer = torch.optim.AdamW(vae_params, lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)

    disc_params = []
    if model.loss.enable_3d:
        disc_params += list(model.loss.discriminator.parameters())
    optimizer_d = torch.optim.AdamW(disc_params, lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)

    loss_scaler = GradScaler()
    total_iters = args.epochs * len(data_loader_train)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iters, eta_min=1e-6)

    # --- Resume ---
    start_epoch = 1
    best_val = float("inf")
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            
            # 兼容仅保存了 state_dict 的旧版 checkpoint
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # 不加载学习率调度器状态，使用新的学习率
                loss_scaler.load_state_dict(checkpoint['scaler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_val = checkpoint.get('best_val', float("inf"))
                model.global_step = checkpoint.get('global_step', 0)
                
                # 重新设置优化器的学习率为当前配置的学习率
                # for param_group in optimizer.param_groups:
                #     param_group['lr'] = args.lr
                # for param_group in optimizer_d.param_groups:
                #     param_group['lr'] = args.lr
            else:
                # 尝试直接加载模型权重 (兼容旧版保存方式)
                model.load_state_dict(checkpoint)
                logger.warning("Loaded checkpoint contained only model weights. Optimizer states were reset.")
            
            logger.info(f"Resumed from epoch {start_epoch-1}")
        else:
            logger.warning(f"No checkpoint found at {args.resume}")

    # --- 训练循环 ---
    logger.info(f"Start training for {args.epochs} epochs")
    
    for epoch in range(start_epoch, args.epochs + 1):
        train_stats = train_one_epoch(
            model, data_loader_train, optimizer, optimizer_d, scheduler,
            device, epoch, loss_scaler, args
        )

        val_stats = validate(
            model, data_loader_val, device, epoch, args.output_dir
        )
        
        val_loss = val_stats["val_rec_loss"] # 通常主要关注重建损失
        logger.info(f"[Val Epoch {epoch}] Rec: {val_loss:.6f}, KL: {val_stats['val_kl_loss']:.6f}")

        # 保存权重 (保存完整状态以便 resume)
        save_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'optimizer_d_state_dict': optimizer_d.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': loss_scaler.state_dict(),
            'best_val': best_val,
            'global_step': model.global_step,
        }

        # 每个 epoch 都保存权重
        torch.save(save_dict, os.path.join(args.output_dir, f"checkpoint-{epoch}.pth"))

        if val_loss < best_val:
            best_val = val_loss
            # Best model 通常只需要权重，但为了统一也保存完整状态
            torch.save(save_dict, os.path.join(args.output_dir, "checkpoint-best.pth"))

    # --- 训练结束：提取 Latent 和 Template ---
    logger.info("Training finished. Loading best model for latent extraction...")
    # 加载 best 权重
    best_ckpt = torch.load(os.path.join(args.output_dir, "checkpoint-best.pth"))
    if 'model_state_dict' in best_ckpt:
        model.load_state_dict(best_ckpt['model_state_dict'])
    else:
        model.load_state_dict(best_ckpt)
        
    extract_latents_and_template(model, data_loader_extract, device, args.output_dir)

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)