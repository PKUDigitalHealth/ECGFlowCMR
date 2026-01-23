import argparse
import os
import datetime
import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
import imageio
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

# 模型导入
from models.ecg_encoder import ECGEncoder
from models.dit_flow import ECG_DiT
from models import disentangle_vae3d 

# 工具导入
from util.ukbb_dataset import UKBBLMDBDataset 

def get_args_parser():
    parser = argparse.ArgumentParser('ECG Flow Matching 训练脚本', add_help=False)
    
    # 训练参数
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr', default=1e-4, type=float, help="学习率（从1e-4提升到2e-4以加速收敛）")
    parser.add_argument('--min_lr', default=1e-6, type=float, help="CosineAnnealingLR 的最小学习率")
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--device', default='cuda:3', help='训练使用的设备')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--grad_clip', default=1.0, type=float, help="梯度裁剪阈值，避免梯度爆炸")
    parser.add_argument('--cfg_prob', default=0.1, type=float, help='Classifier-Free Guidance 训练时的 Drop 概率')
    
    # 超参数
    parser.add_argument('--inference_steps', default=10, type=int, help='Flow Matching 推理时的积分步数')
    parser.add_argument('--guidance_scale', default=2.0, type=float, help='推理时的 CFG 强度 (1.0 为标准，>1.0 加强条件约束)')
    parser.add_argument('--train_num_points', default=1, type=int, help='Rectified Flow 训练随机时间点数量（从1增加到2，增强训练信号）')
    parser.add_argument('--dit_hidden', default=512, type=int, help='DiT 隐层维度（可降至512以节省显存）')
    parser.add_argument('--dit_depth', default=8, type=int, help='DiT 层数（可减小以节省显存）')
    parser.add_argument('--dit_heads', default=8, type=int, help='DiT 注意力头数（与 hidden 成比例，减少可省显存）')
    
    # 路径
    parser.add_argument('--root_dir', default='', help='UKBB LMDB 数据集根目录')
    parser.add_argument('--output_dir', default='', help='结果保存路径')
    parser.add_argument('--log_dir', default='', help='日志保存路径')
    
    # 预训练模型路径
    parser.add_argument('--vae_config', default='config/disentangle_vae3d.yaml', help='VAE 配置文件路径')
    parser.add_argument('--vae_ckpt', default='', help='预训练 VAE 权重路径')
    parser.add_argument('--ecg_ckpt', default='', help='预训练 ECG Encoder 权重路径')
    parser.add_argument('--template_path', default='', help='解剖结构模板路径')
    parser.add_argument('--alpha', default=0.8, type=float, help='初始噪声系数 x0 = template + alpha * noise')
    # 模型配置
    parser.add_argument('--ecg_dim', default=768, type=int)
    parser.add_argument('--latent_channels', default=4, type=int)
    parser.add_argument('--latent_size', default=12, type=int)
    parser.add_argument('--num_frames', default=50, type=int)
    
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

def get_scheduler(optimizer, total_steps, min_lr):
    """余弦退火学习率调度"""
    return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=min_lr)

@torch.no_grad()
def load_vae(config_path, ckpt_path, device):
    conf = OmegaConf.load(config_path)
    model = disentangle_vae3d.__dict__['vae3d'](
        lossconfig=conf.model.params.lossconfig,
        ddconfig=conf.model.params.ddconfig,
        embed_dim=conf.model.params.embed_dim,
    )
    
    if os.path.isfile(ckpt_path):
        print(f"正在加载 VAE: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
             state_dict = checkpoint['model_state_dict']
        else:
             state_dict = checkpoint
             
        model.load_state_dict(state_dict, strict=True)
    else:
        raise FileNotFoundError(f"未找到 VAE 权重: {ckpt_path}")
        
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model

# ==========================================
# 验证与可视化函数
# ==========================================

@torch.no_grad()
def visualize_results(model, ecg_encoder, vae, val_loader, device, epoch, args, template=None, num_samples=8):
    """
    可视化：Flow 推理生成 (X0 -> X1) 并保存为 GIF
    左图: Ground Truth (原始 CMR)
    右图: Flow Prediction
    """
    model.eval()
    ecg_encoder.eval()
    
    save_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取一个 Batch 的数据
    try:
        _, ecg, cmr = next(iter(val_loader))
    except StopIteration:
        return
        
    ecg = ecg.to(device)
    cmr = cmr.to(device)
    
    # 限制样本数量
    B = min(ecg.shape[0], num_samples)
    ecg = ecg[:B]
    cmr = cmr[:B]
    
    use_amp = device.type == "cuda"
    with autocast(enabled=use_amp):
        # 1. 准备条件 (ECG)
        ecg_feat, _ = ecg_encoder(ecg)
        
        # 2. 准备初始状态 (X0)：纯噪声
        if cmr.dim() == 4: cmr = cmr.unsqueeze(1)
        
        # 获取 GT Latent 用于对比
        inputs = vae.get_input(cmr)
        # 编码得到 latent（用于确定形状 & 采样噪声），但左侧展示直接使用原始 CMR
        gt_latent = vae.encode_3d(inputs).mode()
        
        curr_x = torch.randn_like(gt_latent)
        if template is not None:
            curr_x = template.to(device) + args.alpha * curr_x
        
        # 3. Flow Matching 推理 (Euler 积分)
        # 从 t=0 到 t=1
        steps = args.inference_steps
        dt = 1.0 / steps
        
        for i in range(steps):
            # 使用中点法：预测区间 [i/steps, (i+1)/steps] 中点的速度
            # 这样更准确，且能确保覆盖整个 [0, 1] 区间
            t_val = (i + 0.5) / steps
            t = torch.full((B,), t_val, device=device)
            
            # 预测速度 v (显式禁用梯度)
            with torch.no_grad():
                if args.guidance_scale > 1.0:
                    # CFG 推理：v = v_uncond + s * (v_cond - v_uncond)
                    v_cond = model.predict_velocity(curr_x, t, ecg_feat)
                    v_uncond = model.predict_velocity(curr_x, t, torch.zeros_like(ecg_feat))
                    v_pred = v_uncond + args.guidance_scale * (v_cond - v_uncond)
                else:
                    v_pred = model.predict_velocity(curr_x, t, ecg_feat)
            
            # 更新状态 X_{t+1} = X_t + v * dt
            curr_x = curr_x + v_pred * dt
        
        # 4. 解码 Latent -> Image
        # curr_x 是预测的最终 Latent (近似 X1)
        # gt_latent 是真实的 Latent (X1)
        
        # 使用 VAE 解码预测结果；GT 直接使用原始 CMR，避免二次重建导致的模糊
        pred_video = vae.decode(curr_x)
        gt_video = cmr if cmr.dim() == 5 else cmr.unsqueeze(1)
    
    # 5. 生成 GIF
    # 视频维度: (B, C, T, H, W) -> 需要转换为 (B, T, H, W, C) 并归一化到 0-255
    pred_video = torch.clamp(pred_video, -1, 1)
    gt_video = torch.clamp(gt_video, -1, 1)
    
    # 归一化到 [0, 1]
    pred_video = (pred_video + 1) / 2.0
    gt_video = (gt_video + 1) / 2.0
    
    # 转换为 numpy
    pred_np = pred_video.cpu().numpy()
    gt_np = gt_video.cpu().numpy()
    
    for i in range(B):
        frames = []
        for t in range(pred_np.shape[2]):
            # 提取第 t 帧: (C, H, W) -> (H, W, C)
            img_pred = np.transpose(pred_np[i, :, t, :, :], (1, 2, 0))
            img_gt = np.transpose(gt_np[i, :, t, :, :], (1, 2, 0))
            
            # 拼接: 左 GT, 右 Pred
            combined = np.concatenate([img_gt, img_pred], axis=1)
            
            # 转为 uint8
            combined = (combined * 255).astype(np.uint8)
            
            #如果是单通道，转为3通道以便保存
            if combined.shape[-1] == 1:
                combined = np.repeat(combined, 3, axis=-1)
                
            frames.append(combined)
            
        # 保存 GIF
        save_path = os.path.join(save_dir, f"epoch_{epoch+1}_sample_{i}.gif")
        imageio.mimsave(save_path, frames, fps=10)
        
    print(f"已保存 {B} 个可视化样本到 {save_dir}")

@torch.no_grad()
def validate(model, ecg_encoder, vae, val_loader, device, args, template=None):
    model.eval()
    ecg_encoder.eval()
    use_amp = device.type == "cuda"
    
    total_loss = 0.0
    count = 0
    
    for batch in tqdm(val_loader, desc="Validating"):
        _, ecg, cmr = batch
        ecg = ecg.to(device, non_blocking=True)
        cmr = cmr.to(device, non_blocking=True)
        B = ecg.shape[0]
        
        with torch.no_grad():
            with autocast(enabled=use_amp):
                # 1. 准备 Target (X1)
                if cmr.dim() == 4: cmr = cmr.unsqueeze(1)
                inputs = vae.get_input(cmr)
                posterior = vae.encode_3d(inputs)
                x_1 = posterior.mode()
                # 【重要】不归一化x1，保持VAE编码后的真实分布
                
                # 2. 准备 Condition (ECG)
                ecg_feat, _ = ecg_encoder(ecg)
                
    # 3. 构造起点：标准正态分布（与训练时一致）
            if template is not None:
                x_0 = template.to(device) + args.alpha * torch.randn_like(x_1)
            else:
                x_0 = torch.randn_like(x_1)
                
            # 4. 使用 Rectified Flow 风格的多点监督
            loss = model(x_1, ecg_feat, x_0)
        
        total_loss += loss.item() * B
        count += B
        
    avg_loss = total_loss / count
    
    return {"loss": avg_loss}

# ==========================================
# 主函数
# ==========================================

def main(args):
    # Setup
    device = torch.device(args.device)
    use_amp = device.type == "cuda"
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Logging
    logger = logging.getLogger("flow")
    if logger.handlers: logger.handlers.clear()
    logger.setLevel("INFO")
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(os.path.join(args.log_dir, f"train_flow_{current_time}.log"), encoding="utf-8")
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False
    logger.info(f"训练参数: {args}")
    
    # 1. 数据集
    logger.info("正在加载数据集...")
    dataset_train = UKBBLMDBDataset(root_dir=args.root_dir, split='train')
    dataset_val = UKBBLMDBDataset(root_dir=args.root_dir, split='val')
    
    logger.info(f"训练集大小: {len(dataset_train)}, 验证集大小: {len(dataset_val)}")
    
    dataloader_train = DataLoader(
        dataset_train, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=8, 
        pin_memory=True,
        collate_fn=UKBBLMDBDataset.collate_fn
    )
    
    dataloader_val = DataLoader(
        dataset_val, 
        batch_size=args.batch_size, # 验证时可以使用相同或不同的 batch size
        shuffle=False, 
        num_workers=4, 
        pin_memory=True,
        collate_fn=UKBBLMDBDataset.collate_fn
    )
    
    # 2. 模型初始化
    logger.info("正在初始化模型...")
    
    # A. VAE (冻结)
    vae = load_vae(args.vae_config, args.vae_ckpt, device)
    
    # B. ECG Encoder (加载权重)
    ecg_encoder = ECGEncoder(out_dim=args.ecg_dim, target_len=args.num_frames).to(device)
    if os.path.exists(args.ecg_ckpt):
        logger.info(f"从 {args.ecg_ckpt} 加载 ECG Encoder")
        ckpt = torch.load(args.ecg_ckpt, map_location=device)
        
        if isinstance(ckpt, dict) and 'encoder' in ckpt:
            state_dict = ckpt['encoder']
        elif isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        else:
            state_dict = ckpt

        cleaned = {}
        valid_keys = set(ecg_encoder.state_dict().keys())
        for k, v in state_dict.items():
            # 清理前缀
            name = k.replace('module.', '').replace('encoder.', '')
            if name in valid_keys:
                cleaned[name] = v
                
        load_res = ecg_encoder.load_state_dict(cleaned, strict=False)
        if load_res.missing_keys:
            logger.warning(f"ECG Encoder 缺失键: {load_res.missing_keys}")
        if load_res.unexpected_keys:
            logger.warning(f"ECG Encoder 多余键: {load_res.unexpected_keys}")
    
    # C. DiT (可训练)
    dit = ECG_DiT(
        input_size=args.latent_size,
        num_frames=args.num_frames,
        in_channels=args.latent_channels,
        ecg_dim=args.ecg_dim,
        hidden_size=args.dit_hidden,
        depth=args.dit_depth,
        num_heads=args.dit_heads,
        train_num_points=args.train_num_points
    ).to(device)
    
    # 3. 优化器 & 调度器
    # 仅优化 DiT 和 ECG Encoder (如果需要微调)
    optimizer = optim.AdamW(list(dit.parameters()) + list(ecg_encoder.parameters()), 
                            lr=args.lr, weight_decay=args.weight_decay)
    
    scaler = GradScaler(enabled=use_amp)
    total_steps = args.epochs * len(dataloader_train)
    scheduler = get_scheduler(optimizer, total_steps, args.min_lr)
    
    start_epoch = 0

    # 4. 训练循环
    logger.info("=" * 80)
    logger.info("训练配置信息:")
    logger.info(f"  总训练步数: {total_steps}")
    logger.info(f"  学习率调度: CosineAnnealingLR (T_max={total_steps}, eta_min={args.min_lr})")
    logger.info(f"  初始学习率: {args.lr}")
    logger.info(f"  train_num_points: {args.train_num_points}")
    
    # 加载 Template
    template = None
    if args.template_path and os.path.exists(args.template_path):
        logger.info(f"正在加载解剖结构模板: {args.template_path}")
        try:
            template = torch.load(args.template_path, map_location='cpu')
            template.requires_grad = False
            
            # 形状调整以支持广播
            if template.dim() == 3: # (C, H, W) -> (C, 1, H, W)
                template = template.unsqueeze(1)
                logger.info(f"Adjusted template shape to {template.shape} for broadcasting")
                
            logger.info("模板加载成功 (Load to CPU)")
        except Exception as e:
            logger.error(f"模板加载失败: {e}")
            template = None
    else:
        logger.warning(f"未找到模板文件: {args.template_path}，将使用纯噪声初始化")

    logger.info("=" * 80)
    logger.info("开始训练...")
    best_val_loss = float('inf')
    
    # 如果恢复训练，尝试从同目录下找 best checkpoint 更新 best_val_loss，或者简单地保留 inf
    # 这里简单保留 inf，确保新训练至少会保存一次 best
    
    for epoch in range(start_epoch, args.epochs):
        dit.train()
        ecg_encoder.train()
        
        running_stats = {"loss": 0.0, "count": 0}
        print_freq = 20
        
        # 记录epoch开始时间
        epoch_start_time = time.time()
        
        # --- 训练阶段 ---
        for step, batch in enumerate(dataloader_train):
            global_step = epoch * len(dataloader_train) + step
            _, ecg, cmr = batch
            ecg = ecg.to(device, non_blocking=True)
            cmr = cmr.to(device, non_blocking=True)
            
            B = ecg.shape[0]
            
            # --- A. 准备 Target Latent (X1) ---
            with torch.no_grad():
                if cmr.dim() == 4: cmr = cmr.unsqueeze(1)
                
                inputs = vae.get_input(cmr)
                posterior = vae.encode_3d(inputs)
                x_1 = posterior.mode() # (B, 4, 50, 12, 12)
                # 【重要】不归一化x1，保持VAE编码后的真实分布
                # x1的方差约0.8是VAE编码后的正常现象，Flow Matching应该学习这个真实分布
            
            # --- B. 前向与损失 (AMP) ---
            with autocast(enabled=use_amp):
                # 条件 (ECG)
                ecg_feat, _ = ecg_encoder(ecg) # ecg_feat: (B, 50, 768)
                
                # [改进] Classifier-Free Guidance (CFG) Training
                # 随机 Drop 条件，使模型能同时学习无条件分布和有条件分布
                # 这能增强推理时对条件的响应能力，从而提升 SSIM 和结构一致性
                if args.cfg_prob > 0:
                    # 1.0 - cfg_prob 的概率保持 (keep)，cfg_prob 的概率置零 (drop)
                    mask_prob = 1.0 - args.cfg_prob
                    # 生成 mask (B, 1, 1)，并在 batch 维度独立采样
                    mask = torch.bernoulli(torch.full((B, 1, 1), mask_prob, device=device))
                    ecg_feat = ecg_feat * mask
                
    # 构造起点：使用标准正态分布
            # Flow Matching会学习从N(0,1)到x1真实分布的流
            # 如果x1方差是0.8，模型会学习收缩的流，这是合理的
            if template is not None:
                x_0 = template.to(device) + args.alpha * torch.randn_like(x_1)
            else:
                x_0 = torch.randn_like(x_1)
                
            # Rectified Flow 风格的多点监督
            total_loss = dit(x_1, ecg_feat, x_0)
            
            # --- E. 反向传播 (AMP) ---
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            # 先反缩放再裁剪梯度，稳住训练
            scaler.unscale_(optimizer)
            clip_grad_norm_(list(dit.parameters()) + list(ecg_encoder.parameters()), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # --- F. 记录 ---
            current_lr = scheduler.get_last_lr()[0]
            running_stats["loss"] += total_loss.item()
            running_stats["count"] += 1
            
            if step % print_freq == 0:
                avg_loss = running_stats["loss"] / running_stats["count"]
                # 计算ETA（预计剩余时间）
                elapsed_time = time.time() - epoch_start_time
                if step > 0:
                    # 计算每个step的平均时间
                    time_per_step = elapsed_time / (step + 1)
                    # 计算剩余steps
                    remaining_steps = len(dataloader_train) - step - 1
                    # 计算剩余时间（秒）
                    eta_seconds = time_per_step * remaining_steps
                    # 格式化为 HH:MM:SS
                    eta_hours = int(eta_seconds // 3600)
                    eta_minutes = int((eta_seconds % 3600) // 60)
                    eta_secs = int(eta_seconds % 60)
                    eta_str = f"{eta_hours:02d}:{eta_minutes:02d}:{eta_secs:02d}"
                else:
                    eta_str = "计算中..."
                
                # 打印当前 batch 的 x0/x1 统计信息，便于监控分布是否异常
                x0_mean, x0_std = x_0.mean().item(), x_0.std().item()
                x1_mean, x1_std = x_1.mean().item(), x_1.std().item()
                
                logger.info(
                    f"Epoch [{epoch+1}/{args.epochs}] Step [{step}/{len(dataloader_train)}] "
                    f"Loss: {avg_loss:.4f} "
                    f"x0均值/方差: {x0_mean:.4f}/{x0_std:.4f} | x1均值/方差: {x1_mean:.4f}/{x1_std:.4f} | "
                    f"LR: {current_lr:.2e} | ETA: {eta_str}"
                )
                running_stats = {k: 0.0 for k in running_stats}; running_stats["count"] = 0
        
        # --- 验证阶段 ---
        logger.info(f"正在验证 Epoch {epoch+1} ...")
        val_stats = validate(dit, ecg_encoder, vae, dataloader_val, device, args, template=template)
        logger.info(
            f"Epoch [{epoch+1}/{args.epochs}] Val Loss: {val_stats['loss']:.4f}"
        )
        
        # --- 可视化 ---
        if (epoch + 1) % 1 == 0: # 每个 Epoch 都进行可视化 (或者设置间隔)
            logger.info("正在生成可视化 GIF...")
            visualize_results(dit, ecg_encoder, vae, dataloader_val, device, epoch, args, template=template)
        
        # --- 保存 Checkpoint ---
        save_dict = {
            'epoch': epoch,
            'model_state_dict': dit.state_dict(),
            'encoder_state_dict': ecg_encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_stats['loss']
        }
        
        # 1. 保存当前 Epoch
        ckpt_path = os.path.join(args.output_dir, f"checkpoint_flow_{epoch+1}.pth")
        torch.save(save_dict, ckpt_path)
        logger.info(f"已保存 Checkpoint: {ckpt_path}")
        
        # 2. 保存最佳模型
        if val_stats['loss'] < best_val_loss:
            best_val_loss = val_stats['loss']
            best_ckpt_path = os.path.join(args.output_dir, "checkpoint_flow_best.pth")
            torch.save(save_dict, best_ckpt_path)
            logger.info(f"发现更优模型 (Loss: {best_val_loss:.4f}), 已更新 Best Checkpoint")

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)