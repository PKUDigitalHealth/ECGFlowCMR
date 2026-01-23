import math
import sys
import os
import time
import logging
import numpy as np
import torch
import imageio
import datetime
from typing import Iterable
import util.misc as misc  # 假设你保留了原有的工具库

def train_one_epoch(model,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    optimizer_d: torch.optim.Optimizer,
                    scheduler,
                    device: torch.device, 
                    epoch: int, 
                    loss_scaler,
                    args=None):
    """
    单个 epoch 的训练逻辑：包含 VAE 重建损失与 Discriminator 对抗损失的双重优化。
    """
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    logger = logging.getLogger("vae")

    # 训练状态累计
    running_stats = {
        "loss": 0.0, "rec": 0.0, "kl": 0.0, "disc": 0.0, "count": 0
    }

    start_time = time.time()
    end_iter = time.time()
    
    # ------------------------------------------------------------------
    #  Training Loop
    # ------------------------------------------------------------------
    for data_iter_step, batch in enumerate(data_loader):
        
        # 1. 解包数据 (配合 UKBBLMDBDataset)
        # batch: (eids, ecg, cmr) -> 我们只需要 cmr
        _, _, samples = batch
        samples = samples.to(device, non_blocking=True)

        # 2. 维度检查与调整: (B, T, H, W) -> (B, C=1, T, H, W)
        # 我们的 VAE 严格需要 5D 输入
        if samples.dim() == 4:
            samples = samples.unsqueeze(1).contiguous()
        
        # 获取模型标准输入 (确保内存连续)
        inputs = model.get_input(samples)

        # --------------------------------------------------------------
        #  Part A: 训练 VAE (Generator)
        # --------------------------------------------------------------
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            # 前向传播
            reconstructions, posterior = model(inputs)

            # 仅在首个 batch 记录一次重建尺寸，便于排查尺寸对齐问题
            if data_iter_step == 0:
                logger.info(f"[Train] input shape: {inputs.shape}, recon shape: {reconstructions.shape}")
            
            # 计算 Generator Loss (Reconstruction + KL + Generator part of GAN)
            # 注意：optimizer_idx = 0 表示训练生成器
            aeloss, log_dict_ae = model.loss(
                inputs, reconstructions, posterior, 
                optimizer_idx=0, 
                global_step=model.global_step,
                last_layer=model.get_last_layer(), 
                split="train"
            )

        # 反向传播 VAE
        loss_scaler.scale(aeloss).backward()
        if args.grad_clip is not None:
            loss_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        loss_scaler.step(optimizer)
        loss_scaler.update()

        # --------------------------------------------------------------
        #  Part B: 训练 Discriminator
        # --------------------------------------------------------------
        optimizer_d.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            # 计算 Discriminator Loss
            # 注意：optimizer_idx = 1 表示训练判别器
            discloss, log_dict_disc = model.loss(
                inputs, reconstructions, posterior, 
                optimizer_idx=1, 
                global_step=model.global_step,
                last_layer=model.get_last_layer(), 
                split="train"
            )

        # 反向传播 Discriminator
        loss_scaler.scale(discloss).backward()
        if args.grad_clip is not None:
            loss_scaler.unscale_(optimizer_d)
            # 注意：这里只裁剪判别器的参数
            if hasattr(model.loss, 'discriminator'):
                torch.nn.utils.clip_grad_norm_(model.loss.discriminator.parameters(), args.grad_clip)
        loss_scaler.step(optimizer_d)
        loss_scaler.update()

        # --------------------------------------------------------------
        #  Logging & Updates
        # --------------------------------------------------------------
        model.global_step += 1
        loss_value = aeloss.item()

        # 安全检查：防止 Loss 变成 NaN
        if not math.isfinite(loss_value):
            logger.error(f"Loss is {loss_value} at epoch {epoch}, step {data_iter_step}. Skipping this batch.")
            optimizer.zero_grad(set_to_none=True)
            optimizer_d.zero_grad(set_to_none=True)
            continue
            # sys.exit(1)

        # 获取当前 LR
        lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]["lr"]

        # 更新统计
        running_stats["loss"] += loss_value
        running_stats["rec"] += log_dict_ae.get('train/rec_loss', 0.0)
        running_stats["kl"] += log_dict_ae.get('train/kl_loss', 0.0)
        running_stats["disc"] += log_dict_disc.get('train/disc_loss', 0.0)
        running_stats["count"] += 1

        # Tensorboard / MetricLogger 更新
        metric_logger.update(loss=loss_value)
        metric_logger.update(rec_loss=log_dict_ae.get('train/rec_loss', 0.0))
        metric_logger.update(kl_loss=log_dict_ae.get('train/kl_loss', 0.0))
        metric_logger.update(disc_loss=log_dict_disc.get('train/disc_loss', 0.0))
        metric_logger.update(lr=lr)

        # Step Scheduler per Iteration (Batch)
        if scheduler is not None:
            scheduler.step()

        # 打印日志 (每 print_freq 步)
        if data_iter_step % print_freq == 0:
            if misc.is_main_process():
                avg_loss = running_stats["loss"] / running_stats["count"]
                avg_rec = running_stats["rec"] / running_stats["count"]
                avg_kl = running_stats["kl"] / running_stats["count"]
                avg_disc = running_stats["disc"] / running_stats["count"]
                
                logger.info(
                    f"Epoch [{epoch}] [{data_iter_step}/{len(data_loader)}] "
                    f"L_Total: {avg_loss:.4f} | Rec: {avg_rec:.4f} | KL: {avg_kl:.6f} | Disc: {avg_disc:.4f} | LR: {lr:.2e}"
                )
                # 重置
                running_stats = {k: 0.0 for k in running_stats}
                running_stats["count"] = 0

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate(model,
             data_loader: Iterable,
             device: torch.device,
             epoch: int,
             output_dir: str,
             max_vis: int = 8):
    """
    验证函数：计算验证集 Loss 并生成对比 GIF。
    """
    model.eval()
    logger = logging.getLogger("vae")
    
    # 累加器
    val_loss_dict = {"rec": 0.0, "kl": 0.0, "disc": 0.0, "total": 0.0, "count": 0}

    def _prep_video(x: torch.Tensor) -> np.ndarray:
        """
        工具函数：将 5D Tensor (C, T, H, W) 转换为 GIF 友好的 numpy 数组 (T, H, W, 3)。
        针对 CMR (灰度) 进行优化。
        """
        x = x.detach().cpu()
        
        # 1. 维度处理 (C, T, H, W) -> (T, C, H, W)
        if x.dim() == 4: # (C, T, H, W)
            x = x.permute(1, 0, 2, 3) 
        elif x.dim() == 3: # (T, H, W)
            x = x.unsqueeze(1) # (T, 1, H, W)
        
        # 2. 归一化反转 [-1, 1] -> [0, 255]
        x = torch.clamp(x, -1.0, 1.0)
        x = (x + 1.0) / 2.0
        x = (x.numpy() * 255.0).astype(np.uint8) # (T, C, H, W)
        
        # 3. 通道处理 (若是单通道，复制为 3 通道 RGB)
        if x.shape[1] == 1:
            x = np.repeat(x, 3, axis=1) # (T, 3, H, W)
            
        # 4. 最终调整为 imageio 格式 (T, H, W, C)
        return np.transpose(x, (0, 2, 3, 1))

    # ------------------------------------------------------------------
    #  Validation Loop
    # ------------------------------------------------------------------
    for i, batch in enumerate(data_loader):
        _, _, samples = batch
        samples = samples.to(device, non_blocking=True)
        if samples.dim() == 4:
            samples = samples.unsqueeze(1).contiguous()
            
        inputs = model.get_input(samples)

        with torch.cuda.amp.autocast():
            reconstructions, posterior = model(inputs)

            # 仅在首个 batch 记录一次重建尺寸，便于排查尺寸对齐问题
            if i == 0:
                logger.info(f"[Val] input shape: {inputs.shape}, recon shape: {reconstructions.shape}")
            
            # 计算 Loss (split='val')
            aeloss, log_dict_ae = model.loss(
                inputs, reconstructions, posterior, 0, model.global_step,
                last_layer=model.get_last_layer(), split="val"
            )
            discloss, log_dict_disc = model.loss(
                inputs, reconstructions, posterior, 1, model.global_step,
                last_layer=model.get_last_layer(), split="val"
            )

        # 累计 Loss
        bs = inputs.size(0)
        val_loss_dict["rec"] += log_dict_ae.get('val/rec_loss', 0.0) * bs
        val_loss_dict["kl"] += log_dict_ae.get('val/kl_loss', 0.0) * bs
        val_loss_dict["disc"] += log_dict_disc.get('val/disc_loss', 0.0) * bs
        val_loss_dict["total"] += (aeloss.item() + discloss.item()) * bs
        val_loss_dict["count"] += bs

        # --------------------------------------------------------------
        #  Visualization (Save GIFs) - 只在第一个 Batch 做
        # --------------------------------------------------------------
        if i == 0:
            save_dir = os.path.join(output_dir, "val_viz")
            os.makedirs(save_dir, exist_ok=True)
            
            num_to_save = min(max_vis, bs)
            logger.info(f"Saving {num_to_save} validation GIFs to {save_dir}...")
            
            for idx in range(num_to_save):
                # 准备真实视频 (GT) 和 重建视频 (Recon)
                gt_vid = _prep_video(inputs[idx])      # (T, H, W, 3)
                recon_vid = _prep_video(reconstructions[idx]) # (T, H, W, 3)
                
                # 左右拼接
                combined_vid = np.concatenate([gt_vid, recon_vid], axis=2) # 沿 Width 拼接
                
                # 保存 GIF
                save_path = os.path.join(save_dir, f"epoch_{epoch:03d}_sample_{idx:02d}.gif")
                imageio.mimsave(save_path, list(combined_vid), format='GIF', fps=10, loop=0)

    # ------------------------------------------------------------------
    #  Summary
    # ------------------------------------------------------------------
    count = max(1, val_loss_dict["count"])
    stats = {k: v / count for k, v in val_loss_dict.items() if k != "count"}
    
    return {
        "val_loss": stats["total"],
        "val_rec_loss": stats["rec"],
        "val_kl_loss": stats["kl"],
        "val_disc_loss": stats["disc"],
    }