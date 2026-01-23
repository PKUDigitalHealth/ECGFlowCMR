import argparse
import os
import torch
import logging
from tqdm import tqdm
import numpy as np
import imageio
import time
from torch.cuda.amp import autocast
from omegaconf import OmegaConf

# 模型
from models.ecg_encoder import ECGEncoder
from models.dit_flow import ECG_DiT
from models import disentangle_vae3d

# 数据集
from util.ukbb_dataset import UKBBLMDBDataset
from torch.utils.data import DataLoader


def get_args_parser():
    parser = argparse.ArgumentParser("ECG Flow Matching 推理脚本", add_help=False)
    # 数据 & 路径
    parser.add_argument("--root_dir", type=str, default="", help="UKBB LMDB 根目录")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="推理数据划分")
    parser.add_argument("--output_dir", type=str, default="/nvme2/chenggf/fangxiaocheng/ECGFlowCMR/results/ecg_flow_cmr_0.6", help="输出根目录")
    parser.add_argument("--log_level", type=str, default="INFO", help="日志等级")

    # 预训练/权重
    parser.add_argument("--flow_ckpt", type=str, default="", help="Flow + ECG 编码器权重")
    parser.add_argument("--vae_config", type=str, default="", help="VAE 配置")
    parser.add_argument("--vae_ckpt", type=str, default="", help="VAE 权重")
    parser.add_argument("--ecg_ckpt", type=str, default="", help="预训练 ECG Encoder 权重路径")
    parser.add_argument("--template_path", type=str, default="", help="解剖模板")

    # 推理超参
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--alpha_noise", type=float, default=0.6, help="初始噪声系数 (需与训练时 alpha 一致)")
    parser.add_argument("--inference_steps", type=int, default=10)
    parser.add_argument('--guidance_scale', default=2.0, type=float, help='推理时的 CFG 强度 (1.0 为标准，>1.0 加强条件约束)')
    parser.add_argument('--train_num_points', default=1, type=int, help='模型初始化参数，需与训练保持一致')

    # 模型形状配置（需与训练脚本保持一致）
    parser.add_argument("--ecg_dim", type=int, default=768)
    parser.add_argument("--latent_channels", type=int, default=4)
    parser.add_argument("--latent_size", type=int, default=12)
    parser.add_argument("--num_frames", type=int, default=50)
    
    # DiT 结构配置 (必须与训练 checkpoint 一致)
    parser.add_argument('--dit_hidden', default=512, type=int)
    parser.add_argument('--dit_depth', default=8, type=int)
    parser.add_argument('--dit_heads', default=8, type=int)

    # 可视化
    parser.add_argument("--max_batches", type=int, default=-1, help="最多推理多少个 batch；-1 表示全量")
    parser.add_argument("--fps", type=int, default=10)

    # 随机种子
    parser.add_argument("--seed", type=int, default=42, help="随机种子，用于可重现的推理结果")

    return parser


@torch.no_grad()
def load_vae(config_path, ckpt_path, device):
    conf = OmegaConf.load(config_path)
    model = disentangle_vae3d.__dict__["vae3d"](
        lossconfig=conf.model.params.lossconfig,
        ddconfig=conf.model.params.ddconfig,
        embed_dim=conf.model.params.embed_dim,
    )
    if os.path.isfile(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=True)
    else:
        raise FileNotFoundError(f"未找到 VAE 权重: {ckpt_path}")
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model


def setup_logger(level: str):
    logger = logging.getLogger("infer")
    if logger.handlers:
        logger.handlers.clear()
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    logger.propagate = False
    return logger


def prepare_models(args, device, logger):
    # 1. Load VAE
    vae = load_vae(args.vae_config, args.vae_ckpt, device)

    # 2. Load ECG Encoder
    ecg_encoder = ECGEncoder(out_dim=args.ecg_dim, target_len=args.num_frames).to(device)
    
    # 3. Load DiT
    # [修正] 使用与训练脚本一致的参数初始化 DiT，移除不存在的 phase_window_size
    # 确保 hidden_size, depth, num_heads 与训练配置一致
    dit = ECG_DiT(
        input_size=args.latent_size,
        num_frames=args.num_frames,
        in_channels=args.latent_channels,
        ecg_dim=args.ecg_dim,
        hidden_size=args.dit_hidden,
        depth=args.dit_depth,
        num_heads=args.dit_heads,
        train_num_points=args.train_num_points # 保持与训练初始化参数一致
    ).to(device)

    # 4. Load Weights
    if not os.path.isfile(args.flow_ckpt):
        raise FileNotFoundError(f"未找到 Flow 检查点: {args.flow_ckpt}")
    
    logger.info(f"正在加载 Flow Checkpoint: {args.flow_ckpt}")
    ckpt = torch.load(args.flow_ckpt, map_location=device)

    # A. 加载 DiT
    model_sd = ckpt.get("model_state_dict", ckpt)
    dit.load_state_dict(model_sd, strict=True) # 建议 strict=True 以确保结构匹配

    # B. 加载 ECG Encoder
    # 仅从 ecg_ckpt 加载，不使用 flow checkpoint 中的 encoder 权重
    if os.path.isfile(args.ecg_ckpt):
        logger.info(f"正在加载 ECG Checkpoint: {args.ecg_ckpt}")
        ecg_ckpt = torch.load(args.ecg_ckpt, map_location=device)
        
        if isinstance(ecg_ckpt, dict) and 'encoder' in ecg_ckpt:
            state_dict = ecg_ckpt['encoder']
        elif isinstance(ecg_ckpt, dict) and 'model_state_dict' in ecg_ckpt:
            state_dict = ecg_ckpt['model_state_dict']
        else:
            state_dict = ecg_ckpt

        cleaned_sd = {}
        valid_keys = set(ecg_encoder.state_dict().keys())
        for k, v in state_dict.items():
            name = k.replace('module.', '').replace('encoder.', '')
            if name in valid_keys:
                cleaned_sd[name] = v
        
        load_res = ecg_encoder.load_state_dict(cleaned_sd, strict=False)
        logger.info(f"已加载预训练 ECG Encoder. Missing: {load_res.missing_keys}")
    else:
        logger.warning(f"未找到 ECG Encoder 权重: {args.ecg_ckpt}，将使用随机初始化！")

    dit.eval()
    ecg_encoder.eval()
    for p in dit.parameters():
        p.requires_grad = False
    for p in ecg_encoder.parameters():
        p.requires_grad = False

    # 5. Load Template
    if args.template_path and os.path.isfile(args.template_path):
        template = torch.load(args.template_path, map_location=device)
        if template.dim() == 3: # (C, H, W) -> (C, 1, H, W) 适配广播
             template = template.unsqueeze(1)
        # 移除显式 repeat，依赖 PyTorch 广播机制 (B, C, T, H, W)
        # if template.shape[2] == 1: 
        #    template = template.repeat(1, 1, args.num_frames, 1, 1)
        logger.info("已加载解剖模板")
    else:
        logger.warning(f"未找到模板: {args.template_path}，将使用纯随机噪声")
        template = None
        
    return vae, ecg_encoder, dit, template


@torch.no_grad()
def run_inference(args):
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(args.log_level)
    device = torch.device(args.device)

    # 设置随机种子以获得可重现的结果
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    logger.info(f"设置随机种子: {args.seed}")

    # 数据
    dataset = UKBBLMDBDataset(root_dir=args.root_dir, split=args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=UKBBLMDBDataset.collate_fn,
    )
    logger.info(f"推理数据集 {args.split}: {len(dataset)} 个样本，batch={args.batch_size}")

    # 模型
    vae, ecg_encoder, dit, template = prepare_models(args, device, logger)

    save_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(save_dir, exist_ok=True)

    processed_batches = 0
    use_amp = device.type == "cuda"

    all_gts = []
    all_fakes = []
    batch_times = []  # 存储前5个batch的推理时间

    for batch_idx, batch in enumerate(tqdm(loader, desc="Infer")):
        if args.max_batches > 0 and processed_batches >= args.max_batches:
            break
        processed_batches += 1

        # 记录前5个batch的开始时间
        if batch_idx < 5:
            batch_start_time = time.time()

        eids, ecg, cmr = batch
        ecg = ecg.to(device, non_blocking=True)
        cmr = cmr.to(device, non_blocking=True)
        
        # Ground Truth 准备
        if cmr.dim() == 4:
            cmr = cmr.unsqueeze(1) # (B, 1, T, H, W)

        B = ecg.shape[0]
        with autocast(enabled=use_amp):
            # 1. ECG 条件提取
            ecg_feat, _ = ecg_encoder(ecg) # 忽略 ecg_phase，因为 DiT 不需要

            # 2. 准备初始噪声 x0
            # 使用 VAE 编码 GT 仅为了获取正确的 Latent 形状参考
            inputs = vae.get_input(cmr)
            gt_latent = vae.encode_3d(inputs).mode() 

            noise = torch.randn_like(gt_latent)
            
            if template is not None:
                # 模板 + 噪声 (Rectified Flow 训练时的设定)
                x0 = template.to(device) + args.alpha_noise * noise
            else:
                x0 = noise

            curr_x = x0

            # 3. Flow Matching 推理过程 (Euler / Midpoint 积分)
            steps = args.inference_steps
            dt = 1.0 / steps
            
            for i in range(steps):
                # Midpoint method (与训练可视化一致)
                t_val = (i + 0.5) / steps
                t = torch.full((B,), t_val, device=device)
                
                # CFG 推理逻辑
                if args.guidance_scale > 1.0:
                     # v = v_uncond + s * (v_cond - v_uncond)
                    v_cond = dit.predict_velocity(curr_x, t, ecg_feat)
                    v_uncond = dit.predict_velocity(curr_x, t, torch.zeros_like(ecg_feat))
                    v_pred = v_uncond + args.guidance_scale * (v_cond - v_uncond)
                else:
                    v_pred = dit.predict_velocity(curr_x, t, ecg_feat)
                
                curr_x = curr_x + v_pred * dt

            # 4. 解码
            pred_video = vae.decode(curr_x)
            gt_video = cmr

        # 后处理
        pred_video = torch.clamp(pred_video, -1, 1)
        gt_video = torch.clamp(gt_video, -1, 1)
        pred_video = (pred_video + 1) / 2.0
        gt_video = (gt_video + 1) / 2.0

        # 收集结果 (CPU numpy)
        all_fakes.append(pred_video.cpu().numpy())
        all_gts.append(gt_video.cpu().numpy())

        # 记录前5个batch的推理时间
        if batch_idx < 5:
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            batch_times.append(batch_time)

            # 实时输出每个batch的时间
            logger.info(f"Batch {batch_idx+1} 推理时间: {batch_time:.4f} 秒, 平均每个样本: {batch_time/args.batch_size:.4f} 秒")

        # 处理完第5个batch后输出统计
        if batch_idx == 4 and len(batch_times) == 5:
            avg_batch_time = np.mean(batch_times)
            std_batch_time = np.std(batch_times)
            logger.info(f"=== 前5个batch推理时间统计 ===")
            logger.info(f"平均batch推理时间: {avg_batch_time:.4f} ± {std_batch_time:.4f} 秒")
            logger.info(f"平均每个样本推理时间: {avg_batch_time/args.batch_size:.4f} 秒")
            logger.info(f"推理速度: {1.0/(avg_batch_time/args.batch_size):.2f} samples/秒")

    # 保存 NPY 文件
    if all_fakes:
        all_fakes = np.concatenate(all_fakes, axis=0)
        all_gts = np.concatenate(all_gts, axis=0)
        
        fake_path = os.path.join(args.output_dir, "fake.npy")
        gt_path = os.path.join(args.output_dir, "gt.npy")
        
        np.save(fake_path, all_fakes)
        np.save(gt_path, all_gts)
        
        logger.info(f"推理完成，NPY 文件保存完毕:")
        logger.info(f"  Fake: {fake_path}, Shape: {all_fakes.shape}")
        logger.info(f"  GT:   {gt_path}, Shape: {all_gts.shape}")
    else:
        logger.warning("未生成任何样本！")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    run_inference(args)
