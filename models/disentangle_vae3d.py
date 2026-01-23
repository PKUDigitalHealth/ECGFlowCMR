"""
VAE3D 模型定义 (完整版)
适配: Anisotropic Compression (Time-Keep f_t=0, Space-Compress f_s=8)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import numpy as np
from einops import rearrange
import math
import logging
from ldm.modules.diffusionmodules.model import make_conv, nonlinearity, ResnetBlock, Normalize
from ldm.util import instantiate_from_config, default
from itertools import repeat

# 尝试导入 xformers 进行加速
try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except ImportError:
    XFORMERS_IS_AVAILBLE = False
    print("No module 'xformers'. Proceeding without it.")


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv, out_channels=None, keep_temp=False, mode='3d'):
        super().__init__()
        self.with_conv = with_conv
        self.keep_temp = keep_temp
        out_channels = in_channels if out_channels is None else out_channels
        
        if self.with_conv:
            self.conv = make_conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, mode=mode)

    def forward(self, x):
        # 针对 3D 数据的上采样逻辑
        if self.keep_temp:
            # 时间不变 (1.0), 空间翻倍 (2.0)
            x = F.interpolate(x, scale_factor=(1.0, 2.0, 2.0), mode="nearest")
        else:
            # 时间空间同时翻倍
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
            
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv, mode='3d', keep_temp=False):
        super().__init__()
        self.with_conv = with_conv
        self.keep_temp = keep_temp
        
        if with_conv:
            if keep_temp:
                # 仅下采样空间: Kernel(1,3,3), Stride(1,2,2)
                kernel_size = (1, 3, 3)
                stride = (1, 2, 2)
                padding = (0, 1, 1)
            else:
                # 下采样时间与空间: Kernel(3,3,3), Stride(2,2,2)
                kernel_size = (3, 3, 3)
                stride = (2, 2, 2)
                padding = (1, 1, 1)
                
            self.conv = make_conv(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, mode=mode)

    def forward(self, x):
        if self.with_conv:
            x = self.conv(x)
        else:
            # AvgPool 回退逻辑
            if self.keep_temp:
                x = F.avg_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            else:
                x = F.avg_pool3d(x, kernel_size=2, stride=2)
        return x


class MemoryEfficientAttnBlock(nn.Module):
    """
    3D 自注意力模块，支持 xformers 加速。
    """
    def __init__(self, in_channels, num_heads=8, mode='3d', **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.norm = Normalize(in_channels)
        
        # 1x1x1 Convs for Q, K, V, Output
        self.q = make_conv(in_channels, in_channels, kernel_size=1, stride=1, padding=0, mode=mode)
        self.k = make_conv(in_channels, in_channels, kernel_size=1, stride=1, padding=0, mode=mode)
        self.v = make_conv(in_channels, in_channels, kernel_size=1, stride=1, padding=0, mode=mode)
        self.proj_out = make_conv(in_channels, in_channels, kernel_size=1, stride=1, padding=0, mode=mode)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        
        # Compute Q, K, V
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # Rearrange for Attention: B, C, T, H, W -> B, (T*H*W), C
        b, c, t, h, w = q.shape
        q, k, v = map(lambda t: rearrange(t, 'b c t h w -> b (t h w) c').contiguous(), (q, k, v))

        # Split Heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.num_heads), (q, k, v))

        if XFORMERS_IS_AVAILBLE:
            out = xformers.ops.memory_efficient_attention(q, k, v)
        else:
            # Standard Scaled Dot-Product Attention
            d = q.shape[-1]
            attn = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(d), dim=-1)
            out = attn @ v

        # Merge Heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self.num_heads)
        
        # Reshape back to 3D
        out = rearrange(out, 'b (t h w) c -> b c t h w', t=t, h=h, w=w)
        
        out = self.proj_out(out)
        return x + out


class Encoder(nn.Module):
    def __init__(self, *, ch, ch_mult=(1, 2, 4, 8), num_res_blocks, 
                 resolution, in_channels, z_channels, out_z=True, 
                 f_t=0, f_s=None, mode='3d', dropout=0.0, use_checkpoint=False, **kwargs):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.out_z = out_z
        self.mode = mode
        
        # 默认空间压缩层数为总层数-1 (因为最后一层是bottleneck)
        f_s = self.num_resolutions - 1 if f_s is None else f_s
        
        # Initial Convolution
        self.conv_in = make_conv(in_channels, ch, kernel_size=3, stride=1, padding=1, mode=mode)
        
        self.down = nn.ModuleList()
        block_in = ch
        
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, 
                                         dropout=dropout, use_checkpoint=use_checkpoint, mode=mode))
                block_in = block_out
                
                # 在低分辨率层添加 Self-Attention (通常是最后两层)
                if i_level >= self.num_resolutions - 2:
                    attn.append(MemoryEfficientAttnBlock(block_in, mode=mode))
                else:
                    attn.append(nn.Identity())
                    
            down = nn.Module()
            down.block = block
            down.attn = attn
            
            # Downsampling logic
            # 如果 f_t=0，意味着所有层都 keep_temp=True (只压缩空间)
            # 如果 f_t > 0，则前 f_t 层压缩时间
            if i_level != self.num_resolutions - 1:
                # 判断当前层是否需要保留时间维度
                # 如果 config f_t=0, 则 keep_temp 始终为 True
                keep_temp = True if f_t == 0 else (i_level >= f_t)
                
                down.downsample = Downsample(block_in, with_conv=True, mode=mode, keep_temp=keep_temp)
            else:
                down.downsample = nn.Identity()
                
            self.down.append(down)

        # Middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout, use_checkpoint=use_checkpoint, mode=mode)
        self.mid.attn_1 = MemoryEfficientAttnBlock(block_in, mode=mode)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout, use_checkpoint=use_checkpoint, mode=mode)

        # Output
        self.norm_out = Normalize(block_in)
        self.conv_out = make_conv(block_in, 2*z_channels if out_z else z_channels, 
                                  kernel_size=3, stride=1, padding=1, mode=mode)

    def forward(self, x):
        h = self.conv_in(x)
        first_log = not getattr(self, "_logged", False)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                h = self.down[i_level].attn[i_block](h)
            if first_log and i_level == 0:
                logging.getLogger("vae").info(f"[Encoder] level {i_level} pre-down shape: {h.shape}")
            h = self.down[i_level].downsample(h)
            if first_log:
                logging.getLogger("vae").info(f"[Encoder] level {i_level} post-down shape: {h.shape}")

        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        if first_log:
            logging.getLogger("vae").info(f"[Encoder] mid shape: {h.shape}")
            self._logged = True
        
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder3D(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks, 
                 resolution, z_channels, f_t=0, mode='3d', dropout=0.0, use_checkpoint=False, **kwargs):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        
        block_in = ch * ch_mult[-1]
        
        # Initial Conv
        self.conv_in = make_conv(z_channels, block_in, kernel_size=3, stride=1, padding=1, mode=mode)

        # Middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout, use_checkpoint=use_checkpoint, mode=mode)
        self.mid.attn_1 = MemoryEfficientAttnBlock(block_in, mode=mode)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout, use_checkpoint=use_checkpoint, mode=mode)

        self.up = nn.ModuleList()
        
        # Reverse loop for Upsampling
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            
            # +1 block for symmetry
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout, use_checkpoint=use_checkpoint, mode=mode))
                block_in = block_out
                
                if i_level >= self.num_resolutions - 2:
                    attn.append(MemoryEfficientAttnBlock(block_in, mode=mode))
                else:
                    attn.append(nn.Identity())
            
            up = nn.Module()
            up.block = block
            up.attn = attn
            
            # Upsampling logic (Symmetric to Encoder)
            if i_level > 0:
                # Encoder: i_level 0->1 (down), 1->2 (down)...
                # Decoder: i_level 2->1 (up), 1->0 (up)
                # 这里的逻辑需要小心：i_level 对应的是 Target Level
                # 如果 f_t=0, 始终 keep_temp=True
                keep_temp = True if f_t == 0 else (i_level > f_t) # 简化逻辑，假设 Encoder 逻辑如上
                
                up.upsample = Upsample(block_in, with_conv=True, mode=mode, keep_temp=keep_temp)
            else:
                up.upsample = nn.Identity()
            
            self.up.insert(0, up)

        self.norm_out = Normalize(block_in)
        self.conv_out = make_conv(block_in, out_ch, kernel_size=3, stride=1, padding=1, mode=mode)

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        first_log = not getattr(self, "_logged", False)
        if first_log:
            logging.getLogger("vae").info(f"[Decoder] input shape: {h.shape}")

        # Iterate from bottom (lowest res) to top
        for i_level in reversed(range(len(self.up))):
            for i_block in range(len(self.up[i_level].block)):
                h = self.up[i_level].block[i_block](h)
                h = self.up[i_level].attn[i_block](h)
            h = self.up[i_level].upsample(h)
            if first_log:
                logging.getLogger("vae").info(f"[Decoder] level {i_level} post-up shape: {h.shape}")

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if first_log:
            logging.getLogger("vae").info(f"[Decoder] output shape: {h.shape}")
            self._logged = True
        return h


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        self.deterministic = deterministic
        
    def sample(self):
        # Reparameterization trick
        x = self.mean + self.std * torch.randn_like(self.mean)
        return x
        
    def mode(self):
        return self.mean

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                # KL to Standard Normal N(0, I)
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3, 4])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3, 4])


class VAE3d(nn.Module):
    """
    3D VAE 主类
    """
    def __init__(self, lossconfig, ddconfig, embed_dim, 
                 ckpt_path=None, ignore_keys=[], **kwargs):
        super().__init__()
        self.global_step = 0
        
        # 实例化 Loss
        self.loss = instantiate_from_config(lossconfig)
        
        # 3D Encoder & Decoder
        # out_z=True 表示 Encoder 输出均值和方差
        self.encoder_3d = Encoder(out_z=True, **ddconfig)
        self.decoder = Decoder3D(**ddconfig)
        
        # Latent 空间对齐卷积 (1x1 Conv3d)
        # encoder_out_ch -> 2 * embed_dim (for mu, logvar) -> embed_dim (for decoder)
        # 明确使用 stride=1，避免默认 stride=2 造成隐藏的降采样
        self.quant_conv = make_conv(2*ddconfig["z_channels"], 2*embed_dim, kernel_size=1, stride=1, padding=0, mode='3d')
        self.post_quant_conv = make_conv(embed_dim, ddconfig["z_channels"], kernel_size=1, stride=1, padding=0, mode='3d')
        
        self.embed_dim = embed_dim
        
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"Deleting key {k} from state_dict.")
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def encode_3d(self, x):
        h = self.encoder_3d(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        """
        解码: 将 embed_dim 空间的 latents 转换为 z_channels 空间，然后解码为图像
        
        Args:
            z: latents in embed_dim space, shape (B, embed_dim, T, H, W)
        Returns:
            decoded images, shape (B, out_ch, T, H, W)
        """
        z = self.post_quant_conv(z)  # embed_dim -> z_channels
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        """
        前向传播: Input -> Encoder -> Sample -> Decoder -> Recon
        
        正确的流程：
        1. encode_3d: 编码到 embed_dim 空间
        2. sample/mode: 从后验分布采样或取均值
        3. decode: 内部会做 post_quant_conv 转换到 z_channels 空间，然后解码
        """
        posterior = self.encode_3d(input)
        first_log = not getattr(self, "_logged_forward", False)
        if first_log:
            logging.getLogger("vae").info(f"[VAE] posterior mean shape: {posterior.mean.shape}")
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        if first_log:
            logging.getLogger("vae").info(f"[VAE] z sample shape: {z.shape}")
            self._logged_forward = True
        # post_quant_conv 应该在 decode 方法中统一处理，避免重复转换
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch):
        # 确保 Contiguous 内存布局
        return batch.to(memory_format=torch.contiguous_format)


def vae3d(**kwargs):
    return VAE3d(**kwargs)