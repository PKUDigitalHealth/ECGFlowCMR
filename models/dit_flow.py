import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange

# ==========================================
# 1. 基础组件
# ==========================================
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)

# ==========================================
# 2. 核心组件：全功能 DiT Block
# ==========================================
class DiTBlock(nn.Module):
    """
    改进后的 DiT Block，包含：
    1. Spatial Self-Attention (独立权重)
    2. Temporal Self-Attention (独立权重)
    3. Cross-Attention (ECG Condition)
    4. MLP
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        
        # 1. Spatial Self-Attention
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn_spatial = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        
        # 2. Temporal Self-Attention
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn_temporal = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        
        # 3. Cross-Attention (Latent query ECG)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn_cross = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        
        # 4. MLP
        self.norm4 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size)
        )
        
        # adaLN modulation
        # 我们有4个子模块，每个模块需要 (shift, scale, gate) 3个参数
        # 总共 4 * 3 * hidden_size = 12 * hidden_size
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 12 * hidden_size, bias=True)
        )

    def forward(self, x, c, ecg_feat, T, H, W):
        # x input: (B, T, L, D) -> handled internally as needed
        # c: (B, D) global condition
        # ecg_feat: (B, T, D) sequence condition
        
        B, _, L, D = x.shape
        
        # 生成调制参数 (B, 12, D) -> chunk -> 12个 (B, D)
        chunks = self.adaLN_modulation(c).chunk(12, dim=1)
        (shift_s, scale_s, gate_s,   # Spatial
         shift_t, scale_t, gate_t,   # Temporal
         shift_c, scale_c, gate_c,   # Cross
         shift_m, scale_m, gate_m) = chunks # MLP
         
        # -------------------------------------------------
        # 1. Temporal Self-Attention
        # Shape: (B*L, T, D)
        # -------------------------------------------------
        x_temporal = rearrange(x, 'b t l d -> (b l) t d')
        
        shift_t_exp = shift_t.repeat_interleave(L, dim=0).unsqueeze(1)
        scale_t_exp = scale_t.repeat_interleave(L, dim=0).unsqueeze(1)
        gate_t_exp  = gate_t.repeat_interleave(L, dim=0).unsqueeze(1)
        
        x_norm = self.norm2(x_temporal) * (1 + scale_t_exp) + shift_t_exp
        attn_out, _ = self.attn_temporal(x_norm, x_norm, x_norm)
        x_temporal = x_temporal + gate_t_exp * attn_out
        
        # -------------------------------------------------
        # 2. Cross-Attention (Condition Injection)
        # Query: Video Latent (Temporal view) -> (B*L, T, D)
        # Key/Value: ECG Features -> Need to repeat for each spatial location -> (B*L, T, D)
        # -------------------------------------------------
        # 我们在时间维度做 Cross Attention，这样每个像素的时间变化都能参考 ECG
        
        shift_c_exp = shift_c.repeat_interleave(L, dim=0).unsqueeze(1)
        scale_c_exp = scale_c.repeat_interleave(L, dim=0).unsqueeze(1)
        gate_c_exp  = gate_c.repeat_interleave(L, dim=0).unsqueeze(1)
        
        # 准备 KV: ECG feat (B, T, D) -> (B*L, T, D)
        ecg_kv = ecg_feat.repeat_interleave(L, dim=0)
        
        x_norm = self.norm3(x_temporal) * (1 + scale_c_exp) + shift_c_exp
        # Query=x, Key=ECG, Value=ECG
        attn_out, _ = self.attn_cross(x_norm, ecg_kv, ecg_kv)
        x_temporal = x_temporal + gate_c_exp * attn_out
        
        # Restore shape for Spatial Attention
        x = rearrange(x_temporal, '(b l) t d -> b t l d', b=B, l=L)

        # -------------------------------------------------
        # 3. Spatial Self-Attention
        # Shape: (B*T, L, D)
        # -------------------------------------------------
        # 将 Spatial 放在最后，以修复由 Cross/Temporal 引入的空间不一致性，提升 SSIM
        x_spatial = rearrange(x, 'b t l d -> (b t) l d')
        
        # AdaLN Mod
        # expand parameters to match (B*T, 1, D)
        shift_s_exp = shift_s.repeat_interleave(T, dim=0).unsqueeze(1)
        scale_s_exp = scale_s.repeat_interleave(T, dim=0).unsqueeze(1)
        gate_s_exp  = gate_s.repeat_interleave(T, dim=0).unsqueeze(1)
        
        x_norm = self.norm1(x_spatial) * (1 + scale_s_exp) + shift_s_exp
        attn_out, _ = self.attn_spatial(x_norm, x_norm, x_norm)
        x_spatial = x_spatial + gate_s_exp * attn_out
        
        x = rearrange(x_spatial, '(b t) l d -> b t l d', b=B, t=T)
        
        # -------------------------------------------------
        # 4. MLP
        # -------------------------------------------------
        shift_m = shift_m.unsqueeze(1).unsqueeze(1) # (B, 1, 1, D)
        scale_m = scale_m.unsqueeze(1).unsqueeze(1)
        gate_m  = gate_m.unsqueeze(1).unsqueeze(1)
        
        x_norm = self.norm4(x) * (1 + scale_m) + shift_m
        mlp_out = self.mlp(x_norm)
        x = x + gate_m * mlp_out
        
        return x

# ==========================================
# 3. 主模型：优化后的 ECG 条件 DiT
# ==========================================
class ECG_DiT(nn.Module):
    def __init__(self, 
                 input_size=12,    
                 num_frames=50,    
                 in_channels=4,    
                 hidden_size=512,  # 默认 512，与 args 一致
                 depth=8, 
                 num_heads=8,
                 ecg_dim=768,
                 train_num_points=4):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.input_size = input_size
        self.num_frames = num_frames
        self.train_num_points = train_num_points
        self.hidden_size = hidden_size
        
        # 1. Patch Embedding
        self.x_embedder = nn.Linear(in_channels, hidden_size)
        
        # 2. 条件嵌入
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        # ECG 投影 (用于 Cross Attn)
        self.ecg_proj = nn.Linear(ecg_dim, hidden_size) if ecg_dim != hidden_size else nn.Identity()
        
        # Global ECG 投影 (用于 AdaLN)
        self.ecg_global_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frames, input_size * input_size, hidden_size))
        
        # 3. Transformer 块
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])
        
        # 4. 最终层
        self.final_adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_linear = nn.Linear(hidden_size, in_channels, bias=True)
        
        self.initialize_weights()

    def initialize_weights(self):
        # 初始化位置嵌入
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # 初始化嵌入层
        nn.init.normal_(self.x_embedder.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # 零初始化所有 adaLN 调制的最后一层 -> 使得初始状态下 block 近似恒等映射
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_adaLN_modulation[-1].bias, 0)

    def predict_velocity(self, x, t, ecg_feat):
        """
        x: (B, 4, 50, 12, 12)
        t: (B,)
        ecg_feat: (B, 50, 768)
        """
        B, C, T, H, W = x.shape
        L = H * W
        
        # --- 1. Embedding ---
        x = rearrange(x, 'b c t h w -> b t (h w) c')
        x = self.x_embedder(x) # (B, T, L, D)
        x = x + self.pos_embed
        
        # --- 2. Conditions ---
        # A. Time Condition
        t_emb = self.t_embedder(t) # (B, D)
        
        # B. ECG Condition Processing
        ecg_feat_proj = self.ecg_proj(ecg_feat) # (B, 50, D)
        
        # C. Fuse Global Conditions (Time + ECG Global)
        # 相比之前的 +0.3 硬编码，这里使用可学习的投影
        ecg_global = ecg_feat_proj.mean(dim=1) # (B, D)
        ecg_global_emb = self.ecg_global_proj(ecg_global)
        
        c = t_emb + ecg_global_emb # (B, D) 融合条件

        # --- 3. Blocks ---
        for block in self.blocks:
            x = block(x, c, ecg_feat_proj, T, H, W)
            
        # --- 4. Final Head ---
        shift, scale = self.final_adaLN_modulation(c).chunk(2, dim=1)
        x = self.final_norm(x)
        x = x * (1 + scale[:, None, None, :]) + shift[:, None, None, :]
        x = self.final_linear(x)
        
        x = rearrange(x, 'b t (h w) c -> b c t h w', h=H, w=W)
        return x

    def forward(self, target, ecg_feat, x0=None, train_num_points=None):
        """
        Rectified Flow 训练
        """
        b = target.shape[0]
        device = target.device
        num_points = train_num_points or self.train_num_points
        total_loss = 0.0
        
        for _ in range(num_points):
            t = torch.rand((b,), device=device)
            
            if x0 is None:
                x0_sample = torch.randn_like(target)
            else:
                x0_sample = x0
            
            t_reshaped = t.view(b, 1, 1, 1, 1)
            # Rectified Flow: 直线路径插值
            xt = (1.0 - t_reshaped) * x0_sample + t_reshaped * target
            drift = target - x0_sample
            
            # 预测 Vector Field
            pred = self.predict_velocity(xt, t, ecg_feat)
            
            # MSE Loss
            loss_step = (pred - drift) ** 2
            total_loss = total_loss + loss_step.mean()
        
        return total_loss / num_points

    @torch.no_grad()
    def sample(self, shape, num_steps=10, ecg_feat=None, x0=None):
        device = next(self.parameters()).device
        dt = 1.0 / num_steps
        
        if x0 is None:
            x = torch.randn(shape, device=device)
        else:
            x = x0
        
        timesteps = torch.linspace(0, 1, num_steps + 1, device=device)[:-1]
        for t_val in timesteps:
            t_batch = torch.full((shape[0],), t_val, device=device)
            v = self.predict_velocity(x, t_batch, ecg_feat)
            x = x + dt * v
        return x

class PhysicsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, v_pred, ecg_raw):
        v_energy = torch.mean(v_pred ** 2, dim=[1, 3, 4]) 
        ecg_diff = torch.abs(ecg_raw[:, :, 1:] - ecg_raw[:, :, :-1])
        ecg_energy = torch.mean(ecg_diff, dim=1)
        
        ecg_energy = ecg_energy.unsqueeze(1)
        ecg_energy_ds = F.interpolate(ecg_energy, size=50, mode='linear', align_corners=False).squeeze(1)
        
        v_norm = (v_energy - v_energy.mean(1, keepdim=True)) / (v_energy.std(1, keepdim=True) + 1e-6)
        e_norm = (ecg_energy_ds - ecg_energy_ds.mean(1, keepdim=True)) / (ecg_energy_ds.std(1, keepdim=True) + 1e-6)
        
        return F.mse_loss(v_norm, e_norm)

