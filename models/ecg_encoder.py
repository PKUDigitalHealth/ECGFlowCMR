import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ECGEncoder(nn.Module):
    def __init__(self, in_channels=12, base_filters=64, out_dim=768, target_len=50):
        super().__init__()
        self.target_len = target_len # CMR 最终需要的帧数 (50)
        
        # --- 1. Backbone (Stem) ---
        # Stem 部分保持不变，提供 4 倍下采样
        # 5000 -> Conv(s=2) -> 2500 -> MaxPool(s=2) -> 1250
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, 7, stride=2, padding=3, bias=False), 
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1) 
        )
        
        # --- ResNet Layers ---
        # 目标总下采样: 8x
        # Stem 已经贡献了 4x，所以 Layers 只需要再贡献 2x
        
        # Layer 1: stride=1 (1250 -> 1250)
        self.layer1 = self._make_layer(base_filters, base_filters, 2, stride=1)
        
        # Layer 2: stride=2 (1250 -> 625) <--- 这里发生了剩下的 2x 下采样
        self.layer2 = self._make_layer(base_filters, base_filters*2, 2, stride=2) 
        
        # Layer 3: stride=1 (625 -> 625) <--- [修改点] 原来是 stride=2，现在改为 1
        self.layer3 = self._make_layer(base_filters*2, base_filters*4, 2, stride=1) 
        
        # Layer 4: stride=1 (625 -> 625)
        self.layer4 = self._make_layer(base_filters*4, base_filters*8, 2, stride=1) 
        
        # 此时 output length = 625 (即 5000 / 8)
        
        final_ch = base_filters * 8
        
        # --- 2. Heads ---
        self.feature_proj = nn.Conv1d(final_ch, out_dim, 1)
        
        # Phase Head
        self.phase_head = nn.Sequential(
            nn.Conv1d(final_ch, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 2, 1) 
        )

    def _make_layer(self, in_ch, out_ch, blocks, stride=1):
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        layers = []
        layers.append(ResBlock1D(in_ch, out_ch, stride=stride, downsample=downsample))
        for _ in range(1, blocks):
            layers.append(ResBlock1D(out_ch, out_ch))
        return nn.Sequential(*layers)

    def detect_best_cycle(self, phase_sin, phase_cos):
        # 逻辑不变，但输入的 phase_sin/cos 长度现在是 625
        phase = torch.atan2(phase_sin, phase_cos) 
        diff = phase[:, 1:] - phase[:, :-1]
        
        batch_indices = []
        B, L = phase.shape
        
        for b in range(B):
            jumps = torch.where(diff[b] < -3.0)[0]
            
            if len(jumps) < 2:
                start = L // 3
                end = 2 * L // 3
            else:
                mid_idx = len(jumps) // 2
                start = jumps[mid_idx-1] + 1 if mid_idx > 0 else 0
                end = jumps[mid_idx] + 1
                
                if end - start < 10: 
                    start, end = 0, L 
            # 确保返回 Python int，避免 .item() 对纯 int 报错
            start = int(start)
            end = int(end)
            batch_indices.append((start, end))
            
        return batch_indices

    def roi_align(self, feats, phase, indices):
        # 逻辑不变，F.interpolate 会自动处理从变长到固定长度(50)的转换
        B, C, L = feats.shape
        aligned_feats = []
        aligned_phase = []
        
        for b in range(B):
            start, end = indices[b]
            
            f_crop = feats[b:b+1, :, start:end] 
            p_crop = phase[b:b+1, :, start:end] 
            
            f_resized = F.interpolate(f_crop, size=self.target_len, mode='linear', align_corners=False)
            p_resized = F.interpolate(p_crop, size=self.target_len, mode='linear', align_corners=False)
            
            aligned_feats.append(f_resized)
            aligned_phase.append(p_resized)
            
        return torch.cat(aligned_feats, dim=0), torch.cat(aligned_phase, dim=0)

    def forward(self, x, force_indices=None):
        # 1. Backbone Extraction
        # x: (B, 12, 5000)
        feats_long = self.stem(x)         # -> (B, 64, 1250)
        feats_long = self.layer1(feats_long) # -> (B, 64, 1250)
        feats_long = self.layer2(feats_long) # -> (B, 128, 625)
        feats_long = self.layer3(feats_long) # -> (B, 256, 625) [长度保持]
        feats_long = self.layer4(feats_long) # -> (B, 512, 625)
        
        # 2. Heads
        sem_long = self.feature_proj(feats_long) # (B, 768, 625)
        phase_long = self.phase_head(feats_long) # (B, 2, 625)
        
        if force_indices == "return_raw":
            return sem_long, phase_long

        # 3. Cycle Detection (在长度为 625 的序列上找 R 波)
        if force_indices is None:
            with torch.no_grad(): 
                indices = self.detect_best_cycle(phase_long[:, 0], phase_long[:, 1])
        else:
            indices = force_indices
            
        # 4. ROI Align
        # 从 625 长度中切出一小段，插值到 50
        sem_out, phase_out = self.roi_align(sem_long, phase_long, indices)
        
        phase_angle = torch.atan2(phase_out[:, 0, :], phase_out[:, 1, :])
        
        return sem_out.transpose(1, 2), phase_angle

class ECGDecoder(nn.Module):
    def __init__(self, in_dim=768, out_channels=12, base_filters=64):
        super().__init__()
        
        # Encoder 是 8x 下采样，Decoder 需要 8x 上采样
        # Input: (B, 768, 625)
        
        # 1. 投影层
        self.proj = nn.Conv1d(in_dim, base_filters * 8, 1)
        
        # 2. 上采样层 (Upsampling Layers)
        # 625 -> 1250 (2x)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(base_filters * 8, base_filters * 4, 3, padding=1),
            nn.BatchNorm1d(base_filters * 4),
            nn.ReLU(inplace=True)
        )
        
        # 1250 -> 2500 (2x)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(base_filters * 4, base_filters * 2, 3, padding=1),
            nn.BatchNorm1d(base_filters * 2),
            nn.ReLU(inplace=True)
        )
        
        # 2500 -> 5000 (2x)
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(base_filters * 2, base_filters, 3, padding=1),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True)
        )
        
        # 3. 输出层 (Refinement)
        self.final_conv = nn.Conv1d(base_filters, out_channels, kernel_size=7, padding=3)

    def forward(self, x):
        x = self.proj(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.final_conv(x)
        return x

def random_masking(x, mask_ratio):
    """
    对时序特征进行随机 Mask。
    x: (B, C, L)
    mask_ratio: float, 例如 0.5 表示遮挡 50%
    """
    if mask_ratio <= 0:
        return x, None
    
    B, C, L = x.shape
    # 生成随机掩码矩阵 (B, 1, L)
    # keep_prob = 1 - mask_ratio
    mask = torch.rand(B, 1, L, device=x.device) > mask_ratio
    
    # 将 mask 广播到所有通道
    # 被 mask 的地方置为 0 (或者可以用一个可学习的 mask token 替换，这里用 0 简化)
    x_masked = x * mask.float()
    
    return x_masked, mask