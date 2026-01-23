"""
Matches the pretrained weight file i3d_pretrained_400.pt.
"""

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_padding_shape(
    in_size: int,
    kernel_size: int,
    stride: int,
) -> Tuple[int, int]:
    """Compute front/back padding to mimic TensorFlow 'SAME' padding."""
    out_size = math.ceil(float(in_size) / float(stride))
    pad_needed = max(0, (out_size - 1) * stride + kernel_size - in_size)
    pad_front = pad_needed // 2
    pad_back = pad_needed - pad_front
    return pad_front, pad_back


def _pad_3d(
    x: torch.Tensor,
    kernel_size: Sequence[int],
    stride: Sequence[int],
) -> torch.Tensor:
    """Apply dynamic SAME padding on 3D input (T, H, W)."""
    (pad_t_front, pad_t_back) = _get_padding_shape(x.shape[2], kernel_size[0], stride[0])
    (pad_h_front, pad_h_back) = _get_padding_shape(x.shape[3], kernel_size[1], stride[1])
    (pad_w_front, pad_w_back) = _get_padding_shape(x.shape[4], kernel_size[2], stride[2])
    # F.pad expects reverse order: (w_front, w_back, h_front, h_back, t_front, t_back)
    return F.pad(
        x,
        [pad_w_front, pad_w_back, pad_h_front, pad_h_back, pad_t_front, pad_t_back],
    )


class MaxPool3dSamePadding(nn.MaxPool3d):
    """3D max pool with dynamic SAME padding."""

    def __init__(self, kernel_size, stride=(1, 1, 1)):
        super().__init__(kernel_size=kernel_size, stride=stride, padding=0, ceil_mode=False)
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        self.stride = stride if isinstance(stride, tuple) else (stride,) * 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _pad_3d(x, self.kernel_size, self.stride)
        return super().forward(x)


class Unit3Dpy(nn.Module):
    """Basic 3D convolution unit with optional batch norm and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int] = (1, 1, 1),
        stride: Tuple[int, int, int] = (1, 1, 1),
        activation_fn=F.relu,
        use_batch_norm: bool = True,
        use_bias: bool = False,
        name: str = 'unit_3d',
    ):
        super().__init__()
        self._name = name
        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=use_bias,
        )
        self.bn = nn.BatchNorm3d(out_channels, eps=1e-3, momentum=0.001) if use_batch_norm else None
        self.activation_fn = activation_fn
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _pad_3d(x, self.kernel_size, self.stride)
        x = self.conv3d(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        return x


class Mixed(nn.Module):
    """Inception block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: Tuple[int, int, int, int, int, int],
    ):
        super().__init__()
        b0, b1a, b1b, b2a, b2b, b3 = out_channels

        self.branch0 = Unit3Dpy(in_channels, b0, kernel_size=(1, 1, 1), name='branch0')

        self.branch1a = Unit3Dpy(in_channels, b1a, kernel_size=(1, 1, 1), name='branch1a')
        self.branch1b = Unit3Dpy(b1a, b1b, kernel_size=(3, 3, 3), name='branch1b')

        self.branch2a = Unit3Dpy(in_channels, b2a, kernel_size=(1, 1, 1), name='branch2a')
        self.branch2b = Unit3Dpy(b2a, b2b, kernel_size=(3, 3, 3), name='branch2b')

        self.branch3a = MaxPool3dSamePadding(kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.branch3b = Unit3Dpy(in_channels, b3, kernel_size=(1, 1, 1), name='branch3b')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b0 = self.branch0(x)
        b1 = self.branch1b(self.branch1a(x))
        b2 = self.branch2b(self.branch2a(x))
        b3 = self.branch3b(self.branch3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


class InceptionI3d(nn.Module):
    """
    Inception I3D network. Returns logits by default.
    """

    def __init__(
        self,
        num_classes: int = 400,
        spatial_squeeze: bool = True,
        in_channels: int = 3,
        dropout_keep_prob: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.spatial_squeeze = spatial_squeeze
        self.dropout_keep_prob = dropout_keep_prob

        self.end_points = nn.ModuleDict()

        self.end_points['Conv3d_1a_7x7'] = Unit3Dpy(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=(7, 7, 7),
            stride=(2, 2, 2),
            name='Conv3d_1a_7x7',
        )
        self.end_points['MaxPool3d_2a_3x3'] = MaxPool3dSamePadding(
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
        )
        self.end_points['Conv3d_2b_1x1'] = Unit3Dpy(
            in_channels=64,
            out_channels=64,
            kernel_size=(1, 1, 1),
            name='Conv3d_2b_1x1',
        )
        self.end_points['Conv3d_2c_3x3'] = Unit3Dpy(
            in_channels=64,
            out_channels=192,
            kernel_size=(3, 3, 3),
            name='Conv3d_2c_3x3',
        )
        self.end_points['MaxPool3d_3a_3x3'] = MaxPool3dSamePadding(
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
        )
        self.end_points['Mixed_3b'] = Mixed(192, (64, 96, 128, 16, 32, 32))
        self.end_points['Mixed_3c'] = Mixed(256, (128, 128, 192, 32, 96, 64))
        self.end_points['MaxPool3d_4a_3x3'] = MaxPool3dSamePadding(
            kernel_size=(3, 3, 3),
            stride=(2, 2, 2),
        )
        self.end_points['Mixed_4b'] = Mixed(480, (192, 96, 208, 16, 48, 64))
        self.end_points['Mixed_4c'] = Mixed(512, (160, 112, 224, 24, 64, 64))
        self.end_points['Mixed_4d'] = Mixed(512, (128, 128, 256, 24, 64, 64))
        self.end_points['Mixed_4e'] = Mixed(512, (112, 144, 288, 32, 64, 64))
        self.end_points['Mixed_4f'] = Mixed(528, (256, 160, 320, 32, 128, 128))
        self.end_points['MaxPool3d_5a_2x2'] = MaxPool3dSamePadding(
            kernel_size=(2, 2, 2),
            stride=(2, 2, 2),
        )
        self.end_points['Mixed_5b'] = Mixed(832, (256, 160, 320, 32, 128, 128))
        self.end_points['Mixed_5c'] = Mixed(832, (384, 192, 384, 48, 128, 128))

        self.build()

        # Final pooling and logits
        self.avg_pool = nn.AvgPool3d(kernel_size=(2, 7, 7), stride=(1, 1, 1), padding=0)
        self.dropout = nn.Dropout(1 - self.dropout_keep_prob)
        self.logits = Unit3Dpy(
            in_channels=1024,
            out_channels=self.num_classes,
            kernel_size=(1, 1, 1),
            activation_fn=None,
            use_bias=True,
            name='logits',
        )
        self.softmax = nn.Softmax(dim=1)

    def build(self):
        """Utility to register end_points modules in order."""
        for name, module in self.end_points.items():
            self.add_module(name, module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass returning Logits (for classification)."""
        for name, module in self.end_points.items():
            x = module(x)
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = self.logits(x)
        if self.spatial_squeeze:
            x = x.squeeze(3).squeeze(3)
        x = x.mean(2)  # temporal average
        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extraction method for FVD. Returns 1024-dim features.
        """
        for name, module in self.end_points.items():
            x = module(x)
        
        # Post-conv AvgPool
        x = self.avg_pool(x)  # Shape: (B, 1024, T_down, 1, 1)
        
        # Squeeze spatial dimensions
        x = x.squeeze(3).squeeze(3)
        
        # Temporal Average Pooling (Standard FVD approach)
        x = x.mean(2)  # Shape: (B, 1024)
        
        return x