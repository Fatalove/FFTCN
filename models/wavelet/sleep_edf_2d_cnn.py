"""里程碑 6 练习：实现 Sleep-EDF 的 2D-CNN 时频特征分支。

这个文件有意保留为用户练习骨架。带完整中文注释的独立参考实现位于：

    learning_guides/milestone_06/reference_solution.py

正式训练代码不会导入参考答案。

输入输出契约：
    输入：float32 CWT 批次，形状为 [B, 1, 30, 60]
          30 是尺度/近似频率轴，60 是压缩后的时间轴
    特征模式：输出 [B, 216]
    预训练模式：输出五分类 logits [B, 5]
"""

from __future__ import annotations

import torch
from torch import nn


class Conv2dBlock(nn.Module):
    """二维卷积、批归一化和 ReLU 组成的基础块。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """按 Conv2d -> BatchNorm2d -> ReLU 的顺序处理输入。"""

        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x


class SleepEDFWaveFeatureNet(nn.Module):
    """仓库版 FFTCN 的 2D-CNN 时频特征网络练习实现。"""

    def __init__(self, n_classes: int = 5) -> None:
        super().__init__()

        "用户练习：创建四个卷积池化块、预训练分类头和输出模式标志"
        self.features = nn.Sequential(
            Conv2dBlock(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            Conv2dBlock(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            Conv2dBlock(
                in_channels=32,
                out_channels=48,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            Conv2dBlock(
                in_channels=48,
                out_channels=48,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            Conv2dBlock(
                in_channels=48,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            Conv2dBlock(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            Conv2dBlock(
                in_channels=64,
                out_channels=72,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            Conv2dBlock(
                in_channels=72,
                out_channels=72,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )


        self.classifier=nn.Sequential(
            Conv2dBlock(
                in_channels=72,
                out_channels=n_classes,
                kernel_size=1,
                stride=1,
                padding=0,

            ),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier_enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """根据当前模式返回 [B,216] 特征或 [B,5] logits。"""

        x = self.features(x)
        if self.classifier_enabled:
            x = self.classifier(x)

        return torch.flatten(x, start_dim=1)

    def pretrain(self) -> None:
        """启用五分类头；不要在这里调用 ``train()``。"""
        self.classifier_enabled = True

    def finetune(self) -> None:
        """关闭五分类头，直接返回 216 维时频特征。"""
        self.classifier_enabled = False
