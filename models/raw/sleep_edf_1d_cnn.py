"""里程碑 5 练习：实现 Sleep-EDF 的 1D-CNN 时域特征分支。

这个文件有意保留为用户练习骨架。带完整中文注释的独立参考实现位于：

    learning_guides/milestone_05/reference_solution.py

正式训练代码不会导入参考答案。

输入输出契约：
    输入：float32 EEG 批次，形状为 [B, 1, 3000]
    特征模式：输出 [B, 256]
    预训练模式：输出五分类 logits [B, 5]
"""

from __future__ import annotations

import torch
from torch import nn


class Conv1dBlock(nn.Module):
    """一维卷积、批归一化和 ReLU 组成的基础块。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()

        """raise NotImplementedError(
            "用户练习：依次创建 Conv1d、BatchNorm1d 和 ReLU 子模块"
        )"""
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """按 Conv1d -> BatchNorm1d -> ReLU 的顺序处理输入。"""

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class SleepEDFRawFeatureNet(nn.Module):
    """仓库版 FFTCN 的 1D-CNN 时域特征网络练习实现。"""

    def __init__(self, n_classes: int = 5) -> None:
        super().__init__()

        """raise NotImplementedError(
            "用户练习：创建时域特征提取器、预训练分类头和模式标志"
        )"""

        self.features = nn.Sequential(
            Conv1dBlock(1, 128, kernel_size=50, stride=25, padding=0),
            nn.MaxPool1d(kernel_size=8, stride=8, padding=0),
            nn.Dropout(p=0.5),
            Conv1dBlock(128, 128, kernel_size=8, stride=1, padding=3),
            Conv1dBlock(128, 128, kernel_size=8, stride=1, padding=3),
            Conv1dBlock(128, 128, kernel_size=8, stride=1, padding=3),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=0),
            nn.Dropout(p=0.5),


        )

        self.classifier = nn.Sequential(
            Conv1dBlock(128, n_classes, kernel_size=2, stride=1, padding=0),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier_enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """根据当前模式返回 [B, 256] 特征或 [B, 5] logits。"""

        x = self.features(x)

        if self.classifier_enabled:
            x = self.classifier(x)

        return torch.flatten(x, start_dim=1)


    def pretrain(self) -> None:
        """启用五分类头；不要在这里调用 ``train()``。"""

        self.classifier_enabled = True

    def finetune(self) -> None:
        """关闭五分类头，直接返回 256 维时域特征。"""

        self.classifier_enabled = False
