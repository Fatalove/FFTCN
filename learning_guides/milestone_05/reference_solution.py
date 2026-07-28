"""里程碑 5 参考答案。

本文件只用于学习，不会被正式训练代码或用户练习文件导入。
"""

from __future__ import annotations

import torch
from torch import nn


class Conv1dBlock(nn.Module):
    """按照 Conv1d -> BatchNorm1d -> ReLU 组织一个基础卷积块。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()

        # 卷积负责提取局部时域模式，并按 stride 改变序列长度。
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        # BatchNorm1d 对每个输出通道做批归一化，通道数必须等于 out_channels。
        self.batch_norm = nn.BatchNorm1d(out_channels)

        # ReLU 给连续卷积层引入非线性；它不改变张量形状。
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回形状为 [B, out_channels, L_out] 的特征图。"""

        # 先做卷积：通道数和时间长度可能改变。
        x = self.conv(x)

        # 再做批归一化：形状保持不变。
        x = self.batch_norm(x)

        # 最后做 ReLU：形状仍保持不变。
        return self.activation(x)


class SleepEDFRawFeatureNet(nn.Module):
    """把 [B,1,3000] EEG 转成 [B,256] 特征或 [B,5] logits。"""

    def __init__(self, n_classes: int = 5) -> None:
        super().__init__()

        # features 完全对齐仓库 models/raw/parameter.py 的 1D-CNN 参数。
        # 长度依次为：3000 -> 119 -> 14 -> 13 -> 12 -> 11 -> 2。
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

        # 预训练头把 [B,128,2] 变成 [B,n_classes,1]。
        # AdaptiveAvgPool1d(1) 对当前仓库参数看似多余，但保留它可对齐原结构。
        self.classifier = nn.Sequential(
            Conv1dBlock(128, n_classes, kernel_size=2, stride=1, padding=0),
            nn.AdaptiveAvgPool1d(1),
        )

        # 默认与原仓库一致：新建网络时先处于预训练输出模式。
        self.classifier_enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """根据模式返回分类 logits 或时域特征。"""

        # 主干输出 [B,128,2]，其中 128*2=256。
        x = self.features(x)

        # 预训练时经过分类头，输出暂为 [B,5,1]；微调时跳过它。
        if self.classifier_enabled:
            x = self.classifier(x)

        # 从通道维开始展平：[B,5,1] -> [B,5]，或 [B,128,2] -> [B,256]。
        return torch.flatten(x, start_dim=1)

    def pretrain(self) -> None:
        """启用五分类头，但不改变 Dropout/BatchNorm 的训练状态。"""

        self.classifier_enabled = True

    def finetune(self) -> None:
        """关闭分类头，让后续融合网络取得 256 维时域特征。"""

        self.classifier_enabled = False
