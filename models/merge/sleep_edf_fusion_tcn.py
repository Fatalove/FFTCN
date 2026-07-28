"""里程碑 7 练习：实现 FFTCN 的特征融合与非因果 TCN。

这个文件有意保留为用户练习骨架。带完整中文注释的独立参考实现位于：

    learning_guides/milestone_07/reference_solution.py

正式训练代码不会导入参考答案。

输入输出契约：
    raw_epochs:  [B, T, 1, 3000]
    wave_epochs: [B, T, 1, 30, 60]
    融合特征：   [B, T, 472]，其中 472 = 256 + 216
    TCN 输出：    [B, T, 128]
    logits:      [B, T, 5]
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm

from models.raw.sleep_edf_1d_cnn import SleepEDFRawFeatureNet
from models.wavelet.sleep_edf_2d_cnn import SleepEDFWaveFeatureNet


class SamePadConv1d(nn.Conv1d):
    """非因果的一维 same-padding 卷积。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对序列两侧补零后执行卷积，使输出长度为 ceil(L/stride)。"""
        "用户练习：完成非因果 same padding 卷积"
        input_len = x.shape[-1]
        stride = self.stride[0]
        kernel_size = self.kernel_size[0]
        dilation = self.dilation[0]

        output_len = math.ceil(input_len / stride)

        effective_kernel = 1 + (kernel_size - 1) * dilation

        total_padding = max(
            0,
            (output_len - 1) * stride + effective_kernel - input_len
        )

        left_padding = total_padding // 2
        right_padding = total_padding - left_padding
        x = F.pad(x, [left_padding, right_padding])

        return F.conv1d(
            x,
            self.weight,
            self.bias,
            self.stride,
            padding=0,
            dilation=self.dilation,
            groups=self.groups
        )


class TemporalResidualBlock(nn.Module):
    """两个同 dilation 卷积组成的 TCN 残差块。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        "用户练习：创建两次 weight-normalized 卷积、ReLU、Dropout 和残差投影"
        self.conv1 = weight_norm(
            SamePadConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation,
            )
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            SamePadConv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation,
            )
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.main_path = nn.Sequential(
            self.conv1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.relu2,
            self.dropout2,
        )

        self.projection = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if out_channels != in_channels
            else None
        )

        self.output_relu = nn.ReLU()



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回与输入序列长度相同的残差块输出。"""

        "用户练习：完成主路径与残差路径相加"
        main = self.main_path(x)
        residual = x if self.projection is None else self.projection(x)

        residual = self.output_relu(main + residual)

        return residual


class NonCausalTCN(nn.Module):
    """dilation 为 1、2、4、8 的四层非因果 TCN。"""

    def __init__(
        self,
        input_channels: int = 472,
        channels: tuple[int, ...] = (128, 128, 128, 128),
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        "用户练习：按 2**level 创建四个 TemporalResidualBlock"
        blocks: list[nn.Module] = []
        for level, out_channels in enumerate(channels):
            dilation = 2 ** level

            in_channels = input_channels if level == 0 else channels[level - 1]
            blocks.append(
                TemporalResidualBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )

        self.network = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """沿睡眠时间轴 T 建模，将 [B,T,F=472] 转成 [B,T,F=128]。"""

        "用户练习：完成两次维度交换和 TCN 前向传播"
        # Conv1d 的卷积核只沿最后一轴滑动；模型要学习连续睡眠时段的关系，
        # 所以必须把时间轴 T 放到最后，把每个时段的 472 个特征放到通道轴。
        # [B,T,F=472] -> [B,F=472,T]
        x = x.permute(0, 2, 1)
        x = self.network(x)
        # TCN 已沿 T 融合前后时段信息；换回每个时段对应一个 128 维特征向量的表示。
        # [B,F=128,T] -> [B,T,F=128]
        return x.permute(0, 2, 1)


class FFTCNFusionTCN(nn.Module):
    """融合 1D/2D 分支特征，并用非因果 TCN 输出逐时段 logits。"""

    def __init__(self, n_classes: int = 5) -> None:
        super().__init__()

        "用户练习：创建两个特征分支、472->128 TCN 和 128->5 分类头"
        self.raw_feature_net = SleepEDFRawFeatureNet(n_classes=n_classes)
        self.wave_feature_net = SleepEDFWaveFeatureNet(n_classes=n_classes)

        self.raw_feature_net.finetune()
        self.wave_feature_net.finetune()

        self.tcn = NonCausalTCN(
            input_channels=472,
            channels=(128, 128, 128, 128),
            kernel_size=3,
            dropout=0.2,
        )

        self.classifier = nn.Linear(128, n_classes)

    def forward(
        self,
        raw_epochs: torch.Tensor,
        wave_epochs: torch.Tensor,
    ) -> torch.Tensor:
        """返回形状为 [B,T,5] 的逐时段分类 logits。"""

        "用户练习：展平 B/T、提取并恢复特征、拼接、TCN 和分类"
        batch_size, sequence_length = raw_epochs.shape[:2]

        raw_flat = raw_epochs.reshape(batch_size * sequence_length,
                                      *raw_epochs.shape[2:])
        raw_features = self.raw_feature_net(raw_flat)
        raw_features = raw_features.reshape(batch_size, sequence_length, 256)

        wave_flat = wave_epochs.reshape(batch_size * sequence_length,
                                      *wave_epochs.shape[2:])
        wave_features = self.wave_feature_net(wave_flat)
        wave_features = wave_features.reshape(batch_size, sequence_length, 216)

        features = torch.cat((raw_features, wave_features), dim=-1)

        contextual_features = self.tcn(features)
        logits = self.classifier(contextual_features)

        return logits
