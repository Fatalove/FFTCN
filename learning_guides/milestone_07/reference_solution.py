"""里程碑 7 参考答案。

本文件只用于学习。正式练习文件不会导入它。
前两个特征分支复用已经通过验收的里程碑 5、6 正式实现。
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
    """在序列左右补零的非因果一维卷积。"""

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
        # 父类卷积固定使用 padding=0，因为实际左右补零量要在 forward 中
        # 根据输入长度、stride、kernel_size 和 dilation 动态计算。
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
        """返回长度为 ceil(L_in / stride) 的非因果卷积结果。"""

        # 当前输入格式是 [B,C,L]，最后一维才是需要补零的序列长度。
        input_length = x.shape[-1]

        # nn.Conv1d 会把这些标量保存成单元素 tuple，所以取索引 0。
        stride = self.stride[0]
        kernel_size = self.kernel_size[0]
        dilation = self.dilation[0]

        # same padding 希望输出长度等于 ceil(L_in / stride)。
        output_length = math.ceil(input_length / stride)

        # 扩张卷积的有效卷积核长度：k_eff = dilation*(k-1)+1。
        effective_kernel = dilation * (kernel_size - 1) + 1

        # 由卷积输出公式反推总补零量。stride=1 时，本题可简化为
        # total_padding = dilation*(kernel_size-1)。
        total_padding = max(
            0,
            (output_length - 1) * stride + effective_kernel - input_length,
        )

        # 非因果卷积把补零分到左右两侧，因此输出 t 可以同时使用过去和未来。
        # 总量为奇数时，把多出的 1 放到右侧，与仓库实现一致。
        left_padding = total_padding // 2
        right_padding = total_padding - left_padding
        x = F.pad(x, (left_padding, right_padding))

        # 已经手工补零，所以 functional conv1d 的 padding 必须保持 0。
        return F.conv1d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            groups=self.groups,
        )


class TemporalResidualBlock(nn.Module):
    """两个同 dilation 的卷积与一条残差路径组成一个 TCN 块。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        # 第一次扩张卷积把 in_channels 映射为 out_channels。
        # weight_norm 将卷积权重重参数化为“方向 weight_v + 大小 weight_g”。
        self.conv1 = weight_norm(
            SamePadConv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation,
            )
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # 第二次扩张卷积保持 out_channels；同一块内两次卷积使用相同 dilation。
        self.conv2 = weight_norm(
            SamePadConv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation,
            )
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # 主路径的执行顺序与仓库一致：卷积 -> ReLU -> Dropout，重复两次。
        self.main_path = nn.Sequential(
            self.conv1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.relu2,
            self.dropout2,
        )

        # 残差相加要求通道数一致。第一块需要 472 -> 128 的 1x1 投影；
        # 后三块都是 128 -> 128，可以直接使用原输入。
        self.projection = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

        # 主路径与残差相加后再执行一次 ReLU。
        self.output_relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """保持序列长度，并返回 [B,out_channels,T]。"""

        # 主路径包含两次非因果 same-padding 扩张卷积，长度始终为 T。
        main = self.main_path(x)

        # 若通道不同，用 1x1 卷积调整残差通道；否则直接使用 x。
        residual = x if self.projection is None else self.projection(x)

        # 两条路径形状相同，逐元素相加后激活。
        return self.output_relu(main + residual)


class NonCausalTCN(nn.Module):
    """输入/输出通道后置、内部通道前置的四层非因果 TCN。"""

    def __init__(
        self,
        input_channels: int = 472,
        channels: tuple[int, ...] = (128, 128, 128, 128),
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        blocks: list[nn.Module] = []
        for level, out_channels in enumerate(channels):
            # 仓库使用 2**level，因此四个 dilation 实际为 1、2、4、8。
            dilation = 2**level

            # 第一块接收融合后的 472 通道；后续块接收前一块的 128 通道。
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

        # Conv1d 的卷积核只沿最后一轴滑动；模型要学习连续睡眠时段的关系，
        # 所以必须把时间轴 T 放到最后，把每个时段的 472 个特征放到通道轴。
        # [B,T,F=472] -> [B,F=472,T]
        x = x.permute(0, 2, 1)

        # 四个残差块沿 T 卷积、保持 T 不变，并把特征通道从 472 映射为 128。
        x = self.network(x)

        # TCN 已沿 T 融合前后时段信息；换回每个时段对应一个 128 维特征向量的表示。
        # [B,F=128,T] -> [B,T,F=128]
        return x.permute(0, 2, 1)


class FFTCNFusionTCN(nn.Module):
    """完整执行双分支特征提取、融合、TCN 和逐时段分类。"""

    def __init__(self, n_classes: int = 5) -> None:
        super().__init__()

        # 复用里程碑 5、6 已通过测试的特征网络。
        self.raw_feature_net = SleepEDFRawFeatureNet(n_classes=n_classes)
        self.wave_feature_net = SleepEDFWaveFeatureNet(n_classes=n_classes)

        # 融合阶段不使用两个分支各自的预训练分类头，分别取得 256/216 维特征。
        self.raw_feature_net.finetune()
        self.wave_feature_net.finetune()

        # 256 + 216 = 472，四层 TCN 再映射为 128 维上下文特征。
        self.tcn = NonCausalTCN(
            input_channels=472,
            channels=(128, 128, 128, 128),
            kernel_size=3,
            dropout=0.2,
        )

        # nn.Linear 作用在最后一维，所以可直接把 [B,T,128] 映射为 [B,T,5]。
        self.classifier = nn.Linear(128, n_classes)

    def forward(
        self,
        raw_epochs: torch.Tensor,
        wave_epochs: torch.Tensor,
    ) -> torch.Tensor:
        """返回 [B,T,5]，保留批次维和序列维。"""

        # 两种输入共享同一个 B 和 T。先记住它们，随后合并这两个维度，
        # 让此前只处理单时段的 CNN 一次处理 B*T 个时段。
        batch_size, sequence_length = raw_epochs.shape[:2]

        # [B,T,1,3000] -> [B*T,1,3000] -> [B*T,256]。
        raw_flat = raw_epochs.reshape(batch_size * sequence_length, *raw_epochs.shape[2:])
        raw_features = self.raw_feature_net(raw_flat)

        # 恢复序列结构：[B*T,256] -> [B,T,256]。
        raw_features = raw_features.reshape(batch_size, sequence_length, 256)

        # [B,T,1,30,60] -> [B*T,1,30,60] -> [B*T,216]。
        wave_flat = wave_epochs.reshape(
            batch_size * sequence_length,
            *wave_epochs.shape[2:],
        )
        wave_features = self.wave_feature_net(wave_flat)

        # 恢复序列结构：[B*T,216] -> [B,T,216]。
        wave_features = wave_features.reshape(batch_size, sequence_length, 216)

        # 沿最后的特征维拼接：[B,T,256] + [B,T,216] -> [B,T,472]。
        fused = torch.cat((raw_features, wave_features), dim=-1)

        # 非因果 TCN 融合前后时段上下文：[B,T,472] -> [B,T,128]。
        contextual_features = self.tcn(fused)

        # 对每个时段的 128 维表示使用同一个线性层，得到 [B,T,5]。
        return self.classifier(contextual_features)
