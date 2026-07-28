"""里程碑 6 参考答案。

本文件只用于学习，不会被正式训练代码或用户练习文件导入。
"""

from __future__ import annotations

import torch
from torch import nn


class Conv2dBlock(nn.Module):
    """按照 Conv2d -> BatchNorm2d -> ReLU 组织一个基础卷积块。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
    ) -> None:
        # 先初始化 nn.Module。只有完成这一步，下面赋给 self 的层才会被
        # PyTorch 正确登记，才能参与参数更新、保存和设备迁移。
        super().__init__()

        # Conv2d 接收 [B, in_channels, H, W]。
        # 本题 H 是 30 个 Morlet 尺度，W 是 60 个压缩时间格。
        # kernel_size 决定一次观察多大的局部区域；stride 决定滑动步长；
        # padding 决定是否在边缘补零。主干使用 k=3、s=1、p=1，
        # 因此 H/W 不变，只把通道数从 in_channels 改为 out_channels。
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        # BatchNorm2d 为每个输出通道维护独立的缩放、偏移和运行统计量，
        # 所以这里必须传 out_channels，而不是 H 或 W。
        # 它只调整数值分布，不改变 [B, C, H, W] 的形状。
        self.batch_norm = nn.BatchNorm2d(out_channels)

        # ReLU 把负数截为 0，为网络加入非线性；它同样不改变形状。
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回形状为 [B,out_channels,H_out,W_out] 的特征图。"""

        # 第一步：卷积提取局部时频特征，并按构造参数改变通道数。
        x = self.conv(x)

        # 第二步：对卷积产生的每个通道做批归一化，形状保持不变。
        x = self.batch_norm(x)

        # 第三步：执行 ReLU 并返回。一个 Conv2dBlock 到这里才算结束。
        x = self.activation(x)
        return x


class SleepEDFWaveFeatureNet(nn.Module):
    """把 [B,1,30,60] CWT 图转成 [B,216] 特征或 [B,5] logits。"""

    def __init__(self, n_classes: int = 5) -> None:
        # 初始化父类，使 features、classifier 等子网络能被 PyTorch 登记。
        super().__init__()

        # 主干由四个“卷积池化块”组成。
        # 这里说“每块两次卷积”，是指连续创建两个不同的 Conv2dBlock；
        # 每个 Conv2dBlock 内部又依次执行 Conv2d、BatchNorm2d、ReLU。
        self.features = nn.Sequential(
            # ===== 第 1 块：输入 [B,1,30,60] =====
            # 第 1 次卷积：1 -> 32 通道；k=3,s=1,p=1 保持 30x60。
            Conv2dBlock(1, 32, kernel_size=3, stride=1, padding=1),

            # 第 2 次卷积：继续提取特征，通道保持 32，空间仍为 30x60。
            Conv2dBlock(32, 32, kernel_size=3, stride=1, padding=1),

            # 第 1 次池化：两个空间轴都减半，30x60 -> 15x30。
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            # ===== 第 2 块：输入 [B,32,15,30] =====
            # 第 1 次卷积：32 -> 48 通道，空间保持 15x30。
            Conv2dBlock(32, 48, kernel_size=3, stride=1, padding=1),

            # 第 2 次卷积：通道保持 48，空间保持 15x30。
            Conv2dBlock(48, 48, kernel_size=3, stride=1, padding=1),

            # 第 2 次池化：15x30 -> 7x15；15/2 向下取整为 7。
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            # ===== 第 3 块：输入 [B,48,7,15] =====
            # 第 1 次卷积：48 -> 64 通道，空间保持 7x15。
            Conv2dBlock(48, 64, kernel_size=3, stride=1, padding=1),

            # 第 2 次卷积：通道保持 64，空间保持 7x15。
            Conv2dBlock(64, 64, kernel_size=3, stride=1, padding=1),

            # 第 3 次池化：7x15 -> 3x7，两个奇数长度都向下取整。
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            # ===== 第 4 块：输入 [B,64,3,7] =====
            # 第 1 次卷积：64 -> 72 通道，空间保持 3x7。
            Conv2dBlock(64, 72, kernel_size=3, stride=1, padding=1),

            # 第 2 次卷积：通道保持 72，空间保持 3x7。
            Conv2dBlock(72, 72, kernel_size=3, stride=1, padding=1),

            # 第 4 次池化：3x7 -> 1x3，得到 [B,72,1,3]。
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        # 分类头只在分支预训练时使用。
        self.classifier = nn.Sequential(
            # 1x1 卷积不混合相邻位置，只把通道从 72 映射为 n_classes。
            # [B,72,1,3] -> [B,5,1,3]。
            Conv2dBlock(72, n_classes, kernel_size=1, stride=1, padding=0),

            # 对剩余的 1x3 空间位置求自适应平均，得到 [B,5,1,1]。
            # 如果缺少这一步，直接展平会得到 5x1x3=15 维，而不是 5 维。
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # 这个布尔值只决定 forward 是否经过 classifier。
        # 它不控制 BatchNorm 的状态；训练/推理由 model.train()/eval() 控制。
        # 与原仓库一致，新建网络默认处于预训练输出模式。
        self.classifier_enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """根据模式返回分类 logits 或 216 维时频特征。"""

        # 输入约定为 [B,1,30,60]。
        # 经过四个卷积池化块后得到 [B,72,1,3]。
        x = self.features(x)

        # 预训练模式：经过分类头，[B,72,1,3] -> [B,5,1,1]。
        # 特征模式：跳过分类头，保留 [B,72,1,3]。
        if self.classifier_enabled:
            x = self.classifier(x)

        # start_dim=1 表示保留第 0 维批次 B，只展平后面的维度：
        # 预训练模式 [B,5,1,1] -> [B,5]；
        # 特征模式 [B,72,1,3] -> [B,216]，因为 72*1*3=216。
        return torch.flatten(x, start_dim=1)

    def pretrain(self) -> None:
        """启用五分类头，但不改变 BatchNorm 的训练/推理状态。"""

        # 只打开分类头，不调用 self.train()，避免把两个不同概念混在一起。
        self.classifier_enabled = True

    def finetune(self) -> None:
        """关闭分类头，让融合网络取得 216 维时频特征。"""

        # 关闭分支分类头，供后续融合网络取得 216 维时频特征。
        # 这里同样不调用 self.eval()；外部训练循环负责训练/推理状态。
        self.classifier_enabled = False
