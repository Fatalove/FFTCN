# 里程碑 7：特征融合与非因果 TCN

目标文件：`models/merge/sleep_edf_fusion_tcn.py`

参考答案：`learning_guides/milestone_07/reference_solution.py`

本练习把前两个里程碑的单时段特征网络连接起来：1D-CNN 每个时段输出 256 维，2D-CNN 每个时段输出 216 维，拼接为 472 维；随后使用四个非因果 TCN 残差块建模 50 个连续时段的上下文，最后输出每个时段的五分类 logits。

> **训练策略范围说明**：本里程碑只实现模型前向，因此没有出现类别过采样。过采样不是被删除，而是属于里程碑 8“两步训练”的数据加载与训练策略。仓库只在 `seq_len=1` 的 1D-CNN、2D-CNN **训练集预训练**中启用 `offset_resample`；验证集、测试集以及 `seq_len=50` 的完整 FFTCN 微调都不启用。这样既不污染评估分布，也不会用单时段重采样破坏连续序列。

## 1. 读题与题意分析

输入：

```text
raw_epochs:  [B,T,1,3000]
wave_epochs: [B,T,1,30,60]
```

其中默认 `T=50`。前两个 CNN 都只认识单个时段，因此先把 `B` 和 `T` 合并：

```text
[B,T,1,3000]  -> [B*T,1,3000]  -> 1D-CNN -> [B*T,256]
[B,T,1,30,60] -> [B*T,1,30,60] -> 2D-CNN -> [B*T,216]
```

再恢复序列维并融合：

```text
[B*T,256] -> [B,T,256]
[B*T,216] -> [B,T,216]

cat(dim=-1): [B,T,256] + [B,T,216] -> [B,T,472]
```

TCN 用前后时段上下文生成 128 维表示：

```text
[B,T,472] -> 非因果 TCN -> [B,T,128] -> Linear -> [B,T,5]
```

仓库原始 `MergeSleepNet.forward` 最后返回 `[B*T,5]`，而本练习按计划保留 `[B,T,5]`。训练交叉熵前仍可把它 reshape 为 `[B*T,5]`，分类计算本身没有变化；保留序列维更容易检查数据流，也方便后续 NumPy 推理对齐。

## 2. 必要 Python/PyTorch 基础

### 2.1 `Conv1d` 的轴顺序

TCN 的设计目标是让卷积核沿睡眠时间轴 `T` 滑动，从而融合连续 epoch 的上下文。
`Conv1d` 只沿输入的最后一轴滑动，因此进入 TCN 前，必须把 `T` 放到最后一轴。

融合特征首先按容易理解的“时间在前、特征在后”格式保存：

```text
[B,T,F]
```

其中 `F=472` 是每个 epoch 的融合特征数。为了让 `T` 成为卷积滑动轴，需要转换成
PyTorch `Conv1d` 使用的 `[B,C,L]` 格式；此时 `C=F`、`L=T`：

```text
[B,T,F=472] -> [B,C=472,L=T]
```

因此进入 TCN 前交换特征轴和时间轴：

```python
x = x.permute(0, 2, 1)  # [B,T,F] -> [B,F,T]，让卷积核沿 T 滑动
```

TCN 结束后，再换回“每个时间步对应一个特征向量”的格式，供线性分类头使用：

```python
x = x.permute(0, 2, 1)  # [B,F,T] -> [B,T,F]
```

`permute` 只改变轴的解释顺序，不改变元素总数。

### 2.2 same padding

扩张卷积的有效卷积核长度为：

```text
k_effective = dilation × (kernel_size - 1) + 1
```

本题 `kernel_size=3`、`stride=1`，想让输入输出长度相同，需要的总补零量是：

```text
total_padding = dilation × (3 - 1) = 2 × dilation
```

非因果卷积把它平均分到左右两侧：

| dilation | 有效卷积核 | 左补零 | 右补零 |
|---:|---:|---:|---:|
| 1 | 3 | 1 | 1 |
| 2 | 5 | 2 | 2 |
| 4 | 9 | 4 | 4 |
| 8 | 17 | 8 | 8 |

因此 dilation 为 8 并不是创建长度 17 的权重。实际仍然只有 3 个权重位置，只是相邻权重之间间隔 8 个时间步。

### 2.3 因果与非因果

以 `kernel_size=3, dilation=1` 为例：

```text
因果卷积计算 y[t]：使用 x[t-2], x[t-1], x[t]
非因果卷积计算 y[t]：使用 x[t-1], x[t], x[t+1]
```

睡眠分期可以离线处理整夜记录，所以仓库设置 `causal=False`，允许当前时段使用未来时段信息。

### 2.4 残差连接与 1×1 投影

残差块计算：

```text
output = ReLU(main_path(x) + residual(x))
```

第一次 TCN 块的输入是 472 通道，主路径输出是 128 通道，不能直接相加：

```text
[B,472,T] + [B,128,T]  # 错误
```

因此残差路径使用 `kernel_size=1` 的卷积：

```text
[B,472,T] -> 1×1 Conv1d -> [B,128,T]
```

后三块都是 128 通道，可以直接使用原输入作为残差。

### 2.5 weight normalization

`weight_norm` 不处理输入数据，而是把卷积权重重新参数化为：

```text
weight = magnitude × normalized_direction
```

在 PyTorch 状态中通常表现为 `weight_g` 和 `weight_v`。它与 BatchNorm 不同：BatchNorm 归一化激活值，weight normalization 重新表达权重参数。

### 2.6 `reshape`、`cat` 和 `Linear`

```python
raw_flat = raw_epochs.reshape(B * T, 1, 3000)
```

让单时段 CNN 一次处理 `B*T` 个时段。

```python
fused = torch.cat((raw_features, wave_features), dim=-1)
```

沿最后的特征轴拼接，`256+216=472`，不会改变 `B` 和 `T`。

```python
logits = classifier(contextual_features)
```

`nn.Linear(128,5)` 自动作用在最后一维，因此 `[B,T,128]` 可直接变成 `[B,T,5]`。

## 3. 手工推演

### 3.1 一个未来信息示例

假设序列只有 5 个位置：

```text
索引： 0  1  2  3  4
输入： 0  0  0  1  0
```

使用权重全为 1 的非因果 `kernel_size=3` 卷积计算位置 2：

```text
y[2] = x[1] + x[2] + x[3]
     = 0 + 0 + 1
     = 1
```

位置 2 看到了未来位置 3，因此这是非因果卷积。因果卷积在位置 2 只能看到位置 0、1、2，结果会是 0。

### 3.2 四个 TCN 块的感受野

每个块有两次 `kernel_size=3` 卷积。一次 dilation 为 `d` 的卷积使感受野增加：

```text
(kernel_size - 1) × dilation = 2d
```

一个块有两次卷积，所以增加 `4d`：

| 块 | dilation | 累计感受野 |
|---:|---:|---:|
| 初始 | - | 1 |
| 1 | 1 | `1+4×1=5` |
| 2 | 2 | `5+4×2=13` |
| 3 | 4 | `13+4×4=29` |
| 4 | 8 | `29+4×8=61` |

理论感受野为 61 个时段，大于默认序列长度 50；序列边缘超出的部分由补零提供。

注意论文把扩张间隔描述为 `0、1、2、4`，而仓库代码把 `2**level` 直接传给 PyTorch 的 `dilation`，实际参数是 `1、2、4、8`。本项目实现仓库行为。

### 3.3 小批次形状推演

假设 `B=2, T=3`：

```text
raw:  [2,3,1,3000]  -> [6,1,3000]  -> [6,256] -> [2,3,256]
wave: [2,3,1,30,60] -> [6,1,30,60] -> [6,216] -> [2,3,216]

融合： [2,3,256] + [2,3,216] -> [2,3,472]
TCN：  [2,3,472] -> [2,3,128]
分类： [2,3,128] -> [2,3,5]
```

## 4. 带详细注释的完整核心代码

下面的完整代码逐段解释计算作用、张量形状和设计原因。请先理解前三节，再遮住本节，在正式练习文件中自己实现。

```python
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
        # 父类卷积固定 padding=0，因为左右补零量在 forward 中动态计算。
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
        """返回长度为 ceil(L_in/stride) 的非因果卷积结果。"""

        # 输入是 [B,C,L]，最后一维是序列长度。
        input_length = x.shape[-1]

        # Conv1d 会把整数参数保存成单元素 tuple，因此读取第 0 项。
        stride = self.stride[0]
        kernel_size = self.kernel_size[0]
        dilation = self.dilation[0]

        # same padding 的目标输出长度。
        output_length = math.ceil(input_length / stride)

        # 扩张后的有效卷积核长度，但实际可训练权重仍只有 kernel_size 个。
        effective_kernel = dilation * (kernel_size - 1) + 1

        # 从卷积长度公式反推总补零量。
        total_padding = max(
            0,
            (output_length - 1) * stride + effective_kernel - input_length,
        )

        # 非因果卷积把补零分到左右两边；奇数时多出的 1 放在右边。
        left_padding = total_padding // 2
        right_padding = total_padding - left_padding
        x = F.pad(x, (left_padding, right_padding))

        # 输入已手工补零，所以 functional conv1d 的 padding 必须为 0。
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

        # 第一次扩张卷积：in_channels -> out_channels，序列长度保持 T。
        # weight_norm 重新参数化卷积权重，和仓库实现保持一致。
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

        # 第二次扩张卷积：保持 out_channels 和 T；同一块使用相同 dilation。
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

        # 主路径顺序：卷积、ReLU、Dropout，再重复一次。
        self.main_path = nn.Sequential(
            self.conv1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.relu2,
            self.dropout2,
        )

        # 第一块需要把残差从 472 投影为 128 通道；后三块通道相同。
        self.projection = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

        # 主路径与残差相加后再做一次 ReLU。
        self.output_relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """保持序列长度并返回 [B,out_channels,T]。"""

        # 两次 same-padding 卷积保证主路径长度仍为 T。
        main = self.main_path(x)

        # 通道相同时直接使用 x，否则用 1×1 卷积投影。
        residual = x if self.projection is None else self.projection(x)

        # 两条路径形状相同，逐元素相加。
        return self.output_relu(main + residual)


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

        blocks: list[nn.Module] = []
        for level, out_channels in enumerate(channels):
            # 2**level 产生仓库实际使用的 dilation：1、2、4、8。
            dilation = 2**level

            # 第一块输入 472 通道；后续块输入前一块的 128 通道。
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
    """双分支特征提取、融合、非因果 TCN 和逐时段分类。"""

    def __init__(self, n_classes: int = 5) -> None:
        super().__init__()

        # 复用里程碑 5、6 已验收的特征网络。
        self.raw_feature_net = SleepEDFRawFeatureNet(n_classes=n_classes)
        self.wave_feature_net = SleepEDFWaveFeatureNet(n_classes=n_classes)

        # 融合阶段关闭分支自己的五分类头，取得 256/216 维特征。
        self.raw_feature_net.finetune()
        self.wave_feature_net.finetune()

        # 256+216=472；TCN 将每个时段映射成 128 维上下文特征。
        self.tcn = NonCausalTCN(
            input_channels=472,
            channels=(128, 128, 128, 128),
            kernel_size=3,
            dropout=0.2,
        )

        # Linear 作用在最后一维：[B,T,128] -> [B,T,5]。
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, raw_epochs: torch.Tensor, wave_epochs: torch.Tensor) -> torch.Tensor:
        """返回保留批次和序列维的 [B,T,5] logits。"""

        # 记录共同的 B 和 T，随后让单时段 CNN 处理 B*T 个时段。
        batch_size, sequence_length = raw_epochs.shape[:2]

        # [B,T,1,3000] -> [B*T,1,3000] -> [B*T,256]。
        raw_flat = raw_epochs.reshape(batch_size * sequence_length, *raw_epochs.shape[2:])
        raw_features = self.raw_feature_net(raw_flat)

        # [B*T,256] -> [B,T,256]。
        raw_features = raw_features.reshape(batch_size, sequence_length, 256)

        # [B,T,1,30,60] -> [B*T,1,30,60] -> [B*T,216]。
        wave_flat = wave_epochs.reshape(
            batch_size * sequence_length,
            *wave_epochs.shape[2:],
        )
        wave_features = self.wave_feature_net(wave_flat)

        # [B*T,216] -> [B,T,216]。
        wave_features = wave_features.reshape(batch_size, sequence_length, 216)

        # 最后一维拼接：[B,T,256] + [B,T,216] -> [B,T,472]。
        fused = torch.cat((raw_features, wave_features), dim=-1)

        # [B,T,472] -> [B,T,128]。
        contextual_features = self.tcn(fused)

        # 对每个时段共享同一个线性分类头：[B,T,128] -> [B,T,5]。
        return self.classifier(contextual_features)
```

同一份实现保存在 `reference_solution.py`，正式练习文件不会导入它。

## 5. 建议实现顺序

1. 只实现 `SamePadConv1d`，通过长度和非因果性两个测试；
2. 实现单个 `TemporalResidualBlock`，确认通道投影与残差相加；
3. 实现 `NonCausalTCN`，检查 dilation 列表和 `[B,50,472] -> [B,50,128]`；
4. 最后接入两个已完成的特征分支，实现融合与分类；
5. 运行全部测试并检查三条梯度路径。

## 6. 聚焦测试命令

只测试 same padding：

```powershell
python -m unittest tests.test_sleep_edf_fusion_tcn.SleepEDFFusionTCNTest.test_same_padding_preserves_length_for_all_dilations -v
```

只测试非因果行为：

```powershell
python -m unittest tests.test_sleep_edf_fusion_tcn.SleepEDFFusionTCNTest.test_same_padding_is_noncausal_and_sees_the_next_position -v
```

运行全部里程碑 7 测试：

```powershell
python -m unittest tests.test_sleep_edf_fusion_tcn -v
```

## 7. 常见错误解释

### 用固定 `padding=dilation` 后认为已经理解 same padding

对于本题 `kernel_size=3,stride=1`，固定写法确实能保持长度；但仓库函数支持一般输入，练习要求从输出长度公式反推总补零量，以理解奇数补零和 stride 的影响。

### 只在序列左边补零

这会变成因果卷积，位置 `t` 无法看到未来。仓库明确使用 `causal=False`，左右都要补零。

### 残差相加时报 472 和 128 不匹配

第一块遗漏了 `1×1 Conv1d` 投影。残差连接不会自动改变通道数。

### TCN 输出变成 `[B,128,T]`

进入 Conv1d 前做了第一次 `permute`，结束后忘记换回 `[B,T,128]`。

### 拼接后得到 50×50 或 512 维

应沿最后的特征维 `dim=-1` 拼接；256 与 216 相加得到 472。不要沿时间维或批次维拼接。

### 最终输出是 `[B*T,5]`

这是原仓库接口。当前教学契约要求 `[B,T,5]`，不要在 `forward` 最后展平；训练计算交叉熵时再 reshape。

### 内部属性命名不同

不影响测试。测试检查 same padding、非因果行为、dilation、形状和梯度，不要求使用参考答案的内部变量名。

## 8. 工程加固（选读）

生产代码还可以检查两种输入的 `B/T` 是否一致、序列是否正好为 50、输入是否含 NaN/Inf，以及设备/dtype 是否匹配。本次主答案只保留复现正确性需要的数据流和结构，不把额外防御逻辑混入核心实现。

过采样同样不应写入模型 `forward`。它必须由训练 DataLoader 在构造训练样本时完成，详细行为、随机种子、训练/验证隔离和计数验证留到里程碑 8。
