# 里程碑 6：2D-CNN 时频分支

目标文件：`models/wavelet/sleep_edf_2d_cnn.py`

参考答案：`learning_guides/milestone_06/reference_solution.py`

本练习只实现 FFTCN 的 2D-CNN 时频特征分支。输入是 `[B,1,30,60]` 的 CWT 小波图；特征模式输出 `[B,216]`，预训练模式输出 `[B,5]`。本里程碑不进入 TCN、特征融合或训练循环。

## 1. 读题与题意分析

里程碑 4 已把一个 30 秒 EEG 时段转换为：

```text
[1,30,60]
```

加入批次维后是：

```text
[B,1,30,60]
```

四个轴依次表示：

| 轴 | 含义 |
|---|---|
| `B` | 批次中的样本数 |
| `1` | 小波图通道数 |
| `30` | Morlet 尺度轴，可近似理解为频率方向 |
| `60` | 压缩后的时间轴 |

2D-CNN 沿尺度和时间两个方向同时寻找局部模式。仓库实际启用四个卷积池化块，输出 `[B,72,1,3]`，展平后为 `72×1×3=216`。

## 2. 必要 Python/PyTorch 基础

### `Conv2d` 的输入形状

PyTorch 二维卷积固定使用：

```text
[批次 B, 通道 C, 高度 H, 宽度 W]
```

本练习约定 `H=30` 是尺度轴，`W=60` 是时间轴。不要为了“看起来更像图片”而随意交换二者。

### 二维输出尺寸公式

高度和宽度分别使用同一个公式：

```text
H_out = floor((H_in + 2*p_h - k_h) / s_h) + 1
W_out = floor((W_in + 2*p_w - k_w) / s_w) + 1
```

本题卷积使用 `kernel_size=3, stride=1, padding=1`：

```text
floor((H + 2 - 3) / 1) + 1 = H
```

所以卷积保持空间尺寸不变，只改变通道数。

池化使用 `kernel_size=2, stride=2, padding=0`：

```text
floor((L - 2) / 2) + 1 = floor(L / 2)
```

因此奇数长度会向下取整，例如 `15 -> 7`，不是 `15 -> 8`。

### `BatchNorm2d` 与展平

`BatchNorm2d(72)` 中的 72 是通道数，不是高度或宽度。最终使用：

```python
torch.flatten(x, start_dim=1)
```

保留批次维，把后面的通道和二维空间全部展平。

### 两组模式开关

- `model.train()/eval()` 控制 BatchNorm 的运行方式；
- `model.pretrain()/finetune()` 控制是否经过分支自己的五分类头。

二者相互独立。

## 3. 手工推演

每个块包含两次(一个卷积块中顺序放了两个独立的 Conv2d 层)保持尺寸的 `3×3` 卷积，再进行一次 `2×2` 最大池化：

| 阶段 | 通道 | 尺度轴 H | 时间轴 W | 输出形状 |
|---|---:|---:|---:|---|
| 输入 | 1 | 30 | 60 | `[B,1,30,60]` |
| 第 1 块卷积后 | 32 | 30 | 60 | `[B,32,30,60]` |
| 第 1 次池化后 | 32 | 15 | 30 | `[B,32,15,30]` |
| 第 2 块卷积后 | 48 | 15 | 30 | `[B,48,15,30]` |
| 第 2 次池化后 | 48 | 7 | 15 | `[B,48,7,15]` |
| 第 3 块卷积后 | 64 | 7 | 15 | `[B,64,7,15]` |
| 第 3 次池化后 | 64 | 3 | 7 | `[B,64,3,7]` |
| 第 4 块卷积后 | 72 | 3 | 7 | `[B,72,3,7]` |
| 第 4 次池化后 | 72 | 1 | 3 | `[B,72,1,3]` |

于是：

```text
72 × 1 × 3 = 216
```

如果错误地把输入轴换成 `[B,1,60,30]`，四次池化后会得到 `[B,72,3,1]`。展平后仍然碰巧是 216，因此只检查最终维数无法发现轴交换；测试还必须检查最后一张特征图的二维几何形状。

## 4. 论文 432 维与仓库 216 维的区别

论文描述的是 `200×30` 压缩小波图、五个卷积池化块和 432 维小波特征；当前仓库训练路径使用 `[1,30,60]`，且 `WaveFeatureNet` 实际只启用四个卷积池化块，因此本项目实现 216 维仓库版。

需要注意：当前 `models/wavelet/parameter.py` 虽然还保留了候选第五卷积与池化参数，但第五池化的 `kernel_size=1`，直接取消 `network.py` 中的注释不会改变当前 `1×3` 空间尺寸，也不会自动得到论文的 432 维。论文与仓库的差异是输入压缩尺寸、启用块数和池化配置共同造成的，不能简化为“少了一块”。

## 5. 带注释的完整核心代码

下面的注释不仅说明“这行做什么”，还标出了每层的输入输出形状和这样设置参数的原因。四个块中的八次卷积也逐一编号，避免把“两次卷积”误解为重复调用同一层。

```python
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
        # 初始化 nn.Module，使下面的层能参与训练、保存和设备迁移。
        super().__init__()

        # 输入形状是 [B,in_channels,H,W]。
        # 主干使用 k=3,s=1,p=1，因此 H/W 不变，只改变通道数。
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        # BatchNorm2d 的参数是输出通道数，不是高度或宽度。
        # 它调整数值分布，但不改变张量形状。
        self.batch_norm = nn.BatchNorm2d(out_channels)

        # ReLU 把负值截为 0，引入非线性，形状仍保持不变。
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 第一步：卷积提取局部时频特征。
        x = self.conv(x)

        # 第二步：按输出通道做批归一化。
        x = self.batch_norm(x)

        # 第三步：执行 ReLU；到这里才完成一次 Conv2dBlock。
        x = self.activation(x)
        return x


class SleepEDFWaveFeatureNet(nn.Module):
    """把 [B,1,30,60] CWT 图转成 [B,216] 特征或 [B,5] logits。"""

    def __init__(self, n_classes: int = 5) -> None:
        # 初始化父类，使主干和分类头都能被 PyTorch 登记。
        super().__init__()

        # 每块包含两个不同的 Conv2dBlock，随后接一次 MaxPool2d。
        self.features = nn.Sequential(
            # ===== 第 1 块：输入 [B,1,30,60] =====
            # 第 1 次卷积：1 -> 32 通道，空间保持 30x60。
            Conv2dBlock(1, 32, kernel_size=3, stride=1, padding=1),
            # 第 2 次卷积：32 -> 32 通道，空间仍为 30x60。
            Conv2dBlock(32, 32, kernel_size=3, stride=1, padding=1),
            # 第 1 次池化：30x60 -> 15x30。
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            # ===== 第 2 块：输入 [B,32,15,30] =====
            # 第 1 次卷积：32 -> 48 通道，空间保持 15x30。
            Conv2dBlock(32, 48, kernel_size=3, stride=1, padding=1),
            # 第 2 次卷积：48 -> 48 通道，空间保持 15x30。
            Conv2dBlock(48, 48, kernel_size=3, stride=1, padding=1),
            # 第 2 次池化：15x30 -> 7x15，15/2 向下取整为 7。
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            # ===== 第 3 块：输入 [B,48,7,15] =====
            # 第 1 次卷积：48 -> 64 通道，空间保持 7x15。
            Conv2dBlock(48, 64, kernel_size=3, stride=1, padding=1),
            # 第 2 次卷积：64 -> 64 通道，空间保持 7x15。
            Conv2dBlock(64, 64, kernel_size=3, stride=1, padding=1),
            # 第 3 次池化：7x15 -> 3x7，两个奇数长度都向下取整。
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            # ===== 第 4 块：输入 [B,64,3,7] =====
            # 第 1 次卷积：64 -> 72 通道，空间保持 3x7。
            Conv2dBlock(64, 72, kernel_size=3, stride=1, padding=1),
            # 第 2 次卷积：72 -> 72 通道，空间保持 3x7。
            Conv2dBlock(72, 72, kernel_size=3, stride=1, padding=1),
            # 第 4 次池化：3x7 -> 1x3，得到 [B,72,1,3]。
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        # 分类头只在预训练模式启用。
        self.classifier = nn.Sequential(
            # 1x1 卷积只改变通道：[B,72,1,3] -> [B,5,1,3]。
            Conv2dBlock(72, n_classes, kernel_size=1, stride=1, padding=0),
            # 对 1x3 三个位置求平均：[B,5,1,3] -> [B,5,1,1]。
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # 默认启用分类头；这个标志不控制 BatchNorm 的 train/eval 状态。
        self.classifier_enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 主干把 [B,1,30,60] 转成 [B,72,1,3]。
        x = self.features(x)

        # 预训练时经过分类头得到 [B,5,1,1]；特征模式跳过分类头。
        if self.classifier_enabled:
            x = self.classifier(x)

        # 保留批次维：预训练得到 [B,5]，特征模式得到 [B,216]。
        return torch.flatten(x, start_dim=1)

    def pretrain(self) -> None:
        # 只启用分类头，不在这里调用 train()。
        self.classifier_enabled = True

    def finetune(self) -> None:
        # 关闭分类头，让后续融合网络取得 216 维特征；不调用 eval()。
        self.classifier_enabled = False
```

同一份完整实现也保存在 `reference_solution.py`。正式练习文件不会导入它。

## 6. 建议实现顺序

1. 完成 `Conv2dBlock`，先确认 `3×3, padding=1` 保持 `30×60`；
2. 依次搭建四个卷积池化块，每加一个池化就手工检查一次二维尺寸；
3. 确认主干输出 `[B,72,1,3]` 后，再实现展平和 216 维特征模式；
4. 添加五分类头与两个模式切换方法；
5. 运行全部聚焦测试。

## 7. 聚焦测试命令

只运行基础卷积块测试：

```powershell
python -m unittest tests.test_sleep_edf_2d_cnn.SleepEDF2DCNNTest.test_conv_block_preserves_spatial_shape_and_module_order -v
```

运行全部里程碑 6 测试：

```powershell
python -m unittest tests.test_sleep_edf_2d_cnn -v
```

## 8. 常见错误解释

### 得到 `[B,72,3,1]`

输入的 30 和 60 被交换，或网络内部做了不必要的 `transpose`。最终展平维度仍是 216，所以要检查池化后的二维形状。

### 15 池化后得到 8

最大池化公式需要向下取整：`floor(15/2)=7`。

### 预训练输出是 `[B,15]`

只做了 `1×1` 分类卷积而没有 `AdaptiveAvgPool2d((1,1))`，于是 `[B,5,1,3]` 被直接展平成了 15 维。

### 内部属性命名与参考答案不同

这不是错误。聚焦测试检查模块行为、类型和顺序，不要求你使用 `batch_norm/activation` 等特定内部命名。只有公开接口或 checkpoint 兼容需要固定命名时，名称才属于契约。

## 9. 工程加固（选读）

生产代码还可以显式拒绝非 `[B,1,30,60]` 输入、检查 dtype/设备以及 NaN/Inf。本练习通过中间几何形状测试保证轴语义，主答案不加入与核心结构无关的大量防御代码。
