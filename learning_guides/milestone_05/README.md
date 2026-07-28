# 里程碑 5：1D-CNN 时域分支

目标文件：`models/raw/sleep_edf_1d_cnn.py`

参考答案：`learning_guides/milestone_05/reference_solution.py`

本练习只实现 FFTCN 的 1D-CNN 时域特征分支。输入是 `[B,1,3000]` 的 EEG 批次；预训练模式输出 `[B,5]`，特征模式输出 `[B,256]`。本里程碑不进入 2D-CNN、TCN、数据加载或训练循环。

## 1. 读题与题意分析

这条分支有两个用途：

1. **分支预训练**：主干后接五分类头，直接学习睡眠分期，输出 `[B,5]`；
2. **融合微调**：移除分支自己的分类头，只把 256 维时域特征交给后续融合网络。

核心数据流是：

```text
[B,1,3000]
  -> Conv/BN/ReLU
  -> Pool/Dropout
  -> 3 x Conv/BN/ReLU
  -> Pool/Dropout
  -> [B,128,2]
  -> 展平为 [B,256]                         特征模式
  -> Conv/BN/ReLU + AdaptiveAvgPool + 展平  预训练模式 [B,5]
```

这里采用当前 GitHub 仓库版参数。论文版和仓库版在部分卷积核与最终融合维度上不同；本项目已经确定复现仓库的 472 维融合版，因此时域特征必须是 256 维。

## 2. 必要 Python/PyTorch 基础

### `nn.Module`

每个网络模块都继承 `nn.Module`。层在 `__init__` 中创建，张量流在 `forward` 中描述：

```python
class Example(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.ReLU()

    def forward(self, x):
        return self.layer(x)
```

把层保存为 `self.xxx` 很重要：PyTorch 才能登记参数、把参数移动到 GPU，并在反向传播后保存梯度。

### `Conv1d` 的三个维度

PyTorch 一维卷积输入固定使用：

```text
[批次 B, 通道 C, 序列长度 L]
```

本题输入 `[B,1,3000]` 表示一个批次中有 `B` 个样本，每个样本有 1 个 EEG 通道、3000 个采样点。

卷积或池化后的长度公式是：

```text
L_out = floor((L_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
```

本题 dilation 都是 1，可以简化为：

```text
L_out = floor((L_in + 2*padding - kernel_size) / stride) + 1
```

### `BatchNorm1d`、`ReLU` 和 `Dropout`

- `BatchNorm1d(128)` 的 128 必须与前一层输出通道数一致；
- `ReLU` 只改变数值，不改变形状；
- `Dropout` 在 `model.train()` 时随机置零，在 `model.eval()` 时关闭，也不改变形状。

### `train()/eval()` 与 `pretrain()/finetune()`

它们控制不同事情：

- PyTorch 自带的 `train()/eval()`：控制 BatchNorm、Dropout 的运行行为；
- 本练习的 `pretrain()/finetune()`：控制输出是否经过五分类头。

因此，调用 `finetune()` 不应该偷偷调用 `eval()`，调用 `pretrain()` 也不应该偷偷调用 `train()`。

## 3. 手工推演

先按公式手算每一层长度。通道变化单独写在第二列：

| 操作 | 通道 | 长度计算 | 输出长度 |
|---|---:|---|---:|
| 输入 | 1 | - | 3000 |
| Conv1d `k=50,s=25,p=0` | 128 | `floor((3000-50)/25)+1` | 119 |
| MaxPool `k=8,s=8` | 128 | `floor((119-8)/8)+1` | 14 |
| Conv1d `k=8,s=1,p=3` | 128 | `14+6-8+1` | 13 |
| Conv1d `k=8,s=1,p=3` | 128 | `13+6-8+1` | 12 |
| Conv1d `k=8,s=1,p=3` | 128 | `12+6-8+1` | 11 |
| MaxPool `k=4,s=4` | 128 | `floor((11-4)/4)+1` | 2 |

所以主干最终是：

```text
[B,128,2] -> flatten -> [B,256]
```

预训练分类头再做一次 `kernel_size=2` 的卷积：

```text
[B,128,2] -> [B,5,1] -> flatten -> [B,5]
```

这也解释了为什么不能把某个卷积的 padding 随手改成“same”：只要最终长度不再是 2，特征维度就不会是 256。

## 4. 带注释的完整核心代码

```python
import torch
from torch import nn


class Conv1dBlock(nn.Module):
    """按照 Conv1d -> BatchNorm1d -> ReLU 组织一个基础卷积块。"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        # 卷积提取局部时域模式，并改变通道数或序列长度。
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        # 批归一化的通道数必须等于卷积输出通道数。
        self.batch_norm = nn.BatchNorm1d(out_channels)

        # ReLU 引入非线性，但不改变形状。
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        return self.activation(x)


class SleepEDFRawFeatureNet(nn.Module):
    """把 [B,1,3000] EEG 转成 [B,256] 特征或 [B,5] logits。"""

    def __init__(self, n_classes=5):
        super().__init__()

        # 长度流：3000 -> 119 -> 14 -> 13 -> 12 -> 11 -> 2。
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

        # 分类头只在分支预训练时启用：[B,128,2] -> [B,5,1]。
        self.classifier = nn.Sequential(
            Conv1dBlock(128, n_classes, kernel_size=2, stride=1, padding=0),
            nn.AdaptiveAvgPool1d(1),
        )

        # 原仓库中新建特征网络默认处于预训练输出模式。
        self.classifier_enabled = True

    def forward(self, x):
        # 主干固定输出 [B,128,2]。
        x = self.features(x)

        # 预训练时得到 [B,5,1]；特征模式跳过分类头。
        if self.classifier_enabled:
            x = self.classifier(x)

        # [B,5,1] -> [B,5]，或 [B,128,2] -> [B,256]。
        return torch.flatten(x, start_dim=1)

    def pretrain(self):
        # 只控制分类头开关，不改变 BatchNorm/Dropout 状态。
        self.classifier_enabled = True

    def finetune(self):
        # 融合微调阶段需要 256 维时域特征，因此关闭分支分类头。
        self.classifier_enabled = False
```

完整版本也单独保存在 `reference_solution.py`。请理解前三节后，再在正式练习文件中自己敲写或独立重写。

## 5. 建议实现顺序

1. 完成 `Conv1dBlock.__init__` 和 `forward`；
2. 只运行卷积块测试，确认 `[2,1,3000] -> [2,128,119]`；
3. 创建 `self.features`，暂时用手工张量检查 `[B,128,2]`；
4. 创建 `self.classifier` 和 `classifier_enabled`；
5. 完成网络 `forward`、`pretrain()` 和 `finetune()`；
6. 运行全部里程碑 5 测试。

## 6. 聚焦测试命令

只运行第一个卷积块测试：

```powershell
python -m unittest tests.test_sleep_edf_1d_cnn.SleepEDF1DCNNTest.test_conv_block_shape_and_module_order -v
```

运行全部里程碑 5 测试：

```powershell
python -m unittest tests.test_sleep_edf_1d_cnn -v
```

## 7. 常见错误解释

### 得到 384 维而不是 256 维

通常是三层 `kernel_size=8, padding=3` 卷积的长度算错，或某层误用了 `padding=4`。仓库版最终必须是 `[B,128,2]`。

### `BatchNorm1d` 报通道数错误

`BatchNorm1d` 的参数应等于前一层卷积的 `out_channels`，不是时间长度。

### `finetune()` 后 Dropout 仍然生效

这是正常现象。`finetune()` 只关闭分类头；是否启用 Dropout 由外部训练循环调用 `model.train()` 或 `model.eval()` 决定。

### 梯度是 `None`

检查是否把层保存在 `self.features` 中、是否调用了 `loss.backward()`，以及是否在前向计算外层使用了 `torch.no_grad()`。

## 8. 工程加固（选读）

生产代码还可以显式检查：

- 输入是否严格为三维 `[B,1,3000]`；
- 输入 dtype 和设备是否与模型参数一致；
- 输入或中间特征是否含 NaN/Inf；
- 自定义类别数是否为正整数。

这些检查不属于本次核心算法主线。当前主答案只保留网络结构、形状流、模式切换和可训练性。
