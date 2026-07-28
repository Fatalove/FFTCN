# 里程碑 8：两步训练、过采样与可复现检查点

当前状态：已完成。用户正式实现已通过 6/6 聚焦测试、全项目 32/32 回归及真实 Sleep-EDF 小样本训练链路冒烟。

正式练习文件：`training/sleep_edf_two_stage.py`<br>
行为测试：`tests/test_sleep_edf_two_stage.py`<br>
独立参考答案：`learning_guides/milestone_08/reference_solution.py`

## 0. 本里程碑的边界

里程碑 5–7 实现的是模型如何从输入计算 logits；里程碑 8 处理的是模型**怎样被训练**。
过采样、学习率、训练/验证切换和检查点都不应写进 `FFTCNFusionTCN.forward()`。

本里程碑先用小数组和小模型验证训练语义，不直接启动完整 Sleep-EDF 长训练。通过代码练习后，
再使用真实 DataLoader 依次执行：

```text
raw 单 epoch 训练集（偏移过采样）  ──> 1D-CNN 预训练权重
wave 单 epoch 训练集（偏移过采样） ──> 2D-CNN 预训练权重
                                      │
                                      ▼
未过采样的连续 50-epoch 训练序列 ──> 完整 FFTCN 差分学习率微调
                                      │
验证集选择 best checkpoint ──────────┘
                                      │
模型选择结束后，测试集只评估一次 <────┘
```

## 1. 问题分析

### 1.1 为什么不能只进行一次端到端训练

Sleep-EDF 中 N2 等类别明显多于 N1。若直接在原始分布上端到端训练，两个 CNN 很容易优先学会多数类，
而 N1 等少数类的局部波形和时频特征学习不足。

论文采用两步训练：

1. **分支预训练**：把每个 30 秒 epoch 当作独立五分类样本，对训练集执行偏移过采样，分别训练 1D-CNN 和 2D-CNN；
2. **融合微调**：加载两个分支的预训练权重，用保持真实顺序的连续 50-epoch 序列训练 CNN + TCN + 分类头。

两个分支的预训练是并列关系，不是“先训练 1D-CNN，再把其输出交给 2D-CNN”。

### 1.2 为什么过采样不能用于 50-epoch 序列

TCN 要学习真实的睡眠阶段转移，例如：

```text
W -> N1 -> N2 -> N2 -> N3
```

若把少数类单个 epoch 随机复制后拼成序列，可能得到：

```text
N1 -> W -> N3 -> N1 -> REM
```

这种组合没有真实时间关系，会改变 TCN 的学习目标。因此仓库在 `balance=True` 时强制
`seq_len == 1`；融合微调的 `seq_len=50` 必须 `balance=False`。

### 1.3 仓库的 `offset_resample` 不会严格平衡类别

仓库不是把每一类都补到多数类数量，而是：

```text
保留所有原样本
+ 为每个已有类别额外生成 n_max 个偏移窗口
```

若原始类别数量为：

```text
N2: 1000
N1: 200
```

其中 `n_max=1000`，过采样后变成：

```text
N2: 1000 + 1000 = 2000
N1:  200 + 1000 = 1200
```

类别比例由 `5:1` 缩小为约 `1.67:1`，但并没有变成 `1:1`。因此准确表述是“减轻类别不平衡”。

### 1.4 偏移窗口的含义

Sleep-EDF 采样率为 100 Hz，仓库使用 `offset=300`：

```text
300 samples / 100 samples/s = 3 s
```

假设原 epoch 从采样点 6000 开始，随机偏移为 `-120`，新窗口就从 5880 开始，仍截取 3000 点。
标签继续使用原 epoch 的类别。这样生成的不是完全相同的复制样本，而是时间边界略有变化的窗口。

### 1.5 微调阶段不是冻结特征分支

仓库融合微调的默认参数为：

```text
base_learning_rate = 1e-5
lamb               = 1e-2
```

因此：

```text
TCN + 最终分类头学习率 = 1e-5
1D-CNN 分支学习率       = 1e-5 * 1e-2 = 1e-7
2D-CNN 分支学习率       = 1e-5 * 1e-2 = 1e-7
```

两个分支仍然计算梯度并更新，只是更新得更慢。`lr=1e-7` 与“冻结”不同；冻结意味着参数不更新。

仓库还使用 `gamma=0.95` 的指数学习率调度器，每个 epoch 后把所有参数组的学习率乘以 0.95，
因此基础学习率与分支学习率的比例始终保持不变。

### 1.6 默认没有额外的类别权重

原仓库交叉熵权重为：

```text
weight_c = (N_total / N_c) ** alpha
```

融合训练默认 `alpha=0`，任何正数的零次幂都是 1，所以默认等价于普通交叉熵。
本里程碑先忠实保留这个实验语义，不擅自叠加新的类别权重策略。

## 2. 必要 Python、NumPy 与 PyTorch 基础

### 2.1 `dataclass` 用来保存数据策略

```python
# @dataclass 根据字段自动生成初始化、显示和比较方法；frozen=True 禁止创建后改字段。
@dataclass(frozen=True)
class LoaderPolicy:
    stage: str
    split: str
    sequence_length: int
    balance: bool
    input_mode: str
```

它把“训练哪个阶段、读取哪个 split、是否过采样”等信息绑定在一起。`frozen=True` 表示创建后不应被意外修改。

### 2.2 局部随机数生成器

```python
rng = np.random.default_rng(seed)
selected = rng.choice(candidates, size=n_max, replace=True)
```

不要在教学工具内部反复调用 `np.random.seed()` 修改全局随机状态。局部 `rng` 的优点是：

- 相同 seed 得到相同样本；
- 不干扰项目其他随机过程；
- 测试可以独立复现。

`replace=True` 表示有放回抽样：同一个原 epoch 可以被选中多次，但每次随机偏移可能不同。

### 2.3 为什么先把整条记录展平

原输入：

```text
[N,1,3000]
```

这里每个轴的含义是：

- `N`：这条睡眠记录被切成了多少个 30 秒 epoch；
- `1`：单通道 EEG；
- `3000`：一个 epoch 内的采样点数，即 `30 秒 × 100 Hz`。

代码先计算“一个 epoch 一共有多少个元素”：

```python
epoch_length = int(np.prod(epochs.shape[1:]))
```

逐部分拆开：

```text
epochs.shape       = (N, 1, 3000)  # 整批数据的形状
epochs.shape[1:]   = (1, 3000)     # 去掉第 0 轴 N，只保留单个 epoch 的形状
np.prod((1,3000)) = 3000           # 将单个 epoch 的各维长度相乘
epoch_length       = 3000           # 一个 epoch 展平后的元素数
```

不能直接写 `epochs.shape[0]`，因为它表示 epoch 的数量 `N`，而不是每个 epoch 的长度。
使用 `np.prod(epochs.shape[1:])`，是为了兼容单个 epoch 仍带有通道轴的情况；例如
`[N, 1, 3000]` 和 `[N, 3000]` 算出的 `epoch_length` 都是 3000。
外层的 `int(...)` 把 NumPy 整数转换为普通 Python 整数，方便后面参与索引和切片计算。

将整条记录视为连续信号：

```python
flattened_record = epochs.reshape(-1)
```

`reshape(-1)` 中的 `-1` 表示“元素总数由 NumPy 自动推算”。它只改变观察数据的形状，
不会打乱元素的先后顺序：

```text
[N,1,3000] -> [N × 3000]
```

因此，原来按时间连续排列的第 0、1、2……个 epoch，会在一维数组中首尾相接。

某个 epoch 的原始起点是：

```text
epoch_index * 3000
```

加入随机偏移后：

```text
start = epoch_index * 3000 + shift
```

再读取 `flattened_record[start:start+3000]`。这里展平的算法目的，是允许窗口跨越原 30 秒分段边界，
而不是为了满足某个 API 的形式要求。

### 2.4 如何筛选当前类别的安全候选索引

对于一维标签：

```python
labels = np.array([0, 0, 2, 1, 2])
class_label = 2
candidates = np.flatnonzero(labels == class_label)
```

分两步理解：

```text
labels == 2                  -> [False, False, True, False, True]
np.flatnonzero(...)          -> [2, 4]
```

`flatnonzero` 直接返回非零或 `True` 元素的一维索引，因此比一维场景中的
`np.where(...)[0]` 更直观。

随后需要排除整条记录的首尾位置：

```python
excluded_boundary_indices = np.array([0, len(labels) - 1])
candidates = np.setdiff1d(candidates, excluded_boundary_indices)
```

`np.setdiff1d(A, B)` 表示“保留 A 中不属于 B 的元素”。在上例中：

```text
candidates                    = [2,4]
excluded_boundary_indices     = [0,4]
筛选结果                       = [2]
```

注意，`[0,4]` 表示“需要排除的集合”。如果改为内部索引 `[1,2,3]`，就应使用
`np.intersect1d` 取交集，而不能继续使用 `setdiff1d`。

### 2.5 优化器参数组

PyTorch 的一个优化器可以包含多个参数组：

```python
optimizer = torch.optim.Adam([
    {"params": new_parameters, "lr": 1e-5},
    {"params": raw_parameters, "lr": 1e-7},
    {"params": wave_parameters, "lr": 1e-7},
])
```

必须保证：

- 每个可训练参数恰好出现一次；
- 不能遗漏分类头；
- 不能让同一个参数同时出现在两个组中。

### 2.6 `pretrain()/finetune()` 与 `train()/eval()` 不同

本项目中的：

```python
feature_net.pretrain()
feature_net.finetune()
```

控制的是**是否使用分支自己的五分类头**。

PyTorch 中的：

```python
model.train()
model.eval()
```

控制的是 Dropout 与 BatchNorm 的训练/推理行为。两组方法不能互相替代。

### 2.7 训练步骤与验证步骤

训练步骤需要：

```text
train 模式 -> 前向 -> loss -> 清梯度 -> 反向 -> optimizer.step
```

验证步骤需要：

```text
eval 模式 -> no_grad -> 前向 -> loss
```

验证阶段禁止执行 `backward()` 和 `optimizer.step()`，否则验证集会间接参与参数学习。

完整实现中使用下面的判断统一两种流程：

```python
training = optimizer is not None
model.train(training)

with torch.set_grad_enabled(training):
    ...
```

这里有两个容易混淆但彼此独立的开关：

- `model.train(training)` 控制 Dropout、BatchNorm 等模块采用训练行为还是推理行为；
- `torch.set_grad_enabled(training)` 控制 PyTorch 是否建立反向传播需要的计算图。

模型输出的 `loss` 不是普通数字，而是一个零维张量。例如：

```text
tensor(0.8321, grad_fn=<NllLossBackward0>)
```

其中 `grad_fn` 表示它仍连接着用于反向传播的计算图。训练完成后执行：

```python
return float(loss.detach())
```

分为两步：

```text
loss.detach()  -> tensor(0.8321)  # 得到与计算图断开的零维张量
float(...)     -> 0.8321          # 转成普通 Python 浮点数
```

所以该函数返回的是“当前 batch 的平均交叉熵损失”，可用于日志记录和跨 batch 求平均；
它不是预测标签，也不是五个类别的概率。返回普通浮点数还能避免日志列表无意中长期保留计算图。

### 2.8 为什么保存 `state_dict`

原仓库使用 `torch.save(model)` 保存整个对象，这会绑定当时的 Python 类路径。教学实现改为保存：

```text
model_state_dict
optimizer_state_dict
config
label_names
stage / epoch / validation_loss
```

加载时先由当前代码创建模型，再恢复参数。这样更清楚、更便于迁移，也能明确记录实验配置。

相关 API 的职责分别是：

```text
model.state_dict()              读取“参数/缓冲区名称 -> 张量”的映射
model.load_state_dict(state)    把保存的模型状态填入已创建的同结构模型
optimizer.state_dict()          读取 Adam 动量等优化器状态
optimizer.load_state_dict(...)  恢复优化器，使训练能够从原进度继续
```

`torch.save(...)` 和 `torch.load(...)` 负责把包含这些状态及实验元数据的字典写入磁盘、再读回来。
`load_training_checkpoint` 最终返回整个检查点字典，是为了让调用者除了恢复模型外，还能读取
`config`、`label_names`、`stage`、`epoch` 和 `validation_loss`。

## 3. 手工推演

### 3.1 偏移过采样计数

假设一条记录只有两个类别：

```python
labels = [0, 0, 0, 1, 1]
```

原始计数：

```text
class 0: 3
class 1: 2
n_max:   3
```

算法为每个类别额外生成 3 个窗口：

```text
class 0: 3 + 3 = 6
class 1: 2 + 3 = 5
```

最终为 `6:5`，差距缩小但不严格相等。

### 3.2 两步数据策略表

| 阶段 | split | seq_len | balance | 输入 |
|---|---|---:|---:|---|
| raw 预训练 | train | 1 | True | 原始 EEG |
| raw 预训练 | valid | 1 | False | 原始 EEG |
| wave 预训练 | train | 1 | True | CWT 图 |
| wave 预训练 | valid | 1 | False | CWT 图 |
| 融合微调 | train | 50 | False | 原始 EEG + CWT 图 |
| 融合微调 | valid | 50 | False | 原始 EEG + CWT 图 |
| 最终评估 | test | 50 | False | 原始 EEG + CWT 图 |

### 3.3 一个 batch 的训练形状

分支预训练：

```text
raw:  [B,1,3000]  -> [B,5]
wave: [B,1,30,60] -> [B,5]
label: [B]
```

融合微调：

```text
raw:    [B,50,1,3000]
wave:   [B,50,1,30,60]
logits: [B,50,5]
label:  [B,50]
```

计算交叉熵前，把所有预测位置展平：

```text
[B,50,5] -> [B*50,5]
[B,50]   -> [B*50]
```

这里展平的是“批次中的预测位置”，不会改变 TCN 已经融入 logits 的上下文信息。

## 4. 带详细注释的完整核心代码

下面代码完整覆盖本里程碑的行为契约。请先理解前三节，再遮住本节，在正式练习文件中实现。

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer


# 类别名称的排列顺序就是模型输出索引的含义：logits[...,0] 对应 W，
# logits[...,1] 对应 N1，以此类推。保存检查点时也要保存这份顺序。
LABEL_NAMES = ("W", "N1", "N2", "N3", "REM")


@dataclass(frozen=True)
class LoaderPolicy:
    """描述一个阶段的数据来源和是否允许改变类别分布。"""

    stage: str  # 训练阶段：pretrain、finetune 或 test。
    split: str  # 数据划分：train、valid 或 test。
    sequence_length: int  # 一次输入包含几个连续 epoch；预训练为 1，融合为 50。
    balance: bool  # 是否允许偏移过采样；只能用于单 epoch 训练集。
    input_mode: str  # 使用 raw、wave，还是同时使用两种输入 both。


# 实验配置同样保持不可变，避免训练途中无意修改复现条件。
@dataclass(frozen=True)
class TwoStageConfig:
    """足以复现实验语义的最小配置。"""

    seed: int = 42  # 控制可复现抽样。
    sequence_length: int = 50  # 融合阶段一次读取 50 个连续 epoch。
    base_learning_rate: float = 1e-5  # 新加入的 TCN 和分类头使用的学习率。
    feature_learning_rate_scale: float = 1e-2  # 预训练分支学习率的缩放比例。
    scheduler_gamma: float = 0.95  # 每轮结束后，学习率乘以 0.95。
    offset_samples: int = 300  # 偏移窗口最多移动的元素数；100 Hz 时约为 3 秒。


def build_stage_policies(sequence_length: int = 50) -> dict[str, LoaderPolicy]:
    """明确每个阶段的数据角色，阻止过采样污染验证、测试和连续序列。"""

    # 返回值是“策略名称 -> LoaderPolicy”的字典。LoaderPolicy 的五个位置参数依次是：
    # stage、split、sequence_length、balance、input_mode。
    return {
        # 第一步：两个 CNN 都用单个 epoch 做五分类预训练。
        # 只有训练集允许偏移过采样；验证集必须保持真实类别分布。
        "raw_pretrain_train": LoaderPolicy("pretrain", "train", 1, True, "raw"),
        "raw_pretrain_valid": LoaderPolicy("pretrain", "valid", 1, False, "raw"),
        "wave_pretrain_train": LoaderPolicy("pretrain", "train", 1, True, "wave"),
        "wave_pretrain_valid": LoaderPolicy("pretrain", "valid", 1, False, "wave"),
        # 第二步：融合模型必须读取连续 sequence_length 个 epoch。
        # 即使是训练集也不能做单 epoch 随机过采样，否则会破坏序列上下文。
        "fusion_finetune_train": LoaderPolicy(
            "finetune", "train", sequence_length, False, "both"
        ),
        "fusion_finetune_valid": LoaderPolicy(
            "finetune", "valid", sequence_length, False, "both"
        ),
        # 测试集只在模型选择完成后用于最终评估，永远不参与参数更新或过采样。
        "fusion_test": LoaderPolicy("test", "test", sequence_length, False, "both"),
    }


def offset_resample_record(
    epochs: np.ndarray,
    labels: np.ndarray,
    offset: int = 300,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """在单条记录内部生成带时间偏移的新 epoch。

    与仓库语义一致：保留所有原样本，然后为每个已有类别额外生成
    ``n_max`` 个窗口。它会减轻类别不平衡，但不会把最终类别数严格补齐。
    """

    # 统一转成 NumPy 数组，确保后面可以使用 .shape、布尔比较和 reshape。
    # 预期形状：epochs=[N,...]，labels=[N]；第 i 个 epoch 对应 labels[i]。
    epochs = np.asarray(epochs)
    labels = np.asarray(labels)
    if labels.ndim != 1 or epochs.shape[0] != labels.shape[0]:
        raise ValueError("epochs 和 labels 的第一维必须一一对应")

    # 第 0 轴 N 表示 epoch 数量，不属于“单个 epoch”的内部形状，所以使用 shape[1:]。
    # 例如 epochs.shape=[N,1,3000]：shape[1:]=(1,3000)，乘积为 3000，
    # 表示一个 epoch 展平后包含 3000 个元素。int 将 NumPy 整数转成 Python 整数。
    epoch_length = int(np.prod(epochs.shape[1:]))
    # offset 的单位是“展平后的元素个数”。偏移量必须小于一个完整 epoch，
    # 否则新窗口可能不再以原 epoch 附近的信号为主体。
    if offset < 0 or offset >= epoch_length:
        raise ValueError("offset 必须满足 0 <= offset < 单个 epoch 的元素数")
    # 空记录没有类别可统计，也没有窗口可生成；返回副本以免调用者误以为返回值
    # 与传入数组是同一个可修改对象。
    if len(labels) == 0:
        return epochs.copy(), labels.copy()

    # 创建仅供本函数使用的随机数生成器：同一 seed 会选中相同 epoch 并生成相同偏移，
    # 同时不会改变项目其他代码使用的 NumPy 全局随机状态。
    rng = np.random.default_rng(seed)

    # labels.tolist() 把 NumPy 数组转为 Python 列表；dict.fromkeys(...) 在保留
    # 首次出现顺序的同时去重。例如 [0,0,2,1,2] -> [0,2,1]。
    class_labels = list(dict.fromkeys(labels.tolist()))

    # labels == label 产生布尔数组，True 表示该位置属于当前类别；np.sum 统计 True。
    # 例如 labels=[0,0,2,1,2] 时，class_counts={0:2, 2:2, 1:1}。
    class_counts = {label: int(np.sum(labels == label)) for label in class_labels}
    # 找出原始数据中样本最多的类别数，后面每个类别都会额外生成 n_max 个偏移窗口。
    n_max = max(class_counts.values())

    # 先把所有原始样本放进结果。这里用“数组列表”暂存，是为了循环中逐批 append，
    # 最后再一次 concatenate；这比每生成一批就反复扩展大数组更清楚。
    balanced_epochs = [epochs.copy()]
    balanced_labels = [labels.copy()]

    # 将按时间排列的整条记录从 [N,...] 接成一维连续信号。
    # -1 表示让 NumPy 自动计算总长度；例如 [N,1,3000] -> [N*3000]。
    # 元素顺序不会被打乱。这样偏移后的新窗口才能跨越原有 30 秒 epoch 的边界。
    flattened_record = epochs.reshape(-1)

    # 整条记录之外没有可供偏移窗口读取的信号，因此首、尾 epoch 不能作为候选中心：
    # 首 epoch 向左偏移会越过记录起点，尾 epoch 向右偏移会越过记录终点。
    # 这里只列出“需要排除”的两个边界索引，而不是列出中间所有有效索引。
    # 例如共有 5 个 epoch 时，需要排除 [0,4]，有效索引自然剩下 [1,2,3]。
    excluded_boundary_indices = np.array([0, len(labels) - 1])

    # 每次循环只为一个睡眠类别生成新的偏移窗口。
    for class_label in class_labels:
        # labels == class_label 得到布尔掩码；flatnonzero 直接返回其中 True 的一维索引。
        # 例如 labels=[0,0,2,1,2]、class_label=2 时，结果为 [2,4]。
        candidates = np.flatnonzero(labels == class_label)

        # setdiff1d(A,B) 表示“保留 A 中不属于 B 的元素”。这里从当前类别的索引中
        # 删除首尾边界。例如 [2,4] 去掉 [0,4] 后，只剩安全候选 [2]。
        candidates = np.setdiff1d(candidates, excluded_boundary_indices)

        # 某个类别如果只出现在首尾位置，就没有能够安全左右偏移的中心，直接跳过。
        if len(candidates) == 0:
            continue

        # 从安全候选中抽取 n_max 个“中心 epoch”。replace=True 表示有放回抽样，
        # 因而候选数少于 n_max 时也能抽够，同一个中心 epoch 也可能被重复选中。
        # 例如 candidates=[2]、n_max=3 时，selected 只能是 [2,2,2]。
        selected = rng.choice(candidates, size=n_max, replace=True)

        # 为每次抽中的中心生成一个随机移动量。offset=0 时必须单独处理，因为
        # rng.integers(0,0) 没有可抽取的整数，会报错。
        if offset == 0:
            shifts = np.zeros(n_max, dtype=np.int64)
        else:
            # integers 的上界不包含在内，因此这里的范围是 [-offset, offset)。
            # 例如 offset=2 时，可能得到 -2、-1、0、1。
            shifts = rng.integers(-offset, offset, size=n_max)

        # 在一维连续记录中，第 i 个原 epoch 的起点是 i * epoch_length；
        # 再加 shift，便得到新窗口的起点。例如 i=2、长度=3000、shift=-100，
        # 新起点就是 2*3000-100=5900，即从原边界前 100 个元素开始读取。
        starts = selected * epoch_length + shifts

        # 对每个新起点执行三步：
        # 1. 截取连续的 epoch_length 个元素，保证新窗口长度与原 epoch 相同；
        # 2. reshape 回单个 epoch 的原形状，例如 [3000] -> [1,3000]；
        # 3. stack 在第 0 轴叠放全部窗口，得到 [n_max,1,3000]。
        shifted = np.stack(
            [
                flattened_record[start : start + epoch_length].reshape(epochs.shape[1:])
                for start in starts
            ],
            axis=0,
        )

        # 新窗口虽然发生了小幅时间移动，但仍继承其“中心 epoch”的睡眠阶段。
        # np.full 创建长度为 n_max 的标签数组，并保持原 labels 的数据类型。
        shifted_labels = np.full(n_max, class_label, dtype=labels.dtype)

        # 暂存当前类别生成的数据和标签；所有类别处理完成后再统一拼接。
        balanced_epochs.append(shifted)
        balanced_labels.append(shifted_labels)

    # 每个列表的第一个元素是全部原始样本，后续元素是各类别的新样本。
    # concatenate 默认沿第 0 轴连接，因此最终仍保持 epochs=[样本数,...]、labels=[样本数]。
    return np.concatenate(balanced_epochs), np.concatenate(balanced_labels)


def run_classification_step(
    model: nn.Module,
    inputs: Sequence[torch.Tensor],
    targets: torch.Tensor,
    optimizer: Optimizer | None = None,
    return_correct_count: bool = False,
) -> float | tuple[float, int]:
    """运行一次分类步骤，并按需返回同一次前向的正确位置数。

    参数来源：
        model: 当前阶段的 raw、wave 或 fusion 模型，由上层训练编排传入。
        inputs: 当前 DataLoader batch 的模型输入；单分支含一个张量，融合分支
            含 raw、wave 两个张量，顺序与 ``model(*inputs)`` 一致。
        targets: 与 logits 预测位置一一对应的类别索引，形状为 ``[B]`` 或
            ``[B,T]``。
        optimizer: 训练时由上层传入当前优化器；验证时为 ``None``。
        return_correct_count: ``run_epoch()`` 显示累计 ACC 时传 ``True``；
            既有只需要 loss 的调用保持默认 ``False``。

    返回去向：
        默认返回 Python float 的 batch 平均交叉熵；请求正确数时返回
        ``(batch_loss, correct_count)``，供 ``run_epoch()`` 跨 batch 累加。
    """

    # 本函数用“是否传入优化器”区分两种用途：
    # - optimizer 不为 None：训练，需要反向传播并更新参数；
    # - optimizer 为 None：验证，只计算损失，不更新参数。
    training = optimizer is not None

    # train(True) 递归开启训练模式，Dropout 会随机失活、BatchNorm 会更新统计量；
    # train(False) 等价于 eval()。它只控制模块行为，并不单独决定是否记录梯度。
    model.train(training)

    # set_grad_enabled(True) 让训练前向过程建立计算图，以便 loss.backward()；
    # False 则让验证过程不建立计算图，从而节省内存和计算量。
    with torch.set_grad_enabled(training):
        # inputs 是张量序列，* 会把它解包成位置参数：
        # inputs=(raw,)       -> model(raw)
        # inputs=(raw, wave)  -> model(raw, wave)
        logits = model(*inputs)

        # 预训练输出可能是 [B,5]，融合输出是 [B,T,5]。
        # 最后一轴始终是 5 个类别分数；reshape 把前面的所有预测位置合并：
        # [B,5] -> [B,5]，或 [B,T,5] -> [B*T,5]。
        flat_logits = logits.reshape(-1, logits.shape[-1])

        # 标签必须按同样顺序展平：[B] -> [B]，或 [B,T] -> [B*T]。
        # 这样 flat_targets 的第 i 个标签正好对应 flat_logits 的第 i 行。
        flat_targets = targets.reshape(-1)

        # cross_entropy 将每行 5 个 logits 与对应类别索引比较，并对所有位置取平均。
        # loss 是形如 tensor(0.83, grad_fn=...) 的零维张量，不是 Python float。
        loss = F.cross_entropy(flat_logits, flat_targets)

        if optimizer is not None:
            # PyTorch 默认会累加梯度，因此每一步反向传播前先清除上一步的梯度。
            optimizer.zero_grad()
            # 从 loss 沿计算图反向计算每个可训练参数的梯度。
            loss.backward()
            # 优化器读取这些梯度，并按各参数组的学习率更新模型参数。
            optimizer.step()

    # detach() 得到一个不再连接计算图的零维张量；float(...) 再把它转换为普通
    # Python 浮点数，例如 tensor(0.83) -> 0.83。默认只返回本 batch 平均损失，
    # 因而里程碑 8 原有调用不需要改变。
    batch_loss = float(loss.detach())
    if not return_correct_count:
        return batch_loss

    # 训练进度需要 ACC 时，沿 flat_logits 最后一轴的类别分数取 argmax：
    # [预测位置数,类别数] -> [预测位置数]，再与同形状 flat_targets 比较。
    # sum 得到本 batch 正确位置数；item/int 转为 Python 整数供跨 batch 累加。
    # 这里复用本次前向的 logits，不会为了展示指标再执行一次模型前向。
    correct_count = int(
        (flat_logits.detach().argmax(dim=-1) == flat_targets).sum().item()
    )
    return batch_loss, correct_count


def transfer_pretrained_features(
    fusion_model: nn.Module,
    raw_pretrained: nn.Module,
    wave_pretrained: nn.Module,
) -> None:
    """迁移预训练权重，并让融合模型分支输出特征而非五分类 logits。"""

    # state_dict() 是“参数/缓冲区名称 -> 张量”的映射。load_state_dict(...) 按名称
    # 将两个独立预训练模型的权重复制到融合模型中对应的特征分支；网络结构必须匹配。
    fusion_model.raw_feature_net.load_state_dict(raw_pretrained.state_dict())
    fusion_model.wave_feature_net.load_state_dict(wave_pretrained.state_dict())

    # 这里的 finetune() 是本项目特征网络自定义的方法：关闭分支自己的五分类头，
    # 让 raw 分支输出 256 维特征、wave 分支输出 216 维特征供融合模型使用。
    # 它与 PyTorch 的 model.train()/model.eval() 不是同一件事。
    fusion_model.raw_feature_net.finetune()
    fusion_model.wave_feature_net.finetune()


def build_finetune_optimizer(
    model: nn.Module,
    base_learning_rate: float,
    feature_learning_rate_scale: float,
) -> torch.optim.Adam:
    """新模块用基础学习率，两个预训练分支用缩小后的学习率继续更新。"""

    # parameters() 返回一次性迭代器；转成 list 后既能组合、检查身份，也能交给优化器。
    raw_parameters = list(model.raw_feature_net.parameters())
    wave_parameters = list(model.wave_feature_net.parameters())

    # id(parameter) 是该 Parameter 对象在当前 Python 进程中的唯一身份标识。
    # 用集合保存两个预训练分支的身份，后面可从完整模型参数中准确排除它们。
    feature_ids = {id(parameter) for parameter in raw_parameters + wave_parameters}

    # model.parameters() 包含融合模型的全部参数。列表推导式只保留身份不在
    # feature_ids 中的参数，所以 new_parameters 就是新加入的 TCN 和最终分类头。
    new_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in feature_ids
    ]

    # 例如 base_learning_rate=1e-5、scale=1e-2 时，预训练分支学习率为 1e-7。
    # 学习率较小表示“继续缓慢更新”，并不等于冻结参数。
    feature_learning_rate = base_learning_rate * feature_learning_rate_scale

    # Adam 接收三个参数组。每个可训练参数恰好出现一次：新模块使用基础学习率，
    # 两个预训练分支使用缩小后的学习率。
    return torch.optim.Adam(
        [
            {"params": new_parameters, "lr": base_learning_rate},
            {"params": raw_parameters, "lr": feature_learning_rate},
            {"params": wave_parameters, "lr": feature_learning_rate},
        ]
    )


def save_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    config: Mapping[str, Any],
    label_names: Sequence[str],
    stage: str,
    epoch: int,
    validation_loss: float,
) -> None:
    """保存参数和实验元数据，而不是序列化整个 Python 模型对象。"""

    # 允许调用者传入字符串或 Path，并统一转换为 Path 对象。
    path = Path(path)

    # parents=True 会递归创建缺失的父目录；exist_ok=True 表示目录已存在时不报错。
    path.parent.mkdir(parents=True, exist_ok=True)

    # torch.save 将下面的字典序列化到 path。这里只保存可恢复的状态和元数据，
    # 不保存整个 model 对象，因此不会绑定当前 Python 类的导入路径。
    torch.save(
        {
            # 模型 state_dict 保存所有可学习参数以及 BatchNorm 等模块的缓冲区。
            "model_state_dict": model.state_dict(),
            # 优化器 state_dict 保存 Adam 动量等状态，恢复后才能连续训练。
            "optimizer_state_dict": optimizer.state_dict(),
            # 转成普通 dict/list，固定本次实验配置与类别索引顺序。
            "config": dict(config),
            "label_names": list(label_names),
            # 记录训练进度和验证损失，便于判断从哪个阶段、哪一轮继续。
            "stage": stage,
            "epoch": int(epoch),
            "validation_loss": float(validation_loss),
        },
        path,
    )


def load_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """恢复状态；模型结构仍由当前代码显式创建。"""

    # torch.load 读取保存的字典。map_location="cpu" 可把原本位于 GPU 的张量
    # 映射到 CPU，避免在没有相同 GPU 环境的机器上加载失败。
    checkpoint = torch.load(Path(path), map_location=map_location)

    # 调用者已经用当前代码创建好模型结构；这里仅把保存的参数和缓冲区填回模型。
    model.load_state_dict(checkpoint["model_state_dict"])

    # 推理时可以不传优化器；只有需要继续训练时，才恢复 Adam 等优化器状态。
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # 返回完整字典，让调用者继续读取 config、label_names、stage、epoch 和验证损失。
    return checkpoint
```

## 5. 建议实现顺序

1. 实现 `build_stage_policies()`，先明确每个阶段能否过采样；
2. 实现 `offset_resample_record()`，用手工示例核对最终计数是 `6:5`；
3. 实现 `run_classification_step()`，确认训练损失下降、验证不更新参数；
4. 实现 `transfer_pretrained_features()`，核对两个分支权重完全一致且输出 256/216 维特征；
5. 实现 `build_finetune_optimizer()`，确认所有参数恰好进入一个参数组；
6. 实现检查点保存和恢复，确认配置、标签、epoch 和验证损失没有丢失；
7. 运行全部聚焦测试并把结果交给 Codex 验收。

## 6. 聚焦测试

只检查数据策略与过采样：

```powershell
python -m unittest tests.test_sleep_edf_two_stage.SleepEDFTwoStageTest.test_stage_policies_keep_balancing_out_of_validation_test_and_sequences tests.test_sleep_edf_two_stage.SleepEDFTwoStageTest.test_offset_resample_is_reproducible_and_reduces_without_equalizing -v
```

只检查训练步骤：

```powershell
python -m unittest tests.test_sleep_edf_two_stage.SleepEDFTwoStageTest.test_training_step_decreases_loss_and_validation_does_not_update -v
```

运行里程碑 8 全部测试：

```powershell
python -m unittest tests.test_sleep_edf_two_stage -v
```

正式练习当前保留 `NotImplementedError`，所以在你实现前测试失败是预期现象。

## 7. 常见错误

### 把所有类别补到严格相等

那是另一种常见过采样策略，但不符合仓库 `offset_resample`。本题必须保留原样本，并为每个类别额外增加 `n_max` 个窗口。

### 对验证集或测试集使用 `balance=True`

这会改变评估分布，使指标不能代表真实数据。验证和测试必须始终禁用过采样。

### 对 `seq_len=50` 过采样

单 epoch 随机重采样会破坏真实的连续睡眠序列。本题只有两个分支的 `seq_len=1` 训练集允许过采样。

### 把较小学习率理解为冻结

只要参数仍在优化器中且 `requires_grad=True`，它就会继续更新。真正冻结通常需要移出优化器或设置 `requires_grad=False`。

### 在验证步骤调用 `optimizer.step()`

这样验证集就参与了训练。验证步骤应当 `eval()`、关闭梯度，并且不接收优化器。

### 直接保存整个模型对象

`torch.save(model)` 会依赖原来的 Python 类路径。本教学契约保存 `state_dict` 和元数据，加载时显式创建同结构模型。

### 只保存权重，不保存标签顺序

五个 logits 的下标必须稳定对应 `W、N1、N2、N3、REM`。缺少标签元数据会让部署阶段无法可靠解释输出。

### 把 `feature_net.finetune()` 当成 `model.eval()`

前者关闭分支自己的分类头，后者关闭 Dropout 并切换 BatchNorm 行为。两者职责完全不同。

## 8. 与原仓库的有意差异

训练数学与数据策略保持一致，但检查点格式做了明确改进：

| 项目 | 原仓库 | 教学实现 |
|---|---|---|
| 过采样 | 每记录 `offset_resample` | 相同计数和偏移语义，改用局部 RNG |
| 微调 | 三个 Adam 参数组 | 相同参数分组和学习率比例 |
| 检查点 | `torch.save(model)` | 保存 `state_dict`、配置和标签元数据 |
| 测试 | 训练脚本末尾测试 | 明确测试集只在模型选择完成后使用 |

局部 RNG 和 `state_dict` 不改变模型训练目标，但提高了复现性和可移植性。

## 9. 工程加固（选读）

生产训练系统还可以进一步加入：

- 校验每个 split 的受试者 ID 集合无交集；
- DataLoader 使用独立 `torch.Generator` 和 worker seed；
- 记录代码版本、依赖版本、设备、数据哈希和 checkpoint 哈希；
- 原子写入检查点，避免中断产生半文件；
- 断点续训时恢复 scheduler、随机数状态和 GradScaler；
- 对 NaN/Inf、空类别和损坏文件进行显式检查。

这些属于工程加固，不混入本次核心练习。
