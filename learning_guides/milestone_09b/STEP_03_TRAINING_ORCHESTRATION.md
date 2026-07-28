# 里程碑 9B 内部步骤三：完整训练编排

步骤一、二已经把 Sleep-EDF 缓存转换成可复现的 batch。本步骤把里程碑 5–8 的模型与训练核心接口连接成完整运行链：

```text
逐记录偏移过采样缓存
  -> raw 分支预训练 -> raw validation best
  -> wave 分支预训练 -> wave validation best
  -> state_dict 迁移到融合模型
  -> T=50 融合微调 -> fusion validation best
```

本步骤不读取 test split，不启动全量缓存构建或长训练。它们属于步骤四的预检与正式实验。

正式练习：`training/sleep_edf_full_run.py`<br>
行为自测：`tests/test_sleep_edf_training_orchestration.py`<br>
独立参考：`learning_guides/milestone_09b/reference_training_orchestration.py`

## 1. 问题分析

### 1.1 为什么还需要平衡预训练缓存

原仓库先在**每条记录内部**对 raw EEG 做时间偏移过采样，再由偏移后的 raw 计算 CWT。现有 `wave.npy` 只对应未偏移的原始 epoch，不能直接与新增标签配对。

因此步骤三新增 `pretrain_train` 缓存：

```text
train/raw.npy + train/labels.npy + train/manifest.json
    -> 每条记录分别 offset_resample_record()
    -> pretrain_train/raw.npy
    -> 对同一批平衡 raw 计算 CWT
    -> pretrain_train/wave.npy + labels.npy + manifest.json
```

全量训练集预计从 `154,128` 增加到 `494,617` 个单 epoch 样本。平衡 raw 约 `5.53 GiB`，平衡 wave 约 `1.66 GiB`。代码一次只处理一条记录并写入 memmap，不把约 `7.19 GiB` 全部装进内存。

### 1.2 三个训练阶段

| 阶段 | train 数据 | validation 数据 | 输出 |
|---|---|---|---|
| raw 预训练 | `pretrain_train`，`T=1`，raw | 原始 `validation`，`T=1` | raw best checkpoint |
| wave 预训练 | `pretrain_train`，`T=1`，wave | 原始 `validation`，`T=1` | wave best checkpoint |
| 融合微调 | 原始 `train`，`T=50`，both | 原始 `validation`，`T=50` | fusion best checkpoint |

过采样只改变两个分支的训练集。validation 始终保持真实类别分布，融合阶段始终保持真实连续序列。

### 1.3 为什么迁移 best 而不是 final

每个 epoch 产生一个 validation loss。严格变小时才覆盖 `best.pt`；`last.pt` 每轮覆盖，只用于中断恢复。

```text
epoch 0: validation=0.80 -> best=epoch 0
epoch 1: validation=0.65 -> best=epoch 1
epoch 2: validation=0.73 -> best 仍是 epoch 1，last=epoch 2
```

raw/wave 预训练结束后必须重新加载各自的 `best.pt`，再通过 `state_dict` 迁移到融合模型。否则迁移的是内存中最后一轮参数，而不是 validation 选出的参数。

### 1.4 为什么不能平均 batch loss

交叉熵返回当前 batch 内所有预测位置的平均损失。最后一个 batch 可能更小，融合阶段每个样本又含 `T` 个标签，因此 epoch loss 必须按 `targets.numel()` 加权：

```text
epoch_loss = Σ(batch_loss × batch_position_count) / Σ(batch_position_count)
```

### 1.5 断点恢复必须保存什么

恢复下一轮训练不仅需要模型参数，还需要：

- optimizer 状态：例如 Adam 的一阶、二阶动量；
- scheduler 状态：下一轮应从正确学习率继续；
- 已完成 epoch：恢复后从 `epoch + 1` 开始；
- 当前 best validation loss：防止较差模型覆盖旧 best；
- history：日志不能在恢复后从头开始。

设备顺序也是恢复契约的一部分。`optimizer.load_state_dict()` 会参照当前模型参数
所在设备安放 Adam 动量，因此应先执行 `model.to(device)`，再恢复 optimizer。
如果等到进入 batch 循环才移动模型，模型参数和梯度会在 CUDA，而已经恢复的
`exp_avg`、`exp_avg_sq` 仍可能留在 CPU，第一个 `optimizer.step()` 就会失败。

### 1.6 函数参数从哪里来

先看调用链，再读单个函数，参数来源会更清楚：

```text
步骤四创建 FullTrainingConfig
  -> build_balanced_pretrain_cache(config.data_cache_dir, config.seed, ...)
  -> run_full_training(config) 创建 6 个 Dataset/DataLoader
       -> run_two_stage_training(config, loaders)
            -> fit_stage(stage, model, train_loader, validation_loader, ...)
                 -> run_epoch(model, loader, device, optimizer 或 None)
                      -> run_classification_step(model, inputs, targets, optimizer)
                 -> save_stage_checkpoint(...) / load_stage_checkpoint(...)
```

| 函数 | 主要输入从哪里来 | 结果交给谁 |
|---|---|---|
| `_balanced_record_count` | 缓存函数按 manifest 切出的一条记录标签 | 缓存函数用它预分配 memmap |
| `build_balanced_pretrain_cache` | `FullTrainingConfig` 的缓存目录、seed、offset | 步骤四得到平衡缓存及 manifest |
| `run_epoch` | `fit_stage` 传来的模型、Loader、device、可选 optimizer | `fit_stage` 得到 train/validation loss |
| `save/load_stage_checkpoint` | `fit_stage` 当前模型状态与恢复路径 | 断点续训或重新加载 validation best |
| `fit_stage` | `run_two_stage_training` 传来的单阶段全部对象 | 返回该阶段的 best、last 和 history |
| `_resolve_device` | `FullTrainingConfig.device` | 三个阶段共同使用的 `torch.device` |
| `run_two_stage_training` | config 和六个 Loader | 返回 raw、wave、fusion 三阶段结果 |
| `run_full_training` | 步骤四创建的 config | 建立 Loader 并把完整训练结果交回步骤四 |

## 2. 必要 Python 与 PyTorch 基础

### 2.1 两遍扫描与 memmap

`.npy` memmap 创建时必须知道总形状。第一遍只由标签计算每条记录过采样后的数量；第二遍才调用 `offset_resample_record()`，并把一条记录的结果写入已经分配好的全局区间。

### 2.2 `state_dict`

`model.state_dict()` 保存参数和缓冲区；optimizer 与 scheduler 也各自拥有 `state_dict()`。恢复时必须把它们加载回对应对象，而不是重新创建默认状态。

### 2.3 `range(start_epoch, epochs)`

若 `last.pt` 保存的最后完成轮次为 1，则：

```python
start_epoch = checkpoint["epoch"] + 1  # 2
for epoch in range(start_epoch, epochs):
    ...
```

这样不会重复训练 epoch 1。

### 2.4 为什么训练进度使用 batch

模型不是等整个 epoch 结束后才一次更新，而是每从 DataLoader 取得一个 batch，
就执行一次前向传播、反向传播和 `optimizer.step()`。因此用
`tqdm(train_loader, unit="batch")` 包装 Loader，进度条的分母就是
`len(train_loader)`，每完成一次真实参数更新便前进一格。

validation 虽然不更新参数，也仍然逐 batch 前向并汇总损失，所以使用独立进度条。
阶段开始时先单独显示 `==================== RAW PRETRAIN ====================`，
后续进度行以 `Epoch 1/20 | Train` 或 `Epoch 1/20 | Valid` 开头。这样阶段名
不在每个 epoch 重复，epoch 位置也保持在最容易扫描的行首。固定 `ncols=100`
限制进度条宽度，避免宽屏终端把条形区域拉得过长；进度条中的
`125/3865 batch` 表示当前 epoch 内真正执行到哪里。
进度行中的 `loss` 是截至当前 batch 的所有预测位置累计平均损失，`acc` 是
累计正确位置数除以累计位置数。两者都随 batch 推进实时更新；ACC 复用当前
前向已经产生的 logits，不额外执行模型前向。包装 Loader 只观察迭代进度，
不改变 batch 内容、shuffle 顺序或训练数学。

## 3. 手工推演

假设两条记录过采样后分别有 12、9 个样本：

```text
record_a -> [0,12)
record_b -> [12,21)
```

raw、wave、labels 必须共同写入这些区间。wave 的第 `i` 项必须由平衡 raw 的第 `i` 项计算，不能从原始 `train/wave.npy` 复制。

再手算 `run_epoch()` 为什么同时需要 `weighted_loss` 和 `position_count`。假设只有两个 batch：

```text
batch 1: 2 个标签位置，batch_loss = 0.4
batch 2: 1 个标签位置，batch_loss = 1.0

weighted_loss  = 0.4×2 + 1.0×1 = 1.8   # 所有位置的损失总和（分子）
position_count = 2 + 1 = 3             # 所有标签位置总数（分母）
epoch_loss     = 1.8 / 3 = 0.6
```

如果只计算两个 batch 平均值 `(0.4 + 1.0) / 2 = 0.7`，第二个只有 1 个位置的小 batch 会和第一个 2 个位置的 batch 获得相同权重。因此 `position_count = 0` 不是多余变量：它专门保存最终除法所需、且无法从 `weighted_loss` 反推出的分母。

若训练三轮并在第二轮后中断：

```text
last.pt: epoch=1, scheduler 已衰减两次, history 有 2 行
恢复后: start_epoch=2
完成后: last.pt epoch=2, history 有 3 行
```

## 4. 带详细注释的完整核心代码

下面代码只包含步骤三新增逻辑；`FullTrainingConfig`、Dataset/DataLoader 和里程碑 8 的训练核心接口直接复用。

```python
from __future__ import annotations

# json 写 manifest/history；Path 统一磁盘路径；typing 描述 batch、配置和回调接口。
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

# NumPy 负责 memmap/标签统计，PyTorch 负责模型、优化器和 DataLoader 类型。
import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

# 复用已完成里程碑的 CWT、三个模型和步骤二缓存读取接口；不导入原仓库 Loader。
from data.sleep_edf_cwt import morlet_cwt_epoch
from models.merge.sleep_edf_fusion_tcn import FFTCNFusionTCN
from models.raw.sleep_edf_1d_cnn import SleepEDFRawFeatureNet
from models.wavelet.sleep_edf_2d_cnn import SleepEDFWaveFeatureNet
from training.sleep_edf_full_run import (
    FullTrainingConfig,
    SleepEDFSequenceDataset,
    StageTrainingResult,
    build_reproducible_loader,
    load_record_spans,
)
# 复用里程碑 8 已验证的偏移过采样、单 batch 更新、权重迁移和差分学习率逻辑。
from training.sleep_edf_two_stage import (
    LABEL_NAMES,
    build_finetune_optimizer,
    offset_resample_record,
    run_classification_step,
    transfer_pretrained_features,
)


# WaveTransform 明确约束“一个 raw epoch -> 一个 wave epoch”，也允许行为测试注入替身。
WaveTransform = Callable[[np.ndarray], np.ndarray]


def _balanced_record_count(labels: np.ndarray) -> int:
    """计算一条记录经过偏移过采样后的总样本数。

    参数来源：
        labels: 一条记录的标签，形状为 ``[N]``。它由
            ``build_balanced_pretrain_cache()`` 按 manifest 的 ``[start:stop)``
            区间从 ``train/labels.npy`` 中取出。

    返回去向：
        返回该记录的“原始 N 项 + 新增偏移项”总数。上游缓存构建函数先收集
        每条记录的这个数，再求和得到 memmap 第一轴必须预分配的总长度。
    """

    # np.unique 同时给出本记录中实际出现的类别和各类别原始数量。
    # counts.max() 对应 offset_resample_record 中每个可采样类别要新增的 n_max 项。
    unique_labels, counts = np.unique(labels, return_counts=True)
    n_max = int(counts.max())

    # extra_count 只累计“将要新增”的样本数；原始 N 项留到返回时再加。
    # 独立保存这个累加器，才能逐类别累加而不丢失前面类别的贡献。
    extra_count = 0

    # 记录的第一个和最后一个 epoch 不能作为偏移中心，否则平移后的 3000 点窗口
    # 可能越过当前记录边界。这里保存它们的局部索引，供每个类别共同排除。
    boundary_indices = np.array([0, len(labels) - 1])

    # 逐个检查本记录中实际出现的类别，判断它是否拥有至少一个内部候选中心。
    for label in unique_labels:
        # 先取得当前类别在本记录中的全部局部 epoch 索引。
        candidates = np.flatnonzero(labels == label)
        # 再删除首尾边界，只保留 offset_resample_record 真正可以抽样的位置。
        candidates = np.setdiff1d(candidates, boundary_indices)

        # 只要该类别还有内部候选，源算法就会有放回地产生 n_max 个新窗口；
        # 因此总长度增加 n_max，而不是增加“原类别距离 n_max 还差多少”。
        if len(candidates) > 0:
            extra_count += n_max

    # len(labels) 是全部保留的原始样本，extra_count 是各可采样类别新增项之和。
    # int() 把 NumPy 整数转成普通 Python int，供 shape 和 JSON 元数据直接使用。
    return int(len(labels) + extra_count)


def build_balanced_pretrain_cache(
    cache_root: str | Path,
    seed: int,
    offset_samples: int = 300,
    wave_transform: WaveTransform = morlet_cwt_epoch,
) -> Path:
    """逐记录偏移过采样，并写出对齐的 ``pretrain_train`` 缓存。

    参数来源：
        cache_root: 训练缓存根目录。正式运行时来自
            ``FullTrainingConfig.data_cache_dir``，其下已经有步骤一生成的
            ``train/raw.npy``、``labels.npy`` 和 ``manifest.json``。
        seed: 本次实验的基础随机种子，来自 ``FullTrainingConfig.seed``。
            第 ``record_index`` 条记录实际使用 ``seed + record_index``，使每条
            记录可复现且不共享同一串随机抽样结果。
        offset_samples: 偏移窗口最多移动的采样点数，正式值来自
            ``FullTrainingConfig.offset_samples``，默认 300 点即 100 Hz 下 3 秒。
        wave_transform: 把单个 raw epoch ``[1,3000]`` 转成 CWT ``[1,30,60]``
            的函数。正式运行使用里程碑 4 的 ``morlet_cwt_epoch``；测试传入
            微型替身，以便只验证 raw/wave/label 对齐而不计算真实 CWT。

    返回去向：
        返回新 ``pretrain_train/manifest.json`` 的路径。步骤四构建全量缓存时
        用它记录产物位置；随后 ``run_full_training()`` 通过同目录读取三个
        ``.npy`` 文件，不需要把全量约 7.19 GiB 数据同时装入内存。
    """

    # 调用者可以传字符串或 Path。统一转成 Path 后，后续都用 / 组合子路径。
    cache_root = Path(cache_root)
    # source_dir 指向步骤一已经生成的未过采样训练缓存，是本函数唯一的数据来源。
    source_dir = cache_root / "train"
    # output_dir 单独保存分支预训练所需的平衡缓存，不覆盖融合训练仍要使用的 train。
    output_dir = cache_root / "pretrain_train"
    # parents=True 允许缓存根目录层级尚未创建；exist_ok=True 允许断点前已建目录。
    output_dir.mkdir(parents=True, exist_ok=True)

    # raw/labels 只以内存映射方式打开：它们仍留在磁盘，切片到某条记录时才读取。
    # 两者共享全局 epoch 轴；manifest 的 RecordSpan 指明每条夜晚的 [start:stop)。
    source_raw = np.load(source_dir / "raw.npy", mmap_mode="r")
    source_labels = np.load(source_dir / "labels.npy", mmap_mode="r")
    record_spans = load_record_spans(source_dir / "manifest.json")

    # open_memmap 创建文件前必须知道第一轴总长度，因此先做一次只统计标签的扫描。
    # balanced_lengths 按 manifest 顺序保存每条记录的输出长度，便于最后求总和。
    balanced_lengths: list[int] = []
    for span in record_spans:
        # span 来自 manifest；复制出独立 [N] 数组后即可关闭当前 memmap 切片视图。
        labels = np.array(source_labels[span.start : span.stop], copy=True)
        # 每条记录单独计数，保证偏移过采样的候选位置不会跨越夜晚边界。
        balanced_lengths.append(_balanced_record_count(labels))
    # 所有记录输出长度之和就是三个新缓存共享的全局 epoch 数。
    total_epochs = sum(balanced_lengths)

    # raw_shape 保留原始单 epoch 的内部轴，例如 [N,1,3000] -> [total,1,3000]；
    # 这样不把 1 个 EEG 物理通道和 3000 个时间采样点写死在复制逻辑中。
    raw_shape = (total_epochs, *source_raw.shape[1:])
    # 里程碑 4 的固定 CWT 契约是每个 epoch [1,30,60]：1 通道、30 频率、60 时间格。
    wave_shape = (total_epochs, 1, 30, 60)
    # 三个 open_memmap 立即在磁盘建立固定形状的 .npy 文件；w+ 表示创建并可写。
    # raw 用 float32 保持模型输入精度，第一轴位置 i 与 labels/wave 的 i 对齐。
    balanced_raw = np.lib.format.open_memmap(
        output_dir / "raw.npy", "w+", np.float32, shape=raw_shape
    )
    # wave 体积更大，沿用步骤一缓存的 float16 存储；Dataset 取样时再转 float32。
    balanced_wave = np.lib.format.open_memmap(
        output_dir / "wave.npy", "w+", np.float16, shape=wave_shape
    )
    # 分类标签是类别索引，使用 PyTorch 交叉熵需要的 int64，并只占一维样本轴。
    balanced_labels = np.lib.format.open_memmap(
        output_dir / "labels.npy", "w+", np.int64, shape=(total_epochs,)
    )

    # records 将记录每条夜晚写入新缓存后的边界，稍后写进新的 manifest。
    records: list[dict[str, Any]] = []
    # cursor 是下一条记录在全局输出数组中的起点。它从 0 开始，每写完一条记录
    # 就移动到该记录 stop；没有这个游标，就无法把变长记录连续且不重叠地拼接。
    cursor = 0
    try:
        # enumerate 同时给出稳定的记录序号和原始边界；序号只用于派生局部 seed。
        for record_index, span in enumerate(record_spans):
            # 从磁盘只复制当前记录的 raw 为 [N,1,3000] float32。一次处理一条记录，
            # 所以内存峰值由最大单条夜晚决定，而不是由 494,617 个总样本决定。
            raw_record = np.array(
                source_raw[span.start : span.stop], dtype=np.float32, copy=True
            )
            # 用完全相同的 [start:stop) 复制 [N] 标签，保持 raw_record[i] 与标签 i 对齐。
            labels_record = np.array(
                source_labels[span.start : span.stop], dtype=np.int64, copy=True
            )
            # 既有里程碑 8 函数在当前记录内部生成“原样本 + 偏移样本”。
            # offset_samples 决定最大位移，seed + record_index 使各记录独立可复现。
            record_raw, record_labels = offset_resample_record(
                raw_record,
                labels_record,
                offset=offset_samples,
                seed=seed + record_index,
            )

            # 当前记录输出区间是 [cursor:stop)；长度必须等于过采样后标签数。
            stop = cursor + len(record_labels)
            # raw 和 label 写入同一区间，因此新缓存中相同全局索引仍表示同一 epoch。
            balanced_raw[cursor:stop] = record_raw
            balanced_labels[cursor:stop] = record_labels

            # 原始 train/wave.npy 没有新增偏移样本，不能直接复制。这里逐个读取
            # record_raw 的 [1,3000] epoch，并为同一输出索引重新计算 [1,30,60] CWT。
            for local_index, raw_epoch in enumerate(record_raw):
                # np.asarray 统一测试替身和正式 CWT 的返回类型，并在写盘前转为 float32；
                # open_memmap 随后按 balanced_wave 的 float16 存储类型完成转换。
                wave_epoch = np.asarray(wave_transform(raw_epoch), dtype=np.float32)
                # cursor + local_index 把记录内索引映射为全局索引，与 raw/label 对齐。
                balanced_wave[cursor + local_index] = wave_epoch

            # 保存新旧边界和本记录 seed，既支持 Dataset 防跨记录索引，也支持复现实验。
            records.append(
                {
                    "record_id": span.record_id,
                    "start": cursor,
                    "stop": stop,
                    "source_start": span.start,
                    "source_stop": span.stop,
                    "seed": seed + record_index,
                }
            )
            # 下一条记录必须从当前 stop 开始；更新后所有记录在新缓存中首尾相接。
            cursor = stop

        # flush 把操作系统尚未落盘的修改写入三个 .npy 文件；只有三者都完成，
        # 后面写出的 manifest 才能描述一组已经实际存在的对齐缓存。
        balanced_raw.flush()
        balanced_wave.flush()
        balanced_labels.flush()
    finally:
        # 无论循环成功还是 CWT 中途报错，都关闭输入和输出 memmap。
        # Windows 会锁住仍映射的文件，不关闭会妨碍步骤四重建或清理缓存。
        for array in (
            balanced_raw,
            balanced_wave,
            balanced_labels,
            source_raw,
            source_labels,
        ):
            # 只对真正的 memmap 且仍有底层映射的对象调用 close，避免关闭普通 ndarray。
            if isinstance(array, np.memmap) and array._mmap is not None:
                array._mmap.close()

    # manifest 把三个数组的共同样本数、形状、dtype、过采样参数和记录边界固化。
    # shape 中的 NumPy 整数转成 Python int，确保 json.dump 可以直接序列化。
    manifest = {
        "format_version": 1,
        "split": "pretrain_train",
        "sample_count": total_epochs,
        "raw_shape": [int(size) for size in raw_shape],
        "raw_dtype": "float32",
        "wave_shape": [int(size) for size in wave_shape],
        "wave_dtype": "float16",
        "labels_shape": [total_epochs],
        "labels_dtype": "int64",
        "offset_samples": int(offset_samples),
        "base_seed": int(seed),
        "records": records,
    }
    # manifest_path 是本函数唯一返回值；下游以它确认 pretrain_train 缓存位置。
    manifest_path = output_dir / "manifest.json"
    # ensure_ascii=False 保留可读记录名，indent=2 生成便于人工审查的结构化文件。
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    # 返回路径而不是把大型数组返回到内存；训练阶段会由 Dataset 再按需映射读取。
    return manifest_path
```

```python
def run_epoch(
    model: nn.Module,
    batches: Iterable[tuple[Sequence[torch.Tensor], torch.Tensor]],
    device: str | torch.device,
    optimizer: Optimizer | None = None,
) -> float:
    """运行一个训练或验证 epoch，并返回所有预测位置的平均损失。

    参数来源：
        model: 当前阶段要运行的模型。它由 ``run_two_stage_training()`` 创建，
            并经 ``fit_stage()`` 传入；可能是 raw CNN、wave CNN 或融合模型。
        batches: 当前阶段的 DataLoader/可迭代 batch。它由
            ``run_full_training()`` 建立，再由 ``fit_stage()`` 选择 train 或
            validation loader 传入。每项是 ``(inputs, targets)``：单分支
            ``inputs`` 含一个张量，融合分支含 raw、wave 两个张量；``targets``
            在预训练时为 ``[B]``，融合训练时为 ``[B,T]``。
        device: 本轮使用的 CPU/GPU，来自 ``config.device`` 经
            ``_resolve_device()`` 解析后的 ``torch.device``。
        optimizer: 训练轮由 ``run_two_stage_training()`` 创建并经 ``fit_stage()``
            传入；验证轮显式传 ``None``。既有 ``run_classification_step()``
            正是通过它是否为 ``None`` 决定反向传播还是只做前向验证。

    返回去向：
        返回整个 epoch 按标签位置数加权的平均交叉熵。``fit_stage()`` 分别把它
        记为 ``train_loss`` 或 ``validation_loss``；后者还用于选择 ``best.pt``。
    """

    # model 可能刚在 CPU 上创建，而 batch 会被移到 device；先把参数和缓冲区
    # 移到同一设备，后续前向计算才不会出现 CPU/GPU 混用错误。
    model.to(device)

    # weighted_loss 保存 epoch 损失公式的分子：Σ(batch平均损失 × batch位置数)。
    # 它必须从 0.0 开始，才能把每个 batch 对总损失的贡献逐次相加。
    weighted_loss = 0.0
    # position_count 保存同一公式的分母：本 epoch 已处理的标签位置总数。
    # 独立计数是因为最后一个 batch 可能较小，不能用 batch 数代替样本位置数。
    position_count = 0
    # correct_count 保存同一批位置中预测正确的数量，用于显示累计分类准确率。
    correct_count = 0
    # DataLoader 每次产生一个 batch；循环结束才表示 train 或 validation 的整轮完成。
    for inputs, targets in batches:
        # inputs 保持模型调用顺序：(raw,)、(wave,) 或 (raw, wave)。逐个移动张量
        # 而不把它们堆叠，才能继续由 run_classification_step 用 model(*inputs) 解包。
        moved_inputs = tuple(tensor.to(device) for tensor in inputs)
        # targets 与模型输出必须位于同一设备；形状仍保持 [B] 或 [B,T]，
        # 后续分类步骤会把预测位置和标签统一展平，而这里不改变其时间语义。
        moved_targets = targets.to(device)

        # 复用里程碑 8 的单 batch 核心：它设置 train/eval 状态、计算交叉熵，
        # 并在 optimizer 不为 None 时执行清梯度、反向传播和参数更新。
        batch_loss, batch_correct_count = run_classification_step(
            model,
            moved_inputs,
            moved_targets,
            optimizer=optimizer,
            return_correct_count=True,
        )

        # moved_targets.numel() 是当前 batch 的真实预测位置数：预训练为 B，
        # 融合训练为 B×T。int() 把零维计数转换成普通 Python 整数用于累加。
        batch_positions = int(moved_targets.numel())
        # batch_loss 是“当前 batch 的平均值”。先乘 batch_positions 还原为该 batch
        # 所有位置的损失和，再加入分子，避免小 batch 与满 batch 获得相同权重。
        weighted_loss += batch_loss * batch_positions
        # 同时把本 batch 的位置数加入分母；最终分子和分母覆盖完全相同的数据。
        position_count += batch_positions
        # batch_correct_count 来自同一次模型前向，不会为显示 ACC 重复计算模型。
        correct_count += batch_correct_count
        # fit_stage 传入 tqdm 包装后的 Loader 时，实时显示截至当前 batch 的
        # 累计 loss/ACC；普通 list 等 iterable 没有 set_postfix，因此保持原行为。
        set_postfix = getattr(batches, "set_postfix", None)
        if callable(set_postfix):
            set_postfix(
                loss=f"{weighted_loss / position_count:.4f}",
                acc=f"{100.0 * correct_count / position_count:.2f}%",
                refresh=False,
            )

    # 循环结束后，用所有位置的损失和除以所有位置数，得到整个 epoch 的平均损失。
    # 返回普通 float，便于 fit_stage 比较 validation loss 并写入 JSON history。
    return weighted_loss / position_count


def save_stage_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any,
    stage: str,
    epoch: int,
    best_validation_loss: float,
    history: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> None:
    """保存能够继续当前训练阶段的完整 checkpoint。

    参数来源：
        path: ``fit_stage()`` 生成的 ``best.pt`` 或 ``last.pt`` 路径。
        model: ``fit_stage()`` 当前正在训练的阶段模型。
        optimizer: ``run_two_stage_training()`` 为该模型建立、并已训练到当前轮的
            Adam；其动量状态决定恢复后的下一次更新。
        scheduler: 与 optimizer 配套的 ExponentialLR；其内部轮次决定恢复后的
            下一轮学习率。
        stage: ``fit_stage()`` 的阶段名，如 ``raw_pretrain``。
        epoch: ``fit_stage()`` 当前循环中刚刚完成的零起始轮次。
        best_validation_loss: 截至当前轮的最小 validation loss，由
            ``fit_stage()`` 持续维护，恢复后继续用于判断是否覆盖 best。
        history: 从第 0 轮到当前轮的日志行，由 ``fit_stage()`` 逐轮追加。
        config: ``FullTrainingConfig.to_metadata()`` 产生的普通字典，用来说明
            这个 checkpoint 对应的数据路径、seed、epoch 数和学习率等实验配置。

    返回去向：
        没有业务返回值；副作用是把字典写入 ``path``。``load_stage_checkpoint()``
        稍后读取它，用于断点续训或加载 validation-best 模型做权重迁移。
    """

    # 调用者既可传字符串也可传 Path；统一后才能稳定取得 parent 目录。
    path = Path(path)
    # best/last 所在的阶段目录可能尚不存在，因此在写文件前创建完整父目录。
    path.parent.mkdir(parents=True, exist_ok=True)
    # checkpoint 只保存 state_dict 和普通元数据，不保存完整 Python 模型对象。
    # 这样恢复时仍由当前代码创建模型结构，再把参数和训练状态装回去。
    torch.save(
        {
            # 模型参数和 BatchNorm 缓冲区决定恢复后的前向结果。
            "model_state_dict": model.state_dict(),
            # optimizer 状态包含参数组、当前学习率和 Adam 一/二阶动量。
            "optimizer_state_dict": optimizer.state_dict(),
            # scheduler 状态保存已衰减到哪一轮，避免恢复后学习率轨迹重置。
            "scheduler_state_dict": scheduler.state_dict(),
            # stage 让一个 checkpoint 自己说明属于 raw、wave 还是 fusion 阶段。
            "stage": stage,
            # int() 固化刚完成的零起始轮次；fit_stage 恢复时从 epoch + 1 开始。
            "epoch": int(epoch),
            # float() 把可能的 NumPy/张量标量变成可直接比较和记录的 Python 值。
            "best_validation_loss": float(best_validation_loss),
            # 每行复制成普通 dict，避免把调用方可变 Mapping 对象直接留在快照中。
            "history": [dict(row) for row in history],
            # 配置也复制为普通 dict，使 checkpoint 保留创建它的实验上下文。
            "config": dict(config),
            # list() 固化类别索引顺序，防止日后解释 logits 时标签名称错位。
            "label_names": list(LABEL_NAMES),
        },
        # torch.save 把上面的单个字典序列化到 best.pt 或 last.pt。
        path,
    )


def load_stage_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """读取 checkpoint，并把其中需要的训练状态装回现有对象。

    参数来源：
        path: 断点续训时来自 ``fit_stage(resume_from=...)``；预训练结束加载 best
            时来自 ``StageTrainingResult.best_checkpoint``。
        model: 调用方已经按当前代码创建的模型。模型总是恢复，因为续训和权重
            迁移都依赖它。
        optimizer: 只有断点续训时由 ``fit_stage()`` 传入；只加载 best 做迁移时
            为 ``None``，因为旧 Adam 动量不会进入融合阶段的新 optimizer。
        scheduler: 与 optimizer 同理，只在继续同一训练阶段时传入。
        map_location: checkpoint 张量加载到的设备；由当前训练 device 传入，默认
            CPU 便于在没有原训练 GPU 的机器上读取。

    返回去向：
        返回完整 checkpoint 字典。``fit_stage()`` 从中取得已完成 epoch、旧 best
        和 history；``run_two_stage_training()`` 只利用已经装入 model 的 best 权重。
    """

    # Path() 统一路径类型；map_location 决定文件中的张量先落到 CPU 还是当前 GPU。
    checkpoint = torch.load(Path(path), map_location=map_location)
    # 模型状态在续训和“只加载 best”两种路径都必须恢复，因此无条件执行。
    model.load_state_dict(checkpoint["model_state_dict"])
    # optimizer 只在继续同一阶段时传入；恢复它才能延续 Adam 动量和参数组学习率。
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # scheduler 也只在续训时恢复，使下一次 step() 接着旧轮次衰减而不是从头开始。
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    # 把元数据交还调用方，因为恢复游标、旧 best 和 history 不属于 model 状态。
    return checkpoint


def fit_stage(
    stage: str,
    model: nn.Module,
    train_loader: Iterable[tuple[Sequence[torch.Tensor], torch.Tensor]],
    validation_loader: Iterable[tuple[Sequence[torch.Tensor], torch.Tensor]],
    optimizer: Optimizer,
    scheduler: Any,
    epochs: int,
    device: str | torch.device,
    output_dir: str | Path,
    config: Mapping[str, Any],
    resume_from: str | Path | None = None,
) -> StageTrainingResult:
    """完成一个训练阶段的 epoch 循环、选优、日志和断点恢复。

    参数来源：
        stage: ``run_two_stage_training()`` 传入的阶段名，决定输出子目录和日志标识。
        model: 当前阶段模型，由 ``run_two_stage_training()`` 创建或由测试注入。
        train_loader: ``run_full_training()`` 创建的对应训练 DataLoader；raw/wave
            阶段读取平衡 ``pretrain_train``，fusion 阶段读取原始连续 train。
        validation_loader: 同阶段的 validation DataLoader，始终保持真实类别分布。
        optimizer: 当前阶段的 Adam，由 ``run_two_stage_training()`` 创建。
        scheduler: 与 optimizer 配套的 ExponentialLR，也由上游创建。
        epochs: 该阶段计划训练的总轮数，来自 FullTrainingConfig 的对应字段；
            它是“最终应完成的总轮数”，不是恢复后额外再训练的轮数。
        device: ``config.device`` 经 ``_resolve_device()`` 解析后的实际设备；
            恢复 optimizer 前先把 model 移到这里。
        output_dir: ``FullTrainingConfig.output_dir``，三个阶段都在其下建独立目录。
        config: ``FullTrainingConfig.to_metadata()`` 的结果，随 checkpoint 保存。
        resume_from: 可选的 ``last.pt`` 路径，来自步骤四的恢复选择；首次训练为 None。

    返回去向：
        返回 ``StageTrainingResult``，其中包含 best/last/history 路径和内存中的完整
        history。``run_two_stage_training()`` 用预训练结果的 best 路径重新加载权重，
        步骤四用三个阶段结果定位最终模型和训练曲线。
    """

    # 每个阶段使用独立目录，避免 raw、wave、fusion 的 best/last 文件互相覆盖。
    stage_dir = Path(output_dir) / stage
    # best_path 只在 validation loss 严格改善时覆盖，用于模型选择和后续迁移。
    best_path = stage_dir / "best.pt"
    # last_path 每轮覆盖，始终指向最近完整轮次，用于中断后继续训练。
    last_path = stage_dir / "last.pt"
    # history_path 保存人可读 JSON 曲线，不必打开二进制 checkpoint 才能画图。
    history_path = stage_dir / "history.json"
    # 首次进入阶段时目录尚不存在；parents=True 同时创建 output_dir 的缺失层级。
    stage_dir.mkdir(parents=True, exist_ok=True)

    # optimizer 在上游模型仍位于 CPU 时创建。load_state_dict() 会根据 optimizer
    # 当前参数的设备安放 Adam 动量，所以必须在恢复前先移动模型。若延迟到
    # run_epoch()，模型参数虽会到 CUDA，已恢复的 exp_avg/exp_avg_sq 仍留在 CPU，
    # 第一个 optimizer.step() 就会因梯度和动量不在同一设备而失败。
    model.to(device)

    # start_epoch 是本次调用应开始训练的零起始轮次。首次训练从 0 开始；
    # 若恢复 checkpoint，下面会改成“最后已完成 epoch + 1”。
    start_epoch = 0
    # 初始 best 设为正无穷，保证首次得到的有限 validation loss 一定成为 best。
    best_validation_loss = float("inf")
    # history 保存每个已完成 epoch 的损失和学习率；首次训练从空列表开始追加。
    history: list[dict[str, Any]] = []
    # resume_from 不为 None 表示继续同一阶段；模型、优化器、scheduler 和日志
    # 必须来自同一个 last checkpoint，才能保持训练轨迹连续。
    if resume_from is not None:
        checkpoint = load_stage_checkpoint(
            resume_from,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )
        # checkpoint["epoch"] 记录的是已完成轮次，所以加 1 才是下一个待训练轮次。
        start_epoch = int(checkpoint["epoch"]) + 1
        # 继承历史最小 validation loss，防止恢复后较差模型覆盖恢复前的 best.pt。
        best_validation_loss = float(checkpoint["best_validation_loss"])
        # 复制旧 history 后继续 append，最终曲线才包含中断前后的完整轮次。
        history = [dict(row) for row in checkpoint["history"]]

    # 阶段名是这组 epoch 的公共上下文，只在真正还有训练任务时显示一次。
    # 后续进度行便可从 epoch 开始，不必在几十轮中重复 raw/wave/fusion 名称。
    if start_epoch < epochs:
        stage_title = stage.replace("_", " ").upper()
        tqdm.write(f"\n{'=' * 20} {stage_title} {'=' * 20}")

    # range 的终点 epochs 不包含在内。例如从 2 恢复、epochs=5 时只训练 2/3/4。
    for epoch in range(start_epoch, epochs):
        # optimizer 可能有多个参数组：融合新层和两个预训练分支使用不同学习率。
        # 在 scheduler.step() 之前记录，得到的是“本轮实际用于更新”的学习率。
        learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
        # DataLoader 的 len() 是本轮 batch 总数；进度条每完成一个 batch 自动推进。
        with tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs} | Train",
            unit="batch",
            ncols=100,
            disable=None,
        ) as train_batches:
            # 传入 optimizer 表示训练轮；每个 batch 都会执行一次参数更新。
            train_loss = run_epoch(
                model, train_batches, device=device, optimizer=optimizer
            )

        # validation 同样按 batch 前向，但 optimizer=None，因此不会更新模型。
        with tqdm(
            validation_loader,
            desc=f"Epoch {epoch + 1}/{epochs} | Valid",
            unit="batch",
            ncols=100,
            disable=None,
        ) as validation_batches:
            validation_loss = run_epoch(
                model, validation_batches, device=device, optimizer=None
            )
        # 一行 history 把轮次、两类损失和本轮学习率绑定在一起，供恢复和画曲线。
        # epoch/loss 已经是普通 Python 数值，learning_rates 也是 float 列表，可写 JSON。
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rates": learning_rates,
            }
        )
        # 当前轮的训练和验证都已经完成，现在把 scheduler 推进到“下一轮”的学习率。
        # 随后保存 checkpoint，恢复时 optimizer/scheduler 就可直接开始下一轮。
        scheduler.step()

        # 只有 validation loss 严格小于历史 best 才更新模型选择结果；相等时保留
        # 更早 checkpoint，使选择规则确定。test split 从未进入这个比较。
        if validation_loss < best_validation_loss:
            # 先更新数值，再把新 best 连同该数值写入 best.pt。
            best_validation_loss = validation_loss
            save_stage_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                stage,
                epoch,
                best_validation_loss,
                history,
                config,
            )

        # 无论本轮是否改善 validation，都覆盖 last.pt；断点恢复关心“最近完成状态”，
        # 而不是只关心最佳泛化状态。
        save_stage_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            stage,
            epoch,
            best_validation_loss,
            history,
            config,
        )
        # history.json 与 last.pt 同步更新，训练中断后磁盘上仍有截至本轮的可读日志。
        with history_path.open("w", encoding="utf-8") as file:
            json.dump(history, file, ensure_ascii=False, indent=2)

    # 返回路径而不是重新读取 checkpoint；调用方可按用途选择 best 或 last。
    # tuple(history) 把最终日志暴露为不可变序列，避免调用方意外修改内部列表。
    return StageTrainingResult(
        stage=stage,
        best_checkpoint=best_path,
        last_checkpoint=last_path,
        history_path=history_path,
        history=tuple(history),
    )


def _resolve_device(device: str) -> torch.device:
    """把实验配置中的设备字符串解析为 PyTorch 设备对象。

    参数来源：
        device: 来自 ``FullTrainingConfig.device``；默认 ``"auto"``，也可由步骤四
            预检明确指定 ``"cpu"`` 或 ``"cuda:0"``。

    返回去向：
        返回 ``torch.device`` 给 ``run_two_stage_training()``，再传入
        ``fit_stage()``、``run_epoch()`` 和 checkpoint 加载函数统一控制设备。
    """

    # auto 表示由当前机器决定：CUDA 可用时使用第一块 GPU，否则回退到 CPU。
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 非 auto 时尊重调用者明确给出的设备字符串，并统一包装成 torch.device。
    return torch.device(device)


def run_two_stage_training(
    config: FullTrainingConfig,
    loaders: Mapping[str, DataLoader],
    raw_model: nn.Module | None = None,
    wave_model: nn.Module | None = None,
    fusion_model: nn.Module | None = None,
    resume_checkpoints: Mapping[str, str | Path] | None = None,
) -> dict[str, StageTrainingResult]:
    """连接三个阶段：raw/wave 预训练、best 权重迁移和融合微调。

    参数来源：
        config: 步骤四创建的 ``FullTrainingConfig``；正常入口由
            ``run_full_training()`` 传入，提供设备、轮数、学习率、输出目录等。
        loaders: ``run_full_training()`` 创建的六个 DataLoader 映射。固定键分别是
            raw/wave/fusion 三阶段的 train 与 validation。
        raw_model: 可选 raw CNN。正式运行不传，由本函数创建
            ``SleepEDFRawFeatureNet``；行为测试可注入微型模型。
        wave_model: 与 raw_model 同理，正式运行创建 ``SleepEDFWaveFeatureNet``。
        fusion_model: 可选融合模型；正式运行创建 ``FFTCNFusionTCN``，测试可注入。
        resume_checkpoints: 可选的阶段名到 ``last.pt`` 路径映射，由步骤四恢复入口
            提供；首次训练为 None，缺少某阶段键就从该阶段第 0 轮开始。

    返回去向：
        返回三个阶段名到 ``StageTrainingResult`` 的映射。``run_full_training()``
        原样返回给步骤四；其中 fusion 的 best checkpoint 用于最终独立 test 评估。
    """

    # 把 config.device 的人类配置解析为统一 torch.device，供三个阶段共同使用。
    device = _resolve_device(config.device)
    # checkpoint 需要普通字典而不是 dataclass/Path；to_metadata 同时固化全部实验参数。
    metadata = config.to_metadata()
    # 将 None 归一化为空映射，后面每个阶段都可直接 .get() 而无需重复分支判断。
    resume_checkpoints = {} if resume_checkpoints is None else resume_checkpoints
    # 正式运行按固定结构创建三个模型；仅当测试或调用方显式传入时才复用现有对象。
    raw_model = SleepEDFRawFeatureNet() if raw_model is None else raw_model
    wave_model = SleepEDFWaveFeatureNet() if wave_model is None else wave_model
    fusion_model = FFTCNFusionTCN() if fusion_model is None else fusion_model

    # ---- 阶段 1：raw CNN 单 epoch 五分类预训练 ----
    # pretrain() 打开 raw 分支自己的五分类头，使前向输出 [B,5] 而不是 256 维特征。
    raw_model.pretrain()
    # 预训练整个 raw CNN；初始学习率来自统一 pretrain_learning_rate 配置。
    raw_optimizer = torch.optim.Adam(
        raw_model.parameters(), lr=config.pretrain_learning_rate
    )
    # 每完成一轮，ExponentialLR 把所有参数组学习率乘 scheduler_gamma。
    raw_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        raw_optimizer, gamma=config.scheduler_gamma
    )
    # fit_stage 使用平衡 raw train、真实分布 raw validation，并负责 best/last/history。
    raw_result = fit_stage(
        "raw_pretrain",
        raw_model,
        loaders["raw_pretrain_train"],
        loaders["raw_pretrain_validation"],
        raw_optimizer,
        raw_scheduler,
        config.raw_pretrain_epochs,
        device,
        config.output_dir,
        metadata,
        resume_checkpoints.get("raw_pretrain"),
    )
    # fit_stage 结束时内存模型停在最后一轮，不一定是 validation 最佳轮；
    # 因此从 raw_result.best_checkpoint 重新装入模型参数，再准备迁移。
    load_stage_checkpoint(raw_result.best_checkpoint, raw_model, map_location=device)

    # ---- 阶段 2：wave CNN 单 epoch五分类预训练 ----
    # wave 分支与 raw 分支相互独立，使用自己的模型、optimizer、scheduler 和日志目录。
    # pretrain() 打开 wave 分支五分类头，使每个 CWT epoch 输出 [B,5] logits。
    wave_model.pretrain()
    # 为 wave CNN 单独创建 Adam，避免与 raw optimizer 共享参数或动量状态。
    wave_optimizer = torch.optim.Adam(
        wave_model.parameters(), lr=config.pretrain_learning_rate
    )
    # wave scheduler 只跟踪 wave optimizer，按相同 gamma 独立衰减。
    wave_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        wave_optimizer, gamma=config.scheduler_gamma
    )
    # fit_stage 使用平衡 wave train、真实分布 wave validation，并写入 wave 子目录。
    wave_result = fit_stage(
        "wave_pretrain",
        wave_model,
        loaders["wave_pretrain_train"],
        loaders["wave_pretrain_validation"],
        wave_optimizer,
        wave_scheduler,
        config.wave_pretrain_epochs,
        device,
        config.output_dir,
        metadata,
        resume_checkpoints.get("wave_pretrain"),
    )
    # 同样丢弃内存中的 final 状态，明确恢复由 validation 选出的 wave best。
    load_stage_checkpoint(wave_result.best_checkpoint, wave_model, map_location=device)

    # ---- 阶段 3：迁移两个 best 分支并微调融合模型 ----
    # transfer_pretrained_features 按 state_dict 名称复制两个 best CNN，并调用各分支
    # finetune() 关闭预训练分类头，使它们分别输出每 epoch 256/216 维特征。
    transfer_pretrained_features(fusion_model, raw_model, wave_model)
    # 新 TCN/分类头用 fusion_learning_rate；预训练分支只用其乘 scale 的较小学习率。
    fusion_optimizer = build_finetune_optimizer(
        fusion_model,
        config.fusion_learning_rate,
        config.feature_learning_rate_scale,
    )
    # fusion optimizer 含三个学习率参数组；同一个 scheduler 按相同比例衰减三组。
    fusion_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        fusion_optimizer, gamma=config.scheduler_gamma
    )
    # fusion fit_stage 使用原始 T=50 连续 train/validation，不使用过采样序列或 test。
    fusion_result = fit_stage(
        "fusion_finetune",
        fusion_model,
        loaders["fusion_finetune_train"],
        loaders["fusion_finetune_validation"],
        fusion_optimizer,
        fusion_scheduler,
        config.fusion_finetune_epochs,
        device,
        config.output_dir,
        metadata,
        resume_checkpoints.get("fusion_finetune"),
    )

    # 保留三个阶段结果，既能追溯两个分支的来源，也能让步骤四定位 fusion best。
    return {
        "raw_pretrain": raw_result,
        "wave_pretrain": wave_result,
        "fusion_finetune": fusion_result,
    }


def run_full_training(
    config: FullTrainingConfig,
    resume_checkpoints: Mapping[str, str | Path] | None = None,
) -> dict[str, StageTrainingResult]:
    """从磁盘缓存建立六个 DataLoader，并启动完整两步训练。

    参数来源：
        config: 步骤四在显存、耗时和恢复预检通过后创建的
            ``FullTrainingConfig``。其中 ``data_cache_dir`` 已包含 train、
            validation 和本步骤生成的 pretrain_train 缓存。
        resume_checkpoints: 步骤四可选的“阶段名 -> last.pt”映射；
            首次训练为 None。这里只把它透传给
            ``run_two_stage_training()``，不在 Dataset/DataLoader 层加载断点。

    返回去向：
        返回 ``run_two_stage_training()`` 产生的三个阶段结果，供步骤四加载
        fusion validation-best checkpoint、执行一次正式 test 并生成报告。

    运行边界：
        本函数只创建 train/validation Loader，绝不创建 test Loader；test 只能在
        所有模型选择完成后由步骤四单独读取。
    """

    # datasets 用固定名称把“阶段 + train/validation”映射到实际磁盘数据。
    # 后面的 batch_sizes、loaders 和 run_two_stage_training 都复用这些相同名称。
    datasets = {
        # raw 预训练的训练数据来自逐记录过采样后的 pretrain_train；T=1 表示
        # 每个 Dataset 样本是单个 [1,3000] epoch，input_mode=raw 只返回 raw。
        "raw_pretrain_train": SleepEDFSequenceDataset(
            config.data_cache_dir, "pretrain_train", 1, "raw"
        ),
        # raw validation 使用未过采样的真实分布，防止平衡数据影响模型选择。
        "raw_pretrain_validation": SleepEDFSequenceDataset(
            config.data_cache_dir, "validation", 1, "raw"
        ),
        # wave 预训练读取与平衡 raw 同索引生成的 [1,30,60] CWT，同样使用 T=1。
        "wave_pretrain_train": SleepEDFSequenceDataset(
            config.data_cache_dir, "pretrain_train", 1, "wave"
        ),
        # wave validation 读取原始 validation CWT，仍不进行任何类别过采样。
        "wave_pretrain_validation": SleepEDFSequenceDataset(
            config.data_cache_dir, "validation", 1, "wave"
        ),
        # 融合训练回到未过采样 train；both 按 (raw,wave) 返回同一记录内的
        # T=config.sequence_length 个连续 epoch，默认 T=50。
        "fusion_finetune_train": SleepEDFSequenceDataset(
            config.data_cache_dir, "train", config.sequence_length, "both"
        ),
        # 融合 validation 使用相同 T 和输入顺序，但数据来自独立 validation split。
        "fusion_finetune_validation": SleepEDFSequenceDataset(
            config.data_cache_dir, "validation", config.sequence_length, "both"
        ),
    }
    # 三类模型的显存占用不同，因此配置分别提供 raw、wave、fusion batch size。
    # train 与对应 validation 使用同一 batch size，名称必须与 datasets 完全一致。
    batch_sizes = {
        "raw_pretrain_train": config.raw_batch_size,
        "raw_pretrain_validation": config.raw_batch_size,
        "wave_pretrain_train": config.wave_batch_size,
        "wave_pretrain_validation": config.wave_batch_size,
        "fusion_finetune_train": config.fusion_batch_size,
        "fusion_finetune_validation": config.fusion_batch_size,
    }
    # loaders 最终交给 run_two_stage_training；先建空字典，再逐个 Dataset 配置，
    # 比字典推导式更清楚地展示训练集 shuffle、验证集不 shuffle 的分支。
    loaders: dict[str, DataLoader] = {}
    for name, dataset in datasets.items():
        # 六个名称中只有训练入口以 _train 结尾。训练打乱能改变 batch 组合；
        # validation 保持固定顺序，使每轮比较不混入无意义的顺序变化。
        shuffle = name.endswith("_train")
        # build_reproducible_loader 内部使用 config.seed 创建局部 torch.Generator，
        # 因而相同配置可复现 shuffle 顺序，又不会修改全局 PyTorch 随机状态。
        loaders[name] = build_reproducible_loader(
            dataset,
            batch_size=batch_sizes[name],
            shuffle=shuffle,
            seed=config.seed,
            num_workers=config.num_workers,
        )

    # Dataset 持有输入缓存的 memmap。用 try/finally 保证训练成功、报错或中断时
    # 都进入释放路径，尤其避免 Windows 长时间锁住约 7.19 GiB 缓存文件。
    try:
        # 把完整配置和六个 Loader 交给两步训练编排；返回值原样交给步骤四。
        return run_two_stage_training(
            config,
            loaders,
            resume_checkpoints=resume_checkpoints,
        )
    finally:
        # 每个 Dataset 都独立打开 raw/wave/labels memmap，因此必须逐个调用 close。
        for dataset in datasets.values():
            dataset.close()
```

## 5. 用户编码范围与统一自测

请在 `training/sleep_edf_full_run.py` 中保留已经完成的配置和 Dataset/DataLoader，一次完成步骤三新增接口：

1. `_balanced_record_count()` 与 `build_balanced_pretrain_cache()`；
2. `run_epoch()`；
3. `save_stage_checkpoint()` 与 `load_stage_checkpoint()`；
4. `fit_stage()`；
5. `_resolve_device()`；
6. `run_two_stage_training()`；
7. `run_full_training()`。

整组完成后统一运行：

```powershell
python -m unittest tests.test_sleep_edf_training_orchestration -v
```

4 条测试分别验证逐记录平衡缓存、按位置加权的 epoch loss、validation 选优与断点恢复、best 权重迁移与两步训练顺序。测试使用微型数据，不生成全量缓存，也不启动长训练。

测试全部通过只表示步骤三行为自测完成，不代表整个里程碑 9B 已验收。

## 6. 常见错误与测试含义

- 对拼接后的全局 train 一次过采样：偏移窗口可能跨越两条夜晚记录；
- 用原始 `wave.npy` 配新增过采样标签：wave 与 raw/label 不再描述同一 epoch；
- 直接平均各 batch loss：最后一个小 batch 权重过大；
- 用 final 分支权重做迁移：绕过 validation 模型选择；
- 只保存模型参数：Adam 动量、scheduler 进度和 history 在恢复后丢失；
- 从已保存的 epoch 再训练一次：恢复轨迹重复一轮；
- 在步骤三建立 test Loader：test 会提前进入模型选择过程；
- 训练异常后不关闭 Dataset：Windows 可能继续锁住大缓存文件。
