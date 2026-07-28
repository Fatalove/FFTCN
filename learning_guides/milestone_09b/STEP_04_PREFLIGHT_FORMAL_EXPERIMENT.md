# 里程碑 9B 内部步骤四：预检与正式实验

本步骤仍属于里程碑 9B，完成这里的局部自测不等于 9B 已验收。正式验收仍要等全量缓存、预检、固定 seed 0 完整训练、一次正式 test、报告和全部回归共同完成。

## 1. 本步骤的三类范围

### 用户练习的核心代码

你需要在 `training/sleep_edf_formal_experiment.py` 中实现 6 个函数：

1. `seed_experiment()`：固定一次实验涉及的随机源；
2. `build_all_caches()`：串联三个基础 split 和平衡 `pretrain_train` 缓存；
3. `overfit_single_batch()`：重复同一 batch，验证前向、损失、反向和更新链；
4. `evaluate_fusion_best()`：模型选择完成后，只测试融合 validation-best；
5. `build_formal_report()`：保存指标、配置、曲线、耗时、显存和 checkpoint 来源；
6. `run_formal_experiment()`：连接完整训练、一次 test 和报告写入。

步骤三的恢复实现已经存在。本步骤只需要让 `run_full_training(..., resume_checkpoints=...)` 把映射透传给 `run_two_stage_training()`；该接口接线已放入正式文件，不要求你重写训练循环。

### Codex 已准备的辅助材料

- 最小正式骨架与数据类型；
- 7 条行为测试；
- 独立参考答案；
- 本机磁盘、CWT 耗时和默认 batch 显存的初步测量；
- 全量运行前的固定检查顺序。

### 工程加固（选读，不进入当前核心练习）

命令行参数解析、缓存哈希、JSON Schema、损坏文件自动恢复、`persistent_workers` 和多进程 worker 随机种子都不进入本步骤主代码。它们不会改变当前固定运行链，加入后只会遮挡实验语义。

## 2. 问题分析：为什么不能直接开始 90 个 epoch

这一步不是再写一套训练器，而是给已经完成的训练器加一条可信的实验外壳。

### 2.1 固定运行链

```text
处理后 train/validation/test NPZ
  -> 三个基础 raw/labels/wave 缓存
  -> train 逐记录偏移过采样后的 pretrain_train 缓存
  -> 同一 batch 过拟合
  -> 三阶段各跑完整 epoch
  -> 从各阶段 last.pt 恢复到下一轮
  -> seed 0：raw 20 轮 -> wave 20 轮 -> 融合 50 轮
  -> 取 fusion_finetune/best.pt（只由 validation loss 选出）
  -> 创建 test Dataset 并正式评估一次
  -> formal_test_report.json
```

单 batch 可更新、单 epoch 可运行和完整训练结果是三类证据：前两项只能排除局部运行错误，不能替代完整训练及其正式指标。

### 2.2 为什么缓存必须按固定顺序构建

每个基础 split 都先由 `build_raw_label_cache()` 写出 `raw.npy`、`labels.npy` 和 `manifest.json`，再由 `build_wave_cache()` 从这份 raw 逐项计算 `wave.npy`。这样相同索引 `i` 下的 raw、wave 和 label 才描述同一个 epoch。

`pretrain_train` 不能复用基础 train 的 wave：偏移过采样已经改变了 raw 窗口，旧 wave 仍描述偏移前的信号。必须先在每条记录内部得到“原 raw + 偏移 raw”，再由平衡后的每个 raw 重新计算 wave，否则两个分支会在同一索引读取不同的 EEG 片段。

### 2.3 同一 batch 过拟合在检查什么

过拟合预检不是测泛化能力，而是隔离训练链：固定输入和标签后，检查前向、损失、反向传播和优化器更新能否共同让损失下降。每轮从 DataLoader 取新 batch 会同时改变数据和模型参数，无法判断损失变化究竟来自哪一部分。

因此预检要保存完整 `losses`，并比较末值与首值。真实网络含 dropout，所以只要求总体下降，不要求相邻两步都严格下降。

### 2.4 `last.pt`、`best.pt` 与 test 的边界

`last.pt` 保存最近一轮的训练状态，供中断恢复继续使用；`best.pt` 是 validation loss 选出的模型，供正式 test 使用。二者可能来自不同 epoch，不能互换。

test Dataset 只能在 validation 已经完成模型选择后创建。正式 test 只评估融合 `best.pt` 一次，不能根据 test 指标重新挑 epoch、换 checkpoint 或重跑后择优报告。

### 2.5 为什么缓存和训练耗时要分开

缓存构建主要花在 raw 整理与 CWT 预计算，训练耗时主要花在参数优化。报告若只保存二者之和，就无法判断时间消耗来自数据准备还是模型训练，因此必须分别记录 `cache_build_seconds` 和 `training_seconds`。

### 2.6 当前本机初步资源事实

2026-07-22 的只读预检显示：

- D 盘约有 75.69 GiB 可用；全部新缓存预计约 10.0 GiB，其中平衡 `pretrain_train` 约 7.19 GiB，所以存储可接受。
- 20 个 CWT epoch 的测量约为 0.00541 秒/epoch；约 690,096 个 CWT 的粗略估计约 1.04 小时。它只是估算，正式报告记录真实耗时。
- RTX 3050 Laptop 4 GB 上，默认 batch 的单步峰值分配显存初测约为 raw 37.49 MiB、wave 209.71 MiB、fusion 2870.12 MiB。fusion 是正式预检重点。

这些数值只说明可以进入预检，不是全量实验已完成的证据。

## 3. 必要的 Python / NumPy / PyTorch 基础

### 3.1 三类随机源为什么要同时固定

`random.seed(seed)`、`np.random.seed(seed)` 和 `torch.manual_seed(seed)` 控制的是不同生成器。只固定其中一个，不能约束其余库的随机行为。CUDA 可用时还要调用 `torch.cuda.manual_seed_all(seed)`。

这里不额外强制 cuDNN 确定性算法，因为当前目标是遵循原仓库训练逻辑；DataLoader 的 shuffle 已由步骤二的局部 `torch.Generator` 固定。

### 3.2 `perf_counter()` 如何计算耗时

`perf_counter()` 是适合测量经过时间的单调时钟。先保存开始时刻，任务结束后再调用一次，两次结果相减就是该任务的耗时。缓存和训练各自保存起点，才能得到两个独立时间。

### 3.3 CUDA 峰值统计的含义

`reset_peak_memory_stats(device)` 清除此前运行留下的峰值记录；`max_memory_allocated(device)` 返回本次区间内 PyTorch 张量实际分配的最大字节数。除以 `1024**2` 才得到 MiB。CPU 没有这个指标，所以报告用 `None`，不是伪造 0 MiB。

### 3.4 路径和 NumPy 数值为什么要转换

`Path`、NumPy 标量和数组不能直接作为普通 JSON 字段。报告中的路径用 `str(path.resolve())`，时间和显存用 `float()`，曲线用普通 `list[dict]`。指标数组的转换继续交给已经验证过的 `build_evaluation_report()`。

## 4. 手工推演：用一个最小例子跟踪状态

这一节不再解释设计理由，而是用具体小数据跟踪代码运行后，各变量和产物怎样变化。

### 4.1 从基础 train 到 `pretrain_train`

假设 train 只有一条记录 `A`，包含三个 epoch：

```text
raw    = [A0, A1, A2]
labels = [ 0,  1,  0]
```

`build_raw_label_cache()` 先把它写入基础 train 缓存，并在 manifest 中记录：

```text
record A -> [0, 3)
```

`build_wave_cache()` 随后按相同索引计算：

| 索引 `i` | `train/raw[i]` | `train/wave[i]` | `train/labels[i]` |
|---:|---|---|---:|
| 0 | `A0` | `CWT(A0)` | 0 |
| 1 | `A1` | `CWT(A1)` | 1 |
| 2 | `A2` | `CWT(A2)` | 0 |

在这个例子中，两个类别的原始数量分别是 2 和 1，所以 `n_max=2`。类别 0 的两个样本都位于记录边界，没有安全的偏移中心，因此不新增；类别 1 的内部样本 `A1` 可以作为中心，源算法会有放回地抽取 `n_max=2` 次。假设固定 seed 下得到两个偏移窗口 `A1'`、`A1''`，则 `pretrain_train` 写成：

| 索引 `i` | `pretrain_train/raw[i]` | `pretrain_train/wave[i]` | `pretrain_train/labels[i]` |
|---:|---|---|---:|
| 0 | `A0` | `CWT(A0)` | 0 |
| 1 | `A1` | `CWT(A1)` | 1 |
| 2 | `A2` | `CWT(A2)` | 0 |
| 3 | `A1'` | `CWT(A1')` | 1 |
| 4 | `A1''` | `CWT(A1'')` | 1 |

因此偏移项的 wave 来自偏移后的 `A1'`、`A1''`，不是从基础 train 复制 `CWT(A1)`。

此时两个类别的最终数量是 2 和 3，并不严格相等。“平衡缓存”沿用阶段名称，实际遵循原仓库规则：保留全部原样本，并为每个存在安全内部中心的类别额外生成 `n_max` 个窗口。

### 4.2 逐轮跟踪同一 batch 的 `losses`

损失值由模型实际运行产生，不能脱离模型参数手算；这里手工跟踪的是每轮返回值怎样进入 `losses`。假设四次 `run_epoch(model, [batch], ..., optimizer)` 依次返回：

| 更新完成次数 | 当前返回的 `loss` | 执行 `losses.append(loss)` 后 |
|---:|---:|---|
| 0 | — | `[]` |
| 1 | 1.20 | `[1.20]` |
| 2 | 0.91 | `[1.20, 0.91]` |
| 3 | 0.70 | `[1.20, 0.91, 0.70]` |
| 4 | 0.56 | `[1.20, 0.91, 0.70, 0.56]` |

最后手动代入行为判据：

```text
losses[-1] < losses[0]
0.56       < 1.20
True
```

所以这个例子通过“同一 batch 的总体损失下降”预检。

### 4.3 从 validation 历史选出正式 test checkpoint

假设融合阶段历史为：

| epoch | validation loss |
|---:|---:|
| 0 | 0.80 |
| 1 | 0.65 |
| 2 | 0.72 |

逐项比较可知最小值是 `0.65`，对应 epoch 1，因此：

```text
fusion_finetune/best.pt -> epoch 1
fusion_finetune/last.pt -> epoch 2
```

若训练中断后恢复，应读取最新的 `last.pt`，从 epoch 2 之后继续；训练全部结束后的正式 test 则把 epoch 1 的 `best.pt` 传给 `evaluate_fusion_best()`。test 只产生这一组 ACC、Macro-F1、kappa、每类指标和混淆矩阵，不再改变 checkpoint 选择。

### 4.4 从起止时刻算出报告耗时

假设计时器返回：

```text
cache_start   = 100.0
cache_end     = 3700.0
training_start = 4000.0
training_end   = 22000.0  # 三阶段训练返回、正式 test 开始之前
```

分别做减法：

```text
cache_build_seconds = 3700.0 - 100.0  = 3600.0
training_seconds    = 22000.0 - 4000.0 = 18000.0
```

若训练和随后 test 的 CUDA 峰值为 `2870.12 MiB`，报告中的运行信息就是：

```python
runtime = {
    "cache_build_seconds": 3600.0,
    "training_seconds": 18000.0,
    "peak_cuda_memory_mib": 2870.12,
}
```

### 4.5 推演结果分别流向哪里

缓存构建和同 batch 过拟合属于正式训练前的外部预检，不是 `run_formal_experiment()` 内部自动执行的步骤：

```text
build_all_caches()
  -> CacheBuildResult.elapsed_seconds = 3600.0

对 raw、wave、fusion 分别运行 overfit_single_batch()
  -> 三个 OverfitResult
  -> 只用于确认损失下降和显存可接受
```

预检全部通过后，缓存耗时才作为参数交给正式入口；三个 `OverfitResult` 不进入正式报告：

```text
run_formal_experiment(cache_build_seconds=3600.0)
  -> seed_experiment(config.seed)
  -> run_full_training()
  -> 取 fusion_finetune/best.pt
  -> evaluate_fusion_best() 一次
  -> build_formal_report()
  -> formal_test_report.json
```

最终 `FormalExperimentResult` 保存三阶段结果、正式 test 指标、报告路径、训练耗时和峰值显存；大型缓存和模型张量不放入这个结果对象。

## 5. 带完整因果注释的核心代码

下面代码与独立参考答案保持相同接口和调用关系。正式练习文件不能导入参考答案；请理解运行链后自行写入骨架。

```python
from __future__ import annotations

# dataclass 保存有明确字段的运行结果；Path 统一处理文件系统路径。
from dataclasses import dataclass
from pathlib import Path
# platform/random/time 分别提供环境版本、Python 随机源和单调计时。
import platform
import random
from time import perf_counter
# Any/Callable/Mapping/Sequence 描述报告、CWT 回调、阶段映射和模型输入。
from typing import Any, Callable, Mapping, Sequence

# NumPy 控制数值随机源；PyTorch 控制模型、设备、优化器和 CUDA 显存。
import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

# 缓存链先生成基础 raw/labels，再从同一 raw 顺序生成 wave。
from data.sleep_edf_cwt import morlet_cwt_epoch
from data.sleep_edf_training_cache import build_raw_label_cache, build_wave_cache
# 评估链返回指标对象，并复用里程碑 9 的报告格式与 JSON 写入函数。
from evaluation.sleep_edf_metrics import (
    ClassificationMetrics,
    build_evaluation_report,
    evaluate_model,
    save_evaluation_report,
)
# 融合 validation-best 必须装入与训练时相同的重构模型结构。
from models.merge.sleep_edf_fusion_tcn import FFTCNFusionTCN
# 步骤四只组织已完成的缓存读取、epoch、checkpoint 与完整训练接口。
from training.sleep_edf_full_run import (
    FullTrainingConfig,
    SleepEDFSequenceDataset,
    StageTrainingResult,
    _resolve_device,
    build_balanced_pretrain_cache,
    build_reproducible_loader,
    load_stage_checkpoint,
    run_epoch,
    run_full_training,
)


# WaveTransform 表示“单个 raw epoch -> 单个 wave epoch”，并允许自测注入轻量替身。
WaveTransform = Callable[[np.ndarray], np.ndarray]
# TrainingBatch 与 Dataset/DataLoader 公开契约相同：inputs 是 1/2 个张量，targets 是标签。
TrainingBatch = tuple[Sequence[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class CacheBuildResult:
    """保存全量缓存构建产物的位置和耗时。"""

    # manifest_paths 记录 train/validation/test 基础缓存的元数据入口。
    manifest_paths: Mapping[str, Path]
    # balanced_manifest_path 单独指向只供分支预训练使用的 pretrain_train。
    balanced_manifest_path: Path
    # elapsed_seconds 记录整个缓存阶段耗时，后续写入正式报告。
    elapsed_seconds: float


@dataclass(frozen=True)
class OverfitResult:
    """保存重复训练同一 batch 时的损失轨迹与峰值显存。"""

    # losses[i] 是第 i 次更新时该固定 batch 的位置平均交叉熵。
    losses: tuple[float, ...]
    # CUDA 时记录当前预检的峰值分配显存；CPU 没有对应数值。
    peak_cuda_memory_mib: float | None


@dataclass(frozen=True)
class FormalExperimentResult:
    """保存三阶段训练、一次 test 与报告的正式结果。"""

    # stage_results 保留三个阶段的 best/last/history 产物。
    stage_results: Mapping[str, StageTrainingResult]
    # metrics 是融合 validation-best 在固定 test 上的唯一正式指标。
    metrics: ClassificationMetrics
    # report_path 指向已写入上述训练来源和指标的 JSON。
    report_path: Path
    # training_seconds 只计算三阶段训练，不混入事前缓存构建时间。
    training_seconds: float
    # peak_cuda_memory_mib 覆盖当次训练与随后 test 的峰值分配显存。
    peak_cuda_memory_mib: float | None


def seed_experiment(seed: int) -> None:
    """固定 Python、NumPy 和 PyTorch 的实验随机源。

    参数来源：
        seed: 来自 ``FullTrainingConfig.seed`` 的普通 Python 整数。
            ``run_formal_experiment()`` 在创建模型和 DataLoader 之前传入；
            独立预检脚本也使用同一值。

    返回去向：
        没有返回值。函数改变三类随机生成器的后续初始状态，
        使接下来的模型初始化、NumPy 操作和 PyTorch 采样都从同一
        seed 开始。
    """

    # random.seed 固定之后任何基于 Python random 的随机选择。
    random.seed(seed)
    # np.random.seed 固定使用 NumPy 全局生成器的后续操作。
    np.random.seed(seed)
    # torch.manual_seed 固定 CPU 随机状态，也是模型参数初始化的主入口。
    torch.manual_seed(seed)

    # 只在 CUDA 可用时固定所有 GPU 生成器；CPU 环境不需要 GPU 状态。
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_all_caches(
    processed_root: str | Path,
    config: FullTrainingConfig,
    wave_transform: WaveTransform = morlet_cwt_epoch,
) -> CacheBuildResult:
    """按源仓库运行顺序构建三个 split 和平衡预训练缓存。

    参数来源：
        processed_root: 里程碑 3 生成的处理后数据根目录，其下固定有
            ``train``、``validation`` 和 ``test`` 三个记录级 NPZ 目录。
        config: 步骤四运行配置。``data_cache_dir`` 决定缓存写入位置，
            ``seed`` 和 ``offset_samples`` 分别交给平衡缓存的局部随机源
            与偏移范围。
        wave_transform: 把一个 raw ``[1,3000]`` epoch 变为 wave
            ``[1,30,60]`` 的函数。正式运行使用里程碑 4 的 Morlet CWT；
            行为自测传入轻量替身，避免把测试变成长时 CWT。

    返回去向：
        返回三个基础 split manifest、``pretrain_train`` manifest 和构建耗时。
        后续预检用 manifest 核对样本轴，并把耗时写入正式报告。
    """

    # Path 统一用户传入的字符串/路径，后续才能用 / 拼接三个 split。
    processed_root = Path(processed_root)
    # start_time 是缓存阶段起点；保留它才能在全部完成后计算总耗时。
    start_time = perf_counter()
    # manifest_paths 逐 split 累积产物路径，供返回值统一记录。
    manifest_paths: dict[str, Path] = {}

    # 三个 split 使用同一数据链，区别只是输入子目录和输出 split 名。
    for split in ("train", "validation", "test"):
        # 先把记录级 NPZ 按时间顺序写成 split 级 raw/labels/manifest。
        manifest_path = build_raw_label_cache(
            processed_root / split,
            config.data_cache_dir,
            split,
        )
        # 再从刚写入的 raw 逐 epoch 生成 wave，因此 raw/wave/label 的索引一致。
        build_wave_cache(config.data_cache_dir, split, wave_transform)
        # 保存当前 split 的 manifest，后面的返回值要一次交付三个路径。
        manifest_paths[split] = manifest_path

    # 分支预训练遵循源逻辑：先在每条 train 记录内对 raw 偏移过采样，
    # 然后由平衡后的每个 raw 重新生成 wave，并写入独立 pretrain_train。
    balanced_manifest_path = build_balanced_pretrain_cache(
        config.data_cache_dir,
        seed=config.seed,
        offset_samples=config.offset_samples,
        wave_transform=wave_transform,
    )

    # 当前时刻减起点得到四组缓存的总耗时；float() 确保报告可 JSON 序列化。
    elapsed_seconds = float(perf_counter() - start_time)
    # 返回小型路径/耗时对象，而不把约 10 GiB 数组装入内存。
    return CacheBuildResult(
        manifest_paths=manifest_paths,
        balanced_manifest_path=balanced_manifest_path,
        elapsed_seconds=elapsed_seconds,
    )


def overfit_single_batch(
    model: nn.Module,
    batch: TrainingBatch,
    optimizer: Optimizer,
    device: str | torch.device,
    steps: int,
) -> OverfitResult:
    """在同一个 batch 上重复更新模型，作为长训练前的最小预检。

    参数来源：
        model: 待检查的 raw、wave 或融合模型，由预检脚本按正式
            阶段创建。
        batch: 对应训练 DataLoader 取出的第一个 ``(inputs, targets)``；
            inputs 是长度 1 或 2 的张量序列，targets 是 ``[B]`` 或 ``[B,T]``。
        optimizer: 与正式阶段相同的优化器，它持有 model 参数并完成更新。
        device: 来自 ``config.device`` 的 CPU/GPU 位置。
        steps: 同一 batch 重复训练的次数，由预检配置给出。

    返回去向：
        返回每次更新后的平均损失和 CUDA 峰值显存；CPU 运行时峰值
        为 None。预检用损失轨迹判断训练链是否能学习，用峰值判断
        当前 batch size 是否适合 4 GB 显存。
    """

    # steps<=0 时不会发生任何更新，却可能被误读为“预检完成”，因此必须拒绝。
    if steps <= 0:
        raise ValueError("steps 必须为正整数")

    # torch.device 统一字符串或已有设备对象，供显存 API 判断设备类型。
    resolved_device = torch.device(device)
    # CUDA 时先清零“自上次重置后的峰值”，否则会把早先模型的显存算进本预检。
    if resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)

    # losses 按步数累积同一 batch 的损失，用来观察它是否随参数更新下降。
    losses: list[float] = []
    # 每次都把原 batch 包成仅含一项的 iterable，复用步骤三真实 epoch 代码。
    for _ in range(steps):
        loss = run_epoch(model, [batch], resolved_device, optimizer)
        # run_epoch 返回 Python float；保存它而不保存张量，避免保留计算图。
        losses.append(loss)

    # CPU 没有 CUDA 显存，用 None 明确表示“不适用”，不伪造 0 MiB。
    peak_cuda_memory_mib: float | None = None
    if resolved_device.type == "cuda":
        # max_memory_allocated 返回字节；除以 1024**2 转换为报告易读的 MiB。
        peak_cuda_memory_mib = float(
            torch.cuda.max_memory_allocated(resolved_device) / 1024**2
        )

    # tuple 固定预检轨迹，返回后不应被运行脚本意外改写。
    return OverfitResult(tuple(losses), peak_cuda_memory_mib)


def evaluate_fusion_best(
    config: FullTrainingConfig,
    checkpoint_path: str | Path,
) -> ClassificationMetrics:
    """只加载融合阶段 validation-best checkpoint 并正式评估 test。

    参数来源：
        config: ``run_formal_experiment()`` 使用的固定 seed 0 完整配置；
            其缓存根目录下已有 test split，``sequence_length`` 固定融合长度。
        checkpoint_path: ``run_full_training()`` 返回的
            ``results['fusion_finetune'].best_checkpoint``；该路径只由 validation
            损失选出，不是 ``last.pt`` 也不是 test 选出的路径。

    返回去向：
        返回 ACC、Macro-F1、kappa、每类指标与混淆矩阵，交给
        ``build_formal_report()``。函数在返回前关闭 test Dataset 的 memmap。
    """

    # _resolve_device 与训练入口使用同一 auto/CPU/CUDA 规则，避免加载到错误设备。
    device = _resolve_device(config.device)
    # 创建与融合 checkpoint 结构一致的重构模型；构造器已将两个分支切到 finetune 模式。
    model = FFTCNFusionTCN()
    # load_stage_checkpoint 只恢复模型权重；正式 test 不需要 optimizer/scheduler 状态。
    load_stage_checkpoint(checkpoint_path, model, map_location=device)

    # test Dataset 只在 validation 已经选定 checkpoint 后才创建，阻止 test 参与模型选择。
    test_dataset = SleepEDFSequenceDataset(
        config.data_cache_dir,
        "test",
        config.sequence_length,
        "both",
    )
    # 评估顺序不改变指标；shuffle=False 保留 manifest 时间顺序，便于后续定位错误。
    test_loader = build_reproducible_loader(
        test_dataset,
        batch_size=config.fusion_batch_size,
        shuffle=False,
        seed=config.seed,
        num_workers=config.num_workers,
    )

    try:
        # evaluate_model 内部使用 eval/no_grad，将 [B,T,5] logits 展平为全部预测位置。
        return evaluate_model(model, test_loader, n_classes=5, device=device)
    finally:
        # 无论评估成功还是异常，都关闭 test 的 raw/wave/labels memmap，避免 Windows 锁文件。
        test_dataset.close()


def build_formal_report(
    config: FullTrainingConfig,
    stage_results: Mapping[str, StageTrainingResult],
    metrics: ClassificationMetrics,
    training_seconds: float,
    peak_cuda_memory_mib: float | None,
    checkpoint_path: str | Path,
    cache_build_seconds: float | None = None,
) -> dict[str, Any]:
    """把正式 test 指标、训练曲线和可复核上下文组装成报告。

    参数来源：
        config: 正式实验使用的 ``FullTrainingConfig``。
        stage_results: ``run_full_training()`` 返回的三个阶段结果，
            每项包含 best/last/history 路径和逐轮损失曲线。
        metrics: ``evaluate_fusion_best()`` 在固定 test split 上返回的指标。
        training_seconds: ``run_formal_experiment()`` 在完整训练前后计时得到的秒数。
        peak_cuda_memory_mib: 完整训练与 test 过程的 CUDA 峰值显存；
            CPU 运行时为 None。
        checkpoint_path: 本次 test 真正加载的融合 validation-best 路径。
        cache_build_seconds: ``build_all_caches()`` 记录的可选缓存构建耗时；
            恢复旧缓存直接训练时可保持 None。

    返回去向：
        返回可 JSON 序列化的字典，由 ``run_formal_experiment()``
        保存为 ``formal_test_report.json``，最终作为 9B 整体验收证据。
    """

    # Path.resolve 把相对 checkpoint 固定为可复核的绝对路径，str() 使它可写入 JSON。
    checkpoint_path = Path(checkpoint_path)
    resolved_checkpoint = str(checkpoint_path.resolve())
    # metadata 填满里程碑 9 报告必需的数据集、split、seed、checkpoint 和环境字段。
    metadata = {
        "dataset_name": "Sleep-EDF-153",
        "split_name": "test",
        "split_strategy": "subject-disjoint fixed 62/8/8; split seed 42",
        "random_seed": config.seed,
        "checkpoint_path": resolved_checkpoint,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "device": str(_resolve_device(config.device)),
    }
    # 先复用已验证的评估报告构造器，得到声明边界、总指标、每类指标和混淆矩阵。
    report = build_evaluation_report(metrics, metadata)

    # training_stages 逐阶段保存路径和曲线，才能证明 best 来自 validation 历史。
    training_stages: dict[str, dict[str, Any]] = {}
    for stage, result in stage_results.items():
        # dict(row) 复制每轮普通字典，list 容器可直接 JSON 序列化为曲线点列。
        history = [dict(row) for row in result.history]
        # 每个阶段同时记录 best、last 和独立 history 文件，区分选优权重与恢复入口。
        training_stages[stage] = {
            "best_checkpoint": str(result.best_checkpoint.resolve()),
            "last_checkpoint": str(result.last_checkpoint.resolve()),
            "history_path": str(result.history_path.resolve()),
            "history": history,
        }

    # config.to_metadata 保存全部训练超参数；这是复现该次单 seed 运行的入口。
    report["training_config"] = config.to_metadata()
    # data_semantics 固化本次实验的 split 规模、T=50 有效位置和已确认的紧凑时间线限制。
    # 如果不记录它，读者会误以为 18,036 个处理后 test epoch 全部进入了评估。
    report["data_semantics"] = {
        "subject_counts": {"train": 62, "validation": 8, "test": 8},
        "processed_epoch_counts": {
            "train": 154128,
            "validation": 23315,
            "test": 18036,
        },
        "formal_sequence_length": config.sequence_length,
        # support 按真实类别累计被评估位置；求和比硬编码 17,800 更能反映当次报告。
        "test_evaluated_positions": int(metrics.support.sum()),
        "sequence_policy": (
            "source-compatible compacted timeline; non-overlapping sequences "
            "are formed within each record"
        ),
        "known_limitation": (
            "110 of 3836 T=50 sequences cross at least one removed internal "
            "epoch gap (2.87%)"
        ),
    }
    # training_stages 是三条损失/学习率历史与 checkpoint 来源的结构化集合。
    report["training_stages"] = training_stages
    # runtime 将训练、可选缓存时间和峰值显存分开，避免把不同阶段混成一个数。
    report["runtime"] = {
        "cache_build_seconds": (
            None if cache_build_seconds is None else float(cache_build_seconds)
        ),
        "training_seconds": float(training_seconds),
        "peak_cuda_memory_mib": (
            None
            if peak_cuda_memory_mib is None
            else float(peak_cuda_memory_mib)
        ),
    }
    # selection_policy 明示模型选择依据和 test 次数，防止把 test 指标用于反向选模。
    report["selection_policy"] = {
        "criterion": "minimum validation loss",
        "tested_checkpoint": resolved_checkpoint,
        "formal_test_evaluations": 1,
    }
    # 返回完整普通字典，下游 save_evaluation_report 不需要知道训练对象类型。
    return report


def run_formal_experiment(
    config: FullTrainingConfig,
    resume_checkpoints: Mapping[str, str | Path] | None = None,
    cache_build_seconds: float | None = None,
) -> FormalExperimentResult:
    """运行三阶段训练，然后只对融合 validation-best 做一次 test。

    参数来源：
        config: 预检全部通过后确认的正式配置；9B 正式运行固定
            ``seed=0``、raw/wave 各 20 轮、融合 50 轮。
        resume_checkpoints: 可选的“阶段名 -> last.pt”映射，由恢复预检
            或中断后的重启命令提供；首次运行为 None。
        cache_build_seconds: 本次全量缓存构建耗时，来自
            ``CacheBuildResult.elapsed_seconds``，仅用于报告记录。

    返回去向：
        返回三阶段 checkpoint/history、正式 test 指标、报告路径、
        训练耗时与峰值显存。这个对象只描述单 seed、单数据集结果。
    """

    # 在创建任何模型或 DataLoader 前固定随机源，确保初始化和 shuffle 都受 config.seed 控制。
    seed_experiment(config.seed)
    # 用步骤三的共享规则解析 auto，后面训练、test 和显存统计共用该设备。
    device = _resolve_device(config.device)
    # CUDA 时从零开始记录本次正式运行峰值；CPU 运行不调用 CUDA API。
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # training_start 单独保存三阶段训练起点，不把事前缓存或事后 test 混入。
    training_start = perf_counter()
    # run_full_training 创建六个 train/validation Loader，并按 raw -> wave -> fusion 顺序训练。
    stage_results = run_full_training(
        config,
        resume_checkpoints=resume_checkpoints,
    )
    # 训练返回时 validation 选模已结束；此时才停止训练计时。
    training_seconds = float(perf_counter() - training_start)

    # 三阶段结果中只选融合阶段 best；不使用 last，也不查看 test 后再换 checkpoint。
    checkpoint_path = stage_results["fusion_finetune"].best_checkpoint
    # evaluate_fusion_best 在此只调用一次，它是当次模型选择后的唯一正式 test。
    metrics = evaluate_fusion_best(config, checkpoint_path)

    # CPU 用 None 表示无 CUDA 统计；CUDA 则在 test 后读峰值，覆盖完整正式运行。
    peak_cuda_memory_mib: float | None = None
    if device.type == "cuda":
        peak_cuda_memory_mib = float(
            torch.cuda.max_memory_allocated(device) / 1024**2
        )

    # 报告同时接收指标、三阶段曲线、时间、显存和真正测试的 checkpoint 路径。
    report = build_formal_report(
        config=config,
        stage_results=stage_results,
        metrics=metrics,
        training_seconds=training_seconds,
        peak_cuda_memory_mib=peak_cuda_memory_mib,
        checkpoint_path=checkpoint_path,
        cache_build_seconds=cache_build_seconds,
    )
    # 固定报告路径使运行命令和 9B 验收都能直接找到唯一正式 test 产物。
    report_path = config.output_dir / "formal_test_report.json"
    # 评估工具负责创建父目录并以 UTF-8/JSON 写入，这里不重复文件 I/O。
    save_evaluation_report(report_path, report)

    # 返回结构化摘要；大型模型和数据张量不进入结果对象，避免延长显存占用。
    return FormalExperimentResult(
        stage_results=stage_results,
        metrics=metrics,
        report_path=report_path,
        training_seconds=training_seconds,
        peak_cuda_memory_mib=peak_cuda_memory_mib,
    )
```

## 6. 用户编码与行为自测

请一次完成 `training/sleep_edf_formal_experiment.py` 中 6 个 `NotImplementedError`。主代码只保留一个新增检查：`steps > 0`。若不检查，零次更新会返回空轨迹，却可能被误报为“过拟合预检完成”；这是当前固定运行链会产生的静默语义错误，不是泛化式防御。

完成后运行：

```powershell
python -m unittest tests.test_sleep_edf_formal_experiment -v
```

7 条行为分别检查：

1. Python / NumPy / PyTorch 随机序列可重复；
2. 三个 split 与平衡缓存的 raw/wave/labels 样本轴对齐；
3. 同一 batch 重复更新后损失下降，CUDA 峰值在更新前重置、更新后读取；
4. 恢复映射被传入步骤三训练编排，且 Dataset 最终关闭；
5. test 评估加载指定的融合 best，并在结束时关闭 test memmap；
6. 报告含配置、曲线、耗时、显存和 validation-best 来源；
7. 完整入口只把融合 validation-best 交给 test 一次。

## 7. 代码自测通过后的真实预检顺序

真实运行使用仓库中的固定入口 `scripts/run_milestone_09b.py`。全量缓存写盘、真实 batch 的模型更新、完整 epoch 和正式 90 轮训练均由用户逐项启动；脚本不会自动串联长阶段。准确命令、固定路径、产物检查和中断恢复规则见 `learning_guides/milestone_09b/OPERATOR_GUIDE.md`。

顺序保持为：

1. `cache`：构建全量缓存，记录真实耗时、四个 manifest 与构建后 D 盘余量；
2. `overfit`：从真实 Loader 各取一个 raw、wave、fusion batch，分别重复训练，记录损失轨迹和峰值显存；
3. `epoch1`：三阶段各完成 1 个完整 epoch，确认 history、best.pt、last.pt 均生成；
4. `resume2`：在同一预检目录从三阶段 `last.pt` 恢复到总轮数 2，确认 history 继续且 scheduler 学习率没有重置；
5. `formal`：在全新正式目录中固定 seed 0 运行 20/20/50；不得从预检 checkpoint 开始；
6. `formal-resume`：只在正式训练中断后使用同一正式目录的 `last.pt` 恢复，不是首次启动命令；
7. 正式训练与 validation 选模结束后，入口只评估一次 fusion validation-best，并保存 `formal_test_report.json`。

每个长阶段都由用户入口在当前终端显示进度：缓存按四段 CWT epoch 连续计数；
训练的每个 epoch 分别按 train/validation batch 更新，并在描述中显示当前阶段和
`当前 epoch/总 epoch`。`status` 只用于可选旁路检查。每一步完成后检查对应
JSON/history/checkpoint，再进入下一步。若默认 fusion batch 32 出现 OOM，
先讨论，不自行改变实验配置。
