"""里程碑 9B 内部步骤四：预检与正式实验练习。

本文件只保留步骤四的最小公开接口。完整教程与独立参考实现位于：

    learning_guides/milestone_09b/

正式练习不得导入独立参考答案，也不导入原仓库 Loader、trainer
或已训练 checkpoint。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import platform
import random
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from data.sleep_edf_cwt import morlet_cwt_epoch
from data.sleep_edf_training_cache import build_raw_label_cache, build_wave_cache
from evaluation.sleep_edf_metrics import (
    ClassificationMetrics,
    build_evaluation_report,
    evaluate_model,
    save_evaluation_report,
)
from models.merge.sleep_edf_fusion_tcn import FFTCNFusionTCN
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


WaveTransform = Callable[[np.ndarray], np.ndarray]
TrainingBatch = tuple[Sequence[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class CacheBuildResult:
    """保存全量缓存构建产物的位置和耗时。"""

    manifest_paths: Mapping[str, Path]
    balanced_manifest_path: Path
    elapsed_seconds: float


@dataclass(frozen=True)
class OverfitResult:
    """保存重复训练同一 batch 时的损失轨迹与峰值显存。"""

    losses: tuple[float, ...]
    peak_cuda_memory_mib: float | None


@dataclass(frozen=True)
class FormalExperimentResult:
    """保存三阶段训练、一次 test 与报告的正式结果。"""

    stage_results: Mapping[str, StageTrainingResult]
    metrics: ClassificationMetrics
    report_path: Path
    training_seconds: float
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

    "步骤四练习：固定实验随机源"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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

    "步骤四练习：串联基础缓存和平衡缓存"
    processed_root = Path(processed_root)
    start_time = perf_counter()
    manifest_paths = {}
    for split in ("train", "validation", "test"):
        manifest_path = build_raw_label_cache(
            processed_root / split,
            config.data_cache_dir,
            split,
        )

        build_wave_cache(
            config.data_cache_dir,
            split,
            wave_transform,
        )

        manifest_paths[split] = manifest_path

    balanced_manifest_path = build_balanced_pretrain_cache(
        config.data_cache_dir,
        seed=config.seed,
        offset_samples=config.offset_samples,
        wave_transform=wave_transform
    )

    elapsed_seconds = float(perf_counter() - start_time)

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

    "步骤四练习：重复同一 batch 并记录损失/显存"
    if steps <= 0:
        raise ValueError("steps must be positive")

    resolved_device = torch.device(device)

    # 必须在训练前清空旧峰值，否则循环结束后再重置会抹掉本次预检的真实峰值。
    if resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)

    losses = []
    for i in range(steps):
        loss = run_epoch(
            model,
            [batch],
            resolved_device,
            optimizer
        )
        losses.append(loss)

    peak_cuda_memory_mib = None
    if resolved_device.type == "cuda":
        peak_cuda_memory_mib = float(
            torch.cuda.max_memory_allocated(resolved_device) / 1024**2
        )

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

    "步骤四练习：只评估融合 validation-best"
    device = _resolve_device(config.device)
    model = FFTCNFusionTCN()
    load_stage_checkpoint(checkpoint_path, model, map_location=device)

    test_dataset = SleepEDFSequenceDataset(
        config.data_cache_dir,
        split="test",
        sequence_length=config.sequence_length,
        input_mode="both"
    )

    test_dataloader = build_reproducible_loader(
        test_dataset,
        batch_size=config.fusion_batch_size,
        shuffle=False,
        seed=config.seed,
        num_workers=config.num_workers,
    )

    try:
        return evaluate_model(
            model,
            test_dataloader,
            n_classes=5,
            device=device,
        )
    finally:
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

    "步骤四练习：组装指标、曲线和实验来源"
    checkpoint_path = Path(checkpoint_path)
    resolved_checkpoint = str(checkpoint_path.resolve())
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

    report = build_evaluation_report(
        metrics,
        metadata,
    )

    training_stages = {}
    for stage_name, stage_result in stage_results.items():
        history = [dict(row) for row in stage_result.history]
        training_stages[stage_name] = {
            "best_checkpoint": str(stage_result.best_checkpoint.resolve()),
            "last_checkpoint": str(stage_result.last_checkpoint.resolve()),
            "history_path": str(stage_result.history_path.resolve()),
            "history": history,
        }

    report["training_config"] = config.to_metadata()
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

    "步骤四练习：训练、一次 test 与报告落盘"
    seed_experiment(config.seed)
    device = _resolve_device(config.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    training_start = perf_counter()
    stage_results = run_full_training(
        config,
        resume_checkpoints,
    )
    training_seconds = float(perf_counter() - training_start)

    checkpoint_path = stage_results["fusion_finetune"].best_checkpoint
    metrics = evaluate_fusion_best(
        config,
        checkpoint_path,
    )

    peak_cuda_memory_mib: float | None = None
    if device.type == "cuda":
        peak_cuda_memory_mib = float(
            torch.cuda.max_memory_allocated(device) / 1024 ** 2
        )

    report = build_formal_report(
        config,
        stage_results,
        metrics,
        training_seconds,
        peak_cuda_memory_mib,
        checkpoint_path,
        cache_build_seconds,
    )

    report_path = config.output_dir / "formal_test_report.json"
    save_evaluation_report(report_path, report)

    return FormalExperimentResult(
        stage_results=stage_results,
        metrics=metrics,
        report_path=report_path,
        training_seconds=training_seconds,
        peak_cuda_memory_mib=peak_cuda_memory_mib,
    )
