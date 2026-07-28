"""里程碑 9B 真实预检与正式实验的用户运行入口。

本脚本只负责把已经完成的教学重构函数按固定顺序连接起来。它不改模型、
损失、优化器或数据语义，也不会自动连续执行多个长阶段；用户每次必须明确
选择一个 stage，并在检查该阶段产物后再运行下一条命令。
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from tqdm import tqdm


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from models.merge.sleep_edf_fusion_tcn import FFTCNFusionTCN
from models.raw.sleep_edf_1d_cnn import SleepEDFRawFeatureNet
from models.wavelet.sleep_edf_2d_cnn import SleepEDFWaveFeatureNet
from data.sleep_edf_cwt import morlet_cwt_epoch
from training.sleep_edf_formal_experiment import (
    build_all_caches,
    overfit_single_batch,
    run_formal_experiment,
    seed_experiment,
)
from training.sleep_edf_full_run import (
    FullTrainingConfig,
    SleepEDFSequenceDataset,
    StageTrainingResult,
    _resolve_device,
    build_reproducible_loader,
    run_full_training,
)
from training.sleep_edf_two_stage import build_finetune_optimizer


# 所有路径固定在当前仓库内，避免不同命令意外读写不同数据或 checkpoint。
PROCESSED_ROOT = REPOSITORY_ROOT / "datasets" / "sleep-edf-153-processed-v1"
CACHE_ROOT = REPOSITORY_ROOT / "datasets" / "sleep-edf-153-training-cache-v1"
ARTIFACT_ROOT = REPOSITORY_ROOT / "reproduction_artifacts" / "milestone_09b"
REAL_RUN_ROOT = ARTIFACT_ROOT / "real_run"
PREFLIGHT_OUTPUT = REAL_RUN_ROOT / "preflight_training"
FORMAL_OUTPUT = ARTIFACT_ROOT / "formal_seed_0"

CACHE_REPORT = REAL_RUN_ROOT / "cache_build.json"
OVERFIT_REPORT = REAL_RUN_ROOT / "overfit.json"
EPOCH1_REPORT = REAL_RUN_ROOT / "epoch1.json"
RESUME2_REPORT = REAL_RUN_ROOT / "resume2.json"

STAGE_NAMES = ("raw_pretrain", "wave_pretrain", "fusion_finetune")
EXPECTED_CACHE_COUNTS = {
    "train": 154128,
    "validation": 23315,
    "test": 18036,
    "pretrain_train": 494617,
}
OVERFIT_STEPS = 20


class _CacheWaveProgress:
    """在不改缓存算法的前提下，为四段逐 epoch CWT 显示一个连续进度条。"""

    def __init__(
        self,
        wave_transform: Callable[[np.ndarray], np.ndarray],
        stage_counts: Mapping[str, int] = EXPECTED_CACHE_COUNTS,
    ) -> None:
        self._wave_transform = wave_transform
        self._stages = tuple(stage_counts.items())
        self._completed = 0
        self._shown_stage: str | None = None
        self._bar = tqdm(
            total=sum(stage_counts.values()),
            unit="epoch",
            dynamic_ncols=True,
            disable=None,
        )

    def __call__(self, raw_epoch: np.ndarray) -> np.ndarray:
        """先执行原 Morlet CWT，再把成功完成的一个 epoch 计入进度。"""

        boundary = 0
        current_stage = self._stages[-1][0]
        for stage, count in self._stages:
            boundary += count
            if self._completed < boundary:
                current_stage = stage
                break

        if current_stage != self._shown_stage:
            self._bar.set_description(f"cache {current_stage} CWT")
            self._shown_stage = current_stage

        wave_epoch = self._wave_transform(raw_epoch)
        self._completed += 1
        self._bar.update(1)
        return wave_epoch

    def close(self) -> None:
        """异常或成功结束时都关闭进度条，避免终端停在半行。"""

        self._bar.close()


def _config(
    output_dir: Path,
    *,
    raw_epochs: int,
    wave_epochs: int,
    fusion_epochs: int,
) -> FullTrainingConfig:
    """创建只改变轮数和输出目录的固定正式配置。"""

    return FullTrainingConfig(
        data_cache_dir=CACHE_ROOT,
        output_dir=output_dir,
        seed=0,
        offset_samples=300,
        sequence_length=50,
        raw_pretrain_epochs=raw_epochs,
        wave_pretrain_epochs=wave_epochs,
        fusion_finetune_epochs=fusion_epochs,
        raw_batch_size=128,
        wave_batch_size=128,
        fusion_batch_size=32,
        pretrain_learning_rate=1e-5,
        fusion_learning_rate=1e-5,
        feature_learning_rate_scale=1e-2,
        scheduler_gamma=0.95,
        num_workers=0,
        device="auto",
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """把阶段摘要写成 UTF-8 JSON，供用户和 Codex 在下一步前检查。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _read_json(path: Path) -> dict[str, Any]:
    """读取前一阶段产生的 JSON 摘要。"""

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _free_gib() -> float:
    """返回当前仓库所在 D 盘的可用磁盘 GiB。"""

    return round(shutil.disk_usage(REPOSITORY_ROOT.anchor).free / 1024**3, 2)


def _cache_status() -> dict[str, Any]:
    """只读检查四组缓存的 manifest 与三个 NPY 契约是否完整。"""

    splits: dict[str, Any] = {}
    all_complete = True
    for split, expected_count in EXPECTED_CACHE_COUNTS.items():
        split_dir = CACHE_ROOT / split
        required = {
            name: split_dir / name
            for name in ("raw.npy", "wave.npy", "labels.npy", "manifest.json")
        }
        files_exist = {name: path.is_file() for name, path in required.items()}
        sample_count: int | None = None
        manifest: dict[str, Any] = {}
        if files_exist["manifest.json"]:
            try:
                manifest = _read_json(required["manifest.json"])
                sample_count = int(manifest["sample_count"])
            except (OSError, KeyError, TypeError, ValueError):
                # 中断写盘可能留下暂时不可读的 manifest；status 应报告未完成，
                # 而不是妨碍用户查看其余阶段。
                sample_count = None

        array_contracts: dict[str, Any] = {}
        for name in ("raw", "wave", "labels"):
            expected_shape = manifest.get(f"{name}_shape")
            expected_dtype = manifest.get(f"{name}_dtype")
            actual_shape: list[int] | None = None
            actual_dtype: str | None = None
            valid = False
            array: np.ndarray | None = None
            if files_exist[f"{name}.npy"] and expected_shape and expected_dtype:
                try:
                    array = np.load(required[f"{name}.npy"], mmap_mode="r")
                    actual_shape = [int(size) for size in array.shape]
                    actual_dtype = str(array.dtype)
                    valid = (
                        actual_shape == [int(size) for size in expected_shape]
                        and actual_dtype == str(expected_dtype)
                        and actual_shape[0] == expected_count
                    )
                except (OSError, ValueError):
                    valid = False
                finally:
                    if isinstance(array, np.memmap) and array._mmap is not None:
                        array._mmap.close()
            array_contracts[name] = {
                "valid": valid,
                "shape": actual_shape,
                "dtype": actual_dtype,
            }

        records = manifest.get("records", [])
        next_start = 0
        records_cover_cache = bool(records)
        try:
            for record in records:
                if int(record["start"]) != next_start:
                    records_cover_cache = False
                    break
                next_start = int(record["stop"])
            records_cover_cache = records_cover_cache and next_start == expected_count
        except (KeyError, TypeError, ValueError):
            records_cover_cache = False

        complete = (
            all(files_exist.values())
            and sample_count == expected_count
            and all(contract["valid"] for contract in array_contracts.values())
            and records_cover_cache
        )
        all_complete = all_complete and complete
        splits[split] = {
            "complete": complete,
            "sample_count": sample_count,
            "expected_sample_count": expected_count,
            "files": files_exist,
            "arrays": array_contracts,
            "records_cover_cache": records_cover_cache,
        }

    return {
        "complete": all_complete,
        "cache_root": str(CACHE_ROOT),
        "splits": splits,
    }


def _require_cache() -> None:
    """拒绝在缓存缺失或样本数不符时进入真实模型预检。"""

    status = _cache_status()
    if not status["complete"]:
        raise RuntimeError(
            "训练缓存尚未完整通过固定样本数检查；请先运行 cache，并检查 "
            f"{CACHE_REPORT}"
        )


def _history_status(output_dir: Path) -> dict[str, Any]:
    """汇总一个训练目录中三个阶段的 history/checkpoint 进度。"""

    stages: dict[str, Any] = {}
    for stage in STAGE_NAMES:
        stage_dir = output_dir / stage
        history_path = stage_dir / "history.json"
        try:
            history = _read_json(history_path) if history_path.is_file() else []
        except (OSError, ValueError):
            # 训练进程重写 JSON 的瞬间，另一个终端可能恰好读到半份内容。
            history = []
        stages[stage] = {
            "completed_epochs": len(history),
            "last_epoch": None if not history else int(history[-1]["epoch"]),
            "best_exists": (stage_dir / "best.pt").is_file(),
            "last_exists": (stage_dir / "last.pt").is_file(),
            "history_path": str(history_path),
        }
    return stages


def show_status() -> None:
    """打印只读状态；长训练期间可在另一个终端重复运行。"""

    processed_counts = {
        split: len(tuple((PROCESSED_ROOT / split).glob("*.npz")))
        for split in ("train", "validation", "test")
    }
    payload = {
        "fixed_paths": {
            "processed_root": str(PROCESSED_ROOT),
            "cache_root": str(CACHE_ROOT),
            "preflight_output": str(PREFLIGHT_OUTPUT),
            "formal_output": str(FORMAL_OUTPUT),
        },
        "processed_record_counts": processed_counts,
        "free_disk_gib": _free_gib(),
        "cache": _cache_status(),
        "reports": {
            "cache_build": CACHE_REPORT.is_file(),
            "overfit": OVERFIT_REPORT.is_file(),
            "epoch1": EPOCH1_REPORT.is_file(),
            "resume2": RESUME2_REPORT.is_file(),
            "formal_test": (FORMAL_OUTPUT / "formal_test_report.json").is_file(),
        },
        "preflight_training": _history_status(PREFLIGHT_OUTPUT),
        "formal_training": _history_status(FORMAL_OUTPUT),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def run_cache() -> None:
    """构建一次全量缓存；非空目录必须先人工核对，不自动覆盖或删除。"""

    if CACHE_ROOT.exists() and any(CACHE_ROOT.iterdir()):
        raise RuntimeError(
            f"缓存目录已经非空：{CACHE_ROOT}。为避免覆盖约 10 GiB 数据，"
            "请先检查现有内容，不要直接重跑。"
        )

    for split in ("train", "validation", "test"):
        if not (PROCESSED_ROOT / split).is_dir():
            raise FileNotFoundError(f"缺少处理后数据目录：{PROCESSED_ROOT / split}")

    before_gib = _free_gib()
    config = _config(
        FORMAL_OUTPUT,
        raw_epochs=20,
        wave_epochs=20,
        fusion_epochs=50,
    )
    progress = _CacheWaveProgress(morlet_cwt_epoch)
    try:
        result = build_all_caches(
            PROCESSED_ROOT,
            config,
            wave_transform=progress,
        )
    finally:
        progress.close()
    cache_status = _cache_status()
    if not cache_status["complete"]:
        raise RuntimeError("缓存函数已返回，但固定样本数或必需文件检查未通过")

    payload = {
        "processed_root": str(PROCESSED_ROOT.resolve()),
        "cache_root": str(CACHE_ROOT.resolve()),
        "manifest_paths": {
            split: str(path.resolve())
            for split, path in result.manifest_paths.items()
        },
        "balanced_manifest_path": str(result.balanced_manifest_path.resolve()),
        "elapsed_seconds": float(result.elapsed_seconds),
        "free_disk_gib_before": before_gib,
        "free_disk_gib_after": _free_gib(),
        "cache_status": cache_status,
    }
    _write_json(CACHE_REPORT, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _run_one_overfit_stage(stage: str, config: FullTrainingConfig) -> dict[str, Any]:
    """为一个分支创建真实首 batch，并在关闭 Dataset 前完成固定 batch 预检。"""

    seed_experiment(config.seed)
    if stage == "raw":
        dataset = SleepEDFSequenceDataset(CACHE_ROOT, "pretrain_train", 1, "raw")
    elif stage == "wave":
        dataset = SleepEDFSequenceDataset(CACHE_ROOT, "pretrain_train", 1, "wave")
    else:
        dataset = SleepEDFSequenceDataset(
            CACHE_ROOT, "train", config.sequence_length, "both"
        )

    try:
        if stage == "raw":
            batch_size = config.raw_batch_size
            model = SleepEDFRawFeatureNet()
            model.pretrain()
            optimizer = torch.optim.Adam(
                model.parameters(), lr=config.pretrain_learning_rate
            )
        elif stage == "wave":
            batch_size = config.wave_batch_size
            model = SleepEDFWaveFeatureNet()
            model.pretrain()
            optimizer = torch.optim.Adam(
                model.parameters(), lr=config.pretrain_learning_rate
            )
        else:
            batch_size = config.fusion_batch_size
            model = FFTCNFusionTCN()
            optimizer = build_finetune_optimizer(
                model,
                config.fusion_learning_rate,
                config.feature_learning_rate_scale,
            )

        loader = build_reproducible_loader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            seed=config.seed,
            num_workers=config.num_workers,
        )
        batch = next(iter(loader))
        result = overfit_single_batch(
            model,
            batch,
            optimizer,
            device=_resolve_device(config.device),
            steps=OVERFIT_STEPS,
        )
        losses = [float(loss) for loss in result.losses]
        return {
            "batch_size": batch_size,
            "input_shapes": [list(tensor.shape) for tensor in batch[0]],
            "target_shape": list(batch[1].shape),
            "steps": OVERFIT_STEPS,
            "losses": losses,
            "first_loss": losses[0],
            "last_loss": losses[-1],
            "loss_decreased": losses[-1] < losses[0],
            "peak_cuda_memory_mib": result.peak_cuda_memory_mib,
        }
    finally:
        dataset.close()


def run_overfit() -> None:
    """依次执行 raw、wave、fusion 真实首 batch 过拟合，不保存模型。"""

    _require_cache()
    config = _config(
        PREFLIGHT_OUTPUT,
        raw_epochs=1,
        wave_epochs=1,
        fusion_epochs=1,
    )
    payload: dict[str, Any] = {
        "device": str(_resolve_device(config.device)),
        "stages": {},
    }
    for stage in ("raw", "wave", "fusion"):
        print(f"[overfit] 开始 {stage}，固定同一 batch 更新 {OVERFIT_STEPS} 次")
        result = _run_one_overfit_stage(stage, config)
        payload["stages"][stage] = result
        _write_json(OVERFIT_REPORT, payload)
        print(
            f"[overfit] {stage}: {result['first_loss']:.6f} -> "
            f"{result['last_loss']:.6f}, "
            f"peak={result['peak_cuda_memory_mib']} MiB"
        )
        if not result["loss_decreased"]:
            raise RuntimeError(f"{stage} 的固定 batch 损失末值未低于首值")

        # 三个模型必须分别测峰值；释放上一模型，避免活跃张量污染下一阶段。
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"三种真实 batch 预检通过，摘要：{OVERFIT_REPORT}")


def _stage_results_payload(
    results: Mapping[str, StageTrainingResult],
) -> dict[str, Any]:
    """把训练结果压缩为路径和普通 history，避免序列化模型对象。"""

    return {
        stage: {
            "best_checkpoint": str(result.best_checkpoint.resolve()),
            "last_checkpoint": str(result.last_checkpoint.resolve()),
            "history_path": str(result.history_path.resolve()),
            "history": [dict(row) for row in result.history],
        }
        for stage, result in results.items()
    }


def _require_fresh_output(output_dir: Path, command: str) -> None:
    """首次训练不得覆盖已有 checkpoint；恢复必须使用专门命令。"""

    if output_dir.exists() and any(output_dir.rglob("*")):
        raise RuntimeError(
            f"输出目录已非空：{output_dir}。不要重新运行 {command}；"
            "请先检查状态并使用对应恢复命令。"
        )


def run_epoch1(*, resume: bool) -> None:
    """从头运行三阶段 1 epoch，或从同一预检目录的已有 last.pt 恢复。"""

    _require_cache()
    if not OVERFIT_REPORT.is_file():
        raise RuntimeError(f"缺少真实 batch 预检报告：{OVERFIT_REPORT}")
    overfit_payload = _read_json(OVERFIT_REPORT)
    if not all(
        overfit_payload["stages"][stage]["loss_decreased"]
        for stage in ("raw", "wave", "fusion")
    ):
        raise RuntimeError("三种真实 batch 尚未全部通过总体损失下降检查")

    if EPOCH1_REPORT.is_file():
        raise RuntimeError(f"1 epoch 摘要已经存在：{EPOCH1_REPORT}")
    if resume:
        if not PREFLIGHT_OUTPUT.exists():
            raise RuntimeError("预检目录尚不存在；首次运行请使用 epoch1")
        resume_checkpoints = _last_checkpoints(
            PREFLIGHT_OUTPUT, require_all=False
        )
    else:
        _require_fresh_output(PREFLIGHT_OUTPUT, "epoch1")
        resume_checkpoints = None

    config = _config(
        PREFLIGHT_OUTPUT,
        raw_epochs=1,
        wave_epochs=1,
        fusion_epochs=1,
    )
    seed_experiment(config.seed)
    results = run_full_training(
        config,
        resume_checkpoints=resume_checkpoints,
    )
    payload = {
        "config": config.to_metadata(),
        "stages": _stage_results_payload(results),
    }
    for stage, result in results.items():
        if [int(row["epoch"]) for row in result.history] != [0]:
            raise RuntimeError(f"{stage} 的 1 epoch history 不是 [0]")

    _write_json(EPOCH1_REPORT, payload)
    print(f"三阶段 1 epoch 预检完成，摘要：{EPOCH1_REPORT}")


def _last_checkpoints(output_dir: Path, *, require_all: bool) -> dict[str, Path]:
    """收集同一输出目录中已经存在的 last.pt。"""

    checkpoints = {
        stage: output_dir / stage / "last.pt"
        for stage in STAGE_NAMES
        if (output_dir / stage / "last.pt").is_file()
    }
    if require_all and set(checkpoints) != set(STAGE_NAMES):
        missing = sorted(set(STAGE_NAMES) - set(checkpoints))
        raise RuntimeError(f"恢复预检缺少阶段 last.pt：{missing}")
    return checkpoints


def run_resume2() -> None:
    """在同一预检目录把三个阶段从 epoch 0 恢复到总轮数 2。"""

    _require_cache()
    if not EPOCH1_REPORT.is_file():
        raise RuntimeError(f"缺少 1 epoch 摘要：{EPOCH1_REPORT}")
    resume = _last_checkpoints(PREFLIGHT_OUTPUT, require_all=True)
    config = _config(
        PREFLIGHT_OUTPUT,
        raw_epochs=2,
        wave_epochs=2,
        fusion_epochs=2,
    )
    seed_experiment(config.seed)
    results = run_full_training(config, resume_checkpoints=resume)

    payload = {
        "config": config.to_metadata(),
        "resume_checkpoints": {
            stage: str(path.resolve()) for stage, path in resume.items()
        },
        "stages": _stage_results_payload(results),
    }
    for stage, result in results.items():
        history = list(result.history)
        epochs = [int(row["epoch"]) for row in history]
        if epochs != [0, 1]:
            raise RuntimeError(f"{stage} 的恢复 history 不是 [0, 1]：{epochs}")
        for old_lr, new_lr in zip(
            history[0]["learning_rates"], history[1]["learning_rates"]
        ):
            if not math.isclose(
                float(new_lr),
                float(old_lr) * config.scheduler_gamma,
                rel_tol=1e-9,
                abs_tol=0.0,
            ):
                raise RuntimeError(f"{stage} 的 scheduler 学习率没有连续衰减")

    _write_json(RESUME2_REPORT, payload)
    print(f"三阶段恢复到总轮数 2，摘要：{RESUME2_REPORT}")


def _cache_build_seconds() -> float:
    """正式报告复用真实缓存耗时，不用估算值替代。"""

    if not CACHE_REPORT.is_file():
        raise RuntimeError(f"缺少缓存构建摘要：{CACHE_REPORT}")
    return float(_read_json(CACHE_REPORT)["elapsed_seconds"])


def run_formal(*, resume: bool) -> None:
    """从头正式训练，或只在同一正式目录恢复中断的训练。"""

    _require_cache()
    if not RESUME2_REPORT.is_file():
        raise RuntimeError("尚未完成 1→2 epoch 恢复预检，不能启动正式训练")
    report_path = FORMAL_OUTPUT / "formal_test_report.json"
    if report_path.is_file():
        raise RuntimeError(
            f"正式报告已经存在：{report_path}。为避免重复正式 test，拒绝再次运行。"
        )

    if resume:
        formal_status = _history_status(FORMAL_OUTPUT)
        target_epochs = {
            "raw_pretrain": 20,
            "wave_pretrain": 20,
            "fusion_finetune": 50,
        }
        if all(
            formal_status[stage]["completed_epochs"] >= target
            for stage, target in target_epochs.items()
        ):
            raise RuntimeError(
                "三阶段 history 已达到 20/20/50，但正式报告缺失。此时无法仅凭"
                "文件判断 test 是否已经开始；请先人工核对，不要自动重复评估。"
            )

        resume_checkpoints = _last_checkpoints(FORMAL_OUTPUT, require_all=False)
        if not resume_checkpoints:
            existing_files = (
                tuple(path for path in FORMAL_OUTPUT.rglob("*") if path.is_file())
                if FORMAL_OUTPUT.exists()
                else ()
            )
            if not FORMAL_OUTPUT.exists():
                raise RuntimeError("正式目录尚不存在；首次运行请使用 formal")
            if existing_files:
                raise RuntimeError(
                    "正式目录已有文件但没有可恢复的 last.pt；请先人工检查，"
                    "不要自动覆盖"
                )
            # 若只创建了空阶段目录便中断，还没有模型状态可恢复；重新以同一
            # seed 从头开始等价于首次训练，同时保留正式目录不跨到预检目录。
            resume_checkpoints = None
    else:
        _require_fresh_output(FORMAL_OUTPUT, "formal")
        resume_checkpoints = None

    config = _config(
        FORMAL_OUTPUT,
        raw_epochs=20,
        wave_epochs=20,
        fusion_epochs=50,
    )
    result = run_formal_experiment(
        config,
        resume_checkpoints=resume_checkpoints,
        cache_build_seconds=_cache_build_seconds(),
    )
    print(
        json.dumps(
            {
                "report_path": str(result.report_path.resolve()),
                "training_seconds": float(result.training_seconds),
                "peak_cuda_memory_mib": result.peak_cuda_memory_mib,
                "accuracy": float(result.metrics.accuracy),
                "macro_f1": float(result.metrics.macro_f1),
                "kappa": float(result.metrics.kappa),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    """只解析固定阶段名，不开放会改变正式实验语义的超参数。"""

    parser = argparse.ArgumentParser(
        description="FFTCN 里程碑 9B：由用户分阶段运行真实预检与正式实验"
    )
    parser.add_argument(
        "stage",
        choices=(
            "status",
            "cache",
            "overfit",
            "epoch1",
            "epoch1-resume",
            "resume2",
            "formal",
            "formal-resume",
        ),
        help="每次只运行一个阶段；长阶段完成后先检查产物，再进入下一阶段",
    )
    return parser.parse_args()


def main() -> None:
    """分派用户明确选择的单个阶段，不自动串联长操作。"""

    stage = parse_args().stage
    if stage == "status":
        show_status()
    elif stage == "cache":
        run_cache()
    elif stage == "overfit":
        run_overfit()
    elif stage == "epoch1":
        run_epoch1(resume=False)
    elif stage == "epoch1-resume":
        run_epoch1(resume=True)
    elif stage == "resume2":
        run_resume2()
    elif stage == "formal":
        run_formal(resume=False)
    elif stage == "formal-resume":
        run_formal(resume=True)
    else:
        raise ValueError(f"未知阶段：{stage}")


if __name__ == "__main__":
    main()
