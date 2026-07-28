"""用少量真实 Sleep-EDF 数据贯通原仓库训练代码，不修改核心模型模块。"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pyedflib
import torch

# 直接运行 ``python scripts/...py`` 时，Python 只把 scripts/ 加入 sys.path；
# 原仓库使用从项目根目录开始的绝对导入，因此显式加入仓库根目录。
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data.loader import Sleep_Loader
from models.merge.model import MergeModel
from models.raw.model import RawModel
from models.wavelet.model import WaveletModel


LABEL_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}
DEMO_RECORDS = ("SC4001", "SC4002", "SC4021", "SC4022", "SC4031", "SC4032")
RATIO = [0.5, 0.25, 0.25]


def find_file(root: Path, record_id: str, suffix: str) -> Path:
    matches = sorted(root.glob(f"{record_id}*{suffix}"))
    if len(matches) != 1:
        raise RuntimeError(f"{record_id} {suffix} 匹配到 {len(matches)} 个文件")
    return matches[0]


def expanded_labels(hypnogram: Path, epoch_count: int) -> np.ndarray:
    labels = np.full(epoch_count, -1, dtype=np.int64)
    with pyedflib.EdfReader(str(hypnogram)) as reader:
        onsets, durations, descriptions = reader.readAnnotations()
    for onset, duration, description in zip(onsets, durations, descriptions):
        label = LABEL_MAP.get(str(description), -1)
        start = int(round(float(onset) / 30.0))
        count = int(round(float(duration) / 30.0))
        stop = min(epoch_count, start + count)
        if start < epoch_count:
            labels[start:stop] = label
    return labels


def choose_five_class_window(labels: np.ndarray, length: int = 50) -> int:
    expected = set(range(5))
    for start in range(0, len(labels) - length + 1):
        window = labels[start : start + length]
        if set(window.tolist()) == expected:
            return start
    raise RuntimeError(f"找不到包含五类的连续 {length} 段窗口")


def prepare_demo_data(edf_root: Path, output: Path) -> list[dict[str, object]]:
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for record_id in DEMO_RECORDS:
        psg = find_file(edf_root, record_id, "-PSG.edf")
        hypnogram = find_file(edf_root, record_id, "-Hypnogram.edf")
        with pyedflib.EdfReader(str(psg)) as reader:
            labels = reader.getSignalLabels()
            channel_index = labels.index("EEG Fpz-Cz")
            sampling_rate = float(reader.getSampleFrequency(channel_index))
            if sampling_rate != 100.0:
                raise RuntimeError(f"{record_id} 采样率不是 100 Hz：{sampling_rate}")
            total_samples = int(reader.getNSamples()[channel_index])
            epoch_count = total_samples // 3000
            stage_labels = expanded_labels(hypnogram, epoch_count)
            start_epoch = choose_five_class_window(stage_labels)
            signal = reader.readSignal(
                channel_index,
                start=start_epoch * 3000,
                n=50 * 3000,
            )

        x = signal.astype(np.float32).reshape(50, 1, 3000)
        y = stage_labels[start_epoch : start_epoch + 50]
        if x.shape != (50, 1, 3000) or set(y.tolist()) != set(range(5)):
            raise AssertionError(f"{record_id} 演示窗口契约失败")
        target = output / f"{record_id}.npz"
        np.savez(target, x=x, y=y)
        records.append(
            {
                "record_id": record_id,
                "start_epoch": start_epoch,
                "shape": list(x.shape),
                "class_counts": {
                    str(label): int(np.sum(y == label)) for label in range(5)
                },
                "file": str(target.resolve()),
            }
        )
    return records


def parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def finite_float_or_none(value: torch.Tensor) -> float | None:
    number = float(value.detach().cpu())
    return number if np.isfinite(number) else None


def run_walkthrough(data_root: Path, output_root: Path, device: torch.device) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    seed = 7
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    timings: dict[str, float] = {}
    started = time.perf_counter()

    raw_train = Sleep_Loader(data_root, "train", 16, ratio=RATIO, seed=seed, balance=False)
    raw_valid = Sleep_Loader(data_root, "valid", 16, ratio=RATIO, seed=seed, balance=False)
    raw_model = RawModel(device, str(output_root), name="1D-CNN")
    phase = time.perf_counter()
    raw_model.pretrain(raw_train, raw_valid, n_epoch=1, learn_rate=1e-4, gamma=0.95)
    raw_model.save()
    timings["raw_pretrain_seconds"] = time.perf_counter() - phase

    wave_train = Sleep_Loader(
        data_root, "train", 16, ratio=RATIO, seed=seed, balance=False,
        wave="only", waveshape=(30, 60),
    )
    wave_valid = Sleep_Loader(
        data_root, "valid", 16, ratio=RATIO, seed=seed, balance=False,
        wave="only", waveshape=(30, 60),
    )
    wave_model = WaveletModel(device, str(output_root), name="2D-CNN")
    phase = time.perf_counter()
    wave_model.pretrain(wave_train, wave_valid, n_epoch=1, learn_rate=1e-4, gamma=0.95)
    wave_model.save()
    timings["wave_pretrain_seconds"] = time.perf_counter() - phase

    raw_checkpoint = output_root / "RawModel" / "1D-CNN" / "network.pth"
    wave_checkpoint = output_root / "WaveletModel" / "2D-CNN" / "network.pth"
    merge_model = MergeModel(
        device,
        str(output_root),
        str(raw_checkpoint),
        str(wave_checkpoint),
        name="FFTCN",
    )
    merge_train = Sleep_Loader(
        data_root, "train", 1, seq_len=50, ratio=RATIO, seed=seed,
        balance=False, wave="with", waveshape=(30, 60),
    )
    merge_valid = Sleep_Loader(
        data_root, "valid", 1, seq_len=50, ratio=RATIO, seed=seed,
        balance=False, wave="with", waveshape=(30, 60),
    )
    merge_test = Sleep_Loader(
        data_root, "test", 1, seq_len=50, ratio=RATIO, seed=seed,
        balance=False, wave="with", waveshape=(30, 60),
    )
    phase = time.perf_counter()
    merge_model.finetune(
        merge_train,
        merge_valid,
        n_epoch=1,
        learn_rate=1e-4,
        lamb=1e-2,
        gamma=0.95,
        alpha=0,
        early_stoping=True,
    )
    merge_model.save()
    timings["merge_finetune_seconds"] = time.perf_counter() - phase

    logits, targets = merge_model.test_res(merge_test)
    metrics = merge_model.test(merge_test)
    raw_batch, raw_targets = next(iter(raw_train))
    wave_batch, wave_targets = next(iter(wave_train))
    merge_raw, merge_wave, merge_targets = next(iter(merge_test))
    merge_model.sleep_net.eval()
    with torch.no_grad():
        merge_raw_device = merge_raw.to(device)
        merge_wave_device = merge_wave.to(device)
        batch_size, sequence_length = merge_raw_device.shape[:2]
        raw_flat = merge_raw_device.reshape(batch_size * sequence_length, 1, 3000)
        wave_flat = merge_wave_device.reshape(batch_size * sequence_length, 1, 30, 60)
        raw_features = merge_model.raw_feature_net(raw_flat)
        wave_features = merge_model.wave_feature_net(wave_flat)
        fused = torch.cat(
            [
                raw_features.reshape(batch_size, sequence_length, -1),
                wave_features.reshape(batch_size, sequence_length, -1),
            ],
            dim=-1,
        )
        tcn_features = merge_model.sleep_net.sleep_net(fused)
        traced_logits = merge_model.sleep_net.classifier(
            tcn_features.reshape(batch_size * sequence_length, -1)
        )

    timings["total_training_and_evaluation_seconds"] = time.perf_counter() - started
    return {
        "notice": "教学贯通运行；仅 6 条记录、每阶段 1 轮，不代表论文复现指标。",
        "device": str(device),
        "seed": seed,
        "split_file_counts": {"train": 3, "valid": 1, "test": 2},
        "batch_shapes": {
            "raw_pretrain_input": list(raw_batch.shape),
            "raw_pretrain_target": list(raw_targets.shape),
            "wave_pretrain_input": list(wave_batch.shape),
            "wave_pretrain_target": list(wave_targets.shape),
            "merge_raw_input": list(merge_raw.shape),
            "merge_wave_input": list(merge_wave.shape),
            "merge_target": list(merge_targets.shape),
            "merge_logits_flattened": list(logits.shape),
            "merge_targets_flattened": list(targets.shape),
        },
        "pipeline_shapes": {
            "raw_epoch_flat": list(raw_flat.shape),
            "raw_features": list(raw_features.shape),
            "wave_epoch_flat": list(wave_flat.shape),
            "wave_features": list(wave_features.shape),
            "fused_sequence": list(fused.shape),
            "tcn_features": list(tcn_features.shape),
            "logits_flattened": list(traced_logits.shape),
        },
        "parameter_counts": {
            "raw_feature_net": parameter_count(raw_model.feature_net),
            "wave_feature_net": parameter_count(wave_model.feature_net),
            "complete_merge_net": parameter_count(merge_model.sleep_net),
        },
        "demo_metrics": {
            "accuracy": finite_float_or_none(metrics.accuracy_score()),
            "macro_f1": finite_float_or_none(metrics.f1_score()[1]),
            "kappa": finite_float_or_none(metrics.cohen_kappa_score()),
        },
        "checkpoints": {
            "raw": str(raw_checkpoint.resolve()),
            "wave": str(wave_checkpoint.resolve()),
            "merge": str((output_root / "MergeModel" / "FFTCN" / "network.pth").resolve()),
        },
        "timings": timings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--edf-root",
        type=Path,
        default=Path("datasets/sleep-edf-expanded-1.0.0/sleep-cassette"),
    )
    parser.add_argument("--demo-data", type=Path, default=Path("walkthrough/data"))
    parser.add_argument("--outputs", type=Path, default=Path("walkthrough/outputs"))
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("reproduction_artifacts/code_walkthrough/run_summary.json"),
    )
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = prepare_demo_data(args.edf_root, args.demo_data)
    print(f"已准备 {len(records)} 个真实 EEG 演示文件：{args.demo_data.resolve()}")
    if args.prepare_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("原始 data.loader 将 CWT 固定在 cuda:0，本次贯通需要 CUDA")
    result = run_walkthrough(args.demo_data, args.outputs, torch.device("cuda:0"))
    result["demo_records"] = records
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"贯通完成，摘要：{args.summary.resolve()}")


if __name__ == "__main__":
    main()
