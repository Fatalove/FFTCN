"""Milestone 1B: run the repository training path on processed Sleep-EDF-153.

This script is a boundary adapter, not a model rewrite. It keeps the original
RawModel, WaveletModel, MergeModel, CWT implementation, losses, optimizers, and
network definitions, while replacing only the hard-coded trainer entry points
with explicit command-line arguments and the already verified subject-level
Sleep-EDF split directories.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data.resample import offset_resample
from data.wavelet_torch import cwt
from models.merge.model import MergeModel
from models.raw.model import RawModel
from models.wavelet.model import WaveletModel


SPLIT_DIRS = {
    "train": "train",
    "valid": "validation",
    "test": "test",
}


@dataclass(frozen=True)
class RunConfig:
    processed_root: Path
    output_root: Path
    summary_path: Path
    device: torch.device
    seed: int
    raw_epochs: int
    wave_epochs: int
    merge_epochs: int
    raw_batch_size: int
    wave_batch_size: int
    merge_batch_size: int
    learning_rate: float
    gamma: float
    lamb: float
    alpha: float
    balance_pretrain: bool
    seq_len: int
    max_files_per_split: int
    max_epochs_per_file: int
    dry_run: bool


class SourceSplitDataset(Dataset):
    """Read the verified Sleep-EDF split directories with original data semantics."""

    def __init__(
        self,
        processed_root: Path,
        split: str,
        seq_len: int,
        *,
        balance: bool = False,
        seed: int = 0,
        wave: str = "not",
        waveshape: tuple[int, int] = (30, 60),
        device: torch.device = torch.device("cuda:0"),
        max_files: int = 0,
        max_epochs_per_file: int = 0,
    ) -> None:
        super().__init__()
        if split not in SPLIT_DIRS:
            raise ValueError(f"unknown split: {split}")
        split_dir = processed_root / SPLIT_DIRS[split]
        if not split_dir.exists():
            raise FileNotFoundError(split_dir)

        # 原代码：data/loader.py 读取一个单层 root_path，再用 ratio 按文件名切分 train/valid/test。
        # 修改代码：1B 直接读取里程碑 3 已验收的 train/validation/test 子目录，保留 x/y NPZ 契约。
        files = sorted(split_dir.glob("*.npz"))
        if max_files:
            files = files[:max_files]
        if not files:
            raise RuntimeError(f"{split_dir} contains no NPZ files")

        self.seq_len = seq_len
        self.wave = wave
        self.files = files
        self.raw: list[torch.Tensor] = []
        self.spectrum: list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []
        self.all_counter: Counter[int] = Counter()
        self.wavelet = cwt(1 / 100, 3000, device=device)

        np.random.seed(seed)
        pbar = tqdm(files, desc=f"{split}:{wave}", ncols=0)
        for path in pbar:
            with np.load(path) as loaded:
                data = loaded["x"]
                label = loaded["y"]
            if max_epochs_per_file:
                data = data[:max_epochs_per_file]
                label = label[:max_epochs_per_file]
            if balance:
                if seq_len != 1:
                    raise ValueError("offset_resample is only valid for seq_len=1")
                data, label = offset_resample(data, label, 300)

            sample_n = len(label) // seq_len
            if sample_n == 0:
                continue
            data = data[: sample_n * seq_len]
            label = label[: sample_n * seq_len]
            self.all_counter += Counter(label.tolist())
            pbar.set_postfix({"name": path.name, "samples": sample_n})

            if seq_len != 1:
                data = data.reshape((-1, seq_len, *data.shape[1:]))
                label = label.reshape((-1, seq_len))

            data_tensor = torch.tensor(data, dtype=torch.float)
            label_tensor = torch.tensor(label, dtype=torch.long)
            if wave != "only":
                self.raw.extend(data_tensor)
            if wave != "not":
                spectra = self.wavelet(data_tensor, waveshape[1])
                self.spectrum.extend(spectra.cpu().to(torch.float16))
            self.targets.extend(label_tensor)

        if not self.targets:
            raise RuntimeError(f"{split} split produced no samples")
        print(f"{split}:{wave} total:", self.all_counter)

    def __getitem__(self, index: int):
        if self.wave == "with":
            item = [self.raw[index], self.spectrum[index].float()]
        elif self.wave == "only":
            item = [self.spectrum[index].float()]
        elif self.wave == "not":
            item = [self.raw[index]]
        else:
            raise ValueError(f"unknown wave mode: {self.wave}")
        item.append(self.targets[index])
        return item

    def __len__(self) -> int:
        return len(self.targets)


def source_loader(
    config: RunConfig,
    split: str,
    batch_size: int,
    *,
    seq_len: int = 1,
    balance: bool = False,
    wave: str = "not",
) -> DataLoader:
    dataset = SourceSplitDataset(
        config.processed_root,
        split,
        seq_len,
        balance=balance,
        seed=config.seed,
        wave=wave,
        waveshape=(30, 60),
        device=config.device,
        max_files=config.max_files_per_split,
        max_epochs_per_file=config.max_epochs_per_file,
    )
    # 原代码：Sleep_Loader(..., shuffle=True) 对 train/valid/test 都启用 shuffle。
    # 修改代码：保持 shuffle=True，额外固定 generator，便于 1B 结果复核。
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    return DataLoader(dataset, batch_size, shuffle=True, generator=generator)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def safe_metrics(outputs: torch.Tensor, targets: torch.Tensor, n_class: int = 5) -> dict[str, object]:
    pred = torch.argmax(outputs.detach().cpu().reshape(-1, n_class), dim=1)
    true = targets.detach().cpu().flatten()
    mask = (true >= 0) & (true < n_class)
    pred = pred[mask]
    true = true[mask]
    indices = n_class * true + pred
    cm = torch.bincount(indices, minlength=n_class * n_class).reshape(n_class, n_class).float()

    correct = torch.diag(cm)
    pred_count = cm.sum(dim=0)
    true_count = cm.sum(dim=1)
    precision = torch.where(pred_count > 0, correct / pred_count, torch.zeros_like(correct))
    recall = torch.where(true_count > 0, correct / true_count, torch.zeros_like(correct))
    f1 = torch.where(
        precision + recall > 0,
        2 * precision * recall / (precision + recall),
        torch.zeros_like(precision),
    )
    total = cm.sum()
    acc = correct.sum() / total
    pe = pred_count.mul(true_count).sum() / (total * total)
    kappa = torch.where(1 - pe != 0, (acc - pe) / (1 - pe), torch.tensor(0.0))
    return {
        "confusion_matrix": cm.int().tolist(),
        "accuracy": finite_or_none(acc.item()),
        "precision": [finite_or_none(v.item()) for v in precision],
        "recall": [finite_or_none(v.item()) for v in recall],
        "f1": [finite_or_none(v.item()) for v in f1],
        "macro_f1": finite_or_none(f1.mean().item()),
        "kappa": finite_or_none(kappa.item()),
        "support": [int(v.item()) for v in true_count],
    }


def count_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def first_shapes(model: MergeModel, loader: DataLoader, device: torch.device) -> dict[str, list[int]]:
    raw, wave, targets = next(iter(loader))
    model.sleep_net.eval()
    with torch.no_grad():
        raw_device = raw.to(device)
        wave_device = wave.to(device)
        batch_size, seq_len = raw_device.shape[:2]
        raw_flat = raw_device.reshape(batch_size * seq_len, 1, 3000)
        wave_flat = wave_device.reshape(batch_size * seq_len, 1, 30, 60)
        raw_features = model.raw_feature_net(raw_flat)
        wave_features = model.wave_feature_net(wave_flat)
        fused = torch.cat(
            [
                raw_features.reshape(batch_size, seq_len, -1),
                wave_features.reshape(batch_size, seq_len, -1),
            ],
            dim=-1,
        )
        tcn_features = model.sleep_net.sleep_net(fused)
        logits = model.sleep_net.classifier(tcn_features.reshape(batch_size * seq_len, -1))
    return {
        "merge_raw_input": list(raw.shape),
        "merge_wave_input": list(wave.shape),
        "merge_target": list(targets.shape),
        "raw_epoch_flat": list(raw_flat.shape),
        "raw_features": list(raw_features.shape),
        "wave_epoch_flat": list(wave_flat.shape),
        "wave_features": list(wave_features.shape),
        "fused_sequence": list(fused.shape),
        "tcn_features": list(tcn_features.shape),
        "logits_flattened": list(logits.shape),
    }


def run(config: RunConfig) -> dict[str, object]:
    set_seed(config.seed)
    started = time.perf_counter()
    timings: dict[str, float] = {}
    config.output_root.mkdir(parents=True, exist_ok=True)

    print("=========================================raw pretrain=========================================")
    raw_model = RawModel(config.device, str(config.output_root), name="1D-CNN")
    raw_train = source_loader(
        config,
        "train",
        config.raw_batch_size,
        balance=config.balance_pretrain,
        wave="not",
    )
    raw_valid = source_loader(config, "valid", config.raw_batch_size, wave="not")
    phase = time.perf_counter()
    raw_model.pretrain(raw_train, raw_valid, config.raw_epochs, config.learning_rate, config.gamma)
    raw_model.save()
    timings["raw_pretrain_seconds"] = time.perf_counter() - phase

    print("======================================wavelet pretrain======================================")
    wave_model = WaveletModel(config.device, str(config.output_root), name="2D-CNN")
    wave_train = source_loader(
        config,
        "train",
        config.wave_batch_size,
        balance=config.balance_pretrain,
        wave="only",
    )
    wave_valid = source_loader(config, "valid", config.wave_batch_size, wave="only")
    phase = time.perf_counter()
    wave_model.pretrain(wave_train, wave_valid, config.wave_epochs, config.learning_rate, config.gamma)
    wave_model.save()
    timings["wave_pretrain_seconds"] = time.perf_counter() - phase

    raw_checkpoint = config.output_root / "RawModel" / "1D-CNN" / "network.pth"
    wave_checkpoint = config.output_root / "WaveletModel" / "2D-CNN" / "network.pth"
    print("==========================================merge finetune==========================================")
    merge_model = MergeModel(
        config.device,
        str(config.output_root),
        str(raw_checkpoint),
        str(wave_checkpoint),
        name="FFTCN",
    )
    merge_train = source_loader(
        config,
        "train",
        config.merge_batch_size,
        seq_len=config.seq_len,
        wave="with",
    )
    merge_valid = source_loader(
        config,
        "valid",
        config.merge_batch_size,
        seq_len=config.seq_len,
        wave="with",
    )
    phase = time.perf_counter()
    merge_model.finetune(
        merge_train,
        merge_valid,
        config.merge_epochs,
        config.learning_rate,
        config.lamb,
        config.gamma,
        config.alpha,
        early_stoping=True,
    )
    merge_model.save()
    timings["merge_finetune_seconds"] = time.perf_counter() - phase

    print("============================================test============================================")
    merge_test = source_loader(
        config,
        "test",
        config.merge_batch_size,
        seq_len=config.seq_len,
        wave="with",
    )
    outputs, targets = merge_model.test_res(merge_test)
    metrics = safe_metrics(outputs, targets)
    # 原代码：model.test(...) 保存 metrics.xlsx，但 precision/recall 零分母时可能产生 NaN。
    # 修改代码：仍调用原 test 生成 Excel；JSON 摘要另用 safe_metrics 记录零分母为 0 的可解析指标。
    merge_model.test(merge_test)
    timings["total_seconds"] = time.perf_counter() - started

    return {
        "notice": "1B source-level reproduction precheck; dry_run metrics are not paper-comparable.",
        "dry_run": config.dry_run,
        "device": str(config.device),
        "seed": config.seed,
        "processed_root": str(config.processed_root.resolve()),
        "output_root": str(config.output_root.resolve()),
        "epochs": {
            "raw": config.raw_epochs,
            "wavelet": config.wave_epochs,
            "merge": config.merge_epochs,
        },
        "batch_size": {
            "raw": config.raw_batch_size,
            "wavelet": config.wave_batch_size,
            "merge": config.merge_batch_size,
        },
        "limits": {
            "max_files_per_split": config.max_files_per_split,
            "max_epochs_per_file": config.max_epochs_per_file,
            "balance_pretrain": config.balance_pretrain,
        },
        "split_counters": {
            "raw_train": dict(raw_train.dataset.all_counter),
            "raw_valid": dict(raw_valid.dataset.all_counter),
            "wave_train": dict(wave_train.dataset.all_counter),
            "wave_valid": dict(wave_valid.dataset.all_counter),
            "merge_train": dict(merge_train.dataset.all_counter),
            "merge_valid": dict(merge_valid.dataset.all_counter),
            "merge_test": dict(merge_test.dataset.all_counter),
        },
        "sample_counts": {
            "raw_train": len(raw_train.dataset),
            "raw_valid": len(raw_valid.dataset),
            "wave_train": len(wave_train.dataset),
            "wave_valid": len(wave_valid.dataset),
            "merge_train": len(merge_train.dataset),
            "merge_valid": len(merge_valid.dataset),
            "merge_test": len(merge_test.dataset),
        },
        "shape_trace": first_shapes(merge_model, merge_test, config.device),
        "parameter_counts": {
            "raw_feature_net": count_parameters(raw_model.feature_net),
            "wave_feature_net": count_parameters(wave_model.feature_net),
            "complete_merge_net": count_parameters(merge_model.sleep_net),
        },
        "metrics": metrics,
        "checkpoints": {
            "raw": str(raw_checkpoint.resolve()),
            "wavelet": str(wave_checkpoint.resolve()),
            "merge": str((config.output_root / "MergeModel" / "FFTCN" / "network.pth").resolve()),
            "merge_best": str((config.output_root / "MergeModel" / "FFTCN" / "best_network.pth").resolve()),
        },
        "timings": timings,
    }


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", type=Path, default=Path("datasets/sleep-edf-153-processed-v1"))
    parser.add_argument("--output-root", type=Path, default=Path("source_reproduction/outputs"))
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("reproduction_artifacts/source_reproduction/run_summary.json"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raw-epochs", type=int, default=20)
    parser.add_argument("--wave-epochs", type=int, default=20)
    parser.add_argument("--merge-epochs", type=int, default=50)
    parser.add_argument("--raw-batch-size", type=int, default=128)
    parser.add_argument("--wave-batch-size", type=int, default=128)
    parser.add_argument("--merge-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lamb", type=float, default=1e-2)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--max-files-per-split", type=int, default=0)
    parser.add_argument("--max-epochs-per-file", type=int, default=0)
    parser.add_argument("--no-balance-pretrain", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use tiny data and 1 epoch per phase to validate code wiring only.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> RunConfig:
    dry_run = bool(args.dry_run)
    return RunConfig(
        processed_root=args.processed_root,
        output_root=args.output_root,
        summary_path=args.summary,
        device=choose_device(args.device),
        seed=args.seed,
        raw_epochs=1 if dry_run else args.raw_epochs,
        wave_epochs=1 if dry_run else args.wave_epochs,
        merge_epochs=1 if dry_run else args.merge_epochs,
        raw_batch_size=min(args.raw_batch_size, 16) if dry_run else args.raw_batch_size,
        wave_batch_size=min(args.wave_batch_size, 16) if dry_run else args.wave_batch_size,
        merge_batch_size=1 if dry_run else args.merge_batch_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        lamb=args.lamb,
        alpha=args.alpha,
        balance_pretrain=False if dry_run else not args.no_balance_pretrain,
        seq_len=args.seq_len,
        max_files_per_split=1 if dry_run and args.max_files_per_split == 0 else args.max_files_per_split,
        max_epochs_per_file=0 if dry_run and args.max_epochs_per_file == 0 else args.max_epochs_per_file,
        dry_run=dry_run,
    )


def main() -> None:
    config = build_config(parse_args())
    result = run(config)
    config.summary_path.parent.mkdir(parents=True, exist_ok=True)
    config.summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"summary: {config.summary_path.resolve()}")


if __name__ == "__main__":
    main()
