"""Sleep-EDF 标签清洗、首尾 W 裁剪与受试者划分（里程碑 3 练习）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from data.sleep_edf_reader import RawSleepEDFRecord, SleepEDFPaths


LABEL_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}
INVALID_LABELS = {"Movement time", "Sleep stage ?"}
DEFAULT_SPLIT_SEED = 42


@dataclass(frozen=True)
class ProcessedSleepEDFRecord:
    """已映射为五分类且完成首尾 W 裁剪的单条记录。"""

    paths: SleepEDFPaths
    sampling_rate: float
    eeg: np.ndarray  # float32, [N, 1, 3000]
    labels: np.ndarray  # int64, [N]，只允许 0..4


@dataclass(frozen=True)
class SubjectSplit:
    """受试者级划分；同一受试者的所有夜晚只能属于一个集合。"""

    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]


def preprocess_record(
    record: RawSleepEDFRecord,
    max_wake_epochs: int = 60,
) -> ProcessedSleepEDFRecord:
    """同步清洗 EEG/标签并裁剪过量的首尾 W。

    用户练习伪代码：
    1. 用 ``LABEL_MAP`` 构造有效标签 mask；
    2. 对 EEG 和标签应用同一个 mask，删除 Movement/Unknown；
    3. 找到映射后首个和末个非 W 标签；若不存在则显式报错；
    4. 从首个非 W 向前、末个非 W 向后各保留最多 60 段；
    5. 验证 dtype、形状、有限值和标签集合，再构造返回值。

    本项目为对齐原仓库，删除无效时段后采用紧凑序列。它可能把无效时段
    两侧在数组中接到一起；该限制已记录，后续构造 TCN 序列时需再次评估。
    """

    eeg = np.asarray(record.eeg)
    raw_labels = np.asarray(record.labels)

    descriptions = [str(label) for label in raw_labels]

    valid_mask = np.fromiter(
        (description in LABEL_MAP.keys() for description in descriptions),
        dtype=np.bool_,
        count=len(descriptions),
    )

    filtered_eeg = eeg[valid_mask]
    filtered_labels = raw_labels[valid_mask]

    mapped_labels = np.fromiter(
        (LABEL_MAP[label] for label in filtered_labels),
        dtype=np.int64,
    )

    non_wake = np.flatnonzero(mapped_labels != 0)
    if len(non_wake) == 0:
        raise ValueError("No non-wake labels found in record")

    begin =max(0, int(non_wake[0] - max_wake_epochs))
    end = min(len(mapped_labels), int(non_wake[-1]) + max_wake_epochs + 1)

    processed_eeg = np.ascontiguousarray(filtered_eeg[begin: end], dtype=np.float32)
    processed_labels = np.ascontiguousarray(mapped_labels[begin: end], dtype=np.int64)

    if len(processed_eeg) != len(processed_labels):
        raise ValueError("eeg and labels must have the same length")
    if not np.isfinite(processed_eeg).all():
        raise ValueError("eeg must be finite")

    return ProcessedSleepEDFRecord(
        paths=record.paths,
        sampling_rate=record.sampling_rate,
        eeg=processed_eeg,
        labels=processed_labels,

    )




def split_subjects(
    records: Sequence[SleepEDFPaths],
    seed: int = DEFAULT_SPLIT_SEED,
) -> SubjectSplit:
    """按受试者进行可复现的 8:1:1 划分。

    固定契约：先对去重且排序的 subject_id 使用局部随机数生成器洗牌；78 名
    受试者取 62/8/8。不得使用或修改全局随机状态，也不得按记录文件切分。
    """
    subjects = sorted(set(record.subject_id for record in records))
    if len(subjects) < 3:
        raise ValueError("Subjects must have at least 3 subjects")

    rng = np.random.RandomState(seed)
    rng.shuffle(subjects)

    train_count = round(0.8 * len(subjects))
    val_count = round(0.1 * len(subjects))

    train = tuple(subjects[:train_count])
    validation = tuple(subjects[train_count:train_count + val_count])
    test = tuple(subjects[train_count + val_count:])

    if set(train) & set(validation)  or set(train) & set(test) or set(validation) & set(test):
        raise ValueError("Overlapping subjects in train, validation, or test")

    return SubjectSplit(train=train, validation=validation, test=test)



def save_processed_record(record: ProcessedSleepEDFRecord, output_dir: Path) -> Path:
    """保存与原 ``Sleep_Loader`` 兼容的 ``x/y`` NPZ，并返回文件路径。

    文件名使用 ``record_id.npz``；``x`` 必须为 float32，``y`` 必须为 int64。
    """
    eeg, labels = record.eeg, record.labels

    if eeg.dtype != np.float32:
        eeg = np.ascontiguousarray(eeg, dtype=np.float32)

    if labels.dtype != np.int64:
        labels = np.ascontiguousarray(labels, dtype=np.int64)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{record.paths.record_id}.npz"
    np.savez(output_path, x=eeg, y=labels)
    return output_path
