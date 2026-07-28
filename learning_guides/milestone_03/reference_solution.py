"""里程碑 3：核心代码优先的完整参考答案。

这里只保留理解算法与通过当前练习测试所需的代码。
额外输入校验和生产环境防御策略请查看教程的“工程加固（选读）”。
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import numpy as np

from data.sleep_edf_preprocessor import (
    DEFAULT_SPLIT_SEED,
    LABEL_MAP,
    ProcessedSleepEDFRecord,
    SubjectSplit,
)
from data.sleep_edf_reader import RawSleepEDFRecord, SleepEDFPaths


def preprocess_record(
    record: RawSleepEDFRecord,
    max_wake_epochs: int = 60,
) -> ProcessedSleepEDFRecord:
    """映射标签、同步删除无效时段，并裁剪过量首尾 W。"""

    # 只有 LABEL_MAP 中的标签有效；这个 mask 同时用于 EEG 和标签。
    descriptions = np.asarray([str(label) for label in record.labels])
    valid_mask = np.asarray(
        [description in LABEL_MAP for description in descriptions],
        dtype=bool,
    )
    filtered_eeg = record.eeg[valid_mask]
    filtered_descriptions = descriptions[valid_mask]

    # 将有效字符串标签映射为 0..4；S3/S4 都由 LABEL_MAP 映射为 3。
    mapped_labels = np.asarray(
        [LABEL_MAP[description] for description in filtered_descriptions],
        dtype=np.int64,
    )

    # 找首末非 W。全 W 时没有可保留的睡眠主体，必须单独处理。
    non_wake = np.flatnonzero(mapped_labels != 0)
    if len(non_wake) == 0:
        raise ValueError("record contains no sleep epochs")

    # 两端最多各保留 max_wake_epochs 个 W；右端 +1 是因为切片不含 end。
    begin = max(0, int(non_wake[0]) - max_wake_epochs)
    end = min(len(mapped_labels), int(non_wake[-1]) + max_wake_epochs + 1)

    processed_eeg = np.ascontiguousarray(filtered_eeg[begin:end], dtype=np.float32)
    processed_labels = np.ascontiguousarray(mapped_labels[begin:end], dtype=np.int64)

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
    """按受试者而不是按夜晚记录进行可复现的 8:1:1 划分。"""

    # subject_id 去重并排序；排序使同一 seed 的输入顺序固定。
    subjects = sorted({record.subject_id for record in records})

    # 使用局部随机源，不修改程序其他部分的全局随机状态。
    rng = random.Random(seed)
    rng.shuffle(subjects)

    train_count = round(len(subjects) * 0.8)
    validation_count = round(len(subjects) * 0.1)

    return SubjectSplit(
        train=tuple(subjects[:train_count]),
        validation=tuple(subjects[train_count : train_count + validation_count]),
        test=tuple(subjects[train_count + validation_count :]),
    )


def save_processed_record(
    record: ProcessedSleepEDFRecord,
    output_dir: Path,
) -> Path:
    """按原 Sleep_Loader 的 x/y 契约保存 ``record_id.npz``。"""

    # 创建输出目录，并使用唯一 record_id，避免同一受试者两晚互相覆盖。
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{record.paths.record_id}.npz"

    np.savez(output_path, x=record.eeg, y=record.labels)
    return output_path
