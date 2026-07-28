from __future__ import annotations

import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from data.sleep_edf_preprocessor import (
    ProcessedSleepEDFRecord,
    preprocess_record,
    save_processed_record,
    split_subjects,
)
from data.sleep_edf_reader import RawSleepEDFRecord, SleepEDFPaths


def synthetic_raw_record() -> RawSleepEDFRecord:
    labels = (
        ["Sleep stage W"] * 70
        + [
            "Sleep stage 1",
            "Sleep stage 2",
            "Sleep stage 3",
            "Sleep stage 4",
            "Sleep stage R",
            "Movement time",
            "Sleep stage ?",
        ]
        + ["Sleep stage W"] * 70
    )
    epoch_ids = np.arange(len(labels), dtype=np.float32)
    eeg = np.broadcast_to(epoch_ids[:, None, None], (len(labels), 1, 3000)).copy()
    paths = SleepEDFPaths(
        record_id="SC9991",
        subject_id="SC999",
        night_id="1",
        psg_path=Path("SC9991E0-PSG.edf"),
        hypnogram_path=Path("SC9991EC-Hypnogram.edf"),
    )
    return RawSleepEDFRecord(paths, 100.0, eeg, np.asarray(labels, dtype=object))


def synthetic_reference_count_record() -> RawSleepEDFRecord:
    """用小型内存夹具保留 SC4001 验收摘要中的五类计数契约。

    首尾各 70 个 W 会分别裁成 60 个；中间再放 68 个 W。S3/S4 各 110
    个并在预处理时合并为 N3，因此输出总数为 841。
    """

    labels = (
        ["Sleep stage W"] * 70
        + ["Sleep stage 1"] * 58
        + ["Sleep stage W"] * 68
        + ["Sleep stage 2"] * 250
        + ["Sleep stage 3"] * 110
        + ["Sleep stage 4"] * 110
        + ["Sleep stage R"] * 125
        + ["Sleep stage W"] * 70
    )
    paths = SleepEDFPaths(
        record_id="SC4001",
        subject_id="SC400",
        night_id="1",
        psg_path=Path("SC4001E0-PSG.edf"),
        hypnogram_path=Path("SC4001EC-Hypnogram.edf"),
    )
    eeg = np.zeros((len(labels), 1, 3000), dtype=np.float32)
    return RawSleepEDFRecord(paths, 100.0, eeg, np.asarray(labels, dtype=object))


def synthetic_dataset_paths() -> list[SleepEDFPaths]:
    """构造 153 条记录/78 名受试者的内存路径，用于划分行为测试。"""

    records: list[SleepEDFPaths] = []
    for subject_number in range(400, 478):
        nights = ("1", "2") if subject_number < 475 else ("1",)
        for night_id in nights:
            record_id = f"SC{subject_number:03d}{night_id}"
            records.append(
                SleepEDFPaths(
                    record_id=record_id,
                    subject_id=record_id[:5],
                    night_id=night_id,
                    psg_path=Path(f"{record_id}E0-PSG.edf"),
                    hypnogram_path=Path(f"{record_id}EC-Hypnogram.edf"),
                )
            )
    return records


class SleepEDFPreprocessorTest(unittest.TestCase):
    def test_mapping_removal_and_wake_trim_stay_synchronized(self) -> None:
        processed = preprocess_record(synthetic_raw_record())

        self.assertEqual(processed.eeg.shape, (125, 1, 3000))
        self.assertEqual(processed.labels.shape, (125,))
        self.assertEqual(processed.eeg.dtype, np.float32)
        self.assertEqual(processed.labels.dtype, np.int64)
        self.assertEqual(set(processed.labels.tolist()), set(range(5)))
        self.assertEqual(Counter(processed.labels.tolist()), Counter({0: 120, 3: 2, 1: 1, 2: 1, 4: 1}))
        self.assertEqual(float(processed.eeg[0, 0, 0]), 10.0)
        self.assertEqual(float(processed.eeg[-1, 0, 0]), 136.0)

    def test_reference_counts_from_synthetic_record(self) -> None:
        processed = preprocess_record(synthetic_reference_count_record())
        self.assertEqual(processed.eeg.shape, (841, 1, 3000))
        self.assertEqual(
            Counter(processed.labels.tolist()),
            Counter({0: 188, 1: 58, 2: 250, 3: 220, 4: 125}),
        )

    def test_subject_split_is_reproducible_and_disjoint(self) -> None:
        records = synthetic_dataset_paths()
        first = split_subjects(records, seed=42)
        second = split_subjects(records, seed=42)
        self.assertEqual(first, second)
        self.assertEqual((len(first.train), len(first.validation), len(first.test)), (62, 8, 8))
        train, validation, test = set(first.train), set(first.validation), set(first.test)
        self.assertFalse(train & validation)
        self.assertFalse(train & test)
        self.assertFalse(validation & test)
        self.assertEqual(train | validation | test, {record.subject_id for record in records})
        assignment = {
            subject: split_name
            for split_name, subjects in (
                ("train", train),
                ("validation", validation),
                ("test", test),
            )
            for subject in subjects
        }
        for record in records:
            self.assertIn(record.subject_id, assignment)

    def test_saved_npz_matches_original_loader_contract(self) -> None:
        processed = preprocess_record(synthetic_raw_record())
        with tempfile.TemporaryDirectory() as directory:
            path = save_processed_record(processed, Path(directory))
            self.assertEqual(path.name, "SC9991.npz")
            with np.load(path) as saved:
                self.assertEqual(set(saved.files), {"x", "y"})
                np.testing.assert_array_equal(saved["x"], processed.eeg)
                np.testing.assert_array_equal(saved["y"], processed.labels)


if __name__ == "__main__":
    unittest.main()
