from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from data.sleep_edf_reader import SleepEDFPaths, discover_records, read_record


def synthetic_record_ids() -> list[str]:
    """构造与 Sleep-EDF-153 规模一致的 153 条记录和 78 名受试者。

    前 75 名受试者各有两个夜晚，最后 3 名各有一个夜晚：
    ``75 * 2 + 3 = 153``。这里只测试文件配对与受试者语义，不伪装成
    正式 Sleep-EDF 文件内容。
    """

    record_ids: list[str] = []
    for subject_number in range(400, 478):
        nights = ("1", "2") if subject_number < 475 else ("1",)
        for night_id in nights:
            record_ids.append(f"SC{subject_number:03d}{night_id}")
    return record_ids


class FakeEdfReader:
    """提供 ``read_record`` 所需的最小 pyEDFlib 行为。"""

    instances: list["FakeEdfReader"] = []

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.closed = False
        self.__class__.instances.append(self)

    def getSignalLabels(self) -> list[str]:
        return ["EEG Fpz-Cz"]

    def getSampleFrequency(self, channel_index: int) -> float:
        if channel_index != 0:
            raise AssertionError("测试夹具只提供第 0 个 EEG 通道")
        return 100.0

    def readSignal(self, channel_index: int) -> np.ndarray:
        if channel_index != 0:
            raise AssertionError("测试夹具只提供第 0 个 EEG 通道")
        # 三个完整 epoch 后追加 17 点，验证读取器会丢弃不足 30 秒的尾部。
        return np.arange(3 * 3000 + 17, dtype=np.float64)

    def readAnnotations(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.asarray([0.0, 30.0, 60.0]),
            np.asarray([30.0, 30.0, 60.0]),
            np.asarray(["Sleep stage W", "Sleep stage 2", "Sleep stage R"], dtype=object),
        )

    def close(self) -> None:
        self.closed = True


class SleepEDFReaderTest(unittest.TestCase):
    def test_discover_all_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for record_id in synthetic_record_ids():
                (root / f"{record_id}E0-PSG.edf").touch()
                (root / f"{record_id}EC-Hypnogram.edf").touch()

            records = discover_records(root)
            self.assertEqual(len(records), 153)
            self.assertEqual(len({record.record_id for record in records}), 153)
            self.assertEqual(len({record.subject_id for record in records}), 78)
            self.assertEqual(records[0].record_id, "SC4001")
            for record in records:
                self.assertTrue(record.psg_path.is_file())
                self.assertTrue(record.hypnogram_path.is_file())

    def test_read_record_aligns_mocked_psg_and_annotations(self) -> None:
        paths = SleepEDFPaths(
            record_id="SC4001",
            subject_id="SC400",
            night_id="1",
            psg_path=Path("SC4001E0-PSG.edf"),
            hypnogram_path=Path("SC4001EC-Hypnogram.edf"),
        )
        FakeEdfReader.instances.clear()
        with patch("data.sleep_edf_reader.pyedflib.EdfReader", FakeEdfReader):
            loaded = read_record(paths)

        self.assertEqual(loaded.paths.record_id, "SC4001")
        self.assertEqual(loaded.paths.subject_id, "SC400")
        self.assertEqual(loaded.paths.night_id, "1")
        self.assertEqual(loaded.sampling_rate, 100.0)
        self.assertEqual(loaded.eeg.shape, (3, 1, 3000))
        self.assertEqual(loaded.labels.shape, (3,))
        self.assertEqual(loaded.eeg.dtype, np.float32)
        self.assertTrue(np.isfinite(loaded.eeg).all())
        np.testing.assert_array_equal(
            loaded.labels,
            np.asarray(["Sleep stage W", "Sleep stage 2", "Sleep stage R"], dtype=object),
        )
        self.assertEqual(len(FakeEdfReader.instances), 2)
        self.assertTrue(all(reader.closed for reader in FakeEdfReader.instances))


if __name__ == "__main__":
    unittest.main()
