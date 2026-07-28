from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np


# 正式测试默认导入用户练习；Codex 准备阶段可切换到独立参考答案，
# 让两份实现接受完全相同的可观察行为契约。
if os.environ.get("FFTCN_MILESTONE_09B_CACHE_REFERENCE") == "1":
    from learning_guides.milestone_09b.reference_solution import (
        build_raw_label_cache,
        build_wave_cache,
    )
else:
    from data.sleep_edf_training_cache import (
        build_raw_label_cache,
        build_wave_cache,
    )


class SleepEDFTrainingCacheTest(unittest.TestCase):
    def _write_processed_split(self, root: Path) -> Path:
        """写入两个极小 NPZ；创建顺序故意与文件名排序相反。"""

        split_dir = root / "processed" / "train"
        split_dir.mkdir(parents=True)

        raw_b = np.stack(
            [
                np.full((1, 3000), 20.0, dtype=np.float32),
                np.full((1, 3000), 21.0, dtype=np.float32),
            ]
        )
        labels_b = np.array([3, 4], dtype=np.int64)
        np.savez(split_dir / "record_b.npz", x=raw_b, y=labels_b)

        raw_a = np.stack(
            [
                np.full((1, 3000), 0.0, dtype=np.float32),
                np.full((1, 3000), 1.0, dtype=np.float32),
                np.full((1, 3000), 2.0, dtype=np.float32),
            ]
        )
        labels_a = np.array([0, 1, 2], dtype=np.int64)
        np.savez(split_dir / "record_a.npz", x=raw_a, y=labels_a)
        return split_dir

    def test_raw_label_cache_preserves_sorted_record_order_and_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            processed_split = self._write_processed_split(root)
            cache_root = root / "cache"

            manifest_path = build_raw_label_cache(
                processed_split,
                cache_root,
                split="train",
            )
            raw = np.load(cache_root / "train" / "raw.npy")
            labels = np.load(cache_root / "train" / "labels.npy")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(raw.shape, (5, 1, 3000))
        self.assertEqual(raw.dtype, np.dtype(np.float32))
        self.assertEqual(labels.shape, (5,))
        self.assertEqual(labels.dtype, np.dtype(np.int64))
        self.assertEqual(raw[:, 0, 0].tolist(), [0.0, 1.0, 2.0, 20.0, 21.0])
        self.assertEqual(labels.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(manifest["sample_count"], 5)
        self.assertEqual(
            manifest["records"],
            [
                {
                    "record_id": "record_a",
                    "source_file": "record_a.npz",
                    "start": 0,
                    "stop": 3,
                },
                {
                    "record_id": "record_b",
                    "source_file": "record_b.npz",
                    "start": 3,
                    "stop": 5,
                },
            ],
        )

    def test_raw_label_cache_rejects_misaligned_record(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            processed_split = root / "processed" / "train"
            processed_split.mkdir(parents=True)
            raw = np.zeros((2, 1, 3000), dtype=np.float32)
            labels = np.zeros((1,), dtype=np.int64)
            np.savez(processed_split / "broken.npz", x=raw, y=labels)

            with self.assertRaisesRegex(ValueError, "样本轴不一致"):
                build_raw_label_cache(processed_split, root / "cache", "train")

    def test_raw_label_cache_rejects_invalid_raw_shape(self) -> None:
        """错误 epoch 形状不能依靠 NumPy 广播静默写入固定缓存。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            processed_split = root / "processed" / "train"
            processed_split.mkdir(parents=True)

            # 每个 epoch 只有 1 个采样点；若生产者不先检查 [N,1,3000]，
            # NumPy 会把最后一轴从 1 广播成 3000，产生形状正确但内容错误的缓存。
            raw = np.zeros((2, 1, 1), dtype=np.float32)
            labels = np.zeros((2,), dtype=np.int64)
            np.savez(processed_split / "broken_shape.npz", x=raw, y=labels)

            # 这里只约束可观察行为：生产者必须拒绝错误形状；
            # 不限制用户采用哪种内部判断写法或错误消息。
            with self.assertRaises(ValueError):
                build_raw_label_cache(processed_split, root / "cache", "train")

    def test_wave_cache_uses_same_global_epoch_order_and_updates_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            processed_split = self._write_processed_split(root)
            cache_root = root / "cache"
            build_raw_label_cache(processed_split, cache_root, "train")

            seen_markers: list[float] = []

            def fake_wave_transform(raw_epoch: np.ndarray) -> np.ndarray:
                marker = float(raw_epoch[0, 0])
                seen_markers.append(marker)
                return np.full((1, 30, 60), marker, dtype=np.float32)

            wave_path = build_wave_cache(
                cache_root,
                "train",
                wave_transform=fake_wave_transform,
            )
            wave = np.load(wave_path)
            manifest = json.loads(
                (cache_root / "train" / "manifest.json").read_text(encoding="utf-8")
            )

        self.assertEqual(seen_markers, [0.0, 1.0, 2.0, 20.0, 21.0])
        self.assertEqual(wave.shape, (5, 1, 30, 60))
        self.assertEqual(wave.dtype, np.dtype(np.float16))
        self.assertEqual(wave[:, 0, 0, 0].tolist(), seen_markers)
        self.assertEqual(manifest["wave_shape"], [5, 1, 30, 60])
        self.assertEqual(manifest["wave_dtype"], "float16")


if __name__ == "__main__":
    unittest.main()
