from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np


# 正式测试默认导入用户练习；Codex 准备阶段可设置环境变量，让同一行为契约
# 验证独立参考答案，而不是把参考答案复制进正式实现。
if os.environ.get("FFTCN_MILESTONE_09B_REFERENCE") == "1":
    from learning_guides.milestone_09b.reference_data_loading import (
        FullTrainingConfig,
        RecordSpan,
        SleepEDFSequenceDataset,
        build_reproducible_loader,
        build_sequence_positions,
        load_record_spans,
    )
else:
    from training.sleep_edf_full_run import (
        FullTrainingConfig,
        RecordSpan,
        SleepEDFSequenceDataset,
        build_reproducible_loader,
        build_sequence_positions,
        load_record_spans,
    )


class SleepEDFFullRunDataTest(unittest.TestCase):
    def _write_cache(self, root: Path) -> Path:
        split_dir = root / "train"
        split_dir.mkdir(parents=True)

        raw = np.arange(8 * 1 * 4, dtype=np.float32).reshape(8, 1, 4)
        wave = np.arange(8 * 1 * 2 * 3, dtype=np.float16).reshape(8, 1, 2, 3)
        labels = np.array([0, 1, 2, 3, 4, 0, 1, 2], dtype=np.int64)
        np.save(split_dir / "raw.npy", raw)
        np.save(split_dir / "wave.npy", wave)
        np.save(split_dir / "labels.npy", labels)

        manifest = {
            "split": "train",
            "records": [
                {"record_id": "record_a", "start": 0, "stop": 5},
                {"record_id": "record_b", "start": 5, "stop": 8},
            ],
        }
        with (split_dir / "manifest.json").open("w", encoding="utf-8") as file:
            json.dump(manifest, file)
        return split_dir

    def test_config_metadata_is_json_ready_and_preserves_training_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = FullTrainingConfig(
                data_cache_dir=root / "cache",
                output_dir=root / "outputs",
                seed=17,
                offset_samples=321,
                sequence_length=50,
                fusion_batch_size=4,
            )
            metadata = config.to_metadata()
            encoded = json.dumps(metadata)

        self.assertIsInstance(encoded, str)
        self.assertEqual(metadata["seed"], 17)
        self.assertEqual(metadata["offset_samples"], 321)
        self.assertEqual(metadata["sequence_length"], 50)
        self.assertEqual(metadata["fusion_batch_size"], 4)
        self.assertIsInstance(metadata["data_cache_dir"], str)
        self.assertIsInstance(metadata["output_dir"], str)

    def test_manifest_and_sequence_index_never_cross_record_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            split_dir = self._write_cache(Path(temp_dir))
            spans = load_record_spans(split_dir / "manifest.json")

        self.assertEqual(
            spans,
            (
                RecordSpan("record_a", 0, 5),
                RecordSpan("record_b", 5, 8),
            ),
        )
        positions = build_sequence_positions(spans, sequence_length=2)
        self.assertEqual(
            [(p.record_id, p.start, p.stop) for p in positions],
            [
                ("record_a", 0, 2),
                ("record_a", 2, 4),
                ("record_b", 5, 7),
            ],
        )

    def test_single_epoch_branches_return_model_ready_shapes_and_dtypes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_cache(root)

            raw_dataset = SleepEDFSequenceDataset(root, "train", 1, "raw")
            wave_dataset = SleepEDFSequenceDataset(root, "train", 1, "wave")

            raw_inputs, raw_target = raw_dataset[0]
            wave_inputs, wave_target = wave_dataset[0]

            # Windows 不允许删除仍被 memmap 打开的文件；Dataset 契约要求显式释放。
            raw_dataset.close()
            wave_dataset.close()

        self.assertEqual(tuple(raw_inputs[0].shape), (1, 4))
        self.assertEqual(tuple(wave_inputs[0].shape), (1, 2, 3))
        self.assertEqual(str(raw_inputs[0].dtype), "torch.float32")
        self.assertEqual(str(wave_inputs[0].dtype), "torch.float32")
        self.assertEqual(str(raw_target.dtype), "torch.int64")
        self.assertEqual(int(raw_target), int(wave_target))

    def test_sequence_dataset_drops_each_record_tail_without_joining_nights(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_cache(root)
            dataset = SleepEDFSequenceDataset(root, "train", 2, "both")

            positions = [dataset.position_at(i) for i in range(len(dataset))]
            inputs, targets = dataset[2]
            dataset_length = len(dataset)
            dataset.close()

        self.assertEqual(dataset_length, 3)
        self.assertEqual(
            [(p.record_id, p.start, p.stop) for p in positions],
            [
                ("record_a", 0, 2),
                ("record_a", 2, 4),
                ("record_b", 5, 7),
            ],
        )
        self.assertEqual(tuple(inputs[0].shape), (2, 1, 4))
        self.assertEqual(tuple(inputs[1].shape), (2, 1, 2, 3))
        self.assertEqual(targets.tolist(), [0, 1])

    def test_loader_shuffle_order_is_reproducible_for_the_same_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_cache(root)
            dataset = SleepEDFSequenceDataset(root, "train", 1, "raw")

            first = build_reproducible_loader(dataset, 3, True, seed=23)
            second = build_reproducible_loader(dataset, 3, True, seed=23)

            def collect_first_samples(loader) -> list[float]:
                order: list[float] = []
                for inputs, _targets in loader:
                    raw_batch = inputs[0]
                    order.extend(raw_batch[:, 0, 0].tolist())
                return order

            first_order = collect_first_samples(first)
            second_order = collect_first_samples(second)
            dataset.close()

        self.assertEqual(first_order, second_order)
        self.assertCountEqual(first_order, [float(i * 4) for i in range(8)])


if __name__ == "__main__":
    unittest.main()
