from __future__ import annotations

import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from training.sleep_edf_full_run import FullTrainingConfig
from training.sleep_edf_two_stage import offset_resample_record


MODULE_NAME = os.environ.get(
    "FFTCN_MILESTONE_09B_STEP3_MODULE",
    "training.sleep_edf_full_run",
)
orchestration = importlib.import_module(MODULE_NAME)


class ScriptedValidationModel(nn.Module):
    """让三轮 validation loss 按“好、差、中等”变化的微型模型。"""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("completed_epochs", torch.tensor(0, dtype=torch.long))
        self.validation_scores = (3.0, 1.0, 2.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = len(inputs)
        if self.training:
            self.completed_epochs.add_(1)
            score = self.dummy
        else:
            index = int(self.completed_epochs) - 1
            score = self.dummy * 0.0 + self.validation_scores[index]
        zero = self.dummy * 0.0
        return torch.stack(
            (score.expand(batch_size), zero.expand(batch_size)), dim=1
        )


class TinyBranch(nn.Module):
    """提供与真实特征分支相同 pretrain/finetune 接口的微型网络。"""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 2)
        self.classifier_enabled = True

    def pretrain(self) -> None:
        self.classifier_enabled = True

    def finetune(self) -> None:
        self.classifier_enabled = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs.reshape(len(inputs), -1)[:, :1])


class TinyFusion(nn.Module):
    """包含两个可迁移分支和一个新分类头的微型融合模型。"""

    def __init__(self) -> None:
        super().__init__()
        self.raw_feature_net = TinyBranch()
        self.wave_feature_net = TinyBranch()
        self.classifier = nn.Linear(4, 2)

    def forward(
        self, raw_inputs: torch.Tensor, wave_inputs: torch.Tensor
    ) -> torch.Tensor:
        features = torch.cat(
            (
                self.raw_feature_net(raw_inputs),
                self.wave_feature_net(wave_inputs),
            ),
            dim=1,
        )
        return self.classifier(features)


class SleepEDFFullTrainingOrchestrationTest(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(20260721)
        torch.manual_seed(20260721)

    def _write_base_train_cache(self, root: Path) -> tuple[np.ndarray, np.ndarray]:
        train_dir = root / "train"
        train_dir.mkdir(parents=True)
        raw_a = np.arange(5 * 8, dtype=np.float32).reshape(5, 1, 8)
        raw_b = (100 + np.arange(4 * 8, dtype=np.float32)).reshape(4, 1, 8)
        labels_a = np.array([0, 0, 0, 1, 1], dtype=np.int64)
        labels_b = np.array([1, 1, 2, 2], dtype=np.int64)
        raw = np.concatenate((raw_a, raw_b))
        labels = np.concatenate((labels_a, labels_b))
        np.save(train_dir / "raw.npy", raw)
        np.save(train_dir / "labels.npy", labels)
        manifest = {
            "split": "train",
            "records": [
                {"record_id": "record_a", "start": 0, "stop": 5},
                {"record_id": "record_b", "start": 5, "stop": 9},
            ],
        }
        (train_dir / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return raw, labels

    def test_balanced_cache_resamples_each_record_before_building_wave(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw, labels = self._write_base_train_cache(root)

            def fake_wave(raw_epoch: np.ndarray) -> np.ndarray:
                marker = float(raw_epoch[0, 0])
                return np.full((1, 30, 60), marker, dtype=np.float32)

            manifest_path = orchestration.build_balanced_pretrain_cache(
                root,
                seed=17,
                offset_samples=2,
                wave_transform=fake_wave,
            )
            actual_raw = np.load(root / "pretrain_train" / "raw.npy")
            actual_wave = np.load(root / "pretrain_train" / "wave.npy")
            actual_labels = np.load(root / "pretrain_train" / "labels.npy")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        expected_a = offset_resample_record(raw[:5], labels[:5], offset=2, seed=17)
        expected_b = offset_resample_record(raw[5:], labels[5:], offset=2, seed=18)
        expected_raw = np.concatenate((expected_a[0], expected_b[0]))
        expected_labels = np.concatenate((expected_a[1], expected_b[1]))
        np.testing.assert_array_equal(actual_raw, expected_raw)
        np.testing.assert_array_equal(actual_labels, expected_labels)
        np.testing.assert_array_equal(actual_wave[:, 0, 0, 0], actual_raw[:, 0, 0])
        self.assertEqual(manifest["records"][0]["stop"], len(expected_a[1]))
        self.assertEqual(manifest["records"][1]["start"], len(expected_a[1]))
        self.assertEqual(manifest["sample_count"], len(expected_labels))

    def test_epoch_loss_is_weighted_by_prediction_positions(self) -> None:
        model = nn.Identity()
        first_logits = torch.tensor([[2.0, 0.0]])
        first_targets = torch.tensor([0])
        second_logits = torch.tensor(
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        )
        second_targets = torch.tensor([0, 1, 0])
        batches = [
            ((first_logits,), first_targets),
            ((second_logits,), second_targets),
        ]

        actual = orchestration.run_epoch(model, batches, device="cpu")
        expected = F.cross_entropy(
            torch.cat((first_logits, second_logits)),
            torch.cat((first_targets, second_targets)),
        )
        self.assertAlmostEqual(actual, float(expected), places=6)

    def test_fit_stage_uses_validation_best_and_resumes_scheduler_history(self) -> None:
        model = ScriptedValidationModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
        batch = ((torch.ones(2, 1),), torch.zeros(2, dtype=torch.long))

        with tempfile.TemporaryDirectory() as temp_dir:
            first = orchestration.fit_stage(
                stage="scripted",
                model=model,
                train_loader=[batch],
                validation_loader=[batch],
                optimizer=optimizer,
                scheduler=scheduler,
                epochs=2,
                device="cpu",
                output_dir=temp_dir,
                config={"seed": 0},
            )

            resumed_model = ScriptedValidationModel()
            resumed_optimizer = torch.optim.SGD(resumed_model.parameters(), lr=9.0)
            resumed_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                resumed_optimizer, gamma=0.5
            )
            resumed = orchestration.fit_stage(
                stage="scripted",
                model=resumed_model,
                train_loader=[batch],
                validation_loader=[batch],
                optimizer=resumed_optimizer,
                scheduler=resumed_scheduler,
                epochs=3,
                device="cpu",
                output_dir=temp_dir,
                config={"seed": 0},
                resume_from=first.last_checkpoint,
            )
            best = torch.load(resumed.best_checkpoint, map_location="cpu")
            last = torch.load(resumed.last_checkpoint, map_location="cpu")
            history = json.loads(resumed.history_path.read_text(encoding="utf-8"))

        self.assertEqual(best["epoch"], 0)
        self.assertEqual(last["epoch"], 2)
        self.assertEqual(len(history), 3)
        self.assertEqual(
            [row["learning_rates"] for row in history],
            [[0.1], [0.05], [0.025]],
        )
        self.assertEqual(int(resumed_model.completed_epochs), 3)

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA 检查跨设备恢复")
    def test_fit_stage_restores_adam_state_on_training_device(self) -> None:
        """CPU 断点恢复到 CUDA 后，Adam 动量必须与模型参数位于同一设备。"""

        batch = ((torch.ones(2, 1),), torch.zeros(2, dtype=torch.long))
        model = nn.Linear(1, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

        with tempfile.TemporaryDirectory() as temp_dir:
            first = orchestration.fit_stage(
                stage="adam_resume",
                model=model,
                train_loader=[batch],
                validation_loader=[batch],
                optimizer=optimizer,
                scheduler=scheduler,
                epochs=1,
                device="cpu",
                output_dir=temp_dir,
                config={"seed": 0},
            )

            resumed_model = nn.Linear(1, 2)
            resumed_optimizer = torch.optim.Adam(
                resumed_model.parameters(), lr=9.0
            )
            resumed_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                resumed_optimizer, gamma=0.5
            )
            resumed = orchestration.fit_stage(
                stage="adam_resume",
                model=resumed_model,
                train_loader=[batch],
                validation_loader=[batch],
                optimizer=resumed_optimizer,
                scheduler=resumed_scheduler,
                epochs=2,
                device="cuda:0",
                output_dir=temp_dir,
                config={"seed": 0},
                resume_from=first.last_checkpoint,
            )

        self.assertEqual([row["epoch"] for row in resumed.history], [0, 1])
        self.assertEqual(next(resumed_model.parameters()).device.type, "cuda")
        for state in resumed_optimizer.state.values():
            self.assertEqual(state["exp_avg"].device.type, "cuda")
            self.assertEqual(state["exp_avg_sq"].device.type, "cuda")

    def test_fit_stage_reports_train_and_validation_progress_by_batch(self) -> None:
        """阶段标题只显示一次，进度行从 epoch 开始并包装真实 batch。"""

        class RecordingProgress:
            def __init__(self, batches, **options) -> None:
                self.batches = batches
                self.options = options
                self.metric_updates: list[tuple[str, str, bool]] = []

            def __enter__(self):
                return self

            def __exit__(self, *exc_info) -> None:
                return None

            def __iter__(self):
                return iter(self.batches)

            def set_postfix(
                self,
                *,
                loss: str,
                acc: str,
                refresh: bool = True,
            ) -> None:
                self.metric_updates.append((loss, acc, refresh))

        progress_bars: list[RecordingProgress] = []

        def record_progress(batches, **options):
            progress = RecordingProgress(batches, **options)
            progress_bars.append(progress)
            return progress

        model = nn.Linear(1, 2)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.copy_(torch.tensor([1.0, 0.0]))
        # 学习率为 0 使预测始终为类别 0：两个训练 batch 的累计 ACC 应由
        # 第一批 100% 变为 50%，从而区别于第二批自身的 0%。
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
        train_batches = [
            ((torch.ones(2, 1),), torch.zeros(2, dtype=torch.long)),
            ((torch.zeros(2, 1),), torch.ones(2, dtype=torch.long)),
        ]
        validation_batches = [train_batches[0]]

        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(
                orchestration,
                "tqdm",
                side_effect=record_progress,
            ) as progress_factory:
                result = orchestration.fit_stage(
                    stage="raw_pretrain",
                    model=model,
                    train_loader=train_batches,
                    validation_loader=validation_batches,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epochs=1,
                    device="cpu",
                    output_dir=temp_dir,
                    config={"seed": 0},
                )

        progress_factory.write.assert_called_once_with(
            "\n==================== RAW PRETRAIN ===================="
        )
        self.assertEqual(len(progress_bars), 2)
        self.assertEqual(
            [progress.options["desc"] for progress in progress_bars],
            [
                "Epoch 1/1 | Train",
                "Epoch 1/1 | Valid",
            ],
        )
        self.assertEqual(
            [progress.options["unit"] for progress in progress_bars],
            ["batch", "batch"],
        )
        self.assertEqual(
            [progress.options["ncols"] for progress in progress_bars],
            [100, 100],
        )
        self.assertEqual(
            [len(progress.batches) for progress in progress_bars],
            [2, 1],
        )
        # train 的两个 batch、validation 的一个 batch 都必须在迭代器仍打开时
        # 实时更新；不允许等 run_epoch 返回、tqdm 已关闭后再调用无效的 postfix。
        self.assertEqual(
            [len(progress.metric_updates) for progress in progress_bars],
            [2, 1],
        )
        self.assertTrue(
            all(
                not refresh
                for progress in progress_bars
                for _, _, refresh in progress.metric_updates
            )
        )
        self.assertEqual(
            progress_bars[0].metric_updates[-1][0],
            f"{result.history[0]['train_loss']:.4f}",
        )
        self.assertEqual(
            progress_bars[1].metric_updates[-1][0],
            f"{result.history[0]['validation_loss']:.4f}",
        )
        self.assertEqual(
            [progress.metric_updates[-1][1] for progress in progress_bars],
            ["50.00%", "100.00%"],
        )

    def test_two_stage_training_transfers_validation_best_before_fusion(self) -> None:
        raw_model = TinyBranch()
        wave_model = TinyBranch()
        fusion_model = TinyFusion()
        branch_batch = ((torch.tensor([[-1.0], [1.0]]),), torch.tensor([0, 1]))
        fusion_batch = (
            (torch.tensor([[-1.0], [1.0]]), torch.tensor([[1.0], [-1.0]])),
            torch.tensor([0, 1]),
        )
        loaders = {
            "raw_pretrain_train": [branch_batch],
            "raw_pretrain_validation": [branch_batch],
            "wave_pretrain_train": [branch_batch],
            "wave_pretrain_validation": [branch_batch],
            "fusion_finetune_train": [fusion_batch],
            "fusion_finetune_validation": [fusion_batch],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config = FullTrainingConfig(
                data_cache_dir=Path(temp_dir) / "cache",
                output_dir=Path(temp_dir) / "outputs",
                raw_pretrain_epochs=1,
                wave_pretrain_epochs=1,
                fusion_finetune_epochs=1,
                pretrain_learning_rate=1e-2,
                # 本测试只观察迁移起点；融合学习率设为 0，避免随后的一轮微调
                # 改变分支参数而掩盖“迁移的是 validation best”这一行为。
                fusion_learning_rate=0.0,
                device="cpu",
            )
            results = orchestration.run_two_stage_training(
                config,
                loaders,
                raw_model=raw_model,
                wave_model=wave_model,
                fusion_model=fusion_model,
            )

        self.assertEqual(
            set(results), {"raw_pretrain", "wave_pretrain", "fusion_finetune"}
        )
        for expected, actual in zip(
            raw_model.state_dict().values(),
            fusion_model.raw_feature_net.state_dict().values(),
        ):
            self.assertTrue(torch.equal(expected, actual))
        for expected, actual in zip(
            wave_model.state_dict().values(),
            fusion_model.wave_feature_net.state_dict().values(),
        ):
            self.assertTrue(torch.equal(expected, actual))
        self.assertFalse(fusion_model.raw_feature_net.classifier_enabled)
        self.assertFalse(fusion_model.wave_feature_net.classifier_enabled)


if __name__ == "__main__":
    unittest.main()
