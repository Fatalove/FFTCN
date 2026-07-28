from __future__ import annotations

import importlib
import inspect
import json
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch import nn

from evaluation.sleep_edf_metrics import ClassificationMetrics
from training import sleep_edf_full_run as full_run
from training.sleep_edf_full_run import FullTrainingConfig, StageTrainingResult


# 默认检查用户正式练习；Codex 准备材料时可让同一行为契约运行独立参考。
MODULE_NAME = os.environ.get(
    "FFTCN_MILESTONE_09B_STEP4_MODULE",
    "training.sleep_edf_formal_experiment",
)
formal = importlib.import_module(MODULE_NAME)


def _metrics() -> ClassificationMetrics:
    """返回一个可直接用于报告行为测试的五类指标。"""

    support = np.array([3000, 2000, 7000, 3000, 2800], dtype=np.int64)
    return ClassificationMetrics(
        confusion_matrix=np.diag(support),
        support=support,
        precision=np.ones(5, dtype=np.float64),
        recall=np.ones(5, dtype=np.float64),
        f1=np.ones(5, dtype=np.float64),
        accuracy=1.0,
        macro_f1=1.0,
        kappa=1.0,
    )


def _stage_result(root: Path, stage: str) -> StageTrainingResult:
    """构造带一轮历史的微型阶段结果。"""

    stage_dir = root / stage
    return StageTrainingResult(
        stage=stage,
        best_checkpoint=stage_dir / "best.pt",
        last_checkpoint=stage_dir / "last.pt",
        history_path=stage_dir / "history.json",
        history=(
            {
                "epoch": 0,
                "learning_rates": [1e-5],
                "train_loss": 1.0,
                "validation_loss": 0.9,
            },
        ),
    )


class SleepEDFFormalExperimentTest(unittest.TestCase):
    def test_seed_experiment_repeats_python_numpy_and_torch_draws(self) -> None:
        formal.seed_experiment(17)
        first = (random.random(), float(np.random.rand()), torch.rand(3))

        formal.seed_experiment(17)
        second = (random.random(), float(np.random.rand()), torch.rand(3))

        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        self.assertTrue(torch.equal(first[2], second[2]))

    def test_build_all_caches_preserves_split_and_balanced_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            processed_root = root / "processed"
            cache_root = root / "cache"

            for split_index, split in enumerate(("train", "validation", "test")):
                split_dir = processed_root / split
                split_dir.mkdir(parents=True)
                raw = (
                    split_index * 10000
                    + np.arange(5 * 3000, dtype=np.float32)
                ).reshape(5, 1, 3000)
                labels = np.array([0, 0, 0, 1, 1], dtype=np.int64)
                np.savez(split_dir / f"{split}_record.npz", x=raw, y=labels)

            def fake_wave(raw_epoch: np.ndarray) -> np.ndarray:
                return np.full(
                    (1, 30, 60),
                    float(raw_epoch[0, 0]),
                    dtype=np.float32,
                )

            config = FullTrainingConfig(
                data_cache_dir=cache_root,
                output_dir=root / "outputs",
                seed=17,
                offset_samples=2,
                device="cpu",
            )
            result = formal.build_all_caches(
                processed_root,
                config,
                wave_transform=fake_wave,
            )

            for split in ("train", "validation", "test"):
                raw = np.load(cache_root / split / "raw.npy")
                wave = np.load(cache_root / split / "wave.npy")
                labels = np.load(cache_root / split / "labels.npy")
                self.assertEqual(len(raw), len(wave))
                self.assertEqual(len(raw), len(labels))
                # wave 按固定契约以 float16 落盘，因此只检查索引标记
                # 在 float16 量化精度内对齐，不强求与 float32 raw 逐位相等。
                np.testing.assert_allclose(
                    wave[:, 0, 0, 0], raw[:, 0, 0], rtol=1e-3, atol=8.0
                )
                self.assertEqual(result.manifest_paths[split].name, "manifest.json")

            balanced_raw = np.load(cache_root / "pretrain_train" / "raw.npy")
            balanced_wave = np.load(cache_root / "pretrain_train" / "wave.npy")
            balanced_labels = np.load(cache_root / "pretrain_train" / "labels.npy")
            self.assertEqual(len(balanced_raw), len(balanced_wave))
            self.assertEqual(len(balanced_raw), len(balanced_labels))
            np.testing.assert_allclose(
                balanced_wave[:, 0, 0, 0],
                balanced_raw[:, 0, 0],
                rtol=1e-3,
                atol=8.0,
            )
            self.assertEqual(
                result.balanced_manifest_path,
                cache_root / "pretrain_train" / "manifest.json",
            )
            self.assertGreaterEqual(result.elapsed_seconds, 0.0)

    def test_overfit_single_batch_reuses_data_and_reduces_loss(self) -> None:
        torch.manual_seed(7)
        model = nn.Linear(1, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
        batch = (
            (torch.tensor([[-1.0], [1.0]]),),
            torch.tensor([0, 1], dtype=torch.long),
        )

        result = formal.overfit_single_batch(
            model,
            batch,
            optimizer,
            device="cpu",
            steps=20,
        )

        self.assertEqual(len(result.losses), 20)
        self.assertLess(result.losses[-1], result.losses[0])
        self.assertIsNone(result.peak_cuda_memory_mib)

        # CUDA 峰值必须在第一步训练前重置，并在全部更新完成后读取。
        cuda_events: list[str] = []
        with (
            mock.patch.object(
                formal,
                "run_epoch",
                side_effect=lambda *args, **kwargs: cuda_events.append("run") or 1.0,
            ),
            mock.patch.object(
                formal.torch.cuda,
                "reset_peak_memory_stats",
                side_effect=lambda *args, **kwargs: cuda_events.append("reset"),
            ),
            mock.patch.object(
                formal.torch.cuda,
                "max_memory_allocated",
                side_effect=lambda *args, **kwargs: (
                    cuda_events.append("max") or 2 * 1024**2
                ),
            ),
        ):
            cuda_result = formal.overfit_single_batch(
                model,
                batch,
                optimizer,
                device="cuda",
                steps=2,
            )

        self.assertEqual(cuda_events, ["reset", "run", "run", "max"])
        self.assertEqual(cuda_result.peak_cuda_memory_mib, 2.0)

    def test_full_training_forwards_resume_mapping_and_closes_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = FullTrainingConfig(
                data_cache_dir=root / "cache",
                output_dir=root / "outputs",
                device="cpu",
            )
            resume = {"fusion_finetune": root / "fusion" / "last.pt"}
            dataset = mock.Mock()
            expected = {"result": "sentinel"}

            with (
                mock.patch.object(
                    full_run,
                    "SleepEDFSequenceDataset",
                    return_value=dataset,
                ),
                mock.patch.object(
                    full_run,
                    "build_reproducible_loader",
                    return_value=object(),
                ),
                mock.patch.object(
                    full_run,
                    "run_two_stage_training",
                    return_value=expected,
                ) as two_stage,
            ):
                actual = full_run.run_full_training(
                    config,
                    resume_checkpoints=resume,
                )

        self.assertIs(actual, expected)
        self.assertEqual(
            two_stage.call_args.kwargs["resume_checkpoints"],
            resume,
        )
        self.assertEqual(dataset.close.call_count, 6)

    def test_evaluate_fusion_best_loads_selected_checkpoint_and_closes_test(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = FullTrainingConfig(
                data_cache_dir=root / "cache",
                output_dir=root / "outputs",
                sequence_length=50,
                fusion_batch_size=32,
                device="cpu",
            )
            checkpoint_path = root / "fusion_finetune" / "best.pt"
            model = mock.Mock()
            dataset = mock.Mock()
            loader = object()
            metrics = _metrics()

            with (
                mock.patch.object(formal, "FFTCNFusionTCN", return_value=model),
                mock.patch.object(formal, "load_stage_checkpoint") as load,
                mock.patch.object(
                    formal,
                    "SleepEDFSequenceDataset",
                    return_value=dataset,
                ) as dataset_type,
                mock.patch.object(
                    formal,
                    "build_reproducible_loader",
                    return_value=loader,
                ) as build_loader,
                mock.patch.object(
                    formal,
                    "evaluate_model",
                    return_value=metrics,
                ) as evaluate,
            ):
                actual = formal.evaluate_fusion_best(config, checkpoint_path)

        load.assert_called_once_with(
            checkpoint_path,
            model,
            map_location=torch.device("cpu"),
        )
        dataset_type.assert_called_once()
        dataset_call = inspect.signature(full_run.SleepEDFSequenceDataset).bind(
            *dataset_type.call_args.args,
            **dataset_type.call_args.kwargs,
        )
        self.assertEqual(dataset_call.arguments["cache_root"], config.data_cache_dir)
        self.assertEqual(dataset_call.arguments["split"], "test")
        self.assertEqual(dataset_call.arguments["sequence_length"], 50)
        self.assertEqual(dataset_call.arguments["input_mode"], "both")
        build_loader.assert_called_once_with(
            dataset,
            batch_size=32,
            shuffle=False,
            seed=config.seed,
            num_workers=config.num_workers,
        )
        evaluate.assert_called_once_with(
            model,
            loader,
            n_classes=5,
            device=torch.device("cpu"),
        )
        dataset.close.assert_called_once_with()
        self.assertIs(actual, metrics)

    def test_formal_report_records_curves_runtime_and_selection_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = FullTrainingConfig(
                data_cache_dir=root / "cache",
                output_dir=root / "outputs",
                seed=0,
                device="cpu",
            )
            stage_results = {
                stage: _stage_result(root, stage)
                for stage in (
                    "raw_pretrain",
                    "wave_pretrain",
                    "fusion_finetune",
                )
            }
            checkpoint_path = stage_results["fusion_finetune"].best_checkpoint

            report = formal.build_formal_report(
                config=config,
                stage_results=stage_results,
                metrics=_metrics(),
                training_seconds=123.5,
                peak_cuda_memory_mib=2870.125,
                checkpoint_path=checkpoint_path,
                cache_build_seconds=3600.0,
            )
            encoded = json.dumps(report)

        self.assertIsInstance(encoded, str)
        self.assertEqual(report["metadata"]["random_seed"], 0)
        self.assertEqual(report["runtime"]["training_seconds"], 123.5)
        self.assertEqual(report["runtime"]["cache_build_seconds"], 3600.0)
        self.assertEqual(report["runtime"]["peak_cuda_memory_mib"], 2870.125)
        self.assertEqual(
            report["selection_policy"]["tested_checkpoint"],
            str(checkpoint_path.resolve()),
        )
        self.assertEqual(report["selection_policy"]["formal_test_evaluations"], 1)
        self.assertEqual(report["data_semantics"]["test_evaluated_positions"], 17800)
        self.assertIn("110 of 3836", report["data_semantics"]["known_limitation"])
        self.assertEqual(
            report["training_stages"]["fusion_finetune"]["history"][0][
                "validation_loss"
            ],
            0.9,
        )

    def test_formal_run_tests_fusion_validation_best_exactly_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = FullTrainingConfig(
                data_cache_dir=root / "cache",
                output_dir=root / "outputs",
                seed=0,
                device="cpu",
            )
            stage_results = {
                stage: _stage_result(root, stage)
                for stage in (
                    "raw_pretrain",
                    "wave_pretrain",
                    "fusion_finetune",
                )
            }
            selected = stage_results["fusion_finetune"].best_checkpoint
            resume = {"raw_pretrain": root / "raw_pretrain" / "last.pt"}
            metrics = _metrics()
            report = {"formal": True}

            with (
                mock.patch.object(
                    formal,
                    "run_full_training",
                    return_value=stage_results,
                ) as train,
                mock.patch.object(
                    formal,
                    "evaluate_fusion_best",
                    return_value=metrics,
                ) as evaluate,
                mock.patch.object(
                    formal,
                    "build_formal_report",
                    return_value=report,
                ),
                mock.patch.object(formal, "save_evaluation_report") as save,
            ):
                result = formal.run_formal_experiment(
                    config,
                    resume_checkpoints=resume,
                    cache_build_seconds=99.0,
                )

        train.assert_called_once()
        train_call = inspect.signature(full_run.run_full_training).bind(
            *train.call_args.args,
            **train.call_args.kwargs,
        )
        self.assertIs(train_call.arguments["config"], config)
        self.assertEqual(train_call.arguments["resume_checkpoints"], resume)
        evaluate.assert_called_once_with(config, selected)
        save.assert_called_once_with(
            config.output_dir / "formal_test_report.json",
            report,
        )
        self.assertIs(result.metrics, metrics)
        self.assertEqual(result.report_path, config.output_dir / "formal_test_report.json")


if __name__ == "__main__":
    unittest.main()
