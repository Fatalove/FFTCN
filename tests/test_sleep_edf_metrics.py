from __future__ import annotations

import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn


MODULE_NAME = os.environ.get(
    "FFTCN_MILESTONE9_MODULE", "evaluation.sleep_edf_metrics"
)
metrics_module = importlib.import_module(MODULE_NAME)


class SleepEDFMetricsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.y_true = np.array([0, 0, 0, 1, 1, 2], dtype=np.int64)
        self.y_pred = np.array([0, 0, 1, 1, 0, 0], dtype=np.int64)

    def test_prepare_labels_flattens_pairs_and_rejects_invalid_inputs(self) -> None:
        actual_true, actual_pred = metrics_module._prepare_labels(
            [[0, 1], [2, 0]], [[0, 2], [2, 1]], n_classes=3
        )
        np.testing.assert_array_equal(actual_true, [0, 1, 2, 0])
        np.testing.assert_array_equal(actual_pred, [0, 2, 2, 1])
        self.assertEqual(actual_true.dtype, np.int64)
        self.assertEqual(actual_pred.dtype, np.int64)

        with self.assertRaises(ValueError):
            metrics_module._prepare_labels([0, 1], [0], n_classes=3)
        with self.assertRaises(ValueError):
            metrics_module._prepare_labels([], [], n_classes=3)
        with self.assertRaises(ValueError):
            metrics_module._prepare_labels([0, 3], [0, 1], n_classes=3)

    def test_confusion_matrix_uses_true_rows_and_predicted_columns(self) -> None:
        actual = metrics_module.build_confusion_matrix(
            self.y_true, self.y_pred, n_classes=3
        )
        expected = np.array(
            [
                [2, 1, 0],
                [1, 1, 0],
                [1, 0, 0],
            ],
            dtype=np.int64,
        )
        np.testing.assert_array_equal(actual, expected)

    def test_hand_computed_metrics_and_zero_division_policy(self) -> None:
        metrics = metrics_module.compute_classification_metrics(
            self.y_true, self.y_pred, n_classes=3
        )

        np.testing.assert_array_equal(metrics.support, [3, 2, 1])
        np.testing.assert_allclose(metrics.precision, [0.5, 0.5, 0.0])
        np.testing.assert_allclose(metrics.recall, [2 / 3, 0.5, 0.0])
        np.testing.assert_allclose(metrics.f1, [4 / 7, 0.5, 0.0])
        self.assertAlmostEqual(metrics.accuracy, 0.5)
        self.assertAlmostEqual(metrics.macro_f1, 5 / 14)
        self.assertAlmostEqual(metrics.kappa, 0.1)
        self.assertTrue(np.isfinite(metrics.precision).all())
        self.assertTrue(np.isfinite(metrics.f1).all())

    def test_accuracy_can_hide_minority_class_failure(self) -> None:
        y_true = np.array([0] * 9 + [1])
        y_pred = np.zeros(10, dtype=np.int64)
        metrics = metrics_module.compute_classification_metrics(
            y_true, y_pred, n_classes=2
        )

        self.assertAlmostEqual(metrics.accuracy, 0.9)
        self.assertAlmostEqual(metrics.f1[0], 18 / 19)
        self.assertEqual(metrics.f1[1], 0.0)
        self.assertLess(metrics.macro_f1, 0.5)

    def test_kappa_is_explicitly_undefined_without_class_variation(self) -> None:
        metrics = metrics_module.compute_classification_metrics(
            [0, 0, 0], [0, 0, 0], n_classes=2
        )
        self.assertEqual(metrics.accuracy, 1.0)
        self.assertIsNone(metrics.kappa)

        # kappa=None 不应只停留在指标对象中；报告层必须保留 None，
        # 这样 json.dump 才会按既定契约将其写成 null，而不是转换时报错。
        metadata = {
            "dataset_name": "Sleep-EDF-153",
            "split_name": "test",
            "split_strategy": "subject-disjoint fixed 62/8/8",
            "random_seed": 42,
            "checkpoint_path": "checkpoints/best.pt",
            "python_version": "3.10.13",
            "torch_version": "1.12.1",
            "device": "cpu",
        }
        report = metrics_module.build_evaluation_report(
            metrics, metadata, label_names=("W", "N1")
        )
        self.assertIsNone(report["overall"]["cohen_kappa"])

    def test_evaluate_model_flattens_sequence_positions_without_updates(self) -> None:
        model = nn.Identity()
        model.train()
        logits = torch.tensor(
            [
                [[4.0, 1.0, 0.0], [0.0, 3.0, 1.0]],
                [[0.0, 1.0, 5.0], [2.0, 1.0, 0.0]],
            ]
        )
        targets = torch.tensor([[0, 1], [2, 1]])

        metrics = metrics_module.evaluate_model(
            model, [((logits,), targets)], n_classes=3
        )

        self.assertFalse(model.training)
        self.assertAlmostEqual(metrics.accuracy, 0.75)
        np.testing.assert_array_equal(metrics.support, [1, 2, 1])

    def test_report_round_trip_preserves_metadata_and_claim_boundary(self) -> None:
        metrics = metrics_module.compute_classification_metrics(
            self.y_true, self.y_pred, n_classes=3
        )
        metadata = {
            "dataset_name": "Sleep-EDF-153",
            "split_name": "test",
            "split_strategy": "subject-disjoint fixed 62/8/8",
            "random_seed": 42,
            "checkpoint_path": "checkpoints/best.pt",
            "python_version": "3.10.13",
            "torch_version": "1.12.1",
            "device": "cuda:0",
        }
        report = metrics_module.build_evaluation_report(
            metrics, metadata, label_names=("W", "N1", "N2")
        )

        self.assertEqual(report["metadata"], metadata)
        self.assertEqual(report["claim_boundary"], metrics_module.CLAIM_BOUNDARY)
        self.assertEqual([row["label"] for row in report["per_class"]], ["W", "N1", "N2"])

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nested" / "report.json"
            metrics_module.save_evaluation_report(path, report)
            loaded = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(loaded, report)


if __name__ == "__main__":
    unittest.main()
