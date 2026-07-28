from __future__ import annotations

import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from models.merge.sleep_edf_fusion_tcn import FFTCNFusionTCN
from models.raw.sleep_edf_1d_cnn import SleepEDFRawFeatureNet
from models.wavelet.sleep_edf_2d_cnn import SleepEDFWaveFeatureNet
from training.sleep_edf_two_stage import (
    LABEL_NAMES,
    build_finetune_optimizer,
    build_stage_policies,
    load_training_checkpoint,
    offset_resample_record,
    run_classification_step,
    save_training_checkpoint,
    transfer_pretrained_features,
)


class SleepEDFTwoStageTest(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(20260715)
        torch.manual_seed(20260715)

    def test_stage_policies_keep_balancing_out_of_validation_test_and_sequences(self) -> None:
        policies = build_stage_policies(sequence_length=50)

        balanced = [policy for policy in policies.values() if policy.balance]
        self.assertEqual(len(balanced), 2)
        self.assertTrue(all(policy.stage == "pretrain" for policy in balanced))
        self.assertTrue(all(policy.split == "train" for policy in balanced))
        self.assertTrue(all(policy.sequence_length == 1 for policy in balanced))
        self.assertEqual({policy.input_mode for policy in balanced}, {"raw", "wave"})

        protected = [
            policy
            for policy in policies.values()
            if policy.split in {"valid", "test"} or policy.sequence_length > 1
        ]
        self.assertTrue(protected)
        self.assertTrue(all(not policy.balance for policy in protected))

        fusion = [policy for policy in policies.values() if policy.input_mode == "both"]
        self.assertEqual({policy.split for policy in fusion}, {"train", "valid", "test"})
        self.assertTrue(all(policy.sequence_length == 50 for policy in fusion))

    def test_offset_resample_is_reproducible_and_reduces_without_equalizing(self) -> None:
        epochs = np.arange(5 * 8, dtype=np.float32).reshape(5, 1, 8)
        labels = np.array([0, 0, 0, 1, 1], dtype=np.int64)
        epochs_before = epochs.copy()
        labels_before = labels.copy()

        actual_x, actual_y = offset_resample_record(
            epochs, labels, offset=2, seed=17
        )
        repeated_x, repeated_y = offset_resample_record(
            epochs, labels, offset=2, seed=17
        )

        np.testing.assert_array_equal(epochs, epochs_before)
        np.testing.assert_array_equal(labels, labels_before)
        np.testing.assert_array_equal(actual_x, repeated_x)
        np.testing.assert_array_equal(actual_y, repeated_y)
        np.testing.assert_array_equal(actual_x[: len(epochs)], epochs)
        np.testing.assert_array_equal(actual_y[: len(labels)], labels)

        # 原始最大类有 3 个；每个已有类别都额外增加 3 个。
        # 因而最终是 6 和 5，只减小差距，并没有严格变成相同数量。
        self.assertEqual(Counter(actual_y.tolist()), Counter({0: 6, 1: 5}))

    def test_training_step_decreases_loss_and_validation_does_not_update(self) -> None:
        model = nn.Linear(2, 2)
        inputs = (
            torch.tensor(
                [[-2.0, -1.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 1.0]]
            ),
        )
        targets = torch.tensor([0, 0, 1, 1])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

        initial_loss = run_classification_step(model, inputs, targets)
        for _ in range(40):
            run_classification_step(model, inputs, targets, optimizer)
        final_loss = run_classification_step(model, inputs, targets)
        self.assertLess(final_loss, initial_loss * 0.25)

        reported_loss, correct_count = run_classification_step(
            model,
            inputs,
            targets,
            return_correct_count=True,
        )
        self.assertAlmostEqual(reported_loss, final_loss)
        self.assertEqual(correct_count, 4)

        before_validation = [parameter.detach().clone() for parameter in model.parameters()]
        run_classification_step(model, inputs, targets)
        after_validation = list(model.parameters())
        for before, after in zip(before_validation, after_validation):
            self.assertTrue(torch.equal(before, after))

    def test_pretrained_feature_weights_transfer_and_switch_to_feature_mode(self) -> None:
        raw = SleepEDFRawFeatureNet()
        wave = SleepEDFWaveFeatureNet()
        with torch.no_grad():
            for parameter in raw.parameters():
                parameter.fill_(0.125)
            for parameter in wave.parameters():
                parameter.fill_(-0.25)

        fusion = FFTCNFusionTCN()
        transfer_pretrained_features(fusion, raw, wave)

        for expected, actual in zip(
            raw.state_dict().values(), fusion.raw_feature_net.state_dict().values()
        ):
            self.assertTrue(torch.equal(expected, actual))
        for expected, actual in zip(
            wave.state_dict().values(), fusion.wave_feature_net.state_dict().values()
        ):
            self.assertTrue(torch.equal(expected, actual))

        fusion.eval()
        with torch.no_grad():
            raw_features = fusion.raw_feature_net(torch.randn(2, 1, 3000))
            wave_features = fusion.wave_feature_net(torch.randn(2, 1, 30, 60))
        self.assertEqual(tuple(raw_features.shape), (2, 256))
        self.assertEqual(tuple(wave_features.shape), (2, 216))

    def test_optimizer_uses_base_lr_for_new_modules_and_scaled_lr_for_branches(self) -> None:
        model = FFTCNFusionTCN()
        base_lr = 1e-3
        scale = 1e-2
        optimizer = build_finetune_optimizer(model, base_lr, scale)

        parameter_lr: dict[int, float] = {}
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                self.assertNotIn(id(parameter), parameter_lr)
                parameter_lr[id(parameter)] = float(group["lr"])

        self.assertEqual(parameter_lr.keys(), {id(p) for p in model.parameters()})
        for parameter in model.raw_feature_net.parameters():
            self.assertEqual(parameter_lr[id(parameter)], base_lr * scale)
        for parameter in model.wave_feature_net.parameters():
            self.assertEqual(parameter_lr[id(parameter)], base_lr * scale)

        feature_ids = {
            id(parameter)
            for parameter in list(model.raw_feature_net.parameters())
            + list(model.wave_feature_net.parameters())
        }
        for parameter in model.parameters():
            if id(parameter) not in feature_ids:
                self.assertEqual(parameter_lr[id(parameter)], base_lr)

    def test_checkpoint_round_trip_restores_state_and_metadata(self) -> None:
        model = nn.Linear(3, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        inputs = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 0, 1])
        run_classification_step(model, (inputs,), targets, optimizer)
        expected_state = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "checkpoint.pt"
            save_training_checkpoint(
                path=path,
                model=model,
                optimizer=optimizer,
                config={"seed": 42, "sequence_length": 50},
                label_names=LABEL_NAMES,
                stage="fusion_finetune",
                epoch=3,
                validation_loss=0.75,
            )

            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()

            restored_optimizer = torch.optim.Adam(model.parameters(), lr=9e-3)
            checkpoint = load_training_checkpoint(path, model, restored_optimizer)

        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, expected_state[name]))
        self.assertEqual(checkpoint["config"], {"seed": 42, "sequence_length": 50})
        self.assertEqual(checkpoint["label_names"], list(LABEL_NAMES))
        self.assertEqual(checkpoint["stage"], "fusion_finetune")
        self.assertEqual(checkpoint["epoch"], 3)
        self.assertEqual(checkpoint["validation_loss"], 0.75)
        self.assertTrue(restored_optimizer.state_dict()["state"])


if __name__ == "__main__":
    unittest.main()
