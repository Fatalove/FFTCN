from __future__ import annotations

import unittest

import torch
from torch import nn

from models.raw.sleep_edf_1d_cnn import Conv1dBlock, SleepEDFRawFeatureNet


class SleepEDF1DCNNTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260713)

    def test_conv_block_shape_and_module_order(self) -> None:
        block = Conv1dBlock(1, 128, kernel_size=50, stride=25)

        # 属性名属于实现细节；这里只检查真正影响算法的模块类型和执行顺序。
        child_modules = list(block.children())
        self.assertEqual(len(child_modules), 3)
        self.assertIsInstance(child_modules[0], nn.Conv1d)
        self.assertIsInstance(child_modules[1], nn.BatchNorm1d)
        self.assertIsInstance(child_modules[2], nn.ReLU)

        x = torch.randn(2, 1, 3000)
        actual = block(x)
        self.assertEqual(tuple(actual.shape), (2, 128, 119))

    def test_feature_mode_returns_256_values(self) -> None:
        model = SleepEDFRawFeatureNet()
        model.finetune()
        model.eval()

        x = torch.randn(3, 1, 3000)
        with torch.no_grad():
            feature_map = model.features(x)
            features = model(x)

        self.assertEqual(tuple(feature_map.shape), (3, 128, 2))
        self.assertEqual(tuple(features.shape), (3, 256))
        self.assertTrue(torch.isfinite(features).all())

    def test_pretrain_mode_returns_five_logits(self) -> None:
        model = SleepEDFRawFeatureNet(n_classes=5)
        model.pretrain()
        model.eval()

        x = torch.randn(3, 1, 3000)
        with torch.no_grad():
            logits = model(x)

        self.assertEqual(tuple(logits.shape), (3, 5))
        self.assertTrue(torch.isfinite(logits).all())

    def test_output_mode_is_independent_from_train_eval_mode(self) -> None:
        model = SleepEDFRawFeatureNet()
        model.eval()

        model.finetune()
        self.assertFalse(model.classifier_enabled)
        self.assertFalse(model.training)

        model.pretrain()
        self.assertTrue(model.classifier_enabled)
        self.assertFalse(model.training)

    def test_core_convolution_receives_finite_gradient(self) -> None:
        model = SleepEDFRawFeatureNet()
        model.pretrain()
        model.train()

        x = torch.randn(4, 1, 3000)
        targets = torch.tensor([0, 1, 2, 3])
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, targets)
        loss.backward()

        # 不依赖用户给首层卷积取什么属性名，直接找到网络中的第一个 Conv1d。
        first_convolution = next(
            module for module in model.modules() if isinstance(module, nn.Conv1d)
        )
        gradient = first_convolution.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
