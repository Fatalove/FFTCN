from __future__ import annotations

import unittest

import torch
from torch import nn

from models.wavelet.sleep_edf_2d_cnn import Conv2dBlock, SleepEDFWaveFeatureNet


class SleepEDF2DCNNTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260713)

    def test_conv_block_preserves_spatial_shape_and_module_order(self) -> None:
        block = Conv2dBlock(1, 32, kernel_size=3, stride=1, padding=1)

        # 只验证真正影响算法的模块类型与顺序，不限制用户给属性取什么名字。
        child_modules = list(block.children())
        self.assertEqual(len(child_modules), 3)
        self.assertIsInstance(child_modules[0], nn.Conv2d)
        self.assertIsInstance(child_modules[1], nn.BatchNorm2d)
        self.assertIsInstance(child_modules[2], nn.ReLU)

        x = torch.randn(2, 1, 30, 60)
        actual = block(x)
        self.assertEqual(tuple(actual.shape), (2, 32, 30, 60))

    def test_feature_mode_returns_216_values_with_correct_axis_geometry(self) -> None:
        model = SleepEDFWaveFeatureNet()
        model.finetune()
        model.eval()

        pools = [module for module in model.modules() if isinstance(module, nn.MaxPool2d)]
        self.assertEqual(len(pools), 4)

        final_pool_shapes: list[tuple[int, ...]] = []
        handle = pools[-1].register_forward_hook(
            lambda _module, _inputs, output: final_pool_shapes.append(tuple(output.shape))
        )
        try:
            with torch.no_grad():
                features = model(torch.randn(3, 1, 30, 60))
        finally:
            handle.remove()

        # 正确轴顺序应得到 [B,72,1,3]；若把 30 和 60 交换，会变成 [B,72,3,1]。
        self.assertEqual(final_pool_shapes, [(3, 72, 1, 3)])
        self.assertEqual(tuple(features.shape), (3, 216))
        self.assertTrue(torch.isfinite(features).all())

    def test_pretrain_mode_returns_five_logits(self) -> None:
        model = SleepEDFWaveFeatureNet(n_classes=5)
        model.pretrain()
        model.eval()

        with torch.no_grad():
            logits = model(torch.randn(3, 1, 30, 60))

        self.assertEqual(tuple(logits.shape), (3, 5))
        self.assertTrue(torch.isfinite(logits).all())

    def test_output_mode_is_independent_from_train_eval_mode(self) -> None:
        model = SleepEDFWaveFeatureNet()
        model.eval()
        x = torch.randn(2, 1, 30, 60)

        model.finetune()
        with torch.no_grad():
            features = model(x)
        self.assertEqual(tuple(features.shape), (2, 216))
        self.assertFalse(model.training)

        model.pretrain()
        with torch.no_grad():
            logits = model(x)
        self.assertEqual(tuple(logits.shape), (2, 5))
        self.assertFalse(model.training)

    def test_core_convolution_receives_finite_gradient(self) -> None:
        model = SleepEDFWaveFeatureNet()
        model.pretrain()
        model.train()

        x = torch.randn(4, 1, 30, 60)
        targets = torch.tensor([0, 1, 2, 3])
        loss = nn.functional.cross_entropy(model(x), targets)
        loss.backward()

        first_convolution = next(
            module for module in model.modules() if isinstance(module, nn.Conv2d)
        )
        gradient = first_convolution.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
