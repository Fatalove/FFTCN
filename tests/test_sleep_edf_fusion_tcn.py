from __future__ import annotations

import unittest

import torch
from torch import nn

from models.merge.sleep_edf_fusion_tcn import (
    FFTCNFusionTCN,
    NonCausalTCN,
    SamePadConv1d,
    TemporalResidualBlock,
)
from models.raw.sleep_edf_1d_cnn import SleepEDFRawFeatureNet
from models.wavelet.sleep_edf_2d_cnn import SleepEDFWaveFeatureNet


class SleepEDFFusionTCNTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260714)

    def test_same_padding_preserves_length_for_all_dilations(self) -> None:
        x = torch.randn(2, 3, 11)

        for dilation in (1, 2, 4, 8):
            with self.subTest(dilation=dilation):
                conv = SamePadConv1d(
                    3,
                    4,
                    kernel_size=3,
                    stride=1,
                    dilation=dilation,
                )
                self.assertEqual(tuple(conv(x).shape), (2, 4, 11))

    def test_same_padding_is_noncausal_and_sees_the_next_position(self) -> None:
        conv = SamePadConv1d(1, 1, kernel_size=3, bias=False)
        with torch.no_grad():
            conv.weight.fill_(1.0)

        # 只有 t=3 有值。非因果 k=3 卷积在计算 t=2 时能看到未来的 t=3。
        x = torch.zeros(1, 1, 5)
        x[0, 0, 3] = 1.0
        actual = conv(x)

        self.assertEqual(float(actual[0, 0, 2]), 1.0)

    def test_residual_block_projects_channels_and_preserves_length(self) -> None:
        block = TemporalResidualBlock(
            in_channels=472,
            out_channels=128,
            kernel_size=3,
            dilation=2,
            dropout=0.0,
        )
        block.eval()

        with torch.no_grad():
            actual = block(torch.randn(2, 472, 17))

        self.assertEqual(tuple(actual.shape), (2, 128, 17))
        self.assertTrue(torch.isfinite(actual).all())

        kernel_sizes = [
            module.kernel_size
            for module in block.modules()
            if isinstance(module, nn.Conv1d)
        ]
        self.assertEqual(kernel_sizes.count((3,)), 2)
        self.assertEqual(kernel_sizes.count((1,)), 1)

    def test_tcn_uses_expected_dilations_and_preserves_sequence_length(self) -> None:
        tcn = NonCausalTCN(dropout=0.0)
        tcn.eval()

        with torch.no_grad():
            actual = tcn(torch.randn(2, 50, 472))

        self.assertEqual(tuple(actual.shape), (2, 50, 128))

        dilations = [
            module.dilation[0]
            for module in tcn.modules()
            if isinstance(module, SamePadConv1d)
        ]
        self.assertEqual(dilations, [1, 1, 2, 2, 4, 4, 8, 8])

    def test_full_fusion_shapes_are_472_to_128_to_5(self) -> None:
        model = FFTCNFusionTCN()
        model.eval()

        tcn = next(
            module for module in model.modules() if isinstance(module, NonCausalTCN)
        )
        tcn_inputs: list[tuple[int, ...]] = []
        tcn_outputs: list[tuple[int, ...]] = []
        pre_handle = tcn.register_forward_pre_hook(
            lambda _module, inputs: tcn_inputs.append(tuple(inputs[0].shape))
        )
        post_handle = tcn.register_forward_hook(
            lambda _module, _inputs, output: tcn_outputs.append(tuple(output.shape))
        )

        try:
            with torch.no_grad():
                logits = model(
                    torch.randn(2, 50, 1, 3000),
                    torch.randn(2, 50, 1, 30, 60),
                )
        finally:
            pre_handle.remove()
            post_handle.remove()

        self.assertEqual(tcn_inputs, [(2, 50, 472)])
        self.assertEqual(tcn_outputs, [(2, 50, 128)])
        self.assertEqual(tuple(logits.shape), (2, 50, 5))
        self.assertTrue(torch.isfinite(logits).all())

    def test_gradient_reaches_both_feature_branches_and_tcn(self) -> None:
        model = FFTCNFusionTCN()
        model.train()

        logits = model(
            torch.randn(2, 4, 1, 3000),
            torch.randn(2, 4, 1, 30, 60),
        )
        targets = torch.tensor([[0, 1, 2, 3], [4, 3, 2, 1]])
        loss = nn.functional.cross_entropy(logits.reshape(-1, 5), targets.reshape(-1))
        loss.backward()

        raw_branch = next(
            module for module in model.modules()
            if isinstance(module, SleepEDFRawFeatureNet)
        )
        wave_branch = next(
            module for module in model.modules()
            if isinstance(module, SleepEDFWaveFeatureNet)
        )
        tcn_conv = next(
            module for module in model.modules() if isinstance(module, SamePadConv1d)
        )

        raw_conv = next(
            module for module in raw_branch.modules() if isinstance(module, nn.Conv1d)
        )
        wave_conv = next(
            module for module in wave_branch.modules() if isinstance(module, nn.Conv2d)
        )

        for name, parameters in (
            ("raw", list(raw_conv.parameters(recurse=False))),
            ("wave", list(wave_conv.parameters(recurse=False))),
            ("tcn", list(tcn_conv.parameters(recurse=False))),
        ):
            gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
            with self.subTest(branch=name):
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
                self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0.0)


if __name__ == "__main__":
    unittest.main()
