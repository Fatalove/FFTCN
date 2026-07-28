from __future__ import annotations

import unittest

import numpy as np
import torch

from data.sleep_edf_cwt import morlet_cwt_epoch
from data.wavelet_torch import cwt as TorchCWT


class SleepEDFCWTTest(unittest.TestCase):
    def test_shape_dtype_range_and_finite_values(self) -> None:
        rng = np.random.default_rng(123)
        examples = [
            np.zeros(3000, dtype=np.float32),
            np.ones(3000, dtype=np.float32) * 3.5,
            rng.normal(size=3000).astype(np.float32),
        ]

        for epoch in examples:
            with self.subTest(kind=float(epoch[0])):
                spectrum = morlet_cwt_epoch(epoch)
                self.assertEqual(spectrum.shape, (1, 30, 60))
                self.assertEqual(spectrum.dtype, np.float32)
                self.assertTrue(np.isfinite(spectrum).all())
                self.assertGreaterEqual(float(spectrum.min()), 0.0)
                self.assertLessEqual(float(spectrum.max()), 1.0)

    def test_matches_repository_torch_cwt_for_random_epoch(self) -> None:
        rng = np.random.default_rng(20260709)
        epoch = rng.normal(size=3000).astype(np.float32)

        expected = TorchCWT(1 / 100, 3000, device=torch.device("cpu"))(epoch, 60).numpy()
        actual = morlet_cwt_epoch(epoch)

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_accepts_single_channel_epoch_shape(self) -> None:
        rng = np.random.default_rng(42)
        epoch = rng.normal(size=(1, 3000)).astype(np.float32)

        from_flat = morlet_cwt_epoch(epoch.reshape(3000))
        from_channel = morlet_cwt_epoch(epoch)

        np.testing.assert_array_equal(from_channel, from_flat)

    def test_is_deterministic(self) -> None:
        rng = np.random.default_rng(7)
        epoch = rng.normal(size=3000).astype(np.float32)

        first = morlet_cwt_epoch(epoch)
        second = morlet_cwt_epoch(epoch)

        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
