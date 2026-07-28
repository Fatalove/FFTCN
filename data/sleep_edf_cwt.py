"""里程碑 4 练习：对单个 Sleep-EDF 时段计算 Morlet CWT。

这个文件有意保留为练习骨架。带完整注释的参考实现位于：

    learning_guides/milestone_04/reference_solution.py

输入输出契约：
    输入：一个包含 3000 个采样点的 EEG 时段，常见形状为 [3000] 或 [1, 3000]
    输出：float32 小波时频图，形状为 [1, 30, 60]
"""

from __future__ import annotations

import numpy as np


def morlet_cwt_epoch(
    epoch: np.ndarray,
    sampling_rate: float = 100.0,
    output_time_bins: int = 60,
) -> np.ndarray:
    """将一个 30 秒 EEG 时段转换为归一化后的 Morlet CWT 小波图。

    需要实现与 ``data.wavelet_torch.cwt`` 对齐的核心行为：

    1. 构造 30 个 Morlet 频域核；
    2. 对 EEG 时段做 FFT，并与这些频域核相乘；
    3. 做逆 FFT，得到复数形式的小波系数；
    4. 计算 ``log2(abs(wave) ** 2 + 1e-10)``；
    5. 每 50 个时间点求一次平均，把 3000 个采样点压缩成 60 列；
    6. 对整张 [30, 60] 小波图做一次 min-max 归一化；
    7. 以 float32 返回形状 [1, 30, 60]。

    如果输入导致小波图所有值相同，应返回全零图，而不是产生 NaN。
    """


    x = np.asarray(epoch, dtype=np.float32).reshape(-1)
    n = x.shape[0]
    dt = 1.0 / sampling_rate

    scale_count = 30
    dj = 0.25
    morlet_k0 = 6.0
    scales = 2 * dt * 2.0 ** (np.arange(scale_count, dtype=np.float32) * dj)
    scales = scales.reshape(scale_count, 1)

    kplus = np.arange(1, int(n / 2) + 1, dtype=np.float32)
    kplus = kplus * np.float32(2 * np.pi / (n * dt))

    kminus = np.arange(1, int((n - 1) / 2) + 1, dtype=np.float32)
    kminus = np.sort(-kminus * np.float32(2 * np.pi / (n * dt)))

    k = np.concatenate(([np.float32(0.0)],  kplus, kminus)).reshape(1, n)

    positive_freq = (k > 0.0).astype(np.float32)
    exponent = -((scales * k - morlet_k0) ** 2) / 2.0 * positive_freq

    norm = (
        np.sqrt(scales * k[0, 1]).astype(np.float32)
        * np.float32(np.pi ** -0.25)
        * np.float32(np.sqrt(n))
    )

    daughter = norm * np.exp(exponent).astype(np.float32) * positive_freq

    signal_fft = np.fft.fft(x)
    wave = np.fft.ifft(signal_fft * daughter)
    power = np.abs(wave) ** 2
    log_power = np.log2(power + 1e-10)
    compressed = log_power.reshape(scale_count, output_time_bins, -1).mean(axis=-1)

    min_value = compressed.min()
    max_value = compressed.max()

    denominator = max_value - min_value
    if denominator == 0:
        normalized = np.zeros_like(compressed, dtype=np.float32)
    else:
        normalized = ((compressed - min_value) / denominator).astype(np.float32)

    return normalized[np.newaxis, :, :]
