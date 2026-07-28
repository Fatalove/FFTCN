# 里程碑 4：Morlet CWT 核心练习

目标文件：`data/sleep_edf_cwt.py`

参考答案：`learning_guides/milestone_04/reference_solution.py`

本练习只解决一个问题：把一个 30 秒 EEG 片段从 `[3000]` 转成时频图 `[1,30,60]`。不要进入 CNN、TCN 或训练。

注意论文版和 GitHub 仓库版在这里不一致：论文文字描述为窗口长度 `m=15`，把 `3000×30` 小波图压缩为 `200×30`；但当前 GitHub 源码实际训练入口写死 `waveshape=(30, 60)`，`data.wavelet_torch.cwt.compress()` 会把最后一维压缩成 60 列，也就是每 50 个时间点求平均。本练习为了能对齐原仓库训练代码和已完成的 checkpoint，采用仓库版 `[1,30,60]`，不是论文文字版 `200×30`。

## 1. 读题与题意分析

输入：

```text
epoch: float32, [3000] 或 [1,3000]
```

输出：

```text
time_spectrum: float32, [1,30,60]
```

必须对齐原仓库 `data.wavelet_torch.cwt` 的实际训练行为：

1. 采样率 100 Hz，所以 `dt = 1 / 100`；
2. 用 30 个 Morlet 尺度；
3. 用 FFT 做频域卷积；
4. 取 `log2(abs(wave) ** 2 + 1e-10)`；
5. 把 3000 个时间点按每 50 点平均，压缩成 60 列；
6. 对整张 `[30,60]` 做一次 min-max 归一化；
7. 增加通道维度，返回 `[1,30,60]`。

和原仓库的一个小差异：练习版要求常量/全零输入不能产生 NaN。如果归一化时最大值等于最小值，直接返回全零谱图。

## 2. 必要 Python/NumPy 基础

### reshape

```python
x = np.asarray(epoch, dtype=np.float32).reshape(-1)
```

`reshape(-1)` 的意思是拉平成一维。这样 `[3000]` 和 `[1,3000]` 都会变成 `[3000]`。

### 广播

```python
scales.shape  # [30, 1]
k.shape       # [1, 3000]
scales * k    # [30, 3000]
```

这里不需要手写 30 次循环。NumPy 会自动把 30 个尺度和 3000 个频率点组合成 30 行核函数。

### FFT / IFFT

```python
signal_fft = np.fft.fft(x)
wave = np.fft.ifft(signal_fft * daughter)
```

`fft` 把时域信号变到频域。乘上 Morlet 频域核后，再用 `ifft` 回到时域，就得到每个尺度下的小波系数。

### min-max 归一化

```python
normalized = (compressed - compressed.min()) / (compressed.max() - compressed.min())
```

这里是整张图一起归一化，不是每个尺度单独归一化。

## 3. 手工推演

真实输入有 3000 点，手算太大。先看简化版：

```text
假设某个尺度的 log_power 有 12 个时间点：
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

如果要压缩成 3 格，每格包含 4 个点：
第 1 格 = mean([1,2,3,4]) = 2.5
第 2 格 = mean([5,6,7,8]) = 6.5
第 3 格 = mean([9,10,11,12]) = 10.5
```

本题真实情况是：

```text
3000 点 / 60 格 = 每格 50 点
```

所以代码是：

```python
compressed = log_power.reshape(30, 60, -1).mean(axis=-1)
```

`-1` 自动变成 50。

## 4. 带注释的完整核心代码

```python
def morlet_cwt_epoch(epoch, sampling_rate=100.0, output_time_bins=60):
    # 把输入转成 NumPy 数组，并统一为 float32，方便和原仓库 Torch 版本的精度习惯接近。
    # reshape(-1) 会把 [3000] 或 [1, 3000] 都拉平成一维 [3000]，减少形状分支。
    x = np.asarray(epoch, dtype=np.float32).reshape(-1)

    # n 是当前时段的采样点数量；Sleep-EDF 中 30 秒 * 100 Hz 通常为 3000。
    n = x.shape[0]

    # dt 是采样时间间隔。采样率为 100 Hz 时，两个相邻采样点相隔 0.01 秒。
    dt = 1.0 / sampling_rate

    # scale_count 控制小波图的尺度数量，也就是输出 [30, 60] 里的 30 行。
    scale_count = 30

    # dj 是尺度的指数步长；原仓库固定为 0.25。
    dj = 0.25

    # morlet_k0 是 Morlet 小波的中心频率参数；原仓库固定为 6.0。
    morlet_k0 = 6.0

    # 生成 30 个尺度：j = 0..29，scale_j = 2 * dt * 2 ** (j * dj)。
    # np.arange(scale_count) 生成 [0, 1, ..., 29]，每个 j 对应一个尺度。
    scales = 2 * dt * 2.0 ** (np.arange(scale_count, dtype=np.float32) * dj)

    # 把 scales 从 [30] 改成 [30, 1]，后面才能和 [1, 3000] 的频率轴广播相乘。
    scales = scales.reshape(scale_count, 1).astype(np.float32)

    # 构造 FFT 频率轴的正频率部分，索引为 1 到 n/2。
    # 这里只先生成“频率编号”，还不是实际角频率。
    kplus = np.arange(1, int(n / 2) + 1, dtype=np.float32)

    # 把正频率编号换算成角频率，单位是 rad/s，公式为 2*pi*k/(n*dt)。
    kplus = kplus * np.float32(2 * np.pi / (n * dt))

    # 构造 FFT 频率轴的负频率部分，数量与原仓库保持一致。
    kminus = np.arange(1, int((n - 1) / 2) + 1, dtype=np.float32)

    # 负频率取负号后排序，使其顺序与原仓库 Torch CWT 的频率轴一致。
    kminus = np.sort(-kminus * np.float32(2 * np.pi / (n * dt)))

    # 拼接完整频率轴：[0, 正频率..., 负频率...]，再 reshape 成 [1, n]。
    # reshape 成 [1, n] 是为了后面和 [30, 1] 的 scales 自动广播成 [30, n]。
    k = np.concatenate(([np.float32(0.0)], kplus, kminus)).reshape(1, n)

    # Morlet 频域核只保留正频率；k > 0 的位置为 1，其他位置为 0。
    # 这相当于 Heaviside 阶跃函数，用来屏蔽零频和负频率。
    positive_frequency = (k > 0.0).astype(np.float32)

    # 计算 Morlet 频域核的指数部分。
    # scales * k 会广播成 [30, 3000]，表示 30 个尺度分别作用到 3000 个频率点。
    # 减去 morlet_k0 后平方，得到 Morlet 核在频域中的高斯形状。
    exponent = -((scales * k - morlet_k0) ** 2) / 2.0 * positive_frequency

    # 计算 Morlet 核的归一化系数。
    # k[0, 1] 是第一个正频率，原仓库使用它参与尺度归一化。
    norm = (
        # sqrt(scales * k[0, 1]) 让不同尺度下的能量尺度保持一致。
        np.sqrt(scales * k[0, 1]).astype(np.float32)
        # pi ** -0.25 是 Morlet 小波公式中的常数项。
        * np.float32(np.pi ** -0.25)
        # sqrt(n) 对应 FFT 长度带来的幅值缩放。
        * np.float32(np.sqrt(n))
    )

    # daughter 就是最终的 30 个 Morlet 频域核，形状为 [30, 3000]。
    # np.exp(exponent) 生成高斯频域形状，再乘 positive_frequency 去掉非正频率。
    daughter = norm * np.exp(exponent).astype(np.float32) * positive_frequency

    # 对原始 EEG 时段做 FFT，把时域信号 x 转成频域表示。
    signal_fft = np.fft.fft(x)

    # 频域卷积：signal_fft 形状是 [3000]，daughter 形状是 [30, 3000]。
    # NumPy 会广播成 30 行，每一行表示一个尺度下的频域乘积。
    # ifft 后回到时域，得到复数小波系数 wave，形状为 [30, 3000]。
    wave = np.fft.ifft(signal_fft * daughter)

    # 小波系数是复数，abs(wave) 得到幅值，平方后得到功率。
    power = np.abs(wave) ** 2

    # 对功率取 log2，压缩数值范围；加 1e-10 是为了避免 log2(0) 得到 -inf。
    log_power = np.log2(power + 1e-10)

    # 把 log_power 从 [30, 3000] reshape 成 [30, 60, 50]。
    # 这里的 -1 会自动推导为 50，因为 3000 / 60 = 50。
    # 再对最后一维求平均，就得到 [30, 60] 的压缩小波图。
    compressed = log_power.reshape(scale_count, output_time_bins, -1).mean(axis=-1)

    # 找到整张 [30, 60] 小波图中的最小值，用于 min-max 归一化。
    min_value = compressed.min()

    # 找到整张 [30, 60] 小波图中的最大值，用于 min-max 归一化。
    max_value = compressed.max()

    # 分母是最大值和最小值的差；如果为 0，说明整张图所有值都一样。
    denominator = max_value - min_value

    # 如果整张图所有值都一样，直接返回全零图，避免除以 0 产生 NaN。
    if denominator == 0:
        normalized = np.zeros_like(compressed, dtype=np.float32)

    # 正常情况下，执行 min-max 归一化，把数值缩放到 [0, 1]。
    else:
        normalized = ((compressed - min_value) / denominator).astype(np.float32)

    # 增加最前面的通道维度，使输出从 [30, 60] 变成 CNN 需要的 [1, 30, 60]。
    return normalized[np.newaxis, :, :]
```

## 5. 聚焦测试命令

只测试里程碑 4：

```powershell
python -m unittest tests.test_sleep_edf_cwt -v
```

全部预处理 + CWT 测试：

```powershell
python -m unittest tests.test_sleep_edf_preprocessor tests.test_sleep_edf_cwt -v
```

## 6. 常见错误解释

### 输出是 `[30,60]`，不是 `[1,30,60]`

少了最后这一行：

```python
return normalized[np.newaxis, :, :]
```

### 与原仓库 Torch CWT 对不上

优先检查三点：

1. 原 Torch 版没有减均值，`data/wavelet.py` 的 NumPy 旧版才有 `Y - mean(Y)`；
2. 必须是 `log2(abs(wave) ** 2 + 1e-10)`；
3. 归一化是在压缩后对整张 `[30,60]` 做，不是先归一化再压缩。

### 全零输入出现 NaN

原因是：

```python
max_value - min_value == 0
```

本练习要求直接返回全零谱图。这是数值稳定契约，不是额外防御代码。

## 7. 工程加固（选读）

正式生产代码还可以检查：

- 输入长度是否严格为 3000；
- `output_time_bins` 是否整除输入长度；
- 输入是否含 NaN/Inf；
- 采样率是否和数据读取阶段一致。

这些检查有用，但不是本次练习主线。当前重点是理解 CWT 的数据流。
