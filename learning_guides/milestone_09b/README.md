# 里程碑 9B 内部步骤一：训练缓存准备

步骤一已于 2026-07-21 通过 4/4 行为自测。本文件保留为步骤一学习材料；当前步骤二教程见 `STEP_02_DATASET_DATALOADER.md`。

正式练习文件：`data/sleep_edf_training_cache.py`<br>
行为测试：`tests/test_sleep_edf_training_cache.py`<br>
独立参考答案：`learning_guides/milestone_09b/reference_solution.py`

## 0. 一个里程碑，四个内部步骤

9B 不再拆成 9B.1、9B.2 等小里程碑，而是按真实运行链在同一个里程碑内完成四步：

1. **训练缓存准备**：实验配置；记录级 NPZ → raw/labels/manifest → 重构 CWT wave；
2. **数据读取**：manifest → 不跨记录的 Dataset/DataLoader；
3. **训练编排**：两分支预训练 → best checkpoint → 权重迁移 → 融合微调；
4. **正式验证**：小规模与显存预检 → seed 0 完整训练 → 一次正式测试与报告。

多随机种子均值与标准差是 9B 完成后的可选稳定性扩展，不再单列里程碑，也不阻塞 GitHub v0.1。

本指南只覆盖步骤一，不实现 Dataset，也不启动模型训练：

```text
milestone 3 的记录级 NPZ
        ↓ build_raw_label_cache
raw.npy + labels.npy + manifest.json
        ↓ build_wave_cache（只调用重构 morlet_cwt_epoch）
wave.npy
        ↓ 步骤二才开始
Dataset / DataLoader
```

你此前已经完成的 `FullTrainingConfig.to_metadata()` 保留在 `training/sleep_edf_full_run.py`。尚未写完的 `load_record_spans()` 也原样保留，但它属于步骤二，现在先暂停。

## 1. 问题分析

### 1.1 输入从哪里来

步骤一读取里程碑 3 已生成的目录：

```text
datasets/sleep-edf-153-processed-v1/
├── train/
│   ├── SC4001.npz
│   └── ...
├── validation/
└── test/
```

每个 `record_id.npz` 代表一个夜晚，内部契约是：

```text
x: [N,1,3000] float32
y: [N]        int64
```

`x[i]` 与 `y[i]` 必须属于同一个 30 秒 epoch。步骤一不重新读取 EDF、不重新划分受试者，也不改变标签。

### 1.2 三种 manifest 不能混为一谈

本项目中容易遇到三个名字相近但用途不同的文件：

| 文件 | 来源 | 用途 |
|---|---|---|
| `reproduction_artifacts/milestone_03/full_dataset_summary.json` | 里程碑 3 | 记录全量预处理与固定 split 的统计证据 |
| PhysioNet 下载 manifest / `SHA256SUMS.txt` | 数据发布方 | 下载文件清单和完整性校验 |
| `cache/<split>/manifest.json` | 本步骤的 `build_raw_label_cache()` | 保存拼接缓存中每条夜晚的 `[start,stop)` 边界 |

`load_record_spans()` 以后读取的是第三种。它不是手工下载的文件，而是本步骤自动生成的训练缓存元数据。

### 1.3 为什么要先生成 raw/labels/manifest

如果训练时每取一个 epoch 都重新打开记录级 NPZ，一条记录会被反复解析。我们先把同一个 split 的记录按文件名排序并拼成两个标准 `.npy`：

```text
raw.npy:    [总 N,1,3000] float32
labels.npy: [总 N]        int64
```

后续 Dataset 可以用 `mmap_mode='r'` 按需读取，不必把约 19.5 万个 epoch 一次装入 RAM，更不会一次装入 4 GB 显存。

拼接会隐藏夜晚边界，所以同一过程必须生成 `manifest.json`：

```json
{
  "record_id": "SC4001",
  "source_file": "SC4001.npz",
  "start": 0,
  "stop": 841
}
```

### 1.4 为什么采用两遍扫描

`.npy` 内存映射文件创建时必须先知道完整形状，但总 epoch 数要把所有记录长度相加后才能得到。

- 第一遍：逐个打开 NPZ，只检查形状并累计每条记录长度；
- 分配：创建固定形状的 `raw.npy` 和 `labels.npy`；
- 第二遍：再次逐个打开 NPZ，把数据复制到对应 `[start,stop)`。

这会多读一次磁盘，却避免把所有记录同时堆进内存；对当前 4 GB 显存机器更稳妥。注意这里控制的是 CPU 内存和磁盘 I/O，GPU 尚未参与。

### 1.5 为什么 wave 必须后生成

`wave[i]` 必须对应 `raw[i]` 和 `labels[i]`。因此先固定全局 raw 顺序，再按 `epoch_index=0...N-1` 调用里程碑 4 的 `morlet_cwt_epoch()`：

```text
raw[i]:  [1,3000]  ──CWT──>  wave[i]: [1,30,60]
labels[i] 保持不变
```

正式代码不调用原仓库的 `data/wavelet.py` 或原 Loader。测试会注入一个极快的假变换，只验证顺序、形状和对齐；正式构建时省略该参数，才执行真实重构 CWT。

`wave.npy` 以 float16 存盘以减少磁盘占用；步骤二的 Dataset 会在进入模型前复制为 float32。这里改变的是存储精度，不改变样本顺序。

## 2. 必要 Python 与 NumPy 基础

### 2.1 排序后的 `Path.glob`

```python
record_paths = tuple(sorted(source_dir.glob("*.npz")))
```

文件系统不保证目录项返回顺序。显式排序后，`record_id` 的拼接顺序才能跨运行保持一致；文件名的 `.stem` 是去掉 `.npz` 后的记录 ID。

### 2.2 NPZ 容器与上下文管理器

```python
with np.load(record_path) as record:
    raw = record["x"]
    labels = record["y"]
```

NPZ 是含多个命名数组的容器。`with` 结束后关闭当前文件，避免遍历 153 条记录时积累文件句柄。

### 2.3 `open_memmap` 与普通 `np.memmap`

```python
raw_cache = np.lib.format.open_memmap(
    "raw.npy", mode="w+", dtype=np.float32, shape=(total_epochs, 1, 3000)
)
```

`open_memmap` 创建的是带标准 `.npy` 头的内存映射文件，后续能直接用 `np.load(..., mmap_mode='r')` 读取。`w+` 表示创建/覆盖并允许读写。

### 2.4 左闭右开区间与同步切片

若第一条记录长度为 3，第二条长度为 2：

```text
record_a -> [0,3)
record_b -> [3,5)
```

写入时 raw 和 labels 必须使用完全相同的 `start:stop`。下一条记录的 `start` 等于上一条的 `stop`，于是区间既不重叠也无空洞。

### 2.5 可注入函数

```python
WaveTransform = Callable[[np.ndarray], np.ndarray]
```

`build_wave_cache()` 默认参数是真实 CWT；测试可传入同接口的轻量函数。可注入不等于改变正式算法：正式调用不传参数时仍固定使用重构 CWT。

## 3. 手工推演

假设输入目录有两个文件，而且创建顺序是 `record_b` 后 `record_a`：

```text
record_a.npz: x 有 3 个 epoch，y=[0,1,2]
record_b.npz: x 有 2 个 epoch，y=[3,4]
```

排序后先处理 a，再处理 b。第一遍得到：

```text
record_lengths = [(record_a,3), (record_b,2)]
total_epochs = 5
```

第二遍从 `start=0` 开始：

```text
record_a: stop=0+3=3，写 [0,3)，然后 start=3
record_b: stop=3+2=5，写 [3,5)，然后 start=5
```

最终结果：

```text
raw[:,0,0] = [a0,a1,a2,b0,b1]
labels       = [0, 1, 2, 3, 4]
records      = [a:[0,3), b:[3,5)]
```

wave 循环仍按索引 `0,1,2,3,4` 写入，所以 `raw[i]`、`wave[i]`、`labels[i]` 三者保持对齐。步骤二以后只需读取 manifest，就能恢复 a 与 b 的夜晚边界。

## 4. 带详细注释的完整核心代码

下面是步骤一的完整实现。请先读完前三节和代码注释，再遮住本节，在正式练习文件中按第 5 节顺序实现。

```python
"""里程碑 9B 内部步骤一参考答案：构建完整训练使用的顺序缓存。"""

from __future__ import annotations

# json 保存记录边界和数组元数据；Path 统一字符串路径与 Windows/Linux 路径对象。
import json
from pathlib import Path
# Callable 描述可注入的 CWT 函数，便于正式运行和轻量测试共享同一数据契约。
from typing import Callable

# NumPy 读取记录级 NPZ，并创建可由 Dataset 内存映射读取的标准 NPY 文件。
import numpy as np

# 默认只调用里程碑 4 的教学重构 CWT，不调用原仓库 wavelet/loader 实现。
from data.sleep_edf_cwt import morlet_cwt_epoch


# 该函数类型的输入是单个 [1,3000] EEG epoch，输出是 [1,30,60] CWT 图。
WaveTransform = Callable[[np.ndarray], np.ndarray]


def build_raw_label_cache(
    processed_split_dir: str | Path,
    cache_root: str | Path,
    split: str,
) -> Path:
    """按记录顺序构建 raw/labels 缓存与记录边界 manifest。

    输入目录中的每个 ``record_id.npz`` 必须包含 ``x:[N,1,3000]`` 和
    ``y:[N]``。函数按文件名排序后拼接记录，生成：

    - ``raw.npy``：``[总 epoch 数,1,3000] float32``；
    - ``labels.npy``：``[总 epoch 数] int64``；
    - ``manifest.json``：每条记录在上述样本轴中的 ``[start,stop)``。

    返回 manifest 路径，供 CWT 缓存构建和后续 Dataset 读取。
    """

    # source_dir 是里程碑 3 生成的某个 split 目录；每个 NPZ 对应一个夜晚。
    source_dir = Path(processed_split_dir)
    # Dataset 后续按 cache_root/split 查找四个缓存文件，因此生产者在这里固定目录契约。
    split_dir = Path(cache_root) / split
    split_dir.mkdir(parents=True, exist_ok=True)

    # 文件名就是 record_id；排序使拼接顺序不依赖操作系统返回目录项的顺序。
    record_paths = tuple(sorted(source_dir.glob("*.npz")))
    # 第一遍只收集每条记录的长度，先算出总 N，才能创建固定形状的 .npy memmap。
    # 每次只打开一条记录，不把 Sleep-EDF-153 全部 EEG 同时装入内存。
    record_lengths: list[tuple[Path, int]] = []
    total_epochs = 0
    for record_path in record_paths:
        # np.load 返回 NPZ 容器；with 会在当前记录检查完后立即关闭其文件句柄。
        with np.load(record_path) as record:
            raw = record["x"]
            labels = record["y"]

            # 模型的时域分支要求一个 epoch 是 [单通道,3000 采样点]；
            # 第 0 轴 N 才是稍后要跨记录拼接的样本轴。
            if raw.ndim != 3 or raw.shape[1:] != (1, 3000):
                raise ValueError(f"{record_path.name} 的 x 必须是 [N,1,3000]")
            # 一维标签的第 i 项必须和 raw 的第 i 个 epoch 描述同一时刻。
            if labels.ndim != 1 or len(labels) != len(raw):
                raise ValueError(f"{record_path.name} 的 x/y 样本轴不一致")
            # int() 把 NumPy shape 整数变成普通 Python int，供 range 和 JSON 使用。
            epoch_count = int(len(raw))
            record_lengths.append((record_path, epoch_count))
            total_epochs += epoch_count

    # open_memmap 直接创建带 .npy 头信息的磁盘数组；这里只分配文件，不占用同等 RAM。
    raw_path = split_dir / "raw.npy"
    labels_path = split_dir / "labels.npy"
    raw_cache = np.lib.format.open_memmap(
        raw_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_epochs, 1, 3000),
    )
    labels_cache = np.lib.format.open_memmap(
        labels_path,
        mode="w+",
        dtype=np.int64,
        shape=(total_epochs,),
    )

    # 第二遍才把每条记录复制到已经分配好的全局样本轴，并同步建立记录边界。
    records: list[dict[str, object]] = []
    start = 0
    for record_path, epoch_count in record_lengths:
        # 当前记录占用左闭右开区间 [start,stop)；长度严格等于 epoch_count。
        stop = start + epoch_count
        with np.load(record_path) as record:
            # 显式 dtype 统一缓存契约；raw 与 labels 使用相同切片，避免样本错位。
            raw_cache[start:stop] = np.asarray(record["x"], dtype=np.float32)
            labels_cache[start:stop] = np.asarray(record["y"], dtype=np.int64)

        # manifest 保留原始记录身份和来源文件；后续长度 50 序列只能在该区间内切分。
        records.append(
            {
                "record_id": record_path.stem,
                "source_file": record_path.name,
                "start": start,
                "stop": stop,
            }
        )
        # 下一条记录紧接当前 stop，因此 manifest 不会出现重叠或空洞。
        start = stop

    # flush 先把尚在缓冲区的数据写入磁盘，再关闭 Windows 持有的映射文件句柄。
    raw_cache.flush()
    labels_cache.flush()
    raw_cache._mmap.close()
    labels_cache._mmap.close()

    # manifest 同时描述数组整体和每条记录区间；所有 shape 元素转为普通 int 后可写 JSON。
    manifest = {
        "format_version": 1,
        "split": split,
        "sample_count": total_epochs,
        "raw_shape": [total_epochs, 1, 3000],
        "raw_dtype": "float32",
        "labels_shape": [total_epochs],
        "labels_dtype": "int64",
        "records": records,
    }
    manifest_path = split_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        # ensure_ascii=False 保留可读文本；indent=2 让记录边界可以人工复核。
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    # 下一环节以该路径为入口，先读 manifest，再从 raw.npy 构建 wave.npy。
    return manifest_path


def build_wave_cache(
    cache_root: str | Path,
    split: str,
    wave_transform: WaveTransform = morlet_cwt_epoch,
) -> Path:
    """按 raw 全局顺序生成 wave 缓存，并把其元数据写回 manifest。

    默认对每个 ``raw[index]`` 调用里程碑 4 的重构 Morlet CWT。输出
    ``wave.npy`` 为 ``[N,1,30,60] float16``；Dataset 稍后读取时再转回
    float32 供模型计算。函数返回生成的 wave 缓存路径。
    """

    # raw/labels/manifest 已由上一个函数写入同一个 split 目录。
    split_dir = Path(cache_root) / split
    manifest_path = split_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    # mmap_mode='r' 只映射 raw.npy；循环中每次复制一个 epoch 给 CWT，控制内存峰值。
    raw = np.load(split_dir / "raw.npy", mmap_mode="r")
    # CWT 固定把一个 EEG epoch 映射成 [单通道,30 个尺度,60 个时间格]。
    wave_path = split_dir / "wave.npy"
    wave_shape = (len(raw), 1, 30, 60)
    wave_cache = np.lib.format.open_memmap(
        wave_path,
        mode="w+",
        dtype=np.float16,
        shape=wave_shape,
    )

    try:
        for epoch_index in range(len(raw)):
            # copy=True 解除当前 epoch 与只读磁盘映射的联系；CWT 输入保持 [1,3000] 语义。
            raw_epoch = np.array(raw[epoch_index], dtype=np.float32, copy=True)
            # 注入函数与正式 CWT 共享 [1,3000] -> [1,30,60] 契约，便于快速行为测试。
            wave_epoch = np.asarray(wave_transform(raw_epoch), dtype=np.float32)
            if wave_epoch.shape != (1, 30, 60):
                raise ValueError("wave_transform 必须返回 [1,30,60]")

            # 使用同一个 epoch_index 写入，保证 raw[i]、wave[i] 和 labels[i] 始终对齐。
            # float16 只用于减少磁盘占用；后续 Dataset 会在模型边界恢复 float32。
            wave_cache[epoch_index] = wave_epoch

        # 所有 epoch 成功后才把 wave 数据刷新到磁盘；失败时 manifest 不会宣称 wave 已完成。
        wave_cache.flush()
    finally:
        # 无论 CWT 是否抛错，都释放两个 memmap，避免 Windows 锁住半成品缓存。
        wave_cache._mmap.close()
        if isinstance(raw, np.memmap) and raw._mmap is not None:
            raw._mmap.close()

    # 把 wave 元数据追加到原 manifest；records 区间保持不变，因此三种数组共享样本轴。
    manifest["wave_shape"] = [int(size) for size in wave_shape]
    manifest["wave_dtype"] = "float16"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    # 步骤二的 Dataset 将从 cache_root/split 同时映射 raw、wave、labels 和 manifest。
    return wave_path
```

## 5. 用户编码范围与自测

两个函数按依赖顺序编写，但作为同一组练习一次完成、一起检查：

1. `build_raw_label_cache()`：先做两遍 NPZ 扫描，生成 raw/labels/manifest；
2. `build_wave_cache()`：完成前一个函数后，继续逐 epoch 生成 wave 并更新 manifest。

两个函数都完成后统一运行：

```powershell
python -m unittest tests.test_sleep_edf_training_cache -v
```

正式练习文件不得导入 `learning_guides.milestone_09b.reference_solution`。全量 Sleep-EDF 真实 CWT 留到后续预检；步骤一先让 4 条小型行为测试全部通过。

这些测试是编码过程中的自测，不是两个函数各自的验收节点，也不会单独改变里程碑 9B 的状态。内部步骤只负责组织学习与运行顺序；正式验收仍按整个里程碑 9B 进行。

## 6. 常见错误与测试含义

- 不排序 `glob()`：同一批记录在不同系统上的拼接顺序可能不同；
- 用 `np.concatenate` 一次拼全部记录：会显著提高 CPU 内存峰值，失去 memmap 设计意义；
- raw 与 labels 分别维护不同游标：会造成 `raw[i]` 与 `labels[i]` 错位；
- 不检查 raw 精确形状：`[N,1,1]` 等错误数组可能被 NumPy 广播后静默写入 `[N,1,3000]` 缓存；
- manifest 的 `stop` 写成最后一个有效索引：本项目使用左闭右开区间，`stop` 应是最后索引加一；
- 用受试者 ID 代替 `record_path.stem`：同一受试者两个夜晚会失去独立边界；
- wave 重新遍历记录并使用另一种排序：会破坏三种数组的全局索引对齐；
- 测试时直接调用真实 CWT：虽然算法正确，却会把毫秒级契约测试变成长时间数值计算；
- 生成 wave 后不更新 manifest：下游无法确认 wave 文件是否完整以及预期形状；
- 把 float16 wave 直接送进模型：存盘可用 float16，但步骤二必须转换为 float32。

## 7. 工程加固（选读）与下一步

核心行为通过后可以再考虑：

- 临时文件加原子重命名，避免中断后留下看似完整的缓存；
- 为源 NPZ、raw 和 wave 保存 SHA-256；
- 在 manifest 记录采样率、CWT 参数和代码版本；
- 支持从某个 epoch 继续构建 wave；
- 添加进度条和耗时统计；
- 检查异常标签、NaN/Inf 和剩余磁盘空间。

这些不进入当前主答案，以免遮挡“排序 → 两遍扫描 → 同步写入 → 边界记录 → 按索引 CWT”的核心链路。

步骤一通过后，仍在同一个里程碑 9B 内进入步骤二：阅读 `STEP_02_DATASET_DATALOADER.md`，回到已保留的 `training/sleep_edf_full_run.py`，从 `load_record_spans()` 开始读取本步骤生成的 manifest，再实现 Dataset/DataLoader。只有步骤二完成后才进入训练编排。
