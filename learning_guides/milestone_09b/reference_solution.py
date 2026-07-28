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
