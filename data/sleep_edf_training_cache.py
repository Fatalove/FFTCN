"""里程碑 9B 内部步骤一练习：构建完整训练使用的顺序缓存。

运行链从这里开始：

    记录级 NPZ
        -> raw.npy + labels.npy + manifest.json
        -> wave.npy
        -> 后续 Dataset/DataLoader

正式练习不会导入独立参考答案。完整中文讲解位于：

    learning_guides/milestone_09b/
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from data.sleep_edf_cwt import morlet_cwt_epoch

import json

# wave_transform 接收一个 [1,3000] EEG epoch，返回 [1,30,60] CWT 图。
WaveTransform = Callable[[np.ndarray], np.ndarray]


def build_raw_label_cache(
    processed_split_dir: str | Path,
    cache_root: str | Path,
    split: str,
) -> Path:
    """按记录顺序构建 raw/labels 缓存与记录边界 manifest。

    参数：
        processed_split_dir: 某个 split 的记录级 ``record_id.npz`` 目录。
        cache_root: split 级训练缓存的根目录。
        split: 写入 ``cache_root/split`` 的 split 名称。

    返回：
        生成的 ``manifest.json`` 路径。
    """

    "用户练习：两遍扫描 NPZ，构建 raw/labels/manifest"

    source_dir = Path(processed_split_dir)
    split_dir = Path(cache_root) / split
    split_dir.mkdir(parents=True, exist_ok=True)

    record_paths = tuple(sorted(source_dir.glob("*.npz")))

    record_lens: list[tuple[Path, int]] = []
    total_epochs = 0
    for record_path in record_paths:
        with np.load(record_path) as record:
            raw = record["x"]
            labels = record["y"]

            if raw.ndim != 3 or raw.shape[1:] != (1, 3000):
                raise ValueError(f"{record_path.name} 的 x 必须是 [N,1,3000]")

            if labels.ndim != 1 or len(labels) != len(raw):
                raise ValueError(f"{record_path.name} 的 x/y 样本轴不一致")

            epoch_count = int(len(raw))
            record_lens.append((record_path, epoch_count))
            total_epochs += epoch_count

    raw_path = split_dir / "raw.npy"
    labels_path = split_dir / "labels.npy"
    raw_cache = np.lib.format.open_memmap(
        raw_path,
        mode='w+',
        dtype=np.float32,
        shape=(total_epochs, 1, 3000),
    )
    labels_cache = np.lib.format.open_memmap(
        labels_path,
        mode='w+',
        dtype=np.int64,
        shape=(total_epochs,),

    )

    records: list[dict[str, object]] = []
    start = 0
    for record_path, epoch_count in record_lens:
        stop = start + epoch_count
        with np.load(record_path) as record:
            raw_cache[start:stop] = np.asarray(record["x"], dtype=np.float32)
            labels_cache[start:stop] = np.asarray(record["y"], dtype=np.int64)

        records.append({
            "record_id": record_path.stem,
            "source_file": record_path.name,
            "start": start,
            "stop": stop,
        })
        start = stop

    raw_cache.flush()
    labels_cache.flush()
    raw_cache._mmap.close()
    labels_cache._mmap.close()

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
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest_path







def build_wave_cache(
    cache_root: str | Path,
    split: str,
    wave_transform: WaveTransform = morlet_cwt_epoch,
) -> Path:
    """按 raw 全局顺序生成 wave 缓存，并把其元数据写回 manifest。

    ``wave_transform`` 默认使用里程碑 4 的重构 Morlet CWT；测试可注入一个
    形状相同的轻量函数，从而只验证数据顺序和缓存契约。

    返回：
        生成的 ``wave.npy`` 路径。
    """

    "用户练习：逐 epoch 生成 wave.npy 并更新 manifest"

    split_dir = Path(cache_root) / split
    manifest_path = split_dir / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    raw = np.load(split_dir / "raw.npy", mmap_mode="r")

    wave_path = split_dir / "wave.npy"
    wave_shape = (len(raw), 1, 30, 60)
    wave_cache = np.lib.format.open_memmap(
        wave_path,
        mode='w+',
        dtype=np.float16,
        shape=wave_shape,
    )

    try:
        for epoch_index in range(len(raw)):
            raw_epoch = np.array(raw[epoch_index], dtype=np.float32, copy=True)
            wave_epoch = np.asarray(wave_transform(raw_epoch), dtype=np.float32)
            if wave_epoch.shape != (1, 30, 60):
                raise ValueError(f"wave_transform output shape {wave_epoch.shape} not expected (1, 30, 60)")

            wave_cache[epoch_index] = wave_epoch

        wave_cache.flush()
    finally:
        wave_cache._mmap.close()
        if isinstance(raw, np.memmap) and raw._mmap is not None:
            raw._mmap.close()

    manifest["wave_shape"] = [int(size) for size in wave_shape]
    manifest["wave_dtype"] = "float16"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return wave_path
