"""Sleep-EDF-153 原始记录读取接口（里程碑 2 用户练习）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyedflib

@dataclass(frozen=True)
class SleepEDFPaths:
    """一条夜间记录及其配套人工标注。"""

    record_id: str
    subject_id: str
    night_id: str
    psg_path: Path
    hypnogram_path: Path


@dataclass(frozen=True)
class RawSleepEDFRecord:
    """尚未清洗、但已按 30 秒严格对齐的单条记录。"""

    paths: SleepEDFPaths
    sampling_rate: float
    eeg: np.ndarray  # float32, [N, 1, 3000]
    labels: np.ndarray  # str, [N]；本里程碑保留 W/R/1/2/3/4/M/?


def discover_records(root: Path) -> list[SleepEDFPaths]:
    """发现并一一配对 153 个 PSG/Hypnogram 记录。

    文件名契约：记录 ID 为前 6 个字符，例如 ``SC4001``；受试者 ID
    为前 5 个字符，例如 ``SC400``；第 6 个字符表示夜晚 ID。

    用户练习伪代码：
    1. 分别枚举 ``*-PSG.edf`` 与 ``*-Hypnogram.edf``；
    2. 用文件名前 6 个字符建立两个字典；
    3. 检查键集合完全相等且无重复；
    4. 按 record_id 排序并构造 ``SleepEDFPaths``。
    """

    psg_files = list(root.glob('*-PSG.edf'))
    hyp_files = list(root.glob('*-Hypnogram.edf'))

    psg_dict = {}
    for p in psg_files:
        record_id = p.name[:6]
        psg_dict[record_id] = p

    hyp_dict = {}
    for h in hyp_files:
        record_id = h.name[:6]
        hyp_dict[record_id] = h

    assert set(psg_dict.keys()) == set(hyp_dict.keys())
    assert len(psg_dict) == len(psg_files)
    assert len(hyp_dict) == len(hyp_files)

    records = []
    for record_id in sorted(psg_dict.keys()):
        subject_id = record_id[:5]
        night_id = record_id[5]
        records.append(
            SleepEDFPaths(
                record_id=record_id,
                subject_id=subject_id,
                night_id=night_id,
                psg_path=psg_dict[record_id],
                hypnogram_path=hyp_dict[record_id],
            )
        )


    return records


def read_record(
    paths: SleepEDFPaths,
    channel: str = "EEG Fpz-Cz",
    epoch_seconds: int = 30,
) -> RawSleepEDFRecord:
    """读取一条记录并生成未清洗、时间对齐的 30 秒 EEG 时段。

    约束：
    - 验证目标通道存在且采样率为 100 Hz；
    - 每段严格为 3000 点，输出 ``float32 [N, 1, 3000]``；
    - 按 annotation onset/duration 展开原始标签；
    - 以 PSG 的完整 30 秒段数为上限，裁掉超出信号末尾的标注；
    - 不在本里程碑合并 S3/S4、删除 M/? 或裁剪首尾 W。

    用户练习伪代码：
    1. 用 ``pyedflib.EdfReader`` 读取 PSG 头和目标通道；
    2. 计算 ``N = floor(信号采样点数 / 3000)`` 并切段；
    3. 从 Hypnogram 读取 onset、duration、description；
    4. 将每条持续标注展开到对应 30 秒索引，并截断到 ``N``；
    5. 断言每个时段恰有一个标签，且信号数与标签数相等。
    """

    reader = pyedflib.EdfReader(str(paths.psg_path))

    channel_labels = reader.getSignalLabels()
    assert channel in channel_labels, f"Channel {channel} not found in {paths.psg_path}"
    ch_idx = channel_labels.index(channel)

    sfreq = reader.getSampleFrequency(ch_idx)
    assert sfreq == 100, f"Sampling rate {sfreq} Hz not 100 Hz in {paths.psg_path}"

    signal = reader.readSignal(ch_idx)
    reader.close()

    samples_per_epoch = epoch_seconds * int(sfreq)
    N = len(signal) // samples_per_epoch

    signal = signal[:N * samples_per_epoch]
    eeg = signal.reshape(N, 1, samples_per_epoch).astype(np.float32)

    hyp_reader = pyedflib.EdfReader(str(paths.hypnogram_path))
    onsets, durations, descriptions = hyp_reader.readAnnotations()
    hyp_reader.close()

    labels = np.full(N, "?", dtype=object)
    for onset, duration, description in zip(onsets, durations, descriptions):
        start_epoch = int(onset) // epoch_seconds
        end_epoch = int(onset + duration) // epoch_seconds

        start_epoch = max(0, start_epoch)
        end_epoch = min(N, end_epoch)

        for i in range(start_epoch, end_epoch):
            labels[i] = description

    assert eeg.shape[0] == labels.shape[0]
    assert eeg.shape == (N, 1, samples_per_epoch), f"eeg shape {eeg.shape} not (N, 1, samples_per_epoch)"
    assert not np.any(labels == "?"), "Labels contain '?'"

    return RawSleepEDFRecord(
        paths=paths,
        sampling_rate=sfreq,
        eeg=eeg,
        labels=labels,
    )
