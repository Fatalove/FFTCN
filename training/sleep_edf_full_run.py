"""里程碑 9B 的共享实验配置、数据读取与完整训练编排练习。

步骤一训练缓存和步骤二 Dataset/DataLoader 已通过行为自测。当前步骤三
复用这些接口以及里程碑 8 的训练核心策略，完成记录内偏移过采样缓存、
epoch 循环、validation 选优、scheduler、日志、断点恢复和两步训练编排。

正式练习代码不会导入独立参考答案。完整中文讲解位于：

    learning_guides/milestone_09b/
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data.sleep_edf_cwt import morlet_cwt_epoch
from models.merge.sleep_edf_fusion_tcn import FFTCNFusionTCN
from models.raw.sleep_edf_1d_cnn import SleepEDFRawFeatureNet
from models.wavelet.sleep_edf_2d_cnn import SleepEDFWaveFeatureNet
from training.sleep_edf_two_stage import (
    LABEL_NAMES,
    build_finetune_optimizer,
    offset_resample_record,
    run_classification_step,
    transfer_pretrained_features,
)

InputMode = Literal["raw", "wave", "both"]
WaveTransform = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class FullTrainingConfig:
    """保存重构模型一次完整训练所需的最小可复现实验配置。"""

    data_cache_dir: Path
    output_dir: Path
    seed: int = 0
    offset_samples: int = 300
    sequence_length: int = 50
    raw_pretrain_epochs: int = 20
    wave_pretrain_epochs: int = 20
    fusion_finetune_epochs: int = 50
    raw_batch_size: int = 128
    wave_batch_size: int = 128
    fusion_batch_size: int = 32
    pretrain_learning_rate: float = 1e-5
    fusion_learning_rate: float = 1e-5
    feature_learning_rate_scale: float = 1e-2
    scheduler_gamma: float = 0.95
    num_workers: int = 0
    device: str = "auto"

    def __post_init__(self) -> None:
        """拒绝无法形成有效训练循环的非正长度、轮数或 batch size。"""

        positive_integers = {
            "sequence_length": self.sequence_length,
            "raw_pretrain_epochs": self.raw_pretrain_epochs,
            "wave_pretrain_epochs": self.wave_pretrain_epochs,
            "fusion_finetune_epochs": self.fusion_finetune_epochs,
            "raw_batch_size": self.raw_batch_size,
            "wave_batch_size": self.wave_batch_size,
            "fusion_batch_size": self.fusion_batch_size,
        }
        invalid = [name for name, value in positive_integers.items() if value <= 0]
        if invalid:
            raise ValueError(f"以下配置必须为正整数：{invalid}")
        if self.num_workers < 0:
            raise ValueError("num_workers 不能为负数")

    def to_metadata(self) -> dict[str, Any]:
        """返回可直接写入 JSON 或 checkpoint 的普通 Python 字典。"""

        metadata = asdict(self)
        metadata["data_cache_dir"] = str(self.data_cache_dir.resolve())
        metadata["output_dir"] = str(self.output_dir.resolve())
        return metadata


@dataclass(frozen=True)
class RecordSpan:
    """描述一条记录在 split 级拼接数组中的左闭右开区间。"""

    record_id: str
    start: int
    stop: int


@dataclass(frozen=True)
class SequencePosition:
    """描述一个 Dataset 样本来自哪条记录及其全局左闭右开区间。"""

    record_id: str
    start: int
    stop: int


def load_record_spans(manifest_path: str | Path) -> tuple[RecordSpan, ...]:
    """读取 manifest，并返回按时间顺序连续覆盖缓存数组的记录边界。"""

    "用户练习：读取 records，并验证区间连续且非空"
    path = Path(manifest_path)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    rows = payload.get("records", [])
    if not rows:
        raise ValueError("manifest 中必须包含 records 字段")

    expected_start = 0
    spans: list[RecordSpan] = []

    for row in rows:
        start = int(row["start"])
        stop = int(row["stop"])
        record_id = str(row["record_id"])

        if start != expected_start:
            raise ValueError(f"manifest 中的记录区间必须从 0 开始并连续排列")



        spans.append(RecordSpan(record_id, start, stop))
        expected_start = stop

    return tuple(spans)






def build_sequence_positions(
    record_spans: Sequence[RecordSpan],
    sequence_length: int,
) -> tuple[SequencePosition, ...]:
    """在每条记录内部构造非重叠序列，丢弃不足一个完整序列的尾部。"""

    "用户练习：按记录建立序列索引，禁止跨记录拼接"
    if sequence_length <= 0:
        raise ValueError("sequence_length 必须为正整数")

    positions: list[SequencePosition] = []
    for span in record_spans:
        record_length = span.stop - span.start
        usable_length = (record_length // sequence_length) * sequence_length

        for i in range(0, usable_length, sequence_length):
            global_start = span.start + i
            global_stop = global_start + sequence_length
            positions.append(
                SequencePosition(span.record_id, global_start, global_stop)
            )

    return tuple(positions)


class SleepEDFSequenceDataset(Dataset):
    """从 split 级内存映射缓存读取单时段或连续序列。"""

    def __init__(
        self,
        cache_root: str | Path,
        split: str,
        sequence_length: int,
        input_mode: InputMode,
    ) -> None:
        """打开缓存数组，核对样本轴，并建立不跨记录的样本索引。"""

        "用户练习：打开 raw/wave/labels 缓存并建立 positions"

        if input_mode not in {"raw", "wave", "both"}:
            raise ValueError(f"input_mode 必须是 {'raw', 'wave', 'both'} 中的一个")

        self.split_dir = Path(cache_root) / split
        self.input_mode = input_mode
        self.sequence_length = sequence_length

        self.labels = np.load(self.split_dir / "labels.npy", mmap_mode="r")

        self.raw = None
        if input_mode in {"raw", "both"}:
            self.raw = np.load(self.split_dir / "raw.npy", mmap_mode="r")

        self.wave = None
        if input_mode in {"wave", "both"}:
            self.wave = np.load(self.split_dir / "wave.npy", mmap_mode="r")

        self.record_spans = load_record_spans(self.split_dir / "manifest.json")

        self.positions = build_sequence_positions(
            self.record_spans, sequence_length
        )




    def __len__(self) -> int:
        """返回当前 split 中完整单时段或完整序列的数量。"""

        "用户练习：返回序列索引数量"
        return len(self.positions)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """返回模型输入元组和对应标签，且将缓存数值复制为可训练张量。"""

        "用户练习：切片、转换 dtype，并处理 T=1 的序列轴"
        position = self.positions[index]

        start = position.start
        stop = position.stop

        label_array = np.array(self.labels[start:stop], dtype=np.int64, copy=True)
        targets = torch.from_numpy(label_array)

        inputs: list[torch.Tensor] = []
        if self.raw is not None:
            raw_array = np.array(self.raw[start:stop], dtype=np.float32, copy=True)
            inputs.append(torch.from_numpy(raw_array))

        if self.wave is not None:
            wave_array = np.array(self.wave[start:stop], dtype=np.float32, copy=True)
            inputs.append(torch.from_numpy(wave_array))

        if self.sequence_length == 1:
            inputs = inputs[0]
            targets = targets[0]

        return tuple(inputs), targets

    def position_at(self, index: int) -> SequencePosition:
        """返回指定样本的记录身份与区间，供错误分析和按记录评估使用。"""

        "用户练习：从 positions 返回对应位置"
        return self.positions[index]

    def close(self) -> None:
        """关闭 NumPy 内存映射文件，释放 Windows 持有的缓存文件句柄。"""

        "用户练习：关闭 labels/raw/wave 中存在的 memmap"
        for npy in (self.raw, self.wave, self.labels):
            if isinstance(npy, np.memmap) and npy._mmap is not None:
                npy._mmap.close()



def build_reproducible_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
) -> DataLoader:
    """创建只由局部 generator 控制 shuffle 顺序的 DataLoader。"""

    "用户练习：创建局部 torch.Generator 并传给 DataLoader"
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        drop_last=False,
    )


@dataclass(frozen=True)
class StageTrainingResult:
    """保存一个训练阶段的最佳/恢复检查点和逐轮历史。"""

    stage: str
    best_checkpoint: Path
    last_checkpoint: Path
    history_path: Path
    history: tuple[dict[str, Any], ...]


def _balanced_record_count(labels: np.ndarray) -> int:
    """计算一条记录经过偏移过采样后的总样本数。

    参数来源：
        labels: 一条记录的标签，形状为 ``[N]``。它由
            ``build_balanced_pretrain_cache()`` 按 manifest 的 ``[start:stop)``
            区间从 ``train/labels.npy`` 中取出。

    返回去向：
        返回该记录的“原始 N 项 + 新增偏移项”总数。上游缓存构建函数先收集
        每条记录的这个数，再求和得到 memmap 第一轴必须预分配的总长度。
    """

    "用户练习：根据各类别数量和可用内部中心，计算原样本与新增偏移样本总数"
    # unique_labels 是本记录实际出现的类别；counts 是这些类别各自的原始数量。
    # n_max 对应 offset_resample_record 为每个可采样类别新增的样本数。
    unique_labels, counts = np.unique(labels, return_counts=True)
    n_max = int(np.max(counts))

    # extra_count 只累计新增项；原始 len(labels) 留到返回时再加。
    extra_count = 0
    # 首尾 epoch 不能作为偏移中心，否则新窗口可能越过当前记录边界。
    boundary_indices = np.array([0, len(labels) - 1])

    # 逐类判断是否至少有一个排除首尾后的内部候选中心。
    for label in unique_labels:
        # 先取得当前类别全部局部索引，再删除记录边界索引。
        candidates = np.flatnonzero(labels == label)
        candidates = np.setdiff1d(candidates, boundary_indices)
        # 只要仍有候选，既有过采样算法就会为该类别新增 n_max 项。
        if len(candidates) > 0:
            extra_count += n_max

    # 返回“全部保留的原始项 + 各可采样类别新增项”，供 memmap 预分配第一轴。
    return int(len(labels) + extra_count)



def build_balanced_pretrain_cache(
    cache_root: str | Path,
    seed: int,
    offset_samples: int = 300,
    wave_transform: WaveTransform = morlet_cwt_epoch,
) -> Path:
    """逐记录偏移过采样，并写出对齐的 ``pretrain_train`` 缓存。

    参数来源：
        cache_root: 训练缓存根目录。正式运行时来自
            ``FullTrainingConfig.data_cache_dir``，其下已有步骤一生成的 train 缓存。
        seed: 来自 ``FullTrainingConfig.seed`` 的基础随机种子；每条记录再加上
            ``record_index``，形成独立且可复现的局部 seed。
        offset_samples: 来自 ``FullTrainingConfig.offset_samples`` 的最大偏移点数，
            默认 300 点即 100 Hz 下 3 秒。
        wave_transform: 把 raw ``[1,3000]`` 变为 CWT ``[1,30,60]`` 的函数；
            正式运行使用 ``morlet_cwt_epoch``，微型测试可以传入替身。

    返回去向：
        返回 ``pretrain_train/manifest.json`` 路径；训练 Dataset 随后从同目录映射
        raw、wave 和 labels，不把全量缓存一次装入内存。
    """

    "用户练习：逐记录生成平衡 raw/labels，再由同一批 raw 生成 wave"
    # 统一路径类型后，input_dir 指向原始 train，output_dir 单独保存平衡预训练缓存。
    cache_root = Path(cache_root)
    input_dir = cache_root / "train"
    output_dir = cache_root / "pretrain_train"
    # 只创建输出目录，不覆盖融合阶段仍要读取的原始 train 缓存。
    output_dir.mkdir(parents=True, exist_ok=True)

    # raw/labels 以内存映射读取，manifest 提供每条记录不可跨越的 [start:stop) 边界。
    input_raw = np.load(input_dir / "raw.npy", mmap_mode="r")
    input_labels = np.load(input_dir / "labels.npy", mmap_mode="r")
    record_spans = load_record_spans(input_dir / "manifest.json")

    # open_memmap 必须预先知道总长度，因此第一遍只统计各记录过采样后的长度。
    balanced_lens: list[int] = []
    for span in record_spans:
        # 用 manifest 的当前记录区间取得独立 [N] 标签，再计算该记录输出长度。
        labels = np.array(input_labels[span.start:span.stop], copy=True)
        balanced_lens.append(_balanced_record_count(labels))

    # 三个输出缓存共享同一第一轴；总长度是所有记录输出长度之和。
    total_epochs = sum(balanced_lens)

    # raw、wave、labels 的索引 i 必须描述同一个 epoch；这里先固定三者磁盘形状。
    raw_shape = (total_epochs, *input_raw.shape[1:])
    wave_shape = (total_epochs, 1, 30, 60)
    # raw 保持 float32，供模型读取时不损失原始时域精度。
    balanced_raw = np.lib.format.open_memmap(
        output_dir / "raw.npy",
        mode="w+",
        dtype=np.float32,
        shape=raw_shape,
    )

    # wave 沿用 float16 磁盘存储以减小体积；Dataset 取样时再转为 float32。
    balanced_wave = np.lib.format.open_memmap(
        output_dir / "wave.npy",
        mode="w+",
        dtype=np.float16,
        shape=wave_shape,
    )

    # 交叉熵目标是类别索引，因此标签缓存使用一维 int64。
    balanced_labels = np.lib.format.open_memmap(
        output_dir / "labels.npy",
        mode="w+",
        dtype=np.int64,
        shape=(total_epochs,),
    )

    # records 收集新缓存中的记录边界；cursor 保存下一条记录的全局写入起点。
    # cursor 必须独立维护，因为每条记录过采样后的长度不同。
    records: list[dict[str, Any]] = []
    cursor = 0
    try:
        # 第二遍逐记录真正过采样；record_index 同时用于派生本记录局部 seed。
        for record_index, span in enumerate(record_spans):
            # raw 和 labels 用完全相同的原始区间复制到内存，保持样本一一对应。
            raw_record = np.array(
                input_raw[span.start:span.stop], dtype=np.float32, copy=True
            )
            labels_record = np.array(
                input_labels[span.start:span.stop], dtype=np.int64, copy=True
            )
            # 既有里程碑 8 函数只在当前记录内部生成“原始项 + 偏移项”。
            record_raw, record_labels = offset_resample_record(
                raw_record,
                labels_record,
                offset_samples,
                seed=seed + record_index,
            )

            # 当前记录在新缓存占用 [cursor:stop)，raw 和 labels 写入同一区间。
            stop = cursor + len(record_labels)
            balanced_raw[cursor:stop] = record_raw
            balanced_labels[cursor:stop] = record_labels

            # 原始 wave 没有新增偏移项，因此必须由平衡后的每个 raw_epoch 重新计算。
            for i, raw_epoch in enumerate(record_raw):
                wave_epoch = np.asarray(wave_transform(raw_epoch), dtype=np.float32)
                # cursor + i 把记录内索引映射成与 raw/label 一致的全局索引。
                balanced_wave[cursor + i] = wave_epoch

            # 保存新旧区间和本记录 seed，供 Dataset 建序列边界和实验追溯。
            records.append(
                {
                    "record_id": span.record_id,
                    "start": cursor,
                    "stop": stop,
                    "input_start": span.start,
                    "input_stop": span.stop,
                    "seed": seed + record_index,
                }
            )
            # 下一条记录紧接当前 stop 写入，记录之间连续但不会互相重叠。
            cursor = stop

        # 将尚在操作系统缓存中的三个输出数组修改全部落盘。
        balanced_raw.flush()
        balanced_wave.flush()
        balanced_labels.flush()

    finally:
        # 成功或异常都关闭输入/输出 memmap，避免 Windows 持续锁住缓存文件。
        for npy in (
            balanced_raw,
            balanced_wave,
            balanced_labels,
            input_raw,
            input_labels
        ):
            # 普通 ndarray 没有底层映射；只关闭仍处于打开状态的 memmap。
            if isinstance(npy, np.memmap) and npy._mmap is not None:
                npy._mmap.close()


    # manifest 固化共同样本数、三个数组契约、过采样配置和新记录边界。
    manifest = {
        "format_version": 1,
        "split": "pretrain_train",
        "sample_count": total_epochs,
        "raw_shape": raw_shape,
        "raw_dtype": "float32",
        "wave_shape": wave_shape,
        "wave_dtype": "float16",
        "labels_shape": [total_epochs],
        "labels_dtype": "int64",
        "offset_samples": offset_samples,
        "base_seed": seed,
        "records": records,
    }

    # 把可读 JSON 写到新缓存目录，并返回路径供步骤四记录产物位置。
    manifest_path = output_dir / "manifest.json"
    # ensure_ascii=False 保留可读文本，indent=2 便于人工核对缓存元数据。
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # 返回小型 manifest 路径而不是大型数组；训练 Dataset 会按需映射三个 .npy。
    return manifest_path


def run_epoch(
    model: nn.Module,
    batches: Iterable[tuple[Sequence[torch.Tensor], torch.Tensor]],
    device: str | torch.device,
    optimizer: Optimizer | None = None,
) -> float:
    """运行一个训练或验证 epoch，并返回所有预测位置的平均损失。

    参数来源：
        model: ``run_two_stage_training()`` 创建、再由 ``fit_stage()`` 传入的
            raw CNN、wave CNN 或融合模型。
        batches: ``run_full_training()`` 创建、由 ``fit_stage()`` 选中的 train 或
            validation DataLoader；每项为 ``(inputs, targets)``。
        device: ``config.device`` 经 ``_resolve_device()`` 解析后的 CPU/GPU。
        optimizer: 训练轮传入当前阶段 optimizer；验证轮传 None。既有分类步骤
            依靠它决定是否反向传播和更新参数。

    返回去向：
        返回按标签位置数加权的 epoch 平均交叉熵；``fit_stage()`` 把它记录为
        train/validation loss，并用 validation loss 选择 ``best.pt``。
    """

    "用户练习：移动 batch、调用分类步骤，并按 targets.numel() 汇总损失"
    # model 可能刚在 CPU 上创建，而 batch 会被移到 device；先把参数和缓冲区
    # 移到同一设备，后续前向计算才不会出现 CPU/GPU 混用错误。
    model.to(device)

    # weighted_loss 保存 epoch 损失公式的分子：Σ(batch平均损失 × batch位置数)。
    # 它从 0.0 开始，随后加入每个 batch 对总损失的贡献。
    weighted_loss = 0.0
    # position_count 保存同一公式的分母：本 epoch 已处理的标签位置总数。
    # 独立计数是因为最后一个 batch 可能较小，不能用 batch 数代替位置数。
    position_count = 0
    # correct_count 统计同一批已处理位置中预测正确的数量，用于实时累计 ACC。
    correct_count = 0

    # DataLoader 每次产生一个 batch；循环结束才表示 train 或 validation 整轮完成。
    for inputs, targets in batches:
        # inputs 保持 (raw,)、(wave,) 或 (raw,wave) 的模型调用顺序；逐个移动
        # 而不堆叠，后续分类步骤才能继续用 model(*inputs) 正确解包。
        moved_inputs = tuple(input.to(device) for input in inputs)
        # 标签也移到同一设备，但仍保持预训练 [B] 或融合训练 [B,T] 的语义形状。
        moved_targets = targets.to(device)

        # 复用里程碑 8 的单 batch 核心：optimizer 非 None 时更新参数，为 None 时验证。
        batch_loss, batch_correct_count = run_classification_step(
            model,
            moved_inputs,
            moved_targets,
            optimizer,
            return_correct_count=True,
        )

        # numel() 是当前 batch 的真实标签位置数：预训练为 B，融合训练为 B×T。
        batch_position_count = int(moved_targets.numel())
        # batch_loss 是 batch 平均值；乘位置数还原成该 batch 的损失总和并加入分子。
        weighted_loss += batch_loss * batch_position_count
        # 把相同 batch 的位置数加入分母，使最终分子与分母覆盖完全相同的数据。
        position_count += batch_position_count
        # 正确位置数使用同一组展平后的 logits/targets，不做额外模型前向。
        correct_count += batch_correct_count
        # fit_stage 会把 tqdm 进度条作为 batches 传入；每完成一个 batch，就显示
        # 截至当前位置的累计 loss 和 ACC。refresh=False 让 tqdm 按自身刷新频率
        # 绘制，避免为了显示指标而额外刷新数千次终端。
        set_postfix = getattr(batches, "set_postfix", None)
        if callable(set_postfix):
            set_postfix(
                loss=f"{weighted_loss / position_count:.4f}",
                acc=f"{100.0 * correct_count / position_count:.2f}%",
                refresh=False,
            )

    # 所有位置的损失总和 ÷ 所有位置数，就是整个 epoch 的平均交叉熵。
    return weighted_loss / position_count



def save_stage_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any,
    stage: str,
    epoch: int,
    best_validation_loss: float,
    history: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> None:
    """保存能够继续当前训练阶段的完整 checkpoint。

    参数来源：
        path: ``fit_stage()`` 生成的 ``best.pt`` 或 ``last.pt`` 路径。
        model: ``fit_stage()`` 当前正在训练的阶段模型。
        optimizer/scheduler: ``run_two_stage_training()`` 为该阶段创建并传入的
            Adam 与 ExponentialLR。
        stage/epoch: ``fit_stage()`` 的阶段名和当前刚完成的零起始轮次。
        best_validation_loss/history: ``fit_stage()`` 截至当前轮维护的最佳验证损失
            与逐轮日志。
        config: ``FullTrainingConfig.to_metadata()`` 生成的实验配置字典。

    返回去向：
        无业务返回值；把模型和恢复训练需要的全部状态写入 path，供
        ``load_stage_checkpoint()`` 断点续训或加载 validation-best 权重。
    """

    "用户练习：保存步骤三断点恢复所需的完整 state_dict 与元数据"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "stage": stage,
            "epoch": epoch,
            "best_validation_loss": float(best_validation_loss),
            "history": [dict(row) for row in history],
            "config": dict(config),
            "label_names": list(LABEL_NAMES),
        },
        path,
    )


def load_stage_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """读取 checkpoint，并把其中需要的训练状态装回现有对象。

    参数来源：
        path: 断点续训时来自 ``fit_stage(resume_from=...)``；加载 best 时来自
            ``StageTrainingResult.best_checkpoint``。
        model: 调用方已经按当前代码创建的阶段模型，任何加载场景都必须恢复。
        optimizer/scheduler: 只在继续同一训练阶段时由 ``fit_stage()`` 传入；
            只加载 best 做权重迁移时保持 None。
        map_location: 当前训练设备；默认 CPU 便于跨设备读取。

    返回去向：
        返回完整 checkpoint 字典；``fit_stage()`` 从中读取恢复游标、旧 best 和
        history，预训练结束加载 best 时则只使用已装入 model 的参数。
    """

    "用户练习：按需恢复 optimizer/scheduler，同时始终恢复模型"
    path = Path(path)
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint




def fit_stage(
    stage: str,
    model: nn.Module,
    train_loader: Iterable[tuple[Sequence[torch.Tensor], torch.Tensor]],
    validation_loader: Iterable[tuple[Sequence[torch.Tensor], torch.Tensor]],
    optimizer: Optimizer,
    scheduler: Any,
    epochs: int,
    device: str | torch.device,
    output_dir: str | Path,
    config: Mapping[str, Any],
    resume_from: str | Path | None = None,
) -> StageTrainingResult:
    """完成一个训练阶段的 epoch 循环、选优、日志和断点恢复。

    参数来源：
        stage/model: ``run_two_stage_training()`` 给出的阶段名和当前阶段模型。
        train_loader/validation_loader: ``run_full_training()`` 创建的对应 DataLoader。
        optimizer/scheduler: 上游为当前模型创建的 Adam 和 ExponentialLR。
        epochs: FullTrainingConfig 中该阶段计划完成的总轮数。
        device/output_dir/config: 分别来自解析后的设备、配置输出目录和
            ``config.to_metadata()``；恢复 optimizer 前先把 model 移到该设备。
        resume_from: 步骤四可选的 ``last.pt`` 路径；首次训练为 None。

    返回去向：
        返回 best/last/history 路径和完整历史；预训练结果的 best 路径供
        ``run_two_stage_training()`` 重新加载，fusion best 供步骤四正式评估。
    """

    "用户练习：完成 epoch 循环、validation 选优、scheduler、日志和恢复"
    stage_dir = Path(output_dir) / stage
    best_path = stage_dir / "best.pt"
    last_path = stage_dir / "last.pt"
    history_path = stage_dir / "history.json"
    stage_dir.mkdir(parents=True, exist_ok=True)

    # Adam 恢复动量时会参照当前模型参数所在设备；必须在读取 checkpoint 前移动模型。
    # 若等到 run_epoch 才移动，模型会在 CUDA，而恢复出的 exp_avg 仍在 CPU。
    model.to(device)

    start_epoch = 0
    best_validation_loss = float("inf")
    history = []

    if resume_from is not None:
        checkpoint = load_stage_checkpoint(
            resume_from, model, optimizer, scheduler, map_location=device
        )

        start_epoch = checkpoint["epoch"] + 1
        best_validation_loss = checkpoint["best_validation_loss"]
        history = checkpoint["history"]

    # 阶段名是整段训练的上层上下文，只在阶段开始时显示一次。每个 batch
    # 进度条便只需保留 epoch 和 Train/Valid，避免几十轮重复同一长前缀。
    if start_epoch < epochs:
        stage_title = stage.replace("_", " ").upper()
        tqdm.write(f"\n{'=' * 20} {stage_title} {'=' * 20}")

    for epoch in range(start_epoch, epochs):
        learning_rates = [float(p["lr"]) for p in optimizer.param_groups]
        # 一个 epoch 内真正重复执行的是 batch；直接包装 DataLoader 后，进度条
        # 的分母就是 len(train_loader)，每完成一次参数更新便推进一个 batch。
        with tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs} | Train",
            unit="batch",
            ncols=100,
            disable=None,
        ) as train_batches:
            train_loss = run_epoch(
                model, train_batches, device, optimizer,
            )

        # validation 同样按 batch 前向计算；单独显示进度可以区分模型更新已经结束，
        # 当前正在汇总用于选择 best checkpoint 的验证损失。
        with tqdm(
            validation_loader,
            desc=f"Epoch {epoch + 1}/{epochs} | Valid",
            unit="batch",
            ncols=100,
            disable=None,
        ) as validation_batches:
            validation_loss = run_epoch(
                model, validation_batches, device, optimizer=None,
            )

        history.append(
            {
                "epoch": epoch,
                "learning_rates": learning_rates,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )

        scheduler.step()

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            save_stage_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                stage,
                epoch,
                best_validation_loss,
                history,
                config,
            )

        save_stage_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            stage,
            epoch,
            best_validation_loss,
            history,
            config,
        )

        with open(history_path, "w",encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    return StageTrainingResult(
        stage=stage,
        best_checkpoint=best_path,
        last_checkpoint=last_path,
        history_path=history_path,
        history=tuple(history),
    )





def _resolve_device(device: str) -> torch.device:
    """把配置设备字符串解析成三个训练阶段共同使用的 ``torch.device``。

    参数来源：
        device: ``FullTrainingConfig.device``，通常为 ``"auto"``，也可由步骤四
            显式指定 ``"cpu"`` 或 ``"cuda:0"``。

    返回去向：
        返回值由 ``run_two_stage_training()`` 传入 ``fit_stage()``、``run_epoch()``
        和 checkpoint 加载函数，统一控制模型与 batch 所在设备。
    """

    "用户练习：auto 优先选择可用 CUDA，否则使用 CPU；显式设备直接转换"
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device)


def run_two_stage_training(
    config: FullTrainingConfig,
    loaders: Mapping[str, DataLoader],
    raw_model: nn.Module | None = None,
    wave_model: nn.Module | None = None,
    fusion_model: nn.Module | None = None,
    resume_checkpoints: Mapping[str, str | Path] | None = None,
) -> dict[str, StageTrainingResult]:
    """连接 raw/wave 预训练、best 权重迁移和融合微调。

    参数来源：
        config: ``run_full_training()`` 传入的完整实验配置。
        loaders: ``run_full_training()`` 创建的六个 train/validation Loader 映射。
        raw_model/wave_model/fusion_model: 正式运行保持 None 并由本函数创建；
            行为测试可注入微型模型。
        resume_checkpoints: 步骤四可选的阶段名到 ``last.pt`` 路径映射。

    返回去向：
        返回三个阶段的 ``StageTrainingResult``；fusion best 随后进入正式 test。
    """

    "用户练习：连接三个 fit_stage，并确保迁移的是两个 validation best"
    device = _resolve_device(config.device)
    metadata = config.to_metadata()
    resume_checkpoints = {} if resume_checkpoints is None else resume_checkpoints
    raw_model = SleepEDFRawFeatureNet() if raw_model is None else raw_model
    wave_model = SleepEDFWaveFeatureNet() if wave_model is None else wave_model
    fusion_model = FFTCNFusionTCN() if fusion_model is None else fusion_model

    raw_model.pretrain()
    raw_optimizer = torch.optim.Adam(
        raw_model.parameters(), lr=config.pretrain_learning_rate
    )
    raw_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        raw_optimizer, gamma=config.scheduler_gamma
    )

    raw_result = fit_stage(
        "raw_pretrain",
        raw_model,
        loaders["raw_pretrain_train"],
        loaders["raw_pretrain_validation"],
        raw_optimizer,
        raw_scheduler,
        config.raw_pretrain_epochs,
        device,
        config.output_dir,
        metadata,
        resume_checkpoints.get("raw_pretrain"),
    )


    wave_model.pretrain()
    wave_optimizer = torch.optim.Adam(
        wave_model.parameters(), lr=config.pretrain_learning_rate
    )
    wave_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        wave_optimizer, gamma=config.scheduler_gamma
    )
    wave_result = fit_stage(
        "wave_pretrain",
        wave_model,
        loaders["wave_pretrain_train"],
        loaders["wave_pretrain_validation"],
        wave_optimizer,
        wave_scheduler,
        config.wave_pretrain_epochs,
        device,
        config.output_dir,
        metadata,
        resume_checkpoints.get("wave_pretrain"),
    )

    load_stage_checkpoint(raw_result.best_checkpoint, raw_model, map_location=device)
    load_stage_checkpoint(wave_result.best_checkpoint, wave_model, map_location=device)

    transfer_pretrained_features(
        fusion_model,
        raw_model,
        wave_model,
    )
    fusion_optimizer = build_finetune_optimizer(
        fusion_model,
        config.fusion_learning_rate,
        config.feature_learning_rate_scale
    )

    fusion_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        fusion_optimizer, gamma=config.scheduler_gamma
    )

    fusion_result = fit_stage(
        "fusion_finetune",
        fusion_model,
        loaders["fusion_finetune_train"],
        loaders["fusion_finetune_validation"],
        fusion_optimizer,
        fusion_scheduler,
        config.fusion_finetune_epochs,
        device,
        config.output_dir,
        metadata,
        resume_checkpoints.get("fusion_finetune"),
    )

    return {
        "raw_pretrain": raw_result,
        "wave_pretrain": wave_result,
        "fusion_finetune": fusion_result,
    }




def run_full_training(
    config: FullTrainingConfig,
    resume_checkpoints: Mapping[str, str | Path] | None = None,
) -> dict[str, StageTrainingResult]:
    """从磁盘缓存建立六个 DataLoader，并启动完整两步训练。

    参数来源：
        config: 步骤四预检通过后创建的 FullTrainingConfig；其 data_cache_dir
            已包含 train、validation 和 pretrain_train 缓存。
        resume_checkpoints: 步骤四恢复预检传入的“阶段名 -> last.pt”
            映射；首次训练为 None。本函数将它继续交给
            ``run_two_stage_training()``，不在 DataLoader 层解析 checkpoint。

    返回去向：
        返回 ``run_two_stage_training()`` 的三个阶段结果，供步骤四加载 fusion
        validation-best checkpoint 并进行一次正式 test。这里不创建 test Loader。
    """

    "步骤四接口：把 resume_checkpoints 透传给两步训练"
    datasets = {
        "raw_pretrain_train": SleepEDFSequenceDataset(
            config.data_cache_dir, "pretrain_train", 1, "raw"
        ),
        "raw_pretrain_validation": SleepEDFSequenceDataset(
            config.data_cache_dir, "validation", 1, "raw"
        ),
        "wave_pretrain_train": SleepEDFSequenceDataset(
            config.data_cache_dir, "pretrain_train", 1, "wave"
        ),
        "wave_pretrain_validation": SleepEDFSequenceDataset(
            config.data_cache_dir, "validation", 1, "wave"
        ),
        "fusion_finetune_train": SleepEDFSequenceDataset(
            config.data_cache_dir, "train", config.sequence_length, "both"
        ),
        "fusion_finetune_validation": SleepEDFSequenceDataset(
            config.data_cache_dir, "validation", config.sequence_length, "both"
        )
    }

    batch_size = {
        "raw_pretrain_train": config.raw_batch_size,
        "raw_pretrain_validation": config.raw_batch_size,
        "wave_pretrain_train": config.wave_batch_size,
        "wave_pretrain_validation": config.wave_batch_size,
        "fusion_finetune_train": config.fusion_batch_size,
        "fusion_finetune_validation": config.fusion_batch_size,
    }

    loaders = {}
    for name, dataset in datasets.items():
        shuffle = name.endswith("_train")
        loaders[name] = build_reproducible_loader(
            dataset,
            batch_size=batch_size[name],
            shuffle=shuffle,
            seed=config.seed,
            num_workers=config.num_workers,
        )

    try:
        return run_two_stage_training(
            config,
            loaders,
            resume_checkpoints=resume_checkpoints,
        )
    finally:
        for dataset in datasets.values():
            dataset.close()
