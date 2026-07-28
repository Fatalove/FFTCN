"""里程碑 9B 内部步骤二参考答案：Dataset/DataLoader 与配置契约。"""

from __future__ import annotations

# dataclass 定义不可变实验契约；asdict 将配置复制为可序列化字典的起点。
from dataclasses import asdict, dataclass
# json 读取缓存 manifest；Path 统一 Windows/Linux 路径并生成绝对实验路径。
import json
from pathlib import Path
# Any 描述元数据值；Literal 限定输入模式；Sequence 接受列表或元组记录边界。
from typing import Any, Literal, Sequence

# NumPy 读取内存映射缓存；PyTorch 负责张量和可复现 DataLoader。
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# 三种取值直接决定 Dataset 为模型准备哪一组输入张量。
InputMode = Literal["raw", "wave", "both"]


@dataclass(frozen=True)
class FullTrainingConfig:
    """保存重构模型一次完整训练所需的最小可复现实验配置。"""

    # 两个路径分别指向训练缓存根目录和本次实验输出目录。
    data_cache_dir: Path
    output_dir: Path
    # seed 控制 DataLoader shuffle 等随机过程；offset_samples 固化步骤三逐记录
    # 偏移过采样的最大采样点位移；序列长度 50 对应融合 TCN 输入。
    seed: int = 0
    offset_samples: int = 300
    sequence_length: int = 50
    # 三个轮数对应 raw 预训练、wave 预训练和融合微调，默认保持仓库工程配置。
    raw_pretrain_epochs: int = 20
    wave_pretrain_epochs: int = 20
    fusion_finetune_epochs: int = 50
    # 三阶段的显存占用不同，所以分别记录 batch size，不能只保存一个公共值。
    raw_batch_size: int = 128
    wave_batch_size: int = 128
    fusion_batch_size: int = 32
    # 两个分支预训练共享基础学习率；融合阶段另存基础学习率和分支缩放比例。
    pretrain_learning_rate: float = 1e-5
    fusion_learning_rate: float = 1e-5
    feature_learning_rate_scale: float = 1e-2
    # scheduler_gamma 是每轮指数衰减系数；num_workers/device 记录运行环境入口。
    scheduler_gamma: float = 0.95
    num_workers: int = 0
    device: str = "auto"

    def __post_init__(self) -> None:
        """拒绝无法形成有效训练循环的非正长度、轮数或 batch size。"""

        # 把所有必须大于 0 的整数字段放入同一张表，避免为每个字段重复写判断。
        positive_integers = {
            "sequence_length": self.sequence_length,
            "raw_pretrain_epochs": self.raw_pretrain_epochs,
            "wave_pretrain_epochs": self.wave_pretrain_epochs,
            "fusion_finetune_epochs": self.fusion_finetune_epochs,
            "raw_batch_size": self.raw_batch_size,
            "wave_batch_size": self.wave_batch_size,
            "fusion_batch_size": self.fusion_batch_size,
        }
        # value <= 0 的字段无法产生有效序列、epoch 循环或 batch，因此集中报告名称。
        invalid = [name for name, value in positive_integers.items() if value <= 0]
        if invalid:
            raise ValueError(f"以下配置必须为正整数：{invalid}")
        # PyTorch 约定 num_workers=0 表示主进程加载，负数没有合法含义。
        if self.num_workers < 0:
            raise ValueError("num_workers 不能为负数")

    def to_metadata(self) -> dict[str, Any]:
        """返回可直接写入 JSON 或 checkpoint 的普通 Python 字典。"""

        # asdict 把不可变 dataclass 复制为普通字典，供后续序列化和 checkpoint 使用。
        metadata = asdict(self)
        # JSON 不认识 Path 对象；resolve 先固定绝对位置，再转成普通字符串保存。
        metadata["data_cache_dir"] = str(self.data_cache_dir.resolve())
        metadata["output_dir"] = str(self.output_dir.resolve())
        # 其余字段已经是 int、float 或 str，可直接由 json.dump/torch.save 处理。
        return metadata


@dataclass(frozen=True)
class RecordSpan:
    """描述一条记录在 split 级拼接数组中的左闭右开区间。"""

    record_id: str  # 例如 SC4001，用于防止序列跨过夜晚边界。
    start: int  # 该记录第一个 epoch 在拼接数组中的全局索引。
    stop: int  # 该记录最后一个 epoch 之后的位置，不属于该记录。


@dataclass(frozen=True)
class SequencePosition:
    """描述一个 Dataset 样本来自哪条记录及其全局左闭右开区间。"""

    # record_id 让预测或错误样本可以追溯到具体夜晚。
    record_id: str
    # start 和 stop 直接作为 raw/wave/labels 共用的左闭右开切片。
    start: int
    stop: int


def load_record_spans(manifest_path: str | Path) -> tuple[RecordSpan, ...]:
    """读取 manifest，并返回按时间顺序连续覆盖缓存数组的记录边界。"""

    # Path 统一字符串/Path 输入；UTF-8 允许 manifest 保存中文说明而不依赖系统编码。
    path = Path(manifest_path)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    # records 的每一项都描述一条独立夜晚，列表顺序就是拼接缓存中的时间块顺序。
    rows = payload["records"]

    # expected_start 表示下一条记录必须从哪里开始；首条记录应覆盖全局索引 0。
    expected_start = 0
    spans: list[RecordSpan] = []
    for row in rows:
        # JSON 数值显式转 int，保证后续 range 和 NumPy 切片接收普通整数。
        start = int(row["start"])
        stop = int(row["stop"])
        record_id = str(row["record_id"])

        # start 必须紧接上一条 stop，才能证明缓存没有重叠、空洞或顺序错乱。
        if start != expected_start:
            raise ValueError("manifest 中的记录区间必须从 0 开始并连续排列")
        # RecordSpan 将无结构字典变成具名、不可变的记录边界对象。
        spans.append(RecordSpan(record_id=record_id, start=start, stop=stop))
        # 下一条记录必须从当前 stop 开始，保证两条记录不会重叠或留出未知区间。
        expected_start = stop

    # tuple 表示训练期间不会增删记录边界；下游只读这些区间构造序列。
    return tuple(spans)


def build_sequence_positions(
    record_spans: Sequence[RecordSpan],
    sequence_length: int,
) -> tuple[SequencePosition, ...]:
    """在每条记录内部构造非重叠序列，丢弃不足一个完整序列的尾部。"""

    # 长度决定每个 Dataset 样本包含多少连续 epoch；0 或负数无法形成切片。
    if sequence_length <= 0:
        raise ValueError("sequence_length 必须为正整数")

    positions: list[SequencePosition] = []
    for span in record_spans:
        # 当前记录的 epoch 数只由本记录 stop-start 得到，不能与下一条记录合并。
        record_length = span.stop - span.start
        # 整除后再乘回序列长度，只保留能组成完整非重叠序列的前缀。
        usable_length = (record_length // sequence_length) * sequence_length

        # local_start 是记录内部索引，以 sequence_length 为步长避免序列重叠。
        for local_start in range(0, usable_length, sequence_length):
            # Dataset 最终切 split 级数组，所以要加 span.start 转成全局索引。
            global_start = span.start + local_start
            global_stop = global_start + sequence_length
            positions.append(
                SequencePosition(
                    record_id=span.record_id,
                    start=global_start,
                    stop=global_stop,
                )
            )

    # 每个位置都完全落在某条记录内部；不足 T 的尾部没有进入返回结果。
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

        # input_mode 决定模型需要 raw、wave 还是两者；其他字符串没有明确数据语义。
        if input_mode not in {"raw", "wave", "both"}:
            raise ValueError("input_mode 必须是 raw、wave 或 both")

        # 每个 split 有独立目录，防止训练 Dataset 意外读取 validation/test 数组。
        self.split_dir = Path(cache_root) / split
        self.input_mode = input_mode
        self.sequence_length = sequence_length

        # 标签对三种输入都必需；mmap_mode='r' 只映射文件，不一次读入全部内存。
        self.labels = np.load(self.split_dir / "labels.npy", mmap_mode="r")

        # raw/wave 只在当前模式需要时打开，避免 wave 预训练无意义地映射 raw 文件。
        self.raw = None
        if input_mode in {"raw", "both"}:
            self.raw = np.load(self.split_dir / "raw.npy", mmap_mode="r")
            # raw 和 labels 必须共享 epoch 样本轴，否则相同切片会取到不同样本。
            if len(self.raw) != len(self.labels):
                raise ValueError("raw 与 labels 的样本轴长度必须一致")

        self.wave = None
        if input_mode in {"wave", "both"}:
            self.wave = np.load(self.split_dir / "wave.npy", mmap_mode="r")
            # wave 单独生成；长度不一致通常表示它没有随 raw/labels 一起重建。
            if len(self.wave) != len(self.labels):
                raise ValueError("wave 与 labels 的样本轴长度必须一致")

        # manifest 保留记录边界；序列索引只能由这些边界分别生成。
        self.record_spans = load_record_spans(self.split_dir / "manifest.json")
        # 最后一条 stop 应等于缓存 N，证明 manifest 完整覆盖标签样本轴。
        if self.record_spans[-1].stop != len(self.labels):
            raise ValueError("manifest 记录总长度必须等于 labels 样本数")
        self.positions = build_sequence_positions(
            self.record_spans,
            sequence_length=sequence_length,
        )

    def __len__(self) -> int:
        """返回当前 split 中完整单时段或完整序列的数量。"""

        # positions 中一项就是 __getitem__ 可读取的一个训练/评估样本。
        return len(self.positions)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """返回模型输入元组和对应标签，且将缓存数值复制为可训练张量。"""

        # 先取得同一条记录内部的 [start,stop)，raw/wave/label 共用完全相同切片。
        position = self.positions[index]
        start, stop = position.start, position.stop

        # np.array(...,copy=True) 把只读 memmap 切片复制成可写连续数组；
        # dtype=int64 对应 PyTorch 交叉熵要求的类别索引类型 torch.long。
        label_array = np.array(self.labels[start:stop], dtype=np.int64, copy=True)
        targets = torch.from_numpy(label_array)

        # inputs 按模型参数顺序收集：raw 在前，wave 在后。
        inputs: list[torch.Tensor] = []
        if self.raw is not None:
            # raw 数值统一为 float32；形状从缓存 [T,1,L] 原样进入张量。
            raw_array = np.array(self.raw[start:stop], dtype=np.float32, copy=True)
            inputs.append(torch.from_numpy(raw_array))
        if self.wave is not None:
            # wave 即使以 float16 节省磁盘，也在进入模型前恢复 float32 计算类型。
            wave_array = np.array(self.wave[start:stop], dtype=np.float32, copy=True)
            inputs.append(torch.from_numpy(wave_array))

        if self.sequence_length == 1:
            # T=1 只用于单分支预训练；去掉唯一输入的序列轴但保留输入列表。
            # DataLoader 随后添加 B 轴，得到 [B,1,L] 或 [B,1,F,T_wave]。
            inputs[0] = inputs[0][0]
            # 同步去掉标签的 T=1 轴，使 DataLoader 汇总后得到 [B] 而不是 [B,1]。
            targets = targets[0]

        # 输入始终使用元组：raw -> (raw,)，wave -> (wave,)，both -> (raw,wave)。
        # 下游可统一执行 model(*inputs)，targets 则用于交叉熵和评估。
        return tuple(inputs), targets

    def position_at(self, index: int) -> SequencePosition:
        """返回指定样本的记录身份与区间，供错误分析和按记录评估使用。"""

        # 不重新计算边界，直接返回与 __getitem__ 使用的同一不可变索引对象。
        return self.positions[index]

    def close(self) -> None:
        """关闭 NumPy 内存映射文件，释放 Windows 持有的缓存文件句柄。"""

        # 三个属性都可能是 np.memmap 或 None；用同一循环避免遗漏某种 input_mode。
        for array in (self.labels, self.raw, self.wave):
            # 只有当前模式实际打开的 memmap 才具有需要关闭的底层 _mmap 句柄。
            if isinstance(array, np.memmap) and array._mmap is not None:
                array._mmap.close()


def build_reproducible_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
) -> DataLoader:
    """创建只由局部 generator 控制 shuffle 顺序的 DataLoader。"""

    # 局部 Generator 只控制这个 Loader 的采样顺序，不重置项目的全局 RNG。
    generator = torch.Generator()
    generator.manual_seed(seed)

    # drop_last=False 保留验证/测试的所有完整序列；预训练也不丢最后一个小 batch。
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        drop_last=False,
    )
