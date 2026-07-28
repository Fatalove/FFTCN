# 里程碑 9B 内部步骤二：Dataset 与 DataLoader

步骤一已经生成并验证以下缓存契约：

```text
cache_root/<split>/
├── raw.npy
├── wave.npy
├── labels.npy
└── manifest.json
```

步骤二负责把这些磁盘缓存转换成模型可以按 batch 读取的样本。它仍属于里程碑 9B，不是新的子里程碑，也不单独进行正式验收。

正式练习：`training/sleep_edf_full_run.py`<br>
行为自测：`tests/test_sleep_edf_full_run.py`<br>
独立参考：`learning_guides/milestone_09b/reference_data_loading.py`

## 0. 本步骤学什么

1. `load_record_spans()`：从 manifest 恢复每条夜晚的全局边界；
2. `build_sequence_positions()`：只在单条记录内部建立非重叠序列；
3. `SleepEDFSequenceDataset`：用 memmap 切片构造 raw、wave 和标签张量；
4. `build_reproducible_loader()`：用局部随机生成器固定 shuffle 顺序。

`FullTrainingConfig.to_metadata()` 已经完成，本步骤不重写它。

## 1. 问题分析

### 1.1 为什么不能直接把整个缓存当成连续序列

`raw.npy`、`wave.npy` 和 `labels.npy` 按记录顺序拼在同一个全局样本轴上：

```text
record_a 的 epoch | record_b 的 epoch | record_c 的 epoch | ...
```

数组在磁盘上连续，不代表不同夜晚在生理时间上连续。如果直接每 50 个 epoch 切一次，某个样本可能从 `record_a` 的末尾跨到 `record_b` 的开头。TCN 会把两个不同夜晚误当成一条连续睡眠序列。

因此，Dataset 不能只知道总样本数，还必须读取 manifest 中每条记录的 `[start, stop)`。

### 1.2 原仓库怎样构造序列

原仓库 `data/loader.py` 对每个记录分别执行：

```python
sample_n = len(label) // seq_len
data = data[: sample_n * seq_len].reshape(-1, seq_len, ...)
label = label[: sample_n * seq_len].reshape(-1, seq_len)
```

这表示：

- 每条记录单独计算能组成多少个完整序列；
- 序列互不重叠；
- 每条记录不足 `seq_len` 的尾部单独丢弃；
- 绝不把上一条记录的尾部和下一条记录的开头拼起来。

本步骤不把所有数据先复制进 Python 列表，而是先生成轻量的 `SequencePosition` 索引，真正读取样本时再从 memmap 切片。

### 1.3 实际训练使用的三种输入模式

两条分支预训练和融合微调需要不同输入：

| `input_mode` | `sequence_length` | Dataset 输入元组 | 用途 |
|---|---:|---|---|
| `raw` | 1 | `(raw,)` | 1D-CNN 预训练 |
| `wave` | 1 | `(wave,)` | 2D-CNN 预训练 |
| `both` | 50 | `(raw, wave)` | 完整 FFTCN 融合微调 |

输入始终返回元组，使后续训练代码可以统一写成 `model(*inputs)`，不必为单输入和双输入建立两套 batch 协议。

### 1.4 `T=1` 与 `T=50` 的形状差异

缓存切片最初都带序列轴：

```text
raw:    [T,1,L]
wave:   [T,1,F,T_wave]
labels: [T]
```

融合模型需要保留 `T=50`：

```text
一个 Dataset 样本     DataLoader 加 batch 轴
raw  [50,1,3000]  ->  [B,50,1,3000]
wave [50,1,30,60] ->  [B,50,1,30,60]
y    [50]         ->  [B,50]
```

但两个预训练分支的模型接口没有显式序列轴。`sequence_length=1` 时，Dataset 必须去掉仅用于分组的第 0 轴：

```text
一个 Dataset 样本  DataLoader 加 batch 轴
raw  [1,3000]   -> [B,1,3000]
wave [1,30,60]  -> [B,1,30,60]
y    标量        -> [B]
```

这里去掉的是“长度为 1 的睡眠序列轴”，不是 EEG 单通道轴。

### 1.5 为什么读取 memmap 后还要复制

`np.load(..., mmap_mode="r")` 只建立磁盘映射，不把整个数组装入内存。切片得到的数组仍可能引用只读 memmap。

本步骤使用：

```python
np.array(memmap_slice, dtype=..., copy=True)
```

它只复制当前样本，并完成模型边界的 dtype 转换：

- raw：存储和模型计算均为 `float32`；
- wave：磁盘为 `float16`，进入模型前转为 `float32`；
- labels：转为 `int64`，对应交叉熵要求的类别索引 `torch.long`。

### 1.6 为什么使用局部随机生成器

直接调用 `torch.manual_seed(seed)` 会改动整个进程的全局随机状态，可能影响模型初始化、dropout 或其他实验。

本步骤只为当前 DataLoader 创建局部生成器：

```python
generator = torch.Generator()
generator.manual_seed(seed)
```

相同 Dataset、相同 seed 和相同 Loader 参数会得到相同 shuffle 顺序，同时不重置项目的其他随机过程。

## 2. 必要 Python、NumPy 与 PyTorch 基础

### 2.1 dataclass 作为具名边界

```python
@dataclass(frozen=True)
class RecordSpan:
    record_id: str
    start: int
    stop: int
```

与三元组相比，`span.start` 和 `span.stop` 更清楚。`frozen=True` 表示边界建立后不应在训练期间被修改。

### 2.2 左闭右开区间

`[start, stop)` 包含 `start`，不包含 `stop`，长度为：

```python
record_length = stop - start
```

它与 NumPy 的 `array[start:stop]` 完全一致，也让下一条记录可以从上一条的 `stop` 开始。

### 2.3 整除截断

```python
usable_length = (record_length // sequence_length) * sequence_length
```

`//` 先计算完整序列数，再乘回序列长度。例如 `5 // 2 == 2`，因此只使用前 `4` 个 epoch，尾部 `1` 个单独丢弃。

### 2.4 Dataset 的重点接口

- `__init__()`：打开当前模式需要的缓存并建立序列位置；
- `__len__()`：告诉 DataLoader 有多少个可读取样本；
- `__getitem__(index)`：把某个位置转换成模型输入和标签；
- `position_at(index)`：保留样本所属记录和区间，供错误分析使用；
- `close()`：本项目额外提供，用于释放 Windows 仍持有的 memmap 文件句柄。

### 2.5 `torch.from_numpy`

`torch.from_numpy(array)` 从 NumPy 数组创建张量。因为本步骤已经通过 `np.array(..., copy=True)` 得到当前样本的独立数组，所以张量不会继续依赖只读磁盘映射。

## 3. 手工推演

假设 manifest 中有两条记录：

```text
record_a -> [0,5)
record_b -> [5,8)
```

当 `sequence_length=2`：

```text
record_a 长度 5：usable=4 -> [0,2), [2,4)，丢弃索引 4
record_b 长度 3：usable=2 -> [5,7)，丢弃索引 7
```

最终 positions 是：

```text
record_a [0,2)
record_a [2,4)
record_b [5,7)
```

注意不存在 `[4,6)`。虽然它在全局数组上是合法切片，却跨越了两条夜晚记录。

若读取第三个位置 `[5,7)`，并使用 `input_mode="both"`：

```text
raw[5:7]    -> [2,1,L]
wave[5:7]   -> [2,1,F,T_wave]
labels[5:7] -> [2]
```

三者使用同一个 `start:stop`，所以 EEG、CWT 和标签保持逐 epoch 对齐。

## 4. 带详细注释的完整核心代码

下面代码完整覆盖步骤二。`FullTrainingConfig` 是前序已经完成的代码，因此不在这里重复；请保留正式文件中的现有实现。

```python
from __future__ import annotations

# dataclass 把记录边界和序列位置表示为具名不可变对象。
from dataclasses import dataclass
# json 读取步骤一生成的 manifest；Path 统一字符串与路径对象。
import json
from pathlib import Path
# Literal 描述三种输入模式；Sequence 允许列表或元组形式的记录边界。
from typing import Literal, Sequence

# NumPy 负责 memmap 和当前样本复制；PyTorch 负责张量与批加载。
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# raw、wave、both 决定 Dataset 为下游模型准备一条还是两条输入分支。
InputMode = Literal["raw", "wave", "both"]


@dataclass(frozen=True)
class RecordSpan:
    """描述一条记录在 split 级拼接缓存中的左闭右开区间。"""

    # record_id 保存夜晚身份；序列索引依靠它解释样本来自哪条记录。
    record_id: str
    # start 是该记录第一个 epoch 的全局索引，属于当前记录。
    start: int
    # stop 是该记录最后一个 epoch 之后的位置，不属于当前记录。
    stop: int


@dataclass(frozen=True)
class SequencePosition:
    """描述一个 Dataset 样本来自哪条记录及其全局切片区间。"""

    # record_id 让预测或错误样本可以追溯到具体夜晚。
    record_id: str
    # start 和 stop 直接作为 raw/wave/labels 共用的左闭右开切片。
    start: int
    stop: int


def load_record_spans(manifest_path: str | Path) -> tuple[RecordSpan, ...]:
    """读取 manifest，返回按缓存顺序连续排列的记录边界。"""

    # Path 允许调用者传字符串或 Path；UTF-8 与步骤一的写入编码保持一致。
    path = Path(manifest_path)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    # records 的顺序就是 raw/wave/labels 在全局样本轴上的记录拼接顺序。
    rows = payload["records"]

    # 第一个区间应从 0 开始；之后每条记录应紧接上一条记录的 stop。
    expected_start = 0
    spans: list[RecordSpan] = []
    for row in rows:
        # JSON 数值转为 Python int，供切片、range 和 dataclass 稳定使用。
        start = int(row["start"])
        stop = int(row["stop"])
        record_id = str(row["record_id"])

        # 若 start 与 expected_start 不同，记录区间就存在重叠、空洞或乱序。
        # 继续训练会使全局缓存中的某些 epoch 被重复读取或无人归属。
        if start != expected_start:
            raise ValueError("manifest 中的记录区间必须从 0 开始并连续排列")
        # 把无结构 JSON 字典转换为具名、不可变的 RecordSpan。
        spans.append(RecordSpan(record_id=record_id, start=start, stop=stop))
        # 下一条记录必须从当前 stop 开始，维持全局样本轴连续覆盖。
        expected_start = stop

    # tuple 表明训练期间只读边界，不会动态增删记录。
    return tuple(spans)


def build_sequence_positions(
    record_spans: Sequence[RecordSpan],
    sequence_length: int,
) -> tuple[SequencePosition, ...]:
    """在每条记录内部建立非重叠序列，分别丢弃不足一段的尾部。"""

    # 0 或负数不能表示睡眠序列长度，并会让 range 的步长失去合法语义。
    if sequence_length <= 0:
        raise ValueError("sequence_length 必须为正整数")

    positions: list[SequencePosition] = []
    for span in record_spans:
        # 当前夜晚长度只由自身边界计算，不能借用下一条记录的 epoch 补尾部。
        record_length = span.stop - span.start
        # 先整除再乘回，只保留能组成完整非重叠序列的记录前缀。
        usable_length = (record_length // sequence_length) * sequence_length

        # local_start 是记录内部位置；步长等于序列长度，所以相邻样本不重叠。
        for local_start in range(0, usable_length, sequence_length):
            # memmap 使用全局样本轴，因此要加 span.start 转成全局切片索引。
            global_start = span.start + local_start
            global_stop = global_start + sequence_length
            positions.append(
                SequencePosition(
                    record_id=span.record_id,
                    start=global_start,
                    stop=global_stop,
                )
            )

    # 每个位置都完全位于一条记录内；每条记录不足 T 的尾部没有进入结果。
    return tuple(positions)


class SleepEDFSequenceDataset(Dataset):
    """从 split 级 memmap 缓存按序列位置返回模型输入和标签。"""

    def __init__(
        self,
        cache_root: str | Path,
        split: str,
        sequence_length: int,
        input_mode: InputMode,
    ) -> None:
        """打开所需缓存，核对共享样本轴，并建立不跨记录的索引。"""

        # Literal 只帮助静态检查；运行时仍要拒绝没有数据语义的其他字符串。
        if input_mode not in {"raw", "wave", "both"}:
            raise ValueError("input_mode 必须是 raw、wave 或 both")

        # 每个 split 使用独立目录，防止训练 Dataset 意外映射 validation/test 缓存。
        self.split_dir = Path(cache_root) / split
        self.input_mode = input_mode
        self.sequence_length = sequence_length

        # 三种模式都需要标签；mmap_mode='r' 只映射文件，不整体读入内存。
        self.labels = np.load(self.split_dir / "labels.npy", mmap_mode="r")

        # 只打开当前模型分支需要的缓存，避免无意义地占用文件句柄。
        self.raw = None
        if input_mode in {"raw", "both"}:
            self.raw = np.load(self.split_dir / "raw.npy", mmap_mode="r")
            # raw/labels 分别存盘；样本数不同会使同一个切片对应不同 epoch。
            if len(self.raw) != len(self.labels):
                raise ValueError("raw 与 labels 的样本轴长度必须一致")

        self.wave = None
        if input_mode in {"wave", "both"}:
            self.wave = np.load(self.split_dir / "wave.npy", mmap_mode="r")
            # wave 在 raw 之后单独生成；样本数不同通常表示仍残留旧缓存。
            if len(self.wave) != len(self.labels):
                raise ValueError("wave 与 labels 的样本轴长度必须一致")

        # manifest 恢复夜晚边界；序列索引只能在每个边界内部生成。
        self.record_spans = load_record_spans(self.split_dir / "manifest.json")
        # 最后一条记录的 stop 应等于 N，否则 manifest 没有完整覆盖缓存样本轴。
        if self.record_spans[-1].stop != len(self.labels):
            raise ValueError("manifest 记录总长度必须等于 labels 样本数")
        self.positions = build_sequence_positions(
            self.record_spans,
            sequence_length=sequence_length,
        )

    def __len__(self) -> int:
        """返回当前 split 中可以读取的完整单时段或序列数量。"""

        # positions 中的一项就是 __getitem__ 能返回的一个 Dataset 样本。
        return len(self.positions)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """按同一位置切 raw/wave/labels，并转换成模型所需张量。"""

        # position 同时决定三种缓存的切片，保持第 i 个 EEG、CWT 和标签对齐。
        position = self.positions[index]
        start, stop = position.start, position.stop

        # 当前标签切片最初是 [T]；复制后摆脱只读 memmap。
        # int64 对应交叉熵要求的类别索引 torch.long。
        label_array = np.array(self.labels[start:stop], dtype=np.int64, copy=True)
        targets = torch.from_numpy(label_array)

        # inputs 按模型参数顺序收集：raw 在前，wave 在后。
        inputs: list[torch.Tensor] = []
        if self.raw is not None:
            # raw 切片保持 [T,1,L]，并统一为模型计算使用的 float32。
            raw_array = np.array(self.raw[start:stop], dtype=np.float32, copy=True)
            inputs.append(torch.from_numpy(raw_array))
        if self.wave is not None:
            # wave 存盘为 float16；这里只复制当前 [T,1,F,T_wave] 并恢复 float32。
            wave_array = np.array(self.wave[start:stop], dtype=np.float32, copy=True)
            inputs.append(torch.from_numpy(wave_array))

        if self.sequence_length == 1:
            # T=1 只用于单分支预训练；去掉唯一输入的序列轴，但保留输入列表。
            # raw [1,1,L] -> [1,L]；wave [1,1,F,T_wave] -> [1,F,T_wave]。
            inputs[0] = inputs[0][0]
            # 标签也从 [1] 变成标量；DataLoader 汇总后形成 [B]。
            targets = targets[0]

        # raw -> (raw,)，wave -> (wave,)，both -> (raw,wave)；
        # 下游训练循环可以统一调用 model(*inputs)，targets 用于损失和指标。
        return tuple(inputs), targets

    def position_at(self, index: int) -> SequencePosition:
        """返回指定样本的记录身份和区间，供按记录评估与错误分析使用。"""

        # 直接返回 __getitem__ 使用的同一索引对象，避免另算一套边界。
        return self.positions[index]

    def close(self) -> None:
        """关闭已打开的 memmap，释放 Windows 持有的缓存文件句柄。"""

        # 不同 input_mode 会使 raw 或 wave 为 None；统一循环可避免遗漏。
        for array in (self.labels, self.raw, self.wave):
            # 只关闭当前模式实际打开并仍持有底层映射句柄的数组。
            if isinstance(array, np.memmap) and array._mmap is not None:
                array._mmap.close()


def build_reproducible_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
) -> DataLoader:
    """创建由局部随机生成器控制 shuffle 顺序的 DataLoader。"""

    # 局部 Generator 只控制这个 Loader 的采样顺序，不重置全局 torch RNG。
    generator = torch.Generator()
    generator.manual_seed(seed)

    # drop_last=False 保留所有已构造的完整序列；最后一个 batch 可以小于 batch_size。
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        drop_last=False,
    )
```

## 5. 用户编码范围与统一自测

请在 `training/sleep_edf_full_run.py` 中保留已经完成的 `FullTrainingConfig.to_metadata()`，一次完成本步骤剩余接口：

1. 补完已经开始的 `load_record_spans()`；
2. 实现 `build_sequence_positions()`；
3. 实现 `SleepEDFSequenceDataset` 的初始化、长度、取样、位置查询和关闭；
4. 实现 `build_reproducible_loader()`。

不要逐函数提交检查。全部完成后统一运行：

```powershell
python -m unittest tests.test_sleep_edf_full_run -v
```

5 条测试分别覆盖：

- 已完成配置可序列化；
- manifest 与序列索引不跨记录；
- `T=1` 的 raw/wave 分支形状和 dtype；
- `T>1` 时每记录单独丢尾部；
- 相同 seed 的 DataLoader shuffle 顺序一致。

测试全部通过只代表步骤二行为自测完成，不代表整个里程碑 9B 已验收。

## 6. 常见错误与测试含义

- 直接对全局 `N` 每 50 个切片：会把不同夜晚拼成同一 TCN 序列；
- 使用滑动窗口步长 1：会偏离原仓库的非重叠 `reshape` 语义并改变样本数量；
- 把记录尾部留给下一条记录补齐：同样会跨夜晚；
- `T=1` 时删除两个轴：会把 raw 从 `[1,1,3000]` 错变成 `[3000]`，丢掉 EEG 通道轴；
- wave 保持 float16 进入模型：可能改变当前明文训练的数值语义；
- raw、wave 和 labels 使用不同切片：三种数据会发生样本错位；
- 返回裸张量而不是输入元组：后续 `model(*inputs)` 无法统一单分支与双分支；
- 使用全局 `torch.manual_seed()`：会无意改变模型初始化和 dropout 等其他随机过程；
- 不关闭 memmap：Windows 临时测试目录或缓存重建时可能无法删除文件。
