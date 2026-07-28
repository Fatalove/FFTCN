# 里程碑 3：ACM 式完整讲解

目标文件：`data/sleep_edf_preprocessor.py`

本教程采用新的固定顺序：

1. 读题并分析题意；
2. 学习必要的 Python/NumPy 基础；
3. 手工推演小样本；
4. 直接阅读带详细注释的完整代码；
5. 运行聚焦测试并理解错误。

完整代码不会自动覆盖练习文件。建议看完一个函数后先遮住代码，用自己的话复述“输入如何一步步变成输出”，再决定是手敲、模仿还是独立重写。

---

## 第一题：`preprocess_record`

### 1. 读题：这个函数到底做什么

输入 `RawSleepEDFRecord`：

```text
eeg:    float32 [N,1,3000]
labels: 字符串 [N]
```

输出 `ProcessedSleepEDFRecord`：

```text
eeg:    float32 [M,1,3000]
labels: int64 [M]，只允许 0..4
```

处理顺序不能随意改变：

1. 检查 EEG 和标签原本是否一一对应；
2. 删除 Movement/Unknown，并同步删除同位置 EEG；
3. 把剩余标签映射成五分类；
4. 找到首末非 W，只保留两侧最多 60 个 W；
5. 再次检查 shape、dtype 和有限值。

映射规则：

| 原始标签 | 输出 |
|---|---:|
| Sleep stage W | 0 |
| Sleep stage 1 | 1 |
| Sleep stage 2 | 2 |
| Sleep stage 3 / 4 | 3 |
| Sleep stage R | 4 |
| Movement time / Sleep stage ? | 删除 |

### 2. 必要的 Python/NumPy 基础

#### 布尔 mask

```python
values = np.array([10, 20, 30])
mask = np.array([True, False, True])
print(values[mask])  # [10, 30]
```

本题最重要的同步原则：

```python
filtered_eeg = eeg[valid_mask]
filtered_labels = labels[valid_mask]
```

EEG 和标签必须使用同一个 mask。

#### 字典映射

```python
number = LABEL_MAP["Sleep stage 2"]  # 2
```

#### 找非零位置

```python
non_wake = np.flatnonzero(mapped_labels != 0)
```

#### Python 切片右端不包含

```python
values[2:5]  # 包含下标 2、3、4，不包含 5
```

因此要包含最后一个非 W，还需要额外 `+1`。

### 3. 手工推演

测试数据：

```text
70 W
N1, N2, S3, S4, REM, Movement, Unknown
70 W
```

删除两个无效时段后：

```text
70 W + 5 个睡眠时段 + 70 W
```

两端最多各保留 60 个 W：

```text
60 + 5 + 60 = 125 段
```

测试把每段 EEG 填成自己的原始序号。正确输出第一段是 10，最后一段是 136；否则说明 EEG 和标签没有同步切片。

### 4. 带注释的完整代码

```python
def preprocess_record(record, max_wake_epochs=60):
    # 只有 LABEL_MAP 中的标签有效；EEG 和标签共用同一个 mask。
    descriptions = np.asarray([str(label) for label in record.labels])
    valid_mask = np.asarray(
        [description in LABEL_MAP for description in descriptions],
        dtype=bool,
    )
    filtered_eeg = record.eeg[valid_mask]
    filtered_descriptions = descriptions[valid_mask]

    # 字符串映射为 0..4；S3/S4 都由 LABEL_MAP 映射为 3。
    mapped_labels = np.asarray(
        [LABEL_MAP[description] for description in filtered_descriptions],
        dtype=np.int64,
    )

    # 找首末非 W。全 W 时无法定义睡眠主体，所以必须单独处理。
    non_wake = np.flatnonzero(mapped_labels != 0)
    if len(non_wake) == 0:
        raise ValueError("record contains no sleep epochs")

    # 两端最多各保留 max_wake_epochs 个 W；end +1 是切片规则所需。
    begin = max(0, int(non_wake[0]) - max_wake_epochs)
    end = min(len(mapped_labels), int(non_wake[-1]) + max_wake_epochs + 1)

    # EEG 和标签继续使用同一个范围切片。
    processed_eeg = np.ascontiguousarray(filtered_eeg[begin:end], dtype=np.float32)
    processed_labels = np.ascontiguousarray(mapped_labels[begin:end], dtype=np.int64)

    return ProcessedSleepEDFRecord(
        paths=record.paths,
        sampling_rate=record.sampling_rate,
        eeg=processed_eeg,
        labels=processed_labels,
    )
```

相同的最小准确实现见 `reference_solution.py`。额外 shape、采样率、NaN 和未知标签检查放在文末“工程加固（选读）”。

### 5. 聚焦测试与错误解释

```powershell
python -m unittest `
  tests.test_sleep_edf_preprocessor.SleepEDFPreprocessorTest.test_mapping_removal_and_wake_trim_stay_synchronized -v
```

再测真实记录：

```powershell
python -m unittest `
  tests.test_sleep_edf_preprocessor.SleepEDFPreprocessorTest.test_sc4001_reference_counts -v
```

常见错误：

- 输出不是 125 段：检查右端 `+1` 和两端 60 W；
- 第一段不是 10、最后不是 136：EEG 和标签没有共用同一 mask/切片；
- S4 输出为 4：S3/S4 应合并为 N3=3；
- `IndexError`：全 W 时没有先判断 `len(non_wake)`；
- `TypeError: ... not 'generator'`：`np.asarray` 不会自动展开生成器；改用 `[... for ...]` 列表推导式，或使用 `np.fromiter`。

---

## 第二题：`split_subjects`

### 1. 读题

153 条记录来自 78 名受试者。多数受试者有两个夜晚，同一个人的所有夜晚必须进入同一集合。

固定结果：train=62 人、validation=8 人、test=8 人。

### 2. 必要的 Python 基础

集合去重并排序：

```python
subjects = sorted({record.subject_id for record in records})
```

局部随机源：

```python
rng = random.Random(seed)
rng.shuffle(subjects)
```

局部 `Random` 不会修改程序其他部分的全局随机状态。

### 3. 手工推演

不要对 153 个文件直接洗牌。先把文件变成 78 个唯一 subject ID，再洗牌和切分。

```text
round(78×0.8) = 62
round(78×0.1) = 8
78-62-8        = 8
```

### 4. 带注释的完整代码

```python
def split_subjects(records, seed=DEFAULT_SPLIT_SEED):
    # subject_id 去重；先排序消除文件枚举顺序的影响。
    subjects = sorted({record.subject_id for record in records})

    # 使用局部随机源，保证可重复且不污染全局随机状态。
    rng = random.Random(seed)
    rng.shuffle(subjects)

    # round 使 78 人得到 62/8，测试集接收全部剩余人员。
    train_count = round(len(subjects) * 0.8)
    validation_count = round(len(subjects) * 0.1)

    return SubjectSplit(
        train=tuple(subjects[:train_count]),
        validation=tuple(subjects[train_count:train_count + validation_count]),
        test=tuple(subjects[train_count + validation_count:]),
    )
```

### 5. 聚焦测试与错误解释

```powershell
python -m unittest `
  tests.test_sleep_edf_preprocessor.SleepEDFPreprocessorTest.test_subject_split_is_reproducible_and_disjoint -v
```

常见错误：

- 数量变成记录数：你使用了 `record_id` 而不是 `subject_id`；
- 同一 seed 结果变化：去重后没有先排序；
- 两晚进入不同集合：你先切文件，再提取 subject ID；
- 其他代码随机结果变化：你使用了全局 `random.seed()`。

---

## 第三题：`save_processed_record`

### 1. 读题

原 `Sleep_Loader` 只接受：

```text
文件名: record_id.npz
x: float32 [N,1,3000]
y: int64 [N]
```

### 2. 必要的 Python/NumPy 基础

```python
output_dir.mkdir(parents=True, exist_ok=True)
np.savez(output_path, x=eeg, y=labels)
```

- `parents=True`：缺少的父目录一起创建；
- `exist_ok=True`：目录已存在也允许重跑；
- `x=` 和 `y=` 会成为 NPZ 内部键名。

读取 NPZ 时使用：

```python
with np.load(output_path) as saved:
    print(saved["x"].shape)
```

Windows 不允许删除仍被打开的文件；忘记关闭 `np.load()` 句柄可能触发 WinError 32。

### 3. 手工推演

假设 `record_id="SC4001"`，输出目录为 `processed/`：

```text
processed/SC4001.npz
├─ x -> EEG 数组
└─ y -> 标签数组
```

如果键名写成 `eeg/labels`，文件虽然能保存，但原 Loader 会报找不到 `x/y`。

### 4. 带注释的完整代码

```python
def save_processed_record(record, output_dir):
    # 创建目录，并用唯一 record_id 避免两个夜晚互相覆盖。
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{record.paths.record_id}.npz"

    # 原 Loader 固定读取键 x/y。
    np.savez(output_path, x=record.eeg, y=record.labels)
    return output_path
```

### 5. 聚焦测试与错误解释

```powershell
python -m unittest `
  tests.test_sleep_edf_preprocessor.SleepEDFPreprocessorTest.test_saved_npz_matches_original_loader_contract -v
```

常见错误：

- `FileNotFoundError`：没有先 `mkdir`；
- Loader 找不到数组：键名不是 `x/y`；
- dtype 测试失败：保存前没有固定 `float32/int64`；
- 临时目录无法删除：`np.load()` 文件句柄没有关闭；
- 两个夜晚只剩一个文件：文件名误用了 `subject_id`，应使用唯一的 `record_id`。

---

## 工程加固（选读）

先掌握上面的核心代码，再考虑以下生产环境检查：

- 验证输入采样率确实为 100 Hz；
- 验证 EEG shape 为 `[N,1,3000]` 且标签长度为 N；
- 对 `LABEL_MAP` 和 `INVALID_LABELS` 之外的新标签显式报错；
- 检查 EEG 是否包含 NaN/Inf；
- 小数据集切分时检查是否产生空集合；
- 保存前再次验证 dtype、shape 和 EEG/标签数量。

这些检查能提高工程可靠性，但不是理解本里程碑三条核心算法链路的前提，因此不放进主参考答案。

---

## 最终联调

```powershell
python `
  -m unittest tests.test_sleep_edf_preprocessor -v
```

预期 4 个测试全部通过。

推荐学习动作：

1. 看完一个完整函数；
2. 遮住代码，口头复述数据流；
3. 在练习文件中手敲或模仿实现；
4. 运行该函数的聚焦测试；
5. 对照 traceback 和参考代码解释差异；
6. 最后再运行全部测试。

## 学习记录模板

```text
我最初不理解的代码：
这段代码的输入：
这段代码的输出：
它为什么不能省略：
如果写错，哪个测试会失败：
以后遇到相似问题，我会先检查：
```
