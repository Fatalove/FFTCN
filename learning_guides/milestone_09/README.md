# 里程碑 9：工程评估与可复核报告

当前状态：已完成。用户实现通过 7/7 聚焦测试和全项目 39/39 回归；另外使用里程碑 1B 原仓库完整训练得到的 best checkpoint 做了评估接口兼容性复核。这里没有重新训练里程碑 2--9 的教学重构模型，也不产生一组新的模型性能指标。

正式练习文件：`evaluation/sleep_edf_metrics.py`<br>
行为测试：`tests/test_sleep_edf_metrics.py`<br>
独立参考答案：`learning_guides/milestone_09/reference_solution.py`

## 0. 本里程碑的边界

本阶段实现“如何在固定测试集上正确评估一个给定模型并保存报告”，不重新训练模型，也不把单次工程结果包装成论文复现结论。本阶段展示的 79.18% Accuracy 来自里程碑 1B 已有 checkpoint；它只用于验证新旧评估实现对同一组预测的计算一致性，不代表教学重构模型已经获得该性能。

输入输出主线：

```text
测试 batch
    -> model.eval() + no_grad()
    -> logits [...,5]
    -> argmax 得到预测标签
    -> 汇总全部真实/预测标签
    -> 混淆矩阵
    -> ACC、每类 Precision/Recall/F1、Macro-F1、kappa
    -> 携带环境、划分、种子和 checkpoint 的 JSON 报告
```

正式报告必须保留下面的结论边界：

> 这是 Sleep-EDF-153 固定受试者划分上的单数据集工程评估结果，不等同于论文十折交叉验证或三数据集完整复现。

## 1. 问题分析

### 1.1 为什么不能只看 Accuracy

假设测试集有 9 个 W、1 个 N1，模型把所有样本都预测成 W：

```text
Accuracy = 9/10 = 90%
N1 F1    = 0
Macro-F1 ≈ 47.37%
```

90% 看起来很高，但模型完全不会识别 N1。Macro-F1 先分别计算每一类 F1，再让每类以相同权重参与平均，因此更容易暴露少数类失败。

### 1.2 混淆矩阵的两个轴

本项目固定：

```text
行：真实类别
列：预测类别
```

因此 `cm[1,2]` 表示“真实 N1 被预测为 N2 的数量”。如果交换两个轴，Precision 和 Recall 也会被交换。

对于类别 `c`：

```text
TP_c = cm[c,c]
真实数量 = 第 c 行之和 = TP_c + FN_c
预测数量 = 第 c 列之和 = TP_c + FP_c
```

### 1.3 Precision、Recall 和 F1

```text
Precision_c = TP_c / (TP_c + FP_c)
Recall_c    = TP_c / (TP_c + FN_c)
F1_c        = 2 * Precision_c * Recall_c / (Precision_c + Recall_c)
```

- Precision 回答：预测成该类的样本里，有多少是真的？
- Recall 回答：真实属于该类的样本里，有多少被找到了？
- F1 同时惩罚低 Precision 和低 Recall。

如果模型从未预测某一类，该类 Precision 的分母为 0。本教学契约把该类 Precision 和 F1 记为 0，避免原仓库直接相除产生 NaN。

### 1.4 Cohen's kappa

Accuracy 只计算观察一致率 `p0`。kappa 还减去由真实/预测类别边际分布产生的随机期望一致率 `pe`：

```text
kappa = (p0 - pe) / (1 - pe)
```

当全部真实标签和预测标签都只有同一个类别时，`pe=1`，分母为 0。此时 kappa 数学上不可定义，报告使用 JSON 的 `null`，而不是伪造 0 或 1。

### 1.5 为什么指标必须和元数据一起保存

相同的 79% Accuracy 可能来自完全不同的实验：

- 文件级随机划分或受试者级无泄漏划分；
- best checkpoint 或最后一轮 checkpoint；
- 不同随机种子；
- CPU 或不同 CUDA/PyTorch 环境；
- 单数据集固定划分或十折交叉验证。

脱离这些上下文，指标不能被可靠复核。因此报告必须记录数据集、split、划分策略、seed、checkpoint、Python、PyTorch 和设备。

## 2. 必要 Python、NumPy 与 PyTorch 基础

### 2.1 用一维编号统计二维混淆矩阵

类别数为 `C` 时，将二维坐标编码为：

```python
encoded = y_true * C + y_pred
```

例如 `C=3`，真实类别 1、预测类别 2：

```text
encoded = 1*3+2 = 5
```

使用：

```python
np.bincount(encoded, minlength=C**2).reshape(C, C)
```

即可统计所有“真实类别—预测类别”组合。

### 2.2 `axis=0` 与 `axis=1`

混淆矩阵形状为 `[真实类别,预测类别]`：

```python
support = cm.sum(axis=1)          # 压缩列，得到每个真实类别数量
predicted_count = cm.sum(axis=0)  # 压缩行，得到每个预测类别数量
```

这里不要只背“0 是列、1 是行”。更可靠的理解是：`sum(axis=k)` 会消掉第 `k` 个轴。

### 2.3 安全除法

```python
precision = np.divide(
    tp,
    predicted_count,
    out=np.zeros(C, dtype=np.float64),
    where=predicted_count != 0,
)
```

- `where` 指定哪些位置允许相除；
- `out` 提供其他位置的默认结果；
- 因而“从未预测该类”得到 0，而不是 NaN。

### 2.4 序列 logits 的展平

FFTCN 教学接口输出：

```text
logits:  [B,T,5]
targets: [B,T]
```

先在最后一个类别轴取最大值：

```python
predictions = torch.argmax(logits, dim=-1)  # [B,T]
```

然后同步展平：

```text
[B,T] -> [B*T]
```

TCN 已经在 logits 中融合了上下文；这里展平只是汇总预测位置，不会删除已经计算好的上下文信息。

### 2.5 JSON 可保存类型

`json.dump` 不认识 NumPy 数组和 NumPy 标量，所以报告中需要转换：

```python
cm.tolist()
float(metric)
int(support)
```

## 3. 手工推演

使用三个类别：

```python
y_true = [0, 0, 0, 1, 1, 2]
y_pred = [0, 0, 1, 1, 0, 0]
```

逐个放入混淆矩阵：

```text
             预测 0  预测 1  预测 2
真实 0          2       1       0
真实 1          1       1       0
真实 2          1       0       0
```

总共 6 个位置，对角线正确数为 `2+1+0=3`：

```text
Accuracy = 3/6 = 0.5
```

类别 0：

```text
Precision_0 = 2/(2+1+1) = 1/2
Recall_0    = 2/(2+1)   = 2/3
F1_0        = 4/7
```

类别 1：

```text
Precision_1 = 1/2
Recall_1    = 1/2
F1_1        = 1/2
```

类别 2 从未被正确预测，三项指标均为 0：

```text
Macro-F1 = (4/7 + 1/2 + 0) / 3 = 5/14 ≈ 0.3571
```

真实边际为 `[3,2,1]`，预测边际为 `[4,2,0]`：

```text
pe    = (3*4 + 2*2 + 1*0) / 6² = 4/9
kappa = (1/2 - 4/9) / (1 - 4/9) = 0.1
```

## 4. 带详细注释的完整核心代码

下面是完整核心实现。请先理解前三节，再遮住本节，在正式练习文件中重写。

```python
from __future__ import annotations

# dataclass 用来定义只保存指标结果的数据对象；json 负责写出可复核报告。
from dataclasses import dataclass
import json
# Path 统一处理 Windows/Linux 路径和父目录创建。
from pathlib import Path
# 类型标注说明 batch、元数据和标签名称的接口，不改变运行结果。
from typing import Any, Iterable, Mapping, Sequence

# NumPy 负责混淆矩阵和指标公式，PyTorch 负责模型前向评估。
import numpy as np
import torch
from torch import nn


# 顺序就是模型 logits 最后一轴的语义：索引 0 到 4 依次对应这五类。
LABEL_NAMES = ("W", "N1", "N2", "N3", "REM")
# 报告必须携带实验边界，防止把固定划分的单数据集结果误称为论文完整复现。
CLAIM_BOUNDARY = (
    "这是 Sleep-EDF-153 固定受试者划分上的单数据集工程评估结果，"
    "不等同于论文十折交叉验证或三数据集完整复现。"
)
# 只有指标而没有这些上下文时，别人无法确认数据、环境和权重来自哪次实验。
REQUIRED_METADATA_FIELDS = (
    "dataset_name", "split_name", "split_strategy", "random_seed",
    "checkpoint_path", "python_version", "torch_version", "device",
)


# frozen=True 禁止创建后重新绑定字段，避免报告构造期间意外替换某项指标。
@dataclass(frozen=True)
class ClassificationMetrics:
    """保存一次多分类评估的全部数值结果，供报告函数统一读取。"""

    confusion_matrix: np.ndarray  # [C,C]；行是真实类别，列是预测类别。
    support: np.ndarray  # [C]；每个真实类别的数量，即混淆矩阵行和。
    precision: np.ndarray  # [C]；每类 TP/(TP+FP)，分母是预测为该类的数量。
    recall: np.ndarray  # [C]；每类 TP/(TP+FN)，分母是真实为该类的数量。
    f1: np.ndarray  # [C]；每类 Precision 与 Recall 的调和平均。
    accuracy: float  # 标量；全部预测位置中分类正确的比例。
    macro_f1: float  # 标量；各类别 F1 的算术平均，每类权重相同。
    kappa: float | None  # 标量或 None；无类别变化时不可定义。


def _prepare_labels(y_true, y_pred, n_classes):
    """把真实/预测标签整理为一一对应的一维整数数组。"""

    # 指标只关心“每个预测位置的类别编号”。先转成 int64，再把 [B,T] 等形状
    # 按原元素顺序展平为 [N]；真实标签与预测标签执行完全相同的变换。
    true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)

    # 类别数决定混淆矩阵的边长 C；非正数无法表示有效分类任务。
    if n_classes <= 0:
        raise ValueError("n_classes 必须为正整数")
    # 两个一维数组形状相同，才保证第 i 个预测对应第 i 个真实标签。
    if true.shape != pred.shape:
        raise ValueError("y_true 和 y_pred 必须包含相同数量的预测位置")
    # 空数组会让 Accuracy 和 kappa 的总样本数分母为 0，因此不能继续计算。
    if true.size == 0:
        raise ValueError("评估标签不能为空")
    # 合法类别编号为 0 到 C-1；越界真实标签会被统计到错误的矩阵位置。
    if np.any((true < 0) | (true >= n_classes)):
        raise ValueError("y_true 包含类别范围之外的标签")
    # 预测标签使用同一检查，保证 true*C+pred 始终落在 [0,C²)。
    if np.any((pred < 0) | (pred >= n_classes)):
        raise ValueError("y_pred 包含类别范围之外的标签")

    # 返回形状相同的 [N] 数组，后续可逐位置组合真实/预测类别。
    return true, pred


def build_confusion_matrix(y_true, y_pred, n_classes=5):
    """返回 [C,C] 混淆矩阵，其中行是真实类别、列是预测类别。"""

    # 先统一 dtype、形状和标签范围，避免矩阵计数悄悄接受错位或越界输入。
    true, pred = _prepare_labels(y_true, y_pred, n_classes)
    # 对每个预测位置，把二维坐标 (真实类别,预测类别) 编成一维编号：
    # encoded=true*C+pred。例如 C=3 时 (1,2) -> 1*3+2=5。
    encoded = true * n_classes + pred
    # bincount 统计编号 0 到 C²-1 各出现几次；minlength 保证缺失类别仍占格子。
    counts = np.bincount(encoded, minlength=n_classes**2)
    # reshape 按行优先恢复 [真实类别,预测类别]，不会交换两个轴。
    confusion_matrix = counts.reshape(n_classes, n_classes)

    # cm[i,j] 表示真实 i 被预测为 j 的样本数。
    return confusion_matrix


def compute_classification_metrics(y_true, y_pred, n_classes=5):
    """从离散标签计算 ACC、每类指标、Macro-F1 和 Cohen's kappa。"""

    # 所有指标都从同一个混淆矩阵派生，避免不同公式使用不同标签集合。
    cm = build_confusion_matrix(y_true, y_pred, n_classes)
    # 计数保留整数；参与除法的副本转 float64，避免整数除法和精度损失。
    cm_float = cm.astype(np.float64)
    # 对角线 cm[c,c] 是类别 c 预测正确的数量，即每类真正例 TP，形状 [C]。
    tp = np.diag(cm_float)
    # 行语义是真实类别：消掉预测列轴 axis=1，得到每类真实数量 TP+FN。
    support = cm.sum(axis=1)
    # 列语义是预测类别：消掉真实行轴 axis=0，得到每类预测数量 TP+FP。
    predicted_count = cm.sum(axis=0)

    # Precision_c=TP_c/(TP_c+FP_c)，分母是第 c 列和 predicted_count[c]。
    # 某类从未被预测时分母为 0：out 预置 0，where 只在分母非零处相除，
    # 因而该类 Precision 为 0，不会像原仓库直接相除那样产生 NaN。
    precision = np.divide(
        tp, predicted_count,
        out=np.zeros(n_classes, dtype=np.float64),
        where=predicted_count != 0,
    )
    # Recall_c=TP_c/(TP_c+FN_c)，分母是第 c 行和 support[c]。
    # 测试集没有该真实类别时同样按 0 处理，保持输出长度为 C 且数值有限。
    recall = np.divide(
        tp, support,
        out=np.zeros(n_classes, dtype=np.float64),
        where=support != 0,
    )
    # F1 的分母是 Precision+Recall；两者都为 0 时，F1 也应为 0。
    pr_sum = precision + recall
    f1 = np.divide(
        2.0 * precision * recall, pr_sum,
        out=np.zeros(n_classes, dtype=np.float64),
        where=pr_sum != 0,
    )

    # 混淆矩阵所有格子的和就是参与评估的预测位置总数 N。
    total = int(cm.sum())
    # 对角线 TP 之和是预测正确数，所以 Accuracy=p0=正确数/N。
    accuracy = float(tp.sum() / total)
    # 对 [C] 的每类 F1 直接算术平均，让五个睡眠阶段权重相同。
    macro_f1 = float(f1.mean())
    # pe=sum_c(真实比例_c*预测比例_c)。support 与 predicted_count 是两组边际计数；
    # 点积后除以 N²，等价于二者先除以 N 再逐类别相乘求和。
    pe = float(np.dot(support, predicted_count) / total**2)
    # kappa=(p0-pe)/(1-pe)，其中 p0 就是上面的 accuracy。
    denominator = 1.0 - pe
    # pe=1 时分母为 0，kappa 数学上不可定义；None 会在 JSON 中保存为 null。
    kappa = None if np.isclose(denominator, 0.0) else float(
        (accuracy - pe) / denominator
    )

    # 使用具名结果对象统一返回 [C,C] 矩阵、[C] 指标和三个总体标量。
    return ClassificationMetrics(
        confusion_matrix=cm,
        support=support,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        macro_f1=macro_f1,
        kappa=kappa,
    )


def evaluate_model(model, batches, n_classes=5, device="cpu"):
    """汇总所有测试 batch 的预测；不建立梯度，也不更新模型参数。"""

    # 模型参数和输入必须位于同一设备；这里只移动模型，不改变权重数值。
    model.to(device)
    # eval() 让 Dropout 停止随机丢弃，并让 BatchNorm 使用保存的运行统计量。
    model.eval()
    # batch 数量未知，先逐批保存一维 NumPy 标签，最后统一 concatenate。
    true_parts = []
    pred_parts = []
    # no_grad 不建立反向图，减少显存和计算，同时保证这里不学习测试集。
    with torch.no_grad():
        for inputs, targets in batches:
            # inputs 是张量序列：单分支可能是 (raw,)，融合模型是 (raw,wave)。
            # 逐个移到模型所在设备，并保持原输入顺序。
            device_inputs = tuple(tensor.to(device) for tensor in inputs)
            # *device_inputs 把元组解包成位置参数，例如 model(raw,wave)。
            logits = model(*device_inputs)
            # 最后一轴必须是类别分数轴 C；否则 argmax 会作用在错误语义的轴。
            if logits.shape[-1] != n_classes:
                raise ValueError("logits 最后一维必须等于 n_classes")
            # 只沿最后的类别轴取最大分数：[B,C]->[B] 或 [B,T,C]->[B,T]。
            # TCN 已经完成上下文建模，这里不会删除 logits 中已有的时序信息。
            predictions = torch.argmax(logits, dim=-1)
            # 指标只需要一一对应的预测位置，因此展平为 [N_batch]；
            # NumPy 只能直接接收 CPU 张量，所以先调用 cpu()。
            pred_parts.append(predictions.reshape(-1).cpu().numpy())
            # targets 使用相同展平顺序，保证第 i 个预测仍对应第 i 个真实标签。
            true_parts.append(targets.reshape(-1).cpu().numpy())
    # 没有 batch 时 concatenate 和指标分母都无定义，明确报错便于定位数据入口。
    if not true_parts:
        raise ValueError("测试批次不能为空")
    # 沿唯一的样本轴连接各 batch，得到整个测试集的 [N_total] 标签。
    all_true = np.concatenate(true_parts)
    all_pred = np.concatenate(pred_parts)
    # 小数组测试与真实模型评估复用同一纯 NumPy 指标定义。
    return compute_classification_metrics(all_true, all_pred, n_classes)


def build_evaluation_report(metrics, metadata, label_names=LABEL_NAMES):
    """把指标、类别语义和实验上下文整理成可 JSON 序列化的字典。"""

    # 每个 [C] 指标位置都必须有类别名称，否则报告会把数值贴到错误标签上。
    if len(label_names) != len(metrics.f1):
        raise ValueError("label_names 数量必须与指标类别数一致")
    # 按固定清单找出缺失字段；它们共同说明结果来自哪次可复现实验。
    missing = [name for name in REQUIRED_METADATA_FIELDS if name not in metadata]
    if missing:
        raise ValueError(f"评估元数据缺少字段：{missing}")

    # 将四个 [C] 数组按相同索引组织成“每类一条记录”，方便阅读和制表。
    per_class = []
    for i, label_name in enumerate(label_names):
        per_class.append({
            "label": label_name,
            # NumPy 整数/浮点数不是标准 JSON 类型，转成 Python int/float。
            "support": int(metrics.support[i]),
            "precision": float(metrics.precision[i]),
            "recall": float(metrics.recall[i]),
            "f1": float(metrics.f1[i]),
        })
    # 报告分为结论边界、实验上下文、总体指标、分类指标和混淆矩阵。
    return {
        "claim_boundary": CLAIM_BOUNDARY,
        # Mapping 可能不是普通字典；dict() 生成 JSON 可直接处理的浅拷贝。
        "metadata": dict(metadata),
        "overall": {
            "accuracy": metrics.accuracy,
            "macro_f1": metrics.macro_f1,
            # None 会由 json.dump 写成 null，明确表示 kappa 不可定义。
            "cohen_kappa": metrics.kappa,
        },
        "per_class": per_class,
        # NumPy [C,C] 数组先确保为整数，再转嵌套列表供 JSON 序列化。
        "confusion_matrix": metrics.confusion_matrix.astype(int).tolist(),
    }


def save_evaluation_report(path, report):
    """创建父目录，并将评估报告保存为人类可读的 UTF-8 JSON。"""

    # 同时接受字符串和 Path；统一转换后才能使用 parent、mkdir 和 open。
    output_path = Path(path)
    # parents=True 递归创建多级目录；exist_ok=True 允许目录已经存在。
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # UTF-8 保留中文；with 在写入完成或发生异常时都会关闭文件。
    with output_path.open("w", encoding="utf-8") as file:
        # Mapping 转普通字典；ensure_ascii=False 保留中文，indent=2 便于审查。
        json.dump(dict(report), file, ensure_ascii=False, indent=2)
```

完全相同接口的独立答案保存在 `reference_solution.py`，正式模块不会导入它。

## 5. 用户练习顺序

一次只完成一个函数：

1. `_prepare_labels`：先完成 dtype、展平、配对和标签范围检查；
2. `build_confusion_matrix`：再让手工矩阵测试通过；
3. `compute_classification_metrics`：核对 ACC、每类指标、Macro-F1、kappa；
4. `evaluate_model`：验证 `eval/no_grad` 和 `[B,T,5]` 展平；
5. `build_evaluation_report`：补齐结论边界和八项元数据；
6. `save_evaluation_report`：完成 UTF-8 JSON 往返。

聚焦测试示例：

```powershell
python -m unittest `
  tests.test_sleep_edf_metrics.SleepEDFMetricsTest.test_prepare_labels_flattens_pairs_and_rejects_invalid_inputs -v
```

全部完成后：

```powershell
python -m unittest tests.test_sleep_edf_metrics -v
```

## 6. 常见错误与测试含义

- 混淆矩阵转置：会让 Precision/Recall 语义交换；先检查 `cm[真实,预测]`。
- 直接使用 `/`：某类没有预测时产生 NaN；使用 `np.divide(..., where=...)`。
- 对所有样本 F1 做加权平均：那不是 Macro-F1；本阶段每类权重相同。
- 先展平 logits 再对错误轴 `argmax`：类别轴始终是最后一轴，应先 `argmax(dim=-1)`。
- 测试阶段遗漏 `eval()`：Dropout 和 BatchNorm 行为仍处于训练状态。
- 测试阶段调用优化器：会污染测试集，应完全禁止。
- 只保存指标数字：缺少 split、seed 和 checkpoint 后无法复核。
- 把固定划分结果称为论文复现指标：实验协议不同，禁止直接等同。

## 7. 工程加固（选读）

核心练习通过后，可以另外考虑：

- 使用 JSON Schema 校验报告字段；
- 同时输出 CSV/XLSX 和混淆矩阵图片；
- 保存 Git commit、依赖锁文件哈希和 checkpoint SHA-256；
- 按受试者分别计算指标及均值/标准差；
- 使用 bootstrap 置信区间；
- 多次随机种子实验后汇总均值和标准差。

这些内容不会进入本轮核心答案，以免遮挡混淆矩阵和指标定义。
