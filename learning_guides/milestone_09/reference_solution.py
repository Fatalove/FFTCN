"""里程碑 9 独立参考答案；正式评估模块不会导入本文件。"""

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
    "dataset_name",
    "split_name",
    "split_strategy",
    "random_seed",
    "checkpoint_path",
    "python_version",
    "torch_version",
    "device",
)


# frozen=True 禁止创建后重新绑定字段，避免报告构造期间意外替换某项指标。
@dataclass(frozen=True)
class ClassificationMetrics:
    """保存一次多分类评估的全部数值结果，供报告函数统一读取。"""

    confusion_matrix: np.ndarray  # [C,C]；行是真实类别，列是预测类别。
    support: np.ndarray  # [C]；每个真实类别在测试集中出现的次数，即矩阵行和。
    precision: np.ndarray  # [C]；每类 TP/(TP+FP)，分母是预测为该类的数量。
    recall: np.ndarray  # [C]；每类 TP/(TP+FN)，分母是真实为该类的数量。
    f1: np.ndarray  # [C]；每类 Precision 与 Recall 的调和平均。
    accuracy: float  # 标量；全部预测位置中分类正确的比例。
    macro_f1: float  # 标量；各类别 F1 的算术平均，每类权重相同。
    kappa: float | None  # 标量或 None；校正随机一致性，无类别变化时不可定义。


def _prepare_labels(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
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
    # 预测标签使用同一范围检查，保证一维编码 true*C+pred 始终落在 [0,C²)。
    if np.any((pred < 0) | (pred >= n_classes)):
        raise ValueError("y_pred 包含类别范围之外的标签")

    # 返回的是形状相同的 [N] 数组，后续可直接逐位置组合真实/预测类别。
    return true, pred


def build_confusion_matrix(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    n_classes: int = 5,
) -> np.ndarray:
    """返回 [C,C] 混淆矩阵，其中行是真实类别、列是预测类别。"""

    # 先统一 dtype、形状和标签范围，避免矩阵计数悄悄接受错位或越界输入。
    true, pred = _prepare_labels(y_true, y_pred, n_classes)

    # 对每个预测位置，把二维坐标 (真实类别,预测类别) 编成一维编号：
    # encoded=true*C+pred。例如 C=3 时 (1,2) -> 1*3+2=5。
    encoded = true * n_classes + pred
    # bincount 统计编号 0 到 C²-1 各出现几次；minlength 保证即使某类没有出现，
    # 结果仍有 C² 个格子。reshape 按行优先恢复为 [真实类别,预测类别]。
    counts = np.bincount(encoded, minlength=n_classes**2)
    confusion_matrix = counts.reshape(n_classes, n_classes)

    # 返回 int64 计数矩阵；cm[i,j] 表示真实 i 被预测为 j 的样本数。
    return confusion_matrix


def compute_classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    n_classes: int = 5,
) -> ClassificationMetrics:
    """从离散标签计算 ACC、每类指标、Macro-F1 和 Cohen's kappa。"""

    # 所有指标都从同一个混淆矩阵派生，避免各公式使用了不同的标签过滤结果。
    cm = build_confusion_matrix(y_true, y_pred, n_classes)
    # 计数本身保留整数；参与除法的副本转 float64，避免整数除法和精度损失。
    cm_float = cm.astype(np.float64)

    # 对角线 cm[c,c] 是类别 c 预测正确的数量，即每类真正例 TP，形状 [C]。
    true_positive = np.diag(cm_float)
    # 行语义是真实类别，所以消掉“预测列轴 axis=1”后得到每类真实数量 TP+FN。
    support = cm.sum(axis=1)
    # 列语义是预测类别，所以消掉“真实行轴 axis=0”后得到每类预测数量 TP+FP。
    predicted_count = cm.sum(axis=0)

    # Precision_c=TP_c/(TP_c+FP_c)，代码分母正是第 c 列和 predicted_count[c]。
    # 某类从未被预测时分母为 0：out 先放置 0，where 只在分母非零处相除，
    # 因而该类 Precision 明确记为 0，不会像原仓库直接相除那样产生 NaN。
    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros(n_classes, dtype=np.float64),
        where=predicted_count != 0,
    )
    # Recall_c=TP_c/(TP_c+FN_c)，分母是第 c 行和 support[c]。
    # 测试集没有该真实类别时同样按 0 处理，使输出保持有限且长度始终为 C。
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros(n_classes, dtype=np.float64),
        where=support != 0,
    )
    # F1 的分母是 Precision+Recall；两者都为 0 时，F1 也应为 0。
    precision_recall_sum = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        precision_recall_sum,
        out=np.zeros(n_classes, dtype=np.float64),
        where=precision_recall_sum != 0,
    )

    # 混淆矩阵所有格子的和就是参与评估的预测位置总数 N。
    total = int(cm.sum())
    # 对角线 TP 之和是预测正确数，所以 Accuracy=p0=正确数/N。
    accuracy = float(true_positive.sum() / total)
    # 对 [C] 的每类 F1 直接算术平均，让 W、N1、N2、N3、REM 权重相同。
    macro_f1 = float(f1.mean())

    # 随机期望一致率 pe=sum_c(真实比例_c*预测比例_c)。
    # support 是真实边际计数，predicted_count 是预测边际计数；点积后除以 N²，
    # 等价于先把二者分别除以 N，再逐类别相乘求和。
    expected_agreement = float(np.dot(support, predicted_count) / (total**2))
    # kappa=(p0-pe)/(1-pe)，其中 p0 就是上面算出的 accuracy。
    denominator = 1.0 - expected_agreement
    # 若真实和预测都只有同一个类别，pe=1，分母为 0，kappa 数学上不可定义。
    # 使用 None 可在 JSON 中保存为 null，比伪造 0 或 1 更诚实。
    kappa = None
    if not np.isclose(denominator, 0.0):
        kappa = float((accuracy - expected_agreement) / denominator)

    # 使用具名字段返回，确保报告函数能清楚区分 [C,C] 矩阵、[C] 指标和标量。
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


def evaluate_model(
    model: nn.Module,
    batches: Iterable[tuple[Sequence[torch.Tensor], torch.Tensor]],
    n_classes: int = 5,
    device: str | torch.device = "cpu",
) -> ClassificationMetrics:
    """汇总所有测试 batch 的预测；不建立梯度，也不更新模型参数。"""

    # 模型参数和输入必须位于同一设备；这里只移动模型，不改变其权重数值。
    model.to(device)
    # eval() 让 Dropout 停止随机丢弃，并让 BatchNorm 使用已保存的运行统计量。
    model.eval()
    # batch 数量未知，先逐批保存一维 NumPy 标签，最后统一 concatenate。
    true_parts: list[np.ndarray] = []
    pred_parts: list[np.ndarray] = []

    # 测试阶段不需要反向传播。no_grad 不建立计算图，减少显存和计算开销，
    # 同时从机制上保证这里不会通过 loss.backward() 学习测试集。
    with torch.no_grad():
        for inputs, targets in batches:
            # inputs 是张量序列：单分支可能是 (raw,)，融合模型是 (raw,wave)。
            # 逐个移到模型所在设备，但保持元组顺序，供下一行的 * 解包使用。
            device_inputs = tuple(tensor.to(device) for tensor in inputs)
            # *device_inputs 把元组还原成位置参数，例如 model(raw,wave)。
            logits = model(*device_inputs)
            # 最后一轴必须是类别分数轴 C；否则 argmax 会在错误语义的轴上取最大值。
            if logits.shape[-1] != n_classes:
                raise ValueError("logits 最后一维必须等于 n_classes")

            # 模型已经沿时间轴完成上下文计算。现在只在最后的类别轴取最大分数：
            # [B,C] -> [B]，或 [B,T,C] -> [B,T]，不会破坏 TCN 已融入的上下文。
            predictions = torch.argmax(logits, dim=-1)
            # 评估公式只需要一一对应的标签位置，所以把 [B] 或 [B,T] 展平为 [N_batch]。
            # detach 在 no_grad 中已隐含；仍需先移到 CPU，才能安全转换成 NumPy 数组。
            pred_parts.append(predictions.reshape(-1).cpu().numpy())
            # targets 执行同样的展平顺序，保证第 i 个预测仍对应第 i 个真实标签。
            true_parts.append(targets.reshape(-1).cpu().numpy())

    # 没有任何 batch 时，后续 concatenate 和指标分母都没有定义，明确报错更易定位。
    if not true_parts:
        raise ValueError("测试批次不能为空")
    # 沿唯一的一维样本轴连接各 batch，得到整个测试集的 [N_total] 标签。
    all_true = np.concatenate(true_parts)
    all_pred = np.concatenate(pred_parts)
    # 统一交给纯 NumPy 指标函数，确保小数组测试和真实模型评估使用同一套定义。
    return compute_classification_metrics(all_true, all_pred, n_classes=n_classes)


def build_evaluation_report(
    metrics: ClassificationMetrics,
    metadata: Mapping[str, Any],
    label_names: Sequence[str] = LABEL_NAMES,
) -> dict[str, Any]:
    """把指标、类别语义和实验上下文整理成可 JSON 序列化的报告字典。"""

    # 每个 [C] 指标位置都必须有一个类别名称，否则报告会把数值贴到错误标签上。
    if len(label_names) != len(metrics.f1):
        raise ValueError("label_names 数量必须与指标类别数一致")
    # 按固定清单找出缺失字段；这些字段共同说明结果来自哪次可复现实验。
    missing = [name for name in REQUIRED_METADATA_FIELDS if name not in metadata]
    if missing:
        raise ValueError(f"评估元数据缺少字段：{missing}")

    # 将四个 [C] 数组按相同索引组织成“每类一条记录”，方便阅读和后续制表。
    per_class = []
    for index, label_name in enumerate(label_names):
        per_class.append(
            {
                "label": label_name,
                # NumPy 整数/浮点数不是标准 JSON 类型，转成 Python int/float。
                "support": int(metrics.support[index]),
                "precision": float(metrics.precision[index]),
                "recall": float(metrics.recall[index]),
                "f1": float(metrics.f1[index]),
            }
        )

    # 报告分成结论边界、实验上下文、总体指标、分类指标和原始混淆矩阵五部分。
    return {
        "claim_boundary": CLAIM_BOUNDARY,
        # Mapping 可能不是普通字典；dict() 生成 JSON 能直接处理的浅拷贝。
        "metadata": dict(metadata),
        "overall": {
            "accuracy": metrics.accuracy,
            "macro_f1": metrics.macro_f1,
            # kappa 为 None 时，json.dump 会将其写成 null，明确表示不可定义。
            "cohen_kappa": metrics.kappa,
        },
        "per_class": per_class,
        # NumPy [C,C] 数组先确保为整数，再转嵌套列表供 JSON 序列化。
        "confusion_matrix": metrics.confusion_matrix.astype(int).tolist(),
    }


def save_evaluation_report(path: str | Path, report: Mapping[str, Any]) -> None:
    """创建父目录，并把评估报告保存为人类可读的 UTF-8 JSON。"""

    # 同时接受字符串和 Path；统一转换后可使用 parent、mkdir 和 open。
    output_path = Path(path)
    # parents=True 递归创建多级目录；exist_ok=True 允许目录已经存在。
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # UTF-8 保留中文；with 在写入完成或异常时都会关闭文件。
    with output_path.open("w", encoding="utf-8") as file:
        # Mapping 先转普通字典；ensure_ascii=False 直接保存中文，indent=2 便于人工审查。
        json.dump(dict(report), file, ensure_ascii=False, indent=2)
