"""里程碑 9 练习：实现 Sleep-EDF 五分类工程评估。

本文件只保存正式用户练习，不导入学习指南中的参考答案。
完整中文教程与独立参考实现位于：

    learning_guides/milestone_09/
"""

from __future__ import annotations

from dataclasses import dataclass
import json

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn


LABEL_NAMES = ("W", "N1", "N2", "N3", "REM")
CLAIM_BOUNDARY = (
    "这是 Sleep-EDF-153 固定受试者划分上的单数据集工程评估结果，"
    "不等同于论文十折交叉验证或三数据集完整复现。"
)
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


@dataclass(frozen=True)
class ClassificationMetrics:
    """保存一次多分类评估的数值结果。"""

    confusion_matrix: np.ndarray
    support: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray
    accuracy: float
    macro_f1: float
    kappa: float | None


def _prepare_labels(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """把真实/预测标签整理为一一对应的一维整数数组。"""

    # 练习目标：
    # 1. 将 y_true 和 y_pred 转成 int64，并按原顺序展平为 [N]；
    # 2. 检查 n_classes 为正数、两个数组等长且非空；
    # 3. 检查所有类别编号都位于 [0,n_classes)；
    # 4. 返回可逐位置配对的 true 和 pred。

    true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)

    if n_classes <= 0:
        raise ValueError("n_classes 必须为正整数")

    if true.shape != pred.shape:
        raise ValueError("真实标签和预测标签必须等长")

    if true.size == 0:
        raise ValueError("真实标签和预测标签都不能为空")

    if np.any(true < 0) or np.any(true >= n_classes):
        raise ValueError("真实标签必须在 [0,n_classes) 范围内")

    if np.any(pred < 0) or np.any(pred >= n_classes):
        raise ValueError("预测标签必须在 [0,n_classes) 范围内")

    return true, pred



def build_confusion_matrix(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    n_classes: int = 5,
) -> np.ndarray:
    """构造行表示真实类别、列表示预测类别的混淆矩阵。"""

    true, pred = _prepare_labels(y_true, y_pred, n_classes)
    encoded = true * n_classes + pred
    counts = np.bincount(encoded, minlength=n_classes**2)
    confusion_matrix = counts.reshape(n_classes, n_classes)
    return confusion_matrix


def compute_classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    n_classes: int = 5,
) -> ClassificationMetrics:
    """由离散标签计算 ACC、每类指标、Macro-F1 和 Cohen's kappa。"""

    cm = build_confusion_matrix(y_true, y_pred, n_classes)
    cm_float = cm.astype(np.float64)
    tp = np.diag(cm_float)
    true_count = cm.sum(axis=1)
    pred_count = cm.sum(axis=0)

    total = int(cm.sum())
    accuracy = float(tp.sum() / total)

    precision = np.divide(
        tp, pred_count,
        where=pred_count != 0,
        out=np.zeros(n_classes, dtype=np.float64)
    )

    recall = np.divide(
        tp, true_count,
        where=true_count != 0,
        out=np.zeros(n_classes, dtype=np.float64)
    )

    pr_sum = precision + recall
    f1 = np.divide(
        2 * precision * recall, pr_sum,
        where=pr_sum != 0,
        out=np.zeros(n_classes, dtype=np.float64),
    )

    macro_f1 = float(f1.mean())

    pe = float(np.dot(true_count, pred_count) / total**2)

    denominator = 1.0 - pe
    kappa = None if denominator == 0 else float((accuracy - pe) / denominator)

    return ClassificationMetrics(
        confusion_matrix=cm,
        support=true_count,
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
    """在不更新参数的前提下汇总所有 batch，并计算测试指标。"""

    model.to(device)
    model.eval()

    true_parts = []
    pred_parts = []

    with torch.no_grad():
        for x, y in batches:
            device_x = tuple(tensor.to(device) for tensor in x)
            logits = model(*device_x)

            if logits.shape[-1] != n_classes:
                raise ValueError(f"logits 最后一维必须为 {n_classes}，但为 {logits.shape[-1]}")

            pred = torch.argmax(logits, dim=-1)
            pred_parts.append(pred.reshape(-1).cpu().numpy())
            true_parts.append(y.reshape(-1).cpu().numpy())

    if not true_parts:
        raise ValueError("批次不能为空")

    all_true = np.concatenate(true_parts)
    all_pred = np.concatenate(pred_parts)

    return compute_classification_metrics(all_true, all_pred, n_classes)



def build_evaluation_report(
    metrics: ClassificationMetrics,
    metadata: Mapping[str, Any],
    label_names: Sequence[str] = LABEL_NAMES,
) -> dict[str, Any]:
    """把指标和实验上下文整理为可保存、可复核的报告字典。"""

    if len(label_names) != len(metrics.f1):
        raise ValueError("类别名称数量必须与指标类别数一致")

    missing = [name for name in REQUIRED_METADATA_FIELDS if name not in metadata]

    if missing:
        raise ValueError(f"缺少必填元数据字段：{', '.join(missing)}")

    per_class=[]
    for i, label_name in enumerate(label_names):
        per_class.append({
            "label": label_name,
            "support": int(metrics.support[i]),
            "precision": float(metrics.precision[i]),
            "recall": float(metrics.recall[i]),
            "f1": float(metrics.f1[i]),
        })

    return {
        "claim_boundary": CLAIM_BOUNDARY,
        "metadata": dict(metadata),
        "overall": {
            "accuracy": float(metrics.accuracy),
            "macro_f1": float(metrics.macro_f1),
            "cohen_kappa": metrics.kappa,
        },
        "per_class": per_class,
        "confusion_matrix": metrics.confusion_matrix.astype(int).tolist(),
    }







def save_evaluation_report(path: str | Path, report: Mapping[str, Any]) -> None:
    """将评估报告保存为 UTF-8 JSON。"""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(dict(report), f, ensure_ascii=False, indent=2)
