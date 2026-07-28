"""用里程碑 9 指标接口复核里程碑 1B 已训练 checkpoint 的预测结果。

本脚本不训练里程碑 2--9 的教学重构模型。它加载的是里程碑 1B 运行原仓库
完整训练流程后得到的模型权重，目的只是验证新评估/报告代码对同一模型、同一
测试集能否得到与原评估代码一致的指标。
"""

from __future__ import annotations

import argparse
import platform
from pathlib import Path
import sys
from typing import Iterable, Sequence

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


from data.loader import Sleep_Loader
from evaluation.sleep_edf_metrics import (
    LABEL_NAMES,
    build_evaluation_report,
    evaluate_model,
    save_evaluation_report,
)
def parse_args() -> argparse.Namespace:
    """读取评估所需路径、设备和固定测试数据策略。"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=Path,
        default=REPOSITORY_ROOT / "datasets" / "sleep-edf-153-processed-v1",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "source_reproduction"
            / "full_outputs"
            / "MergeModel"
            / "FFTCN"
            / "best_network.pth"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "reproduction_artifacts"
            / "milestone_09"
            / "full_test_report.json"
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def adapt_original_batches(
    loader: Iterable[Sequence[torch.Tensor]],
) -> Iterable[tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]]:
    """把原 Loader 的 ``raw,wave,target`` 改成 ``((raw,wave),target)``。"""

    # 原 Loader 用列表返回三个张量；evaluate_model 统一要求
    # “输入张量序列 + 真实标签”的二元结构。这里只改变容器，不改变张量数值。
    for raw, wave, targets in loader:
        yield (raw, wave), targets


def main() -> None:
    """复核里程碑 1B checkpoint，并写出明确记录来源的 JSON 报告。"""

    args = parse_args()
    device = (
        "cuda:0"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )

    # 这个文件来自里程碑 1B 的原仓库完整训练，不是里程碑 2--9 教学重构模型
    # 重新训练得到的 checkpoint。原仓库保存的是完整模型对象，所以这里仍用
    # torch.load 读取；里程碑 8 的教学 checkpoint 则优先用 state_dict。
    model = torch.load(args.checkpoint, map_location=device)

    # 测试集读取固定的受试者无交集 test 子目录。wave="with" 同时返回原始 EEG
    # 和 CWT 图；seq_len=50 保持融合模型训练时的连续序列契约。
    test_loader = Sleep_Loader(
        str(args.data_path),
        set_name="test",
        batch_size=args.batch_size,
        seq_len=args.sequence_length,
        ratio=[0.8, 0.1, 0.1],
        balance=False,
        seed=args.seed,
        wave="with",
        waveshape=(30, 60),
    )

    metrics = evaluate_model(
        model,
        adapt_original_batches(test_loader),
        n_classes=len(LABEL_NAMES),
        device=device,
    )
    metadata = {
        "dataset_name": "Sleep-EDF-153",
        "split_name": "test",
        "split_strategy": "subject-disjoint fixed 62/8/8",
        "random_seed": args.seed,
        "checkpoint_path": str(args.checkpoint.resolve()),
        # 以下字段把“使用哪套模型代码训练、权重来自哪一阶段、是否重新训练”
        # 明确写入报告，避免把评估兼容性验证误读为第二次模型性能复现。
        "model_implementation": "original repository MergeSleepNet",
        "checkpoint_origin": "milestone 1B full original-repository training",
        "reconstructed_model_trained": False,
        "training_run_reused": True,
        "evaluation_purpose": (
            "validate milestone 9 metric/report compatibility against "
            "milestone 1B predictions"
        ),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device": device,
    }
    report = build_evaluation_report(metrics, metadata, label_names=LABEL_NAMES)
    save_evaluation_report(args.output, report)

    print(f"report={args.output}")
    print(f"accuracy={metrics.accuracy:.6f}")
    print(f"macro_f1={metrics.macro_f1:.6f}")
    print(f"cohen_kappa={metrics.kappa}")


if __name__ == "__main__":
    main()
