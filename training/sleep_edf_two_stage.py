"""里程碑 8 练习：实现 FFTCN 的两步训练核心策略。

这个文件只保留影响实验语义的核心内容：

1. 定义预训练、微调和测试阶段的数据策略；
2. 对单条记录的单 epoch 样本执行可复现的偏移过采样；
3. 运行一次训练或验证分类步骤；
4. 把两个预训练特征分支迁移到融合模型；
5. 为新模块与预训练分支建立差分学习率；
6. 保存和恢复 state_dict、配置及标签元数据。

完整中文讲解和独立参考答案位于：

    learning_guides/milestone_08/

正式练习代码不会导入参考答案。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer


# 类别名称的顺序对应模型最后一轴的索引：0=W、1=N1、2=N2、3=N3、4=REM。
LABEL_NAMES = ("W", "N1", "N2", "N3", "REM")


# @dataclass 会根据下面的字段自动生成 __init__、__repr__、相等比较等方法；
# frozen=True 表示对象创建后字段不可重新赋值，防止训练策略在运行中被意外改变。
@dataclass(frozen=True)
class LoaderPolicy:
    """描述一个训练阶段应读取哪个 split，以及是否允许过采样。"""

    stage: str  # pretrain、finetune 或 test。
    split: str  # train、valid 或 test。
    sequence_length: int  # 一次输入包含几个连续 epoch。
    balance: bool  # 是否允许偏移过采样；连续序列、验证集和测试集必须为 False。
    input_mode: str  # raw、wave 或同时使用两种输入 both。


# 配置也使用不可变 dataclass：创建时可覆盖默认值，创建后不再随意修改。
@dataclass(frozen=True)
class TwoStageConfig:
    """里程碑 8 使用的最小可复现实验配置。"""

    seed: int = 42  # 控制随机抽样，使同一配置可以复现。
    sequence_length: int = 50  # 融合阶段一次读取 50 个连续 epoch。
    base_learning_rate: float = 1e-5  # 新 TCN 和分类头使用的学习率。
    feature_learning_rate_scale: float = 1e-2  # 预训练分支学习率的缩放比例。
    scheduler_gamma: float = 0.95  # 每轮结束后，学习率乘以 0.95。
    offset_samples: int = 300  # 偏移窗口最多移动的元素数；100 Hz 时约为 3 秒。


def build_stage_policies(sequence_length: int = 50) -> dict[str, LoaderPolicy]:
    """返回两步训练全部数据入口的策略，保证验证/测试不被过采样。"""

    "用户练习：定义两个分支预训练和融合微调的数据策略"
    # 字典的键是便于调用者查找的策略名称，值是该入口的五项设置：
    # stage、split、sequence_length、balance、input_mode。
    policies: dict[str, LoaderPolicy] = {
        # 第一步：raw 和 wave 分支都以单个 epoch 进行预训练。
        # 只有训练集可以平衡类别；验证集必须保持真实分布。
        "raw_pretrain_train": LoaderPolicy("pretrain", "train", 1, True, "raw"),
        "raw_pretrain_valid": LoaderPolicy("pretrain", "valid", 1, False, "raw"),
        "wave_pretrain_train": LoaderPolicy("pretrain", "train", 1, True, "wave"),
        "wave_pretrain_valid": LoaderPolicy("pretrain", "valid", 1, False, "wave"),

        # 第二步：融合模型读取连续序列，并同时使用 raw 与 wave 两种输入。
        # 融合训练需要保留连续 sequence_length 个 epoch 的真实顺序，不能单独过采样。
        "fusion_finetune_train": LoaderPolicy("finetune", "train", sequence_length, False, "both"),
        # 验证集只评估当前模型，不改变类别分布。
        "fusion_finetune_valid": LoaderPolicy("finetune", "valid", sequence_length, False, "both"),
        # 测试集仅用于模型选择完成后的最终评估。
        "fusion_test": LoaderPolicy("test", "test", sequence_length, False, "both"),

    }

    # 返回完整策略表；调用者根据当前阶段的键取得对应 LoaderPolicy。
    return policies


def offset_resample_record(
    epochs: np.ndarray,
    labels: np.ndarray,
    offset: int = 300,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """复现仓库的记录内偏移过采样，但使用局部随机数生成器。"""

    "用户练习：实现只用于 seq_len=1 训练集的偏移过采样"
    # labels 必须是 [N]；epochs 的第 0 轴也必须是 N，保证 epochs[i] 对应 labels[i]。
    if labels.ndim != 1 or epochs.shape[0] != labels.shape[0]:
        raise ValueError("epochs and labels 必须一一对应")

    # shape[1:] 是单个 epoch 的内部形状。例如 [N,1,3000] -> (1,3000)，
    # np.prod 再把各维相乘，得到一个 epoch 展平后的 3000 个元素。
    epoch_length = int(np.prod(epochs.shape[1:]))
    # offset 的单位是元素数，必须小于一个完整 epoch。
    if offset < 0 or offset >= epoch_length:
        raise ValueError(f"offset 必须在 0 和 {epoch_length - 1} 之间")

    # 空记录无法统计类别或生成窗口，直接返回独立副本。
    if len(labels) == 0:
        return epochs.copy(), labels.copy()

    # tolist() 转为 Python 列表；dict.fromkeys(...) 去重但保留首次出现顺序。
    class_labels = list(dict.fromkeys(labels.tolist()))

    # labels == label 产生布尔掩码，np.sum 统计 True 的数量，即该类别样本数。
    class_counts = {label: int(np.sum(labels == label)) for label in class_labels}

    # 后面每个已有类别都会额外生成 n_max 个偏移窗口。
    n_max = max(class_counts.values())

    # 先保留所有原始样本；循环生成的新样本稍后追加，最后统一拼接。
    balanced_epochs = [epochs.copy()]
    balanced_labels = [labels.copy()]

    # 把按时间排列的 [N,...] 连接成一维连续记录，允许新窗口跨越原 epoch 边界。
    flattened_record = epochs.reshape(-1)

    # 只列出需要排除的首尾索引；例如 N=5 时排除 [0,4]，内部 [1,2,3] 仍可使用。
    excluded_boundary_indices = np.array([0, len(labels) - 1])

    # 局部 RNG 使相同 seed 可复现，同时不修改 NumPy 全局随机状态。
    rng = np.random.default_rng(seed)
    for class_label in class_labels:
        # flatnonzero 直接返回属于当前类别的所有一维位置。
        candidates = np.flatnonzero(labels == class_label)

        # 从当前类别位置中删除首尾边界；setdiff1d(A,B) 表示保留 A 中不属于 B 的元素。
        candidates = np.setdiff1d(candidates, excluded_boundary_indices)

        # 类别若只出现在边界，就没有安全的偏移中心。
        if len(candidates) == 0:
            continue

        # 有放回抽取 n_max 个中心，所以候选较少时同一 epoch 可以重复出现。
        selected = rng.choice(candidates, size=n_max, replace=True)

        # offset=0 必须单独处理；否则从 [-offset, offset) 生成每个中心的移动量。
        if offset == 0:
            shifts = np.zeros(n_max, dtype=np.int64)
        else:
            shifts = rng.integers(-offset, offset, size=n_max)

        # 原起点是“epoch 索引 × 单个 epoch 长度”，再加随机移动量得到新起点。
        starts = selected * epoch_length + shifts

        # 从每个新起点截取 epoch_length 个元素，恢复为单个 epoch 原形状，
        # 再沿第 0 轴堆叠为 [n_max,...]。
        shifted = np.stack(
            [
                flattened_record[start : start + epoch_length].reshape(epochs.shape[1:])
                for start in starts
            ],
            axis=0,
        )

        # 新窗口继承其中心 epoch 的类别；数据类型与原标签保持一致。
        shifted_labels = np.full(n_max, class_label, dtype=labels.dtype)
        balanced_epochs.append(shifted)
        balanced_labels.append(shifted_labels)

    # 沿样本轴连接“全部原样本 + 各类别新样本”，并返回一一对应的数据和标签。
    return np.concatenate(balanced_epochs), np.concatenate(balanced_labels)


def run_classification_step(
    model: nn.Module,
    inputs: Sequence[torch.Tensor],
    targets: torch.Tensor,
    optimizer: Optimizer | None = None,
    return_correct_count: bool = False,
) -> float | tuple[float, int]:
    """运行一次分类步骤，并按需返回同一次前向的正确位置数。

    参数来源：
        model: 当前阶段的 raw、wave 或 fusion 模型，由上层训练编排传入。
        inputs: 当前 DataLoader batch 的模型输入；单分支含一个张量，融合分支
            含 raw、wave 两个张量，顺序与 ``model(*inputs)`` 一致。
        targets: 与 logits 预测位置一一对应的类别索引，形状为 ``[B]`` 或
            ``[B,T]``。
        optimizer: 训练时由上层传入当前优化器；验证时为 ``None``。
        return_correct_count: ``run_epoch()`` 显示累计 ACC 时传 ``True``；
            既有只需要 loss 的调用保持默认 ``False``。

    返回去向：
        默认返回 Python float 的 batch 平均交叉熵；请求正确数时返回
        ``(batch_loss, correct_count)``，供 ``run_epoch()`` 跨 batch 累加。
    """

    "用户练习：实现训练/验证共用的单步分类逻辑"
    # 是否传入优化器决定本次是训练还是验证。
    training = optimizer is not None

    # train(True) 开启训练模式；train(False) 等价于 eval()，影响 Dropout/BatchNorm。
    model.train(training)

    # 训练时建立计算图；验证时关闭梯度以减少内存与计算。
    with torch.set_grad_enabled(training):
        # *inputs 解包输入：例如 (raw,) -> model(raw)，(raw,wave) -> model(raw,wave)。
        logits = model(*inputs)

        # 将 [B,5] 或 [B,T,5] 统一整理成 [预测位置数,5]；标签同步整理成一维。
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_targets = targets.reshape(-1)

        # 返回前，loss 仍是连接计算图的零维张量，例如 tensor(0.83, grad_fn=...)。
        loss = F.cross_entropy(flat_logits, flat_targets)

        if training:
            # 清梯度 -> 反向计算梯度 -> 根据梯度更新参数。
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # detach() 断开计算图，float(...) 将零维张量转成普通 Python 浮点数。
    # 默认仍只返回本 batch 的平均交叉熵，保持里程碑 8 的原有调用接口。
    batch_loss = float(loss.detach())
    if not return_correct_count:
        return batch_loss

    # 训练进度需要 ACC 时，沿 flat_logits 最后一轴的类别分数取 argmax：
    # [预测位置数,类别数] -> [预测位置数]，再与同形状 flat_targets 比较。
    # sum 得到本 batch 正确位置数；item/int 转为 Python 整数供跨 batch 累加。
    # 这里复用本次前向的 logits，不会为了展示指标再执行一次模型前向。
    correct_count = int(
        (flat_logits.detach().argmax(dim=-1) == flat_targets).sum().item()
    )
    return batch_loss, correct_count


def transfer_pretrained_features(
    fusion_model: nn.Module,
    raw_pretrained: nn.Module,
    wave_pretrained: nn.Module,
) -> None:
    """把两个预训练特征网络的 state_dict 迁移到融合模型。"""

    # 练习目标：
    # 1. 取得 raw_pretrained 和 wave_pretrained 的 state_dict；
    # 2. 按名称加载到 fusion_model 的对应分支；
    # 3. 调用两个分支自定义的 finetune()，让其输出特征而不是五分类 logits。
    # 该函数原地修改 fusion_model，不需要返回新模型，所以返回类型是 None。

    fusion_model.raw_feature_net.load_state_dict(raw_pretrained.state_dict())
    fusion_model.wave_feature_net.load_state_dict(wave_pretrained.state_dict())

    fusion_model.raw_feature_net.finetune()
    fusion_model.wave_feature_net.finetune()



def build_finetune_optimizer(
    model: nn.Module,
    base_learning_rate: float,
    feature_learning_rate_scale: float,
) -> torch.optim.Adam:
    """为新 TCN/分类头和两个预训练分支创建差分学习率 Adam。"""

    # 练习目标：
    # 1. 分别收集 raw、wave 两个预训练分支的 Parameter 对象；
    # 2. 从完整模型参数中排除这两组，得到新 TCN 与分类头参数；
    # 3. 创建三个互不重复的参数组，新模块用 base_learning_rate，
    #    两个预训练分支用 base_learning_rate * feature_learning_rate_scale；
    # 4. 返回持有这三个参数组的 Adam 优化器。
    raw_params = list(model.raw_feature_net.parameters())
    wave_params = list(model.wave_feature_net.parameters())

    params_ids = {id(param) for param in raw_params + wave_params}
    new_params = [
        param for param in model.parameters() if id(param) not in params_ids
    ]

    feature_learning_rate = base_learning_rate * feature_learning_rate_scale

    optimizer = torch.optim.Adam(
        [
            {"params": new_params, "lr": base_learning_rate},
            {"params": raw_params, "lr": feature_learning_rate},
            {"params": wave_params, "lr": feature_learning_rate},
        ],
    )
    return optimizer





def save_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    config: Mapping[str, Any],
    label_names: Sequence[str],
    stage: str,
    epoch: int,
    validation_loss: float,
) -> None:
    """保存可移植训练检查点，不保存整个 Python 模型对象。"""

    # 练习目标：创建父目录，并用 torch.save 保存一个字典。字典至少包含：
    # model/optimizer 的 state_dict、config、label_names、stage、epoch、validation_loss。
    # 该函数只把检查点写入 path，因此没有业务返回值，返回类型是 None。

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": dict(config),
            "label_names": list(label_names),
            "stage": stage,
            "epoch": int(epoch),
            "validation_loss": float(validation_loss),
        },
        path,
    )


def load_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """恢复模型和可选优化器状态，并返回完整检查点元数据。"""

    # 练习目标：
    # 1. 用 map_location 读取检查点字典；
    # 2. 必须恢复 model_state_dict；仅在 optimizer 不为 None 时恢复优化器状态；
    # 3. 返回完整字典，使调用者还能读取配置、标签顺序和训练进度。

    checkpoint = torch.load(Path(path), map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
