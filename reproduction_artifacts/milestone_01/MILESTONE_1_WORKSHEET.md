# 里程碑 1：环境与形状练习

## 代码变更说明

`scripts/milestone_1_check.py` 是本项目新增的只读诊断脚本。原仓库只在 README 中写了 Python/PyTorch 版本，没有可执行的环境证明；因此把版本、解释器路径、NVIDIA 驱动、PyTorch CUDA runtime 和关键包统一输出为 JSON。

为什么不把环境检查写入模型入口：模型代码应该负责计算，环境诊断应能在 PyTorch 缺失或 CUDA 不可用时仍给出结构化报告。将两者分离后，同一脚本也可以比较不同 Conda 环境，而不会触发训练副作用。

验证方法：报告中的 `python.executable` 必须指向目标环境，且实际执行一次 CUDA 张量或模型前向；仅看到 `nvidia-smi` 不能证明 PyTorch CUDA 可用。

## A. 环境记录

创建并激活独立的 `fftcn` 环境后运行：

```powershell
python .\scripts\milestone_1_check.py `
  --output .\reproduction_artifacts\milestone_01\environment_report.json
```

请确认报告中的 `python.executable` 指向 `fftcn` 环境，并记录：

| 项目 | 结果 |
|---|---|
| Python |  |
| PyTorch |  |
| PyTorch 编译时 CUDA runtime |  |
| `torch.cuda.is_available()` |  |
| GPU |  |
| 驱动 |  |
| NumPy |  |
| SciPy |  |

## B. 形状推导

卷积或池化的单轴输出公式（dilation 默认为 1）：

```text
L_out = floor((L_in + 2P - D(K - 1) - 1) / S + 1)
```

请根据 `models/raw/parameter.py` 独立填写：

| 1D 层 | 通道数 | 输入长度 | K | S | P | 输出长度 |
|---|---:|---:|---:|---:|---:|---:|
| Conv1 | 128 | 3000 | 50 | 25 | 0 |  |
| MaxPool1 | 128 |  | 8 | 8 | 0 |  |
| Conv2 | 128 |  | 8 | 1 | 3 |  |
| Conv3 | 128 |  | 8 | 1 | 3 |  |
| Conv4 | 128 |  | 8 | 1 | 3 |  |
| MaxPool2 | 128 |  | 4 | 4 | 0 |  |

最终时域特征维数：`通道数 × 最终长度 = ______`。

请根据 `models/wavelet/parameter.py` 独立填写。每组两次卷积均使用
`K=3, S=1, P=1`，不改变高宽；随后池化使用 `K=2, S=2, P=0`：

| 2D 层组 | 输出通道 | 输入高×宽 | 池化后高×宽 |
|---|---:|---:|---:|
| Block 1 | 32 | 30×60 |  |
| Block 2 | 48 |  |  |
| Block 3 | 64 |  |  |
| Block 4 | 72 |  |  |

最终时频特征维数：`通道数 × 最终高 × 最终宽 = ______`。

融合特征维数：`时域特征 + 时频特征 = ______`。

## C. 验收解释题

1. 为什么项目环境不能直接依赖 Conda `base`？
2. NVIDIA 驱动、系统 CUDA Toolkit、PyTorch 自带 CUDA runtime 分别负责什么？
3. 论文版 816 维与仓库版 472 维分别由哪两条分支组成，差异来自哪些结构变更？
