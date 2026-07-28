# FFTCN：Sleep-EDF-153 单数据集教学复现

本仓库是在上游 FFTCN 代码基础上完成的教学重构，目标是用清晰、可测试的
PyTorch 代码学习单通道 EEG 自动睡眠分期的完整数据流：

```text
30 秒 raw EEG
  ├─> 1D-CNN 时域分支 ──────────────> 256 维时域特征
  └─> Morlet CWT ─> 2D-CNN 时频分支 ─> 216 维时频特征
                                      │
                                      ▼
                              472 维融合特征
                                      │
                                      ▼
                         TCN 序列建模（T = 50）
                                      │
                                      ▼
                           W / N1 / N2 / N3 / REM
```

当前发布范围只覆盖 **Sleep-EDF-153**。仓库提供可读的模型实现、数据处理与训练
入口、无需真实数据的短测试，以及一次固定随机种子完整训练的可复核证据。

## English overview

This repository is an educational fork of FFTCN focused on a reproducible
single-dataset pipeline for Sleep-EDF-153. It provides readable PyTorch
implementations, data and training entry points, short CPU-compatible tests,
and evidence from one fixed-seed full training run. It is not a strict
reproduction of the paper's multi-dataset or cross-validation protocol.

## 1. 项目定位与结论边界

本项目适合：

- 学习 1D-CNN、2D-CNN、特征融合与 TCN 如何组成睡眠分期模型；
- 理解 Sleep-EDF 的读取、标签规范化、受试者级划分和序列边界；
- 学习两分支预训练、checkpoint 迁移、融合微调和 validation-best 选择；
- 从测试、实验报告和训练日志复核一次完整工程运行。

本项目目前不包含：

- SHHS-1 或 ISRUC-S1 的完整复现；
- 论文十折交叉验证或三数据集严格协议复现；
- 多随机种子均值、标准差或稳定性结论；
- 同态加密、可信执行环境或其他正式隐私计算实现；
- 可安装的 PyPI 包或生产部署接口。

因此，当前合法结论是：

> FFTCN 在 Sleep-EDF-153 上的单数据集、单种子工程复现。

短测试通过只说明代码接口和行为契约成立，不等于模型已经完成训练，也不等于获得
论文指标。

## 2. 模型与数据契约

### 2.1 输入、特征与输出

| 对象 | 形状或数值 |
|---|---|
| 单个 raw epoch | `[1, 3000]`，100 Hz、30 秒 |
| 连续 raw 序列 | `[T, 1, 3000]` |
| 单个 CWT | `[1, 30, 60]` |
| 连续 CWT 序列 | `[T, 1, 30, 60]` |
| 1D-CNN 时域特征 | `[T, 256]` |
| 2D-CNN 时频特征 | `[T, 216]` |
| 融合特征 | `[T, 472]` |
| TCN 表征 / logits | `[T, 128]` / `[T, 5]` |
| 正式融合序列长度 | `T = 50` |
| 睡眠阶段 | W=0、N1=1、N2=2、N3=3、REM=4 |

这里的 472 是每个 epoch 的融合特征数，不是 472 个 EEG 物理通道。

### 2.2 数据语义

- 数据来源：Sleep-EDF Expanded v1.0.0 sleep-cassette；
- 规模：153 条记录、78 名受试者；
- 固定划分：seed 42，train/validation/test 为 62/8/8 名受试者，集合两两无交集；
- 标签处理：S3/S4 合并为 N3，删除 Movement/Unknown；
- 处理后 epoch 数：train 154,128，validation 23,315，test 18,036；
- T=50 序列不跨记录，每条记录不足 50 的尾部独立丢弃；
- 正式 test 最终包含 17,800 个有效分类位置。

## 3. 仓库结构

```text
data/                     Sleep-EDF 读取、预处理、CWT 与训练缓存
models/                   1D-CNN、2D-CNN、融合 TCN 及上游模型代码
training/                 两步训练、Dataset/DataLoader、checkpoint 与正式实验
evaluation/               混淆矩阵、Accuracy、Macro-F1 与 Cohen kappa
scripts/                  下载、预处理、评估和分阶段训练入口
tests/                    无需真实数据的行为测试
learning_guides/          从数据到完整训练的中文教学材料
reproduction_artifacts/   小型 JSON/Markdown 验证证据，不包含模型权重
REPRODUCTION_PLAN.md       现行范围、进度、实验事实与验收协议
AI_AGENT_HANDOFF.md        跨智能体上下文重建与教学协作协议
```

建议先读 [学习指南索引](learning_guides/README.md)，再沿着
`data → models → training → evaluation` 阅读源码。

## 4. 环境

已验证环境：

- Windows 10；
- Python 3.10.13；
- PyTorch 1.12.1 + CUDA 11.6；
- NumPy 1.22.3；
- SciPy 1.10.1；
- NVIDIA RTX 3050 Laptop 4 GB。

完整直接依赖写在 `environment.yml`。新环境使用：

```powershell
conda env create -f environment.yml
conda activate d21
```

已有 `d21` 环境可更新为：

```powershell
conda env update -n d21 -f environment.yml --prune
conda activate d21
```

`environment.yml` 固定的是本项目验证过的直接依赖组合，不是跨所有操作系统的完整
lock file。CPU 可以运行短测试；正式训练配置使用 CUDA。

## 5. 无需数据的短检查

在仓库根目录运行：

```powershell
python -m unittest discover -s tests -v
```

这组测试使用临时小数组和小模型，不会：

- 下载 Sleep-EDF；
- 创建全量缓存；
- 读取正式 checkpoint；
- 启动完整训练；
- 修改正式 test 评估记录。

里程碑 9B 整体验收时，该短回归为 69/69 通过。详细证据见
[9B 验收摘要](reproduction_artifacts/milestone_09b/validation_summary.json)。

## 6. 数据获取与预处理

数据来自 PhysioNet 的
[Sleep-EDF Expanded v1.0.0](https://physionet.org/content/sleep-edfx/1.0.0/)。
请遵守数据来源页面的使用要求。

### 6.1 下载并校验

```powershell
python scripts/download_sleep_edf_153.py
```

该命令会联网下载 sleep-cassette 的 153 条 PSG 及对应 Hypnogram，并用官方
SHA-256 清单校验文件。默认目录：

```text
datasets/sleep-edf-expanded-1.0.0/
```

`datasets/` 已被 Git 忽略，数据不会进入公开源码历史。

### 6.2 构建固定预处理数据

```powershell
python scripts/build_sleep_edf_processed.py
```

该命令读取完整 EDF，执行标签规范化、Wake 裁剪和固定受试者划分，默认写入：

```text
datasets/sleep-edf-153-processed-v1/
```

下载和预处理都不是短检查：它们会产生较大的网络、磁盘和运行时间开销。

## 7. 缓存、预检与正式训练

完整入口是：

```text
scripts/run_milestone_09b.py
```

必须按阶段由用户逐项启动，不要写成自动串联的一条长命令：

```powershell
# 1. 从预处理 NPZ 构建 raw / labels / wave / manifest 缓存。
python scripts/run_milestone_09b.py cache

# 2. 在三种真实 batch 上执行短过拟合和显存预检。
python scripts/run_milestone_09b.py overfit

# 3. raw / wave / fusion 各运行一个 epoch，验证完整链路。
python scripts/run_milestone_09b.py epoch1

# 4. 从已验证 checkpoint 恢复到总轮数 2。
python scripts/run_milestone_09b.py resume2

# 5. 从全新 seed 0 初始化启动正式 20 / 20 / 50 轮训练。
python scripts/run_milestone_09b.py formal
```

`cache`、`epoch1`、`resume2` 和 `formal` 都会执行真实磁盘写入或模型更新。
`formal` 不是快速演示；开始前请阅读
[真实预检与正式训练指南](learning_guides/milestone_09b/OPERATOR_GUIDE.md)。

只有运行中断且存在符合契约的 `last.pt` 时，才使用：

```powershell
python scripts/run_milestone_09b.py epoch1-resume
python scripts/run_milestone_09b.py formal-resume
```

不要为了观察指标重复运行正式 test。

## 8. 正式结果

| 实现与证据 | Accuracy | Macro-F1 | Cohen kappa |
|---|---:|---:|---:|
| 原仓库完整训练 checkpoint 的接口复核 | 0.7917977528 | 0.7010606829 | 0.7141909029 |
| 教学重构 seed 0 正式训练 | 0.7941573034 | 0.7055190242 | 0.7169520827 |

教学重构的训练轮数为 raw/wave/fusion `20/20/50`。raw 和 wave 的
validation-best 权重先迁移到融合模型；融合模型再按 validation loss 选择 best；
所有模型选择结束后，fusion validation-best 才在固定 test 上正式评估一次。

两行结果都不是论文十折均值，也不能用来证明教学重构“优于论文”或具有多随机
种子稳定性。正式报告：

- [9B 正式 test 报告](reproduction_artifacts/milestone_09b/formal_seed_0/formal_test_report.json)
- [9B 验收摘要](reproduction_artifacts/milestone_09b/validation_summary.json)
- [T=50 序列语义审计](reproduction_artifacts/milestone_09b/sequence_semantics_audit.md)

## 9. 论文、上游仓库与教学代码的差异

本仓库的目标是学习并重构实际可运行的模型代码。论文描述与发布代码不一致时，
默认记录差异并实现仓库行为。当前固定实现包括：

- raw `[1,3000]` 经 Morlet CWT 得到 `[1,30,60]`；
- 1D-CNN / 2D-CNN 特征维为 256 / 216；
- 每个 epoch 的融合特征维为 472；
- TCN dilation 为 1 / 2 / 4 / 8；
- Sleep-EDF 使用固定 62/8/8 受试者划分，而不是论文十折协议；
- 正式结果只来自 seed 0 的一次完整训练。

这类差异不会被隐藏，也不会包装成论文协议严格复现。

## 10. 教学材料与证据

- [现行复现计划](REPRODUCTION_PLAN.md)
- [智能体项目交接协议](AI_AGENT_HANDOFF.md)
- [学习指南索引](learning_guides/README.md)
- [里程碑证据日志](reproduction_artifacts/MILESTONE_LOG.md)
- [复现产物说明](reproduction_artifacts/README.md)

大型数据、缓存、模型权重、IDE 配置、生成的二进制结果、论文全文和个人文献
阅读/复现规划不进入 v0.1 源码发布。

## 11. AI 辅助开发披露

本项目在 2026 年 7 月的教学重构过程中使用 **OpenAI Codex** 作为交互式编程与
教学辅助工具。其参与范围包括：

- 协助维护复现计划、教学协议和跨阶段上下文；
- 解释模型、数据、训练与 Git 概念，准备教学指南和命令；
- 辅助编写测试、项目文档和确定性的工程修正；
- 检查用户运行后产生的日志、指标、checkpoint 元数据和验收证据。

用户负责确定项目范围和验收标准，完成或审核核心模型、数据与训练代码，显式启动
长时间实验，并复核最终代码、文档、结果和结论。Codex 不是作者或最终责任主体；
智能体生成的文字也不作为独立实验依据。相关输出只有经过源码、行为测试和保存
产物核对后才会进入项目。

仓库中测试注释、学习指南和里程碑证据保留具体的 Codex 分工，是为了使开发过程
可审计，而不是把工作表述成未经辅助的纯人工成果。若将本项目用于课程作业、论文
或其他正式提交，仍须单独遵守所在学校、课程、会议或期刊的 AI 使用与披露规则；
本节不能替代提交材料中要求的 AI 使用声明。

为了让 Codex、Claude 等智能体在新会话中可靠恢复项目背景，本仓库提供
[智能体项目交接协议](AI_AGENT_HANDOFF.md)。它不会传递隐藏聊天记录或自动复制
“记忆”，而是要求智能体从现行计划、指南、代码和证据文件重新建立可核查上下文。

## 12. 上游归属与许可证

本项目基于 [bjm-123/FFTCN](https://github.com/bjm-123/FFTCN) 进行教学重构。
上游 README 还注明其 GitHub 仓库是原
[Gitee waveletnet 项目](https://gitee.com/abmoon/waveletnet/tree/master) 的再发布。

仓库保留上游 MIT 许可证及 `Copyright (c) 2024 BJM@XJTU`。教学重构新增了
Sleep-EDF-153 数据链、可读模型实现、行为测试、训练编排、实验证据和中文学习
材料。具体许可条款见 [LICENSE](LICENSE)。

本仓库用于代码学习与工程复核，不提供医疗诊断建议，也不宣称已经实现患者 EEG
隐私保护。
