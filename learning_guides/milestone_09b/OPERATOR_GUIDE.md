# 里程碑 9B：真实预检与正式训练运行指南

这份手册只负责运行已经完成并通过全套短测试的教学重构代码。所有会构建
全量缓存或更新真实模型参数的命令都由你在本机 PowerShell 中亲自启动；
Codex 负责准备命令、检查每阶段产物并判断是否可以进入下一项。

> 当前状态（2026-07-25）：这些阶段已全部完成，里程碑 9B 已整体验收通过。
> 本手册保留为运行链记录，不是当前待执行清单。正式报告已经存在，禁止重复
> 执行 `formal`、`formal-resume` 或手工再次评估 test。

## 1. 运行位置与实验配置

请先进入克隆后的仓库根目录；下文所有项目文件路径都相对于该目录。

| 用途 | 运行约定 |
|---|---|
| Python | 先激活由 `environment.yml` 建立的 `d21` 环境，再使用 `python` |
| 处理后记录 | `datasets/sleep-edf-153-processed-v1/` |
| 全量训练缓存 | `datasets/sleep-edf-153-training-cache-v1/` |
| 预检摘要 | `reproduction_artifacts/milestone_09b/real_run/` |
| 1→2 epoch 恢复预检 | `reproduction_artifacts/milestone_09b/real_run/preflight_training/` |
| 正式 seed 0 输出 | `reproduction_artifacts/milestone_09b/formal_seed_0/` |
| 最终正式报告 | `reproduction_artifacts/milestone_09b/formal_seed_0/formal_test_report.json` |

正式配置固定为 seed 0、偏移 300 点、序列长度 50、raw/wave/fusion
轮数 `20/20/50`、batch size `128/128/32`、学习率 `1e-5`、
特征分支学习率比例 `1e-2`、scheduler gamma `0.95`、`num_workers=0`。
运行入口不允许从命令行改这些值。

## 2. 开始前只运行短检查

```powershell
python -m unittest tests.test_sleep_edf_formal_experiment -v
python -m unittest tests.test_milestone_09b_runner -v
python scripts\run_milestone_09b.py status
```

前两条分别检查步骤四核心逻辑和固定运行入口。第三条只读取路径、磁盘、缓存和训练产物，
不会创建缓存或更新模型。首次启动正式训练前应看到处理后记录数为
train/validation/test=`122/16/15`，四组缓存均为 complete，overfit 报告已存在，
三阶段恢复预检均已完成 epoch 0–1，best/last/history 和 `resume2.json` 已通过核验；
正式输出目录尚不存在，正式训练、正式 test 和最终报告均未生成。

当前正式训练、一次正式 test 和报告均已完成；`status` 已核对三阶段
`20/20/50`、best/last/history 与报告齐全。第 3 节命令只记录当时的操作顺序，
不要再次执行。

## 3. 已完成阶段的原运行顺序

以下命令是 9B 已完成实验的可复核运行顺序，不是当前指令。原执行时没有把命令
放进同一个脚本或并行运行，而是每完成一项先核对产物再进入下一项。

长阶段会直接在当前终端显示进度。缓存进度条按全部 CWT epoch 连续计数并显示
当前 split；训练在每个 epoch 内分别显示 train 和 validation 的 batch 进度。
每个阶段先显示一次 `==================== RAW PRETRAIN ====================`；
随后以 `Epoch 1/20 | Train` 和 `Epoch 1/20 | Valid` 开头的进度行分别显示
训练和验证。`125/3865 batch` 表示当前 epoch 已完成 125 个训练 batch；
进度行还显示百分比、持续时间、速度、累计 `loss` 和累计 `acc`。二者都在
运行中按 batch 更新，分别表示截至当前所有预测位置的平均损失和准确率。
进度条宽度固定为 100 列，避免在宽屏终端被无意义拉长。无需依靠另一个终端
才能判断主进程是否仍在推进。

### 3.1 构建全量缓存

```powershell
python scripts\run_milestone_09b.py cache
```

这一步预计新增约 10 GiB，并包含约 690,096 次 CWT；先前微型基准粗估约
1.04 小时，实际以本次记录为准。当前终端会依次显示
`cache train/validation/test/pretrain_train CWT`，并以 epoch 为单位更新同一条
总进度。成功后检查：

- `real_run/cache_build.json` 已生成；
- train/validation/test 样本数分别为 `154128/23315/18036`；
- `pretrain_train` 样本数为 `494617`；
- 四组缓存都具有对齐的 `raw.npy/wave.npy/labels.npy/manifest.json`。

若命令中断，缓存目录可能只有部分文件。不要直接重跑或手工删除，把状态输出
发给 Codex 后再决定如何处理。

### 3.2 三种真实 batch 过拟合与显存

```powershell
python scripts\run_milestone_09b.py overfit
```

入口会依次取 raw、wave、fusion 的真实首 batch，各自在同一 batch 上更新
20 次，记录首末损失和 CUDA 峰值。成功后检查
`real_run/overfit.json` 中三段 `loss_decreased` 都是 `true`。

若 fusion 出现 OOM，不要自行调小 batch 32；把完整错误和已经写出的
raw/wave 结果发给 Codex，先讨论是否改变源仓库配置。

### 3.3 三阶段各跑 1 epoch

```powershell
python scripts\run_milestone_09b.py epoch1
```

这一步会真实更新模型，但不会读取 test。成功后
raw/wave/fusion 三个目录都应产生 `best.pt`、`last.pt`、`history.json`，
且每份 history 只有 epoch 0。运行时每个阶段依次出现一条 train batch
进度和一条 validation batch 进度，描述中的 `[1/1]` 表示当前预检 epoch。

若 `epoch1` 在完整生成 `real_run/epoch1.json` 前中断，不要重新运行首次命令，
改用同一预检目录恢复：

```powershell
python scripts\run_milestone_09b.py epoch1-resume
```

已经生成 `last.pt` 的阶段会从该断点恢复；若只来得及创建空阶段目录、还没有
任何 checkpoint，则会以同一 seed 从头开始该预检。

### 3.4 在同一目录恢复到总轮数 2

```powershell
python scripts\run_milestone_09b.py resume2
```

这里的“2”是计划总轮数，不是额外再跑 2 轮。入口固定复用 3.3 的输出目录，
并核对 history 为 `[0,1]`、scheduler 学习率连续乘以 `0.95`。

预检 checkpoint 只证明恢复链可用，不能作为正式训练起点。
当前实测三阶段 history 均为 `[0,1]`，best/last 位于 epoch 1，恢复预检已经通过。

### 3.5 从头启动正式 20/20/50

只有 3.1–3.4 全部检查通过后才运行：

```powershell
python scripts\run_milestone_09b.py formal
```

正式训练使用全新的 `formal_seed_0` 目录，从随机 seed 0 初始化开始，不加载
预检 checkpoint。它依次完成 raw 20、wave 20、fusion 50 轮；随后只加载
fusion validation-best，对固定 test 正式评估一次并写出报告。不要在命令
结束后再手工调用第二次 test。

### 3.6 只有正式训练中断时才恢复

```powershell
python scripts\run_milestone_09b.py formal-resume
```

恢复只读取同一个 `formal_seed_0` 目录中已经存在的各阶段 `last.pt`。不要把
`preflight_training` 的 checkpoint 复制或传给正式目录。若正式报告已经存在，
入口会拒绝再次运行，防止重复正式 test。

若中断发生在首个 `last.pt` 之前，但正式目录里只有脚本创建的空阶段目录，
`formal-resume` 会以同一 seed 从头开始。若三个 history 已达到 20/20/50 而
报告缺失，入口会停止：这时无法只凭文件判断 test 是否已经开始，必须先人工
核对，不能自动重试。

当前 checkpoint 遵循原仓库式运行链，只保存模型、优化器、scheduler、epoch
和 history，不保存 Python/NumPy/PyTorch 与 DataLoader 的随机状态。因此恢复
保证参数和学习率状态连续，但不承诺与“从未中断”的 shuffle/dropout 轨迹逐位
一致。发生恢复时，正式报告中的训练耗时和峰值显存只覆盖完成实验的最后一次
进程；提交验收时必须同时注明曾经恢复，不能把它误写成完整不中断运行的总耗时。

## 4. 可选的旁路状态检查

当前运行终端已经显示进度条。只有需要核对磁盘、checkpoint 或 history 文件时，
才在另一个 PowerShell 终端运行以下只读命令：

```powershell
python scripts\run_milestone_09b.py status
```

它会显示每个阶段已完成的 epoch 数、最后 epoch、best/last 是否存在以及
history 路径。也可以直接查看某一阶段的最新 history：

```powershell
Get-Content -LiteralPath "reproduction_artifacts\milestone_09b\formal_seed_0\raw_pretrain\history.json"
```

训练进入 wave 或 fusion 后，把目录名改成 `wave_pretrain` 或
`fusion_finetune`。查看 GPU 状态可在另一终端执行：

```powershell
nvidia-smi
```

## 5. 每一步需要交回检查的内容

| 阶段 | 交回内容 |
|---|---|
| cache | 终端末尾输出、`real_run/cache_build.json` |
| overfit | 终端输出、`real_run/overfit.json` |
| epoch1 | `real_run/epoch1.json` |
| resume2 | `real_run/resume2.json` |
| formal / formal-resume | 当前 `status` 输出；完成后提交 `formal_test_report.json` |

局部预检通过不等于里程碑完成。本次是在全量缓存、全部预检、固定 seed 0 完整
训练、一次正式 test、最终报告与全部回归共同通过后，才将整个里程碑 9B
标记为已完成。
