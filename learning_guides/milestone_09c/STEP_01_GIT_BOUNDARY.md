# 里程碑 9C 步骤一：Git 基线与发布边界

本步骤只整理本地 Git 的“下一次提交候选”，不创建 fork、不修改 remote、不提交、
不 push，也不发布版本。它是 9C 的内部学习步骤，不单独改变里程碑正式状态。

本文件一次性给出步骤一的完整操作链。请连续完成全部练习和最终自测，中途无需
停下交回；全部完成后只申请一次“步骤一综合检查”。

## 1. 问题分析

当前目录同时承担两种职责：

1. 本地实验工作区：允许保存数据、缓存、checkpoint、论文和 IDE 配置；
2. 公开源码仓库：只应保存别人理解、复核和运行项目所必需的材料。

二者不能简单地“全部提交”。数据和模型产物很大；完整论文未必具有再分发授权；
IDE/工具配置只对本机有效；生成 checkpoint 和指标表也会让代码提交难以审查。

2026-07-27 的只读盘点还发现：

- 本地 `origin` 指向作者仓库 `https://github.com/bjm-123/FFTCN.git`；
- 当前分支为 `main`，与作者 `origin/main` 对齐；
- 工作区同时有 staged、unstaged 和 untracked 文件；
- `.idea/`、生成 checkpoint、完整论文等文件已经进入暂存区；
- `.gitignore` 已排除大型 Sleep-EDF 数据和 9B checkpoint，但发布边界仍不完整。

因此第一步不是 `git commit`，而是理解 Git 的三个区域，并先让 ignore 规则表达
“什么永远不应进入公开版本”。

## 2. 必要 Git 基础

### 2.1 工作区、暂存区和 HEAD

```text
工作区 working tree
    │ git add
    ▼
暂存区 index（下一次提交的候选快照）
    │ git commit
    ▼
HEAD（当前已经提交的快照）
```

- 工作区是磁盘上现在看到的文件；
- 暂存区不是“文件备份”，而是下一次提交准备采用的内容快照；
- HEAD 是当前提交，不会因为编辑文件或执行 `git add` 自动变化。

`.gitignore` 只阻止尚未跟踪的文件进入候选集合。一个文件若已经被跟踪或暂存，
后来再写入 ignore，Git 不会自动把它从暂存区移走。因此本步骤先补规则，随后在
同一个步骤内恢复暂存区基线，再根据发布意图重建暂存集。

### 2.2 `git status --short` 的两列

短状态的第一列表示暂存区相对 HEAD，第二列表示工作区相对暂存区：

```text
XY path
││
│└─ 工作区变化
└── 暂存区变化
```

当前仓库中三个典型例子：

- `AM file`：文件已经按“新增”进入暂存区，但暂存后又在工作区修改；
- `AD file`：文件已经按“新增”进入暂存区，但工作区中后来删除；
- `?? file`：未跟踪，尚未进入暂存区。

这解释了为什么不能只看磁盘文件，也不能直接执行一次全仓库 `git add .`。

### 2.3 `.gitignore` 与删除的区别

ignore 不会删除本地文件。它只是告诉 Git：这些本地材料不应成为新提交内容。

例如忽略 `*.pth` 后，模型 checkpoint 仍保留在磁盘上，只是不再作为公开源码的
候选。下一内部动作使用 `git restore --staged` 时，也只调整暂存区，不删除工作区
文件。

### 2.4 `origin` 与 `upstream`

远程名称只是本地别名：

```text
作者仓库 upstream  ──拉取更新──>  本地仓库  ──推送──>  用户 fork origin
```

当前尚未创建用户 fork，所以现在不改远程。到步骤三才会先核对精确 URL，再把
作者仓库命名为 `upstream`、用户 fork 命名为 `origin`。

## 3. 手工推演

假设工作区有三个文件：

```text
models/net.py
datasets/train.edf
result/best.pth
```

公开版本需要模型源码，但不需要本地数据和 checkpoint：

| 文件 | 本地是否保留 | Git 是否发布 | 原因 |
|---|---:|---:|---|
| `models/net.py` | 是 | 是 | 模型实现是学习主线 |
| `datasets/train.edf` | 是 | 否 | 数据由下载脚本获取，体积大 |
| `result/best.pth` | 是 | 否 | 训练产物，不属于 v0.1 源码 |

正确顺序是：

1. 先在 `.gitignore` 表达后两类排除规则；
2. 用 `git check-ignore --no-index` 验证规则命中；
3. 再检查暂存区，移出先前误暂存的本地材料；
4. 最后才选择性暂存源码。

如果颠倒为“先 `git add .`、再考虑排除”，暂存区会混入大量不应发布的文件。

## 4. 本项目的完整核心规则

保留 `.gitignore` 中已有内容，并在文件末尾亲手补入下面这一组规则：

```gitignore
# JetBrains IDE 与本地 Codex/agent 配置只服务当前机器，不进入公开版本。
.idea/
.agents/
.codex/

# 数据全部由公开下载与预处理脚本重建，仓库不携带任何数据实体或官方副本。
datasets/

# 原仓库路径复现实验产生的模型、曲线和指标文件是本地运行产物。
source_reproduction/

# 所有模型权重和二进制训练结果都不进入 Git 历史或 v0.1 release。
*.pt
*.pth
*.npz
*.xlsx

# 论文与个人文献阅读/复现规划只作为本地学习资料，不随公开源码分发。
背景/*.pdf
背景/*.docx
背景/deep-research-report.md
```

这些规则不会排除：

- Python 源码、测试和运行脚本；
- `learning_guides/` 教学材料；
- `REPRODUCTION_PLAN.md`；
- 小型 Markdown/JSON 验证证据；
- 上游 `LICENSE`。

## 5. 完整用户练习

### 5.1 补全并自测发布边界

打开根目录 `.gitignore`，保留已有规则，在末尾加入第 4 节完整规则并保存。当前
工作区已经完成这一小项并通过一次过程性行为查看，因此无需重写；保留现有内容，
直接继续本节后续操作即可。

用下面的只读命令观察规则行为：

```powershell
git check-ignore -v --no-index `
  ".idea/FFTCN.iml" `
  "datasets/sleep-edf-expanded-1.0.0/RECORDS" `
  "source_reproduction/dry_run_outputs/RawModel/1D-CNN/network.pth" `
  "reproduction_artifacts/milestone_09b/formal_seed_0/fusion_finetune/best.pt" `
  "背景/01 A Feature Fusion Model Based on Temporal Convolutional Network for Automatic Sleep Staging Using Single-Channel EEG.pdf"
```

`--no-index` 很重要：当前部分文件已经暂存，普通 `check-ignore` 会跳过已跟踪候选；
加入该参数后才能单独验证 ignore 规则本身。命令只读，不修改工作区、暂存区或提交。

预期每个路径都显示命中的 `.gitignore` 行。若某个路径没有输出，先检查规则，不要
用 `git add` 试错。

### 5.2 把暂存区恢复到 HEAD 基线

`.gitignore` 行为自测完成后，把暂存区恢复到当前 `HEAD` 基线。这里的
“复位”不是删除文件，也不是丢弃代码修改：

```text
暂存区：恢复为 HEAD 中已经提交的内容
工作区：保持现在的全部新增、修改和删除
本地数据/模型：继续留在磁盘，由 .gitignore 隐藏
```

如需对照，可以先查看执行前的暂存快照：

```powershell
git --no-pager diff --cached --name-status
```

然后执行：

```powershell
git restore --staged :/
```

`:/` 表示当前仓库根目录下的全部路径；`--staged` 明确只恢复暂存区。不要把它
替换为 `git reset --hard`，后者会丢弃工作区修改。

执行后运行两条只读检查：

```powershell
git --no-pager diff --cached --name-status
git status --short
```

每条 Git 命令都必须以 `git` 开头。PowerShell 自己也有一个名为 `diff` 的别名；
如果漏写 `git`，出现 `InputObject / SideIndicator` 表格，说明执行成了
`Compare-Object`，不是 Git 出错。该误操作只比较两个字符串，不会改变仓库。

预期行为：

1. 第一条不再输出任何 staged 差异；
2. 源码、指南、计划等修改仍以未暂存 `M` 或未跟踪 `??` 保留；
3. 数据、IDE、checkpoint、完整论文和 `source_reproduction/` 生成物不再出现在
   普通状态中；
4. 磁盘上的本地实验文件仍然存在。

这一步不会创建提交，可以继续下一小节，无需停下交回检查。

### 5.3 按发布清单重建暂存快照

现在只把已经确认属于 v0.1 发布边界的路径加入暂存区：

```powershell
git add -- `
  ".gitignore" `
  "REPRODUCTION_PLAN.md" `
  "data" `
  "evaluation" `
  "learning_guides" `
  "models" `
  "reproduction_artifacts" `
  "scripts" `
  "tests" `
  "training"
```

这里没有使用 `git add .`，因为我们要让暂存快照直接表达发布意图。现有 ignore
规则仍会阻止上述目录中的 checkpoint、`source_reproduction/` 生成物和二进制
结果被重新加入。

本步暂不加入：

- 根 `README.md`：它将在步骤二按现行环境和运行链完整改写；
- 根 `README.pdf`：它是上游已跟踪文件，保留还是移出由步骤二的项目表面整理决定；
- `test01.py`：它是本地临时练习文件，先留在工作区，步骤二再分类；
- `背景/deep-research-report.md`：个人文献阅读与复现规划，只保留在本机；
- `LICENSE`：它已经存在于 HEAD 且没有修改，无需重复暂存。

`git add` 只更新暂存区，不会提交或上传任何内容。

当前 Windows 全局 Git 配置为 `core.autocrlf=true`。暂存 LF 文本时可能逐文件提示
`LF will be replaced by CRLF the next time Git touches it`；这是工作区换行策略提醒，
不是 `git add` 失败。步骤一不修改用户的全局配置，也不为消除提示而重写全仓库
换行符。

## 6. 整步最终自测

完成第 5 节全部操作后，再统一运行：

```powershell
git --no-pager -c core.quotePath=false diff --cached --name-status
git -c core.whitespace=cr-at-eol diff --cached --check
git -c core.quotePath=false status --short

$staged = git -c core.quotePath=false diff --cached --name-only
$forbidden = $staged | Where-Object {
  $_ -match '(^|/)(\.idea|\.agents|\.codex|datasets|source_reproduction|walkthrough|__pycache__)/' -or
  $_ -match '^背景/.*\.(pdf|docx)$' -or
  $_ -match '\.(pt|pth|npz|xlsx|pyc)$'
}
$forbidden
```

预期结果：

1. 暂存快照包含源码、测试、脚本、指南、计划和允许发布的小型 Markdown/JSON；
2. `$forbidden` 没有任何输出；
3. 带 `core.whitespace=cr-at-eol` 的空白检查没有输出；该临时参数只把仓库已有的
   CRLF 行尾视为合法，真实尾随空格和多余文件尾空行仍会被报告；
4. `.idea/`、数据、checkpoint、完整论文和生成输出没有重新进入暂存区；
5. `README.md`、`README.pdf` 和 `test01.py` 暂未进入本次新增暂存内容，等待步骤二。

若自测与预期不符，不要用 `git add .`、`git reset --hard` 或删除本地实验文件来
“清空问题”；保留现场，在整步完成说明中报告差异即可。

## 7. 唯一交回点与综合检查范围

完成第 5–6 节后，只需告诉 Codex：

> 步骤一已完成，请进行一次综合检查。

无需在 `.gitignore`、暂存区复位或选择性暂存之间分别交回。Codex 将一次性检查：

1. `.gitignore` 是否覆盖固定发布边界且没有误伤发布材料；
2. 暂存快照是否只包含允许发布的候选；
3. 工作区重要修改和本地实验文件是否仍保留；
4. 是否存在空白错误、遗漏路径或误暂存文件；
5. 是否可以进入步骤二。

这次综合检查只决定是否进入 9C 步骤二，不会把 9C 标记为完成；整个里程碑仍在
步骤三结束后统一正式验收。

## 8. 本步骤禁止动作

在步骤一中不要执行：

- `git add .`；
- `git reset --hard` 或任何丢弃工作区修改的命令；
- `git commit`；
- remote 修改；
- push、tag 或 release。
