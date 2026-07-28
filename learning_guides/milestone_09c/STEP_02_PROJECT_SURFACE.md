# 里程碑 9C 步骤二：README、环境与最小复现入口

本步骤把已经通过代码与实验验收的 FFTCN 教学重构，整理成新读者可以理解和验证的
项目表面。只修改本地文档、环境声明和发布候选，不创建 GitHub fork，不修改
remote，不 commit，也不运行数据下载、全量缓存或训练。

本文件一次性给出步骤二的完整操作链。请完成全部编辑、显式暂存和最终自测后，
只申请一次“步骤二综合检查”。

## 1. 问题分析

本步骤开始前，根 `README.md` 仍主要描述上游仓库：

- 声明 Python 3.7、PyTorch 1.8，与已经验证的 `d21` 环境不一致；
- 把 SHHS dataloader 和旧 trainer 当作主要入口，没有说明当前 Sleep-EDF-153 链路；
- 没有说明 2–9B 教学重构代码、固定划分、正式指标和结论边界；
- 要求读者查看 8 MB 的 `README.pdf`，但该二进制副本不属于当前 v0.1 的必要源码；
- 没有可直接创建环境的 `environment.yml`，也没有“无需数据即可运行”的短检查。

步骤一还留下一个有意未暂存的 `test01.py`。它只有一个 NumPy 临时练习，不是项目
测试；若写入公开 `.gitignore`，会把个人文件名误变成所有贡献者的仓库规则。

因此本步骤不是“把 README 写长”，而是建立一条可执行的读者路径：

```text
理解项目边界
  -> 创建/核对 d21 环境
  -> 无数据短测试
  -> 按需下载和预处理 Sleep-EDF-153
  -> 定位缓存、预检和正式训练入口
  -> 从证据文件复核正式指标
```

## 2. 已核对的事实底座

### 2.1 环境事实

| 依赖 | 已验证版本 |
|---|---:|
| Python | 3.10.13 |
| PyTorch / CUDA runtime | 1.12.1 / 11.6 |
| NumPy | 1.22.3 |
| SciPy | 1.10.1 |
| Matplotlib | 3.7.2 |
| tqdm | 4.68.3 |
| pyEDFlib | 0.1.42 |
| requests | 2.31.0 |
| openpyxl | 3.1.5 |

这些版本来自当前 `d21` 环境和
`reproduction_artifacts/milestone_01/environment_d21_report.json`。环境文件描述
已验证组合，不承诺任意新版本都具有相同行为。

### 2.2 数据与结果事实

- 数据集：Sleep-EDF Expanded v1.0.0 sleep-cassette，153 records / 78 subjects；
- 固定受试者划分：seed 42，train/validation/test 为 62/8/8；
- 处理后 epoch：154,128 / 23,315 / 18,036；
- T=50 按记录组序列后，正式 test 位置为 17,800；
- 教学重构正式 seed 0：Accuracy `0.7941573034`、Macro-F1 `0.7055190242`、
  Cohen kappa `0.7169520827`；
- test 只在模型选择结束后评估一次，best checkpoint 只由 validation loss 决定；
- 这是单数据集、单种子工程复现，不是论文十折或三数据集严格复现。

### 2.3 运行入口事实

| 目的 | 入口 |
|---|---|
| 下载并校验官方数据 | `scripts/download_sleep_edf_153.py` |
| 构建固定预处理数据 | `scripts/build_sleep_edf_processed.py` |
| 缓存/预检/正式训练 | `scripts/run_milestone_09b.py` |
| 详细长流程说明 | `learning_guides/milestone_09b/OPERATOR_GUIDE.md` |
| 正式结果 | `reproduction_artifacts/milestone_09b/formal_seed_0/formal_test_report.json` |
| 9B 验收摘要 | `reproduction_artifacts/milestone_09b/validation_summary.json` |

## 3. 必要项目化基础

### 3.1 README 是可执行地图

README 中的每条命令都应回答三个问题：

1. 运行前需要什么；
2. 它会不会联网、写大量文件或更新模型；
3. 成功后读者应看到或得到什么。

短测试、下载、预处理、缓存和训练不能混成一个“运行项目”命令。否则读者无法判断
时间、磁盘、GPU 和 test 使用边界。

### 3.2 `environment.yml` 不是完整 lock file

本项目使用它固定直接依赖和关键版本，让读者复原已验证环境。Conda 仍会解析操作
系统相关的传递依赖；因此 README 应写“已验证组合”，不能写成“所有平台完全一致”。

### 3.3 工作区与公开快照可以不同

- `README.pdf`：上游已跟踪的生成二进制；从 Git 索引移除，但本地文件继续保留；
- `test01.py`：个人临时练习；写入 `.git/info/exclude`，只影响当前克隆；
- `README.md`、`environment.yml`、`AI_AGENT_HANDOFF.md`：所有读者需要，进入公开
  暂存快照。

这三种处理分别对应“公开删除但本地保留”“仅本机忽略”“正式发布”。

## 4. 手工推演

假设新读者只想确认模型代码能导入：

```text
不需要数据
不需要 checkpoint
不需要 GPU 训练
只需要环境 + 源码 + 短测试
```

如果 README 一上来要求运行 `formal`，读者会误以为验证仓库必须重新执行
20/20/50 轮训练。正确分层应是：

| 层级 | 命令性质 | 是否进入本步骤自测 |
|---|---|---:|
| 环境导入 | 只读、秒级 | 是 |
| `unittest` 短回归 | 临时小张量、CPU 可运行 | 是 |
| CLI `--help` | 只读、秒级 | 是 |
| 数据下载 | 联网、大文件 | 否，只写入口 |
| 预处理/缓存 | 大量磁盘与 CWT | 否，只写入口 |
| 正式训练 | GPU、长时间、模型更新 | 否，只写入口 |

## 5. 完整用户练习

### 5.1 分类 `README.pdf` 与 `test01.py`

在根 `.gitignore` 的文档规则后加入：

```gitignore
# 上游 README 的生成 PDF 已由可维护的 Markdown 取代，本地副本不再发布。
README.pdf
```

随后只从 Git 索引移除 PDF，保留磁盘文件：

```powershell
git rm --cached -- README.pdf
```

把个人练习文件写入当前克隆的本地排除表：

```powershell
$exclude = ".git/info/exclude"
if (-not (Select-String -LiteralPath $exclude -Pattern '^test01\.py$' -Quiet)) {
  Add-Content -LiteralPath $exclude -Value "test01.py"
}
```

不要把 `test01.py` 写入公开 `.gitignore`，也不要删除它。`.git/info/exclude` 不会被
commit，不需要 `git add`。

### 5.2 创建根 `environment.yml`

亲手创建文件并填写完整内容：

```yaml
name: d21
channels:
  - pytorch
  - defaults
dependencies:
  - python=3.10.13
  - pytorch=1.12.1
  - cudatoolkit=11.6
  - numpy=1.22.3
  - scipy=1.10.1
  - matplotlib=3.7.2
  - tqdm=4.68.3
  - pip
  - pip:
      - pyEDFlib==0.1.42
      - requests==2.31.0
      - openpyxl==3.1.5
```

字段含义：

- `name` 对应已验证环境名；
- `channels` 让旧版 PyTorch/CUDA 组合从 PyTorch 官方 Conda channel 解析；
- Conda 部分固定训练与数值计算主依赖；
- pip 部分固定 EDF 读取、官方数据下载和旧结果表兼容依赖。

不要把本机绝对路径、GPU 型号或用户目录写进 YAML。

### 5.3 核对由 Codex 重写的根 `README.md`

用户已明确表示暂时无法独立完成 README，并委托 Codex 根据项目事实重写。现有
`README.md` 已由 Codex 完成；本小节不再要求用户重新编写，只需按下面的内容契约
阅读核对。README 以中文为主，并保留一个简短英文概述，已经按以下顺序覆盖：

1. **标题与一句话定位**：Sleep-EDF-153 上的 FFTCN 单数据集教学复现；
2. **English overview**：说明 fork、single-dataset educational reproduction、
   runnable tests、reproducible training pipeline；
3. **范围与非范围**：明确不是论文三数据集/十折严格复现，不宣称隐私保护；
4. **模型数据流**：raw 1D-CNN 256 + CWT 2D-CNN 216 → 472 fusion → TCN → 5 类；
5. **仓库结构**：列出 `data/models/training/evaluation/scripts/tests/learning_guides/
   reproduction_artifacts` 的职责；
6. **环境**：给出新建与激活命令；
7. **无需数据的短检查**：给出 `unittest` 命令并说明不会训练；
8. **数据获取与预处理**：只给入口、默认目录和联网/磁盘提醒；
9. **缓存、预检与正式训练**：列出固定 stage 顺序，并链接操作手册；
10. **正式结果与证据**：写精确指标、test 位置数、模型选择与 test 一次性边界；
11. **论文/上游/教学代码差异**：写固定 CWT、特征、TCN、划分和单种子差异；
12. **AI 辅助开发披露**：说明 Codex 的具体分工、用户责任和正式提交边界；
13. **跨智能体交接**：链接 `AI_AGENT_HANDOFF.md`，要求从现行文件重建上下文，
    不声称自动继承隐藏聊天记忆；
14. **归属与许可证**：保留 BJM@XJTU MIT 版权，说明本仓库是教学 fork。

README 使用下面的英文概述，没有创造新的实验结论：

```markdown
## English overview

This repository is an educational fork of FFTCN focused on a reproducible
single-dataset pipeline for Sleep-EDF-153. It provides readable PyTorch
implementations, data and training entry points, short CPU-compatible tests,
and evidence from one fixed-seed full training run. It is not a strict
reproduction of the paper's multi-dataset or cross-validation protocol.
```

环境命令已经写成：

```powershell
conda env create -f environment.yml
conda activate d21
```

已有 `d21` 的维护命令为：

```powershell
conda env update -n d21 -f environment.yml --prune
```

短检查为：

```powershell
python -m unittest discover -s tests -v
```

数据与完整训练入口为：

```powershell
# 联网并写入大型数据；不是短检查。
python scripts/download_sleep_edf_153.py

# 读取完整 EDF 并写入预处理 NPZ；不是短检查。
python scripts/build_sleep_edf_processed.py

# 下面各阶段必须由用户逐项明确启动，不能自动串联。
python scripts/run_milestone_09b.py cache
python scripts/run_milestone_09b.py overfit
python scripts/run_milestone_09b.py epoch1
python scripts/run_milestone_09b.py resume2
python scripts/run_milestone_09b.py formal
```

结果表包含：

| 实现/证据 | Accuracy | Macro-F1 | Cohen kappa |
|---|---:|---:|---:|
| 原仓库完整训练 checkpoint 的接口复核 | 0.7917977528 | 0.7010606829 | 0.7141909029 |
| 教学重构 seed 0 正式训练 | 0.7941573034 | 0.7055190242 | 0.7169520827 |

表后必须说明两行都不是论文十折均值，不能用它们证明教学重构“优于论文”或具有
多随机种子稳定性。

Codex 已检查 README 不再保留以下过期或不准确内容：

- Python 3.7 / PyTorch 1.8 是当前环境；
- 当前仓库直接提供 SHHS dataloader；
- 阅读 `README.pdf` 才能运行项目；
- 运行旧 `models/*/trainer.py` 是当前教学重构主入口；
- 本项目已经实现患者 EEG 隐私保护。

Codex 已同时创建 `AI_AGENT_HANDOFF.md`。它只保存稳定的协作规则、权威文件顺序、
项目事实和可复制的新会话提示；当前步骤、唯一下一步和训练进度仍只写在
`REPRODUCTION_PLAN.md`，避免不同智能体继承过期状态。

### 5.4 显式暂存步骤二文件

完成编辑后执行：

```powershell
git add -- .gitignore README.md environment.yml AI_AGENT_HANDOFF.md
```

`README.pdf` 的 staged 删除已由 `git rm --cached` 建立；`.git/info/exclude` 是本地
配置，不进入暂存区。不要使用 `git add .`。

## 6. 整步最终自测

所有编辑和暂存完成后，只在末尾统一运行一次。

### 6.1 环境与项目入口

```powershell
python -c "import importlib.metadata as m; names=['torch','numpy','scipy','matplotlib','tqdm','pyedflib','requests','openpyxl']; print({n:m.version(n) for n in names})"
python -m unittest discover -s tests -v
python scripts/download_sleep_edf_153.py --help
python scripts/build_sleep_edf_processed.py --help
python scripts/run_milestone_09b.py --help
```

这些命令不下载数据、不构建缓存、不更新模型。

### 6.2 文档和暂存边界

```powershell
git -c core.whitespace=cr-at-eol diff --cached --check
git --no-pager -c core.quotePath=false diff --cached --name-status
git -c core.quotePath=false status --short

rg -n "Python 3\.7|Pytorch 1\.8|README\.pdf|models/(raw|wavelet|merge)/trainer\.py" README.md
```

最后一条 `rg` 预期没有输出；若只是为了说明“这些旧入口不再使用”而主动引用，
必须让上下文明确是历史差异，不得让读者误认为仍是当前命令。

### 6.3 从精确暂存树构造干净预览

当前还没有 commit，不能直接用普通 `git archive HEAD`。`git write-tree` 会把现有
暂存区写成一个匿名 tree，既不 commit，也不改变分支；随后从该 tree 解压的目录
就是“如果现在提交，新读者会拿到什么”。

```powershell
$tree = git write-tree
$previewRoot = Join-Path ([IO.Path]::GetTempPath()) (
  "fftcn-preview-" + [guid]::NewGuid().ToString("N")
)
$previewRepo = Join-Path $previewRoot "repo"
$archive = Join-Path $previewRoot "snapshot.zip"

New-Item -ItemType Directory -Path $previewRepo | Out-Null
git archive --format=zip --output $archive $tree
Expand-Archive -LiteralPath $archive -DestinationPath $previewRepo

Push-Location $previewRepo
try {
  python -m unittest discover -s tests -v
  python scripts/download_sleep_edf_153.py --help | Out-Null
  python scripts/build_sleep_edf_processed.py --help | Out-Null
  python scripts/run_milestone_09b.py --help | Out-Null

  $required = @(
    "README.md",
    "AI_AGENT_HANDOFF.md",
    "environment.yml",
    "LICENSE",
    "data",
    "models",
    "training",
    "evaluation",
    "scripts",
    "tests",
    "learning_guides",
    "reproduction_artifacts/milestone_09b/validation_summary.json"
  )
  $missing = $required | Where-Object { -not (Test-Path -LiteralPath $_) }
  $missing
  Test-Path -LiteralPath "README.pdf"
  Test-Path -LiteralPath "test01.py"
}
finally {
  Pop-Location
}

"preview=$previewRepo"
```

预期：

1. 当前目录和预览目录的短测试都通过；
2. `$missing` 没有输出；
3. 两次 `Test-Path` 都输出 `False`；
4. 本地原目录中的 `README.pdf` 和 `test01.py` 仍存在；
5. 没有运行任何下载、缓存或训练。

预览目录放在系统临时目录，本步骤不要求删除它。

这里使用 ZIP 而不是 Windows 自带 `tar`：后者可能把 Git archive 中的 UTF-8 中文
路径 `背景/` 解码成乱码，使 README 在预览目录中的相对链接失效。ZIP 配合
`Expand-Archive` 能保留该目录名；把研究路线文件列入 `$required` 也会直接暴露
这类编码问题。

## 7. 唯一交回点与综合检查范围

完成第 5–6 节后，只需告诉 Codex：

> 步骤二已完成，请进行一次综合检查。

Codex 将一次性核对：

1. README 事实、命令、指标、AI 辅助披露、归属和结论边界；
2. `AI_AGENT_HANDOFF.md` 是否能让新智能体从现行文件重建上下文且不复制动态状态；
3. `environment.yml` 与已验证 `d21` 环境；
4. `README.pdf`/`test01.py` 是否按公开与本机边界正确分类；
5. 当前工作区和匿名暂存树的短测试与 CLI；
6. 是否可以进入步骤三。

这不是 9C 正式验收；整个里程碑仍在步骤三完成后统一验收。

## 8. 本步骤禁止动作

- 不执行 `git add .`；
- 不删除本地 `README.pdf` 或 `test01.py`；
- 不真正下载 Sleep-EDF；
- 不构建全量预处理或训练缓存；
- 不启动 overfit、epoch、resume 或 formal 训练；
- 不创建/修改 remote；
- 不 commit、push、tag 或 release。
