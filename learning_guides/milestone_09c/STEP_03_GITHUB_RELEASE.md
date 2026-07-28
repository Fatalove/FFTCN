# 里程碑 9C 步骤三：一次 CPU CI 与 v0.1 发布

本步骤只完成一次公开交付，不再增加练习分支、Pull Request 或第二轮 CI。

本指南中的仓库、邮箱和环境命令均使用通用名称或占位符；个人 Git 客户端、代理、
证书和本机绝对路径不属于项目内容，不应写入公开仓库。

剩余主线只有六步：

```text
确认公开 commit 身份
  → 初始化当前终端的 Git 联网组件
  → 阅读 CPU CI
  → 本地自测并 commit
  → push main，等待一次 CI
  → 创建 v0.1 tag 和 Release
```

全部完成后只申请一次 **里程碑 9C 整体验收**。

## 1. CPU CI 在做什么

CI 是 Continuous Integration（持续集成）。可以把本项目的 CPU CI 理解为：

> GitHub 临时提供一台没有本机数据的全新 CPU 电脑，自动检查公开代码的最小行为。

`main` 被 push 后，它依次完成：

1. 下载这次公开 commit；
2. 安装 Python 3.10、PyTorch 1.12.1 CPU 版和直接依赖；
3. 运行无需 Sleep-EDF 数据的短测试；
4. 检查三个公开命令能否显示 `--help`。

| CI 会做 | CI 不会做 |
|---|---|
| 安装 CPU 依赖 | 不使用 GPU |
| 运行临时数组和小模型测试 | 不下载 Sleep-EDF |
| 检查 Python/CLI 接口 | 不构建正式缓存 |
| 留下一份公开通过/失败日志 | 不训练模型、不重复正式 test |

它不是 demo，也不是模型的一部分。它只回答：换到一台干净电脑后，公开仓库的最小
代码检查还能不能运行。步骤二的干净预览曾发现 4 项测试暗中读取本机
`datasets/`，这正是保留一次 CPU CI 的实际理由。

## 2. 只需要理解的五个 Git 概念

| 名称 | 本步骤中的含义 |
|---|---|
| `commit` | 把当前暂存快照保存为本地历史 |
| `push` | 把本地 commit 上传到用户自己的 fork |
| `main` | 本次直接发布的主分支 |
| `v0.1 tag` | 给通过 CI 的 commit 一个固定版本名 |
| `Release` | 在 GitHub 上给 `v0.1` 建立说明和下载页 |

当前 remote 已经配置正确，不要重复执行 `remote rename` 或 `remote add`。

## 3. 本步骤的数据流

```text
本地暂存快照
    │
  commit
    ↓
本地 main ──push──> 用户 fork 的 main
                         │
                         └──自动运行一次 CPU CI
                                   │
                                通过后
                                   ↓
                            v0.1 tag + Release
```

## 4. 完整用户操作

### 4.1 决定公开 commit 身份

```powershell
# 读取未来公开 commit 中显示的作者姓名；这条命令不修改任何内容。
git config --get user.name

# 读取未来公开 commit 中记录的邮箱；公开仓库的 commit 历史会包含它。
git config --get user.email
```

这里的邮箱是 **Git commit 的作者元数据**，不会改变代码、测试、CI 或模型结果。

| 选择 | 效果 | 代价或风险 |
|---|---|---|
| 普通邮箱 | GitHub 能用已验证邮箱关联账号 | 邮箱会写进公开 Git 历史，可能被检索、复制或用于垃圾邮件 |
| GitHub `noreply` | GitHub 仍能关联账号和贡献记录，同时隐藏真实收件邮箱 | 别人不能通过 commit 元数据直接给真实邮箱发信 |

GitHub 页面上的 `Private` 只表示不在个人资料页公开该邮箱；如果本地 Git 仍使用普通
邮箱，它依然会进入新 commit 的元数据。更改配置也只影响**之后创建的新 commit**。

**本项目推荐使用 GitHub `noreply`。** 理由是仓库准备公开，而公开真实收件邮箱
对代码复现没有任何收益。请从 GitHub **Settings → Emails** 复制账号当前显示的
准确地址，不要在公开指南中记录个人地址。

```powershell
# --local 表示只修改当前 FFTCN 仓库，不影响其他 Git 仓库。
# 把占位符替换为 GitHub 页面提供的准确地址；它不是实际收件邮箱。
git config --local user.email "<GitHub 提供的 noreply 地址>"

# 再读一次，确认输出与上一行设置的地址完全相同。
git config --get user.email
```

如果确实需要别人从 commit 直接找到邮箱，也可以使用普通邮箱；这不是功能错误，
只是隐私取舍。对于公开教学仓库通常没有这个必要。

### 4.2 初始化当前终端的 Git 联网组件

公开项目只记录通用联网预检，不记录个人 Git 客户端、代理、证书或安装路径。

```powershell
# 显示当前 Git 版本，确认命令可用。
git --version

# 显示 fetch/push 地址，确认 origin 是用户 fork，upstream push 是 DISABLED。
git remote -v

# 只读取用户 fork 的远程 HEAD；成功会输出 SHA 和 HEAD，不会上传或修改文件。
git ls-remote origin HEAD
```

若最后一条命令失败，请保留完整输出并停止。根据自己的 Git 与网络环境在本机
配置，不要关闭 SSL 校验，也不要把代理、证书、账号或绝对路径提交到公开仓库。

### 4.3 阅读已经准备好的 CPU CI

```powershell
# 按原样读取工作流文件；UTF8 参数保证中文注释正常显示。
Get-Content -LiteralPath ".github/workflows/cpu-tests.yml" -Encoding UTF8
```

阅读时只抓住五层关系：

```text
push main
  → GitHub 读取工作流
  → 建立 Python CPU 环境
  → 安装依赖
  → 短测试与三个 --help
```

文件中的具体字段已经附有逐行中文注释。

### 4.4 运行唯一一轮本地自测，然后 commit

```powershell
# 运行 69 项无数据短测试；它不会启动正式训练。
python -m unittest discover -s tests -v

# 以下三条只解析参数并显示帮助，不下载数据、不建缓存。
python scripts/download_sleep_edf_153.py --help
python scripts/build_sleep_edf_processed.py --help
python scripts/run_milestone_09b.py --help

# 检查暂存文本是否有冲突标记或不合规空白；无输出表示通过。
git -c core.whitespace=cr-at-eol diff --cached --check

# 列出即将进入 commit 的文件及 A/M/D 状态，完成最后一次范围核对。
git --no-pager -c core.quotePath=false diff --cached --name-status

# 把当前暂存快照保存为本地 commit；此时还没有上传 GitHub。
git commit -m "feat: add Sleep-EDF-153 teaching reproduction"

# 检查 tracked 工作区是否干净；正常情况下没有输出。
git status --short

# 显示刚创建的 commit 短 SHA、标题和分支位置。
git log -1 --oneline --decorate
```

### 4.5 push main，并只等待一次 CPU CI

```powershell
# 必须输出用户 fork URL；这是即将接收上传的仓库。
git remote get-url --push origin

# 必须输出 DISABLED；这证明作者仓库不能被误推。
git remote get-url --push upstream

# 把本地 main 上传到用户 fork；-u 同时保存默认跟踪关系。
git push -u origin main
```

`push` 是本步骤第一次对外写入。完成后打开用户 fork 的 **Actions** 页面，只等待
`CPU short tests` 这一轮结束。CUDA 专用恢复测试会跳过，其余测试和三个 CLI 应通过。

如果失败，请保留 Actions URL 和第一条真实错误；不要 force push、删测试或随意升级
依赖来绕过。

### 4.6 CI 通过后创建 v0.1 tag 和 Release

```powershell
# 读取 GitHub 上最新的 origin/main 状态，不修改本地源码。
git fetch origin

# 分别显示本地 main 与远程 main 的完整 SHA；两行必须完全相同。
git rev-parse HEAD
git rev-parse origin/main

# 创建带说明的 v0.1 标签，固定指向当前已通过 CI 的 commit。
git tag -a v0.1 -m "FFTCN Sleep-EDF-153 teaching reproduction v0.1"

# 检查标签指向，不展开文件差异。
git show --no-patch --decorate v0.1

# 只把 v0.1 标签上传到用户 fork。
git push origin v0.1
```

随后在用户 fork 打开 **Releases → Draft a new release**：

- tag：选择 `v0.1`；
- target：确认是用户 fork 的 `main`；
- title：`v0.1 — Sleep-EDF-153 teaching reproduction`；
- 不勾选 prerelease；
- 不上传数据、checkpoint、论文或额外 ZIP/TAR。

Release notes：

```markdown
## FFTCN Sleep-EDF-153 teaching reproduction

This release contains a single-dataset educational reconstruction with
readable data, model, training, and evaluation code; no-data behavioral tests;
Chinese learning guides; and one fixed-seed full-run report.

The recorded seed-0 engineering run reached Accuracy 0.7941573034,
Macro-F1 0.7055190242, and Cohen kappa 0.7169520827 on 17,800 fixed test
positions. This is not a ten-fold, multi-dataset, or multi-seed paper result.

Datasets, caches, checkpoints, generated binaries, and paper copies are not
included.
```

## 5. 整步最终自测

```powershell
# 应输出 main。
git branch --show-current

# tracked 工作区应干净；正常情况下没有输出。
git status --short

# 再次确认 origin/upstream 没有颠倒。
git remote -v

# 查看 main 最新 commit 和 v0.1 标签。
git log -1 --oneline --decorate
git show --no-patch --decorate v0.1

# 分别保存本地、远程 main 和标签指向的完整 SHA。
$head = git rev-parse HEAD
$originMain = git rev-parse origin/main
$tagCommit = git rev-list -n 1 v0.1

# 打印三者；三个 SHA 必须完全相同。
"HEAD=$head"
"origin/main=$originMain"
"v0.1=$tagCommit"
```

## 6. 唯一交回点

全部完成后只发送一次：

```text
步骤三已完成，请进行里程碑 9C 整体验收。
main Actions URL: <URL>
release URL: <URL>
```

Codex 会同时完成步骤三检查和整个 9C 正式验收，不再增加重复检查。

## 7. 禁止动作

- 不重复配置 remote，不向 `upstream` push；
- 不增加 topic branch、Pull Request 或第二轮 CI；
- 不执行 `git add .`、`git push --force` 或 `git reset --hard`；
- 不关闭 SSL 校验，不公开 token、密码或未确认可公开的邮箱；
- 不发布数据、缓存、checkpoint、NPZ/XLSX、论文或生成二进制；
- 不下载 Sleep-EDF、不训练模型、不重复正式 test。
