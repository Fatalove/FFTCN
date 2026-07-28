# 里程碑 2：Sleep-EDF 数据读取

## 代码变更说明

本里程碑没有修改原仓库 `models/` 核心模块，新增和调整均位于数据边界、下载工具与测试层。

| 位置 | 变更 | 为什么需要 | 为什么这样改 | 验证方式 |
|---|---|---|---|---|
| `scripts/download_sleep_edf_153.py` | 新增 | 原仓库没有 Sleep-EDF 下载工具；8 GB 数据需要可恢复和可校验 | 使用官方公开 S3、Range 断点续传、并发下载和逐文件 SHA-256 | 重跑同一命令显示 306 个文件均为 cached 且校验通过 |
| 下载清单来源 | 从 `RECORDS` 改为 `SHA256SUMS.txt` | 实测官方 `RECORDS` 只列 153 个 PSG，不列 Hypnogram | 从完整校验清单筛选 `sleep-cassette/*.edf`，并断言 153+153 | 最终 PSG=153、Hypnogram=153、未配对=0 |
| `data/sleep_edf_reader.py` | 新增并由用户完成核心实现 | 原 Loader 只读 `.npz`，没有 EDF/Hypnogram 对齐接口 | 用 dataclass 固定路径、身份、采样率和 `[N,1,3000]` 契约；先展开原始标签，再按 PSG 有效长度裁齐 | `tests/test_sleep_edf_reader.py` 与 153 条记录时间轴扫描 |
| `tests/test_sleep_edf_reader.py` | 新增 | 仅凭打印结果无法防止配对、形状或时长回归 | 用 SC4001 的已知事实锁定 153 对、78 人、100 Hz 和 2650 段 | `python -m unittest tests.test_sleep_edf_reader -v` |
| `pyEDFlib==0.1.42` | 环境补充 | Python 标准库和原仓库均不能解析 EDF+ annotations | 使用专用 EDF Reader，固定版本避免环境漂移 | 读取 SC4001 通道头、3000 点信号和 154 条 annotations |

可迁移原则：

1. 数据下载完成不等于数据可信，必须使用发布方提供的校验和；
2. 网站的索引清单可能只面向主记录，配套标注要从完整 manifest 复核；
3. 断点续传只能由一个进程拥有同一个 `.part` 文件。若多个进程写同一路径，文件即使大小正确也可能哈希失败；
4. 读取器先输出“未清洗但严格对齐”的稳定契约，标签合并和裁剪放到下一层，避免一个函数同时承担解析、清洗和划分。

## 数据来源与范围

- 来源：PhysioNet Sleep-EDF Database Expanded v1.0.0；
- 本项目范围：`sleep-cassette`，即 153 条 PSG 与 153 个配套 Hypnogram；
- 实际受试者数：78；
- 本地目录：`datasets/sleep-edf-expanded-1.0.0/sleep-cassette/`；
- 下载器：`scripts/download_sleep_edf_153.py`；
- 完整下载已按官方 `SHA256SUMS.txt` 逐文件校验。

## 用户练习

完成 `data/sleep_edf_reader.py` 中的：

1. `discover_records`：按记录 ID 配对 PSG 与 Hypnogram；
2. `read_record`：读取 Fpz-Cz，展开标注，并按 PSG 时长裁齐为 30 秒段。

运行验收：

```powershell
python -m unittest tests.test_sleep_edf_reader -v
```

当前已知对齐事实：`SC4001` 的 PSG 为 79,500 秒，而 Hypnogram 覆盖到
86,400 秒；末尾超出 PSG 的 `Sleep stage ?` 标注必须被裁掉。

这里选择“按 PSG 有效完整时段裁标注”，而不是用 Hypnogram 总时长扩展 EEG。原因是标注可以覆盖信号外的未知区间，但不存在的 EEG 不能通过填零伪造；这一决定会影响样本数，因此必须由测试固定。

## 2026-07-06 验收结果

- 状态：通过；Codex 未修改用户完成的读取器核心代码；
- 单元测试：2/2 通过；
- 全集轻量扫描：153 条记录、78 名受试者，配对、通道、采样率、起始时间和标注覆盖问题均为 0；
- 全部信号长度可整分为 3000，全部 annotation duration 为 30 秒整数倍；
- 151 条记录存在标注延伸到 PSG 末尾之外的情况，当前实现均正确裁掉，且有效 PSG 范围内无标签空洞或重叠；
- 完整读取抽查：SC4001 为 `[2650,1,3000]`，SC4822 为 `[2810,1,3000]`。

非阻塞改进建议：当前输入校验使用 `assert`，在 `python -O` 下会被移除；EDF Reader 也应在异常路径使用 context manager 自动关闭。这些不影响本里程碑数据集上的验收，但正式工程化时应改为显式异常与 `with pyedflib.EdfReader(...)`，并补充负向测试后再修改。
