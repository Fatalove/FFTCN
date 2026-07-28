# FFTCN 里程碑证据日志

本文件记录后续可复核节点的事实证据，不定义当前状态或下一步。当前状态和执行入口只以仓库根目录 `REPRODUCTION_PLAN.md` 为准。

历史完整日志见：`reproduction_artifacts/plan_history/REPRODUCTION_PLAN_FULL_2026-07-20.md`。

## 记录模板

```markdown
## YYYY-MM-DD - 里程碑 N：可复核节点

- 事实状态：
- 用户完成内容：
- Codex 审查结论：
- 执行命令：
- 测试或实验结果：
- 关键形状/指标：
- 实验与结论边界：
- 产物位置：
```

日志不得写成具有执行效力的“下一步”；需要改变当前工作时，直接覆盖 `REPRODUCTION_PLAN.md` 的当前工作块。

## 2026-07-21 - 里程碑 9B：步骤一自测闭环与步骤二教学交付

- 事实状态：里程碑 9B 仍为“等待用户练习”；当前内部步骤切换为 2/4 Dataset/DataLoader，不进行步骤级正式验收。
- 用户完成内容：在 `build_raw_label_cache()` 第一次扫描中加入 raw 精确 `[N,1,3000]` 检查，阻止 `[N,1,1]` 被 NumPy 静默广播写入固定缓存；未加入额外生产防御。
- Codex 审查结论：修改位置和条件正确；步骤一缓存测试 4/4、已完成模块与缓存回归 43/43 通过。全量 discover 中步骤二骨架的 4 个预期失败不属于步骤一回归。
- 步骤二教学范围：读取 manifest 记录边界、构造记录内非重叠序列位置、从 memmap 返回 raw/wave/both 模型张量、显式关闭 Windows memmap，并用局部 `torch.Generator` 构造可复现 DataLoader。
- 教学准备：新增独立步骤二教程；保留用户已完成的 `FullTrainingConfig.to_metadata()` 和部分 `load_record_spans()`；正式骨架、教程与独立参考的接口同步。步骤二参考 5/5 通过，教程 7 个 Python 代码块语法通过。
- 注释质量：按 `COMMENT_QUALITY_GATE.md` 从头审读完整代码和参考答案；记录/全局索引、T=1 与 T>1 轴、memmap 复制、dtype、输入元组顺序、Windows 句柄和局部随机源均在对应代码前解释；工程加固单独列出。
- 实验边界：没有构建 Sleep-EDF-153 全量缓存，没有启动训练，也没有生成 checkpoint 或新指标。
- 产物位置：`learning_guides/milestone_09b/STEP_02_DATASET_DATALOADER.md`、`learning_guides/milestone_09b/reference_data_loading.py`、`training/sleep_edf_full_run.py`、`tests/test_sleep_edf_full_run.py`、`reproduction_artifacts/milestone_09b/validation_summary.json`。

## 2026-07-21 - 里程碑 9B：步骤二教学材料核心化修订与行为自测完成

- 事实状态：里程碑 9B 仍为“等待用户练习”；步骤二行为自测完成，内部指针移到 3/4 完整训练编排的教学准备，不进行步骤级或里程碑级正式验收。
- 用户决定：`T=1` 只用于 raw/wave 单分支预训练，`both` 只用于 `T=50` 融合微调；空记录拒绝和缓存生产者已保证的重复形状检查不进入核心学习代码。
- 修改结果：测试不再构造固定训练链之外的 `T=1 + both`；教程和参考答案删除空区间检查、重复 ndim/通道形状检查、辅助材料清单与工程加固清单，保留跨记录边界、样本轴对齐、T=1 轴变换、dtype、memmap 释放和局部随机源等核心语义。
- 正确性边界：继续检查 raw/wave/labels 样本轴长度与 manifest 总长度，防止分阶段重建缓存后旧 wave 或边界文件造成样本错位；没有覆盖用户正式练习实现。
- 验证结果：独立参考答案 5/5、用户正式实现 5/5、教程 Python 代码块 7/7 可编译、全部既有回归 48/48 通过。该结果只表示步骤二自测完成。
- 实验边界：没有构建全量缓存、启动训练、生成 checkpoint 或产生新指标。
- 产物位置：`learning_guides/milestone_09b/STEP_02_DATASET_DATALOADER.md`、`learning_guides/milestone_09b/reference_data_loading.py`、`tests/test_sleep_edf_full_run.py`、`reproduction_artifacts/milestone_09b/validation_summary.json`。

## 2026-07-20 - 核心计划长期教学规则恢复审计

- 事实状态：里程碑 9B 仍为“等待用户练习”，当前内部步骤仍为 1/4；本次没有推进代码或实验。
- 用户完成内容：指出计划瘦身后可能遗失中文译本优先、避免过度防御和学习核心代码优先等长期规则，要求逐行解析旧版完整计划，并要求 Codex 不得无依据附和。
- Codex 审查结论：旧版 1,130 行已完整核对。中文译本优先原本仍在现行第 4.2 节；最小准确代码原则仍在但缺少可执行判据；“学习核心代码优先”确实没有成为醒目总则。还需纠正“核心仅等于模型层”的过窄理解：数据语义和训练语义在对应里程碑同样属于核心。
- 修改结果：现行第 2.3 节新增学习目标优先级、核心范围划分和正确性检查准入标准；第 2.4 节限制无证据的阻塞测试；第 4.2 节强化中文译本与图片型 PDF 的职责；第 8.2 节增加规则瘦身复核清单。
- 状态一致性复核：当前正式缓存测试为 3/4；唯一失败是错误 raw 形状 `[N,1,1]` 被 NumPy 广播后静默写入 `[N,1,3000]`。现行计划已从过期的“两个函数仍含 `NotImplementedError`”修正为“用户实现已存在，待补 raw 精确形状检查”；这仍只是内部步骤一自测，不是 9B 验收。
- 语义边界：没有修改正式用户代码、测试、数据、模型、9B 状态或实验结果；“工程加固（选读）”默认不进入正式验收。
- 产物位置：`REPRODUCTION_PLAN.md`、`reproduction_artifacts/plan_rule_recovery_audit_2026-07-20.md`。

## 2026-07-20 - 核心计划瘦身与约束消歧

- 事实状态：里程碑 9B 仍为“等待用户练习”，没有改变任何里程碑完成状态。
- 用户完成内容：指出核心约束文件随项目推进不断膨胀，Codex 已开始违反里程碑整体验收等明显规则，要求详细解析并优化。
- Codex 审查结论：旧计划 1,130 行、47,516 个字符；约 80% 内容是详细里程碑手册和历史日志。旧日志中存在已经失效的里程碑 10 入口、逐函数检查要求和被推翻的内部命名约束，与现行规则并列，构成主要干扰源。
- 修改结果：旧文件完整归档；现行主文件压缩为 252 行、9,081 个字符，只保留当前工作、不可违反规范、稳定契约、状态表、9B 契约和维护规则。
- 语义边界：没有修改数据、模型、正式用户代码、测试或实验结果；只重构文档权限和信息层级。
- 新发现：中文译文路径原有重复 `01` 已纠正；明确区分 18,036 个处理后 test epoch 与 `T=50` 截尾后的 17,800 个评估位置；记录内部时间空洞策略登记为待用户决定，未擅自选择。
- 验证结果：主文件列出的现有本地路径全部可访问；前序行为回归 39/39、缓存独立参考 3/3、Dataset/DataLoader 独立参考 5/5 通过。后两组结果只证明教学材料的参考契约有效，不代表用户正式 9B 实现已经完成。
- 产物位置：`REPRODUCTION_PLAN.md`、`reproduction_artifacts/plan_history/`、`reproduction_artifacts/plan_optimization_2026-07-20.md`。

## 2026-07-20 - 里程碑 9B：序列空洞语义定案

- 事实状态：里程碑 9B 仍为“等待用户练习”；本次只确定数据序列语义，没有完成缓存或训练。
- 用户决定：根据论文、中文译本和源代码证据，采用原仓库兼容的紧凑序列；gap-aware 只作 9B 后可选对照。
- 证据：原 Loader 不保存时间位置，直接对每条记录截断并 `reshape`；`195,479` 个清洗 epoch 经每记录 `T=50` 截断后得到 `191,800` 个位置，五类计数与论文 Table I 完全一致。
- 限制：3,836 个非重叠序列中，110 个至少跨越一个被删除的内部无效时段，占约 `2.87%`；最终报告必须披露。
- 文献读取规则：优先检索中文译本；图片型原论文只用于公式、图表、版式和译文歧义核对。
- 产物位置：`reproduction_artifacts/milestone_09b/sequence_semantics_audit.md`。

## 2026-07-21 - 里程碑 9B：步骤三缓存方案定案与教学材料准备

- 事实状态：里程碑 9B 仍为“等待用户练习”；当前内部步骤为 3/4 完整训练编排，不进行步骤级或里程碑级正式验收。
- 用户决定：尽量遵循原仓库运行逻辑，采用全量保存平衡预训练缓存的方案；先在每条记录内部调用既有 `offset_resample_record()` 得到 raw/labels，再由同一批平衡 raw 计算 wave。
- 资源判断：这里的 `7.19 GiB` 指 D 盘磁盘存储，不是 RAM 或显存。检查时 D 盘剩余约 `77.83 GiB`，缓存建成后估计仍剩约 `70.64 GiB`，可接受；逐记录处理并写入 memmap，不把全量缓存同时放进内存。
- 教学准备：在不覆盖步骤二用户实现的前提下，为步骤三加入 9 个正式练习函数骨架（含 `_balanced_record_count()`、`_resolve_device()` 两个私有辅助函数）；新增遵循问题分析、必要基础、手工推演、完整核心代码、用户编码和行为自测顺序的教程，并建立独立参考答案与 4 条行为测试。
- 行为契约：逐记录偏移且 wave/原始 raw/标签严格对齐；epoch loss 按标签位置数加权；只用 validation 选 best，last 保存恢复状态；迁移 raw/wave validation-best 后才进入融合微调；步骤三不建立 test Loader。
- 验证结果：步骤三独立参考 4/4，通过参考入口运行的全套测试 52/52，教程 3/3 Python 代码块可编译；正式骨架、教程和参考答案的 9 个函数签名与调用关系同步。首次准备时漏放 `_balanced_record_count()`，后续全量复核又发现 `_resolve_device()` 未进入正式骨架；两项均已补齐并重新同步。
- 注释质量：按 `COMMENT_QUALITY_GATE.md` 从头审读完整代码与参考答案；记录区间、cache 对齐、位置加权、train/eval、学习率时序、best/last、恢复游标、`state_dict` 迁移和 memmap 释放均在对应代码前解释，没有加入与固定运行链无关的生产防御。
- 实验边界：没有实际生成约 `7.19 GiB` 的平衡缓存，没有启动长训练，没有生成新 checkpoint 或指标；正式步骤三代码仍有预期 `NotImplementedError`，等待用户实现。
- 产物位置：`training/sleep_edf_full_run.py`、`learning_guides/milestone_09b/STEP_03_TRAINING_ORCHESTRATION.md`、`learning_guides/milestone_09b/reference_training_orchestration.py`、`tests/test_sleep_edf_training_orchestration.py`、`reproduction_artifacts/milestone_09b/validation_summary.json`。

## 2026-07-22 - 里程碑 9B：步骤三因果式注释重审

- 事实状态：里程碑 9B 仍为“等待用户练习”，当前内部步骤仍为 3/4；本次只修订教学说明与遗漏骨架，不进行步骤级或里程碑级正式验收。
- 用户反馈：原注释只概括代码行为，没有逐一说明函数参数从运行链哪里传入、每个小模块怎样服务主线，以及 `position_count = 0` 等变量为什么必须单独存在。
- 注释标准修订：`COMMENT_QUALITY_GATE.md` 明确要求每个参数标注上游来源、返回值标注下游用途；每个中间变量说明来源、职责和删除后会失去的信息；累加器、计数器、游标和默认初值必须映射到对应算法公式或状态。
- 教程与参考修订：9 个步骤三函数都加入“参数来源/返回去向”；完整代码逐块解释缓存路径、两遍扫描、memmap 形状、记录游标、设备移动、损失分子/分母、checkpoint 字段、恢复游标、学习率时序、best/last、权重迁移和六个 Loader。`run_epoch()` 新增两个 batch 的数值手算，明确 `weighted_loss` 是分子、`position_count` 是分母。
- 正式练习边界：保留用户已经写出的 `_balanced_record_count()`、`build_balanced_pretrain_cache()` 和 `run_epoch()` 算法逻辑，只补充学习注释；另补齐先前遗漏的 `_resolve_device()` 私有辅助函数骨架。步骤三现在共 9 个练习函数。
- 验证结果：教程 3/3 Python 代码块可编译；正式练习、教程、参考的 9/9 函数签名与参数说明同步；独立参考入口全套测试 52/52 通过。
- 实验边界：没有生成全量平衡缓存、启动训练、生成 checkpoint 或新指标。
- 产物位置：`learning_guides/COMMENT_QUALITY_GATE.md`、`learning_guides/milestone_09b/STEP_03_TRAINING_ORCHESTRATION.md`、`learning_guides/milestone_09b/reference_training_orchestration.py`、`training/sleep_edf_full_run.py`、`REPRODUCTION_PLAN.md`、`reproduction_artifacts/milestone_09b/validation_summary.json`。

## 2026-07-22 - 里程碑 9B：步骤三用户自测完成与步骤四教学准备

- 事实状态：里程碑 9B 仍为“等待用户练习”；步骤三用户正式实现通过 4/4 行为自测，内部指针移到 4/4 预检与正式实验，不进行步骤级或里程碑级正式验收。
- 步骤四用户核心范围：固定 Python/NumPy/PyTorch 随机源；串联 train/validation/test 基础缓存和逐记录平衡 `pretrain_train`；重复同一 batch 过拟合；只加载融合 validation-best 评估 test；组装指标、曲线、环境、耗时、显存与 checkpoint 来源；运行正式实验并落盘报告。
- 接口修订：`run_full_training()` 增加可选 `resume_checkpoints` 参数并透传给已完成的两步训练编排；该修订不重写 epoch 或 checkpoint 算法。
- 教学准备：新增步骤四最小骨架、问题分析、必要基础、手工推演、带因果注释的完整代码、独立参考与 7 条行为测试；三份材料的 3 个结果 dataclass 与 6 个函数签名同步。
- 行为契约：三类随机源可重复；四组缓存样本轴对齐；同 batch 损失可下降；恢复映射不丢失；test 只加载指定融合 best 并关闭 memmap；报告包含紧凑时间线 110/3836 序列限制；正式入口每次只调用一次 test 评估。
- 资源预检：D 盘检查时约有 `75.69 GiB` 可用，预计全部新缓存约 `10.0 GiB`；690,096 个 CWT 按微型基准粗估约 `1.04 h`；4 GB RTX 3050 上默认 batch 单步峰值分配初测约为 raw `37.49 MiB`、wave `209.71 MiB`、fusion `2870.12 MiB`。这些只是进入真实预检的依据。
- 验证结果：步骤四独立参考 7/7，含全部前序的回归 59/59，教程 2/2 Python 代码块可编译，正式/教程/参考接口同步。
- 注释质量：按 `COMMENT_QUALITY_GATE.md` 从导入到返回结果完整审读；每个参数的上游、返回值下游、三类随机源、计时起点、损失轨迹、CUDA 峰值、best/last 分工、test 读取边界和 JSON 类型转换均说明“做什么与为什么”；没有加入 CLI、Schema、缓存哈希或 worker 工程加固。
- 实验边界：没有构建 Sleep-EDF-153 全量缓存，没有运行真实小样本/完整 epoch/恢复预检，没有启动 20/20/50 训练，没有生成新 checkpoint 或正式指标。
- 产物位置：`training/sleep_edf_formal_experiment.py`、`learning_guides/milestone_09b/STEP_04_PREFLIGHT_FORMAL_EXPERIMENT.md`、`learning_guides/milestone_09b/reference_formal_experiment.py`、`tests/test_sleep_edf_formal_experiment.py`、`training/sleep_edf_full_run.py`、`REPRODUCTION_PLAN.md`、`reproduction_artifacts/milestone_09b/validation_summary.json`。

## 2026-07-23 - 里程碑 9B：步骤四用户实现自测完成

- 事实状态：里程碑 9B 仍为“等待用户练习”；四个内部步骤的代码与局部行为自测已完成，但尚未进行真实数据预检、完整训练或整个里程碑验收。
- 用户实现：`training/sleep_edf_formal_experiment.py` 中固定随机源、全量缓存入口、同 batch 过拟合、融合 validation-best 的一次 test、正式报告和完整实验串联 6 个函数已完成。
- 审查修正：`overfit_single_batch()` 原先在训练循环结束后才重置 CUDA 峰值，会清除本次前向/反向/更新产生的真实峰值；现已把重置移到循环之前，并在全部更新后读取。
- 测试契约修正：原测试把 Dataset 和 `run_full_training()` 的位置参数/关键字参数写法当作必须一致；现通过公开签名绑定参数，只检查实际传值与调用语义。
- 验证结果：步骤四用户实现 7/7，全部前序回归 59/59；新增的 CUDA 顺序检查确认“重置 → 固定 batch 更新 → 读取峰值”。
- 实验边界：没有构建约 10.0 GiB 全量缓存，没有运行真实 batch、完整 epoch、恢复预检或 20/20/50 训练，没有生成新 checkpoint 或正式 test 指标。
- 下一运行块：先与用户核对准确路径和命令，再依次执行全量缓存、三种真实 batch 过拟合与显存、完整 epoch、恢复预检、固定 seed 0 完整训练和一次正式 test。

## 2026-07-23 - 里程碑 9B：用户运行入口与规则文件信息前置

- 事实状态：里程碑 9B 仍为“等待用户练习”；四个内部步骤的代码自测完成，真实全量缓存、模型更新、checkpoint 和正式指标仍未产生。
- 修改位置：新增 `scripts/run_milestone_09b.py`、`tests/test_milestone_09b_runner.py` 与 `learning_guides/milestone_09b/OPERATOR_GUIDE.md`；同步 `REPRODUCTION_PLAN.md`、步骤四指南、本目录 README、验证摘要和 `.gitignore`。
- 运行分工：全量缓存、真实 batch、完整 epoch、恢复预检和正式训练均由用户逐项启动；Codex 只准备固定命令、检查 JSON/history/checkpoint，并且没有执行任何长阶段。
- 固定入口：命令顺序为 `cache → overfit → epoch1 → resume2 → formal`；预检和正式输出严格分离，正式训练不得继承预检 checkpoint，正式中断只在原正式目录使用 `formal-resume`。
- 确定性修正：真实 batch 预检先把 `auto` 解析为实际 CPU/CUDA 设备；缓存完成状态同时核对 manifest、可读 NPY 的 shape/dtype 和记录对缓存的连续覆盖；`epoch1-resume` 与正式训练首个 checkpoint 前的恢复不再陷入无法重跑的状态。
- 恢复边界：现有 checkpoint 恢复模型、优化器、scheduler、epoch 和 history，但不保存随机源或 DataLoader generator 状态，因此不宣称恢复轨迹与不中断训练逐位一致；若 20/20/50 history 已完成而正式报告缺失，入口拒绝自动重试 test。
- 规则文件重构：整体目标与路线、当前工作块、唯一下一步和状态表移到文件开头；长期引导式编码、最小正确性检查、行为优先测试、中文译本优先和里程碑整体验收规则完整保留。
- 验证结果：运行入口聚焦测试 7/7、全部短测试 66/66；只读 `status` 确认处理后记录为 122/16/15，缓存、预检产物和正式报告均不存在。
- 实验边界：本节点只准备可审查的用户操作入口和文档，没有构建约 10 GiB 缓存、运行真实模型更新、生成 checkpoint 或评估 test。

## 2026-07-24 - 里程碑 9B：长阶段前台进度显示

- 用户反馈：首次运行 `cache` 后当前终端没有任何进度，原指南要求另开终端反复执行 `status`，无法直观区分长时计算与卡死；这不符合深度学习长任务的常规交互。
- 原因：运行入口先前只负责阶段分派和完成后摘要；缓存核心直到整段结束才写 manifest，训练核心直到每个 epoch 结束才写 history，入口没有把这些内部进度转换成前台显示。
- 修改位置：只修改辅助运行层 `scripts/run_milestone_09b.py`，没有改变缓存、CWT、模型、损失、优化器或 checkpoint 语义。缓存通过包装原 `morlet_cwt_epoch` 显示四段连续 epoch 进度；训练通过只读轮询既有 history 显示当前阶段和 epoch 总进度。
- 文档修订：`learning_guides/milestone_09b/OPERATOR_GUIDE.md` 将当前终端进度条设为默认方案，`status` 降为可选旁路检查；核心计划、步骤四指南、README 和验证摘要同步当前状态。
- 当前运行事实：用户于 2026-07-24 启动的 cache 进程仍在运行。只读快照确认 train/validation/test 基础缓存已写出；`pretrain_train` 快照由 67,059/494,617（13.56%）推进到 153,634/494,617（31.06%），证明进程没有卡住。完整缓存尚无 manifest 和 `cache_build.json`，不能视为完成。
- 当前进程边界：正在运行的 Python 已载入旧代码，无法在不中断计算的情况下注入新进度条；不应为显示目的重启。本次继续通过只读快照核对，之后启动的长阶段会直接显示新进度。
- 验证结果：进度包装与 history 读取新增 2 条测试；运行入口 9/9、全部短测试 68/68 通过。
- 实验边界：缓存仍在由用户进程构建；没有运行真实 batch 预检、完整训练 epoch、恢复预检或正式 test，没有生成教学重构 checkpoint 或新指标。

## 2026-07-24 - 里程碑 9B：全量缓存完成与剩余训练路径性能审计

- 事实状态：全量缓存已由用户完成；`cache_build.json` 记录实际用时 `2489.8038 s`。只读 `status` 确认 train/validation/test/pretrain_train 为 `154128/23315/18036/494617` 项，四组文件、shape、dtype 和记录覆盖均通过；里程碑仍为“等待用户练习”。
- 审计范围：只检查尚未运行的 overfit、完整 epoch、恢复、正式训练和 test 路径；没有删除或重建有效缓存，没有执行 `optimizer.step()`，没有生成 checkpoint 或评估 test。
- 源仓库对照：原 `data/loader.py` 同样使用默认 `num_workers=0`，原训练循环使用 batch 级 `tqdm`。重构代码的验证阶段已正确关闭梯度，测试预测及时移到 CPU，checkpoint 每轮写入规模较小，均未发现与缓存串行 CWT 同等级的低效。
- 真实缓存短测：Dataset 构造 raw/wave 各约 `0.6 s`，fusion 约 `0.006 s`；稳定 batch 准备耗时 raw/wave/fusion 约 `0.0217/0.0091/0.0614 s`。预热后、无优化器更新的 forward+backward 约 `0.0056/0.0297/2.1201 s`，参数最大变化为 `0`。
- 判断：raw 数据准备是后续可选优化点；wave 和尤其 fusion 主要受模型计算限制。保留固定 `num_workers=0` 和 batch size，不在取得 `overfit`、`epoch1` 真实证据前改动运行语义。训练仅按 epoch 前台更新属于可观察性改进候选，不阻塞下一项短预检。
- 唯一下一步：由用户运行 `scripts/run_milestone_09b.py overfit`；检查三段损失下降和真实峰值显存后，再决定是否需要在 `epoch1` 前处理任何阻塞项。

## 2026-07-24 - 里程碑 9B：训练进度改为 batch 级

- 用户决定：模型训练实际按 batch 前向、反向和更新，因此长训练进度也应与该执行单位一致，不再只等待整个 epoch 完成后更新。
- 实现位置：`fit_stage()` 分别用 `tqdm` 包装 train 和 validation Loader；描述包含阶段、数据用途和 `当前 epoch/总 epoch`，进度分母直接来自真实 Loader 的 batch 数，完成行保留该 epoch 的位置平均 loss。
- 冲突清理：删除运行入口中后台轮询 history 的 `_TrainingHistoryProgress` 及线程；`epoch1/resume2/formal` 直接调用完整训练，由核心编排显示 batch 进度，避免两组进度条同时刷新终端。
- 语义边界：没有修改 Dataset、batch 内容、shuffle、模型、损失、反向传播、优化器、scheduler、checkpoint 或恢复规则；缓存和真实模型参数均未改动。
- 同步范围：正式练习、步骤三教程、独立参考、运行手册、步骤四说明、行为测试和现行计划均已同步。
- 验证结果：步骤三正式实现 5/5、独立参考 5/5、运行入口 8/8、默认实现与参考入口下的全部短回归均为 68/68；步骤三教程 3/3 个 Python 代码块可编译。

## 2026-07-24 - 里程碑 9B：三种真实 batch 过拟合与显存预检通过

- 用户运行：`python scripts/run_milestone_09b.py overfit`，在 CUDA 上依次对 raw、wave、fusion 的固定真实首 batch 各更新 20 次；入口正常生成 `real_run/overfit.json`。
- 输入契约：raw 为 `[128,1,3000]`、标签 `[128]`；wave 为 `[128,1,30,60]`、标签 `[128]`；fusion 为 raw `[32,50,1,3000]`、wave `[32,50,1,30,60]`、标签 `[32,50]`。
- 损失结果：raw `1.727863 → 1.676829`，wave `1.757720 → 1.328712`，fusion `1.712032 → 1.547823`；60 个损失值全部有限，三段均满足约定的首末总体下降。raw 单步存在波动，但首末与前后五步均值方向一致，不违反训练模式下的随机层语义或当前预检契约。
- 显存结果：峰值分配显存为 raw `41.85 MiB`、wave `212.04 MiB`、fusion `2882.17 MiB`；fusion 约为 4096 MiB 的 70.37%。该指标不是整卡实时占用，但连续 20 次真实更新成功，证明默认 `B=32、T=50` 在当前环境可运行。
- 实验边界：overfit 不保存模型，尚未完成任何完整 epoch、checkpoint、恢复预检、正式训练或 test；里程碑 9B 仍为“等待用户练习”。
- 唯一下一步：由用户运行 `scripts/run_milestone_09b.py epoch1`，完成后三阶段分别核对 epoch 0 history、best.pt、last.pt、损失和 batch 进度。

## 2026-07-24 - 里程碑 9B：batch 进度实时显示累计平均损失

- 用户反馈：`epoch1` 的 batch 计数正常推进，但运行中的进度行没有损失指标；已完成的 raw/wave 行也没有稳定保留先前仅在 epoch 结束后设置的 postfix。
- 原因：`fit_stage()` 原先必须等待 `run_epoch()` 返回完整 epoch loss 后才调用 `set_postfix()`，因此处理中没有可显示的损失，结束后的单次刷新也不够可靠。
- 修正：`run_epoch()` 每完成一个 batch，便用当前 `weighted_loss / position_count` 更新 `avg_loss`；它是截至当前全部预测位置的累计平均交叉熵，与最终 epoch loss 使用同一公式。`refresh=False` 只更新字段，由 tqdm 按正常频率绘制，避免额外的逐 batch 终端刷新开销。
- 生命周期修正：`tqdm` 在被 `run_epoch()` 完整迭代后已经自动关闭；原先返回后再调用 `set_postfix()` 的两行并非只是“不够可靠”，而是处在无效生命周期中，现已从正式、教程和参考实现删除。测试要求每个 batch 在迭代器打开时更新，且最后一次 `avg_loss` 必须等于 history 的最终 epoch loss。
- 运行边界：用户已经启动的 `epoch1` 进程继续使用启动时载入的旧函数，不应为显示目的中止；本次修改从之后启动的 `resume2/formal` 生效，不改变当前进程产生的模型、损失、checkpoint 或 history。
- 验证结果：正式训练编排 5/5、独立参考 5/5，两个 Python 文件可编译，步骤三教程 3/3 个 Python 代码块可编译，验证摘要 JSON 有效。为避免与用户正在执行的 fusion GPU 训练争抢资源，本次没有重复运行已在上一节点通过的全套 68 条测试。

## 2026-07-24 - 里程碑 9B：三阶段 1 epoch 预检通过

- 用户运行：`python scripts/run_milestone_09b.py epoch1` 正常完成；raw/wave/fusion 的 train batch 数为 `3865/3865/95`，validation batch 数为 `183/183/15`，三阶段均生成 epoch 0 history、best.pt 和 last.pt。
- 损失结果：raw train/validation 为 `1.268184/1.454311`，wave 为 `1.015262/0.960737`，fusion 为 `1.424678/1.161234`；全部为有限值。本步骤只验证完整运行链，不以单个 epoch 判断最终收敛或泛化指标。
- checkpoint 核验：六个 checkpoint 均可在 CPU 加载，stage 和 epoch 字段正确；第一轮的 best 与 last 模型状态一致，checkpoint history 与磁盘 history、`epoch1.json` 一致。
- scheduler 核验：epoch 0 使用的基础学习率为 `1e-5`；保存到 last.pt 的下一轮学习率 raw/wave 为 `9.5e-6`，fusion 参数组为 `9.5e-6/9.5e-8/9.5e-8`，符合 gamma `0.95`。
- loss 显示复盘：本次启动进程使用旧实现，进度条只显示 batch 数。用户正确指出旧代码在 `run_epoch()` 返回后设置 postfix 时 tqdm 已自动关闭；当前磁盘代码已改为在每个 batch 内更新 `avg_loss`，并删除关闭后无效调用。
- 验证结果：修正后的默认实现全套短回归 68/68、独立参考训练编排 5/5、教程 3/3 Python 代码块可编译。
- 实验边界：尚未运行 resume2、正式 20/20/50 或 test；里程碑 9B 仍为“等待用户练习”。
- 唯一下一步：由用户运行 `scripts/run_milestone_09b.py resume2`，只新增 epoch 1，并核对 history `[0,1]`、连续学习率和实时 `avg_loss`。

## 2026-07-24 - 里程碑 9B：修复 CPU 断点恢复到 CUDA 的 Adam 状态设备

- 用户运行：`python scripts/run_milestone_09b.py resume2` 在 raw 第一个 batch 的 `optimizer.step()` 中终止，异常指出 `cuda:0` 与 `cpu` 张量混用。
- 原因：模型和 Adam 最初都在 CPU 创建；旧 `fit_stage()` 先恢复 optimizer，直到 `run_epoch()` 才执行 `model.to(cuda)`。PyTorch 恢复 Adam 状态时按照当时参数设备把 `exp_avg/exp_avg_sq` 放到 CPU，后来移动模型不会连带移动这些 optimizer 状态。
- 修正：`fit_stage()` 在任何 checkpoint 加载前先执行 `model.to(device)`，使 optimizer 恢复时模型参数、Adam 动量和后续梯度落在同一设备。正式实现、步骤三教程和独立参考同步，并补充原因注释。
- 产物边界：异常发生在第一个 raw batch 更新完成前，训练循环尚未保存本轮 checkpoint；没有生成 `resume2.json`，epoch 0 的 best/last/history 没有被覆盖，可继续作为恢复源。
- 验证结果：真实微型 CUDA 复现先确认旧顺序必现同一异常、修正顺序可完成 `step()`；新增 CPU checkpoint → CUDA 恢复测试确认 epoch `[0,1]` 且 Adam 一/二阶动量均在 CUDA。正式与独立参考聚焦测试各 6/6，两种入口的全部短回归各 69/69，教程 3/3 Python 代码块可编译。
- 实验边界：Codex 没有重跑真实 `resume2` 或启动正式训练；当前唯一下一步仍由用户重新运行同一条 `resume2` 命令。

## 2026-07-24 - 里程碑 9B：三阶段恢复到总轮数 2 预检通过

- 用户运行：`python scripts/run_milestone_09b.py resume2` 正常完成；raw/wave/fusion 都从已核验的 epoch 0 断点只新增 epoch 1，生成 `real_run/resume2.json`。
- epoch 1 损失：raw train/validation 为 `1.108070/1.411243`，wave 为 `0.897982/0.914962`，fusion 为 `0.947678/0.882611`；六条前台进度均显示并保留最终累计位置平均 `avg_loss`。
- history 核验：三阶段 report 与磁盘 history 都严格为 `[0,1]`，两轮损失均有限；epoch 1 的 train 和 validation loss 均低于各自 epoch 0。
- checkpoint 核验：三阶段 best.pt 和 last.pt 均可在 CPU 读取且为 epoch 1；三段 validation 都在 epoch 1 改善，因此 best/last 模型状态相同。best、last、磁盘 history 与 `resume2.json` 完全一致。
- scheduler 核验：三段 `scheduler.last_epoch` 均为 2；下一轮 raw/wave 学习率为 `9.025e-6`，fusion 参数组为 `9.025e-6/9.025e-8/9.025e-8`，符合连续两次乘以 `0.95`。
- 正式边界：`formal_seed_0` 当前不存在；正式训练将从全新 seed 0 初始化开始，不加载任何预检 checkpoint。尚未产生正式 checkpoint、正式 test 或新正式指标。
- 当前唯一下一步：由用户运行 `scripts/run_milestone_09b.py formal`，完成固定 20/20/50 训练并只评估一次 fusion validation-best。

## 2026-07-24 - 跨对话恢复规则与长期记忆补充

- 用户要求：在新开对话前回顾本次长会话，把不会随实验进度变化的约束和核心规则写入长期位置，防止后续依赖旧聊天片段猜测。
- 现行规则补充：`REPRODUCTION_PLAN.md` 明确要求新对话、上下文压缩或任务恢复后先完整读取本文件，重点核对 1.2 当前工作块、1.3 状态表及其引用产物；文件可以确定进度时不得反问用户。
- 长期记忆补充：新增 ad-hoc 记忆说明，保存源仓库运行语义优先、长训练由用户启动、长任务在当前终端展示 batch 级实时损失、教学注释的参数上下游与变量职责、核心优先及跨设备 Adam 恢复顺序。
- 信息分工：会变化的 ACTIVE_USER_STAGE 和唯一下一步只保留在现行计划与操作手册，不复制成长期静态进度；长期记忆只保存“重新读取现行计划”的恢复规则，避免过期快照覆盖真实状态。
- 实验边界：本次只更新规则、记忆和日志，没有启动正式训练、更新模型或产生新实验指标；当前唯一下一步仍为用户运行 `scripts/run_milestone_09b.py formal`。

## 2026-07-24 - 里程碑 9B：训练进度视觉层级与累计 ACC

- 用户反馈：现有进度行把 `wave_pretrain`、train/validation 和 epoch 全部平铺在每一行，宽屏动态条形过长；信息虽完整但重复、难扫描。用户指出原仓库的阶段分隔标题、行首 `EPOCH[x/y]` 和实时 ACC 更清晰。
- 原仓库核对：`models/*/trainer.py` 在阶段开始前打印等号分隔标题，`models/base/model.py` 与 `models/merge/model.py` 的训练条以 `EPOCH[x/y]` 开头，并显示当前 batch 的 acc/loss；不足是固定只显示训练条，且 postfix 是单个 batch 指标而非整轮累计指标。
- 展示修正：每个 raw/wave/fusion 阶段只打印一次等号分隔标题；train/validation 行统一以 `Epoch 当前/总数 | Train/Valid` 开头；进度条固定为 100 列，不再随宽屏无限拉长。
- 指标修正：进度 postfix 改为截至当前 batch 的累计位置平均 `loss` 与累计位置 `acc`。`run_classification_step()` 通过可选兼容参数返回同一次前向的正确位置数，默认仍返回原有 float loss；因此没有第二次前向，也未改变模型、梯度、优化器或 checkpoint 语义。
- 同步范围：正式实现、里程碑 8 的公开接口教程/参考、里程碑 9B 的步骤三教程/参考、操作手册、行为测试、验证摘要和现行计划均已同步。
- 验证结果：四个 Python 文件通过编译；正式实现相关测试 12/12、步骤三独立参考 6/6 通过。行为测试用确定性预测验证两个训练 batch 后显示累计 50% 而不是末 batch 0%，并核对 loss、标题、行首与固定宽度。
- 当前运行边界：用户已经启动的正式 Python 进程仍使用启动时加载的旧展示代码，不应为界面改动中止。只读快照确认 raw 20/20、wave 至少 10/20、fusion 0/50；新样式只会在未来新进程或必要的 `formal-resume` 中生效。

## 2026-07-25 - 里程碑 9B：正式训练、一次 test 与整体验收通过

- 正式运行：用户启动的独立 seed 0 正式目录完成 raw/wave/fusion `20/20/50` 轮；磁盘 history、报告内 history、best.pt 与 last.pt 均可读且字段一致。validation-best 分别位于 raw epoch 19、wave epoch 18、fusion epoch 48，对应 validation loss 为 `0.938977/0.698592/0.458227`。
- 模型选择与 test 边界：raw/wave validation-best 通过 `state_dict` 迁移后才进行 fusion 微调；最终只加载 fusion validation-best，并在全部模型选择结束后对固定 test 正式评估一次。报告记录 `formal_test_evaluations=1`，没有加载预检 checkpoint。
- 正式指标：`17,800` 个有效 test 位置上的 Accuracy 为 `0.7941573034`、Macro-F1 为 `0.7055190242`、Cohen kappa 为 `0.7169520827`；5×5 混淆矩阵总和、每类 support、precision、recall、F1 以及三个总体指标均从报告矩阵独立重算一致。
- 数据与实验契约：train/validation/test/pretrain_train 四组缓存样本轴、dtype、manifest 覆盖和 T=50 记录边界通过；受试者 `62/8/8` 两两不相交。报告披露 3,836 个紧凑序列中 110 个跨被删除内部空洞（2.87%），并明确结果只是 Sleep-EDF-153 单数据集、单种子工程复现。
- 运行事实：全量缓存 `2489.8038 s`，正式训练 `2657.7651 s`，峰值 CUDA 分配 `2903.06 MiB`；正式进程已结束，正式报告存在，禁止重复 test。
- 接口同步修正：整体验收发现 `reference_data_loading.py` 的 `FullTrainingConfig` 遗漏正式接口已有的 `offset_samples`。已只在独立参考中补回该字段及其来源/用途注释，没有改变训练、模型、缓存或实验语义；步骤二独立参考随后 5/5 通过。
- 最终验证：项目固定 `d21` 环境下全部短回归 69/69；四个独立参考分别 4/4、5/5、6/6、7/7；四份指南合计 17 个 Python 代码块可编译；21 个公开/练习 callable 契约、`SleepEDFSequenceDataset.__init__` 与 `FullTrainingConfig` 字段在正式实现、完整教程和独立参考间同步。既有四步人工注释质量门槛继续成立。
- 验收结论：第 2.4 节原 8 项统一条件全部满足，里程碑 9B 标记为“已完成”。当前工作块切换为 9C“GitHub 项目化与 v0.1”，但 9C 范围和整体验收条件尚未共同固定，因此保持“未开始”，没有创建远程仓库或发布版本。
- 证据位置：`reproduction_artifacts/milestone_09b/formal_seed_0/formal_test_report.json`、`reproduction_artifacts/milestone_09b/validation_summary.json`、`reproduction_artifacts/milestone_09b/sequence_semantics_audit.md`。

## 2026-07-27 - 里程碑内部步骤改为完整交付与单次综合检查

- 用户纠正：里程碑可以模块化为多个内部步骤，但每个步骤的指导文件必须一次性给全，不能把子操作拆成频繁的强制交回检查。
- 协议修正：每个内部步骤完整覆盖问题分析、必要基础、手工推演、全部用户操作、最终自测、预期结果与禁止动作；用户完成整步后只申请一次步骤综合检查，检查项由 Codex 按步骤契约集中决定。
- 状态边界：步骤综合检查只决定是否进入下一内部步骤，不改变里程碑正式状态；整个里程碑仍只接受一次统一正式验收。
- 9C 步骤一：原先分开的 `.gitignore`、暂存区复位和选择性重建暂存快照已合并为一个完整闭环；先前 `.gitignore` 查看仅算过程反馈，不作为独立检查点。

## 2026-07-27 - 里程碑 9C 步骤一：PowerShell 命令与换行检查修正

- 现象：暂存区复位后漏写 `git`，PowerShell 将 `diff` 解析为 `Compare-Object`；随后 `git add` 输出大量 LF→CRLF 提示。
- 诊断：错误的 `diff` 调用没有修改仓库；`git add` 已成功，114 个暂存候选中没有禁发路径。换行提示来自用户全局 `core.autocrlf=true`，不是添加失败。
- 修正：步骤一指南改用短命令和 `--no-pager`，明确每条子命令必须以 `git` 开头；空白检查以单次命令参数 `core.whitespace=cr-at-eol` 兼容上游已有 CRLF，不改变全局 Git 配置。
- 清理：仅机械删除真实的文件尾多余空行和尾随空格；Markdown 原有强制换行改为显式 `<br>` 或段落分隔，不改变代码、实验或教学语义。

## 2026-07-27 - 里程碑 9C 步骤一综合检查通过

- 暂存边界：114 个候选全部位于固定发布根目录；禁发路径、意外根目录和二进制差异均为 0，最大候选文件约 92 KB。
- ignore 行为：IDE/agent/Codex 配置、数据、导览输出、原仓库生成输出、缓存、checkpoint、NPZ/XLSX 和论文 PDF/DOCX 代表路径全部命中；源码、模型、训练、评估、脚本、测试、指南、计划、许可证和正式 JSON 均未误伤。
- 工作区保留：代表性 Sleep-EDF 数据清单、原仓库 checkpoint、9B 正式 checkpoint 和论文 PDF 仍在磁盘；没有用 ignore 或暂存区复位删除本地产物。
- 质量检查：兼容上游 CRLF 的 staged whitespace check 通过，疑似凭据扫描为 0；唯一未暂存项 `test01.py` 已归类为本机临时练习文件。
- Git 边界：当前分支仍为 `main`，`origin` 仍指向作者 `bjm-123/FFTCN`；没有 commit、remote 修改、push、tag 或 release。
- 结论：步骤一综合检查通过，可以进入步骤二；这是内部步骤推进，不改变 9C 的正式状态。

## 2026-07-27 - 里程碑 9C 步骤二：根 README 重写

- 委托边界：用户明确表示暂时无法独立重写 README，并要求 Codex 根据项目具体情况完成；README 属于项目说明，不替代用户的模型核心代码练习。
- 内容重构：删除上游过期的 Python 3.7/PyTorch 1.8、SHHS dataloader、README.pdf 和旧 trainer 主入口说明；改为中文主文与简短英文概述。
- 事实范围：现稿覆盖 Sleep-EDF-153 单数据集定位、模型/张量契约、62/8/8 划分、环境、无数据短检查、官方数据下载、预处理、分阶段训练、正式指标、论文/代码差异、证据链、上游归属和 MIT 许可证。
- 结论边界：明确 9B 指标来自 seed 0 单次完整训练，test 为 17,800 个有效位置且只评估一次；不表述为论文十折、多数据集、多种子稳定性或隐私保护结论。
- 验证：所有相对链接存在；无本机绝对路径和过期入口措辞；正式 JSON 的 Accuracy/Macro-F1/kappa 与 README 一致；固定 `d21` 环境下无数据短回归 69/69 通过。
- 操作边界：没有下载数据、构建缓存、启动训练、重复 test、修改 remote、commit 或 push。

## 2026-07-27 - 里程碑 9C 步骤二：AI 辅助披露与跨智能体交接

- 用户判断：仓库现有测试、规则和证据直接记录 Codex 分工，不应为了表面整洁而
  隐藏；希望在 README 主动说明其角色，并为 Codex、Claude 等智能体提供可复用
  的项目交接文件。
- 披露边界：README 现已说明 Codex 用于计划、教学、文档、测试、调试、确定性
  工程修正和产物检查；同时明确用户负责范围、核心实现审核、长实验启动、结果
  复核和最终责任，智能体不是作者或最终责任主体，生成文字也不是独立实验依据。
- 交接实现：新增 `AI_AGENT_HANDOFF.md`，以现行计划为动态状态唯一来源，规定
  权威文件顺序、新会话启动、引导式编码、里程碑整体验收、长训练和发布安全边界，
  并提供可直接复制给不同智能体的启动提示。
- 诚信边界：公开仓库披露用于提高可审计性，但不能替代课程、学校、会议或期刊
  另行要求的 AI 使用声明；不得把交接文件描述为自动传递隐藏聊天记忆。
- 操作边界：本次只修改步骤二文档，没有下载数据、构建缓存、启动训练、重复
  test、修改 remote、commit 或 push。

## 2026-07-27 - 里程碑 9C 步骤二综合检查通过

- 环境与入口：固定 `d21` 为 Python 3.10.13、PyTorch 1.12.1、NumPy 1.22.3、
  SciPy 1.10.1；其余直接依赖与 `environment.yml` 一致。当前目录 69/69 短回归
  通过，下载、预处理和 9B 运行器的三个 `--help` 均正常。
- 发现阻塞：第一次匿名暂存树不含本机 `datasets/`，因此 reader/preprocessor
  的 4 项测试出现 1 个失败、3 个错误；它们此前借助本机正式 EDF 才能通过，与
  README 的“无需数据短检查”相冲突。
- 测试修正：reader 改用临时 153 对文件名和 mock `pyedflib.EdfReader`，继续验证
  78 名受试者、PSG/Hypnogram 配对、100 Hz、30 秒切段、标注展开及关闭句柄；
  preprocessor 改用内存标签和 153 条合成路径，保留 841 个参考位置、五类计数、
  62/8/8 可复现受试者划分和互斥契约。正式数据代码与既有实验产物没有改动。
- Windows 预览修正：匿名 tree 的 Git 路径 `背景/` 正确，但 Windows `tar` 会把
  UTF-8 中文目录解成乱码；步骤二指南改用 ZIP + `Expand-Archive`，并把研究路线
  文件加入必需清单。
- 最终预览：匿名 tree `35f9aadef748d5cb4a1eebd4974e3ab69080949e` 的 ZIP
  预览 69/69，三个 CLI 正常，必需项缺失 0、README/AI 交接断链 0、禁发归档 0，
  `README.pdf` 和 `test01.py` 均不在公开快照但仍保留在本机。
- Git 与运行边界：发布候选无禁发路径、二进制新增、未暂存 tracked 修改或空白
  错误；分支仍为 `main`，`origin` 仍是作者仓库。没有下载、缓存、训练、正式
  test、remote 修改、commit、push、tag 或 Release。
- 结论：步骤二综合检查通过，可以进入步骤三；这只推进内部步骤，不改变 9C 的
  正式状态。步骤三完整材料为 `learning_guides/milestone_09c/STEP_03_GITHUB_RELEASE.md`。

## 2026-07-27 - 里程碑 9C 步骤三：发布链去除过度工程化

- 用户反馈：不理解 CPU CI 的实际作用，并指出步骤三包含 topic branch、两次 CI、
  fast-forward 等过多操作；要求所有命令提供足够详细的中文注释。
- 边界复核：最小 CPU CI 是现行 9C 统一通过条件，用于证明公开快照在没有本机
  数据的环境中可运行；它不是 demo、模型模块、训练或正式 test。该检查本身保留。
- 过度部分：topic branch、分支 CI、fast-forward main 和第二次 main CI 不属于
  9C 固定通过条件，对已经验收的个人 fork 首次发布没有增加必要证据，已从主线
  删除。
- 简化主线：创建 fork → 配置 `origin/upstream` → 阅读一个 CI 文件 → 一次
  commit → 一次 `main` push/CI → `v0.1` tag/Release。
- 教学改进：Codex 直接创建 `.github/workflows/cpu-tests.yml`，对触发条件、权限、
  runner、超时、UTF-8、依赖安装、短测试和 CLI 检查逐项添加中文注释；步骤三
  PowerShell/Git 命令也逐行解释作用、是否写入本地及是否产生外部动作。
- 状态边界：没有创建 fork、修改 remote、commit、push、tag 或 Release；9C
  仍为“等待用户练习”。

## 2026-07-27 - 里程碑 9C 步骤三：同步真实 remote 与 Git 联网边界

- 现场状态：用户已经创建公开 fork；`origin` 指向用户 fork，`upstream` fetch
  指向作者仓库且 push 为 `DISABLED`。这些外部与 remote 操作由用户完成，不是
  Codex 代为执行。
- Git 状态：当前仍位于 `main` 的作者起始 commit，发布候选均在暂存区；尚未
  commit、push、tag 或创建 Release。
- 身份边界：当前 Git 邮箱是普通邮箱而非 `noreply`。指南要求用户先决定是否
  接受其进入公开历史；若不接受，必须从 GitHub 页面复制准确地址。日志不记录
  该私人邮箱的具体内容。
- 教学同步：步骤三不再要求重复创建 fork 或配置 remotes；剩余流程从公开邮箱
  选择开始，随后是联网初始化、阅读最小 CI、一次自测/commit、一次 push/CI 和
  `v0.1` Release。9C 仍为“等待用户练习”。

## 2026-07-27 - 里程碑 9C 步骤三：补全选择理由

- 用户反馈：需要用户思考和选择时，只给选项而不解释差异、影响和推荐理由，无法
  满足引导式编码要求。
- 邮箱分析：普通邮箱与 GitHub `noreply` 不影响代码、CI 或模型结果，区别只在
  commit 作者元数据和隐私；公开仓库使用普通邮箱会把真实收件地址写入 Git 历史。
- 当前推荐：使用 GitHub 页面明确提供的账号专属 `noreply` 地址，既保留账号归属
  和贡献记录，也不公开真实收件邮箱。尚未替用户修改 Git 配置，等待用户确认。
- 协议修正：计划现已要求所有真实选择同时给出选项影响、风险/代价、推荐项、
  推荐理由和选定后的准确操作。

## 2026-07-27 - 里程碑 9C 步骤三：commit 邮箱选择完成

- 用户选择：接受推荐，使用 GitHub 提供的账号专属 `noreply` 地址。
- 执行结果：已通过 `git config --local` 写入当前 FFTCN 仓库，并用
  `--show-origin` 确认配置来源为 `.git/config`。
- 影响边界：只影响此仓库之后创建的 commit，不影响其他仓库、GitHub 登录邮箱、
  通知邮箱、代码、测试、CI 或模型结果；尚未创建 commit 或执行任何 GitHub 写入。
- 当前下一步：从步骤三第 4.2 节开始完成 Git 联网初始化和剩余简化发布链。

## 2026-07-28 - 里程碑 9C 步骤三：个人研究规划移出发布边界

- 用户澄清：`背景/deep-research-report.md` 是个人文献阅读与复现规划，不是公开
  模型代码、运行接口或实验结果的必要组成。
- 边界修正：该文件从 v0.1 暂存快照移除，并加入精确 ignore 规则；本机原文件
  保留，不删除、不改写。
- 引用同步：根 README 删除公开链接，步骤一发布清单和步骤二必需文件检查同步
  移除该路径；现行计划只把它标记为本机私有学习参考。
- 影响：公开仓库仍包含代码、测试、学习指南、现行计划和可复核 JSON 证据，
  不影响模型运行、CPU CI 或 9B 结果。
- 提交修正：用户发现问题时本地发布 commit 已创建但尚未 push；Codex 在确认
  `origin/main` 仍位于上游起点后 amend 该本地 commit，使个人研究规划从首次
  公开快照中消失。由于仅看到 commit 输出，未擅自认定步骤三其他命令均已完成。

## 2026-07-28 - 里程碑 9C 步骤三：公开 Git 材料去个性化

- 用户边界：本机 Git 客户端偏好、代理名称/端口、helper/证书路径、完整个人
  `noreply` 地址及排错过程属于私人操作信息，应保存在用户记忆或本机配置中，
  不属于公开项目。
- 清理范围：现行计划删除本机 Git 规则和联网状态；步骤三指南改为通用
  `git --version`、remote 与只读联网检查，并使用邮箱占位符；日志删除本机
  联网诊断细节。
- 保留范围：通用 Git/GitHub 概念、fork 的 `origin/upstream` 角色、CPU CI、
  commit/push/tag/Release 流程继续公开；上游源码自身的历史硬编码路径不属于
  本次定向清理。
- 历史策略：仓库尚未创建 `v0.1` tag/Release，用户已明确授权以
  `--force-with-lease` 重写首次发布 commit，避免个性化内容继续出现在
  `main` 的正常历史中。

## 2026-07-28 - 里程碑 9C：v0.1 发布与整体验收通过

- 公开交付：用户 fork 的 `main`、带说明的 `v0.1` tag 与 GitHub Release 指向
  同一最终快照；`origin` 指向用户 fork，`upstream` 只允许 fetch，commit 使用
  GitHub `noreply`。Release 不是 prerelease，且没有数据、checkpoint、论文或
  额外附件。
- 路径修正：首次整体验收发现三处公开文档仍保留本机仓库绝对路径。经用户确认，
  操作指南统一改为“先进入克隆后的仓库根目录”，项目文件命令改用相对路径；最终
  跟踪树中的该本机仓库路径为 0。
- 发布边界：公开树不含 `datasets/` 实体、checkpoint、IDE/智能体私有配置、个人
  文献阅读规划、完整论文或生成二进制；无超过 1 MiB 的跟踪文件，也未发现公开
  邮箱、代理、Git helper、证书或本机 Git 客户端信息。
- 行为验证：固定 `d21` 环境正式实现短回归 69/69、三个公开 CLI 3/3；四份独立
  参考分别 4/4、5/5、6/6、7/7。`v0.1` 干净归档再次通过 69/69、CLI 3/3、
  README 本地链接 13/13 和必需文件 8/8。
- 教学与证据：四个教学步骤共 26 项正式实现—完整教程—独立参考契约同步，
  `FullTrainingConfig` 17 个字段一致且保留 `offset_samples`；17 个教程 Python
  代码块可编译。9B 正式报告、验收摘要与 README 三项指标一致，既有人工
  `COMMENT_QUALITY_GATE` 审读未被项目化修改破坏。
- 验收结论：里程碑 9C 的七项统一条件全部满足，正式状态更新为“已完成”。当前
  不自动进入里程碑 10；后续工作必须先共同固定新的学习范围与整体验收条件。

## 2026-07-28 - 里程碑 9C：公开 Python 与证据路径可移植性复核

- 用户反馈：虽然仓库根目录已改为相对表述，公开运行指南仍把 Python 固定为本机
  `d21` 解释器安装位置；JSON 证据也保留了本机环境前缀和仓库前缀。这些信息与
  版本、依赖等可复现实验事实不同，不应成为其他用户必须复制的配置。
- 修正范围：公开 PowerShell/Python 命令统一改为环境激活后的 `python`；结构化
  证据保留 Python、PyTorch、CUDA、包版本和实验配置，但把解释器命令记为
  `python`、环境记为 `d21`，并把项目内文件、缓存与 checkpoint 来源改为相对于
  仓库根目录的路径。
- 保留边界：上游原代码自带的 `D:\BJM` / `E:\wty` 硬编码用于说明原仓库的真实
  可移植性问题，不是当前用户配置；本次不借项目发布清理改写上游算法源码。
- 结果：公开跟踪树中的当前用户 Python 安装前缀、环境前缀与仓库绝对前缀均为 0；
  公开路径规范写入现行计划，里程碑 9C 仍保持“已完成”。
