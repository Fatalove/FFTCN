# 里程碑 9B 证据目录

本目录记录 Sleep-EDF-153 教学重构模型的完整训练验证。9B 是一个整体里程碑，四个内部步骤只表示学习与运行依赖，不单独验收。

## 当前进度

- 四个内部步骤的正式练习代码已经完成，步骤四聚焦测试为 7/7；
- 固定运行入口测试为 8/8，步骤三训练编排测试为 5/5，全套短测试为 68/68；
- 处理后数据已存在：train/validation/test 分别有 122/16/15 条记录；
- 全量训练缓存已完成并核验：train/validation/test/pretrain_train 分别为 154128/23315/18036/494617 项，实际构建用时 2489.80 秒；
- raw/wave/fusion 真实 batch 均完成 20 次更新并满足首末损失下降，峰值分配显存为 41.85/212.04/2882.17 MiB；
- 三阶段 epoch 0 均已完成，history、best.pt、last.pt、损失和下一轮学习率均通过核验；
- 恢复预检和正式 20/20/50 训练尚未运行；当前唯一下一步是由用户运行 `resume2`；
- 因此当前仍是“等待用户练习”，里程碑 9B 尚未完成。

## 当前运行链

```text
record-level NPZ
  -> train/validation/test 的 raw + labels + manifest
  -> 由同一 raw 重构 wave
  -> 每条 train 记录先偏移过采样，再生成对齐的 pretrain_train raw/wave/labels
  -> raw/wave validation-best 预训练 checkpoint
  -> state_dict 迁移
  -> T=50 融合微调
  -> validation 选出 fusion best
  -> 固定 test 只正式评估一次
```

真实长操作由用户逐项执行。固定入口为 `scripts/run_milestone_09b.py`，准确命令、路径、每阶段检查项和恢复规则见 `learning_guides/milestone_09b/OPERATOR_GUIDE.md`。脚本不会自动串联长阶段，也不会自动改变 batch size 或覆盖已有缓存。

固定缓存策略遵循源仓库顺序：在每条记录内部先调用偏移过采样得到平衡 raw/labels，再从同一批 raw 计算 wave，最终保存对齐的 `pretrain_train` 缓存。预计全部新缓存约占 D 盘 10 GiB，其中平衡数据区约 7.19 GiB；这是磁盘用量，不是同等大小的 RAM 或显存分配。

## 相关指导与证据

- `learning_guides/milestone_09b/OPERATOR_GUIDE.md`：用户分阶段运行指南；
- `validation_summary.json`：当前代码、测试、数据和实验边界摘要；
- `sequence_semantics_audit.md`：紧凑序列决定及其限制；
- `../MILESTONE_LOG.md`：可复核节点日志。

紧凑序列语义保持不变：删除 Movement/Unknown 后不额外插入 gap 边界，序列不得跨记录，并在每条记录内独立截断为非重叠 T=50。3,836 个序列中有 110 个跨过至少一个被删除的内部无效时段；最终报告必须披露这一限制。
