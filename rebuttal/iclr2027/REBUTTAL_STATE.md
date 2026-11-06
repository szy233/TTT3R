# REBUTTAL STATE — ICLR 2027 Submission 28147 (DDD3R)

- **会场：** ICLR 2027，OpenReview 公开讨论，可上传修订版
- **截止：** 2026-11-18；12 月 16 日出结果
- **阶段：** 初次回应，Phase 3 完成，**停在证据关卡，等待确认实验**
- **更新：** 2026-11-06

## 文件

| 文件 | 内容 |
|---|---|
| `REVIEWS_RAW.md` | 评审原文 |
| `ISSUE_BOARD.md` | 16 项评审问题和 11 项自查问题 |
| `STRATEGY_PLAN.md` | 主题、修订策略、发布节奏、证据计划 |
| `scripts/paired_stats.py` | 从现有日志算配对统计，不运行模型 |
| `results/paired_stats.txt` | 上面脚本的输出 |
| `patches/tau_fix.patch` | τ 更正补丁，待应用到论文仓库 |

上一轮 NeurIPS 8591 的材料在分支 `claude/understand-full-project-14f204` 的 `rebuttal/`，彼此独立。

## 进度

| Phase | 状态 |
|---|---|
| 0–1 初始化、原文存档 | 完成 |
| 2 问题拆分 | 完成 |
| 3 策略 | 完成 |
| 3.5 证据 | **等待确认（AUTO_EXPERIMENT=false）** |
| 4 起草 | 未开始；缺词数上限，且第二轮依赖实验结果 |
| 5–7 校验、压力测试、定稿 | 未开始 |

## τ 更正：补丁已备好，尚未应用

本会话被隔离在自己的 worktree，不能写论文所在的 `understand-full-project-14f204/paper`，所以**论文源码目前没有任何改动**。`patches/tau_fix.patch` 包含：

- `sec/appendix_supp.tex` 第 51 行：brake 改为 τ=1，注明 KITTI 汇总结果用的是 τ=2。
- `sec/appendix_supp.tex` 第 259 行：按 τ=1 重写对比，τ=2 的“约 2 倍以内”不再适用。
- `sec/v2/experiments.tex` 第 10 行：同样更正。

`sec/appendix_supp.tex` 是已提交版和 advisor 版共用的。已上传的 PDF 和本地 `iclr2027.pdf` 都没有变。

## 尚未解决

1. 表 5 KITTI 03、04 列 brake 的 τ 查不到记录，E5 解决。
2. `checklist.tex:51` 还写着 τ=2，ICLR 构建不用这个文件，没有改。
3. 代码默认 `--brake_tau` 仍是 2.0，`run_ddd3r_eval.sh` 显式传 1.0。公开代码前要对齐。
4. 论文 `main` 的 ce69b27 还没 push，上一轮 worktree 里 `analysis/gen_fig1_teaser.py` 也未提交。
