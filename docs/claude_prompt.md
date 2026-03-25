# Claude Code 上下文 Prompt

复制以下内容作为新会话的第一条消息：

---

我在做 TTT3R 项目，目标 NeurIPS，核心是一个三层频域引导框架用于循环3D重建模型的 state/memory 选择性更新（train-free，推理阶段即插即用）。

请先阅读项目根目录的 `CLAUDE.md`，里面有完整的技术细节、实验结果、代码结构和服务器配置。再看 `docs/research_progress.md` 了解实验时间线。

当前进度：
- Layer 1（帧筛选）✅ 已验证，跳35%帧，TTT3R depth -3.1%
- Layer 2（SIASU token级调制）🔄 bug已修复，消融待重跑
- Layer 3（几何一致性gate）✅ 最佳结果，ttt3r_geogate -7.16%（频域版）/ -7.41%（空间域版）
- B2 memory gate 测试过但效果弱（~1%），已放弃

下一步：
1. 重跑 Layer 2 SIASU 消融（`analysis/spectral_ablation.py`）
2. 三层联合实验
3. 论文 outline

服务器地址 10.160.4.14（user: szy），项目在 `/home/szy/research/TTT3R`，模型在 `model/cut3r_512_dpt_4_64.pth`。本地通过 rsync 同步结果。所有实验命令在 `docs/run_experiments.sh`。

请用中文交流，代码用英文。
