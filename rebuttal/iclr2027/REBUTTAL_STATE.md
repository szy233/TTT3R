# REBUTTAL STATE — ICLR 2027 Submission 28147 (DDD3R)

- **会场：** ICLR 2027，OpenReview 公开讨论，可上传修订版
- **截止：** 2026-11-18；12 月 16 日出结果
- **阶段：** 初次回应，Phase 7。**E1–E9 全部完成并裁定（E9 不成立，见下）。E7 要求的改动和写入后独立审阅的修正已于 2026-11-08 写入三条回应（见 `REBUTTAL_DRAFT_rich.md` 的 “E7 裁定后的改动” 和 “写入后的独立审阅”）。E8 要求的两处改动（噪声说明加 “500f”、PSN7 加一句逐格报告）也已于 2026-11-08 写入。修订版已于 2026-11-08 开始，源码在本 worktree 的 `paper/`（分支 `iclr2027-revision`，已 commit 为 `c86fa4c` 并推到私有仓库 `szy233/ddd3r`），全部改动见 `REVISION_LOG.md`，给审稿人的清单见 `CHANGE_LIST.md`。三条回复已于 2026-11-08/09 与作者逐段重过（见 `REBUTTAL_DRAFT_rich.md` 的“与作者逐段重过”）。作者确认 ICLR 可以多轮回复，字数上限不再是硬约束，但要精简。发布顺序：先上传修订版，再发变更清单（单独一条评论），最后发三条回复。三条回复和修订版都尚未发出或上传。rebuttal 材料已于 2026-11-09 按作者要求推到公开仓库 `szy233/TTT3R` 的分支 `claude/project-understanding-b08e9a`（作者知道其中含审稿原文和未发出的回复）**
- **更新：** 2026-11-08

## E6 和 E6d 的裁定

| 预测 | 结果 | 回复必须怎么写 |
|---|---|---|
| P6a，KITTI r_rel 同范数 | 成立，11.88 vs 16.78，10/11，p=0.019 | 可以说同范数下方向降低分段旋转误差。ATE、t_rel 打平，RPE 平移更差（0/11） |
| P6b，TUM 平标量 | 不成立，α=0.10–0.25 打平，α=0.33 ortho 更好 | 必须说缩放也可能进入噪声范围 |
| P6c1–P6c4，α∥=0.15 | 全部成立 | 只作敏感性结果。ScanNet 的害处缩小但未消失，不改默认值 |
| P6d，KITTI 平标量 | 不成立，α=0.10–0.20 打平，最低均值在 α=0.10，位于扫描边界 | 必须说缩放也可能进入噪声范围，不能暗示只有分解做得到 |

另外，同代码下 brake 在 TUM 上略好于 const 0.33（0.0634 vs 0.0648，p=0.039），但 2.3% 的差距小于测得的舍入噪声 4.7%，回复里必须说明在噪声以内。裁定和细节见 `docs/PREREG_E6.md`、`docs/PREREG_E6D.md` 的日期节，数字见 `results/e6.txt`。

## E7 的裁定（2026-11-08）

固定每个 token 的步长为 const α 的步长，只换成分解后的方向。

| 预测 | 结果 | 回复必须怎么写 |
|---|---|---|
| P7a，TUM | 不成立，三个尺度都打平（p≥0.31） | 必须说平范数下方向在 TUM 上没有收益，并把同范数那句限定为 ortho 自己的范数调度 |
| P7b，KITTI r_rel | 三个尺度都成立（8/11、9/11、11/11，p≤0.014），差距都大于 13.4% 的舍入噪声 | TUM 打平，所以“两个数据集都成立”那行不满足，按“其他模式”处理。只能逐格报告并写明代价，不能概括成“方向和尺度可以叠加” |
| P7c，ScanNet | 不成立。两个尺度方向都更差，害处没有随步长缩小（差值 +0.0018，CI [-0.073, +0.068]） | 删掉任何“害处随步长变小”的暗示 |
| ARKitScenes，无预测 | 0.10、0.20 方向更差（2/18、3/18），0.33 打平 | 引用 KITTI 格子的回复必须同时报告这些 |
| 平标量检查 | const 0.10 在四个数据集上都不比 ortho 差，ScanNet 上更好（0.382 vs 0.488） | 任何讨论方向的回复都必须明说 |

舍入噪声（Amendment 1）：一个乘子改 1e-7，均值变化最多 4.7%（TUM ATE）、6.9%（KITTI ATE）、13.4%（KITTI r_rel），只测了一次，ScanNet 和 ARKitScenes 没测。回复必须删掉“2.2% 是我们见过的最大差距”，写出测得的差距，并把低于它的“更好”标为噪声以内。

事后、未登记：ortho 的 RPE 旋转在 TUM、ScanNet、ARKit 上好于 const 0.10，但同步长下方向并不降低 RPE 旋转，所以这个优势来自 ortho 更大的步长。ARKit 上 const 0.33 与 brake 打平（p=0.78）。细节见 `docs/PREREG_E7.md` 的日期节，数字见 `results/e7.txt`、`results/e7_posthoc.txt`。

## E8 的裁定（2026-11-08）

问题是 KITTI 上方向压低累积旋转漂移这个效应，在 ARKitScenes 上是否也存在，以及是否随序列变长而出现。新指标 R_Δ 是跨 Δ 帧的 RPE 旋转，调用与现有 RPE 旋转相同，主指标是 R_100。

| 预测或检查 | 结果 | 回复必须怎么写 |
|---|---|---|
| I1、I2、I3 | 全部通过。在 KITTI 四格里 R_100 都随方向下降（−22% 到 −36%），与 r_rel 一致 | — |
| P8a，ARKit 500 设计 A | 不成立，打平（−4.3%，p=0.62） | 不能说 ARKitScenes 有 KITTI 的旋转效应，只能逐格报告 |
| P8b，ARKit 500 设计 B | 不成立，三个尺度方向都更差（+10.8%、+14.8%、+13.6%，p≤0.022） | 同上 |
| TUM、ScanNet，无预测 | 平范数下 TUM 0.10 和 ScanNet 两个尺度方向更差，其余打平 | 不能做一般性陈述 |
| P8c，ARKit 1000 R_100 | 不成立，三格打平，均值都是方向更高（+4.0% 到 +7.1%） | 不能把任何说法推广到 1000 帧 |
| P8d，ATE 代价随长度缩小 | 不成立（0/3） | 不能说方向的效应取决于序列长度 |
| ARKit 1000 ATE | 三格打平，n=8，均值都是方向更差（+13.6% 到 +23.2%） | 不能说方向改善 ARKitScenes 精度 |

ARKitScenes 1000 帧的舍入噪声是 ATE +0.2%、R_100 −1.0%，这是 ARKitScenes 上唯一一次测量；500 帧仍然没有。所以回复里 “not run on ScanNet or ARKitScenes” 这句已改成 “ARKitScenes 500f”。事后、未登记：1000 帧上 brake 与 TTT3R 打平（0.937 vs 1.030，p=0.38），RPE 旋转更高（0/8，p=0.0078）。const 0.10 在 ARKit 1000 帧上与 ortho 打平，E7 那句仍然成立。细节见 `docs/PREREG_E8.md` 的日期节，数字见 `results/e8a.txt`、`results/e8b.txt`、`results/e8_posthoc.txt`。

## E9：漂移能量路由规则的留出检验（2026-11-09 登记并裁定：不成立）

**裁定：** I1、I2 都通过（I1 的 13 个值逐位复现）。KITTI 的前缀 ē 均值 0.501，正好卡在阈值上，11 条里 5 条走 ortho。在唯一有区分度的格子 KITTI r_rel 上，规则比 ortho 差（15.24 vs 11.88，+28.3%，p=0.031，超过舍入噪声上限 13.4%），所以 P9a 不成立。P9b 成立：规则在 KITTI 和 Sintel 的 ATE 上都不比 brake 差。Sintel 两个分支打平（0.2361 vs 0.2372），没有区分度。ARKitScenes 18 个场景全部走 brake。**回复和修订版保持 “no validated rule”，不加任何文字。** 按登记，这个 rebuttal 期间不再在 KITTI、Sintel、ARKit 上试别的阈值或统计量。如果 qpVA 在后续回合追问配置选择，可以用一句话说明这条规则在留出的 KITTI 上没有通过。

**登记时的设计（原文）：**

回应 qpVA W1（未见过的场景怎么选配置）。规则在 TUM、ScanNet 上探索得到：TTT3R 下前 200 帧的相邻 delta 漂移能量 ē<0.5 走 ortho，否则走 brake τ=1。留出集是 KITTI Odometry（11）和 Sintel（14），ARKitScenes 已被看过（选阈值时用了它的整段 ē），只并列报告、不计入。登记前已知：KITTI 两个分支 ATE 打平、r_rel ortho 更好（p=0.0029），所以 KITTI 上实际检验的是它的 ē 会不会落在 0.5 以下；Sintel 只知道论文表 4 的旧代码均值（brake 0.237、ortho 0.236）。成立就只能写成“在两个留出集上成立的启发式，不算验证过的选择器”；不成立就保持“no validated rule”，并且不再在这些数据上试别的阈值。登记 `docs/PREREG_E9.md`，哈希 `results/e9_registration.sha256`（dbfa8a38…），脚本 `scripts/e9_selector.py`、`scripts/run_e9.sh`，日志 `results/queue_e9.log`。

## 文件

| 文件 | 内容 |
|---|---|
| `REVIEWS_RAW.md` | 评审原文 |
| `ISSUE_BOARD.md` | 16 项评审问题和 11 项自查问题 |
| `STRATEGY_PLAN.md` | 主题、修订策略、发布节奏、证据计划 |
| `scripts/paired_stats.py` | 从现有日志算配对统计，不运行模型 |
| `results/paired_stats.txt` | 上面脚本的输出 |
| `results/kitti_e5b.txt`、`results/kitti_official_metrics.txt` | E5b 全部数字，由 `scripts/kitti_e5b.py` 从日志生成 |
| `results/e1_e4.txt` | E1–E4 全部数字，由 `scripts/e1_e4_rulings.py` 生成 |
| `results/e6.txt`、`results/e6_kitti_metrics.txt` | E6 和 E6d 全部数字，由 `scripts/e6_rulings.py`、`scripts/e6_kitti_metrics.sh` 生成 |
| `docs/PREREG_E6.md`、`docs/PREREG_E6D.md` | E6、E6d 的登记和裁定。分隔线以上由 `results/e6_registration.sha256`、`results/e6d_registration.sha256` 校验，不可改 |
| `results/e7.txt`、`results/e7_kitti_metrics.txt`、`results/e7_posthoc.txt` | E7 全部数字，由 `scripts/e7_rulings.py`、`scripts/e7_kitti_metrics.sh`、`scripts/e7_posthoc.py` 生成 |
| `results/reply_sources.txt` | MDAW 两句的配对检验（TUM 网格各配置对 brake、τ=2 对 const 0.5），未登记，由 `scripts/reply_sources.py` 生成 |
| `docs/PREREG_E7.md` | E7 的登记、Amendment 1、舍入噪声、r_rel 有效性检查和 E7 裁定。分隔线以上由 `results/e7_registration.sha256` 校验，不可改 |
| `docs/PREREG_E8.md` | E8 的登记、一处偏差（ScanNet R_200 用 63 个场景）以及 E8a、E8b 的裁定。分隔线以上由 `results/e8_registration.sha256` 校验，不可改 |
| `results/e8_instrument.txt`、`results/e8a.txt`、`results/e8b.txt`、`results/e8_posthoc.txt` | E8 全部数字，由 `scripts/e8_rot_drift.py` 的 instrument、e8a、e8b、posthoc 四个模式生成。E8b 的运行由 `scripts/run_e8.sh` 启动，日志在 `results/queue_e8.log` |
| `docs/PREREG_E9.md` | E9 的登记和裁定（不成立）。分隔线以上由 `results/e9_registration.sha256` 校验，不可改 |
| `results/e9_instrument.txt`、`results/e9_prefix.json`、`results/e9.txt` | E9 全部数字，由 `scripts/e9_selector.py` 的 prefix、rule 两个模式生成，运行由 `scripts/run_e9.sh` 启动，日志在 `results/queue_e9.log`；Sintel 三个分支在 `results/relpose/sintel/` |
| `PASTE_READY_MDAW.txt`、`PASTE_READY_qpVA.txt`、`PASTE_READY_PSN7.txt` | 要发出的三条评论，每位审稿人一条 |
| `REBUTTAL_DRAFT_rich.md` | 覆盖情况、数字来源、待确认的句子、可选内容 |
| `patches/tau_kitti_fix.patch` | τ 和 KITTI 的合并补丁，已应用到本 worktree 的 `paper/` 克隆（分支 `iclr2027-revision`） |
| `REVISION_LOG.md` | 修订版的逐处改动、原因、数字来源、编号变化和未改事项 |
| `CHANGE_LIST.md` | 给审稿人的英文变更清单，约 3000 字符，随修订版一起发 |
| `results/revision_sources.txt` | 修订版引用、此前结果文件里没有的检验（ARKit const 0.5 的 p 值、ARKit 各方法 RPE），由 `scripts/revision_sources.py` 生成，未登记 |
| `../../paper/` | 论文克隆，基于 `ce69b27`，push 地址已禁用。`latexmk -pdf iclr2027.tex` 构建，25 页，正文 10 页；`\revisionfalse` 去掉蓝色 |

上一轮 NeurIPS 8591 的材料在分支 `claude/understand-full-project-14f204` 的 `rebuttal/`，彼此独立。

## 进度

| Phase | 状态 |
|---|---|
| 0–1 初始化、原文存档 | 完成 |
| 2 问题拆分 | 完成 |
| 3 策略 | 完成 |
| 3.5 证据 | 完成，裁定见 `docs/PREREG_E1_E5.md` |
| 4 起草 | 完成。作者要求逐位回复，每条上限 5000 字符，尽量简短，可用 Markdown 表格 |
| 5–6 校验、压力测试 | 完成。Codex MCP 不可用，改用一个独立审阅代理 |
| 7 定稿 | 完成，等作者确认待批句子后发出 |
| 8 后续回合 | 未开始 |

## τ 和 KITTI 更正：补丁已应用到修订分支

2026-11-08 起，论文在本 worktree 的 `paper/` 克隆里修订（分支 `iclr2027-revision`，2026-11-09 已 commit 为 `c86fa4c` 并推到私有仓库 `szy233/ddd3r`），补丁已应用，之后又按 E6–E8 的裁定改写了 (iii) 等段落，详见 `REVISION_LOG.md`。`understand-full-project-14f204/paper` **仍然没有任何改动**，将来要从克隆里 fetch 过去。`patches/tau_kitti_fix.patch` 取代了旧的 `tau_fix.patch`。两件事必须一起应用：只改 τ 的话，KITTI 那几格仍是 τ=2。

补丁内容（已在临时副本验证，能应用到论文 `ce69b27`，两个版本都能编译，均 19 页）：

- brake 全文改为 τ=1，包括 `sec/appendix_supp.tex` 第 51、259 行和 `sec/v2/experiments.tex` 第 10 行。
- 表 3 的 KITTI 列换成 E5b 的 ATE RMSE，并填上 TTSA3R。
- 表 5 的 const、brake 两行和第二名的下划线。
- 排名表的 KITTI 列、相关句子和表注。
- 建议 (iii) 重写，撤回 “best aggregate” 和 “strongest test”。

`sec/appendix_supp.tex` 是已提交版和 advisor 版共用的。已上传的 PDF 和本地 `iclr2027.pdf` 都没有变。

## 尚未解决

1. `checklist.tex:51` 还写着 τ=2，ICLR 构建不用这个文件，没有改。
2. 代码默认 `--brake_tau` 仍是 2.0，`run_ddd3r_eval.sh` 显式传 1.0。公开代码前要对齐。
3. 论文 `main` 的 ce69b27 还没 push，上一轮 worktree 里 `analysis/gen_fig1_teaser.py` 也未提交。
4. ~~论文的 AI 使用声明前后矛盾（S15）~~ 已于 2026-11-09 定稿：作者确认假设源自作者、翻译一项为未使用、保留厂商名、审阅程序三句属实；讨论期那句补上了 “design”。剩下的只有 OpenReview 的 AI Assistance 勾选，要和声明一致。
5. 补丁只处理 τ 和 KITTI。γ 含义写反、A.8 等公式问题仍按 `STRATEGY_PLAN.md` 的修订项处理。
