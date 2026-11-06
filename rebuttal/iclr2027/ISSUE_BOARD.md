# ISSUE BOARD — ICLR 2027 Submission 28147 (DDD3R)

评审原文见 `REVIEWS_RAW.md`。下文数字来自 `results/paired_stats.txt`，或注明的原始文件。

| ID | 评审 | Rating | Pres. | Info. | Sound. | Conf. | 立场 |
|---|---|---|---|---|---|---|---|
| R1 | qpVA | **2 Weak rejection** | 2 | 2 | 2 | 3 | 负面，理由具体 |
| R2 | PSN7 | 3 Weak acceptance | 3 | 3 | 3 | 2 | 正面，只问一件事 |
| R3 | MDAW | **2 Weak rejection** | 2 | 2 | 2 | 2 | 摇摆，问题都能回答 |

**优先级：R3 > R1 > R2。** 目前是两票拒绝一票接受。R3 的问题最具体也最能回答，R1 与它的关切大部分重合。

**共同点：** 三位都肯定诊断、常数阻尼基线和序列长度实验。分歧不在诊断，而在方向分解的独立价值和公式的严谨性。

---

## 按根因归组

| 组 | 根因 | 涉及问题 | 评审数 |
|---|---|---|---|
| A | 方向分解的独立贡献没有隔离 | R1-C2、R1-C3、R2-C1、R2-Q1、R3-C1、R3-Q3 | 3/3 |
| B | 公式与解释不一致 | R1-C5a、R1-C5b、R3-C2、R3-C3、R3-Q1、R3-Q2a、R3-Q2b | 2/3 |
| C | 没有在未知场景选配置的规则 | R1-C1、R3-C1 末段 | 2/3 |
| D | 门控坍缩只测了帧平均 | R1-C4 | 1/3 |

---

## R1 — qpVA

| ID | 原文要点 | 类型 / 严重度 | 核查结果 | 回应方式 | 状态 |
|---|---|---|---|---|---|
| R1-C1 | "shifts the problem to hyperparameter search ... no reliable rule" | practical / major | **部分成立。** 固定配置已有明显增益，但确实没有验证过的选择规则，在线自适应也都失败了 | narrow_concession + evidence | open，见 T3 |
| R1-C2 | "ScanNet-1000 ... 0.488, worse than both TTT3R (0.406) and TTSA3R (0.378) ... underperforms TTT3R on ARKitScenes" | empirical / critical | **成立，且比他说的更强。** ScanNet 上 ortho 只赢 TTT3R 29/65 场景，p=0.047；赢 brake 4/65 | narrow_concession | open |
| R1-C3 | "added value of directional decomposition remains unclear" | empirical / critical | **成立。** 现有数据无法把方向和幅度分开 | 需要 E1 | needs_user_input |
| R1-C4 | "Stable frame-averaged gates do not rule out useful token-level temporal adaptation" | assumptions / major | **成立。** 论文只测了帧平均的时间方差，以及帧内空间方差，没有测单 token 的时间方差。TTSA3R 在 ScanNet 1000f 赢 TTT3R 43/65，也说明门控设计并非无关紧要 | 需要 E3，否则收窄表述 | needs_user_input |
| R1-C5a | "EMA timing differs between the equations and Figure 2" | theorem_rigor / major | **成立。** 代码和方程 7 都是先更新 d_t 再投影。图 2(c) 的说明写的是 d_{t-1} 进入分解 | 承认并修图注 | open |
| R1-C5b | "γ interpolation direction contradicts Equation (12)" | theorem_rigor / major | **成立**，同 R3-C2 | 承认并修文字 | open |

## R2 — PSN7

| ID | 原文要点 | 类型 / 严重度 | 核查结果 | 回应方式 | 状态 |
|---|---|---|---|---|---|
| R2-C1 | "couldn't clearly tell the improvement is from less pace or corrected direction" | empirical / critical | **成立。** 这是最准确的一句批评 | 需要 E1 | needs_user_input |
| R2-Q1 | "If the per-token update follows the same norm as ortho without direction decomposition, how is the performance on TUM and ScanNet?" | empirical / critical | **没有跑过。** 这正是该做的消融，实现只需几行 | 需要 E1 | needs_user_input |

## R3 — MDAW

| ID | 原文要点 | 类型 / 严重度 | 核查结果 | 回应方式 | 状态 |
|---|---|---|---|---|---|
| R3-C1 | "do not sufficiently establish the independent contribution of directional decomposition" | empirical / critical | **成立**，同 R2-C1 | 需要 E1 | needs_user_input |
| R3-C1b | "default directional hyperparameters were selected using TUM" | empirical / major | **属实，但可回答。** 见 R3-Q3 的嵌套选择 | grounded_evidence | 零成本结果已有 |
| R3-C2 | 方程 12 增大 γ 趋向方向分解，与 A.4 相反；γ=0 特殊处理造成不连续 | theorem_rigor / major | **全部成立。** 方程本身和代码一致，错的是 A.3、A.4 的文字和表 5 标题。ScanNet 数据也符合这个方向：γ=0.5/1/2/3/5 → 0.327/0.358/0.394/0.407/0.456，逐步从 const(0.283) 走向 ortho(0.488)。γ=0 在代码里是“关闭自适应”的开关，不是极限 | 承认，数字不变 | open |
| R3-C3 | 方程 7 先用当前更新算 EMA，所以 β_ema=0 时 d_t 等于当前方向，不是 A.8 假设的上一帧方向 | theorem_rigor / major | **成立。** 代码同样先更新后投影。β_ema=0 时 δ⊥=0，A.8 的 Setup 写错了 | 承认并改 A.8 | open |
| R3-Q1 | γ 自适应的实际实现是什么 | reproducibility / major | 代码：`w = ē^γ`，`α∥ᵗ = w·α⊥ + (1-w)·α∥`，只在 γ>0 时进入。**还有一处评审没发现：** 方程 12 写的是瞬时 e_t，代码用的是它的 EMA | 直接说明 | 可回答 |
| R3-Q2a | 投影到 d_{t-1} 和更新后的 d_t 有什么差别，能否比较 | empirical / major | 实现投影到更新后的 d_t。**比较没有跑过** | 需要 E2 | needs_user_input |
| R3-Q2b | brake 是方程 6 的严格特例，还是只在性质上相关 | theorem_rigor / critical | **不是严格特例。** 只有允许 α⊥=α∥=g_t 随 token 和时间变化才算同一形式，而且 g_t 看的是 δ_{t-1}，不是 d_t。摘要的 "strictly generalizes" 和引言 "strictly generalizes all prior update strategies" 都与 A.8 矛盾 | 承认并改摘要、引言 | open |
| R3-Q3 | TUM 上 ortho 对 brake 的逐序列配对比较、置信区间、显著性；独立选参或留出验证 | empirical / major | **已算出：** ortho 8/8 胜，均值 −14.0%，CI [−0.0132, −0.0049]，精确 Wilcoxon p=0.0078（n=8 能达到的最小值）。留一嵌套选参：留出均值 0.0577 对 brake 0.0634，8/8 胜，CI [−0.0101, −0.0023]，p=0.0078 | grounded_evidence | 可回答 |

---

## 评审没发现、但修订版必须处理的问题

| ID | 问题 | 证据 | 严重度 |
|---|---|---|---|
| S1 | brake 的 τ 写错 | 论文写 τ=2。TUM/ScanNet relpose 与 `inv_t1` 逐位相同，其余任务由脚本参数确认是 τ=1；**KITTI 汇总是 τ=2**（zjc 报告，`--brake_tau 2.0`），表 5 的 KITTI 04 也是 τ=2（复现出 7.05001，τ=1 是 5.556）。真 τ=2 时 brake 输给 const：TUM 0.0822 对 0.0786，ScanNet 0.3107 对 0.2827。**补丁在 `patches/tau_fix.patch`，尚未应用到论文** | critical |
| S2 | KITTI 列的指标标错 | 表 2 的 192.84/161.14/156.61/166.89/154.10 是 **ATE Mean**，论文写的是 ATE RMSE。RMSE 应为 212.59/183.16/179.33/185.68/170.03，排名不变 | major |
| S3 | **表 2 的 KITTI 列很可能跑在另一条 RoPE 路径上** | 数据相同：在本机重建 `kitti_odo_full` 后，结果与 `kitti_odom` 逐位相同。seq 04 RMSE（报告 / 编译版 / 回退加 clamp）：CUT3R 32.45 / 21.85 / 31.31，brake τ=2 6.66 / 7.05 / 6.90，ortho 10.28 / 17.79 / 12.28。zjc 的提交 `cd57a18` 给回退 RoPE 加了 clamp，把 pose token 的位置 −1 改成 0，相当于改变了模型行为。**证据很强但未最终确认，需要问 zjc。** 受影响：表 2 KITTI 列、排名表、建议 (iii)、KITTI 上的 20.1% / 7.6% / 58.6% | critical |
| S4 | 论文完全没有报告 RPE | 相对 CUT3R 的 RPE rot：TUM const −13%、brake +71%、ortho +25%；ScanNet const +91%、brake +158%、ortho +321%。论文却引用了 KITTI 上旋转误差 −58.6% | major |
| S5 | brake “最优”缺乏统计支持 | ScanNet brake 对 const(0.5)：−7.5%，36/65，CI 跨 0，p=0.31。TUM brake 对 const(0.33)：−3.8%，4/8，p=0.55。ScanNet 没跑过 const(0.33) | major |
| S6 | “漂移能量”有三种算法 | 表 3 是逐帧先平均 cos 再平方（对 δ_{t-1}）；方程 12 是对 EMA d_t，代码还再做一次 EMA；Point3R 是逐事件 cos² 的均值。而 A4c 显示 EMA 版方向相反（TUM 0.786 > ScanNet 0.660，90f） | major |
| S7 | FPS 数据矛盾 | 表 `tab:overhead` 称 ortho 比 CUT3R 快 18%（9.95 对 8.44），早先的 S2 实验是 CUT3R 最快（10.75），S2 没合入最新分支。配置按顺序跑，可能是顺序效应，未证实 | minor |
| S8 | “zero added memory” 不准确 | 峰值显存 6.1401 → 6.1428 GB，约多 2.7 MB。附录也写了要存 d_t | minor |
| S9 | 可复现性声明不实 | "All reported numbers come from a single evaluation script"，但 KITTI 来自另一分支的另一批运行 | major |
| S10 | TTSA3R “improving on it in every setting” | Sintel 上 0.20908 对 0.20879，略差；ARKitScenes 9/18，不显著 | minor |
| S11 | 表 5 取整 | TUM γ=2 原始值 0.05569，印成 0.055 | minor |

**已排除：** 视频深度上 “every variant strictly improves”，KITTI 上 const 0.10649 确实小于 TTT3R 0.10741，印出来都是 0.107 只是取整。这项不是问题。

---

## 覆盖情况

共 16 项评审问题：R1 6 项，R2 2 项，R3 8 项。全部有记录。

| 状态 | 数量 | 问题 |
|---|---|---|
| 现在就能答或承认 | 10 | R1-C1、R1-C2、R1-C5a、R1-C5b、R3-C1b、R3-C2、R3-C3、R3-Q1、R3-Q2b、R3-Q3 |
| 需要新实验 | 6 | R1-C3、R1-C4、R2-C1、R2-Q1、R3-C1、R3-Q2a |

其中公式类问题除了回应，还要在修订版里改正。
