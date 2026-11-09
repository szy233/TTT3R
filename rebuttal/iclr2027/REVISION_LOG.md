# 修订版记录 — ICLR 2027 Submission 28147（DDD3R）

2026-11-08 开始。这份文件记录修订版的每一处改动、原因和数字来源。给审稿人看的英文变更清单在 `CHANGE_LIST.md`。

## 在哪里改、怎么构建

- **源码：** 本 worktree 的 `paper/`。它是从 `understand-full-project-14f204/paper` 只读克隆来的，在分支 `iclr2027-revision` 上，基于 `ce69b27`。另一个 worktree 一处都没动。克隆的 push 地址已设为 `DISABLED`，防止误推。
- **基线：** 对 ICLR 构建来说，`ce69b27` 的源码与 9 月 26 日的 `43bf931` 完全相同。ce69b27 只加了 advisor 草稿用的文件。已提交 PDF 的 AI 声明比 43bf931 早一版，所以 pdfdiff 在声明部分会多出一些差异。
- **构建：** 在 `paper/` 里运行 `latexmk -pdf iclr2027.tex`。输出 `paper/iclr2027.pdf`，共 25 页，正文到第 10 页结束，符合讨论期 10 页的上限。只剩基线里原有的两处 2pt 以内的 overfull，没有未定义引用。
- **标蓝：** 改动的文字和新加的表格都是蓝色。`iclr2027.tex` 里把 `\revisiontrue` 改成 `\revisionfalse` 就全部恢复为黑色，正式版用这个开关。
- **全部 diff：** 运行 `git -C paper diff ce69b27 c86fa4c`，新文件 `sec/appendix_revision.tex` 也在其中。
- **已 commit 并推送（2026-11-09）：** `c86fa4c`，推到私有仓库 `szy233/ddd3r` 的分支 `iclr2027-revision`；该仓库的 `main` 没动。克隆的 push 地址仍是 `DISABLED`，推送时直接指定了地址。

## 编号变化

新的正文表插在 §5.2，所以表格编号从 5 开始顺延一位。回复里写的 “Table 5 caption” 现在对应 **Table 6**，变更清单里已写明。

| 内容 | 提交版 | 修订版 |
|---|---|---|
| 方向与步长（新） | — | Table 5，§5.2 |
| γ 扫描 | Table 5 | Table 6 |
| 排名 | Table 6 | Table 7 |
| 开销、7Scenes、更新规则、Point3R | Table 7–10 | Table 8–11 |
| 附录新表 | — | Table 12–19 |
| 消融与推荐 | §5.2 | §5.3 |
| 新附录 | — | A.12 统计规则与噪声，A.13 RPE，A.14 方向与步长全表，A.15 逐 token 门控，A.16 TUM 逐序列与留出选参 |

A.1–A.11、方程编号、图编号和算法编号都没变。回复里引用的 A.4、A.8、Eq. 6/7/12、Algorithm 1、Figure 2(c) 仍然指向同一处。

## 逐处改动

“来源”一列列出的是 `results/` 里的文件。凡是未登记、事后补算的比较，都已在对应结果文件里标明。

### 1. τ 和 KITTI（应用了 `patches/tau_kitti_fix.patch`）

| 位置 | 改动 | 原因 | 来源 |
|---|---|---|---|
| A.2 (iv)、§4.3、§5 开头 | brake 的 τ 一律写 1；A.2 把 brake 描述为相关的标量基线，不再称为 “relaxation” | S1；三条回复都说修订版会更正 τ | `paired_stats.txt` 的 Brake temperature |
| Table 3 KITTI 列 | 换成 E5b 重跑的 ATE RMSE，补上 TTSA3R；图注写明同一协议、彼此差异不显著 | S2、S3 | `kitti_e5b.txt` |
| Table 6（原 Table 5）KITTI 两列 | const 行改为 α=0.5 的 7.10/73.93；brake 行改为 τ=1；下划线随之调整 | S12 | `kitti_e5b.txt` |
| Table 7（原 6）KITTI 列、A.4 正文、§5.1 | KITTI 不再算作 ortho 领先 | E5b | `kitti_e5b.txt` |
| §5.3 (iii) | 撤回 “best aggregate for outdoor driving”，改写为 ATE 不显著、分段旋转误差显著更低，同时写明 RPE 平移代价和绝对误差都很大；比补丁版本更短，删去了 24.6% 那句 | E5b；qpVA 回复已撤回 | `kitti_e5b.txt` |
| A.8 “Why brake is still useful” | 按 τ=1 重写数值对比 | S1 | 补丁中已验证 |
| `checklist.tex:51` | τ=2 改为 1（只有 NeurIPS 构建用到这个文件） | S1 | — |

### 2. γ 与方程 12（A.3、A.4、Table 6）

| 位置 | 改动 | 原因 | 来源 |
|---|---|---|---|
| A.3 方程 12、算法 1 | e_t 改为逐 token 的 EMA ē_t（与 β_ema 相同，初值 ē₂=e₂），与代码一致 | R3-Q1；MDAW 回复 | `model.py` 第 1686–1695、2033–2037 行 |
| A.3 正文 | γ→0⁺ 时趋向各向同性，γ 增大时趋向 ortho；γ=0 是关闭开关，直接运行 ortho，所以只有 γ>0 时才连续 | R3-C2、R1-C5b | 代码；ScanNet 扫描 |
| Table 6 图注、A.4 正文 | 方向改正；删去 “drift-energy profiles” 的说法 | 同上 | — |
| Table 6 TUM γ=2 | 0.055 改为 0.056；下划线移到 γ=0 | S11 | 原始日志 0.05569 |
| Table 6 新增 γ=5 行 | TUM 0.055、ScanNet 0.456；KITTI 没有跑，填 — | 补全趋势 | `eval_results/relpose/*/ttt3r_ortho_steep_v2_g5` |
| Table 7 | 删掉 “Drift regime” 一列；TUM 排名改为 ortho 1、γ 温和 2、brake 3、const 4（γ=2 修正后的结果） | 撤回漂移能量决定配置的说法；S11 | — |
| §4.3 | 补充 γ 的方向和 γ=0 处的不连续 | R3-C2 | — |

### 3. EMA 时序（方程 7、算法 1、图 2）

| 位置 | 改动 | 原因 |
|---|---|---|
| 方程 7 | 加入每步归一化 | S14；代码第 1677 行 |
| §4.1 正文 | 写明第一帧：d₁=δ̂₁，更新量为 α⊥δ₁，不做分解；投影到更新后的 d_t 上 | S14；代码第 1666–1672 行 |
| 算法 1 | 加第一帧分支、归一化、ē_t；γ=0 标注为关闭开关；删掉 “initialized to 0” | S14、R3-Q1 |
| 图 2(c) 画法 | 原始 Δ_t 先进入 EMA 方框，d_{t-1} 从下方输入，方框标为 d_t，再进入 project。由 `analysis/gen_method_diagram.py` 重新生成（只改了 panel c，原图备份在 scratchpad 的 `method_diagram.orig.*`） | R1-C5a；qpVA 回复 |
| 图 2 图注 (c) | 同上 | 同上 |

### 4. brake 与 “strictly generalizes”

| 位置 | 改动 | 原因 |
|---|---|---|
| 摘要 | 删去 “strictly generalizes ... special cases”，改为只有 const 是特例，brake 是用于对比的标量基线 | R3-Q2b；MDAW 回复 |
| 引言、贡献 (2) | 同上；明确 brake 只在 α⊥=α∥=g_t 逐 token、逐帧变化时才符合这个形式，而且 g_t 用的是上一帧 delta | 同上 |
| §4.1、§4.3 | “subsumes all ... special cases” 改掉 | 同上 |
| A.7、Table 10 图注 | “generalizes” 改为 “adds directional control”；补一句 α⊥=α∥=1 时退化为 TTT3R | 同上 |
| A.8 标题与 Setup | 标题改为 “Relation between the Brake and DDD3R”。Setup 原来写 β_ema=0 等于取上一帧方向，这是错的（β_ema=0 时 δ⊥=0），现改为显式使用上一帧 delta 作参照，并说明它不是方程 7 的某个取值 | R3-C3；MDAW 回复 |
| A.8 其余段落 | 小标题改为 “Three differences” 和 “When the two gates agree”；删去 “EMA tracking matters more” 的因果说法；总结改为 “相关的标量基线，ScanNet 上最好” | R3-Q2b |

### 5. 门控坍缩（E3）

| 位置 | 改动 | 来源 |
|---|---|---|
| 摘要、引言 M2、图 1(b) 图注、§4.2 | 限定为帧平均值近似常数；删去 “providing no (real/meaningful) temporal adaptivity” | `e1_e4.txt` E3 |
| §3.1 末句 | “not when to dampen but always too large” 改为 “幅度是一阶问题” | E3 |
| §3.2 | 补充逐 token 的时间标准差（0.038–0.102，帧平均为 0.017–0.031）和冻结测试的结果（两格无损失，两格不确定） | `e1_e4.txt` |
| §3.2 “Adaptivity does not predict” | 删去 “rules out a useful timing role”，改为帧平均方差不能预测收益 | qpVA 回复撤回 |
| §3.2 小标题、末段 | “Why no scalar gate can help” 改为 “What no scalar gate can change”；“This explains why” 改为 “consistent with ... at the frame level” | — |
| A.11 | 补一句：(a) 用的是帧平均轨迹，逐 token 的检验见 A.15 | — |
| 新增 A.15、Table 18 | 方差分解表和冻结测试表 | `e1_e4.txt` |

### 6. 漂移能量与方向（E1b、E6、E7、E8）

| 位置 | 改动 | 来源 |
|---|---|---|
| 摘要 | ortho 在 TUM 上再降 14%，但在 ScanNet、ARKit 上更差；同范数回放显示方向在 TUM 有帮助、在 ScanNet 有害；ortho 在四个位姿基准上都不显著优于 const 0.10 | `e1_e4.txt`、`e7.txt` |
| 引言 M3、贡献 (3) | “harmful repetition / useful refinement” 改为数据集相关；贡献 (3) 写入受控消融的结论 | 同上 |
| §3.3 定义 | 说明连续 delta 的漂移能量（表 2、图 7）与 γ 变体所用的量（相对 EMA 方向的对齐度再取 EMA）是两个不同的量 | S6 |
| §3.3 “dual nature” | 删去 “because ... reflects useful refinement” 和 “consistent signed signal”；补充 ARKit 的漂移能量最高，但同范数下打平，所以漂移能量不能可靠地预测方向效应的正负 | E1b P2b' 不成立 |
| §5.1 | 删去 “consistent with the drift-energy diagnosis”；补充 brake 对 TTT3R 的配对检验（−38.2%、−35.6%）和更高的 RPE 旋转；“since” 改为 “consistent with” | `e1_e4.txt` |
| 图 4 图注 | 删去 “Both crossovers are explained by drift-energy differences” | — |
| 图 3、图 4 | 宽度从 0.84 缩到 0.78 textwidth，为了把正文控制在 10 页内 | — |
| §5.1 3D 重建段 | 删去 “drift energy ... governs the appropriate operating point” | — |
| **新增 §5.2 与 Table 5** | 同范数回放的四数据集结果和配对结论；固定步长、R_100 的摘要；const 0.10 那句（含 TUM 样本量不足的说明） | `e1_e4.txt`、`e6.txt`、`e7.txt`、`e7_posthoc.txt`、`e8a.txt` |
| §5.3 敏感性 | 补充嵌套留一选参（0.0577 vs 0.0634）和 α∥=0.15 的结果；删去 “consistent with its higher drift energy” | `paired_stats.txt`、`e6.txt` |
| §5.3 推荐 (i)(ii) | (i) 默认推荐固定标量阻尼：brake τ=1 或 const 0.33，二者都是在这些基准上设定的，也没有经过验证的选择规则。(ii) ortho 在 TUM 有帮助，但不适合作默认 | `e1_e4.txt`、`e6.txt`、`e7_posthoc.txt` |
| §5.3 Beyond this backbone | 删去 ARKit 的 “correctly predicted” 和 Point3R 的 “correctly indicates”，改为如实写结果，并说明没有在 Point3R 上用方向分解 | — |
| A.4 正文、Table 7 | 删去 “drift energy governs”，改为不声称漂移能量决定配置 | E1b |
| A.9 末句 | 删去 “This is also what §3.3 predicts”，改为没有在 Point3R 上测方向 | — |
| A.10 | 删去 “measured before any pose evaluation ... predicts”；p 值更正为 0.002、0.028；补充 brake 的检验（对 TTT3R 和 const 都打平），以及同范数、固定步长、const 0.33 和 1000 帧的结果；结论改为推荐固定标量，不把漂移能量当选择规则 | `revision_sources.txt`、`e1_e4.txt`、`e7.txt`、`e7_posthoc.txt`、`e8b.txt` |
| 新增 A.14 及 Table 13–17 | 两种设计的定义；同范数全指标；固定步长全指标；const 扫描对 ortho；R_100（含舍入噪声）；ARKit 1000 帧；α∥=0.15；投影时序（E2） | `e1_e4.txt`、`e6.txt`、`e7.txt`、`e8a.txt`、`e8b.txt`、`e8_posthoc.txt` |
| 结论 | 改写为固定标量阻尼最可靠、方向是效应随数据集变化的控制轴；局限补上：没有选择规则、漂移能量不能预测正负、门控结论只在帧平均层面成立、TUM 只有 8 个序列 | — |
| 相关工作末句 | “replacing scalar magnitude control” 改为 “adding ... to” | — |

### 7. 统计规则、RPE、TUM 逐序列（新增）

| 位置 | 内容 | 来源 |
|---|---|---|
| §5 开头 | 一句配对检验规则，指向 A.12 | `stats_rule.py` |
| A.12 | 规则全文（Wilcoxon n≤16 精确、bootstrap 20000、2.2%、不做多重比较校正）；1e-7 扰动的噪声（4.7%、6.9%、13.4%、单序列最多 78%、KITTI r_rel 被判为 better）；ARKit 1000 帧 +0.2%；ScanNet 和 ARKit 500 帧没有测 | `e7.txt`、`e8b.txt`、`paired_stats.txt` |
| A.13、Table 12 | 五个方法在四个数据集上的逐帧 RPE（S4） | `paired_stats.txt`、`kitti_e5b.txt`、`e6.txt`、`revision_sources.txt` |
| A.16、Table 19 | TUM 逐序列、嵌套留一选参；写明网格是在旧版代码上跑的（差 2.2%），8 个配置里只有 3 个胜过 brake | `paired_stats.txt`、`reply_sources.txt` |

### 8. 其他小修

| 位置 | 改动 | 原因 |
|---|---|---|
| 图 1 图注 | (a) 写明 scene0759_00 用的是 \dddbrake{}；(b) 限定为帧平均；(c) “replaces” 改为 “adds” | `analysis/gen_fig1_teaser.py` 第 380 行 |
| §3.2 TTSA3R | “improving on it in every setting” 改为逐项列出显著与不显著的设置 | S10；`paired_stats.txt` |
| §4 开头、A.1、§5.3、A.6、Table 8 | “zero added memory / zero peak memory” 改为一个与状态同尺寸的缓冲区、峰值显存在所报精度下不变；删去 “runs faster ... memory write pressure” 和 “+18%” 一列，说明各配置是先后计时的，不当作加速 | S7、S8 |
| A.5 定性 | 写明轨迹图里有 brake 和 ortho 两条，场景是从 brake 改进最大的场景中选的；“confirm” 改为 “illustrate” | `analysis/viz_traj_comparison.py` |
| A.1 Hyperparameters | 删去 “already achieves strong results across all evaluation tasks” | — |
| AI 使用声明 | 补充讨论期的用途；“every reported number was regenerated ...” 改为如实说明，并承认提交版 KITTI 条目的问题。**DRAFT 注释和三项待作者决定的事项保留** | S15 |
| 可复现性声明 | “single evaluation script” 限定为 CUT3R 骨干的相对位姿；写明 τ=1 由脚本显式传入；指向 A.12 | S9 |

## 独立审阅后的修正（2026-11-08）

审阅代理没有看过我的判断，只拿到了修订后的源码、diff、结果文件、裁定和三条回复。它的结论是：所有新数字都对得上结果文件，方法描述和代码一致，回复里承诺的修改都已兑现。下面这些问题我逐条核实后确认成立，已经改掉。

| 位置 | 问题 | 改法 | 依据 |
|---|---|---|---|
| 引言贡献 (3) | 只写了“方向在 TUM 有帮助”，没有限定在 ortho 自己的步长下；也没有把 KITTI 的结果和 TUM 打平、ARKit 有害、const 0.10 不差放在一起说 | 改为：只在 ortho 自己的步长下帮 TUM；伤 ScanNet，固定小步长时也伤 ARKit；降低 KITTI 分段旋转误差；const 0.10 在四个数据集上都不显著更差 | E6b、E7 |
| 结论 | 只引用了固定步长下 KITTI 的那一格 | 同上，四个数据集一起写 | E7 |
| A.14 α∥=0.15 | KITTI 对 brake 的差距是 12.6%，小于 13.4% 的噪声，原文却当成结论 | 改为在噪声以内，删掉 p | E7 Amendment 1 |
| 摘要 | “worse on ScanNet and ARKitScenes” 读起来是相对 brake，但 ARKit 上 ortho 对 brake 是打平 | 改为 “worse than constant dampening”（两处都有 p≤0.028）；“a further 14%” 改为 “14% below \dddbrake{}”，避免读成叠加；brake 补上 “chosen on these benchmarks” | `e1_e4.txt` |
| 图 1(a) 图片 | 图里把 brake 的轨迹和 0.12 标成 “DDD3R”，旁边 (c) 又写着方向分解 | 标签改为 DDD3R_brake 后重新生成。脚本复制到 scratchpad 运行，只改了输出路径和标签，固定了随机种子；(b)(c) 与原图完全一致，只有点云采样略有不同。原脚本把输出写死在主检出目录，所以没有直接运行 | `analysis/gen_fig1_teaser.py`；生成用的脚本在 scratchpad 的 `teaser/gen_fig1_teaser_brake.py` |
| §3.3 | “On TUM aggressive suppression ... is best” 违反 E6b；“held fixed” 有歧义 | 改为在三种操作点中 ortho 的 ATE 最低；“held fixed” 改为“在 ortho 自己的逐 token 步长下”；ARKit 也限定为 ortho 的范数调度；小标题改为 “Whether suppressing drift helps” | E6b |
| §5、A.2、Table 6 表头 | 仍把 brake 写成 DDD3R 的一个操作点 | 改为分别列出 const、默认方向规则和标量 brake；表头改为 “Configuration” | R3-Q2b |
| §5.2 | “raises ATE on ScanNet ... at 0.10 and 0.20” 会被读成 ScanNet 跑了 0.20 | 改为 “ScanNet at both tested α” | — |
| §5.2 const 0.10 那句 | 只说了 ATE，没说 KITTI 分段旋转误差也打平 | 补上“KITTI 分段旋转误差上也不更差”（p=0.97） | E6d |
| §5.3 (i) | 没写 ScanNet 上 brake 对 const 0.5 打平 | 补上 p=0.31；“the brake's gain” 限定为 TUM | `paired_stats.txt` |
| §5.3 (iii) | 没写 const 0.10–0.20 在分段旋转误差上与 ortho 打平 | 补上，并注明这是次要指标（正文不叙述选指标的过程） | E6d |
| §5.3 Beyond、A.9 | Point3R 写成 “same scenes”，但 0.598 是表 2 的 96 场景均值，Point3R 是 7 个序列，按更新事件平均 | 正文只说“远低于 CUT3R”；A.9 写明各自的样本和估计方式 | A.9 原文；S6 |
| §3.2 标题和首句 | 仍然笼统地说门控坍缩 | 标题改为 “Frame-Averaged Gates Collapse to Constants”，首句改为帧平均 | qpVA 回复 |
| §3.2、§4.1 σ_spatial | 原文写 σ_spatial≈0.15、“大一个数量级”，与 E3 的门控日志矛盾（实测 0.047–0.051，是帧平均时间标准差的 1.7–2.8 倍） | 改为约 0.05、“larger”；§4.1 的 ≫ 改为 >；A.15 补上帧内标准差。原来 0.15 的出处没有查到 | `revision_sources.txt` |
| §3.2 TTSA3R | “its advantage cannot survive accumulation” 与 1000 帧时显著优于 TTT3R 矛盾；不显著的设置漏了 KITTI | 改为“更好的标量调度不能消除累积”；补上 KITTI（p=0.054） | `paired_stats.txt`、`kitti_e5b.txt` |
| A.14 同范数 | 写成记录 ‖δ̃_t‖、回放时再乘一次 β_t；代码实际记录的是 ‖β_t·δ̃_t‖，回放时不再乘门控 | 按代码改写 | `model.py` 的 `_apply_norm_replay` |
| A.14 | 缺 KITTI 旋转指标的有效性说明 | 补上：冻结朝向的轨迹得 21.87，接近 CUT3R 和 TTT3R | `kitti_metric_validity.txt` |
| A.4 KITTI 逐序列 | “strong drift adaptation (γ=2)” 方向说反了，而且单序列差异在噪声以内 | 改为最低值所在的 γ，并注明在噪声以内 | A.12 |
| A.3 | ē_t 与数据集均值 ē 混用同一个符号 | 补一句说明二者不同 | — |
| 算法 1 | 没有返回 ē_t | 已补 | — |
| §4.1 第一帧 | 写成 “the update is α⊥δ₁”，漏了 β | 改为 “the regulated delta” | — |
| Table 6 | 两个 0.055 只有一个加了下划线 | 两个都加 | — |
| A.13 | “higher rotation error” 只是均值比较，没有检验 | 改为 “higher mean” | — |
| §5.3 敏感性 | “a larger α∥ helps” 没有配对检验 | 改为 “lowers the mean” | — |
| §5.1、图 4 图注 | “ortho takes over beyond ~300 frames” 读起来像方向的效应依赖长度 | 改为两条曲线在约 300 帧处相交 | E8 |
| §5.1 3D 重建段 | 漂移能量那句引用的是 §5.2 | 改为引用 §3.3 | — |
| 结论 | “fixed-γ Pareto frontier ... sequence-level property” 在论文里没有依据 | 删除；局限部分随之压缩 | — |
| AI 声明 | “predictions tested in A.10”，但 A.10 已不再按预测来写 | 改为 “predictions for the ARKitScenes evaluation of A.10” | — |
| 可复现性声明 | “rerun for this revision” 是在叙述修订过程 | 删去 | — |
| `CHANGE_LIST.md` | τ 所在小节写成了 §4.3，实际是 §4.2 | 已改，并补上图 1 和 σ 两项 | — |

为把正文控制在 10 页以内，引言贡献 (3) 和结论的新文字又压缩了一次。两个版本的正文现在都在第 10 页最后一行结束，**没有余量**，今后正文每加一句都要同时删掉等长的文字。

不改的一项：σ_temporal≤0.03。这个数和 E3 里 ScanNet 前 20 个场景的 0.0305 只差在两位小数的舍入，而且两者用的场景集合不同。

另外，回复里的 “Table 5 caption” 在修订版里是 Table 6。这一处留到逐段过回复时再改。

## 没有改、需要作者知道的

1. **AI 声明（S15）：** 作者已于 2026-11-09 确认：M1–M3 和漂移能量假设由作者提出（AI 只是细化，写 refine）；翻译一项保持“未使用”；保留厂商名；审阅程序里的三句（读过所有 AI 写的代码、独立推导过方程 (5) 的证明、核对过全部参考文献）都属实。讨论期那句补上了 “design”，因为 E1–E8 的实验设计和预测也由 AI 起草；源码里的 DRAFT 注释换成了确认记录。**OpenReview 的 AI Assistance 勾选需要作者按这段声明自己核对。**
2. **“We will release the code”：** 提交版里已有这句，原样保留。
3. **代码默认 `--brake_tau` 仍是 2.0：** 脚本显式传入 1.0，可复现性声明写的是这一点。公开代码前要对齐。
4. **S13：** 图 1(b) 和 §3.2 里 KITTI 的门控统计 μ∈[0.31,0.35] 只查到一半。从脚本看，图 1(b) 的 KITTI 曲线来自 KITTI raw 的三段 drive（`analysis_results/a1a2_dynamics/a1_kitti_2011_09_26_drive_*`），也就是视频深度用的序列，不是 KITTI Odometry。数值和跑在哪条 RoPE 路径上都没有重算，未改。
5. **图 1 图片：** (a) 的标签已改为 DDD3R_brake；(b) 的面板标题仍是 “Gate collapse”，靠图注限定为帧平均。
6. **没有逐项核实、未改的数字：** 视频深度、7Scenes、Sintel、扩展曲线（曲线缺 tum_s1_150 和 scannet_s3_500 两个 brake 点），以及 FPS 原始数值。
7. **STRATEGY_PLAN 里 “不改标题” 的决定照旧执行。**
8. **E9（2026-11-09）不成立，论文不改：** 用前 200 帧漂移能量 ē 选配置（ē<0.5 走 ortho，否则走 brake τ=1）。这条规则在 TUM、ScanNet 上定下来，再拿到留出的 KITTI Odometry 和 Sintel 上检验：
   - 结果：KITTI 的 ē 正好卡在阈值附近（均值 0.501），11 条里 5 条走 ortho。在唯一有区分度的 r_rel 上，规则比 ortho 差（15.24 vs 11.88，p=0.031），P9a 不成立。
   - 后果：“no validated rule” 原样保留，A.10 也不用改。按登记，这个 rebuttal 期间不再在这三个数据集上试别的阈值或统计量。
   - 顺带核实了第 6 条里的 Sintel 位姿列：当前代码下 ortho 0.2361、brake τ=1 0.2372，与表 4 的 0.236、0.237 一致，说明表里的 brake 是 τ=1；TTT3R 现在是 0.2078，表里 0.209 来自旧版运行（0.20879），相差 −0.46%，未改。
   - 详见 `docs/PREREG_E9.md` 的结果节和 `results/e9.txt`。
