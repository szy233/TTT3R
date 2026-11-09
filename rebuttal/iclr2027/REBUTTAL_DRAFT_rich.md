# 回应草稿说明 — ICLR 2027 Submission 28147

发给评审的文本是三个文件，每位审稿人一条，上限都是 5000 字符：

| 文件 | 字符数 |
|---|---|
| `PASTE_READY_MDAW.txt` | 4953 |
| `PASTE_READY_qpVA.txt` | 4953 |
| `PASTE_READY_PSN7.txt` | 3577 |

E7 的改动、写入后独立审阅的修正和 E8 的改动都已于 2026-11-08 写入，换行符按两个字符计时为 4973、4996、3602（逐段重过后的数字）。MDAW 已没有余量。写入前的版本备份在 scratchpad 的 `PASTE_READY_*.pre_e7.txt`、`PASTE_READY_*.pre_audit.txt` 和 `PASTE_READY_*.pre_e8.txt`。

这份文件只给作者看：覆盖情况、数字来源、发出前要确认的句子和可选内容。

三条都发成 official comment。读者至少包括对应的审稿人和 AC。讨论是双盲的，文本里没有姓名、单位、仓库或机器信息。表格是 Markdown 格式。

## 发出前要确认的句子

| 句子 | 位置 | 为什么要确认 |
|---|---|---|
| "We will post a revised PDF by Nov 18 ..." / "We will add this ablation to a revised PDF by Nov 18." | 三条的结尾 | 公开承诺。修订版还没开始 |
| "... The paper prints τ=2, which the revision corrects." | qpVA 开头段、PSN7 检验段 | 同样依赖修订版。不交修订版时改写这句，τ 的说明本身要留下 |
| "The paper's KITTI Odometry column did use τ=2." | MDAW 的 Q2 | 主动披露 KITTI 的一部分。只说 TUM、ScanNet、ARKitScenes 是 τ=1 而不提 KITTI，读者会以为没有例外，所以建议保留 |
| "We recommend one fixed scalar (brake τ=1 or const 0.33, ... and const 0.33 not yet run on ARKitScenes), and ortho only where validated." | qpVA 的 W1 | 论文立场的调整。E7 已在 ARKitScenes 上跑了 const 0.33（与 brake 打平，p=0.78），括号里的 “not yet run” 已不成立，已删掉 |
| "We had predicted ..." | PSN7 | 属实（本地提交 f123a0c、3b6789b），但评审无法核实，因为分支没有公开 |
| "So direction is a separate lever, not the only route." | MDAW W1、qpVA W3、PSN7 | E7 的 KITTI 模式落在 “其他模式” 那行，不能做概括，所以已删掉这句。E6b、E6d 要求的“缩放也可能进入噪声范围”改由 “Const 0.10 alone is not worse than ortho on any of the four datasets” 满足 |
| "A 1e-7 change to one multiplier ... moved means by up to 4.7% ..., 6.9% ... and 13.4% ..." | 三条的检验说明 | Amendment 1 的第一行要求写出测得的差距。只测了一次，是下限 |
| "We keep decomposition as a separately tested control axis ..." | MDAW、qpVA 结尾 | 论文立场的调整 |
| "... so we withdraw “best aggregate for outdoor driving”." | qpVA W3 | 公开撤回论文的建议 (iii)，修订版要同步 |

## 覆盖情况

16 项全部有回答。

| 问题 | 位置 |
|---|---|
| R1-C1 选择规则 | qpVA W1 |
| R1-C2 ortho 不稳定 | qpVA W2 |
| R1-C3 方向的价值 | qpVA W3 |
| R1-C4 门控坍缩 | qpVA W4 |
| R1-C5a EMA 时序、R1-C5b γ 方向 | qpVA W5 |
| R2-C1、R2-Q1 同范数 | PSN7 |
| R3-C1 方向的独立贡献 | MDAW W1 |
| R3-C1b 超参数来自 TUM、R3-Q3 配对统计和留出 | MDAW 的逐序列表 |
| R3-C2、R3-Q1 γ 的实现和不连续 | MDAW 的 γ |
| R3-C3 A.8 的 Setup、R3-Q2a d_{t-1} 对 d_t | MDAW 的 EMA |
| R3-Q2b brake 是否特例 | MDAW 的 Brake |

## 数字来源

| 内容 | 来源 |
|---|---|
| E1b、E2、E3、E4，ortho 对评审点名的方法，brake 对 TTT3R，RPE | `results/e1_e4.txt`，由 `scripts/e1_e4_rulings.py` 生成 |
| TUM 逐序列、留出值、嵌套留一、2.2%、真 τ=2 | `results/paired_stats.txt`，由 `scripts/paired_stats.py` 生成 |
| KITTI 同范数和平标量、TUM 平标量、α∥=0.15、const 0.33 对 brake 在 TUM 的结果 | `results/e6.txt`，由 `scripts/e6_rulings.py` 生成；裁定在 `docs/PREREG_E6.md`、`docs/PREREG_E6D.md` |
| γ 扫描 | `eval_results/relpose/scannet_s3_1000/ttt3r_ortho_steep_v2_g*` |
| 公式、图 2(c)、表 5、A.3、A.4、A.8 的编号和原文 | 提交版 PDF，已对照 |

## 独立审阅后的改动

一个没有看过我结论的审阅代理逐条核对了合并版。它重算了结果文件，数字没有出错。根据它的意见：

- 删掉了 “None of 58 other logged configurations ...”。这句没有信息量：59 个日志运行里，没有一个在 ScanNet 上低于 brake，却有 33 个在 TUM 上低于 brake，brake 本身就是被挑出来的最优者。
- “Direction has an independent effect” 改为实际发生的登记结果：方向在 TUM 有帮助，在 ScanNet 有害，但 ScanNet 的损失不全来自方向。
- 标量阻尼的结论加了限定：只说 1000 帧的增益，ARKitScenes 打平，RPE rot 变差。
- brake 的尺度效应从“就是尺度效应”改成“不再归功于时间自适应”，并给出 const 0.5 的 p 值。
- 打平的结果补上 CI，这是登记要求。
- 冻结门控注明是 oracle，TTSA3R 注明是融合门控。
- 选参的结论收窄，并说明 8 个配置里有 4 个输给 brake。

拆成三条之后，内容沿用上一版，新增的是 ARKitScenes 上对 TTSA3R 的比较和 TUM 逐序列的留出值。最新的三条文本另外审过一次，见下面“第二次独立审阅”。所有小数都由脚本对照过结果文件，但这个检查只确认数字存在，放对了位置靠人工核对。

逐条中文对照时又改了 7 处，作者已确认：

- qpVA、PSN7 补上 brake 用 τ=1 而论文印 τ=2 的说明。拆成三条后，这一点原本只留在 MDAW 那条。
- PSN7 补上 “tied” 的定义。
- qpVA 的 W1 写明 const 0.33 还没在 ARKitScenes 上跑过。
- MDAW 的 “Only the γ sweep uses this branch” 改写。原句紧跟 γ=0，会被读成只有扫描用 γ=0，而 ortho 主结果都是 γ=0。
- MDAW 的 KITTI 写明是 KITTI Odometry。视频深度的 KITTI 是 τ=1。
- MDAW 的 EMA 表写明 CI 方向，改用结果文件里的 updated − previous。
- MDAW 的 “pace” 改为 “norm”。pace 是 PSN7 的用词。

E6、E6d 裁定后的改动，作者已同意写入：

- “direction helps on TUM only” 改为方向在同范数下有独立作用，符号随数据变化。
- MDAW、qpVA 新增 KITTI 同范数结果及其代价，并写明 r_rel 是看过标量对比后才选的；同时写明平标量可能进入噪声范围。PSN7 加简短版。
- MDAW、qpVA 新增 α∥=0.15 的敏感性结果，并写明是事后选的、brake 仍更好、KITTI 上不如默认值。PSN7 只写 TUM 和 ScanNet。
- qpVA 的 const 0.33：TUM 从打平改为 brake 略好（同代码，p=0.039），结论从“不再归功于时间自适应”改为“大部分归功于尺度”。
- qpVA 的 W2 限定为默认 α∥=0.05，并撤回 “best aggregate”。PSN7 改为 “changes part of the conclusion”。
- MDAW 为腾空间删掉两处：“Wins over brake” 一行（可从逐序列表看出）、“p=0.0078 is the smallest possible”（改成表内标注）。

## 第二次独立审阅

E6d 改写写入后，一个没看过我判断的代理审了三条文本，结论是需要修订。每一项我都对照日志或登记原文核实过，复算脚本是一次性的，结果记在这里。

**已改：**

- 逐句规则：真 τ=2 对 const 0.5 是打平（p=0.195、0.071），不是更差。4 个配置对 brake 也是打平，均值更高（p=0.15–0.95），不是输。
- RPE：TUM 上 brake 对 TTT3R（p=0.74）、ortho 对同范数（p=0.31）配对都打平，所以改说“均值更高”。
- qpVA：const 0.5 在 ARKitScenes 上好于 TTT3R（13/18，p=0.028），所以把“标量阻尼打平”改成只说 brake。
- “direction has its own effect” 改为“同范数下方向在 TUM 有帮助、在 ScanNet 有害，且同范数也输给 brake”。这是 E1b 实际发生的那一行要求的说法。
- α∥=0.15：写明对比值，“不如默认值”改成明确的 “worse”，ARKitScenes 补上“与同范数和 brake 打平”。
- MDAW 的 17.66 是 τ=1 重跑，原句读起来像 τ=2。
- qpVA 的 α∥=0.15 段移到 W3 之后，因为它用到 W3 才定义的 “same norm”。
- 术语：t_rel 改 “segment translation”，flat α 改 “const”。
- qpVA 撤回 “best aggregate” 的理由改为“ATE 上只胜过 CUT3R”，这是 E5b 的登记裁定。
- “new runs” 改 “runs”，因为 γ 扫描、τ=2 和部分基线来自日志。
- A.3 的 γ=0 句从“错误”名单里拿掉：按代码的开关它是对的。
- PSN7 补上事后选定、r_rel 选择时机、扫描边界等披露，并按 E1 的登记报告 `normmatch`。
- `p≥0.11` 改 `p>0.10`，实际最小值是 0.1094。

**有意没改或做了取舍：**

- MDAW 的打平结果没有全部带 CI，登记要求带上，但放不下。完整数字在 `results/e6.txt`、`results/e1_e4.txt`。
- E6b、E6d 的 Holm p 只记在登记文件的日期节，回复开头已写明 “no multiplicity correction”。
- 为控制长度，MDAW 删了 “β_ema=0.9: 0.0755”；qpVA 删了 “just above our 2.2% floor” 和 “with a documented ScanNet failure”（W2 的表已经写明）。

**审阅者也没能核实：** 论文各图是否都用 γ=0、OpenReview 的表格渲染、KITTI 上回放后的实际范数（靠构造和单元测试保证）、Nov 18 交修订版是否可行。

修改之后只重跑了字符数和数字检查，没有再审第二遍。

## 新增 “What stands”

原来三条几乎全是让步，没有说明论文还有什么站得住。作者同意后，三条都加了 “What stands: training-free scalar dampening lowers ATE against TTT3R at 1000 frames ...”，数字来自 `results/e1_e4.txt` 的 brake 对 TTT3R。MDAW、PSN7 紧接着写明 ARKitScenes 上打平；qpVA 的打平写在 W3。

为腾位置做的取舍：

- α∥=0.15 那段压缩：删掉 TUM 的 0.0545，ARKitScenes 的 0.567 vs 0.609，KITTI 的 15.44 vs 17.66，以及 qpVA 的 0.261、11.88。结论和 p 值都保留。
- MDAW 这段还删了 “brake stays better”。读者可以从 0.399 和表里的 brake 0.2615 看出来。qpVA 保留这句。
- MDAW 的 “The brake gains you cite ...” 由新句取代，“The 17.66 above is τ=1” 改为在正文写 “brake τ=1”。
- qpVA 的 “Brake's ATE gains hold ...” 拆开：收益进结尾，ARKitScenes 打平留在 W3。
- 其余是不影响内容的措辞压缩，例如 “runs on the same checkpoint” 改 “on the same checkpoint”。

这些改动同样只跑过字符数和数字检查，没有再做独立审阅。

## 逐位核对后的改动

作者要求按审稿人逐条核对并给中文对照。三条的数字、判定、代码和论文引用都对得上，唯一没能逐一确认的是论文每张图是否都用 γ=0。作者同意后改了 13 处：

- **qpVA 5 处：** const 0.33 那句的主语改为 “const 0.33 stays slightly worse”，让数字顺序和 CI 方向一致；删掉 “about brake's typical multiplier”，因为结果文件里没有这个说法的实测值；“at our sweep's edge” 改为 “the smallest value we ran”；“shrink CI” 改为 “narrowed by 0.071, CI”；开头改为 “answer each below. All runs use the same checkpoint.”
- **PSN7 3 处：** 结论段拆成两段并调整顺序，“it” 因此改为 “direction”；边界说法改为 “the smallest we ran on KITTI”；ARKitScenes 上的两处打平补 CI，const 补 “p>0.05”。
- **MDAW 5 处：** 开头改为 “All runs use the same checkpoint.”；边界说法改为 “the smallest we ran”；“shrink CI” 同样改写；恢复 “and brake stays better”；为腾位置，标题改为 “W1, Q3. Direction.”，并删掉 “moving”。

这些改动在作者看过对照后写入，并跑过字符数和数字检查，但没有再做独立审阅。

## E7 裁定后的改动（2026-11-08 已写入）

裁定见 `docs/PREREG_E7.md` 的 “Results, 2026-11-08, E7”。改动由 scratchpad 的 `build_e7_candidates.py` 生成，跑过字符数、数字来源、残留措辞和身份信息检查，作者同意后写入。

三条都改的地方：

1. **检验规则那句。** 删掉 “the largest we saw between code versions of one configuration”，加上测得的舍入差距：一个乘子改 1e-7，均值最多变 4.7%（TUM ATE）、6.9%（KITTI ATE）、13.4%（KITTI 旋转误差），只测一次，ScanNet、ARKitScenes 没测，小于这些的差距属于舍入噪声以内。
2. **同范数那句加限定。** “at identical norm” 改为 “at ortho's own norm schedule”（KITTI 那句是 “at ortho's norm schedule”）。这是 P7a 打平那行的要求。
3. **新增 E7 一段。** 保留 const α 的逐 token 步长，只换成分解方向：KITTI 旋转误差在 α=0.10、0.20、0.33 都更低（7.68 vs 12.41、7.88 vs 16.15、9.13 vs 17.61，8/11、9/11、11/11，p≤0.014），代价是 0.10 的 ATE 更差、0.10 和 0.20 的 RPE 平移更差；TUM 打平（p≥0.31）；ScanNet 两个尺度 ATE 都更差（胜 ≤5/65）；ARKit 在 0.10、0.20 更差（≤3/18），0.33 打平。const 0.10 在四个数据集上都不比 ortho 差，ScanNet 上更好（0.382 vs 0.488）。PSN7 多一句：const 0.10 的 RPE 旋转在 TUM、ScanNet、ARKit 更高（p<0.01）。
4. **删掉** “So direction is a separate lever, not the only route.” 和旧的平标量句（被第 3 条取代）。

MDAW、qpVA 另外：

- 删掉 α∥=0.15 整段（登记里只是 “可以说”，事后选的参数，不能回答 MDAW 要的 “独立选出的超参数”），腾出字数。其中 KITTI “beats brake (p=0.0098)” 本来就必须标为噪声以内，删掉后不再需要。
- MDAW 逐序列表的 CI 和 p 两行合成一行，“Selection ... gain. But ...” 合成一句。
- qpVA：W1 删掉 “and const 0.33 not yet run on ARKitScenes”；const 0.33 那段加上 ARKit 与 brake 打平（0.5405 vs 0.5391，p=0.78），TUM 上 “2.3% worse ... within rounding noise”。

没有改的：What stands、修订版承诺、τ 说明、W2/W4/W5、Q1/Q2。

### 写入后的独立审阅（2026-11-08，修正已写入）

审阅代理没有看过我的判断，结论是 “needs revision”。我逐条复核后，下列修正成立，由 scratchpad 的 `build_audit_fixes.py` 生成，作者同意后写入（MDAW 4954、qpVA 4935、PSN7 3367 字符）：

- MDAW 的 “4 of the 8 configs tie brake at higher means” 结果文件里没有来源。重算（`results/reply_sources.txt`）显示只有 3 个配置胜过 brake，另外 5 个打平，其中 1 个均值更低。改为 “only 3 of the 8 configs beat brake”。
- MDAW 的 τ=2 “tied” 原来也只有均值。现在 `results/reply_sources.txt` 有配对检验：TUM p=0.195，ScanNet p=0.071，打平。正文不用改。
- PSN7 的 “direction adds a further gain on TUM” 读起来像 E7 禁止的叠加说法，改为 “helps”。
- const 0.10 那句补上指标：“in ATE”，并加 “at higher RPE rotation except on KITTI (p<0.01)”，三条统一。
- PSN7、MDAW 的 What stands 写明 brake 是在这些基准上定的（“set on these benchmarks”）。
- “KITTI rotation error” 改为 “KITTI segment rotation error”，和 RPE 旋转区分。
- MDAW 的 “mean RPE rotation worsens” 改为 “is higher”，因为没有配对检验。
- qpVA 删掉 “and ortho only where validated”。没有哪个数据集显示 ortho 在主指标上胜过所有平标量。
- qpVA W2 的 “Confirmed for the default α∥=0.05” 改为 “Confirmed on ScanNet. On ARKitScenes it ties TTT3R.”，因为表里 ARKit 对 TTT3R 是打平。
- 为字数做的压缩：MDAW 的 “identical by construction” 去掉 “by construction”，EMA 那两句合成一句，表头并入 “TUM 1000f”；qpVA 开头删掉 “and answer each below”；E7 段开头改为 “With const α's per-token step length, the decomposed direction lowers”。

没采纳：MDAW 补 KITTI τ=1 重跑的说明（回复里没有引用 KITTI 上 brake 的数字，也没有字数）；qpVA、MDAW 点明 “预测失败”（结果和撤回都已写，PSN7 点明了）；MDAW 补 ARKit 对平标量的检验（qpVA 的表里有，MDAW 没有字数）。

## E8 裁定后的改动（2026-11-08 已写入）

E8 的裁定见 `docs/PREREG_E8.md`。现有句子都不违反它，改动只有两处，由 scratchpad 的 `build_e8_candidates.py` 生成，作者同意后写入：

- 三条的噪声说明从 “(one draw, not run on ScanNet or ARKitScenes)” 改为 “… ARKitScenes 500f)”。E8b 在 ARKitScenes 1000 帧上测过舍入噪声（ATE +0.2%），原句字面上已不成立；回复里 ARKitScenes 的数字都是 500 帧，500 帧仍然没测过。为抵字数，MDAW 的 “only 3 of the 8 configs … none of this” 改为 “only 3 of 8 configs … none of it”。
- PSN7 的 E7 段，在 ARKitScenes 的 ATE 之后加一句逐格报告（E8a，平范数，R_100）：“Measured over 100-frame gaps, rotation error with direction also drops on KITTI, but rises on ARKitScenes at all three α, on ScanNet at both and on TUM at 0.10 (other cells tied).” 这句不是一般性陈述，符合登记的 “Report each cell”。

没采纳：MDAW、qpVA 也加这一句（没有字数）。E8b 的 1000 帧结果不进回复，因为全部打平，n=8，放进修订版。

E8 对修订版的约束：不能说方向在 ARKitScenes、TUM 或 ScanNet 上降低旋转漂移，不能说方向的效应取决于序列长度，不能说方向改善 ARKitScenes 精度。

## 与作者逐段重过（2026-11-08 起，已写入的改动）

顺序是 MDAW → qpVA → PSN7。三条共用的段落只过一次，结论同步到另外两条。每一步改动前的版本都备份在 scratchpad：`PASTE_READY_*.pre_pass1.txt` 是第 1 项之前的，`PASTE_READY_*.pre_item4.txt` 是第 4 项之前的。

| 项 | 段落 | 改动 | 原因 |
|---|---|---|---|
| 1 | 开头的检验说明 | 三条共 4 处 "rounding noise" 改为 "numerical noise" | 和修订版 §5、A.12 的用词一致；1e-7 扰动引起的是数值发散，不是舍入本身 |
| 2 | MDAW 同范数段和表 | 不改 | 数字全部对过 |
| 3 | MDAW KITTI 段 | 不改 | — |
| 4 | 固定步长段（三条共用） | const 0.10 那句补上 "nor in KITTI segment rotation error"（12.41 vs 11.88，p=0.97）；为抵字数，三条都删掉 "wins 8/11, 9/11, 11/11"，MDAW 再删掉 "8.04 vs 4.65" | E6d：不能暗示只有分解才能降低 KITTI 旋转误差。这层意思是在把那句统一为 ATE 时丢掉的 |

| 5 | MDAW TUM 逐序列表 | 不改 | — |
| 6 | MDAW γ 段 | 改得更紧凑：写明已在修订版改正，并注明新表号（Table 6）；ScanNet 那句和"连续性"那两句改写或合并 | 作者 2026-11-08 说明：ICLR 可以多轮回复，字数上限不再是约束，但措辞要精简。改后默认修订版先于回复上传 |
| 7 | MDAW EMA 段 | 补上 "all numbers come from the code"；"is wrong" 改为 "was wrong"；与修订合成一句（Eq. 7、算法 1、A.8、图 2(c)）；表前加引导语 | 让 MDAW 知道公式错误不影响结果 |
| 8 | MDAW brake 段 | 写明修订版删去 "strictly generalizes"，并已按 τ=1 重跑 KITTI；"while" 改为 "but"，补上 "A" | 交代已改正，避免读者以为 KITTI 那列仍是 τ=2 |
| 9 | MDAW 结尾 What stands | RPE 旋转的代价并入第一句（与 qpVA、PSN7 一致）；"at 500" 改为 "at 500 frames"；修订版承诺改为“修订版已做这些更正（标蓝），变更清单另发一条评论” | 作者同意先上传修订版、再发回复；变更清单作为面向全体审稿人的单独评论 |
| 10 | MDAW 整条压缩 | 共 8 处：开头合句，噪声范围的限定改为指向 A.12；ARKit 打平去掉 CI；固定步长段 "With … the decomposed direction" 改为 "At … direction"；brake 段缩短；EMA 段的对比表改成一句话；结尾删掉与同范数表重复的 ATE 数值；修订版那句缩短 | 精简，并回到单条 5000 以内（4973） |
| 11 | 共用段同步到 qpVA、PSN7 | 开头三处和固定步长段首句，与 MDAW 用同一写法 | 三条保持一致 |
| q1–q2 | qpVA 开头独有部分、W1、W2 表 | 不改 | — |
| q3 | qpVA W3 同范数段 | "the norm, not the trajectory" 并入段首句，删掉后面单独那句；ARKit 打平去掉 CI | 与 MDAW 写法一致 |
| q4 | qpVA KITTI 段 | 去掉 RPE 平移的具体数字，只留 (0/11)；撤回 "best aggregate" 那句不变 | 与 MDAW 写法一致 |
| q6 | qpVA const 0.33 段 | 合句；括号里的 p 值明确是 brake 对 const 0.5 的差距本身（TUM p=0.016，ScanNet p=0.31）；ARKit 上 brake 与 TTT3R 打平那句挪到结尾 What stands | 原写法有歧义；三条回复结构一致 |
| q7 | qpVA W4 门控坍缩 | 末句补上修订版的处理：把坍缩结论限定在帧平均层面（§3.2、A.15） | 让审稿人知道论文已改、新证据在哪 |
| q8 | qpVA W5 公式 | 写明修订版两处都已改；"are wrong" 改为 "were wrong"；注明新表号；补上 Eq. 12 漏写 EMA，以及所有数字都来自代码 | 与 MDAW 一致，并说明公式错误不影响结果 |
| q9 | qpVA 结尾 | 与 MDAW 相同：删掉 ATE 数值、只留胜场和 p；修订版那句改为已完成；ARKit 那句由 q6 挪入 | 三条一致 |
| q10 | qpVA 整条压缩 | 开头两处合句；const 0.33 段中打平的两组均值去掉；W4 首句、冻结引导句、末句去重；W5 两处缩短；结尾删掉与 W1 重复的 "set on these benchmarks"。W2 的 ARKit CI 按作者意见保留 | 精简，并回到单条 5000 以内（4996） |
| p1 | PSN7 开头、两张表、检验说明 | "pace is identical by construction" 改为 "pace, not trajectory, is identical"，并把"direction 指闭环效果"提到首段，删去后文重复的那句；τ 的两句合并 | 读者在看表前就知道比的是什么；与另两条一致 |
| p2 | PSN7 结论段 | 去掉同范数对 TTT3R 打平时的 CI；"higher mean RPE" 写明为旋转；预测失败那两句合并 | 精简，指标写清楚 |
| p3 | PSN7 KITTI 与 α∥=0.15 | 两处补上 p（0.016、<0.001） | 与回复里其他比较一致 |
| p4 | PSN7 的 R_100 句 | "(other cells tied)" 改为 "(the others tie)" | 去掉内部说法，与修订版一致 |
| p5 | PSN7 闭环标量段 | 删掉三个 ATE 数值，改为说明它没能固定范数、所以改用回放，并指向修订版 A.14 | 这些数值没有控制范数，容易被误读成方向有害的证据 |
| p6 | PSN7 结尾 | 与另两条对齐：删掉与表重复的 ATE 数值，ARKit 去掉 CI；补上定位句；修订版那句改为已完成，并指向 §5.2、Table 5、A.14 | 三条一致；PSN7 关心的消融在哪里可以直接找到 |

改完后的字数：MDAW 4933（换行按两个字符计 4973），qpVA 4953（4996），PSN7 3577（3602）。

**交叉检查（2026-11-09）：** 三条回复中，共用的检验说明、同范数表、固定步长段和结尾的核心句逐字一致；252 个小数都能在 `results/*.txt` 里找到出处；引用的 §3.2、§5.2、A.4、A.8、A.12、A.14、A.15、Eq. 6/7/12、Figure 2(c)、Algorithm 1、Table 5/6 都与修订版 PDF 的编号一致。三条都默认修订版先上传，变更清单（`CHANGE_LIST.md`）另发一条面向全体审稿人的评论。

## 可选内容

需要时再用。

**同范数的闭环变体（评审没有问）：**

> A closed-loop scalar using ortho's gain formula gives TUM 0.0761, ScanNet 0.2858, ARKitScenes 0.5275, but its mean gain runs 23-27% above ortho's, so it does not answer "same norm".

**KITTI 的完整披露：**

> While rechecking, we found that the KITTI column of Table 3 printed ATE mean under an RMSE label and did not come from one protocol. A rerun of all methods under one protocol gives ortho -21.1% against CUT3R on ATE RMSE, significant only against CUT3R, so we withdraw "best aggregate for outdoor driving" and "strongest test of domain shift".

“best aggregate” 的撤回已写进 qpVA，这一段现在只多出 “strongest test of domain shift” 和表 3 的来源说明。

**r_rel 的有效性（未登记，E7 之后）：**

> A trajectory with ground-truth positions and frozen orientation scores 21.87 deg/100m, near TTT3R (20.60), so rotation error does not reward under-rotation here, but the baselines are about as poor as not turning.

**α∥=0.15 的压缩版（为字数从 MDAW、qpVA 删掉）：**

> At α∥=0.15, chosen post hoc, TUM still beats same norm (0.0550 vs 0.0725, p=0.016), ScanNet's loss to same norm narrows but remains, and ARKitScenes improves on the default (p=0.012) but ties same norm and brake.

## 修订版要补的内容

除 `STRATEGY_PLAN.md` 的修订项外，这一轮新增：

- 不能写成只有分解才能达到这些结果：TUM 和 KITTI r_rel 上，α=0.10–0.20 的平标量都与 ortho 打平。KITTI 的扫描停在 α=0.10，那里同时是最低均值所在的边界。
- α∥=0.15 只作敏感性结果，不改默认值。
- E7：固定步长下，方向只在 KITTI 旋转误差上三个尺度都更好，TUM 打平，ScanNet、ARKit 更差。不能写 “方向和尺度可以叠加” 或 “方向在各个步长都有收益”。
- const 0.10 在四个数据集的主指标上都不比 ortho 差。ortho 在 RPE 旋转上更好，但这来自步长，不是方向。
- 舍入噪声：1e-7 的扰动能让 KITTI r_rel 被判为 “更好”（9/11，p=0.0137）。修订版的显著性判定要写出测得的差距，并说明只测了一次。
- brake 相对 const 0.5 的差距，大部分（TUM 约 90%）来自尺度，不是全部。
- A.3 的 “Setting γ=0 recovers the fixed operating point” 只靠代码里的开关成立。按方程 12，γ=0 时 e^0=1，得到的是各向同性。修订时要把开关写明。
- 图 2(c) 的绘图本身也要改，不只是图注。面板里写的是 EMA d_{t-1}。
- 摘要的 “tuned per scene”、引言的 “self-adjust across scenes” 没有证据支持。
- 摘要和 §3.3 中“漂移能量决定方向效应”的说法，以及 §5 “drift energy measured before any pose evaluation correctly predicted which operating point would fail”，都要对照 E1b 重写。
- `sec/analyse.tex:69` 的 “rules out a useful timing role”，以及摘要的 “no real temporal adaptivity” 和引言的 “no meaningful temporal adaptivity”。
- 第一帧的处理：代码用 α⊥δ，算法 1 从 d=0 开始。
- AI 使用声明和可复现性声明（S9、S15）。
