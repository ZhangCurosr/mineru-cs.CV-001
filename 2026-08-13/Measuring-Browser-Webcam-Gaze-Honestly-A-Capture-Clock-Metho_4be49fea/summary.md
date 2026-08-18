---
title: "Measuring-Browser-Webcam-Gaze-Honestly-A-Capture-Clock-Metho"
source: https://arxiv.org/pdf/2608.11566v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 12:33:49"
field: "浏览器端摄像头眼球追踪与延迟计量"
keywords: ["webcam eye tracking", "latency measurement", "rVFC", "weakly-supervised segmentation", "GazeMedSeg", "browser-based gaze"]
innovations: ["基于 rVFC 捕获时钟的浏览器 webcam gaze 延迟测量方法，支持精确配对与可证明下界配对两档策略", "开源 TypeScript 基准测试平台与两套可互换引擎（WebGazer / FaceMesh+KRR）的统一度量管道", "通过固定下游弱监督分割管线量化摄像头 gaze 精度边界： expert EyeLink Dice 0.68 vs. webcam Dice ≈ 0"]
benchmarks: ["Kvasir-SEG polyp segmentation", "16×8 sweep grid accuracy/latency", "12×8 drift grid accuracy/latency", "Grid-resolution scaling (L1–L6)"]
---

# 论文速读：Measuring-Browser-Webcam-Gaze-Honestly-A-Capture-Clock-Metho

## 一句话总结
本文提出了一套基于浏览器 rVFC API 的**捕获时钟**方法，纠正了浏览器摄像头眼球追踪系统中因时间戳误用导致的"虚假零延迟"测量缺陷；在新测得的真实延迟下，评估了两个引擎的精度与下游弱监督医学分割的可行性边界。

## 研究问题与动机
- **测量缺陷普遍存在**：浏览器摄像头眼球追踪管线包含视频解码、推理循环、渲染管线等多个异步阶段，但主流做法是在样本发出时用 `performance.now()` 打时间戳，而非记录帧捕获时刻；缺失的捕获时钟默认为发出时钟，导致报告的推理延迟恒为约 0 ms。
- **真实延迟被严重低估**：论文作者在为自己的论文采集数据时两次撞见此 bug，FaceMesh 引擎报告 0.00 ms 中位推理延迟（7720 个样本），改用 rVFC 后升至 22/27 ms（中位/p₉₅）；WebGazer 同样被纠正为 34/52 ms。
- **延迟差距直接影响交互预算**：20–50 ms 的真实延迟足以决定一个 50 ms 交互延迟预算是否达标，对浏览器端实时应用（手术视频嵌入、crowd-scale 采集）具有重要工程意义。
- **现有工作未使用 rVFC 测量延迟**：尽管 rVFC API 可提供每帧的 `captureTime`/`presentationTime`，但截至本文尚无浏览器眼球追踪工作将其用于推理延迟测量。

## 核心贡献（创新点）
- **C1 捕获时钟方法学**：利用 rVFC API 恢复每帧捕获时钟，对暴露推理流水线的引擎通过 FIFO 队列实现精确帧配对，对隐藏流水线的引擎（如 WebGazer）提供可证明的下界近似；与已有工作本质区别在于首次将 rVFC 用于浏览器眼球追踪的延迟测量，并给出精确配对与下界配对的统一框架。
- **C2 开源 TypeScript 参考实现**：在两个可互换引擎（WebGazer 基线 + 新建 FaceMesh+KRR 管线）上实现度量方法，并提供完整的分析套件；与已有工作的区别在于实现了跨引擎的可比较基准测试平台，且将控制层（One-Euro 滤波、I-VT 固定点分类）标准化。
- **C3 实证发现揭示延迟真实规模与精度边界**：在普通硬件上揭示 20–50 ms 延迟间隙；分离空间散布与时间抖动两个精度维度；下游临床探针显示摄像头眼球追踪精度（4–7°）不足以支撑细粒度病灶标注（Dice≈0 vs. 专家眼动仪 0.68），界定其粗粒度区域提示的适用边界。

## 方法详解
- **捕获时钟来源**：通过 `requestVideoFrameCallback (rVFC)` API 获取每帧元数据，优先使用 `captureTime`（本地摄像头流可用），否则回退至 `presentationTime`（解码帧提交合成的时刻）；回退情况下每个恢复的延迟均为可验证的下界。在收集设备上，`presentationTime` 中位仅落后 `captureTime` 0.6 ms（p₉₅ 0.9 ms，最大 1.7 ms），近似极紧。
- **精确配对（FIFO 队列，适用于暴露流水线的引擎）**：每帧处理前将当前帧时钟推入 FIFO 队列 `q_F`，引擎发出样本时出队并配对：`t_c^(k) = q_F[k]`。要求引擎按到达顺序处理帧且每帧发出一个样本；FaceMesh+KRR 满足此条件。
- **下界配对（适用于不暴露流水线的引擎，如 WebGazer）**：维护标量 `t̃_c = max{τ_i : i ≤ now}`（最新观测帧时钟），每次样本发出时打上该时间戳；由于实际源帧到达时间 `t_c^true ≤ t̃_c`，因此 `t̃_ℓ_I = t_e − t̃_c ≤ t_e − t_c^true = ℓ_I^true`，报告值为可验证下界。
- **延迟定义**：推理延迟 `ℓ_I = t_e − t_c`（引擎从捕获帧产生 gaze 估计的耗时）；管线延迟 `ℓ_P = t_r − t_c`（捕获到渲染交接，`t_r` 在下次 `requestAnimationFrame` 回调采样）。
- **FaceMesh+KRR 引擎**：13 维 MediaPipe 特征向量（每眼水平/垂直虹膜位移 4、两眼眼内角距离 2、左右眼水平不对称 1、归一化虹膜与角坐标 6），经 RBF 核岭回归映射到屏幕坐标；`γ` 由标准化校准特征的 median-pairwise-distance 启发式设定（每次拟合更新），`λ = 10⁻³` 固定。
- **基准协议**：Sweep 任务（全屏 16×8 网格，3 s 停留）；Drift 任务（随机 N=10 子集，2 s 停留，考察校准衰减）；每样本记录 `(t_c, t_e, t_r)` 及目标/gaze 坐标。

## 实验与结果
- **硬件与受试者**：单用户（N=1），14 英寸 MacBook Pro（M4, 2024），集成摄像头，Chrome (arm64) on macOS Sequoia 15.6；距离 60 cm，固定光照与姿态，窗口 1890×1071 CSS px。
- **延迟结果（Table 1）**：
  - FaceMesh+KRR：推理延迟中位 22.0–22.8 ms，p₉₅ 26.8–27.0 ms；管线 p₉₅ 27.3–27.4 ms。
  - WebGazer（下界配对†）：推理延迟中位 32.8–34.0 ms，p₉₅ 50.6–52.0 ms；管线 p₉₅ 51.0–52.2 ms。
  - 朴素测量均报告 ≈0 ms，差距 20–50 ms；FaceMesh p₉₅ 管线延迟 27–28 ms 通过 50 ms 交互预算，WebGazer 下界已超标。
- **精度结果**：FaceMesh 均值误差比 WebGazer 低 3.1°（sweep）和 4.6°（drift），但两者差异落在 ~4.6° 运行间变异带内，**不排名**。两者均无法达到 sub-2° 精度。
- **精度分解（Finding 2–3）**：空间散布（径向 p₉₅ ≈6.13° vs. 6.21°）两引擎无差异，但帧内抖动速度 `v_p99` 相差 1.6–3.5×；每格误差结构不同（FaceMesh 径向扩展，WebGazer 对角线模式）。
- **下游临床探针（GazeMedSeg on Kvasir-SEG）**：保持下游管线完全不变，仅替换 gaze 源；专家 EyeLink 1000 gaze 训练息肉分割器 Dice=0.679，普通摄像头 gaze Dice≈0.000。摄像头固定点落在息肉内仅 17%（EyeLink 为 90%），伪掩码 Dice 0.12–0.17 vs. 0.75–0.78。**此为硬件惩罚的上界**（ annotator 专业性与 viewing 指令同时改变）。
- **网格分辨率扩展（Appendix B）**：均值误差在 L1–L5（pitch 15.0°→2.6°）范围内基本平坦（斜率 ≈0°/degree），Hit rate 随网格变密单调下降；两引擎排序由校准方法决定而非引擎本身。
- **Ablation（Appendix C）**：One-Euro `β`  sweep 运行间变异（~4.6°）淹没参数效应；KRR 核比较显示 RBF > linear > poly2，但 poly2 因 λ 未重新调参导致欠正则化，非干净核对比。

## 相关工作脉络
- **WebGazer [7]**：基于眼 patch 像素的岭回归管线，十年后仍为标准基线；本文在延迟测量方法论上修正其隐性缺陷（报告延迟≈0 ms），精度本工作中其表现与 FaceMesh+KRR 差异在运行间变异带内。
- **TurkerGaze [13]**： crowdsourcing 眼动注意力数据集，使用 WebGazer 管线；本文的方法论改进可直接提升此类 crowd-scale 采集的延迟报告可信度。
- **GazeMedSeg [14]**：MICCAI 2024 的弱监督息肉分割方法，使用专家 EyeLink 1000（1000 Hz，≤0.5°）gaze 达到 Dice 0.778；本文沿相同管线仅替换 gaze 源，量化了摄像头级精度（4–7°）的下限，划定其仅适用于粗粒度区域提示的边界。
- **GazeSAM [11] / Gaze2Segment [5]**：均依赖实验室级红外眼动仪（如 Tobii Pro Nano、EyeLink）；本文工作位于其上游，使 browser webcam gaze 信号本身可用于下游任务而不需专用硬件。
- **MediaPipe FaceMesh [4]**：提供 per-frame 虹膜关键点，是自然特征源；本文首次将其与 KRR 及公开基准测试平台组合发布。
- **rVFC API [12]**：W3C WICG 规范，提供 per-frame `captureTime`/`presentationTime`；本文是已知首个将其用于浏览器眼球追踪推理延迟测量的工作。

## 局限性与未来方向
- **单受试者（N=1）**：任务内引擎对比在 within-subject 设计下稳健，但空间结构结果与精度排名可能受个体头姿/虹膜几何影响；多用户复现（N=5）已在准备中。
- **校准混淆**：网格分辨率实验中两引擎使用各自默认校准（FaceMesh pursuit vs. WebGazer nine-point），交叉对比反映"引擎+校准"而非纯引擎差异； matched-pursuit 对照（L6 ‡ 行）显示 WebGazer 从 7.1° 跳至 11.1°，校准方法影响显著。
- **下游探针的混杂**：摄像头 gaze 实验同时改变了硬件、annotator 专业性与 viewing 指令，观测到的 Dice 差距是硬件惩罚的**上界**，纯硬件效应需 protocol-matched 收集隔离。
- **One-Euro `β` ablation 运行间变异过大**：姿态/光照漂移（~4.6°）淹没滤波参数效应，建议后续研究使用 ≥3 次重复 per condition。
- **poly2 核比较因 λ 未重调而失效**：未做 per-kernel 交叉验证，仅揭示 ablation 设计中的必要控制变量。
- **未来方向**：多用户复现、手术视频标注的临床 pilot（Cholec80 管线 [10]）、校准延迟补偿的敏感性分析（当前固定 100 ms lag）。

## 研究启发与可借鉴点
- **延迟测量方法学可直接迁移**：任何基于浏览器媒体流（`getUserMedia` + canvas/WebGL inference）的实时系统（手势追踪、姿态估计、语音唤醒）都可能遭遇同类"零延迟假象"；rVFC 捕获时钟+FIFO 精确配对/下界配对的两档策略可作为通用诊断模板。
- **精度双维度报告范式**：区分"空间散布"（radial p₉₅ / 每格误差热力图）与"时间抖动"（`v_p99`）避免单一均值掩盖结构差异；建议在眼球追踪/任何坐标回归任务中同时报告。
- **下游弱监督探针作为精度标尺**：将 gaze 信号直接喂入固定下游管线（GazeMedSeg）并观察性能坍塌点，比纯 angular error 更能反映实际可用性边界；此"端到端 feasibility probe"策略可迁移至其他传感信号的价值评估。
- **校准方法敏感性应作为标准对照**：WebGazer 在 pursuit vs. nine-point 校准下误差从 7.1° 跳至 11.1°，提示任何跨引擎比较必须控制校准协议；建议在论文中报告"matched calibration"对照实验。
- **运行间变异带的保守使用**：以 ~4.6° between-run band 作为 refrain-from-ranking 的判定阈值，而非 noise floor 的确证；此保守用法可作为小样本 benchmark 的通用报告规范。

## 关键术语表
- **rVFC（requestVideoFrameCallback）**：浏览器 API，每解码一帧触发回调并携带 `captureTime`/`presentationTime` 元数据，本文核心延迟恢复手段。
- **capture clock**：每帧相机捕获时刻的时间戳，作为推理延迟计算的统一参考原点；本文通过 rVFC 恢复。
- **精确配对 / 下界配对**：前者通过 FIFO 队列将 gaze 样本与源帧精确绑定（适用于暴露流水线的引擎）；后者用最新帧时钟作为代理，报告值恒为真实延迟的下界。
- **One-Euro filter**：基于速度的低通滤波器，用于平滑原始 gaze 信号，参数 `β` 控制平滑强度。
- **I-VT（Ignored Velocity Threshold）**：在线固定点/扫视分类算法，以速度阈值（本文 1200 px/s ≈ 18°/s）判定 fixation。
- **Hit rate（per-cell）**： dwell-window 样本质心落在对应目标格内的比例，衡量闭环分类可用性。
- **`v_p99`（within-fixation jitter velocity）**：固定点内瞬时 gaze 速度的 p₉₉ 分位数，刻画时间精度/抖动。
- **GazeMedSeg**：MICCAI 2024 提出的弱监督医学图像分割方法，将 gaze 点卷积为 Gaussian heatmap 生成 pseudo-mask 训练 U-Net ensemble。

## 可复现要素
- **数据集**：Kvasir-SEG（息肉分割，900 train / 100 test）[3]；GazeMedSeg 数据集 [14]（expert EyeLink gaze annotations）；本文收集的 webcam gaze CSV logs 与所有 per-sample 数据随代码发布。
- **代码/权重**：开源 TypeScript SPA 参考实现 + benchmark harness + 分析套件 + clock_probe.html，随 paper 发布（论文声明 "released in source" / "accompanying source bundle"）。
- **关键超参**：FaceMesh+KRR — `λ = 10⁻³`（固定），`γ` by median-pairwise-distance（每次拟合更新）；One-Euro `β=0.007`（default）；I-VT 速度阈值 1200 px/s；GazeMedSeg downstream — σ=70 px Gaussian，2-level U-Net ensemble，15k iters，`bs=4`，single seed。
- **环境**：Chrome (arm64) on macOS Sequoia 15.6，MediaPipe FaceMesh 0.4.1633559619，WebGazer 3.4.0；14-inch MacBook Pro M4 2024，集成 1080p 摄像头 @ 1280×720。
- **复现耗时**：重跑分析 <30 s（commodity hardware）；重新采集数据约 30 min（ablation sweep）+ 额外 30 min（4-run 矩阵）。
