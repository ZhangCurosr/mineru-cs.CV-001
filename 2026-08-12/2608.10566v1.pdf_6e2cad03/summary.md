---
title: "Iterative Erasure Count Is Not an Afine-Invariant Concept Dimension"
source: https://arxiv.org/pdf/2608.10566v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:05:12"
field: "representation interpretability / probing"
keywords: ["iterative null-space projection", "afine invariance", "linear probing", "concept dimension", "metric transport", "INLP", "LEACE", "gauge freedom"]
innovations: ["population count jumps from 1 to 2 under shear with fixed sufficient and guarding rank", "complete metric-QR trajectory is afine-equivariant under congruence-transported positive-definite metric", "empirical dense-map sensitivity in frozen V-JEPA2 survives restandardization while orthogonal maps are algebraically invariant"]
benchmarks: ["100DOH hand-object contact", "TouchMoment egocentric touch onset", "synthetic Gaussian cross-fit calibration"]
---

# 论文速读：Iterative Erasure Count Is Not an Afine-Invariant Concept Dimension

## 一句话总结
本文证明：通过反复拟合线性探针并投影掉其方向来统计"概念维度"的迭代擦除法（iterative null-space projection），其输出——停止次数与累积编辑秩——在保信息的可逆变换（如剪切变换）下会改变，因此不是概念的内禀维度；真正内禀的是生成维数、充分线性维数、最小监护秩等群体量，而非过程量。

## 研究问题与动机
- **核心问题**：用迭代投影掉分类器方向再重训、计数直到分类失败的方法（INLP 及其变体）被广泛解读为"某个概念的分布式编码维数"，这一解读是否成立？
- **INLP 原文**虽谨慎将其框架为 guardedness，但后续应用将"数十到数百个正交方向"直接等同为"分布式物理变量"；近期视频世界模型论文用正交探针训练次数作为特征维数并解释为 distributed 变量，是本论文的直接针对对象 [13]。
- **混淆两个量纲**：一个标签可由 1 个潜标量生成，却可从多个观测坐标预测（因混叠矩阵关联这些坐标）；反之，一个 rank-one 经验交叉协方差被一个 afine 编辑消去，不能证明找到了唯一的 population-sufficient 方向。
- **未声明测量程序**：停止次数与累积秩依赖特征度量、探针损失、正则化、协方差估计与停止规则，属于过程量，却被当作不依赖坐标系的语义量报告。

## 核心贡献（创新点）
1. **精确整数计数不可识别定理**（Theorem 1/2）：在保持 generating/sufficient/guarding rank 完全相同的情况下，证明 Euclidean 迭代停止次数可以从 1 跳到 2（标量 ridge），累积编辑秩从 2 跳到 4（full-QR 两输出），而 population 概念秩不变——此前文献未给出这种精确整数分离。
2. **完整度量传输下的 afine 等价性定理**（Proposition 2）：对任意正定系数度量 G，按同张量律做 congruence 传输后，整个累积 metric-QR 轨迹在 afine 变换下等价；精确协方差是其自然推论，但不是唯一的"语义度量"。
3. **Moore–Penrose OLS 非等价性的边界刻画**：明确 rank-deficient 后 MP 选最小范数解，通常不按 A 传输，因而不在 Proposition 2 覆盖范围内——揭示了常见实现中的隐蔽假设。
4. **视频分析协议的可复现应力测试**：匹配公开视频分析的 Adam/MSE/全 QR 规范与持有验证阈值，在 n=4000 大样本下身份混合 K=1（20/20 轮），剪切 a∈{.5,.75,1,1.25,2} 每轮均接受 ≥2 次更新，把抽象定理桥接到实际优化器输出。
5. **冻结 V-JEPA2/DINOv2 的密集 afine 应力测试**：在 100DOH 手物接触任务上，经过 log-spaced 密集变换 + 坐标维均值标准差重标准化，rank-10 映射轨迹的 AUROC 与身份显著不同（κ=10 时 +.192 [.110,.275]），而 20 个 Haar 正交映射与未收缩协方差轨迹与身份代数一致——区分各向异性效应与坐标重命名效应。

## 方法详解
- **五个应估计量**（Table 2）：①生成维数（data-generating rule 中潜变量个数）；②充分线性维数（min r 使 Y ⊥ X | XB）；③交叉协方差秩（rank Cov(X, φ(Y))，仅一阶矩）；④迭代擦除停止次数（procedure-defined 停止时间）；⑤线性监护秩（满足指定 guard 条件的 min rank(I−T)）。前三个是 population/latent 量，后两个是 procedure 量。
- **度量传输定义**：设 G_X ≻ 0 为系数度量（等价于对偶空间二次型），Z = XA，则保 w^⊤Gw 需 G_Z = A^⊤G_XA（对偶空间 congruence）；若对 primal 行特征作用则取逆 congruence。Algorithm 1 给出完整 metric-QR 循环：每步 Fit → 残差化 → G-正交化 → 追加列空间 → 累计编辑算子 T_k = I − U_kU_k^⊤G_X；记录迭代计数与 rank(I−T_k) 分离。
- **Proposition 1**：对 d≥2、非零 w，存在可逆 A 使原坐标 Euclid 投影与变换后投影再映射回来不等，尽管 xw=(xA)(A^{−1}w) 对所有 x 成立。取 2×2 shear block [1 0; a 1] 代入即显式得非对角项。
- **Proposition 2（afine 等价）**：若 probe 满足 B_k^Z=A^{−1}B_k、正则化按 A 传输、非子空间 tie-breaking 也等价，则 ∀k：ZT_k^Z=(XT_k)A，从而评分、累积编辑秩、所有基于评分的停止次数在两套参数化下完全一致。证明用归纳：edited features 按 A 对应 → 下次系数块按 A^{−1} 传输 → metric Gram–Schmidt 保持基关系。
- **定理 1（population ridge 剪切计数不可识别）**：S,N iid N(0,1)，X_a=(S+aN,N)，L=sign(S)。EU 正交累积基 + 标量 ridge λ≥0。a=0 时 count=1；∀a≠0 时 count=2，尽管 sufficient dim=1、guard rank=1。证明：Cov(X_a,L)=c(1,0)^⊤，ridge 系数 ∝(1+λ,−a)^⊤；仅当 a=0 时平行于 (1,0)^⊤，第一次 EU 删后剩余协方差非零，第二次非零。
- **定理 2（full-QR 多元计数不可识别）**：S,N iid N(0,I_r)，Y=S，X_a=(S+aN,N)∈R^{2r}。a=0 时 count=r、cum edit rank=r；∀a≠0 时 count=2、cum edit rank=2r（即环境维）。proof：第一个系数块 B=[I, −aI]^⊤ 列空间维 r；其正交补 spanned by C=[aI,I]^⊤，保留坐标 X_aC=aS+(1+a²)N，cross-cov 与 Y 为 aI_r（rank r），第二次 MP-OLS 满列空间把环境维耗尽。
- **有限 circular-QR 桥**：θ~Unif[−π,π]，S=(sin θ,cos θ)，N~N(0,0.5I_2)，X_a=(S+aN,N)。两输出 MSE + Adam lr=1e−3、wd=1e−4、100 epoch、80/20 split。更新接受条件：held-out R²≥.1 且 circular MAE≤80°。n=4000 时身份 K=1（20/20）；剪切 a∈{.5,.75,1,1.25,2} 全部 ≥2。a=1 时 mean K=5.75（range 3–8），2/20 被 censored 于 cap=8；post-first-edit R²=.195，circular MAE=59.5°。
- **视觉案例**：100DOH（hand contact 3/4 vs 0）与 TouchMoment（approach vs onset），V-JEPA2 layer 21/18、DINOv2 layer 12/11。5 组 log-spaced 密集 signed-Hadamard 变换 κ∈{1,2,3,5,10,100,1000}；rank-ten 在重标准化后 AUROC 从 .599(κ=1) 到 .791(κ=10)；未收缩 empirical 协方差下所有 map 轨迹一致（Δ≤2.11e−6），Haar 正交同样一致（Δ≤4.27e−6）。

## 实验与结果
- **人口构造**：
  - 定理 1：λ∈{0,.1,1,10}，a=0→count=1；∀a≠0→count=2（固定 sufficient dim=1, guard=1）。
  - 定理 2：r=2（两输出）→ a=0 cum edit rank=2；a≠0 cum edit rank=4（环境维）。
  - 连续目标 8 维控制：sufficient dim=guard=2；Euclid κ=1→2，κ∈{3,10,100,1000}→8；exact 协方差度量恒为 2。
- **有限 circular 桥**（n=4000, 20 seeds）：
  - a=0 所有 20 轮 K=1；a=.25 同；a∈{.5,.75,1,1.25,2} 每轮均 ≥2。
  - a=1 mean K=5.75（range 3–8, 2/20 censored），post-first R²=.195，circular MAE=59.5°。
  - 小样本 n=1000 时只有 15/20 a=1 接受第 2 次，吻合公式 (11) 的连续衰减。
  - batch size 64/128/256 分离稳定：a=0 的 P(K≥2)=0，a=1 均 =1。
- **冻结特征密集映射**（100DOH val, V-JEPA2 L21）：
  - 重标准化后 mean rank-10 AUROC：κ=1→.599, κ=2→.666, κ=3→.673, κ=5→.745, κ=10→.791, κ=100→.853, κ=1000→.840。
  - rank-10 差值 [identity, mapped] 95% bootstrap：κ=2 [.008,.134]，κ=10 [.110,.275]。
  - 无重标准化：κ=10/100/1000 分别为 .802/.841/.853——效应非单纯坐标缩放。
  - 未收缩 empirical 协方差：κ=1/1000 均为 .518，Δ=0；Haar 正交 20 个 map Δ≤4.27e−6。
- **新攻击者审计**（A/B/C 三分离, 5 partitions × 5 maps）：
  - rank-0 AUROC .822 相同；rank-10 身份均 .665，κ=10 密集均 .776，配对差均值 +.111，25/25 为正。
  - rank-1 差均值 +.033，rank-2 +.060，随秩单调上升。
- **交叉拟合校准**（合成 1024 维, Y=1[X₁≥0]，已知 population guard rank=1）：
  - MP/SAL 在 n_A=100/1000/2000/4000/8000 留下 fresh logistic AUROC .881/.754/.683/.622/.572；population oracle 始终 .499。
  - LEACE-OAS 在 100DOH-max：n_A=100→.876，n_A=6000→.647；TouchMoment：.927→.626。五分区 std<.014。
- **强攻击者保留过程依赖**（Figure 2, 正则化每秩重选）：100DOH Euclid/OAS 经 1 次编辑 .827/.586，经 10 次 .598/.494；TouchMoment 经 1 次 .924/.811，经 10 次 .778/.447——曲线非单调、强度量依赖，不能被压成单一语义计数。

## 相关工作脉络
- **INLP [24]**：迭代零空间投影的原始框架；本文确认其 Linear guardedness 仍可报告，但指出"iteration count 不是 intrinsic protected-attribute dimension"。
- **Mean Projection / SAL / LEACE [11, 29, 6]**：单次协方差感知 one-shot guard，显式 distortion 几何；本文的定位是：不提出新擦除器，而是给 full-QR 多变量情形的精确整数计数分离 + 完整 transported-metric 轨迹定理。
- **Amnesic probing [10]**：用累计子空间作反事实干预；本文指其 removed-direction count 只刻画 procedure 属性。
- **Roeder et al. [26] 线性可识别性**：证明 broad discriminative 族可识别到 invertible linear transform；本文在此基础上给出更窄更硬的离散结果——具体整数的 count 跳变，而 sufficient/guarding rank 不变。
- **Cain [7] gauge freedom**：函数保真的 invertible map 下 Euclid 相似会变；本文补充：一般 metric 依赖本身不蕴含"discrete stopping count 也变"，本文给出精确 counterexample。
- **视频物理可解释论文 [13]**：用正交探针次数估计 feature dimensionality；本文明确其输出的 coordinate-relative 性质，不否认 layer-wise access / tuning geometry / held-out steering 作为 native-basis 证据的价值。
- **RLACE / minimax erasure [25]**：不同目标（attacker-relative 优化）；本文认为其强化了"必须声明 estimand"的需求。
- **TCAV [14]**：CAB 把 human concept 操作化为 feature-direction；本文重申 invertible map 可保持所有线性 score 但改变 Euclid 角度。

## 局限性与未来方向
- **线性范围**：定理与潜构造仅限线性表示与线性干预；非线性、token-wise、空间、时序、因果概念组织不在覆盖范围。
- **单一模型/任务**：视觉案例集中于 V-JEPA2 单一 checkpoint、100DOH 接触标签；DINOv2 仅为 image-only 对照，非时序 video 模型。
- **标签噪声**：100DOH 中 object-box 几乎 label-deterministic；TouchMoment 的 onset/pre-onset 区分含 hand closure、approach、action phase 等混杂——二者都不是纯"物理触"的干净目标。
- **样本依赖**：cross-fit 校准显示 high-dimensional 估计误差即可留下大量 fresh-attacker access（n_A=100 AUROC .881），与"多 population 通路"无法区分。
- **正交 map 与固定密集 map 族**：只覆盖了 five fixed signed-Hadamard SPD maps 和 secondary diagonal family；不能推广到所有 invertible map 分布。
- **验证复用**：层/正则化/rank 选择、C 选择均 reuse validation；无 untouched local holdout 可用；prospective protocol 已冻结但尚未在未见数据集上验证。
- **Moore–Penrose 非等价**：rank-deficient 后 MP 选最小范数解不被 Prop 2 覆盖——常见实现的实际行为未纳入等价定理。

## 研究启发与可借鉴点
1. **三维分离报告范式**：今后报告 probe 类结果时，建议把 generating dimension、sufficient dimension、cross-cov rank、guarding rank 与 iteration count、cumulative edit rank 严格分开，并把 metric/probe loss/正则化/协方差估计/停止规则都列为 estimand 的一部分。
2. **Transported-metric 等价性作为 sanity check**：任何新的 erasure 流程应在同一正定度量下做 congruence 传输，若 Euclid 轨迹与 metric 轨迹不一致则应报告差异来源，而不是把 Euclid 默认当作 canonical。
3. **正交 map 作为零假设对照**：对 Haar 正交族应给出"代数一致"的零假设轨迹，用以区别于真正各向异性效应——本文的 20 Haar map + 未收缩 empirical cov 对比设计值得移植到其它表征审计。
4. **交叉拟合 A/B/C 三分离**：用 disjoint source-video 角色分离 estimation、fit、evaluation，可切断"residual access 被误读为多通路"的伪信号；建议在表征可解释性论文中例行使用。
5. **样本量—估计误差桥接**：本文的 synthetic cross-fit 曲线（n_A=100→.881, n_A=8000→.572 vs oracle .499）直观说明 high-dim estimation error 即可造成"似多通路"的假象，今后任何"多方向/多通路"结论前都应先校准该下限。

## 关键术语表
- **Generating dimension**：数据生成规则中使用潜变量编码 Y 的个数，是 population 层面的概念维数。
- **Sufficient linear dimension**：满足 Y ⊥ X | XB 的最小 r，代表线性充分子空间的 population 维数。
- **Cross-covariance rank**：rank Cov(X, φ(Y))，仅捕获一阶矩，对 φ 的选择敏感，标量情形 ≤1。
- **Linear guarding rank**：在指定攻击者类下满足 guardedness 条件所需的最小 rank(I−T)，依赖攻击者类与风险基线。
- **Cumulative edit rank**：第 k 步编辑后 rank(I−T_k)，仅在每次更新加 1 个独立方向时才等于独立删去方向数。
- **Iterative erasure count**：按声明停止规则接受的更新数，是 procedure-relative 估计量而非语义维数。
- **Metric transport law**：G_Z = A^⊤G_XA 的 congruence 传输律，是 Prop 2 等价性的前提。
- **Orientation-fixed concordance**：AUROC 在 selection half 上固定得分朝向、audit half 不复 flip 后计算；低于 .5 表示 audit half 上方向反转，并非等价于 chance。

## 可复现要素
- **数据集**：100DOH（source [28]，train/val/test 1600/200/200，已公开 split boundary）与 TouchMoment（source [18,17,20]，11564/618/2588，视频不跨 split）；100DOH-max 17206/2126/1958；均为公开数据集。
- **模型权重**：V-JEPA2 facebook/vjepa2-vitl-fpc64-256（冻结）与 DINOv2-Base（冻结）；权重公开。
- **代码/数据**：补充 artifact 含 annotation-safe manifest、提取配置、全部分析脚本、频谱、fold 级与聚合结果、单命令缓存入口；许可媒体/特征缓存/大型预测数组不重新分发。
- **关键超参**：Adam lr=1e−3、wd=1e−4、100 epoch；80/20 split；minibatch 128（验证 64/256）；accepted 条件 R²≥.1 且 circular MAE≤80°；LBFGS logistic 探针；C=0.01 一次选择不再重选；特征维度 1024/768；绝对容忍 1e−8（QR）/1e−6（SVD）；20 seeds/cell；bootstrap 1000 group replicates。
