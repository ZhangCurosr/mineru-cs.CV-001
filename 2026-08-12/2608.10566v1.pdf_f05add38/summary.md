---
title: "Iterative Erasure Count Is Not an Afine-Invariant Concept Dimension"
source: https://arxiv.org/pdf/2608.10566v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:05:24"
field: "表征可解释性与概念擦除"
keywords: ["iterative erasure", "affine equivariance", "concept dimension", "linear probe", "representation interpretability", "INLP", "LEACE"]
innovations: ["证明迭代擦除计数在固定充分维度下可因可逆变换发生精确整数变化", "建立完整转移度量擦除轨迹的仿射等变性定理并以精确协方差为推论", "在冻结视觉特征上以新鲜攻击者审计复现过程依赖性"]
benchmarks: ["100DOH", "TouchMoment", "synthetic Gaussian", "synthetic circular target"]
---

# 论文速读：Iterative Erasure Count Is Not an Afine-Invariant Concept Dimension

## 一句话总结
本文证明了"迭代擦除计数"（iterative erasure count）并非一个仿射不变的概念维度估计量——在相同人口统计学充分维度和最小防护秩下，可逆线性重参数化可使计数发生精确的整数变化；只有当正定度量被协变地传输且探针规则满足等变性时，完整擦除轨迹才是仿射等变的。

## 研究问题与动机
- 线性探针（linear probes）只能回答"标签信息是否可访问"，却无法回答"信息如何被组织/分布"，后者是表征可解释性研究的核心诉求。
- 学界广泛使用迭代零空间投影（INLP 风格）：拟合线性分类器 → 移除其方向 → 重拟合 → 统计停止迭代次数，并将其解读为"概念维度"或"分布物理变量数"。
- 现有做法混淆了**模型定义的人口量**（生成维度、充分线性维度、最小防护秩）与**过程定义量**（停止计数、累积编辑秩），未意识到后者的值同时被特征度量、探针损失、正则化、协方差估计和停止规则决定。
- 先验工作仅给出连续层面的非等变性观察（如 gauge-freedom），未提供精确的整数计数变化反例与完整轨迹等变性定理。

## 核心贡献（创新点）
1. **人口层面整数计数非识别定理**：构建已知秩一充分概念的 Gaussian 混合样本，证明在固定充分线性维度与最小防护秩均为 1 的条件下，Euclidean 迭代计数可从 1 跳变到 2（Theorem 1）；对双输出全 QR 移除，累积编辑秩可从 r 跳变到环境维度 2r（Theorem 2）。
2. **完整转移度量轨迹的仿射等变性定理**（Proposition 2）：只要正定系数度量 G 按协变律传输、探针等变、正则化随 A 传输、非子空间平局打破等变，则任意迭代 k 都有 $Z T_k^Z = (X T_k) A$，精确协方差度量仅为一个特例推论。
3. **有限圆形 Adam/QR 复现桥接**：复现动机论文的 Adam/MSE/全 QR 顺序更新协议，在 n=4000 下 identity 混合 20/20 次停于 K=1，而任意非零剪切 a∈{.5,.75,1,1.25,2} 均 ≥2 次，揭示实际优化器下同样的过程依赖性。
4. **冻结视觉特征的压力测试**：对 V-JEPA2 冻结特征施加预声明的各向异性/正交映射，展示即便重标准化后、使用独立新鲜攻击者，轨迹仍显著变化（100DOH rank-10 均值 AUROC 从 .599→.791，Δ+.192 [.110,.275]）；未收缩经验协方差干预则与等变性推论一致（所有映射轨迹相同，最大预测差异 2.11×10⁻⁶）。

## 方法详解
- **五种"维度"量澄清**（Table 2）：
  - *Generating dimension*：数据生成规则使用的潜变量个数。
  - *Sufficient linear dimension*：最小 r，使存在 B∈R^{d×r} 满足 Y ⊥ X | XB。
  - *Cross-covariance rank*：rank Cov(X, φ(Y))，仅捕获一阶矩。
  - *Iterative erasure count*：在声明停止规则触发前的非坍缩顺序探针更新数。
  - *Cumulative edit rank*：rank(I − T_k)，当 T_k 为幂等投影时等于移除子空间维数；对非投影更新仅表编辑算子秩。
  - *Linear guarding rank*：满足声明防护准则的最小 rank(I − T)。

- **Algorithm 1（累积度量-QR 擦除）**：
  1. 从完整表示固定 G ≻ 0。
  2. 每步拟合系数块 B_k = Probe(XT_{k−1}, Y)。
  3. 将 B_k 关于 U_{k−1} 残差化并做 G-正交化 QR 得到新块 Q_k。
  4. 若 Q_k 为空或停止规则触发则停；否则 U_k = [U_{k−1}, Q_k]，T_k = I − U_k U_k^⊤ G。
  5. 分别记录迭代次数与 rank(I − T_k)。

- **Proposition 2 等变性证明要点**：
  - 若 U_k^Z = A^{−1} U_k，则 (U_k^Z)^⊤ G_Z U_k^Z = I。
  - 由 T_k A = A T_k^Z 得 Z T_k^Z = (X T_k) A，归纳可得所有步scores、累积编辑秩、基于 score 的停止计数在两种参数化下一致。
  - 精确协方差作为特例：G_X = Σ_X ⇒ G_Z = A^⊤ Σ_X A = Σ_Z。
  - OAS/Ledoit–Wolf 收缩或特征值底截断通常破坏该等变性，因其不满足 f(A^⊤ Σ A) = A^⊤ f(Σ) A。

- **Population 构造**：
  - H = (S, N) ∼ N(0, I_d)，Y = 1[S ≥ 0]，X = H A。
  - T_A = A^{−1}(I − ee^⊤)A 秩为 1 且 X T_A ⊥ Y，故最小防护秩 = 1。
  - 对 X_a = (S + aN, N)：ridge 系数 ∝ (1+λ, −a)^⊤；a=0 时与交叉协方差同向，一次移除即独立；a≠0 时第一移除后保留方向 r_λ，其残差协方差 g_λ ≠ 0，导致第二次移除必要，但充分维度和最小防护秩仍为 1。

- **有限圆形 QR 桥接**：
  - θ ∼ Unif[−π, π]，S = (sin θ, cos θ)，N ∼ N(0, .5I_2)，X_a = (S + aN, N)。
  - 按 [13] 规格：Adam lr=10^{−3}、weight decay=10^{−4}、100 epoch、80/20 固定划分；保留条件 R² ≥ .1 且 circular MAE ≤ 80°；20 seeds，上限 8 更新；分别报告 rank(I − T_k) 与 rank(T_k)。

- **视觉压力测试**：
  - 冻结 V-JEPA2 layer 21 / DINOv2 layer 12，LBFGS logistic 探针 + LinearSVC，C=.01 不重选。
  - 密集映射族 A_κ = QD_κQ^⊤（随机符号 Hadamard 基 + log 间距特征值 κ^{−1/2} ~ κ^{1/2}），5 个固定映射/条件；20 个 Haar 正交映射作健全性检验。
  - 重标准化消融：每次映射后重新估计坐标均值/方差再擦除，证实效应非坐标缩放所致。
  - 新鲜攻击者审计：将 1600 个样本划分为 A/B/C 角色（训练集内不重叠），A 学习擦除轨迹与标准法，B 在每个 κ/rank 独立调 C，C 单次评估；25 对差值全正。

## 实验与结果
- **数据集**：100DOH（手-物接触图像，平衡审计子集 1600/200/200）、TouchMoment（第一人称视频接触时刻，11564/618/2588）、合成 Gaussian（d=2, d=8）、合成圆形目标（d=4）。
- **基线对比**：INLP、MP、LEACE、SAL、OAS/LW 收缩协方差、随机子空间、PCA 子空间、标签置换子空间、Haar 正交映射。
- **核心定量结果**：
  - Theorem 1：a=0 时迭代计数=1，a≠0 时=2，充分维度与防护秩恒为 1；对 λ∈{0, .1, 1, 10} 成立。
  - Theorem 2（双输出 r=2）：a=0 累积编辑秩=2，a≠0 时=4（环境维度）；充分维度与防护秩恒为 2。
  - 连续目标 8 维控制：κ=1 时 Euclidean 计数=2，κ∈{3,10,100,1000} 时=8；精确协方差度量下始终=2。
  - 有限圆形 n=4000：a=0 全部 20 次 K=1；a∈{.5,.75,1,1.25,2} 全部 ≥2；a=1 均值 K=5.75（范围 3–8），2/20 被 8 上限右删；post-first-edit 均值 R²=.195，circular MAE=59.5°。
  - 100DOH V-JEPA2 rank-10 + 重标准化：κ=1 均值 AUROC=.599，κ=10 均值=.791，Δ+.192 [.110,.275]；无重标准化 κ=10/100/1000 分别为 .802/.853/.840。
  - 未收缩经验协方差：所有 κ 下 rank-1=.587、rank-10=.518，最大预测差异 2.11×10⁻⁶；20 Haar 正交映射最大差异 4.27×10⁻⁶。
  - 新鲜攻击者审计（100DOH rank-10）：identity 均值 .665，κ=10 均值 .776，Δ+.111，25/25 对全正；rank-1 与 rank-2 亦全正（均值 +.033、+.060）。
  - 合成 rank-1 MP/SAL 跨拟合：n_A=100 时独立 AUROC=.881，n_A=8000 时=.572，而人口 oracle 恒为 .499，说明高维估计误差即可制造大量"残存可访问性"。
- **最强结果**：Theorem 1 在人口层面给出精确整数计数 1 vs 2 的分离；有限圆形桥接在 n=4000 达到 20/20 确定性分离；视觉压力测试在冻结特征上复现过程依赖性（κ=10 密度映射 ΔAUROC +.192）。

## 相关工作脉络
- **INLP [24]**：迭代零空间投影奠基者；本文不否定其线性防护有效性，但指出停止计数不能等同于 intrinsic protected-attribute dimension。
- **MP / SAL / LEACE [11, 29, 6]**：一次性一阶矩或仿射线性防护，显式失真几何；本文承认其价值，但强调精确协方差只是众多等价度量之一，非"语义正确"度量，并给出完整转移轨迹定理与其 rank-deficient 边界。
- **Gauge freedom [7]**：证明 Euclidean 相似性在保函数可逆变换下可变；本文在此基础上给出更紧的离散化结果——整数计数精确变化 + 完整轨迹等变性定理。
- **Video direction study [13]**：动机来源；本文复现其 Adam/MSE/全 QR 协议并在控制分布上复现 count 变化，将其解读为 procedure stress test 而非 contact dimension 估计。
- **Amnesic probing [10] / MNestic [27]**：使用移除方向数衡量干预/对照规模；本文提示这些数目同样是过程相对量，需声明度量与估计量。
- **Concept geometry 不可识别性 [19, 26]**：表示因子及其几何在无归纳偏置下仅可识别至可逆线性变换；本文将其应用于离散计数场景。

## 局限性与未来方向
- 定理与构造仅覆盖线性表示与线性干预，未延伸至非线性、token 级、空间、时序或因果概念组织。
- 视觉案例仅围绕接触 predicate 与单个 V-JEPA2 检查点；DINOv2 为图像编码器，非时序视频模型；100DOH 含对象配置捷径，TouchMoment 含接近/动作阶段混杂。
- 有限圆形实验使用控制分布而非动机论文的视频激活，未复现 steering 行为；mini-batch 大小按公开规范声明 128 并审计 64/256，非精确隐式复现。
- 密集映射族仅为 5 个固定 anisotropic + 20 个正交/对角族的存在性压力测试，非所有可逆变换的分布；"orientation-fixed concordance" AUROC 低于 .5 仅记录方向翻转，翻转会人为制造"超随机"分数。
- 历史 official-test 配置已暴露 1865 条，无 untouched holdout 剩余，所有视觉结论限于过程依赖性证据而非新 benchmark claim。
- 未来方向：将等变性框架推广至非线性擦除、tokenwise/spatiotemporal 干预；建立过程声明模板（度量、探针损失、正则化、协方差估计、停止规则、测试使用策略）成为标配；在更多表示与概念上检验转移度量轨迹的实际区分力。

## 研究启发与可借鉴点
- **过程声明标准化**：任何报告迭代擦除计数的研究必须明确声明度量、探针损失、正则化、协方差估计器、停止规则与测试使用策略，并分离 generating/sufficient/guarding 维度——这是本文提出的规范性建议，可直接作为后续工作的 checklist。
- **转移度量等变性作为验证工具**：若某擦除轨迹声称反映"概念维度"，则在同一表示上用协变传输的 G 重跑应得相同计数；否则结果过程相关。可将其作为表征分析的前置 sanity check。
- **Fresh attacker + role-split 审计设计**：将官方训练集划分为 A（估计干预）、B（训练新鲜攻击者）、C（单次评估）三组且互不相交，能消除 reused C 伪装的稳定性，同时避免使用官方验证/测试——这一跨拟合设计可迁移到任何"擦除后可访问性"评估。
- **未收缩经验协方差的等变性控制**：用无 shrinkage、无 flooring 的样本协方差作干预时，Trajectory 对所有可逆变换保持等变，可作为理论预期的"锚点对照"；任何偏离表明引入了 estimator-dependent 几何。
- **定量分离五种"维度"**：生成维度、充分线性维度、交叉协方差秩、迭代计数、累积编辑秩、防护秩——六种量在理论上可严格区分，实践中不应混用一个名称报告；本文的 Table 2 可直接复用为分析方法学论文的结构化框架。

## 关键术语表
**Iterative erasure count（迭代擦除计数）**：在声明停止规则触发前成功拟合并移除的探针方向总数，是过程相关的停止时间而非概念固有属性。

**Cumulative edit rank（累积编辑秩）**：第 k 步后 rank(I − T_k)，度量累计编辑算子的秩；当且仅当每次更新添加一个独立方向时才等于独立移除维数。

**Sufficient linear dimension（充分线性维度）**：最小 r，使得存在线性子空间 BX 满足 Y ⊥ X | BX，描述概念在表示中线性可充分捕获所需的最小维数。

**Linear guarding rank（线性防护秩）**：满足声明防护准则（如攻击者风险不低于 R_0）的最小 rank(I − T)，依赖攻击者类、损失与基线风险。

**Affine equivariance of trajectory（轨迹仿射等变性）**：当正定度量按协变律 G_Z = A^⊤ G_X A 传输、探针与正则化随之等变时，完整擦除序列在 Z = XA 与原坐标 X 下给出对应相同的 scores 与计数。

**Exact covariance as corollary（精确协方差推论）**：取 G_X = Σ_X 时 Proposition 2 的直接推论；Σ_Z = A^⊤ Σ_X A 自动成立，但 Covariance 并非唯一合法的语义度量。

**Orientation-fixed concordance**：在选定半区固定 score 方向、审计半区禁止翻转后计算的 AUROC；<.5 记录方向翻转而非与随机等价，易被事后翻转伪装为"高于随机"。

**Cross-fit calibration（跨拟合校准）**：将数据划分为估计/训练/评估三组（A/B/C）且互不相交，用以区分有限样本估计误差与真实多重人口路由。

## 可复现要素
- **数据集**：100DOH（公开，含释放的 split 边界）、TouchMoment（公开，来自 HOI4D/TACO）、合成 Gaussian/圆形数据。
- **代码/权重**：补充材料含 annotation-safe manifests、提取配置、所有分析脚本、光谱与折叠级/聚合结果、单命令缓存分析入口；授权媒体、特征缓存与大预测数组不重新分发。冻结表示：facebook/vjepa2-vitl-fpc64-256、DINOv2-Base。
- **关键超参**：Adam lr=10^{−3}、weight decay=10^{−4}、100 epoch、batch 128（审计 64/256）；LBFGS logistic / LinearSVC；C ∈ {.01 为主，跨拟合调 {10^{−3}, 10^{−2}, 10^{−1}, 1}}；特征值底 floor 10^{−4} λ_max；QR 绝对容差 10^{−8}，map rank 绝对 SVD 容差 10^{−6}；20 seeds/cell，上限 8 更新；bootstrap 1000 次组重采样。
