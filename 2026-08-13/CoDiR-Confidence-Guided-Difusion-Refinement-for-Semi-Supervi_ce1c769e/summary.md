---
title: "CoDiR-Confidence-Guided-Difusion-Refinement-for-Semi-Supervi"
source: https://arxiv.org/pdf/2608.11807v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 10:38:04"
field: "半监督医学图像分割"
keywords: ["Semi-supervised learning", "Histopathology segmentation", "Diffusion models", "Pseudo-label refinement", "Mean Teacher", "Confidence gating"]
innovations: ["置信度门控扩散精炼模块：仅对教师低置信度区域进行部分加噪条件去噪，避免全图修正破坏可靠结构", "双源置信度加权策略：融合教师与扩散模型置信度生成像素级监督权重，抑制不可靠伪标签噪声", "在 GlaS/CRAG 上以 10%/20% 标注数据超越已有最强方法，扩散精炼模块单独贡献 +6.36% mDice"]
benchmarks: ["GlaS", "CRAG"]
---

# 论文速读：CoDiR-Confidence-Guided-Diffusion-Refinement-for-Semi-Supervised-Histopathology-Segmentation

## 一句话总结
本文提出 **CoDiR**（Confidence-Guided Diffusion Refinement），一种结合冻结病理基础模型（UNI）与扩散模型伪标签精炼的半监督结肠腺体分割框架；通过仅对教师模型低置信度区域进行扩散去噪修正，有效缓解伪标签确认偏差，在 GlaS 和 CRAG 数据集上以 10%/20% 标注数据达到 88.09%/89.83% 和 89.19%/90.29% mDice，优于已有最强方法。

## 研究问题与动机
- **半监督病理分割的核心瓶颈**：像素级标注成本高昂，而现有伪标签在半监督框架下常包含结构性/边界错误，持续训练会放大误差（确认偏差）。
- **既有方法仅间接改善伪标签质量**：如作者前期 UniSemAlign 通过特征/语义对齐约束伪标签，但并未直接修正掩码中不可靠区域。
- **扩散模型的结构先验未被充分用于半监督伪标签精炼**：已有工作（如 DifRect）对全图进行扩散修正，易破坏高置信度可靠结构；缺乏"选择性修正"机制。
- **病理基础模型（如 UNI）的表征优势已在多任务验证，但在半监督伪标签精炼环节的潜力尚未被系统挖掘**。

## 核心贡献（创新点）
1. **提出 CoDiR 半监督分割框架**：将冻结 UNI ViT-L/16 编码器、DeepLabV3 解码器与 Mean Teacher 范式结合；与 UniSemAlign 的本质区别在于 CoDiR 直接在掩码空间修正低置信度区域，而非仅通过特征对齐间接提升伪标签。
2. **设计置信度门控扩散精炼模块**：仅在教师预测低置信度像素（通过自适应阈值 τ 识别）处进行部分加噪 + K 步条件去噪；相比 SDEdit/DifRect 的全图编辑，该设计避免破坏可靠结构、抑制幻觉。
3. **提出置信度融合与加权策略**：精炼区域使用扩散模型置信度、高置信区域保留教师置信度，二者相乘得到像素级监督权重，使可靠像素贡献更强；区别于仅依赖单一置信度来源的 prior。
4. **在 GlaS 与 CRAG 基准上验证有效性**：10% 标注下 CRAG mDice +1.26%、mJaccard +2.94% 超越上一最强方法；ablation 显示扩散精炼模块单独贡献 +6.36% mDice。

## 方法详解
- **编码器**：冻结的 UNI ViT-L/16（d_img=1024），丢弃 CLS token，patch tokens 重排为 2D 特征网格。
- **解码器**：DeepLabV3-style，ASPP 聚合多尺度上下文并上采样至 H×W  logits。
- **Mean Teacher 机制**：Student f_θ 通过梯度下降更新，Teacher f_ᾱθ 维护 EMA（ᾱθ ← αᾱθ + (1-α)θ，α=0.99），Teacher 仅用于生成伪标签。
- **扩散精炼流程**：
  1. Teacher 对无标签图像 I_u 输出软伪掩码 ȳ = σ(f_ᾱθ(A_w(I_u)))，置信度 c_teacher = max(ȳ, 1-ȳ)。
  2. 按自适应阈值 τ（FreeMatch 风格 EMA 更新，训练区间 [0.60, 0.75]，推理固定 τ=0.75）生成二值精炼掩码 m（c_teacher < τ 的像素）。
  3. 对伪掩码施加部分噪声（α_n=0.2）得到 ỹ_αn，再经 K 步 DDPM 反向去噪（条件为 I_u）得到 y_diff。
  4. 融合：y_r = m⊙y_diff + (1-m)⊙ȳ。
- **置信度加权**：c_diff^eff = m⊙max(y_diff,1-y_diff) + (1-m)⊙c_teacher；像素权重 w = c_teacher·c_diff^eff。
- **损失函数**：
  - 监督损失 L_sup = λ₁L_CE + λ₂L_Dice + λ₃L_clDice + λ₄L_Boundary（λ₁=λ₂=1.0, λ₃=λ₄=0.5）。
  - 无监督精炼损失 L_unsup^refine = E[w·(L_BCE(ŷ_u^w, y_r) + L_Dice(ŷ_u^w, y_r))]。
  - 一致性损失 L_unsup^cons = L_BCE(ŷ_u^s, ŷ_u^w) + L_Dice(ŷ_u^s, ŷ_u^w)。
  - 总无监督损失 L_unsup = λ_u L_unsup^refine + λ_c L_unsup^cons（λ_u=λ_c=0.25）。
  - 总损失 L = 0.5(L_sup + L_unsup)。

## 实验与结果
- **数据集**：GlaS（165 张）、CRAG（213 张），分别采用 10% 和 20% 随机标注。
- **实现**：单卡 NVIDIA RTX PRO 4000 Blackwell（24GB），256×256 输入，AdamW (1e-4)，batch=16，100 轮，10 轮 warmup；扩散 T=100、α_n=0.2。
- **主要结果（mDice/mJaccard）**：
  - **GlaS 10%**：CoDiR 88.09/79.41 vs. UniSemAlign 88.15/78.82（mDice 略低 0.06，mJaccard +0.59）。
  - **CRAG 10%**：CoDiR **89.83/82.46** vs. UniSemAlign 88.57/79.52（+1.26/+2.94）。
  - **GlaS 20%**：CoDiR **89.19/81.09** vs. UniSemAlign 88.58/79.50（+0.61/+1.59）。
  - **CRAG 20%**：CoDiR **90.29/82.93** vs. UniSemAlign 89.40/80.88（+0.89/+2.05）。
  - 在 8 项基准指标中胜 7 项。
- **Ablation 关键数字**：
  -  backbone：UNI 显著优于 ResNet101/200、MedCLIP、CONCH。
  - 模块增量：Mean Teacher 基线 80.95 → +扩散精炼 87.31（+6.36）→ +置信度融合 87.50 → +置信度加权 88.09。
  - τ_max=0.75 最优；α_n=0.20 最优；EMA decay=0.99 最优。

## 相关工作脉络
1. **Mean Teacher / FixMatch / UAMT / CPS**：经典半监督分割基线，依赖伪标签+一致性正则；本文与其定位差异在于引入扩散先验直接修正掩码结构。
2. **UniSemAlign（作者前作, CVPRW'26）**：使用 UNI 编码器+文本/原型语义对齐；本文与其差异是从"特征对齐间接约束伪标签"升级为"掩码空间选择性扩散精炼"。
3. **DifRect（MICCAI'24）**：基于潜在扩散的标签修正；本文与其差异是仅对低置信度区域进行部分加噪精炼，避免全图修正带来的结构破坏。
4. **SDEdit（arXiv'21）**：图像编辑范式；本文借鉴其部分加噪思想但将其适配到伪掩码精炼并引入置信度门控。
5. **CorrMatch / CSDS / DuSSS**：近期半监督分割方法（利用相关性匹配、解耦表征等）；本文定位是在这些方法与 UniSemAlign 基础上，通过扩散结构先验进一步弥补伪标签边界/拓扑误差。
6. **病理基础模型 UNI / CONCH / MedCLIP**：预训练视觉编码器；本文选择 UNI 因其病理特异性表征在 ablation 中表现最优。

## 局限性与未来方向
- 扩散精炼器仅在少量标注数据（10%/20%）上训练，跨器官、跨染色方案的泛化能力未验证。
- 仅在结肠腺体数据集（GlaS、CRAG）上评估，缺乏在其他病理分割任务上的通用性证明。
- 扩散模型引入额外计算开销（K 步去噪），推理/训练效率有待优化。
- 未来方向：更数据高效的精炼策略、结构感知边界优化、跨域/跨染色泛化能力提升。

## 研究启发与可借鉴点
1. **"选择性精炼"范式可迁移**：将扩散模型仅用于低置信度区域修正的思路，可迁移至其他医学图像半监督分割任务（如肝脏、肺结节），降低幻觉风险。
2. **置信度双源融合策略**：教师置信度与扩散置信度相乘得到像素权重的设计，为半监督伪标签质量评估提供了可复用的加权机制。
3. **部分加噪（partial noising）的超参敏感性分析**：α_n=0.2 为最优的经验对同类任务有参考价值；但需根据数据分布重新校准。
4. **冻结病理基础模型 + 轻量解码器 + Mean Teacher 的组合**：在标注极度稀缺场景下可作为强基线架构复用。
5. **与团队方向结合机会**：若团队关注跨中心/跨染色泛化，可将 CoDiR 的精炼模块与 domain adaptation 技术结合，探索"扩散精炼 + 域对齐"的联合框架。

## 关键术语表
- **Mean Teacher**：半监督学习范式，教师网络权重为学生权重的 EMA，用于生成稳定的伪标签。
- **Confidence-Guided Refinement**：基于置信度阈值仅对低置信度区域进行扩散去噪修正的策略。
- **Partial Noising（SDEdit 风格）**：对伪掩码施加中等强度噪声（α_n=0.2）后条件去噪，保留大部分原始结构。
- **UNI ViT-L/16**：大规模预训练的病理视觉基础模型（1024 维 patch tokens），本文作为冻结编码器使用。
- **clDice**：保留管状/腺体拓扑结构的一致性 Dice 损失。
- **Boundary Loss**：针对边界像素的加权损失，提升分割轮廓精度。
- **FreeMatch 自适应阈值**：基于批次均值的 EMA 更新置信度阈值 τ，训练区间 [0.60, 0.75]。
- **Confirmation Bias（确认偏差）**：半监督学习中错误伪标签被反复强化导致性能下降的现象。

## 可复现要素
- **数据集**：GlaS（公开）、CRAG（公开）；半监督设置采用 10% 和 20% 随机标注 split。
- **代码开源**：是，地址 https://github.com/vongla345/codir
- **权重**：冻结的 UNI 编码器权重；扩散精炼器与 Student/Teacher 权重训练后保存（论文未提供预训练权重链接，需自行训练）。
- **关键超参**：α=0.99（EMA）、λ₁=λ₂=1.0、λ₃=λ₄=0.5、λ_u=λ_c=0.25、T=100、α_n=0.2、τ_min=0.60、τ_max=0.75（训练）、τ=0.75（推理）、batch=16、lr=1e-4、100 epochs、10 epochs warmup。
