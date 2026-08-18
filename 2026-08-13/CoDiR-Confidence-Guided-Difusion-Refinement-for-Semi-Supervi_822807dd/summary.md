---
title: "CoDiR-Confidence-Guided-Difusion-Refinement-for-Semi-Supervi"
source: https://arxiv.org/pdf/2608.11807v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 10:36:37"
field: "医学图像半监督分割"
keywords: ["semi-supervised learning", "histopathology segmentation", "diffusion model", "pseudo-label refinement", "Mean Teacher", "confi-dence gating", "pathology foundation model"]
innovations: ["置信度门控扩散精炼模块，仅修正低置信度伪标签区域", "结合冻结 UNI 编码器与 Mean Teacher 的半监督分割框架", "扩散置信度 × 教师置信度的联合加权无监督损失"]
benchmarks: ["GlaS", "CRAG"]
---

# 论文速读：CoDiR-Confidence-Guided-Difusion-Refinement-for-Semi-Supervi

## 一句话总结
本文提出 CoDiR，一种结合冻结 pathology foundation 编码器与置信度门控扩散精炼模块的半监督组织病理学分割框架，选择性地修正 Mean Teacher 伪标签中的低置信度区域，显著减少确认偏差，在 GlaS 和 CRAG 基准上以 10%/20% 标注数据达到或超越 SOTA 性能。

## 研究问题与动机
- 组织病理学图像像素级标注成本高昂，需专家病理学家手动标注形态变异大、染色差异明显的腺体结构，难以规模化获取标注数据。
- 现有半监督方法依赖 Mean Teacher 生成的伪标签直接监督学生网络，但在模糊/重叠腺体区域伪标签不可靠，错误结构会反复强化，导致确认偏差（confirmation bias）并恶化分割质量。
- 前作 UniSemAlign 仅通过语义对齐间接改善伪标签质量，并未在掩码层面直接修正错误区域，边界和结构误差仍难消除。
- 扩散模型具备建模掩码结构先验的能力，但直接全局编辑（如 SDEdit）会破坏可靠预测；需要一种"仅在不确定区域精炼"的机制。

## 核心贡献（创新点）
- 提出 CoDiR 半监督框架，将冻结 UNI ViT-L/16 编码器 + DeepLabV3 解码器 + Mean Teacher 范式统一，直接在掩码层面修正不可靠区域，而此前 UniSemAlign 仅通过特征对齐间接改善伪标签。
- 设计置信度门控扩散精炼模块，基于自适应阈值只精炼低置信度像素，保留高置信度教师预测；与 DifRect 和 SDEdit 式全局编辑相比，避免了确认偏差放大并提升边界质量。
- 构造扩散置信度 × 教师置信度的联合权重机制，使可靠像素对无监督损失贡献更大，抑制噪声伪标签对训练的不利影响。
- 在 GlaS 和 CRAG 上系统验证：10%/20% 标注设置下均超过/追平最强的 UniSemAlign，且消融表明扩散精炼模块贡献最大（+6.36% mDice）。

## 方法详解
- **视觉编码器**：冻结预训练的 UNI ViT-L/16（patch token 数 $N = \frac{H}{16} \times \frac{W}{16}$，$d_{img} = 1024$），丢弃 CLS token，将剩余 patch tokens reshape 为 2D 特征网格。
- **解码器**：DeepLabV3-style 解码器，通过 ASPP 聚合多尺度上下文并上采样，输出 per-pixel logit map $\mathbf{S} \in \mathbb{R}^{B \times 1 \times H \times W}$。
- **Mean Teacher**：学生 $f_\theta$ 通过梯度下降更新；教师 $f_{\bar{\theta}}$ 使用 EMA 更新 $\bar{\theta} \leftarrow \alpha \bar{\theta} + (1-\alpha)\theta$，$\alpha = 0.99$，不直接优化，仅生成伪标签。
- **扩散精炼器**：小型 UNet $\epsilon_\phi$ 仅在标注集 $\mathcal{D}_l$ 上训练，学习条件分布 $p(y|I)$，以 DDPM 形式建模掩码结构先验；采用部分噪声策略（partial noising）将教师伪掩码加噪到 $\alpha_n = 0.2$，再进行 $K$ 步去噪。
- **置信度计算与门控**：教师软伪标签 $\hat{y} = \sigma(f_{\bar{\theta}}(\mathcal{A}_w(I_u)))$，置信度 $c_{teacher} = \max(\hat{y}, 1-\hat{y})$；通过 FreeMatch 风格自适应阈值 $\tau$（EMA 动态更新，钳制在 $[0.60, 0.75]$）得到二值精炼掩码 $m$。
- **伪标签融合**：$y_r = m \odot y_{diff} + (1-m) \odot \hat{y}$，仅低置信度区域替换为扩散输出，高置信度区域保留教师预测。
- **置信度加权**：$w = c_{teacher} \cdot c_{diffusion}^{eff}$，其中精炼区取扩散置信度、非精炼区保留教师置信度，用于无监督损失加权。
- **监督损失**：$\mathcal{L}_{sup} = \lambda_1 \mathcal{L}_{CE} + \lambda_2 \mathcal{L}_{Dice} + \lambda_3 \mathcal{L}_{clDice} + \lambda_4 \mathcal{L}_{Boundary}$，$\lambda_1=\lambda_2=1.0$，$\lambda_3=\lambda_4=0.5$，其中 clDice 保持腺体拓扑、Boundary loss 改善边界。
- **无监督损失**：$\mathcal{L}_{unsup} = \lambda_u \mathbb{E}[w \cdot (\mathcal{L}_{BCE} + \mathcal{L}_{Dice})] + \lambda_c \mathcal{L}_{cons}$，$\lambda_u=\lambda_c=0.25$，一致性项约束弱/强增强预测不变。

## 实验与结果
- **数据集**：GlaS（165 张）与 CRAG（213 张）结肠腺体分割基准；半监督设置随机选取 10% 和 20% 训练样本标注。
- **实现细节**：单卡 NVIDIA RTX PRO 4000 Blackwell（24GB），训练图缩至 256×256，推理按非重叠 patch 拼接；AdamW（lr=$1\times10^{-4}$，batch=16），100 epoch，EMA $\alpha=0.99$，扩散 100 timesteps，$\alpha_n=0.2$；暖启 10 epoch。
- **GlaS（10%）**：CoDiR mDice=88.09%，mJaccard=79.41%，略低于 UniSemAlign（88.15%/78.82%）但 Jaccard 更强。
- **CRAG（10%）**：CoDiR mDice=89.83%（+1.26% vs UniSemAlign 88.57%），mJaccard=82.46%（+2.94% vs 79.52%），八项指标中七项居首。
- **GlaS（20%）**：mDice=89.19%（+0.61%），mJaccard=81.09%（+1.59%），整体超越 UniSemAlign。
- **CRAG（20%）**：mDice=90.29%（+0.89%），mJaccard=82.93%（+2.05%）。
- **最强提升**：CRAG 10% 标注下 mDice +1.26%、mJaccard +2.94%，表明在腺体更大、形状更不规则的数据集上精炼收益更高。
- **消融**：扩散精炼模块贡献最大（+6.36% mDice），置信度融合 +0.19%、置信度加权 +0.59%；不同 backbone 中 UNI 最优（ResNet101/200、MedCLIP、CONCH 均大幅落后）；超参 $\tau_{max}$、$\alpha_n$、EMA decay 均有稳健区间。

## 相关工作脉络
- **FixMatch / UAMT / CPS**：经典半监督分割基线，依赖伪标签置信度阈值或一致性正则；CoDiR 在相似范式上引入扩散精炼，弥补其伪标签不可靠导致的确认偏差。
- **DifRect**：基于 Latent Diffusion 的全局伪标签纠正方法；CoDiR 与之差异在于仅对低置信度区域局部精炼且使用浅层 UNet，避免全局编辑破坏可靠结构。
- **SDEdit**：图像编辑框架，通过加噪+去噪实现全局编辑；CoDiR 借鉴部分噪声策略但引入置信度门控，使精炼具有选择性。
- **UniSemAlign（前作）**：同一团队先前工作，通过文本原型和视觉 encoder 的语义对齐间接提升伪标签质量；CoDiR 转为直接在掩码层面精确修正不可靠区域，属于方法升级。
- **CorrMatch / CSDS / DuSSS**：近年利用特征相关、解耦表示或视觉-语言对齐的 SSL 方法；CoDiR 定位在于结合 foundation encoder + diffusion 结构先验，而非特征对齐。
- **clDice / Boundary loss**：拓扑保持与边界损失在先前的 gland 分割工作中已有应用；CoDiR 将其整合进监督/无监督双阶段训练，强化形态与边界质量。

## 局限性与未来方向
- 扩散精炼器仅在少量标注集上训练，跨器官、跨染色域泛化能力尚未充分验证。
- 实验仅在结肠腺体（GlaS、CRAG）两个数据集上评估，覆盖病理性状有限。
- 当前精炼器为低容量 mini-UNet，边界精度仍有提升空间。
- 论文提出未来将探索更高效、结构感知的精炼策略，以在多样本域中更好地修正不确定区域。

## 研究启发与可借鉴点
- **置信度门控 + 扩散精炼的设计范式**可迁移到其他医学分割任务（如器官、病灶），尤其适用于伪标签在边界/粘连区域误差较大的场景。
- **冻结 pathology foundation encoder（UNI）+ 轻量解码器**的"基础模型 + 下游适配"策略值得在本团队中复现验证，以较低成本获取强语义表征。
- **部分噪声（partial noising）+ 自适应阈值**的组合提供了一种可操作的伪标签质量控制机制，可拓展至其他需要区域选择修正的任务。
- **扩散置信度 × 教师置信度的联合权重**思想可用于任何伪标签质量评估，作为无监督损失的自适应采样信号。
- 未来可与本团队的 foundation model 微调方向结合：探索更多病理 foundation 模型（如 CONCH、MedCLIP）在该框架下的泛化表现。

## 关键术语表
- **Semi-supervised learning（半监督学习）**：同时利用少量标注数据和大量未标注数据进行模型训练的学习范式。
- **Histopathology segmentation（组织病理学分割）**：对病理图像中的腺体、细胞或组织结构进行像素级分割的任务。
- **Diffusion model（扩散模型）**：通过逐步去噪过程建模数据分布的生成模型，可用于结构先验学习。
- **Pseudo-label refinement（伪标签精炼）**：对模型初始预测的掩码进行二次修正，以抑制伪标签中的错误和噪声。
- **Mean Teacher**：通过指数移动平均（EMA）维护教师网络、以教师伪标签监督学生网络的 SSL 框架。
- **Confirmation bias（确认偏差）**：错误伪标签在训练中反复被强化，导致模型性能持续下降的现象。
- **Partial noising（部分噪声策略）**：将图像或掩码加噪至中间水平后再去噪，以平衡保留原有结构与引入新结构的能力。
- **clDice（connected Dice）**：保留管状/腺体拓扑连通性的损失函数，有助于维持分割结构的合理性。

## 可复现要素
- **数据集**：GlaS（公开）、CRAG（公开）；标注比例分别为 10%、20% 随机采样。
- **代码**：已开源，GitHub: https://github.com/vongla345/codir
- **权重**：使用冻结的 UNI ViT-L/16 预训练权重；扩散精炼器小型 UNet 需自行训练。
- **关键超参**：EMA decay $\alpha=0.99$，扩散 timesteps $T=100$，部分噪声水平 $\alpha_n=0.2$，置信度阈值范围 $\tau \in [0.60, 0.75]$，损失权重 $\lambda_1=\lambda_2=1.0$，$\lambda_3=\lambda_4=0.25$，训练 100 epoch，warmup 10 epoch。
- **硬件**：单卡 NVIDIA RTX PRO 4000 Blackwell（24GB）即可运行。
