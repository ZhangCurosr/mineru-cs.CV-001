---
title: "Repurposing-RGB-based-Foundation-Model-for-Depth-Estimation"
source: https://arxiv.org/pdf/2608.11564v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 12:38:58"
field: "多光谱视觉感知"
keywords: ["热成像深度估计", "跨模态知识蒸馏", "层级监督", "DINOv3", "RGB-热图对齐", "MS²数据集"]
innovations: ["提出层级对齐机制同时传递局部结构与全局语义，超越单层特征蒸馏", "首次引入亮度-对比度置信度验证动态过滤低质量 RGB 监督"]
benchmarks: ["MS² (Multi-Spectral Stereo)"]
---

# 论文速读：Repurposing-RGB-based-Foundation-Model-for-Depth-Estimation

## 一句话总结
提出 **RGB-HS** 框架，通过将冻结的 RGB 基础模型（DINOv3）作为教师分支，利用层级对齐（图级结构 + 潜在级语义）与质量验证机制，将 RGB 知识迁移至热成像深度估计学生分支，在 MS² 数据集上实现优于现有 SOTA 的热图深度估计性能，同时保持极低推理复杂度。

## 研究问题与动机
1. **热成像深度估计的模态局限**：热图像缺乏 RGB 的纹理与颜色信息，物体边界模糊、空间梯度弱，直接单图/立体深度估计面临严峻挑战。
2. **现有跨模态迁移方法对基础模型表征利用不足**：GTDE 仅利用 DINOv2 做零样本泛化，忽略 RGB 模态本身；RGB-MDE 虽引入 RGB 分支做蒸馏，但未利用基础模型中间层 token 的多级语义信息。
3. **RGB 监督信号质量不均未被建模**：低光照、雨雾等条件下 RGB 特征可靠性下降，现有方法对所有 RGB 监督一视同仁，可能引入噪声对齐。

## 核心贡献（创新点）
1. **RGB-HS 层级监督框架**：用预训练 DINOv3 替换热图编码器，并引入并行 RGB 分支，通过多级对齐从教师向学生学习结构与语义——与 GTDE/RGB-MDE 仅使用顶层/单一特征的本质区别在于利用了 Transformer 全层级 token。
2. **图级相关性对齐损失（Map-level Correlation Loss）**：强制热图与 RGB 特征图在局部空间依赖上一致，捕捉边缘与深度不连续结构——区别于像素级 L2/MAE 对齐，本方法关注通道内空间结构而非逐像素值。
3. **潜在级 KL 散度 + 余弦相似度联合对齐（Latent-level Alignment）**：通过对全局最大池化后的通道向量做方向与分布双重约束，使热图学生捕捉场景级语义抽象——现有工作（如 RGB-MDE）未对 latent 分布做显式对齐。
4. **亮度-对比度置信度验证模块（Verification）**：以 RGB 图像的亮度均值与对比度标准差融合为质量指标，动态缩放对齐损失权重，过滤低质量 RGB 监督——这是首次在热图深度迁移中引入数据级质量门控。

## 方法详解
**整体架构**：教师-学生范式。冻结的 RGB 教师分支（DINOv3 ViT-B/16）与可训练的热学生分支（共享相同 encoder-decoder 架构，decoder 沿用 MSCRF）并行输入配对 RGB-热图；推理时仅保留热分支。

**特征提取**：对第 $l$ 层 Transformer 输出提取空间 patch tokens $\mathbf{T}_l \in \mathbb{R}^{N_l \times C_l}$，去掉 class/register token，reshape 为 $\mathbf{F}_l \in \mathbb{R}^{C_l \times H_l \times W_l}$。

**图级对齐损失**（Eq.2-3）：
$$\mathcal{L}_{\text{corr}} = \frac{1}{C} \sum_{c=1}^{C} \left\| \mathbf{f}_{\text{THR},c}^\top \mathbf{f}_{\text{THR},c} - \mathbf{f}_{\text{RGB},c}^\top \mathbf{f}_{\text{RGB},c} \right\|_2^2, \quad \mathcal{L}_{\text{map}} = \lambda^l \mathcal{L}_{\text{corr}}$$
计算每通道特征图与其转置的内积（即通道自相关矩阵），最小化热图与 RGB 的结构相似性差异。

**潜在级对齐损失**（Eq.4-7）：先对特征图做全局最大池化聚合为 $\bar{\mathbf{f}} \in \mathbb{R}^C$，再用余弦相似度损失与温度缩放 KL 散度联合约束：
$$\mathcal{L}_{\text{cos}}^g = 1 - \frac{\langle \bar{\mathbf{f}}_{\text{THR}}, \bar{\mathbf{f}}_{\text{RGB}} \rangle}{\|\bar{\mathbf{f}}_{\text{THR}}\|_2 \|\bar{\mathbf{f}}_{\text{RGB}}\|_2}, \quad \mathcal{L}_{\text{KL}}^g = \text{KL}\!\left(\text{Softmax}\!\left(\frac{\bar{\mathbf{f}}_{\text{RGB}}}{T}\right) \bigg\| \text{Softmax}\!\left(\frac{\bar{\mathbf{f}}_{\text{THR}}}{T}\right)\right)$$
$$\mathcal{L}_{\text{latent}} = \lambda_1^g \mathcal{L}_{\text{cos}}^g + \lambda_2^g \mathcal{L}_{\text{KL}}^g$$

**验证模块**（Eq.8-11）：对 RGB 图计算亮度 $C_{\text{BT}}$ 与对比度 $C_{\text{CT}}$，取平均得 $C_{\text{RGB}} \in [0,1]$，用其缩放热图特征：$\mathbf{F}_{\text{THR}} \leftarrow C_{\text{RGB}} \cdot \mathbf{F}_{\text{THR}}$，再代入对齐损失。

**总损失**（Eq.12）：
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{baseline}} + \frac{1}{NL}\sum_{n,l} \left(\mathcal{L}_{\text{map}}^{(n,l)} + \mathcal{L}_{\text{latent}}^{(n,l)}\right)$$
超参数：$\lambda^l = 0.5,\ \lambda_1^g = 1.0,\ \lambda_2^g = 0.1$，AdamW，lr=$1\times10^{-4}$，单卡 RTX PRO 6000。

## 实验与结果
- **数据集**：MS²（Multi-Spectral Stereo），含日间/夜间/雨天三元光照天气划分；训练 26K、验证 4K、测试 17.8K 配对样本。
- **评估指标**：AbsRel、SqRel、RMSE、RMSElog、$\delta<1.25/1.25^2/1.25^3$。
- **最强结果（Stereo）**：AbsRel **0.105**、SqRel **0.572**、RMSE **3.548**、RMSElog **0.134**、$\delta<1.25/\delta<1.25^2/\delta<1.25^3$ = **0.887 / 0.981 / 0.996**，在 MS² 全天气 Avg 上全面超越 MSCRF (Stereo)（分别提升 12.5%、45.9%、30.0%）。
- **关键对比**：RGB-HS (Mono) SqRel=0.650、RMSE=3.806，优于 MSCRF (Stereo) 的 SqRel=1.057、RMSE=5.068，说明层级监督在单目设置下即可匹敌甚至超越前作立体方案。
- **干净版本 MS² 对比（Table II）**：RGB-HS AbsRel=0.072、SqRel=0.283、RMSE=2.595，与 SOTA RGB-MDE (AbsRel=0.072) 持平，但 RMSE 更低（2.595 vs 2.677）。
- **复杂度优势**：模型 249M / FLOPs 0.23T / 推理 44.46ms，显著低于 RGB-MDE（666M / 0.72T / 698ms），甚至小于 MSCRF baseline（284M / 0.32T）。

## 相关工作脉络
1. **MSCRF [9]**：热图深度估计首篇深度学习框架，采用 Swin-L encoder-decoder；本文以其为 baseline 替换编码器并注入层级监督。
2. **GTDE [15]**：两阶段策略，利用 DINOv2 实现零样本热图深度泛化；局限在于未利用 RGB 模态本身且仅取顶层特征。
3. **RGB-MDE [16]**：置信度感知蒸馏，从 RGB-depth 分支向热分支迁移知识；但未利用 Foundation Model 多层 token，也无质量门控。
4. **ThermoStereoRT [30]**：基于知识蒸馏的热立体匹配实时方法；同样缺少对 RGB 特征质量的分层评估。
5. **DepthAnything / DepthAnythingV2 / DINOv3**：RGB 单目深度基础模型系列；本文选用 DINOv3 ViT-B/16 作为教师 backbone，验证其在跨模态迁移中的通用性。

## 局限性与未来方向
1. 验证模块仅依赖亮度-对比度简单统计，未考虑语义级可靠性（如遮挡、反光区域），可能存在误判。
2. 需要配对 RGB-热图数据进行训练，限制了在无配对数据的场景下应用。
3. 推理时完全丢弃 RGB 分支，无法在需要时动态利用高质量 RGB 额外提升性能。
4. 作者指出未来可将层级监督推广至热图语义分割、检测与跟踪等任务，但目前仅验证了深度估计。

## 研究启发与可借鉴点
1. **层级 token 对齐范式可复用于其他跨模态任务**：图级相关性 + 潜在级分布对齐的组合兼顾局部结构与全局语义，可迁移至 RGB-热图分割/检测中，替代简单的 L2 特征蒸馏。
2. **亮度-对比度验证思想的泛化**：用轻量图像质量信号动态调制跨模态蒸馏权重，避免低质量源污染学生表示；该思路可用于任何存在"主模态质量波动"的监督蒸馏场景（如低光照 RGB→红外）。
3. **大模型替换小模型但压缩复杂度的设计**：用 DINOv3 ViT-B（86M）替换 Swin-L（197M）并叠加蒸馏后总模型仅 249M，说明 foundation model 的参数效率远高于从头训练的小模型，值得在团队的热/红外任务中复现此设计模式。
4. **Clean dataset 实验的价值**：论文同时报告干净版 MS² 结果并与 RGB-MDE 对比，为后续工作提供了去除标注噪声后的公平比较基线，建议在后续工作中沿用该对比协议。

## 关键术语表
- **RGB-HS**：Repurposing RGB-based Foundation model for thermal depth estimation with Hierarchical Supervision 的缩写，本文提出的核心框架。
- **DINOv3**：Meta 提出的新一代自监督视觉基础模型（ViT 架构），本文用作 RGB 教师 backbone，在 RGB 单目深度任务上达到 SOTA。
- **Map-level Alignment**：图级对齐，通过对齐热图与 RGB 特征图的通道自相关矩阵来保持局部结构一致性。
- **Latent-level Alignment**：潜在级对齐，通过对齐全局最大池化后通道的方向（余弦）与概率分布（KL 散度）来传递语义抽象。
- **Verification Module**：验证模块，基于 RGB 图像亮度与对比度计算置信度 $C_{\text{RGB}}$，用于动态调制对齐损失权重。
- **MS² Dataset**：Multi-Spectral Stereo 数据集，包含同步采集的立体 RGB、NIR、热图及 LiDAR 深度 GT，覆盖白天/夜间/雨天多种条件。
- **Teacher-Student Paradigm**：教师-学生范式，本文指冻结 RGB DINOv3 为教师、热图编码器为学生的知识迁移机制。
- **AbsRel / SqRel**：平均绝对相对误差与平方相对误差，深度估计常用误差指标，越小越好。

## 可复现要素
- **数据集**：MS²（Multi-Spectral Stereo），公开可用（作者引用 [9] 及官方 split）。
- **代码/权重**：论文未明确声明开源；baseline MSCRF [9] 代码公开，DINOv3 ViT-B/16 预训练权重可通过 timm/huggingface 获取。
- **关键超参**：$\lambda^l=0.5$、$\lambda_1^g=1.0$、$\lambda_2^g=0.1$、AdamW lr=$1\times10^{-4}$、单卡 RTX PRO 6000、PyTorch 实现。
- **Backbone**：DINOv3 ViT-B/16（86M 参数）作为教师与 Student encoder 共享架构。
