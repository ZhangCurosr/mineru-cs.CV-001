---
title: "A-Neighborhood-Attention-Transformer-Network-for-Enhanced-3D"
source: https://arxiv.org/pdf/2608.12274v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 08:11:41"
field: "医学图像分割"
keywords: ["Left Anterior Descending Artery", "3D Medical Image Segmentation", "Neighborhood Attention", "Transformer", "Coronary Artery", "Low-contrast CT", "Parameter-efficient Fine-tuning", "LoRA"]
innovations: ["提出 NA-UNETR：将 Neighbourhood Attention 与 Dilated NA 结合，实现局部精细结构与长程血管连续性的联合建模", "首次将 LoRA 参数高效微调引入 3D 医学 Transformer 分割，冻结编码器注意力层并适配 rank-8 LoRA 适配器", "引入同方差不确定性加权的 Dice-Focal + Hausdorff 复合损失，动态平衡区域重叠与边界精度"]
benchmarks: ["LAD-SEG (institutional, n=20)", "ImageCAS (public, n=1,000)"]
---

# 论文速读：A-Neighborhood-Attention-Transformer-Network-for-Enhanced-3D-Segmentation-of-the-Left-Anterior-Descending-Artery

## 一句话总结
提出了 NA-UNETR，一种结合邻域注意力（NA）与扩张邻域注意力（DiNA）的 3D Transformer 分割框架，通过预训练 + LoRA 参数高效微调策略，在低对比度、重度类别不平衡的非增强 CT 上实现了左前降支（LAD）动脉的精准分割。

## 研究问题与动机
1. **临床需求**：胸部放疗中精确勾画 LAD 动脉对心脏剂量 sparing 至关重要，低至 10 Gy 辐射剂量即可增加冠脉钙化和缺血事件风险。
2. **任务挑战**：LAD 体积微小（平均前景体素仅 540 个，前景占比约 1.7×10⁻⁵）、软组织结构对比度差、边界模糊，且在自由呼吸非增强 CT 上受运动伪影影响严重。
3. **现有方法不足**：传统 CNN（如 nnU-Net）在 LAD 上 Dice 仅约 0.21，远低于心脏腔室分割的 0.85；现有方法难以同时兼顾薄血管的局部精细结构与长程连续性依赖。
4. **小数据瓶颈**：专家标注的 LAD 数据稀缺（仅 20 例），直接训练 Transformer 模型困难，需要有效的迁移学习策略。

## 核心贡献（创新点）
1. **NA-UNETR 架构**：首次将 Neighborhood Attention（NA）和 Dilated NA（DiNA）块引入冠状动脉 3D 分割，在 UNETR 风格骨干网中联合建模局部精细结构与长程上下文。
2. **预训练 + LoRA 高效适配**：在 1,000 例 CTA 数据集上预训练后，冻结编码器注意力层并通过 rank-8 LoRA 适配器仅微调解码器参数，以极小的参数量适应非增强 CT 领域偏移。
3. **不确定性加权复合损失**：提出 Dice-Focal 与边界感知 Hausdorff 损失的组合，通过同方差不确定性（homoscedastic uncertainty）动态平衡两项损失权重，避免手动调参。
4. **动脉中心采样与定制化预处理**：针对重度类别不平衡设计正负样本 1:1 的 patch 采样策略，并配合强度裁剪、局部对比度增强及几何/结构增广，显著提升低对比度血管的可见性。

## 方法详解

### 架构设计
- **Encoder**：4 级 U 形结构，embedding dimension = 48。每级前有重叠 tokenizer（两个 3×3×3 卷积，stride 分别为 2×2×2 和 1×1×1）。每级包含若干 NAT（Neighborhood Attention Transformer）块，每块前设残差卷积（Res-Conv）稳定梯度。NAT 块内交替使用 NA 和 DiNA，kernel size 逐级递减（7, 7, 7, 3, 3）。
- **Decoder**：对称 U 形，每级上采样后与 encoder skip 连接融合，经残差块整合后最终通过 1×1×1 卷积 + sigmoid 输出体素级概率图。
- **DiNA 机制**：在 NA 基础上引入扩张采样与可学习相对位置偏置 $b(i, j)$，使感受野随深度指数扩展，同时保持局部空间归纳偏置。

### 核心公式

**邻域注意力（NA）**：
$$\text{NA}(\mathbf{q}_i) = \sum_{j \in \mathcal{N}(i)} \alpha_{ij} \mathbf{v}_j, \quad \alpha_{ij} = \frac{\exp(\mathbf{q}_i^\top \mathbf{k}_j / \sqrt{d})}{\sum_{l \in \mathcal{N}(i)} \exp(\mathbf{q}_i^\top \mathbf{k}_l / \sqrt{d})}$$

**扩张邻域注意力（DiNA）**：
$$\text{DiNA}_{\delta,k}(\mathbf{q}_i) = \sum_{j \in \mathcal{N}_\delta(i)} \alpha_{ij}^{(\delta)} \mathbf{v}_j, \quad \alpha_{ij}^{(\delta)} = \frac{\exp(\mathbf{q}_i^\top \mathbf{k}_j / \sqrt{d} + b(i,j))}{\sum_{l \in \mathcal{N}_\delta(i)} \exp(\mathbf{q}_i^\top \mathbf{k}_l / \sqrt{d} + b(i,l))}$$

**复合损失（同方差不确定性加权）**：
$$\mathcal{L}_{\text{total}} = \frac{1}{2\sigma_1^2} \mathcal{L}_{\text{Dice-Focal}} + \frac{1}{2\sigma_2^2} \tilde{\mathcal{L}}_{\text{Hausdorff}} + \log \sigma_1 + \log \sigma_2$$
其中 $\sigma_1^2, \sigma_2^2$ 为可学习方差参数，对 Hausdorff 损失添加了零均值高斯噪声 $\epsilon \sim \mathcal{N}(0, \sigma_2^2)$ 以促进探索、防止过拟合早期边界估计。

### 训练策略
- **预训练阶段**：在 ImageCAS（1,000 例 CTA）上训练 100 epochs，固定随机种子进行 5-fold 评估。
- **微调阶段**：冻结编码器注意力层，仅更新解码器参数及 rank-8 LoRA 适配器（$W \rightarrow W + AB$，$A \in \mathbb{R}^{d \times 8}, B \in \mathbb{R}^{8 \times d}$），在 LAD-SEG（20 例）上训练 200 epochs，5-fold 交叉验证。
- **Focal Loss 参数**：$\alpha=0.8$（类别权重背景 0.1/前景 0.9），$\gamma=2$；$\lambda_1=\lambda_2=1$。

## 实验与结果

### 数据集
- **LAD-SEG**：机构私有数据集，20 例自由呼吸非增强 CT（分辨率 1.17×1.17×3.0 mm）， physician 标注 LAD 轮廓。
- **ImageCAS**：公开数据集，1,000 例高分辨率 CTA（0.29–0.43 mm in-plane，0.25–0.45 mm slice spacing），双专家标注。

### 基线模型
- **CNN 类**：U-Net, U-Net++, nnU-Net, MedNeXt
- **Transformer 类**：UNETR, Swin UNETR, Swin UNETR-V2, nnFormer

### 主要结果（LAD-SEG，5-fold 平均）

| 方法 | DSC (%) | HD95 (mm) | ASD (mm) |
|------|---------|-----------|----------|
| nnU-Net | 42.54 ± 2.90 | 39.68 ± 5.92 | 10.37 ± 1.56 |
| Swin UNETR | 44.78 ± 6.08 | 40.65 ± 6.96 | 10.44 ± 1.26 |
| **NA-UNETR（ours）** | **45.64 ± 4.86** | **38.16 ± 4.37** | **10.01 ± 1.39** |
| NA-UNETR (no-dil.) | 44.09 ± 3.89 | 40.10 ± 9.30 | 9.55 ± 3.24 |

- 相对 nnU-Net：**Dice 提升 3.10 pp**；相对 Swin UNETR：**HD95 降低 2.96 mm**。
- 边界精度在所有模型中最优。Mann-Whitney U 检验因样本量小（n=20）未达统计显著（p > 0.05）。

### 主要结果（ImageCAS，4-fold）

| 方法 | DSC (%) | HD95 (mm) | ASD (mm) |
|------|---------|-----------|----------|
| UNet++ | 78.57 ± 0.51 | 9.46 ± 0.84 | 1.15 ± 0.07 |
| Swin UNETR-V2 | 78.03 ± 0.48 | 9.13 ± 0.73 | 1.12 ± 0.06 |
| **NA-UNETR（ours）** | **79.49 ± 0.25** | **8.89 ± 0.30** | **1.02 ± 0.03** |

- 相对 UNet++ **Dice 提升 0.92 pp**，相对 Swin UNETR **HD95 降低约 0.57 mm**，ASD 降低约 9%。
- Mann-Whitney U 检验显示 **p < 0.05**（统计显著）。

### 计算效率
- 参数量 19.6M，FLOPs 314.1B，推理耗时 1.33s/体积，峰值显存 4.17 GB。
- 性能优于 Swin UNETR（5.49 GB VRAM），参数量和训练时间（18.77s/epoch）与 UNETR 相当。

### 消融关键结论
- **DiNA 有效**：去掉扩张模块 Dice 下降 ~1.5 pp。
- **Residual Block**：移除后 Dice 从 45.64% 降至 43.01%。
- **Variable Kernel**：固定 3×3×3 kernel Dice 降低 ~2.5 pp。
- **LoRA Rank**：r=8 最优（Dice 45.64%），r≥16 出现边际下降。
- **预训练必要性**：仅在 LAD-SEG 上训练 Dice 仅 36.39%（vs 45.64%）。
- **动态 Loss 权重**：固定权重 Dice 44.08%（vs 45.64%）。

## 相关工作脉络
1. **nnU-Net / U-Net 系列**：作为医学图像分割基准，但在 LAD 等细薄低对比结构上 Dice 仅 ~0.21，无法捕捉长程血管连续性。
2. **UNETR / Swin UNETR**：首次将 Transformer 引入 3D 医学分割，但 Swin 的 shifted-window 机制在处理跨切片血管连续性时仍受限，本文通过 NA+DiNA 渐进扩展感受野加以改进。
3. **邻域注意力（Hassani et al., 2023）**：原始 NA 工作专注于自然图像，本文首次将其适配到 3D 医学体积的细长血管分割，并引入 DiNA 扩展感受野。
4. **LoRA 微调策略**：源于 NLP 大模型参数高效适配，本文将其引入医学 Transformer 微调，冻结编码器注意力层仅适配 LoRA + 解码器，显著降低过拟合风险。
5. **冠状动脉分割现有研究**：既往工作（van den Bogaard 等）多采用 atlas-based 方法（Dice 0.09–0.27）或 CNN 方案，本文首次系统评估 transformer-based 模型在非增强 CT LAD 分割上的潜力。

## 局限性与未来方向
1. **边界精度受限于成像模态**：LAD-SEG 上 HD95 和 ASD 仍较高（38.16 mm / 10.01 mm），薄血管（仅数个体素宽）的边界预测困难，且受非增强 CT 固有分辨率限制。
2. **数据集规模小**：LAD-SEG 仅 20 例，统计检验效力不足（Mann-Whitney U 未达显著），需要更大规模多中心验证。
3. **跨模态域偏移**：预训练使用 CTA（高对比）→ 微调使用非增强 CT，虽经 LoRA 适配但仍存在残余域偏移，缺乏显式的域自适应机制。
4. **临床部署尚未就绪**：作者明确声明"目前不适合临床部署"，需要前瞻性 workflow 研究评估时间节省、编辑负担和用户接受度。
5. **未来方向**：引入显式域对齐策略缩小 CTA–非增强 CT 差距；扩大标注数据规模；探索多模态融合提升可见性。

## 研究启发与可借鉴点
1. **NA/DiNA 块的设计思路**：对细长管状结构（血管、神经、气管）分割任务具有迁移价值，可通过调节 kernel size 序列和扩张率控制感受野增长曲线，平衡局部细节与全局连续性。
2. **LoRA + 编码器冻结的微调范式**：在 3D 医学 Transformer 微调中极具参考价值——冻结编码器注意力层可大幅减少过拟合风险，rank=8 为实用起点，后续可调参探索。
3. **同方差不确定性加权复合损失**：将区域重叠损失与边界感知损失联合优化的思路可推广至其他细小结构分割任务，避免手动调权。
4. **动脉中心 patch 采样策略**：正负样本 1:1 的 patch 级别平衡方法，结合 Gamma 对比度增强，对处理极端类别不平衡的医学图像有直接借鉴意义。
5. **预训练 → 微调的两阶段范式**：在大规模公开 Coronary CTA 数据集上预训练，再在稀缺目标域上高效微调，是解决小样本 3D 医学分割的有效策略。

## 关键术语表
- **LAD（Left Anterior Descending Artery）**：左前降支动脉，心脏最重要冠状动脉之一，负责供应左心室前壁血液，是胸部放疗心脏剂量 sparing 的关键子结构。
- **Neighborhood Attention（NA）**：一种计算高效的注意力机制，将 token 交互限制在局部 $k \times k \times k$ 窗口内，相比全局自注意力更适合保留局部几何一致性。
- **Dilated NA（DiNA）**：在 NA 基础上引入扩张采样和可学习相对位置偏置，使感受野随网络深度指数扩展，同时保持局部性约束。
- **Homoscedastic Uncertainty Weighting**：通过引入可学习方差参数 $\sigma_1^2, \sigma_2^2$ 自动动态平衡多损失项权重，替代手动调参。
- **LoRA（Low-Rank Adaptation）**：参数高效微调方法，通过低秩矩阵分解（$W + AB$）替换原权重矩阵，仅更新少量参数即可适配下游任务。
- **clDice（Centerline Dice）**：基于预测与 GT skeleton 相交区域的拓扑一致性度量，对细长管状结构的连续性评估优于传统 Dice。
- **HD95（95th-percentile Hausdorff Distance）**：预测边界与 GT 边界间距离的第 95 百分位数，衡量最坏情况下的边界偏差。

## 可复现要素
- **代码开源**：已公开于 https://github.com/rafiibnsultan/NA_UNETR
- **预训练权重**：论文声明将在 ImageCAS 预训练后公开发布。
- **数据集**：LAD-SEG 为机构私有数据集（20 例）；ImageCAS 为公开数据集（1,000 例，https://github.com/ZengAries/ImageCAS）。
- **关键超参**：embedding dim=48；NAT blocks per stage=(3,4,6,18,5)；kernel sizes=(7,7,7,3,3)；LoRA rank r=8；AdamW lr=10⁻⁴，weight decay=10⁻⁵；Focal α=0.8，γ=2；patch size=(96,96,96)；类别权重背景 0.1/前景 0.9。
- **硬件**：8× NVIDIA A100 GPU（40 GB），单卡训练。
- **实现框架**：PyTorch 2.5.1，Python 3.9.21。
