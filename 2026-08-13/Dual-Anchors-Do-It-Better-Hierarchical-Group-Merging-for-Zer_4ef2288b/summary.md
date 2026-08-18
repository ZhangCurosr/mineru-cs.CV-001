---
title: "Dual-Anchors-Do-It-Better-Hierarchical-Group-Merging-for-Zer"
source: https://arxiv.org/pdf/2608.11933v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 12:24:37"
field: "零样本异常检测"
keywords: ["零样本异常检测", "双锚点框架", "层次化分组", "视觉-语言模型", "跨模态对齐", "工业缺陷检测"]
innovations: ["提出双锚点范式，通过层次化分组合并构建图像语义锚点与文本锚点并列对齐", "设计组门控Token精化器利用图像锚点自适应精化全局表示", "构建动态状态提示实现图像条件化的自适应文本生成"]
benchmarks: ["MVTec-AD", "VisA", "MPDD", "BTAD", "DTD-Synthetic", "ISIC", "Kvasir"]
---

# 论文速读：Dual-Anchors-Do-It-Better: Hierarchical Group Merging for Zero-Shot Anomaly Detection

## 一句话总结
本文提出了一种**双锚点框架（Dual-Anchor Framework）**，通过层次化分组合并机制从图像特征中提取正常/异常语义锚点，与文本锚点联合构建动态状态提示，显著缓解了现有零样本异常检测方法对文本提示的过度依赖，在14个工业与医疗基准上实现了最优性能。

---

## 研究问题与动机

- **文本锚点过度依赖问题**：现有CLIP-based ZSAD方法将语义锚点主要置于文本侧，导致性能高度敏感于prompt设计，缺乏稳定的视觉 grounding。
- **单锚点决策边界偏差**：单一文本锚点会导致在未见领域中的决策边界无法完整覆盖异常样本，产生偏向性。
- **图像编码器层级结构未充分利用**：现有方法常将不同层提取的特征视为独立，忽视了图像编码器自局部到全局的层级语义整合潜力。
- **缺乏图像中心的结构化视觉推理**：尽管部分工作引入视觉上下文提示，但仍未建立以视觉结构为核心的语义推理机制。

---

## 核心贡献（创新点）

1. **双锚点范式（Dual-Anchor Framework）**：首次将图像语义锚点与文本锚点并列，构建平衡的跨模态对齐机制，与仅依赖文本锚点的方法形成本质区别。
2. **层次化分组合并（Hierarchical Group Merging）**：提出自顶向下的特征聚合策略，通过可学习组token逐步合并局部到全局视觉特征，生成具有层级异常表征的图像锚点。
3. **组门控Token精化器（Group-Gated Token Refiner, GGTR）**：利用正/异常组token作为视觉锚点对[CLS] token进行门控加权精化，增强全局判别表示。
4. **动态状态提示（Dynamic State Prompt）**：将图像依赖的组token融入文本提示，构建自适应状态感知提示，降低人工prompt工程成本。

---

## 方法详解

### 整体架构
框架包含三个核心模块：
1. **Hierarchical Group Merging**：生成正常/异常组token作为图像锚点
2. **Group-Gated Token Refiner**：利用图像锚点精化全局表示
3. **Dynamic State Prompt**：构建图像条件化的自适应文本提示

### 特征提取
- 采用**DINOv3（ViT-L/16）**作为视觉骨干网络
- 从第6、12、18、24层提取1024维patch token：$F_i^{(l)} \in \mathbb{R}^{M \times C}$
- 文本分支使用**CLIP（ViT-L/14@336px）**获取768维embedding

### 层次化分组合并（HGM）

**四个阶段：**

1. **初始化**：通过cross-attention继承上层分组线索
   $$F_i'^{(l)} = \text{CrossAttn}(F_i^{(l)}, G'^{(h)})$$

2. **分配**：基于Gumbel Softmax实现可微离散分组
   $$\Pi = \text{Softmax}\left(\frac{F_i'^{(l)} W_Q (G^{(h)} W_K)^\top + \Gamma}{\tau}\right)$$
   $$G'^{(h)} = G^{(h)} + \text{Linear}(\Pi(G^{(h)} W_V))$$

3. **合并**：采用二分软匹配策略进行无参数token缩减
   - 计算源集合与目标集合间的余弦相似度矩阵 $S$
   - 为每个源token选择最相似的目标token配对
   - 按相似度分数保留top-r配对，通过平均融合：
     $$\tilde{g}_u = \frac{1}{2}(g_u + g_{v'(u)})$$
   - 未配对token保留，拼接得到下一层输入

4. **更新**（仅最终Merge Block）：将组token重新分配回patch token

**最终输出：**
$$\{G_N, G_A\}, F_{assign} = \sum_{h=1}^{H} \text{MergeBlock}(F^{(h)}, G'^{(h)})$$
其中 $G_N$、$G_A$ 分别为正常和异常组token（图像锚点）

### 组门控Token精化器（GGTR）

利用图像锚点对[CLS] token进行自适应精化：
$$w_n = \frac{t_{cls}^\top G_N}{\|t_{cls}\| \|G_N\|}, \quad w_a = \frac{t_{cls}^\top G_A}{\|t_{cls}\| \|G_A\|}$$
$$t_{cls}^{\text{refined}} = t_{cls} + \sigma(\text{Linear}(\text{Norm}(\text{Concat}(w_n, w_a))))$$

### 动态状态提示

构建图像条件化的文本提示：
$$p_n = [V_1][V_2]\cdots[V_E][W_1][G_N][\text{class}]$$
$$p_a = [V_1'][V_2']\cdots[V_E'][W_1'][G_A][\text{class}]$$
其中 $[V_i]$、$[W_i]$ 为可学习上下文token

### 异常检测与损失函数

**像素级异常图：**
$$\mathcal{A}^{(L+1)} = \text{Softmax}(\text{Up}([\frac{1}{L}\sum_{l=1}^{L} F_i^{(l)}, F_{assign}] \mathcal{T}^\top)))$$
$$\mathcal{M} = \frac{1}{L+1} \sum_{l=1}^{L+1} \frac{\mathcal{A}_a^{(l)} + (1 - \mathcal{A}_n^{(l)})}{2}$$

**图像级异常分数：**
$$p_{cls} = \frac{t_{cls}^{\text{refined}} \cdot T}{\|t_{cls}^{\text{refined}}\|_2 \|T\|_2}$$

**损失函数：**
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{seg}} + \mathcal{L}_{\text{ortho}} + \mathcal{L}_{\text{group}}$$

其中：
- $\mathcal{L}_{\text{cls}} = \text{BCE}(p_{cls}, y)$
- $\mathcal{L}_{\text{seg}} = \text{Dice}(\mathcal{A}^{(L+1)}, S) + \text{Focal}(\mathcal{A}^{(L+1)}, S)$
- $\mathcal{L}_{\text{ortho}} = |\langle G_N, G_A \rangle|^2$（正交正则化）
- $\mathcal{L}_{\text{group}} = 1 - \frac{1}{2}(\cos(G_N, Z_N) + \cos(G_A, Z_A))$（组语义对齐）

---

## 实验与结果

### 数据集
- **工业域（8个）**：MVTec-AD、VisA、MPDD、BTAD、RSDD、KSDD2、DAGM、DTD-Synthetic
- **医疗域（6个）**：ISIC、CVC-ColonDB、CVC-ClinicDB、TN3K、Endo、Kvasir
- **辅助训练**：MVTec-AD（跨域实验）、VisA（MVTec-AD评估时）

### 评估指标
- 图像级：AUROC、F1-score、AP
- 像素级：AUROC、AP

### 主要结果

**图像级性能（Table 1）：**
| 数据集 | 本文方法 | 次优方法 | 提升幅度 |
|--------|----------|----------|----------|
| MVTec-AD | (92.7, 94.0, 97.1) | Bayes-PFL (92.3, 93.1, 96.7) | +0.4 AUROC |
| VisA | (88.3, 85.0, 91.8) | Bayes-PFL (87.0, 84.1, 89.2) | +1.3 AUROC |
| BTAD | (95.9, 91.0, 94.6) | Bayes-PFL (93.2, 91.9, 96.5) | +2.7 AUROC |
| DAGM | (98.0, 96.9, 97.7) | Bayes-PFL (97.7, 95.7, 97.0) | +0.3 AUROC |
| DTD-Synthetic | (98.8, 98.5, 99.6) | Bayes-PFL (95.1, 95.1, 98.4) | +3.7 AUROC |
| **Average** | **(92.9, 91.1, 94.6)** | Bayes-PFL (91.2, 90.3, 92.9) | **+1.7 AUROC** |

**像素级性能（Table 2）：**
- MVTec-AD: (92.4, 50.4) vs 次优 (91.9, 46.7)，AP提升 **+3.7**
- VisA: (96.6, 34.4) vs 次优 (95.6, 29.8)，AP提升 **+4.6**
- BTAD: (97.5, 59.3) vs 次优 (94.0, 47.1)，AP提升 **+12.2**
- DTD-Synthetic: (98.9, 82.7) vs 次优 (97.8, 69.9)，AP提升 **+12.8**
- **Average: (92.9, 54.2) vs 次优 (92.3, 50.3)**，AP提升 **+3.9**

### 消融实验（Table 3）
| 组件 | 图像级 (AUROC, F1, AP) | 像素级 (AUROC, AP) |
|------|------------------------|---------------------|
| w/o HGM | 88.3, 90.8, 93.2 | 82.8, 44.1 |
| w/o GGTR | 88.4, 90.9, 93.5 | 80.6, 46.1 |
| w/o DSP | 90.6, 92.2, 95.5 | 84.6, 44.8 |
| **Ours** | **92.7, 94.0, 97.1** | **92.4, 50.4** |

各组均有显著贡献，HGM对像素级性能提升最大（+8.1 AUROC, +6.3 AP）。

### 组token数量消融（Table 4）
- 最优配置：$[16, 8, 4, 2]$ → 图像级 (88.2, 92.7, 97.1)，像素级 (92.4, 50.4)
- 过多token（$[128, 32, 8, 2]$）反而降低性能，说明层级聚合需适度

---

## 相关工作脉络

1. **WinCLIP [15]**：手工设计正常/异常提示词，直接利用CLIP跨模态对齐能力；本文通过图像锚点减少对提示词的依赖。
2. **AdaCLIP [6]**：引入可学习提示缓解语言偏差；本文进一步构建图像语义锚点实现双锚点对齐。
3. **Bayes-PFL [22]**：将提示空间建模为概率分布；本文采用确定性的层次化分组策略，更直接地建立视觉锚点。
4. **VCP-CLIP [21]**：引入视觉上下文提示；本文的图像锚点提供更结构化的视觉推理框架。
5. **AnomalyCLIP [30]**：物体无关的提示学习；本文聚焦于图像锚点的层次化构建而非提示学习本身。
6. **GroupViT [28]**：将显式分组机制引入ViT用于语义分割；本文将其适配于零样本异常检测任务。
7. **TokenLearner [24] / ToMe [4]**：探索自适应token聚合；本文采用自顶向下合并策略构建层级语义表征。

---

## 局限性与未来方向

- **高级逻辑缺陷处理困难**：与现有ZSAD方法一样，对未见领域中高层逻辑缺陷的检测仍有限制。
- **未来方向**：计划扩展至**query-image少样本设置**，以更好地处理复杂缺陷场景。
- **DINOv3文本对齐较弱**：作者承认DINOv3在文本对齐方面弱于CLIP，但权衡后优先选择视觉表征能力更强的骨干。

---

## 研究启发与可借鉴点

1. **双锚点范式的迁移价值**：将图像锚点与文本锚点并列的思路可推广至其他跨模态任务（如零样本分割、开放词汇检测）。
2. **Gumbel Softmax分组机制**：可微离散分组策略可复用至其他需要结构化token聚合的场景。
3. **层次化合并的拓扑设计**：$[16, 8, 4, 2]$的指数递减组数配置值得参考，可作为其他ViT变体的设计先验。
4. **正交正则化促进语义分离**：$\mathcal{L}_{\text{ortho}}$ 约束两组token正交的思想可用于多类别锚点学习。
5. **组token精化[CLS]的策略**：GGTR的门控加权机制可作为通用的token精化模块嵌入其他模型。

---

## 关键术语表

**Zero-Shot Anomaly Detection (ZSAD)**：在未见领域无需标注数据的情况下识别异常的检测范式。

**Dual-Anchor Framework**：同时利用图像语义锚点和文本锚点实现跨模态对齐的框架。

**Hierarchical Group Merging (HGM)**：自顶向下逐步聚合局部到全局视觉特征的分组合并策略。

**Group-Gated Token Refiner (GGTR)**：利用正/异常组token作为锚点对[CLS] token进行门控加权精化的模块。

**Dynamic State Prompt (DSP)**：融合图像条件化组token的自适应文本提示，替代静态手工prompt。

**Gumbel Softmax Assignment**：通过Gumbel噪声实现可微离散分组的技术。

**Bipartite Soft Matching**：基于余弦相似度的二分图软匹配策略，用于token对的参数无合并。

**Orthogonality Regularizer**：约束正常与异常组token内积接近零的正则化项，促进语义解耦。

---

## 可复现要素

- **数据集**：MVTec-AD、VisA等14个数据集均为公开数据集，配置遵循Qu et al. [22]的协议
- **代码开源情况**：论文未明确声明代码开源状态
- **权重开源情况**：论文未声明预训练权重开源
- **关键超参**：
  - 骨干网络：DINOv3 ViT-L/16
  - 文本编码器：CLIP ViT-L/14@336px
  - 图像尺寸：518×518
  - Patch特征维度：1024（视觉）、768（文本）
  - 组token配置：$[16, 8, 4, 2]$
  - 温度系数τ：论文未明确给出具体数值
