---
title: "Dual-Anchors-Do-It-Better-Hierarchical-Group-Merging-for-Zer"
source: https://arxiv.org/pdf/2608.11933v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 12:27:02"
field: "零样本异常检测"
keywords: ["zero-shot anomaly detection", "dual-anchor", "hierarchical group merging", "vision-language model", "anomaly localization"]
innovations: ["提出双锚点框架，通过分层组并构建图像语义锚点并与文本锚点融合实现平衡对齐", "设计分层组并策略（HGM），自顶向下渐进聚合局部到全局视觉特征形成层次化异常表示", "引入动态状态提示与Group-Gated Token Refiner，降低prompt依赖并增强全局表征判别力"]
benchmarks: ["MVTec-AD", "VisA", "MPDD", "BTAD", "RSDD", "KSDD2", "DAGM", "DTD-Synthetic", "ISIC", "ClinicDB", "ColonDB", "TN3K", "Endo", "Kvasir"]
---

# 论文速读：Dual Anchors, Do It Better: Hierarchical Group Merging for Zero-Shot Anomaly Detection

## 一句话总结
本文提出一种**双锚点（Dual-Anchor）零样本异常检测框架**，通过分层组并机制从图像编码器中提取正常/异常组token作为图像语义锚点，与文本锚点融合形成平衡的对齐范式，显著降低了对文本prompt的依赖，在8个工业和6个医学基准上均取得最优或次优结果。

## 研究问题与动机
1. **过度依赖文本锚点**：现有CLIP-based ZSAD方法将正常/异常的语义锚点主要设在文本端，性能高度敏感于prompt设计与调优。
2. **弱视觉锚定**：单锚点设计导致在未见域中的决策边界无法充分覆盖所有异常样本（图1(b) top）。
3. **忽视图像编码器的层级结构优势**：现有方法通常将不同层的特征独立处理，未充分利用从局部到全局的层次化语义聚合。
4. **缺乏结构化的视觉推理能力**：即使引入视觉上下文提示的方法，仍以文本驱动的对齐为主，缺乏能有效建模层次语义的视觉中心化表示。

## 核心贡献（创新点）
1. **双锚点范式**：通过分层组并构建图像语义锚点（正常/异常组token），与文本锚点互补，形成更平衡的跨模态对齐——区别于仅依赖文本锚点的单锚点方法（如WinCLIP、AnomalyCLIP）。
2. **分层组并（Hierarchical Group Merging）**：自顶向下的渐进式合并策略，将局部到全局的视觉特征聚合为层次化的异常表示——区别于GroupViT等仅做语义分割的分组方法，本文聚焦无监督异常检测且无需显式标注。
3. **动态状态提示（Dynamic State Prompt）**：组token被注入可学习的上下文token中，生成图像条件自适应的提示——区别于AdaCLIP等静态learnable prompt，本文提示随输入图像动态调整。
4. **Group-Gated Token Refiner**：利用组token作为门控信号校准[CLS]全局表示——该机制为图像表示的自适应校准提供了轻量且有效的新途径。

## 方法详解
**整体架构**（图2）：采用冻结的**DINOv3（ViT-L/16）**主干提取多层patch tokens，经分层组并生成组token，再通过GGTR和动态状态提示完成双锚点对齐。

### 关键模块
- **Hierarchical Group Merging**：由多个Merge Block组成，每Block含四个阶段：
  1. **Initialization**：用前一层的group tokens做cross-attention初始化当前层特征（公式1）
  2. **Assignment**：基于Gumbel Softmax实现可微的离散分组分配（公式2-4），温度参数τ控制离散程度
  3. **Merging**：采用**bipartite soft matching**（参考ToMe [4]）进行无参数token降维，按余弦相似度配对并平均合并（公式5-8）
  4. **Update**：最终Merge Block中将group tokens回注给patch tokens，得到最终分配表示$F_{assign}$

- **Image Anchor**：经过H层合并后，得到正常组token $\mathbf{G}_N$ 和异常组token $\mathbf{G}_A$（公式9），作为图像侧的语义锚点。

- **Group-Gated Token Refiner (GGTR)**：计算[CLS] token与$\mathbf{G}_N$、$\mathbf{G}_A$的余弦相似度作为门控权重$w_n, w_a$（公式10），经线性层+sigmoid后以残差方式更新$[\text{CLS}]$（公式11）。

- **Dynamic State Prompt**：将组token嵌入prompt模板：
  $p_n = [V_1][V_2]\cdots[V_E][W_1][\mathbf{G}_N][\text{class}]$，$p_a$同理（公式12），再经CLIP Text Encoder得到文本嵌入$\mathcal{T}=\{t_n, t_a\}$。

### 损失函数
- **像素级**：$\mathcal{L}_{seg} = \text{Dice} + \text{Focal}$（公式16）
- **图像级**：$\mathcal{L}_{cls} = \text{BCE}(p_{cls}, y)$（公式15-16）
- **正交正则**：$\mathcal{L}_{ortho} = |\langle \mathbf{G}_N, \mathbf{G}_A \rangle|^2$，鼓励正常/异常组token语义分离（公式17）
- **分组一致性**：$\mathcal{L}_{group} = 1 - \frac{1}{2}(\cos(\mathbf{G}_N, \mathbf{Z}_N) + \cos(\mathbf{G}_A, \mathbf{Z}_A))$，用ground-truth mask分区并计算均值特征以约束组token（公式18）
- **总损失**：$\mathcal{L}_{total} = \mathcal{L}_{cls} + \mathcal{L}_{seg} + \mathcal{L}_{ortho} + \mathcal{L}_{group}$（公式19）

## 实验与结果
**数据集**：8个工业数据集（MVTec-AD、VisA、MPDD、BTAD、RSDD、KSDD2、DAGM、DTD-Synthetic）+ 6个医学数据集（ISIC、ClinicDB、ColonDB、TN3K、Endo、Kvasir）。辅助训练用MVTec AD；MVTec-AD评测时用VisA训练。

**评估指标**：图像级AUROC/F1-score/AP；像素级AUROC/AP。

**主要结果**（Table 1 & 2）：
- **图像级平均AUROC**：本文**92.9%**，超越第二的Bayes-PFL（91.2%）+1.7pp；平均AP达**94.6%**，超越第二的Bayes-PFL（92.9%）+1.7pp。
- **像素级平均AUROC**：本文**92.9%**，超越第二的Bayes-PFL（92.3%）+0.6pp；平均AP达**54.2%**，超越第二的Bayes-PFL（50.3%）+3.9pp。
- 在VisA上以**88.3**的图像级AUROC刷新最高（第二名Bayes-PFL为87.0）；在DTD-Synthetic上以**98.8**图像级AUROC刷新最高。
- 医学数据集上同样全面领先或接近最优（ISIC像素级AUROC 93.0，CVC-ColonDB像素级AUROC 83.4达最高）。

**消融实验**（Table 3）：
- 去除HGM导致图像级AUROC下降4.4pp、像素级下降9.6pp，验证层级融合的关键作用。
- 去除GGTR导致图像级AUROC下降4.3pp、像素级下降11.8pp。
- 去除Dynamic State Prompt导致图像级AUROC下降2.1pp、像素级下降4.8pp。

**分组数量消融**（Table 4）：最佳配置为$[16, 8, 4, 2]$，过多或过少均损害性能。

## 相关工作脉络
1. **WinCLIP** [15] / **AnomalyCLIP** [30]：纯文本prompt驱动的CLIP-based ZSAD，依赖手工或静态learnable prompt，本文通过引入图像锚点打破单一文本依赖。
2. **AdaCLIP** [6]：引入learnable context tokens，但prompt仍为静态，本文的动态状态提示使prompt随输入图像自适应调整。
3. **Bayes-PFL** [22]：将prompt空间建模为概率分布，本文与之竞争且全面超越，核心差异在于引入结构化的视觉分组而非仅概率采样。
4. **GroupViT** [28]：将分组机制引入ViT用于语义分割，本文借鉴其分组思想但应用于无监督异常检测且无需标注。
5. **TokenMerg (ToMe)** [4]：bipartite soft matching策略的来源，本文将其扩展为多层级联的分层组并流程。
6. **VCP-CLIP** [21]：引入视觉上下文提示，本文认为其仍偏文本驱动，通过显式的分组token构建更结构化的视觉锚点。

## 局限性与未来方向
1. 论文自述：面对**未见域中的高级逻辑缺陷（high-level logical defects）**仍存在困难。
2. 未来方向：扩展到**query-image-based few-shot设置**，以更好地处理上述场景。
3. 隐含局限：使用DINOv3替代CLIP作为主干，牺牲了一定程度的图文对齐能力以换取更强的视觉表征，需权衡。

## 研究启发与可借鉴点
1. **HGM的分层组并思想可迁移**：从局部到全局的渐进式token聚合策略，可借鉴到影像分割、目标检测等需要多尺度语义整合的任务中。
2. **Gumbel Softmax分组分配的稳定性设计**：将soft assignment与cross-attention结合的实现细节，值得在需要可微离散化的其他场景中复现。
3. **动态状态提示机制**：将图像side的learnable tokens注入prompt模板的思路，可推广至其他VLM下游任务的prompt engineering。
4. **双锚点对齐范式**：图像锚点+文本锚点的对称设计可启发自监督/半监督视觉表示学习的对比训练策略。
5. **实验设计值得借鉴**：同时在工业缺陷检测和医学影像诊断两大高价值领域统一验证，且提供像素级定位定性对比，论证充分。

## 关键术语表
**Zero-Shot Anomaly Detection (ZSAD)**：在无目标域标签的情况下，借助视觉-语言模型的跨模态对齐能力实现异常检测与定位。
**Dual-Anchor Framework**：同时利用图像侧语义锚点（正常/异常组token）和文本侧锚点，实现更均衡的跨模态对齐。
**Hierarchical Group Merging (HGM)**：自顶向下的多阶段合并策略，将patch tokens逐层聚合为具有层次语义的group tokens。
**Group Token**：可学习的、代表特定语义概念的token，经分组后充当图像侧的语义锚点（正常$\mathbf{G}_N$、异常$\mathbf{G}_A$）。
**Gumbel Softmax Assignment**：通过Gumbel噪声实现的离散但可微的分组分配机制，支持端到端优化。
**Group-Gated Token Refiner (GGTR)**：利用组token与[CLS]的余弦相似度作为门控权重，自适应校准全局图像表示。
**Dynamic State Prompt**：将图像conditioned的组token融入prompt模板，生成随输入动态调整的文本表示。
**Bipartite Soft Matching**：无参数的token合并策略，通过余弦相似度配对并平均相似token以实现降维。
