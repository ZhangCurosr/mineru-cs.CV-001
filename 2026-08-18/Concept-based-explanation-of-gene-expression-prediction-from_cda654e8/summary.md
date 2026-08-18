---
title: "Concept-based-explanation-of-gene-expression-prediction-from"
source: https://arxiv.org/pdf/2608.16669v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:14:39"
field: "计算病理学与可解释AI"
keywords: ["虚拟空间转录组学", "概念解释", "稀疏自编码器", "层间相关性传播", "数字病理", "结直肠癌"]
innovations: ["将CRP扩展到ViT架构实现概念级归因", "结合注意力感知LRP与RA-SAE建立形态-分子概念图谱", "系统比较相关性vs激活基础概念的忠实现性"]
benchmarks: ["HEST-1k COAD", "TCGA COAD"]
---

# 论文速读：Concept-based explanation of gene expression prediction from H&E images

## 一句话总结
本文提出了一个可解释框架，将注意力感知层间相关性传播（LRP）与松弛原型稀疏自编码器（RA-SAE）概念发现相结合，实现了从H&E图像预测基因表达的同时，建立转录程序与组织形态学特征之间的关联。

## 研究问题与动机
1. 现有数字病理ViT基础模型的预测方法缺乏可解释性，现有解释方法（如注意力热力图、梯度saliency）仅能定位预测区域，无法揭示形态学概念如何贡献于基因表达预测
2. 空间转录组（ST）揭示了肿瘤内显著的空间异质性（如iCMS2/iCMS3共存），但现有模型难以将分子表型与组织形态学特征建立联系
3. 现有概念发现方法（如PCX）在ViT编码器输出上聚类时，概念高度纠缠且缺乏下游预测归因；而基于激活的概念发现通常与空间回归任务脱钩
4. 虚拟空间转录组学方法（如BLEEP、Phoenix）主要关注预测精度，忽略了模型预测所依赖的形态学概念的解释性需求

## 核心贡献（创新点）
1. **将CRP扩展到ViT架构**：通过CP-LRP（保守传播）将LRP适配到Vision Transformer，实现了在ViT中进行概念级别的可归因解释，而现有CRP formulations不直接适用于该架构
2. **结合注意力感知LRP与RA-SAE概念发现**：首次将注意力感知相关性传播与松弛原型稀疏自编码器结合用于虚拟空间转录组学，实现了局部（单样本）和全局（跨样本概念图谱）的解释
3. **建立形态学概念图谱**：提出了post-hoc概念发现框架，无需在训练期间约束模型，从无约束潜在表示中提取可解释形态学概念，并与基因表达预测直接关联
4. **验证了相关性vs激活的概念质量差异**：系统比较了基于相关性（R）和基于激活（Z）的概念，发现相关性方法能建立组织形态学与下游预测之间更直接的关联，且概念使用更紧凑、负干扰更低

## 方法详解
**预测架构**：
- 编码器：Fine-tuned UNI-2h（基于ViT_g/14），DINO-v2路径学基础模型
- 池化：Patch Gated Attention Pooling（隐藏维度128），输出d=1536的patch embedding
- 预测头：每个基因独立的MLP（Linear 1536→256→1），预测log1p基因表达
- 损失函数：联合MSE-Pearson损失（MSE + (1-Pearson r)）
- 两阶段训练：先训练最后10个encoder参数组和MLP基因头，再冻结encoder微调MLP

**概念发现（RA-SAE）**：
- 输入：attention pooling后的embeddings，d_input=1536，per-feature标准化
- 稀疏自编码器：隐藏维度d_hidden=8000（overcomplete），TopK稀疏激活（K=32）
- 原型约束：字典D受限于输入数据的凸包（convex hull），通过n_c=8000个k-means质心确定，引入松弛参数δ=25平衡约束与重建
- 损失函数：L_RA-SAE = L_MSE - λ·L_reanimation，λ=10^-2
- Reanimation loss：鼓励概念不被永久"死亡"，定义为单位batch内concept j未激活的比例加权

**解释生成流程**：
1. 使用LRP γ-rule（γ_conv=0, γ_linear=0.25）计算每个基因signature的relevances
2. 将MLP identity映射替换为RA-SAE，得到概念层面的relevances R(D)
3. 对每个tile进行l1归一化后PCA降维（50维），构建kNN图（k=15, cosine距离）
4. Leiden聚类（resolution=0.5）生成概念簇，选取与簇质心余弦相似度最高的tile作为原型
5. 通过概念最大绝对质量选择关键概念，隔离单个概念后反向传播回输入图像

**分类器**：
- 基于gene signature的argmax分类：raw_i,s = 均值预测表达式
- 校准标量μ_s平衡各signature的表达量级
- 分层分类：stroma/immune → normal（安全边R=2）→ iCMS2/iCMS3

## 实验与结果
**数据集**：
- HEST-1k COAD：n=49 WSIs，训练35/验证9/测试5，交集15,162个基因，813个预测基因（6个signature）
- TCGA COAD：n=440外部验证队列

**预测性能**：
- 基因表达预测：过滤PCC≥0.3后保留622个基因，各signature的PCC中位数0.46–0.68
- 可预测性与空间自相关（Moran's I）、非零GE频率、GE方差正相关
- 组织分类加权F1：空间分辨率分类0.872，聚合bulk分类0.819（HEST-1k）；TCGA COAD weighted F1达0.770–0.842（取决于分类策略）
- iCMS预后分层：Cox回归显示iCMS3与健康不良预后显著相关（HR与测量数据一致），空间分辨率显示iCMS3比例越高风险越大

**概念发现结果**：
- iCMS2原型多由多个概念组合解释（形态较分散）；iCMS3原型常由单一主导概念解释
- 识别出iCMS2特有概念：增大核、明显核仁、管状生长模式、腔内坏死碎屑
- iCMS3特有概念：正常上皮向腺瘤过渡含杯状细胞、大片坏死碎屑（无隐窝）、含肿瘤和免疫细胞的粘液
- 共享概念：增大核伴明显核仁、拉长核伴小核仁
- RA-SAE对比TopK SAE：更少dead codes、更高一致性、更低内在维度

**R vs Z概念比较**：
- 重要性排名高度相关（Spearman=0.942），Top-100 Jaccard=0.587
- R-based概念使用更紧凑（Breadth: 0.193 vs 0.405），负干扰更低（2.6×10^5 vs 1.1×10^6）
- R-based概念在morphological distinctions上更强，而Z-based概念包含更多artifact cluster

## 相关工作脉络
1. **虚拟空间转录组学**（BLEEP、Phoenix、DeepSpot-M）：本文在这些高精度预测框架基础上引入可解释性，填补了"预测强但不可解释"的空白
2. **概念相关性传播（CRP）**（Achtibat et al., Nature MI 2023）：本文将其扩展到ViT架构，并首次应用于空间基因表达预测任务
3. **注意力感知LRP**（CP-LRP, AttnLRP）：本文使用1xt工具包中的CP-LRP实现ViT架构中的连续相关性传播
4. **稀疏自编码器概念发现**（RA-SAE, Fel et al. 2025）：本文将其从ImageNet分类任务扩展到病理学空间回归任务，发现需要更强的松弛参数δ=25
5. **激活-based概念分析**（Kim et al., 2026）：本文对比R和Z两种方式，证明在病理学预测中相关性归因比激活激活更适合建立形态-分子关联
6. **数字病理可解释性**（xMIL等）：本文针对空间分辨率预测任务的需求，提出了从全局概念到局部形态学的多层次解释策略

## 局限性与未来方向
1. **数据集规模有限**：HEST-1k仅49个WSI，概念原型存在较强患者特异性，未来需更大规模数据集提升可重复性
2. **概念发现无标准化策略**：当前缺乏从众多原型和概念中选择最优集合的计算高效方法，影响了R/Z概念的对比可比性
3. **基因级解释被聚合掩盖**：当前方法在signature层面聚合relevances，可能掩盖基因特异性模式，未来需联合建模基因、空间位置和SAE隐维度
4. **相关性传播的可改进性**：顶层架构结构与reference-value based attribution兼容，可进一步提升局部解释的忠实度
5. **概念忠实度评估缺乏统一标准**：概念级解释的faithfulness度量仍是开放问题，需建立更严格的评价框架
6. **理论基础待完善**：病理图像、激活、基因表达签名和相关性之间的几何/拓扑关系尚需统一理论框架

## 研究启发与可借鉴点
1. **RA-SAE在病理学表征中的适配**：论文发现病理学表征需要更强的松弛参数（δ=25 vs 文献报告的1）才能达到满意重建质量，提示不同模态的SAE超参数需针对性调优
2. **CP-LRP对ViT的可解释性价值**：通过reformulation将ViT注意力机制适配到LRP框架，为其他ViT病理模型提供了可直接复用的解释技术路径
3. **post-hoc概念发现保留预测性能**：无需在训练期间引入解释约束，概念从自由学习的潜在表示中提取，这一设计原则适用于各类基础模型的可解释性分析
4. **R vs Z概念的定量比较框架**：论文提出了包括Jaccard、Spearman重要性、负干扰、effective rank等多个维度的系统性比较方法，可作为概念发现工作的评估基准
5. **空间异质性分析的临床价值**：iCMS2/iCMS3的spot-level空间分布与bulk-level分类的差异揭示了预后信息，提示空间分辨率解释可能挖掘传统slide-level分析遗漏的生物学信号

## 关键术语表
**Spatial Transcriptomics (ST)**：空间转录组学，一种保留组织空间位置信息的基因表达测量技术
**iCMS (intrinsic Consensus Molecular Subtypes)**：内在共识分子亚型，基于单细胞转录组分析定义的结直肠癌恶性上皮细胞状态（iCMS2和iCMS3）
**RA-SAE (Relaxed Archetypal Sparse Autoencoder)**：松弛原型稀疏自编码器，通过凸包约束和松弛参数学习的稀疏字典学习框架，用于从潜在表示中提取可解释概念
**LRP (Layer-wise Relevance Propagation)**：层间相关性传播，一种将模型预测输出归因到输入像素的守恒解释方法
**CRP (Concept Relevance Propagation)**：概念相关性传播，将LRP扩展到概念层面，将相关性追溯到学习的潜在概念
**CP-LRP (Conservative LRP)**：保守层间相关性传播，针对Transformer架构改进的LRP变体，确保注意力机制中相关性的保守传播
**Patch Attention**：patch级注意力机制，在ViT的patch tokens上学习的可学习注意力权重，用于多实例学习中的池化
**Morphological Concept Atlas**：形态学概念图谱，通过聚类概念相关性建立的全局可解释表示，将分子表型与组织形态学特征关联

## 可复现要素
- **数据集**：HEST-1k COAD（公开）和TCGA COAD（公开）
- **代码/权重**：论文未提及代码是否开源，UNI-2h编码器为公开模型
- **关键超参**：RA-SAE hidden dim=8000, K=32, δ=25, n_c=8000; LRP γ_conv=0, γ_linear=0.25; patch attention hidden dim=128; MLP dim=256; batch size=512（预测）/4096（SAE）; lr=1e-5/5e-4/1e-4（分部件）
