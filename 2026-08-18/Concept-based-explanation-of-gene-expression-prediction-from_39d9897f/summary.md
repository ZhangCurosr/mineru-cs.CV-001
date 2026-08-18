---
title: "Concept-based-explanation-of-gene-expression-prediction-from"
source: https://arxiv.org/pdf/2608.16669v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:14:26"
field: "计算病理学与可解释AI交叉"
keywords: ["空间转录组学", "可解释AI", "稀疏自编码器", "相关性传播", "计算病理学", "ViT解释性"]
innovations: ["将CRP扩展至ViT架构实现概念级空间预测解释", "结合注意力感知LRP与RA-SAE后验概念发现用于虚拟空间转录组学", "系统比较R-based与Z-based概念并建立多维度评测框架"]
benchmarks: ["HEST-1k COAD", "TCGA COAD"]
---

# 论文速读：Concept-based-explanation-of-gene-expression-prediction-from-H%E2%80%93E-images

## 一句话总结
本文提出了一种可解释框架，将注意力感知的逐层相关性传播（LRP）与松弛原型稀疏自编码器（RA-SAE）的概念发现相结合，用于从 H&E 图像预测结直肠癌空间转录组学（ST），建立了分子表型与组织形态之间的关联。

## 研究问题与动机
- 现有视觉 Transformer（ViT）病理基础模型的explainability方法主要局限于局部热力图，无法揭示形态学概念如何贡献于空间基因表达预测
- 分子状态在同一组织切片内经常变化，slide-level 的全局解释无法捕获空间异质性，需要局部且空间分辨的解释
- 概念基解释框架在基因表达预测中尚未被探索，现有的 heatmap 方法定位预测但不从高层形态概念角度解释预测
- 直接对 ViT 编码器输出神经元归因会导致相关性高度分散在各维度上，概念仍强烈纠缠

## 核心贡献（创新点）
- **将 CRP 扩展至 ViT 架构**：现有 CRP 公式不直接适用于 ViT，本文通过 CP-LRP 实现对注意力、LayerNorm 和 gated MLP 等 ViT 特定组件的连续性相关性传播
- **结合注意力感知 LRP 与 RA-SAE 概念发现用于虚拟 ST**：不同于要求在训练中预设解释结构的 prototype 方法，本文在后验阶段从无约束潜表示中提取概念，保留了模型预测性能
- **建立了相关性驱动的概念图谱，链接分子表型与组织形态学**：通过 Leiden 聚类在 RA-SAE 隐藏层的相关性空间中发现原型图像，并将单个概念定位到输入图像的特定区域

## 方法详解
- **预测架构**：基于 UNI-2h ViT 编码器（ViT_g/14），结合 gated attention pooling（hidden dim=128）得到 d=1536 的 per-tile 向量，每个基因配备独立 MLP head（1536→256→1）。联合优化 MSE + (1-Pearson r) 损失，AdamW 优化器，分两阶段训练：第一阶段解冻最后 10 个 encoder 参数组和 MLP gene heads，第二阶段冻结 encoder 仅微调 MLP heads
- **RA-SAE 概念发现**：在 patch attention pooling 后的嵌入（d_input=1536）上训练，hidden dim=8000（overcomplete dictionary），TopK=32 稀疏激活，relaxation δ=25，n_C=8000 个 k-means 质心。损失函数 L_RA-SAE = L_MSE - λ·L_reanimation（λ=10^-2）
- **解释管线**：对每个基因签名聚合输出应用 LRP γ-rule（γ_conv=0, γ_linear=0.25）计算相关性，将 identity mapping 替换为训练好的 RA-SAE，得到概念层相关性；经 l1-normalize 后做 PCA（50 主成分）+ kNN 图（15 cosine neighbors）+ Leiden 聚类（resolution=0.5）发现原型；对每个概念隔离相关性并通过 SAE encoder、patch attention 和 reformulated ViT 反向传播回输入图像

## 实验与结果
- **数据集**：HEST-1k COAD（n=49 WSIs，训练35/验证9/测试5），TCGA COAD 外部验证（n=440）
- **基因表达预测**：过滤 PCC≥0.3 的 622 个基因，跨不同签名的中位 Pearson 相关系数为 0.46-0.68；可预测性与空间自相关（Moran's I）、基因表达出现频率和方差正相关
- **iCMS 分类**：HEST-1k 空间分辨加权 F1=0.872，聚合加权 F1=0.819；TCGA COAD 聚合加权 F1=0.770（kNN nearest mode），DQ confident 最高达 0.842
- **预后分层**：predicted bulk-level iCMS 状态与 measured 一致地显著分层患者总生存期（Cox regression），spot-level iCMS3 比例呈现剂量效应
- **概念对比**：R-based 概念相比 Z-based（激活）概念具有更低负干扰（2.60×10^5 vs 1.10×6）、更低概念使用广度（0.193 vs 0.405），prototype 更具形态学区分度；两者重要性等级 Spearman 相关=0.942，top-100 Jaccard=0.587

## 相关工作脉络
- **Jamshidi Idaji et al.** 分析了不同解释框架热力图的忠实度，发现 LRP 和 integrated gradients 优于 attention/gradient 方法——本文在其基础上进一步从概念层面解释空间预测
- **Fel et al. (RA-SAE)** 提出松弛原型稀疏自编码器用于 DINOv2/ImageNet 概念发现——本文将其首次应用于计算病理学 ViT 嵌入并比较了 R-based 与 Z-based 概念
- **Achtibat et al. (CRP & AttnLRP)** 提出概念相关性传播和注意力感知 LRP——本文将其扩展适配至 ViT-based 空间回归任务
- **Kim et al.** 在病理学基础模型上使用稀疏自编码器进行概念发现但基于激活而非相关性——本文系统比较了两种范式的差异
- **BLEEP / Phoenix** 实现了从 H&E 预测空间基因表达——本文在这些预测模型之上增加了概念级可解释性分析层

## 局限性与未来方向
- 虚拟空间转录组学受限于训练数据的质量和规模（HEST-1k 仅 49 样本），引入不确定性
- 当前方法跨基因聚合相关性可能模糊基因特异性解释模式，未来需联合建模基因、空间位置和 SAE 隐维度
- 缺乏最优原型/概念提取的计算高效策略，替代聚类方法和重采样稳定性分析有待探索
- 概念基于解释仍缺乏被广泛接受的忠实度度量标准，建立严格评估框架是重要开放挑战
- RA-SAE 在病理学嵌入上需要比原论文报告更弱的 archetypality 约束，需针对生物医学图像嵌入的正则化策略研究

## 研究启发与可借鉴点
- **后验概念发现范式**：不约束训练过程而在推理阶段提取概念，可移植到其他 ViT-based 病理模型和空间预测任务
- **R-based vs Z-based 概念的系统比较方法**：本文提出了多维度对比框架（dictionary coherence、negative interference、effective rank、空间分布等），可作为后续可解释性研究的评测基准
- **CP-LRP 对 ViT 的适配方案**：将 SDPA attention 替换为手动矩阵乘法以实现 Gamma rule 传播，此技术路径可复用于其他 transformer 架构的可解释性分析
- **层次化 argmax 分类器设计**：用于从连续基因表达预测离散组织类型的思路可迁移至其他多分类空间预测任务

## 关键术语表
- **虚拟空间转录组学（Virtual Spatial Transcriptomics）**：从常规 H&E 组织病理图像预测空间基因表达的技术
- **RA-SAE（Relaxed Archetypal Sparse Autoencoder）**：将字典原子约束到输入数据凸包并引入松弛参数的稀疏自编码器，用于从 ViT 潜表示中提取可解释概念
- **LRP（Layer-Wise Relevance Propagation）**：逐层相关性传播，一种将模型输出预测值分解到输入像素的贡献度的解释方法
- **CRP（Concept Relevance Propagation）**：概念相关性传播，将 LRP 相关性追溯通过学习的潜概念，连接局部解释与全局推理模式
- **iCMS（intrinsic Consensus Molecular Subtypes）**：基于单细胞转录组学定义的结直肠癌固有分子亚型（iCMS2/iCMS3），反映肿瘤细胞 intrinsic 转录状态
- **Gated Attention Pooling**：门控注意力池化，MIL 框架下的注意力机制，用于从多个 patch token 聚合得到 tile 级表示
- **Morphological Concept Atlas**：形态学概念图谱，将分子表型与组织病理学特征关联的可解释概念集合

## 可复现要素
- **数据集**：HEST-1k COAD 和 TCGA COAD，均为公开数据集
- **代码/权重**：论文未提及代码开源声明；使用了公开基础模型 UNI-2h
- **关键超参**：RA-SAE hidden dim=8000, TopK=32, δ=25, n_C=8000；encoder lr=1e-5, attention pooling lr=5e-4, gene heads lr=1e-4；weight decay=1e-3；batch size=512（prediction）/4096（SAE）
