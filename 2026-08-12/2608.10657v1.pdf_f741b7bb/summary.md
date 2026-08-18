---
title: "Retrieval-Augmented Vision Foundation Models for Robust Leukemia Cell Classification across Multiple Microscopy Datasets"
source: https://arxiv.org/pdf/2608.10657v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:44:14"
field: "医学图像分析与白血病诊断"
keywords: ["leukemia classification", "vision foundation model", "low-rank adaptation", "retrieval-augmented classification", "domain shift", "held-out evaluation", "cytomorphology"]
innovations: ["两阶段级联流水线解耦白血病检测与ALL/AML亚型分类，充分利用含纯正常细胞的数据集扩充Stage 1训练", "系统benchmark量化预训练特化、LoRA适配与RAC三类因素对跨域性能的独立贡献", "首次用held-out跨域协议揭示单细胞白血病数据中模型依赖背景伪影的捷径学习问题"]
benchmarks: ["ALL-IDB2 (held-out), C-NMC 2019, AML-Cytomorphology, Peripheral Blood Cell Dataset, AML-Cytomorphology MLL Helmholtz"]
---

# 论文速读：Retrieval-Augmented Vision Foundation Models for Robust Leukemia Cell Classification across Multiple Microscopy Datasets

## 一句话总结
本文提出一个跨多数据集的鲁棒白血病细胞分类框架，采用两阶段流水线（Stage 1：白血病/非白血病二分类；Stage 2：ALL/AML亚型分类），在冻结编码器线性探测、LoRA微调与检索增强分类（RAC）三种策略下对比三类预训练特化的视觉基础模型。核心发现是：通用模型经低成本LoRA适配后可逼近甚至在Stage 1上超越领域专用模型，但Stage 2跨域评估揭示了现有单细胞数据集因各亚型来自单一成像源而迫使模型依赖数据集特异性伪影的问题。

## 研究问题与动机
- **核心问题**：预训练特化程度对模型在域偏移（domain shift）下的性能影响有多大？低成本适配技术（PEFT/LoRA、检索增强）能否替代昂贵的领域专用预训练？
- **现有方法不足**：多数白血病分类研究仅使用单一数据集训练与测试，无法揭示跨域泛化缺陷；即使报告高精度，也常依赖数据集特定伪影（如背景、染色），在真实临床场景中崩溃。
- **评估协议缺失**：此前工作（如Ben Rabah & Serag）因类别定义不一致而排除白血病做跨域评估，且内部随机划分无法暴露域外退化现象。
- **亚型分类难题**：现有公开单细胞数据集中，ALL与AML样本分别来自不同成像平台，导致模型难以学习细胞形态学特征而非背景伪影。

## 核心贡献（创新点）
1. **两阶段级联分类流水线**：将白血病诊断分解为"是否存在白血病→ALL/AML亚型"两个独立子任务，Stage 1可利用包含纯正常细胞的源扩充训练数据至122,167张（而非仅72,824张）。
2. **系统benchmark量化预训练特化贡献**：在同一条流水线上公平比较DinoBloom（单细胞域）、BiomedCLIP（生物医学域）与CLIP（通用域），隔离预训练域、适配器策略（linear probe vs. LoRA）与RAC各自贡献。
3. **Held-out数据集评估协议用于白血病**：首次在白血病分类任务中实现跨数据集训练+持出数据集测试，暴露了仅靠内部划分无法发现的数据集特异性捷径问题。
4. **RAC模块的消融与诊断价值**：不仅用于提升性能，还作为诊断工具揭示嵌入空间中细胞是按形态学还是按数据集伪影组织。

## 方法详解
- **两阶段流水线**：Stage 1对122,427张统一标注（normal/ALL/AML）的单细胞图做二分类（leukemia vs. non-leukemia）；Stage 2对Stage 1预测为阳性的72,824张白血病细胞做ALL vs. AML分类，两阶段独立训练与评估。
- **三个编码器**：DinoBloom-B（ViT-B/14，768维嵌入，85.7M参数，预训练于13个单细胞数据集）；BiomedCLIP（ViT-B/16，512维，86.2M参数，预训练于PubMed Central图文对）；CLIP（ViT-B/32，512维，87.8M参数，预训练于自然图文对）。三者参数量差异<2.5%，主要变量为预训练域。
- **Linear Probing**：冻结编码器，仅在预计算嵌入上训练加权逻辑回归分类器，无数据增强，直接隔离编码器表征质量。
- **LoRA适配**：在attention投影层注入低秩矩阵（$h = W_0 x + B A x$），rank $r=8$、scale=16、dropout=0.05，可训练参数仅~300K（占总量~0.35%），AdamW，lr=$1\times10^{-4}$，batch=128，3 epochs，混合精度，轻微增强（水平/垂直翻转+颜色抖动）。
- **RAC模块**：以余弦相似度在L2归一化嵌入库中检索top-$k$（$k=20$）最近邻，生成检索分布 $p_{\mathrm{ret}}$，与线性分类分布 $p_{\mathrm{probe}}$ 在log空间融合：$\mathrm{logits}_{\mathrm{final}} = \log p_{\mathrm{probe}} + \alpha \log p_{\mathrm{ret}}$；linear probe下$\alpha$在$\{0, 0.25, 0.5, 0.75, 1\}$中网格搜索，LoRA下固定$\alpha=0.5$。
- **域偏移缓解预处理**：C-NMC 2019数据集的黑背景被替换为来自健康外周血抹片的随机区域（含RBC），配合随机旋转/缩放与Reinhard颜色归一化。
- **标签统一**：五个数据集统一为三类（normal/ALL/AML），仅保留非歧义核白细胞，排除血小板/成红细胞/ smudge细胞/成熟粒细胞；基因AML亚型坍缩为单类。
- **评估指标**：Accuracy、Macro Recall、Macro F1（等权各类，避免多数类主导）。

## 实验与结果
- **数据集**：5个公开数据集（C-NMC 2019、AML-Cytomorphology、Peripheral Blood Cell、AML-Cytomorphology MLL Helmholtz、ALL-IDB2）合并为122,427张单细胞图，经过滤后Stage 1训练122,167张，Stage 2训练69,400张。
- **Stage 1 Held-out评估**：在ALL-IDB2（n=260）上测试。最佳结果：**DinoBloom + Linear Probe + RAC，Accuracy=0.9423**；CLIP+LoRA达0.9115，超越DinoBloom linear probe（0.8923）。LoRA使CLIP相对linear probe提升+0.2769，DinoBloom仅+0.0385；预训练差距从0.2577收窄至0.0193。
- **RAC贡献**：DinoBloom linear probe +RAC提升+0.0500；CLIP linear probe -RAC下降-0.0769，说明嵌入空间需编码细胞形态学模式RAC才有效。
- **Stage 2 Held-out评估（关键发现）**：在所有配置下，ALL recall≈0.0000，模型几乎全部预测为AML（test set含3,294 AML vs. 130 ALL）。但在同一数据上做80/20随机内部分割时，ALL recall高达0.9988，CLIP冻结甚至达到1.0000 Accuracy——直接证明模型在训练时依赖背景伪影而非细胞形态。
- **控制实验**：同一模型在域内随机划分下ALL recall≥0.99，在held-out下ALL recall=0，揭示先前工作内部划分的虚假高表现。
- **结论数字**：预训练特化带来0.2577的accuracy差距（linear probe），LoRA可将该差距缩小至0.0193；最高性能（0.9423）来自领域专用模型+RAC，无需任何微调。

## 相关工作脉络
- **Prasad & Anbarasi / Pandey et al. / Revathi & Kaliappan**：使用ViT在单一数据集上做ALL分类，缺乏跨域评估，易过拟合数据集特定特征。
- **Wang et al. (ALSNet)**：跨平台180,928张细胞图训练并做held-out评估，但聚焦19类细胞形态学+病例级预测，未涉及LoRA/RAC与两阶段pipeline。
- **Patel et al.**：探索CLIP/BiomedCLIP/PLIP的few-shot提示工程，未涉及编码器适配，无法量化适配对预训练差距的补偿。
- **Naing et al.**：基于ResNet34度量学习做AML检索分类，检索嵌入空间从零训练；本文在预训练VFM嵌入上检索并提供cytomorphological grounding，且结合LoRA/linear probe对比。
- **Ben Rabah & Serag**： benchmark预训练特化影响，但因白血病类别定义不一致而将其排除在跨域评估之外；本文通过标签统一解决了该障碍，同时揭示了第二层障碍（亚型数据源单一）。
- **Ng et al.**：使用DinoBloom+LoRA+检索做WBC分层集成pipeline，但仅一种编码器、无法隔离预训练贡献；hold-out测试来自训练同域。

## 局限性与未来方向
- **Stage 1 held-out集小**：ALL-IDB2仅260张，小幅accuracy波动对应个位数细胞数差异。
- **预训练污染**：DinoBloom预训练已包含三个源数据集（D2/D3/D4），AML评估对领域专用编码器并非完全 unseen。
- **Stage 2根本限制**：当前公开数据集中ALL与AML分别来自单一成像源，模型依赖背景捷径；需新增同平台双亚型 held-out数据集，或扩充多源ALL/AML数据迫使其学习形态学特征。
- **背景归一化缺失**：建议对所有数据集应用统一合成背景，强制编码器关注细胞本身。
- **超参不一致**：LoRA配置的$\alpha$固定为0.5（计算预算限制），而linear probe下$\alpha$网格搜索，两者对比存在偏差。

## 研究启发与可借鉴点
- **Held-out评估作为诊断工具**：跨域held-out测试不仅能测性能，更能揭示模型依赖的捷径特征（如背景），可作为常规评估流程的一部分。
- **低成本适配可大幅缩小预训练差距**：LoRA使通用CLIP在Stage 1上逼近领域模型，提示在资源受限场景下可优先投资适配而非昂贵预训练。
- **RAC有效性依赖嵌入空间质量**：RAC仅在编码器已学到形态学相似性时增益明显；否则引入噪声。可借鉴为"先评估嵌入空间是否编码任务相关特征，再决定是否加检索"的决策流程。
- **两阶段解耦策略可推广**：将复杂诊断分解为粗筛→细分的两级，并利用所有可用标签（包括仅含负样本的数据集）扩充第一阶段训练规模。
- **标签统一协议**：排除歧义细胞类型、统一核白细胞包含准则的做法，可为多源医学图像数据整合提供可复用范式。

## 关键术语表
- **Vision Foundation Model (VFM)**：在大规模图像数据上预训练的视觉模型，可通过适配迁移至下游分类任务。
- **Low-Rank Adaptation (LoRA)**：冻结预训练权重，仅训练注入的低秩更新矩阵以实现参数高效的微调。
- **Linear Probing**：冻结编码器，仅在提取的嵌入上训练轻量线性分类器，用于评估预训练表征质量。
- **Retrieval-Augmented Classification (RAC)**：在推理时检索嵌入空间中top-k最相似训练样本，以其标签的相似度加权投票辅助分类。
- **Domain Shift**：训练与测试数据来自不同采集设备、染色、光照或实验室协议导致的分布差异。
- **Held-out Dataset Evaluation**：评估时完全排除训练集来源数据，以真实测量跨域泛化能力。
- **Cytomorphology**：细胞的形态学特征（胞质、核仁、核膜等），是白血病分型诊断的依据。
- **Macro-averaged Recall/F1**：对各类别等权计算recall或F1后取平均，避免多数类主导指标。

## 可复现要素
- **数据集**：5个公开数据集（C-NMC 2019、AML-Cytomorphology、Peripheral Blood Cell Dataset、AML-Cytomorphology MLL Helmholtz、ALL-IDB2），均公开发布。
- **代码/权重**：DinoBloom-B 来自HuggingFace Hub (MarrLab)；CLIP 来自OpenAI `clip-vit-base-patch32`；BiomedCLIP 通过open_clip加载；权重均为官方公开版本。论文未提供统一代码仓库链接。
- **关键超参**：LoRA rank $r=8$、scale=16、dropout=0.05；lr=$1\times10^{-4}$，batch=128，3 epochs；RAC $k=20$，$\alpha\in\{0,0.25,0.5,0.75,1\}$（linear probe网格搜索，LoRA固定0.5）；输入统一224×224。
- **实验环境**：Windows 11, Intel Core i9-13900HX, RTX 4090 GPU, 32GB RAM；Python 3.12.10, PyTorch 2.5.1 (CUDA 12.1), transformers 5.9.0, PEFT 0.20.0, open-clip 3.3.0。
