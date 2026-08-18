---
title: "Retrieval-Augmented Vision Foundation Models for Robust Leukemia Cell Classification across Multiple Microscopy Datasets"
source: https://arxiv.org/pdf/2608.10657v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:45:23"
field: "医学图像分类与域泛化"
keywords: ["leukemia classification", "vision foundation models", "domain shift", "LoRA", "retrieval-augmented classification", "cytomorphology", "cross-dataset evaluation"]
innovations: ["两阶段级联分类管线结合跨数据集held-out评估协议揭示捷径学习现象", "系统量化专用vs通用视觉基础模型在白血病分类中的预训练专业化影响及LoRA/RAC的补偿能力"]
benchmarks: ["ALL-IDB2 (held-out)", "C-NMC 2019", "AML-Cytomorphology", "AML-Cytomorphology MLL Helmholtz", "Peripheral Blood Cell Dataset"]
---

# 论文速读：Retrieval-Augmented Vision Foundation Models for Robust Leukemia Cell Classification across Multiple Microscopy Datasets

## 一句话总结
本文提出一个两阶段检索增强视觉基础模型框架，在五个异构单细胞显微镜数据集上实现鲁棒的白血病检测与分型（ALL/AML）；通过held-out数据集评估协议揭示了现有方法依赖数据集特有伪影而非细胞形态学特征的隐患，并证明高效的低成本适配（LoRA + RAC）可部分弥补昂贵领域专用预训练。

## 研究问题与动机
- **现实域偏移问题**：临床场景中，采集设备、染色方案、光照条件和操作协议的差异导致单数据集模型泛化性差；现有工作大多使用同源内部分割评估，掩盖了真实域漂移下的性能崩溃。
- **领域专用预训练成本高**：DinoBloom等专用编码器需大量标注单细胞图像预训练，计算资源消耗大；是否存在成本效益更高的替代方案（PEFT + 检索增强）？
- **已有工作的局限**：此前白血病分类研究多局限于单一数据集训练/评估；Wang等虽采用cross-dataset训练，但聚焦细胞形态分类而非白血病亚型判定；Ben Rabah & Serag的基准测试中白血病被排除在外，因跨数据集类别定义不一致。
- **评估协议的诊断价值缺失**：内部80/20随机分割可能产生虚假高准确率，无法暴露模型对数据集特有背景伪影的依赖。

## 核心贡献（创新点）
1. **两阶段级联分类管线**：Stage 1进行二分类（白血病 vs 非白血病），Stage 2条件性地进行ALL/AML亚型分类，将原始任务拆解为两个独立子任务；与已有工作相比，这是首个将两阶段管线应用于白血病跨数据集分类的研究。
2. **系统基准测试预训练专业化影响**：在同一管线中对比DinoBloom（单细胞专用）、BiomedCLIP（生物医学）、CLIP（通用）三个编码器，隔离出预训练域作为主要变异来源（三者参数差异<2.5%）；相比Ben Rabah & Serag仅做线性探针+末层微调，本文进一步量化LoRA和RAC的补偿效果。
3. **Held-out数据集评估协议**：Stage 1在不在预训练语料中的ALL-IDB2上评估，Stage 2用ALL-IDB2 + AML-Cytomorphology组合作为未见域测试集；该协议成功暴露了模型在学习数据集背景捷径而非细胞形态学的现象。
4. **检索增强分类（RAC）模块**：在嵌入空间中检索top-k最相似细胞图像并提供细胞形态学依据的上下文；与Naing等独立训练的度量学习检索不同，本文的RAC在预训练基础模型嵌入空间上运行，并与LoRA/线性探针联合验证。
5. **标签规范化与域偏移缓解预处理**：对五个异构数据集的标签统一映射为三类（normal/ALL/AML），并对C-NMC 2019的黑色背景替换为合成外周血涂片背景以减小域差异。

## 方法详解
- **数据集整合**：合并C-NMC 2019（D1）、AML-Cytomorphology（D2）、Peripheral Blood Cell（D3）、AML-Cytomorphology MLL Helmholtz（D4）、ALL-IDB2（D5）五个公开数据集，经筛选后得到122,427张单细胞图像；所有遗传AML亚型折叠为单一AML类，仅保留明确的有核白细胞。
- **域偏移缓解预处理**：将C-NMC 2019的黑色背景替换为随机选取的健康外周血涂片区域（含RBC），配合旋转、缩放和Reinhard颜色标准化，使图像视觉一致性提升。
- **两阶段管线**：Stage 1用122,167张图像（去掉D5后）训练二分类器（白血病vs非白血病）；Stage 2用69,400张白血病图像（去掉D5 + D2中正常部分）条件性训练ALL vs AML分类器；两个阶段独立训练/评估，互不污染。
- **三种编码器**：DinoBloom-B（ViT-B/14，85.7M参数，单细胞血液学预训练，768维嵌入）；BiomedCLIP（ViT-B/16，86.2M参数，生物医学图文预训练，512维嵌入）；CLIP（ViT-B/32，87.8M参数，自然图像预训练，512维嵌入）。三者仅用vision tower。
- **两种适配策略**：
  - **Linear Probe**：冻结编码器，仅训练顶部逻辑回归分类器，使用类别加权处理不平衡；嵌入一次性提取后离线计算。
  - **LoRA**：在注意力投影层注入低秩更新矩阵，冻结预训练权重；超参：rank r=8，scaling=16，dropout=0.05，AdamW lr=1e-4，batch_size=128，3个epoch混合精度训练；DinoBloom可训练参数296,450，BiomedCLIP 295,938，CLIP 443,394。
- **RAC模块**：对查询图像嵌入，用L2归一化嵌入库做精确余弦相似度搜索（top-k=20），得到检索分布 $p_{\text{ret}}$，与分类分布 $p_{\text{probe}}$ 在log空间融合：$\text{logits}_{\text{final}} = \log p_{\text{probe}} + \alpha \log p_{\text{ret}}$；线性探针配置下α从{0, 0.25, 0.5, 0.75, 1}网格搜索选优，LoRA配置固定α=0.5。
- **评估协议**：Stage 1 held-out测试集=ALL-IDB2（260张，130 ALL + 130 normal）；Stage 2 held-out测试集=ALL-IDB2（130 ALL）+ AML-Cytomorphology（3,294 AML）= 3,424张；控制实验在同域内做80/20随机分层分割评估以诊断捷径学习。指标为Accuracy、Macro-Recall、Macro-F1。

## 实验与结果
- **Stage 1（held-out ALL-IDB2, n=260）**：
  - **最佳结果**：DinoBloom + RAC（linear probe），Accuracy=0.9423，Recall=0.9423，F1=0.9423。
  - DinoBloom linear probe：0.8923；BiomedCLIP linear probe：0.6962；CLIP linear probe：0.6346。
  - **LoRA效应**：CLIP经LoRA后Accuracy达0.9115（较linear probe提升+0.2769），超越DinoBloom frozen（0.8923），Gap从0.2577缩小至0.0193。
  - **RAC效应**：对DinoBloom提升最大（linear probe +0.0500）；对CLIP/BiomedCLIP的linear probe配置反而下降（-0.0769/-0.0039），说明嵌入空间需编码细胞形态学特征时RAC才有效。
- **Stage 2（held-out, n=3,424）**：
  - **性能崩溃**：所有模型在held-out测试上All Recall≈0.0–0.023，几乎全部预测为AML类；DinoBloom/BiomedCLIP Accuracy≈0.962（因AML占主导）。
  - **控制实验揭示捷径学习**：同域80/20分割下，DinoBloom ALL Recall=0.9988，BiomedCLIP=0.9971，CLIP=1.0000——同一模型从"完美"到"崩溃"，证明训练中学到了数据集背景伪影而非细胞形态。
  - RAC诊断：查询ALL图像时，DinoBloom嵌入空间中96.5%的top-20近邻为AML，仅3.5%为ALL，确认编码器将ALL细胞置于AML侧。
- **关键结论**：领域专用预训练的0.2577准确率优势可通过LoRA（+0.2769）基本补偿；最佳结果（0.9423）来自专用编码器+RAC的组合；Stage 2亚型分类在跨域场景下仍存在根本性挑战。

## 相关工作脉络
1. **Ben Rabah & Serag（2026）**：量化基础模型预训练专业化对数字病理的影响，但白血病因类别不一致被排除在跨数据集评估之外，仅做单数据集内部划分；本文完成了其未解决的标签规范化并首次实现白血病held-out跨域评估。
2. **Wang等ALSNet（2026）**：在三个平台180,928张单细胞图像上训练，报告了较强的held-out泛化；但聚焦19类细胞形态分类（聚合并推论至病例级别），而非直接细胞级ALL/AML分型，也未探索LoRA/RAC。
3. **Patel等（2024, 2024）**：评估CLIP/BiomedCLIP在白血病检测/正常vs恶性分类中的few-shot表现，但无编码器适配，也未量化域偏移影响。
4. **Naing等（2023）**：基于ResNet34的深度度量学习检索系统用于AML分期分类，与本文RAC思想类似但嵌入空间从头训练、单数据集80/20评估，无跨域诊断。
5. **Ng等（2026）**：DinoBloom+LoRA+检索推理的层级集成WBC分类管线，但仅用单一编码器无法评估预训练专业化贡献，且held-out测试来自训练域内。
6. **Dasdelen等（2026）cAItomorph**：DinoBloom用于血液恶性肿瘤分类，无适配或检索增强，外部评估仅限AML类别。

## 局限性与未来方向
- **Stage 1 held-out集合规模小**：仅260张图像，少量细胞的误分类即可显著影响准确率数字。
- **预训练数据污染**：DinoBloom预训练语料包含D2/D3/D4三个数据集，使得AML评估对专用编码器并非完全unseen域。
- **Stage 2根本性数据缺陷**：当前公开单细胞数据集中，每个亚型仅来自单一图像采集源，导致编码器可依赖背景伪影而非细胞形态；需更多异构来源数据或跨亚型同源采集数据。
- **未来方向**：（1）收集更多ALL和AML的跨域数据源；（2）对所有数据集统一背景（如统一合成背景），迫使编码器学习细胞形态学特征；（3）探索更鲁棒的域泛化方法。

## 研究启发与可借鉴点
1. **LoRA可有效缩小专用vs通用编码器差距**：CLIP（通用）经LoRA后超越DinoBloom（专用）frozen版本，证明参数高效微调是昂贵的领域专用预训练的可行替代方案，值得在其他医学影像分类任务中验证。
2. **Held-out跨域评估协议作为诊断工具的价值**：同域80/20 vs held-out的对比实验清晰揭示了模型依赖背景捷径的现象，这一评估范式可迁移至任何医学图像分类研究中，用于检验模型是否学到真正的形态学/病理学特征。
3. **RAC的有效性取决于嵌入空间的语义质量**：当编码器编码了目标领域的形态学模式时RAC有效（DinoBloom），否则引入噪声（CLIP linear probe）；这提示RAC应与编码器适配联合设计，而非简单叠加。
4. **标签规范化策略对跨数据集训练至关重要**：本文统一排除platelets、erythroblasts、smudge cells和不一致标注的immature granulocytes，只保留明确的有核白细胞，该方法论可直接借鉴到其他多源医学图像整合任务。
5. **两阶段级联设计的可扩展性**：将复杂分类任务拆分为"检测→分型"两阶段，Stage 1可利用更多样本（122K vs 69K），该设计思路适用于其他需要先做存在性判断再做细粒度分类的临床任务。

## 关键术语表
- **Vision Foundation Models (VFMs)**：在大规模图像数据集上预训练、可迁移至下游任务的视觉深度学习模型（如DINOv2、CLIP系列）。
- **Low-Rank Adaptation (LoRA)**：一种参数高效微调方法，冻结预训练权重并通过低秩矩阵近似权重更新，大幅减少可训练参数量。
- **Retrieval-Augmented Classification (RAC)**：在推理时从嵌入库中检索top-k最相似样本，以相似度加权投票方式增强分类决策的模块。
- **Domain Shift**：训练域与测试域在数据采集设备、染色、光照或协议上的差异导致的分布偏移。
- **Linear Probing**：冻结预训练编码器，仅在提取的嵌入上训练轻量线性分类器以评估表征质量的评估/微调方法。
- **Macro-averaged Recall/F1**：对每个类别单独计算指标后取算术平均，平等对待各类别，适用于类别不平衡场景。
- **Held-out Dataset Evaluation**：在完全未被训练使用的独立数据集上评估模型，用于严格检验跨域泛化能力。
- **Cytomorphology**：细胞形态学，指通过细胞形态特征（胞质、核仁、核膜等）进行疾病诊断的方法。

## 可复现要素
- **数据集**：五个公开数据集（C-NMC 2019、AML-Cytomorphology、Peripheral Blood Cell Dataset、AML-Cytomorphology MLL Helmholtz、ALL-IDB2），均为公开可用；作者已对数据进行清洗和标签规范化。
- **代码/权重**：编码器权重来自官方公开发布（DinoBloom-B从Hugging Face MarrLab仓库加载，CLIP从OpenAI checkpoint，BiomedCLIP从openclip）；Python实现环境：PyTorch 2.5.1 + CUDA 12.1, transformers 5.9.0, PEFT 0.20.0, open_clip 3.3.0等（论文未明确声明整体代码开源仓库）。
- **关键超参**：LoRA rank=8, scaling=16, dropout=0.05, lr=1e-4, batch_size=128, 3 epochs；RAC k=20；线性探针α从{0, 0.25, 0.5, 0.75, 1}网格搜索（LoRA配置固定α=0.5）；图像resize至224×224。
- **硬件**：Intel Core i9-13900HX + RTX 4090 GPU + 32GB RAM。
