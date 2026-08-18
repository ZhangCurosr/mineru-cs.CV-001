---
title: "Retrieval-Augmented Vision Foundation Models for Robust Leukemia Cell Classification across Multiple Microscopy Datasets"
source: https://arxiv.org/pdf/2608.10657v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:44:18"
field: "医学图像分类与跨域泛化"
keywords: ["leukemia classification", "vision foundation models", "domain shift", "LoRA", "retrieval-augmented classification", "cytomorphology", "cross-dataset evaluation"]
innovations: ["双阶段分级分类管线结合跨域保留集评估揭示捷径学习问题", "系统性量化预训练域特异性对白血病分类跨域性能的影响", "证明LoRA高效适配可弥合通用与领域专用模型的性能差距"]
benchmarks: ["ALL-IDB2 (held-out)", "C-NMC 2019", "AML-Cytomorphology", "Peripheral Blood Cell Dataset", "AML-Cytomorphology MLL Helmholtz"]
---

# 论文速读：Retrieval-Augmented Vision Foundation Models for Robust Leukemia Cell Classification across Multiple Microscopy Datasets

## 一句话总结
本文构建了一个双阶段分类框架，利用三种不同预训练域的先验视觉基础模型（DinoBloom/BiomedCLIP/CLIP），结合线性探测、LoRA适配与检索增强分类（RAC）模块，在五个异构单细胞显微镜数据集上实现急性白血病检测与亚型分类，并通过保留集协议量化域偏移下的模型鲁棒性。

## 研究问题与动机
1. **域偏移导致泛化退化**：单数据集训练的模型在面对不同采集设备、染色、光照和协议时，在真实临床场景泛化能力严重下降，血液涂片图像差异尤为突出。
2. **领域专用预训练的价值未明**：针对白血病分类任务， expensive 的单细胞领域预训练 vs. 通用视觉模型 + 高效适配策略（LoRA/RAC）孰优孰劣，缺乏系统性量化分析。
3. **现有文献的评估协议缺陷**：大部分已有工作使用同一域内的随机划分进行评估，无法暴露模型对数据集特定伪影（background artifact）的依赖，导致"虚假高准确率"。
4. **检索增强在细胞学图像分类中的作用未经验证**：RAC已在自然图像长尾分类中验证，但在医学细胞形态学分类中能否有效取决于嵌入空间是否编码了细胞形态特征。

## 核心贡献（创新点）
1. **双阶段分级分类管线**：Stage 1做二分类（白血病 vs. 非白血病，训练数据122,167张），Stage 2条件性进行亚型分类（ALL vs. AML，训练数据69,400张），使Stage 1可利用仅含正常细胞的源数据集扩充训练语料，提升泛化能力。
2. **系统性预训练域影响基准**：在同一双阶段管线中对比三个参数量相近（~86M）的编码器（DinoBloom单细胞预训练 / BiomedCLIP生物医学预训练 / CLIP自然图像预训练），隔离预训练域变量，量化其对跨域泛化的贡献（线性探测下DinoBloom比CLIP高出+0.2577准确率）。
3. **成本高效适配策略的替代价值**：证明LoRA可将通用CLIP的准确率从0.6346提升至0.9115，超越未适配的DinoBloom（0.8923），且RAC+DinoBloom达到最佳0.9423，说明高效适配技术可弥补大部分领域专用预训练差距。
4. **保留集评估协议揭示"捷径学习"问题**：通过严格的跨域评估发现Stage 2亚型分类性能崩溃（ALL召回率≈0），而同样的模型在域内80/20划分可达近完美准确率，揭示模型学习的是背景伪影而非细胞形态特征。

## 方法详解

**双阶段管线架构**：三个编码器（DinoBloom-B / BiomedCLIP / CLIP，均为ViT-Base量级）作为骨干，每个阶段独立设置线性分类头。Stage 1二分类对所有122,427张图；Stage 2子类型分类仅对122,427中的72,824张白血病细胞。

**三种编码器对比**：
- **DinoBloom-B**：DINOv2变体，ViT-B/14，768维嵌入，85.7M参数，预训练于13个单细胞血细胞数据集
- **BiomedCLIP**：ViT-B/16，512维嵌入，86.2M参数，预训练于PubMed Central的百万级医学图文对
- **CLIP**：ViT-B/32，512维嵌入，87.8M参数，预训练于自然图像-文本对

**两种适配策略**：
- **线性探测（Linear Probe）**：冻结编码器，仅在预计算嵌入上训练逻辑回归分类器，使用类别权重处理不平衡，无数据增强
- **LoRA**：在注意力投影层注入低秩矩阵，冻结预训练权重。配置：rank r=8, scale=16, dropout=0.05，可训练参数约296K（DinoBloom），AdamW优化器，lr=1e-4，batch=128，3 epoch混合精度训练，轻微增强（水平/垂直翻转+轻度色彩抖动）

**检索增强分类（RAC）模块**：
- 在编码器嵌入空间中，对查询图像计算L2归一化余弦相似度（等价于点积），检索Top-k=20个最近邻
- 按相似度加权投票生成检索分布 $p_{\text{ret}}$
- 与线性分类器分布 $p_{\text{probe}}$ 在log空间融合：$\text{logits}_{\text{final}} = \log p_{\text{probe}} + \alpha \log p_{\text{ret}}$
- 线性探测下α在{0, 0.25, 0.5, 0.75, 1}网格搜索；LoRA配置下α固定为0.5（计算预算约束）

**域偏移缓解预处理**：
- C-NMC 2019数据集的黑色背景被替换为随机选择的健康外周血涂片背景（主要含RBC），配合旋转、缩放及Reinhard颜色归一化

**标签对齐**：五个数据集的异构标签统一为三类（Normal / ALL / AML），仅保留明确的有核白细胞，排除血小板、红母细胞、smudge细胞及成熟度不一致的中间粒系细胞；AML遗传亚型全部折叠为单一AML类。

## 实验与结果

**数据集**：五个公开单细胞数据集（D1 C-NMC 2019 / D2 AML-Cytomorphology / D3 Peripheral Blood Cell / D4 AML-Cytomorphology MLL Helmholtz / D5 ALL-IDB2），总计122,427张可用图像。

**评估协议**：
- Stage 1保留集：ALL-IDB2（D5，n=260，130 ALL + 130 Normal）
- Stage 2保留集：ALL-IDB2（D5 ALL类）+ AML-Cytomorphology（D2 AML类，n=3,424，130 ALL + 3,294 AML）
- 对照组：Stage 2域内随机80/20划分

**Stage 1关键结果（保留集，n=260）**：

| 模型 | 配置 | 准确率 | 提升 |
|------|------|--------|------|
| DinoBloom | Linear Probe | 0.8923 | — |
| DinoBloom | LP+RAC | **0.9423** | +0.0500 |
| DinoBloom | LoRA | 0.9308 | +0.0385 |
| DinoBloom | LoRA+RAC | 0.9385 | +0.0077 |
| CLIP | LoRA | 0.9115 | +0.2769（vs LP）|
| CLIP | Linear Probe | 0.6346 | — |

- **最佳结果**：DinoBloom + RAC（0.9423准确率）
- LoRA使CLIP超越未适配DinoBloom（0.9115 > 0.8923）
- RAC对CLIP线性探测有负面影响（-0.0769），适配后缓解（-0.0192）

**Stage 2关键发现（保留集，n=3,424）**：
- 所有配置下ALL召回率≈0（几乎全预测为AML），macro- recall≈0.50，说明亚型分类在跨域条件下崩溃
- 但域内80/20划分中DinoBloom/BiomedCLIP召回率达0.99+，CLIP达1.00——揭示模型依赖背景伪影而非细胞形态
- RAC检索分析：对ALL查询图像，96.5%的Top-20邻居为AML，仅3.5%为ALL，说明嵌入空间中ALL细胞被错误地靠近AML区域

## 相关工作脉络

1. **Ben Rabah & Serag (2026)**：量化预训练专业化对数字病理模型的影响（7通用+3领域专用），但将白血病分类因标签不一致而排除在跨数据集协议之外，仅在单个数据集内评估——本文通过标签归一化填补了这一空白。
2. **Wang et al. (ALSNet, 2026)**：跨平台训练180,928张单细胞图像进行形态学分类，报告了良好的保留集性能，但聚焦于19类细胞形态→病例级预测的自顶向下路径，而非白血病检测/分型的端到端分类；本文引入LoRA和RAC适配，提供了对比视角。
3. **Ng et al. (2026)**：使用DinoBloom+LoRA+概念类似RAC的检索推理步骤用于WBC分类，但仅用单一编码器、基于WBCBench的域内hold-out测试，且不涉及白血病诊断——本文扩展至多编码器对比与严格跨域评估。
4. **Naing et al. (2023)**：使用ResNet34+度量学习的图像检索与分类系统用于AML分期，检索机制与本文RAC相似，但嵌入空间从零训练，未使用预训练基础模型，且仅在单数据集上评估——本文展示了在预训练嵌入空间上做检索的有效性条件。
5. **Patel et al. (2024, 2024)**：探索CLIP/BiomedCLIP/PLIP等视觉语言模型用于白血病分类的提示工程与少样本学习，但缺少编码器适配策略的系统性评估——本文补充了LoRA与RAC对VLMs的适配效果量化。
6. **Dasdelen et al. (2026) cAItomorph**：使用DinoBloom对周边血涂片进行分类，但无适配或检索增强，且外部评估仅限AML类——本文系统展示了适配策略在相同编码器上的增益。

## 局限性与未来方向

1. **Stage 1保留集规模小**：ALL-IDB2仅含260张图像，微小准确率差异对应极少样本数，统计置信度受限。
2. **领域专用预训练的污染风险**：DinoBloom的预训练语料包含三个训练数据集（D2/D3/D4），故其AML评估不完全是"完全未见域"。
3. **Stage 2亚型分类跨域崩溃**：由于训练阶段ALL和AML分别来自不同单一数据源，模型依赖背景伪影而非形态特征；解决需同时拥有ALL和AML的双源数据或统一背景归一化。
4. **RAC融合权重α在LoRA配置下固定为0.5**：与线性探测配置下的网格搜索存在不一致性，可能影响结果可比性。
5. **未来方向**：收集同一成像平台下的ALL+AML配对数据作为保留集；对所有数据集应用统一的合成背景归一化；探索更多数据源以强制编码器学习细胞形态特征。

## 研究启发与可借鉴点

1. **保留集跨域评估作为诊断工具**：将域内随机划分与严格保留集评估并列对比，可快速识别模型是否学习了数据源伪影而非任务相关特征——这一双协议设计可直接迁移到其他医学图像分类领域。
2. **LoRA作为低成本预训练替代**：通用CLIP经LoRA适配后可匹敌甚至超越领域专用DinoBloom（线性探测），表明在资源受限场景下，通用基础模型+LoRA是一条可行的替代路径，值得在其他医学子领域（如病理、放射）验证。
3. **RAC有效性的前提条件**：RAC模块仅在编码器嵌入空间已编码任务相关形态特征时才正向有效；若嵌入空间受域伪影主导，RAC反而引入噪声——这为后续工作中"是否加入RAC"提供了明确的决策依据。
4. **背景归一化作为域偏移缓解手段**：将C-NMC的黑色背景替换为真实血涂片背景并做随机变换，是一种简单有效的域对齐预处理，可借鉴到任何多源显微镜图像融合任务。
5. **双阶段解耦设计的扩展价值**：将复杂分类任务分解为"有无疾病→具体分型"两级管线，可有效利用仅有部分标签的数据源，对于标签不完整的多疾病诊断框架具有参考意义。

## 关键术语表

**Vision Foundation Model (VFM)**：在大规模图像数据集上预训练的深度学习模型，可学习通用视觉表示并适配到下游任务，如DinoBloom、CLIP等。

**Domain Shift（域偏移）**：训练数据与测试/部署数据的分布差异，可由采集设备、染色、光照或协议不同引起，是医学图像泛化的核心挑战。

**Low-Rank Adaptation (LoRA)**：一种参数高效微调方法，冻结预训练权重并在注意力层注入低秩矩阵（B·A）进行增量更新，大幅降低可训练参数量。

**Retrieval-Augmented Classification (RAC)**：在推理时从嵌入库中检索Top-k最近邻并按相似度加权投票，将检索结果与分类器输出在log空间融合，辅助决策。

**Linear Probing（线性探测）**：冻结预训练编码器，仅在其提取的嵌入上训练轻量线性分类器，用于评估编码器本身表征质量而不受微调干扰。

**Held-out Dataset Evaluation（保留集评估）**：将某个完整数据集从训练集中排除，在模型完全未见过的域上进行测试，用于严格评估跨域泛化能力。

**Cytomorphology（细胞形态学）**：研究细胞形态特征的科学，在白血病诊断中指通过显微镜观察白细胞的细胞质、核仁、核膜等形态进行分型。

**Parameter-Efficient Fine-Tuning (PEFT)**：仅更新模型少量参数以适应新任务/域的微调范式，避免全参数微调带来的高计算成本和灾难性遗忘风险。

## 可复现要素

- **数据集**：五个公开数据集（C-NMC 2019 / AML-Cytomorphology / Peripheral Blood Cell Dataset / AML-Cytomorphology MLL Helmholtz / ALL-IDB2），均为已发表公共数据，但本文构建了合并后的统一数据集
- **代码**：论文未提及代码开源声明
- **模型权重**：DinoBloom-B（HuggingFace MarrLab仓库）、CLIP（openai/clip-vit-base-patch32）、BiomedCLIP（open_clip）均可通过官方公开渠道获取
- **关键超参**：LoRA rank=8, scale=16, dropout=0.05, lr=1e-4, batch=128, epochs=3, RAC k=20, 输入分辨率224×224
- **硬件**：RTX 4090 GPU + Intel i9-13900HX CPU + 32GB RAM
- **软件栈**：Python 3.12.10, PyTorch 2.5.1 (CUDA 12.1), transformers 5.9.0, PEFT 0.20.0, open_clip 3.3.0
