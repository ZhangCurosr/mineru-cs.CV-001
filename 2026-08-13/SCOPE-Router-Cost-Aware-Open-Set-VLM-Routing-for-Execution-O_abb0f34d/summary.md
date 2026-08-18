---
title: "SCOPE-Router-Cost-Aware-Open-Set-VLM-Routing-for-Execution-O"
source: https://arxiv.org/pdf/2608.12127v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 12:39:38"
field: "多模态模型路由与部署优化"
keywords: ["VLM routing", "model routing", "cost-aware", "open-set", "execution-oriented", "hybrid calibration", "contrastive regularization"]
innovations: ["提出混合校准策略（随机/诊断/多样性）在有限预算下最大化模型Profile区分度", "CRM per-pair独立BCE消除多正样本稀释并将成本编码为连续目标", "RCCR路由一致性对比正则化使相似路由偏好的查询在空间拉近以泛化跨样本"]
benchmarks: ["VLM-ExecRouterBench", "VL-RouterBench", "MMR-Bench"]
---

# 论文速读：SCOPE-Router: Cost-Aware Open-Set VLM Routing for Execution-Oriented Tasks

## 一句话总结

本文针对视觉语言模型（VLM）路由中的三个核心缺陷——执行任务覆盖缺失、开放集能力不足、训练目标与成本脱耦——提出了**VLM-ExecRouterBench**（首个面向执行任务的 VLM 路由评测基准）、**SCOPE-Router**（基于混合校准的开放集双塔路由器）和 **CRM+RCCR**（架构无关的成本感知训练目标），在三个基准上均取得最佳 Rank Score，开放集 OOD 场景下领先第二名 6.75 分。

---

## 研究问题与动机

1. **执行任务路由缺失评测**：现有 VLM 路由基准仅覆盖传统 VQA（视觉推理、OCR、图表理解），而 VLM 在实际部署中越来越多地用于代码生成、工具调用 Agent、多步网络检索等**需要主动执行的流水线**，这些任务的模型能力差异和互补性更强，但尚无路由基准涵盖。

2. **开放集能力薄弱**：新模型不断涌现，但传统路由方法将模型视为固定类别标签，新增模型必须重新训练；已有开放集方法（UniRoute、ICL-Router）缺乏系统性的校准数据选择策略，在有限校准预算下面临模型 profile 区分度低的问题。

3. **训练目标与成本脱耦**：RouterDC 的对比损失仅按正确/错误划分正负样本，忽略成本差异；VL-RouterBench 的 softmax CE 损失在多正样本时稀释梯度；UniRoute 仅在推理阶段作为后处理引入成本，训练过程本身对成本无感知。

4. **VLM 多正样本稀释**：在执行任务中，多个模型可能同时答对同一样本，softmax 归一化导致多正样本信号被稀释，无法表达"在多个正确答案中选择最便宜的"这一核心路由偏好。

---

## 核心贡献（创新点）

1. **VLM-ExecRouterBench**：首个面向执行任务的 VLM 路由基准，覆盖 Code、Agentic、Search 三个领域，11 个候选模型（定价跨度近两个数量级），34k 样本，采用统一的 Routing Input / Execution Context / Verification Rule 三元组结构。*与 VL-RouterBench 的本质区别：从被动问答扩展到主动执行，引入代码单元测试验证、工具调用 Agent 回路和受限语料检索验证。*

2. **SCOPE-Router 开放集双塔路由器**：将开放集路由建模为查询到行为 Profile 的匹配问题，每个模型通过一次性校准（1024 样本）生成 Query-Aware Profile 即可加入路由，无需重新训练。*与 UniRoute/ICL-Router 的本质区别：提出混合校准策略（随机 50%/诊断 30%/多样性 20%），在有限预算下最大化 Profile 区分度；Profile 融合行为统计与语义方向，而非仅用公共信号或聚类。*

3. **CRM+RCCR 成本感知训练目标**：CRM 用 per-pair 独立 sigmoid BCE 替代 softmax CE，将成本偏好编码为连续相关性目标，消除多正样本稀释；RCCR 对具有相似路由偏好的查询施加对比正则化，使它们在路由空间中更近。*与 RouterDC/UniRoute 的本质区别：两个组件互补——CRM 的去归一化评分是 RCCR 分布正则化的前提（RCCR 加在 softmax 上反而降低性能），且目标纯架构无关，可直接替换任何训练路由器的损失函数。*

---

## 方法详解

### 问题形式化

给定查询集合 $\{x_i\}_{i=1}^N$ 和候选 VLM 池 $\mathcal{M} = \{m_1, \dots, m_K\}$，对每个模型执行每个样本，得到：
- **正确率矩阵** $Y \in \{0,1\}^{N \times K}$（$Y_{i,m}=1$ 当模型 $m$ 答对样本 $i$）
- **成本矩阵** $C \in \mathbb{R}_+^{N \times K}$（基于 token 用量 × 百万 token 单价）

路由学习打分函数 $s_\theta(x, m)$，推理时选择 $\hat{m}_i = \arg\max_m s_\theta(x_i, m)$。评估主指标 **Rank Score**：

$$S_\beta = \frac{(1+\beta) \cdot A \cdot \tilde{C}}{\beta \cdot A + \tilde{C}}, \quad \beta = 0.1$$

其中 $A$ 为路由准确率，$\tilde{C}$ 为对数归一化成本分。

### 数据生成：统一三元组结构

每个样本拆分为：
- **Routing Input**：路由器可见的轻量输入（文本 + 可选图像）
- **Execution Context**：送给候选 VLM 的完整上下文（含工具 schema、检索语料等）
- **Verification Rule**：任务专属验证逻辑（代码用单元测试、Agentic 用精确/选择匹配、Search 用答案匹配）

**三领域设计**：
- **Code**：MBPP（入门 Python）+ BigCodeBench（函数调用）+ APPS（竞赛题）+ LiveCodeBench（近期竞赛），用可执行测试验证
- **Agentic**：MMMU、DocVQA、ChartQA、AI2D、OCRBench、MathVista、RealWorldQA，通过轻量多模态 Agent 回路执行（支持图像检查、裁剪/缩放、OCR 工具调用）
- **Search**：BrowseComp-Plus（固定语料深度检索），使用 Qwen3-Embedding-8B 密集检索器，每轮返回 top-5 passage

### 混合校准策略（Hybrid Calibration）

校准集 $|S_{\text{calib}}| = 1024$，按 50%/30%/20% 依次分配：

1. **随机采样（50%）**：均匀覆盖训练分布
2. **诊断采样（30%）**：最大化判别力，每个候选样本得分：

$$d_i = 0.7 \cdot 4p_i(1-p_i) + 0.3 \cdot \text{cost\_spread}_i^{\text{norm}}$$

其中 $p_i = \frac{1}{K}\sum_m Y_{i,m}$ 为答对比例（ disagreement 项在 $p_i=0.5$ 时最大），cost spread 为答对模型间的成本极差。按难度桶（hard 20%/medium 60%/easy 20%）分层采样。

3. **多样性采样（20%）**：在冻结的 query embedding 空间（BGE-M3 + DINOv2-large'，concat-L2 norm）上执行 MiniBatchKMeans，选各簇质心最近样本。

### 模型行为 Profile 构建

对每个模型 $m$，构建 Query-Aware Profile $\mathbf{p}_m \in \mathbb{R}^{3S+8+3D}$：

**行为向量**（3S 维）：
$$\mathbf{p}_m^{\text{behav}} = [\mathbf{y}_m;\; \tilde{\mathbf{c}}_m;\; \mathbf{y}_m \odot (1-\tilde{\mathbf{c}}_m);\; \mathbf{s}_m]$$
- $\mathbf{y}_m$：逐样本正确率（$S$ 维）
- $\tilde{\mathbf{c}}_m$：归一化成本（$S$ 维）
- $\mathbf{y}_m \odot (1-\tilde{\mathbf{c}}_m)$：**性价比向量**（高权重给答对且便宜的样本）
- $\mathbf{s}_m$：8 维摘要统计（准确率、平均成本、正确/错误子集平均成本等）

**语义向量**（3D 维）：
$$\mathbf{p}_m^{\text{sem}} = [\bar{\mathbf{e}}_m^+;\; \bar{\mathbf{e}}_m^-;\; \bar{\mathbf{e}}_m^v]$$
- $\bar{\mathbf{e}}_m^+ = \frac{\sum_i Y_{i,m}\hat{\mathbf{e}}_i}{\sum_i Y_{i,m}}$：模型擅长的语义方向
- $\bar{\mathbf{e}}_m^- = \frac{\sum_i (1-Y_{i,m})\hat{\mathbf{e}}_i}{\sum_i (1-Y_{i,m})}$：模型不擅长的语义方向
- $\bar{\mathbf{e}}_m^v$：高效语义方向（性价比加权平均）

### 双塔路由架构

- **Query Encoder**：BGE-M3（文本）+ DINOv2-large'（视觉），normalize-concat → QueryMLP（2 层，hidden 128，dropout 0.5）→ L2 归一化 → $\mathbf{q}_i \in \mathbb{R}^{64}$
- **Profile Encoder**：ProfileMLP（同上结构）→ $\hat{\mathbf{p}}_m \in \mathbb{R}^{64}$
- **匹配分数**：$s_{i,m} = \mathbf{q}_i^\top \hat{\mathbf{p}}_m / \tau$

### CRM+RCCR 训练目标

**Relevance Target**：
$$R_{i,m} = \mathbb{1}[Y_{i,m}=1] \cdot \exp(-\lambda \cdot \alpha \cdot (C_{i,m} - C_i^{\min}))$$
- 最便宜的正确模型：$R=1$；更贵正确模型：$R \in (0,1)$；错误模型：$R=0$
- 全失败样本：$R_{i,:}=\mathbf{0}$（仅用于 CRM 全负样本，排除 RCCR）

**CRM Loss**（per-pair 独立 sigmoid BCE，替代 softmax CE）：
$$\mathcal{L}_{\text{CRM}} = \frac{1}{BK}\sum_i\sum_m \text{BCE}(\sigma(s_{i,m}), R_{i,m})$$
- 关键创新：每个 pair 梯度独立，多正样本不再被归一化稀释

**RCCR Loss**（路由一致性对比正则化）：
定义 $\tilde{\mathbf{r}}_i = R_{i,:}/\|R_{i,:}\|_1$，样本间相似度 $w_{ij} = \tilde{\mathbf{r}}_i^\top \tilde{\mathbf{r}}_j$，行归一化为 $\hat{w}_{ij}$：

$$\mathcal{L}_{\text{RCCR}} = -\frac{1}{|\hat{\mathcal{V}}|}\sum_{i\in\hat{\mathcal{V}}}\sum_{j\in\mathcal{U}}\hat{w}_{ij}\log\frac{\exp(\mathbf{q}_i^\top\mathbf{q}_j/\tau_s)}{\sum_{\ell\in\mathcal{U},\ell\neq i}\exp(\mathbf{q}_i^\top\mathbf{q}_\ell/\tau_s)}$$

**总损失**：$\mathcal{L} = \mathcal{L}_{\text{CRM}} + 0.1 \cdot \mathcal{L}_{\text{RCCR}}$

---

## 实验与结果

### 基准与设置

- **VLM-ExecRouterBench**（本文新基准）：34k 样本，11 模型，3 领域
- **VL-RouterBench**（已有 VQA 基准）
- **MMR-Bench**（多模态推理基准）
- 基线：4 类共 20+ 方法（训练免基线、特征级路由器、端到端路由器）
- $\beta=0.1$（偏重准确率），$\lambda$ 在验证集上调优

### 主要结果（Rank Score）

| 基准 | SCOPE-Router | 次优方法 | 提升幅度 |
|------|-------------|---------|---------|
| VLM-ExecRouterBench | **80.94 ± 1.22** | CosineCls 79.55 | +1.39 |
| VL-RouterBench | **76.18 ± 1.44** | RouterDC 74.59 | +1.59 |
| MMR-Bench | **61.23 ± 0.75** | UniRoute-KM 59.72 | +1.51 |

- **OOD 泛化**（5 数据集 held-out）：SCOPE-Router 88.14 vs K-means 86.30，**领先 1.84 分**，成本仅为后者的 21.8%
- **开放集 doubly OOD**（5/11 模型 held-out + held-out 数据集）：领先 UniRoute-KM **+6.75 分**，准确率 +7.26pp
- **成本效率**：VS Strongest 基线，准确率仅降 5.21pp，**成本降低 85%**（VLM-ExecRouterBench）；MMR-Bench 上仅用 Strongest **0.8%** 的成本达到可比精度

### 跨架构迁移性

CRM+RCCR 直接替换 RouterDC、ZOOTER、CosineCls、VLC 的原始损失，在 VL-RouterBench 和 VLM-ExecRouterBench 上均提升 Rank Score **+1.25 ~ +6.21 分**，验证了目标的架构无关性。

### 消融要点

- **CRM vs softmax**：CRM 单独提升 +1.0 分，验证多正样本去稀释有效
- **RCCR 的互补性**：RCCR 加在 softmax 上反而降分（74.51→73.41），仅在 CRM 基础上提升 +0.68 分，说明 CRM 的去归一化框架是 RCCR 的前提
- **混合校准**：全组合 76.18 > R+Div 76.00 > Div only 75.69，三者互补
- **编码器鲁棒性**：25 种 BGE/DINO 组合 Span < 0.8 分
- **Query Embedder 冻结 vs 微调**：冻结 76.18 vs 微调 75.68，冻结更优且训练成本更低

---

## 相关工作脉络

1. **RouterBench / LLM-RouterBench / RouterArena**：将 LLM 路由形式化为成本-质量优化，但仅覆盖文本路由，未涉及视觉模态和执行任务。*本文定位：首次将路由评测扩展到 VLM 执行任务。*

2. **VL-RouterBench**：首个 VLM 路由基准（14 数据集、17 模型、统一 Rank Score），但局限于传统 VQA。*本文定位：补充执行任务维度，指出 VQA 与执行任务在模型互补性上的本质差异。*

3. **RouterDC**：双对比损失路由器，按正确率硬性划分正负样本，忽略成本和多正样本信号。*本文定位：CRM 用 per-pair 连续目标替代 hard-label 对比，消除多正稀释。*

4. **UniRoute**：开放集路由，随机划分验证数据构建模型表示，成本仅作为推理后处理。*本文定位：混合校准策略系统性优化校准集选择，成本直接编码进训练目标。*

5. **MMR-Bench**：预算感知多模态推理评测。*本文定位：在 MMR-Bench 上同样取得最佳 Rank Score，证明方法跨基准泛化。*

6. **Routeprofile**：基于 GNN 的冷启动 LLM 路由。*本文定位：profile 基于首轮校准推理而非图结构公共信号，保留样本级细粒度。*

---

## 局限性与未来方向

1. **新模型仍需校准**：开放集虽然免重新训练，但新模型仍需运行校准集生成 profile，性能取决于校准样本能否充分暴露新模型的能力和 failure modes。
2. **单步决策限制**：当前路由器对每个样本做出单一任务级决策，无法适应 Agentic/Search 长轨迹中**不同步骤需要不同模型**的场景（如早期步骤需 OCR、后期步骤需检索综合）。
3. **校准预算固定**：1024 样本的校准预算在当前设置下最优，但未研究更小预算下的 profile 质量退化边界。
4. **未来方向**：（a）从紧凑校准集构建更鲁棒的模型 profile；（b）动态路由策略，在长轨迹中间步骤可 Revision 模型选择。

---

## 研究启发与可借鉴点

1. **混合校准策略的可迁移价值**：随机（分布覆盖）+ 诊断（分歧×成本离散）+ 多样性（嵌入空间均匀）的三阶段分配，可作为任意模型 profiling 任务的通用采样范式，尤其适用于"有限预算下最大化判别力"的场景。

2. **per-pair 独立 BCE 替代 softmax CE 的设计思想**：在多正样本普遍存在的推荐/排序/路由任务中，soft target + per-pair sigmoid 是一个值得推广的损失设计原则，避免归一化稀释。

3. **路由一致性对比正则化（RCCR）的跨样本泛化思路**：将对"相似路由偏好"的查询在 embedding 空间拉近，是一种轻量级的样本结构正则化，可适配到任何基于双塔匹配的排序系统中。

4. **与本团队的结合机会**：如果团队有 LLM/Agent 路由需求，可将 SCOPE-Router 的混合校准 + CRM+RCCR 直接迁移到纯文本 LLM 路由场景（RouterDC、LLM-RouterBench 的基线），预期在多正样本密集的编程/检索任务中获得更大收益。

5. **执行任务的统一三元组接口设计**：Routing Input / Execution Context / Verification Rule 的分离设计，使路由器输入轻量化的同时保证模型评估的完整性，可作为执行型 VLM 评测的标准范式。

---

## 关键术语表

- **VLM-ExecRouterBench**：首个面向代码生成、Agent 工具调用、Web 检索三类执行任务的 VLM 路由评测基准，11 模型 34k 样本。
- **SCOPE-Router**：Selective Calibration for Open-set Profile-Enhanced Routing，基于混合校准双塔匹配的开放集 VLM 路由器。
- **CRM（Cost-aware Relevance Matching）**：成本感知相关性匹配，用 per-pair 独立 sigmoid BCE 将成本偏好编码为连续目标，消除多正样本稀释。
- **RCCR（Routing-Consistency Contrastive Regularization）**：路由一致性对比正则化，对路由偏好相似的查询在 embedding 空间施加对比拉近。
- **Query-Aware Profile**：模型行为 Profile，融合逐样本正确率、成本、性价比向量和语义方向，表征模型"能解什么、解得如何、花多少钱"。
- **混合校准（Hybrid Calibration）**：随机/诊断/多样性三类采样按 50%/30%/20% 分配的校准集构建策略，最大化有限预算下的 profile 区分度。
- **Rank Score**：路由主评估指标，准确率与对数归一化成本的调和均值（$\beta=0.1$ 偏重准确率）。
- **多正样本稀释（Multi-positive Dilution）**：softmax CE 在多模型同时答对时归一化稀释各正确模型信号的训练缺陷。

---

## 可复现要素

- **数据集**：VLM-ExecRouterBench 已公开（HuggingFace: `Kirito-Lab/VLM-ExecRouterBench`），包含 33,966 样本的 JSONL 记录及对齐的正确率/成本矩阵。
- **代码**：已开源（GitHub: `yutao1024/SCOPE-Router`）。
- **关键超参**：校准集 1024 样本（50%/30%/20% 混合）；QueryMLP/ProfileMLP 2 层 hidden 128 dropout 0.5 output 64；$\lambda=10$（成本敏感度）；$\mu=0.1$（RCCR 权重）；$\beta=0.1$（Rank Score）；AdamW lr=0.001 weight_decay=0.003 batch=512；冻结 query embedder（BGE-M3 + DINOv2-large'）；early stopping patience=20 max 100 epochs。
- **编码器默认**：BGE-M3（文本）+ DINOv2-large'（视觉），D=2048。

---
