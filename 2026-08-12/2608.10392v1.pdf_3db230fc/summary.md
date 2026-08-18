---
title: "Share First, Route What Remains:"
source: https://arxiv.org/pdf/2608.10392v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:45:24"
field: "高效MoE架构设计"
keywords: ["Mixture-of-Experts", "Token-Adaptive", "Shared-Residual", "Dynamic Routing", "Domain Generalization", "Efficient Inference"]
innovations: ["揭示共享与路由的依赖关系，提出先共享后路由的有序原则", "通过累积路由质量自适应确定剩余专家数量", "引入Gram正则化促进路由方向多样性"]
benchmarks: ["DomainBed", "GLUE"]
---

# 论文速读：Share First, Route What Remains: A Unified Framework for Token-Adaptive MoE Computation

## 一句话总结
论文提出 UniF-MoE 框架，通过**先提取可重用计算（共享专家）、再基于剩余需求动态路由专家数量**，将共享建模、细粒度计算与动态路由三个独立机制整合为一个有序原则，在保持性能的同时显著降低计算开销。

## 研究问题与动机
- 现有 MoE 方法通常将共享专家、细粒度专家选择和动态路由作为平行机制独立设计，忽略了它们之间的依赖关系。
- 实证观察表明：共激活的专家在特定值通道（value positions）上高度对齐；一旦提取共享响应，专家的偏好会发生显著变化；共享覆盖率越高，恢复原始输出所需的剩余专家数量越少。
- 因此，共享、细粒度计算与动态路由应当是**顺序执行**的三个决策阶段，而非相互独立的超参数调节。

## 核心贡献（创新点）
1. **揭示共享‑路由依赖性**：首次在稀疏上采样 FFN 专家内部发现可重用计算与 Token‑特定计算之间存在路由条件依赖，将共享建模、细粒度计算与动态路由统一为一个有序原则。
2. **提出 UniF‑MoE 统一框架**：通过单一 Token‑依赖预算耦合共享宽度、共享内容与剩余专家数量，实现了专家内（块级）与专家间（数量）稀疏性的协同控制。
3. **累积路由质量确定专家数量**：设计基于累积路由质量的阈值机制，使激活的剩余专家数量自适应剩余计算需求，替代固定 top‑k 路由。
4. **Gram 正则化促进路由多样性**：通过约束路由器嵌入的正交性，保持专家角色的差异化，避免共享与剩余路由坍缩。
5. **性能‑效率双重提升**：在视觉与语言基准上，UniF‑MoE 超越代表性静态/动态 MoE，同时降低激活参数、FLOPs、推理延迟与内存占用。

## 方法详解
### 1. 块级共享‑剩余分区（Blockwise Shared‑Residual Partitioning）
- 每层包含 1 个共享专家 \(E_{\text{shr}}\) 和 K 个剩余专家 \(\{E_1,\ldots,E_K\}\)，所有专家从同一稠密 FFN 初始化，中间宽度 H 划分为 B 个对齐的块，每块宽度 \(M=H/B\)。
- 每个专家的输出可表示为各块输出之和：\(E_j(\mathbf{x}) = \sum_{b=1}^{B} E_j^{(b)}(\mathbf{x})\)。
- 共享专家执行 Token 选定的块集合 \(\mathcal{T}_{\text{shr}}(\mathbf{x})\)，剩余专家仅执行互补块集合 \(\mathcal{T}_{\text{res}}(\mathbf{x})\)，两者无重叠。

### 2. Token‑自适应共享建模（Token‑Adaptive Shared Modeling）
- **共享需求得分**：扩展路由器 \(\mathbf{W}_g^{\star}=[\mathbf{W}_{\text{shr}},\mathbf{W}_g]\) 计算共享得分 \(\alpha(\mathbf{x}) = \tau + (1-2\tau)\sigma(\mathbf{x}\mathbf{W}_{\text{shr}})\)，其中 \(\tau=(B-1)/B^2\) 保证 \(\alpha\in(\tau,1-\tau)\)。
- **共享块数量**：\(b(\mathbf{x}) = \text{round}(B\alpha(\mathbf{x}))\)，确保每 Token 至少使用 1 个共享块和 1 个剩余块。
- **共享块选择**：以每块 up‑projection key 的平均值 \(\mu_b\) 作为原型，优先级 \(u_b(\mathbf{x})=\mathbf{x}\mu_b\)，选取 Top‑\(b(\mathbf{x})\) 个块构成 \(\mathcal{T}_{\text{shr}}(\mathbf{x})\)。

### 3. 累积剩余专家路由（Cumulative Residual‑Expert Routing）
- 剩余需求 \(\beta(\mathbf{x}) = 1 - \alpha(\mathbf{x})\)。
- 路由器产生亲和力分布 \(\mathbf{s}(\mathbf{x})=\text{softmax}(\mathbf{x}\mathbf{W}_g)\)，按降序排列得 \(p_1(\mathbf{x})\ge p_2(\mathbf{x})\ge\cdots\ge p_K(\mathbf{x})\)。
- 激活最小的前缀长度 \(k(\mathbf{x})\) 使得累积亲和力满足：\(\sum_{i=1}^{k(\mathbf{x})} p_i(\mathbf{x}) \ge \beta(\mathbf{x})\)。

### 4. 输出合并与计算度量
- 最终输出：\(\mathbf{y} = \alpha(\mathbf{x}) E_{\text{shr}}^{S}(\mathbf{x}) + \sum_{i=1}^{k(\mathbf{x})} p_i(\mathbf{x}) E_{q_i(\mathbf{x})}^{\mathcal{R}}(\mathbf{x})\)。
- Token 激活总块数：\(C_B(\mathbf{x}) = b(\mathbf{x}) + k(\mathbf{x})[B-b(\mathbf{x})]\)。共享块仅执行一次，剩余块按需路由，避免重复计算。

### 5. 训练目标
- 损失函数：\(\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda_{\text{div}} \mathcal{L}_{\text{div}}\)，其中 \(\mathcal{L}_{\text{div}} = \|(\mathbf{W}_g^{\star})^\top \mathbf{W}_g^{\star} - \mathbf{I}_{K+1}\|_F\) 为 Gram 正则项，鼓励嵌入正交。
- 初始化时 \(\mathbf{W}_g^{\star}\) 列为标准正交基，进一步稳定路由几何。

## 实验与结果
- **数据集**：视觉基准 DomainBed（PACS, VLCS, OfficeHome, TerraIncognita, DomainNet）；语言基准 GLUE（CoLA, MRPC, QNLI, MNLI, RTE）。
- **基线**：Dense DeiT‑S/16 / BERT‑large，静态 MoE（GMoE, EMoE, EMoE‑L），动态 MoE（DynMoE, MASS），领域泛化方法（LFME, DMDA, PC‑MoE）等。
- **主要结果**：
  - **DomainBed 平均准确率 69.5%**，全面领先（PACS 89.6%，VLCS 81.7%，OfficeHome 74.2%，DomainNet 49.4%；TerraInc. 52.6% 略低于 LFME 的 53.4%）。
  - **GLUE 平均得分 82.76%**，在五类任务上均优于所有固定 top‑k 变体及动态 MoE（CoLA 66.83%，MRPC 91.57%，QNLI 93.10%，MNLI 86.84%，RTE 75.47%）。
- **成本对比**（VLCS 上）：相对 top‑2 GMoE，激活参数减少 9.1%，FLOPs 减少 16.1%，推理时间减少 45.2%，内存占用减少 52.7%。
- **消融实验**：移除任一自适应决策（固定 α、前缀块选择、固定 top‑2 剩余专家）均导致性能下降与计算增加，验证三阶段协同的必要性。
- **超参分析**：块数 \(B=8\) 在多数 DomainBed 数据集上表现最佳；Gram 正则权重 \(\lambda_{\text{div}}=0.01\) 平衡多样性与任务损失。

## 相关工作脉络
1. **共享专家方法**（DeepSeekMoE, Union‑of‑Experts）：添加专门共享专家捕获通用知识，但未与剩余路由顺序耦合，易产生重复计算。
2. **细粒度/模块化专家**（Emergent MoE, MoSE, MoNE）：在 FFN 内部变化计算宽度，但独立于共享提取过程，未考虑剩余需求的变化。
3. **动态路由方法**（DynMoE, MASS, Alloc‑MoE）：根据 Token 难度动态调整专家数量，但未以共享覆盖率为条件，可能为已部分计算的响应分配额外专家。
4. **专家专用性增强**（Orthogonality, MP‑MoE）：通过正交约束或协方差选择减少专家重叠，但未整合共享‑剩余依赖的有序决策。
5. **稀疏上采样**（Sparse Upcycling）：从预训练稠密 FFN 复制初始化专家，为本方法的块对齐与通道可比性提供基础。
6. **统一 MoE 框架**：本文首次将共享宽度、共享内容、剩余专家数量三个决策串联为单一 Token‑依赖预算，区别于以往平行机制的设计。

## 局限性与未来方向
- **局限性**：
  - 仅在 Vision（DomainBed）和 Language（GLUE）中等规模基准验证，未在大语言模型生成任务或多模态场景测试。
  - 块划分假设专家内通道可对齐，对于非稀疏上采样（如随机初始化）的专家可能失效。
  - 共享需求得分 \(\alpha(\mathbf{x})\) 仍依赖线性路由器，难以捕获复杂语义模式。
- **未来方向**：
  - 探索通道级而非块级的细粒度共享，进一步细化计算分配。
  - 扩展至预训练语言模型（如 LLaMA‑MoE）与多模态 MoE，验证跨模态泛化能力。
  - 研究自适应块数 \(B\) 或动态分块策略，减少人工调参。
  - 结合稀疏注意力或其他条件计算机制，构建更通用的 Token‑自适应架构。

## 研究启发与可借鉴点
1. **顺序化耦合思想**：将共享提取、细粒度选择、动态路由按依赖顺序串联，可作为设计其他稀疏专家架构的通用原则。
2. **累积路由质量阈值**：以累积亲和力覆盖剩余需求来确定专家数量，逻辑简洁且无需额外网络，可移植至 DynMoE、MASS 等框架。
3. **Gram 正则化技巧**：约束路由器嵌入正交性以促进多样性，是轻量高效的正则化手段，适用于任何基于嵌入的路由器。
4. **诊断性分析范式**：共激活对齐分析、路由 rank 转变、需求 CDF 等方法为理解 MoE 内部机制提供了可复用的实验分析框架。
5. **块级共享‑剩余分区**：平衡选择粒度与计算复杂度，为跨层、跨设备（如训练/推理）的 MoE 优化提供可借鉴的模块化设计。

## 关键术语表
- **Mixture‑of‑Experts (MoE)**：稀疏激活神经网络架构，通过路由器将输入分配给少数专家子网络，以提升容量并维持计算效率。
- **Token‑Adaptive Computation**：根据每个输入 token 的语义需求动态调整计算量（如专家数量、宽度）的机制。
- **Shared Expert**：提取通用知识并跨 token 重用的专家模块，与专用剩余专家互补。
- **Cumulative Routing Mass**：按路由器亲和力降序累加专家权重，直至满足剩余计算需求的阈值方法，用于确定激活专家数。
- **Sparse Upcycling**：从预训练稠密模型复制 FFN 参数初始化稀疏专家，保留通道对齐以支持细粒度共享。
- **Gram Regularizer**：约束路由器嵌入矩阵的 Gram 矩阵接近单位矩阵，促进嵌入正交和路由方向多样性。
- **Blockwise Partitioning**：将 FFN 隐藏维度划分为多个对齐的块，实现共享与剩余计算的细粒度拆分。
- **Residual Demand**：共享计算提取后，仍需由专用专家处理的剩余计算量，决定路由专家的数量。

## 可复现要素
- **数据集**：DomainBed（含 PACS, VLCS, OfficeHome, TerraIncognita, DomainNet）与 GLUE（CoLA, MRPC, QNLI, MNLI, RTE）均为公开基准。
- **代码/权重**：代码开源地址 https://github.com/existence0420/UniF-MoE；论文未提供预训练权重，但给出完整配置与训练细节。
- **关键超参**：
  - 视觉任务：\(K=6\)（剩余专家数），\(B=8\)（块数），dropout=0.1，stochastic depth=0.1。
  - 语言任务：\(K=16\)，\(B=16\)，epoch≤10，batch size=32，learning rate ∈ \{2e‑5, 3e‑5, 5e‑5\}。
  - 正则化：\(\lambda_{\text{div}}=0.01\)，或thonormal 初始化 \(\mathbf{W}_g^{\star}\)。
