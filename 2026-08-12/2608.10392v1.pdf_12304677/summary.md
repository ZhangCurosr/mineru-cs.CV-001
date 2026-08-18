---
title: "Share First, Route What Remains:"
source: https://arxiv.org/pdf/2608.10392v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:44:24"
field: "高效大模型架构"
keywords: ["mixture of experts", "token-adaptive computation", "shared expert", "dynamic routing", "sparse upcycling", "domain generalization"]
innovations: ["提出'先共享后路由'原则，将共享建模、细粒度计算与动态路由统一为有序的单一预算分配", "基于通道对齐性分析的 block 级共享-残差分解，实现 token 自适应的共享宽度与残差专家数联合优化"]
benchmarks: ["DomainBed", "GLUE"]
---

# 论文速读：Share First, Route What Remains: A Unified Framework for Token-Adaptive MoE Computation

## 一句话总结
本文提出 **UniF-MoE**，一种统一的 Token-自适应 MoE 计算框架。其核心原则是"先共享，再路由剩余"：通过将稀疏 upcycling 的 FFN 专家分解为对齐的 key-value 通道块，先用 token 自适应方式提取可复用共享内容，再根据剩余需求动态路由专家数量，实现共享建模、细粒度计算与动态路由三个决策的有序统一。

## 研究问题与动机
1. **现有 MoE 方法的决策割裂问题**：Conventional top-k MoE 将完整专家作为计算单元，对所有 token 分配相同数量的专家，导致简单 token 冗余计算、困难 token 容量不足。
2. **共享专家、细粒度计算、动态路由三者决策相互依赖却被独立设计**：提取可复用计算后，剩余内容发生变化、最优专家选择发生变化、所需专家容量也发生变化；分别决策会导致重复共享工作或为已部分计算的响应分配专家容量。
3. **缺乏对 FFN 内部结构的可解释分析**：传统方法未深入分析稀疏 upcycled FFN 中通道对齐性与路由偏好的内在关系。

## 核心贡献（创新点）
1. **揭示共享与路由的依赖关系并提出"先共享再路由"原则**：通过对共激活专家的 value 通道对齐性进行分析，发现共激活对的 aligned 位置占比高达约 80%，且移除这些位置后仅 5.7% 的 token 保持原有 top-2 专家选择（对角线质量仅 26.9%），证明三个决策应视为一个有序过程而非独立旋钮。
2. **提出 UniF-MoE 统一框架，通过单一 token 依赖预算耦合三个自适应决策**：shared-demand score 确定共享块数与通路权重，key prototypes 选择共享内容，cumulative routing mass 确定剩余专家数，实现 token 级共享宽度、共享内容与剩余专家数量的联合协调。
3. **引入 Gram 正则化促进路由方向多样性**：正交初始化 + Gram 约束使 router embeddings 趋向正交归一，降低平均共激活率 62.9%，在保证有用专家合作的同时减少路由方向的过度相关。
4. **在视觉和语言基准上实现更强的 accuracy–efficiency 权衡**：DomainBed 平均准确率 69.5%，GLUE 平均 82.76%，同时将推理时间降低 45.2%、内存降低 52.7%（相对于 top-2 GMoE）。

## 方法详解
**总体架构**：每层含 1 个共享专家 $E_{\text{shr}}$ 和 $K$ 个残差专家 $\{E_1, \ldots, E_K\}$，所有专家初始化为相同的 dense FFN，中间宽度 $H$ 被划分为 $B$ 个对齐的 block，每个 block 宽度 $M = H/B$。

**第一步：Token-自适应共享建模**
- 共享需求分数：在标准路由器 $\mathbf{W}_g$ 基础上增加一个可学习列 $\mathbf{W}_{\text{shr}}$，得到 $\mathbf{W}_g^\star = [\mathbf{W}_{\text{shr}}, \mathbf{W}_g]$，共享需求分数为 $\alpha(\mathbf{x}) = \tau + (1-2\tau)\sigma(\mathbf{x}\mathbf{W}_{\text{shr}})$，其中 $\tau = \frac{B-1}{B^2}$，保证 $\alpha(\mathbf{x}) \in (\tau, 1-\tau)$。
- 共享块数：$b(\mathbf{x}) = \text{round}(B \alpha(\mathbf{x}))$，确保每个 token 至少使用 1 个共享块并保留至少 1 个残差块。
- 共享块选择：用每个 block 的 up-projection key 均值 $\mu_b = \frac{1}{M}\sum_{h=(b-1)M+1}^{bM} \mathbf{K}_{\text{shr}}[:, h]$ 作为 priority，$u_b(\mathbf{x}) = \mathbf{x}\mu_b$，选择 top-$b(\mathbf{x})$ 个 block 构成共享集合 $\mathcal{T}_{\text{shr}}(\mathbf{x})$，其余为残差集合 $\mathcal{T}_{\text{res}}(\mathbf{x})$。

**第二步：Cumulative Residual-Expert Routing**
- 残差需求：$\beta(\mathbf{x}) = 1 - \alpha(\mathbf{x})$，与共享需求共享单一预算。
- 路由分数：$\mathbf{s}(\mathbf{x}) = \text{softmax}(\mathbf{x}\mathbf{W}_g)$，按降序排列得 $p_1(\mathbf{x}) \geq p_2(\mathbf{x}) \geq \cdots \geq p_K(\mathbf{x})$。
- 激活专家数：$k(\mathbf{x}) = \min\{n : \sum_{i=1}^n p_i(\mathbf{x}) \geq \beta(\mathbf{x})\}$，即最小的累积亲和度前缀满足残差需求。

**第三步：输出合并**
$$\mathbf{y} = \alpha(\mathbf{x}) E_{\text{shr}}^S(\mathbf{x}) + \sum_{i=1}^{k(\mathbf{x})} p_i(\mathbf{x}) E_{q_i(\mathbf{x})}^{\mathcal{R}}(\mathbf{x})$$
其中 $E_{\text{shr}}^S$ 为共享专家在选中 block 上的输出，$E_j^{\mathcal{R}}$ 为残差专家在非共享 block 上的输出。

**训练目标**：
- 任务损失：$\mathcal{L}_{\text{task}}$
- 多样性正则：$\mathcal{L}_{\text{div}} = \|(\mathbf{W}_g^\star)^\top \mathbf{W}_g^\star - \mathbf{I}_{K+1}\|_F$
- 总损失：$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda_{\text{div}} \mathcal{L}_{\text{div}}$，推荐 $\lambda_{\text{div}} = 0.01$

**计算量**：每个 token 激活 $C_B(\mathbf{x}) = b(\mathbf{x}) + k(\mathbf{x})[B - b(\mathbf{x})]$ 个 block，当 $k(\mathbf{x})=1$ 时恰好等于一个完整 FFN。

## 实验与结果
**数据集**：
- 视觉：DomainBed（PACS、VLCS、OfficeHome、TerraIncognita、DomainNet）， backbone 为 ImageNet 预训练的 DeiT-S/16，$K=6, B=8$
- 语言：GLUE（CoLA、MRPC、QNLI、MNLI、RTE），backbone 为 BERT-large，$K=16, B=16$

**主要结果**：
- **DomainBed 平均准确率**：UniF-MoE 达 **69.5%**，超过所有基线（LFME 68.5%、MASS 68.5%、GMoE 67.9%）；在 PACS（89.6%）、VLCS（81.7%）、DomainNet（49.4%）领先，OfficeHome 与 GMoE 并列 74.2%。
- **GLUE 平均分**：UniF-MoE 达 **82.76%**，在所有 5 个任务上均超越包括 BERT-large（81.37%）、MASS（82.19%）、DynMoE（81.64%）和最佳固定 top-k（81.95%）的所有基线。
- **计算成本（VLCS 上）**：相对 top-2 GMoE，激活参数减少 9.1%、FLOPs 减少 16.1%、推理时间减少 45.2%、内存减少 52.7%；推理时间 0.17s/step、推理内存 0.26 GiB，均为所有 MoE 基线最低。
- **消融实验**：逐项移除三个自适应决策均导致性能下降和计算量上升；固定 $\alpha=0.4$ 影响最大（DomainBed 从 69.50% 降至 68.20%）；$B=8$ 或 $16$ 效果最佳。

## 相关工作脉络
1. **共享专家方法（DeepSeekMoE、Union-of-Experts）**：这些方法改进复用或专业化，但使用 token-特定的共享分配来定义专业化应处理的计算。UniF-MoE 使该依赖显式化：共享通路先处理可复用 block 响应，Gram 约束为剩余部分保留不同路由。
2. **动态路由方法（DynMoE、MASS、Alloc-MoE）**：通过累积置信度或增长/剪枝专家池来适应专家数量，但通常独立做出分配且不基于共享响应。UniF-MoE 通过单一 shared-residual 预算有序排列二者。
3. **细粒度计算方法（Emergent MoE、MoNE、MoSE）**：关注 FFN 内部结构或可变宽度，但独立进行两种分配且不与共享响应条件化。UniF-MoE 以共享需求决定共享宽度与内容，以残差需求决定专家数。
4. **专家专业化方法（orthogonality/variance 目标、MP-MoE）**：减少重叠但不用 token-特定共享分配。UniF-MoE 通过提取可复用响应后再考虑专业化。
5. **稀疏 Upcycling（Komatsuzaki et al.）**：从 dense checkpoint 初始化 MoE 专家，使得对齐通道位置可比。本文基于此分析建立方法设计。
6. **FFN 作为 Key-Value Memory（Geva et al.）**：理论基础，将专家输出分解为 channel 响应之和，是本文通道级分解的核心理论支撑。

## 局限性与未来方向
1. **依赖稀疏 upcycling 假设**：分析基于专家从同一 pretrained FFN 初始化的设定，通道对齐性依赖于这一初始化方式；对随机初始化或其他初始化策略的专家不一定适用。
2. **Block 粒度固定**：当前将 FFN 划分为固定数量的 block，block 数 $B$ 需手动选择（$B=4$ 过粗、$B=32$ 过细），缺乏自动确定最优 $B$ 的机制。
3. **仅验证于中等规模模型**：实验仅在 DeiT-S/16 和 BERT-large 上进行，在更大规模 LLM（如 LLaMA-70B 级别）上的泛化能力尚待验证。
4. **串行计算依赖**：共享计算必须在残差路由之前完成，可能在高并行度硬件（如 TPU）上带来一定算力利用率限制。
5. **未探索多共享专家场景**：当前只设一个共享专家，多个共享专家如何协同与分工尚不清楚。

## 研究启发与可借鉴点
1. **"共享-残差"预算拆分思想可迁移至其他 MoE 变体**：将 routing budget 拆分为共享部分与残差部分（$\alpha(\mathbf{x}) + \beta(\mathbf{x}) = 1$），为后续研究如何更精细地划分计算预算提供了简洁且可复用的设计范式。
2. **通道级对齐分析可作为诊断工具**：通过分析 co-activated 专家的 value 通道余弦相似度分布来诊断 MoE 内部的冗余结构，这种方法可应用于其他模型架构的性能分析与改进。
3. **Gram 正则化促进路由多样性**：简单的正交初始化 + Gram 约束即可有效促进路由方向多样性（降低共激活率 62.9%），无需复杂的对比学习或辅助损失，可作为标准技巧集成到其他 MoE 模型中。
4. **Cumulative routing mass 替代 hard top-k**：用累积亲和度前缀来确定激活专家数（而非固定 top-k），实现真正的 token-adaptive 专家数量，可在各种动态 MoE 场景中作为替代方案。
5. **块级共享的内容选择机制（key prototype）**：用 block 级 up-projection key 均值作为 prototype 进行选择，使不同 token 可使用相同共享宽度但不同共享内容，避免固定前缀问题，设计简洁且有效。

## 关键术语表
**Sparse Upcycling**：从 dense pretrained FFN checkpoint 复制初始化 MoE 专家的稀疏化训练方法，使各专家在通道级别具有可比性。

**Shared-residual Partitioning**：将每个专家的隐藏维度划分为对齐的 block，为每个 token 将 block 分配给共享通路或残差通路，确保无重叠。

**Cumulative Routing Mass**：按亲和度降序累积 expert 权重，当累积值首次覆盖残差需求时的专家数，用于确定 token-adaptive 的专家激活数量。

**Token-adaptive Computation**：根据每个 token 的语义需求动态调整共享宽度、共享内容和残差专家数的计算分配机制。

**Gram Regularizer**：对 router embeddings 的 Gram 矩阵施加 Frobenius 范数约束，使其趋向单位矩阵，促进路由方向的正交性与多样性。

**Key Prototype**：每个 block 内 up-projection key 的均值向量，用于评估该 block 对当前 token 的优先级，实现内容级共享选择。

**Residual Demand**：共享需求 $\alpha(\mathbf{x})$ 的补集 $\beta(\mathbf{x}) = 1 - \alpha(\mathbf{x})$，代表需要由残差专家处理的计算量比例。

**Co-activation Frequency**：被路由到同一专家对的 token 比例，与通道对齐性正相关，反映专家间的可复用计算程度。

## 可复现要素
- **数据集**：DomainBed（PACS、VLCS、OfficeHome、TerraIncognita、DomainNet）和 GLUE（CoLA、MRPC、QNLI、MNLI、RTE）；代码已开源，数据集为标准公开数据集。
- **代码开源**：https://github.com/existence0420/UniF-MoE
- **权重开源**：论文未提及模型权重开源情况。
- **关键超参**：
  - 视觉：DeiT-S/16，$d=384$，$H=1536$，$K=6$，$B=8$，hidden dropout=0.1，stochastic-depth=0.1，lr=$1\times10^{-5}$~$5\times10^{-5}$（依数据集），weight decay=$10^{-6}$~0
  - 语言：BERT-large，$K=16$，$B=16$，max sequence length=128，batch size=32，AdamW，lr∈{2e-5, 3e-5, 5e-5}，$\lambda_{\text{div}}=0.01$
  - 训练设备：NVIDIA RTX 3090
