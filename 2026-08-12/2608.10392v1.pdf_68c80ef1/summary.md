---
title: "Share First, Route What Remains:"
source: https://arxiv.org/pdf/2608.10392v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:45:27"
---

# 论文速读：Share First, Route What Remains:

## 一句话总结
本文提出UniF-MoE，将MoE中的共享专家、细粒度块选择与动态路由统一为“先共享、后路由”的有序决策：先提取token可复用的隐藏块响应，再根据剩余计算需求自适应激活残差专家，在提升视觉与语言任务精度的同时显著降低激活参数、FLOPs与推理延迟。

## 研究问题与动机
- 现有MoE研究通常将共享专家设计、expert内部细粒度计算与动态路由作为独立机制开发，忽略了它们之间的内在依赖：一旦提取出可复用计算，剩余输出分布与最优专家选择都会发生变化。
- 传统top-k路由以完整expert为计算单元，对简单token造成共享响应重复计算，对困难token则专家容量不足，缺乏token级的预算分配。
- 独立并行调节三种稀疏性会导致路由器在已部分计算的响应上再次分配expert容量，造成冗余与效率损失。
- 现有工作缺少一个统一的、按序耦合的token-adaptive计算框架，无法协调intra-expert（块级）与inter-expert（路由）稀疏性。

## 核心贡献（创新点）
- **揭示共享-残差依赖关系**：提出“share first, then route what remains”原则，指出可复用计算提取后必须重新评估路由，而非沿用原router决策。与现有并行模块化设计本质不同，本文将其排序为单阶段的因果依赖。
- **统一Token-Adaptive MoE框架（UniF-MoE）**：通过共享需求分数$\alpha(\mathbf{x})$与残差需求$\beta(\mathbf{x})$构建零和预算，将共享块数、共享内容与残差专家数量耦合于单次路由。与仅调整单维度的基线不同，本文一次性协调三层稀疏性。
- **累积路由质量动态激活**：引入按排序权重累积的最小子集选择$k(\mathbf{x})$，使激活专家数严格由剩余需求覆盖阈值决定。区别于DynMoE/MASS的置信度阈值或固定预算，本文直接复用同一router输出而不引入额外网络。
- **Gram正交路由正则**：对router嵌入矩阵施加Gram约束$\|(\mathbf{W}_g^\star)^\top\mathbf{W}_g^\star-\mathbf{I}\|_F$，以极低开销塑造正交路由几何。与MP-MoE等直接约束expert输出的正则不同，本文作用于路由器空间以间接促进专家角色分化。
- **系统性诊断与实证**：在TerraIncognita上完成value alignment、router rank transition与residual demand CDF的联合分析，量化证明共享覆盖率与残差需求呈负相关（相关系数-0.673），为设计提供数据支撑。

## 方法详解
- **Blockwise Shared-Residual Partitioning**：每层包含1个共享专家$E_{\mathrm{shr}}$与K个残差专家$\{E_1,\dots,E_K\}$，每个expert中间层宽度$H$被均分为$B$个对齐块（$M=H/B$）。所有expert从同一稠密FFN初始化，块边界天然对齐。对每个token，共享路径执行选定块子集，残差路径执行互补块，无隐藏位置重叠。
- **Token-Adaptive Shared Modeling**：扩展router矩阵加入共享列$\mathbf{W}_{\mathrm{shr}}$，计算共享需求分数：
  $\alpha(\mathbf{x}) = \tau + (1-2\tau)\sigma(\mathbf{x}\mathbf{W}_{\mathrm{shr}}),\quad \tau=\frac{B-1}{B^2}$
  由$\sigma\in(0,1)$与$\tau$边界保证$\alpha(\mathbf{x})\in(\tau,1-\tau)$，进而块数$b(\mathbf{x})=\mathrm{round}(B\alpha(\mathbf{x}))\in\{1,\dots,B-1\}$。
- **Shared Block Selection via Key Prototypes**：用每块上投影key均值$\mu_b=\frac{1}{M}\sum_{h\in\mathcal{H}_b}\mathbf{K}_{\mathrm{shr}}[:,h]$作为块原型，优先级$u_b(\mathbf{x})=\mathbf{x}\mu_b$。选取Top-$b(\mathbf{x})$块构成$\mathcal{T}_{\mathrm{shr}}(\mathbf{x})$，补集为$\mathcal{T}_{\mathrm{res}}(\mathbf{x})$，使共享内容具token特异性。
- **Cumulative Residual-Expert Routing**：残差需求$\beta(\mathbf{x})=1-\alpha(\mathbf{x})$。对常规router输出$\mathbf{s}(\mathbf{x})=\mathrm{softmax}(\mathbf{x}\mathbf{W}_g)$排序得$p_1\geq\cdots\geq p_K$，激活满足$\sum_{i=1}^n p_i(\mathbf{x})\geq\beta(\mathbf{x})$的最小$n$作为$k(\mathbf{x})$。总系数质量有界：$1\leq\alpha+P_{k}<1+p_k$，避免过度激活。
- **Shared-Residual Output Merging**：
  $E_{\mathrm{shr}}^S=\sum_{b\in\mathcal{T}_{\mathrm{shr}}}E_{\mathrm{shr}}^{(b)},\quad E_j^{\mathcal{R}}=\sum_{b\in\mathcal{T}_{\mathrm{res}}}E_j^{(b)}$
  $\mathbf{y}=\alpha(\mathbf{x})E_{\mathrm{shr}}^S(\mathbf{x})+\sum_{i=1}^{k(\mathbf{x})}p_i(\mathbf{x})E_{q_i}^{\mathcal{R}}(\mathbf{x})$
- **Training Objective**：$\mathcal{L}=\mathcal{L}_{\mathrm{task}}+\lambda_{\mathrm{div}}\|(\mathbf{W}_g^\star)^\top\mathbf{W}_g^\star-\mathbf{I}_{K+1}\|_F$，正交初始化配合Gram正则，保持embedding尺度为1且方向分离，促进稀疏重叠。计算度量：$C_B(\mathbf{x})=b(\mathbf{x})+k(\mathbf{x})[B-b(\mathbf{x})]$，共享块仅计算一次。

## 实验与结果
- **数据集与配置**：视觉 backbone为ImageNet预训练DeiT-S/16（$d=384,H=1536$），转换layer 8&10，$K=6,B=8$；语言 backbone为BERT-large-cased，转换layer 20&22，$K=16,B=16$。
- **基线**：DeiT-S/16、GMoE、EMoE、EMoE-L、LFME、DMDA、PC-MoE、DynMoE、MASS、固定top-k MoE（$k\in\{1,2,4,8\}$）。
- **Vision结果（DomainBed）**：UniF-MoE平均精度**69.5%**，PACS 89.6%、VLCS 81.7%、DomainNet 49.4%获最优，OfficeHome 74.2%并列第一；TerraIncognita（52.6%）略低于LFME（53.4%，依赖显式域专用化）。
- **Language结果（GLUE）**：平均**82.76%**，五项任务全面超越固定top-k best oracle（82.19%）与DynMoE/MASS。
- **效率对比（VLCS）**：相对top-2 GMoE，激活参数减少9.1%（$N_A$: 21.01 vs 23.11），FLOPs减少16.1%，推理时间降低45.2%（0.17s vs 0.31s），推理内存降低52.7%（0.26 GiB vs 0.55 GiB）；全面优于DynMoE。
- **消融**：固定$\alpha$/prefix-block/top-2 residual任一环节均导致精度与计算双降，证明三阶段协同必要性；$\lambda_{div}=0.01$效果最佳；$B=8$在块粒度与复杂度间最优。

## 相关工作脉络
- **DeepSeekMoE / Union-of-Experts**：添加独立共享expert或虚拟共享路由层，但共享内容静态或仅按路由神经元聚合，未按token剩余需求动态分配；本文以value alignment为准则先验识别可复用块。
- **DynMoE / MASS / Alloc-MoE**：动态调整expert数量或预算，但路由决策与共享/细粒度计算解耦；本文用$\beta(x)=1-\alpha(x)$构建共享-残差零和预算，使路由规模严格承接前置共享分配。
- **Emergent MoE / Nested & Slimmable Experts**：在expert内部暴露模块化或可变宽度，但未与跨expert路由顺序耦合；本文通过块划分实现intra/expert稀疏的因果串联。
- **Orthogonality / Variance 正则（Advancing Expert Specialization）**：直接约束expert输出或隐藏状态促分化；本文仅约束router embedding Gram矩阵，以几何约束间接引导稀疏重叠，代价更低。
- **Sparse Upcycling**：将单预训练FF
