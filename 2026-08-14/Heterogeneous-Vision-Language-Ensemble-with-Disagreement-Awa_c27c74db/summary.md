---
title: "Heterogeneous-Vision-Language-Ensemble-with-Disagreement-Awa"
source: https://arxiv.org/pdf/2608.12843v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:02:46"
---

# 论文速读：Heterogeneous-Vision-Language-Ensemble-with-Disagreement-Awa

## 一句话总结
本文针对基于文本的行人体异常检索（TBAPS）任务，提出一种异构视觉-语言模型迭代集成框架，通过分数对齐融合多源互补表征，并引入基于预测分歧感知的VLM选择性交叉编码器重排序机制，在AI City Challenge 2026 Track 4的PAB基准上取得90.92% mAP的最优成绩。

## 研究问题与动机
- **核心问题**：TBAPS要求仅凭自然语言描述，从数万级监控图像库中精确检索出具有特定异常行为的行人，需同时理解细粒度外观、动作语义、物体交互与场景上下文。
- **现有方法不足1**：传统TBPS方法主要依赖静态外观特征（衣着、体型），无法区分视觉相似但语义迥异的行为（如“下楼梯”与“摔下楼梯”）。
- **现有方法不足2**：已有迭代集成框架（如HUI）局限于同构或架构相似的嵌入模型，未能充分利用大规模预训练视觉-语言模型的互补表征。
- **现有方法不足3**：直接融合不同预训练模型的相似度矩阵会因图库排序/索引不一致导致匹配错位；而全量调用高成本VLM交叉编码器重排序推理开销过大，难以落地。

## 核心贡献（创新点）
- **异构嵌入迭代集成**：在HUI基线上依次融合Voyage Multimodal、BGE-VL-v1.5-mmeb与Qwen3-VL-Embedding-8B等多个异构视觉-语言嵌入模型。与已有工作本质区别在于：不再追求单一最强模型，而是显式利用不同训练目标与数据分布带来的表征互补性，以分阶段加权更新缓解几何对齐漂移。
- **分数对齐策略**：针对独立模型图库索引不一致问题，构建内部索引到基准参考索引的映射表，将所有相似度矩阵重排至统一图库顺序后再进行融合。与已有工作本质区别在于：解决了异构双编码器集成中“排序错配”的系统性误差，使分数级融合在任意预训练模型间均可靠有效。
- **分歧感知VLM重排序**：设定零容忍Top-1分歧阈值识别模糊查询，仅将约15.1%的高分歧样本路由至`gemini-3.1-flash-lite`交叉编码器进行细粒度重排，其余保留双编码器结果。与已有工作本质区别在于：打破“全量重排序”与“纯双编码器”的两极做法，用预测不一致性作为疑难代理信号，在推理延迟与长尾召回之间取得高效平衡。
- **RRF跨模态融合**：采用Reciprocal Rank Fusion（k=60）整合异构双编码器排序与VLM重排序列表。与已有工作本质区别在于：免去分数归一化与校准步骤，直接将不同分布的排名列表统一聚合，适合多源API/离线模型混合部署场景。

## 方法详解
- **基线检索框架**：沿用HUI框架，结合Local-Global Hybrid Perspective (LHP) 模块（概率性局域/全局图像变换增强视觉表征）与Unified Image-Text (UIT) 模块（联合优化ITC、ITM、MLM、MIM损失），得到图文嵌入 $\mathbf{z}_q = f_q(q)$、$\mathbf{z}_i = f_i(i)$ 及余弦相似度 $S(i,q) = \frac{\mathbf{z}_i^\top \mathbf{z}_q}{\|\mathbf{z}_i\|_2 \|\mathbf{z}_q\|_2}$。
- **异构迭代集成**：设嵌入模型集合 $\mathcal{M}=\{M_1,...,M_K\}$，各模型独立计算相似度矩阵 $S^{(k)} \in \mathbb{R}^{N_q \times N_g}$。累积矩阵初始化为 $S_{\text{init}}$，按 $S \leftarrow w_k S + (1-w_k) S^{(k)}$ 逐步更新，依次引入Voyage → 中间模型（BGE-VL-v1.5-mmeb等）→ Qwen3-VL-Embedding。
- **分数对齐（Score Alignment）**：为各模型构建内部图库排序到基准参考索引的映射，对 $S^{(k)}$ 的行进行重排，确保所有分数严格对应同一张图库图像，避免错误配对。
- **分歧感知路由（Disagreement-Aware Routing）**：若多个集成配置在Top-1预测上存在任意不一致，则判定该查询为模糊样本。满足条件的查询（299个，占15.1%）被送入`gemini-3.1-flash-lite`交叉编码器，对Top-10候选进行专家级异常行为重排；其余84.9%查询直接沿用双编码器结果。
- **RRF融合**：最终分数计算为 $\text{RRF}(d) = \sum_{r \in \mathcal{R}} \frac{1}{k + \text{rank}(r, d)}$，其中 $k=60$，将主排序 $P_{\text{primary}}$、备选排序 $P_{\text{alt}}$ 与VLM重排序 $P_{\text{VLM}}$ 合并后按分排序输出。

## 实验与结果
- **数据集**：AI City Challenge 2026 Track 4 官方 Pedestrian Anomaly Behavior (PAB) 基准。训练集含超100万合成图文对（1,000常规 + 1,600异常行为）；评测集含1,978条查询与36,773张图库图像（含34,795个真实监控干扰项）。评估指标：mAP、Recall@1/5/10。
- **评估基线**：CLIP, IRRA, X-VLM, RaSa, CMP, AnomalyLMM, SSDC，以及当前最强TBAPS基线HUI (Nguyen et al. [11])。
- **主要结果**：最终提交（Iterative Ensemble + Rerank Fusion）取得 **mAP 90.92%**、**Recall@1 85.13%**、**Recall@5 97.72%**、**Recall@10 98.68%**，刷新该赛道记录。消融表明：加入Voyage带来显著增益；将Qwen置于末层并微调权重（0.88:0.9）可提升+1.29% mAP；BGE-VL-v1.5-mmeb对海量干扰项鲁棒性最佳。
- **效率收益**：分歧路由仅对299个疑难查询调用`gemini-3.1-flash-lite`，相比全量重排序降低 **84.9%** 推理延迟，同时保障高置信度样本的检索效率。

## 相关工作脉络
- **CLIP / BLIP-2 / BEiT-3**：通用视觉-语言预训练范式，本文借鉴其跨模态对齐思想，但定位差异在于将其作为离线嵌入模块接入特定检索集成管线，而非从头预训练。
- **CMP / SSDC / AnomalyLMM**：面向PAB的早期TBAPS方法，多依赖单模型判别或生成式LVLM全量推理；本文采用“双编码器异构集成 + 选择性交叉编码器”的粗到细范式，显著降低延迟并提升长尾召回。
- **HUI (Nguyen et al. [11])**：当前最强基线，提出LHP+UIT与迭代集成；本文在其基础上扩展异构模型库，并引入分数对齐与分歧路由，解决其
