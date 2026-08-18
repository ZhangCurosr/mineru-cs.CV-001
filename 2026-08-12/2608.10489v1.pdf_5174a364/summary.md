---
title: "When Vision Becomes Text: Visual Token Pruning via Cross-Modal Residual Guidance in VLMs"
source: https://arxiv.org/pdf/2608.10489v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:31:45"
---

# 论文速读：When Vision Becomes Text: Visual Token Pruning via Cross-Modal Residual Guidance in VLMs

## 一句话总结
本文提出无需训练的视觉Token压缩方法SIEVE，通过量化LLM前向过程中“文本表示随层深逐渐吸收视觉信息”的跨模态吸收现象（CMA），构建基于Tikhonov正则化最小二乘的跨模态残差（CMR）分数，并结合文本注意力相关性与残差空间多样性贪心筛选，在大幅削减视觉Token数量的同时有效保留任务关键信息。

## 研究问题与动机
- **核心问题**：VLM将高分辨率图像编码为海量视觉Token并送入LLM，导致自注意力计算开销与KV-cache显存占用急剧上升，低损失压缩视觉Token是实现高效推理的关键。
- **现有方法不足**：主流剪枝策略依赖单层内的注意力分数或特征相似度，仅捕获局部单步信号，缺乏对VLM整条推理链路中跨模态信息流动的宏观视角。
- **关键观察**：随着Transformer层数加深，文本Token通过因果自注意力持续聚合视觉信息，视觉表示逐渐被“吸收”进文本子空间，大量视觉Token的独立信息价值随之降低。
- **动机**：从几何表征视角量化这一跨模态吸收过程，以“文本无法解释的残差”作为视觉Token重要性判别依据，弥补单一相似度/注意力信号在任务相关性与全局冗余度量上的盲区。

## 核心贡献（创新点）
- **提出CMA与CMR跨模态残差度量机制**：从几何视角层级别度量视觉表示被文本子空间解释的比例，并在Token级别构建残差比；与现有基于相似度或单点注意力的剪枝标准本质不同，CMR刻画的是贯穿多层推理的跨模态信息流与冗余度。
- **设计SIEVE无训练压缩框架**：联合CMR几何独特性、文本注意力语义相关性与残差空间多样性三者筛选Token；区别于传统特征空间去重，该方法在剔除文本可解释分量后的残差空间中衡量互补性，定位更精准且无需微调。
- **提出FlashAttention兼容的辅助值聚合方案**：通过构造特殊辅助值矩阵 $\widetilde{V}$ 在一次流式前向中统计后文本文本对视觉Token的注意力权重总和；与依赖物化完整注意力矩阵或修改Transformer结构的剪枝方法不同，该设计几乎不引入额外延迟，工程落地更友好。

## 方法详解
- **中心化处理**：以文本Token均值 $\bar{t}$ 为参照，对文本矩阵 $T \in \mathbb{R}^{N_t \times D}$ 与视觉矩阵 $V \in \mathbb{R}^{N_v \times D}$ 同步去中心化得到 $T_c, V_c$，突出视觉Token相对文本子空间的独有偏离。
- **CMR分数构造**：对第 $i$ 个视觉Token，通过带Tikhonov正则的最小二乘求其在文本子空间上的线性组合系数：$\min_{b_i} \|v_{c,i} - b_i T_c\|_2^2 + \lambda \|b_i\|_2^2$。正则系数 $\lambda$ 自适应选取：计算Gram矩阵 $G = T_c T_c^\top$ 的特征值，按能量比例阈值 $\eta$ 截断取第 $k$ 大特征值作为 $\lambda$。重建残差比定义为 $\mathrm{CMR}_i = \|v_{c,i} - \hat{v}_{c,i}\|_2 / \|v_{c,i}\|_2$，值越大说明该Token携带越多的文本不可解释的独特视觉信息。
- **注意力信号构建**：聚合后文本文本Token在顶部 $\rho_h$ 比例注意力头中对各视觉Token的注意力权重之和 $a_i = \sum_{h \in \mathcal{H}_{top}} \sum_{t \in \mathcal{T}} \alpha_{t,i}^h$，反映语义相关性；借助辅助值矩阵 $\widetilde{V}$（视觉位置置单位基向量，其余为零）配合FlashAttention流式计算，避免显式物化 $N \times N$ 注意力矩阵。
- **引导分数融合**：采用乘法融合 $\mathrm{Score}_i = a_i^2 \cdot \mathrm{CMR}_i^2$，兼顾任务相关性与几何独特性，避免加法融合中量纲差异导致的单一信号主导问题。
- **残差空间多样性选择**：计算Token间残差向量的平方余弦相似度 $g_{ij} = (r_i^\top r_j / (\|r_i\|_2 \|r_j\|_2))^2$，贪心迭代选取最高分Token，并按 $\mathrm{Score}_j \leftarrow \mathrm{Score}_j(1 - g_{i^*j})$ 衰减剩余候选分数，逐步压制残差方向重叠的冗余Token，直至达到目标数量 $K$。

## 实验与结果
- **评测设置**
