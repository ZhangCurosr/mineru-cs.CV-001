---
title: "When Vision Becomes Text: Visual Token Pruning via Cross-Modal Residual Guidance in VLMs"
source: https://arxiv.org/pdf/2608.10489v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:31:25"
field: "多模态高效推理"
keywords: ["视觉语言模型", "token剪枝", "跨模态吸收", "训练无关压缩", "残差空间选择"]
innovations: ["提出跨模态残差CMR量化视觉token中无法被文本子空间解释的独特信息", "设计SIEVE训练无关剪枝方法联合CMR、注意力相关性和残差空间多样性三个互补信号", "发现并验证VLM推理中视觉token随层加深被文本子空间渐进吸收的现象"]
benchmarks: ["GQA", "MMBench", "MMBench-CN", "MME", "POPE", "SQA", "VQAv2", "TextVQA"]
---

# 论文速读：When Vision Becomes Text: Visual Token Pruning via Cross-Modal Residual Guidance in VLMs

## 一句话总结
本文提出一种训练无关的视觉token剪枝方法SIEVE，通过跨模态残差（CMR）量化视觉token中无法被文本子空间解释的独特信息，并结合文本注意力相关性与残差空间多样性选择，实现高效视觉token压缩。在LLaVA-NeXT-7B上仅保留11.1%视觉token即可保持97.5%原始性能，实现3.62× prefill加速。

## 研究问题与动机
- **视觉token冗余导致推理成本高昂**：VLM将图像编码为大量视觉token与文本token共同送入LLM，高置信分辨率图像、多图输入和视频理解场景进一步放大计算开销。
- **现有方法仅捕捉单层局部信号**：主流剪枝方法依赖注意力分数（局部相关性）或特征相似度（去重），缺乏对多模态推理全过程的全局视图。
- **视觉信息随层加深被文本子空间渐进吸收**：通过重新审视VLM推理过程，发现文本token通过自注意力持续聚合视觉信息，深层视觉token可被文本子空间更好解释，存在跨模态冗余。
- **需要从"独特性+相关性+互补性"三维选择token**：单纯相似度或注意力指导可能遗漏具有独特视觉信息但语义相关性弱的token，或遗漏相关性强但残差信息少的token。

## 核心贡献（创新点）
1. **提出跨模态吸收（CMA）与跨模态残差（CMR）**：从几何表示视角量化视觉token被文本子空间吸收的程度，与现有基于相似度/注意力的剪枝指导 criterion 本质不同，关注跨模态信息流动的层间演化规律。
2. **提出SIEVE训练无关剪枝框架**：联合CMR（视觉独特性）、文本注意力相关性（任务相关性）和残差空间多样性（互补性）三个互补视角选择token，而非单一信号排序。
3. **发现并验证跨模态吸收现象**：在LLaVA-1.5-7B上分析32层，CMA与层深度Spearman相关系数ρ=+0.99，证明视觉token随层加深逐渐融入文本表示空间。
4. **设计FlashAttention兼容的注意力分数计算**：通过auxiliary value matrix实现双flash聚合，无需物化完整注意力矩阵，保持与高效attention实现的兼容性。

## 方法详解
- **跨模态吸收（CMA）定义**：$\text{CMA} = 1 - \frac{\|V_c - \hat{V}_c\|_F^2}{\|V_c\|_F^2}$，其中$V_c$为中心化视觉token矩阵，$\hat{V}_c$为其在文本子空间上的投影，CMA衡量当前层视觉信息可被文本子空间解释的比例。
- **跨模态残差（CMR）构造**：对第i个中心化视觉token $v_{c,i}$，通过Tikhonov正则化最小二乘投影到文本子空间：$b_i = \arg\min \|v_{c,i} - b_i T_c\|_2^2 + \lambda\|b_i\|_2^2$，其中正则化系数$\lambda = \sigma_k^2 + \epsilon$由文本Gram矩阵$G=T_c T_c^\top$的第k大特征值确定（覆盖能量比例$\eta$）。CMR得分定义为重建残差与原始表示的比值：$\text{CMR}_i = \frac{\|v_{c,i} - \hat{v}_{c,i}\|_2}{\|v_{c,i}\|_2}$，越小说明token越能被文本子空间解释，应优先剪枝。
- **注意力信号构建**：$a_i = \sum_{h \in \mathcal{H}_{\text{top}}} \sum_{t \in \mathcal{T}} \alpha_{t,i}^h$，仅保留对视觉区域注意力质量最高的$\rho_h$比例注意力头，通过辅助value matrix实现双flash聚合以兼容FlashAttention。
- **引导分数融合**：$\text{Score}_i = a_i^2 \cdot \text{CMR}_i^2$，乘法融合确保token同时具备高任务相关性和高残差独特性，避免加法融合中单一信号主导的问题。
- **残差空间多样性选择**：在残差空间$r_i = v_{c,i} - \hat{v}_{c,i}$中计算余弦相似度的平方$g_{ij} = \left(\frac{r_i^\top r_j}{\|r_i\|_2 \|r_j\|_2}\right)^2$，贪婪选择：每次选$\text{Score}_i$最高token，并对剩余候选按$(1-g_{i^*j})$衰减，逐步抑制残差方向相似的冗余token。

## 实验与结果
- **评估架构与基准**：LLaVA-1.5-7B（576视觉token）、LLaVA-NeXT-7B（2880视觉token）、Qwen2.5-VL-7B（1296视觉token，分辨率1008×1008）；八个标准基准：GQA、MMBench、MMBench-CN、MME、POPE、SQA、VQAv2、TextVQA。
- **LLaVA-1.5-7B结果**：保留192 token（33.3%）时平均性能达Vanilla的99.1%；128 token（22.2%）时98.7%；64 token（11.1%）时96.0%，仍超越最强基线SCOPE 0.6pp。
- **LLaVA-NeXT-7B结果**：保留320 token时平均性能达Vanilla的97.5%，超越DART和HoloV分别1.4pp和1.3pp。
- **Qwen2.5-VL-7B结果**：保留33.3% token时99.7%平均性能；22.2%时98.9%；**11.1%时96.7%**，超越FastV和HoloV分别9.2pp和2.3pp。
- **效率提升（LLaVA-NeXT-7B，11.1%保留率）**：prefill加速3.62×，端到端加速2.49×，KV-cache从1156MB降至192MB（6.02×减少）。
- **消融结论**：仅用注意力分数得96.07%，仅用CMR得96.00%，乘法融合得96.59%，加法融合仅96.20%；残差空间多样性选择优于原始空间多样性（-0.89pp vs -6.46pp）。

## 相关工作脉络
- **FastV（ECCV'24）**：基于层数固定比例剪枝（layer 2后保留1/2），属经验性规则剪枝，无内容自适应指导；SIEVE通过CMR和注意力动态选择token。
- **SparseVLM（ICML'25）**：基于attention score剪枝，依赖当前层注意力质量；SIEVE引入跨模态残差视角，捕捉text-subspace不可解释的独特信息。
- **DART（EMNLP'25）**：基于动态阈值剪枝；SIEVE从跨模态吸收现象出发，提供几何解释。
- **VisionZip（CVPR'25）**：长序列中"更长并非必要"的冗余剪枝；SIEVE关注跨模态信息流动而非序列长度本身。
- **HoloV（NeurIPS'25）**：保留视觉整体上下文；SIEVE在保留holistic信息的同时通过残差空间多样性消除冗余。
- **SCOPE（NeurIPS'25）**：显著性-覆盖导向剪枝；SIEVE额外引入跨模态残差作为独立信号，与注意力形成互补。

## 局限性与未来方向
- **剪枝层数未充分讨论**：论文主要在一个固定层剪枝，多层渐进剪枝的策略和收益未系统探索。
- **仅验证图像任务**：未见视频或多图输入上的实验，对长视觉序列的扩展性待验证。
- **正则化系数敏感性**：虽证明对η和$H_{\text{top}}$鲁棒，但未讨论极端场景（如极低能量比或单头注意力）下的表现。
- **未涉及训练优化**：纯训练无关方法，未来可探索与训练过程联合优化的剪枝策略。
- **理论分析有限**：CMA的上界分析依赖视觉token的主成分能量分布，但未给出更严格的理论保证。

## 研究启发与可借鉴点
- **跨模态吸收现象的挖掘思路**：从模型推理过程的层间演化规律出发发现新的剪枝信号，而非局限于单层局部统计量，这一研究范式可迁移到其他多模态压缩任务。
- **残差空间多样性选择的设计**：在去除文本可解释成分后的残差空间中进行冗余度量，比原始特征空间更准确地反映token的互补性，该思路可推广至其他模态的token选择。
- **乘法融合互补信号的策略**：Attention和CMR的overlap仅11.54%，乘法融合优于加法，这一发现为多信号融合提供了实证依据和可复用策略。
- **双flash attention兼容设计**：通过auxiliary value matrix在不物化完整注意力矩阵的前提下提取统计量，对需要attention score的高效VLM优化具有参考价值。
- **CMA的有效尺度解释**：论文指出CMA不应直接按[0,1]解读，而应相对于随机基线（$N_t/D \approx 0.007$）和理论上界（约0.4-0.7）理解，这种相对解释框架值得借鉴。

## 关键术语表
- **Cross Modal Absorption (CMA)**：层级别的跨模态吸收度量，衡量视觉token表示中可被文本子空间解释的信息比例。
- **Cross Modal Residual (CMR)**：token级别的跨模态残差得分，通过Tikhonov正则化投影到文本子空间后计算重建残差与原始表示的比值，量化视觉token的独特性。
- **SIEVE**：本文提出的训练无关视觉token压缩方法，联合CMR、文本注意力相关性和残差空间多样性选择保留token。
- **Text Subspace**：由文本token表示张成的低维子空间，随层加深逐渐吸收更多视觉相关信息。
- **Residual-Space Diversity Selection**：在去除文本可解释成分后的残差空间中计算token间冗余并贪婪选择互补token的策略。
- **Dual-Flash Aggregation**：通过auxiliary value matrix实现FlashAttention兼容的注意力统计提取方法。
- **Energy Ratio (η)**：用于确定Tikhonov正则化系数的文本子空间能量覆盖阈值，控制有效秩。
- **$H_{\text{top}}$**：选出的对视觉区域注意力质量最高的注意力头数量。

## 可复现要素
- **数据集**：GQA、MMBench、MMBench-CN、MME、POPE、SQA、VQAv2、TextVQA（均为公开基准）。
- **代码/权重**：论文未明确声明开源状态，需查看arXiv页面或补充材料。
- **关键超参**：能量比例η（默认0.75最优，测试范围0.65-0.95）、选头数$H_{\text{top}}$（默认12）、保留token比例（11.1%/22.2%/33.3%）。
- **模型**：LLaVA-1.5-7B、LLaVA-NeXT-7B、Qwen2.5-VL-7B。
- **分辨率设置**：Qwen2.5-VL实验固定为1008×1008。
