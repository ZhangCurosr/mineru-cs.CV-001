---
title: "When Vision Becomes Text: Visual Token Pruning via Cross-Modal Residual Guidance in VLMs"
source: https://arxiv.org/pdf/2608.10489v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:32:39"
field: "多模态高效推理"
keywords: ["视觉语言模型", "token 剪枝", "跨模态吸收", "无训练压缩", "VLM 推理加速"]
innovations: ["提出 CMR（跨模态残差）度量视觉 token 中无法被文本子空间解释的信息量", "设计 SIEVE 训练无感知压缩框架，联合残差得分、注意力相关性与残差空间多样性选择", "发现并量化 VLM 中视觉 token 随层深被文本子空间渐进吸收的单调规律"]
benchmarks: ["GQA", "MMBench", "MMBench-CN", "MME", "POPE", "SQA", "VQAv2", "TextVQA"]
---

# 论文速读：When Vision Becomes Text: Visual Token Pruning via Cross-Modal Residual Guidance in VLMs

## 一句话总结
本文提出 SIEVE，一种无需训练的视觉 token 压缩方法，通过**跨模态残差（CMR）**量化视觉 token 中无法被文本子空间解释的信息量，结合文本注意力相关性评分与残差空间多样性选择，在大幅减少视觉 token 数量的同时保持 VLM 推理性能。

## 研究问题与动机
- **VLM 推理成本高**：LLaVA 等 VLM 将图像编码为大量视觉 token 与文本 token 共同输入 LLM，高分辨率图像、多图输入和视频场景下，视觉 token 数量急剧膨胀，导致 self-attention 计算和 KV-cache 开销显著上升。
- **现有方法缺乏全局视角**：主流 token 剪枝方法依赖单层内的注意力分数（如 FastV、SparseVLM）或特征相似度（如 Token Merging），仅捕捉局部层状态，无法反映多模态推理的全程信息流动模式。
- **跨模态吸收现象未被利用**：随着 LLM 层加深，文本 token 通过 self-attention 持续聚合视觉信息，视觉 token 逐步被文本子空间"吸收"，其中冗余 token 占比随层深增加而增大，这一规律可为剪枝提供新的指导信号。
- **几何表征视角的缺失**：现有工作未从子空间可解释性的角度衡量视觉 token 的独特性，缺乏对"哪些视觉信息真正无法被文本表征覆盖"的系统性度量。

## 核心贡献（创新点）
1. **提出 CMA（跨模态吸收度量）与 CMR（跨模态残差）**：从几何表征视角量化视觉 token 被文本子空间吸收的程度，CMR 通过 Tikhonov 正则化最小二乘投影计算每个视觉 token 的残差比例；与已有方法本质区别在于不依赖注意力分数或特征相似度，而是利用跨模态信息吸收的全局规律。
2. **提出 SIEVE 训练无感知压缩框架**：联合 CMR 残差得分、文本-视觉注意力相关性得分与残差空间多样性选择三个互补信号进行 token 筛选；与已有单信号剪枝方法的本质区别在于同时兼顾视觉独特性、任务相关性和 token 间互补性。
3. **发现并验证跨模态吸收的单调规律**：在 LLaVA-1.5-7B 上系统分析 32 层 Transformer，发现 CMA 随层深单调递增（Spearman 相关系数 ρ=+0.99），最终层 CMA 约为随机基线的 29 倍；此前无工作从该角度揭示 VLM 内部跨模态信息流动的几何结构。
4. **设计 FlashAttention 兼容的注意力统计计算策略**：通过辅助 value 矩阵的 dual-flash 聚合策略在不物化完整注意力矩阵的前提下高效计算文本到视觉的注意力得分；与直接提取注意力子矩阵的方法本质不同，保持了 FlashAttention 的显存效率。

## 方法详解

**核心观察与 CMA（层级别度量）**：
- 定义 CMA = 1 − ||V_c − V̂_c||_F² / ||V_c||_F²，其中 V_c 为居中后的视觉 token 矩阵，V̂_c 为其在文本子空间上的投影。CMA 越大表示该层中越多视觉信息可被文本子空间解释。
- 实验发现 CMA 从浅层到深层单调递增：POPE 上从 0.008→0.209，MME 上从 0.009→0.217，TextVQA 上从 0.017→0.396。

**CMR（token 级别度量）**：
- 对第 i 个居中的视觉 token v_{c,i}，求解 Tikhonov 正则化最小二乘：b_i = argmin ||v_{c,i} − b_i T_c||₂² + λ||b_i||₂²，闭式解为 b_i = v_{c,i} T_cᵀ(G + λI)⁻¹，其中 G = T_c T_cᵀ 为文本 Gram 矩阵。
- 正则化系数 λ 自适应确定：对 G 的特征值排序，选取满足能量比阈值 η 的最小 k，令 λ = σ_k² + ε。
- CMR_i = ||v_{c,i} − v̂_{c,i}||₂ / ||v_{c,i}||₂，值越大表示该 token 携带越多的文本无法解释的独特视觉信息。

**注意力信号**：
- a_i = Σ_{h∈H_top} Σ_{t∈T} α_{t,i}^h，仅保留视觉区域注意力质量最高的 top-ρ_h 个 attention head。
- 通过 dual-flash 策略（辅助 value 矩阵标记视觉 token 位置）兼容 FlashAttention，无需物化完整注意力矩阵。

**融合与多样性选择**：
- 综合得分：Score_i = a_i² × CMR_i²（乘性融合，优于加性融合，实验显示 96.59% vs 96.20%）。
- 残差空间多样性贪心选择：在残差空间 r_i = v_{c,i} − v̂_{c,i} 中计算余弦相似度 g_{ij} = (r_iᵀ r_j / (||r_i||₂||r_j||₂))²，每次选取最高分 token 后按 (1 − g_{i*j}) 衰减其余候选者的得分，迭代至预算 K。

## 实验与结果

**实验设置**：三模型（LLaVA-1.5-7B、LLaVA-NeXT-7B、Qwen2.5-VL-7B）× 八大基准（GQA、MMBench、MMBench-CN、MME、POPE、SQA、VQAv2、TextVQA）。

**主要结果**：
- **LLaVA-1.5-7B**：保留 11.1% tokens 时，SIEVE 保持 Vanilla 的 **96.0%** 平均性能，最优超过完整结果基线 SCOPE 0.6pp；保留 22.2% 和 33.3% 时分别保持 **98.7%** 和 **99.1%**。
- **LLaVA-NeXT-7B**（2880 tokens → 320 tokens）：保持 Vanilla **97.5%** 平均性能，较 DART 和 HoloV 分别提升 **1.4pp** 和 **1.3pp**。
- **Qwen2.5-VL-7B**：11.1% 保留率下保持 Vanilla **96.7%** 性能，较 FastV 和 HoloV 分别提升 **9.2pp** 和 **2.3pp**；33.3% 保留率下达 **99.7%**，近乎无损。
- **效率**（LLaVA-NeXT-7B，11.1% 保留率）：**3.62× prefill 加速**、**2.49× 端到端加速**、**6.02× KV-cache 缩减**（1156MB → 192MB）。
- **消融**：纯注意力得分 96.07%、纯 CMR 96.00%、乘性融合 96.59%；仅 Top-K 选择（无多样性）降至 90.13%（−6.46），残差空间多样性优于原始空间多样性（−0.89 vs −6.46）。

## 相关工作脉络

1. **注意力分数剪枝**（FastV/ECCV'24、SparseVLM/ICML'25、Pyramiddrop）：基于单层注意力权重剔除低重要性 token；SIEVE 的不同在于引入跨模态残差作为全局几何信号，弥补注意力分数偏置和不可靠的问题。
2. **特征相似性压缩**（Token Merging/Bolya et al. 2023、Duplication Matters/Wen et al. 2025）：合并相似视觉 token；SIEVE 不依赖 token 间相似度，而是以文本子空间可解释性为判别标准。
3. **多样性/覆盖性选择**（DivPrune/Alvar et al. 2025、SCOPE/NeurIPS'25、Beyond Attention/Zhang et al. 2025a）：关注 token 间的多样性和覆盖范围；SIEVE 在**残差空间**而非原始特征空间做多样性选择，更精准地识别真正互补的视觉信息。
4. **动态/自适应剪枝**（VisionZip/CVPR'25、HoloV/NeurIPS'25、SpecFlow/ICML'26、DART/EMNLP'25）：自适应决定保留比例或从覆盖角度选择；SIEVE 的切入点是从跨模态吸收规律出发，与上述方法正交且可互补。
5. **推理目标引导**（TRIO/Zhang et al. 2026）：基于推理目标指导 token 削减；SIEVE 更侧重于跨模态几何表征分析而非任务目标函数。

## 局限性与未来方向

- **剪枝层选择需经验设定**：SIEVE 仅在少数剪枝层执行评分，未自动学习最优剪枝层位置，可能对不同架构/任务不泛化。
- **残差空间计算复杂度为 O(N_v²·d)**：当视觉 token 极多时（如高分辨率图像、视频），二次复杂度可能成为瓶颈。
- **主要在静态图像基准上评估**：论文未充分验证在视频理解或多图推理场景下的表现。
- **CMA 分析主要在 LLaVA-1.5-7B 上完成**：跨模态吸收规律在不同架构（如 Qwen2.5-VL 使用 Naive Patch Merger）中是否一致仍需验证。
- **正则化超参数 η 虽鲁棒但需设定**：虽然实验表明性能对 η 不敏感，但理论上最优值可能因任务而异。

## 研究启发与可借鉴点

1. **跨模态吸收的几何视角可迁移**：CMA/CMR 的分析框架可用于其他多模态模型（如音频-语言、视频-语言模型），探索跨模态信息流动的普遍规律。
2. **残差空间多样性选择的设计思路值得复用**：先去除共享子空间成分（文本可解释部分），再在残差上做多样性筛选，这一"去共享→选互补"的两阶段策略可推广至多模态检索、多源信息融合等场景。
3. **Tikhonov 正则化子空间投影的计算范式**：闭式解避免了迭代优化，仅增加微量开销，可作为通用的"某 modal 对另一 modal 可解释性度量"工具。
4. **Dual-flash 注意力统计策略**：通过辅助 value 矩阵在不物化注意力矩阵的前提下提取特定 attention 统计量，可应用于其他需要 attention 信号但受限于显存的 token 管理方法。
5. **乘性融合 vs 加性融合的启示**：两个互补信号采用乘性融合能自然抑制单一信号异常高但另一信号接近零的背景 token，这一设计原则可用于多信号 token 评分的融合策略。

## 关键术语表

**CMA（Cross Modal Absorption）**：跨模态吸收，层级别度量，量化某层视觉 token 表征中可由文本子空间解释的比例。

**CMR（Cross Modal Residual）**：跨模态残差，token 级别度量，通过 Tikhonov 正则化子空间投影计算每个视觉 token 中无法被文本子空间解释的残差比例。

**SIEVE**：本文提出的无需训练的视觉 token 压缩方法，联合 CMR、文本注意力相关性得分与残差空间多样性选择进行 token 筛选。

**Tikhonov 正则化最小二乘**：在求解视觉 token 到文本子空间的线性组合系数时引入 L2 正则项，防止低能量方向导致的数值不稳定，可得闭式解。

**Dual-flash 注意力统计**：通过构造辅助 value 矩阵并执行第二次 FlashAttention 前向传播，在不物化完整注意力矩阵的前提下高效聚合文本到视觉的注意力得分。

**残差空间多样性选择**：在去除文本子空间可解释成分后的残差空间中，以广义余弦相似度度量 token 间冗余度，通过贪心迭代选择信息互补的 token 子集。

**Energy Ratio（η）**：用于确定 Tikhonov 正则化系数 λ 的能量阈值，控制文本子空间中纳入的主要方向数量。

**Top-K Head 筛选**：仅保留对视觉区域注意力质量最高的若干 attention head 参与文本相关性评分，以降低噪声 head 的干扰。

## 可复现要素

- **数据集**：GQA、MMBench、MMBench-CN、MME、POPE、SQA、VQAv2、TextVQA（均为公开基准）。
- **代码/权重**：论文未明确声明开源，但 arXiv 论文通常附带代码仓库链接（需进一步确认）。模型权重为公开的 LLaVA-1.5-7B、LLaVA-NeXT-7B、Qwen2.5-VL-7B。
- **关键超参**：能量比 η（实验扫描 0.65/0.75/0.85/0.95，最优 η=0.75）、top attention head 数 H_top（实验扫描 8/12/16/24/32，最优 H_top=12）、剪枝保留率（11.1%/22.2%/33.3%）、正则化稳定项 ε（论文未给出具体值）。
- **硬件环境**：论文未详细说明。
- **输入分辨率**：Qwen2.5-VL 实验固定为 1008×1008（1296 tokens/图）。
