---
title: "When Vision Becomes Text: Visual Token Pruning via Cross-Modal Residual Guidance in VLMs"
source: https://arxiv.org/pdf/2608.10489v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:32:04"
---

# 论文速读：When Vision Becomes Text: Visual Token Pruning via Cross-Modal Residual Guidance in VLMs

## 一句话总结
本文提出 **SIEVE**，一种免训练的视觉 Token 压缩方法，首次从“跨模态渐进吸收”视角重新审视 VLM 推理过程，通过 **Cross-Modal Residual (CMR)** 量化无法被文本子空间解释的独特视觉信息，并结合文本注意力相关性与残差空间多样性选择，在大幅削减视觉 Token 数量的同时保持高推理性能与显著加速。

## 研究问题与动机
- **问题**：VLM 将高分辨率图像编码为海量视觉 Token 送入 LLM，导致自注意力计算、KV-cache 与推理延迟急剧上升。
- **现有方法不足**：主流视觉 Token 剪枝依赖单层注意力权重或特征相似度，仅捕获局部层状态，缺乏对多模态联合推理全生命周期的全局视角；相似度导向易忽略与当前文本上下文相关的 Token，而纯注意力导向又难以区分冗余视觉区域。
- **核心观察**：随着 LLM 层数加深，文本 Token 通过因果自注意力持续聚合视觉信息，视觉表示逐渐被投影到文本子空间内（即“视觉向文本转化”），深层 Token 中存在大量可由文本子空间线性解释的成分，具备冗余剪枝潜力。

## 核心贡献（创新点）
1. **提出跨模态吸收（CMA）与跨模态残差（CMR）度量**：从几何表示视角量化视觉信息被文本子空间渐进吸收的程度；与已有工作本质区别在于，跳出单层相似度/注意力框架，将 Token 重要性重新定义为“无法被当前文本语境解释的残余信息量”。
2. **提出 SIEVE 免训练压缩框架**：联合 CMR（视觉独特性）、文本注意力相关性（任务相关性）与残差空间多样性选择（互补性）进行多信号融合筛选；与已有方法本质区别在于首次在残差空间而非原始特征空间执行多样性去重，更精准地保留真正互补的视觉证据。
3. **工程友好的 FlashAttention 兼容设计**：通过双 Flash 聚合策略在不物化完整注意力矩阵的前提下高效提取 text-to-visual 注意力统计，使方法可直接嵌入现有推理引擎；与已有方法本质区别在于兼顾理论有效性与生产级推理兼容性。
4. **跨架构广泛验证**：在 LLaVA-1.5、LLaVA-NeXT、Qwen2.5-VL 三种架构与八个基准上系统性验证，证明方法对模型结构与分辨率差异具有强泛化性。

## 方法详解
- **跨模态吸收（CMA，层级别）**：  
  定义第 $l$ 层视觉 Token 矩阵 $V_c$ 与其在文本中心化子空间上的投影 $\hat{V}_c$ 之间的 Frobenius 范数比：  
  $\mathrm{CMA} = 1 - \frac{\|V_c - \hat{V}_c\|_F^2}{\|V_c\|_F^2}$。CMA 随层深单调递增（Spearman $\rho=0.99$），证实视觉信息在深度网络中被文本子空间逐步吸收。
- **跨模态残差（CMR，Token 级别）**：  
  对每个中心化视觉 Token $v_{c,i}$，通过 Tikhonov 正则化最小二乘投影到文本子空间：  
  $b_i = \arg\min \|v_{c,i} - b_i T_c\|_2^2 + \lambda\|b_i\|_2^2$，其中正则化系数 $\lambda = \sigma_k^2 + \epsilon$，由文本 Gram 矩阵 $G=T_cT_c^\top$ 的特征值按能量阈值 $\eta$ 截断自适应确定。  
  重构残差归一化得分：$\mathrm{CMR}_i = \frac{\|v_{c,i} - \hat{v}_{c,i}\|_2}{\|v_{c,i}\|_2}$。CMR 越高表示该 Token 携带越多的文本无法解释的独特视觉信息。
- **注意力相关信号**：  
  仅保留视觉注意力质量最高的前 $\rho_h$ 比例 Attention Head，聚合 post-image 文本 Token 对该视觉 Token 的注意力权重之和 $a_i = \sum_{h \in \mathcal{H}_{\mathrm{top}}} \sum_{t \in \mathcal{T}} \alpha_{t,i}^h$。为兼容 FlashAttention，采用辅助值矩阵 $\widetilde{V}$ 的 dual-flash 聚合策略避免显式构造 $N\times N$ 注意力矩阵。
- **引导得分融合**：  
  $\mathrm{Score}_i = a_i^2 \cdot \mathrm{CMR}_i^2$。乘法融合促使选出的 Token 同时具备高任务相关性与高几何独特性，避免单一信号主导或背景噪声干扰（消融显示乘法优于加法）。
- **残差空间多样性贪心选择**：  
  残差向量 $r_i = v_{c,i} - \hat{v}_{c,i}$。冗余度量：$g_{ij} = \left(\frac{r_i^\top r_j}{\|r_i\|_2 \|r_j\|_2}\right)^2$。  
  迭代过程：选取 $\mathrm{Score}$ 最高 Token $i^*$ 加入集合 $\mathcal{S}$，并对剩余候选执行折扣 $\mathrm{Score}_j \leftarrow \mathrm{Score}_j(1 - g_{i^*j})$，直至达到预算 $K$。残差空间去重能更准确识别互补信息而非表面特征相似的冗余 Token。

## 实验与结果
- **实验设置**：模型 LLaVA-1.5-7B、LLaVA-NeXT-7B、Qwen2.5-VL-7B；基准 GQA、MMBench、MMBench-CN、MME、POPE、SQA、VQAv2、TextVQA；基线 FastV、SparseVLM、DART、VisionZip、HoloV、SCOPE、SpecFlow。
- **主要结果**：
  - **LLaVA-1.5-7B**：保留 64 Token（11.1%）时平均性能达 Vanilla 的 **96.0%**，超越最强完整基线 SCOPE 0.6pp。
  - **LLaVA-NeXT-7B**：保留 320 Token 时平均性能达 Vanilla 的 **97.5%**，较 DART 与 HoloV 分别提升 1.4pp 与 1.3pp。
  - **Qwen2.5-VL-7B**：保留 11.1% Token 时平均性能达 Vanilla 的 **96.7%**，较 FastV 与 HoloV 分别提升 9.2pp 与 2.3pp，体现强抗压缩能力。
- **效率收益**（LLaVA-NeXT-7B）：Prefill 加速 **3.62×**，端到端加速 **2.49×**，KV-cache 缩减 **6.02×**（1156MB → 192MB）。
- **消融结论**：仅用 Attention 或仅用 CMR 分别达 96.07% / 96.00%，乘法融合达 96.59%；残差空间多样性选择（96.59%）显著优于原始空间多样性（95.70%）与直接 Top-K（90.13%）；超参 $\eta$ 与 $H_{\mathrm{top}}$ 在宽范围内表现稳定（均值波动 <0.6pp）。

## 相关工作脉络
- **Attention-score pruning**（FastV, SparseVLM, Pyramiddrop）：依赖单层自注意力权重剪枝，但注意力分数易受噪声干扰且无法区分“高权重但信息冗余”的 Token；本文在其基础上引入残差几何信号校正。
- **Similarity-based reduction**（Token Merging, VisionZip）：通过聚类/相似度合并视觉 Token，但未考虑 Token 与当前文本查询的动态相关性；本文强调任务相关性与视觉独特性的联合优化。
- **Diversity/Coverage selection**（SCOPE, HoloV, DivPrune）：在原始特征空间做多样性约束，共享背景分量易主导余弦相似度；本文将其迁移至残差空间，使多样性度量聚焦于真正互补的视觉信号。
- **Cross-modal flow / adaptive dropping**（Dynamic-LLaVA, TRIO）：关注推理链或目标驱动的动态裁剪；本文提供正交的几何解释视角，可与上述方法结合形成双层筛选。
- **定位差异**：现有方法多为“静态局部信号”导向，本文首次将“跨模态渐进吸收”作为全局推理过程的显式建模对象，使剪枝依据从“当前层谁重要”转变为“哪些信息已被文本吸收、哪些仍需保留”。

## 局限性与未来方向
- **计算开销边界**：残差 Gram 矩阵构建复杂度为 $O(N_v^2 d)$，在极高帧率视频或多图长序列场景下评分阶段可能成为瓶颈；未来可探索低秩近似或采样估计。
- **固定剪枝层假设**：实验主要在固定层执行一次裁剪，未充分探索多层渐进式裁剪或动态层选择的潜力；未来可研究自适应剪枝层调度。
- **仅验证 7B 规模**：未在更大参数规模（如 70B+）或不同视觉编码器架构（如 SigLIP、PaLI）上验证，泛化边界尚待扩展。
- **超参仍需人工标定**：虽然 $\eta$ 与 $H_{\mathrm{top}}$ 鲁棒，但最优配置依赖任务分布；未来可引入自动调参或元学习机制。

## 研究启发与可借鉴点
1. **“吸收-残差”范式可迁移**：将跨模态信息流抽象为子空间投影与残差度量，适用于任何多模态融合架构（如音频-语言、图-语言）的 Token/Head 压缩。
2. **残差空间多样性去重**：在剔除共享子空间分量后再做冗余度量，是解决“高相似但低互补”问题的有效通用技巧，可推广至视觉 Transformer 的 Patch 选择与 LLM 的 Context Pruning。
3. **乘法融合异质信号**：`Score = attn^2 × residual^2` 的二次幂乘法能天然压制单边异常高值（背景噪声或孤立高残差点），比线性加权更稳健，值得在多信号排序场景中复用。
4. **Dual-flash 统计提取设计**：通过辅助值矩阵 $\widetilde{V}$ 绕过显式注意力矩阵物化，既保留 FlashAttention 的内存效率又实现精细统计，对工程化部署具有直接参考价值。

## 关键术语表
- **CMA (Cross Modal Absorption)**：层级别指标，衡量该层视觉 Token 表示可被文本子空间线性解释的比例，值越大说明跨模态信息融合越充分。
- **CMR (Cross Modal Residual)**：Token 级别得分，通过 Tikhonov 正则化最小二乘投影计算重构残差归一化值，量化无法被文本子空间解释的独特视觉信息。
- **SIEVE**：本文提出的免训练视觉 Token 压缩方法，联合 CMR、文本注意力相关性与残差空间多样性贪心选择实现高效剪枝。
- **Residual-Space Diversity Selection**：在去除文本可解释成分后的残差向量空间中使用平方余弦相似度度量冗余，并基于引导得分进行迭代折扣贪心筛选。
- **Dual-Flash Aggregation**：构造辅助值矩阵 $\widetilde{V}$ 并通过标准 FlashAttention 前向计算提取 text-to-visual 注意力统计，避免显式物化完整 $N\times N$ 注意力矩阵。
- **Tikhonov Regularized Least Squares**：为稳定求解视觉 Token 到文本子空间的线性组合系数而引入 $\ell_2$ 正则的最小二乘方法，正则化系数 $\lambda$ 由文本子空间能量谱自适应确定。
- **Cross-Modal Absorption Trend**：观察到的现象，即随着 LLM 层数加深，因果自注意力使文本 Token 持续聚合视觉信息，导致视觉 Token 表示逐渐靠近文本 PCA 平面。

## 可复现要素
- **数据集**：GQA、MMBench、MMBench-CN、MME、POPE、SQA、VQAv2、TextVQA（均为公开基准）。
- **代码/权重**：论文未明确声明代码开源状态；模型权重为官方 released 版本（LLaVA-1.5-7B、LLaVA-NeXT-7B、Qwen2.5-VL-7B）。
- **关键超参**：能量阈值 $\eta \in \{0.65, 0.75, 0.85, 0.95\}$（默认推荐 0.75）、保留图像注意力 Head 数 $H_{\mathrm{top}} \in \{8, 12, 16, 24, 32\}$（默认推荐 12）、正则化小常数 $\epsilon$（数值稳定用）、残差去重相似度阈值 $g_{ij} \in [0,1]$（隐式通过 $(1-g_{i^*j})$ 缩放）。

<!--META
{"keywords": ["Visual Token Pruning", "VLM Efficiency", "Cross-Modal Absorption", "Training-free Compression", "
