---
title: "InSight-doc: Agentic Visual Perception for Long-Document Understanding"
source: https://arxiv.org/pdf/2608.10628v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:12:01"
---

# 论文速读：InSight-doc: Agentic Visual Perception for Long-Document Understanding

## 一句话总结
本文提出 InSight-doc，一个无需外部检索器、端到端的智能体视觉感知框架，将视觉分辨率视为推理时可自适应调度的资源；模型从低分辨率文档全景出发，在多轮图文交错推理中动态放大（zoom-in）关键区域，在 8B 规模下显著提升长文档 VQA 准确率的同时，将幻觉率降低逾 40% 并节省 41%–68% 的推理延迟。

## 研究问题与动机
1. **上下文腐烂与计算开销**：多模态大语言模型（MLLM）处理长文档时通常需直接输入高分辨率全页，导致视觉 token 暴增与 O(N²) 注意力计算，并引发 context rot（随上下文变长性能急剧下降）。
2. **现有粗到细方法的局限性**：CogDoc、Doc-V* 等虽采用低分辨率概览+高分辨率聚焦策略，但主要定位至整页级别，且 Doc-V* 高度依赖外部检索器（ColQwen2.5），检索误差难以在生成阶段修复。
3. **多跳与跨页证据聚合困难**：真实长文档证据常分散于远距离、非相邻页面，静态输入或单轮检索无法支持多跳推理与多轮自修正。
4. **缺乏端到端主动感知训练范式**：现有视觉搜索工作多集中于单张自然图像或固定尺度，针对“严格 token 预算下跨多页动态获取子区域证据”的训练数据与算法仍属空白。

## 核心贡献（创新点）
1. **提出 InSight-doc 智能体框架**：在统一的 multimodal chain-of-thought 中 interleaved 执行文本推理与 `zoom_in` 工具调用，实现子页面级（region-level）精准定位，彻底去除外部检索模块。
2. **构建大规模主动感知训练语料**：汇集 arXiv、DUDE、DocVQA 等多源文档，经三阶段级联过滤合成 17.9K 带 region-level zoom-in 轨迹的 SFT 数据与 19.2K 困难/不可答 RL 样本，并设计基于实体/数值突变的合成负样本策略。
3. **给出推理成本的理论上界分析**：推导相对序列长度与相对延迟的闭合上界公式，证明在典型长文档参数下该方法可将 token 量压缩至基线的 7.8%–45.0%，延迟上界可降至 32.5% 以内。
4. **端到端 SFT+RL 训练闭环验证**：仅使用二值准确率奖励的 GRPO 算法即显著提升证据定位质量与轨迹稳定性，消融冗余/卡死行为，使 8B 开放模型在多项基准上超越 GPT/Gemini 闭源旗舰。

## 方法详解
- **多轮交互式推理范式**：初始化视觉上下文 $\mathcal{I}_{ctx}^{(0)} = \{\tilde{I}_k^{(0)}\}$，其中 $\tilde{I}_k^{(0)} = \mathsf{resize}(I_k, r \cdot \mathsf{size}(I_k))$，$r \leq 1$ 为初始缩放因子。模型在每个时间步 $t$ 基于当前上下文生成思维链，并可选地发出工具调用 `zoom_in(k, d, b)`（$k$ 为目标图像索引，$d$ 为自然语言区域描述，$b$ 为边界框）。
- **高分辨率裁剪与上下文追加**：若调用合法，从原始高分源 $I_{s(k)}$ 裁剪得到 $I_{crop}^{(t)}$，并按公式 $\tilde{I}_{crop}^{(t)} = \mathsf{resize}(I_{crop}^{(t)}, c \cdot r \cdot \mathsf{size}(I_{crop}^{(t)}))$ 放大（$c>1$ 为 zoom factor）。裁剪图追加至上下文 $\mathcal{I}_{ctx}^{(t)} = \mathcal{I}_{ctx}^{(t-1)} \cup \{\tilde{I}_{crop}^{(t)}\}$，循环至模型输出 `<answer>` 或达到工具调用上限。
- **推理成本理论分析**：设 $P_0, R_0$ 为基线输入/输出 token 数，$n(r)$ 为 zoom-in 调用次数。推导得相对序列长度上界 $S_r/S_0 \leq x(r) + \kappa^{-1}y(r)$ 与相对延迟上界 $T_r/T_0 \leq w_p x(r)^2 + w_c x(r)y(r) + w_g y(r)^2$，其中 $x(r)=r^2+\delta n(r)$，$y(r)=1+\lambda n(r)$，$\kappa=P_0/R_0$，$(w_p,w_c,w_g)$ 为归一化权重。
- **数据合成与分流**：采用三阶段过滤：(1) Prior-only：20 DPI 下被 Qwen3-VL-8B 正确回答的丢弃；(2) Zoom-free：50/70/100 DPI 下被 Qwen3-VL-32B 无需放大即可回答的丢弃；(3) Zoom-in CoT 构建：利用 InSight-o3 双智能体（vReasoner + vSearcher）生成多跳轨迹，合并为 flat multimodal CoT 作为 SFT 目标；未达正确答案者转入 RL。不可答样本通过 GPT-5-nano 对可答种子进行实体/数值微调生成 hard negatives。
- **SFT+RL 训练流程**：SFT 阶段对 17.9K 轨迹做全参数微调；RL 阶段使用 GRPO 算法，仅以最终答案正确性为 binary reward，配合 weighted refill sampler 按 86% 可答 / 14% 不可答比例采样，KL 正则系数 0.01。

## 实验与结果
- **评估基准**：中短
