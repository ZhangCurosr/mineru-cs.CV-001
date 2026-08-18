---
title: "InSight-doc: Agentic Visual Perception for Long-Document Understanding"
source: https://arxiv.org/pdf/2608.10628v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:10:41"
---

# 论文速读：InSight-doc: Agentic Visual Perception for Long-Document Understanding

## 一句话总结
InSight-doc 提出了一种端到端、无需外部检索器的智能体视觉感知框架，将图像分辨率视为推理时可动态分配的自适应资源；模型从低分辨率文档概览出发，在多轮思维链中按需调用 `zoom_in` 工具局部放大获取高分辨率证据。通过构建 17.9K SFT 轨迹与 19.2K 难样本强化学习数据的主动感知语料，结合 SFT+RL（GRPO）训练，InSight-doc-8B 在长文档 VQA 上显著提升准确率的同时，将幻觉率降低逾 40%，推理延迟减少 41%–68%。

## 研究问题与动机
- 长文档理解通常需将高分辨率页面直接输入 MLLM，导致视觉 token 暴增，引发高昂的计算开销与 KV cache 压力，并加剧“上下文腐烂”（context rot）现象。
- 传统多阶段文档解析管线（布局检测→OCR→阅读顺序重建）级联误差严重，且难以泛化到表格、图表混排的复杂视觉丰富型文档。
- 现有视觉检索增强（RAG）方法依赖外部 embedding 检索器选取 top-k 页面，检索与生成解耦，检索误差难以在后续推理中纠正，且引入额外索引与 API 延迟。
- 粗到细（coarse-to-fine）方案多停留在页面级别的全页拉取，缺乏子页面区域级（region-level）的动态精细化定位能力，易将无关页面内容也纳入上下文。

## 核心贡献（创新点）
1. **端到端无检索器的自适应分辨率智能体框架**：将视觉分辨率作为推理时的动态资源，模型在推理循环中自主决策何时、何处放大，与固定分辨率端到端输入或依赖外部检索器的策略本质不同。
2. **区域级多跳证据获取机制**：通过统一的 `zoom_in(k, d, b)` 工具调用实现子页面级裁剪与高分辨率重建，支持跨页多跳查询与多轮自我修正，区别于 Doc-V* 等仅支持页面级拉取的方法。
3. **三阶段过滤与双智能体轨迹合成的主动感知语料**：构建包含 17.9K SFT 与 19.2K 难样本 RL 的高质量数据集，兼顾可回答与不可回答样本，填补了长文档多步 zoom-in 训练数据的空白。
4. **SFT+RL 联合优化突破精度-效率帕累托前沿**：在仅 8B 参数的前提下，以极低初始分辨率（r=0.25）超越 GPT 系列闭源模型，同时大幅压降幻觉率与推理延迟。

## 方法详解
- **多轮视觉上下文构建**：文档页面以低分辨率因子 $r \leq 1$ 下采样后作为初始视觉上下文 $\mathcal{I}_{\text{ctx}}^{(0)}$。模型每轮输出 CoT（`<think>`）与工具调用 `zoom_in(k, d, b)`，其中 $k$ 为当前上下文图片索引，$d$ 为区域描述，$b$ 为边界框。
- **高分辨率裁剪与递归缩放**：若调用合法，系统从原始高分辨率源图 $I_{s(k)}$ 按 $b$ 裁剪得到 $I_{\text{crop}}^{(t)}$，再按缩放因子 $c>1$ resize 后追加至上下文：$\tilde{I}_{\text{crop}}^{(t)} = \text{resize}(I_{\text{crop}}^{(t)}, c \cdot r \cdot \text{size}(\cdot))$，递归更新 $r \gets c \cdot r$，直至模型回答或达到工具上限。
- **延迟与序列长度理论界**：推导相对序列长度上界 $S_r/S_0 \leq x(r) + \kappa^{-1}y(r)$ 与相对延迟上界 $T_r/T_0 \leq w_p x(r)^2 + w_c x(r)y(r) + w_g y(r)^2$，证明在典型长文档 VQA 参数下，序列长度可降至基线的 7.8%–45.0%，延迟上界可压至约 32.5%。
- **训练流程**：基座模型为 Qwen3-VL-8B-Instruct。第一阶段全参 SFT（17,913 条轨迹，2 epochs，lr=5e-6，cosine decay，warmup=0.05，max seq len=65536）；第二阶段使用 GRPO 进行强化学习，仅设二元准确率奖励，800 步，lr=1e-6，kl coeff=0.01，max prompt=24576，max response=8192，tool limit=10，视觉编码器冻结。

## 实验与结果
- **评测基准**：标准文档 VQA（DUDE, MP-DocVQA）、长文档 VQA（MMLongBench-Doc, LongDocURL）、通用高分辨率 VQA（MME-RealWorld-Lite, O3-Bench）。PDF 统一光栅化为 200 DPI，测试时按 $r \in \{0.25, 0.35, 0.5, 0.7\}$ 下采样输入，judge 为 GPT-5-nano。
- **主要准确率结果**：在 $r=0.25$（50 DPI）下，InSight-doc-8B (SFT+RL) 平均准确率达 66.9%，较 Qwen3-VL-8B 基线提升 +16.4
