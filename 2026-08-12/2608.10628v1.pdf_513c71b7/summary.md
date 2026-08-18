---
title: "InSight-doc: Agentic Visual Perception for Long-Document Understanding"
source: https://arxiv.org/pdf/2608.10628v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:10:24"
field: "多模态长文档理解"
keywords: ["长文档理解", "多模态大语言模型", "视觉搜索", "Agent", "粗到细推理", "上下文腐烂"]
innovations: ["提出无检索器、区域级放大的端到端agent框架InSight-doc", "构建17.9K SFT+19.2K RL的主动感知训练语料", "将视觉分辨率作为自适应推理时资源，显著降低延迟与幻觉"]
benchmarks: ["DUDE", "MP-DocVQA", "MMLongBench-Doc", "LongDocURL", "MME-RealWorld-Lite", "O3-Bench"]
---

# 论文速读：InSight-doc: Agentic Visual Perception for Long-Document Understanding

## 一句话总结
InSight-doc 提出一种端到端、无检索器的 agent 视觉感知框架，将视觉分辨率作为自适应推理时资源，通过"低分辨率概览+选择性区域放大"的粗到细工作流，在长文档 VQA 任务上显著降低幻觉、序列长度与推理延迟，同时提升准确率。

## 研究问题与动机
- **长文档推理成本高**：MLLM 采用注意力机制处理所有 N 个 token，导致 O(N²) 时间复杂度和 O(N) 空间开销，难以扩展至超长文档。
- **上下文腐烂（Context Rot）**：随着 prompt 变长，模型性能急剧下降，归因于稀释的注意力与缺乏真实长上下文训练数据。
- **高分辨率与效率的矛盾**：保留精细视觉信息需高分辨率编码，但会耗尽上下文预算；激进下采样虽节省 token 却可能丢失关键证据。
- **现有方法的局限**：端到端方法计算开销大；视觉检索方法依赖外部检索器，检索错误难以恢复；粗到细方法多停留在页面级，无法实现子页面区域级精确定位。

## 核心贡献（创新点）
1. **提出 InSight-doc 框架**：首个端到端、无检索器、支持区域级放大的 agent 框架，在推理过程中动态获取多尺度视觉证据，而非被动处理固定分辨率文档。
2. **构建主动感知训练语料**：构造 17.9K 高质量 SFT 示例（含区域级放大轨迹）与 19.2K 困难 RL 示例，覆盖可答与不可答问题。
3. **理论分析与实验验证**：提供序列长度与推理延迟的理论上界分析，在多个基准上验证了准确性-效率帕累托前沿的显著提升。
4. **与 Doc-V* 的本质区别**：本文方法完全无外部检索器依赖（94%-99.8% 轨迹由模型自主完成区域级证据获取），而对比方法高度依赖 ColQwen2.5 等检索模型。

## 方法详解
- **初始视觉上下文**：将文档各页图像以 resize 因子 r（r ≤ 1）下采样，初始化视觉上下文空间 $\mathcal{I}_{ctx}^{(0)} = \{\tilde{I}_k^{(0)}\}_{k=1}^N$。
- **放大工具调用**：在步骤 t，模型发出工具调用 zoom_in(k, d, b | I_{ctx}^{(t-1)})，其中 k 为图像索引，d 为自然语言区域描述，b 为边界框。
- **区域裁剪与调整**：从高分辨率源 I_{s(k)} 裁剪区域得 I_crop^{(t)} = crop(I_{s(k)}, b, r)，再按公式 $\tilde{I}_{crop}^{(t)} = resize(I_{crop}^{(t)}, c \cdot r \cdot size(I_{crop}^{(t)}))$ 缩放，其中 c > 1 为放大因子，递归更新 r ← c·r。
- **迭代推理循环**：将裁剪图像追加至上下文 $\mathcal{I}_{ctx}^{(t)} = \mathcal{I}_{ctx}^{(t-1)} \cup \{\tilde{I}_{crop}^{(t)}\}$，重复至模型决定回答或达到工具调用上限（10次）。
- **数据构建流水线**：三阶段过滤（prior-only、zoom-free、zoom-in CoT 构建），使用 InSight-o3 双智能体（vReasoner + vSearcher）生成轨迹，合并为单智能体平坦序列作为 SFT 目标。
- **RL 训练**：采用 GRPO 算法，仅使用二元准确率奖励，加权采样策略平衡可答/不可答样本比例（86%/14%）。

## 实验与结果
- **数据集**：DUDE（5.7页）、MP-DocVQA（7.0页）、MMLongBench-Doc（49.4页）、LongDocURL（85.6页）、MME-RealWorld-Lite、O3-Bench。
- **评估设置**：初始分辨率 r ∈ {0.25, 0.35, 0.5, 0.7}（对应 DPI 50/70/100/140），文档超过模型上下文限制时递归降采样。
- **主要结果**：
  - InSight-doc-8B (SFT+RL) 在 r=0.25 时平均准确率 66.9%，较 Qwen3-VL-8B 提升 16.4 点；r=0.5 时平均 72.6%，提升 4.3 点。
  - 在 MMLongBench-Doc 和 LongDocURL 上分别达 57.8% 和 65.6%，超越先前最强结果 15.7 和 9.3 点。
  - 相比 Qwen3-VL-8B (r=0.7)，InSight-doc (r=0.35) 在最长文档子集上准确率高出 3.0 点，延迟降低 71%（11.2s vs 39.3s）。
- **幻觉抑制**：在不可答题上，InSight-doc-8B (r=0.25) F1 达 69.1 (DUDE) 和 74.4 (MMLongBench-Doc)，较基线提升 24.6/25.9 点。
- **轨迹质量**：RL 训练后证据盒覆盖率提升至 82.3%，冗余轨迹从 14.1% 降至 5.8%，卡住轨迹从 9.7% 降至 0.1%。

## 相关工作脉络
- **端到端方法**（Qwen3-VL、InternVL3）：直接编码所有高分辨率页面，本文方法通过选择性放大降低 token 开销。
- **视觉检索增强方法**（VDocRAG、ColPali）：依赖外部检索器选取 top-k 页面，本文完全无检索器，模型自主决定区域级证据获取。
- **粗到细方法**（CogDoc、Doc-V*）：从低分辨率概览识别相关页面后切换高分辨率，本文扩展到子页面区域级定位，支持多跳推理。
- **视觉搜索方法**（DeepEyes、Mini-o3）：在单张自然图像上学习缩放操作，本文将其扩展至多页面文档场景。
- **长视频理解**（LongVideo-R1、ProVCA）：沿时间轴选择关键帧，本文在空间维度选择子页面区域，两者均体现"选择性获取证据"思想。

## 局限性与未来方向
- 仅在 Qwen3-VL-8B-Instruct 上验证，未测试其他模型架构与规模。
- 仅使用二元准确率奖励，未探索更复杂的奖励设计或高级 RL 方法。
- 未对极端长文档（>150页）进行充分评估。
- 未来可探索多模态奖励信号、跨模型泛化、以及更长的推理步数。

## 研究启发与可借鉴点
- **"分辨率即推理资源"理念**：将视觉分辨率视为可自适应分配的推理时资源，为多模态 agent 设计提供新视角。
- **数据构建策略**：三阶段难度过滤（prior-only → zoom-free → zoom-in）可有效筛选训练样本，避免"太简单"或"太困难"的数据。
- **SFT+RL 联合训练**：SFT 提供初始工具使用能力，RL 进一步优化策略质量，两者结合效果显著优于单一训练阶段。
- **不可答样本处理**：专门构建不可答问题并教授模型正确拒绝，对实际部署至关重要。
- **理论分析结合实际实验**：提供序列长度与延迟的上界分析，增强方法的可解释性与可信度。

## 关键术语表
- **InSight-doc**：本文提出的端到端、无检索器的 agent 视觉感知框架。
- **Context Rot（上下文腐烂）**：长上下文导致模型性能下降的现象。
- **GRPO（Group Relative Policy Optimization）**：用于本模型 RL 训练的强化学习算法。
- **vReasoner / vSearcher**：InSight-o3 中两个解耦的智能体，分别负责推理状态维护和区域定位。
- **Resize factor (r)**：初始图像下采样比例，控制输入视觉 token 数量。
- **Evidence-box coverage**：裁剪区域覆盖标注证据盒的比例，衡量定位精度。
- **Coarse-to-fine**：从粗到细的多尺度推理策略。
- **Agent trajectory**：模型在多轮交互中产生的工具调用序列及中间观察结果。

## 可复现要素
- **代码**：已开源，见 https://github.com/m-Just/InSight-doc
- **数据集**：已公开于 Hugging Face
- **模型权重**：已发布
- **基础模型**：Qwen3-VL-8B-Instruct
- **关键超参**：SFT 学习率 5×10⁻⁶，batch size 32，max sequence length 65,536；RL 学习率 1×10⁻⁶，rollouts per prompt 8，max prompt length 24,576，tool-use limit 10 次
- **训练框架**：verl
