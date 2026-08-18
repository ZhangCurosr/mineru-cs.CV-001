---
title: "VERDICT: Training-Free Step-Wise Verification of Multimodal Reasoning via Disagreement-Aware Consensus"
source: https://arxiv.org/pdf/2608.10665v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-08-18 06:51:41"
---

# 论文速读：VERDICT: Training-Free Step-Wise Verification of Multimodal Reasoning via Disagreement-Aware Consensus

## 一句话总结
提出 VERDICT，一种免训练的逐步验证框架，通过视觉、逻辑、上下文三个专业代理的跨模态分歧感知共识机制，在推理步骤级别动态拦截并替换错误步骤，显著提升了多模态大模型在复杂空间与科学图表推理任务上的准确率与跨任务稳定性。

## 研究问题与动机
- 多模态大模型（MLLMs）生成链式思维（CoT）时易出现视觉幻觉或逻辑跳跃，现有测试时缩放（TTS）方法多在完整响应级别操作（如多数投票、自我评估），无法精准定位并修正中间推理步骤的错误。
- 单视角验证信号难以捕捉多模态推理中视觉接地、形式逻辑与任务语境之间的复杂张力，导致部分 TTS 方法在特定基准上出现性能退化（5 个对比方法中 4 个平均低于未验证 base）。
- 现有步骤级验证严重依赖训练好的过程奖励模型（PRM），成本高且泛化性差，缺乏一种无需额外训练即可适配不同推理模式的通用验证架构。

## 核心贡献（创新点）
- **提出三代理跨模态分工验证架构**：设计 Visual（V）、Logical（L）、Contextual（C）三个独立代理分别评估视觉可验证性、逻辑连贯性与任务相关性，通过异质权重 λᵢ 隐式适配不同推理模式。*区别于单视角 self-critique 或训练型 PRM，本方法以功能正交的代理分工替代单一打分器，实现更细粒度的错误拦截。*
- **定义分歧感知共识的精确不动点**：将多代理评分建模为耦合系统的精确不动点，而非启发式近似；共识缺失即判定步骤需替换。*本质区别于 Self-Consistency 等纯多数投票策略，引入跨模态张力作为验证信号的理论严谨性。*
- **引入分散度 Δ* 与双安全网机制**：Δ* 捕获高置信度但跨模态未对齐的候选，配合 ϵ 容忍度与 τ 置信度阈值形成 fallback ranking 双保险。*弥补了传统 TTS 缺乏不确定性度量与动态容错能力的空白，避免过度过滤或漏过滤。*
- **训练免依赖的全基准正向提升**：在 6 个多模态推理基准上平均达 70.15%，较最强 TTS（Self-Synthesizer 67.11%）提升 +2.71 点，且全部基准无退化。*相比多数 TTS 在部分任务上出现负向增益的现象，展现出更强的跨任务一致性与视觉接地强化能力。*

## 方法详解
- **基础生成模块**：Base 模型采用 Qwen2.5-VL-7B-Instruct，通过 chain-of-thought prompt 生成候选推理步骤，采样参数 temperature=0.8，top-p=0.6，每步最多 1000 token。答案提取由 Gemma-3:12B（via Ollama）独立完成，**不参与验证管线**，实现生成与验证解耦。
- **三代理验证架构**：三个代理均基于 Qwen2.5-VL-7B-Instruct，独立输出 [0,1] 分数：
  - **Visual Agent (V)**：评估步骤中对象识别、空间关系、视觉特征是否可从图像中直接验证。
  - **Logical Agent (L)**：评估推理是否严格从先前步骤逻辑推出，是否有效推进问题解答。
  - **Contextual Agent (C)**：评估步骤是否紧扣原题意图，剔除无关信息或猜测。
- **分歧感知共识（Disagreement-Aware Consensus）**：通过异质 λᵢ 使系统隐式适应不同推理模式，无需显式步骤分类。共识被形式化为耦合评分系统的精确不动点；被拒绝的步骤是验证者在共识压力下无法达成稳定一致的情况。
- **步骤级纠错机制**：当某步骤未通过共识（分歧超出阈值），系统在该步骤级别进行拦截与替换，保留已验证的前序步骤，而非丢弃整条推理轨迹。
- **双安全网与超参控制**：设置 ϵ（分散度容忍度）与 τ（置信度阈值）。ϵ 过小则 fallback ranking 兜底，ϵ 过大则 τ 继续过滤，形成双层容错。默认 ϵ=0.1（保守端），τ=0.6。

## 实验与结果
- **数据集**：6 个公开基准（3DSRBench 2,772 对、CV-Bench 2,638 例、BLINK 3,807 题、MMStar 1,500 样本、AI2D 4,817 张、CV-Bench-3D 深度排序子集），覆盖空间推理、科学示意图、视觉问答等任务。
- **评估基线**：Base Model、Self-Consistency (ICLR'23)、Self-Selector (ICML'24)、Self-Refinement、Self-Synthesizer (ICML'25)、TTAug (ICLR'26)。
- **主要结果（Table 4）**：
  | Method | 3DSR | CV-3D | CV-2D | BLINK | MMStar | AI2D | Avg. |
  |---|---|---|---|---|---|---|---|
  | Base Model | 56.12 | 76.39 | 74.27 | 48.31 | 61.25 | 81.52 | **66.31** |
  | Self-Consistency | 57.08 | 78.12 |
