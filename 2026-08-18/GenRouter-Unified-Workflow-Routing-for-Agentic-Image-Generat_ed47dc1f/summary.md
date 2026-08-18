---
title: "GenRouter-Unified-Workflow-Routing-for-Agentic-Image-Generat"
source: https://arxiv.org/pdf/2608.16721v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:23:18"
field: "智能体图像生成"
keywords: ["Agentic Image Generation", "Workflow Routing", "Text-to-Image", "Multi-Objective Optimization", "Experience-Driven Routing", "GenCanvas"]
innovations: ["首个统一工作流路由框架GenRouter，联合路由工作流模板与生成器后端", "双记忆系统（轨迹记忆+路由记忆）支持经验蒸馏与零样本泛化", "7维任务签名与Pareto过滤实现可解释的多目标路由决策"]
benchmarks: ["WISE", "DPG-Bench", "GenEval2", "OneIG-Bench", "SpatialGenEval", "LongText-Bench"]
---

# 论文速读：GenRouter: Unified Workflow Routing for Agentic Image Generation

## 一句话总结
GenRouter 是首个针对智能体图像生成的统一工作流路由框架，通过构建标准化工作流空间 GenCanvas 并将异构提示词自适应路由至最优执行轨迹，在保持高质量的同时将计算成本降低 95% 以上、延迟降低 65%。

## 研究问题与动机
- **现有智能体图像生成系统高度碎片化**：Mind-Brush 专注外部知识检索，GEMS 专注迭代校验，各框架孤立开发、拓扑固定，难以互融互通（Figure 2 Left）。
- **"一刀切"范式导致严重计算错配（compute-mismatch）**：即便最简单的提示词也被强制投入重型推理/迭代流程，大量浪费算力和引入不可接受的延迟。
- **测试时扩展策略在多约束任务上脆弱**：纯 prompt 预扩充（BeautifulPrompt）或后改写（Idea2img）方法难以应对涉及外部知识、精确文本渲染或多步空间逻辑推理的复杂请求。
- **缺乏可统一抽象的工作流空间**：现有方法未将多样化 agentic pipeline 系统化为通用原语与模板集合，无法在同一结构化空间内实现动态路由。

## 核心贡献（创新点）
1. **GenCanvas 标准化工作流空间**：将智能体图像生成拆解为 8 个通用基础原语（$\pi_{\text{search}}, \pi_{\text{reason}}, \pi_{\text{sketch}}, \pi_{\text{verify}}$ 等）和 9 个可组合拓扑模板；与已有工作本质区别在于提供模块化、可扩展的统一基础设施，而非又一个孤立流水线。
2. **经验驱动的自我演进路由机制 GenRouter**：通过需求分析（Demand Profiling）→ 经验匹配（Memory-Guided Matching）→ Pareto 过滤三层级联决策，自适应地将提示词分配至最优 `(工作流模板, 生成器)` 计划；与 RouteT2I 等静态模型级路由的本质区别在于同时联合路由工作流拓扑和后端生成器。
3. **双记忆系统（轨迹记忆 + 路由记忆）**：轨迹记忆记录实例级执行日志用于精确匹配，路由记忆按任务签名桶聚合并蒸馏统计先验以缓解数据稀疏；冷启动阶段通过系统探索初始化，在线阶段持续更新，实现无需手动重训练的自主进化。
4. **严格的成本/延迟隔离评估协议**：将生成器自身的推理成本和延迟从路由开销中剥离，确保跨工作流比较的公平性；这一设计在同类工作中较为少见。

## 方法详解
**GenCanvas 原语层**（$\Pi = \{\pi_{\text{rewrite}}, \pi_{\text{decompose}}, \pi_{\text{search}}, \pi_{\text{reason}}, \pi_{\text{skill}}, \pi_{\text{verify}}, \pi_{\text{refine}}, \pi_{\text{sketch}}\}$）：每个原语封装原子认知/操作能力，输出 JSON 契约，解耦执行逻辑与底层模型。

**工作流模板层**（$\mathcal{W}$，4 层级共 9 个模板）：
- **语义对齐级**：`DirectGen`（直接调用生成器）、`RewriteGen`（$\pi_{\text{rewrite}} \to g$）
- **外部锚定级**：`SearchGen`（文本搜索→改写→生成）、`RefGen`（视觉参考搜索→改写→生成）
- **结构推理级**：`ReasonGen`、`SkillGen`、`SketchGen`（$\pi_{\text{sketch}} \to g$，生成 SVG/HTML/CSS 代码）
- **迭代精修级**：`VerifyGen`（decompose → Loop(g → verify → refine)）、`HybridGen`（{search, sketch} → Loop(g → verify → refine)）

**GenRouter 三层路由**：
1. **需求分析（Demand Profiling）**：轻量 LLM（Qwen3.5-4B）提取 7 维任务签名 $\mathbf{z}(x) \in \{0,1,2,3,4,5\}^7$，涵盖语义表达、事实锚定、视觉参考、逻辑推理、组合启发、评估批判、空间布局；基于签名门控剪枝候选集，例如 `HybridGen` 仅在 $\sum_{k \in \mathcal{H}} \mathbb{I}(z_k \geq 3) \geq 2$ 时激活。
2. **经验匹配（Memory-Guided Matching）**：融合轨迹记忆 $\mathcal{M}_{\text{traj}}$（实例级精确匹配，top-k 相似历史）和路由记忆 $\mathcal{M}_{\text{route}}$（桶级聚合统计），以置信度 $\alpha_p$ 加权：$\hat{S}_p = \alpha_p \hat{S}_{\text{traj}}(p) + (1-\alpha_p)\hat{S}_{\text{route}}(p)$；冷启动时退化为确定性阈值先验。
3. **Pareto 过滤（Pareto Filtering）**：计划 $p_a$ 支配 $p_b$ 当且仅当 $\hat{S}_{p_a} \geq \hat{S}_{p_b}$ 且 $\hat{C}_{p_a} \leq \hat{C}_{p_b}$ 且 $\hat{L}_{p_a} \leq \hat{L}_{p_b}$，至少一处严格优于；在非支配 Pareto 集 $\mathcal{P}_{\text{pareto}}$ 中最大化标量效用：$\hat{p} = \arg\max_{p \in \mathcal{P}_{\text{pareto}}} \hat{U}_p$，其中 $\hat{U}_p = \hat{S}_p - \lambda_c \hat{C}_p - \lambda_l \hat{L}_p$（$\lambda_c=5.0, \lambda_l=0.0006$）。

## 实验与结果
**数据集/基线**：WISE、DPG-Bench、GenEval2、OneIG-Bench（EN/CN）、LongText、SpatialGenEval、ArtiMuse；基线包括 Qwen-Image/Z-Image 原生模型、Mind-Brush、SCOPE、GEMS。

**主要结果**（平均 5 项基准）：
- 与 GEMS 相比，GenCanvas + GenRouter **性能最高（71.3% vs 68.7%）**，成本降低 **>95%**（$2.97 vs $59.70），延迟降低 **65%**（4.68h vs 13.62h）。（Table 3）
- 在 WISE 上达到 0.88（与 SCOPE 持平），但成本仅为 $1.57（SCOPE 为 $1.97）；在 GenEval2 上达到 71.6（超越 GEMS 的 70.4），成本大幅降低（$2.03 vs $40.43）。（Table 2）
- 混合测试集上：`HybridGen` 单一模板性能 73.53 但成本 $6.27/延迟 8.78h；GenRouter 达到同等性能 73.52 但成本仅 $1.37/延迟 1.76h，位于最优 Pareto 前沿。（Figure 5 Middle）

**演进与泛化**：经历三个基准的经验累积后，未见混合集性能从 73.5 提升至 75.2，成本降 8.7%，延迟降 7.9%；零样本从 WISE 迁移至 DPG-Bench 时，冻结经验下即达 87.1 分，成本仅为 LLM-as-Router 的一半（$1.51 vs $2.40）。（Figure 5 Right, Figure 6）

## 相关工作脉络
1. **GEMS [15] / SCOPE [22]**：静态多步骤智能体工作流，分别采用迭代校验和结构化技能编排；本文将其映射为 GenCanvas 中的 `VerifyGen` 和 `HybridGen` 模板，并主张动态路由替代固定拓扑。
2. **Mind-Brush [14] / Gen-Searcher [9]**：专注外部知识检索的原生工作流；本文将其抽象为 `SearchGen`/`RefGen` 模板，并在统一空间内与其他模板协同路由。
3. **GenClaw [38]**：通过可执行视觉代码控制空间布局；本文抽象为 `SketchGen` 模板（$\pi_{\text{sketch}} \to g$），纳入统一模板库。
4. **RouteT2I [34]**：在边缘/云端生成器之间静态调度请求；本文强调自身是首个**工作流级**路由框架，联合路由工作流拓扑+生成器后端，超越单纯的模型选择。
5. **Evoroute [42] / PORT [31]**：经验驱动的语言模型路由系统；本文延续经验路由范式，但首次将其扩展至智能体图像生成领域的工作流-生成器联合空间。
6. **GenEval / DPG-Bench / WISE**：评测基准；本文在这些基准上建立统一比较标准，同时提出成本/延迟隔离评估协议以解决跨工作流公平比较难题。

## 局限性与未来方向
- **冷启动依赖少量校准样本**：仅 10 个校准提示词进行穷举探索（Appendix B.1），对长尾或极复杂任务的先验可能不足。
- **7 维任务签名可能不够全面**：签名空间为人工设计的固定维度，未能覆盖所有潜在生成需求；极端 prompt 可能被误分类。
- **原语后端性能瓶颈**：`reason`/`verify` 等原语统一实例化为 Kimi K2.5，若后端模型本身存在缺陷将限制整体上限。
- **未来方向**：可扩展至视频/3D 生成领域；引入强化学习替代启发式路由策略；自动化扩展原语库与模板集；支持实时流式生成场景。

## 研究启发与可借鉴点
1. **任务签名（Task Signature）的可迁移设计**：7 维正交需求分解思路可用于任何需要多约束决策的 AI Agent 系统，为"需求分析→能力匹配"提供可复用的形式化范式。
2. **双记忆蒸馏架构**：轨迹记忆（实例级精确）+ 路由记忆（桶级聚合）的互补设计，有效兼顾精度与鲁棒性，可推广至多 Agent 协作、模型选择等经验驱动场景。
3. **Pareto 过滤替代标量加权**：在成本-质量-延迟多目标优化中，先用 Pareto 支配剔除劣解再标量化排序，避免线性加权对异常值的敏感，该方法论可迁移至多目标路由问题。
4. **工作流拓扑空间有界化**：无约束原语路由导致执行不稳定（性能 65.0 vs 73.52，Table D.1），证明**预定义拓扑模板**对智能体系统的稳定性至关重要；这一发现对 Agent 框架设计有普遍指导意义。
5. **成本/延迟隔离评估协议**：将基础设施开销与模型推理开销剥离，为公平评测路由类系统提供了可复用的评估规范。

## 关键术语表
- **GenCanvas**：将智能体图像生成拆解为 8 个通用原语和 9 个拓扑模板的统一工作流抽象空间与开源代码库。
- **GenRouter**：基于经验驱动的工作流路由器，通过需求分析→经验匹配→Pareto 过滤三层机制为每个提示词动态选择最优 `(工作流, 生成器)` 计划。
- **任务签名（Task Signature）$\mathbf{z}(x)$**：7 维向量，量化提示词在语义表达、事实锚定、视觉参考、逻辑推理、组合启发、评估批判、空间布局七个维度的需求强度（0-5 分）。
- **双记忆系统**：轨迹记忆 $\mathcal{M}_{\text{traj}}$（实例级执行日志）与路由记忆 $\mathcal{M}_{\text{route}}$（桶级聚合统计）的组合，支持精确匹配与稀疏缓解。
- **基础原语（Foundational Primitive）**：GenCanvas 中定义的原子操作算子，包括 search、reason、sketch、verify、refine、decompose、skill、rewrite 共 8 种。
- **Pareto 支配**：计划 $p_a$ 在所有维度不劣于 $p_b$ 且至少一项严格优于，则 $p_a$ 支配 $p_b$；非支配集合构成 Pareto 前沿。
- **compute-mismatch**："一刀切"工作流范式导致的简单提示被强制投入高成本流程的资源浪费现象。
- **PrimitiveTrace**：每个原语执行产生的标准化日志，包含后端模型、token 消耗、货币成本与延迟，用于统一会计与经验蒸馏。

## 可复现要素
- **数据集**：WISE、DPG-Bench、GenEval2、OneIG-Bench、LongText、SpatialGenEval、ArtiMuse（均为公开基准）；混合测试集由论文构造（9 个基准共 500 条，权重见 Table 4）。
- **代码**：开源，GitHub https://github.com/EnVision-Research/GenRouter。
- **权重**：生成器后端使用 Qwen-Image-2512 和 Z-Image-Turbo（公开模型）；需求分析器使用 Qwen3.5-4B（本地运行）。
- **关键超参**：$\lambda_c = 5.0$、$\lambda_l = 0.0006$；冷启动探索 prompt 数 $n=10$；每 50 次执行批量评估一次；热工作流激活阈值：$\sum_{k \in \mathcal{H}} \mathbb{I}(z_k \geq 3) \geq 2$。
- **原语后端**：通用语言/视觉原语（reason、verify 等）统一使用 Kimi K2.5；外部搜索使用 Serper Search。
