---
title: "Diagram-MMU-A-Multi-Modal-Benchmark-for-Scientific-Diagrams"
source: https://arxiv.org/pdf/2608.12262v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-08-18 10:55:57"
---

# 论文速读：Diagram-MMU-A-Multi-Modal-Benchmark-for-Scientific-Diagrams

## 一句话总结
本文提出 **Diagram-MMU**，首个面向科学图表的统一多模态基准，涵盖图表解析（D2C-P）、代码编辑（D2C-E）与问答推理（DQA）三大任务及16种能力设置，系统性评测了12个主流MLLMs的TikZ代码生成能力，揭示了大模型“视觉推理强、结构化编码弱”的能力鸿沟，并首次量化了工具使用行为（采用率、调用强度、上下文退化）对性能的非线性影响。

## 研究问题与动机
- **推理-编码能力失衡**：现有MLLMs在科学图表的视觉理解与问答任务上表现优异，但将图表转换为可执行代码（如TikZ）并进行精确编辑的能力显著薄弱，缺乏统一的评测基准加以诊断。
- **Agentic设置的作用机制不明**：工具调用、上下文利用、状态管理与规划等增强手段对不同任务类型的影响存在异质性，尚缺系统性对照实验。
- **领域覆盖与评估粒度不足**：既往工作多聚焦通用插图或单一图表类型，缺乏将Chemistry（chemfig）与Circuit（circuitikz）纳入的跨领域评测，且缺少对象级保真度与修改精度的细分指标。
- **工具行为缺乏可量化诊断**：现有评测仅关注最终准确率，未深入分析模型工具采用策略、调用模式及其引发的context rot等隐性退化现象。

## 核心贡献（创新点）
1. **构建首个多任务科学图表基准Diagram-MMU**：统一整合D2C-P、D2C-E与DQA三类任务及16种评测Setting，填补了“理解→生成→编辑”全链路评测的空白。
2. **设计正交分解的Agentic能力评测框架**：将工具使用解耦为Context Utilization、Tool Use、State Management、Planning四类Setting，精准定位各模型在多步协作中的薄弱环节。
3. **提出细粒度多维评估体系**：同步覆盖图像级（SSIM/CLIP/LPIPS/FID）、代码级（CrystalBLEU）与对象级（F1_type/text/color/bbox）指标，并引入preserve-only与edit-only拆分以区分“未改到位”与“改坏原内容”。
4. **首次纳入Chemistry与Circuit领域并建立专业验证流程**：由13名研究生交叉验证18,305道样本，显著提升化学分子结构与电路图代码生成的评测可信度。
5. **构建工具行为量化分析范式**：引入Adoption、Intensity、响应率/执行率变化及context rot诊断，为MLLM工具调用效率优化提供可复用的观测维度。

## 方法详解
- **任务定义**：
  - **D2C-P**：输入科学图表图像，输出可执行的TikZ代码。
  - **D2C-E**：给定原始图表图像、基础TikZ代码与自然语言修改指令，输出增量编辑后的TikZ代码。
  - **DQA**：基于图表内容回答描述性或what-if推理问题。
- **16种评测Setting**：基础设置（S1/Direct coding、S6/Direct editing、S12/Direct answering）与四类Agentic设置（S2/S7/S13上下文利用、S3/S8/S14/S16工具使用结合TikZ Search MCP Server、S4/S9/S15状态管理、S5/S11/S16多步规划）。
- **评估指标体系**：
  - 图像级：SSIM、CLIP Score、LPIPS、FID
  - 代码级：CrystalBLEU
  - 对象级：F1<sub>type</sub>、F1<sub>text</sub>、F1<sub>color</sub>（CIEDE2000）、F1<sub>bbox</sub>（IoU≥0.3）→ 综合为F1<sub>avg</sub>
  - DQA：Accuracy，由Qwen3-Next-80B-A3B-Instruct作为Judge进行二值评分
  - 编辑任务拆分：preserve-only（保留部分保真度）与edit-only（修改部分准确性）
- **工具使用分析**：统计各模型在每类Setting下的Tool Adoption率、Intensity（平均每问题调用次数n̄与有调用问题平均值n̄+），并追踪响应率与执行率的相对变化以量化工具增益或干扰。

## 实验与结果
- **数据集规模**：3,744张人工筛选科学图表，覆盖Charts、Planar Geometry、3D Shapes、Graph、Chemistry、Circuit六个领域，共18,305道人工验证评测样本。
- **评测模型**：12个主流MLLMs（闭源：Gemini-3.1/3.0 Pro/Flash、GPT-5.2、Claude-4.6 Opus、Seed-2.0 Pro；开源：Qwen3.5-397B-A17B、Qwen3-VL-235B-A22B、Kimi-K2.5、Qwen3-VL-8B、InternVL3-38B、专为此任务微调的TikZero+ 10B）。
- **基础能力表现**：
  - D2C-P物体级F1<sub
