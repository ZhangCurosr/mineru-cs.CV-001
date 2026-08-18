---
title: "Diagram-MMU-A-Multi-Modal-Benchmark-for-Scientific-Diagrams"
source: https://arxiv.org/pdf/2608.12262v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-08-18 10:53:31"
---

# 论文速读：Diagram-MMU-A-Multi-Modal-Benchmark-for-Scientific-Diagrams

## 一句话总结
本文提出首个面向科学图表的多模态大模型基准 Diagram-MMU，包含3,744张TikZ图表与18,305个评估样本，覆盖解析(D2C-P)、编辑(D2C-E)与问答(DQA)三类任务及16种基础/agentic评估设置；研究揭示当前MLLMs存在“推理强、编码弱”的能力不对称，对象级语义解析与空间定位仍是显著瓶颈。

## 研究问题与动机
- 科学图表依赖几何拓扑与领域符号的语义表达，传统OCR或自然图像MLLMs仅能捕获像素/文本表层，无法准确还原结构关系，导致在ER图、化学式、电路图等领域专属任务上错误率高。
- 现有基准多依赖Python/SVG代码表示，未采用学术界标准的TikZ/PGF格式，且仅覆盖单一任务，缺乏对agentic能力（上下文利用、工具使用、状态管理、规划）的系统性评估。
- “Vibe Writing”范式要求用户输入自然语言意图即可生成可编译学术图表，但该能力高度依赖强大的图表→代码生成与编辑能力，目前缺乏统一评测基准制约模型迭代。

## 核心贡献（创新点）
- 构建首个基于TikZ/PGF格式的科学图表多模态基准，打通解析、编辑与问答全链路评估；与以往仅关注图像描述或单一代码生成任务的工作不同，本文强调“符号感知”与“agentic工作流”双重维度。
- 提出图像级-代码级-对象级三层细粒度指标体系，引入F1_type/text/color/bbox等对象级度量；与现有工作多依赖SSIM/CLIP等宏观指标相比，本文揭示视觉相似≠语义正确的关键洞察。
- 设计16种渐进式评估设置(S1-S16)，系统划分并量化基础能力与四类agentic能力；此前基准多为单次直接生成/问答，本文首次将工具调用、状态记忆与多步规划纳入标准化评测协议。
- 实证揭示MLLMs在科学图表上的“推理强、编码弱”不对称现象（DQA最高86%，D2C-P对象级F1仅31-57%），为后续模型训练与评测提供明确的差距地图。

## 方法详解
- **数据集构建**：从TikZ/PGF官方手册(3,540条)与社区资源(3,309条)收集6,849条代码，经简化、编译验证、去重得到3,744个独特图表，覆盖Charts、Planar Geometry、3D Shapes、Graph、Chemistry、Circuit六个领域。每图生成5个实例(1 D2C-P + 2 D2C-E + 2 DQA)，总计18,305样本，由13名研究生交叉验证；另提供Mini split(300图/1,500实例)用于快速评测。
- **任务设计**：
  - D2C-P：给定图表与预设preamble，模型直接生成可编译TikZ代码，GT为原始源码，无需agentic生成。
  - D2C-E：给定图表与自然语言编辑指令，通过agentic流水线生成修改后代码并要求保留未受影响元素；采用双验证器(GPT-5.2与Gemini-3 Pro交叉校验)加最多3次重试，最终人工复核。编辑维度含Color(2)、Text(4)、Scope(8)、Layout(3)共17个模板。
  - DQA：分描述型(23模板)与推理型(Standard 18、What-if 19模板)；What-if问题与D2C-E编辑操作刻意对齐以测试因果推理；答案分OI-NUM/OI-TERM/OI-LIST三类，由LLM judge评分。
- **评估指标**：图像级(SSIM、CLIP Score、LPIPS、FID)、代码级(CrystalBLEU)、对象级(F1_type、F1_text、F1_color[CIEDE2000]、F1_bbox[IoU≥0.3]，平均为F1_avg)。
- **Agentic设置**：16种配置(S1-S16)逐步开启上下文检索、TikZ搜索工具(MCP server形式，按需查询避免context rot)、状态记忆与任务规划模块，对比vanilla生成与增强型agent表现。

## 实验与结果
- **评测基线**：闭源模型Gemini-3.1 Pro/Gemini-3.0 Pro/Flash、GPT-5.2、Claude-4.6 Opus、Seed-2.0 Pro；开源模型Qwen3.5-397B-A17B、Kimi-K2.5、Qwen3-VL系列、InternVL3-38B、特化模型TikZero+。
- **DQA结果**：闭源模型整体显著领先，Gemini-3.0 Pro达86.46%（最高），Gemini-3.1 Pro 86.29%；开源最强Qwen3.5-397B为83.42%，Qwen3-VL-8B仅47.12%；3D Shapes为全任务最难领域（Gemini-3.0 Flash 67.69%，GPT-5.2 63.96%）。
- **D2C-P对象级**：闭源模型F1_avg聚类于47.57-54.64，开源跨度大(31.47-57.48)，特化模型TikZero+仅15.43%。
- **D2C-E与定位瓶颈**：edit-only CrystalBLEU仅0.51%-2.98%；bbox定位为所有模型最弱维度（GPT-5.2: 8.0，Claude-4.6 Opus: 9.2），新增元素定位极差(edit-only bbox 2.3-3.0)。
- **Agentic能力增益**：D2C-E上下文利用稳定提升F1_avg(+4.1~10.6)；工具使用仅Claude-4.6 Opus持续获益(+2.2/+4.4)，Gemini-3.1 Pro因过度检索出现context rot(-4.3)；状态管理仅Claude-4.6 Opus(+2.5/+2.6)与GPT-5.2显著提升；规划能力pass@1→pass@2增益最大，k>4后饱和。
- **结论**：基础解析失败会导致D2C-E与DQA同步下降(gap 20-40分)；Gemini-3.0 Pro在三类任务上表现最均衡；当前模型在符号感知、空间定位与agentic工具协同方面仍有巨大提升空间。

## 相关工作脉络
- 与依赖OCR/NLP或自然图像MLLMs的早期图表理解工作对比：本文指出此类方法仅捕获文本标注而忽略几何拓扑，在领域专属图表上错误率高，强调必须转向符号感知与代码级生成范式。
- 与TikZero等特化图表代码生成模型对比：特化模型在对象级F1上反而更低(15.43%)，表明单一任务微调可能损害通用结构理解与编辑泛化能力，凸显综合基准的评测价值。
- 与MMMU、ChartQA等多模态VLM基准对比：现有基准多聚焦图像描述或数值问答，缺乏可编译代码生成(D2C)与agentic工作流评估，本文填补基础能力+工具智能体双维度评测空白。
- 与WebArena、GAIA等通用agentic基准对比：本文首次将agentic评估引入科学图表领域，聚焦TikZ搜索工具、状态管理与编辑规划，而非通用网页/代码代理
