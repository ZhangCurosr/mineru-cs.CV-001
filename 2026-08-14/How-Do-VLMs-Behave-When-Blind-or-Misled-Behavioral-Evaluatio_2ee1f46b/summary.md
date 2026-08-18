---
title: "How-Do-VLMs-Behave-When-Blind-or-Misled-Behavioral-Evaluatio"
source: https://arxiv.org/pdf/2608.13267v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:48:05"
field: "多模态大模型评估与可靠性"
keywords: ["VLM", "科学图表理解", "行为可靠性", "幻觉评估", "多模态基准", "Admittance-Resistance-Inductance", "选择性模糊"]
innovations: ["提出 A-R-I 行为框架，首次将承认不确定性、抵抗误导上下文、有界推断三者统一为可量化的行为评估维度", "设计选择性模糊探针区分不可恢复与可推断两类条件，同时测量 Admittance 与 Inductance", "揭示描述质量与行为可靠性的根本脱钩：GPT-5.2 MQM 第一但主动承认率仅 8%，Gemini 3.1 Pro MQM 第二但全维度行为最优"]
benchmarks: ["SCIFIGBENCH", "ChartQA", "SciFIBench", "CharXiv", "ChartMuseum", "ChartQAPro"]
---

# 论文速读：How Do VLMs Behave When Blind or Misled? Behavioral Evaluation of VLMs on Scientific Figures

## 一句话总结
论文提出了 **SCIFIGBENCH**——一个面向科学图表理解的 VLM 诊断基准，首次系统性地评估模型在视觉证据缺失或误导性上下文下的**行为可靠性**（Admittance-Resistance-Inductance 框架），揭示出描述质量最高的模型（GPT-5.2）在不一致情况下会以 96% 的概率编造内容，而行为可靠性第二的 Gemini 3.1 Pro 则有 71% 的概率承认不确定性。

## 研究问题与动机
1. **现有基准的盲区**：当前 VLM 图表理解基准（如 ChartQA、SciFIBench、CharXiv）主要评估感知与推理准确率，缺乏对模型在**视觉证据缺失或误导性语境下行为可靠性**的系统测量。
2. **科学场景的严重性**：科学图表承载论文核心定量证据，VLM 的误读会导致下游摘要、对比和科学推理产生严重后果；高准确率不等于高行为可靠性。
3. **感知与行为脱节**：同样在 MQM 描述质量上得分相近的模型（GPT-5.2: 91.6 vs Gemini 3.1 Pro: 90.2），在面对模糊/遮挡元素时行为截然相反，现有评估体系完全无法捕捉这一差异。
4. **部署风险**：若仅依据传统准确率指标选择 VLM 部署到科学工作流中，可能将"自信的错误"系统性地嵌入科研管道。

## 核心贡献（创新点）
1. **提出 SCIFIGBENCH 基准**：覆盖 250 张 arXiv 科学图表，包含 1,000 道人工审核的推理题、600+ 小时人工标注，并通过图像变换、选择性模糊、标题偏置探针等扩展至超过 34,000 个评估样本；与已有工作的本质区别在于**首次将行为可靠性（而非仅准确率）纳入统一评估体系**。
2. **提出 A-R-I（Admittance–Resistance–Inductance）行为框架**：将行为分解为三个正交维度——承认不确定性的认知诚实（Admittance）、抵抗误导性上下文的鲁棒性（Resistance）、从部分证据做有界推断的能力（Inductance）；与已有不确定性评估的本质区别在于**区分了证据是否可恢复的两种模糊条件，并分别测量**，而非笼统地评估"不确定时怎么说"。
3. **设计了四种可控压力测试探针族**：图像变换（噪声/旋转/低对比度）、标题偏置（嵌入 2–3 条合理但错误的断言）、虚假前提探针（不存在元素/矛盾数值/不可回答）、选择性模糊（不可恢复 vs 可推断）；与已有工作的本质区别在于**每个探针家族均锚定具体的部署失败模式**（如 OCR 丢失、多模态 RAG 中的错误图注检索、agent 间传播的错误前提）。

## 方法详解
**数据集构建**：从 187 篇 arXiv 论文（2023–2025）中选取 250 张英文科学图表（条形图 99、折线图 99、饼图 52），由两名受过训练的标注员（94% 一致性）编写专家描述， disagreements 由第三方仲裁。

**感知评估（MQM 适配）**：针对每种图表类型手工编写检查清单（条形图 14 项、折线图 15 项、饼图 11 项），每项标记 Major/Minor 严重级别；GPT-4o judge 逐条评估覆盖度与正确性，规则引擎映射误差到 Accuracy/Completeness/Clarity 三维扣分；最终分数：`MQM = max(0, 100 − P × 100 / (N × 5))`，满分 100。

**推理评估**：GPT-4o 生成候选问题（每类 3 个，覆盖 Counting/Computation/Comparison/Pattern Analysis），Mistral Large 3 验证答案可从图中独立得出且无歧义，最终保留 1,000 题，双判官（GPT-4o + Mistral Large 3）交叉评分。

**行为评估（A-R-I 框架）**：
- **Admittance（承认不确定性）**：对 228 个不可恢复的选中模糊目标，测试模型在主动提问和被动描述两种模式下是否承认视觉证据缺失；评分维度：admits（是/否）、fabricates（是/否）、correct（编造内容是否正确）。
- **Resistance（抵抗误导）**：四类探针——Inexist（预设不存在的元素）、Contra（嵌入 20–30% 偏差的错误数值）、Unanswerable（超出图表范围的领域问题）、Caption Bias（标题嵌入 2–3 条虚假断言）；评分 1.0（抵抗）/ 0.5（犹豫）/ 0.0（接受/编造）。
- **Inductance（有界推断）**：对 215 个从上下文可推断的选中模糊目标，测量模型在有依据前提下正确推断的比例，而非凭空编造。

**图像变换**：高斯噪声（σ=25）、低对比度（α=0.3）、旋转（15°）、论文页嵌入（in-paper）；共 1,243 个变换样本。

## 实验与结果
**评估模型**：8 个模型（GPT-5.2、Gemini 3.1 Pro、Llama 4 Maverick、Qwen3-VL-235B/30B/8B、Gemma-3-27B-IT、Phi-4 Multimodal），temperature=0，GPT-4o 作为自动判官。

**感知结果（MQM）**：GPT-5.2 最高（91.6，CI [90.4, 92.8]），Gemini 3.1 Pro 次之（90.2），两者差距统计显著但实际微小（Cliff's δ=0.09）；Phi-4 最低（62.2）。旋转是最具破坏性的变换（均降 19.4 分），噪声几乎无影响。

**推理结果**：Gemini 3.1 Pro 综合最强（81.0%），GPT-5.2 次之（78.4%）；两者与其他模型差距远大于感知质量的差距。Phi-4 灾难性低（8.6%），Gemma 3 27B 也远低于一线（27.2%）。

**行为结果（核心发现）**：
- **Resistance**：Gemini 3.1 Pro 最高（0.91），GPT-5.2 次之（0.81），Phi-4 最低（0.21）；Gemini vs GPT-5.2 差距显著（p<0.001）。
- **Active Admittance**：Gemini 3.1 Pro 唯一全面承认不确定性的模型（71%），GPT-5.2 仅 8%，其余模型均≤19%；GPT-5.2 在模糊区域**以 96% 概率编造不可读内容**。
- **Inductance 正确率**：Gemini（66%）> GPT-5.2（59%），表明两者在上下文允许时均能做合理推断，但 Gemini 更少幻觉。
- **Caption Bias Resistance**：Gemini 和 GPT-5.2 并列最高（0.89），Phi-4 极低（0.05），几乎完全跟随修改后的标题。
- **关键洞察**：MQM 排名第 1 的 GPT-5.2 在主动 Admittance 上仅排第 4；MQM 排名第 2 的 Gemini 3.1 Pro 在所有行为维度上均排第 1，揭示了**感知质量与行为可靠性之间存在根本性脱钩**。

## 相关工作脉络
1. **ChartQA / ChartBench / MathVista**：聚焦封闭形式任务与聚合准确率，不评估不确定性与行为可靠性，SCIFIGBENCH 在其基础上引入行为维度。
2. **CharXiv / SciFIBench / ChartMuseum**：专注科学图表理解但均无对抗性压力测试，SCIFIGBENCH 通过选择性模糊和虚假前提探针补充此缺口。
3. **ChartQAPro / EncQA / MultiChartQA**：引入不可答题但缺乏系统性行为框架；SCIFIGBENCH 的 A-R-I 将"抵抗-承认-推断"统一为一个可量化的三维框架。
4. **CHAOS / CHART-NOISe**：评估扰动下的性能退化，但侧重于鲁棒性而非认知诚实性；SCIFIGBENCH 的行为维度测量的是模型**主动选择**（是否承认不知道）而非被动性能下降。
5. **Honesty/Abstention 研究（BeHonest 等）**：评估 LLM 何时该回答、何时该放弃；SCIFIGBENCH 将这一思想扩展到多模态图表场景，并区分了被动描述 vs 主动提问两种模式。
6. **Sycophancy 研究（SycEval）**：评估模型顺从用户主张的倾向；SCIFIGBENCH 的 Caption Bias 探针验证了标题依赖是训练 artifact 而非能力限制，与纯 LLM 场景的发现形成对比。

## 局限性与未来方向
1. **图表类型覆盖有限**：仅包含条形图、折线图和饼图，散点图、热力图、网络图和示意图等科学文献常见类型未覆盖。
2. **语言仅限英文**：非英语语料库的泛化性未经验证。
3. **无法分离因素**：top 模型 MQM≥90 说明其对虚假前提的失败更可能源于指令遵循压力而非视觉能力缺陷，但基准无法对此进行决定性区分；也未定位到具体组件（视觉编码器 vs 语言模型）的责任。
4. **自动化评估依赖 GPT-4o**：虽有人工验证（Krippendorff's α=0.91）和跨判官鲁棒性测试（Mistral Large 3），但绝对分数的偏移（均值 -15 分）仍然存在。

## 研究启发与可借鉴点
1. **A-R-I 框架可直接迁移到其他多模态领域**：医学影像、卫星图像等场景中模型同样需要在证据不足时承认不确定性，而非自信编造；框架的三个维度可作为通用行为评估标准。
2. **"选择性模糊"的实验设计极具复用价值**：通过区分"不可恢复"vs "可推断"两类模糊目标，同时测量 Admittance 和 Inductance，避免了传统"鲁棒性"测试的单一退化曲线，为后续工作提供了精细的诊断工具。
3. **Caption Bias 探针揭示了 instruction tuning 的关键风险**：Phi-4 和 Gemma-3 在同等感知能力下表现截然不同的标题依赖，提示我们在微调阶段需显式引入"视觉优先于文本"的对齐信号，而非仅优化准确率。
4. **主动 vs 被动行为的差异值得深入研究**：GPT-5.2 被动承认率（23%）远高于主动承认率（8%），而 Gemini 两端均高；这种"必须回答"偏见与 RLHF 训练压力直接相关，可作为未来对齐研究的诊断基准。
5. **MQM 适配到科学图表评估的方法论**：检查清单+严重级别+绑定验证（label-value binding error）的组合是可复用的自动化评估范式，适用于其他需要细粒度开放描述评估的场景。

## 关键术语表
**SCIFIGBENCH**：面向科学图表理解的 VLM 诊断基准，联合评估感知、推理与行为可靠性，含 250 张图表及超过 34,000 个评估样本。
**A-R-I 框架（Admittance–Resistance–Inductance）**：行为可靠性三维评估框架，分别衡量模型承认不确定性的诚实度、抵抗误导上下文的能力、从部分证据做有界推断的质量。
**MQM（Multidimensional Quality Metrics）**：源自翻译质量评估的评分框架，本文适配为图表描述的质量评估标准，通过检查清单和加权扣分计算 0–100 分。
**Selective Blur**：通过 OCR 定位并模糊图表特定元素的测试技术，分为 Admittance（不可恢复）和 Inductance（上下文可推断）两类。
**Resistance Probes**：包括 Inexist（预设不存在元素）、Contra（嵌入错误数值）、Unanswerable（超出图表范围）三类虚假前提探针。
**Caption Bias**：在图表标题中嵌入 2–3 条合理但错误的断言，测试模型是遵循视觉证据还是盲从文本上下文。
**Presupposition Embedding**：通过定指短语（如 "the benchmark line"）将虚假前提嵌入问题中，是最强的欺骗向量，比显式错误更难抵抗。

## 可复现要素
- **数据集**：250 张来自 arXiv 的科学图表，论文声明数据集、评估脚本、模型输出和提示词将在发表后开源；项目网站：https://scifigbench.nlp4sci.com/
- **代码**：脚本位于 `scripts/experiments/`、`scripts/capability_generation/`、`scripts/adversarial_transforms/` 目录，论文声明将开源
- **关键超参**：所有模型 temperature=0，max tokens=2,048（Gemini 用 16,000），随机种子=42
- **API 版本**：GPT-4o judge 使用 Azure OpenAI API version 2024-12-01-preview；商业模型经 Azure，开源模型经 OpenRouter
- **判决模型**：GPT-4o（主）+ Mistral Large 3（交叉验证）
- **图像变换参数**：高斯噪声 σ=25，低对比度 α=0.3/β=50，旋转 15°
