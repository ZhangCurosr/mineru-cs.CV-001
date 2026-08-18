---
title: "Matched-Outcomes-Divergent-Gaze-How-Foveated-MLLMs-Search-Co"
source: https://arxiv.org/pdf/2608.16514v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:26:05"
field: "多模态视觉认知建模"
keywords: ["visual search", "foveated vision", "multimodal LLMs", "eye movements", "COCO-Search18", "human-model alignment", "scanpath prediction"]
innovations: ["三轴解耦：decision/finding/gaze可分离，证明MLLMs类人outcome但非类人process", "无序列瓶颈的null model：单次通过架构达成类人准确率但产生低熵大振幅scanpath", "降解sweeps证明不存在人类-like搜索的操作点"]
benchmarks: ["COCO-Search18"]
---

# 论文速读：Matched-Outcomes-Divergent-Gaze-How-Foveated-MLLMs-Search-Co

## 一句话总结
本研究通过将被视网膜输入匹配至人类明视度水平的三种通用MLLMs（Qwen3.5-35B-A3B、GLM-4.6V-Flash、Gemma-4-E4B）在COCO-Search18上进行逐注视点驱动搜索，发现模型在目标存在决策与目标获取效率上匹配或超越人类，但注视过程呈现低熵、大振幅、高自洽性的非人类签名，证明零样本MLLMs适合 outcome 与空间问题，却无法模拟人类序列采样过程。

## 研究问题与动机
1. **核心问题**：当MLLMs获得与人类相同的明视度限制（foveated）输入时，它们的搜索行为是否像人类一样？这关系到将MLLMs作为人类视觉模型使用的合理性，以及注意力对齐分数的解释力。
2. **现有方法不足**：当前研究常用 answer-alignment（答案对齐）和 saliency metrics（显著图重叠）来评估模型是否"像人一样看"，但这些指标仅计算在 outcome 或时间折叠的空间图上，无法捕捉 gaze process（注视过程）的时序差异。
3. **理论假设**：若视网膜输入匹配，则搜索行为也应匹配——这一假设支撑着将MLLMs作为人类观察者替代品的常见做法，需要可证伪的检验。
4. **实践动机**：区分"看得对"与"看得像人"，为视觉AI的 human-like vision 认证提供方法论基础。

## 核心贡献（创新点）
1. **三维解耦框架**：将"类人搜索"解耦为 decision（目标存在判断）、finding（目标获取效率）、gaze（注视动力学）三个独立轴，揭示三者可分离。
2. **过程轴发现**：证明在匹配人类明视度条件下，MLLMs共享一种低熵、大振幅、高自洽性的注视签名，与人类的交互一致性（cross-seed ScanMatch 0.84/0.91/0.71）远高于人类间一致性（0.53）。
3. **降解 sweeps 结论**：遍历合成降解条件（gist-k）证明不存在任何操作点能使模型同时在人类水平成功率和人类水平搜索模式上运行。
4. **度量批判**：指出 answer-alignment 和 saliency scores 对过程轴盲视，不能 certify 类人视觉；零样本MLLMs仅适合 outcome 和 spatial 问题，不适合 process 和 temporal 问题。
5. **Null model 提案**：提出无序列瓶颈但达成类人 outcome 的搜索器可作为 null model，用于隔离序列性带来的行为成本。

## 方法详解
1. **数据与参考**：使用 COCO-Search18 验证集，包含141个目标存在（target-present）和144个目标不存在（target-absent）场景，每场景10条人类 scanpaths，单位为 (scene, target) trial。
2. **明视度支架（Foveation Bracket）**：在注视点应用确定性渲染器，共9种条件：
   - **sharp**：无衰减
   - **geisler–perry (GP)**：Geisler & Perry 明视度衰减函数（唯一匹配人类的校准条件）
   - **gist-k**：高斯退化，k ∈ {8, 16, 24, 32, 48, 128} 缩放外围截止（合成条件）
   - **crop**：仅保留 ~2.5° 半径的明视盘
3. **搜索循环**：每 episode 从强制中心注视开始，模型每步观察当前注视点渲染的场景（历史 glimpse 保留在 context 中），返回单一指令：LOOK(x,y)、FOUND(x,y) 或 ABSENT。无搜索策略施加，50 glimpse 上限从不强制终止。
4. **指令读取**：每生成以单一指令行结束，坐标归一化到单位正方形，原点左上角，y向下增长。仅消费最终指令行。
5. **模型**：Qwen3.5-35B-A3B（35B MoE，3B活跃，reasoning-tuned）、GLM-4.6V-Flash（≈9B，reasoning-tuned）、Gemma-4-E4B（≈4B，instruction-tuned，thinking禁用）。
6. **度量体系**：
   - Decision：存在准确率、信号检测 d′
   - Finding：TFP@n（第n次扫视命中概率）、TFP-end、NFix-TP/TA（中位注视数）
   - Gaze：Clif's δ（7个统计量）、ScanMatch相似度、PCA降维

## 实验与结果
1. **实验规模**：3模型 × 285场景 × 5 seeds × 9条件 = 38,475 episodes，全部完成。
2. **Decision结果**：近天花板检测（target-present 0.99/0.99/0.97），d′ 3.84/4.23/3.14 vs 人类2.91，表现优于人类。
3. **Finding结果**：首次扫视命中率 TFP@1 为 0.97/0.97/0.80，远超人类0.49；最终成功率 TFP-end 0.98/0.98/0.93 与人类0.93相当；中位注视数 2/2/3 少于人类3。
4. **Gaze结果（GP条件）**：
   - 熵更低（δ = -0.67/-0.65/-0.27），空间采样更集中
   - 扫视振幅更大（δ = +0.50/+0.61/+0.23），直接跳跃至目标
   - 自洽性极高：cross-seed ScanMatch 0.84/0.91/0.71，远超人类间天花板0.53
5. **降解 sweeps**：在 k=32（Qwen）和 k=16（Gemma）时 TFP@1 接近人类0.49，但 TFP-end 已降至0.71/0.75，证明无法同时匹配成功率和搜索模式。
6. **最强结果**：GLM-4.6V-Flash 在 TFP@1（0.97）和 cross-seed 自洽性（0.91）上最优；Gemma-4-E4B 在多个 gaze 统计量上最接近人类。

## 相关工作脉络
1. **人类搜索模型**：COCO-Search18 锚定了 scanpath predictor、inverse-RL、transformer 模型；经典搜索混合并行外围评估与串行焦点检查。本文与之定位差异：直接比较 MLLM 过程而非预测。
2. **明视度架构**：Geisler–Perry 衰减提供人类匹配渲染器；生物约束网络在无训练时即可产生类人 scanpaths。本文发现对通用 MLLM 此路不通。
3. **MLLM vs 人类视觉**：先前工作发现模型"see but do not perceive"、attention similarity 与 task performance dissociate。本文差异：测量逐注视点过程，带明确停止决策。
4. **Agentic搜索训练MLLMs**：LLM-guided search、tree-based zoom、RL-focused 等方法优化"看哪里"以提升准确率。本文定位：刻画通用模型在固定明视约束下如何探索，无需搜索训练。
5. **计算效率方法**：token merging、adaptive tokenizer 等分配 compute 而非 acuity。本文明确区分：这是预算分配机制，非外围视觉模型。

## 局限性与未来方向
1. **未评估搜索训练 agent**：search-trained agentic、pointing-native、frontier closed-source 模型未纳入，是最相关的下一步比较对象。
2. **跨模型混淆**：architecture、recipe、sparsity 共变，趋势仅作描述性报告，无法分离独立效应。
3. **prompt 形状效应**：gaze signature 在单一 prompt 下测量，memory clause 可能部分塑造 refixation 行为。
4. **固定时长效能**：fixation durations 仅为人类数据，模型侧无时间度量。
5. **序列采样因果测试缺失**：通过消除法（acuity 和 spatial prior 已排除）推断缺少序列采样，但未直接操纵架构验证。

## 研究启发与可借鉴点
1. **三轴解耦评估框架**：可将 decision/finding/gaze 解耦思路迁移至其他视觉认知任务评估，区分"做对"与"怎么做"。
2. **明视度支架设计**：GP 衰减 + gist-k 合成梯度 + crop 极端条件的 bracket 设计，可作为视觉模型感知限制的标准化测试协议。
3. **cross-seed 自洽性作为过程度量**：引入 agent↔agent ScanMatch 度量确定性，弥补传统 agent↔human 相似度的不足。
4. **Null model 概念**：无序列瓶颈但类人 outcome 的搜索器作为基准，可用于量化序列采样的行为成本。
5. **降解 sweeps 方法**：通过连续变化输入质量而非二元比较，揭示模型能力边界与失败模式。

## 关键术语表
- **Foveated Vision（明视度视觉）**：仅在注视点附近高分辨率、外围低分辨率的视觉系统特性，人类眼动受此限制呈串行。
- **Scanpath（扫描路径）**：注视点序列，反映视觉探索的时序结构。
- **TFP@n（Target Fixation Probability）**：第n次扫视前已命中目标的概率，衡量搜索效率。
- **Clif's δ**：非参数效应量统计量，度量两组分布的差异方向和幅度。
- **ScanMatch**：基于 Needleman-Wunsch 全局对齐的注视序列相似度度量。
- **Gist-k**：人为退化的外围视觉条件，k 越大外围信息越粗糙。
- **Single-pass Architecture（单次通过架构）**：并行编码器一次性解析清晰帧，无需序列采样。
- **Serial Sampling（序列采样）**：人类视觉的串行过程，需多次注视逐步确认目标。

## 可复现要素
- **数据集**：COCO-Search18（公开），使用 validation split 的141 target-present + 144 target-absent 场景
- **代码/权重**：论文未提及代码开源声明；模型为 Qwen3.5-35B-A3B、GLM-4.6V-Flash、Gemma-4-E4B（均来自官方）
- **关键超参**：temperature=0.6、5 seeds、50 glimpse 上限、hit tolerance=1°、gaze entropy 网格 14×9
- **渲染器**：Geisler-Perry 衰减函数 Eq.(S1)， viewing distance=0.6m，resolution=30 px/deg
- **Prompt**：见 Supplementary Sec.S12，逐行固定格式输出指令
