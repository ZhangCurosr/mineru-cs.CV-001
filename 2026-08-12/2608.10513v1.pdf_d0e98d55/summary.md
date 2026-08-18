---
title: "SafeCap: Improving LVLM Safety with Image Captioning Reinforcement Learning"
source: https://arxiv.org/pdf/2608.10513v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:35:38"
field: "多模态大模型安全对齐"
keywords: ["LVLM安全对齐", "强化学习", "图像描述", "多模态安全", "GRPO", "caption中介奖励"]
innovations: ["将LVLM安全对齐形式化为自描述问题，训练模型先输出安全相关caption再回答问题", "引入caption中介强化学习奖励，利用冻结纯文本LLM评估caption支持的安全一致回答", "提出DirectCap/Prism多协议评估框架，在11个基准上证明安全增益同时保持视觉效用"]
benchmarks: ["MM-SafetyBench", "MSSBench", "VLSBench", "FigStep", "MIS-Test", "MM-Vet", "BLINK", "MMVP", "ERQA", "VPCT", "MMStar"]
---

# 论文速读：SafeCap: Improving LVLM Safety with Image Captioning Reinforcement Learning

## 一句话总结
SafeCap 提出一种基于强化学习的框架，通过训练 LVLM 在生成最终回答前先输出安全相关的图像描述（caption），利用冻结的纯文本 LLM 对 caption 进行安全一致性评估来引导策略优化，从而在多模态安全基准上显著提升模型安全性而不损失视觉效用。

## 研究问题与动机
- **视觉输入绕过文本安全对齐**：LVLM 继承自语言骨干的安全拒绝策略常被视觉输入削弱或绕过，因为有害指令可隐藏在图像文本（OCR）、对象语义或排版中，纯文本侧的安全机制无法感知。
- **现有 caption 中介防御存在缺陷**：ECSO 等推理时防御将图像转为文本描述再利用语言侧安全机制，但图像转文本过程会丢失细粒度视觉细节（如安全相关物体、可见文字），导致下游 LLM 做出错误安全判断；且该方法依赖下游语言模型本身具备安全识别能力。
- **训练时安全对齐面临安全-效用权衡**：现有安全 SFT/DPO 方法若仅优化拒绝行为易引发过度拒绝（over-refusal），损害对良性多模态请求的理解与响应能力。
- **缺乏端到端可学习的自描述安全路径**：现有方法多为推理时包装（inference-time wrapper）或静态偏好数据微调，未让 LVLM 自身学会在回答前主动暴露安全关键视觉证据。

## 核心贡献（创新点）
- **安全感知自描述 formulation**：将 LVLM 安全对齐形式化为自描述问题，模型在生成最终回答前必须先输出结构化 caption 作为安全证据通道；与 ECSO 等推理时方法本质区别在于，SafeCap 训练的是 LVLM 自身的双步生成路径而非外挂包装。
- **Caption 中介强化学习奖励设计**：引入双信号奖励——直接评估策略模型最终回答的安全/效用，以及利用冻结纯文本 LLM 评估 caption 是否支持安全对齐的独立回答，并与组件级归一化结合；与 CapRL 等仅优化 caption 质量的 VQA 信号本质区别在于，本文引入二值安全一致性判断使 caption 成为安全推理的证据接口。
- **多协议评估与消融验证**：提出 Direct / DirectCap / Prism 三种推理协议进行系统评估，证明 SafeCap 在目标 DirectCap 路径上获得最强且最一致的安全增益（4B-Base 提升 19.0 分），同时保持视觉效用不下降；与单一协议评估的本质区别在于提供了训练损伤、caption 可迁移性、原生能力保持的多维度诊断。

## 方法详解
- **问题设定与输出格式**：每条训练样本为图像-问题对 $(x, q)$，策略模型 $\pi_\theta$ 生成结构化响应 $y = \<\mathtt{Caption}\>c\</\mathtt{caption}\>a$，其中 $c$ 为图像描述、$a$ 为最终回答，对应 DirectCap 操作模式。
- **训练目标**：基于 GRPO（Shao et al. 2024）在 SPA-VL 数据集上训练，每 prompt 采样 $K=8$ 组 rollouts，组合三个奖励信号构造优势函数：
  $$A_i = w_{\mathrm{tmp}} \tilde{r}_{i,\mathrm{tmp}} + w_{\mathrm{cap}} \tilde{r}_{i,\mathrm{cap}} + w_{\mathrm{ans}} \tilde{r}_{i,\mathrm{ans}}$$
  其中 $\tilde{r}$ 为组内归一化后的奖励分量，用于 PPO-clipped GRPO 损失更新。
- **模板奖励 $R_{\mathrm{tmp}}$**：规则解析提取 `<caption>` 标签，要求恰好一个完整 caption 块、caption 和 answer 均非空、生成在长度限制内终止，否则跳过下游奖励仅保留结构失败信号。
- **回答奖励 $R_{\mathrm{ans}}$**：使用 judge 模型对回答打分，保留分数 $u(a) \in [0,5]$ 衡量信息保留程度，风险分数 $h(a) \in [0,5]$ 衡量有害内容；采用指数风险折扣函数 $S(u,h) = u \cdot \gamma^h$（$\gamma=0.35$），相比线性奖励 $u-\beta h$ 更激进地惩罚高风险回答同时保留安全有用回答的正值。
- **Caption 奖励 $R_{\mathrm{cap}}$**：将 caption $c$ 输入冻结纯文本 LLM（Qwen3-4B）生成答案 $a_f$，judge 对 caption 评分描述覆盖度 $u(c)$（不检查视觉事实正确性），再使用二值安全对齐 judge 判断 $g(q,a,a_f) \in \{0,1\}$——当策略答案与冻结 LLM 答案在安全状态上一致时得 1 分；最终 $R_{\mathrm{cap}} = g(q,a,a_f) \cdot u(c)$。
- **组件级组归一化**：每个奖励分量在 rollout 组内独立做均值-方差归一化 $\tilde{r}_{i,k} = (r_{i,k} - \mu_{G,k}) / (\sigma_{G,k} + \epsilon)$，防止不同尺度分量耦合影响信用分配，借鉴 GDPO 的去耦归一化思想但无需额外白化步骤。
- **推理协议**：DirectCap（目标协议，模型先写 caption 再回答）；Direct（移除 caption 要求测试原生回答是否受损）；Prism（将生成的 caption 输入冻结纯文本 LLM 测试 caption 的可迁移性）。

## 实验与结果
- **数据集与基线**：使用公开 SPA-VL 多模态安全对齐数据集训练，在 5 个安全基准（MM-SafetyBench、MSSBench、VLSBench、FigStep、MIS-Test）和 6 个视觉效用基准（MM-Vet、BLINK、MMVP、ERQA、VPCT、MMStar）上评测；基线包括 safety SFT、DPO 及同数据训练的 SafeGRPO。
- **零训练协议对比**：Direct 在指令微调的 2B/4B 模型上平均最优，DirectCap 在部分效用指标上具竞争力，Prism 受限于冻结 LLM 对 caption 完整性的依赖整体分数偏低，表明需通过训练内化 caption 中介的安全推理。
- **SafeCap 主要结果**：在 2B/2B-Base/4B/4B-Base 四种设置下，DirectCap 的 11 基准综合平均分分别提升 5.29 / 5.57 / 5.48 / 8.57 分；其中 4B-Base 模型的 DirectCap 安全平均分（S-Avg）提升 18.96 分（从 40.43 到 59.39），视觉效用平均分（V-Avg）几乎不变（53.13 → 53.05）。
- **与 SFT/DPO/SafeGRPO 对比**：在相同 Qwen3.5-4B-Base 初始化和 SPA-VL 数据上，SafeCap DirectCap S-Avg 达 59.39，显著高于 SFT（43.19）和 DPO（41.36）；在 SafeGRPO 的 SafeTag-VL-3K 数据上，SafeCap DirectCap S-Avg 为 55.06 vs SafeGRPO 的 41.27，同时 V-Avg 保持相近（51.34 vs 50.56）。
- **消融结果**：移除直接回答奖励导致 DirectCap 安全下降 6.49 分，移除 caption 奖励下降 2.53 分，移除归一化下降 2.02 分；风险系数 $\gamma=0.35$ 为平衡选择，更小 $\gamma$ 加速安全收敛但可能过度抑制效用。
- **训练稳定性**：三种子seed的 100-step 实验显示 DirectCap S-Avg 为 50.83±1.14，较零训练提升 10.40±1.14 分（p=0.004），效用变化 +1.19±1.54 分不显著。

## 相关工作脉络
- **推理时 caption 中介防御（ECSO, Gou et al. 2024）**：将不安全图像转为 query-aware 文本描述再利用语言侧安全机制；本文定位差异在于训练 LVLM 自身的双步生成路径而非推理时外挂包装，避免了图像转文本的信息丢失和下游模型能力依赖。
- **多模态安全基准与攻击（MM-SafetyBench, VLSBench, FigStep, MIS）**：揭示了视觉输入可绕过文本安全对齐的各种攻击模式（排版嵌入、对象暗示、良性提示+危险图像）；本文与之互补，提供训练时内化安全推理的方法而非仅依赖评测。
- **训练时安全对齐（SPA-VL, VLGuard, MM-RLHF）**：提供偏好数据或安全微调范式；本文定位差异在于不依赖私有数据，仅用公开 SPA-VL 数据配合强化学习即可超越 SFT/DPO 同类方法，且显式建模 caption 作为安全证据通道。
- **Caption 中心预处理与 RL（CapRL, Xing et al. 2025）**：将开放域 caption 质量转为解耦 VQA 信号减少 reward hacking；本文受其启发但将 caption 奖励从感知质量扩展到安全一致性判断，使 caption 成为跨模型转移的安全证据接口。
- **安全 SFT/DPO（Zong et al. 2024, Zhang et al. 2024）**：通过人工标注偏好对优化安全行为；本文与它们的对比实验显示，在相同数据和步数下 SafeCap 的 DirectCap 增益远超 SFT/DPO（+18.96 vs +2.76/+0.93），表明 caption 中介 RL 比直接偏好优化更能同时兼顾安全与效用。
- **SafeGRPO（Rong et al. 2025）**：基于规则治理的强化学习安全对齐；本文在相同数据和初始化上的对比实验（SafeTag-VL-3K）显示 SafeCap DirectCap 显著优于 SafeGRPO Base 推理（55.06 vs 41.27），凸显自描述路径的价值。

## 局限性与未来方向
- **冻结 LLM 能力边界**：caption 奖励的有效性受限于用于评估的冻结纯文本 LLM 的安全推理能力；Prism 评估中更换更强的 Qwen3-14B 可提升 4.72 分，说明 caption 本身能提供的安全增益上限由下游 reasoner 决定。
- **caption 事实性无显式监督**：由于冻结 LLM 无法看到图像，caption 奖励只能检查描述覆盖度和安全一致性，无法检测事实性错误；论文通过人工监控未发现系统性幻觉，但未建立防止 hallucination 的机制。
- **奖励的可验证性不足**：安全性和风险奖励依赖固定规则 judge 打分，尚未实现完全可验证的安全奖励，属于社区开放问题。
- **规模限制**：实验仅在 Qwen3.5-2B/4B 系列上进行，更大模型家族和规模的泛化需未来验证。
- **缺少 caption SFT 预训练**：论文指出在 RL 之前进行 caption 中介的 SFT 是 promising extension，但当前缺乏合适的监督数据。

## 研究启发与可借鉴点
- **中间表示作为安全证据通道**：将安全推理拆解为"先提取证据（caption）→ 后做决策（answer）"的两阶段模式，可将此思路迁移至其他需要逐步推理的安全/可信场景（如代码生成安全审查、医学影像报告生成）。
- **跨模型一致性奖励设计**：利用冻结模型作为独立裁判来评估中间产出的质量，避免奖励黑客问题；此设计可推广至其他需要自我反思或自我校验的 RL 训练场景。
- **组件级归一化在多奖励 RL 中的应用**：将不同尺度的奖励分量独立归一化后再组合，能有效防止某一强信号淹没其他信号，可复用于任何多目标 RLHF/RLAIF 训练。
- **多协议诊断框架**：同时评估原生能力（Direct）、学习协议（DirectCap）和迁移能力（Prism），为方法验证提供完整画像而非单一指标，值得在多模态对齐工作中借鉴。
- **指数风险折扣函数**：$u \cdot \gamma^h$ 形式的奖励设计比线性惩罚更能平衡安全与效用，可探索迁移至纯文本安全对齐或代码安全审查任务。

## 关键术语表
**SafeCap**：一种基于强化学习的 LVLM 安全对齐框架，通过训练模型生成安全相关的图像描述（caption）后再回答，利用 caption 作为安全证据通道。
**DirectCap**：SafeCap 的目标推理协议，模型先生成带标签的图像描述再输出最终回答，实现端到端自描述安全推理。
**Prism 协议**：诊断性评估协议，将模型生成的 caption 输入冻结纯文本 LLM 仅凭文本生成答案，检验 caption 是否携带足够的可迁移安全证据。
**SPA-VL**：公开的 multimodal safety-alignment 数据集，包含多模态安全偏好数据，本文唯一使用的训练数据源。
**GRPO（Group Relative Policy Optimization）**：DeepSeekMath 提出的强化学习优化算法，基于组内相对优势估计策略梯度，本文用于安全对齐训练。
**Caption-mediated reward**：通过将生成的 caption 输入冻结 LLM 并检查其答案与策略答案的安全一致性，结合描述覆盖度评分构成的复合奖励信号。
**Exponential risk-discount**：奖励函数 $S(u,h) = u \cdot \gamma^h$，以指数形式随风险分数衰减效用贡献，相比线性惩罚更激进地抑制高风险回答。
**Component-wise group normalization**：在每个 rollout 组内对各奖励分量独立做均值-方差归一化，防止多奖励信号间的尺度耦合。

## 可复现要素
- **数据集**：SPA-VL 多模态安全对齐数据集（公开，https://arxiv.org/abs/2406.12030），代码库链接 https://github.com/Safe-VLM/SafeCap
- **代码/权重**：代码已开源（GitHub: Safe-VLM/SafeCap），论文未明确声明模型权重是否公开
- **关键超参**：GRPO 每 prompt 采样 K=8 rollouts，batch size=64，mini-batch size=64，学习率 $5 \times 10^{-7}$，weight decay=0.01，gradient clipping=1.0，entropy regularization=0，最大 token 数 4096，训练 200 steps；奖励权重 $(w_{\mathrm{tmp}}, w_{\mathrm{cap}}, w_{\mathrm{ans}}) = (0.5, 0.5, 1.0)$，风险折扣系数 $\gamma = 0.35$，PPO clip 阈值 $\delta = 0.2$
- **冻结 LLM**：Qwen3-4B（caption 奖励使用），judge 模型 gpt-oss-20b（确定性解码）
- **训练硬件**：8 × NVIDIA H200 GPUs
