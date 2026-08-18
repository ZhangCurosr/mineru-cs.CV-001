---
title: "SafeCap: Improving LVLM Safety with Image Captioning Reinforcement Learning"
source: https://arxiv.org/pdf/2608.10513v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:35:17"
field: "多模态大模型安全对齐"
keywords: ["LVLM安全", "强化学习对齐", "图像描述", "多模态安全", "DirectCap", "GRPO"]
innovations: ["将安全对齐建模为自我描述任务，通过中间caption显式化安全证据", "引入直接答案奖励与冻结LLM的caption中介一致性奖励双信号优化", "提出逐组件组归一化与指数风险折扣reward设计避免多信号耦合与敷衍拒绝"]
benchmarks: ["MM-SafetyBench", "MSSBench", "VLSBench", "FigStep", "MIS-Test", "MM-Vet", "BLINK", "MMVP", "ERQA", "VPCT", "MMStar"]
---

# 论文速读：SafeCap: Improving LVLM Safety with Image Captioning Reinforcement Learning

## 一句话总结
SafeCap 提出一种基于强化学习的框架，训练 LVLM 先生成安全感知的图像 caption，再基于该 caption 生成最终回答；通过"直接答案奖励 + 冻结 LLM 的 caption 中介奖励"双信号优化，在保持视觉理解能力的同时显著提升多模态安全对齐性能。

## 研究问题与动机
- **视觉输入可绕过纯文本安全对齐**：有害指令可隐藏在图片 OCR 文本、图形化提示或物体语义中，使语言侧已对齐的安全策略失效（FigStep、HADES、MIS 等攻击）。
- **推理时 caption 中介防御（如 ECSO）的信息损失**：现有方法将图像转为文字再交由冻结 LLM 判断，但这一转换过程会丢失细粒度视觉细节，导致下游安全决策不可靠。
- **训练时安全微调易损视觉能力**：直接 SFT/DPO 优化安全行为可能导致"过度拒绝"或视觉理解退化，需要在安全与效用之间取得平衡。
- **缺乏对"感知→对齐"联合优化的系统方法**：当前方法要么只做输出端拒绝，要么做推理时包装，未将"安全相关的视觉证据显式化"作为可训练接口。

## 核心贡献（创新点）
1. **安全感知自我描述形式化**：将 LVLM 安全对齐建模为"先生成 tagged caption，再作答"的自述任务，使中间 caption 成为安全证据通道，而非事后解释。
2. **Caption 中介强化学习目标**：引入结合直接答案评估与冻结 LLM 一致性评估的双信号奖励，并通过逐组件组归一化避免多尺度信号耦合，区别于仅依赖输出端拒绝监督的训练方式。
3. **多协议评估与消融体系**：提供 Direct/DirectCap/Prism 三种推理协议下的系统性评估，揭示训练对不同操作点的差异化增益，并给出奖励组件、风险系数、替代 reward 形式的完整消融。

## 方法详解
- **输入-输出形式化**：对图像-问题对 $(x, q)$，策略模型 $\pi_\theta$ 生成结构化响应 $y = \langle\mathtt{caption}\rangle c \langle/\mathtt{caption}\rangle a$，其中 $c$ 为图像描述，$a$ 为最终答案。
- **训练算法**：使用 GRPO（Shao et al., 2024），每样本采样 $K=8$ 组 rollout，PPO clipped 目标，无显式 KL 惩罚，学习率 $5\times10^{-7}$，训练 200 步。
- **优势函数**：$A_i = w_{\text{tmp}}\tilde{r}_{i,\text{tmp}} + w_{\text{cap}}\tilde{r}_{i,\text{cap}} + w_{\text{ans}}\tilde{r}_{i,\text{ans}}$，权重设为 $(0.5, 0.5, 1.0)$。
- **模板奖励** $R_{\text{tmp}}$：规则解析，要求恰好一个完整 caption 块且 caption/answer 非空、在长度限制内终止，否则跳过下游奖励。
- **直接答案奖励** $R_{\text{ans}} = u(a)\cdot\gamma^{h(a)}$：judge 给出保留分 $u\in[0,5]$ 和风险分 $h\in[0,5]$，$\gamma=0.35$ 控制风险抑制强度；乘法形式避免线性奖励下低信息 caption 被"敷衍拒绝"策略占优。
- **Caption 奖励** $R_{\text{cap}} = g(q, a, a_f)\cdot u(c)$：将 caption 输入冻结 LLM（Qwen3-4B）得到 $a_f$；judge 评估 caption 的描述覆盖率 $u(c)$ 及 binary 安全一致性 $g\in\{0,1\}$（策略回答与冻结 LLM 回答在"是否识别风险"上是否一致）。
- **逐组件组归一化**：$\tilde{r}_{i,k} = (r_{i,k} - \mu_{G,k})/(\sigma_{G,k}+\epsilon)$，借鉴 GDPO 的去耦思想，防止不同方差分量在信用分配阶段耦合。

## 实验与结果
- **数据集**：SPA-VL 公共安全偏好数据集，无私有数据；训练 200 步，8×H200 GPU。
- **评估基准**：5 个安全基准（MM-SafetyBench、MSSBench、VLSBench、FigStep、MIS-Test）+ 6 个视觉效用基准（MM-Vet、BLINK、MMVP、ERQA、VPCT、MMStar）。
- **主要结果（DirectCap 协议）**：
  - 2B：S-Avg +5.29；2B-Base：+5.57；4B：+5.48；**4B-Base：+19.0**（S-Avg 从 40.43 升至 59.39），V-Avg 基本持平（53.13→53.05）。
  - 随机种子鲁棒性（3 次，100-step）：S-Avg $50.83\pm1.14$，V-Avg $54.32\pm1.54$，t 检验 $p=0.004$。
- **对比基线（同 backbone Qwen3.5-4B-Base、同数据 SPA-VL）**：
  - SFT DirectCap S-Avg 仅 43.19，DPO 仅 41.36，SafeCap 达 59.39；V-Avg 三组均明显下降或持平，仅 SafeCap 保持接近零训练水平。
  - 在 SafeGRPO 的 SafeTag-VL-3K 数据上，SafeCap DirectCap S-Avg 55.06 vs SafeGRPO 41.27，V-Avg 51.34 vs 50.56。
- **消融**：移除 answer reward 对 DirectCap 安全降幅最大（-6.49）；移除 caption reward 也显著降安全（-2.53）和效用（-2.41）；移除归一化对 Direct/DirectCap 安全影响大（-2.74/-2.02）；$\gamma=0.35$ 为平衡选择，更小 $\gamma$ 收敛更快但效用略降。
- **最强结果**：Qwen3.5-4B-Base DirectCap，S-Avg 59.39，相对零训练 DirectCap 提升 18.96 点，V-Avg 53.05 几乎不变。

## 相关工作脉络
1. **ECSO（Gou et al., 2024）**：推理时 image-to-text 变换后交由冻结 LLM 判断；本文定位——ECSO 是 wrapper 方法且依赖下游 LLM 的安全能力，SafeCap 将 caption 生成作为可训练的内在路径，而非外部包装。
2. **SPA-VL / VLGuard / MM-RLHF（Zhang et al., 2024; Zong et al., 2024; Zhang et al., 2025b）**：训练时安全 SFT/RLHF 方法；本文定位——这些方法直接优化输出端行为，未显式建模"视觉证据→安全决策"的中间表征通道。
3. **FigStep / HADES / MIS（Gong et al., 2023; Li et al., 2024; Ding et al., 2025）**：展示视觉嵌入攻击使安全对齐失效；本文定位——这些工作揭示问题，SafeCap 提供解决路径。
4. **CapRL（Xing et al., 2025）**：将 caption 质量解耦为 VQA 信号以减少 reward hacking；本文定位——CapRL 关注感知质量，SafeCap 将其扩展至安全一致性信号（冻结 LLM 安全状态对齐）。
5. **SafeGRPO（Rong et al., 2025）**：基于规则治理的 RL 安全对齐；本文定位——SafeGRPO 需额外安全标注，SafeCap 仅需 SPA-VL 公开数据，且 DirectCap 协议表现显著更优。
6. **GDPO（Liu et al., 2026）**：组奖励去耦归一化；本文定位——借鉴其逐组件归一化思想，但不引入 batch-level whitening，适配 GRPO 训练路径。

## 局限性与未来方向
- **评估规模有限**：实验仅在小模型（2B/4B）上完成，更大模型及更多模型家族需后续验证。
- **Caption 奖励受冻结 LLM 能力制约**：Prism 诊断表明，相同 caption 切换 Qwen3-4B→Qwen3-14B 可提升 S-Avg 4.72 点，说明下游 LLM 安全能力是关键瓶颈。
- **无法检测 caption 事实错误**：冻结 LLM 仅见文本，reward 不能识别 hallucination，仅依赖 manual monitoring 未发现系统性幻觉，但缺乏显式约束。
- **SFT 预训练环节缺失**：作者指出 caption 中介的 SFT 作为 RL 前缀有潜力，但缺乏合适的监督数据。
- **Risk 奖励可验证性不足**：依赖人工指定 rubric 的 judge，而非可自动验证的安全 reward。

## 研究启发与可借鉴点
1. **中间表征作为安全证据通道**：将"感知→对齐"解耦为显式的中间生成步骤，可迁移至多轮对话安全、医疗/法律多模态决策等需可解释证据链的场景。
2. **去耦归一化 + 乘法风险折扣的 reward 设计**：逐组件组归一化避免多信号尺度耦合；乘法形式 $u\cdot\gamma^h$ 比线性 $u-\beta h$ 更能抑制"低信息敷衍拒绝"策略，可复用于多目标 RLHF 场景。
3. **多协议诊断范式**：Direct/DirectCap/Prism 三类推理路径可系统分离"原生能力保留""自述能力提升""跨模型迁移性"，为后续工作提供标准化的诊断框架。
4. **无需私有数据的公开数据训练**：仅用 SPA-VL 即取得显著收益，降低工业落地门槛；可直接迁移到团队其他 VLM 安全对齐任务。
5. **与团队方向的结合机会**：若团队关注多模态Agent安全或可视化推理，可将 SafeCap 的 caption 中介奖励适配到"推理轨迹显式化"场景，结合自回归 reasoning 路径做安全对齐。

## 关键术语表
- **LVLM（Large Vision-Language Model）**：将大语言模型扩展至视觉输入的多模态模型。
- **DirectCap**：SafeCap 目标推理协议，模型先生成 tagged caption 再作答。
- **Prism 协议**：诊断协议，将 LVLM 生成的 caption 输入冻结纯文本 LLM 作答，评估 caption 的可迁移性。
- **Caption-mediated reward**：通过冻结 LLM 与安全一致性判断评估 caption 是否支持安全对齐决策的奖励信号。
- **Exponential risk-discount**：$S(u,h)=u\cdot\gamma^h$，乘法风险抑制函数，避免线性 reward 下低质量 rollouts 的相对优势。
- **Component-wise group normalization**：在每组 rollout 内对每个 reward 分量独立标准化，防止多信号尺度耦合。
- **SPA-VL**：公开的 multimodal safety preference 对齐数据集，本文主要训练数据来源。
- **GRPO（Group Relative Policy Optimization）**：基于 group-relative advantage 的 PPO 变体，本文所用 RL 算法。

## 可复现要素
- **数据集**：SPA-VL（公开，https://github.com/...），SafeTag-VL-3K（SafeGRPO 配套，论文使用其发布版本）。
- **代码/权重**：代码开源 https://github.com/Safe-VLM/SafeCap，项目页 https://safe-vlm.github.io/SafeCap/；论文未提供模型权重下载链接。
- **关键超参**：$K=8$ rollouts，batch size 64，mini-batch 64，lr $5\times10^{-7}$，weight decay 0.01，gradient clipping 1.0，entropy reg 0，max tokens 4096，steps 200；$\gamma=0.35$，$(w_{\text{tmp}}, w_{\text{cap}}, w_{\text{ans}})=(0.5,0.5,1.0)$；冻结 LLM 用 Qwen3-4B，judge 用 gpt-oss-20b；KL loss 与 KL reward penalty 均关闭。
