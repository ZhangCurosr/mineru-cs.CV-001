---
title: "FOR-TEXT-IMAGE-TO-VIDEO-DIFFUSION-MODELS"
source: https://arxiv.org/pdf/2608.13205v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:59:33"
field: "视频生成模型"
keywords: ["TI2V", "self-distillation", "on-policy distillation", "flow matching", "video generation", "hybrid-policy", "diffusion models"]
innovations: ["提出HPSD混合策略自蒸馏框架，解决TI2V场景下on-policy蒸馏的条件-状态不匹配问题", "首次系统形式化特权条件激发的能力内化任务，实现跨条件生成能力迁移", "设计子轨迹长度K作为离/在线策略插值参数，在保留教师先验与策略对齐间取得平衡"]
benchmarks: ["VBench", "VideoAlign", "VisionReward", "UnifiedReward-v2", "HPS Score", "CLIP Score"]
---

# 论文速读：FOR-TEXT-IMAGE-TO-VIDEO-DIFFUSION-MODELS

## 一句话总结
论文提出 **Hybrid-Policy Self-Distillation (HPSD)**，通过"教师锚定+学生演化"的混合策略设计，将 TI2V 模型在特权条件（高质量首帧+增强提示）下激发的生成能力内化到模型的基础 T2V 生成能力中，同时规避了 off-policy SFT 的暴露偏差与 on-policy 蒸馏的条件-状态不匹配问题。

---

## 研究问题与动机

1. **能力鸿沟**：TI2V 模型在特权条件（高质量首帧、增强 prompt）下生成的视频质量显著优于基础 T2V 模式，但这部分"特权能力"未被内化，导致纯文本驱动的 T2V 基线能力受限。
2. **Off-policy SFT 的暴露偏差**：直接在教师生成的固定轨迹上微调学生，监督信号来自固定离线分布，随着学生策略演化，固定标签与学生实际访问的状态逐渐偏离，无法提供精确的状态感知修正。
3. **On-policy 蒸馏的条件-状态不匹配（Condition-State Mismatch）**：在 TI2V 场景下，将教师查询于学生访问的中间状态时，教师要求首帧为干净条件 $c_{img}$，而学生生成的是无首帧约束的自由运行内容，导致强制拼接产生无效混合输入，教师输出错误速度场，误导去噪过程。
4. **核心科学问题**：如何在不引发条件-状态冲突的前提下，让学生既能继承教师特权条件的先验内容，又能接受与其当前策略对齐的精细修正？

---

## 核心贡献（创新点）

1. **发现并形式化"条件激发能力"（Condition-Elicited Capability）概念**：首次系统揭示 TI2V 模型中特权条件与基础模式的能力差距，并将"能力内化"定义为自蒸馏新任务。与已有工作仅关注蒸馏加速或质量提升不同，本文聚焦统一架构下跨条件能力的迁移。

2. **提出 Hybrid-Policy Self-Distillation (HPSD)**：学生从教师轨迹锚定点出发、用自己的策略完成子轨迹演化，并在终端接受速度级监督。与 off-policy 的区别在于保留了策略感知的精细修正；与 on-policy 的区别在于锚定点避免了条件-状态冲突。

3. **揭示 TI2V 场景下 on-policy 蒸馏的失效机制**：形式化定义并可视化展示了条件-状态不匹配现象（Fig.3），指出现有 OPD 方法直接迁移到 TI2V 会导致生成崩溃（如早期帧模糊、内容不连贯）。

4. **双模型验证+TI2V 模式协同增益**：在 WAN-2.2-TI2V-5B 和 LTX-2.3 两个代表性统一模型上验证，HPSD 不仅大幅提升 T2V 基线，也同步增强 TI2V 模式表现，证明共享权重的整体能力提升。

---

## 方法详解

### 整体架构
HPSD 采用"单模型双角色"设置：同一 TI2V 模型在不同条件下分别扮演教师与學生。教师运行于 TI2V 特权模式（增强 prompt $c_{txt}^+$ + 高质量首帧 $c_{img}$），学生运行于基础 T2V 模式（原始 prompt $c_{txt}$）。

### 关键步骤

**Step 1：特权条件构建（离线阶段）**
- 使用外部 LLM（Qwen3.6-27B）将原始 prompt 重写为增强 prompt $c_{txt}^+$，并生成首帧描述 $c_{ff}$
- 使用辅助 T2I 模型（Z-Image-Turbo）渲染高质量首帧 $c_{img}$
- 所有特权条件预计算后缓存，不参与训练梯度

**Step 2：离策略锚点轨迹（Off-Policy Anchor Trajectory）**
- 教师沿特权条件 $(c_{txt}^+, c_{img})$ 完整 roll-out 去噪轨迹，得到 anchor 状态集合 $\{\hat{x}_{t_i}^{Tea}\}$
- 将教师锚点状态转换为兼容学生输入的格式：对教师始终干净的首帧槽重新加噪至当前时间步 $t_i$
  $$x_{t_i}^{Tea} = [(1-t_i)c_{img} + t_i\epsilon,\; \hat{x}_{t_i}^{Tea,(2:F)}]$$
  使所有帧共享同一噪声水平，匹配学生在 T2V 推理时看到的输入格式。

**Step 3：混合策略子轨迹（Hybrid-Policy Sub-Trajectory）**
- 学生从每个转换后的锚点 $x_{t_i}^{Tea}$ 出发，用自己当前策略 $v_\phi$ 继续去噪 $K$ 步，得到子轨迹终端状态 $x_{t_{i+K}}^{Hyb}$
- 在终端状态处重新施加干净首帧 $c_{img}$，构造教师查询输入：
  $$\hat{x}_{t_{i+K}}^{Hyb} = [c_{img},\; x_{t_{i+K}}^{Hyb,(2:F)}]$$
- 损失函数（仅对第 2 帧及之后生效）：
  $$\mathcal{L}_{HPSD} = \mathbb{E}_{t_i \sim \mathcal{A}} \| v_\phi(x_{t_{i+K}}^{Hyb}, t_{i+K}|c_{txt}) - \text{sg}[\tilde{v}(\hat{x}_{t_{i+K}}^{Hyb}, t_{i+K}|c_{txt}^+, c_{img})] \|_{(2:F)}^2$$

**Step 4：教师 EMA 更新**
- 教师 $ \tilde{v} $ 通过学生权重 $v_\phi$ 的指数移动平均（EMA decay=0.999）更新，保证目标分布稳定。

### 关键设计洞察
- **子轨迹长度 $K$**：插值参数——$K=0$ 退化为纯离策略（SFT），$K$ 增大趋近纯 on-policy；实验表明 $K=3$ 为最佳折中。
- **仅监督第 2~F 帧**：因教师首帧槽始终为干净条件，其速度无去噪含义，故跳过第 1 帧。
- **无梯度中间步骤**：教师 roll-out 和学生前 K 步演化均在 inference 模式下执行，不存储计算图，避免额外显存开销。

---

## 实验与结果

### 实验设置
- **训练数据集**：~50K prompts（来自 Pref-GRPO 数据集），覆盖多样主题
- **Base 模型**：WAN-2.2-TI2V-5B（1280×704，50 steps）、LTX-2.3（768×512，30 steps）
- **训练配置**：8×H200 GPU，batch size=1/GPU，500 steps，AdamW，lr=$1\times10^{-4}$，LoRA（r=32, α=64），bf16
- **评估集**：500 条 prompt（来自 VideoDPO + VideoFeedback）
- **评估指标**：VideoAlign、VisionReward、UR-v1/v2（Alignment/Physics/Style）、HPS、CLIP Score、VBench（SC/BC/MS/DD/AQ/IQ）

### 主要结果（WAN-2.2）
| 方法 | VideoAlign | UR-v2-A | UR-v2-P | UR-v2-S |
|------|-----------|---------|---------|---------|
| Vanilla T2V | 0.5335 | 2.802 | 3.167 | 3.100 |
| SFT | 1.2046 | 2.854 | 3.181 | 3.207 |
| On-Policy | 0.2613 | 2.724 | 3.171 | 3.204 |
| **HPSD（Ours）** | **1.8753** | **2.890** | **3.203** | **3.275** |

- **VideoAlign 最强结果**：1.8753，较第二名 SFT（1.2046）**提升约 55%**
- HPSD 在 WAN-2.2 上 **8 项指标全部第一**

### 主要结果（LTX-2.3）
| 方法 | VideoAlign | UR-v2-A |
|------|-----------|---------|
| Vanilla T2V | 0.2307 | 2.877 |
| SFT | 0.9584 | 2.873 |
| **HPSD（Ours）** | **1.5244** | **2.887** |

- HPSD 在 LTX-2.3 上 **7/8 项指标第一**
- On-Policy 蒸馏在 LTX-2.3 上 VideoAlign 仅 0.0242（严重崩溃），印证条件-状态不匹配危害

### VBench 关键维度
- **Subject Consistency（SC）**：HPSD 在 WAN-2.2 上达 0.9722（vs Vanilla 0.9654）
- **Aesthetic Quality（AQ）**：HPSD 在 WAN-2.2 上达 0.6343（vs Vanilla 0.5773）
- **Dynamic Degree（DD）**略低，但作者论证反映更好的时序一致性和物理合理性（UR-v2-P 提升佐证）

### 消融结论
- $K=3$ 最优（Tab.3a）：过大导致偏离开启锚点太远，过小则偏离 on-policy 收益
- 特权条件叠加有效：仅 $+c_{img}$ 提升明显，$+c_{txt}^+ + c_{img}$ 达到峰值（Tab.3b）
- TI2V 模式同步增益：VideoAlign 从 0.7831 → 1.2139（**约 +55%**）（Tab.3c）

---

## 相关工作脉络

1. **Supervised Fine-Tuning on Teacher Outputs（Off-policy SFT）**：直接监督教师生成终点的去噪版本。本质区别：HPSD 引入学生策略演化，使监督状态与学生访问分布对齐，而非停留在固定离线分布。

2. **On-Policy Distillation (OPD) for Diffusion Models**（如 Jiang et al., 2026; Fang et al., 2026）：在 student-visited states 上查询教师提供密集监督。本质区别：HPSD 通过锚点机制避免 TI2V 特有的条件-状态不匹配问题，而 OPD 方法在该场景下直接失效。

3. **D-OPSD（Jiang et al., 2026）**：自蒸馏方法，需要多模态编码器注入图像条件。本质区别：HPSD 无需额外编码模块，利用统一 TI2V 架构自身支持的条件切换；且 D-OPSD 只能蒸馏文本特权条件，无法处理图像条件。

4. **Flow Matching / Diffusion Distillation（Lipman et al., 2022; Yin et al., 2024b）**：关注步数压缩（step distillation）或分布匹配。本质区别：HPSD 关注的是**能力内化**而非推理加速，针对统一多模态架构的跨条件迁移问题。

5. **Prompt Enhancement / Inference-time Engineering**（如 Wang et al., 2025a; Cheng et al., 2025）：在推理时通过提示优化提升生成质量。本质区别：HPSD 将特权条件的增益**内化进模型权重**，无需推理时额外工程。

6. **LLM On-Policy Distillation**（如 Shenfeld et al., 2026; Yang et al., 2026）：将 OPD 范式从语言模型迁移到视觉生成。本质区别：HPSD 针对 diffusion/flow models 的连续状态空间与多条件架构重新设计了锚定机制。

---

## 局限性与未来方向

1. **辅助模型依赖**：特权条件构建需调用外部 LLM 和 T2I 模型，增加数据合成成本与流水线复杂度；虽为一次性离线开销，但仍需部署高质量辅助模型。
2. **额外前向传播开销**：训练时需多轮教师/学生 roll-out，相比 SFT 和 OPD 增加计算量；虽不需存储梯度，但延迟上升。
3. **子轨迹长度 K 需调优**：K 值选择影响离/在线平衡，不同模型规模/分辨率下的最优 K 可能不同。
4. **未来方向**：
   - 探索更轻量/蒸馏化的辅助条件合成模型，降低离线成本
   - 将 HPSD 推广至其他多条件统一生成架构（如 3D、音频-视频联合生成）
   - 探索 K 的自适应学习机制，而非固定超参
   - 研究特权条件的自动搜索/优化，而非依赖人工 prompt engineering

---

## 研究启发与可借鉴点

1. **"锚定+演化"混合策略范式可迁移**：HPSD 的核心思想——"从高质量先验状态出发，沿学生自身策略演化后接受监督"——可推广到其他需要跨条件能力迁移的生成任务（如图像修复、风格迁移、3D 生成）。

2. **条件-状态不匹配的识别与规避策略**：本文形式化诊断了多条件模型中 on-policy 蒸馏的失效机理，为后续研究提供了"先诊断再设计"的方法论借鉴，避免盲目套用 LLM 领域的 OPD 经验。

3. **子轨迹长度 $K$ 的插值设计**：通过单一超参在离策略与在线策略间连续调节，为蒸馏方法的设计提供了一个简洁可控的"安全阀"，值得在其他蒸馏框架中借鉴。

4. **TI2V 模式协同增益的发现**：针对 T2V 优化的蒸馏方法同时提升了 TI2V 模式，提示共享权重架构中不同条件模式间存在隐式知识耦合，可作为后续"一体化能力增强"研究的切入点。

5. **实验设计亮点**：使用多种 reward model（VideoAlign、VisionReward、UR-v2）+ 结构化 VBench 多维度评估，并辅以丰富的 qualitative comparison，为视频生成蒸馏论文的实验规范提供了参考。

---

## 关键术语表

**TI2V（Text-Image-to-Video）**：统一支持文本到视频和图像到视频生成的扩散模型架构，共享权重集，可选首帧图像作为条件。

**Privileged Condition（特权条件）**：能显著激发模型更强生成能力的额外输入，包括高质量首帧图像和增强 prompt，二者组合效果最佳。

**Condition-Elicited Capability（条件激发能力）**：由特权条件激活、远超基础模式的生成能力；本文目标将其内化入模型通用权重。

**Condition-State Mismatch（条件-状态不匹配）**：on-policy 蒸馏在 TI2V 中的特有失败模式——教师被查询于学生生成的混合状态时，强制干净首帧与学生自由内容冲突，导致教师输出错误速度场。

**Hybrid-Policy（混合策略）**：学生从教师锚点出发、用自己的策略演化子轨迹的设计，同时融合离策略的内容先验和在线策略的状态对齐优势。

**Off-Policy Anchor Trajectory（离策略锚点轨迹）**：教师在特权条件下生成的完整去噪轨迹，其中间状态经格式转换后作为学生 roll-out 的起点。

**Sub-Trajectory Length K**：控制学生从锚点出发后自行演化步数的超参，$K=0$ 退化为纯 off-policy，$K$ 越大越趋近 on-policy。

**EMA Teacher（指数移动平均教师）**：教师权重通过学生权重的滑动平均实时更新，保证目标分布稳定，避免 on-policy 训练中的目标震荡。

---

## 可复现要素

- **数据集**：~50K prompts（来自 Pref-GRPO 数据集），论文声明代码将开源
- **代码**：论文声明"HPSD code will be released at HPSD Repo"，截至论文提交时尚未公开
- **权重**：Base 模型 WAN-2.2-TI2V-5B 和 LTX-2.3 为开源模型；LoRA 权重应随代码一并发布
- **关键超参**：
  - 训练步数：500 steps
  - Batch size：1 per GPU × 8 GPUs
  - Learning rate：$1\times10^{-4}$
  - LoRA：r=32, α=64
  - Sub-trajectory length K：3
  - Anchor steps |A|：6
  - EMA decay：0.999
  - Mixed precision：bfloat16
  - 训练帧数：33 frames
- **辅助模型**：Qwen3.6-27B（prompt rewriter）、Z-Image-Turbo（首帧生成器）；附录 D 验证了 Flux.2-Klein-4B 的泛化性

---
