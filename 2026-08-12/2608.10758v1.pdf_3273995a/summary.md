---
title: "Where To Look? : Causal Tracing of Vision Encoders in VLM"
source: https://arxiv.org/pdf/2608.10758v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 08:03:52"
field: "多模态可解释性"
keywords: ["causal tracing", "vision-language models", "mechanistic interpretability", "spatial grounding", "visual representations", "activation patching"]
innovations: ["首次系统量化视觉token因果贡献与空间IoU对齐的关系，发现强因果性token常位于目标区域外", "提出因果-空间失配概念并通过跨6个主流VLM的实证验证其普遍性", "设计受控字符串追踪任务分离外观线索与结构信息，揭示模型对颜色的依赖"]
benchmarks: ["CLIP causal tracing实验", "DeepSeek-VL/Qwen-2.5/LLaVA系列/InternVL/SmolVLM跨模型评估", "String Puzzle Task（同色/异色条件）"]
---

# 论文速读：Where To Look? : Causal Tracing of Vision Encoders in VLM

## 一句话总结
论文通过因果追踪（causal tracing）方法系统研究VLM视觉编码器的内部表示，发现高度因果性的视觉token往往不与目标查询区域在空间上对齐；同时，通过受控的字符串追踪任务揭示模型依赖外观线索（如颜色）而非结构信息来完成推理，表明**强多模态性能并不等同于空间局部化的因果表示**。

## 研究问题与动机
- **性能-可解释性鸿沟**：前沿VLM在各类基准上取得优异表现，但新基准常出现性能骤降，引发疑问——模型是真正习得了鲁棒视觉表征，还是利用了数据集特定的捷径？
- **因果追踪的视觉扩展**：因果追踪技术已从语言模型扩展到VLM（如Palit et al., 2023），但现有工作并未直接检验"因果重要token是否与空间相关区域对齐"这一核心问题。
- **视觉 grounding 与因果影响脱节**：视觉定位（visual grounding）研究与因果分析各自独立发展，缺乏将二者统一考察的系统性框架。
- **结构保留 vs 外观依赖**：VLM在连接复杂视觉结构时表现不佳（Esmaeilkhani & Latecki, 2026），需探究视觉编码器是否能在去除外观线索后仍保留结构信息。

## 核心贡献（创新点）
1. **将激活修补（activation patching）扩展至视觉编码器**，构建因果贡献Γ与空间IoU对齐的量化关联框架，首次系统度量视觉token因果重要性与空间定位的关系。
2. **发现并实证"因果-空间失配"现象**：高度因果性token通常不在目标区域内，表明因果重要性（causal importance）与空间接地（spatial grounding）是视觉表征的两个正交属性。
3. **引入受控字符串追踪任务（string-puzzle）**，通过同色/异色条件的对照实验，证明模型依赖低层外观线索（颜色）而非结构连通性进行推理。
4. **跨模型、跨损坏类型、跨随机种子的稳健性验证**：从CLIP到DeepSeek-VL、Qwen-2.5、LLaVA系列、InternVL、SmolVLM，因果-空间相关性在所有模型中均保持低位（绝对值<0.1），且不受DDPM/blur损坏类型影响。
5. **提出纵向评估框架假设**：随着模型能力提升，视觉表征的空间局部化程度并未同步提升，暗示更强的VLM性能来源于分布式视觉信息利用而非更精确的空间编码。

## 方法详解
### 因果贡献度量
采用三阶段前向传播计算token级因果贡献Γ(l,t)：
- **Clean run**：输入原始图像I和文本查询T，记录相似度s_clean
- **Corrupted run**：输入加噪图像（Gaussian noise/DDPM/blur），记录s_corr
- **Patched run**：在损坏图像上，将指定层l和token t的激活替换为clean run中对应激活，记录s_patch

$$\Gamma_{l,t} = \frac{s_{patch} - s_{corr}}{s_{clean} - s_{corr}}$$
Γ≈1表示该激活几乎完全恢复相似度差距（强因果效应），Γ≈0表示无影响。

### 空间对齐度量
- 每个ViT token对应固定大小的图像patch P_t
- ground-truth边界框B来自referencing expression标注
- 空间对齐用IoU度量：$$\mathrm{IoU}(P_t, B) = \frac{|P_t \cap B|}{|P_t \cup B|}$$
- 核心指标：$$\mathrm{Corr}(\Gamma, \mathrm{IoU})$$ —— 因果贡献与空间重叠的全局相关性

### 逐层分析
计算每层l的独立相关性：$$\mathrm{Corr}_l = \mathrm{Corr}(\Gamma_{l,t}, \mathrm{IoU}_t)$$，观察空间接地是否在特定层逐渐增强。

### 字符串追踪任务
- **同色设置**：所有字符串颜色相同，唯一可靠线索为几何连通性
- **异色设置**：每根字符串颜色不同，颜色可作为额外的判别性外观线索
- 通过固定几何拓扑仅改变外观属性，隔离模型对结构 vs 外观的依赖程度
- 评估指标：endpoint accuracy（预测正确端点的比例）

## 实验与结果
### 数据集与模型
- **基础实验**：CLIP，100个样本
- **扩展实验**：DeepSeek-VL、Qwen-2.5、LLaVA-NeXT、LLaVA-1.5、InternVL、SmolVLM
- **损坏类型**：DDPM噪声 + Gaussian blur
- **重复性**：DeepSeek-VL/Qwen/LLaVA-NeXT/LLaVA各10次随机种子；InternVL 6次

### 核心结果（因果-空间相关性 Corr(Γ, IoU)）

| 模型 | DDPM | Blur |
|------|------|------|
| CLIP | 0.062 | — |
| DeepSeek-VL | 0.0531±0.0334 | 0.0520±0.0041 |
| Qwen-2.5 | 0.0912±0.0422 | 0.0294±0.0100 |
| LLaVA-NeXT | 0.0419±0.0166 | -0.0217±0.0052 |
| LLaVA-1.5 | 0.0509±0.0329 | 0.0460±0.0001 |
| InternVL | 0.0397±0.0224 | 0.0121±0.0033 |
| SmolVLM | 0.0011±0.0000 | -0.0019±0.0000 |

- **关键发现**：所有模型的Corr均在[-0.02, 0.09]区间，呈现弱相关甚至负相关
- **强因果token ≠ 空间对齐**：如Qwen-2.5在DDPM下max Γ_top-1 = 0.5365±0.3822，但对应IoU极低；LLaVA-1.5 max Γ达0.9451，却与目标区域无空间关联
- **损坏鲁棒性**：DDPM和blur两种损坏类型下趋势一致，仅LLaVA-NeXT在blur下出现微弱负相关（-0.0217）
- **随机种子稳定性**：多种子实验结果一致，排除偶然性

### 字符串追踪结果
- **同色设置**：Gemini-3.1 Pro在交叉/重叠字符串场景中表现显著下降
- **异色设置**：相同几何结构下性能明显提升
- **结论**：颜色线索大幅辅助追踪任务，而纯结构信息不足以支撑可靠推理

## 相关工作脉络
1. **Palit et al. (2023)**：将因果追踪应用于BLIP，首次展示中间视觉表示可因果影响多模态预测；本文扩展其方法至vision encoder层级，并首次引入空间IoU对齐度量。
2. **Esmaeilkhani & Latecki (2026)**：证明VLM中attention与空间相关性可分离，并提出显式grounding方法；本文不从attention角度出发，而是直接测量causal contribution与spatial grounding的关联。
3. **Schaumlöffel et al. (2026)**：研究VLM中物体定位机制；本文聚焦更基础的表征问题——视觉token的因果重要性是否隐含空间信息。
4. **Mechanistic interpretability in VLMs**：因果追踪原由 language model interpretability 发展而来（如Pearlmutter et al.），本文将其适配到vision encoder并建立与空间定位的定量联系。
5. **Visual grounding studies**：传统视觉定位研究关注模型是否关注正确区域；本文指出"关注"（attention/causal）与"空间对齐"（IoU overlap）可完全解耦，为grounding研究提供新的诊断维度。

## 局限性与未来方向
- **仅覆盖ViT架构**：当前分析局限于ViT-based编码器（因token与patch一一对应），未验证SigLIP等非标准架构下的泛化性。
- **损坏类型有限**：仅测试DDPM和blur两种损坏方式，其他损坏（如遮挡、裁剪、对抗扰动）的影响未探索。
- **字符串任务样本量小**：初步观察基于单一模型（Gemini-3.1 Pro）的少量实例，缺乏统计显著性。
- **纵向评估未完成**：跨模型代际的演化分析仅提出计划，尚未执行。
- **未来方向**：扩展至SigLIP等架构、跨模型世代纵向比较、探索因果-空间对齐与模型能力的关系、分析结构信息损失发生在vision encoder还是language model推理阶段。

## 研究启发与可借鉴点
1. **因果追踪+空间IoU的联合度量框架**可直接迁移至团队后续对视觉-语言对齐机制的研究中，提供一种同时刻画"因果影响"和"空间精确性"的双重诊断工具。
2. **受控外观-结构剥离实验设计**（同色/异色字符串）为研究模型结构推理能力提供了简洁可控的测试范式，可用于评估不同视觉编码器的结构保留能力。
3. **逐层相关性分析**（Corr_l across layers）可帮助定位视觉表征中空间信息的演化阶段，识别哪些层负责/不负责空间接地，为模型改进提供干预目标。
4. **强因果token≠空间对齐**的发现提示团队：在依赖视觉token进行下游任务（如VQA、grounding）时，需警惕模型可能利用非局部外观线索而非真正的空间推理。
5. **破坏性干预鲁棒性评估**（多种损坏类型+多随机种子）的实验设计规范值得借鉴，确保因果结论不被特定干预手段或随机性偏差所影响。

## 关键术语表
**Causal Tracing（因果追踪）**：通过向中间激活注入扰动或替换，测量特定神经元/token对模型输出的因果影响机制。
**Activation Patching（激活修补）**：将干净样本的某层某token激活替换到损坏样本对应位置，观察输出变化以量化因果贡献。
**Causal Contribution Γ（因果贡献）**：修复后相似度差距恢复的比例，Γ=1表示完全恢复，Γ=0表示无因果效应。
**IoU（Intersection-over-Union）**：token对应图像patch与ground-truth目标边界框的重叠比率，用于量化空间对齐程度。
**Spatial Grounding（空间接地）**：模型视觉表征与图像空间区域的对齐程度，即模型是否"看"在正确的地方。
**Vision Encoder（视觉编码器）**：将输入图像转换为token序列的模块（如ViT、SigLIP），是VLM理解视觉信息的前置组件。
**String Puzzle Task（字符串追踪任务）**：通过重叠/交叉字符串的端点匹配任务，控制性地分离结构信息与外观线索的研究工具。
**Causal-Spatial Mismatch（因果-空间失配）**：高度因果性token与目标区域空间位置不对齐的现象，本文的核心发现。

## 可复现要素
- **数据集**：CLIP实验使用100个样本（具体数据集名称论文未明确提及）；字符串追踪任务由自定义Python生成器合成
- **代码开源**：论文未提及代码开源
- **权重**：使用的模型（CLIP、DeepSeek-VL、Qwen-2.5、LLaVA系列、InternVL、SmolVLM、Gemini-3.1 Pro）均为公开可用模型
- **关键超参**：损坏类型（DDPM/Gaussian blur）、token top-k选择策略、随机种子重复次数（10次/6次）；论文未明确报告学习率、batch size等训练超参（因使用预训练模型）
