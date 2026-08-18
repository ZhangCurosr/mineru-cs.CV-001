---
title: "Where To Look? : Causal Tracing of Vision Encoders in VLM"
source: https://arxiv.org/pdf/2608.10758v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 08:02:54"
field: "多模态大模型可解释性"
keywords: ["causal tracing", "vision-language models", "mechanistic interpretability", "spatial grounding", "visual representations"]
innovations: ["将因果追踪扩展到VLM视觉编码器并量化因果贡献与空间对齐的关系", "通过string-tracing受控任务揭示模型对外观线索的结构依赖", "建立跨层因果-空间关联分析方法"]
benchmarks: ["CLIP", "DeepSeek-VL", "Qwen-2.5", "LLaVA-NeXT", "LLaVA-1.5", "InternVL", "SmolVLM"]
---

# 论文速读：Where To Look? : Causal Tracing of Vision Encoders in VLM

## 一句话总结
本文通过因果追踪（causal tracing）方法分析VLM视觉编码器的内部表示，发现高因果贡献的视觉token往往并不位于查询相关的目标空间区域内，且模型严重依赖外观线索而非结构信息完成视觉推理任务，揭示了现代VLM在"看到-使用-推理"视觉结构之间存在显著差距。

## 研究问题与动机
- 前沿VLM在基准测试上表现优异，但新基准常导致性能大幅下降，难以判断模型是否真正习得了鲁棒的视觉表征还是利用了数据集捷径
- 现有因果追踪方法主要用于语言模型，缺乏对VLM视觉编码器内部计算机制的系统性分析
- 视觉定位（visual grounding）研究显示注意力与空间相关性可能不一致，但因果影响与空间定位的关系尚未被直接探究
- 模型在去除外观线索后能否保留视觉结构信息，以及这种信息如何被下游语言模型利用，尚不明确

## 核心贡献（创新点）
- **扩展因果追踪到VLM视觉编码器**：将activation patching技术适配到视觉编码器，定义因果贡献度Γ并建立其与空间对齐（IoU）的量化关联框架，区别于以往仅关注语言模型或单独研究因果/定位的工作
- **揭示因果重要性与空间定位的解耦现象**：首次在CLIP及多个大型VLM（DeepSeek-VL、Qwen-2.5、LLaVA系列、InternVL等）中发现高因果贡献token与目标区域的IoU相关性极低（约0.05-0.09），证明强多模态性能不必然伴随空间局部化的因果表征
- **提出结构化视觉推理的受控评估范式**：设计string-tracing任务，在保持几何连通性不变的情况下操控外观线索（颜色一致vs.颜色区分），量化模型对外观线索的依赖程度，揭示视觉编码器在保留长程结构信息方面的局限性
- **建立因果-空间关联的跨层分析框架**：提出逐层计算Corr_l(Γ, IoU)的方法，追踪空间 grounding 在视觉编码器各层的发育规律，为ViT架构VLM提供可复用的表征可解释性分析工具

## 方法详解
- **因果贡献度量**：对图像I和文本查询T，计算图像-文本相似度s(I,T)=⟨f_img(I), f_text(T)⟩；通过三次前向传播（clean/corrupted/patched）定义层-token对的因果贡献Γ_{l,t}=(s_patch−s_corr)/(s_clean−s_corr)，值接近1表示该激活对相似度有强因果影响
- **空间对齐量化**：将ViT每个token映射到对应图像patch区域P_t，与referencing expression的地框B计算IoU(P_t,B)=|P_t∩B|/|P_t∪B|，然后计算Corr(Γ,IoU)衡量因果重要性与空间重叠的关联强度
- **逐层分析**：对每层l独立计算Corr_l=Corr(Γ_{l,t}, IoU_t)，识别空间grounding在哪一层开始与因果重要性对齐
- **String-tracing任务**：生成合成拼图，多条视觉上重叠的弦连接端点对；模型需从一端追踪到另一端；设置同色条件（仅依赖几何结构）和异色条件（颜色提供额外线索），通过端点准确率Accuracy=(1/N)ΣI(ŷ_i=y_i)评估
- **两种图像损坏方式**：DDPM噪声破坏和模糊破坏，用于创建clean-corrupted对比状态，验证结果的鲁棒性

## 实验与结果
- **CLIP基础实验**：100个样本上Corr(Γ,IoU)=0.062，高因果token（如Γ=0.718）对应的IoU仅0.049，证明因果重要性≠空间定位
- **大型VLM扩展**：DDPM破坏下DeepSeek-VL相关系数0.0531±0.0334、Qwen-2.5为0.0912±0.0422、LLaVA-NeXT为0.0419±0.0166、LLaVA为0.0509±0.0329、InternVL为0.0397±0.0224；blur破坏下数值更低甚至为负
- **最高因果贡献存在**：Qwen-2.5在DDPM下Top-1最大Γ达0.5365±0.3822，blur下达0.8034±0.0868，说明低相关系数并非因缺乏高因果token，而是因果重要token未稳定对应目标区域
- **跨随机种子稳定性**：DeepSeek-VL、Qwen-2.5、LLaVA-NeXT、LLaVA各10次运行结果一致，排除随机性干扰
- **String-tracing任务**：Gemini-3.1 Pro在同色条件下难以正确追踪端点，而异色条件下表现显著更好，表明模型依赖颜色等外观线索而非纯结构信息

## 相关工作脉络
- **Palit et al. (2023)**：最早将因果追踪扩展到BLIP等VLM，证明中间视觉表示与多模态预测存在因果关联，本文在此基础上引入空间对齐维度的量化分析
- **Esmaeilkhani & Latecki (2026)**：指出VLM难以遵循连通视觉结构，并提出显式定位方法；本文通过string-tracing任务进一步验证并量化了结构推理的局限性
- **Schaumlöffel et al. (2026)**：研究VLM中物体定位机制；本文与之互补，区分了"因果重要性"与"空间grounding"两个不同属性
- **CLIP (Radford et al., 2021)**：作为基础实验对象验证方法论可行性，后续扩展至DeepSeek-VL、Qwen-2.5、LLaVA系列、InternVL、SmolVLM等现代VLM
- **ViT (Dosovitskiy et al., 2020)**：提供token-patch空间映射的基础架构假设，本文明确讨论该方法在SigLIP等非标准ViT架构中的泛化限制

## 局限性与未来方向
- 当前分析主要基于ViT架构，尚未扩展到SigLIP等新兴视觉编码器架构，无法确定因果-空间解耦是通用属性还是架构特有条件
- 纵向评估计划覆盖从早期到SOTA的多代VLM，但现阶段仅包含2023年后的模型，缺少历史对比
- String-tracing实验仅在单个proprietary模型（Gemini-3.1 Pro）上验证，需要扩展到开源模型族
- 因果追踪依赖图像损坏操作，不同损坏方式（DDPM vs. blur）导致测量值有波动，单一损坏机制可能不足
- 无法区分视觉编码器信息丢失与语言模型推理能力不足在string任务失败中的相对贡献

## 研究启发与可借鉴点
- **因果追踪与空间对齐的联合分析框架**可直接迁移到其他多模态模型的表征可解释性研究，如多模态大语言模型（MLLMs）或 diffusion models 的文本引导生成
- **String-tracing受控任务设计**提供了剥离外观线索、测试纯结构推理能力的通用范式，可推广到几何推理、拓扑关系理解等视觉认知能力评估
- **逐层Corr_l分析**可作为VLM视觉编码器设计的诊断工具，帮助识别哪些层贡献了空间定位能力，指导编码器架构改进
- 本文发现的"因果-空间解耦"现象暗示可通过引入显式空间约束训练（如 grounding loss）或修改预训练目标来改善VLM的空间推理能力
- 外观线索依赖的发现提示可在数据增强或对抗训练中引入外观扰动，迫使模型学习更鲁棒的结构化表征

## 关键术语表
**Causal Tracing（因果追踪）**：通过替换模型内部激活值来量化特定表征对输出的因果影响的可解释性方法
**Activation Patching（激活修补）**：将 corrupted 输入经过某层的激活替换为 clean 输入对应层的激活，以测量该位置的因果贡献
**Intersection-over-Union (IoU)**：衡量token对应图像patch与目标边界框空间重叠程度的指标
**Spatial Grounding（空间定位）**：模型表征与输入图像中任务相关空间区域的对齐程度
**Vision Encoder（视觉编码器）**：将输入图像转换为token序列的神经网络模块，本文主要基于ViT架构
**String-tracing Task（弦追踪任务）**：要求模型沿连续几何结构从一端追踪到另一端的合成视觉推理任务
**Corruption Robustness（损坏鲁棒性）**：测试不同图像损坏方式（DDPM噪声、模糊）下因果分析结果的稳定性

## 可复现要素
- **数据集**：合成生成的string-tracing拼图数据集，自定义Python生成器控制弦数量、轨迹、端点位置和视觉属性
- **代码/权重**：论文未提及代码开源情况；使用的VLM包括CLIP、DeepSeek-VL、Qwen-2.5、LLaVA-NeXT、LLaVA、InternVL-2、SmolVLM等公开模型
- **关键超参**：DDPM和blur两种损坏方式；10个随机种子重复实验（部分模型6次）；字符串颜色一致性条件对比
