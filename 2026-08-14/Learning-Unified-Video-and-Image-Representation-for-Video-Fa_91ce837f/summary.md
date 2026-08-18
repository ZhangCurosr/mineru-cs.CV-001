---
title: "Learning-Unified-Video-and-Image-Representation-for-Video-Fa"
source: https://arxiv.org/pdf/2608.13064v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:48:29"
field: "视频人脸伪造检测"
keywords: ["Face Forgery Detection", "Video Deepfake Detection", "Unified Representation", "Partial Forgery", "Multiple Instance Learning", "Feature Alignment", "Pseudo Labeling"]
innovations: ["提出统一视频-图像表示框架UVIF，利用图像细粒度标注缓解部分伪造视频帧级监督缺失问题", "设计伪标签生成与视频导向特征对齐策略，桥接图像与视频帧分布并提升帧级判别力"]
benchmarks: ["ForgeryNet", "DFDC"]
---

# 论文速读：Learning-Unified-Video-and-Image-Representation-for-Video-Face-Forgery-Detection

## 一句话总结
本文提出 UVIF（Unified Video and Image Representation for face Forgery detection）框架，通过将标注图像作为细粒度监督信号引入视频人脸伪造检测任务，在统一编码器内联合建模视频与图像，以解决现有方法难以有效检测**部分伪造视频**（即视频中仅含少量篡改帧）的问题。

## 研究问题与动机
1. **部分伪造视频的检测难题**：现实场景中伪造视频往往只含少量篡改帧（部分伪造），而现有方法普遍假设视频中所有帧均为伪造，导致对部分伪造视频的泛化能力有限。
2. **现有视频检测方法缺乏帧级监督**：视频类方法（如 TSM、VideoSwin）仅使用视频级标签训练，无法区分真帧与伪造帧；在部分伪造视频下，梯度会同时将真实帧推向伪造分布，破坏帧级表示学习。
3. **现有图像类方法存在标签噪声**：图像类方法将视频所有帧视为同一标签，在部分伪造视频下会将真实帧错误标注为伪造，引入训练噪声。
4. **图像标注资源丰富且与视频帧表征相似**：现有人脸伪造数据集（如 ForgeryNet）提供了大量带精确标注的静态人脸图像，图像与单帧视频在表示学习上具有共性，可作为细粒度监督信号补充帧级标注的缺失。

## 核心贡献（创新点）
1. **提出 UVIF 统一框架**：在单一模型内联合处理视频与图像输入，利用图像类标注为视频部分伪造检测提供细粒度监督，而不仅依赖视频级标签。与以往仅用视频数据或简单拼接两种模态不同，本文通过共享编码器与多任务学习实现表示层面的统一。
2. **设计伪标签生成过程**：通过图像分类头对弱增强视频帧生成软伪标签，再指导强增强帧的帧级分类头学习，从而在无帧级真实标签情况下为视频帧引入细粒度监督。与单纯 MIL 弱监督不同，该方法借助图像监督生成更可靠的帧级伪标签。
3. **提出视频导向的特征对齐策略**：以图像特征为锚点，在编码器中间层进行跨分布的特征插值，并强制插值结果偏向图像分布，缩小图像与视频帧之间的分布差异。与 manifold mixup 仅在同类分布内插值不同，本文的插值显式桥接图像与视频两类分布，并保持对图像头的兼容性。
4. **系统验证数据效率与跨骨干通用性**：实验表明仅需 10k 标注图像即可显著提升性能，且在 ForgeryNet 和 DFDC 上与多种 SOTA 视频/图像/MIL 基线相比取得领先，同时推理时不增加额外计算开销。

## 方法详解
1. **统一视频与图像建模**  
   - 使用共享参数的统一编码器 $\phi(\cdot)$，输入为 4D 张量 $X \in \mathbb{R}^{T \times 3 \times H \times W}$（图像时 $T=1$）。  
   - 采用标准 2D CNN 或 Transformer 骨干，视频输入额外接入时序融合模块（CNN 用 TSM，Transformer 用 temporal attention）。  
   - 编码器输出平均池化后分别进入视频分类头 $h_v(\cdot)$ 和图像分类头 $h_s(\cdot)$，计算视频级交叉熵 $\mathcal{L}_{\text{video}}$ 与图像级交叉熵 $\mathcal{L}_{\text{image}}$，实现多任务联合优化。

2. **伪标签生成（Bridging Video and Image Representation）**  
   - 对每个视频帧 $v_t$ 生成弱增强视图 $v_t^{\text{weak}}$ 与强增强视图 $v_t^{\text{strong}}$。  
   - 弱增强视图经编码器得到特征后由图像头 $h_s$ 预测软伪标签 $\tilde{y}_t$；强增强视图经编码器得到特征后由新引入的帧头 $h_f$（结构与图像头相同但参数独立）预测帧级概率 $p_t$。  
   - 伪标签损失：$\mathcal{L}_{\text{pseudo}} = \frac{1}{T} \sum_{t=1}^T \mathcal{H}(\text{stopgrad}(\tilde{y}_t), p_t)$，其中 stop-gradient 防止训练坍塌。推理阶段丢弃帧头，仅影响编码器表示学习。

3. **视频导向的特征对齐（Video-Oriented Feature Alignment）**  
   - 随机采样编码器层 $\ell \in \{0,1,2\}$，分别从图像数据集 $\mathcal{D}_s$ 和视频帧中提取中间特征，构建图像特征集合 $\mathcal{Z}_s$ 与视频帧特征集合 $\mathcal{Z}_f$（伪造帧通过伪标签置信度 $\tilde{y}_{t,1} > \tau$ 筛选，$\tau=0.9$）。  
   - 以图像特征 $z_a^\ell \in \mathcal{Z}_s$ 为锚点，目标特征 $z_q^\ell$ 从 $\mathcal{Z}_s \cup \mathcal{Z}_f$ 中采样，进行特征插值：$z_m^\ell = \lambda z_a^\ell + (1-\lambda) z_q^\ell$，标签 $y_m = \lambda y_a + (1-\lambda) y_q$。  
   - 当目标为视频帧特征时，强制 $\lambda > 0.5$（从 $\text{Beta}(1,1)$ 截断采样），使插值特征更靠近图像分布，保持对图像头的兼容性。每轮构造 $N_{\text{align}}=64$ 个插值样本，计算对齐损失 $\mathcal{L}_{\text{align}} = \gamma \mathcal{H}(y_m, h_s(\phi_{\ell+1:}(z_m^\ell)))$，$\gamma=0.5$。

4. **总体优化目标**  
   - 端到端联合优化：$\mathcal{L} = \mathcal{L}_{\text{video}} + \mathcal{L}_{\text{image}} + \mathcal{L}_{\text{pseudo}} + \mathcal{L}_{\text{align}}$。推理时仅使用统一编码器与视频头，无额外计算开销。

## 实验与结果
- **数据集**：ForgeryNet（>220k 视频、>290 万图像，多数为部分伪造）、DFDC preview（1131 真实 + 4113 伪造视频，多数部分伪造）。评估指标为视频级 Accuracy 与 AUC。
- **主要结果（ForgeryNet）**：UVIF-Swin-S 达到 **88.94% Acc / 95.69% AUC**，优于所有对比方法；UVIF-Swin-T 达到 88.02% Acc / 95.10% AUC，相比同骨干 VideoSwin-T（83.89% / 90.53%）提升约 **+4.13% Acc / +4.57% AUC**，且参数量与 FLOPs 不变。
- **DFDC 结果**：UVIF-Swin-S 达到 **90.73% Acc / 96.05% AUC**，UVIF-Swin-T 达 88.67% Acc / 95.26% AUC，均超过原 SOTA 方法。
- **数据效率**：仅使用 10k 图像即可显著提升性能（86.36% vs 基线 83.89%），100k 图像后趋于饱和，即使使用全部 2.3M 图像也无进一步增益。
- **消融结论**：时序融合（+8.85% Acc）、图像监督、伪标签、特征对齐逐层递进贡献；特征对齐中图像锚点约束与图像偏向插值（$\lambda>0.5$）缺一不可；不同骨干（ResNet、ConvNeXt、Swin）均获得稳定提升。

## 相关工作脉络
1. **图像类人脸伪造检测**（如 Mesonet、FaceForensics++、Effort）：以单帧图像为输入进行分类，但在部分伪造视频下因全帧赋相同标签引入噪声；本文通过统一建模与特征对齐缓解该问题。
2. **视频类人脸伪造检测**（如 TSM、SlowFast、VideoSwin、TimeSformer、UniFormer）：直接处理视频序列建模时序线索，精度更高但缺乏帧级监督，部分伪造帧易被均匀梯度拉偏；本文在相同骨干上引入图像细粒度监督与伪标签机制。
3. **多示例学习（MIL）方法**（如 S-MIL、TransMIL、DSMIL、ACMIL）：将视频视为 bag、帧视为实例，仅依赖 bag 级标签；其受限于无法获取实例级标签，部分伪造检测能力提升有限；本文借助外部图像标注绕过实例标签缺失瓶颈。
4. **统一多模态架构**（如 Omnivore、ImageBind、CLIP）：通过单模型处理多种模态并映射到统一特征空间；本文借鉴该思路，但在人脸伪造检测任务中面向“视频-图像跨域表示对齐”设计特定的伪标签与特征插值策略。
5. **Manifold Mixup / 特征空间正则化**：在隐藏层对同类样本进行插值以正则化表示；本文将其扩展为跨分布（图像↔视频帧）插值，并通过锚点约束与方向偏向解决分布偏移下的兼容性难题。

## 局限性与未来方向
1. **依赖图像标注数据**：方法 effectiveness 建立在可用的带标注图像集合之上，若图像与视频所用伪造类型差异较大，虽可通过 feature alignment 缓解，但仍需一定数量的匹配/互补图像。
2. **伪标签置信度阈值敏感**：视频帧伪标签筛选依赖于置信度阈值 $\tau$（文中设为 0.9），在伪造帧比例极低或图像-视频分布差异较大时可能影响特征集合 $\mathcal{Z}_f$ 的质量。
3. **未考虑帧间时序一致性显式建模**：当前对齐与伪标签主要作用于单帧表征，尚未显式建模时序上下文在部分伪造检测中的作用。
4. **未来方向**：作者提出从弱监督学习视角进一步探索，并开发面向伪造分析的多模态基础模型。

## 研究启发与可借鉴点
1. **用易获细粒度标注弥补难获标注**：将图像级精确标注迁移到视频帧级监督，适用于视频任务中帧级标注稀缺但静态图像标注丰富的场景，具备较高的复用价值。
2. **跨模态特征插值 + 锚点约束**：通过以源域特征为锚点、目标域特征为插值对象并偏向源域的方式，可在不破坏源域分类能力的同时促进目标域适配，适用于各类图像-视频跨域迁移任务。
3. **弱/强增强一致性训练结构**：伪标签生成采用 weak/strong augmentation pair，可与 FixMatch 等半监督范式结合，用于视频异常检测、动作识别等需要帧级学习的任务。
4. **多任务统一编码器 + 独立分类头设计**：共享编码器保留通用表征能力，不同任务头分别优化，便于在不增加推理开销的前提下扩展新任务。

## 关键术语表
**UVIF**：Unified Video and Image Representation for face Forgery detection，本文提出的统一视频-图像表示人脸伪造检测框架。  
**部分伪造视频**：视频中仅包含少量篡改帧而非全部帧均被伪造的视频，是本文重点解决的检测难点。  
**Multiple Instance Learning (MIL)**：弱监督学习范式，将视频视为 bag、帧视为 instance，仅使用 bag 级标签进行训练。  
**Pseudo Labeling**：利用已训练模型对无标签样本生成软标签作为监督信号，本文用于为视频帧生成细粒度伪标签。  
**Video-Oriented Feature Alignment**：以图像特征为锚点、对图像与视频帧中间层特征进行偏向图像分布的插值，以缩小跨域分布差异。  
**ForgeryNet**：包含 >220k 视频与 >2.9M 图像的大规模人脸伪造检测基准数据集，多数视频为部分伪造。  
**DFDC**：Deepfake Detection Challenge 的 preview 数据集，包含大量部分伪造视频，广泛用于伪造检测评测。  
**Temporal Shift Module (TSM)**：通过时序通道移位实现轻量级视频时序建模的模块，本文用于 CNN 骨干的时序融合。

## 可复现要素
- **数据集**：ForgeryNet（公开）、DFDC preview（公开）。
- **代码与权重**：代码已开源，见 https://github.com/haotianll/UVIF；论文未明确说明预训练权重是否公开。
- **关键超参**：采样帧数 32、时间步长 4、图像 batch 256/迭代、学习率 0.01、动量 0.9、权重衰减 0.0001、训练迭代 ForgeryNet 50K / DFDC 20K；对齐超参 $\mathcal{I}=\{0,1,2\}$、$\tau=0.9$、$N_{\text{align}}=64$、$\gamma=0.5$；$\lambda$ 采样方式：图像目标 $\text{Beta}(1,1)$，视频帧目标 $\text{Beta}(1,1)$ 截断为 $\lambda>0.5$。
