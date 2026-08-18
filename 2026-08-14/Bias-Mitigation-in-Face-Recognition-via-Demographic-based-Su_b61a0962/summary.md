---
title: "Bias-Mitigation-in-Face-Recognition-via-Demographic-based-Su"
source: https://arxiv.org/pdf/2608.12971v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 13:43:04"
field: "人脸识别公平性与去偏"
keywords: ["人脸识别", "公平性", "监督对比学习", "Demographic bias", "ArcFace", "ISO 19795-10", "FPD/FND"]
innovations: ["提出 Demographic-based Supervised Contrastive (DeSCon) 损失，联合 ArcFace 与 SupCon 优化嵌入尾部分布", "设计 All/WG/Hard 三种人口感知非匹配对采样策略，Hard 变体在多种条件下最稳定提升公平性"]
benchmarks: ["RFW", "BFW", "VGGFace2", "IJB-C", "LFW", "CPLFW", "CALFW", "AgeDB", "CFP-FP"]
---

# 论文速读：Bias-Mitigation-in-Face-Recognition-via-Demographic-based-Su

## 一句话总结
本文提出 **DeSCon**（Demographic-based Supervised Contrastive）损失函数，将监督对比学习与基于质心的 margin 分类（ArcFace）相结合，并通过三种人口统计学感知的非匹配对采样策略（All/WG/Hard）定向优化 score 分布尾部，从而在不牺牲整体识别性能的前提下显著降低人脸识别系统在不同族裔/性别群体间的公平性差异。

## 研究问题与动机
- **部署场景的特殊性**：人脸验证系统需在极低的 False Match Rate（FMR）工作点运行，对应于非匹配分数分布的**尾部区域**；单纯平衡训练数据的类分布只能改善分布均值，无法充分修正尾部行为的不公平性。
- **现有去偏方法的局限**：后处理分数归一化需要部署时已知人口属性且无法提升整体性能；去特征化方法（如 SensitiveNets）往往以牺牲识别精度为代价；仅依赖数据平衡仍不够充分（文献 [3, 29] 及本文实验均验证）。
- **评估指标的非标准性**：多数工作仍按各 demographic group 单独优化阈值报告 accuracy，不符合真实部署中单一全局阈值的约束，也无法反映国际标准的公平性要求。
- **对比学习在公平性方向的空白**：Supervised Contrastive Learning 已用于开放集识别、属性分类等，但基于 demographic 标签的 pair selection 策略用于公平性人脸识别尚属首次探索。

## 核心贡献（创新点）
- **提出 DeSCon 损失**：将 SupCon 与 ArcFace 联合优化，通过 SoftMax 型聚合在近距离非匹配对上的更强梯度，直接塑形非匹配分布尾部，区别于仅关注类间均值分离的传统 margin loss。
- **设计三种 demographic-aware 采样策略**：DeSCon-All（全局非匹配对）、DeSCon-WG（同族裔组内非匹配对）、DeSCon-Hard（同组内最困难 K 个非匹配对），系统探究局部平衡与全局分离的权衡。
- **揭示 Hard 采样的鲁棒优势**：DeSCon-Hard 在平衡与非平衡训练、多测试协议下均稳定降低 FPD 与 FND，优于其他变体及 SOTA in-processing 方法（RamFace、MixFairFace）。
- **建立标准化公平性评估基准**：严格按 ISO/IEC 19795-10 使用 Gini 系数（FPD/FND）评估，并重实现多个基线方法，指出 ACC/STD 指标易产生乐观假象，呼吁社区采用国际标准。

## 方法详解
- **联合损失函数**：
  $$\mathcal{L}_{\mathrm{total}} = \frac{1}{B}\sum_{b=1}^{B} w_d\big[\mathcal{L}_{\mathrm{arc}}(x_b) + \lambda\cdot\mathcal{L}_{\mathrm{sup}}(x_b)\big]$$
  其中 $\mathcal{L}_{\mathrm{arc}}$ 为 ArcFace 加性角边界损失，$\mathcal{L}_{\mathrm{sup}}$ 为 Supervised Contrastive loss；$\lambda=1$，尺度参数 $s$ 与 ArcFace 保持一致。
- **人口加权补偿**：当训练集 demographic 不平衡时，采用 $w_d = \frac{N}{D N_d}$ 对样本加权，$N$ 为总样本数，$D$ 为族裔组数，$N_d$ 为第 $d$ 组样本数。
- **三种非匹配对采样策略**：
  - **DeSCon-All**：分母包含批次内所有不同身份的嵌入（跨组+组内）。
  - **DeSCon-WG**：仅包含与 anchor 同 demographic group 的非匹配对，强调组内细粒度分离。
  - **DeSCon-Hard**：从同组非匹配对中选取与 anchor 相似度最高的前 $K=10$ 个作为 hard negatives，最激进地优化尾部。
- **Batch 构造**：每个 batch 显式保证各 demographic group 均存在 mated pairs，以支撑稳定的对比优化。

## 实验与结果
- **数据集**：训练使用 BUPT-BalancedFace（1.3M 图像，28k 身份，4 族裔平衡）与 BUPT-GlobalFace（不平衡）；测试使用 RFW（平衡族裔）、BFW（平衡族裔+性别）、VGGFace2（不平衡）、IJB-C 及 LFW/CPLFW/CALFW/AgeDB/CFP-FP 标准基准。
- **主干网络**：IResNet50 / IResNet100，从头训练，$s=64$、$m=0.5$、$M=512$、$B=256$、30 epochs、LR=0.1（12/20 epoch 衰减 10 倍）。
- **RFW 主结果（IResNet50）**：DeSCon-Hard 取得 TMR=74.69%，FPD=0.5766，FND=0.1191，相比 ArcFace（FPD=0.6372，FND=0.1348）分别降低 9.5%/11.9%；同时 MixFairFace 在 FND 上最优（0.0855）但 TMR 损失较大。
- **BFW/VGGFace2 验证**：DeSCon-Hard 在 IResNet50 上为唯一同时降低 FPD 与 FND 的变体；IResNet100 上三者均改善，但 Hard 仍提供最均衡的 trade-off。
- **不平衡训练鲁棒性**：在 BUPT-GlobalFace 上，DeSCon-Hard 在 RFW 取得最高 TMR（78.87%）与最低 FPD（0.4553）、FND（0.0625），显著优于 RamFace/MixFairFace。
- **可迁移性**：替换为 AdaFace 基线后，所有 DeSCon 变体仍改善公平性，证明方法与 ArcFace 无强绑定。

## 相关工作脉络
- **ArcFace / MagFace / AdaFace**：基于质心的角边界分类损失，本文以其为基底加入对比正则项，而非替代；与这些方法的区别在于显式优化非匹配分布尾部而非仅类间分离。
- **RamFace [71]**：按种族自适应调整 margin，属于在分类头层面做组别差异化；DeSCon 则在嵌入空间层面通过对比学习施加组内 hard negative 约束，机制不同。
- **MixFairFace [64]**：通过 adapter 进行特征混合去偏；本文方法无需额外模块，纯损失设计，且训练开销与 ArcFace 基本持平（<1%）。
- **Labelless [45] / SensitiveNets [43]**：无需 demographic 标签或主动剥离敏感信息；DeSCon 明确利用训练时的人口标签指导 pair selection，推理时无需标签。
- **FairScoreNormalization / ScoreNorm**：后处理归一化需部署时已知人口属性；DeSCon 为 in-processing 方法，端到端学习公平嵌入，不依赖运行时元信息。
- **SupCon [27]**：原始监督对比学习假设 batch 内同身份为 positive；本文扩展其 negative 采样逻辑，引入 demographic 约束以实现公平性导向的表示学习。

## 局限性与未来方向
- **训练规模受限**：受限于 BUPT 数据集的 demographic 标注，未在 WebFace-260M 等超大规模数据集上验证，整体 TMR 低于工业级模型。
- **依赖准确的人口标签**：当前方法需可靠族裔/性别标注；若标签噪声高或不可用（如 Labelless 所指出的场景），策略效能可能下降。
- **未探索跨属性交互**：仅评估 ethnicity 与 gender，年龄、配饰等其他 demographic 维度的公平性尚未涉及。
- **未来方向**：结合自动或估计的人口属性标注，将 DeSCon 扩展至百万级身份的大规模训练；探索与其他 debiasing 技术（如 cross-domain、adversarial）的融合。

## 研究启发与可借鉴点
- **尾部优化思路**：将公平性关注点从"分布均值对齐"转向"尾部行为控制"，这一视角可迁移至其他 biometric 或分类任务中需要低误判率的场景。
- **Hard demographic pair mining**：同组内 hardest negative 的选择策略简单高效，可作为通用组件嵌入任意对比学习框架，增强组内 discriminability。
- **标准化评估实践**：全面采用 ISO/IEC 19795-10 的 Gini 指标并重现实验，为社区提供了可复现的公平性评测模板，值得在后续工作中沿用。
- **与强基线兼容**：证明 DeSCon 可无缝替换 ArcFace 接入 AdaFace 等更强 margin 损失，提示公平性正则化不应与特定分类器耦合，有利于模块化设计。

## 关键术语表
- **DeSCon**：Demographic-based Supervised Contrastive loss，本文提出的结合人口感知对比学习的公平性损失。
- **FPD / FND**：False Positive/Negative Differential，基于 Gini 系数计算的组间 FMR/FNMR 不平等度量，值越小越公平。
- **TMR**：True Match Rate，在固定全局阈值下的真正匹配率，反映整体验证性能。
- **ArcFace**：Additive Angular Margin Loss，在单位超球面上引入加性角边界的分类损失，构成本文基线。
- **SupCon**：Supervised Contrastive Learning，利用身份标签拉近同类、推开异类的对比损失。
- **BUPT-BalancedFace / GlobalFace**：包含族裔标注的人脸数据集，前者平衡、后者不平衡，用于训练。
- **RFW / BFW**：Racial Faces in the Wild 与 Balanced Faces in the Wild，提供族裔/性别标注的公平性评测基准。
- **IJB-C**：IARPA Janus Benchmark C，大规模无约束人脸验证基准，用于检验模型泛化能力。

## 可复现要素
- **数据集**：BUPT-BalancedFace、BUPT-GlobalFace、RFW、BFW、VGGFace2、IJB-C、LFW、CPLFW、CALFW、AgeDB、CFP-FP（多为公开或可向作者申请）
- **代码**：论文声明"Source code is available upon request"（需向作者索取）
- **关键超参**：$s=64$、$m=0.5$、$M=512$、batch size=256、$\lambda=1$、$K=10$、30 epochs、初始 LR=0.1（12/20 epoch ×0.1）
- **评估协议**：ISO/IEC 19795-10，固定阈值报告 TMR/FMR/FNMR；ACC/STD 仅作参考
