---
title: "Bias-Mitigation-in-Face-Recognition-via-Demographic-based-Su"
source: https://arxiv.org/pdf/2608.12971v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 13:41:38"
field: "人脸识别公平性"
keywords: ["人脸验证公平性", "监督对比学习", "bias mitigation", "Demographic-based Supervised Contrastive Learning", "ISO/IEC 19795-10", "FPD/FND", "ArcFace"]
innovations: ["提出 DeSCon 损失，将 Supervised Contrastive Learning 与 demographic 感知 pair 采样结合以优化非匹配分数分布尾部", "设计 DeSCon-Hard/WG/All 三种采样策略，证明 within-group hard negative mining 在平衡/不平衡训练下均最稳健", "采用 ISO/IEC 19795-10 标准的 FPD/FND/Gini 统一评测公平性，揭示 ACC/STD 指标的局限性"]
benchmarks: ["RFW", "BFW", "VGGFace2", "LFW", "CPLFW", "CALFW", "AgeDB", "CFP-FP", "IJB-C"]
---

# 论文速读：Bias-Mitigation-in-Face-Recognition-via-Demographic-based-Su

## 一句话总结
本文提出 DeSCon（Demographic-based Supervised Contrastive Learning），通过在人脸特征学习中引入基于人口统计信息的监督对比损失，针对非匹配分数分布的尾部进行正则化，在保持验证性能的同时显著缓解人脸认证系统中的 demographic 偏见。

## 研究问题与动机
- 人脸验证系统在实际部署中工作于极低 FMR 操作点，即非匹配分数分布的尾部；现有方法（如类别平衡、后处理归一化）难以针对性优化尾部行为。
- 仅对训练数据做 demographic 平衡并不能完全消除偏见，且可能降低整体验证性能。
- 现有去偏方法（如去除敏感信息）往往牺牲识别精度；基于 margin loss 的方法缺乏对同组内难负样本的显式区分能力。
- 多数公平性评估采用 per-group 阈值，与真实部署中使用单一全局阈值的场景不符，缺乏符合 ISO/IEC 19795-10 标准的统一评估。

## 核心贡献（创新点）
- **提出 DeSCon 损失函数**：将 Supervised Contrastive Learning 与 ArcFace 结合，在 embedding 空间直接对齐同类嵌入、推开异类嵌入，补强了 centroid-based 损失对尾部行为的控制。
- **设计三种人口统计感知采样策略**：DeSCon-All（跨组）、DeSCon-WG（同组）、DeSCon-Hard（同组 hardest pair），系统探究不同 non-mated pair 选择对公平性的影响。
- **提出 demographics-weighted loss 加权**：在训练数据 demographic 不平衡时，通过 $w_d = N/(D \cdot N_d)$ 补偿各组样本分布差异，无需 staged training。
- **提供符合 ISO/IEC 19795-10 的公平性评测基线**：统一使用全局阈值 $\tau$，以 FPD/FND 和 Gini 系数衡量公平性，揭示 ACC/STD 指标在公平性评估中的不可靠性。
- **验证 DeSCon-Hard 的鲁棒性**：在 balanced 与 imbalanced 训练条件下均稳定降低 FPD 与 FND，并与 AdaFace 等更强 margin loss 兼容。

## 方法详解
- **核心架构**：在 ArcFace 基础上联合优化 Supervised Contrastive Loss（SupCon），二者共享归一化 embedding 空间与尺度参数 $s$。
- **SupCon 损失**：$\mathcal{L}_{\mathrm{sup}}(x_a) = -\frac{1}{|P(a)|}\sum_{x_p \in P(a)}\log\frac{e^{s\cdot\cos(x_a,x_p)}}{\sum_{x \in P(a)\cup N(a)}e^{s\cdot\cos(x_a,x)}}$，其中 $P(a)$ 为同类 mated 对，$N(a)$ 为非匹配对。
- **三种 non-mated pair 采样策略**：
  - **DeSCon-All**：batch 内所有非匹配样本（无论 demographic）。
  - **DeSCon-WG**：仅采样与 anchor 同 demographic 组的非匹配样本。
  - **DeSCon-Hard**：选取同组内最相似的 top-K（$K=10$）非匹配样本，聚焦分布尾部。
- **总体损失**：$\mathcal{L}_{\mathrm{total}} = \frac{1}{B}\sum_{b=1}^B w_d[\mathcal{L}_{\mathrm{arc}}(x_b) + \lambda\cdot\mathcal{L}_{\mathrm{sup}}(x_b)]$，$\lambda=1$，$w_d$ 为 demographic 组权重。
- **batch 构造**：显式保证每组 demographic 均有 mated pair 参与对比学习，实现 uniform coverage。
- **与 centroid-based 方法的本质区别**：SupCon 的 SoftMax 对所有非匹配对聚合梯度，对邻近负样本施加更强推力，直接作用于非匹配分数分布尾部。

## 实验与结果
- **数据集**：训练使用 BUPT-BalancedFace（4 族裔平衡，1.3M 图像）与 BUPT-GlobalFace（不平衡）；评估使用 RFW、BFW、VGGFace2 及 LFW/CPLFW/CALFW/AgeDB/CFP-FP/IJB-C。
- **主要结果（RFW，IResNet50）**：
  - ArcFace 基线：TMR 74.09%，FPD 0.6372，FND 0.1348
  - **DeSCon-Hard**：TMR 74.69%（+0.60%），FPD 0.5766（-0.0606），FND 0.1191（-0.0157）
  - DeSCon-All：TMR 76.13%，FPD 0.6974
  - DeSCon-WG：TMR 76.87%，FPD 0.8186
- **最强结果**：DeSCon-Hard 在多种设置下稳定取得最佳公平性-性能权衡；IResNet100 上 TMR 达 75.78%，FPD 0.6372，FND 0.1325。
- **Imbalanced 训练（BUPT-GlobalFace）**：DeSCon-Hard 在 RFW 上 TMR 78.87%，FPD 0.4553，显著优于基线与其他方法。
- **消融**：替换 SupCon 为 Triplet/MS 损失导致 TMR 大幅下降；DeSCon-Hard 优于 DeSCon-MS 与 DeSCon-SH。
- **与 AdaFace 兼容性**：DeSCon-WG 在 AdaFace 上 TMR 80.14%，FPD 0.5766，FND 0.0835。

## 相关工作脉络
- **ArcFace / MagFace / AdaFace**：基于角 margin 的分类损失，本文扩展其仅作用于类中心，缺乏对同组内 hard 非匹配对的显式区分。
- **RamFace / MixFairFace / Labelless**：in-processing 去偏方法，分别通过种族自适应 margin、特征混合、无标签自适应学习去偏；本文强调它们未系统评估 ISO 标准下的公平性。
- **DeFT / FairScoreNormalization**：pre-/post-processing 方案，需 demographic 信息或额外计算，不提升整体 TMR；本文方法在 training 阶段直接学习公平表征。
- **Supervised Contrastive Learning（SupCon）**：用于 facial attribute classification 与 open-set recognition，本文首次将其与 demographic-based pair selection 结合用于公平性人脸验证。
- **ISO/IEC 19795-10 公平性评估**：本文采用 FPD/FND/Gini 替代 per-group ACC，指出后者高估公平性，推动评估标准对齐国际规范。

## 局限性与未来方向
- 训练依赖含 demographic 标注的中等规模数据集（BUPT），未在 WebFace-260M 等大规模无标注数据集上验证。
- 仅评估了 ethnicity，gender/age 等其他 demographic 维度留待未来。
- DeSCon-WG 与 DeSCon-All 的 FPD 增益存在 backbone 与测试集依赖性，稳定性不如 DeSCon-Hard。
- 未来方向包括：估计或自动推断大规模数据的 demographic 标签以扩展至 WebFace 规模，以及探索其他敏感属性（gender、age）的公平性优化。

## 研究启发与可借鉴点
- **SupCon 用于尾部正则化**：SoftMax 对比损失对邻近负样本的强梯度可自然作用于极低 FMR 操作点，这一机制可迁移至其他 biometric verification 任务。
- **Demographic-guided hard negative mining**：Hard within-group 采样策略概念可推广至表格/语音/医疗等存在 group disparity 的学习场景。
- **ISO 标准评测范式**：采用全局阈值 + FPD/FND/Gini 的评估框架可作为后续工作的标准 baseline，避免 per-group 阈值的乐观偏差。
- **Loss 加权补偿 imbalance**：$w_d = N/(D \cdot N_d)$ 的简单 weighting 策略无需重采样即可缓解 demographic skew，可与现有 margin loss 无缝集成。
- **可复用代码结构**：DeSCon 与 ArcFace/AdaFace 等 base loss 兼容，其 modular 设计便于后续研究者替换 backbone 或采样策略。

## 关键术语表
- **DeSCon（Demographic-based Supervised Contrastive Learning）**：本文提出的公平性人脸识别损失，结合 ArcFace 与 SupCon，通过 demographic 感知的非匹配对采样降低组间误差差异。
- **Supervised Contrastive Learning（SupCon）**：利用 identity 标签在同一 batch 内拉近同类样本、推开异类样本的对比学习损失。
- **FPD（False Positive Differential）**：各 demographic 组 FMR 差异的 Gini 系数度量，值越低表示跨组 false match 越公平。
- **FND（False Negative Differential）**：各 demographic 组 FNMR 差异的 Gini 系数度量，反映跨组 false non-match 公平性。
- **IResNet**：基于 ResNet 的 face recognition backbone，包含 IResNet50 与 IResNet100 等变体。
- **RFW（Racial Faces in the Wild）**：包含四个族裔平衡子集的公平性验证 benchmark，专用于评估种族偏见。
- **ArcFace**：在 hypersphere 上引入 additive angular margin 的分类损失，增强 intra-class 紧凑性与 inter-class 可分性。
- **BUPT-BalancedFace / BUPT-GlobalFace**：分别为 4 族裔平衡与不平衡的人脸识别训练数据集。

## 可复现要素
- **数据集**：BUPT-BalancedFace、BUPT-GlobalFace、RFW、BFW、VGGFace2、LFW、CPLFW、CALFW、AgeDB、CFP-FP、IJB-C（均为公开数据集，部分需申请）。
- **代码/权重**：论文声明 "Source code is available upon request"，未公开；模型权重未开源。
- **关键超参**：scale $s=64$，margin $m=0.5$，embedding dim $M=512$，batch size $B=256$，$\lambda=1$，$K=10$，训练 30 epochs，初始 LR=0.1，epoch 12/20 各衰减 0.1 倍。
- **Backbone**：IResNet50、IResNet100，从 scratch 训练。
