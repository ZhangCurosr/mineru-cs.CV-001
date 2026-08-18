---
title: "Class Activation Mapping in Explainable Computer Vision: A Method-Centered Review of CNN, Transformer, and Foundation-Model-Era Visual Explanations"
source: https://arxiv.org/pdf/2608.12299v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 08:07:01"
---

# 论文速读：Class Activation Mapping in Explainable Computer Vision: A Method-Centered Review of CNN, Transformer, and Foundation-Model-Era Visual Explanations

## 一句话总结
本文是一篇以方法演进为主线、覆盖2016年至今57篇严格筛选文献的综述，系统梳理了Class Activation Mapping（CAM）从CNN梯度加权、无梯度扰动、高分辨率融合、Transformer Token归因到基础模型时代提示/先验融合的全链条方法体系，并指出当前评估协议碎片化与因果忠实度缺失是制约CAM从“可视化热图”走向“可信证据”的核心瓶颈。

## 研究问题与动机
1. **方法繁荣但缺乏统一梳理**：CAM已从最初的CNN全局平均池化定位工具，扩展为涵盖梯度基、分数基、消融基、高分辨率、Token级、因果去偏、基础模型先验的庞大方法族，但既有综述多聚焦单一架构或应用层可视化，缺乏以“归因机制×架构依赖×评估目标”为主轴的严格方法学期梳理。
2. **梯度信号的固有缺陷**：一阶梯度易受饱和、噪声与浅层碎裂影响，Integrated Grad-CAM、Relevance-CAM、ShapleyCAM等尝试改进，但未形成对“局部灵敏度≠因果贡献”这一本质问题的系统性回应。
3. **评估指标不可比**：不同论文使用不同的目标层、输入尺寸、阈值策略、扰动基线与后处理（CRF/SAM等），导致Faithfulness、Localization、RUNTIME等指标跨方法对比存在严重protocol heterogeneity问题。
4. **架构演进带来归因单元迁移**：ViT将归因单元从卷积通道移至Patch/Class Token与Attention头；CLIP/DINO/SAM等基础模型进一步引入文本提示、自监督语义邻接图与分割先验，传统CAM的“单类得分解释”范式面临重构压力。

## 核心贡献（创新点）
1. **构建严格的方法学期语料库与检索协议**：通过IEEE/ACM/Elsevier/Springer及顶级会议筛选，最终纳入57篇以“生成、改进、评估或理论分析CAM样式热图”为核心贡献的论文，排除纯应用可视化与预2016文献。
2. **提出八维CAM方法分类学**：按归因机制与架构依赖划分为CNN梯度/重levance、Transformer注意力/相关性传播、无梯度/扰动类、注意力-梯度混合、高分辨率、Token/Patch级、基础模型时代、架构感知/训练耦合共8类，并明确每类的最佳适用场景与 recurring gap。
3. **跨家族定量对比与趋势归纳**：在共享protocol下比较Grad-CAM类与无梯度/混合类的运行时、Insertion/Deletion忠实度与Localization能量分布，揭示“新方法不必然双重优越”“高分辨率与忠实度可兼得”等反直觉结论。
4. **倡导标准化评估卡片与因果/去偏融合路径**：明确提出最小评估报告规范（目标层、归一化、前向/反向次数、是否使用外部先验等），并将CI-CAM/C-CAM/Debiased-CAM等因果干预方法确立为下一代CAM的核心方向。

## 方法详解
CAM系列方法的核心统一形式为 $L^{c} = h\left(\sum_{k} \alpha_{k}^{c} A^{k}\right)$，其中 $A^{k}$ 为选定层的特征图/Token激活，$\alpha_{k}^{c}$ 为类别 $c$ 的归因权重，$h(\cdot)$ 通常为非负激活（如ReLU）。本文按 $\alpha_{k}^{c}$ 的估计方式与目标定义划分方法族：

1. **梯度基CAM**：Grad-CAM用目标分对特征图的平均梯度代替原CAM的类别权重；Grad-CAM++引入正偏高阶偏导改善多目标覆盖；Integrated Grad-CAM沿输入-基准路径积分降低饱和敏感；Relevance-CAM以层间相关性传播替代梯度，缓解中间层梯度碎裂；LIFT-CAM以DeepLIFT近似SHAP系数赋予线性组合更严谨的可加归因语义；FAM将解释目标从类别logit迁移至特征向量本身，适配Re-ID与自监督表征分析。
2. **无梯度/扰动基CAM**：Score-CAM以激活图为掩码做多次前向推理，用类别置信度变化作权重；Ablation-CAM直接度量特征图移除后的分数边际下降；Eigen-CAM放弃类别特异性，取特征主成分生成类无关激活图；ReciproCAM通过对中间特征图做空间扰动实现远快于Score-CAM的轻量化归因；ScoreCAM++引入tanh门控强化高/低优先级激活区分度。
3. **混合/因果/去偏CAM**：SEAM加入自监督等变约束与像素相关模块，提升WSSS伪标签一致性；AdvCAM通过对抗性扰动强制模型关注原本忽略的区域；C2AM在无标签数据上训练类无关对比激活图以分离前景/背景；CI-CAM通过因果干预切断目标-背景共现混淆（如鸟/树枝、鸭/水）；C-CAM将因果链建模扩展至医学图像解剖共现场景；Debiased-CAM引入多任务去偏头以抑制模糊、色温、昼夜等分布偏移。
4. **高分辨率CAM**：LayerCAM仅取逐空间位置的正梯度，使浅层细节与深层语义可安全融合；F-CAM引入可训练引导上采样解码器输出全分辨率热图；Poly-CAM通过多层早期/晚期卷积特征融合保持Insertion/Deletion忠实度；RPIM基于超像素构建区域内整合与区域间扩散机制提升像素级一致性；GFR-CAM利用Gram-Schmidt正交化生成层次化解释分量，缓解单一主成分的“解释隧道效应”；Finer-CAM将解释目标由“支持某类”改为“目标类vs参考类”的logit差，显著提升细粒度定位。
5. **Token/Patch级与Transformer CAM**：Transformer attribution以Deep Taylor Decomposition将相关性经注意力残差路径传播，避免单纯Rollout attention的线性假设；TS-CAM耦合Token语义与自注意力图突破CNN局部感受野限制；MCTformer引入多Class Token生成类专属定位图；CTI在同图/跨图视角注入Class Token提升全局-局部一致性；Prompt-CAM学习类专属Prompt使细粒度特征通过Prompt-Query注意力显式暴露。
6. **基础模型时代CAM**：gScoreCAM仅用梯度筛选Top通道、用Score-CAM式前向分数加权，将CLIP定位速度提升约8倍；S2C在训练阶段通过SAM-segment contrasting将分割先验转移至分类器；DINO semantic guider将CAM种子在DINO自监督邻接图上扩散构建类感知亲和区域图；DiffCAM通过特征分布差异生成 saliency，绕过决策边界梯度依赖；MetaCAM对多种CAM输出做Top-k共识与自适应阈值融合，验证集成一致性优于单图。

## 实验与结果
本文作为综述不独立开展实验，但系统汇总了各子方法在共享Protocol下的对比结果，关键数值如下：
- **运行时**：gScoreCAM在CLIP设置下将Score-CAM的运行时间缩短约8倍；ReciproCAM相比Score-CAM快148倍且ADCC表现强劲。
- **定位指标**：LayerCAM在PASCAL VOC val/test上取得60.8/61.4 mIoU（VGG16）；CI-CAM在CUB Top-1定位达到58.39%；SEAM在PASCAL VOC上将CRF前伪标签mIoU提升至55.41%。
- **忠实度趋势**：ILSVRC2012/VGG16协议下，LayerCAM与Poly-CAM在保持Insertion性能的同时显著改善Deletion，说明空间细节提升与忠实度并非零和博弈；LIFT-CAM在ImageNet能量定位中贡献更紧凑的类内证据。
- **细粒度改进**：Finer-CAM作为通用包装器，在CUB-200与Cars数据集上对Grad-CAM/LayerCAM/Score-CAM均有稳定定位提升，验证“logit差”目标的泛用性。
- **评估碎片化结论**：不同论文使用的层、阈值、后处理（CRF/SAM/正则化）与扰动基线差异巨大，跨方法绝对数值不可直接排名；作者建议仅在同一Protocol内比较。

## 相关工作脉络
1. **Grad-CAM [2]**：奠定后验梯度加权CAM范式，本文将其视为所有后续梯度基方法的起点，本文系统区分了其“局部灵敏度”与“因果贡献”的本质差异。
2. **Score-CAM [8] / Ablation-CAM [9]**：开启无梯度归因路线，本文将其定位为“避免梯度饱和但牺牲计算效率”的对立统一组，并串联出ScoreCAM++、ReciproCAM、Cluster-CAM等轻量化演进。
3. **LayerCAM [16] / F-CAM [18] / Poly-CAM [37]**：聚焦高分辨率与浅层-深层融合，本文指出其共同gap为“分辨率提升可能放大语义模糊带来的噪声”，并以Finer-CAM说明目标重构比单纯上采样更根本。
4. **Transformer attribution [32] / TS-CAM [31]**：将归因单元从卷积通道迁移至Token与残差注意力流，本文强调“Attention不等于解释”，必须通过相关性传播或Token-语义耦合才能产出类专属定位。
5. **CI-CAM [45] / C-CAM [26] / Debiased-CAM [44]**：引入因果干预与分布去偏，本文将其确立为CAM从“可视化”迈向“忠实证据”的关键转折，并与单纯追求热图锐度的方法形成对比。
6. **gScoreCAM [47] / S2C [42] / DINO guider [43] / DiffCAM [50]**：代表基础模型时代CAM的外部先验融合路线，本文明确指出这类方法的热图“部分所有权”归属预训练模型而非待解释分类器，要求后续工作披露提示词/参考集来源。

## 局限性与未来方向
1. **评估标准化缺失**：现有论文目标层、输入尺度、阈值策略、扰动基线与后处理不统一，跨方法对比困难；作者建议强制推行“最小评估卡片”。
2. **因果忠实度不足**：自然图像与医学图像中普遍存在目标-背景/解剖共现混杂，现有CAM多为相关性热点而非必要性证据，需结合干预数据集、反事实编辑与因果指标联合验证。
3. **无梯度方法效率瓶颈**：Score-CAM/Ablation-CAM等前向/消融成本过高，自适应切换（梯度可靠时用梯度，不确定时切扰动）是可行方向。
4. **提示词/参考集敏感性**：Finer-CAM依赖参考类、gScoreCAM依赖文本提示、DiffCAM依赖参考分布，未来需将Prompt/Reference鲁棒性纳入标准评测协议。
5. **人类验证与客观指标的平衡**：人类偏好不能替代Deletion/Insertion与Sanity Check，需建立多目标联合评估框架以避免“好看但不忠实”的热图误导用户。
6. **综述本身的边界**：本文严格排除纯应用与架构训练耦合型WSOL方法（如Hide-and-Seek、ACoL、SPG等），故定量对比仅限于后验可解释方法族，未能覆盖端到端定位-解释联合优化路线。

## 研究启发与可借鉴点
1. **目标重构范式**：Finer-CAM将解释目标从“支持某类”改为“类间logit差”，该思路可直接迁移至细粒度分类、开放词汇检测与多标签定位任务，作为通用后验包装器使用。
2. **评估卡片标准化实践**：论文提出的“最小报告规范”（目标层、归一化、前/反向次数、外部先验来源、metric列表）可直接嵌入团队内部benchmark pipeline，提升方法对比的可复现性与可发表性。
3. **因果去偏与CAM融合**：CI-CAM/C-CAM的do-calculus干预思路表明，CAM可与结构因果模型（SCM）结合；团队可探索在医学影像或工业缺陷检测中引入干预变量，剥离设备伪影/背景共现干扰。
4. **基础模型先验的显式解耦**：gScoreCAM/S2C/DINO guider证明外部先验可显著提升弱监督定位，但必须显式记录提示词与参考集；团队在调用CLIP/SAM/DINO辅助归因时，可设计“先验消融实验”量化外部模型对热图的贡献占比。
5. **多方法集成与共识机制**：MetaCAM的Top-k共识+自适应阈值策略为CAM集成提供了轻量方案；团队可将其扩展为“梯度/无梯度/高分辨率三系并行+一致性过滤”的生产级解释管线。

## 关键术语表
**CAM-style map**：针对目标类别/概念/提示/检测头输出的空间、Token或Patch级解释热图。
**Attribution weight ($\alpha_k^c$)**：用于加权融合特征图、通道、Token或区域的归因系数。
**Gradient-based CAM**：通过对目标分数关于特征或注意力的导数计算归因权重的一类方法。
**Gradient-free CAM**：不依赖反向梯度，改用前向分数变化、消融、掩码、PCA、聚类或扰动估计权重的一类方法。
**Faithfulness**：衡量高亮证据是否真正影响模型输出的指标，常用Deletion/Insertion AUC、Average Drop/Increase等。
**Localization**：衡量高亮证据与边界框、掩码或专家标注的对齐程度，常用mIoU、DSC、Top-1/Top-5定位率。
**WSOL / WSSS**：弱监督物体定位与弱监督语义分割，CAM常作为伪标签种子驱动密集预测头训练。
**Sanity check**：随机化模型权重或标签后验证热图是否发生显著变化的测试，用于排除“无意义但视觉上合理”的伪解释。

## 可复现要素
- **数据集**：ILSVRC/ImageNet、PASCAL VOC、CUB-200-2011、COCO、ProMRI、ACDC、CHAOS、RSNA/SIIM、Cat/Dog等（各子方法独立使用，综述未统一评测）。
- **代码/权重开源状态**：论文未提供统一代码库；所综述57篇方法的代码与权重开源情况以各原文为准，部分为CVPR/ICCV/ECCV/TPAMI等期刊配套开源，部分需联系作者获取。
- **关键超参**：选取的特征层/Token层、正则化/阈值策略（如Rethinking CAM的百分位阈值、F-C
