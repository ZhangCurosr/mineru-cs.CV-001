---
title: "Class-Activation-Mapping-in-Explainable-Computer-Vision-A-Me"
source: https://arxiv.org/pdf/2608.12299v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 11:21:05"
field: "可解释计算机视觉"
keywords: ["Class Activation Mapping", "explainable AI", "visual explanation", "gradient-based attribution", "gradient-free CAM", "Vision Transformer interpretation", "foundation model explainability"]
innovations: ["构建57篇严格方法中心CAM文献库及PRISMA-inspired筛选协议", "提出归因机制×架构依赖性×评估目标的三维分类体系", "揭示CAM从单类CNN定位到比较性/多层级/基础模型感知解释的演进主线"]
benchmarks: ["ImageNet/ILSVRC2012", "CUB-200-2011", "PASCAL VOC 2007", "COCO", "ProMRI/ACDC/CHAOS", "OpenImages"]
---

# 论文速读：Class-Activation-Mapping-in-Explainable-Computer-Vision-A-Me

## 一句话总结
本文是一篇以方法为中心的综述论文，系统梳理了自2016年原始CAM以来57篇CAM类方法的演进脉络，构建了按归因机制、架构依赖性和评估目标三维划分的分类体系，并对比了CNN、Transformer及基础模型时代各类方法的优劣势与开放挑战。

## 研究问题与动机
- **CAM方法碎片化严重**：CAM已从单一的CNN全局平均池化分类器扩展为涵盖梯度类、无梯度类、高分辨率、token级、因果去偏、基础模型时代的广泛方法族，但缺乏系统性的方法中心化梳理。
- **评估协议不统一**：现有论文使用不同的目标层、输入尺寸、阈值、扰动基线和后处理，导致跨方法比较困难，忠实度（faithfulness）、定位（localization）和鲁棒性常被不同协议测量。
- **CAM被滥用为应用可视化**：大量应用论文仅使用Grad-CAM作为展示图，并未推动方法发展，需区分"方法贡献论文"与"纯应用论文"。
- **新兴架构带来的解释挑战**：Vision Transformer和CLIP/DINO/SAM等基础模型的引入使CAM的解释单元从卷积通道变为token/patch/attention，原有方法设计范式需要重新审视。

## 核心贡献（创新点）
- **构建严格的方法中心文献库**：通过PRISMA-inspired的搜索协议从2016年至今筛选出57篇严格的方法贡献论文，排除了仅应用Grad-CAM可视化的论文和纯综述论文。
- **提出多维分类体系**：将CAM方法按归因机制（梯度/无梯度/混合/高分辨率/token级/基础模型时代/架构感知）系统分类，揭示了各类别的优势用例与 recurring gap。
- **揭示演进主线与趋势判断**：明确指出CAM研究正从"解释单个CNN层中一个类别的分数"转向"比较性、多层级、概率性、token感知和基础模型感知"的解释范式。
- **开展跨家族定量对比分析**：在共享协议下对比了梯度类与无梯度/混合CAM的计算成本、忠实度和定位性能，得出三条实践结论（Grad-CAM仍是快速诊断首选；无梯度方法常在忠实度上更优但成本更高；高分辨率和基础模型方法需特殊评估注意）。
- **提出标准化评估卡片建议**：呼吁未来工作报告最小化评估信息卡（模型与层、目标定义、归一化方式、前向/反向传播次数、是否使用CRF/SAM/先验、使用哪些指标），以推动可复现比较。

## 方法详解

**通用CAM形式**：$L^c = h(\sum_k \alpha_k^c A^k)$，其中 $A^k$ 是第k个特征图，$\alpha_k^c$ 是重要性权重，$h$ 通常为非负激活函数（如ReLU）。不同方法的核心差异在于 $\alpha_k^c$ 的估计方式和 $A^k$ 的提取位置。

**梯度类方法**：
- **Grad-CAM**：用目标分数对特征图的平均梯度代替原始CAM的架构权重，实现通用post-hoc解释。
- **Grad-CAM++**：用正偏置高阶偏导数的加权组合改进多对象覆盖。
- **Integrated Grad-CAM**：通过路径积分减少局部梯度饱和导致的敏感度低估。
- **Relevance-CAM**：用逐层相关性传播（LRP）替代梯度，解决中间层梯度不可靠问题。
- **LIFT-CAM**：用DeepLIFT近似SHAP风格系数，为CAM权重提供可加性归因理论基础。
- **Transformer归因**：Chefer等提出基于Deep Taylor Decomposition的相关性传播方法，通过注意力连接和跳跃连接解释ViT，而非简单可视化注意力权重。

**无梯度/扰动类方法**：
- **Score-CAM**：用激活图作为mask进行前向得分加权，避免梯度依赖，但需大量前向传播。
- **Ablation-CAM**：直接测量移除特征图后的分数下降，估计边际贡献。
- **ReciproCAM**：对中间特征图进行空间扰动，比Score-CAM快148倍，ADCC性能强劲。
- **ScoreCAM++**：用tanh门控激活图改进归一化，区分高/低优先级区域。

**高分辨率方法**：
- **LayerCAM**：用各空间位置的正梯度加权，融合浅层细节与深层语义。
- **Poly-CAM**：融合早晚期卷积层特征生成高分辨率CAM，保持竞争性插入/删除忠实度。
- **Finer-CAM**：将解释目标从"为何此类分高"改为"此类相较于参考类的logit差异"，提升细粒度定位。

**基础模型时代方法**：
- **gScoreCAM**：用梯度选择top通道+Score-CAM前向得分加权解释CLIP，将运行时降低约8倍。
- **S2C**：将SAM分割先验通过对比学习 transfer 到分类器。
- **DiffCAM**：通过特征分布差异而非决策边界梯度生成显著图。
- **MetaCAM**：通过top-k共识和自适应阈值集成多种CAM方法。

**因果/去偏方法**：
- **CI-CAM**：用因果干预减少对象-上下文混淆（如鸟与树枝）。
- **C-CAM**：针对医学图像中器官共现和模糊边界建模类别因果和解剖因果链。
- **Debiased-CAM**：训练多输入多任务模型处理模糊、色温、日夜变化等图像扰动偏差。

## 实验与结果

**文献库规模**：57篇方法贡献论文，来源限于IEEE/CVF、ACM、Elsevier、Springer及其会议（CVPR、ICCV、ECCV、WACV、ICASSP、ICIP、ACM MM、CHI等），2016年至今。

**分类分布**：最大类别为无梯度/扰动CAM、CNN解释、高分辨率CAM和注意力-梯度混合方法。

**关键定量结果**（均在源论文协议下报告）：
- **ReciproCAM**：在ILSVRC2012上比Score-CAM快148倍，ADCC表现强劲。
- **SEAM**：PASCAL VOC上伪标签mIoU达55.41%（CRF前）。
- **LayerCAM**：VGG16上PASCAL VOC val/test mIoU为60.8/61.4。
- **CI-CAM**：CUB Top-1定位58.39%，ImageNet可比。
- **gScoreCAM**：在COCO上BoxAcc接近Score-CAM但运行时大幅降低（约8倍加速）。
- **Finer-CAM**：在CUB-200和Cars上显著改进Grad-CAM/LayerCAM/Score-CAM的定位分数。
- **ScoreCAM++**：ImageNet/VGG-19上平均下降更低、置信度提升优于Score-CAM和梯度基线。
- **Relevance-CAM**：ResNet-50上在Average Increase指标上优于Grad-CAM和Grad-CAM++。
- **LIFT-CAM**：ImageNet上插入/删除AUC优于Grad-CAM和Grad-CAM++。

**交叉对比结论**：Grad-CAM类仍是最快最便捷的诊断工具；无梯度和可加性归因方法在忠实度/定位上通常更优但成本更高；高分辨率和基础模型方法需特别注意其可能反映外部先验而非原分类器决策。

## 相关工作脉络
- **原始CAM (Zhou et al., CVPR 2016)**：提出GAP+线性分类器架构使CNN可定位，本文所有后续工作均以此为起点，但本文强调该方法对架构的强依赖是后续工作的主要改进动因。
- **Grad-CAM (Selvaraju et al., ICCV 2017)**：将CAM从特定架构推广为通用post-hoc工具，本文将其定位为"变革性转折点"，后续梯度类方法均在其基础上改进。
- **Score-CAM (Wang et al., CVPRW 2020) 与 Ablation-CAM (Desai & Ramaswamy, WACV 2020)**：分别开创无梯度CAM路线，本文强调这两篇改变了CAM的设计范式——从梯度敏感性转向前向评估和边际贡献。
- **TS-CAM (Gao et al., ICCV 2021) 与 MCTformer (Xu et al., CVPR 2022)**：将CAM扩展到Vision Transformer，本文指出这标志着解释单元从卷积通道到token交互的根本转变。
- **Finer-CAM (Zhang et al., CVPR 2025) 与 DiffCAM (Li et al., CVPR 2025)**：代表近期两股趋势——目标重构（对比解释）和分布差异（非决策边界梯度），本文将其定位为CAM方法论的进一步扩展。
- **gScoreCAM (Chen et al., ACCV 2022/2023) 与 MetaCAM (Dick et al., Scientific Reports 2026)**：分别代表基础模型时代解释的两条路径——高效适配CLIP和集成泛化，本文将其纳入同一框架讨论其共同依赖外部先验的问题。

## 局限性与未来方向
- **评估标准化缺失**：不同论文使用不同协议，跨方法比较困难，需建立最小化评估卡片。
- **因果忠实度挑战**：自然图像中混杂因素普遍，医学图像中解剖共现严重，现有因果方法（CI-CAM、C-CAM）仍局限于特定假设。
- **高效无梯度解释**：Score-CAM等成本过高限制部署，需探索自适应解释策略（可靠时用电梯度，不确定时切换扰动）。
- **基础模型敏感性问题**：Finer-CAM依赖参考类、gScoreCAM依赖prompt、DiffCAM依赖参考分布，需将prompt/参考敏感度纳入标准鲁棒性测试。
- **人类验证不足**：视觉悦目的图可能不忠实，需结合删除/插入、sanity check、因果干预和任务结果的多维度人类中心评估。
- **综述范围局限**：仅包含57篇严格方法论文，排除了大量应用论文和预2016工作，可能遗漏部分重要发展。

## 研究启发与可借鉴点
- **目标重构思路**：Finer-CAM将解释目标从"为何此类高"改为"此类相对参考类为何更高"，这一对比式目标设计可迁移到细粒度分类、异常检测和迁移学习等场景。
- **评估卡片规范化**：提出的最小化评估报告框架（模型层、目标定义、归一化、计算次数、先验使用、指标）可直接作为团队CAM研究的标准报告模板。
- **自适应解释策略**：论文提出的"梯度可靠时用电梯度、不确定时切扰动"的自适应框架，可作为后续研究的可实现方向——设计不确定性估计模块来动态选择解释方法。
- **集成思想的启示**：MetaCAM通过top-k共识集成多种CAM提升鲁棒性，提示团队可在多模型/多尺度CAM集成方面探索更系统的聚合方法。
- **基础模型解释的范式转移**：CLIP/DINO/SAM辅助的CAM表明外部先验既是增强来源也是忠实度风险，团队在开发新解释方法时应显式报告并评估外部先验的贡献度。

## 关键术语表
- **Class Activation Mapping (CAM)**：将模型内部证据转化为热力图，高亮支持目标类别或概念的图像区域、卷积通道、token或patch。
- **Attribution Weight ($\alpha_k^c$)**：用于加权组合特征图/通道/token的系数，是各类CAM方法的核心设计变量。
- **Faithfulness（忠实度）**：衡量高亮证据是否真正影响模型输出的指标，常用删除/插入测试评估。
- **Localization（定位）**：衡量高亮证据是否与边界框、掩码或专家标注对齐，常用mIoU/DSC/Top-1定位评估。
- **WSOL/WSSS**：弱监督对象定位/弱监督语义分割，CAM常作为伪标签种子用于训练密集预测器。
- **Sanity Check**：测试当模型权重或标签随机化时解释是否随之随机变化，用于验证解释方法的合理性。
- **Token/Patch CAM**：针对Vision Transformer的解释方法，通过类token、patch token或注意力矩阵生成解释。
- **Foundation-Model-Era CAM**：利用CLIP、DINO、SAM等基础模型先验改进CAM的方法，解释目标可扩展到open-vocabulary概念。

## 可复现要素
- **数据集**：ImageNet (ILSVRC2012)、CUB-200-2011、PASCAL VOC 2007、COCO、Cars、Aircraft、ProMRI、ACDC、CHAOS、OpenImages等（论文声明部分数据集公开可用）
- **代码/权重**：论文未明确声明开源状态，各方法代码见原文引用；MetaCAM和SSG-CAM发表于Scientific Reports（通常开源）
- **关键超参**：论文未统一给出超参，因涉及57篇独立论文；建议参看各原始论文；共享评估协议包括：输入尺寸（ ImageNet标准224或原始分辨率）、目标层选择、归一化方式（min-max vs tanh门控）、阈值方法（max vs percentile）
