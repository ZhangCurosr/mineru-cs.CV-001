---
title: "PolypVision: A Three-Stage Hierarchical Deep Learning Framework for Classification and Segmentation of Colorectal Polyps"
source: https://arxiv.org/pdf/2608.10649v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:42:51"
field: "医学图像分析"
keywords: ["colorectal polyp", "deep learning", "segmentation", "classification", "EfficientNetV2", "UNet++", "transfer learning", "Focal Loss"]
innovations: ["三阶段层级迁移学习框架耦合分类与分割任务", "设备无关设计无需硬件校准", "渐进式知识传递降低各阶段数据需求"]
benchmarks: ["Kvasir-SEG", "CVC-ClinicDB", "PolypGen"]
---

# 论文速读：PolypVision: A Three-Stage Hierarchical Deep Learning Framework for Classification and Segmentation of Colorectal Polyps

## 一句话总结
PolypVision 是一个三阶段分层深度学习框架，通过共享 EfficientNetV2-M 骨干网络和渐进式迁移学习，在结直肠癌筛查中联合完成息肉二元分类、像素级分割与腺瘤亚型分类，在三个公开数据集上达到 AUC ~0.99 的分类精度和 mAP@50 94.4% 的检测性能。

## 研究问题与动机
- 结直肠癌（CRC）是全球第三高发病率、第二高癌症相关死亡率的恶性肿瘤，绝大多数病例起源于癌前息肉。
- 当前内窥镜下息肉分类存在显著观察者间差异，即使是经验丰富的消化科医师也存在不一致性。
- 现有深度学习方法大多将检测、分割和分类视为独立任务，未能利用跨任务的互补信息。
- 临床工作流需要从粗粒度（良恶性）到细粒度（组织学亚型）的渐进式分析，但现有方法缺乏这种层级化设计。

## 核心贡献（创新点）
- **三阶段层级管道**：将二分类、分割和腺瘤亚型分类通过共享 EfficientNetV2-M 表示耦合，相比独立任务方法能复用跨任务知识。
- **渐进式迁移学习策略**：Stage 1 → Stage 2（编码器初始化）→ Stage 3（骨干初始化），显著降低各阶段对标注数据的需求。
- **任务特异性损失函数**：分类阶段使用 Focal Loss（γ=2, α=0.25）处理类别不平衡，分割阶段使用 Dice+BCE 联合损失，Stage 3 额外应用 MixUp 增强。
- **设备无关设计**：无需针对特定内窥镜硬件或窄带成像系统进行校准，适用于多种临床设备。
- **开放可及性**：提供免费 Web 应用 https://polypvision.com（DataBioX 项目），支持实时息肉分析。
- **SOTA 性能**：在 Kvasir-SEG、CVC-ClinicDB、PolypGen 三个公开数据集上达到 AUC ~0.99 和 mAP@50 94.4%。

## 方法详解

**系统架构**：三阶段层级流水线，每阶段的训练骨干网络传递给下一阶段。

**骨干网络**：EfficientNetV2-M（5280 万参数），基于 ImageNet-21k 预训练，结合 Fused-MBConv 模块实现高效特征提取。分类阶段附加 Global Average Pooling + Dropout + 线性头；分割阶段作为 UNet++ 编码器。

**Stage 1 - 二元分类**：
- 输入：[B, 3, 224, 224]
- 任务：腺瘤 vs. 增生性分类，同时预测 Paris 分型（病变形态）和 JNet 分型（pit pattern + 血管模式）
- 输出：全局平均池化特征 [B, feature_dim] → 任务特异性 Dropout → 线性头
- 损失：Focal Loss（γ=2, α=0.25）
- 传递：训练好的 Stage 1 骨干作为 Stage 2 编码器初始化

**Stage 2 - 分割 + 切除术推荐**：
- 架构：UNet++ 解码器（嵌套跳跃连接 + 渐进式特征融合）+ EfficientNetV2-M 编码器（来自 Stage 1）
- 输出：二值分割掩码 [B, 1, H, W] + 辅助分类头（冷钳息肉切除术/EMR/ESD）
- 损失：Dice Loss + BCE Loss（分割）+ Cross-Entropy（切除术推荐）
- 灵活性：编码器可冻结或微调
- 传递：训练后编码器传递给 Stage 3

**Stage 3 - 腺瘤亚型分类**：
- 输入：仅接收 Stage 1 筛选出的腺瘤图像
- 任务：管状 / 管状绒毛状 / 绒毛状三分类
- 架构：EfficientNetV2-M（来自 Stage 2）+ GAP + Dropout + 线性头
- 输出：logits [B, 3]
- 增强：Focal Loss + MixUp 数据增强
- 层间学习率衰减：新初始化头部 1×10⁻³，迁移骨干层 1×10⁻⁴ ~ 5×10⁻⁵

**训练配置**：AdamW 优化器（λ=1×10⁻⁴），余弦学习率调度 + warm restarts，随机翻转/旋转/颜色抖动，基于验证 AUC/Dice 的早停。

## 实验与结果

**数据集**：
| 数据集 | 任务 | 模态 | 规模 | 年份 |
|--------|------|------|------|------|
| PolypGen | 分割/检测 | 图像+视频 | 3,762 张（单帧+序列） | 2021 |
| Kvasir-SEG | 分割/检测 | 图像 | 1,000 张 | 2019 |
| CVC-ClinicDB | 分割 | 图像 | 612 张（29 视频序列） | 2015 |
| ERCPMP | 分类 | 图像+视频 | 419 张+37 视频 | 2024 |

**检测结果（mAP@50）**：
- Kvasir-SEG：PolypVision 94.4%，优于 YOLOv8-s（91.16%）、CRH-YOLO（90.7%）、YOLOv5（81.0%），略低于 YOLO-LAN（96.19%）
- CVC-ClinicDB：PolypVision 未报告（表中为空）

**分类结果（AUC）**：
- Kvasir-SEG：~0.99，超越 ResNet-50/VGG16（0.91–0.98）
- CVC-ClinicDB：~0.99，匹配 InceptionV3/ResNet（~0.99）

**可解释性**：Grad-CAM 显示模型在正确分类时关注病灶特征（黏膜 pit pattern、表面纹理、病变边界），误分类案例中注意力分散至周围组织。

## 相关工作脉络
- **YOLO 系列检测**：YOLOv8-s（91.16%）、YOLO-LAN（96.19%）在 Kvasir-SEG 上表现强劲，PolypVision 以 94.4% 接近 SOTA 但提供多任务整合。
- **UNet++ 分割**：ResUNet++ 在 CVC-ClinicDB 上达到 82.9% mAP，PolypVision 通过层级迁移学习实现更优的端到端性能。
- **传统分类方法**：ResNet-50/VGG16（AUC 0.91–0.98）、InceptionV3（AUC ~0.99），PolypVision 达到同等水平但扩展至多任务场景。
- **多任务学习**：医学图像中联合优化相关任务已显示收益，但胃肠道病理的层级分类仍缺乏探索。
- **迁移学习**：小规模标注数据场景下的有效策略，本文通过三阶段渐进式迁移实现知识复用。

## 局限性与未来方向
- **误差传播**：Stage 3 仅处理 Stage 1 筛选的腺瘤图像，Stage 1 错误会向下游传播。
- **端到端优化缺失**：当前为顺序训练而非联合优化，可能限制全局性能。
- **缺乏不确定性感知**：未探索置信度估计或可靠的不确定性量化。
- **外部验证不足**：需在多个临床中心的前瞻性队列中验证泛化能力。
- **未来方向**：端到端联合训练、不确定性感知推理、多中心外部验证。

## 研究启发与可借鉴点
- **层级迁移学习范式**：三阶段"粗→细"的知识传递策略可用于其他需要渐进式诊断的医疗 AI 任务。
- **任务特异性损失设计**：Focal Loss + Dice/BCE 的组合在类别不平衡和分割任务中具有通用参考价值。
- **设备无关设计**：无需硬件校准的策略提升了临床部署可行性，可作为标准化方案。
- **Grad-CAM 临床可信度验证**：通过可视化确认模型关注医学相关特征，对监管审批有帮助。
- **开源 Web 应用模式**：DataBioX 项目的免费 tier 设计为技术转化提供了可借鉴路径。

## 关键术语表
- **EfficientNetV2-M**：复合缩放的高效卷积神经网络，5280 万参数，作为本文三阶段的共享骨干。
- **UNet++**：嵌套跳跃连接的 U-Net 变体，通过渐进式特征融合提升分割精度。
- **Focal Loss**：解决类别不平衡的损失函数，通过调制因子降低易分样本权重（γ=2, α=0.25）。
- **Paris 分型**：病变形态学分类系统（有蒂、无蒂、平坦、凹陷），指导内镜评估。
- **JNet 分型**：基于 pit pattern 和血管模式的分类系统，评估病变恶性潜能。
- **Grad-CAM**：梯度加权类激活映射，可视化 CNN 决策依据的空间区域。
- **EMR/ESD**：内镜黏膜切除术/内镜黏膜下剥离术，不同类型的息肉切除技术。
- **MixUp 增强**：线性插值训练样本的数据增强技术，提升模型泛化能力。

## 可复现要素
- **数据集**：全部公开可用（Kvasir-SEG, CVC-ClinicDB, PolypGen, ERCPMP）
- **代码/权重**：论文未提及开源代码仓库，但提供 Web 应用 https://polypvision.com
- **关键超参**：
  - Focal Loss: γ=2, α=0.25
  - 优化器: AdamW, λ=1×10⁻⁴
  - 学习率: 头部 1×10⁻³，骨干 1×10⁻⁴~5×10⁻⁵
  - 输入尺寸: 分类 224×224，分割可变分辨率
  - 增强: 随机翻转/旋转/颜色抖动，Stage 3 额外 MixUp
