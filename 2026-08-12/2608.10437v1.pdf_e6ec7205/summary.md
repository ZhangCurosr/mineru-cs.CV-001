---
title: "MammoMix: Leveraging Mixture of Experts for Robust Mammogram Breast Detection"
source: https://arxiv.org/pdf/2608.10437v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:53:34"
---

# 论文速读：MammoMix: Leveraging Mixture of Experts for Robust Mammogram Breast Detection

## 一句话总结
本文提出 MammoMix，一种基于 Mixture-of-Experts（MoE）范式的乳腺钼靶病灶检测框架，通过将专家模型按数据源进行领域专业化训练，并结合 MoCAE 校准模块与精细化融合策略，有效缓解多源异构乳腺影像跨域部署时的性能衰退与置信度失真问题。

## 研究问题与动机
- **跨域泛化瓶颈**：乳腺X光影像受设备型号、成像协议、人群特征及病灶形态影响极大，单一检测器在多中心数据合并训练或跨数据集测试时，AUC 与敏感度常显著下降。
- **单模型容量冲突**：统一模型难以同时拟合异构分布，容易过拟合样本占优的数据集（如 DDSM），稀释对小样本子集（如 DMID）的敏感度。
- **传统集成缺陷**：标准 ensemble 缺乏可学习的路由机制，且直接融合原始置信度会导致“高置信但低质量”的专家主导输出，破坏融合公平性。
- **临床可靠性需求**：真实筛查场景中图像来源混杂且未知，检测系统不仅需高精度，更需输出分数与实际预测可靠性对齐，以支撑临床决策信任。

## 核心贡献（创新点）
- **首个面向乳腺钼靶的 MoE 检测框架**：训练 3 个独立 YOLOS 专家分别专攻 CSAW、DDSM、DMID，实现领域特征隔离学习，避免分布冲突。
- **CNN 门控路由机制**：设计轻量 CNN 分类器输出图像所属数据集的概率分布 $p(d|I)$，在推理时动态选取最匹配专家，实现自适应领域切换。
- **MoCAE 置信度校准模块**：引入 Random Forest 回归器将专家原始置信度映射为真实 $\text{IoU}_{\max}$ 期望值，In-domain 与 Out-of-domain 分别以实际 IoU 和 0 为监督信号，防止分布外过置信。
- **Soft NMS + Score Voting 融合策略**：以高斯衰减替代硬截断抑制重叠框，并基于校准得分与框间 IoU 进行加权坐标重构，显著提升密集/簇状病灶的定位精度与召回率。

## 方法详解
- **骨干网络（YOLOS-base）**：纯 ViT 架构，输入固定 resize 至 $640 \times 640$，patch size $16 \times 16$ 生成 $N=1600$ 个 token，拼接 100 个 learnable `[DET]` 查询向量，经 12 层 Multi-head Self-Attention + FFN 输出 100 个候选框及二分类概率（cancer / no-object），无需 Anchor 与 NMS。
- **MoMo（合并训练基线）**：单 YOLOS 模型直接在 CSAW+DDSM+DMID 聚合数据上训练，用于验证“简单数据堆叠”与“专家隔离”的效果差异。
- **Simple MoE**：三个专家独立训练后，训练一个 CNN 门控网络 $G: \mathcal{X} \rightarrow \{1,2,3\}$ 预测图像来源分布；推理时取 $\arg\max_d p(d|I)$ 路由至对应专家输出，不做分数融合。
- **MoCAE 校准与融合流程**：
  1. **校准数据构造**：每个专家在全部三个子集验证集上提取 ResNet-18 512维 embedding 与原始置信度 $s$，拼接为 513 维特征；同域样本目标为 $\text{IoU}_{\max}$，跨域样本目标强制为 0。
  2. **RF 校准器训练**：3 个独立的 Random Forest（300棵树）最小化 $\text{MSE} = \frac{1}{N}\sum (C([f_k, s_k]) - \text{IoU}_{\max,k})^2$，输出校准得分 $\tilde{s}$。
  3. **Soft NMS**：对合并预测集，若 $\text{IoU}(b_i, b_j) > \theta$，则更新 $\tilde{s}_i \leftarrow \tilde{s}_i \cdot \exp\left(-\frac{\text{IoU}(b_i,b_j)^2}{\sigma_{\text{nms}}}\right)$，取 $\sigma_{\text{nms}}=0.08$。
  4. **Score Voting**：对存活框计算权重 $w_{ij} = \tilde{s}_i \cdot \exp\left(-\frac{(1-\text{IoU}(b_i,b_j))^2}{\sigma_{\text{nms}}}\right)$，重构坐标 $b_i' = \sum_{j\neq i} w_{ij} b_j / \sum_{j\neq i} w_{ij}$。
- **训练配置**：AdamW（lr=$5\times10^{-5}$, wd=$10^{-4}$），cosine schedule + warmup 0.05 + 1 restart，batch=8（gradient accumulation=2），200 epochs（early stopping patience=20），数据划分约 64%/16%/20%。

## 实验与结果
- **数据集**：CSAW（472）、DDSM（1,333）、DMID（201），总计 2,006 张带像素级 mask 的乳腺影像。
- **评估基线**：DETR、RT-DETRv2、MoMo（YOLOS 全量合并）、Simple MoE、YOLOS 单专家。
- **主要数值与结论**：
  - **CSAW**：YOLOS 单专家 mAP@50-95 最优（0.3292），MoCAE（0.3044）略低，但在 **small lesion** 上显著反超（0.3119 vs 0.2106）。
  - **DDSM**（较同质）：DETR 保持领先（0.1995），YOLOS 单专家与 Simple MoE 并列（0.1973），MoCAE 为 0.1357，说明单一强模型在均匀数据上已足够，MoE 优势不明显。
  - **DMID**（小样本且多样）：YOLOS 单专家 0.3364，Simple MoE 0.3348，MoCAE 0.3204；MoCAE 在 **large lesion** 上登顶（0.4910），并在 **small lesion** 恢复检测能力（0.0505 vs 单专家 0.0000）。
- **整体结论**：MoE+MoCAE 在异构、小样本、跨域场景下显著提升鲁棒性与可靠性，尤其对微小与簇状病灶检测增益明显；单源同质数据集仍可依赖强单模型。

## 相关工作脉络
- **DETR / RT-DETRv2**：端到端 Transformer 检测代表，本文指出其在 COCO 等大基准上表现优异，但医学影像中收敛慢、算力要求高；YOLOS 因纯 ViT 轻量且免 NMS 被选作骨干。
- **Faster R-CNN / YOLO 系列**：传统 CNN 检测器在 DDSM/CSAW 上广泛验证，依赖局部感受野，对全局上下文（如腺体密度与病灶关系）建模受限。
- **MoE 架构理论（Masoudnia et al.）**：综述指出 MoE 擅长子任务/子域划分，但医学影像中鲜有落地，且现有多模型集成缺乏可学习门控与联合优化。
- **MoCaE（Oksuz et al.）**：原始工作提出校准专家融合，但为 post-hoc 后处理；本文将其嵌入 MoE 训练管线，与领域专业化协同优化。
- **域自适应/泛化方法**：迁移学习、对抗特征对齐等常用于跨中心乳腺影像，但易引发过拟合或特征坍塌；本文从“专家隔离+校准融合”另辟蹊径，避开显式分布对齐。

## 局限性与未来方向
- **推理延迟较高**：3 个专家+校准模块使单图推理耗时约 8 秒（约为单 YOLOS
