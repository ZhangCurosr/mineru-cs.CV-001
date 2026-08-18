---
title: "EGM-Det-Entropy-Guided-Multimodal-Adaptive-Fusion-for-UAV-RG"
source: https://arxiv.org/pdf/2608.11685v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 11:29:49"
---

# 论文速读：EGM-Det-Entropy-Guided-Multimodal-Adaptive-Fusion-for-UAV-RG

## 一句话总结
提出 EGM-Det，一种基于信息熵引导的 RGB-IR 模态自适应融合框架，用于 UAV 视角目标检测；通过 EOGF 模块构建浅层熵先验指导局部对齐与空-通道门控融合，并引入双教师跨模态蒸馏与学生门控熵自适应加权，显式建模空间变化的模态可靠性，在 DroneVehicle、LLVIP 和 VEDAI 三个基准上均取得 SOTA 性能。

## 研究问题与动机
- UAV 视角下 RGB 与红外图像的空间可靠性差异显著：不同成像机制导致纹理、对比度与热响应异构，且同一区域内某模态可靠而另一模态模糊，呈现强内容依赖性。
- 现有 RGB-IR 检测器多采用静态权重、交叉注意力或 Transformer 交互进行特征融合，但模态选择过程仅由最终检测损失隐式优化，缺乏对“何时何地应信任哪种模态”的显式建模。
- 公开数据集（如 DroneVehicle）存在跨模态配对不一致（未匹配实例、框位/朝向偏差、单模态标注）与类别级混淆（car/freight car/truck/van），直接影响检测监督与教师置信度信号的可靠性。
- 传统检测蒸馏侧重于特征或预测头模仿，未针对多模态融合中的“模态偏好行为”进行选择性蒸馏，也未考虑学生决策不确定性的动态监督需求。

## 核心贡献（创新点）
1. **熵引导的偏移-门控自适应融合模块（EOGF）**：利用输入级强度、局部熵与跨模态差异构建浅层熵先验，联合可微局部偏移对齐与空间-通道门控实现多尺度融合；与现有静态/纯注意力融合的本质区别在于显式引入信息熵作为可靠性先验，动态感知场景属性并选择性聚合线索。
2. **双教师模态偏好蒸馏策略**：将冻结的 RGB/IR 单模态教师密集分类置信度归一化为软模态偏好分布 $Q_l$，作为学生空间门控 $G_l$ 的显式监督目标；与已有蒸馏工作的本质区别在于不要求学生模仿融合特征或预测结果，而是直接蒸馏“模态选择行为”本身。
3. **学生门控熵自适应重加权机制**：结合教师响应可靠性 $R_l$ 与学生门控熵 $H_l$ 构造像素级权重 $\omega_l$，在模糊区域强化 KL 监督、在确信区域放松约束；与均匀权重蒸馏的本质区别在于 uncertainty-aware 的细粒度正则化，避免对已稳定区域的过度扰动。
4. **DroneVehicle 标注重构与统一评测协议**：系统性修正跨模态不一致与类别级标签错误，建立可复现的 paired RGB-IR 评估流程；使 EGM-Det 在 DroneVehicle（refined）、LLVIP、VEDAI 三项基准上均达 SOTA，并揭示标注噪声对多模态训练的级联影响。

## 方法详解
- **双分支骨干**：RGB 与 IR 图像分别经对称骨干网络提取多尺度特征 $F_l^{rgb}$ 与 $F_l^{ir}$，仅在选定层级通过 EOGF 交互，保留模态特异性表示。
- **EOGF 模块**：
  1. **熵先验构建**：输入归一化强度 $I^{rgb}, I^{ir}$ 与局部熵图 $H^{rgb}, H^{ir}$，结合跨模态强度差 $|I^{rgb}-I^{ir}|$ 经轻量估计器 $\Phi_{prior}$ 生成先验 $P$，下采样适配各层级 $P_l$。
  2. **局部偏移对齐**：特征投影至共享嵌入空间 $Z_l^{rgb}, Z_l^{ir}$，结合 $P_l$ 与特征差异预测偏移场 $\Delta_l$ 与对齐置信度 $C_l$，通过可微形变 $\mathcal{W}(\cdot,\cdot)$ 对 RGB 特征做局部补偿（非全局配准）。
  3. **空-通道门控融合**：预测空间门控 $G_l = [g_l^{rgb}, g_l^{ir}]$（同位置 Softmax 归一）与通道门控 $c_l^{rgb}, c_l^{ir}$，融合公式为 $F_l^{fus} = g_l^{rgb} \odot c_l^{rgb} \odot \tilde{F}_l^{rgb} + g_l^{ir} \odot c_l^{ir} \odot F_l^{ir}$。
- **选择性跨模态蒸馏**：
  1. 教师最大类别置信度密集图 $\bar{M}_l^m$ 经 $\epsilon$-平滑归一化得软偏好目标 $Q_l$，仅监督空间门控。
  2. 教师可靠性 $R_l = \max(\bar{M}_l^{rgb}, \bar{M}_l^{ir})$；学生门控熵 $H_l = -\frac{1}{\log 2}\sum g \log g$ 量化决策模糊度。
  3. 自适应权重 $\omega_l = \text{clip}(1 + \beta_{pos}[\text{sg}(H_l)-\tau]_+ - \beta_{neg}[\tau-\text{sg}(H_l)]_+, \omega_{min}, \omega_{max})$（含 stop-gradient 防篡改），门控损失为 $\mathcal{L}_{gate}^{adapt} = \text{KL}(Q_l \| G_l)$ 以 $R_l
