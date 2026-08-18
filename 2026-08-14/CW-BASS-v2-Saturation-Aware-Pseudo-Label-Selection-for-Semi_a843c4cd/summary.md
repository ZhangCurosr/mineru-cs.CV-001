---
title: "CW-BASS-v2-Saturation-Aware-Pseudo-Label-Selection-for-Semi"
source: https://arxiv.org/pdf/2608.12773v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 13:43:37"
---

# 论文速读：CW-BASS-v2-Saturation-Aware-Pseudo-Label-Selection-for-Semi

## 一句话总结
本文针对基础模型（DINOv2）教师置信度饱和导致传统自适应阈值失效的问题，提出 CW-BASS v2。该方法通过留出校准切片无偏估计伪标签噪声，引入自自适应置信度下界，并设计一个单次诊断门控机制，根据置信集合可靠性 $\pi_{kept}$ 在严格过滤与自适应规则之间自动切换，避免确认偏差。

## 研究问题与动机
1. **半监督语义分割（SSSS）的核心瓶颈**：自训练依赖教师生成的伪标签，但伪标签含噪，模型若盲目信任错误预测会引发确认偏差，因此“信任哪些伪标签”是几乎所有 SSSS 方法要解决的根本问题。
2. **ResNet 时代共识在基础模型下失效**：FlexMatch、FreeMatch 等自适应/课程阈值规则依赖教师置信度分布的较宽动态范围；DINOv2 教师输出高度饱和（98% 像素置信度≥0.95），导致动态阈值的选择信号坍缩，规则反而引入大量噪声。
3. **In-batch 噪声估计存在向下偏差**：直接在训练集上估计伪标签错误率会因学生模型已过拟合这些像素而低估噪声，形成自强化循环，使自适应阈值不断下调。
4. **缺乏 regime 感知的统一诊断**：现有方法要么一味收紧阈值（如 UniMatch V2），要么盲目沿用旧自适应规则，缺少基于教师自身校准状态的 principled 选择机制。

## 核心贡献（创新点）
1. **饱和度感知门控选择机制**：通过单次前向在留出校准集上测量 $\pi_{kept}=Pr[correct|c\geq\tau]$，以操作阈值 $\tau$ 为界自动判定严格过滤或自适应下界，决策边界无需针对 mIoU 调参，跨六类 DINOv2 教师 blind 预测准确。
2. **无偏留出校准诊断**：将 5% 标注集划为校准切片，仅用于噪声率估计而不参与监督梯度；从理论上证明该校准估计在给定教师条件下无偏，打破了 in-batch 反馈驱动的向下偏差循环。
3. **具有稳定性保证的自自适应置信度下界**：设计 $\tau_k^{floor}=s\cdot\bar{c}_t\cdot(\mu_k/\max_j\mu_j)$ 并证明其将保留率约束在固定分位数 $1-G_k(sr_k)$，理论保证保留率不会随教师变自信而坍缩至 1。
4. **首次针对基础模型教师的阈值机制级审计**：完整拆解“置信度饱和→动态范围坍缩→掩码洪水→早期峰值后衰退”的失败链条，量化各规则在匹配 batch 下的性能边界，明确严格阈值在饱和 regime 的必要性。

## 方法详解
- **整体架构**：继承 UniMatch/CW-BASS 的 weak-to-strong consistency、EMA 教师与双强视图（图像空间 CutMix + 特征通道 Dropout）框架，保留置信加权交叉熵 $\mathcal{L}_{cw}$ 与 Sobel 边界辅助项。
- **留出校准（Held-out Calibration）**：$\mathcal{L}=\mathcal{L}_{tr}\sqcup\mathcal{L}_{cal}$，$\alpha=5\%$。每 epoch 仅对 $\mathcal{L}_{cal}$ 跑一次教师前向，统计每类在截止阈值 $\tau$ 下的伪标签噪声率 $\widehat{\varepsilon}_k(\tau)=1-n_k^c(\tau)/n_k(\tau)$。该估计与优化过程统计独立，满足条件无偏性（Proposition 1）。
- **自自适应置信度下界**：维护教师平均置信度的 EMA $\bar{c}_t$ 与各分类别运行均值 $\mu_k$，定义 $\tau_k^{floor}=s\cdot\bar{c}_t\cdot(\mu_k/\max_j\mu_j)$（$s=0.95$）。最终阈值取 $\tau_k^{final}=\max(\tau^{dyn},\tau_k^{floor})$，保留掩码 $\mathcal{M}=\{(h,w):c_{h,w}\geq\tau_{\hat{y}_{h,w}}^{final}\}$。Theorem 1 证明在缩放族假设下 $\bar{c}_t$ 因子约去，保留率被钉在固定分位数且严格小于 1。
- **饱和度门控（Saturation Gate）**：在 $\mathcal{L}_{cal}$ 上计算 $\pi_{kept}=Pr[\hat{y}=y|c\geq 0.95]$。若 $\pi_{kept}\geq 0.95$ 则启用严格阈值 $\tau=0.95$；否则启用自适应下界（Equation 10）。该门控仅依赖教师一次前向，边界即操作阈值本身，不依赖下游 mIoU 调参。
- **损失函数**：$\mathcal{L}=\frac{1}{2}(\mathcal{L}_x+\frac{1}{2}(\mathcal{L}_s+\mathcal{L}_{fp}))$，其中 $\mathcal{L}_s=\mathcal{L}_{cw}+\beta_b\mathbb{E}_{(h,w)\in B}[CE]$，$\gamma=1,\beta_b=0.5$。

## 实验与结果
- **数据集与协议**：Pascal VOC 2012（1/8=183 张，1/4=366 张）、Cityscapes、ADE20K（150 类）；骨干网 DINOv2-S/B/L + DPT-lite 解码器；所有规则共享 backbone、优化器（AdamW）、LR（backbone $5\times10^{-6}$ / decoder $2\times10^{-4}$）、crop、batch=16 与数据划分，仅变动阈值规则。
- **Pascal VOC 1/8**：严格阈值达到 87.40 mIoU（3 seed 均值 86.19±1.82），与 UniMatch V2 报告值 87.9 接近；所有自适应规则（Dynamic/Floor/Per-class/SoftMatch/FreeMatch）均低于严格规则，落在 79–85 区间，且全部呈现早期峰值后衰退。
- **Cityscapes**：严格规则与自适应规则基本持平（差距 ≤0.7 mIoU），门控正确选择严格阈值。
- **ADE20K 1/8**：教师置信集合可靠性较低（$\pi_{kept}\approx 89\%$），门控激活自适应下界，达到 50.58 mIoU，比严格阈值（49.10）提升 +1.5（单 seed），超越 UniMatch V2-B 报告的 49.8。
- **门控验证**：Table VII 显示 6 个 DINOv2 教师的 $S=\Pr[c\geq 0.95]$ 均≥82%，但 $\pi_{kept}$ 区分出 Pascal/Cityscapes（≈98%）与 ADE20K（≈89%），门控以 $\tau=0.95$ 为界 blind 做出正确 strict/floor 判定。
- **最强结果与提升**：Pascal VOC 1/8 复现 SOTA 操作点（87.4）；ADE20K 单 seed 在自适应 regime 实现 +1.5 mIoU 提升。

## 相关工作脉络
1. **UniMatch V2**：确立 DINOv2 骨干上严格固定阈值（$\tau=0.95$）+ 双强视图的当前 SOTA；本文
