---
title: "InterPruner: Interactive Structured Pruning via Taylor-Implicit Criterion and Language-Prior Modulator for Multimodal Object Detection"
source: https://arxiv.org/pdf/2608.10724v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:59:44"
---

# 论文速读：InterPruner: Interactive Structured Pruning via Taylor-Implicit Criterion and Language-Prior Modulator for Multimodal Object Detection

## 一句话总结
本文提出了 InterPruner，首个针对 RGB-红外双模态目标检测器的交互式结构化通道剪枝框架。通过泰勒隐式准则(TIC)量化通道贡献、模态交互冗余分析器(MIRA)捕捉跨模态补偿性、场景先验通道锚点(SPCA)引入语言先验动态调节，在 FLIR 与 DroneVehicle 数据集上实现了高精度与低计算开销的平衡，甚至在 50% 剪枝率下于 FLIR 数据集上获得 +0.6% 的 mAP 提升。

## 研究问题与动机
- **跨模态冗余未被建模**：现有单模态剪枝方法假设通道重要性可独立评估，直接移植至双分支 RGB-红外检测器会破坏模态间的特征对齐与互补性，导致性能骤降。
- **固定准则无法适应动态场景**：遥感场景中光照、天气、背景复杂度变化剧烈，通道冗余分布具有显著的场景依赖性，静态剪枝标准难以兼顾不同环境下的最优保留策略。
- **真实剪枝代价高昂**：精确评估通道重要性通常需完整重训练或大量微调，在双分支大模型上计算开销不可接受，亟需低成本的损失近似机制。
- **CNN 骨干仍占主导**：尽管 ViT/Mamba 在多模态感知中兴起，但遥感高分辨率图像下 CNN 仍是兼顾效率与多尺度特征的主流选择，其通道级剪枝需求迫切。

## 核心贡献（创新点）
1. **泰勒隐式准则(TIC)**：利用高阶泰勒展开与隐函数定理近似剪枝后的真实损失变化，无需完整重训练即可准确量化各通道对检测损失的贡献。与仅依赖单次梯度或 BN 缩放系数的现有方法本质不同，显式建模了参数补偿带来的非线性损失曲率。
2. **模态交互冗余分析器(MIRA)**：通过交替遮蔽单一模态并执行短步数梯度适应，度量残存分支通道响应的变化幅度，显式识别跨模态可补偿的冗余通道。与忽略模态间依赖的单分支剪枝方法本质不同。
3. **场景先验通道锚点(SPCA)**：利用冻结的多模态大语言模型(Qwen2.5-VL)生成结构化场景描述，并通过轻量化 MLP 映射为通道级权重，实现基于语义上下文的动态剪枝决策。与静态固定剪枝策略本质不同。
4. **统一交互式通道选择机制**：将通道贡献、跨模态冗余与场景权重融合，结合等宽约束与贪心预算分配，实现双分支协同剪枝，填补了 RGB-红外结构化剪枝领域的空白。

## 方法详解
- **问题形式化**：给定预训练模型参数 $\mathbf{W}$ 与通道二进制掩码 $\mathbf{M}$，目标是在满足剪枝率 $\rho$ 的可行掩码集合 $\mathcal{S}$ 中，最小化检测损失 $\mathcal{L}(\mathbf{W}^*(\widehat{\mathbf{M}}), \widehat{\mathbf{M}})$，其中 $\mathbf{W}^*$ 为掩码固定后的最优参数。
- **泰勒隐式准则(TIC)**：定义即时损失变化 $\Delta \mathcal{L}_{el}$ 与真实损失变化 $\Delta \mathcal{L}_{rl}$，后者可分解为即时变化与参数恢复损失 $\Delta \mathcal{L}_{rec}$。利用隐函数定理将参数位移 $\Delta \mathbf{W}$ 展开为一阶 $\Delta \mathbf{W}^{(1)} = -\mathbf{H}^{-1}\mathbf{g}$ 与二阶项，代入泰勒展开消去高阶项后得到：
  $C_q = \Delta \mathcal{L}_{el}^q - \left(\frac{1}{2}\mathbf{g}_q^T \mathbf{H}_q^{-1}\mathbf
