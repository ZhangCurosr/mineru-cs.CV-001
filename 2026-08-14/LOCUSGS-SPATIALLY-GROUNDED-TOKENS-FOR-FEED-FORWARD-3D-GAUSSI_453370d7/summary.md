---
title: "LOCUSGS-SPATIALLY-GROUNDED-TOKENS-FOR-FEED-FORWARD-3D-GAUSSI"
source: https://arxiv.org/pdf/2608.12825v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:05:10"
---

# 论文速读：LOCUSGS-SPATIALLY-GROUNDED-TOKENS-FOR-FEED-FORWARD-3D-GAUSS

## 一句话总结
LocusGS 为基于查询的前馈 3D Gaussian Splatting 方法引入显式的 3D 锚点状态（中心与支撑半径），通过锚点引导的自/交叉注意力与锚点中心化偏移解码，使每个查询令牌专精于紧凑的局部空间区域，在相同 Gaussian 预算下显著提升新视角合成质量与空间组织结构。

## 研究问题与动机
- 现有 query-based 前馈 3DGS 方法（如 TokenGS）使用固定数量的可学习 latent query 聚合多视角特征并解码 Gaussian，解耦了输出预算与输入分辨率/视角数。
- 实证观察表明：同一 query 解码出的 Gaussian 往往空间高度离散，散布于场景不同区域，缺乏明确的空间职责划分与局部聚集性。
- 根因分析：纯 latent 特征表示的 query 没有显式指定其在 3D 空间中的作用位置与覆盖范围，导致跨视角特征聚合与 Gaussian 生成均缺乏空间先验约束。
- 本文动机：为每个 Gaussian query 附加可学习的 3D 锚点状态，并在整个解码流水线（特征聚合、token 交互、Gaussian 生成）中持续使用该空间状态，从而建立 query 级的空间专业化。

## 核心贡献（创新点）
1. **显式 3D 锚点状态增强**：为每个 query 维护中心 $\mu \in \mathbb{R}^3$ 与标量支撑半径 $r \in \mathbb{R}_+$，使查询具备可解释的 3D 空间坐标与作用域；与 TokenGS 等纯隐式 embedding 方案相比，将几何先验显式嵌入查询表示。
2. **锚点引导的交叉注意力几何偏置**：在 standard content-based cross-attention 中注入基于“锚点-射线”最短距离的高斯惩罚项，使 query 优先聚合空间几何一致的多视角 patch 特征；与仅依赖视觉相似度的基线相比，有效抑制外观驱动的跨视角歧义。
3. **锚点中心化高斯解码**：最终层不回归绝对 Gaussian 位置，而是预测相对锚点的局部偏移并由锚点半径缩放，强制同 query 生成的 Gaussian 聚集在球状局部邻域；与自由坐标解码相比，显著提升 token 级空间紧凑度。
4. **多层渲染监督与渐进锚点细化**：对中间解码层同步施加图像重建与可见性正则化，并按层权重加权；配合逐层残差更新锚点参数，实现从粗到精的空间结构学习。

## 方法详解
- **编码阶段**：多视角图像经共享 patch embedding 提取视觉特征，相机位姿编码为 patch-level Plücker ray 嵌入后叠加至对应 patch 特征，作为 decoder 的 memory。
- **初始锚点参数**：第 $i$ 个 query 的初始状态为 $(\mu_i^0, \rho_i^0)$，$\mu_i^0$ 在归一化 3D 空间随机初始化，原始半径 $\rho_i^0$ 通过逆 softplus 映射至期望初始支撑尺度，激活半径 $r_i^l = \mathrm{softplus}(\rho_i^l) + \epsilon$ 保证恒正。
- **锚点感知自注意力**：由当前锚点中心计算位置嵌入 $\mathbf{p}_i^l = \mathrm{MLP}(\mathrm{PE}(\mu_i^l))$，叠加至 query 特征后执行标准自注意力，使 token 间交互受 3D 位置约束而非仅依赖 latent 语义。
- **锚点-射线几何偏置交叉注意力**：计算锚点 $\mu_i^l$ 到第 $j$ 个图像 patch 对应 Plücker 射线 $\ell_j$ 的最短距离 $D(\mu_i^l, \ell_j) = \|\mu_i^l \times \bar{\mathbf{d}}_j - \bar{\mathbf{m}}_j\|_2$，构造偏置 $b_{ij}^l = -\frac{1}{2}\left(\frac{D(\mu_i^l, \ell_j)}{\sigma_0 r_i^l}\right)^2$，加入注意力 logits：$\alpha_{ij}^l = \mathrm{softmax}_j\left(\frac{(\bar{\mathbf{q}}_i^l)^\top \mathbf{k}_j}{\sqrt{d}} + \gamma b_{ij}^l\right)$，其中 $\gamma$ 为可学习非负尺度，$\sigma_0=0.1$ 为固定基础带宽。偏置值截断至 $[-20, 0]$ 保证数值稳定。
- **动态锚点细化**：每层输出后，轻量预测头生成残差 $\Delta\mu_i^l = f_\mu(q_i^{l+1})$，$\Delta\rho_i^l = f_\rho(q_i^{l+1})$，按 $\mu_i^{l+1} = \mu_i^l + \Delta\mu_i^l$、$\rho_i^{l+1} = \rho_i^l + \Delta\rho_i^l$ 累加更新，再重新计算激活半径。
- **锚点中心化解码**：最后一层预测 $K$ 个局部偏移 $\delta_{i,k} = f_\delta(q_i^L)$，最终 Gaussian 中心为 $\mu_{i,k}^G = \mu_i^L + r_i^L \delta_{i,k}$；尺度、旋转、颜色、不透明度沿用标准 head 预测。
- **多层监督损失**：在选定层 $l_m$ 解码出中间 Gaussian 集合 $\mathcal{G}^{l_m}$ 与锚点，计算 $\mathcal{L}^{l_m} = \mathcal{L}_{\mathrm{rec}}^{l_m} + \lambda_G \mathcal{L}_{\mathrm{vis}}(\{\mu_{i,k}^{G,l_m}\}) + \lambda_A \mathcal{L}_{\mathrm{vis}}(\{\mu_i^{l_m}\})$，总损失 $\mathcal{L} = \sum_m w_m \mathcal{L}^{l_m}$（$w_m$ 随
