---
title: "PolarSym-Polar-Geometry-aware-Attention-for-CAD-Floorplan-Pa"
source: https://arxiv.org/pdf/2608.11793v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 12:35:56"
---

# 论文速读：PolarSym-Polar-Geometry-aware-Attention-for-CAD-Floorplan-Pa

## 一句话总结
本文提出 PolarSym 框架，通过将 Transformer 注意力机制显式解耦为“方向”与“距离”两个独立的极坐标几何先验分支，解决了现有 CAD 户型图解析方法在处理长距离对称与重复布局时语义注意力失效的问题，在公开基准上显著提升了解析精度与训练收敛速度。

## 研究问题与动机
1. **核心问题**：现有基于 Transformer 的 CAD 平面解析方法（如 SymPoint 系列）主要依赖语义相似性构建注意力，空间几何关系仅通过隐式位置编码表达，缺乏对建筑布局内在轴对称/重复结构的显式建模。
2. **长距对称匹配缺陷**：面对镜像结构、重复房间或远距离对称目标时，纯语义注意力易退化为主要关注局部邻近区域，难以建立稳定可靠的长程结构对应关系。
3. **笛卡尔坐标与单调假设的局限**：传统位置编码将方向与距离混合于统一坐标系，且距离感知注意力常默认“距离越近相关性越高”的单调衰减关系，与建筑平面图中“远距离对称部件高度相关”的几何规律相悖。
4. **训练效率瓶颈**：现有方法往往依赖长时间训练（如 SymPoint V1 需 1000 epoch）或复杂图结构来弥补几何建模不足，在受限算力下收敛慢、数据利用率低。

## 核心贡献（创新点）
1. **提出极坐标系几何注意力框架 PolarSym**：首次在注意力计算层将建筑组件的几何关系显式解耦为方向与距离两个互补因子，而非将其混合为统一的位置嵌入，为 CAD 平面解析提供了统一的几何建模范式。
2. **设计方向–距离协同建模机制**：提出基于 SF-RoPE 的方向建模分支捕捉全局方向一致性，以及基于 RDB 的径向建模分支学习非单调的长程距离响应，并通过动态门控策略实现两者的自适应融合。
3. **轻量级架构兼容与高效优化**：几何先验以加法方式直接调制原始注意力分数，无需修改骨干网络；在相同训练配置下全面超越 SymPoint V2 复现基线，并展现出更快的收敛速度与更稳定的优化动态。

## 方法详解
- **整体流程**：输入为 Point Transformer 骨干提取的 N 个点特征 $\mathbf{F} \in \mathbb{R}^{N \times D}$ 与二维坐标 $\mathbf{P} \in \mathbb{R}^{N \times 2}$，经线性投影得到 $\mathbf{Q}, \mathbf{K}, \mathbf{V}$。PolarSym 解码器在保留原始语义注意力的基础上，并行注入方向与径向两个几何分支，最终通过动态门控融合输出几何感知注意力图。
- **语义注意力基准**：$S^{\mathrm{sem}} = \frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}$，仅基于特征相似度计算，无法捕捉几何对称性。
- **方向建模分支 (SF-RoPE)**：针对传统 RoPE 仅做相邻特征对旋转的局限，SF-RoPE 将特征维度平分为两半并进行跨半区旋转。旋转角度 $\theta_k = \lambda \phi(\pmb{p}_i)$ 由物理坐标映射的极角决定，不同 attention head 交替采用水平/垂直坐标编码。方向增强项定义为 $\Delta^{\mathrm{dir}} = S^{\mathrm{dir}} - S^{\mathrm{sem}}$，确保语义表征与方向几何严格解耦。
- **径向建模分支 (RDB)**：摒弃距离-相关性单调假设，RDB 将两点欧氏距离 $d_{ij}$ 通过傅里叶特征映射 $\gamma(d_{ij})$ 转换为多频带表示，再经轻量 MLP 预测径向注意力偏置 $B^{\mathrm{rad}}$。该设计使网络能学到具有多个局部峰值的非单调距离响应，从而为远距离但结构对称的组件分配高注意力。
- **动态门控融合策略**：最终注意力 $S = S^{\mathrm{sem}} + \alpha \Delta^{\mathrm{dir}} + \beta B^{\mathrm{rad}}$，其中 $\alpha = \operatorname{tanh}(g_\alpha)$，$\beta = \operatorname{tanh}(g_\beta)$ 为可学习门控系数。为稳定训练，初始化时设 $g_\alpha = 2.0
