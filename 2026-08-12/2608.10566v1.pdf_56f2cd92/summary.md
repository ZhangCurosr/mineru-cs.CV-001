---
title: "Iterative Erasure Count Is Not an Afine-Invariant Concept Dimension"
source: https://arxiv.org/pdf/2608.10566v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:04:21"
---

# 论文速读：Iterative Erasure Count Is Not an Afine-Invariant Concept Dimension

## 一句话总结
本文从理论上严格证明：常用迭代擦除计数（iterative erasure count）并非仿射不变的概念维度估计量；在保持预测信息与充分维度不变的可逆重新参数化下，该计数与累积编辑秩会发生整数跳变。通过总体构造定理、有限Adam/QR复现实验与冻结视觉特征的压力测试，文章明确区分了过程相对估计量与总体语义维度，并给出完整度量大小的仿射等变性定理及其协方差特例推论。

## 研究问题与动机
- **核心问题**：神经网络表征中究竟有多少个方向用于编码某一概念（如手-物接触、物理变量）？现有工作普遍采用迭代零空间投影（如INLP及其视频分析延伸）拟合线性探针后移除方向，将停止次数或累积移除秩直接解读为概念的内蕴维度。
- **方法混淆**：现有做法将“模型定义的总体量”（生成维度、充分线性维度、最小守卫秩）与“过程定义的统计量”（停止计数、累积编辑秩）混为一谈，忽略了协方差估计、正则化、度量选择与停权规则对计数的支配作用。
- **理论缺口**：先前研究（如LEACE/MP）仅指出一阶矩守卫或欧氏非等价性，Gauge-freedom 工作仅说明欧氏相似性随可逆变换改变，均缺乏对“离散整数计数跳变”的精确刻画，也未给出完整擦除轨迹的等变条件。
- **应用误导风险**：视频世界模型等后续工作将数十个正交探针方向直接解释为“分布式物理变量”，若该计数本身不具备坐标无关性，则此类语义推断缺乏几何基础。

## 核心贡献（创新点）
1. **证明总体Ridge探针下迭代计数不可识别**：构造已知充分维度与守卫秩均为1的高斯生成过程，展示在 $a=0$ 时计数为1、$a\neq 0$ 时计数严格跳变为2，且该跳变对 $\lambda \in \{0, 0.1, 1, 10\}$ 均成立。
2. **建立完整度量大小的仿射等变性定理**：证明当正定系数度量、探针规则、正则化与平局规则沿可逆变换一致 transported 时，整个累积擦除轨迹（得分、秩、计数）严格等变；精确协方差度量仅为该定理的一个特例推论，而非唯一语义几何。
3. **全QR多输出整数跳变反例**：针对 Motivating 视频分析的 full-QR 移除代数（两输出MSE），证明累积编辑秩从 $r$ 跳变为 $2r$（环境维度），而充分维度与最小守卫秩恒为 $r$。
4. **预声明映射压力测试与交叉拟合审计**：在冻结 V-JEPA2/DINOv2 特征上施加稠密各向异性、对角与正交映射族，结合三角色 cross-fit 协议，实证分离有限样本估计残留与真实多路径可访问性，明确迭代计数仅为过程相对 estimand。

## 方法详解
- **五类“维度”量纲分离**：明确区分 generating dimension（生成规则使用的潜变量数）、sufficient linear dimension（满足 $Y \perp X \mid XB$ 的最小 $r$）、cross-covariance rank（$\text{rank Cov}(X, \phi(Y))$，标量编码下 $\le 1$）、iterative erasure count（停止时间）与 linear guarding rank（满足声明守卫条件的最小干预秩）。强调后三者依赖攻击者类别、损失与基线风险，不得混用同一名称。
- **欧氏擦除非不变性（Prop 1）**：对行特征 $x$ 作可逆变换 $z = xA$，原坐标欧氏投影 $x(I - ww^\top/(w^\top w))$ 与变换后投影再映射回原坐标 $xA(I - w_z w_z^\top/(w_z^\top w_z))A^{-1}$ 一般不相等，除非 $A$ 保持相关欧氏度量。
- **完整轨迹等变性（Prop 2）**：给定 $G_X \succ 0$，在迭代 $k$ 中使 probe 返回系数块 $B_k$，追加至 $G_X$-正交基 $U_k$，编辑算子 $T_k = I - U_k U_k^\top G_X$。若对 $Z = XA$ 传输度量 $G_Z = A^\top G_X A$，且 probe 等变（$B_k^Z = A^{-1}B_k$）、正则化随 $A$ 传输、tie-breaking 等变，则归纳可得 $ZT_k^Z = (XT_k)A$，计数与编辑秩完全一致。
- **总体 Rank-One 构造（Theorem 1）**：设 $S, N \overset{iid}{\sim} \mathcal{N}(0,1)$，$Y=\mathbb{I}[S\ge 0]$，$X_a=(S+aN, N)$。对任意有限 $\lambda \ge 0$，保持 Euclidean-orthonormal 累积基 $U_k$，迭代拟合 population ridge LS 并 residualize。$a=0$ 时计数为1；$a\neq 0$ 时第一次移除后残差方向与交叉协方差不再平行，第二次编辑必然发生，计数严格为2。
- **全QR多输出构造（Theorem 2）**：$Y=S \in \mathbb{R}^r$，$X_a=(S+aN, N) \in \mathbb{R}^{2r}$。按文献 [13] 的全系数列 QR 移除代数，$a=0$ 时累积编辑秩为 $r$；$a\neq 0$ 时第一块系数的正交补张成
