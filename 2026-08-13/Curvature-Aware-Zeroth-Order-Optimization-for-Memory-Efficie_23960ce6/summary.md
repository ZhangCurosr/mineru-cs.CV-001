---
title: "Curvature-Aware-Zeroth-Order-Optimization-for-Memory-Efficie"
source: https://arxiv.org/pdf/2608.12279v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 11:21:39"
field: "测试时适配与零阶优化"
keywords: ["test-time adaptation", "zeroth-order optimization", "memory-efficient", "curvature-aware", "forward-only", "edge deployment"]
innovations: ["揭示TTA中Hessian持久低秩与慢变性质，为ZO方差缩减提供几何先验", "提出CAZO曲率感知零阶优化，通过EMA对角Hessian估计实现各向异性扰动采样", "在不使用反向传播的前提下实现70%+内存降低与SOTA适配精度"]
benchmarks: ["ImageNet-C severity-5", "ImageNet-R", "ImageNet-V2", "ImageNet-Sketch"]
---

# 论文速读：Curvature-Aware Zeroth-Order Optimization for Memory-Efficient Test-Time Adaptation

## 一句话总结
本文提出 CAZO（曲率感知零阶优化），通过揭示测试时适配过程中 Hessian 矩阵的持久低秩与慢变性质，利用滑动平均估计的对角 Hessian 构造各向异性扰动采样，在仅前向传播的前提下显著降低零阶梯度估计的方差，实现内存开销降低约 70% 的同时达到 SOTA 适配精度。

## 研究问题与动机
- **测试时适配（TTA）的内存瓶颈**：现有 TTA 方法普遍依赖反向传播（BP）进行微调，在边缘设备等资源受限场景下内存与计算开销难以接受。
- **零阶（ZO）方法的梯度估计方差过高**：标准 ZO 方法仅通过前向函数评估估计梯度，其方差随参数维度线性放大（$O(d/k)$），导致在高维适配任务中收敛缓慢、效果差。
- **现有 BP-free 方法对损失地形利用不足**：FOA 等方法采用进化策略，ZOA 面向量化模型，均未有效利用适配过程中损失曲面的几何先验来降低采样方差。
- **低资源部署需求**：需要一种完全无需反向传播、内存占用极低、同时保持高适配精度的方案，以支持端侧部署。

## 核心贡献（创新点）
- **揭示 TTA 中 Hessian 的持久低秩与慢变性质**：实验证明适配过程中 Hessian 的主要特征值高度集中（Top 20 特征值解释 >96% 方差），且主曲率子空间跨步骤投影比稳定在 ~0.9，为方差缩减提供理论依据。
- **提出 CAZO 曲率感知零阶优化方法**：通过滑动平均（EMA）估计对角 Hessian，构造各向异性扰动协方差矩阵，使采样在尖锐方向衰减、平坦方向放大，从而显著降低梯度估计方差。
- **提供非凸收敛性保证**：在标准光滑性与方差假设下，推导 CAZO 达到 $O(1/\sqrt{T})$ 收敛率，且收敛常数显式依赖于曲率边界 $\beta_l$ 与 $\beta_u$。
- **实现 70%+ 内存降低与 SOTA 性能**：在 ImageNet-C 等多个基准上，CAZO 以 <1,700 MB 内存（较 TENT 等 BP 方法降低 >70%）取得 69.0% 平均精度，超越所有 BP-free 与 BP-based 基线。

## 方法详解
- **零阶梯度估计基础**：采用对称两点估计，对参数 $\theta_t$ 施加扰动 $u_i$，通过 $\hat{g}(\theta_t) = \frac{1}{k}\sum_{i=1}^k \frac{\mathcal{L}(\theta_t + \epsilon u_i) - \mathcal{L}(\theta_t - \epsilon u_i)}{2\epsilon} u_i$ 估计梯度。
- **曲率感知扰动采样**：扰动不再来自各向同性 $\mathcal{N}(0, I)$，而是来自预处理高斯分布 $\mathcal{N}(0, \tilde{H}_t^{-1})$，其中 $\tilde{H}_t$ 为对角 Hessian 近似矩阵，沿高曲率方向衰减扰动、沿低曲率方向放大扰动。
- **对角 Hessian 的 EMA 估计**：使用逐元素平方的零阶梯度进行滑动平均更新 $D_t = (1-\nu)D_{t-1} + \nu \hat{g}^2(\theta_{t-1})$，再归一化得到 $\tilde{H}_t = \text{diag}\left(\frac{D_t}{1-(1-\nu)^t}\right)$，计算与内存开销仅为 $O(d)$。
- **复合损失函数**：采用无监督熵损失 + 基于干净数据特征统计的 MSE 特征对齐损失，适配器仅更新轻量 adapter 参数，预训练权重冻结。
- **收敛性分析**：Lemma 1 给出梯度估计的偏差界 $\mathbb{E}[\hat{g}] = \tilde{H}_t^{-1}\nabla\mathcal{L} + \mathcal{O}(\epsilon)$ 与方差上界；Theorem 1 证明在恰当学习率下达到 $\mathcal{O}(1/\sqrt{T})$ 收敛率，曲率条件良好时常数项更小。

## 实验与结果
- **数据集与模型**：ImageNet-C（15 类失真 × 5 个严重级别，评估 severity-5）、ImageNet-R/V2/Sketch；源模型为 ViT-B/16。
- **主要结果（ImageNet-C, severity-5）**：CAZO 平均精度 **69.0%**，超越 FOA（65.8%，+3.2%）、ZOA（67.5%，+1.5%），以及 BP-based 方法 TENT（59.8%）、SAR（62.7%）、CoTTA（61.9%）。
- **持续适配（CTTA）结果**：CAZO 达 **65.3%**，超越 LCoTTA（62.3%）、ETA（61.7%）、SAR（61.6%）。
- **内存效率**：CAZO（k=20）仅需 **1,695 MB**，相较 TENT 的 6,404 MB、SAR 的 6,405 MB、CoTTA 的 17,773 MB，降低 **4–10 倍**；且内存几乎不随 k 变化。
- **量化兼容性**：8-bit 量化下 CAZO 达 67.8%，6-bit 下仍保持 61.2%，显著优于所有对比方法。
- **消融结果**：适配器最佳位置为 layer 3；降采样比 384 最优；k=20 时精度最高但 k=2 即可接近 FOA 性能（65.2% vs 65.8%）；EMA 系数 ν=0.8 最佳；扰动尺度 ϵ=0.1 最优。

## 相关工作脉络
- **TENT（2021）**：基于 BP 的熵最小化 TTA 方法，通过更新 BN 参数实现适配；CAZO 与其定位差异在于完全不使用反向传播，内存降低 70%+。
- **FOA（2024）**：基于 CMA-ES 进化策略的 BP-free TTA，仅前向传播；CAZO 与之同为 BP-free，但引入曲率感知各向异性采样替代随机进化搜索，在相当内存预算下精度更高（69.0% vs 65.8%）。
- **ZOA（2025，并发工作）**：面向量化模型的 ZO TTA，使用域知识偏移库；CAZO 与 ZOA 均使用两点 ZO 更新，但 CAZO 聚焦曲率感知采样以通用提升 ZO 效率，在相近内存下精度更强。
- **MeZO（2023）**：证明 ZO 微调收敛性取决于有效内在维度而非原始维度；CAZO 在此基础上进一步利用 Hessian 低秩结构进行各向异性采样，实现更激进的方差缩减。
- **LCoTTA（2025）**：发现熵最小化梯度的主子空间并投影更新以稳定持续适配；CAZO 与之灵感相通（均发现低维结构），但 CAZO 作用于 ZO 扰动方向而非 BP 梯度，且完全无反向传播。
- **Entropy Minimization TTA 系列（SHOT, DELTA, EATA, DeYO, RoTTA）**：均为 BP-based 方法，通过不同正则/筛选机制缓解熵最小化的退化问题；CAZO 从优化器设计层面绕过 BP，从根本上降低内存开销。

## 局限性与未来方向
- **ViT 架构适配为主**：实验主要集中在 Vision Transformer，对 CNN 或其他架构的泛化性有待验证（虽补充实验涉及 DeiT/Swin-Tiny，但覆盖有限）。
- **扰动数量 k 与推理延迟的权衡**：k=20 时精度最优但需 40 次前向传播，实际部署需在精度与延迟间做取舍。
- **EMA 平滑系数 ν 的敏感性**：ν 过大（如 1.0）导致不稳定，过小则响应滞后，需根据具体场景调参。
- **未探索多任务/多模态场景**：当前仅在图像分类 TTA 上验证，向多模态或序列任务迁移需进一步研究。
- **曲率估计的对角近似可能损失跨维度交互信息**：完整 Hessian 逆的近似虽节省内存，但在强耦合参数场景下可能不够精确。

## 研究启发与可借鉴点
- **Hessian 低秩性质的实验分析方法**：可通过特征值谱分析 + 跨步投影比 $\rho_t^{(r)}$ 来验证任何优化过程中的低维结构，该方法论可直接迁移到其他优化场景。
- **对角 Hessian 的 EMA 估计策略**：用滑动平均逐元素平方梯度近似对角曲率，计算简洁、内存低，可推广至其他 ZO 优化任务（如 LLM 微调、神经架构搜索）。
- **各向异性扰动采样设计**：将曲率先验融入 ZO 采样分布的思路，可与贝叶斯优化、进化策略等方法结合，探索更高效的黑盒优化。
- **轻量 Adapter + 冻结主干的 TTA 范式**：在早期 Transformer 层插入低维适配器（layer 3、降采样比 384）是一种高效适配架构，值得在其他视觉/多模态任务中复现。
- **曲率感知 ZO 与量化部署的结合**：CAZO 在 6/8-bit 量化下仍保持优势，表明该方法与边缘 AI 部署管线高度兼容，可进一步探索与量化感知训练的结合。

## 关键术语表
- **Test-Time Adaptation (TTA)**：测试时适配，指模型在推理阶段利用无标签测试数据在线调整自身参数，以应对训练-测试分布偏移。
- **Zeroth-Order (ZO) Optimization**：零阶优化，仅通过函数值评估（前向传播）估计梯度进行参数更新，无需反向传播。
- **Random Gradient Estimation (RGE)**：随机梯度估计，标准 ZO 方法，通过对称两点扰动估计梯度，方差随参数维度线性增长。
- **Hessian Matrix**：海森矩阵，损失函数关于参数的二阶偏导矩阵，刻画损失曲面的局部曲率结构。
- **Exponential Moving Average (EMA)**：指数移动平均，用于平滑序列数据的滑动平均策略，此处用于在线估计对角 Hessian。
- **Adapter**：适配器模块，插入预训练 Transformer 层间的轻量参数模块，通过低秩变换实现高效参数微调。
- **Anisotropic Perturbation**：各向异性扰动，指沿不同参数维度以不同强度施加的随机扰动，由曲率信息驱动。
- **Continual Test-Time Adaptation (CTTA)**：持续测试时适配，模型在无重置条件下连续处理多个域偏移样本的适配场景。

## 可复现要素
- **数据集**：ImageNet-C（公开）、ImageNet-R（公开）、ImageNet-V2 Matched-Frequency（公开）、ImageNet-Sketch（公开）。
- **代码开源**：是，GitHub: https://github.com/Hollyming/CAZO。
- **关键超参**：batch size=64，learning rate=0.01，perturbation number k=20（默认），EMA coefficient ν=0.8，perturbation scale ϵ=0.1，adapter downsampling ratio=384，adapter layer=3（ViT-B/16），random seeds={42, 2020, 2025, 1234, 8}。
- **模型**：ViT-B/16，预训练于 ImageNet-21K/ImageNet-1K。
- **硬件**：单卡 NVIDIA H20 GPU。
