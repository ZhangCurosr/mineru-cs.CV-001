---
title: "Curvature-Aware-Zeroth-Order-Optimization-for-Memory-Efficie"
source: https://arxiv.org/pdf/2608.12279v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 10:44:16"
field: "边缘设备自适应学习"
keywords: ["Test-Time Adaptation", "Zeroth-Order Optimization", "Memory-Efficient", "Hessian Low-Rank", "BP-Free"]
innovations: ["揭示TTA过程中Hessian的低秩与缓变结构", "提出基于EMA对角Hessian估计的曲率感知各向异性扰动采样", "在减少70%内存开销的同时实现SOTA适配精度"]
benchmarks: ["ImageNet-C", "ImageNet-R", "ImageNet-V2", "ImageNet-Sketch"]
---

# 论文速读：Curvature-Aware Zeroth-Order Optimization for Memory-Efficient Test-Time Adaptation

## 一句话总结
本文针对边缘设备测试时适应（TTA）场景，提出**CAZO**（曲率感知零阶优化）方法，通过发现并利用TTA过程中损失函数Hessian的低秩与缓变特性，将零阶优化的梯度估计方差大幅降低，在减少70%内存开销的同时实现了SOTA的适配精度。

## 研究问题与动机
1. **现有TTA方法的内存瓶颈**：主流TTA方法依赖反向传播（BP）微调，在边缘设备上内存开销过大（如CoTTA需17,773 MB），难以部署于资源受限场景。
2. **零阶方法的高方差缺陷**：零阶优化（ZO）仅需前向计算，内存友好，但梯度估计方差随参数维度线性增长（$O(d/k)$），导致收敛慢、精度差，无法直接用于高精度TTA。
3. **缺乏对损失曲率的系统性分析**：现有ZO方法采用各向同性随机扰动，未利用TTA损失地形中的结构信息（如低秩Hessian），导致采样效率低下。

## 核心贡献（创新点）
1. **揭示TTA中Hessian的低秩与缓变结构**：实证发现适配过程中Hessian特征值呈显著低秩分布（Top 20特征值解释96%+方差），且主曲率子空间在适配步骤间保持稳定（投影比~0.9）。
2. **提出CAZO曲率感知零阶优化框架**：通过滑动平均（EMA）估计对角Hessian，构建各向异性扰动协方差矩阵，使采样在高曲率方向衰减、低曲率方向放大，有效降低梯度估计方差。
3. **理论保证与高效实现**：在标准光滑假设下证明CAZO达到$O(1/\sqrt{T})$收敛率，且使用对角近似避免高维逆矩阵计算，内存开销降低70%+。
4. **全面SOTA性能**：在ImageNet-C、ImageNet-R/V2/Sketch等基准上超越所有BP-free和BP-based基线，其中ImageNet-C（severity-5）达69.0%平均准确率，较ZOA/FOA分别提升+1.5%/+3.2%。

## 方法详解
**整体框架**：冻结预训练ViT-B/16权重，仅在ViT的第3层插入轻量adapter模块，通过CAZO仅用前向计算更新adapter参数。

**关键设计**：
1. **对角Hessian估计**：使用滑动平均EMA跟踪梯度元素级平方来近似对角Hessian：
   - $D_t = (1-\nu)D_{t-1} + \nu \hat{g}^2(\theta_{t-1})$
   - $\tilde{H}_t = \text{diag}\left(\frac{D_t}{1-(1-\nu)^t}\right)$
2. **各向异性扰动采样**：采样方向服从$\mathcal{N}(0, \tilde{H}_t^{-1})$，实现"沿高曲率方向缩小、沿低曲率方向放大"的自适应采样。
3. **双点对称梯度估计**：
   - $\hat{g}(\theta_t) = \frac{1}{K}\sum_{i=1}^{K}\frac{\mathcal{L}(\theta_t+\epsilon u_i) - \mathcal{L}(\theta_t-\epsilon u_i)}{2\epsilon}u_i$
   - 其中$u_i \sim \mathcal{N}(0, \tilde{H}_t^{-1})$
4. **复合损失函数**：结合无监督熵损失（熵最小化）与特征对齐MSE损失（使用干净数据特征统计）。
5. **收敛性保证**：在L-光滑和数据方差假设下，证明CAZO达到$O(1/\sqrt{T})$收敛率，且常数项依赖于曲率界$\beta_l, \beta_u$。

## 实验与结果
**数据集与模型**：ImageNet-C（15类腐蚀，severity-5）、ImageNet-R/V2/Sketch；ViT-B/16预训练模型。

**主要结果**：
| 设置 | 方法 | 平均准确率 | 内存(MB) | 时间(s) |
|------|------|-----------|----------|---------|
| ImageNet-C (reset) | CAZO | **69.0%** | 1,695 | 3,127 |
| ImageNet-C (reset) | FOA | 65.8% | 1,553 | 2,885 |
| ImageNet-C (reset) | ZOA | 67.5% | 1,660 | 398 |
| ImageNet-C (reset) | TENT | 59.8% | 6,404 | 210 |
| ImageNet-C (CTTA) | CAZO | **65.3%** | - | - |
| ImageNet-C (CTTA) | LCoTTA | 62.3% | - | - |
| ImageNet-R/V2/Sketch | CAZO | 63.5% | - | - |

**关键发现**：
- CAZO比最佳BP-free方法FOA/A ZOA分别高+3.2%/+1.5%，同时比BP方法TENT/SAR等高+6.3%/+7.1%。
- 内存开销从TENT的6,404 MB降至1,695 MB（**减少约73%**）。
- 在8-bit/6-bit量化下仍保持领先（67.8%/61.2%），证明对边缘部署的有效性。
- 仅k=2扰动即可达65.2%准确率，k=20时达69.0%。

## 相关工作脉络
1. **TENT [51]**：最早基于BP的熵最小化TTA方法，更新BN统计量；CAZO在不使用BP的前提下实现更高精度。
2. **FOA [40]**：基于CMA-ES的进化策略BP-free TTA，CAZO相比其更轻量且精度更高（+3.2%）。
3. **ZOA [7]**：同时期提出的ZO TTA方法，面向量化模型，强调领域知识管理；CAZO专注于曲率感知采样以提升效率。
4. **MeZO [33]**：LLM微调中的零阶方法，证明收敛取决于内在有效维度；CAZO进一步利用Hessian低秩结构指导采样。
5. **LCoTTA [11]**：发现熵最小化梯度的低维子空间并投影；CAZO从Hessian角度发现相似性质并用于ZO采样。
6. **熵最小化类TTA**（SHOT, EATA, SAR等）：依赖BP且内存开销大，CAZO以零阶方式实现更好精度。

## 局限性与未来方向
1. **Hessian近似为对角形式**：忽略特征值方向相关性，可能在高维场景中精度受限；未来可探索低秩全矩阵近似。
2. **实验局限于视觉Transformer**：未验证在卷积网络或大语言模型上的泛化性。
3. **扰动数量k与精度的权衡**：k越大精度越高但耗时增加，实际部署需根据硬件约束平衡。
4. **适配器位置固定**：当前默认layer-3，未深入探索多层适配或不同架构下的最优位置。
5. **未考虑动态步长/学习率调度**：EMA系数ν固定为0.8，自适应调度可能进一步提升性能。

## 研究启发与可借鉴点
1. **Hessian低秩性的广泛利用**：本研究揭示的"TTA过程中Hessian低秩且缓变"性质可推广至其他优化场景（如联邦学习、在线学习），作为降维或加速收敛的先验。
2. **对角Hessian的滑动平均估计**：用EMA跟踪$\hat{g}^2$作为曲率代理的轻量方案，可复用于其他零阶优化任务（如黑盒对抗攻击、超参优化）。
3. **Adapter位置选择原则**：ViT中early layer（如layer-3）更适合快速域对齐，这一经验可迁移至DeiT、Swin等其他Transformer变体。
4. **BP-free高维优化的方差缩减策略**：曲率感知各向异性采样思路可与Sparsified ZO、SignZO等技术结合，进一步降低通信/计算开销。
5. **内存-精度权衡分析框架**：表4展示了完整的复杂度对比（FP次数、内存、时间），为后续工作提供可复用的评测范式。

## 关键术语表
**Test-Time Adaptation (TTA)**：测试时适应，指模型在推理阶段仅用无标签测试数据在线适配，无需重新训练。

**Zeroth-Order (ZO) Optimization**：零阶优化，仅通过函数估值（前向计算）估计梯度，无需反向传播，适合内存受限场景。

**Hessian Matrix**：海森矩阵，损失函数关于参数的二阶偏导矩阵，描述损失地形的局部曲率。

**Exponential Moving Average (EMA)**：指数移动平均，用于平滑序列数据的加权滚动平均，此处用于跟踪对角Hessian估计。

**Anisotropic Perturbation**：各向异性扰动，根据不同方向的曲率大小调整采样方差，区别于各向同性的均匀扰动。

**Adapter**：适配器，插入预训练模型中的轻量可训练模块，用于高效参数微调。

**Continual TTA (CTTA)**：持续测试时适应，模型在无重置情况下连续处理多个域偏移数据流的设定。

**Effective Rank**：有效秩，基于特征值分布的Shannon熵定义的"Hessian实质非零维度数"。

## 可复现要素
- **数据集**：ImageNet-C（公开）、ImageNet-R/V2/Sketch（公开）；训练数据ImageNet（公开）。
- **代码**：已开源，https://github.com/Hollyming/CAZO。
- **模型**：ViT-B/16（公开权重）。
- **关键超参**：batch_size=64，learning_rate=0.01，perturbation数k=20（默认），EMA系数ν=0.8，扰动尺度ε=0.1，adapter下采样比率=384，adapter位置=layer-3。
- **硬件**：单卡NVIDIA H20 GPU。
