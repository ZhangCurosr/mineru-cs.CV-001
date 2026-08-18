---
title: "ENCORE: Efficient Noise Context-Aware Representation for Low-Dose CT Denoising"
source: https://arxiv.org/pdf/2608.10343v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:42:08"
field: "医学图像重建与去噪"
keywords: ["低剂量CT去噪", "噪声上下文感知", "自协方差估计", "动态卷积", "Noise2Noise", "Cornish-Fisher展开", "零样本条件去噪"]
innovations: ["提出基于Cornish-Fisher偏度校正的CT噪声合成方法，替代传统高斯近似以提升低光子计数场景的真实性", "设计自协方差预处理与FlyingConv动态卷积模块，将噪声功率与空间相关性作为稳定先验嵌入去噪网络", "实现零样本条件去噪：通过缩放推理时噪声上下文强度动态控制输出纹理与去噪强度"]
benchmarks: ["Mayo2016 (10%/25% dose)", "Mayo2020 (cross-vendor)", "Tabletop real-world cone-beam CT phantom"]
---

# 论文速读：ENCORE: Efficient Noise Context-Aware Representation for Low-Dose CT Denoising

## 一句话总结
论文提出 ENCORE（Efficient Noise COntext-aware REpresentation）框架，通过显式提取 CT 噪声的局部功率和相关性上下文，并结合自适应卷积模块 FlyingConv，实现低剂量 CT 去噪；同时支持零样本条件下对输出纹理/噪声强度的动态控制，并在整个重建-去噪流水线中通过定制 CUDA 内核实现高效推理。

## 研究问题与动机
1. **CT 噪声的非平稳性与空间相关性**：传统去噪网络假设 i.i.d. 高斯噪声，而 CT 原始投影数据包含泊松量子噪声与高斯电子噪声，且反投影操作会引入复杂的空间相关性，导致标准架构在 CT 图像上表现受限。
2. **已有噪声增强方法（如 NADD）的不足**：NADD 直接拼接随机生成的噪声图作为额外输入，噪声图存在较大的统计方差，缺乏预处理稳定化，难以让网络可靠地捕捉 CT 噪声特性。
3. **缺乏面向 CT 噪声特性的架构设计**：现有工作多聚焦于损失函数或自监督训练策略，而从网络结构层面直接建模 CT 噪声特征的研究较少；静态卷积权重无法适应随解剖区域变化的噪声上下文。
4. **临床配对数据稀缺但物理先验可复用**：N2N 等自监督策略可缓解配对数据需求，但需结合更精确的物理噪声模型（超越简单高斯近似）以提升泛化能力。

## 核心贡献（创新点）
1. **基于 Cornish-Fisher 展开的噪声合成改进**：用偏度校正项替代传统高斯近似，使生成的训练对更贴近真实泊松噪声分布，本质区别在于显式建模了低光子计数下的分布偏度，而非仅匹配方差。
2. **噪声自协方差预处理提取稳定上下文**：将随机噪声图转换为局部噪声功率与相关性的自协方差图（signed-log 归一化），为网络提供稳定、模型友好的噪声统计先验，优于 NADD 直接输入原始噪声图的方案。
3. **FlyingConv 自适应卷积模块**：设计基于解剖特征与自协方差图联合预测的动态卷积核，并按组共享权重以降低显存开销，本质区别在于同时利用噪声上下文与图像内容调制卷积权重，而非仅依赖图像局部特征。
4. **零样本条件去噪能力**：通过缩放推理时噪声上下文强度（调整 $d_{\text{target}}$），可在不重新训练的情况下动态控制去噪强度与纹理保留程度，填补了自监督 CT 去噪中缺乏推理期可控性的空白。
5. **全流程定制 CUDA 内核加速**：针对 FBP 重建、自协方差估计与 FlyingConv 模块重写 GPU 内核，充分利用缓存与纹理内存单元，显著降低端到端延迟，而非仅优化理论 MACs。

## 方法详解
### 1) 噪声合成与训练对生成
- 高剂量投影 $P_{\text{ND}}$ 建模为 $P_{\text{ND}} = \text{Pois}(P_{\text{clean}}) + \mathcal{N}(0, \sigma_e^2)$。
- 目标剂量水平 $d$ 下的低剂量投影：
  $$P_{\text{LD}} \approx d\big(P_{\text{ND}} + a(\text{Pois}(P_{\text{ND}}) - P_{\text{ND}} + b\,\mathcal{N}(0, \sigma_e^2))\big), \quad a=\sqrt{1/d-1},\; b=\sqrt{1/d+1}$$
- 训练对 $(P_{\text{sLD}}, P_{\text{ID}})$ 通过独立噪声采样获得；噪声图 $\mathbf{X}_n \sim \text{FBP}(-\log(P_{\text{Lower}}/P_{\text{LD}}))$ 用于上下文估计。
- **改进点**：将 $\mathcal{N}(0, P)$ 替换为经 Cornish-Fisher 展开推导的偏度校正项 $\mathcal{W}(0, P)$，使生成噪声的偏度与目标泊松分布一致（见附录公式 (7)-(10)）。

### 2) 零样本推理扩展
- 推理时通过目标剂量 $d_{\text{target}} > d$ 缩放噪声强度：
  $$P_{\text{Lower}} = P_{\text{LD}} + \mathcal{W}(0, P_{\text{LD}}(1-r)) + \mathcal{N}(0, \sigma_e^2(1-r^2)), \quad r = d/d_{\text{target}}$$
- $d_{\text{target}} = \infty$ 对应去噪最强（无残留噪声），$d_{\text{target}} = 1$ 对应输出保留正常剂量水平噪声；由此实现零样本纹理调控。

### 3) 自协方差估计预处理
- 对 $N$ 个噪声图 $\mathbf{X}_n$ 计算空间滞后 $p$、窗口大小 $w$ 的自协方差图 $\mathbf{V} \in \mathbb{R}^{p \times p \times H \times W}$：
  $$\mathbf{V}(i,j,x,y) = \frac{1}{Nw^2}\sum_n \sum_{(u,v)\in\Omega} \mathbf{X}_n(x+u, y+v)\cdot\mathbf{X}_n(x+u+i-\lfloor p/2\rfloor, y+v+j-\lfloor p/2\rfloor)$$
- 默认设置 $p=5,\; w=5,\; N=1$；采用 signed-log 归一化 $\text{sign}(\mathbf{V})\cdot\log(1+|\mathbf{V}|)$ 压缩动态范围。

### 4) FlyingConv 模块
- 输入：$4\times4$ 平均池化后的特征图 $\mathbf{F}$ 与自协方差图 $\mathbf{V}$ 拼接。
- Predictor：两个 FasterNet 块 + $2\times2$ 平均池化 + $1\times1$ 卷积，输出自适应核权重张量 $\mathbf{W} \in \mathbb{R}^{k^2 \times C/g \times H/8 \times W/8}$（默认组数 $g=2$）。
- 自适应核与特征图分辨率不匹配时，将插值与加权求和融合到单次执行循环，避免额外显存峰值。
- 末尾接 $1\times1$ 卷积做通道融合；网络中交替使用 FlyingConv 与普通卷积层。

### 5) 损失函数与训练设置
- 因采用 N2N 自监督框架，损失仅使用 MSE；训练 300 epoch，patch 256×256，batch=32，Adam 初始 LR=$2\times10^{-4}$，每 60 epoch 减半。

## 实验与结果
### 数据集
- **Mayo2016**：Siemens Healthineers  scanner，10 患者 1 mm 层厚 NDCT；8 患者 4,773 切片用于训练/验证，2 患者 1,093 切片测试。模拟剂量 $d \in \{0.25, 0.10\}$。
- **Mayo2020**：GE Healthcare Discovery CT750 HD，5 患者共 663 腹部切片，用于跨厂商泛化评估。
- **Tabletop 真实数据**：在体模上使用自建锥束 CT 系统采集，评估真实物理条件鲁棒性。

### 基线方法
- Vanilla（DnCNN/UNet/NAFNet/Uformer 原始配置）
- +NADD（噪声增强深度去噪）
- +COV（噪声图替换为自协方差图）
- +ENCORE（完整框架）

### 主要定量结果（Base UNet）
- **Mayo2016 @10% 剂量**：+ENCORE 达到 **PSNR 24.07 dB / SSIM 0.8119 / AUHOC 0.3985**，显著优于 Vanilla（24.21/0.8153/0.3913）与 +NADD（24.31/0.8168/0.3869），且统计检验 $p<0.01$。
- **Mayo2020 @10% 剂量**：+ENCORE 达到 **PSNR 25.94 dB / SSIM 0.8610 / AUHOC 0.3959**，优于所有基线。
- **MACs 效率**：Base UNet+ENCORE 仅需 43.17 G MACs，显著低于 +COV（69.47 G）与 NAFNet/Uformer 同级配置。
- **延迟**：端到端延迟（含 FBP）约 10.14 ms（Base UNet+ENCORE），与 +COV（10.32 ms）相当，但 PSNR 更高。

### 核心结论
- 自协方差预处理（+COV）已明显优于 NADD；FlyingConv 进一步提升去噪质量并降低计算量。
- 超低剂量（10%）与轻量级（Lite）配置下提升更为显著。
- 通过调整 $d_{\text{target}}$ 可在 PSNR/AUHOC 与 SSIM 间动态权衡，解决默认设置下 SSIM 略低的平滑-细节权衡问题。

### 加速效果
- 定制 CUDA 内核相比朴素实现：FBP 加速 **4.8×**，协方差估计加速 **2.9×**，FlyingConv 去噪模块加速 **3.2×**；显存占用分别降低 **4.0×** 与 **5.6×**。

## 相关工作脉络
1. **Noise2Noise (Lehtinen et al., 2018)**：无配对数据的自监督去噪基础方法；本文在其框架上引入物理噪声模型改进，提升噪声合成的真实性。
2. **Half2Half (Yuan et al., 2020)**：单扫描生成统计独立训练对；本文采用类似 N2N 思路但改用 Cornish-Fisher 偏度校正替代高斯近似。
3. **Wang & Wang (2022)**：基于高斯近似的任意剂量模拟与独立噪声对生成；本文指出其在低光子计数/高密度骨结构下失效，并以偏度校正项修正。
4. **NADD (Kristof et al., 2025)**：将噪声功率与相关性图作为额外输入通道；本文认为其直接使用随机噪声图导致训练不稳定，提出自协方差预处理以提供模型友好型上下文。
5. **MalleConv (Jiang et al., 2022)**：动态卷积根据图像上下文变化权重；本文 FlyingConv 在此基础上进一步融合噪声自协方差信息，面向 CT 特有噪声模式。
6. **AUHOC (Kc et al., 2026)**：频率域 patch 相似度度量结构幻觉；本文将其作为关键评估指标，强调去噪结果不应引入虚假结构。

## 局限性与未来方向
1. **训练仅依赖模拟数据**：端到端泛化到真实临床扫描仍需进一步验证；真实散射、束硬伪影等物理效应未在噪声模型中完全建模。
2. **SSIM 在特定情形下略低**：默认 $d_{\text{target}}=\infty$ 在均匀解剖区域（如肝脏）易过度平滑；需依赖零样本调控或调参以平衡 SSIM。
3. **延迟优化受限于显存带宽**：FlyingConv 动态权重调制增加显存访问开销，MACs 减少未完全转化为延迟下降，需更高效的后端实现。
4. **自协方差估计参数需手工设定**：当前默认 $p=5, w=5, N=1$，对不同扫描协议/剂量水平的自适应能力未充分探索。
5. **未来方向**：将散射、束硬化等物理因素纳入噪声模型；探索端到端可微重建-去噪联合优化；适配更高带宽 GPU 架构以释放 FlyingConv 效率潜力。

## 研究启发与可借鉴点
1. **物理噪声模型的显式偏度校正**：Cornish-Fisher 展开可将高阶统计量注入合成噪声，适用于其他成像模态（如 MRI、超声）中对非高斯噪声的建模。
2. **噪声上下文作为稳定先验**：将随机噪声增强转为局部自协方差估计，为网络提供确定性统计信号，可迁移至其他非平稳噪声去除任务。
3. **动态卷积与物理先验融合**：FlyingConv 将噪声统计图与图像特征共同输入 Predictor，为"模型架构 × 物理域"交叉设计提供了可复用的模块范式。
4. **零样本推理期可控去噪**：通过单一缩放因子 $d_{\text{target}}$ 实现纹理/噪声级别的交互式调节，无需额外训练，对临床工作流中不同诊断需求（如骨骼 vs. 软组织）极具实用价值。
5. **全流程定制内核的评测规范**：本文不仅报告 MACs，更测量含 FBP 的端到端延迟并开源定制 CUDA 代码，为影像重建任务的性能基准提供了可复现方法论。

## 关键术语表
**Noise2Noise (N2N)**：无需干净参考图、仅用多幅含噪图像互相监督训练的自监督去噪范式。
**Cornish-Fisher 展开**：通过对正态变量施加多项式变换来匹配目标分布偏度与峰度的数学工具。
**自协方差图 (Autocovariance map)**：描述局部噪声在不同空间滞后下的相关性结构，作为噪声上下文先验。
**FlyingConv**：本文提出的动态卷积模块，根据图像特征与噪声上下文图实时生成卷积核权重。
**AUHOC**：基于频率域 patch 相似度的结构幻觉评估指标，越低表示幻觉越少。
**signed-log 归一化**：$\text{sign}(x)\cdot\log(1+|x|)$，用于压缩自协方差图的动态范围并稳定训练。
**NADD (Noise-Augmented Deep Denoising)**：将合成的噪声图直接拼接为额外输入通道的 CT 去噪方法。
**MACs**：Multiply-Accumulate Operations，衡量神经网络理论计算量的常用单位。

## 可复现要素
- **数据集**：Mayo2016（公开）、Mayo2020（公开）、Tabletop 自建体模数据（未公开）。
- **代码**：全流程开源，仓库地址 https://github.com/minwoo-yu/ENCORE.git。
- **关键超参**：$p=5$（空间滞后 patch 大小），$w=5$（聚合窗口大小），$N=1$（噪声图数量），$g=2$（FlyingConv 分组数），$d_{\text{target}}=\infty$（默认零样本设置），batch size=32，patch=256×256，epoch=300，初始 LR=$2\times10^{-4}$，每 60 epoch 减半。
- **硬件**：NVIDIA RTX A6000 GPU。
- **定制内核**：FBP、协方差估计与 FlyingConv 均采用 C++/CUDA 实现，已集成至开源仓库。
