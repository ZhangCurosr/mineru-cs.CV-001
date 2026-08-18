---
title: "ENCORE: Efficient Noise Context-Aware Representation for Low-Dose CT Denoising"
source: https://arxiv.org/pdf/2608.10343v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:42:09"
field: "医学图像去噪"
keywords: ["低剂量CT", "去噪", "噪声上下文", "自监督学习", "自适应卷积", "FlyingConv"]
innovations: ["将噪声自协方差图作为模型友好的上下文输入，替代原始随机噪声图", "设计FlyingConv模块，根据局部解剖和噪声上下文动态生成卷积权重", "引入Cornish-Fisher偏度校正替代高斯近似，提升低剂量物理一致性"]
benchmarks: ["Mayo2016", "Mayo2020"]
---

# 论文速读：ENCORE: Efficient Noise Context-Aware Representation for Low-Dose CT Denoising

## 一句话总结
本文提出 ENCORE 框架，通过将 CT 噪声的局部自协方差（噪声功率与空间相关性）显式建模并嵌入到自适应卷积模块中，在保持计算效率的同时显著提升了低剂量 CT 去噪质量，并支持推理时的零样本条件去噪控制。

## 研究问题与动机
1. **CT 噪声非 i.i.d. 特性被忽视**：现有主流去噪模型多采用自然图像通用架构，假设噪声为独立同分布（i.i.d.），而 CT 原始投影包含泊松量子噪声与高斯电子噪声，经反投影后产生强空间相关性和非平稳噪声，导致标准模型性能受限。
2. **噪声增强方法存在统计方差问题**：NADD 等方法直接将随机生成的噪声图拼接到输入，每次仿真噪声分布的波动未被稳定化，导致网络难以可靠地学习 CT 噪声统计特性。
3. **高斯近似在低剂量下失效**：传统训练对（如 Half2Half、[5]）用高斯分布近似泊松量子噪声，在光子饥饿（光子数低）场景下产生统计偏差，影响模型物理一致性。
4. **缺乏灵活的去噪控制能力**：现有方法训练后去噪强度固定，难以在推理时根据临床需求动态调节输出纹理（降噪 vs. 细节保留）。

## 核心贡献（创新点）
1. **重新设计基于 Cornish-Fisher 展开的噪声合成流程**：将训练配对生成中的高斯近似替换为偏度校正项 $\mathcal{W}$，使合成噪声更符合实际泊松统计，尤其在低光子数和高衰减区域（如骨骼）显著提升 SSIM。
2. **噪声自协方差预处理方法**：从单张噪声图估计局部噪声功率与相关性的自协方差图 $\mathbf{V}$，并以 signed-log 归一化压缩动态范围，相比 NADD 的原始噪声图提供更稳定、更易被模型利用的上下文信号。
3. **FlyingConv 自适应卷积模块**：设计在线权重调制卷积，融合解剖特征与自协方差图动态生成空间自适应核，相比 NADD 仅扩展输入通道的方式，从架构层面解锁噪声上下文的潜在能力，同时通过分组策略和插值-卷积融合降低显存访问开销。
4. **零样本条件去噪能力**：通过缩放噪声图强度（参数 $d_{\text{target}}$）在推理时无需重新训练即可动态调节输出残留噪声水平，灵活平衡 PSNR（像素精度）与 SSIM（纹理保留）。
5. **全栈 CUDA 自定义内核优化**：对 FBP 重建、自协方差估计和 FlyingConv 均定制加速内核，相比原生实现实现 2.9×–4.8× 吞吐量提升。

## 方法详解
1. **噪声合成与训练对生成**：
   - 高剂量投影：$P_{\mathrm{ND}} = \mathrm{Pois}(P_{\mathrm{clean}}) + \mathcal{N}(0, \sigma_{\mathrm{e}}^2)$。
   - 目标低剂量 $d$ 下的投影生成：$P_{\mathrm{LD}} = d(P_{\mathrm{ND}} + a(\mathrm{Pois}(P_{\mathrm{ND}}) - P_{\mathrm{ND}} + b\mathcal{N}(0, \sigma_{\mathrm{e}}^2)))$，其中 $a = \sqrt{1/d - 1}, b = \sqrt{1/d + 1}$。
   - 使用 Cornish-Fisher 展开将泊松噪声的高斯项替换为 $\mathcal{W}(0, P)$，保留偏度信息，生成统计独立的 $P_{\mathrm{sLD}}$ 和 $P_{\mathrm{ID}}$ 对，满足 Noise2Noise (N2N) 训练要求。
   - 噪声图生成：$X_n \sim \mathrm{FBP}(-\log(P_{\mathrm{Lower}}/P_{\mathrm{LD}}))$，其中 $P_{\mathrm{Lower}} = P_{\mathrm{LD}} + \mathcal{W}(0, P_{\mathrm{LD}}) + \mathcal{N}(0, \sigma_{\mathrm{e}}^2)$。
   - 推理时零样本控制：引入缩放因子 $r = d / d_{\text{target}}$，令 $P_{\text{Lower}} = P_{\text{LD}} + \mathcal{W}(0, P_{\text{LD}}(1-r)) + \mathcal{N}(0, \sigma_{\text{e}}^2(1-r^2))$，调节输出噪声强度。

2. **噪声自协方差估计**：
   - 计算噪声图 $\mathbf{X}_n$ 的自协方差图 $\mathbf{V} \in \mathbb{R}^{p \times p \times H \times W}$：
     $$\mathbf{V}(i,j,x,y) = \frac{1}{Nw^2}\sum_{n=1}^{N}\sum_{(u,v)\in\Omega}\mathbf{X}_n(x+u, y+v)\cdot\mathbf{X}_n(x+u+\lfloor p/2\rfloor, y+v+\lfloor p/2\rfloor)$$
   - 默认参数：$p=5, w=5, N=1$，利用 CT 噪声在毫米级 ROI 内近似平稳的特性。
   - 采用 signed-log 归一化：$\mathrm{sign}(\mathbf{V})\cdot\log(1+|\mathbf{V}|)$ 压缩动态范围。

3. **FlyingConv 模块**：
   - 将 $4\times4$ 平均池化的特征图 $\mathbf{F}$ 与自协方差图 $\mathbf{V}$ 融合，经轻量 Predictor（两个 FasterNet 块 + $2\times2$ 平均池化 + $1\times1$ 卷积）生成自适应核权重 $\mathbf{W}$。
   - 采用分组深度可分离卷积（$g=2$ 默认），每个预测权重在 $g$ 个连续通道上共享，降低显存访问开销。
   - 将插值与卷积融合为单一执行循环，利用 GPU 纹理内存单元（TMU）加速，减少显存带宽瓶颈。
   - 网络中 FlyingConv 与标准卷积交替堆叠，以兼顾空间自适应性与通道间交互。

4. **全栈硬件加速**：
   - 基于 LEAP 库定制 FBP 重建 CUDA 内核，最大化 GPU cache 利用率，减少 FBP 与后续模块间的显存传输。
   - 自协方差估计与 FlyingConv 模块均通过自定义内核实现，显著提升吞吐量。

## 实验与结果
1. **数据集**：
   - **Mayo2016**：Siemens 扫描仪数据，4773 片训练/验证（8 例），1093 片测试（2 例），模拟剂量 $d \in \{0.25, 0.1\}$。
   - **Mayo2020**：GE Healthcare 扫描仪数据，663 片腹部扫描，用于跨厂商泛化评估。
   - **Tabletop 真机数据**：自研锥形束 CT 系统扫描体模，验证方法在物理成像条件下的鲁棒性。

2. **评估基线**：DnCNN、UNet、NAFNet、Uformer 的 Vanilla、+NADD、+COV（自协方差预处理）、+ENCORE 配置；MACs 与真实推理延迟在 NVIDIA RTX A6000 上测量。

3. **主要结果**（UNet Base，Mayo2016 10% 剂量）：
   - **+ENCORE**：PSNR 24.07 / SSIM 0.8119 / AUHOC 0.3985，MACs 43.17G，延迟 10.14ms。
   - 相比 Vanilla UNet（PSNR 24.27 / SSIM 0.8169）在低剂量下 SSIM 略低（因默认 $d_{\text{target}}=\infty$ 过平滑），但 AUHOC 更低（伪影更少）；通过调节 $d_{\text{target}}$ 可恢复 SSIM。
   - 相比 +NADD（PSNR 25.57 / SSIM 0.8552，Base DnCNN）在同等架构下提升约 0.3-0.5 dB PSNR，MACs 降低 35%。
   - **Lite 变体优势更显著**：UNet Lite +ENCORE PSNR 24.38，MACs 仅 17.57G，延迟 5.93ms，性价比最优。
   - 统计检验（Wilcoxon signed-rank，Holm 校正，$p<0.01$）确认 +ENCORE 在多数指标上显著优于所有变体。

4. ** Ablation 关键结论**：
   - Cornish-Fisher 偏度校正在高衰减切片（骨骼）SSIM 上提升 0.0014。
   - 零样本控制：$d_{\text{target}} \in [0.75, 1.0]$ 时 SSIM 改善，输出纹理更接近临床 NDCT。
   - 自协方差参数：$w=5, p=5, N=1$ 为最优/够用配置；$N$ 增大仅增加延迟。
   - FlyingConv 分组大小 $g=2$ 在效率与性能间取得最佳平衡。
   - 自定义 CUDA 内核：FBP 加速 4.8×，协方差估计加速 2.9×，FlyingConv 加速 3.2×。

## 相关工作脉络
1. **Noise2Noise (Lehtinen et al.)**：提出无需干净参考的训练框架，本文沿用其思想但引入物理噪声建模替代纯数据驱动。
2. **NADD (Kristof et al.)**：首个将噪声图拼接至 CT 去噪输入的方法，本文指出其原始噪声图的随机波动问题，并提出预处理的自协方差图与自适应架构作为根本性改进。
3. **Half2Half (Yuan et al.)**：从单次 CT 生成独立训练对，本文与其目标一致但通过 Cornish-Fisher 展开改进其高斯近似带来的统计偏差。
4. **MalleConv (Jiang et al.)**：动态卷积权重自适应局部图像内容，本文扩展此思想，将权重调制信号从纯图像上下文扩展至噪声上下文（解剖+物理）。
5. **Gaussian Poisson 近似 (Wang & Wang, SPIE 2022)**：本文的 baseline 噪声合成方法，指出其在低剂量和高衰减区域失效，引入偏度校正。

## 局限性与未来方向
1. **噪声模型简化**：当前估计仅考虑量子噪声与电子噪声，未建模散射（scatter）和束硬化（beam hardening）等物理效应，导致仿真与真实数据间存在分布偏移。
2. **SSIM 权衡问题**：默认 $d_{\text{target}}=\infty$ 设置在均匀解剖区域（如肝脏）出现过平滑导致 SSIM 下降，需依赖零样本控制调节，未能端到端自动平衡。
3. **MACs 与延迟不一致**：FlyingConv 虽减少 MACs，但动态权重生成带来显存带宽瓶颈，实际延迟优势未能完全释放，依赖更高带宽 GPU 或进一步优化内存访问模式。
4. **仅模拟数据训练**：方法在模拟数据上训练，虽在真机数据上展现鲁棒性，但未验证跨中心、跨扫描协议的大规模泛化能力。

## 研究启发与可借鉴点
1. **噪声上下文先验的显式建模**：将物理噪声统计（自协方差）作为中间表示接入网络，比单纯扩展输入通道更高效，可迁移至其他具有非平稳噪声的成像模态（如超声、MRI）。
2. **零样本条件控制设计**：通过单一超参 $d_{\text{target}}$ 实现推理时去噪强度的灵活调节，无需多模型或微调，为医学影像后处理提供实用工具。
3. **Cornish-Fisher 偏度校正思路**：当分布近似失效时，用数学展开保留高阶统计量（偏度），比换更复杂模型更轻量且物理可解释，值得在其他逆问题中探索。
4. **全栈自定义内核优化实践**：从重建到推理的端到端 CUDA 定制，揭示了理论计算复杂度与实际延迟的差异来源（显存带宽），为工程导向研究提供标杆。
5. **FlyingConv 模块化设计**：将自适应核生成与插值融合为单次 GPU 操作，避免中间显存分配，可作为通用模块集成到后续变体中。

## 关键术语表
**ENCORE**：Efficient Noise COntext-aware REpresentation，本文提出的低剂量 CT 去噪框架，显式建模并整合噪声上下文信息。
**Noise2Noise (N2N)**：无需干净参考，通过两个不同噪声实现的配对进行自监督训练的方法。
**自协方差图 (Autocovariance Map)**：描述噪声图局部空间相关性和功率的预计算地图，作为模型友好的噪声上下文输入。
**FlyingConv**：在线权重调制卷积模块，根据局部解剖和噪声上下文动态生成卷积核权重。
**Cornish-Fisher 展开**：通过偏度校正将高斯变量转换为匹配目标分布（泊松）的数学技巧。
**零样本条件去噪 (Zero-shot Conditional Denoising)**：推理时通过缩放噪声图强度动态控制输出去噪强度，无需重新训练。
**AUHOC**：Hallucination Operating Characteristic 曲线下面积，衡量结构幻觉程度的频率域评估指标，越低越好。
**MACs**：Multiply-Accumulate Operations，衡量模型理论计算复杂度的指标。

## 可复现要素
- **数据集**：Mayo2016（公开）、Mayo2020（公开）、Tabletop 真机数据（自采集）
- **代码**：已开源，https://github.com/minwoo-yu/ENCORE.git
- **关键超参**：$p=5, w=5, N=1, g=2$；学习率 $2\times10^{-4}$，每 60 epoch 减半；Batch size 32；256×256 patch；300 epochs
- **硬件**：NVIDIA RTX A6000 GPU
