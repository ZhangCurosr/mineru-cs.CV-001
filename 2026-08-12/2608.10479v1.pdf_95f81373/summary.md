---
title: "Bridging Event Streams and DiT: Event-Guided Video Frame Interpolation"
source: https://arxiv.org/pdf/2608.10479v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:30:50"
field: "事件视觉与视频生成"
keywords: ["视频帧插值", "事件相机", "扩散Transformer", "IWE", "光流对齐", "适配器微调"]
innovations: ["提出轻量适配器将IWE与双向稀疏光流注入预训练DiT，避免端到端重型训练", "通过CMax将异步事件转化为LDM兼容的边缘-运动联合控制信号", "构建迄今最大规模合成事件-视频数据集EvPexels并开源"]
benchmarks: ["BS-ERGB", "DAVIS", "Pexels"]
---

# 论文速读：Bridging Event Streams and DiT: Event-Guided Video Frame Interpolation

## 一句话总结
本文提出一种基于适配器的轻量级框架，将事件相机流转化为 IWE（Image Warped Events）和双向稀疏光流，注入预训练的 DiT（Diffusion Transformer）视频插值模型，显著减少大时间间隔与复杂运动场景下的运动模糊、结构失真与时间不一致伪影，在多个真实/合成基准上取得 SOTA 性能。

## 研究问题与动机
- **核心问题**：基于 Latent Diffusion Model（LDM）的视频帧插值（VFI）在处理大时间间隔和复杂运动时，仍易产生运动模糊、对象形变、身份不一致和结构伪影，尤其在中间帧退化明显。
- **现有帧基方法局限**：GI、FCVG、Wan2.1 FLF2V 等仅依赖首尾帧引导生成，缺乏细粒度运动线索，难以维持时序连贯性。
- **事件数据集成难点**：事件相机虽能提供高时间分辨率、大动态范围的异步像素级亮度变化信号，但其稀疏/异步特性与主流生成模型所用的密集网格表示不兼容，且大规模配对 event-video 数据集稀缺，端到端训练不可行。
- **已有事件辅助方法的不足**：VDM-EVFI 等采用 ControlNet 式重型设计，将原始事件转成 voxel grid 并从头训练大量参数，计算开销大且泛化受限。

## 核心贡献（创新点）
1. **提出适配器框架而非端到端重写**：在预训练 DiT（Wan2.1 FLF2V）基础上仅引入轻量 IWE 编码器和流对齐融合适配器，配合 LoRA 微调，不改动主干架构即可注入事件引导信号。
2. **桥接事件视觉与 LDM 控制信号**：利用对比最大化（CMax）从事件流同时提取双向稀疏光流和 IWE，将异步稀疏事件转化为与扩散模型控制信号（边缘图、光流场）高度兼容的空间-时序对齐表示。
3. **构建大规模合成数据集 EvPexels**：基于 Pexels 视频与 Vid2e 模拟器生成 1,100 场景、约 39 万 RGB 帧的合成 event-video 数据，为事件感知插值模型训练提供迄今最大规模的专用数据集，并将开源。
4. **全面的 SOTA 验证**：在 BS-ERGB（真实）及 DAVIS、Pexels（合成）三个基准的 ×24 插值设置下，感知指标（LPIPS/FID/FVD）全面领先，综合性能最优。

## 方法详解
- **基础模型**：采用 Wan2.1 开源的 FLF2V（DiT 架构）作为视频扩散主干，输入首尾帧 $I_0, I_1$，在潜空间生成固定长度 81 帧视频。
- **事件表征提取**：对 $I_0$ 与 $I_1$ 之间的事件流按时间分段，对每段 $[k-1, k]$ 和 $[k+1, k]$ 分别应用 CMax 方法 [24]，得到稀疏前向/后向光流 $\mathbf{f}_{k-1 \to k}, \mathbf{f}_{k+1 \to k}$ 及对应的 IWE 图 $\mathcal{W}^{k-1\to k}, \mathcal{W}^{k+1\to k}$。
- **IWE 空间条件注入**：设计轻量 3D CNN 编码器将 IWE 编码为特征 $\mathbf{F}_{\mathcal{W}}^k$，通过逐元素加法注入输入潜表示；为适配新模态，对所有 DiT 块施加 LoRA 微调。
- **流对齐与融合适配器**：在选定的 DiT 块（论文选取前 2 个块）之前，将 patch token 重排为帧级特征，利用光流 warp 相邻帧特征 $\mathbf{F}_{x_t}^{k-1}, \mathbf{F}_{x_t}^{k+1}$ 对齐到当前帧 $k$，再经卷积融合网络 $G(\cdot)$ 聚合，残差加回原 DiT 特征，增强时空一致性。
- **训练细节**：学习率 $1\times10^{-4}$，输入 resize 至 832×480，8×NVIDIA A800 GPU，全局 batch size 8，共 4,000 步；光流注入模块仅作用于子集 DiT 块以平衡效率与性能。

## 实验与结果
- **数据集**：训练使用真实 BS-ERGB（48 clips, 970×625@28fps）+ 合成 EvPexels（1,100 clips, 389,761 frames, 704×480）；测试在 BS-ERGB test（26 clips）、DAVIS（50 clips, 25 frames）、Pexels（30 clips, 25 frames）。
- **评估设置**：×24 插值，指标包括 PSNR、SSIM、LPIPS、FID、FVD。
- **主要结果**：
  - **BS-ERGB**：LPIPS 0.132（SOTA）、FID 8.168（SOTA）、FVD 117.368（SOTA）；PSNR 23.261（第三）、SSIM 0.704（第二）。生成式方法在感知质量上显著优于传统事件方法（如 CBMNet-Large PSNR 25.306 但 LPIPS 0.169、FVD 228.753）。
  - **DAVIS**：PSNR 25.544、SSIM 0.799、LPIPS 0.115、FID 13.367、FVD 158.557，五项指标均最佳，超越 VDM-EVFI-Wan2.1（LPIPS 0.123、FVD 165.236）。
  - **Pexels**：PSNR 29.089、SSIM 0.858、LPIPS 0.080、FID 16.319、FVD 151.345，同样五项全优，相对次优方法提升明显。
- **消融结论**：
  - 完整模型 vs 移除 IWE/flow：PSNR 23.072 vs 17.390，LPIPS 0.126 vs 0.241，证明两者缺一不可。
  - Flow warping 优于直接拼接输入：warping 版 LPIPS 0.126 vs 直接输入 0.126（FVD 121.6 vs 123.3），显式时序对齐更有效。
  - 注入位置：前两块利于重建指标（PSNR/SSIM），后两块利于感知指标，本文取前者保质量。
  - 事件表征对比：IWE+Flow 显著优于 Edge-based（CUBE）和 Event Voxel Stack（VDM-EVFI）方案。

## 相关工作脉络
- **传统 VFI**：RIFE、Time Lens、CBMNet-Large、TimeLens-XL 依赖事件提供精确运动估计，但在场景剧变（新对象出现）时仍显不足，且像素级优化导致感知质量受限。
- **扩散 VFI**：TRF、GI、ViBiD、FCVG、Wan2.1 FLF2V 利用预训练 I2V 模型进行插值，无需复杂定制架构，但未充分利用事件提供的细粒度时序线索。
- **事件辅助扩散 VFI**：VDM-EVFI [4] 采用 ControlNet 风格，将事件转为 voxel grid 并重型训练；本文本质差异在于将事件转换为与 DiT 原生控制信号兼容的 IWE+Flow，通过适配器轻量注入，避免端到端事件中心训练。
- **事件-图像对齐技术**：CMax [24,26] 与 edge-based IWE 生成 [12] 为本方法的事件表征提供理论基础，桥接了事件视觉与经典 diffusion control signal。
- **定位差异**：本文聚焦"预训练 DiT + 轻量适配器 + 事件语义转译"范式，在保持生成模型强大先验的同时引入事件运动先验，兼顾效率与质量。

## 局限性与未来方向
- 光流与 IWE 的精度依赖 CMax 质量，在极低纹理/快速非刚性形变场景下估计可能退化，进而影响 adapter 效果。
- 仅在前 2 个 DiT 块注入流对齐信息，未探索全层或多尺度注入策略，可能存在信息瓶颈。
- EvPexels 为合成数据，真实事件-视频配对数据仍稀缺，域间隙可能限制真实场景泛化。
- 当前固定 81 帧输出长度，对超长序列或变长插值的扩展性未验证。
- 未来可在实时事件流在线处理、跨域自适应、多事件传感器融合、以及延伸至视频编辑/补全等下游任务方面展开。

## 研究启发与可借鉴点
- **适配器范式**：不改主干、仅插入轻量模块（编码器+warping+fuse）+ LoRA 微调的策略，可迁移至其他预训练生成模型的条件增强任务（如文生视频、图像编辑）。
- **IWE 作为边缘-like 控制信号**：将异步事件转为类 Canny 边缘图再经 3D CNN 编码注入潜空间，是一种高效的多模态条件融合思路，可与 ControlNet/T2I-Adapter 类设计互换借鉴。
- **光流 warping 特征对齐**：在 DiT 中间层用稀疏光流 warping 相邻帧 latent 特征并残差融合，有效增强时序一致性，该机制可直接复用于视频补全、视频稳像、temporal consistency regularization 等方向。
- **合成 event-video 数据构建 pipeline**：TransNet V2 片段选取 + Vid2e 模拟器的事件合成流程可复用于其他事件视觉任务的数据集构建。
- **多基准联合评测策略**：同时覆盖真实（BS-ERGB）与合成（DAVIS/Pexels）并在感知/失真双维度报告，为领域 benchmark 设计提供范式参考。

## 关键术语表
- **IWE (Image Warped Events)**：通过对比最大化将事件流在参考时刻对齐后生成的类边缘图像，富含场景结构与运动边界信息。
- **DiT (Diffusion Transformer)**：基于纯 Transformer 架构的扩散模型，本文以 Wan2.1 FLF2V 为骨干用于视频帧插值。
- **CMax (Contrast Maximization)**：通过最大化图像对比度同时估计光流与生成 IWE 的无监督事件处理方法。
- **FLF2V**：Wan2.1 开源框架中的关键帧插值（Fixed-Length Frame-to-Video）方法，本文为其 adapter 底座。
- **LoRA (Low-Rank Adaptation)**：低秩适应微调技术，本文用于在所有 DiT 块上高效适配新事件条件而不冻结主干。
- **VFI (Video Frame Interpolation)**：视频帧插值，指在首尾帧之间合成中间帧的任务。
- **BS-ERGB**：真实采集的高速事件-视频配对数据集（970×625, 28fps），含 48 训练 clip 与 26 测试 clip。
- **EvPexels**：本文构建的合成 event-video 数据集（1,100 场景、~39 万帧），专为事件驱动插值训练设计，计划开源。

## 可复现要素
- **数据集**：BS-ERGB（公开）、DAVIS（公开）、Pexels（公开视频源）、EvPexels（论文声明将开源并附工具）。
- **代码/权重**：项目主页 https://joseph-lin-tech.github.io/BridgeEventDiT-VFI/，代码与模型权重预计随论文出版开源；Wan2.1 FLF2V 为开源底座。
- **关键超参**：学习率 $1\times10^{-4}$、输入分辨率 832×480、训练 4,000 步、batch size 8、8×A800 GPU、光流适配器注入前 2 个 DiT 块、LoRA 应用于全部 DiT 块；其余超参遵循 FLF2V 原始配置。
