---
title: "Bridging Event Streams and DiT: Event-Guided Video Frame Interpolation"
source: https://arxiv.org/pdf/2608.10479v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:30:51"
field: "事件视觉与视频生成"
keywords: ["Video Frame Interpolation", "Event Camera", "Diffusion Transformer", "IWE", "Adapter-based Fine-tuning", "Wan2.1", "Temporal Consistency"]
innovations: ["将事件流通过CMax转化为IWE与双向光流，以轻量适配器注入预训练DiT模型进行帧插值", "提出IWE空间编码器与光流对齐-融合双路适配器，无需端到端事件中心训练", "构建EvPexels大规模合成event-video数据集并开源"]
benchmarks: ["BS-ERGB", "DAVIS", "Pexels"]
---

# 论文速读：Bridging Event Streams and DiT: Event-Guided Video Frame Interpolation

## 一句话总结
本文提出一种基于适配器的轻量微调框架，将从事件相机流中提取的 Image Warped Events（IWE）和双向稀疏光流注入预训练的 DiT-based 视频扩散模型（Wan2.1 FLF2V），以事件提供的高时间分辨率运动线索弥补仅依赖首尾帧进行帧插值的不足，显著提升插帧的清晰度与时序一致性。

## 研究问题与动机
1. **现有 LDM-based VFI 方法的局限**：FCVG、Wan2.1 FLF2V 等仅依赖起止帧引导插值，面对大时间间隔与复杂高速运动时易产生运动模糊、物体形变、身份不一致和结构伪影。
2. **事件数据与生成模型的表示鸿沟**：事件相机提供的异步、稀疏像素级变化信号，无法直接对齐主流扩散模型所需的密集网格表示（如 edge map、flow field）。
3. **配对数据稀缺制约端到端训练**：大规模 event-video 配对数据集匮乏，使得从头训练事件辅助扩散模型不切实际。
4. **已有事件辅助扩散方法的代价过高**：如 VDM-EVFI 采用 ControlNet-style 重型架构，需端到端事件中心化训练，计算与数据开销大。

## 核心贡献（创新点）
1. **适配器式事件注入框架**：在冻结的预训练 DiT 基础上仅增加 IWE 编码器与光流对齐-融合适配器，辅以 LoRA，无需改动主干架构即可实现事件引导插帧，区别于 VDM-EVFI 等需要重型 ControlNet 的端到端方案。
2. **事件流到扩散控制信号的桥接**：通过对比最大化（CMax）从原始事件流中提取 IWE 与双向稀疏光流，将异步稀疏事件转化为与 edge map/flow field 语义兼容的中间表征，首次系统性地打通事件视觉与 DiT 扩散生成之间的接口。
3. **双路适配器的即插即用设计**：IWE 编码器以 3D CNN 提取边缘一致性空间特征并加至输入潜层；光流对齐-融合模块在选定 DiT 块前对相邻帧特征做 warp 对齐后聚合为残差修正，二者分别负责空间结构引导与时序一致性增强。
4. **大规模合成数据集 EvPexels**：构建 1,100 个多样化运动场景、共约 39 万 RGB 帧的合成 event-video 数据集，填补事件驱动帧插值训练数据的空白，并将开源。

## 方法详解
- **基础模型**：采用 Wan2.1 FLF2V（DiT-based I2V 扩散模型），以首帧 $I_0$ 和尾帧 $I_1$ 为条件在潜空间生成中间帧序列。训练目标沿用原始 MSE：$\mathcal{L} = \mathbb{E}\|u(x_t^k, c_{txt}, t;\theta) - v_t\|^2$，扩展为事件辅助形式后加入 $\mathcal{E}$ 条件。
- **事件表示提取**：将 $I_0$ 与 $I_1$ 之间的事件流划分为多个时段 $[k{-}1,k]$ 和 $[k{+}1,k]$，利用 CMax 方法分别计算前向/后向稀疏光流 $\mathbf{f}_{k{-}1\to k}$、$\mathbf{f}_{k+1\to k}$ 及对应 IWE 图 $\mathcal{W}^{k{-}1\to k}$、$\mathcal{W}^{k+1\to k}$。
- **IWE 空间注入**：IWE 图经轻量 3D CNN 编码器得到特征 $\mathbf{F}_{\mathcal{W}}^k$，通过逐元素加法注入视频潜输入的初始层；所有 DiT 块附加 LoRA 层以适配新模态。
- **光流对齐-融合模块**：在每个选定 DiT 块前，将 patch tokens 还原为帧级特征，利用估计光流 warp 相邻帧特征 $\mathbf{F}_{x_t}^{k-1}$、$\mathbf{F}_{x_t}^{k+1}$ 至当前帧，再经卷积融合网络 $G(\cdot)$ 聚合后以残差形式加入当前帧特征：$\mathbf{F}_{\mathrm{fused}}^k = G(\mathbf{F}_{x_t}^k, \mathbf{F}_{x_t}^{k-1\to k}, \mathbf{F}_{x_t}^{k+1\to k})$。
- **训练配置**：学习率 $1\times10^{-4}$，输入分辨率 832×480，在 8×NVIDIA A800 上训练 4,000 步、batch size 8；光流对齐模块插入前两个 DiT 块（共 40 块）。

## 实验与结果
- **数据集**：真实 BS-ERGB（970×625，28fps）、合成 EvPexels（704×480）、DAVIS（50 clips）、Pexels（30 clips）；统一 ×24 插值设置。
- **基线**：传统 VFI（RIFE、TimeLens、CBMNet-Large、TimeLens-XL）、扩散 VFI（TRF、GI、ViBiD、FCVG、Wan2.1 FLF2V）、事件扩散 VFI（VDM-EVFI-Wan2.1）。
- **BS-ERGB**：本方法感知指标全面最优（LPIPS 0.132、FID 8.168、FVD 117.368）；失真指标 PSNR 23.261（第三）、SSIM 0.704（第二），略低于强调像素重建的传统方法。
- **DAVIS**：本方法 PSNR 25.544、SSIM 0.799、LPIPS 0.115、FID 13.367、FVD 158.557，全部指标最优。
- **Pexels**：本方法 PSNR 29.089、SSIM 0.858、LPIPS 0.080、FID 16.319、FVD 151.345，全部指标最优。
- **消融结论**：① 移除 IWE 或光流 warp 均导致明显下降；② 直接将 flow 作为输入（不 warp）劣于 warp 对齐；③ 注入位置前两块优于中/后两块（重建指标优先）；④ IWE+Flow 表示显著优于 Edge-based（CUBE）与 Event Voxel Stack（VDM-EVFI）。

## 相关工作脉络
1. **传统 VFI（RIFE、AMT 等）**：依赖显式光流估计与补偿，擅长简单运动但对大位移与 Occlusion 处理乏力，本文方法在 perceptual 指标上远超此类。
2. **事件辅助传统 VFI（TimeLens、CBMNet、TimeLens-XL）**：利用事件提供细粒度运动，PSNR/SSIM 表现强，但易在动态前景区域产生伪影；本文继承其运动引导思路并将其接入扩散生成管线。
3. **LDM-based VFI（MCVD、LDMVFI、VIDIM、DreamMover、TRF、GI、ViBiD、FCVG、Wan2.1 FLF2V）**：以预训练 I2V 模型为核心，本文在此基础上进一步引入事件信号弥补仅首尾帧引导的信息瓶颈。
4. **事件辅助扩散 VFI（VDM-EVFI）**：采用 ControlNet 风格将 event voxel grid 编码为条件信号，需从头训练且模型厚重；本文通过 CMax 预提取 IWE+光流并仅以轻量适配器注入，避免事件端到端训练与架构改造。

## 局限性与未来方向
1. 实验仅覆盖 ×24 插值，对更大时间间隔或极端运动的泛化能力未验证。
2. 事件表征完全依赖 CMax 预估计的光流质量，光流估计误差将直接传导至插帧结果。
3. EvPexels 为合成数据（Vid2e 模拟器生成事件），与真实事件相机分布存在域偏移，模型在真实事件数据上的鲁棒性有待检验。
4. 适配器仅插入前两个 DiT 块，未探索更灵活的注入策略（如自适应深度、多尺度融合）。

## 研究启发与可借鉴点
1. **事件到扩散控制的通用桥接范式**：CMax 提取 IWE+光流再注入预训练扩散模型的思路可迁移至事件驱动的视频生成、去模糊、超分等下游任务。
2. **轻量适配器 + LoRA 的新模态接入策略**：在冻结 DiT 主干的基础上通过局部模块微调融合异步传感器数据，兼顾性能与参数效率，适用于多模态生成模型扩展。
3. **EvPexels 构建流程的可复用性**：TransNet V2 抽帧 → Vid2e 事件合成 → 按运动多样性筛选的数据 pipeline 可直接复用于其他事件-视频联合预训练研究。
4. **潜特征 warp 对齐的时序一致性机制**：在 DiT 块前对相邻帧 latent 做光流 warp 后残差融合的设计，可推广至任何基于 transformer 的时序生成模型。

## 关键术语表
**Event Camera（事件相机）**：异步记录像素级亮度变化的传感器，提供微秒级时间分辨率与高动态范围，擅长捕捉高速运动。
**IWE（Image Warped Events）**：通过时间对齐事件流生成的类边缘图像，对场景轮廓与物体边界高度敏感，可作为扩散模型的结构性控制信号。
**DiT（Diffusion Transformer）**：以 Transformer 为骨干的扩散模型架构，近年成为视频生成（如 Wan2.1）的主流底座。
**Contrast Maximization（CMax）**：通过最大化图像对比度来联合估计光流与生成 IWE 的事件处理方法。
**FLF2V（Forward-Last Frame to Video）**：Wan2.1 中的帧插值变体，以首尾帧为条件生成固定长度视频序列。
**BS-ERGB**：真实捕获的高速事件-视频配对基准数据集（970×625，28fps），广泛用于事件辅助 VFI 评估。
**EvPexels**：本文构建的 1,100 场景合成 event-video 数据集，用于事件感知插帧模型的训练与泛化验证。

## 可复现要素
- **数据集**：BS-ERGB（公开）、EvPexels（论文声明将随项目页面开源）、DAVIS（公开）、Pexels 视频（公开可下载）。
- **代码/权重**：项目页面 https://joseph-lin-tech.github.io/BridgeEventDiT-VFI/，论文声明开源数据集与工具；基座模型 Wan2.1 FLF2V 为开源。
- **关键超参**：学习率 $1\times10^{-4}$，输入分辨率 832×480，训练 4,000 步、batch size 8、8×A800；光流对齐模块注入前 2 个 DiT 块；LoRA 应用于全部 DiT 块。
