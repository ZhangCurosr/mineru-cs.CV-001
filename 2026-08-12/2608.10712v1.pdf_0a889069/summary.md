---
title: "Compact Feed-Forward 3D Gaussians via Saliency-Guided Primitive Merging"
source: https://arxiv.org/pdf/2608.10712v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:53:10"
field: "3D 视觉与神经渲染"
keywords: ["3D Gaussian Splatting", "Feed-Forward Reconstruction", "Superpixel Segmentation", "Primitive Merging", "Novel View Synthesis", "Model Compression"]
innovations: ["显著性引导的内容自适应超像素分组替代均匀体素化", "Feature Gaussian + Slot Decoder 实现推理时可调节 LOD 的质量-效率折中", "骨干无关的跨视图学习合并模块，冻结 FF backbone 即插即用"]
benchmarks: ["DL3DV-Bench", "MipNeRF360", "Tanks and Temples"]
---

# 论文速读：Compact Feed-Forward 3D Gaussians via Saliency-Guided Primitive Merging

## 一句话总结
本文提出一种结构感知的超像素引导原语合并流水线，可将任意前馈（Feed-Forward）3D Gaussian Splatting 方法输出的每像素原语压缩至原有的 **约 1/20**，同时在多个基准上保留甚至提升新视图合成质量，且可作为骨干网络无关的后处理模块使用。

## 研究问题与动机
- **冗余严重**：前馈 3DGS 方法直接预测每像素 Gaussian，产生大量高度冗余、位置次优的原语，导致渲染延迟高、内存占用大，难以扩展到大规模场景和机器人/自动驾驶等下游任务。
- **现有压缩方法各有局限**：均匀体素化无法感知区域视觉复杂度差异；迭代细化易过拟合输入视图；基于检测的方法仅在初始重建阶段操作，缺乏跨视图合并能力；图融合方法依赖全局统一阈值，对内容不敏感。
- **缺乏骨干网络无关的通用压缩方案**：前述方法均与重建模型深度耦合，无法在不重训的前提下受益于更强的基础模型（foundation model）。
- **稀疏视图下泛化性不足**：冗余原语会引入高频伪影和半透明雾状噪声，不利于模型对未见视图的泛化。

## 核心贡献（创新点）
1. **内容自适应的显著性引导超像素分组**：用 Shi-Tomasi 角点响应引导 BASS 超像素种子初始化和分布，使纹理区域获得细粒度分块、平坦区域粗粒度合并，区别于传统 SLIC/BASS 均匀种子方案。
2. **Feature Gaussian + Set Transformer 编码-解码架构**：将可变大小的超像素组压缩为单个"特征高斯"（含空间参数、基础色和潜变量 $z_j$），通过 SAB+PMA 实现排列不变性聚合；解码器采用 slot 设计将单个 FG 扩展为 K 个输出 Gaussian，支持推理时可控 LOD 调节，而已有方法无此在线质量-效率旋钮。
3. **跨视图匹配与学习合并模块**：基于 3D kNN + AABB-IoU + 特征余弦相似度的双门控匹配机制，结合 SAB+PMA 融合的 merger，实现对重叠视角冗余 FG 的联合压缩，区别于 ReSplat/VolSplat 等单视图或固定网格方案。
4. **零样本即插即用骨干无关范式**：流水线仅训练新增模块，冻结任意 FF 骨干网络（DA3 / DepthSplat / AnySplat），可直接附加到最新大模型上，对比 ReSplat/VolSplat 需联合训练 backbone 的方案更为灵活。

## 方法详解
整体流程分为 **Encode → Merge → Decode** 三个阶段：

### 4.1 显著性引导的超像素分组
- 对每张输入视图 $\mathbf{I}_\nu$，计算 Shi-Tomasi 角点响应 $\lambda_\nu$（结构张量的最小特征值）；
- 以高响应区密集、低响应区稀疏的方式初始化 BASS 种子，替代传统均匀六边形网格；
- BASS 迭代优化，生成颜色均匀、空间连续、形状规则且大小随局部复杂度自适应的超像素分区图 $\mathbf{M}_\nu$，将同组内每像素 Gaussian 归入 $S_j$。

### 4.2 Feature Gaussian Encoder $\mathcal{M}^{\mathrm{enc}}$
- 对组 $S_j$ 内所有 Gaussian 做 MLP token 化后，经若干 **Set Attention Block (SAB)** 交互；
- **Pooling by Multihead Attention (PMA)** 层用可学习 query $\mathcal{Q}$ 将变长集合聚合为单 token；
- 输出一个 **Feature Gaussian** $\tilde{\mathcal{G}}_j = (\boldsymbol{\mu}_j, \mathbf{s}_j, \mathbf{q}_j, \mathbf{c}_j, \mathbf{z}_j)$，其中：
  - $\boldsymbol{\mu}_j$ 为透明度加权质心偏移；$\mathbf{c}_j$ 为组内漫反射色加权平均；
  - $\mathbf{z}_j \in \mathbb{R}^d$ 为潜变量，保存组内几何与外观细节；
  - 尺度 $\mathbf{s}_j$ 与旋转 $\mathbf{q}_j$ 暂不输出，推迟至 decoder 阶段，防止 encoder 通过"压缩几何"逃避跨视图匹配。

### 4.3 LOD Decoder $\mathcal{M}_K^{\mathrm{dec}}$
- 使用 slot-based 设计：latents + base color → K 个 slot token → 逐 slot 自注意力协调 → 输出头预测每个 Gaussian 的参数；
- $\boldsymbol{\mu}_i, \mathbf{c}_i$ 预测为相对于 FG 的偏移；推理时按 $\alpha_i < \tau_a$ 裁剪剔除低不透明度 Gaussian；
- 训练时维护 K=1/2/4 三个解码头，共享同一 encoder/merger，推理时按需选择——相当于一个无重训的质量-效率旋钮。

### 4.4 跨视图匹配与合并
- 对每个 FG 做 3D kNN（排除同输入视图候选）；
- **Match gate** 双重门控：$\cos(\mathbf{z}_{j_1}, \mathbf{z}_{j_2}) > \tau_f$ 且 AABB-IoU $> \tau_g$；
- 取连通分量作为 merge group，送入与 encoder 同构的 merger（SAB+PMA）；
-  merger 输出对首个 FG 的残差更新 $(\Delta \boldsymbol{\mu}, \Delta \mathbf{z}, \Delta m, \Delta \mathbf{c}_0, \Delta \mathbf{s}, \Delta \mathbf{q})$，所有投影零初始化以保证训练起始为恒等变换。

### 4.5 Refiner $\mathcal{M}^{\mathrm{ref}}$
- 对每个 FG 在其 kNN 邻域内做 self-attention，用邻居的相对位姿与属性作为 context token；
- 残差预测 FG 参数更新，专门针对横跨多个超像素组的薄结构（围栏、电线杆等）和深度边界进行修复；
- 同样零初始化保证训练稳定性。

### 4.6 训练目标
$$
\mathcal{L} = \lambda_{\text{MSE}} \mathcal{L}_{\text{MSE}} + \lambda_{\text{SSIM}} \mathcal{L}_{\text{SSIM}} + \lambda_{\text{LPIPS}} \mathcal{L}_{\text{LPIPS}} + \lambda_{\text{teach}} \mathcal{L}_{\text{teach}} + \lambda_{\text{div}} \mathcal{L}_{\text{div}}
$$
- **Photometric loss**：渲染图与 ground-truth 的 MSE / SSIM / LPIPS，覆盖输入视图与 Held-out 新视图。
- **Teacher loss**：早期因随机初始化导致几何坍塌，对每个超像素组做矩匹配（moment-matching）近似目标 $\boldsymbol{\mu}_j^{\text{teach}}, \Sigma_j^{\text{teach}}$，与 pipeline 输出做 geodesic quaternion + L1 加权损失，随训练 decay。
- **Decoder diversification**：对 $K>1$ 施加目标 AABB-IoU 的吸引/排斥正则，惩罚组内不透明度差异过大，防止单原语主导；同样随训练 decay。

## 实验与结果
**数据集与基准**：训练集 DL3DV-10K [20]；评测集 DL3DV-Bench（140 scene，未见）、MipNeRF360、Tanks & Temples；输入视角数 3/6/9/12。

**骨干网络**：DepthSplat [44]（有姿态）、AnySplat [11]（无姿态）、Depth Anything 3 (DA3) [19]（foundation model，统一深度/姿态/高斯预测）。

**主要结果（Table 1）**：
- **DA3 + Ours (K=1, $r_c = 4.4\%$)**：DL3DV-Bench **18.37 dB / 0.578 SSIM**；MipNeRF360 **17.87 dB / 0.514 SSIM**；Tanks & Temples **16.00 dB / 0.562 SSIM**。
  - 关键发现：DA3 未合并仅为 16.82 dB，**合并后反而提升到 17.87 dB**（MipNeRF360），说明超像素合并起到了正则化作用，去除了半透明雾状和高频噪声。
- **DepthSplat + Ours (K=2, $r_c = 10.5\%$)**：DL3DV-Bench **17.41 dB**，显著优于未合并 15.78 dB 与 Mom. Match 13.16 dB。
- **vs. ReSplat**：同量级 $r_c$（6.2% vs. 4.4%~8.9%），本文方法在所有骨干上 PSNR/SSIM 均占优；ReSplat 无法迁移至更强 foundation model。
- **vs. VolSplat**：VolSplat 在 MipNeRF360 仅 11.81 dB PSNR（严重退化），作者归因于其仅在 RealEstate-10K 训练。
- **线上增量重建（Table 2, MipNeRF360）**：K=1 在累积 84 视角后仅剩 **1.0M 原语**（朴素基线 16.1M），PSNR 15.33 vs. 15.27，FPS 从 60 提升至 **365（6.1×加速）**。

**Ablation（AnySplat, K=1, Table 3）**：
- w/o refiner：PSNR 14.14 → 13.49 dB，refiner 贡献显著；
- w/o cross-view：PSNR 几乎不变（14.13 vs. 14.14），但 $r_c$ 由 4.4% 升至 5.3%，说明跨视图合并主要用于进一步降冗余而非保质量；
- 更大超像素（$r_c=2.0\%$）：PSNR 13.97，验证细粒度分割保质量的结论；
- 与 K=4 对比：更大超像素 $r_c=8.0\%$ / PSNR=14.45 与 K=2 $r_c=8.9\%$ / PSNR=14.45 几乎持平，两种调节机制互补。

## 相关工作脉络
1. **PixelSplat / MVSplat / AnySplat / DepthSplat**：前馈 3DGS 代表工作，均输出每像素 Gaussian；本文不做重新设计骨干，而是作为通用后处理模块附加其输出之上。
2. **ReSplat [43]**：在 16× 子采样空间预测并迭代细化，实现 $r_c=6.2\%$；但需联合训练，且无法直接接入新 foundation model。
3. **VolSplat [38]**：体素对齐的 FF 3DGS，均匀离散化丧失区域自适应能力；本文在 MipNeRF360/T&T 上全面超越。
4. **Off The Grid [25] / Fuse-and-Refine [39]**：分别基于检测放置与图卷积融合；前者无法跨视图合并，后者依赖全局统一阈值；本文用显著性引导 + 学习融合规避两类局限（论文因代码不可用未纳入定量对比）。
5. **FreeSplat [40] / Gaussian Graph Network [50]**：基于图/深度的跨视图融合；均用固定距离阈值，对纹理密集/平坦区域一视同仁；本文用 BASS 超像素提供内容感知的分组粒度。
6. **ZPressor [37]**：信息瓶颈压缩多视图输入，属输入侧压缩；与本文输出侧原语合并正交，可组合使用。

## 局限性与未来方向
- **依赖骨干质量**：若初始 FF 输出原语质量差，合并无法恢复；属于"garbage in, garbage out"式局限。
- **合并引入额外延迟**：pipeline 引入 ~540 ms 开销（论文未详述），对于极低延迟应用受限。
- **LPIPS 提升有限**：过度合并会损失高频纹理细节，LPIPS 下降明显，需在 PSNR/SSIM 与感知质量之间权衡。
- **仅静态场景**：当前不支持动态场景的时序一致性。
- **超像素为算法驱动**：BASS + 显著性初始化是启发式方案，未与端到端可微分组联合优化。
- **作者 Future Work 建议**：① 扩展至动态场景（维持时序一致性）；② 用学习型分组替代算法超像素，与合并流水线联合训练。

## 研究启发与可借鉴点
1. **显著性引导的自适应分组思路可迁移**：Shi-Tomasi 角点响应作为超像素种子引导的策略，可直接推广至 NeRF/3DGS 以外的其他点云/原语后处理（如 LiDAR 点云聚类、特征金字塔压缩）。
2. **特征 Gaussian 概念具有通用性**：将一组结构化对象编码为一个"带潜变量的原子"，再通过 slot decoder 扩展，可在其它多视图聚合、跨视角特征融合任务中复用（如多视角图像拼接、视频时序特征压缩）。
3. **LOD 解码器作为推理时质量-效率旋钮**：无需重训即可在 K=1/2/4 间切换，对部署侧灵活适配（边缘设备用 K=1，服务器用 K=4）极具工程价值，可借鉴至 NeRF 轻量化、SLAM 建图压缩等场景。
4. **Teacher + Decaying Regularizer 训练技巧**：moment-matching teacher 缓解早期几何坍塌，diversification 正则鼓励 slot 分化——两者均 schedule decay，这套"软引导 + 逐步放手"策略可用于其它 set-to-set 压缩任务。
5. **零样本骨干无关验证范式**：在三个独立 FF backbone 上统一评测，证明方法泛化性而非过拟合单一架构，这种"跨架构泛化"的 eval 策略可作为后续工作的 benchmark 模板。

## 关键术语表
**3D Gaussian Splatting (3DGS)**：用各向异性 3D 高斯椭球集合表示场景，并通过可微分光栅化实现实时渲染的显式表示方法。

**Feed-Forward (FF) 3DGS**：绕过 per-scene 优化，直接从输入视图回归 Gaussian 参数的泛化重建方法，核心是 ViT backbone + 预测头。

**Feature Gaussian (FG)**：本文引入的中间表示，用单组空间参数 + 基础色 + 潜向量 $\mathbf{z}$ 编码一个超像素组的全部信息，供后续匹配、合并与解码使用。

**BASS (Bayesian Adaptive Superpixel Segmentation)**：通过迭代最大化后验实现颜色均匀、空间紧凑、边界规则的超像素分割算法；本文将其种子初始化替换为显著性引导版本。

**Shi-Tomasi 角点响应**：图像结构张量最小特征值，用于衡量局部纹理复杂度；高响应区密集布种、低响应区稀疏布种以引导超像素分布。

**Set Transformer (SAB + PMA)**：排列不变的注意力架构；SAB 进行 token 两两交互，PMA 用可学习 query 池化为单一输出，本文用作 encoder/merger 核心。

**Level-of-Detail (LOD) Decoder**：共享 encoder/merger 下不同 K 头的解码器，推理时通过选择 K 值在保真度与渲染开销间折中，无需重新训练。

**AABB-IoU**：Axis-Aligned Bounding Box 交集-over-并集，本文用于衡量两个 Feature Gaussian 几何重叠程度，作为跨视图匹配的门控条件之一。

## 可复现要素
- **训练数据集**：DL3DV-10K [20]，公开。
- **评测数据集**：DL3DV-Bench（140 scene）、MipNeRF360、Tanks & Temples，均公开。
- **代码开源**：论文未提及（"Due to unavailable public code" 提及部分 baseline，但对本方法未声明开源）；权重未提及。
- **关键超参**：AdamW，300,000 步，cosine schedule，最大 LR $10^{-6}$；K=1/2/4 三头联合训练；match gate 阈值 $\tau_f, \tau_g$、opacity prune 阈值 $\tau_a$、各 loss 权重 $\lambda$ 见 supplementary。
- **骨干实现**：DepthSplat / AnySplat / DA3 均已开源（对应引用）。
