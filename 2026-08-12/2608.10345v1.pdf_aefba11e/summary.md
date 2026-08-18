---
title: "CasDeblurGS: Cascaded 2D-to-3D Multi-View Consistency for 3D Gaussian Splatting from Two Blurry Images"
source: https://arxiv.org/pdf/2608.10345v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:28:25"
field: "神经渲染与三维重建"
keywords: ["3D Gaussian Splatting", "Deblurring", "Sparse Reconstruction", "Pose-free", "Multi-view Consistency"]
innovations: ["提出无位姿级联2D-to-3D引导框架，仅需两张模糊图像重建3DGS场景", "设计OCGM模块通过forward-backward一致性过滤遮挡/错误对应", "利用provisional 3D重渲染提供dense global guidance实现渐进式恢复"]
benchmarks: ["Deblur-NeRF"]
---

# 论文速读：CasDeblurGS: Cascaded 2D-to-3D Multi-View Consistency for 3D Gaussian Splatting from Two Blurry Images

## 一句话总结
本文提出了 CasDeblurGS，一种从无相机外参的仅两张运动模糊图像中恢复3D Gaussian Splatting 场景的级联2D-to-3D框架，通过 occlusion-aware 对应关系过滤建立局部可靠的2D引导，再利用 pose-free 3DGS 重渲染提供全局3D引导，实现高质量的新视角合成。

## 研究问题与动机
- **极端稀疏视角 + 运动模糊的联合挑战**：实际捕捉常受限于时间、传感器条件或平台稳定性，只能获取极少视角（如两张），且手持成像、平台振动、低光照会导致严重运动模糊，同时破坏细节和跨视角对应关系。
- **现有方法依赖过多冗余假设**：已有 blur-aware neural rendering 方法通常需要 20–40 张输入图像来联合稳定模糊和几何估计，在极端两视角下无法满足。
- **可靠几何初始化难以获取**：从模糊帧中通过标准 SfM 恢复相机位姿已很困难，当视图冗余和几何初始化均薄弱时，问题变为 ill-posed，导致不稳定和虚假几何。
- **纯2D去模糊不足以保证3D一致性**：提升2D图像质量本身不保证新视角合成的3D一致性，而现有 diffusion-based 方法在严重模糊下也会因几何匹配线索坍塌而失效。

## 核心贡献（创新点）
1. **提出无位姿前馈式 CasDeblurGS 框架**：仅需两张模糊图像和已知内参即可重建 coherent 3D 场景，无需输入视角位姿、辅助锐利图像或 per-scene 测试时优化。
2. **设计级联 2D-to-3D 多视角一致性引导策略**：Stage 1 通过 occlusion-aware cross-view guidance module (OCGM) 建立局部可靠对应关系，Stage 2 利用 frozen pose-free 3DGS 重渲染提供密集全局3D引导，形成互补。
3. **证明渐进式引导显著优于单次去模糊再重建**：与相同 backbone 搭配 DAVANet/Difix3D+ 的基线相比，CasDeblurGS 在真实场景提升 PSNR 1.19 dB，在合成场景提升 2.11 dB，验证了渐进式 2D→3D 引导的有效性。

## 方法详解
- **整体架构**：三个阶段级联——Stage 1（局部2D引导与去模糊）、Stage 2（全局3D引导与去模糊）、最终3D表示构建，所有模块推理时冻结，无 per-scene 优化。
- **Stabilizer 预处理**：使用冻结的 NAFNet 稳定器 $S_{\phi}$ 对两张模糊输入分别处理得到对齐友好观察 $C_1, C_2$，不作为最终恢复结果。
- **OCGM（Occlusion-aware Cross-view Guidance Module）**：
  - 使用冻结的 RAFT-Large 估计双向光流 $F_{1\to2}, F_{2\to1}$。
  - 通过 forward-backward consistency 计算残差 $e(x) = \|F_{br}(x) + F_{rb}(x')\|_2$。
  - 采用 motion-adaptive 阈值 $T(x) = \tau + \alpha(\|F_{br}(x)\|_2 + \|F_{rb}(x')\|_2)$，设置 $\tau=1.0, \alpha=0.01$。
  - 生成 validity mask $M_{br}(x) = \mathbb{I}[\text{in-bounds}(x')] \cdot \mathbb{I}[e(x) \leq T(x)]$。
  - 构造 masked warp $W_{br} = M_{br} \odot \mathcal{W}(C_r, F_{br})$。
- **Stage 1 2D-guided deblurring**：网络 $\mathcal{D}_{\theta}^{2D}$（NAFNet架构，base width 64）接收 7-channel 输入（原始模糊图 + masked warp + validity mask），输出中间恢复图 $\tilde{S}_1, \tilde{S}_2$。
- **Stage 2 3D-guided deblurring**：
  - 将中间恢复图输入冻结的 pose-free 3DGS backbone（NoPoSplat）得到 provisional 3D表示 $\mathcal{G}$。
  - 在输入视点处 re-render 得到 $R_1, R_2$ 作为全局3D引导。
  - 网络 $\mathcal{D}_{\psi}^{3D}$ 接收 6-channel 输入（原始模糊图 + re-render引导），输出最终恢复图 $\hat{S}_1, \hat{S}_2$。
- **最终3D表示**：将最终恢复图再次通过 frozen backbone 得到 $\mathcal{G}^{\star}$，用于 novel-view synthesis。
- **训练策略**：两阶段 stage-wise 训练，每阶段 400K 迭代，AdamW (lr=$10^{-3}$)，loss = $0.9\mathcal{L}_{\ell_1} + 0.1\mathcal{L}_{\text{perc}}$。

## 实验与结果
- **数据集**：Deblur-NeRF 的 camera-motion-blur subset，含 5 个 synthetic 和 10 个 real-world 场景，85 个固定 two-view tuples（25 synthetic + 60 real-world）。
- **评估基线**：SE-GS, GAURA, CoherentGS, DAVANet+NoPoSplat, Difix3D++NoPoSplat。
- **主要结果**：
  - **Real-world**：PSNR 21.13 dB（最优），SSIM 0.700，LPIPS 0.228，较最强基线 CoherentGS 提升 +1.19 dB PSNR。
  - **Synthetic**：PSNR 23.41 dB（最优），SSIM 0.796，LPIPS 0.165，较最强基线 Difix3D+ 提升 +2.11 dB PSNR。
- **Abation 关键数字**：
  - Blurry inputs → +Stabilizer: +0.23 dB
  - +Stabilizer → +Stage 1 (2D): +0.58 dB（最大增量）
  - +Stage 1 → +Stage 2 (3D): +0.25 dB
  - Cross-view matches（BlurBasket）：69 → 77 → 96 → 112
  - Camera reprojection：valid points +33%，reprojection error -9.7%，inlier ratio 0.9168 → 0.9343。
- **Runtime**：平均 65.52s，快于 CoherentGS (96.14s) 31.8%，快于 Difix3D+ (70.85s) 7.5%。

## 相关工作脉络
- **Blur-aware NeRF/3DGS**：Bad-NeRF/BAD-Gaussians 等方法通过 SE(3) 插值/Bézier 曲线建模运动轨迹，但需大量多视角冗余，在两视角下不可行。
- **Sparse blurry reconstruction with 2D priors**：HQGS 利用 2D edges/semantics，S2Gaussian 解决 feature-space 不一致，CoherentGS 使用视频 diffusion priors——依赖局部2D线索，难以保证全局3D一致性。
- **Feed-forward 3DGS**：PixelSplat、MVSplat、NoPoSplat 等方法假设 sharp inputs，GAURA 处理 blurry inputs 但需 known poses 且用 kernel-based synthetic blur，在两视角下未验证。
- **Multi-view image deblurring**：利用 disparity/depth 进行恢复，但在 extreme two-view 设置下因高细节丢失和位姿/深度不可用而受限。
- **3D-consistent restoration**：3DENHANCER、SiR-DIF 等 diffusion-based 方法，但在严重模糊导致几何匹配线索坍塌时脆弱。
- **本文定位**：与上述方法本质区别在于——无需 external generative priors、无需 per-scene optimization、无需 camera poses，通过级联 2D→3D 引导在无位姿条件下实现稳定两视角模糊重建。

## 局限性与未来方向
- 假设已知相机内参和静态场景，仅适用于两视角设置。
- 极端模糊、大遮挡或重叠区域有限时，correspondence guidance 可能退化。
- 未来工作将扩展到未知内参、动态场景和更广泛的稀疏视角设置。

## 研究启发与可借鉴点
1. **级联 2D→3D 引导策略可迁移**：OCGM 的 forward-backward consistency 掩码机制可复用于其他 sparse-view 或多视角恢复任务，作为可靠对应关系过滤的通用模块。
2. **Frozen backbone + stage-wise trainable restoration 的设计范式**：固定通用 3DGS backbone、仅训练轻量级去模糊网络的模式，可降低训练成本并保证推理效率，适合资源受限场景。
3. **Re-render 作为全局引导的思想**：利用 provisional 3D 表示的重渲染结果提供 dense global guidance，解决了纯 2D 方法缺乏全局一致性的问题，可推广到其他神经渲染任务。
4. **Motion-adaptive threshold 设计**：$T(x) = \tau + \alpha\|F(x)\|_2$ 根据运动幅度自适应调整容错阈值，比固定阈值更鲁棒，可借鉴于其他光流/对应关系过滤场景。

## 关键术语表
- **3D Gaussian Splatting (3DGS)**：使用各向异性 3D 高斯分布集合表示场景的神经渲染技术，支持高效可微分渲染。
- **Pose-free 3DGS**：无需输入相机外参，仅凭图像和已知内参直接预测 3D 高斯表示的 feed-forward 方法（如 NoPoSplat）。
- **OCGM (Occlusion-aware Cross-view Guidance Module)**：通过双向光流估计和 forward-backward consistency 过滤遮挡/错误对应，生成 masked cross-view warp 的模块。
- **Forward-backward consistency**：验证光流可靠性的方法，计算正向流与反向流的残差，残差小于阈值则视为有效对应。
- **Motion-adaptive threshold**：根据光流位移大小动态调整的一致性阈值，公式为 $T(x) = \tau + \alpha(\|F_{br}\| + \|F_{rb}\|)$。
- **Re-rendering as guidance**：将 provisional 3D 表示在输入视点处重新渲染，生成 dense global 3D 引导图用于最终恢复。
- **Deblur-NeRF**：包含合成和真实世界运动模糊场景的数据集，用于评估模糊图像下的 3D 重建性能。
- **Stage-wise training**：先训练 Stage 1 网络并冻结，再用其输出训练 Stage 2 网络的渐进式训练策略。

## 可复现要素
- **数据集**：Deblur-NeRF camera-motion-blur subset（5 synthetic + 10 real-world scenes），训练数据来自 DeepDeblurRF 的合成相机运动模糊数据集；评估用固定的 85 个 two-view tuples，索引映射在 supplementary Tab.1 中提供。**数据集公开**。
- **代码/权重**：论文未明确声明开源状态，但提供了 NoPoSplat checkpoint 和 NAFNet pretrained weights 的引用。**代码开源情况论文未提及**。
- **关键超参**：
  - OCGM: $\tau = 1.0$, $\alpha = 0.01$
  - 训练: 400K iterations, AdamW lr=$10^{-3}$, weight decay=$10^{-3}$, $\beta_1=\beta_2=0.9$
  - Cosine decay to min lr=$10^{-7}$, batch size=32/GPU, 2×A100
  - Loss: $\lambda_1=0.9, \lambda_p=0.1$
  - 输入尺寸: 256×256
  - 网络架构: NAFNet base width=64, encoder [1,1,1,28], middle block×1, decoder [1,1,1,1]
