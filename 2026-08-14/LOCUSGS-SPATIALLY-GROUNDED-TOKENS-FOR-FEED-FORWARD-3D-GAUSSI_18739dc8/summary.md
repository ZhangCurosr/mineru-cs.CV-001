---
title: "LOCUSGS-SPATIALLY-GROUNDED-TOKENS-FOR-FEED-FORWARD-3D-GAUSSI"
source: https://arxiv.org/pdf/2608.12825v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:04:21"
field: "3D 视觉与神经渲染"
keywords: ["3D Gaussian Splatting", "Feed-forward Reconstruction", "Query-based Methods", "Spatial Grounding", "Novel View Synthesis"]
innovations: ["引入显式3D anchor状态（中心+半径）为query提供空间定位先验", "提出anchor-to-ray几何偏置引导cross-attention进行几何一致的跨视图特征聚合", "以anchor为中心、半径为缩放因子进行Gaussian局部偏移解码"]
benchmarks: ["DL3DV", "RealEstate10K"]
---

# 论文速读：LOCUSGS - SPATIALLY GROUNDED TOKENS FOR FEED-FORWARD 3D GAUSSIAN SPLATTING

## 一句话总结
本文提出 LocusGS，为基于可学习 query 的 feed-forward 3D Gaussian Splatting 方法引入显式的 3D anchor 状态（中心点 + 支持半径），通过逐层渐进式细化 anchor 并引导跨视图特征聚合与 Gaussian 解码，显著提升了渲染质量与 Gaussian 空间组织的结构连贯性。

## 研究问题与动机
- 现有 query-based feed-forward 3DGS（如 TokenGS）将场景表示为固定数量的可学习 query，每个 query 聚合多视图证据后解码一组 Gaussian。但同一 query 解码出的 Gaussian 往往分散在场景中相距较远的区域，缺乏 query 级别的空间一致性。
- 根本原因在于现有方法将 query 仅表示为隐式 latent 特征，没有显式指定该 query 在 3D 空间中的操作位置及其覆盖范围，导致聚合的证据和解码的 Gaussian 均不受局部区域的显式约束。
- 这种空间弥散分布与场景几何结构对齐较差，限制了 token-based 3DGS 表示的可解释性与重建精度。

## 核心贡献（创新点）
1. **显式 3D anchor 状态**：每个 Gaussian query 被扩充为一个 learnable 的 3D anchor 状态（中心 $\mu$ + 支持半径 $r$），使 query 具备明确的空间位置与覆盖范围感知，区别于 TokenGS 等仅有 latent 特征的 query 表示。
2. **Anchor-guided 交叉注意力**：提出 anchor-to-ray 几何偏置，衡量 anchor 中心与图像 patch 对应 Plücker 射线之间的最短距离，并将其作为附加偏置注入 cross-attention logits，引导 query 优先聚合与其 3D 位置几何一致的视图证据，与纯内容驱动的 attention 形成本质区别。
3. **Anchor-centered Gaussian 解码**：最终 Gaussian 中心以 anchor 为中心、以半径为缩放因子预测局部偏移（$\mu^G = \mu^L + r^L \delta$），使同一 query 解码出的 Gaussian 形成局部紧凑的 3D 分组，而非自由散射在全局空间中。
4. **多层渲染监督**：利用 anchor 的逐层渐进细化特性，在多个 decoder 层施加中间渲染损失（含 Gaussian 中心与 anchor 中心的可见性正则），加速收敛并提升优化稳定性。

## 方法详解
- **整体架构**：编码器将多视图图像 tokenize 为 patch-level 视觉 token，相机位姿以 patch-level Plücker 射线嵌入注入；解码器维护 $N$ 个 Gaussian query，每个 query $i$ 在 layer $l$ 处具有 token 特征 $\mathbf{q}_i^l$ 与 anchor 状态 $\mathbf{a}_i^l = (\mu_i^l, r_i^l)$。
- **Anchor 初始化**：anchor 中心在归一化 3D 场景空间内随机初始化，半径经 inverse softplus 转换得到初始未约束参数 $\rho$。
- **Self-attention**：由 anchor 中心经正弦位置编码 + MLP 生成空间位置嵌入 $\mathbf{p}_i^l$，叠加到 query 特征上（$\tilde{\mathbf{q}}_i^l = \mathbf{q}_i^l + \mathbf{p}_i^l$），使 query 间交互依赖其 3D 空间位置。
- **Cross-attention（Anchor-aware）**：定义 anchor-to-ray 几何偏置 $b_{ij}^l = -\frac{1}{2}\left(\frac{D(\mu_i^l, \ell_j)}{\sigma_0 r_i^l}\right)^2$，其中 $D(\cdot,\cdot)$ 为 3D 点到 Plücker 射线的最短距离，$\sigma_0=0.1$ 为固定带宽超参。交叉注意力 logit 为 $\alpha_{ij}^l = \mathrm{softmax}_j\left(\frac{(\bar{\mathbf{q}}_i^l)^\top \mathbf{k}_j}{\sqrt{d}} + \gamma b_{ij}^l\right)$，$\gamma$ 为 learnable scale（softplus 约束为正）。
- **动态 Anchor 细化**：每层更新后，token 特征经轻量 head 预测 anchor 中心与未约束半径的残差：$\Delta\mu_i^l = f_\mu(\mathbf{q}_i^{l+1})$，$\Delta\rho_i^l = f_\rho(\mathbf{q}_i^{l+1})$，按 $\mu^{l+1} = \mu^l + \Delta\mu^l$，$\rho^{l+1} = \rho^l + \Delta\rho^l$ 更新，激活半径 $r = \mathrm{softplus}(\rho) + \epsilon$。
- **Anchor-centered Gaussian 解码**：每 token 解码 $K=64$ 个 Gaussian，中心为 $\mu_{i,k}^G = \mu_i^L + r_i^L \delta_{i,k}$，其余属性（尺度、旋转、颜色、不透明度）由标准 token-based 预测头输出。
- **多层监督损失**：对监督层集合 $\mathcal{S} = \{l_1,\dots,l_M\}$（含最后一层），每层损失为 $\mathcal{L}^{l_m} = \mathcal{L}_{\mathrm{rec}}^{l_m} + \lambda_G \mathcal{L}_{\mathrm{vis}}(\{\mu_{i,k}^{G,l_m}\}) + \lambda_A \mathcal{L}_{\mathrm{vis}}(\{\mu_i^{l_m}\})$，总损失按层序号加权求和。$\lambda_{\mathrm{SSIM}}=0.2$，$\lambda_G=1.0$，$\lambda_A=0.1$。

## 实验与结果
- **数据集**：RE10K（2 view, 256×256）与 DL3DV（4 view 训练，2/4/6 view 测试，256×256 / 448×256）。
- **基线**：TokenGS（同 token 数与 Gaussian 预算的强 baseline）、MVSplat、DepthSplat、GS-LRM 等。
- **RE10K 结果**（2 view）：1024 token 变体 PSNR=28.50、SSIM=0.909，超越 TokenGS 1024（28.02/0.896）+0.48dB PSNR；4096 token 变体 PSNR=28.89、SSIM=0.916，同样超越 TokenGS 4096（28.41/0.903）。1024 token 变体已超越 GS-LRM（28.10/0.892），且 Gaussians 数量仅为其一半。
- **DL3DV 结果**（4 view 训练，4 view 测试）：LocusGS 4096 token 达到 PSNR=24.80、SSIM=0.812，超越 TokenGS 4096（23.44/0.757）达 +1.36dB / +0.055 SSIM。在 2 view 和 6 view 测试设置下同样保持一致性提升。
- **Token 级 Gaussian 分散度**（$C_{\mathrm{centroid}}$）：RE10K 从 TokenGS 的 5.1164 降至 0.1978；DL3DV 从 0.5881 降至 0.0433，表明同一 token 对应的 Gaussian 组空间紧凑度显著提升。
- **收敛性**：LocusGS 训练更快收敛，在训练/验证集 PSNR 均持续优于 TokenGS。
- **推理延迟**：LocusGS 参数量 241.5M（TokenGS 222.0M），纯前向时间 407.3ms（TokenGS 341.0ms），开销增加约 19%。

## 相关工作脉络
- **TokenGS [8]**：本文主要对比基线，采用纯 latent query 聚合多视图特征并解码 Gaussian 组；LocusGS 在其基础上引入显式 3D anchor 状态，使 query 具备空间定位能力。
- **MVSplat [6] / DepthSplat [7]**：dense pixel-aligned feed-forward 3DGS，Gaussian 预算随输入视图数和分辨率线性增长；LocusGS 保持固定预算，解耦于输入规模。
- **GS-LRM [24]**：基于大重建模型的 feed-forward 3DGS，使用预训练几何 backbone；LocusGS 无需额外预训练几何模型，在相同 token/Gaussian 预算下即可超越。
- **DAB-DETR [20]**：将 object query 与动态细化的 anchor box 结合用于 2D 目标检测；LocusGS 借鉴此思想但在 3D 空间中维护中心+半径状态，服务于场景级 Gaussian 重建而非 2D 检测。
- **Scaffold-GS [21]**：用稀疏 3D anchor 组织局部 Gaussian，但 anchor 为外部初始化而非从多视图特征中端到端学习；LocusGS 的 anchor 由多视图图像特征逐步细化，是 query 自身的状态。
- **GlobalSplat [9] / C3G [19]**：同类 query-based 方法，分别采用解耦几何/外观分支和预训练 VGGT 编码器；本文选择 TokenGS 作为受控对比，以隔离空间锚定带来的增益。

## 局限性与未来方向
- 当前方法假设输入视图为已知位姿（calibrated），anchor-aware 聚合依赖相机射线，扩展到 pose-free 场景或联合处理位姿不确定性是自然方向。
- 每个 token 仅用中心+标量半径描述局部空间支持，属于各向同性表示，难以刻画细长结构或几何复杂区域；未来可采用各向异性支持或可见性感知的不确定性表示。
- 推理延迟较 TokenGS 增加约 19%，在极端实时场景下仍需进一步优化。

## 研究启发与可借鉴点
- **显式空间先验引入 query-based 表示**：将 3D anchor 状态（中心+半径）与 latent query 结合的设计范式，可迁移到其他 query-based 3D 重建任务（如神经辐射场、点云生成），为隐式表征注入可解释的空间结构。
- **Geometric bias 注入 cross-attention**：通过几何关系（点-射线距离）构造 attention bias 的设计，可在任意需要多视图信息聚合的 Transformer 解码器中复用，减少纯外观驱动的歧义聚合。
- **多层中间监督策略**：对渐进细化结构施加多层渲染损失并按层加权，是一种有效的训练稳定技巧，适用于任何逐层 refinements 的网络架构。
- **Radius-scaled offset 解码**：以 anchor 为中心、半径为缩放因子预测局部偏移，是一种保持局部紧凑性的解码范式，可推广至 3D 物体/场景的 token-based 表示学习。

## 关键术语表
- **Feed-forward 3DGS**：直接从输入视图一步预测 3D Gaussian 表示的方法，避免逐场景迭代优化，实现快速推理。
- **Gaussian Query**：可学习的查询向量，通过 Transformer 解码器聚合多视图特征并解码一组 3D Gaussian 基元。
- **Anchor State**：每个 query 附带的显式 3D 空间状态，由中心点 $\mu$ 和支持半径 $r$ 构成，提供 query 的空间定位与覆盖范围先验。
- **Anchor-to-ray Geometric Bias**：衡量 anchor 中心与图像 patch Plücker 射线之间最短距离的负指数偏置，注入 cross-attention logits 以引导几何一致的证据聚合。
- **Plücker Ray**：用六维坐标 $(\mathbf{m}, \mathbf{d})$ 表示的 3D 射线，其中 $\mathbf{d}$ 为方向、$\mathbf{m}=\mathbf{o}\times\mathbf{d}$ 为力矩向量，常用于高效计算点-线距离。
- **Token-level Gaussian Dispersion ($C_{\mathrm{centroid}}$)**：衡量同一 query 解码出的所有 Gaussian 中心到其质心的平均距离，值越小表示空间组织越紧凑。
- **Multi-layer Rendering Supervision**：在多个 decoder 层施加中间渲染损失，使渐进细化的 anchor 和 Gaussian 均受到直接监督，加速训练收敛。

## 可复现要素
- **数据集**：RE10K、DL3DV（均为公开数据集）。
- **代码**：论文项目页面 https://leo-frank.github.io/LocusGS_viewer，代码开源情况论文未明确声明。
- **权重**：论文未提及开源。
- **关键超参**：base bandwidth $\sigma_0=0.1$；每 token 解码 Gaussian 数 $K=64$；base 学习率 $4\times10^{-4}$（warmup 2000 iters），finetune 学习率 $4\times10^{-5}$（warmup 400 iters）；base/finetune 各 300/20 epochs；$\lambda_{\mathrm{SSIM}}=0.2$，$\lambda_G=1.0$，$\lambda_A=0.1$。
- **硬件**：NVIDIA A100 40GB GPU。
- **优化器**：AdamW。
