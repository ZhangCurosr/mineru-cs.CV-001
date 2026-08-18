---
title: "CasDeblurGS: Cascaded 2D-to-3D Multi-View Consistency for 3D Gaussian Splatting from Two Blurry Images"
source: https://arxiv.org/pdf/2608.10345v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:42:39"
field: "稀疏视角去模糊与3D重建"
keywords: ["3D Gaussian Splatting", "Motion Deblurring", "Sparse View Reconstruction", "Pose-Free Neural Rendering", "Multi-View Consistency", "Feed-Forward 3D Reconstruction"]
innovations: ["提出级联2D-to-3D多视角一致性框架，从两帧模糊图像在无外参与无逐场景优化下完成3DGS重建", "设计OCGM遮挡感知跨视角引导模块，结合运动自适应前向-后向一致性掩码筛选可靠光流对应", "利用冻结pose-free 3DGS主干重渲染提供全局3D引导，实现两级渐进去模糊与一致重建"]
benchmarks: ["Deblur-NeRF (real-world)", "Deblur-NeRF (synthetic)"]
---

# 论文速读：CasDeblurGS: Cascaded 2D-to-3D Multi-View Consistency for 3D Gaussian Splatting from Two Blurry Images

## 一句话总结
本文提出 CasDeblurGS，一种从两帧运动模糊图像（仅已知内参、无外参）进行免相机位姿、免逐场景优化的级联式 feed-forward 3DGS 重建框架；通过"局部 2D 对应引导→全局 3D 重渲染引导"的两阶段级联策略，在 Deblur-NeRF 真实/合成数据上分别将 PSNR 提升 1.19 dB 和 2.11 dB。

## 研究问题与动机
- **极端稀疏视角+严重模糊**：实际采集常受限于时间/稳定性，仅有两帧含运动模糊的图像可用，模糊进一步削弱本就稀缺的跨视角几何约束，使联合估计模糊与场景结构严重不适定。
- **既有方法依赖多视角冗余**：如 BAD-NeRF、Crim-GS 等轨迹优化方法需 20–40 张输入图才能稳定联合估计模糊轨迹与几何，在两张视角下几乎失效。
- **既有方法依赖准确相机位姿或辅助先验**：GAURA 等方法假设已知位姿或使用核模拟合成模糊；CoherentGS 依赖视频扩散先验与交替逐场景优化；极端两视角下 SfM 初始化不可行。
- **纯 2D 去模糊无法保证 3D 一致性**：HQGS、S2Gaussian 等依赖局部 2D 线索，在两张视角缺乏全局 3D 约束时易产生结构失真与跨视图不一致。

## 核心贡献（创新点）
- **级联 2D-to-3D 多视角一致性框架**：提出由局部可靠 2D 对应到全局 3D 引导的渐进恢复范式，与依赖外部扩散先验或大量冗余视角的方法形成本质区别。
- **OCGM（遮挡感知跨视角引导模块）**：在冻结 stabilizer + RAFT 基础上，引入前向-后向一致性掩码与运动自适应阈值，对不可靠对应进行筛选；比直接用原始模糊图估计光流的基线更具几何可信度。
- **冻结 pose-free 3DGS 主干的阶段性复用**：在 Stage 2 中以中间恢复图构建临时 3D 表示并做输入视角重渲染，得到稠密全局 3D 引导；与"先 2D 去模糊再通用重建"的控制基线相比，全局 3D 反馈是关键增益来源。
- **在极端两视角模糊设定下实现 pose-free、免逐场景优化、免外部 sharp 图像**：推理仅需两帧模糊图与内参；相比 CoherentGS/Difix3D+ 等方法对相机位姿或扩散先验的依赖更低。

## 方法详解
**总体流程**：给定两帧模糊图像 $\{ (B_i, K_i) \}_{i=1}^2$，三阶段级联：Stage 1 建立局部 2D 引导并输出中间恢复 $\tilde{S}_1, \tilde{S}_2$；Stage 2 将中间恢复送入冻结的 pose-free 3DGS 主干 $h_\eta$（NoPoSplat）生成临时高斯集合 $\mathcal{G}$，将其重渲染回输入视角得到全局 3D 引导 $R_1, R_2$，再经 3D 引导去模糊网络 $\mathcal{D}_\psi^{3D}$ 得到最终恢复 $\hat{S}_1, \hat{S}_2$；最后用同一 $h_\eta$ 从 $\hat{S}_1, \hat{S}_2$ 生成输出 3D 表示 $\mathcal{G}^\star$ 供新视角合成，全程无逐场景优化。

**Stage 1：局部 2D 引导与去模糊**
- **Stabilizer 预处理**：$C_i = S_\phi(B_i)$，使用冻结预训练 NAFNet（GoPro checkpoint），仅用于生成利于对齐的观察，不作为最终去模糊结果。
- **OCGM 生成掩码交叉视图 warp**：
  - 双向光流：$F_{1\to2} = \text{RAFT}(C_2, C_1), F_{2\to1} = \text{RAFT}(C_1, C_2)$。
  - 前向-后向残差：$e(x) = \|F_{b\to r}(x) + F_{r\to b}(x')\|_2$，其中 $x'=x+F_{b\to r}(x)$。
  - 运动自适应阈值：$T(x) = \tau + \alpha(\|F_{b\to r}(x)\|_2 + \|F_{r\to b}(x')\|_2)$，默认 $\tau=1.0, \alpha=0.01$。
  - 有效性掩码：$M_{br}(x) = \mathbb{1}[\text{in-bounds}(x')] \cdot \mathbb{1}[e(x) \le T(x)]$。
  - 掩码 warp：$W_{br} = M_{br} \odot \mathcal{W}(C_r, F_{br})$，$\mathcal{W}$ 为反向 warp。
- **2D 引导去模糊**：$\tilde{S}_1 = \mathcal{D}_\theta^{2D}(\text{concat}(B_1, W_{1\to2}, M_{1\to2}))$，$\tilde{S}_2$ 对称。网络为 NAFNet（base width 64），输入 7 通道（3 模糊 + 3 warp + 1 掩码）。

**Stage 2：全局 3D 引导与去模糊**
- 临时 3D 表示：$\mathcal{G} = h_\eta((\tilde{S}_1, K_1), (\tilde{S}_2, K_2))$，$h_\eta$ 为冻结的 NoPoSplat（RealEstate10K 预训练）。
- 输入视角重渲染：$R_i = \text{Render}(\mathcal{G}, v_i)$，$v_i$ 为主干 canonical 坐标系中的内部视角，无需真实外参。
- 3D 引导去模糊：$\hat{S}_i = \mathcal{D}_\psi^{3D}(\text{concat}(B_i, R_i))$，输入 6 通道（3 模糊 + 3 重渲染）。

**最终 3D 表示**：$\mathcal{G}^\star = h_\eta((\hat{S}_1, K_1), (\hat{S}_2, K_2))$，直接用于可微分 3DGS rasterizer 的新视角合成，无后续优化。

**训练细节**：两阶段依次训练，每阶段 400K 迭代 AdamW（lr=1e-3，wd=1e-3，$\beta_{1,2}=0.9$），余弦退火至 1e-7，batch=32/GPU，双 A100；损失 $\mathcal{L}=0.9\mathcal{L}_{\ell_1}+0.1\mathcal{L}_{\text{perceptual}}$（VGG19）。所有冻结模块（$S_\phi$, RAFT, $h_\eta$）全程不更新。

## 实验与结果
- **数据集与协议**：Deblur-NeRF 相机运动模糊子集，5 个合成 + 10 个真实场景；85 个固定两视角元组（25 合成 + 60 真实），每张图 resize + center-crop 至 $256\times256$；每个元组由两帧模糊输入和一路 held-out 目标视图构成。
- **评估基线**：SE-GS（逐场景优化）、GAURA（feed-forward，训练/推理用 8–12 视角）、CoherentGS（扩散先验 + 交替优化）、DAVANet + 共享 NoPoSplat、Difix3D+ + 共享 NoPoSplat。
- **主要结果（Table 1）**：
  - 真实场景：CasDeblurGS 获 **PSNR 21.13 / SSIM 0.700 / LPIPS 0.228**，较最强基线（CoherentGS: 19.90 / 0.660 / 0.292）提升 **+1.19 dB PSNR、+0.040 SSIM、-0.064 LPIPS**。
  - 合成场景：**PSNR 23.41 / SSIM 0.796 / LPIPS 0.165**，较最强基线（Difix3D+: 21.30 / 0.673 / 0.262）提升 **+2.11 dB PSNR、+0.109 SSIM、-0.094 LPIPS**。
- **Ablation（Table 2，真实）**：Blurry→+Stab(20.30)→+2D(20.88)→+3D(21.13)，Stage 1 贡献最大增量 +0.58 dB，Stage 2 再增 +0.25 dB。
- **跨视角对应分析（Fig.4）**：BlurBasket 匹配数从 69（模糊）→77（stabilizer）→96（Stage1）→112（Stage2）。
- **相机重投影分析（Table 3）**：Stage 2 有效三角化点数 +33%，1px inlier 数 +33%，中位重投影误差降低 9.7%，inlier ratio 0.9168→0.9343。
- **运行时（Supp Table 3）**：平均 65.52s，比 Difix3D+（70.85s）快 7.5%，比 CoherentGS（96.14s）快 31.8%；PSNR 高出 GAURA +4.48 dB。
- **额外诊断（Supp Table 2）**：去掉 stabilizer 直接从模糊估计光流下降 0.33 dB；纯 3D-only（跳过 Stage 1）仅 20.26 dB，证明局部 2D 引导不可替代。

## 相关工作脉络
- **NeRF/3DGS 模糊建模（BAD-NeRF、Bad-Gaussians、Crim-GS、Comogaussian）**：基于连续运动轨迹（SE(3)/Bézier/Neural ODE）联合优化模糊与几何，依赖大量视角冗余；本文在两张视角+无外参与轨迹优化无法工作的设定下提出 feed-forward 方案。
- **稀疏模糊 3DGS（HQGS、S2Gaussian、CoherentGS）**：前者依赖 2D 边缘/语义先验，后者依赖视频扩散先验与交替优化；本文完全不依赖外部生成先验，仅在推理时利用冻结主干重渲染提供全局 3D 反馈。
- **Feed-forward 3DGS（PixelSplat、MVSplat、GAURA、NoPoSplat、Pf3Splat、AnySplat）**：多数假设锐利输入；GAURA 虽处理模糊但依赖已知位姿且在两视角下未验证；本文在 pose-free 设定下将 feed-forward 3DGS 扩展到两视角模糊情形。
- **多视角去模糊（Disparity-aware 等方法）**：依赖视差/深度估计；在两视角模糊场景下高频率细节退化严重、可靠外参与深度不可用；本文以光流 + 遮挡掩码替代显式深度线索。
- **3D 一致扩散增强（3DENHANCER、SIR-DIF、Difix3D+）**：3D 一致性扩散在严重模糊导致几何匹配线索崩溃时脆弱；本文用冻结 3DGS 重渲染作为轻量 3D 一致性信号，避免推理时扩散的开销。

## 局限性与未来方向
- **仅适用于静态场景**：运动物体导致光流与重渲染不一致。
- **依赖已知相机内参**：未知内参下的推广未验证。
- **两视角极端设定下的对应引导可能退化**：在极度模糊、大遮挡或小重叠区域，OCGM 的可靠性下降。
- **作者声明的未来方向**：未知内参、动态场景、更广泛的稀疏视角设定。

## 研究启发与可借鉴点
- **"先本地 2D 再全局 3D"的级联范式具有迁移价值**：可在更多极端少视角任务（如双目/单目、含遮挡）中复用：先用局部对应建立可信引导，再经轻量 3D 主干回投影得到全局反馈。
- **OCGM 的设计可移植到其他多视角对齐任务**：运动自适应的前向-后向一致性阈值 + 掩码 warp 的组合对光流质量要求较高的 sparse-view matching 同样适用。
- **冻结 pose-free 3DGS 主干的"重渲染即引导"策略**：避免逐场景优化即可获取全局 3D 一致性信号，可拓展到其他 feed-forward 3D 表示（如 volume/pixel hybrid）作为 guidance source。
- **控制实验设计值得借鉴**：用同一 frozen backbone 串联不同 2D 去模糊器构造"去模糊+重建"基线（DAVANet/ Difix3D+），有效剥离 2D 恢复与 3D 级联增益；本团队在评估新 3D 方法时可借鉴该对照范式。
- **Camera reprojection analysis 作为 3D 一致性的间接验证**：通过 triangulate + reprojection error 量化 Stage 2 带来的几何一致性提升，比单纯 PSNR 更能支撑方法动机。

## 关键术语表
- **CasDeblurGS**：本文提出的级联 2D-to-3D 多视角一致性 3DGS 去模糊重建框架，从两帧模糊图像在无外参与无逐场景优化条件下输出新视角合成结果。
- **OCGM（Occlusion-aware Cross-view Guidance Module）**：基于冻结 stabilizer + 双向 RAFT 光流 + 前向-后向一致性掩码 + 运动自适应阈值的遮挡感知跨视角引导模块。
- **Pose-free Feed-forward 3DGS**：无需输入视角外参、直接从稀疏图像与其内参预测 3D Gaussian 集合的 feed-forward 重建范式（本文采用 NoPoSplat）。
- **NoPoSplat**：本文采用的 pose-free 3DGS 主干，以第一视角建立 canonical 坐标系直接预测共享空间内的高斯集合。
- **Forward-Backward Consistency Mask**：通过比较 $F_{b\to r}$ 与 $F_{r\to b}$ 往返残差剔除遮挡/错误匹配的像素有效性掩码。
- **Motion-Adaptive Threshold**：阈值 $T(x)=\tau+\alpha(\|F_{b\to r}\|+\|F_{r\to b}\|)$，随位移幅度自适应放宽一致性容忍度以避免过度拒真。
- **Deblur-NeRF**：包含 5 个合成 + 10 个真实场景的相机运动模糊神经渲染基准数据集，本文主要评测集。
- **Novel-View Synthesis（NVS）**：利用重建的 3D 表示在新观测视角下渲染图像的评估任务。

## 可复现要素
- **数据集**：Deblur-NeRF 相机运动模糊子集（5 合成 + 10 真实），评估用的 85 个固定元组映射见 Supplementary Tab.1；训练数据为 DeepDeblurRF 合成的 6DoF 运动模糊数据集（65 train + 10 val）。
- **代码/权重**：论文未明确说明开源；NoPoSplat checkpoint（RealEstate10K 预训练）、RAFT-Large（Torchvision）、NAFNet（GoPro checkpoint）均为公开权重；项目主页 https://haeyun-choi.github.io/Cascaded2D3D_page/。
- **关键超参**：OCGM 参数 $\tau=1.0, \alpha=0.01$；训练 lr=1e-3、wd=1e-3、$\beta_{1,2}=0.9$、余弦退火至 1e-7、400K 迭代、batch=32/GPU、双 A100；损失权重 $\lambda_{\ell_1}=0.9, \lambda_p=0.1$；NAFNet base width=64，编码器配置 [1,1,1,28]，解码器 [1,1,1,1]。
