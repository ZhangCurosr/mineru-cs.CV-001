---
title: "CasDeblurGS: Cascaded 2D-to-3D Multi-View Consistency for 3D Gaussian Splatting from Two Blurry Images"
source: https://arxiv.org/pdf/2608.10345v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:43:51"
---

# 论文速读：CasDeblurGS: Cascaded 2D-to-3D Multi-View Consistency for 3D Gaussian Splatting from Two Blurry Images

## 一句话总结
本文针对仅含两张运动模糊图像且无相机外参的极端稀疏场景，提出级联式 2D-to-3D 多视图一致性框架 CasDeblurGS，通过遮挡感知光流掩码构建局部可靠 2D 引导、再利用冻结的免位姿 3DGS 骨干重渲染提供全局密集 3D 引导，在无测试时优化、无外部先验的前提下实现高质量新视角合成与几何一致的 3D 重建。

## 研究问题与动机
- **核心问题**：仅凭两张运动模糊图像（已知内参）恢复连贯 3D 场景并生成新视角，且不依赖输入相机位姿、辅助清晰图像或逐场景测试时优化。
- **现有方法不足**：传统模糊神经渲染方法依赖大量多视图冗余（通常 20–40 张）与精确 SfM 位姿初始化，在两视图下严重不适定，易产生不稳定几何与漂浮伪影。
- **现有方法不足**：基于扩散先验或 2D 结构提示的方法（如 CoherentGS、HQGS）在极端稀疏约束下难以保证全局 3D 一致性，常出现结构畸变与跨视图不匹配。
- **现有方法不足**：通用 feed-forward 3DGS 方法假设输入清晰，对运动模糊敏感；已知稀疏去模糊方法（如 GAURA）仍需已知位姿且未验证两视图极限场景。

## 核心贡献（创新点）
- **提出无位姿级联 2D-to-3D 引导框架**：首次面向“两图+已知内参”极端设定实现端到端 feed-forward 重建，全程冻结骨干网络，无需测试时优化或外部生成先验。
- **设计遮挡感知跨视图引导模块（OCGM）**：结合稳相器与双向 RAFT 光流，引入运动自适应前向-后向一致性阈值生成有效性掩码，精准过滤模糊导致的遮挡与误匹配 warp。
- **级联局部 2D 与全局 3D 一致性**：Stage 1 通过掩码跨视图 warp 恢复中间图像，Stage 2 将中间结果送入冻结免位姿 3DGS 骨干并重渲染，获得密集全局 3D 引导完成最终恢复，二者互补提升渲染质量与几何一致性。
- **在 Deblur-NeRF 上建立新 SOTA**：真实场景 PSNR 提升 1.19 dB，合成场景提升 2.11 dB；重投影分析与对应点可视化证实了几何一致性的显著改善。

## 方法详解
- **整体架构**：输入为两张模糊图像及内参 $\{(B_i, K_i)\}_{i=1}^2$。全程冻结预训练稳相器 $S_\phi$（NAFNet）、光流估计器 RAFT-Large、免位姿 3DGS 骨干 $h_\eta$（NoPoSplat）；仅离线训练 2D 引导去模糊网络 $\mathcal{D}_\theta^{2D}$ 与 3D 引导去模糊网络 $\mathcal{D}_\psi^{3D}$，推理阶段完全无逐场景优化。
- **Stage 1 局部 2D 引导与去模糊**：
  1. **稳相预处理**：$C_i = S_\phi(B_i)$ 仅用于生成对齐友好的观测，不作为最终恢复结果。
  2. **双向光流估计**：$F_{1\to2} = \text{RAFT}(C_2, C_1)$，$F_{2\to1} = \text{RAFT}(C_1, C_2)$。
  3. **前向-后向一致性校验**：计算残差 $e(x) = \|F_{br}(x) + F_{rb}(x')\|_2$，采用运动自适应阈值 $T(x) = \tau + \alpha(\|F_{br}(x)\| + \|F_{rb}(x')\|)$ 生成有效性掩码 $M_{br}$，过滤越界、遮挡与大位移误匹配。
  4. **掩码跨视图 warp**：$W_{br} = M_{br} \odot \mathcal{W}(C_r, F_{br})$，与原始模糊图、掩码拼接后输入 $\mathcal{D}_\theta^{2D}$，得到中间恢复图 $(\tilde{S}_1, \tilde{S}_2)$。
- **Stage 2 全局 3D 引导与去模糊**：
  1. 将 $(\tilde{S}_1, \tilde{S}_2)$ 与内参输入冻结骨干 $h_\eta$，构建临时规范空间 3D 高斯表示 $\mathcal{G}$。
  2. 在规范坐标系内对两输入视角 $v_1, v_2$ 进行重渲染，得到全局 3D 引导图 $(R_1, R_2)$。
  3. 将原始模糊图与重渲染引导图拼接后输入 $\mathcal{D}_\psi^{3D}$，输出最终恢复图 $(\hat{S}_1, \hat{S}_2)$。
- **最终 3D 表示**：将 $(\hat{S}_1, \hat{S}_2)$ 再次传入同一冻结骨干 $h_\eta$，得到用于新视角合成的 3D 高斯表示 $\mathcal{G}^\star$。
- **训练与损失**：两阶段网络均使用 AdamW（lr=$10^{-3}$，weight decay=$10^{-3}$，400K 迭代，余弦衰减至 $10^{-7}$），损失函数为 $\mathcal{L} = 0.9\mathcal{L}_{\ell_1} + 0.1\mathcal{L}_{\text{perc}}$（VGG19 感知损失），Stage 1 训练完成后冻结再训 Stage 2。

## 实验与结果
- **数据集与协议**：Deblur-NeRF 数据集（5 合成、10 真实场景），固定 85 组两视图测试元组（25 合成 + 60 真实），图像统一缩放裁剪至 $256\times256$。
- **评估基线**：SE-GS、GAURA、CoherentGS、DAVANet+NoPoSplat、Difix3D+&NoPoSplat。
- **主要数值结果**：
  - **真实场景**：PSNR 21.13 / SSIM 0.700 / LPIPS 0.228，较最强基线（Difix3D+ 19.94 dB）**提升 +1.19 dB**。
  - **合成场景**：PSNR 23.41 / SSIM 0.796 / LPIPS 0.165，较最强基线（Difix3D+ 21.30 dB）**提升 +2.11 dB**。
  - 消融表明：稳相器(+0.23 dB) → Stage 1 2D引导(+0.58 dB) → Stage 2 3D引导(+0.25 dB) 递进有效；Stage 2 使有效三角测量点数增加约 33%，中位重投影误差降低 9.7%，内点率从 0.9168 升至 0.9343。
  - 推理平均耗时 65.52 秒，快于 CoherentGS（96.14s）与 Difix3D+（70.85s），在质量与速度间取得更好平衡。

## 相关工作脉络
- **轨迹优化型去模糊神经渲染**（Bad-NeRF、Bags、Crim-GS）：依赖多视图冗余与连续位姿轨迹优化，在 <10 视图下极不稳定；本文完全摒弃轨迹优化，改用冻结骨干+级联引导。
- **稀疏视图 feed-forward 3DGS**
