---
title: "Towards Color-Faithful Low-Light Image Enhancement via Adaptive Color Debiasing and Saturation Rectification"
source: https://arxiv.org/pdf/2608.10512v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:00:38"
field: "低光照图像增强"
keywords: ["low-light image enhancement", "color correction", "adaptive cylindrical color transform", "color debiasing", "saturation rectification", "AdaLAB", "CAGE"]
innovations: ["提出 AdaLAB 圆柱自适应色彩空间和 AdaCCT 自适应圆柱色彩变换，实现颜色保真低光照增强", "前向变换通过色调定向色度去偏和亮度感知色度缩放抑制嵌入的颜色偏差", "反向变换通过超色域亮度补偿校正饱和度异常，避免直接裁剪导致的颜色伪影"]
benchmarks: ["LOLv1", "LOLv2-real", "LOLv2-synthetic", "SDSD-indoor", "SDSD-outdoor", "SID"]
---

# 论文速读：Towards Color-Faithful Low-Light Image Enhancement via Adaptive Color Debiasing and Saturation Rectification

## 一句话总结
本文提出 CAGE 框架，通过在圆柱自适应 LAB 色彩空间 AdaLAB 中引入自适应色彩去偏（forward transform）和色域协调的饱和度校正（inverse transform），实现低光照图像增强的颜色保真，有效缓解全局颜色偏差与局部饱和度异常问题。

## 研究问题与动机
- **低光照成像的颜色偏差问题**：低信噪比、相机硬件限制和机内自动处理流程常导致颜色偏移，直接提亮后嵌入的颜色偏差更加明显，难以同时恢复亮度与真实颜色。
- **现有 RGB 方法亮度-颜色耦合**：多数 LLIE 方法直接在 RGB 空间增强，亮度增强会导致更严重的颜色失真，如颜色偏移和饱和度异常。
- **仅解耦亮度与色度不够**：Retinex 分解或 HSV 类色彩空间虽能分离亮度和色度，但无法消除已嵌入的颜色偏差，偏差在整个增强管线中传播，最终表现为全局颜色偏差与局部欠/过饱和。
- **CYUV/HSV 的局限性**：YUV 分离不够充分，HSV 存在色调不连续和黑色平面噪声问题，HVI 的非可微变换使颜色轨迹一致性差；CIELab 提供更平滑的颜色轨迹和更一致的颜色偏差组织结构，但单纯对称可逆变换无法决定校正方向或调整失真幅度。

## 核心贡献（创新点）
- **提出 AdaLAB 圆柱自适应色彩空间**：L 和 AB 分别继承 CIELab 的亮度和两个对抗色度轴，提供解耦且图像特定的基础，支持均匀颜色偏移和色度平面缩放的建模。
- **设计 AdaCCT 自适应圆柱色彩变换**：通过学习色调偏移幅度和色度缩放强度，自适应预测色调偏移方向和亮度敏感性，实现前向变换的颜色去偏与反向变换的饱和度校正。
- **前向变换抑制嵌入颜色偏差**：通过色度平面平移和缩放重新组织颜色分布，在骨干网络增强前压制低光照嵌入的颜色偏差。
- **反向变换实现色域协调饱和度校正**：通过将超色域亮度补偿替代直接裁剪，将色度盈余转换为亮度补偿，避免高饱和区域的可见颜色偏差伪影。
- **通用兼容性与显著性能提升**：CAGE 可与多种 LLIE 骨干网络（Retinexformer、DarkIR、HVI-CIDNet）集成，在 LOLv1、LOLv2、SDSD、SID 等基准上实现一致的保真度提升。

## 方法详解
CAGE 的整体框架如下：

$$\hat{y} = G^{-1}(F_{\theta}(G(x)))$$

其中 $x$ 为输入低光照图像，$F_{\theta}$ 为增强骨干网络，$G(\cdot)$ 和 $G^{-1}(\cdot)$ 分别为前向和反向 AdaCCT 变换。

**（1）图像自适应圆柱参数预测**

从降采样输入 $x_{\downarrow128}$ 通过轻量级 CNN $H(\cdot)$ 提取紧凑表示 $z$：

$$z = H(x_{\downarrow128})$$

预测三类参数：

- **亮度敏感性顶点**：通过线性层预测 $p$ 个区间 logits $\{\hat{v}_k\}_{k=1}^{p}$，归一化后累积得到亮度采样顶点 $\{v_k\}_{k=0}^{p}$，满足单调递增。
- **色度缩放强度**：通过全局交互机制学习 $\{\hat{c}_k\}_{k=0}^{p}$，经指数衰减加权得到 $c_k$，softplus 激活确保正值：
  $$\tilde{c}_k = \sum_{j=0}^{p} \hat{c}_j e^{-\tau|j-k|}, \quad c_k = \text{softplus}(\tilde{c}_k + 1)$$
- **色调偏移方向与幅度**：线性层预测共享 2D 色调偏移方向向量 $d$，幅度 $\{m_k\}$ 同样通过全局交互学习，最终得到顶点级色调偏移向量 $\{d_k\}$：
  $$m_k = \sum_{j=0}^{p} \hat{m}_j e^{-\tau|j-k|}, \quad d_k = m_k \cdot d$$

**（2）前向变换（RGB → AdaLAB）**

- **色调定向色度去偏**：将 RGB 转换到 LAB 空间后分离亮度图 $l_l$ 和色度图 $u_l$，根据 $l_l$ 对 $\{d_k\}$ 进行线性插值得到 $d_l$，计算相似度权重 $s_l$ 后移位色度图：
  $$d_l = \text{Interp}(l_l; \{v_k\}, \{d_k\}), \quad \tilde{u}_l = u_l - s_l \cdot d_l$$
- **亮度感知色度缩放**：插值得到 $c_l$，缩放色度图获得 AdaLAB 表示：
  $$\hat{u}_l = c_l \cdot \tilde{u}_l, \quad \hat{x}_l = [l_l; \hat{u}_l]$$

**（3）反向变换（AdaLAB → RGB）**

- **逆转色度缩放**：从增强后的亮度图 $\hat{l}_h$ 插值得到 $c_h$，还原色度：
  $$\tilde{u}_h = \hat{u}_h / (c_h + \epsilon)$$
- **超色域亮度补偿（OOGLC）**：先按 $\hat{l}_h$ 裁剪色度得到 $u_c$，将超出色域部分转换为亮度补偿：
  $$l_h = \hat{l}_h + \gamma \|u_c - \tilde{u}_h\|_2, \quad u_h = \text{GamutClipping}(u_c, l_h)$$
- 最终输出：$\hat{y}_h = [\alpha_l l_h; \alpha_c u_h]$，转换为 RGB。

**（4）训练目标**

$$L_{\text{total}} = \lambda \cdot L(\hat{y}_{\text{AdaLAB}}, y_{\text{AdaLAB}}) + L(\hat{y}, y)$$

其中 $y_{\text{AdaLAB}}$ 是将 GT 映射到基础 LAB 空间并应用输入低光照图像预测的亮度感知色度缩放得到（不含色调偏移）。

## 实验与结果
- **数据集**：LOLv1（485 训练/15 测试）、LOLv2-real（689/100）、LOLv2-synthetic（900/100）、SDSD-indoor（1655/308）、SDSD-outdoor（2650/500）、SID（2099/598），以及 DICM、LIME、MEF、NPE、VV 等无参考数据集。
- **评估指标**：PSNR、SSIM、LPIPS（成对数据集）；BRISQUE、NIQE（无参考数据集）。
- **基线方法**：RetinexNet、KinD、DRBN、MIRNet、ZeroDCE、EnlightenGAN、LLFlow、SNR-Net、BreaD、FourLLIE、LLFormer、QuadPrior、CWNet、Retinexformer、DarkIR、HVI-CIDNet 等。
- **主要结果**：
  - 在 LOLv1 上，CAGE 与 Retinexformer 集成后 PSNR 达 27.09 dB（+1.01 dB），SSIM 0.871，LPIPS 0.090；与 DarkIR 集成后 PSNR 26.03 dB（+0.71 dB）；与 HVI-CIDNet 集成后 PSNR 27.25 dB（+1.22 dB）。
  - 在 LOLv2-real 上，CAGE+Retinexformer 达 24.10 dB（+2.21 dB）。
  - 在挑战性数据集 SDSD-indoor 上，CAGE+Retinexformer 达 29.98 dB（+1.78 dB），CAGE+DarkIR 达 30.36 dB（+0.42 dB）。
  - 在 SID 上，CAGE+Retinexformer 达 22.64 dB（+0.59 dB）。
  - 额外参数量仅 0.07M（约 HVI-CIDNet 的 3.5%），FLOPs 增加不足 0.01G（小于 HVI-CIDNet 的 0.1%）。
- **消融实验**：
  - 完整的圆柱自适应参数（色调偏移+色度缩放+亮度敏感性）效果最佳，移除色调偏移导致 PSNR 下降 0.49 dB。
  - AdaLAB 色彩空间优于 RGB、YUV、HVI 和基础 LAB。
  - 超色域亮度补偿优于简单 RGB/LAB 色域裁剪。
- **无参考评估**：在 DICM、LIME、MEF、NPE、VV 上，CAGE 在 15 个骨干-数据集组合中有 12 个降低 BRISQUE、14 个降低 NIQE。
- **主观评估**：12 位参与者超过 1500 次评分，CAGE 集成方法在所有三个骨干网络上均获得更高平均主观评分。

## 相关工作脉络
- **Retinex-based 方法**（如 RetinexNet、Retinexformer）：通过分解光照与反射率解耦亮度恢复与结构保持，但不显式校正嵌入的颜色偏差，偏差在增强过程中传播。
- **YUV 类方法**（如 BreaD）：线性分离亮度与色度，但分离不够充分，色度校正受限于光照主导。
- **HSV/HVI 类方法**（如 HVI-CIDNet）：强分离亮度与色度，但存在色调不连续、黑色平面噪声、非可微变换等问题，导致颜色轨迹不稳定。
- **3D LUT 色彩变换**（如 AdaICF）：支持内容感知的颜色映射，但主要用于通用图像调整而非低光照特定颜色恢复。
- **HVI-CIDNet+**：基于 HVI 空间的改进，但仍受限于 HSV 的结构缺陷。
- **本工作定位**：CAGE 不同于仅解耦亮度和色度的方法，显式建模并校正嵌入的颜色偏差，通过前向变换抑制偏差、反向变换校正饱和度异常，提供了更稳定的颜色保真增强方案。

## 局限性与未来方向
- **共享色调偏移方向的局限性**：当前方法预测单一图像级色调偏移方向，适用于颜色偏差方向全局一致的场景；对于混合光照下空间分离区域存在不同颜色偏移方向的情况，相同方向分配可能导致局部残余颜色偏差。
- **未来方向**：将局部色度内容纳入方向预测，允许相似亮度的区域获得不同的色调偏移方向，以处理空间变化的颜色偏差模式。

## 研究启发与可借鉴点
- **色彩空间选择策略**：CIELab 相比 HSV 提供更平滑的颜色轨迹和更一致的颜色偏差组织结构，在颜色保真任务中值得优先考虑。
- **亮度感知参数化设计**：通过亮度敏感性顶点和线性插值机制，实现沿亮度轴的自适应参数分布，有效捕捉低光照下非均匀的颜色变化模式。
- **全局交互机制替代局部依赖**：采用指数衰减全局交互替代传统 1D LUT 的局部相邻样本依赖，提升跨亮度区间的参数一致性和泛化能力。
- **超色域补偿替代直接裁剪**：将超出有效色域的色度转换为亮度补偿而非直接截断，避免了高饱和区域的可见颜色偏差伪影，为色域处理提供了新思路。
- **即插即用架构设计**：CAGE 作为前向/反向变换模块，可与多种 LLIE 骨干网络无缝集成而保持其原始结构不变，提供了高度通用的增强框架。

## 关键术语表
**CAGE**：一种圆柱色彩校正框架，包含自适应色彩去偏和色域协调饱和度校正，用于颜色保真的低光照图像增强。
**AdaLAB**：圆柱自适应 LAB 色彩空间，继承 CIELab 的亮度与色度轴，支持图像特定的均匀颜色偏移和色度缩放建模。
**AdaCCT**：自适应圆柱色彩变换，包含前向和反向变换，实现 RGB 与 AdaLAB 之间的转换及颜色校正。
**色调定向色度去偏**：在 LAB 色度平面上沿预测的色调偏移方向进行移位，抑制嵌入的颜色偏差。
**亮度感知色度缩放**：根据亮度图插值得到缩放强度，重新组织不同亮度平面的颜色分布。
**超色域亮度补偿（OOGLC）**：将超出有效 RGB 色域的色度转换为亮度补偿，实现饱和度与亮度的平衡重建。
**全局交互**：通过指数衰减加权使每个亮度顶点与所有其他顶点交互，提升参数一致性。
**GT Mean Evaluation**：针对 LO Lv1 等小测试集的评估策略，将预测图像的灰度均值对齐到 GT 后再计算指标。

## 可复现要素
- **数据集**：LOLv1、LOLv2-real、LOLv2-synthetic、SDSD-indoor、SDSD-outdoor、SID、DICM、LIME、MEF、NPE、VV（均为公开数据集）。
- **代码/权重**：代码已开源，地址为 https://yangzhichen763.github.io/CAGE/；论文未提及预训练权重的具体下载链接。
- **关键超参**：亮度区间数 $p = 32$，相似度权重范围 $\delta_1 = 0.2$、$\delta_2 = 1.0$，超色域补偿比例 $\gamma = 1.0$，损失平衡系数 $\lambda = 1.0$，最终亮度/饱和度调整参数 $\alpha_l = 1.0$、$\alpha_c = 1.0$（LOLv1 上 $\alpha_l = 1.3$，LOLv2-real 上 $\alpha_l = 1.1$、$\alpha_c = 0.8$）。
- **训练环境**：NVIDIA A40 GPU，CUDA 11.6，PyTorch 1.13.0。
