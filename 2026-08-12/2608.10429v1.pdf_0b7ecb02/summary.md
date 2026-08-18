---
title: "Lesion-Aware Adaptive Fourier Neural Operator for CT-to-PSMA PET Synthesis in Prostate Cancer"
source: https://arxiv.org/pdf/2608.10429v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:51:05"
field: "医学图像跨模态合成与定量评估"
keywords: ["CT-to-PET synthesis", "PSMA PET", "prostate cancer", "lesion-aware learning", "adaptive Fourier neural operator", "radiomics reproducibility", "total lesion activity", "proxy conditioning"]
innovations: ["以放射组学空间趋势为动机推导可微 CT 对比度与紊乱度双代理，推理期无需分割", "引入病灶级 TLA 对数归一化损失 + 瘤核对比度损失 + 指数距离加权瘤周损失联合监督", "以 ICC 瘤核/瘤周放射组学重复性与病灶 TLA 误差为补充指标，超越全容积 SSIM/PSNR 评测局限"]
benchmarks: ["TCIA PSMA-PET-CT-Lesions", "SSIM / PSNR / MAE", "SUVmax & SUVmean percentage error", "Total Lesion Activity (TLA) error", "Radiomics ICC (FO/GLCM/GLRLM/GLSZM)"]
---

# 论文速读：Lesion-Aware Adaptive Fourier Neural Operator for CT-to-PSMA PET Synthesis in Prostate Cancer

## 一句话总结
本文提出 LAFNO（Lesion-Aware Adaptive Fourier Neural Operator），通过两项 CT 衍生代理通道（对比度代理、紊乱度代理）和病灶级监督损失，引导 3D U-Net+AFNO 瓶颈结构从 CT 合成 PSMA-PET；在 TCIA 数据集上，全容积 SSIM 保持竞争力（$^{18}$F: 0.960，$^{68}$Ga: 0.938），同时将每例患者 TLA 误差降至 48.3% / 64.0%，并在肿瘤核心放射组学 ICC 上全面超越基线。

## 研究问题与动机
- **病灶体素占比极小却被全局损失淹没**：全身 PSMA-PET 中肿瘤体素仅占微小比例，但承载临床关键活性信号；使用 L1/MSE 等全局损失时，模型易偏向健康背景组织，即使 SSIM/PSNR 较高仍可能低估病灶活性或丢失肿瘤结构。
- **放射组学可作为生物学条件但提取成本高**：放射组学能把解剖影像转为可量化的肿瘤强度与纹理特征，直接用于条件化生成有潜力；然而传统提取依赖勾画的病灶 ROI，耗时长、流程依赖分割且存在观察者变异。
- **CT 软组织对比度有限，尤其是小结节与微小病变**：CT 难以区分细小或低摄取病灶对应的多种合理 PET 摄取模式（1 对多映射），导致基于单一 CT 强度的合成模型稳定性不足。
- **瘤周微环境（TME）信息被忽视**：既往工作多聚焦瘤内核，但侵袭与耐药不仅取决于肿瘤细胞本身，还受瘤周界面的纹理与密度异质性影响；缺少对 0–10 mm 瘤周带的显式监督不利于临床可信的合成图像。

## 核心贡献（创新点）
1. **以放射组学分析为发现步骤、以可微 CT 代理为落地路径**：先用 radiomics 在 TCIA PSMA-PET-CT-Lesions 上刻画“瘤核→瘤周”的衰减梯度与纹理异质性趋势，再将其替换为两个无需标注、可直接从 CT 计算的代理通道；与 Kim et al. 使用手工 radiomics 作为 GAN/diffusion 条件的方式相比，本文避免推理阶段任何分割与特征提取。
2. **LAFNO 的 AFNO 瓶颈处注入双代理**：对比度代理经 AvgPool 保留平滑梯度，紊乱度代理经 MaxPool 保留局部峰值，随后拼接到 256 维瓶颈特征并由 1×1×1 卷积投影；本质区别在于用物理尺度匹配的轻量前处理替代高维条件输入，仍保留长程谱混合能力。
3. **病灶级 TLA + 瘤核对比度 + 指数距离加权瘤周三项联合监督**：相比仅靠 L1 的全容积目标，该组合把优化重心从“整体相似度”转向“小病灶活性守恒与局部结构一致”；其中 TLA 项用对数压缩与分母归一化避免大灶主导梯度，这一设计区别于常见的逐体素肿瘤 mask L1。
4. **评测体系从全容积指标扩展到 lesion-level TLA 与 radiomics ICC**：除了 SSIM/PSNR/MAE，还报告 SUVmax/SUVmean 百分比误差、病灶占比阈值曲线，以及 FO/GLCM/GLRLM/GLSZM 在瘤核与 0–10 mm 瘤周的 ICC，填补了以往 CT→PET 合成多停留在视觉质量评测的空白。

## 方法详解
- **CT 放射组学分析驱动代理定义**：对 480 例患者的 10,209 个连通病灶，在瘤核与 0–5/5–10/10–20 mm 壳层提取 radiomics；发现沿核→周方向 firstorder_Mean / firstorder_Median / ngtdm_Contrast 单调递减，而 firstorder_Entropy 从核向 0–5 mm 壳显著上升并趋于稳定；这两条趋势直接对应 contrast 与 disorder 代理的设计动机。
- **对比度代理（contrast proxy）**：$C(\mathbf{x}) = \mathrm{CT}(\mathbf{x}) - G_{\sigma} * \mathrm{CT}(\mathbf{x})$，$\sigma=5$ mm，用可分离 1D 高斯卷积实现；模拟核到周的衰减梯度。
- **紊乱度代理（disorder proxy）**：$D(\mathbf{x}) = \overline{\mathrm{CT}^2}_w(\mathbf{x}) - \left(\overline{\mathrm{CT}}_w(\mathbf{x})\right)^2$，窗口 $w=7.5$ mm，用均匀滤波实现局部方差；捕捉瘤周纹理异质性。
- **LAFNO 架构**：3D U-Net 编码器（3 级 stride-2，$64^3 \to 8^3$，256 通道）+ AFNO 瓶颈（4 个 AFNO block，谱通道混合 + 残差 MLP）+ 转置卷积解码 + Sigmoid 输出；代理在瓶颈处注入：$\mathbf{b}' = W_{258 \to 256}[\mathbf{b} \| \mathrm{AvgPool}_8(C) \| \mathrm{MaxPool}_8(D)]$。
- **训练策略**：单卡 A100-40GB，200 epoch；Adam($lr=2\times10^{-4}, \beta_1=0.5, \beta_2=0.999$)，50 epoch 后线性衰减；70% patch 以病灶为中心（随机抖动）+30% 均匀采样，匹配滑动窗口推理分布。
- **损失函数**：$\mathcal{L} = \mathcal{L}_{\mathrm{L1}} + 0.05\,\mathcal{L}_{\mathrm{TLA}} + 0.02\,\mathcal{L}_{c} + 0.05\,\mathcal{L}_{\mathrm{peri}}$。
  - $\mathcal{L}_{\mathrm{TLA}}$：按连通病灶归一化对数 summed-SUV 误差，分母 $\log(1+N_k \cdot \mathrm{SUV}_{\max})$ 让不同大小病灶梯度可比。
  - $\mathcal{L}_c$：对预测与真值 PET 在 SUV 空间施加同式对比度算子，仅在肿瘤 mask 内取平均，约束瘤核内局部摄取结构。
  - $\mathcal{L}_{\mathrm{peri}}$：0–10 mm 瘤周环内指数距离加权 L1，权重 $w(\mathbf{x})=\exp(-d(\mathbf{x})/5\text{mm})$，靠近边界的体素惩罚更强。
- **数据预处理**：CT 裁剪到 [-1000,1000] HU 线性缩到 [0,1]；PET 转为 SUV 后作 log-normalize：$\mathrm{PET}(\mathbf{x}) = \log(1+\mathrm{SUV}(\mathbf{x}))/\log(1+\mathrm{SUV}_{\max})$；proxy 在重采样前计算，再以 99 百分位归一。

## 实验与结果
- **数据集**：TCIA PSMA-PET-CT-Lesions（Jeblick et al., 2026）；$^{18}$F-PSMA $n=335$（测试 $N=47$）、$^{68}$Ga-PSMA $n=204$（测试 $N=30$），含手动勾画肿瘤 mask。
- **基线**：AFNO-L1（同架构仅 L1）、3D Pix2Pix（PatchGAN）、FlowLet（3D Haar 小波流匹配）、cWDM（3D Haar 小波条件扩散）。
- **全容积指标**：LAFNO 在 $^{18}$F 上 SSIM=0.960、PSNR=35.98、MAE=0.0038；在 $^{68}$Ga 上 SSIM=0.938、PSNR=35.10、MAE=0.0048。略逊于 AFNO-L1/cWDM 的全局 SSIM/PSNR，属于有意权衡。
- **SUV 误差**：cWDM 在 $^{18}$F 上 SUVmax/SUVmean 百分比误差最低；LAFNO 处于竞争区间，$^{18}$F 的 SUVmax 误差 55.05%、SUVmean 8.61%；$^{68}$Ga 的 SUVmax 误差 23.60%、SUVmean 13.41%。
- **TLA（主要优势）**：LAFNO 在 $^{18}$F 上病灶级 |err|=52.7%、患者级 |err|=48.3%，≤25%/≤50%/≤75% 比例分别为 24.0%/50.9%/81.0%，均为最优；$^{68}$Ga 上病灶级 |err|=54.4%、患者级 |err|=64.0%，≤50%/≤75% 比例 31.3%/61.4% 同样领先。
- **Radiomics ICC（核心亮点）**：在所有 tracer×feature-class×region 组合中，LAFNO 在**肿瘤核心**取得最高 ICC；$^{18}$F 下瘤核 FO/GLCM/GLRLM/GLSZM 分别为 0.38/0.62/0.78/0.71，全面超过 AFNO-L1（0.24/0.39/0.59/0.51）。瘤周 0–10 mm 环在 $^{18}$F 上也普遍领先，但 $^{68}$Ga 呈现 tracer 依赖性混合结果（FO/GLCM/GLRLM 被 cWDM/FlowLet 超越）。
- **消融要点**：仅加 proxy 使患者级 TLA 误差从 62.2%→56.5%；再加 $\mathcal{L}_{\mathrm{TLA}}$ 降至 50.5%；再加 $\mathcal{L}_c$ 降至 47.8%，GLSZM 瘤核 ICC 从 0.624→0.760；最终加入 $\mathcal{L}_{\mathrm{peri}}$ 微调至 48.3%，对瘤周 GLSZM 0–3 mm 带 ICC 提升显著（0.637→0.849），但对其他特征影响有限。

## 相关工作脉络
- **Kim et al. (2025, WACV) 放射组学条件肿瘤合成**：用 GAN 生成肿瘤 mask、扩散模型按 size/shape/texture 条件合成纹理；本文沿袭“radiomics→条件信号”的思想，但以 CT 代理直接替代手工特征，省去推理期分割与特征提取。
- **Bhaskara & Oderinde (2025) AFNO 用于合成盆腔 CT**：验证 AFNO 瓶颈在跨模态医学图像翻译中的有效性；本文将其迁移至 CT→PSMA-PET，并在瓶颈处引入双代理与病灶损失，形成 LAFNO。
- **U-Net / Pix2Pix 系列医学图像到图像翻译**：传统 Encoder-Decoder+Skip 的通用范式；本文与之的差别在于用 AFNO 替代普通瓶颈 MLP，并以 lesion-aware 目标纠正全局重建偏差。
- **FlowLet (Danese et al., 2026)**：3D Haar 小波空间条件流匹配；作为生成类基线，在 SSIM/PSNR 上表现中等，但在病灶 TLA 与瘤周 ICC 上落后于 LAFNO。
- **cWDM (Friedrich et al., 2024)**：3D Haar 小波条件扩散；在全容积与 SUV 误差上有竞争力，但在病灶级 TLA 和瘤核 radiomics 保真度上不及 LAFNO，体现扩散/流匹配 vs. 重建+病灶监督的侧重差异。
- **Abtahi et al. (2026) Fine-UNet 分割与定量**：聚焦 PSMA-PET/CT 病灶分割与生存分层；本文与其互补——不依赖精确分割即可在合成过程中获得病灶级活性约束。

## 局限性与未来方向
- **数据集偏向高负荷、骨转移主导的可见病灶**：容易勾画的病灶被充分覆盖，而微小软组织灶、弱摄取结节与生理性摄取可能代表性不足。
- **proxy 无法完全刻画肿瘤异质性与亚型差异**：相同 CT 外观下不同组织学/代谢状态对应不同 PET 摄取模式（1 对多），单一线性代理条件存在上限。
- **$^{68}$Ga-PSMA 的整体表现弱于 $^{18}$F-PSMA**：更高正电子能量带来更长飞行距离与更严重的部分容积效应，加之多扫描仪、多重建参数的采集异质性，共同削弱该示踪剂的预测一致性。
- **瘤周 fidelity 仍具挑战性**：即使引入指数衰减加权 $\mathcal{L}_{\mathrm{peri}}$，紧邻边界的重复性仍未达最高，且加入瘤周项后瘤核 ICC 略有回落，多目标间平衡有待优化。
- **膀胱高摄取引起的溢出与噪声**：盆腔区 bladder activity 会干扰前列腺床附近合成质量；建议未来采用标准化膀胱管理（如导尿管引流）降低 spillover。
- **未来方向**：更大规模、富注释（含坏死/硬化/软组织/炎症/生理性摄取亚型）的数据集；MRI 或 CT/MRI 联合代理条件（图 10 已提示 T2-weighted MRI 上代理信号更强）；偏容积校正、tracer/扫描器标准化与距离相关瘤周效应的显式建模。

## 研究启发与可借鉴点
- **“放射组学发现 → 可微代理落地”的范式可迁移**：先在标注数据上做空间/纹理趋势分析，再用轻量化卷积/池化算子编码为额外通道，可避免推理期分割依赖；这一思路适用于其他 CT/MRI→功能成像的合成任务。
- **病灶级对数归一化 summed-activity 损失设计**：$\mathcal{L}_{\mathrm{TLA}}$ 的分母按 $N_k \cdot \mathrm{SUV}_{\max}$ 归一、分子做 log 压缩，能缓解大灶梯度主导与小灶梯度淹没问题，可推广至其他稀疏阳性体素场景（如肺结节、肝转移）。
- **AFNO 瓶颈 + 跨模态条件注入的组合**：若任务中存在低维但稳定的外部信号（如密度/梯度/距离场），可考虑在谱混合前先投影融合，比全分辨率通道拼接更节省显存。
- **指数距离加权瘤周损失的结构**：将监督强度随解剖距离指数衰减，既保留了边界附近的局部约束，又避免了远场背景噪声反向干扰；可用于任何需要兼顾“核心准确性”与“界面连续性”的合成任务。
- **多目标消融与 tracer 分层分析值得效仿**：论文按组件顺序累计消融，并结合 $^{18}$F/$^{68}$Ga 两组分别评估；这种“单一变量 + 跨示踪剂对照”的策略能清晰定位每部分的真实贡献与失败边界。

## 关键术语表
- **PSMA PET**：靶向前列腺特异性膜抗原的正电子发射断层显像，用于前列腺癌分期与复发监测的关键分子影像。
- **Synthetic PET**：由解剖模态（CT/MRI）经深度学习合成的伪 PET 图像，用于降低辐射剂量、缓解扫描仪供需压力。
- **LAFNO**：本文提出的病灶感知自适应傅里叶神经算子网络，将 CT 双代理注入 AFNO 瓶颈并以病灶级损失监督。
- **Adaptive Fourier Neural Operator (AFNO)**：在频域做通道间混合并接残差 MLP 的算子块，擅长捕捉长程空间依赖。
- **Total Lesion Activity (TLA)**：单病灶内 SUV 求和乘以体素体积的活性指标，反映病灶整体代谢负荷。
- **Contrast proxy**：CT 原始强度减去 $\sigma=5$ mm 高斯模糊后的残差通道，近似瘤核到瘤周的密度衰减梯度。
- **Disorder proxy**：CT 在 $w=7.5$ mm 滑动窗内的局部方差通道，刻画瘤周纹理异质性。
- **Radiomics reproducibility (ICC)**：真实 PET 与合成 PET 中提取的影像组学特征在受试者间的一致性，常用组内相关系数 ICC 衡量。

## 可复现要素
- **数据集**：TCIA PSMA-PET-CT-Lesions（Jeblick et al., 2026），公开；含 $^{18}$F-PSMA（335 例）与 $^{68}$Ga-PSMA（204 例）及手动肿瘤 mask。
- **代码/权重**：论文未提供开源声明与链接，未明确是否开源。
- **关键超参**：patch 尺寸 $64^3$；训练 200 epoch；Adam($lr=2\times10^{-4}, \beta_1=0.5, \beta_2=0.999$)，50 epoch 后线性衰减；70% 病灶中心 patch+30% 均匀采样；proxy 参数 $\sigma=5$ mm、窗口 $w=7.5$ mm；损失权重 $\lambda_{\mathrm{TLA}}=0.05$、$\lambda_c=0.02$、$\lambda_{\mathrm{peri}}=0.05$。
- **硬件**：单卡 NVIDIA A100-40 GB。
