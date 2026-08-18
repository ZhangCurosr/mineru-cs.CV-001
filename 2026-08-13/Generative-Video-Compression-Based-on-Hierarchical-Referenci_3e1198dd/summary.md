---
title: "Generative-Video-Compression-Based-on-Hierarchical-Referenci"
source: https://arxiv.org/pdf/2608.11618v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 11:51:10"
field: "生成式视频压缩"
keywords: ["generative video compression", "diffusion model", "hierarchical reference structure", "perceptual quality", "neural video coding", "flow matching"]
innovations: ["首次统一将分层引用结构应用于潜码编码与扩散生成重建两阶段", "提出HTCM从短程局部细节与长程全局语义中提取互补时序上下文", "设计HA-Adapter通过分层注意力掩码阻断低质量帧向高质量帧的伪影传播"]
benchmarks: ["HEVC B", "UVG", "MCL-JCV"]
---

# 论文速读：Generative-Video-Compression-Based-on-Hierarchical-Referenci

## 一句话总结
本文提出 **GVCHR**（Generative Video Compression based on Hierarchical Referencing），首次在基于扩散模型的视频压缩框架中统一设计**分层引用结构**：在潜码阶段将层级引用与层级质量结构耦合，在生成重建阶段通过分层注意力适配器防止低质量帧的伪影传播，从而在多个基准上取得最先进压缩性能（相比 GNVC-VD，LPIPS BD-rate 提升 50.5%，DISTS 提升 54.0%）。

---

## 研究问题与动机

1. **潜码阶段缺乏引用-质量耦合设计**：现有扩散视频压缩方法（如 DiffVC、GNVC-VD）要么采用简单的逐帧引用链，要么引入帧级质量变化但未显式关联帧的引用角色，导致真正重要的参考帧未能分配更多比特。
2. **生成重建阶段伪影传播严重**：解码后的有损潜帧被注入视频 DiT 时，通常采用全注意力或逐帧条件注入，允许低质量高帧任意参与跨帧交互，使伪影在去噪过程中放大。
3. **分层设计仅被部分探索**：传统视频编码和保真导向 NVC 中广泛使用的分层引用/质量结构，在扩散 GVC 中尚未被系统性地贯穿编码与生成两个阶段。
4. **感知质量与编码效率的矛盾**：现有方法在极低码率下往往因缺乏有效的参考机制而导致伪影累积，限制了生成先验潜力的发挥。

---

## 核心贡献（创新点）

1. **统一的分层引用框架**：首次将分层引用结构同时应用于潜码编码与生成重建两个阶段，使高质量参考帧在编码时获得更多比特、在去噪时被优先复用，本质区别在于以往方法仅在编码侧做质量分配、生成侧完全忽略层级依赖。
2. **分层时间上下文挖掘（HTCM）**：提出从短程参考（最近同层/下层帧的局部细节）与长程参考（所有最低层帧的全局语义记忆）中提取互补时序上下文，区别于以往仅依赖最近邻或全帧引用的单一策略。
3. **分层注意力适配器（HA-Adapter）**：将编码侧的层级结构延伸至视频 DiT，通过分块注意力掩码限制每帧仅能 attend 到同层或更低层参考帧，从根本上切断低质量帧向高质量帧的伪影传播路径。
4. **多阶段训练策略**：设计了从纯潜码训练→协同适配扩散模型→像素域重建优化的 6 阶段训练流程，实现生成先验与编码失真的渐进对齐。
5. **SOTA 实验结果**：在 HEVC B、UVG、MCL-JCV 三个基准上，相比最新方法 GNVC-VD，平均 LPIPS BD-rate 降低 50.5%、DISTS 降低 54.0%，同时保持较快的编解码速度。

---

## 方法详解

### 整体架构
GVCHR 由**分层潜码编码器**和**基于扩散的生成重建模块**组成：
- 输入视频 V 分离为关键帧与灰度中间帧（颜色信息在生成阶段恢复，节省码率）。
- 3D VAE 编码器将视觉信号转为潜帧序列 $\{\mathbf{x}_t\}_{t=0}^{1+T/4}$。
- 分层潜码器 + HTCM 对潜帧进行条件编码，输出重建潜帧 $\{\hat{\mathbf{x}}_t\}$ 及其特征 $\{\hat{\mathbf{f}}_t\}$。
- HA-Adapter 将解码后的潜帧与特征注入视频 DiT，通过分层注意力引导去噪，最终由 VAE 解码器得到像素域重建 $\hat{\mathbf{V}}$。

### 分层引用结构
将潜帧序列组织为多层（文中测试 2/3/4 层），其中：
- 低层帧被更高频率地复用为参考帧，分配更多比特（质量更高）。
- 高层帧编码更紧凑、 distortion 更大，但不作为更低层的参考，从而限制误差累积。
- 以 GOP=32（潜 GOP=8）、潜 mini-GOP=4 为例，权重配置为 $[w_t] = [0.5, 0.9, 0.5, 1.2]$，对应层级 ID $\{L_t\} = \{2,1,2,0\}$。

### HTCM：分层时间上下文挖掘
对当前潜帧 $\mathbf{x}_t$：
- **短期上下文** $\mathbf{c}_t^S = \hat{\mathbf{f}}_s$，其中 $s = \max\{i \mid i < t, L_i \le L_t\}$，即最近的同层或更低层帧特征，承载局部细节。
- **长期上下文**：利用**门控槽注意力（GSA）**对历史最低层帧特征 $\{\hat{\mathbf{f}}_l \mid L_l=0\}$ 进行递归更新，维护固定大小的记忆 $\mathbf{m}_l$，编码层 ID、帧索引和 QP 的元嵌入 $\mathbf{e}_l$。
- 长程表示通过 cross-attention 与短期上下文对齐：$\mathbf{c}_t^L = \text{Attention}(\mathbf{W}_q^C(\mathbf{c}_t^S + \mathbf{e}_s), \mathbf{W}_k^C \mathbf{m}_l, \mathbf{W}_v^C \mathbf{m}_l)$。
- 最终由上下文细化网络融合 $\mathbf{c}_t^S$ 与 $\mathbf{c}_t^L$，输出条件特征 $\mathbf{c}_t$ 与熵估计参数 $\mathbf{c}_t^e$。

编码公式：
$$\mathbf{y}_t = g_a(\mathbf{x}_t \mid \mathbf{c}_t), \quad \hat{\mathbf{y}}_t = \lfloor \mathbf{y}_t \rceil, \quad \hat{\mathbf{x}}_t, \hat{\mathbf{f}}_t = g_s(\hat{\mathbf{y}}_t \mid \mathbf{c}_t)$$
$$\mu, \sigma = g_{ep}(\mathbf{y}_t \mid \mathbf{c}_t^e)$$

### HA-Adapter：分层注意力生成控制
- 控制信号嵌入器：将 $\hat{\mathbf{x}}_t$ 与 $\hat{\mathbf{f}}_t$ 拼接后 patchify 为帧级 token $\{\hat{\mathbf{x}}_t'\}$。
- **分块注意力掩码** $\mathbf{M}$ 约束每帧仅 attend 到同层或更低层参考帧：
$$\mathbf{M}_{i,j} = \begin{cases} -\infty & \text{if } L_i < L_j \\ \mathbf{0} & \text{if } L_i \ge L_j \end{cases}$$
- 该设计阻止低质量高层帧污染高质量低层帧的去噪过程，同时利用 block attention 加速推理。
- 去噪流程采用 flow-matching 形式：$\mathbf{z}_{\tau-1} = \mathbf{z}_\tau + (\sigma_{\tau-1}-\sigma_\tau) \cdot \hat{\epsilon}(\mathbf{z}_\tau, \tau, \mathbf{u})$，其中 $\mathbf{u} = \text{HA-Adapter}(\{\hat{\mathbf{x}}_t\}, \{\hat{\mathbf{f}}_t\})$。

### 多阶段训练（Table I）
| 阶段 | Iter | 帧数 | LR | 训练模块 | RD损失项 |
|------|------|------|-----|----------|----------|
| 1-3 | 30K/20K/20K | 5→17→33 | $10^{-4}$ | 潜码器 | $L_{Codec}$ |
| 4 | 30K | 33 | $10^{-4}$ | 潜码器+HA-Adapter | $L_{Codec}+L_{CFM}$ |
| 5 | 40K | 33 | $5\times10^{-5}$ | 潜码器+HA-Adapter | $0.1 L_{Codec}+0.1 L_{CFM}+L_{Rec}+0.1 L_{LPIPS}$ |
| 6 | 60K | 33 | $2\times10^{-5}$ | 潜码器+HA-Adapter | $0.025 L_{Codec}+L_{Rec}+0.1 L_{LPIPS}$ |

五个 λ 值：[160, 289, 635, 1397, 3072]；QP offset 配置：$w_0=8.0$ 对应 offset 8，mini-GOP 内 $[0.5,0.9,0.5,1.2]$ 对应 offset $[0,2,0,4]$。

---

## 实验与结果

### 数据集与评估协议
- 训练集：从 Pexels 筛选约 480K 视频（VBench 美学分 >4.5，CLIP-IQA 清晰度 >0.65，16:9 比例）。
- 测试集：HEVC B、UVG、MCL-JCV（每序列前 96 帧，GOP=32 设置）。
- 评估指标：LPIPS、DISTS、FID、FloLPIPS（时序一致性）、PSNR、MS-SSIM；码率单位 BPP。
- 锚点：以开源 GLC-Video 为 BD-rate 计算基准（因 VTM 与 GVC 方法 RD 曲线重叠不足）。

### 主要结果（Table II）
相比 GLC-Video：
- **HEVC B**：LPIPS -63.7% / DISTS N.A. / FID -49.3% / FloLPIPS -30.7%
- **UVG**：LPIPS -65.0% / DISTS -74.0% / FloLPIPS -51.4%
- **MCL-JCV**：LPIPS -36.4% / DISTS -65.8% / FID -22.2% / FloLPIPS +9.7%
- **三个数据集平均 LPIPS BD-rate 提升 55.0%**

相比 GNVC-VD（Table III，以最新闭源方法为基准）：
- **平均 LPIPS BD-rate 降低 50.5%，DISTS 降低 54.0%**
- GOP 24 设置：LPIPS -47.0%，DISTS -49.1%
- 编解码速度：GOP 32 下 GVCHR 编码更快（396ms vs 439ms）、解码更快（2814ms vs 4225ms），得益于 block attention 加速。

### 消融实验（Table IV/V）
- 去除短程参考：LPIPS 退化 44.5%
- 去除长程参考：LPIPS 退化 21.4%
- 去除关键帧采样：LPIPS 退化 11.4%
- 3 层设置优于 2/4 层；仅分层质量无分层引用（DiffVC 式）效果次之；编码或生成侧移除分层引用均显著下降。
- 增大 GOP 尺寸可进一步提升性能，但解码时间随 attention 二次增长。

---

## 相关工作脉络

1. **GNVC-VD [13]**（CVPR 2026，最新 GVC 方法）：采用视频 DiT 在 GOP 内联合增强潜帧，但未探索分层引用与质量结构的统一设计；本文在其基础上引入层级感知，BD-rate 大幅领先。
2. **DiffVC [12]**：结合帧级潜扩散与图像扩散先验，使用了简单的逐帧引用链与继承的层级质量结构，但未将层级延伸至生成阶段；本文与之相比在引用结构与生成侧均有创新。
3. **DCVC-FM [7] / DCVC-DC [6]**：保真导向 NVC 的分层质量设计先驱；本文借鉴其质量分配思想，但将其与扩散生成重建统一，面向感知质量而非 PSNR。
4. **EHVC [17] / HLVC [30]**：NVC 中的分层多参考/分层质量研究；本文将其思想迁移至扩散 GVC 领域，并新增生成侧分层注意力机制。
5. **GLC-Video [22]**：开源 GVC 基线，使用 VQ-VAE 潜空间变换编码；本文以其为锚点评测，显示 GVCHR 在感知指标上的显著优势。
6. **VACE [10]**：提供预训练的 video DiT 与初始 adapter 权重；本文在其 adapter 基础上改造为分层注意力变体（HA-Adapter）。

---

## 局限性与未来方向

1. **GOP 越大解码延迟越高**：attention 计算复杂度随 GOP 平方增长，当前默认 latent GOP=8 为性能与效率的折中；长序列场景下仍需进一步优化。
2. **仅支持固定 GOP 结构**：分层配置（层级数、权重）为手动设定，缺乏自适应层级划分能力。
3. **闭源基线无法完整复现**：GNVC-VD 代码未公开，部分对比数值来自论文报告，可能存在统计波动。
4. **颜色恢复依赖关键帧数量**：灰度中间帧的颜色恢复能力受限于关键帧密度，极端压缩下可能产生色彩不一致。
5. **未来方向**：可扩展至更深的层级划分、自适应层级分配、结合 motion-aware 分层策略、以及向实时流式视频传输场景延伸。

---

## 研究启发与可借鉴点

1. **层级感知的质量-引用耦合设计**：将帧的"引用重要性"与"分配比特量"显式关联的思路可迁移至图像压缩、点云压缩等多粒度条件生成任务。
2. **短程+长程互补上下文提取**：HTCM 中 GSA 记忆机制 + cross-attention 对齐的方案，适用于长序列时序建模中局部细节与全局语义的联合编码。
3. **分层注意力掩码防止误差传播**：HA-Adapter 的分块 attention mask 设计是一种通用的"信息流隔离"技术，可用于任何需要分层条件注入的扩散生成模型。
4. **多阶段渐进训练策略**：从纯编码器→加入扩散适配→像素域优化的六阶段方案，是解决生成先验与编码失真对齐的有效范式，值得在其他 GVC 或生成压缩任务中借鉴。
5. **关键帧采样 + 灰度中间帧的颜色生成恢复**：这一码率节省技巧可与任意生成模型结合，进一步拓展超低码率视频生成的可行性。

---

## 关键术语表

**Generative Video Compression (GVC)**：利用生成模型（如 GAN/扩散模型）先验，在极低码率下重建 perceptually pleasing 视频的新兴压缩范式。

**Hierarchical Reference Structure**：将 GOP 内帧组织为多层，低层帧被高频复用为参考、高层帧不能反向引用低层的帧间依赖结构。

**Hierarchical Quality Structure**：与引用层级耦合的比特分配策略——越重要的参考帧分配越多比特、质量越高。

**Hierarchical Temporal Context Mining (HTCM)**：从短期参考（最近同层/下层帧局部细节）与长期参考（最低层帧的全局语义记忆）中提取互补时序条件的模块。

**Gated Slot Attention (GSA)**：一种线性时间复杂度的序列记忆更新机制，用于递归维护固定大小记忆以聚合历史信息。

**Hierarchical Attentive Adapter (HA-Adapter)**：将编码侧层级结构延伸至视频 DiT 的适配器，通过分块注意力掩码限制跨帧引用方向，防止伪影传播。

**Flow-Matching**：扩散建模的一种形式化方法，学习将高斯噪声沿连续轨迹映射至数据流形的速度场。

**BD-rate / BD-Metric**：Bjøntegaard 导出率失真参数，用于量化两条 RD 曲线的平均性能差异（百分比下降表示改进）。

---

## 可复现要素

| 要素 | 状态 |
|------|------|
| 训练数据集 | Pexels 筛选约 480K 视频（VBench >4.5，CLIP-IQA >0.65），非标准公开数据集 |
| 测试数据集 | HEVC B、UVG、MCL-JCV（公开） |
| 代码开源声明 | 论文未明确提及代码开源计划 |
| 权重开源声明 | HA-Adapter 与视频 DiT 使用 VACE [10] 预训练权重初始化；潜码器从零训练；论文未声明整体权重发布 |
| 关键超参 | λ ∈ {160, 289, 635, 1397, 3072}；latent GOP=8，mini-GOP=4；GOP=32；patch size 256×256/448×448；LR 1e-4→2e-5 衰减 |
| 硬件环境 | NVIDIA H20 GPU（推理测速） |
| 框架依赖 | VACE [10] 预训练权重、DCVC-RT [8] 潜码器架构 |

---
