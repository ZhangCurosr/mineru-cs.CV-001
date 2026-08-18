---
title: "HarmoniDPO-Video-guided-Audio-Generation-via-Preference-Opti"
source: https://arxiv.org/pdf/2608.11913v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 11:58:15"
field: "视频到音频生成"
keywords: ["video-to-audio generation", "diffusion models", "direct preference optimization", "cross-modal alignment", "RLHF", "test-time scaling"]
innovations: ["首次将Online-DPO引入V2A生成任务", "双视频特征表示保留时间动态", "双尺度扩散搜索测试时优化"]
benchmarks: ["VGGSound", "AVSync15"]
---

# 论文速读：HarmoniDPO-Video-guided-Audio-Generation-via-Preference-Opti

## 一句话总结
本文提出HarmoniDPO框架，首次将基于偏好的优化（Online-DPO）引入视频到音频（V2A）生成任务，通过双视频特征表示和偏好对齐显著提升了音频-视频同步性与主观感知质量。

## 研究问题与动机
1. **时间动态信息丢失**：现有V2A方法通常将视频压缩为单个特征表示，导致时间动态和细粒度视觉信息大量丢失。
2. **重建损失与感知质量脱节**：传统L1/L2重建损失无法有效捕捉人类听觉感知质量，难以区分"可接受"与"高质量"音频。
3. **一对多映射下的偏好选择缺失**：V2A具有一对多特性（同一视频可匹配多个合理音频），现有方法无法在多解中选择符合人类偏好的最优输出。
4. **离线DPO的分布偏移问题**：静态偏好数据集随模型进步迅速过时，导致策略分布与偏好模型分布产生偏移。

## 核心贡献（创新点）
1. **双视频特征表示**：结合全局视频上下文特征（InternVid）与细粒度逐帧特征（CLIP），保留时间动态和语义细节，本质区别于现有方法的单特征压缩。
2. **在线直接偏好优化（Online-DPO）**：首次在V2A任务中应用RLHF思想，通过迭代生成候选音频并基于自动奖励模型进行偏好对齐，区别于依赖预收集静态数据的离线DPO方法。
3. **VA-DPO损失函数**：引入奖励幅度感知机制，显式建模偏好强度而非仅做二元选择，比标准DPO更敏感地利用奖励差异信息。
4. **双尺度扩散搜索（DDS）**：提出测试时缩放算法，通过混合大小步长自适应探索潜在空间，平衡局部开发（exploitation）与全局探索（exploration）。

## 方法详解

### 1. 基础架构
以预训练的Tango-2文本到音频扩散模型为基础，冻结主模型参数，仅微调新增的投影头和交叉注意力层。音频信号经预训练VAE编码器压缩为低维潜在表示$z_0 = \mathcal{E}_\theta(x_a)$，在潜在空间进行去噪生成。

### 2. 双视频条件输入
- **全局特征**：InternVid视频编码器处理均匀采样的M帧，输出$f_g \in \mathbb{R}^{d_g}$
- **逐帧特征**：CLIP图像编码器处理N=250帧，输出$e_i \in \mathbb{R}^{d_f}$
- **特征融合**：投影对齐到公共维度$d_c$后拼接为$F = [f'_g; e'_1; \ldots; e'_N]$，经RoPE位置编码和多头自注意力机制处理时序关系，最终通过交叉注意力注入U-Net扩散模型

### 3. 可选文本条件
FLAN-T5文本编码器提取$f_t$，经投影后也通过交叉注意力融入生成过程。

### 4. Online-DPO流程
**奖励建模**采用三维自动评估：
- 音频-视觉对应性：CAV-MAE相似度$R_{av}$
- 音频-文本一致性：CLAP相似度$R_{at}$
- 内在音频质量：Meta Audiobox-Aesthetics预测器$R_{quality}$

综合奖励：$R(y) = w_{av}R'_{av}(y) + w_{at}R'_{at}(y) + w_{quality}R'_{quality}(y)$

**迭代训练**：
- 每轮从当前策略$\pi_t$生成N个候选音频
- 按奖励选取最优$y_w = \arg\max R(y_i)$和最差$y_l = \arg\min R(y_i)$构建偏好对
- 参考模型$\pi_{ref}$每轮训练后更新

### 5. VA-DPO损失
$$\mathcal{L}_{VA-DPO} = -\mathbb{E}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} - \lambda(R(y_w)-R(y_l))\right)\right]$$
其中$\lambda$平衡DPO项与奖励差距项。

### 6. DDS测试时缩放
维护候选集$P_0$，每迭代对每个候选生成两个变体：$x_s = \beta_s x + \sqrt{1-\beta_s^2}\eta$（保守步长）和$x_l = \beta_l x + \sqrt{1-\beta_l^2}\eta$（激进步长），按CLIP/CLAP评分择优保留。

## 实验与结果

### 数据集
训练：VGGSound（遵循官方train/test split）
评估：VGGSound + AVSync15（150个时间同步挑战性样本）

### 主要结果（VGGSound）
| 方法 | MKL↓ | CLIP↑ | FID↓ | FAD↓ | CLAP↑ |
|------|------|-------|------|------|-------|
| FoleyCrafter | 2.56 | 10.70 | 19.67 | 2.78 | 25.3 |
| Frieren | 2.58 | 11.83 | 12.48 | 3.32 | 24.7 |
| **HarmoniDPO (aligned + DDS)** | **1.82** | **13.65** | **6.42** | **1.59** | **32.57** |

完整模型相比FoleyCrafter：MKL降低28.9%，CLIP提升27.8%，FAD降低42.8%，CLAP提升28.9%。

### AVSync15结果
- MKL: 1.332（最佳）
- CLIP: 14.38（最佳）
- FID: 31.59（最佳）
- Onset Acc: 32.53（最佳）
- Onset AP: 69.97（最佳）

### 消融结论
- InternVid特征比CLIP平均池化MKL提升6.3%，比纯帧特征提升20.1%
- 双特征融合比单特征MKL降低0.515，CLIP提升23.0%
- 用户研究：8候选配置在OVL（3.92）和REL（3.95）上达到最佳平衡

## 相关工作脉络
1. **FoleyCrafter**：使用IP-adapter连接视频语义+音频事件检测，本文改用双视频特征表示，并通过偏好优化替代纯重建训练
2. **Diff-Foley**：采用对比学习CAVP编码器实现同步音频合成，本文引入InternVid时序感知编码+在线偏好对齐
3. **Tango-2**：本文使用的base模型，其DPO对齐思路被本文扩展至V2A领域并引入在线版本
4. **Diffusion-DPO**：首次将DPO应用于图像生成，本文将其推广至跨模态V2A任务并解决离线DPO的分布偏移问题
5. **RLHF在LLM中的应用**：InstructGPT等建立SFT→RLHF范式，本文首次将此范式引入音频生成并改造为online版本

## 局限性与未来方向
1. **数据规模限制**：VGGSound仅约20万样本，每视频仅10秒，限制长视频泛化能力
2. **单参考音频问题**：现有数据集仅提供单一参考音频，无法充分利用V2A的一对多特性
3. **自动奖励的局限性**：CAV-MAE/CLAP/Audiobox等指标可能与人类主观判断存在偏差
4. **未来方向**：构建更大规模长视频数据集、开发更精确的人类偏好模拟奖励模型、探索多模态偏好优化

## 研究启发与可借鉴点
1. **Online-DPO范式可迁移**：其"生成候选→自动评估→偏好优化"的闭环框架可复用于文本到图像、视频编辑等其他生成任务
2. **多维度自动奖励设计**：将AV对应性、文本一致性、内在质量解耦评估的思路，可指导其他跨模态任务的质量优化
3. **测试时缩放策略**：DDS的双尺度搜索机制为扩散模型推理阶段提供了无需额外训练的计算效率优化方案
4. **双特征融合模式**：全局+局部的互补表征设计可推广至视频到音乐、视频到3D等其他跨模态生成任务

## 关键术语表
**Video-to-Audio (V2A) Generation**：从视频生成与之同步的音频的跨模态生成任务
**Latent Diffusion Model (LDM)**：在压缩潜在空间进行去噪扩散的生成模型
**Direct Preference Optimization (DPO)**：将偏好学习转化为二分类问题、无需显式奖励模型的对齐方法
**Online-DPO**：在训练过程中动态生成偏好对并迭代表对调整的DPO变体
**Audio-Visual Correspondence**：生成音频与输入视频在时序和语义层面的匹配程度
**Dual-scale Diffusion Search (DDS)**：通过混合大小步长自适应探索潜在空间的测试时优化算法
**CAV-MAE**：对比式音视频掩码自编码器，用于评估音频-视觉对齐质量
**CLAP**：大规模对比式语言-音频预训练模型，用于评估音频-文本一致性

## 可复现要素
- **数据集**：VGGSound（公开）、AVSync15（公开）
- **代码/权重**：论文未提及开源计划
- **关键超参**：学习率$1\times10^{-4}$、batch size 128、N=250帧、候选数8、$\beta_s/\beta_l$为不同步长参数、奖励权重$w_{av}=w_{at}=w_{quality}=1$
- **硬件**：64×H800 GPU
- **Base模型**：Tango-2（公开）、InternVid（公开）、CLIP/OpenCLIP（公开）、FLAN-T5（公开）
