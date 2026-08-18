---
title: "FLEET-Token-Based-Feature-Extraction-for-Event-Camera-based"
source: https://arxiv.org/pdf/2608.16523v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:17:59"
field: "事件相机强化学习感知表征"
keywords: ["Event Camera", "Reinforcement Learning", "Feature Extractor", "Cross-Attention", "Random Fourier Features", "Tokenization"]
innovations: ["Perceiver交叉注意力将变长事件序列压缩为固定大小隐token，解耦计算成本与传感器分辨率", "纯端到端RL奖励驱动训练，无需辅助重建损失，表征与控制目标对齐", "RFF时空编码+时间门控融合，突破MLP频谱偏置，精确解析高频空间细节"]
benchmarks: ["Event-CarEnv Racing", "Event-CarEnv Parking", "Event Inverted Double Pendulum"]
---

# 论文速读：FLEET: Token-Based Feature Extraction for Event Camera-based Reinforcement Learning

## 一句话总结
FLEET 是一种面向事件相机强化学习的端到端可训练特征提取器，通过随机傅里叶特征编码与 Perceiver 风格交叉注意力机制，直接将原始异步事件序列压缩为固定大小的隐式 token，解耦了推理计算量与传感器分辨率，在所有对比任务上超越了现有最强基线。

## 研究问题与动机
1. **稀疏事件数据与现有架构的模态鸿沟**：事件相机输出异步、稀疏、不规则的时空事件流，标准深度学习 CNN 架构针对密集同步图像设计，无法直接处理；现有方法将事件聚合为稀疏张量（如 RVT），导致控制频率下运动模糊破坏时序精度，高频率下计算成本随传感器分辨率线性/二次增长。
2. **生成式基线的方法论缺陷**：eVAE 等变分自编码器依赖离线轨迹数据预训练，其辅助重建损失使表示偏向视觉保真度而非控制相关特征，产生目标函数错位；此外两阶段冻结架构在离线数据无法覆盖在线部署动态时面临严重分布偏移。
3. **缺乏轻量级端到端方案**：现有工作要么基于网格聚合浪费算力，要么依赖辅助预训练任务，尚无线性复杂度、可与 RL 联合优化的事件序列直接处理架构。

## 核心贡献（创新点）
1. **空间分辨率解耦的特征提取器**：提出 Perceiver 风格交叉注意力机制，将变长事件序列映射到固定数量的可学习隐向量，使后续主干处理的计算复杂度与传感器分辨率完全无关；与已有网格化 CNN 的本质区别在于避免了像素级离散化与算力分辨率耦合。
2. **任务对齐的表示学习**：纯基于 RL 奖励信号进行端到端训练，无需辅助重建损失；与 eVAE 等生成式方法相比，表征完全由时序差分更新驱动，避免了目标错位问题。
3. **频谱偏置缓解**：引入随机傅里叶特征（RFF）编码事件坐标，突破标准 MLP 的高频平滑偏置，可精确解析小障碍物等高频时空细节；与位置编码或可学习 LUT 的本质区别在于 RFF 无需训练即可逼近任意高频函数。
4. **新基准与频率鲁棒性验证**：构建 Event-CarEnv（基于 CarEnv 扩展的事件相机自动驾驶基准），系统证明 FLEET 在观测频率变化（$\Delta t \in \{0.1, 0.05, 0.025\}$ s）下性能稳定，而 CNN 基线显著退化。

## 方法详解
- **事件序列采样与定长化**：在采样窗口 $\tau$ 内，将原始事件序列 $\mathcal{E}_\tau$（长度 $D$）等分成 $M$ 个 bin，每个 bin 随机采样一个事件构成定长序列 $\mathcal{E}_{\text{batched}}$（长度 $M$）；若事件不足 $M$ 个则补零并 mask。
- **随机傅里叶特征编码**：事件的时间 $t \in \mathbb{R}$ 和空间坐标 $\mathbf{u} \in \mathbb{R}^2$ 归一化至 $[-1,1]$ 后，通过 RFF 映射到 $d_{\text{embed}}$ 维空间：$\text{RFF}(\mathbf{x}) = [\sin(\pi \mathbf{x} B), \cos(\pi \mathbf{x} B)]$，其中 $B_{ij} \sim \mathcal{N}(0, \sigma^2)$，时空分量使用不同缩放系数（$\sigma_{\text{temporal}} = 5.0$，$\sigma_{\text{spatial}} = 20.0$）。
- **时间门控融合**：时间嵌入经线性投影 + sigmoid 生成门控向量，与空间嵌入做逐元素乘积：$\mathbf{z}_k = \text{RFF}(\mathbf{u}_k) \otimes \text{sigmoid}(W_{\text{gate}} \text{RFF}(t_k) + \mathbf{b}_{\text{gate}})$，实现时空信息的动态耦合。
- **Perceiver 交叉注意力降维**：$N$ 个可学习隐向量作为 Query，嵌入事件序列作为 Key/Value，经多头交叉注意力（$h=4$ 头）压缩为 $N$ 个隐 token；使用 padding mask 忽略填充事件。
- **MLP 主干与全局平均池化**：$N$ 个 token 经三层 MLP 编码器（宽度 $2d_{\text{embed}}, 2d_{\text{embed}}, d_{\text{embed}}$）后，全局平均池化为单维特征向量。
- **端到端 RL 集成**：特征向量与状态向量拼接后输入 PPO Actor/Critic；对 Actor 输入施加 stop-gradient，特征提取器参数仅由 Critic 的价值函数损失更新，避免 Actor-Critic 目标冲突。

## 实验与结果
- **数据集/环境**：自建 Event-CarEnv（含 Racing 竞速与 Parking 泊车两个场景，96×64 分辨率，仿真事件相机基于 ESIM 模型生成）；Event Inverted Double Pendulum（128×128 分辨率静态相机）。
- **评估指标**：回合回报的区间均值（IQM）与 95% 置信区间（CI95），50000 次 bootstrap 迭代，10 个随机种子。
- **主要结果（Event-CarEnv Racing，$\Delta t=0.05$，$10^6$ 步）**：FLEET IQM = **46.96**，大幅领先所有基线；DMRCNN 第二（12.94，仅为 FLEET 的 27.6%）；NatureCNN（RGB）为 23.89，FLEET 超越 RGB 变体 **~50%**。
- **Parking 场景**：FLEET IQM = **5.67**，ImpalaCNN 达 5.18（91% 相对水平），事件密度更高时 CNN 基线表现改善。
- **Inverted Double Pendulum**：FLEET IQM = **6390.41**（远超事件基线 <1200），但仍低于 NatureCNN（RGB）的 8980.69（因该任务全局视野，天然有利于密集帧感知）。
- **分辨率不变性**：FLEET 在 96×64 至 384×256 四档分辨率上 IQM 保持 40.48–46.96，GFLOPs 恒为 83.28，VRAM 恒为 4.09 GB；384×256 下推理速度比 DMRCNN 快 310×，比 NatureCNN 快 400×。
- **频率鲁棒性**：在 Racing 场景 $\Delta t \in \{0.1, 0.05, 0.025\}$ 三档下，FLEET IQM 维持在 40.33–46.96；同条件下 NatureCNN 从 19.02 骤降至 5.87，DMRCNN 从 21.00 降至 6.78。

## 相关工作脉络
1. **RVT/Grid-based CNNs**（Gehrig & Scaramuzza, 2023；Xu et al., 2024 DMRCNN）：将事件聚合为稀疏时空张量后使用 CNN 处理；FLEET 完全跳过聚合步骤，直接处理原始事件序列，解耦了计算量与分辨率。
2. **eVAE**（Vemprala et al., 2021）：基于 VAE 的两阶段预训练表示学习方法，依赖离线轨迹数据且目标函数偏向视觉重建；FLEET 端到端训练，无需预训练，表征完全对齐控制任务。
3. **Event Transformer**（Sabater et al., 2022）：监督分类任务中的 Perceiver 交叉注意力方案，但仍需对聚合张量进行 patchify 和位置编码；FLEET 直接作用于原始事件，跳过网格化与 patchification。
4. **LFF-DS**（Schier et al., 2023）：基于 Deep Sets 的事件集合编码器，在 CarEnv 上有基础表现但无法学习有效策略；FLEET 引入交叉注意力后性能大幅提升，证明 token 化机制的关键作用。
5. **事件相机 SNN 方法**（Kou et al., 2025 等）：利用脉冲神经网络处理事件流，但训练困难且依赖专用神经形态硬件；FLEET 采用标准 MLP + 注意力架构，可在通用 GPU 上高效运行。

## 局限性与未来方向
1. **Sim-to-Real 差距未验证**：所有实验均在仿真环境（CarEnv + ESIM 事件模型）中进行，尚未在真实事件相机上部署或验证策略迁移性。
2. **事件序列截断的信息损失**：当单步事件数远大于 $M$ 时，需丢弃部分事件（按 bin 随机采样），在密集场景下可能遗漏关键事件；论文提及将探索高效的采样与过滤策略。
3. **高对比度静态场景下事件稀疏**：在低动态环境中事件流稀少，固定长度 $M$ 的采样可能导致大量填充事件，影响效率（虽在 Parking 场景中仍表现良好）。
4. **未来方向**：智能事件采样/过滤以缩减序列长度；闭环 sim-to-real 迁移；扩展至更多复杂机器人控制任务。

## 研究启发与可借鉴点
1. **"序列直接处理 + 交叉注意力降维"范式**：对任何稀疏非结构化感知数据（如激光雷达点云、异步触觉传感器），均可借鉴"原始数据 → RFF 编码 → 交叉注意力 → 固定 token"的 pipeline，避免强制网格化的信息损失。
2. **Stop-gradient 解耦特征提取器与 Actor**：仅通过 Critic 的 TD 误差更新表征参数，避免 Actor-Critic 目标冲突，这一设计对任何端到端 RL 视觉骨干均有参考价值。
3. **任务对齐表征优于重建表征**：生成式预训练 + 下游微调的两阶段范式在 RL 中可能因分布偏移和目标错位而失效；纯奖励驱动的端到端训练值得在其他感知-控制联合任务中验证。
4. **RFF 在坐标编码中的有效性**：对于需要精确定位信息（如障碍物边界、曲率检测）的控制任务，RFF 相较于可学习 LUT/标准位置编码能更稳定地保留高频细节，可作为坐标嵌入的默认选择。
5. **频率鲁棒性评估的必要性**：控制频率变化是现实系统的核心需求，将多频率鲁棒性纳入基准评估体系，可比单一固定频率的实验更具说服力。

## 关键术语表
- **Event Camera**：异步事件相机，像素级响应亮度变化并仅在超过阈值时输出事件（时间戳、坐标、极性），具有微秒级延迟和高动态范围。
- **Random Fourier Features (RFF)**：通过高斯采样的随机投影将低维坐标映射到高维正弦/余弦特征空间，使 MLP 能无偏地学习高频函数。
- **Cross-Attention（Perceiver）**：用固定数量的可学习 Query 向量对变长输入序列进行注意力压缩，输出维度与输入长度无关。
- **Temporal Gating**：用时间嵌入经 sigmoid 生成的门控向量对空间嵌入进行逐元素调制，实现时空信息的动态融合。
- **RVT (Recurrent Vision Transformer)**：将事件聚合为稀疏时空张量后通过 CNN/Transformer 处理的代表方法，计算量与传感器分辨率耦合。
- **eVAE**：基于变分自编码器的事件相机表示学习方法，需离线轨迹数据预训练，以重建损失为优化目标。
- **IQM (Interquartile Mean)**：去极值后的平均值，作为 RL 算法比较的稳健聚合指标，减少极端种子对结论的影响。
- **Event-CarEnv**：本文构建的高吞吐量事件相机自动驾驶 RL 基准，基于 CarEnv 模拟器扩展 RGB 与事件相机传感器。

## 可复现要素
- **数据集**：Event-CarEnv（基于 CarEnv [43] 自制），Racing 和 Parking 两个场景；Event Inverted Double Pendulum（自建）。
- **代码开源**：是，公开于 https://github.com/tgottwald/FLEET（MIT 风格仓库）。
- **权重**：论文未单独提供预训练权重，代码开源支持完整复现。
- **关键超参**：$M = 2048$（序列长度），$d_{\text{embed}} = 128$，$N = 32$（隐 token 数），$h = 4$（注意力头数），$\sigma_{\text{temporal}} = 5.0$，$\sigma_{\text{spatial}} = 20.0$，PPO 学习率 $3 \times 10^{-4}$，MLP 隐层 256 单元 × 2 层，训练步数 Racing/Parking $10^6$，IDP $5 \times 10^6$。
- **框架**：JAX + Flax。
- **硬件**：NVIDIA RTX 3090 GPU，32 GB RAM。
