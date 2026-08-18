---
title: "DreamFly: Causal Memory and Receding-Horizon Difusion Planning for Aerial Vision-Language Navigation"
source: https://arxiv.org/pdf/2608.12308v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 08:09:29"
---

# 论文速读：DreamFly: Causal Memory and Receding-Horizon Difusion Planning for Aerial Vision-Language Navigation

## 一句话总结
本文提出 DreamFly，一个基于 Dream-VLA 骨干的无人机视觉-语言导航（VLN）框架，通过**因果对齐的历史记忆**、**退视界扩散规划（plan-K, execute-one）**与**LiteStop 显式终止模块**，在严格时序边界下联合建模历史上下文、短期动作依赖与终止决策，显著提升了航拍 VLN 在部分可观测环境中的闭环导航性能。

## 研究问题与动机
- **核心问题**：航拍 VLN 是部分可观测的闭环序列决策任务，需在不丢失历史地标线索的前提下规划短期动作，并可靠判断导航是否结束。
- **现有方法不足**：
  1. 传统 VLA/自回归策略主要依赖当前观测与隐式隐状态，易丢失已观测地标；且历史记忆缺乏明确的时序访问边界，推理时可能出现当前步信息向历史分支泄漏。
  2. 单步动作预测缺乏 lookahead；而开环执行整个动作块会削弱闭环视觉反馈的在线纠错能力。
  3. 终止动作与运动动作的风险高度不对称：提前停止会不可逆地结束导航，但现有方法常将终止耦合在动作生成头中，缺乏独立的终止校准机制。
  4. 航拍环境具有大尺度 3D 结构、高度变化强耦合水平/垂直运动、视觉遮挡严重等特性，对跨步视觉推理与在线误差纠正提出更严苛要求。

## 核心贡献（创新点）
1. **因果对齐的历史记忆（Causally Aligned Historical Memory）**：历史记忆 $M_{<t}$ 仅由当前决策步 $t$ 之前的观测构建，采用 read-before-write 协议严禁当前步信息进入历史分支。与已有工作相比，现有记忆机制（如关键帧选择、网格投影）多聚焦“保留什么内容”，本文首次显式界定历史分支的**时序访问边界**以保证闭环一致性。
2. **退视界扩散规划（Receding-Horizon Diffusion Planning）**：基于双向扩散主干联合预测 $K$ 步动作块，但在线推理时仅执行第一步并依据新观测重新规划（plan-K, execute-one）。与 ACT/Bidirectional Decoding 等动作分块方法相比，本文利用离散扩散的并行生成能力捕获短期动作依赖，同时通过逐帧重规划严格维持闭环反馈。
3. **LiteStop 显式终止模块**：从扩散策略初始全掩码状态的 action logits 中抽取规划表示 $\mathbf{H}_t^{(0)}$，独立训练轻量级 MLP 头估计停止概率，且主导航策略全程冻结。与将 Stop 视为普通动作标记的生成式策略相比，该设计将终止决策与运动规划解耦，避免提前/延迟停止风险污染已学习的动作分布。

## 方法详解
- **问题设定**：离散动作空间包含 Stop、多步长前进、左转/右转、上升/下降、侧移。成功标准：最终位置与目标 Euclidean 距离 $<20$ m。
- **因果记忆构造**：
  - 使用冻结的 CLIPSeg 密集路由与 OWLv2 区域路由提取指令条件候选特征，OWLv2 区域特征映射至 CLIPSeg 特征空间并聚合。
  - 候选特征按视觉/空间一致性关联为 active tracks，稳定证据沉淀为 anchor+prototype 双表示，每步最多写入 2 个槽位。
  - 记忆槽固定 16 个、维度 512。决策步 $t$ 的记忆表示为 $\{(\mathbf{r}_{<t}^j, \mu_{<t}^j)\}_{j=1}^{16}$，仅当 $\mu_{<t}^j=1$ 时参与后续交叉注意力。
  - 记忆与当前视觉特征通过门控交叉注意力融合：$\widetilde{\mathbf{Z}}_t = \mathbf{Z}_t + \mathbf{M}_{\mathrm{img}} \odot \mathbf{G}_t \odot (\mathbf{C}_t W_O)$，其中 $W_O$ 与门控参数零初始化，保证训练初期记忆适配器退化为恒等映射。
- **退视界扩散规划**：
  - 训练时所有 $K$ 个动作位置替换为 [MASK]，仅对有效前缀（长度 $L_t = \min(K, T-t)$）计算 CrossEntropy，超长位置填充 Stop 并被 validity mask $v_{t,h}$ 排除在损失外。
  - 引入几何核重加权 CAR：$c_{t,h} = \max(10^{-6}, \sum_{i\in\mathcal{U}_t} \kappa_{i,q_{t,h}})$ 与 horizon decay $\gamma^h$，损失函数为 $\mathcal{L}_{\mathrm{act}} = \frac{\sum v_{t,h} c_{t,h} \gamma^h \mathrm{CE}_\mathcal{V}(\mathbf{z}_{t,h}, \chi(\bar{a}_{t,h}^\star))}{\sum v_{t,h} c_{t,h} \gamma^h}$，强调近端动作。
  - 推理时采用 monotonic origin sampler 迭代离散扩散，前向传播结束后缓存初始全掩码 logits 矩阵 $\mathbf{H}_t^{(0)} \in \mathbb{R}^{K\times|\mathcal{A}|}$。
- **LiteStop 解耦终止**：
  - 将 $\mathbf{H}_t^{(0)}$ 展平后通过 LayerNorm+MLP 输出标量 logit $\ell_t^{\mathrm{stop}}$，经 sigmoid 得停止概率 $p_t^{\mathrm{stop}}$。
  - 训练损失为类别不平衡加权的 BCE：$\mathcal{L}_{\mathrm{stop}} = -\frac{1}{|\mathcal{B}|}\sum [4.0\, y_t^{\mathrm{stop}}\log p_t^{\mathrm{stop}} + (1-y_t^{\mathrm{stop}})\log(1-p_t^{\mathrm{stop}})]$，主策略参数全程冻结。
  - 推理时先完成完整扩散生成，再读取缓存的 $\mathbf{H}_t^{(0)}$ 评估 $d_t^{\mathrm{stop}} = \mathbb{I}[p_t^{\mathrm{stop}} \geq \eta_{\mathrm{stop}}]$，最终终止信号为 $d_t^{\mathrm{term}} = d_t^{\mathrm{stop}} \vee \mathbb{I}[\hat{a}_t^0 = a_{\mathrm{stop}}]$。若未终止，仅执行 $\hat{a}_t^0$ 并获取 $O_{t+1}$ 重新规划。

## 实验与结果
- **数据集**：OpenFly。训练集含 20 个子集、85,785 条轨迹、1,356,622 个决策步。测试集：test-seen（UE BigCity + 6 个 AirSim 城市场景，1,392 条）与 test-unseen（UE SmallCity，404 条）。成功阈值 20 m。
- **评估指标**：NE（↓）、SR（↑）、OSR（↑）、SPL（↑）。
- **基线**：Random、Action Sampling、Seq2Seq、CMA、AerialVLN、OpenFly-Agent。
- **主要结果（Table 2）**：DreamFly 在 test-seen 与 test-unseen 上四项指标均最优：
  - test-seen：NE=44.87m，**SR=32.04%**，OSR=46.77%，**SPL=28.22%**（较最强基线 OpenFly-Agent 的 SR 22.63% 提升 +9.41pp，SPL 20.42% 提升 +7.80pp）。
  - test-unseen：NE=45.29m，**SR=29.46%**，OSR=46.78%，**SPL=23.54%**。
- **消融（Table 3）**：从 Dream-VLA 基线（SR 21.55%）逐步加入因果记忆（→24.11%）、动作块规划（→27.73%）、LiteStop（→31.46%），各项指标稳定上升；距离分区分析显示 LiteStop 对短距导航增益最大，历史记忆对中距导航贡献显著。

## 相关工作脉络
1. **OpenFly [4]**：本文实验基准平台，提供标准化航拍 VLN 轨迹与评估协议；本文在其公开 checkpoint 上直接对比 OpenFly-Agent，并复用其数据管线。
2. **Dream-VLA [33]**：本文策略骨干，提供双向扩散与离散动作分块生成能力；本文在其上引入记忆融合与退视界推理适配航拍场景。
3. **LongFly [8] / OpenFly-Agent [4]**：侧重历史视觉压缩与关键帧选择，解决“保留多少历史”问题；本文补充了“何时允许历史信息
