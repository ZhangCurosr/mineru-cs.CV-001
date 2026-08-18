---
title: "DreamFly-Causal-Memory-and-Receding-Horizon-Difusion-Plannin"
source: https://arxiv.org/pdf/2608.12308v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 11:28:18"
---

# 论文速读：DreamFly: Causal Memory and Receding-Horizon Diffusion Planning for Aerial Vision-Language Navigation

## 一句话总结
本文提出 DreamFly，一种基于离散扩散策略的无人机视觉-语言导航（Aerial VLN）闭环框架，通过因果对齐的历史记忆、滑窗扩散规划与解耦的显式终止模块（LiteStop），在部分可观测的城市三维环境中联合建模历史上下文、短程前瞻动作结构与可靠停障判断。

## 研究问题与动机
- **历史信息易丢失与泄漏风险**：无人机单步自第一人称观测视野受限，易遗忘已观察过的地标；若历史记忆构建未严格界定时序边界，当前步观测可能“泄漏”至历史分支，破坏闭环因果一致性。
- **单步预测缺乏前瞻，开环执行鲁棒性差**：仅预测下一步动作无法捕获短期动作依赖性；若直接开环执行整段动作块，则丧失对新增视觉反馈的响应能力。
- **终止信号耦合于动作生成导致风险不对称**：运动偏差可通过后续重规划修正，但提前停障会不可逆地结束导航回合；将终止视为普通动作 token 会模糊提前/推迟停障的差异化风险。
- **地面 VLN 无法直接迁移至空中平台**：无人机需协同控制水平位移、垂直升降与视角调整，且城市三维结构复杂、遮挡严重，对时序推理、3D 空间建模与在线误差校正提出更严格要求。

## 核心贡献（创新点）
1. **因果对齐的历史记忆机制**：构建仅包含当前决策步之前观测的固定容量记忆 $M_{<t}$，采用 read-before-write 协议严防当前步信息渗入历史分支，并通过轻量门控交叉注意力融合至当前视觉表征。
   *本质区别*：不同于 HAMT 保留全量历史或 GridMM 依赖显式空间网格投影，本文严格限定记忆的时序访问边界，使训练（专家轨迹前缀）与部署（在线累积）共享同一因果约束。
2. **滑窗扩散规划（Receding-Horizon Diffusion Planning）**：基于双向扩散骨干联合预测 K 步离散动作块，引入有效前缀监督与 horizon-aware 几何核加权；推理时仅执行首动作，收到新观测后立即重新规划。
   *本质区别*：区别于 ACT 的时间集成或 Diffusion Policy 的连续轨迹执行，本文将未来动作定位为辅助规划变量而非开环指令，在保留闭环反馈的同时显式建模短程动作依赖。
3. **LiteStop 解耦终止控制模块**：从扩散策略初始全掩码状态的 K×|A| logit 网格中提取规划表征，训练独立终止头估计停障概率，主策略完全冻结。
   *本质区别*：避免终止信号与运动动作在同一生成管线中相互干扰，以零参数扰动代价提供专用停障校准目标，显著降低提前停障的不可逆风险。

## 方法详解
- **问题设定**：给定语言指令 $I$ 与初始位姿 $s_0=(p_0,\psi_0)$，每步接收自第一人称 RGB 观测 $O_t$，从离散动作空间 $\mathcal{A}$（含 Stop、前后/左右/垂直/横向移动）中选择 $a_t$。成功标准为最终位置距目标 Euclidean 距离 < 20 m。
- **因果历史记忆构建**：
  - 使用冻结的 CLIPSeg 密集路由器与 OWLv2 区域路由器提取指令相关候选；OWLv2 区域特征直接映射至 CLIPSeg 特征空间：$\mathbf{f}(b)=\mathrm{Norm}\left(\sum_g \mathrm{area}(G_g\cap b)\mathbf{v}_g\right)$。
  - 候选经视觉/空间一致性关联为 active tracks，稳定证据晋升为 prototype，单次高置信度候选可单步写入。
  - 记忆槽固定 16 个，每槽表示为 $\mathbf{r}_{<t}^j=[\mathbf{e}^{j,\mathrm{anc}};\mathbf{e}^{j,\mathrm{pro}};\rho^j;\log(1+\delta^j)]$，含 anchor、prototype、存在标记与距上次更新步数。
  - 记忆融合：以当前视觉 token $\mathbf{Z}_t$ 为 query，记忆嵌入 $\mathbf{E}_{<t}$ 为 key/value 做掩码多头交叉注意力 $\mathbf{C}_t=\mathrm{MHA}(\mathbf{Z}_tW_Q,\mathbf{E}_{<t}W_K,\mathbf{E}_{<t}W_V;\pmb{\mu}_{<t})$，经门控残差更新 $\widetilde{\mathbf{Z}}_t=\mathbf{Z}_t+\mathbf{M}_{img}\odot\mathbf{G}_t\odot(\mathbf{C}_tW_O)$，初始化恒等映射避免破坏预训练分布。
- **滑窗扩散规划**：
  - 训练时将 K 个动作位置全部替换为 [MASK]，单次双向前向联合预测。损失函数为 $\mathcal{L}_{\mathrm{act}}=\frac{\sum_{t,h}v_{t,h}c_{t,h}\gamma^h\mathrm{CE}_\mathcal{V}(\mathbf{z}_{t,h},\chi(\bar{a}^\star_{t,h}))}{\sum_{t,h}v_{t,h}c_{t,h}}$，其中 $\gamma=0.7$ 强化近期动作，$c_{t,h}$ 为 CAR 几何核权重，$v_{t,h}$ 屏蔽超出轨迹长度的尾部 padding。
  - 推理采用单调原点采样器逐步去噪，最终解码得到动作块 $\hat{\mathbf{a}}_t=[\hat{a}_t^0,\dots,\hat{a}_t^{K-1}]$；仅执行 $\hat{a}_t^0$，获得 $O_{t+1}$ 后重新规划，形成 receding-horizon 闭环。
- **LiteStop 终止控制**：
  - 复用初始全掩码前向的 logit 网格 $\mathbf{H}_t^{(0)}\in\mathbb{R}^{K\times|\mathcal{A}|}$，经 LN+MLP 映射为停障标量：$\ell_t^{\mathrm{stop}}=g_{\mathrm{stop}}(\mathbf{H}_t^{(0)})$，$p_t^{\mathrm{stop}}=\sigma(\ell_t^{\mathrm{stop}})$。
  - 训练时冻结全部策略参数，仅优化 $\mathcal{L}_{\mathrm{stop}}$（正类权重 $\lambda_+=4.0$），标签由专家动作是否为 Stop 决定，不依赖地理成功信号。
  - 推理阈值 $\eta_{\mathrm{stop}}=0.50$，最终终止决策 $d_t^{\mathrm{term}}=d_t^{\mathrm{stop}}\vee\mathbb{I}[\hat{a}_t^0=a_{\mathrm{stop}}]$，优先在动作执行前拦截。

## 实验与结果
- **数据集与划分**：基于 OpenFly 基准，经四项标准化处理（修正 Forward 6m 编码、重映射非标准标签、剔除预设关键帧、记录相对位姿元数据）。训练集 85,785 轨迹 / 1,356,622 决策步；测试集 1,796 轨迹（test-seen 1,392，test-unseen 404，覆盖 UE BigCity、UE SmallCity 与多个 AirSim 城市环境）。
- **评估基线**：Random、Action Sampling、Seq2Seq、CMA、AerialVLN、OpenFly-Agent。
- **主要指标**：NE↓、SR↑、OSR↑、SPL↑。
- **核心结果**：
  - DreamFly 在 test-seen 取得 SR=32.04%、SPL=28.22%、NE=44.87m；在 test-unseen 取得 SR=29.46%、SPL=23.54%、NE=45.29m，全面超越所有对比方法。
  - 相对最强基线 OpenFly-Agent，seen/unseen 的 SR 分别提升 +9.41pp / +15.35pp，SPL 提升 +7.80pp / +11.05pp，NE 显著降低。
- **消融结论**：逐条引入 Causal Memory、Action Chunk、LiteStop
