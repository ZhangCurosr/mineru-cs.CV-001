---
title: "FUSE-Active-Functional-Afordance-Grounding-through-Adaptive"
source: https://arxiv.org/pdf/2608.12683v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:59:22"
field: "具身主动感知与功能定位"
keywords: ["affordance grounding", "active perception", "3D Gaussian Splatting", "semantic-geometric exploration", "embodied AI"]
innovations: ["提出主动功能可供性定位新任务", "设计语义-几何联合不确定性建模与自适应快-慢双模式切换框架"]
benchmarks: ["Habitat桌面功能定位基准"]
---

# 论文速读：FUSE-Active-Functional-Afordance-Grounding-through-Adaptive

## 一句话总结
本文提出**主动功能可供性定位（Active Functional Affordance Grounding）**新任务及**FUSE**框架，通过自适应切换快速 amortized 规划与显式语义-几何探索，在有遮挡的具身场景中高效定位满足功能查询的物体。

## 研究问题与动机
- **功能理解需求**：具身智能体需根据物体的功能而非身份进行交互，但现有方法在功能线索被遮挡或不完整时缺乏主动获取观测的决策机制。
- **被动方法的局限**：现有可供性定位方法（如 CRAFT、CRAFT-E）仅从固定视角进行预测，可能在证据不足时就做出错误判断。
- **主动识别任务的差异**：现有主动感知/搜索方法通常针对预定义类别或实例，而功能定位需从语言查询生成候选假设并持续获取证据直到可靠定位。
- **计算效率瓶颈**：完全显式探索需在每步更新3D空间表示并重新计算不确定性，计算开销大；完全 amortized 规划虽快但在模糊状态下易出错。

## 核心贡献（创新点）
1. **提出主动功能可供性定位任务**：定义了一个新的序列感知任务，要求智能体在部分可观测场景下根据功能查询主动发现并空间定位目标物体，而非针对预定义类别。
2. **构建 Habitat 基准与评估协议**：发布基于程序化组装桌面场景的 benchmark，包含静态、规范视角、VLM 主动、搜索式、amortized 及自适应方法等多条评测轨道。
3. **FUSE 自适应证据获取框架**：设计置信度门控机制，在 amortized 快速规划（低熵时）与显式语义-几何探索（高熵时）之间自适应切换，平衡精度与计算成本。
4. **语义-几何联合不确定性建模**：将功能不确定性分解为 SAM3 语义证据（候选物体定位）与 SSNR 几何不确定性（3DGS 重建偏差），融合形成证据获取图指导视角选择。

## 方法详解
**问题建模**：给定功能查询 $q$、初始观测 $x_0$ 和可行感知动作集 $\mathcal{A}$，目标是最小化未解决的功能不确定性：
$$\min_{T, a_{1:T}, o^*} \mathcal{U}_{\text{func}}(\mathcal{O}_q, \mathcal{B}_T) + \lambda \sum_{t=1}^{T} C(a_t, \mathcal{B}_t), \quad \text{s.t. } \mathcal{V}(o^*, \mathcal{O}_q, \mathcal{B}_T) \geq \tau$$

**功能不确定性分解**：
$$\mathbf{U}_t^{\text{func}}(\mathcal{O}_q) = \mathbf{U}_t^{\text{sem}}(\mathcal{O}_q) + \mathbf{U}_t^{\text{geo}}$$
- **语义不确定性**：将候选假设输入 SAM3，得到像素级掩码和置信度，构建语义证据图 $\mathbf{S}_t$，$\mathbf{U}_t^{\text{sem}} = 1 - \mathbf{S}_t$。
- **几何不确定性**：比较当前观测 $x_t$ 与 3DGS 渲染图像 $\hat{x}_t$ 的梯度差异，计算局部结构信噪比（SSNR），反转后归一化为 $\mathbf{U}_t^{\text{geo}}$。

**证据获取图**：
$$\mathbf{E}_t(\mathcal{O}_q) = \mathbf{S}_t(\mathcal{O}_q) + 1 - \text{Norm}\left(\frac{\|\nabla x_t\|^2}{\|\nabla x_t - \nabla \hat{x}_t\|^2 + \epsilon}\right)$$

**空间证据表示**：使用增量 3D Gaussian Splatting (3DGS) 累积多视角观测，支持跨视角推理、可见性/遮挡预测。

**终止准则**：融合 SAM3 置信度 $s_{t,i}$ 与深度邻近度 $d_{t,i}$（在 0.3m–1.2m 范围内最佳），$r_{t,i} = 0.7s_{t,i} + 0.3d_{t,i}$，连续 $K=5$ 步无改善则停止。

**Amortized Planner**：基于 CLIP ViT-B/16 提取图像/文本特征，经三层 MLP ($1056 \to 512 \to 512 \to 1$) 预测动作值，训练目标为动作后的证据增益（语义+几何+可见性+定位+距离）。

**FUSE 自适应门控**：计算动作分布熵 $H_t$，当 $H_t \leq \eta=0.8$ 时选择 amortized 动作，否则触发 3DGS 联合优化并执行显式探索。

## 实验与结果
**数据集**：基于 Habitat-Sim v0.3.3 的程序化桌面场景，6 个独立场景配置，100 个随机 episode，每 episode 含 1 目标 + 7–14 干扰物，初始视角保证目标部分遮挡。

**评估指标**：Success Rate（IoU ≥ 0.5）、Avg. IoU、Avg. Steps、Target Pixel Ratio、Center Score、Distance Score、Final SAM3 Score。

| 方法 | Success ↑ | Avg. IoU ↑ | Steps ↓ | Final SAM3 Score ↑ |
|------|-----------|------------|---------|-------------------|
| Static | 42.00% | 41.25% | 0 | 48.20% |
| Random Active | 56.00% | 55.03% | 12.20 | 58.17% |
| VLM Active | 65.00% | 64.01% | 7.65 | 58.15% |
| Explicit Exploration | 70.00% | 68.77% | 8.30 | 64.20% |
| Amortized Planner | 66.00% | 64.98% | 8.50 | 62.69% |
| **FUSE** | **72.00%** | **70.91%** | **9.32** | **66.61%** |
| Oracle-Label FUSE | 77.00% | 75.86% | 9.98 | 72.36% |

**核心结果**：
- FUSE 取得最高非 oracle 成功率 **72%**，优于 Explicit Exploration (70%) 和 Amortized Planner (66%)。
- 相比 Explicit Exploration，FUSE 在保持相近精度下实现 **1.33× 加速**，仅 33.41% 决策调用显式探索，每 episode 平均 2.97 次 3DGS 更新。
- **知识源鲁棒性**：FUSE 在不同上游知识源（CRAFT-E、Gemini、GPT、Llama 等）下均有效，CRAFT-E 取得最高非 oracle IoU (70.91%)。

**消融**：
- 语义仅（53%）< Random Active（56%），证明几何不确定性是必要补充。
- 几何仅（65%）> 语义仅，说明减少几何不确定性提供更强的探索信号。
- 像素级 SAM3（70%）远优于 patch-level CLIP（34%），强调细粒度语义的重要性。
- 深度邻近度加入后 success 提升约 5pp，避免过早终止于远距离弱分辨率视角。

## 相关工作脉络
- **CRAFT/CRAFT-E**：神经符号可供性定位方法，从单视角预测任务相关物体/部件，但假设已有充分视觉证据，不解决"证据不足时应看哪里"的问题。
- **Next-Best-View/主动抓取**：如 Lei et al. (2026)、Shi et al. (2025)、Tong et al. (2026)，目标已在先验确定，聚焦改进已有目标的感知或抓取策略，而非从无到有的目标发现。
- **Active Object Search**：Chaplot et al. (2020) 等任务针对预定义类别搜索，目标身份已知；本文目标由查询诱导，身份需在探索中涌现。
- **3D Gaussian Splatting 主动感知**：Kerbl et al. (2024) 等利用 3DGS 进行显式场景重建与主动探索，目标是通用场景覆盖，本文聚焦功能导向的语义-几何联合证据获取。
- **VLM 主动感知**：VLM Active 使用 Gemini 控制相机，虽能用少量动作达到 65% 成功率，但其 Final SAM3 Score (58.15%) 显著低于 FUSE (66.61%)，表明"看向目标中心"不等于"获取功能定位所需证据"。

## 局限性与未来方向
- **仿真环境限制**：实验仅在 Habitat-Sim 中进行，尚未在真实机器人平台验证。
- **场景类型受限**：当前为静态桌面场景，未涉及动态环境或复杂室内布局。
- **功能查询依赖上游知识源**：最终成功率上限受候选假设质量制约（oracle 达 77%，CRAFT 仅 53%），FUSE 无法补救知识源遗漏的正确目标。
- **计算开销仍集中在 SAM3**：SAM3 grounding 占 58.94% 总延迟，是主要瓶颈。
- **未来方向**：真实机器人部署、动态环境扩展、更丰富操作任务、长程 agent 联合学习功能推理/主动感知/动作执行。

## 研究启发与可借鉴点
1. **语义-几何联合不确定性建模**：将功能不确定性分解为语义证据（SAM3）和几何重构偏差（SSNR），为"证据在哪不足"提供空间化量化指标，可迁移至其他主动感知任务。
2. **置信度门控的快-慢双模式切换**：用 entropy 作为不确定性代理，自动决定何时调用轻量预测何时触发昂贵显式推理，是一种高效的资源分配策略，可推广至其他主动决策系统。
3. **功能导向而非身份导向的目标发现**：候选标签由查询诱导（如"cut"→knife/cutter 假说），而非搜索预定义类，对开放世界具身理解有启示意义。
4. **深度邻近度终止准则**：将距离先验（0.3m–1.2m）融入停止判断，避免"看到但太远无法操作"的假阳性，可应用于其他视觉-grounded 任务。
5. **与不同知识源解耦**：证明感知规划模块与候选生成模块可独立优化，便于模块化升级上游推理器。

## 关键术语表
- **Affordance（可供性）**：物体对智能体行为潜能的感知属性（如"可抓握"、"可坐"），取决于智能体能力与物体特性。
- **Functional Affordance Grounding**：将功能语言查询映射到场景中满足该功能的物体/部件的空间定位。
- **3D Gaussian Splatting (3DGS)**：基于可微渲染的增量 3D 场景表示方法，用于存储和更新空间证据。
- **SSNR (Structural Signal-to-Noise Ratio)**：局部结构信噪比，通过比较观测与渲染图像梯度衡量几何重构不确定性。
- **Amortized Planning**：通过离线训练的神经网络直接预测感知动作价值，避免在线重复优化，加速决策。
- **Candidate Hypothesis**：由功能查询诱导的潜在目标物体假设集合，身份未知，需在探索中验证。
- **Explicit Exploration**：每步更新 3D 空间表示并重算语义-几何不确定性的完整主动搜索策略。
- **Entropy Gate**：基于动作分布熵的阈值门控，判断何时从快速 amortized 模式切换到显式推理模式。

## 可复现要素
- **数据集**：基于 Habitat-Sim v0.3.3 + iTHOR 场景的 6 个桌面场景配置，论文声明将发布基准和评估协议。
- **代码/权重**：论文声明"will release"，附录提供了硬件/软件版本、3DGS 参数、COLMAP 设置等细节。
- **关键超参**：熵阈值 $\eta = 0.8$，停止耐心 $K = 5$，感知预算 $T = 30$，3DGS 更新迭代数 400（初始 1000），CLIP ViT-B/16 冻结，MLP 学习率 $10^{-3}$，batch size 64，20 epochs。
