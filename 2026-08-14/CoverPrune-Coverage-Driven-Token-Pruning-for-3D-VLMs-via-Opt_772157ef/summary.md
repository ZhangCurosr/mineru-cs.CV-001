---
title: "CoverPrune-Coverage-Driven-Token-Pruning-for-3D-VLMs-via-Opt"
source: https://arxiv.org/pdf/2608.13226v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:57:55"
field: "3D 视觉语言模型高效推理"
keywords: ["Visual Token Pruning", "Optimal Transport", "3D Vision-Language Models", "Spatial Reasoning", "Inference-time Acceleration"]
innovations: ["将3D VLM token剪枝从多样性最大化重新定义为最优传输驱动的视觉证据覆盖问题", "提出FST多维传输代价与信息感知容量的联合设计", "设计SGS贪心算法与轻量级块结构OT近似两种高效求解方案"]
benchmarks: ["ScanQA", "SQA3D", "Scan2Cap", "VSI-Bench"]
---

# 论文速读：CoverPrune-Coverage-Driven-Token-Pruning-for-3D-VLMs-via-Opt

## 一句话总结
CoverPrune 将 3D 视觉语言模型的推理时 token 剪枝从"最大化多样性"重新定义为"保留视觉证据覆盖"，通过最优传输（OT）框架在无训练前提下选出紧凑且代表性强的 token 子集。该方法在激进剪枝预算（10% tokens）下仍能保持接近完整 token 的空间推理性能，显著优于现有 SOTA 剪枝方法。

## 研究问题与动机
- **3D VLM 的 token 爆炸问题**：3D 视觉语言模型通过视频/多视图/3D 表示注入几何信号，单输入可生成数百至数千视觉 tokens，attention 的二次复杂度导致推理严重瓶颈。
- **多样性剪枝在 3D 场景下的失效**：现有方法（如 Diversity-based pruning）以特征相似性为冗余代理，剔除最相似 token 以最大化分散度；但在 3D 中，代表主流模式的"原型 token"与大量同类 token 相似，易被过早剔除，导致保留集偏向异常值而非代表性观测。
- **多视图一致性与几何结构破坏**：3D 环境中的 token 常编码重复的多视图观测，尽管在特征空间中看似冗余，但共同构成几何结构的重建证据；仅按相似性剪枝会破坏多视图对应关系，损害距离、顺序等空间推理。
- **注意力分数的不可靠性**：Attention-based 剪枝依赖早期层注意力质量，但注意力分数可能受 attention sinks 和 prompt 依赖性干扰，跨层/解码步的注意力模式不稳定。

## 核心贡献（创新点）
- **剪枝范式重构**：提出从"最大化 token 多样性"到"保留视觉证据覆盖"的范式转移，将剪枝建模为最优传输问题，使保留 token 作为原型分布表示质量以覆盖原 token 集。
- **FST 多维传输代价设计**：提出 Feature-Spatial-Temporal（FST）代价矩阵，联合建模语义相似度、3D 空间邻近性和时间一致性，确保选取的 token 同时满足语义对齐、几何完整与时间顺序。
- **半松弛 OT + 贪心子集选择**：将组合子集选择问题转化为半松弛 OT（Semi-Relaxed OT）形式，利用子模最大化性质设计 Spatial-Guided Greedy Selection（SGS）算法，在保留理论保证的前提下避免全局 OT 重算。
- **轻量级变体 CoverPrune-Lite**：基于 Morton 码空间填充曲线构建 3D 感知排序，将 token 分组后在每个组内做局部最优原型选择，将推理复杂度从 O(N³) 降至 O(N log N)，剪枝耗时仅 0.41s（vs. 2.53s）。
- **无需训练且即插即用**：方法完全 inference-time 训练无关，插入在 visual-geometric encoder 与 3D VLM 之间，跨 GS-Reasoner 和 VLM-3R 两个 SOTA 基座模型验证。

## 方法详解
**问题设定**：设输入视频提取 N 个视觉 token 集合 T = {t_j}，每个 token 含特征 f_j ∈ R^d、3D 坐标 x_j ∈ R^3、时间戳 τ_j。给定保留比 R，需选 K = ⌈RN⌉ 个 token 构成子集 S 以最大化覆盖函数 Cover(S; T)。

**OT 形式化**：将 S 视为源支撑（prototypes），T 视为目标支撑，定义代价矩阵 C(S, T)，通过 OT 计划 P 衡量两者差异：
- L_OT(S; T) = min_{P≥0} ⟨C(S,T), P⟩ s.t. P1 = u, P^T 1 = v
- 其中 u 为均匀源容量（原型平等），v 为信息感知目标容量（非均匀，反映 token 重要度差异）。

**FST 传输代价（Feature-Spatial-Temporal Cost）**：
- d_f(s,t) = 1 - cos(f_s, f_t)：特征空间余弦距离
- d_x(s,t) = ||x_s - x_t||_2：3D 欧氏距离
- d_τ(s,t) = ReLU(τ_s - τ_t)：时间代价（惩罚用晚帧 token 覆盖早帧 token）
- C_ij = λ_f d̂_f + λ_x φ_κ(d̂_x) + λ_τ d̂_τ，其中 φ_κ(x) = log(1+κx)/log(1+κ) 扩展近场距离动态范围。

**FST 目标容量（Informativeness-aware Capacity）**：
- 对每个目标 token t_j，计算其到 3D 最近邻 n 个 token 的平均 FST 差异作为局部独特性分数 r_j
- v_j = φ(r_j) / Σ φ(r_l)，通过单调映射 φ 控制容量向信息性 token 集中，防止密集冗余区域主导覆盖目标。

**SGS 优化算法（Spatial-Guided Greedy Selection）**：
- 采用 Semi-Relaxed OT：P1 = u, P^T 1 ≤ v（允许目标侧未用完容量）
- 贪心步骤 ℓ：先求解 SOT_ε 获得当前传输计划 P_ℓ，计算目标残差容量 r_ℓ = [v - P_ℓ^T 1]_+
- 选择下一个 token：t_ℓ* = argmin_{t∈T\S_ℓ} Σ_{t_j∈M_g(t)} r_ℓ,j C(t, t_j)，其中 M_g(t) 为 t 在目标集中 3D 空间最近的 g 个 token
- 利用局部性先验避免全局 OT 重算，将贪心每一步的计算限制在小邻域内。

**CoverPrune-Lite（块结构 OT 近似）**：
- 使用 Morton 码空间填充曲线对所有目标 token 做 3D 感知排序，使相邻 token 在 3D 空间近邻
- 按容量 v 将排序后的 token 分为 K 个非重叠组 G_q，每组累积容量 ≈ 1/K
- 每个组内选原型：s_q = argmin_{t∈G_q} Σ_{t_j∈G_q} v_j C(t, t_j)
- 从 OT 视角：块对角约束 P_blk 强制每组的运输计划局部化，全局问题解耦为 K 个独立组内选择。

## 实验与结果
**数据集与基准**：
- ScanQA（3D QA）、SQA3D（场景问答）、Scan2Cap（3D 描述生成）——通用 3D 任务
- VSI-Bench——复杂时空推理基准，含 Object Count、Relative Distance、Relative Direction、Route Plan、Object Size、Room Size、Absolute Distance、Appearance Order 共 8 个子任务

**基线方法**：VisionZip (CVPR'25)、FastVID (NeurIPS'25)、DTC (CVPR'25)、EgoPrune (arXiv'25)，均为 training-free 方法。

**主要结果**（GS-Reasoner 基座，20% 保留率）：
- VSI-Bench 平均得分：CoverPrune = 59.76，CoverPrune-Lite = 59.43，VisionZip = 57.55，FastVID = 55.52，DTC = 56.57，EgoPrune = 49.44，Vanilla = 64.70（100%）
- CoverPrune 在 20% 保留率下保持 Vanilla 的 **92.4%** 性能
- 10% 保留率下：CoverPrune = 56.83（相对 Vanilla 87.8%），显著领先 DTC（51.66）和 EgoPrune（44.71）
- 5% 保留率下：CoverPrune-Lite = 52.88 略超 CoverPrune = 52.66，说明轻量变体在极端压缩下仍稳健

**ScanQA/Scan2Cap/SQA3D 结果**（10% 保留率）：
- CoverPrune 在 Scan2Cap Acc% = 99.04%，ScanQA Acc% = 100.00%，SQA3D EM% = 58.37%，全面超 baseline

**效率分析**（VSI-Bench, 10% 保留率）：
| 方法 | Tokens | Prun. Time (s) | Rel. Acc |
|------|--------|----------------|----------|
| DTC | 628 | 3.47 | 79.85% |
| CoverPrune | 628 | 2.53 | 87.84% |
| CoverPrune-Lite | 628 | 0.41 | 88.01% |

**消融**（R=20%, GS-Reasoner）：
- 去 feature cost：Overall ↓3.58（最大贡献）
- 去 geometry cost：Overall ↓0.74
- 去 time cost：Overall ↓0.80
- 去 FST capacity：Overall ↓0.18
- CoverPrune-Lite 在 Room Size（全局布局）上优于 CoverPrune，CoverPrune 在 Relative Direction（精细关系）上更优。

## 相关工作脉络
- **VisionZip (CVPR'25)**：结合 attention 重要性估计与特征多样性启发式，属于通用多模态剪枝；本文认为其 attention-based + diversity 混合策略在 3D 空间推理中不够对齐，缺乏几何/时间感知。
- **FastVID (NeurIPS'25)**：动态密度剪枝用于视频 LLM；针对视频理解而非 3D 空间推理，未利用 3D 坐标和几何结构信息。
- **DTC (CVPR'25)**：基于 voxel grounding 的 diversity-driven token compression，专门针对 3D QA；但其多样性目标在激进剪枝下破坏多视图对应，本文的覆盖视角弥补此缺陷。
- **EgoPrune (arXiv'25)**：利用 SfM pose 对齐重叠区域后再过滤冗余；依赖精确位姿估计，本文仅用 3D 坐标即可工作，更通用。
- **ToSA (IROS'25)**：引入空间感知信号的安全 token merging；属于 token merging 而非 pruning，且为微调方法。
- **SPOT (ECML PKDD'21)** 与 **Partial Wasserstein Covering (AAAI'22)**：将 OT 用于原型选择的相关理论工作，本文将其适配到 inference-time token pruning 场景并引入 FST 代价与容量设计。

## 局限性与未来方向
- **计算开销仍存**：CoverPrune 的 SGS 贪心需迭代求解 SOT，最坏情况仍较重；CoverPrune-Lite 虽快但近似误差在部分精细任务（如 Relative Direction）上有轻微损失。
- **依赖于 3D 坐标估计**：方法假设 token 有对应 3D 坐标（通过 SfM 或几何基础模型估计），在坐标估计质量差的场景下性能可能下降。
- **参数设置**：λ_f、λ_x、λ_τ 等权重在当前实验中设为 1，可能需针对不同场景调优。
- **未来方向**（论文自述）：将覆盖驱动的剪枝范式扩展到通用 2D VLMs，并探索与训练过程的联合优化。

## 研究启发与可借鉴点
- **覆盖 vs 多样性的视角转换**：将"保留最具代表性的样本"形式化为 OT 覆盖问题，为其他领域的样本/特征选择提供了可复用的理论框架。
- **多域代价联合设计**：FST 代价（特征+空间+时间）的设计思路可迁移到任何需要保持多维一致性的序列选择任务。
- **半松弛 OT 的贪心近似**：SGS 利用局部性先验将全局 OT 降为邻域局部选择，证明了在空间结构中近似子模最大化的高效性，可应用于大规模图/点云数据选择。
- **Morton 码排序 + 容量引导分组**：CoverPrune-Lite 的块结构近似策略，为需要快速局部匹配的 OT 类问题提供了 O(N log N) 的实用方案。
- **无需训练的即插即用范式**：在 encoder 与 LLM 之间插入的 plug-and-play 设计，便于在不同 3D VLM 架构间移植。

## 关键术语表
**Optimal Transport (OT)**：最优传输，衡量两个分布间最小代价匹配的理论框架，本文将其用于建模 token 覆盖关系。
**Semi-Relaxed OT**：半松弛最优传输，放松目标侧容量约束为不等式，使源支撑（选出的 token 子集）成为优化变量。
**Feature-Spatial-Temporal (FST) Cost**：FST 传输代价，联合特征余弦距离、3D 欧氏距离和时间顺序惩罚的多域代价函数。
**Informativeness-aware Capacity**：信息感知容量，基于 token 局部独特性评分分配的目标权重，使代表性 token 获得更多传输质量。
**Spatial-Guided Greedy Selection (SGS)**：空间引导贪心选择，利用 3D 邻域局部性近似子模最大化，逐步构建覆盖子集。
**Morton Code**：Morton 码（Z-order curve），一种空间填充曲线，将高维坐标映射为一维序以保持空间局部性。
**VSI-Bench**：Visual-Spatial Intelligence Benchmark，面向 egocentric 室内扫描视频的复杂时空推理基准。
**Token Pruning**：推理时 token 剪枝，在不重训练模型的前提下动态剔除冗余视觉 token 以加速推理。

## 可复现要素
- **数据集**：ScanQA、SQA3D、Scan2Cap、VSI-Bench（均为公开基准）
- **代码**：已开源，GitHub: https://github.com/Brucess/CoverPrune
- **权重**：使用 GS-Reasoner 和 VLM-3R 默认权重，无额外训练
- **关键超参**：λ_f = λ_x = λ_τ = 1；α_f = α_x = α_τ = 1；κ（φ_κ 非线性映射参数，论文未明确给出默认值）；n（局部独特性邻居数，未明确）；g（SGS 邻域大小，未明确）；K（保留 token 数）
