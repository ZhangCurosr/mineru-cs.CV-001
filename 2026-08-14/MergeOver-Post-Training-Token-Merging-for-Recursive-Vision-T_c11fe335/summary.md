---
title: "MergeOver-Post-Training-Token-Merging-for-Recursive-Vision-T"
source: https://arxiv.org/pdf/2608.13141v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:50:06"
field: "高效视觉Transformer"
keywords: ["Vision Transformer", "Token Merging", "Recursive Weight Sharing", "Post-Training Compression", "Edge AI"]
innovations: ["提出MergeOver后训练方法将ToMe集成到递归ViT中无需重新训练", "设计Unmerge Stack解决层次化架构的空间网格恢复问题", "提出stage-wise single-shot调度策略在精度与效率间取得最优平衡"]
benchmarks: ["ImageNet-1K"]
---

# 论文速读：MergeOver: Post-Training Token Merging for Recursive Vision Transformers

## 一句话总结
本文提出 **MergeOver**，一种后训练压缩方法，将 Token Merging (ToMe) 动态序列缩减技术成功集成到递归权重共享的层次化 Vision Transformer（以 SReT 为代表），无需重新训练即可在 ImageNet-1K 上将 top-1 准确率仅损失 1.47 个百分点的同时，显著降低 GPU 峰值激活内存（PAM）并提升 ARM/x86 CPU 推理吞吐。

## 研究问题与动机
1. **Vision Transformer 的资源瓶颈**：ViT 具有二次计算复杂度 $O(N^2)$ 和大量参数，严重限制其在资源受限的边缘硬件上的部署。
2. **递归权重共享的副作用**：SReT 等递归架构虽通过权重复用降低参数量，但重复执行自注意力会大幅增加计算复杂度和峰值激活内存（PAM），形成新的瓶颈。
3. **Token Merging 与递归架构的结构冲突**：现有的 ToMe 等 token 缩减方法与 SReT 的层次化空间金字塔（convolutional pooling layers）、Sliced Group Self-Attention (SGA) 分组约束以及 token permutation 机制存在架构摩擦，直接集成会导致维度不匹配或追踪失效。
4. **后训练压缩路径的空白**：现有高效 ViT 方法（MobileViT、FastViT 等）依赖重新训练或蒸馏，缺乏一种可直接应用于预训练递归模型的后训练压缩方案。

## 核心贡献（创新点）
1. **首次将 ToMe 后训练集成到递归层次化 ViT 中**：提出 MergeOver 框架，在不重新训练的前提下实现动态 token 合并，填补了递归权重共享与 token 合并交叉领域的空白。
2. **Unmerge Tracking Stack 解决空间布局约束**：利用 ToMe 的 BiPartite Soft Matching (BSM) 算法自带可逆 Unmerge 操作，在每阶段结束时将压缩序列恢复为原始 2D 网格，满足层间卷积池化的严格空间对齐要求。
3. **Constraint-Safe Merge-Rate Adjustment 机制**：针对 SGA 的分组整除约束和 BSM 的二分匹配上限，提出安全合并率调整策略，确保 $(N-r)$ 能被所有递归迭代的组号最小公倍数整除且 $r \leq \lfloor N/2 \rfloor$。
4. **Stage-wise Single-Shot Token 调度策略**：提出仅在每阶段第一个 block 执行 token 合并、后续递归迭代保持序列长度不变的调度方案，相比全局 constant/linear 调度和 stage-wise exponential 调度，在精度保持与硬件效率间取得更优平衡。
5. **跨平台基准评估**：在 GPU（RTX 4060 Ti）、x86 CPU（Intel Core Ultra 9 285K）、ARM CPU（Raspberry Pi 5）三种平台上系统评估了不同调度策略的性能，揭示平台与 batch size 对 token 合并收益的显著影响。

## 方法详解
**整体架构**：MergeOver 将 ToMe 模块插入 SReT Transformer block 中 SGA 与 MLP 层之间，通过四个关键技术解决结构兼容性：

1. **Unmerge Stack（空间布局恢复）**：
   - 每个 stage 内的 token 合并操作会将对应的 Unmerge 函数压入 LIFO 栈
   - 在进入层间卷积池化层前，按严格逆序应用所有 Unmerge 操作，将压缩序列恢复至原始长度（如 $N=784$），保证 2D 网格结构完整性
   - 恢复的是空间维度而非特征聚合，即复制合并后的特征到原始空间坐标

2. **Merging Constraints（安全合并率约束）**：
   - 必须满足两个条件：$(N-r) \bmod g = 0$ 且 $N-r \geq g$，其中 $g = \text{LCM}(g_1, g_2, ...)$ 为所有递归迭代组号的最小公倍数
   - 同时需满足 $r \leq \lfloor N/2 \rfloor$（BSM 二分匹配的上限）
   - 将目标合并率 $r_{\text{target}}$ 调整为最近的可行值 $r_{\text{safe}}$

3. **Synchronized Token-Mass Tracking（同步质量追踪）**：
   - 初始化质量张量 $\mathbf{s}$ 为全 1，与特征序列并行流动
   - 在 SGA 的 permutation/inverse-permutation 操作中对 $\mathbf{s}$ 施加完全相同的排列，确保质量指标与对应 token 同步
   - 使用未加权的 Key 矩阵计算 token 相似度（避免高 mass token 主导匹配过程）
   - 合并时采用加权平均：先按 mass 预乘特征，再求和并除以更新后的 mass

4. **Token Reduction Scheduling（调度策略）**：
   - **Constant**: $r_d^{\text{const}} = r$（全局固定合并数）
   - **Linear**: $r_d^{\text{lin}} = \lfloor 2r(1 - \frac{d-1}{D-1}) \rfloor$（全局线性递减）
   - **Stage-wise Exponential**: $r_{s,d}^{\text{exp}} = \lfloor \rho_{\exp} \cdot N_s \cdot \alpha^{d-1} \rfloor$（每阶段指数衰减）
   - **Stage-wise Single-shot**: $r_{s,d}^{\text{shot}} = \lfloor \rho_{\text{shot}} \cdot N_s \rfloor$（仅在每阶段首 block 合并，后续 $d>1$ 时 $r=0$）

## 实验与结果
- **数据集**：ImageNet-1K validation（50,000 张图像，1000 类）
- **基线模型**：SReT-Tiny-Distill（4.76M 参数，1.91G FLOPs，top-1 准确率 77.39%）
- **硬件平台**：
  - GPU：NVIDIA GeForce RTX 4060 Ti
  - x86 CPU：Intel Core Ultra 9 285K（4 线程）
  - ARM CPU：Raspberry Pi 5（16GB RAM）

**最优配置**（$\rho_{\text{shot}} = 0.25$）：
| 平台 | Batch Size | Top-1 Acc. | 变化 | Throughput | 变化 | PAM/ΔRSS | 变化 |
|------|-----------|------------|------|------------|------|----------|------|
| GPU | 1 | 75.92% | -1.47% | 175.33 img/s | -21.7% | 3.90 MB | -37.3% |
| GPU | 16 | 75.92% | -1.47% | 1588.00 img/s | **+21.7%** | 61.27 MB | **-38.4%** |
| x86 CPU | 1 | 75.92% | -1.47% | - | - | 12.27 ms | **-17.9%** |
| x86 CPU | 16 | 75.92% | -1.47% | - | - | 100.98 ms | **-30.0%** |
| ARM CPU | 1 | 75.92% | -1.47% | - | - | 139.42 ms | **-2.4%** |
| ARM CPU | 16 | 75.92% | -1.47% | - | - | 1018.61 ms | **-17.6%** |

**关键结论**：
- 全局调度（constant/linear）在 GPU batch size 16 时几乎无吞吐提升（-10.8% ~ -5.9%），而 stage-wise 调度显著提升
- FLOPs 估计不能准确预测硬件性能（如 $\rho_{\text{shot}}=0.40$ 比 $\rho_{\text{exp}}=0.40$ 有更高吞吐量但 FLOPs 也更高）
- GPU 单流推理（batch size 1）因并行度不足导致吞吐量下降，但在 batch size ≥8 时开始获益
- x86 CPU 上所有调度在 batch size 16 时均加速，stage-wise 调度优势明显（最高 -50.6% 延迟）
- ARM CPU 单图推理收益有限（-2.4%），但批处理时仍有显著改进（-17.6%）

## 相关工作脉络
1. **SReT [7]**：本文的基础递归架构，采用分层空间金字塔和 SGA 机制，参数高效但计算复杂度高；本文在其上叠加 ToMe 实现后训练压缩。
2. **ToMe [10]**：原始 token merging 方法，通过 BSM 实现可逆合并，支持 post-training 应用；本文将其适配到递归层次化架构。
3. **PiToMe [11]**：优化匹配过程的 ToMe 变体，但本文未采用因其不可逆特性不适合层次化网络。
4. **DynamicViT [8] / EViT [9]**：token pruning 方法，前者需重新训练，后者依赖 CLS token 与 SReT 的全局平均池化不兼容。
5. **ToFu [12]**：token fusion 方法，混合 pruning 和 merging，但缺乏双向追踪能力，无法恢复 2D 网格结构。
6. **QUOTA [24] / MADTP++ [15]**：多轴压缩方法，但依赖 destructive pruning 需重新训练，不适用于后训练场景。

## 局限性与未来方向
1. **模型范围受限**：仅在 SReT-Tiny-Distill 上验证，无法分离递归权重共享与层次化金字塔结构的独立效应；需在大模型、非递归层次化 backbone 及标准 ViT 上验证泛化性。
2. **调度策略对比不完整**：全局与 stage-wise 调度在参数化、深度索引和轨迹上存在差异，未在等价还原预算下比较 matching per-stage fractional constant/linear 调度。
3. **CPU 内存测量局限**：ΔRSS 包含 allocator caching 和物理页驻留行为，不能直接等同于 PAM，GPU PAM 与 CPU ΔRSS 不可直接比较。
4. **FLOPs 估计不充分**：thop 无法捕获 BSM 匹配、token-mass 追踪、序列操作和 Unmerge 追踪的计算开销。
5. **缺乏底层优化**：未进行低层 profiling 和 kernel 优化，GPU 单流吞吐下降的原因仅作为假设。
6. **边缘单图推理仍具挑战**：Raspberry Pi 上 batch size 1 仅获得 2.4% 延迟改进，实际单图边缘部署仍需进一步优化。

## 研究启发与可借鉴点
1. **Unmerge Stack 设计范式**：利用可逆 token 操作（如 BSM 的 Unmerge）解决层次化架构中动态序列长度与固定空间网格的冲突，可迁移到其他需要恢复 2D/3D 结构的 token 操作场景。
2. **Stage-wise vs Global 调度对比**：揭示了 hierarchical architecture 中按阶段重置的调度策略优于全局深度索引调度，为其他递归/分层模型的高效 token 管理提供设计原则。
3. **Platform-dependent 性能分析框架**：系统评估 GPU/CPU/ARM 上不同 batch size 的表现，揭示 token 合并收益的平台依赖性和批处理敏感性，为边缘部署决策提供方法论参考。
4. **Constraint-safe Adjustment 机制**：将数学约束（整除性、二分匹配上限）嵌入 post-training 压缩流程，确保架构兼容性而不依赖重新训练，可推广到其他结构受限的压缩场景。
5. **与互补技术的集成潜力**：论文指出 MergeOver 可与 FlashAttention、后训练量化（FQ-ViT）等技术结合，为多轴压缩研究提供扩展基线。

## 关键术语表
- **Vision Transformer (ViT)**：将 Transformer 架构应用于图像识别任务的模型，将图像分割为 patch 序列并进行全局自注意力计算。
- **Sliced Recursive Transformer (SReT)**：基于 PiT 的层次化递归 ViT，通过权重共享重复使用同一 Transformer block，并采用 Sliced Group Self-Attention (SGA) 和卷积池化实现空间金字塔。
- **Token Merging (ToMe)**：一种后训练压缩方法，通过 BiPartite Soft Matching (BSM) 将相似 token 合并，支持可逆的 Unmerge 操作以恢复序列长度。
- **BiPartite Soft Matching (BSM)**：ToMe 的核心算法，将 token 分为两组并计算软匹配对，实现可微分的 token 合并与去合并。
- **Peak Activation Memory (PAM)**：推理过程中激活值所需的最大显存/内存，是影响边缘部署的关键资源约束。
- **Unmerge Stack**：MergeOver 引入的 LIFO 数据结构，存储每步合并对应的 Unmerge 函数，在阶段结束时按逆序恢复原始 2D 空间网格。
- **Token Mass Vector (s)**：并行于特征序列的质量追踪张量，记录每个 token 代表的原始 patch 数量，用于加权平均合并和 proportional attention 计算。
- **Stage-wise Single-shot Schedule**：MergeOver 提出的最优调度策略，仅在每个 stage 的第一个 block 执行 token 合并，后续递归迭代保持序列长度不变。

## 可复现要素
- **数据集**：ImageNet-1K validation set（ILSVRC 2012），公开可用
- **代码**：开源，DOI: 10.5281/zenodo.21888951（参考文献 [16]）
- **预训练权重**：SReT-Tiny-Distill 权重从作者公开仓库 [25] 获取
- **关键超参**：
  - $\rho_{\text{shot}} = 0.25$（最优配置）
  - $\rho_{\exp} \in \{0.25, 0.40\}$
  - $\alpha = 0.3$
  - $r_{\text{const}}, r_{\text{lin}} \in \{10, 20\}$
- **软件环境**：Python 3.10, CUDA 13.0, PyTorch 2.11.0, torchvision 0.26.0, timm 0.4.12
- **硬件环境**：NVIDIA RTX 4060 Ti, Intel Core Ultra 9 285K (x86), Raspberry Pi 5 (ARM)
