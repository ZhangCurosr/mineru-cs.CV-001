---
title: "MergeOver-Post-Training-Token-Merging-for-Recursive-Vision-T"
source: https://arxiv.org/pdf/2608.13141v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:49:59"
field: "视觉 Transformer 模型压缩"
keywords: ["Vision Transformer", "Token Merging", "Recursive Weight-Sharing", "Model Compression", "Post-Training Optimization", "Edge AI"]
innovations: ["提出首个无需重训练的递归 ViT 后训练 token 合并框架 MergeOver", "设计 Unmerge Stack 解决层级卷积池化的空间网格约束", "提出 stage-wise single-shot 调度策略适配递归层级架构"]
benchmarks: ["ImageNet-1K"]
---

# 论文速读：MergeOver: Post-Training Token Merging for Recursive Vision Transformers

## 一句话总结
MergeOver 是一种**无需重新训练**的后处理压缩方法，将 Token Merging（ToMe）动态令牌合并技术集成到递归权重共享的层级 Vision Transformer（SReT）中，通过 Unmerge tracking stack、约束安全合并率调整和跨空间置换的同步 token mass 追踪，解决了两种范式融合中的结构冲突问题。

## 研究问题与动机
- **ViT 在边缘设备上的部署瓶颈**：Vision Transformers 虽然精度高，但参数量大且计算复杂度随序列长度呈二次增长（O(N²)），严重限制了其在资源受限的硬件上的部署。
- **递归权重共享的双刃剑效应**：SReT 等递归架构通过权重复用大幅减少参数量，但重复执行自注意力操作反而增加了计算复杂度和峰值激活内存（PAM），形成新的瓶颈。
- **现有后训练压缩方法的空白**：已有的高效 ViT 替代方案（如 MobileViT、EfficientViT 等）大多依赖重训练、蒸馏或结构重设计；而后训练方法（如 token pruning、ToMe）主要面向各向同性的标准 ViT，尚未探索与递归层级架构的结合。
- **结构不兼容性**：原生集成两者面临三大冲突——token 合并破坏 SReT 层级卷积池化所需的刚性 2D 空间网格；无约束合并违反 SGA 的严格 group-size 边界；SReT 的空间置换使 ToMe 所需的 token-mass 追踪失效。

## 核心贡献（创新点）
1. **提出 MergeOver，首个无需重训练的递归 ViT 后训练 token 合并框架**：通过将 ToMe 的 BSM 合并与 SReT 的层级递归结构无缝集成，填补了递归权重共享与动态 token 合并交叉领域的空白。
2. **设计了 Unmerge tracking stack 解决空间布局约束**：利用 ToMe 的 Unmerge 操作恢复序列长度，确保压缩后的 token 能在每个 stage 结束前重建为满足卷积池化层所需的刚性 2D 网格。
3. **提出了 stage-wise single-shot token 归约调度策略**：针对 SReT 层级金字塔中每阶段不同的序列长度，设计了仅在每阶段首个 block 执行一次 token 合并的调度方案，相比全局常数和线性调度更适配递归架构。
4. **建立了跨平台基准评测体系**：在 ImageNet-1K 上对 GPU（RTX 4060 Ti）、x86 CPU（Intel Core Ultra 9）和 ARM CPU（Raspberry Pi 5）三种平台进行了系统的吞吐、延迟和内存评估。

## 方法详解

**整体框架**：MergeOver 将 SReT-Tiny-Distill 与 ToMe 进行后训练集成，在 Transformer block 内的 SGA 和 MLP 层之间插入 token 合并模块。

**关键设计一：Unmerge Stack（空间布局恢复）**
- SReT 采用从 PiT 继承的空间金字塔结构，stage 间使用卷积池化层进行下采样，要求输入序列长度必须能重塑为正方形张量（如 14×14 = 196）。
- ToMe 的 BiPartite Soft Matching (BSM) 算法在合并时同时返回一个对应的 Unmerge 函数，该函数通过**复制合并后的特征到原始空间坐标**来恢复序列长度（不撤销聚合，而是还原空间维度）。
- 实现一个 LIFO Unmerge stack：在每个 stage 内执行合并操作时，将对应的 Unmerge 函数压栈；在进入 inter-stage 卷积池化层之前，按严格逆序弹出并应用所有 Unmerge 操作，将压缩序列重建回原始长度。

**关键设计二：合并约束（方程 3）**
由于 SReT 的 SGA 机制和 ToMe 的 BSM 算法，合并数量 r 必须同时满足：
- **Group 可整除约束**：序列长度 N 必须能被 SGA 的 group 数 g 整除（g 为所有递归迭代中 group 数的最小公倍数 LCM）
- **BSM 二分匹配约束**：单次迭代中合并的 token 数 r ≤ ⌊N/2⌋
- 形式化表达：$(N - r) \bmod g = 0$，$N - r \geq g$，$g = \mathrm{LCM}(g_1, g_2)$，$r \leq \lfloor N/2 \rfloor$
- 目标合并数 $r_{\mathrm{target}}$ 需调整为最接近的安全值 $r_{\mathrm{safe}}$。

**关键设计三：Token Mass 同步追踪**
- 初始化全1的 mass tensor $s$，追踪每个 token 代表的原始 patch 数量。
- 使用**未加权的 Key 矩阵（K）**计算 token 相似度进行 BSM 匹配，确保合并决策仅基于语义特征相似性，避免"重" token（高累积 mass）主导匹配过程。
- 合并时使用质量加权平均：先对输入特征张量按对应 mass 进行逐元素预乘，再通过 sum reduction 聚合加权特征和并行 mass 指标，最后除以更新后的 mass tensor 得到正确的加权平均表示。
- 由于 SGA 使用置换（permutation）和逆置换操作在不同递归 pass 间移动 attention 窗口，将 mass tensor 与特征序列通过**完全相同的置换函数并行路由**，确保 mass 指标与对应 token 保持同步。

**关键设计四：Token 归约调度策略**
四种调度策略对比：
- **Constant Schedule**：$r_d^{\mathrm{const}} = r$，每个合并 block 合并固定数量
- **Linear Schedule**：$r_d^{\mathrm{lin}} = \lfloor 2r(1 - \frac{d-1}{D-1}) \rfloor$，从 2r 线性递减至 0
- **Stage-wise Exponential**：$r_{s,d}^{\exp} = \lfloor \rho_{\exp} \cdot N_s \cdot \alpha^{d-1} \rfloor$，每阶段内指数衰减
- **Stage-wise Single-shot**：$r_{s,d}^{\mathrm{shot}} = \lfloor \rho_{\mathrm{shot}} \cdot N_s \rfloor$（仅在第一 block），后续 block 不合并（公式 7），保持固定序列长度

## 实验与结果

**数据集**：ImageNet-1K 验证集（50,000 张图像，1,000 个类别），标准 ViT 预处理流水线（resize 到 256 短边 → 224×224 中心裁剪 → 标准化）。

**基线**：SReT-Tiny-Distill（无压缩递归 ViT 基线，参数 4.76M，FLOPs 1.91G，Top-1 准确率 77.39%）。

**硬件环境**：NVIDIA GeForce RTX 4060 Ti GPU、Intel Core Ultra 9 285K x86 CPU（4 线程）、Raspberry Pi 5 ARM CPU。

**主要结果（Table 1 核心数据）**：

| 配置 | Top-1 Acc (%) | ΔAcc | FLOPs (G) | ΔFLOPs |
|------|-------------|------|-----------|--------|
| SReT baseline | 77.39 | — | 1.91 | — |
| ρ_shot = 0.25（选中） | **75.92** | **-1.47** | 1.49 | -22.0% |
| ρ_shot = 0.40 | 72.36 | -5.03 | 1.25 | -34.3% |
| ρ_exp=0.25, α=0.3 | 74.58 | -2.81 | 1.35 | -29.3% |
| r_const = 10 | 70.95 | -6.44 | 1.32 | -30.8% |
| r_lin = 10 | 74.74 | -2.65 | 1.46 | -23.5% |

**GPU（RTX 4060 Ti，ρ_shot = 0.25）**：
- Batch size 1：吞吐降低 21.7%（175.33 vs 223.83 img/s），PAM 降低 37.3%（3.90 vs 6.22 MB）
- Batch size 16：吞吐**提升 21.7%**（1588.00 vs 1304.58 img/s），PAM**降低 38.4%**（61.27 vs 99.47 MB）

**x86 CPU（Intel Core Ultra 9 285K，ρ_shot = 0.25）**：
- Batch size 1：延迟**降低 17.9%**（12.27 vs 14.94 ms）
- Batch size 16：延迟**降低 30.0%**（100.98 vs 144.32 ms）

**ARM CPU（Raspberry Pi 5，ρ_shot = 0.25）**：
- Batch size 1：延迟降低 2.4%（139.42 vs 142.82 ms）
- Batch size 16：延迟**降低 17.6%**（1018.61 vs 1236.71 ms）

**关键结论**：
- Stage-wise 调度（指数和 single-shot）在 GPU batch=16 和所有 CPU 平台上显著优于全局调度（常数/线性）。
- 更强的压缩（ρ=0.40）带来更高的硬件加速，但伴随更大的精度损失。
- GPU 单流推理（batch=1）因合并/追踪操作的额外开销导致吞吐下降；CPU 平台则统一改善。
- FLOPs 估计不能完全预测硬件性能——BSM 匹配、mass 追踪和 Unmerge 操作的开销未被充分计入。

## 相关工作脉络
1. **SReT (Shen et al., ECCV 2022)**：递归权重共享的层级 ViT 基线，本文在其之上集成 token 合并，解决了原方法中重复自注意力带来的 PAM 和吞吐瓶颈。
2. **ToMe (Bolya et al., 2023)**：后训练 token 合并框架，通过 BSM 实现可逆合并；本文将其适配到层级递归架构，是 ToMe 首次与递归 ViT 的结合。
3. **PiToMe (Tran et al., NeurIPS 2024)**：优化合并匹配光谱保真的 ToMe 变体，保留了可逆性；与本文可互补，未来可探索其在 SReT 上的应用。
4. **Token Pruning 方法（DynamicViT, EViT）**：通过永久丢弃 token 降低计算量，但需要重训练或依赖 CLS token；本文采用合并而非剪枝，避免破坏层级结构且无需重训练。
5. **ToFu (Kim et al., WACV 2024)**：混合 token pruning 和不可逆合并的融合框架；因缺乏可逆 Unmerge 功能，与 SReT 的层级金字塔不兼容，本文凸显了可逆性的重要性。
6. **QUOTA (Li et al., 2026)**：将低比特量化与确定性 token pruning 结合的后训练多轴压缩方法；面向各向同性 ViT，本文填补了其在递归 ViT 领域的空白。

## 局限性与未来方向
- **模型局限性**：仅在 SReT-Tiny-Distill 上验证，无法分离递归权重共享效应与 SReT 层级金字塔结构或其他架构特定属性的影响；需扩展到大模型、非递归层级 backbone 和标准 ViT 上验证泛化性。
- **调度策略对比不完整**：全局调度与 stage-wise 调度在参数化、深度索引和归约轨迹上均不同，未评估等量归约预算下的匹配对比，无法完全隔离调度形状本身的效应。
- **CPU 内存测量局限**：GPU PAM 与 CPU ΔRSS 使用不同测量方式，前者为激活内存，后者包含分配器缓存、后端工作区等进程级内存，不可直接比较。
- **FLOPs 估计不充分**：thop 估计未完全捕捉 BSM 匹配、mass 追踪、序列操作和 Unmerge 追踪的开销；缺乏底层 profiling 和优化。
- **边缘部署挑战**：Raspberry Pi 单流推理（batch=1）仅获得 2.4% 的延迟改善，实际单流边缘部署仍然困难；未来需结合低层 kernel 优化和量化等互补后训练技术。

## 研究启发与可借鉴点
1. **可逆合并解决层级结构约束**：利用 ToMe 的 Unmerge 操作重建 2D 空间网格的思路，可推广到其他需要保持空间结构的层级 ViT 架构（如 Swin、PiT、PVT），为后训练压缩提供了通用解法。
2. **Mass-aware 相似度计算**：使用未加权的 Key 矩阵计算 token 相似度而非依赖 mass 加权，避免了"重"token 主导匹配过程，这一设计对其他基于相似度的 token 选择/合并方法具有借鉴价值。
3. **Stage-wise vs Global 调度的比较框架**：本文系统对比了 4 种调度策略在递归架构上的表现，揭示了 stage-wise 调度对层级结构的适配优势，这一评估范式可直接迁移到其他 ViT 压缩研究中。
4. **跨平台统一评测方法**：同时评估 GPU、x86 CPU 和 ARM CPU 三种平台的吞吐/延迟/内存，并详细分析 batch size 依赖性，为边缘 AI 模型压缩研究提供了可复用的评测方法论。
5. **约束安全的合并率调整**：通过最小公倍数（LCM）和二分匹配上界（⌊N/2⌋）自动将目标合并率调整为安全值的机制，可集成到任何需要满足结构性约束的动态压缩框架中。

## 关键术语表
- **Vision Transformer (ViT)**：将 Transformer 架构应用于图像识别任务的模型，将图像分割为 patch 序列后处理，具有 O(N²) 计算复杂度。
- **SReT (Sliced Recursive Transformer)**：基于 PiT 的层级空间金字塔结构，通过递归重复使用同一 Transformer block 的冻结权重来大幅减少参数量的 ViT 变体。
- **SGA (Sliced Group Self-Attention)**：SReT 中的分组自注意力机制，将 token 序列等分为 g 组并在组内独立计算注意力，支持置换操作以扩展感受野。
- **Token Merging (ToMe)**：后训练压缩方法，通过 BiPartite Soft Matching (BSM) 合并相似 token，并通过 token-mass 向量保持 proportional attention。
- **Unmerge Stack**：LIFO 栈结构，存储每个合并操作对应的 Unmerge 函数，在 stage 结束前逆序应用以恢复原始序列长度，满足卷积池化的空间网格要求。
- **Token Mass Vector (s)**：并行追踪每个 token 所代表原始 patch 数量的向量，用于在合并时进行质量加权平均和 proportional attention 修正。
- **Peak Activation Memory (PAM)**：前向推理过程中激活张量占用的最大显存/内存，是边缘部署的关键约束指标。
- **Stage-wise Single-shot Schedule**：仅在每阶段第一个 block 执行 token 合并（合并比例为 ρ_shot · N_s），后续 block 保持固定序列长度，适配 SReT 层级结构。

## 可复现要素
- **数据集**：ImageNet-1K（ILSVRC 2012），公开可用。
- **代码**：开源，DOI: 10.5281/zenodo.21888951。
- **预训练权重**：SReT 预训练权重从作者公开仓库 [25] 获取。
- **关键超参**：
  - ρ_shot ∈ {0.25, 0.40}（选中 0.25）
  - ρ_exp = 0.25/0.40, α = 0.3
  - r_const ∈ {10, 20}
  - r_lin ∈ {10, 20}
- **环境**：Python 3.10, CUDA 13.0, PyTorch 2.11.0, torchvision 0.26.0, timm 0.4.12。
- **图像处理**：resize 到 256 短边 → 224×224 中心裁剪 → 标准 ImageNet 均值/std 标准化。
