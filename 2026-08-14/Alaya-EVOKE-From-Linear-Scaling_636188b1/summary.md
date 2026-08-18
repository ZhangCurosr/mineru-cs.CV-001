---
title: "Alaya-EVOKE-From-Linear-Scaling"
source: https://arxiv.org/pdf/2608.13546v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 13:40:06"
field: "视频生成与交互式世界模型"
keywords: ["interactive world model", "video generation", "distribution matching distillation", "long-horizon generation", "geometric memory", "few-step diffusion", "chunk-wise sparse attention"]
innovations: ["外部化几何世界状态库使 denoiser 上下文有界且独立于会话时长", "chunk-wise 稀疏注意力 + per-chunk 文本条件实现线性成本的长时序监督", "30秒 self-forced 全窗口分布匹配蒸馏将长程一致性传递到三步 CFG-free 学生"]
benchmarks: ["WBench", "VBench-2.0", "VBench-Long"]
---

# 论文速读：Alaya-EVOKE: From Linear-Scaling Supervision to Endless World

## 一句话总结
Evoke 是一个三步无 CFG 视频世界模型，通过将持久化世界状态外部化到相机索引的几何世界状态库（World State Bank）中，并重新设计支持长时序监督的教师模型（Chunk-wise Sparse Attention + Per-chunk Conditioning），实现了无上下文膨胀的无限时长交互式视频生成。在单张 H200 上每 1.5s 片段生成仅需 2.11s，在 WBench 上达到 SOTA，并在 VBench-2.0/Long 上与多步系统持平。

## 研究问题与动机
- **持久记忆与低延迟的冲突**：现有交互式世界模型将历史帧或 KV cache 保留在 denoiser 上下文中，随着会话时长增长，每步成本线性甚至二次膨胀；窗口截断/缓存驱逐又会导致信息丢失。
- **少步蒸馏的教师能力上限**：低延迟交互通常依赖少步蒸馏（few-step distillation），学生只能继承教师所表达的监督能力；若教师监督 horizon 短、条件固定，则无法传递长程一致性和中途指令响应能力。
- **退化漂移与内容漂移的本质区别**：曝光/饱和度等退化漂移在短窗口内可检测，但场景身份/空间布局的渐进式内容漂移需跨越较长时序才能暴露，传统短窗口监督无法约束后者。
- **位置编码边界限制**：随着生成时长增加，连续时序位置轴可能超出训练所见范围，导致模型性能退化；需要一种与时间无关的持久状态访问机制。

## 核心贡献（创新点）
1. **有界循环生成框架**：将交互式世界生成形式化为有界循环过程，通过外部相机索引的世界状态库（World State Bank）存储场景几何，使 denoiser 上下文长度和位置范围独立于会话时长。与已有工作的区别：此前工作（如 WARP、Matrix-game）要么在 denoiser 侧累积历史，要么仅做短时 warp，本文实现了有界读写 + 淘汰的外部持久几何状态。
2. **面向长时序交互监督的教师重设计**：提出 chunk-wise 稀疏注意力（local context + 远距离关键帧 + linear attention global state），将注意力计算从二次降为线性，同时为每个 chunk 分配独立文本条件，实现长时序监督与动态调度的联合建模。与已有蒸馏工作的区别：此前将教师视为固定高质量打分器，本文显式将教师结构本身作为设计变量。
3. **30 秒长时序分布匹配蒸馏**：在 self-forced rollout 下对 189 latent frames（~31.4s）做全窗口 DMD 监督，将长程一致性和中途条件响应能力传递给三步 CFG-free 学生。与已有工作的区别：多数蒸馏工作监督窗口较短（几秒），本文验证了更长监督 horizon 对光度稳定性的显著提升。

## 方法详解

### 3.1 有界循环会话形式化
在每个循环步骤 $k$，学生生成一个包含 $F=9$ latent frames（1.5s @ 24fps）的 chunk $x_k$，条件为相机轨迹 $\mathcal{P}_k$ 和可变文本 $c_k$：

$$r_k = \text{Read}(M_k, \mathcal{P}_k), \quad x_k \sim p_\theta(\cdot | r_k, h_k, c_k), \quad M_{k+1} = \text{Write}(M_k, x_k, \mathcal{P}_k)$$

其中 $h_k$ 为有界局部历史（19 latent frames ≈ 3.2s，分长/中/短三层），$M_k$ 为世界状态库。会话延长仅增加循环调用次数，不扩大单次调用的上下文或位置范围。

### 3.2 长时序监督分析
定义学生轨迹分布 $q_\theta$ 与数据分布 $p$ 之间的自强制分布匹配目标：

$$\mathcal{L}_W(\theta) = \mathbb{E}_k [D(q_\theta^{(k:k+W-1)} || p^{(k:k+W-1)})]$$

监督 horizon $W$ 决定了两类漂移的可检测性：
- **退化漂移**（曝光/饱和度/纹理退化）：短窗口内可见，但成因可能远在窗口之前；增大 $W$ 扩展了 rollouts 下 conditioning shift 的分布范围。
- **内容漂移**（场景身份/物体外观渐进演化）：局部窗口均合理，需足够长的 $W$ 才能暴露不一致；此类漂移由教师监督约束，空间重访一致性由世界状态库负责。

### 3.3 长时序交互教师（Evoke Teacher）
基于 14B Wan2.2 A14B DiT，每 chunk（9 latent frames）采用 chunk-wise 稀疏注意力：
- **First-frame global sink**：首帧全局接收
- **Local context**：含一帧重叠的局部上下文
- **Spatially compressed nearby frames**：空间压缩的近邻帧
- **Selected distant frames**：少量选择的远距帧
- **Linear-attention global state**：线性注意力累积的全局状态

注意力开销随序列长度线性增长。同一 chunk 划分赋予每个 chunk 独立的文本条件（caption 按 12s 分段映射到 latent chunks）。

**蒸馏目标**（公式 3）：构建 20-chunk self-forced rollout（189 latent frames ≈ 31.4s），教师与 critic 联合打分全部帧：

$$\Delta s = s_{\text{fake}} - s_{\text{real}}, \quad \nu = \text{mean}_\Omega[|\hat{x}_0 - s_{\text{real}}|], \quad \mathcal{L}_{\text{gen}} = \frac{1}{2}\|\hat{x}_0 - (\hat{x}_0 - \frac{\Delta s}{\nu})^{\text{detach}}\|_2^2$$

掩码 $\Omega$ 排除 ground-truth prefix 和第一 chunk $x_1$（避免边界闪烁）。chunk 间历史 detach，梯度图限于单 chunk，但教师/critic 仍对完整 31.4s 轨迹联合评分——监督 horizon ≠ 梯度 horizon。

### 3.4 几何世界状态库
- **Write**：monocular depth model（Depth Anything 3）估计 chunk 内 12 帧深度，反投影为 world-space 几何，追加到 $M_k$
- **Read**：当前相机位姿直接决定查询哪些存储视图；按 co-visibility 排序，选择最多 8 个独立源视图，经 batched projection + z-buffer 渲染为 view-aligned warped observation 和 per-pixel visibility mask
- **Visibility gating**：visibility < 0.5 的像素设 $\sigma = 1$（无信息贡献）；支持区域 $\sigma \in [0, 0.135]$；visibility 信号同时 pooling 到各 history tier 的 patch 分辨率，移除不受支持的 token

当前实现保留 90s 几何（2160 pixel frames，活跃源池 ≤ 720 frames）。

### 3.5 三步有界推理
三阶段 coarse-to-fine latent pyramid（$12\times20, 24\times40, 48\times80$），每阶段一次 CFG-free denoising evaluation。几何条件仅注入最粗阶段，visibility-based token pruning 移除不支持的 warp token。

## 实验与结果

**数据集与训练**：Sekai 视频数据集 + 内部视频数据；学生基于 Helios，教师基于 Wan2.2 A14B。

**WBench 导航 split（n=158）**：Evoke 在 Video Quality avg（82.79）、Setting avg（83.76）、Physical avg（72.06）三组中领先所有已评估的 few-step 系统；Consistency avg（86.87）与最佳持平。Scene（74.68）和 Causal Fidelity（82.44）提升最大。

**VBench-2.0**：总分 66.77，排名 1/10（领先 Veo 3 的 66.72）。

**VBench-Long**：总分 85.11，排名 7/10（与 Veo 3 的 85.06 极接近）。

**长会话稳定性**：8 条 65.5 分钟 rollout（各 2619 chunks），光度统计在短暂瞬态后趋于平稳，无渐进退化；内容描述符 decorrelation 速度与真实视频相当。

**推理速度**：单 H200 @ 384×640，每 1.5s chunk 2.11s（仅 diffusion wall clock）；含几何渲染路径的完整推理开销约为 denoiser 的 1.38 倍（几何路径占单步 38% 的 denoiser 成本）。

**教师 horizon 消融（Fig.6）**：长时序教师蒸馏的学生在光度稳定性上显著优于短时序教师（Wilcoxon p=0.016），亮度稳定在 101% vs 74%；sharpness 无差异。

**几何召回（Fig.7/Sec.4.4）**：retention window ≥ 离开时长时，revisit PSNR 提升 2.3–3.2 dB， plateau 达 15.4–17.8 dB（可识别但非像素级忠实重建）。

**Timed prompt switching**：未锚定内容实现率 67%（天花板 83%），覆盖锚定几何的实现率仅 4%（天花板 17%）；证实文本控制作用于自由演化内容，几何内存抵抗锚定区域被覆盖。

## 相关工作脉络
1. **WARP（Wang & He, 2026）**：warp-based 历史条件，仅支持短程帧间对应；Evoke 将其扩展到持久几何记忆 + 有界上下文循环。
2. **Rubic / Relic（Hong et al., 2025; Feng et al., 2025）**：在 denoiser 侧使用检索/流式条件维持有限上下文；Evoke 将持久状态完全外置，denoiser 上下文恒定。
3. **Diffusion Forcing（Chen et al., 2024）/ Self-Forcing（Huang et al., 2025a）**：自强制 rollouts 用于训练稳健学生；Evoke 在此基础上引入 30s 长窗口全轨迹评分，而非短时 window。
4. **Causal Forcing（Zhu et al., 2026）**：关注双向教师与因果学生的架构不匹配；Evoke 在此基础上进一步关注监督 horizon 和条件调度两个正交设计维度。
5. **Matrix-game / HY-World**：few-step 交互式世界模型，依赖 KV cache 或 context frames 维持历史；Evoke 通过外部几何状态库避免了上下文膨胀。
6. **Gen3C（Ren et al., 2025）/ Lyra 2.0（Shen et al., 2026）**：几何感知的世界生成；Evoke 的不同在于将几何状态作为循环过程中可读写的外部 store，而非仅用于单段生成条件。

## 局限性与未来方向
- **细粒度一致性不足**：当前几何世界状态仅保留粗粒度场景结构，物体身份、外观和局部细节的长期一致性有限； richer object-level / semantic representations 是改进方向。
- **动态状态建模缺失**：持久世界应建模物体运动、状态转换及其长时演化，当前方法仅处理静态几何；显式的动态世界状态表示是重要下一步。
- **推理加速仍有空间**：需要更高压缩比的视频 VAE、更高效的 few-step 生成器和更低成本的几何条件模块，才能实现真正的实时交互。
- **导航与透视能力相对较弱**：WBench 中 Navigation（78.63）和 Perspective（69.74）得分低于其他组，作者归因于当前 camera-control path 的局限。
- **几何记忆为有界而非无限**：仅保留 90s 几何，超出部分不再约束后续生成（退化为 inpainting），并非对每处访问过的地点永久记忆。

## 研究启发与可借鉴点
1. **监督 horizon 作为显式设计变量**：本文系统论证了不同失败模式（退化漂移 vs 内容漂移）需要不同 temporal scale 的约束，教师监督 horizon 不应默认等于训练 clip 长度——可迁移到任何蒸馏/少步生成任务中，通过 sweep W 找到边际收益饱和点。
2. **Chunk-wise sparse attention for long-video teacher**：将序列划分为固定 chunk 并结合 local context + 远距离采样帧 + linear global state，使教师能以线性成本处理长序列——可复用到其他需要长时序监督的扩散模型训练中。
3. **监督 horizon ≠ 梯度 horizon 的技术**：chunk 间 detach 历史限制反向图长度，但教师/critic 仍对完整 rollout 联合打分；这一分离使得长监督信号可用而不引发显存爆炸，对长序列蒸馏有通用参考价值。
4. **Per-chunk conditioning 实现中途事件控制**：将文本条件按 chunk 分配而非全局固定，使同一 rollout 中可插入/撤回指令；这一接口设计可直接应用于交互式视频生成系统的用户控制模块。
5. **几何世界状态库 + Visibility gating 的条件注入策略**：仅在不支持区域提高 warp noise level 而非强制填充，使模型在已知区域保持忠实、在未知区域保留生成自由度——可推广到任何基于几何条件的生成任务。

## 关键术语表
**World State Bank**：相机位姿索引的外部持久几何存储，记录已观测场景的 3D 结构，支持有界读写与淘汰，使 denoiser 上下文不随会话增长。
**Chunk-wise Sparse Attention**：将长序列划分为固定 chunk，每 chunk 仅 attends 局部上下文 + 少量远距帧 + linear global state，使注意力开销从二次降至线性。
**Self-Forced Rollout**：学生在训练时使用自身生成的历史作为条件（而非 ground-truth），模拟部署时的条件偏移，提升鲁棒性（Huang et al., 2025a）。
**Distribution Matching Distillation (DMD)**：通过教师与 critic 的分数差估计分布梯度，将慢教师蒸馏为少步学生（Yin et al., 2024a,b）。
**Degradation Drift vs Content Drift**：前者指曝光/饱和度等低阶统计量的渐进变化（短窗口可检测）；后者指场景身份/物体外观的渐进演化（需长窗口联合比较才能发现）。
**Evocation**：在 ongoing rollout 中通过 timed prompt schedule 引入或撤除文本驱动元素的能力。
**CFG-free**：无需 classifier-free guidance（即不做条件/无条件两次前向），三步即可生成，大幅降低推理开销。
**Warp-conditioning**：将历史帧通过几何变换对齐到当前视角作为条件输入，比直接拼接历史帧更节省上下文。

## 可复现要素
- **数据集**：Sekai 视频数据集（Li et al., 2026）+ 内部视频数据；Sekai 公开可用
- **代码**：https://github.com/SII-YuanyangYin/Evoke（已开源）
- **项目页面**：https://evoke-world.github.io/Evoke/
- **模型权重**：论文未明确提及公开权重链接，项目页面可能有下载
- **关键超参**：chunk 长度 = 9 latent frames（1.5s @ 24fps）；监督 horizon W ≈ 30s（20 chunks）；世界状态库保留 90s 几何；学生三步 coarse-to-fine pyramid（12×20 → 24×40 → 48×80）；几何条件仅注入最粗阶段；depth model 使用 Depth Anything 3
- **推理硬件**：单 H200，384×640 分辨率
- **论文未提及**：具体学习率、optimizer 设置、教师预训练 checkpoint 来源细节、内部数据集规模
