---
title: "Alaya-EVOKE-From-Linear-Scaling"
source: https://arxiv.org/pdf/2608.13546v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 13:40:35"
field: "视频生成与交互世界模型"
keywords: ["interactive world model", "few-step video generation", "distribution matching distillation", "long-horizon supervision", "geometric world state", "sparse attention", "recurrent generation", "evocation"]
innovations: ["将持久世界状态外置到相机索引几何世界状态银行，实现有界 recurrent 生成", "教师模型面向长时交互重设计：分块稀疏注意力+逐块文本条件，支持 30 秒监督时域与 session 中段指令切换", "通过自强制分布匹配将长程监督与动态条件能力蒸馏至三步 CFG-free 学生"]
benchmarks: ["WBench", "VBench-2.0", "VBench-Long"]
---

# 论文速读：Alaya-EVOKE-From-Linear-Scaling

## 一句话总结
Evoke 将交互世界模型的持久场景状态从去噪器上下文解耦，通过外部相机索引的世界状态银行（World State Bank）实现有界 recurrent 生成；同时重新设计教师模型，采用分块稀疏注意力与逐块文本条件进行约 30 秒长程监督，再通过自强制分布匹配蒸馏为一个无 CFG 的三步学生模型，实现小时级连贯生成与 session 中段的响应式指令控制。

## 研究问题与动机
- 交互世界模型需同时支持持久记忆、低延迟响应和长时域生成，三者对底层模型施加相互冲突的要求：将历史保留在去噪器上下文或 KV cache 中会使每步成本随 session 长度线性/二次增长，而 windowing 和 cache eviction 会以丢弃信息为代价。
- 低延迟交互通常依赖少数几步蒸馏生成，学生能力受限于教师表示的时域范围和 conditioning 能力，无法在长 session 中引入新的文本指令或应对持续的内容漂移。
- 现有几何条件方法（如 warp-as-history）仅覆盖短程帧间对应，无法将已观察表面的持久几何状态在相机回归时显式召回。
- 教师通常被视为固定高质量生成器，其注意力结构（双向、二次）、监督时域（短 clip）和 conditioning 模式（全局单一 prompt）并未针对长时交互蒸馏需求进行专门设计。

## 核心贡献（创新点）
- **持久状态外置**：将场景几何维护在外部相机索引的世界状态银行中，去噪器上下文长度和位置范围与 session 时长无关；与现有方法将历史保留在 denoiser 侧或依赖有限检索预算的本质区别在于"用相机位姿寻址几何存储，而非 token 历史扩展"。
- **教师面向长时交互重设计**：提出分块稀疏注意力（chunk-wise sparse attention），每块访问有界局部上下文+少量远距离帧+线性注意力全局状态，使激活内存与计算随序列长度线性增长；与现有长视频稀疏注意力仅在加速目标上不同的本质在于"同步提供逐块独立文本条件，支持 session 中段指令切换"。
- **30 秒自强制分布匹配蒸馏**：在 20 块自强制 rollout（共 189 latent frames ≈ 31.4 s）上进行全窗口 distribution matching，梯度只回传单块（history detach），监督时域覆盖完整 rollout；与常规短窗蒸馏的本质区别在于"监督时域远大于梯度时域，从而暴露局部合理但全局不一致的内容漂移"。
- **Evocation（召唤）能力**：逐块文本条件配合外部几何记忆，使新增/移除元素在锚定场景结构保持不变的前提下可在 ongoing session 中实时生效（unanchored 内容实现率 67%，anchored 区域覆盖仅 4%）。
- **系统性能**：三步 CFG-free 学生在单卡 H200、384×640 下每 1.5 s chunk 耗时 2.11 s；在 WBench 导航 split 上取得 SOTA（80.8 avg），VBench-2.0 得分 66.77（第 1/10），VBench-Long 得分 85.11（第 7/10）。

## 方法详解
- **Recurrent Session Formulation**：每步生成 $F=9$ latent frames（36 pixel frames，1.5 s @ 24 fps）的 chunk $x_k$，条件为相机轨迹 $\mathcal{P}_k$、局部历史 $h_k$（固定 19 latent frames）和文本条件 $c_k$。世界状态银行操作满足 $r_k = \text{Read}(M_k, \mathcal{P}_k)$、$x_k \sim p_\theta(\cdot | r_k, h_k, c_k)$、$M_{k+1} = \text{Write}(M_k, x_k, \mathcal{P}_k)$，上下文和位置范围保持有界。
- **Geometric World State Bank**：新生成 chunk 经单目深度估计（Depth Anything 3）得到 12 帧深度图，反投影追加到世界状态银行；Read 阶段按共视度排序选取至多 8 个源视图，经批处理投影+z-buffer 渲染得到 warped 观测和可见性 mask。可见性 < 0.5 的像素被设为 $\sigma=1$ 不注入视觉信息，支持区域噪声 $\sigma \in [0, 0.135]$；同 mask 在 patch 级用于剔除不支持的 history token。当前实现保留 90 s 几何（2160 pixel frames，源池至多 720 帧）。
- **Chunk-wise Sparse Attention Teacher**：基于 14B Wan2.2 A14B 扩散 transformer，保留高低噪声 expert 按 timestep 选择；每查询 chunk 访问：首帧全局 sink、含一帧重叠的局部上下文、空间压缩的近邻帧、少量选中远距离帧、线性注意力全局状态。Teacher 与 Critic 共享 backbone，LoRA 开关区分二者。监督时域 $W$ 控制训练目标覆盖的 chunk 数。
- **Long-Horizon Distribution Matching**：全窗口 189 latent frames（≈31.4 s）联合打分，score difference $\Delta s = s_{\text{fake}} - s_{\text{real}}$，归一化 $\nu = \text{mean}_\Omega[|\hat{x}_0 - s_{\text{real}}|]$，DMD loss $\mathcal{L}_{\text{gen}} = \frac{1}{2}\|\hat{x}_0 - (\hat{x}_0 - \Delta s / \nu)^{\text{detach}}\|_2^2$。掩码 $\Omega$ 排除 ground-truth prefix 和第一生成 chunk $x_1$（避免边界 flicker）。跨 chunk 历史 detach 使梯度时域限于单块，但监督时域覆盖完整 rollout。辅助 warp-conditioning 损失保持相机可控性。
- **Per-chunk Conditioning**：训练文本按 12 s 分段映射到对应 latent chunk；推理时 timed prompt schedule 允许 session 中段指令变更。
- **三步 CFG-free 推理**：每 chunk 三次无 classifier-free guidance 的去噪评估，粗到细 latent pyramid（$12\times20, 24\times40, 48\times80$），几何条件仅注入最粗阶段。

## 实验与结果
- **数据集**：Sekai 视频数据集 + 内部视频数据。训练使用 6×8 GPU。
- **WBench 导航 split（n=158）**：Evoke 获得 Quality avg 82.79、Setting avg 83.76、Physical avg 72.06，Public leaderboard 平均 80.8 排名第一（超越第二名 H iDream-O1-World 的 80.7）；Scene 74.68、Causal Fidelity 82.44、Segment 100.00 提升显著。Navigation（78.63）和 Perspective（69.74）相对较弱。
- **VBench-2.0**：总分 66.77，排名 1/10，小幅领先 Veo 3（66.72）。
- **VBench-Long**：总分 85.11，排名 7/10，接近 Veo 3（85.06）和 IPOC（85.71）。
- **长时稳定性**：8 条 65.5 分钟 rollout（共 2619 chunks/条），光度统计在初始瞬态后趋于稳定；内容描述符在 session 早期快速变化后趋缓，与真实视频控制曲线相当。非永久身份保持声明，而是"无 runaway 退化"。
- **时延**：单 H200、384×640、三步 CFG-free，每 chunk 2.11 s（纯扩散 wall clock）；几何路径额外增加约 1.84 s（见附录 B）。
- **消融**：长程教师 vs 短程教师对照显示 photometric stability 显著提升（Wilcoxon p=0.016），亮度稳定在 101% vs 74%；内容描述符和 sharpness 无显著分离。监督窗长 W 的 detectability 在 W≥2 chunk 后趋于饱和，无 sharp threshold。

## 相关工作脉络
- **交互式世界模型（保留历史在 denoiser 侧）**：Matrix-Game 2.0、HY-World 1.5、LingBot-World 等通过额外 context frames 或 KV cache 保留历史，并用 windowing/eviction/streaming conditioning 控制增长；Evoke 的本质差异是将持久状态外置到几何存储，denoiser 侧永远有界。
- **几何条件方法**：Warp-as-history（Wang & He, 2026）、Lyra 2.0、Video World Models with Long-Term Spatial Memory（Wu et al., 2026）等将生成观察提升到场景表示并渲染到目标视角；Evoke 扩展为相机位姿寻址的显式有界读写 eviction 世界状态银行，支持持续 session。
- **少步蒸馏**：Distribution Matching Distillation（DMD）、Self-forcing、Consistency Models 等将慢教师压缩为少步学生；Evoke 独特定位是将教师本身作为设计变量，针对长时监督时域和动态 conditioning 重设计，而非仅用教师作为高质量 target source。
- **稀疏注意力长序列建模**：Moga（Jia et al., 2025）、Radial Attention（Li et al., 2025c）、Kimi Linear 等；Evoke 的差异在于将稀疏注意力与逐块独立文本条件耦合，服务于交互式蒸馏监督。
- **教师-学生因果失配**：Causal Forcing（Zhu et al., 2026）关注双向教师与因果学生的不匹配；Evoke 聚焦于监督时域和 conditioning schedule 两个正交设计变量。
- **Few-step 视频生成评测**：VBench-2.0、VBench-Long、WBench 构成评测基线体系；Evoke 以三步无 CFG 逼近多步系统性能，是效率-质量权衡上的新 Pareto 前沿。

## 局限性与未来方向
- 当前几何世界状态仅保留粗粒度场景结构，细粒度物体身份、外观和局部细节的长期一致性仍有限；更丰富的 object-level 或语义世界表示是下一步。
- 持久世界需同时建模静态几何和动态状态（物体运动、状态转换及其长时演化），现有工作尚未覆盖动态世界状态的连续更新表示。
- 推理加速仍有空间：更高压缩比的视频 VAE、更高效少步生成器、更低成本的几何 conditioning 路径均可进一步改进。
- 长程监督窗长的收益存在饱和现象，实际最优 W 仍需经验调参；per-chunk conditioning 的中段响应能力提升在当前样本量下未显示与教师时域的统计显著差异。
- 几何召回在留存窗口不足时退化为 far-pose floor（15.4–17.8 dB 平台），表明当前为"可识别而非像素忠实"重建。

## 研究启发与可借鉴点
- **状态-生成解耦范式**：将持久世界状态外置（几何存储+相机寻址）而非扩展 denoiser 上下文，是可迁移的系统级设计原则，适用于任意需要长 session 记忆的生成任务（如游戏环境生成、虚拟仿真）。
- **监督时域与梯度时域解耦**：history detach 使梯度计算限于单块，而监督信号覆盖完整 rollout，这一"长监督/短梯度"策略可在不突破显存限制的前提下将教师长程能力转移给少步学生。
- **逐块 conditioning 作为交互式控制接口**：将文本 prompt 按 chunk 分段映射到 rollout 各段，并在训练中显式包含 prompt 切换样本，使蒸馏后的学生天然支持 session 中段事件注入/撤回，值得迁移到多模态 agent 场景。
- **可见性掩码驱动的几何条件注入**：基于 z-buffer 可见性动态调节 warp 噪声水平和 history token 有效性，避免不可靠几何污染当前生成——这一"有条件注入"机制可直接用于其他 warp-based 视频生成系统。
- **长程教师蒸馏的价值验证**：Fig. 6 的对照实验明确证明：相同蒸馏配方下，仅更换教师时域即可显著提升学生的光度稳定性，为"教师设计先于蒸馏策略"提供了实证依据。

## 关键术语表
**World State Bank**：外部相机索引的有界几何存储，记录已生成场景的深度反投影结果，支持 Read/Write/eviction 操作。
**Evocation（召唤）**：在 ongoing session 中通过 timed prompt schedule 动态引入或移除文本指定元素的能力，锚定场景结构保持不变。
**Chunk-wise Sparse Attention**：将长序列划分为 chunk，每查询 chunk 仅访问有界局部上下文+少量远距离帧+线性注意力全局状态，使计算和激活内存随序列长度线性增长。
**Self-forced Rollout**：学生以自身生成历史作为后续条件进行的 rollout（非 ground-truth 引导），用于模拟部署时的累积 conditioning shift。
**Distribution Matching Distillation（DMD）**：通过教师- critic score difference 直接优化学生输出分布与教师分布的一致性，无需多次采样估计梯度。
**Degradation Drift**：曝光偏移、饱和度变化、纹理退化等局部统计渐变，可在短窗口内检测但根源可能在前序累积扰动。
**Content Drift**：场景身份、物体外观或空间布局的渐进演变，每短窗口 individually plausible 但长距比较时暴露不一致。
**CFG-free（无 Classifier-Free Guidance）**：不使用条件/无条件双次前向的 guidance 机制，降低推理计算开销。

## 可复现要素
- **数据集**：Sekai 视频数据集（Li et al., 2026）+ 内部视频数据；Sekai 已公开发布，内部数据论文未说明开源情况。
- **代码**：https://github.com/SII-YuanyangYin/Evoke（已开源）。
- **项目页面**：https://evoke-world.github.io/Evoke/。
- **模型权重**：论文未明确说明权重是否单独开源，代码仓库中包含相关实现。
- **关键超参**：三步 CFG-free 去噪、30 秒监督窗（≈20 chunks/189 latent frames）、90 秒世界状态银行保留窗口（2160 pixel frames）、chunk 大小 9 latent frames（1.5 s @ 24 fps）、局部历史 19 latent frames（3.2 s）、至多 8 个检索源视图、可见性阈值 0.5、$\sigma \in [0, 0.135]$（支持区）/ $\sigma=1$（不支持区）、文本分段间隔 12 s、分辨率 384×640。
- **硬件**：训练 6×8 GPU；推理评估在单 H200 上进行。
