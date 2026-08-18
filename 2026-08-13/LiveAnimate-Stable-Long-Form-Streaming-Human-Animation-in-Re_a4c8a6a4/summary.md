---
title: "LiveAnimate-Stable-Long-Form-Streaming-Human-Animation-in-Re"
source: https://arxiv.org/pdf/2608.11745v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 12:32:23"
---

# 论文速读：LiveAnimate-Stable-Long-Form-Streaming-Human-Animation-in-Re

## 一句话总结
本文提出 LiveAnimate，首个在十亿参数规模下同时实现实时流式生成与稳定长程生成的人类动画系统。通过将双向 14B DiT 转换为块因果生成器并结合姿态感知有界 KV 缓存，在 2×H100 GPU 上以约 19.63 FPS 完成三分钟长视频生成，身份与外观在整个滚出过程中保持恒定，而离线基线需 2–5 小时且会出现严重退化。

## 研究问题与动机
1. 现有扩散驱动的人体动画方法均为离线批处理，单片段生成需数分钟至数小时，无法满足直播、虚拟形象、远程Presence等交互式场景的实时响应需求。
2. 百亿/十亿参数全身体扩散模型（如 UniAnimate-DiT、Wan-Animate）虽能生成高质量短片，但缺乏处理开放长度姿态流的能力，无法在推理时持续滚动生成。
3. 即便支持长程生成的 EverAnimate 仍需每块 20 步去噪，距离实时交互仍有数量级差距；且长序列流式生成面临显存/延迟线性增长与身份外观漂移的固有矛盾。
4. 现有自回归视频生成方法多关注通用视频的滚出漂移缓解，未针对姿态驱动场景下的极低延迟预算与 pose-aware 缓存管理进行联合系统设计。

## 核心贡献（创新点）
1. 提出 LiveAnimate 系统，首个在十亿参数规模下联合实现实时流式、低延迟与稳定长程生成的人类动画框架。与已有工作的本质区别在于首次将 streaming、real-time 与 stable long-form 三项约束在同一 14B DiT 上端到端优化，而 prior work 仅覆盖其中 1–2 项。
2. 设计 Reference-Anchored Teacher-Forcing Adaptation，通过全局 Ref Sink 与 ground-truth clean history 将预训练双向 DiT 安全转为块因果生成器。本质区别在于解决了条件生成场景下标准 teacher forcing 无法保留参考身份锚点的缺陷，区别于无条件文本/视频任务的单向因果适配。
3. 提出 Block-wise Self-Forcing Distillation (BS-DMD)，以单块重放与停止梯度策略完成分布匹配蒸馏，将采样步数压缩至 3 步。本质区别在于避免了全 rollout 图驻留导致的显存爆炸，使 14B 模型蒸馏可在单节点 8×80GB GPU 完成，区别于原有 Self Forcing 的全局梯度回传或 Causal Forcing 的严格教师监督路径。
4. 设计 Pose-Retrieval Sink Attention (PR-Sink)，融合 Static Sink、Dynamic Sink 与三槽 Rolling Window，使显存与每块延迟不随流长度增长。本质区别在于引入 pose-aware 检索机制与多样性保持的 Bank Update，专门应对姿态循环重现时的外观恢复，区别于传统固定滚动窗口或纯 attention sink 方案。

## 方法详解
- **两阶段训练管线**：Stage 1 采用 Reference-Anchored Teacher-Forcing Adaptation，将视频按时间划分为 $B_f$ 个潜帧块。每个块 $b$ 去噪时仅 attending 到 ground-truth clean 历史块 $z_{<b}^{\mathrm{GT}}$ 与全局 Ref Sink（参考图 latent 的 KV），Attention 计算如公式 (1) 所示：$\mathrm{Attn}(Q_b, K, V) = \mathrm{softmax}\left(\frac{Q_b [K_{\mathrm{ref}}; K_{<b}]^T}{\sqrt{d}}\right)[V_{\mathrm{ref}}; V_{<b}]$，确保参考图作为永久上下文锚点。Stage 2 采用 BS-DMD，先对 N 个块执行无梯度的 Self-Forcing Rollout 得到轨迹 $(\bar{z}_1, \dots, \bar{z}_N)$，再逐个位置 $T$ 重放并计算 $\mathcal{L}_T = \mathcal{L}_{\mathrm{DMD}}(\tilde{z}^{(T)}; \mathcal{C})$，实现一步蒸馏至 3 步采样。
- **Clean KV Update**：每个块去噪结束后，在干净时间步 $t=0$ 额外执行一次前向传播 $\mathrm{KV}_{\mathrm{cache}} \leftarrow \mathrm{KV}_{\mathrm{cache}} \cup \mathrm{Forward}(\hat{z}_b^0, t=0, \mathcal{C})$，将干净状态的 KV 写入历史
