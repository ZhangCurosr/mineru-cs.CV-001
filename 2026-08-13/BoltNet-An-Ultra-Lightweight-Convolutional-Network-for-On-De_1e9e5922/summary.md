---
title: "BoltNet-An-Ultra-Lightweight-Convolutional-Network-for-On-De"
source: https://arxiv.org/pdf/2608.11844v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 10:32:57"
---

# 论文速读：BoltNet-An-Ultra-Lightweight-Convolutional-Network-for-On-De

## 一句话总结
本文提出 BoltNet，一种结合无参数通道-空间重排（SRB 与 LPS）的超轻量全卷积网络，专为在内存、算力与功耗严格受限的边缘设备上完成高基数、长尾分布的植物物种细粒度识别而设计；在 <2 MB 体积约束下，BoltNet 于 Pl@ntNet-300K 取得最优 F1-score，并在 CPU/GPU/NPU 三类异构硬件上均保持最高的端到端能效比。

## 研究问题与动机
- **细粒度长尾识别与边缘部署的矛盾**：公民科学图像（如 Pl@ntNet-300K）包含 1081 个物种、强标签歧义与陡峭长尾分布（后 80% 物种仅占 11% 样本），传统紧凑模型在稀有类上性能急剧崩塌，而重型骨干网又无法部署于野外终端。
- **部署成本不等于参数量或 FLOPS**：中间激活张量占用、算子数据类型与硬件调度策略共同决定实际延迟与能耗；仅凭理论复杂度指标无法预测跨平台真实表现。
- **现有超轻量模型目标场景错配**：EmergencyNet、TakuNet 等专为低基数应急图像设计，在千级分类任务上精度严重不足；主流 Mobile/EfficientNet 系列在 <2 MB 极限预算下难以兼顾细粒度判别力。
- **亟需统一的跨硬件能效评估范式**：单一 device 测试结果缺乏可迁移性，需在 CPU、GPU、NPU 上同步度量 FPS/W，才能指导实际野外部署选型。

## 核心贡献（创新点）
1. **提出 BoltNet 超轻量全卷积架构**：在 <2 MB 与 <100M
