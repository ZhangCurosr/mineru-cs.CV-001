---
title: "DistilVDR: A Compact End-to-End Visual Document Retriever via Dual-Student Distillation"
source: https://arxiv.org/pdf/2608.10636v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:58:13"
---

# 论文速读：DistilVDR: A Compact End-to-End Visual Document Retriever via Dual-Student Distillation

## 一句话总结
本文提出了 DistilVDR，一个仅 524M 参数的端到端单向量视觉文档检索器；通过从 8B 视觉语言教师进行双侧独立余弦对齐蒸馏，在不使用对比学习、负采样或相关性标签的情况下，在 ViDoRe 基准上达到教师 86.9% 的性能，同时将索引体积缩小 15.6 倍、建库速度提升一个数量级。

## 研究问题与动机
- **现有 SOTA VDR 模型过于臃肿**：当前最优检索器参数量达 2B–8B（如 Qwen3-VL-Embedding-8B、Tomoro-ColQwen3-8B），单文档嵌入需 >16 GB 显存，索引百万文档耗时数十 GPU 小时，难以落地部署。
- **轻量多向量路线代价过高**：colSmol、ModernVBERT 等 <1B 多向量模型虽参数小，但 per-token 存储使索引膨胀约一个数量级，MaxSim 晚期交互打分比单向量点积慢两个数量级；后处理剪枝/融合仍依赖预训练大模型。
- **已有蒸馏路线不完整**：NanoVDR 等先前工作仅蒸馏 70M 查询编码器，文档端仍保留 2B 教师，索引与部署成本未实质性降低，未实现紧凑的端到端单向量检索器。
- **单向量压缩易损失质量**：直接对多向量模型做单向量化（如 BiModernVBERT）在 ViDoRe v1/v2 上分别下降 17.6 与 20.3 NDCG@5；小参数从头+对比学习训练难以匹敌多向量质量，需更高效的蒸馏范式。

## 核心贡献（创新点）
- **双侧独立余弦对齐蒸馏框架**：查询与文档两个学生编码器分别从单个 8B 教师处独立学习 cosine 对齐目标，完全解耦且无需相关性标签或负采样。
- **不对称编码器-only 学生设计**：匹配 VDR 文本查询/图像文档的输入非对称性，将 87% 参数（454M）集中于重型文档塔，查询塔保持轻量 70M DistilBERT，实现容量精准分配。
- **动态分块+缩略图视觉令牌预算控制**：基于页面宽高比自动选择网格布局（最多 6 块 448×448 tile），附加单张全局缩略图，既避免低分辨率丢失小字表格，又防止高分辨率超出文本骨干 8192 token 上下文窗口。
- **统一全链路效率评测协议**：在相同 H200 硬件与评估驱动下复现并对比 12 个 250M–8.8B 公开检索器，同步报告检索质量、查询延迟、建库吞吐、峰值显存、索引体积与打分延迟，填补了此前多向量与单向量效率难以公平对比的空白。

## 方法详解
- **双学生架构**：冻结的 8B Qwen3-VL-Embedding 教师分别对文档图像 $d$ 与带指令前缀 $\pi$ 的查询 $q$ 进行前向编码，输出 4096 维 L2 归一化向量 $\mathbf{T}(d), \mathbf{T}(\pi \circ q)$。两个学生独立投影至同一空间并归一化，部署时教师完全丢弃。
- **文档编码器 $f_d$（454M）**：
  1. **动态分块**：按页面宽高比从候选网格集 $\mathcal{G}=\{(p,q): n_{min}\le pq\le n_{max}\}$ 中选取最匹配布局，切分为 $T_{max}$ 个非重叠 tile，附加一张同分辨率缩略图。
  2. **视觉编码**：InternViT-300M-448 将每 tile 编码为 1024 个 768 维 patch token，拼接后序列最长 7168 tokens。
  3. **上下文融合**：线性投影映射至 ModernBERT-base 嵌入空间，利用其双向注意力与 rotary position embeddings 对纯视觉序列重编码。
  4. **输出投影**：Mean pooling 后经 768→4096 线性层输出最终向量。
- **查询编码器 $f_q$（70M）**：添加与教师一致的指令前缀 $\pi$，经 DistilBERT-base 编码、mean pooling，再通过 768→4096 线性投影输出。
- **蒸馏损失**：两侧完全解耦，仅使用点wise余弦对齐损失：
  $$\mathcal{L}_d = 1 - \langle f_d(d), T(d)\rangle, \quad \mathcal{L}_q = 1 - \langle f_q(\pi \circ q), T(\pi \circ q)\rangle$$
  教师 Embedding 预计算缓存为 float32 数组后，学生训练过程无需反向穿过教师，训练完全并行化。

## 实验与结果
- **数据集**：ViDoRe v1+v2+v3 全套 22 个数据集（涵盖 EN/FR/ES/DE/IT/PT，覆盖科学论文、财务报告、工业手册、ESG、生物医学等多领域）。
- **基线**：复现 12 个公开检索器（SigLIP2-L、BiModernVBERT、colSmol-256M/500M、ColModernVBERT、SauerkrautLM-ColLFM2、DSE-Qwen2、Qwen3-VL-Embedding-2B、ColPali v1.3、Tomoro-ColQwen3-4B/8B、ColNomic-7B）及 8B 教师作为 oracle。
- **主要结果**：
  - DistilVDR-HiRes 平均 NDCG@5 达 **61.74**，以 **+8.73** 点超越最强 sub-1B 基线 colSmol-500M（53.01）；在高分辨率敏感的 v3 上以 47.07 领先次强 baseline 13.55 点。
  - DistilVDR-Fast（视觉令牌预算仅为 HiRes 的 1/3）平均
