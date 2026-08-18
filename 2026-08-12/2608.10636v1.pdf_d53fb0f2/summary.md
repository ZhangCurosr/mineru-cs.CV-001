---
title: "DistilVDR: A Compact End-to-End Visual Document Retriever via Dual-Student Distillation"
source: https://arxiv.org/pdf/2608.10636v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:41:40"
field: "视觉文档检索与模型压缩"
keywords: ["视觉文档检索", "蒸馏", "单向量检索器", "多模态嵌入", "ViDoRe", "Cosine alignment", "Dynamic tiling"]
innovations: ["双边余弦对齐蒸馏：两学生独立回归 8B 教师 embedding，无需负样本或对比项", "输入非对称性匹配的 encoder-only 架构：454M 文档塔 + 70M 查询塔", "固定 visual-token 预算动态分块 + 缩略图，解决低分辨丢失与上下文超限两类失败模式"]
benchmarks: ["ViDoRe v1", "ViDoRe v2", "ViDoRe v3"]
---

# 论文速读：DistilVDR: A Compact End-to-End Visual Document Retriever via Dual-Student Distillation

## 一句话总结
DistilVDR 是一个 524 M 参数的端到端视觉文档检索（VDR）系统，通过对单个 8 B 视觉语言教师做双边余弦对齐蒸馏，产出紧凑的单一向量检索器；HiRes 变体在 ViDoRe v1+v2+v3 上达到 61.74 NDCG@5（教师的 86.9%），以 15.6× 更小的索引和 10× 更快的索引速度，击败所有复现的 sub-1 B 基线。

## 研究问题与动机
- **VDR 被大模型主导**：主流系统 2–8 B 参数，单页编码需 >16 GB VRAM，百万文档语料索引需数十 GPU 小时，难以在生产环境部署。
- **已有压缩路线各有缺陷**：① 从头训练小多向量编码器（如 colSmol、SauerkrautLM-ColLFM2、ModernVBERT）虽参数少，但每 token 存储使索引膨胀 10×、MaxSim 评分延迟膨胀 100×；② 蒸馏路线（如 NanoVDR）仅蒸馏查询端，文档端仍保留 2 B 教师，百万文档索引仍需数十 GPU 小时。
- **单向量压缩代价质量**：BiModernVBERT 比 ColModernVBERT 在 ViDoRe v1/v2 分别低 17.6 / 20.3 NDCG@5，说明小编码器从头对比学习难以学到多向量级质量。
- **缺乏端到端紧凑单向量方案**：现有工作要么只蒸馏一端，要么依赖已有强基座做后处理（剪枝/融合/重排序），未从根本上把教师整条路径替换掉。

## 核心贡献（创新点）
- **双边余弦对齐蒸馏**：两个学生各自独立回归教师缓存 embedding，无需相关性标签、负采样或对比项，使文档侧与查询侧蒸馏完全解耦、可并行训练。
- **输入非对称性匹配的编码器结构**：文档塔 454 M（InternViT-300M + ModernBERT-base）集中视觉容量，查询塔 70 M（DistilBERT-base）仅文本，与 VDR 的"文本查询 × 图像文档"非对称输入天然对齐。
- **固定 visual-token 预算的动态分块**：按页宽高比选择网格布局并追加单缩略图，解决"低分辨丢失小文字/表格"与"高分辨超出文本上下文窗口"两类失败模式。
- **统一基准复现与 profiling**：在相同评估驱动下复现 12 个 250 M–8.8 B 检索器，同时报告 NDCG@5、查询延迟、索引吞吐、峰值 VRAM、索引体积、评分延迟六个维度。
- **双部署变体**：HiRes（6 tile + 缩略图）与 Fast（2 tile + 缩略图）共享同一套编码器与训练，仅发布时切换视觉 token 预算，HiRes 在 v3 上领先所有 sub-1 B 基线 13.55 点。

## 方法详解
- **问题设定**：文本查询 $q$ 与文档图像集 $\{d_i\}$，查询编码器 $f_q \rightarrow \mathbf{q} \in \mathbb{R}^{4096}$，文档编码器 $f_d \rightarrow \mathbf{d}_i \in \mathbb{R}^{4096}$，L2 归一化后得分 $s(q,d_i)=\mathbf{q}^\top \mathbf{d}_i$，文档向量建库时预计算一次。
- **教师**：冻结 Qwen3-VL-Embedding-8B（Li et al., 2026a），输出维度 $k=4096$，仅在训练前跑一次缓存目标 embedding，学生训练阶段完全不访问教师。
- **文档编码器 $f_d$（454 M）**：① 动态分块：按页宽高比从 $\{(p,q): n_{\min}\le pq\le n_{\max}\}$ 选网格，Resize 到 $p^* \times q^*$ 个 $448\times448$ tile，追加单缩略图；② InternViT-300M-448 视觉编码器将每 tile 输出 1024 个 768-d patch token，最多 7168 token；③ 线性投影到 768-d 后输入 ModernBERT-base（8192 上下文）做双向重编码；④ mean pooling → 768→4096 线性层 → L2 归一化。
- **查询编码器 $f_q$（70 M）**：查询文本加教师训练时的 instruction prefix $\pi$，DistilBERT-base 编码 → mean pooling → 768→4096 线性层 → L2 归一化。
- **蒸馏损失**：
  $$\mathcal{L}_d = 1 - \langle f_d(d), T(d) \rangle, \quad \mathcal{L}_q = 1 - \langle f_q(\pi\circ q), T(\pi\circ q) \rangle$$
  两学生前向从不共享，训练完全解耦；监督信号来自教师 embedding 空间（教师自身已用相关性监督训练），故学生无需任何负样本或对比项。
- **数据**：文档 1.20 M 图像（711 K base + 454 K 多域补充 + 31.7 K 金融补充，感知哈希去重）；查询 1.49 M（NanoVDR 训练集 + MarianMT 五语翻译）。
- **训练超参**：文档端 AdamW、峰值 LR $1\times10^{-3}$、batch 256、3 epoch、$2\times$ H200；查询端峰值 LR $5\times10^{-4}$、batch 512、15 epoch、$2\times$ H200；one-cycle schedule、3% warmup。
- **变体差异**：HiRes $T_{\max}=6$（6 tile + 缩略图，worst-case 7168 token）；Fast $T_{\max}=2$（2 tile + 缩略图，3072 token）。
- **对比监督消融**（Sec 5.1）：在 cosine 蒸馏 checkpoint 上加 $\mathcal{L}_{\text{InfoNCE}}$ 或 $\mathcal{L}_{\text{KL}}$ 联合精修 1 epoch，NDCG@5 无提升（InfoNCE 微降，KL 持平），归因于 InfoNCE 与教师校准几何正交、KL 与 cosine 近似冗余。

## 实验与结果
- **基准**：ViDoRe 全套 22 数据集（v1 10 集英文、v2 4 集多语、v3 8 集六大语言专业域），指标 NDCG@5（pytrec_eval，benchmark 内平均）。
- **复现基线**：12 个公开检索器（250 M–8.8 B，单/多向量）+ 8 B 教师作 oracle；Multi-vector 用作者推荐 late-interaction，Single-vector 用 brute-force dot product。
- **检索质量（Table 1）**：
  - HiRes Avg 61.74，Fast Avg 59.98；对比最强 sub-1 B 基线 colSmol-500M（53.01），HiRes +8.73、Fast +6.97。
  - v3 最难：HiRes 47.07 vs colSmol-500M 33.52（+13.55）；Fast 43.66 以 3× 更小 visual-token 预算换取吞吐。
  - 对更大模型：HiRes 超 DSE-Qwen2（2.2 B）1.03、超 ColPali v1.3（2.9 B）1.42；Fast 落后 Qwen3-VL-Embedding-2B 约 6.5 点；落后 4–8 B 多向量 6.95–9.80 点，差距集中在 v2/v3。
  - 教师 Avg 71.05，HiRes 达其 86.9%、Fast 84.4%。
- **差距分解（Table 2）**：T×T=71.05，T×S=65.02（-6.03），S×T=66.36（-4.69），S×S=61.74（-9.31）；两侧误差不线性叠加，存在交互项。
- **效率（Table 3，单 H200, B=8, bf16）**：
  - Fast 文档吞吐 99.04 docs/s、峰值 VRAM 2.10 GB；HiRes 36.82 docs/s、3.07 GB；两者查询 3.4 ms。
  - 索引：Fast/HiRes 16.4 GB/M doc（4096-d float32）；colSmol-500M 等 sub-1 B 多向量 256 GB/M；Tomoro-8B 819 GB/M → 15.6× 更小。
  - 评分：10K 文档 9.6 ms vs 多向量 1.1–3.2 s → 100× 更快；Fast 较 8 B 教师 18× 更快。
- **消融（Table 4）**：
  - Tile 预算：0 tile（单图）较 HiRes 平均低 5.12 点；Fast→HiRes 加 3× token 换 +1.76 平均 / +3.41 v3。
  - 数据规模：25%/50%/75%/100% 单调上升，满数据较 1/4 数据高约 5 点。
  - 输出维：4096→768 索引 5.3× 压缩（16.4→3.07 GB/M），v3 损失 4.77 点、平均 3.19 点。
  - 查询骨干：ModernBERT-base（149 M）67.40 最高但 2.2× 参数、5.1× 延迟；DistilBERT-base（70 M）66.36 取舍。
- **结论**：524 M 单向量在质量-成本 Pareto 前沿上全面优于已发布的 sub-1 B 多向量路线，同时在部署维度（索引体积、评分延迟、编码吞吐、峰值 VRAM）均占优。

## 相关工作脉络
- **ColBERT / 多向量 late interaction**（Khattab & Zaharia, 2020；Santhanam et al., 2022）：Per-token 存储与 MaxSim 评分是 DistilVDR 所要替代的范式，后者以单点积换取 100× 评分加速与 15× 存储压缩。
- **NanoVDR**（Liu et al., 2026）：同团队前期工作，仅蒸馏 70 M 查询端，文档端仍用 2 B 教师；DistilVDR 将其扩展为双边蒸馏、端到端替换教师。
- **ModernVBERT**（Teiletche et al., 2025）：encoder-only 单向量设计，比因果 decoder 高 10.6 NDCG@5；但仍属从头训练小编码器路线，未做蒸馏。
- **SigLIP2 / InternViT**（Tschannen et al., 2025；Chen et al., 2024b）：视觉 backbone 选择依据——SigLIP2-L 单图性能弱（Avg 25.93），InternViT-300M 配合动态分块显著改善。
- **Cross-encoder → bi-encoder 蒸馏**（Hofstätter et al., 2021；Chen et al., 2024a）：文本检索成熟范式；本文将其推广至跨模态（图像↔文本）非对称场景，目标是几何 embedding 对齐而非标量相关性分数。
- **OCR-free 文档理解**（Donut、Pix2Struct、UDOP、LayoutLMv3）：共享"直接入图避免 OCR 误差累积"前提，但面向生成/抽取任务；DistilVDR 面向最近邻检索，输出单 dense vector。
- **Universal multimodal embedders**（VLM2Vec、UniIR）：多模态泛化方向；DistilVDR 放弃广度换 VDR 子域深度，在 sub-1 B 量级做到更强。

## 局限性与未来方向
- **实验控制不足**：基线均为官方 release 原样评估，未用相同预算重训；缺少"同架构从头对比学习"的控制实验，无法严格隔离蒸馏贡献。
- **单教师固定**：仅用 Qwen3-VL-Embedding-8B，未扫教师规模/家族，强/弱教师对最终质量的边界未知。
- **与 4–8 B 多向量仍有 7–10 点差距**：主要落在 v3，未能分离 late interaction 与模型容量的贡献。
- **仅 ViDoRe 评测**：缺少企业内部语料、扫描/OCR 劣化页、生产 query 分布的验证。
- **学生上限受教师约束**：双边独立蒸馏无法超越被替换侧的教师，若需突破需更强监督信号（cross-encoder 分数、多向量教师）。
- **未利用跨模态耦合**：两学生独立训练，丢弃了教师隐含的 query-document 流形耦合；联合蒸馏是自然扩展。
- **分块规则内容无关**：$T_{\max}$ 固定，密集小字页与单图页预算相同；可基于低成本页面复杂度估计做自适应分配。
- **查询塔仅限拉丁语系**：英文 + MarianMT 五语翻译，中文/日文/阿拉伯文及图像条件查询不在范围内。
- **索引压缩未探索**：仅测 Matryoshka 截断（4096→768），标量量化、乘积量化、二值哈希正交未测，当前 16.4 GB/M 是上界。

## 研究启发与可借鉴点
- **双边解耦蒸馏简化训练**：缓存教师 embedding、两学生独立训练，避免 cross-encoder 蒸馏的复杂度与显存压力，可扩展到任意双塔多模态检索。
- **输入非对称性指导架构非对称**：VDR 的"文本查询 × 图像文档"自然对应"轻查询塔 × 重文档塔"，这一设计原则可迁移到其他模态不对等检索（如 audio-text、3D-text）。
- **固定 visual-token 预算 + 缩略图**：解决高分辨与上下文窗口的 trade-off，可在任意视觉编码器上复用；thumbnail 提供全局布局先验成本低收益大。
- **cosine alignment 优于 InfoNCE/KL-ranking**：在已有教师几何校准的场景下，对比项可能正交或冗余；提示我们在蒸馏设置中优先验证点wise 目标再叠加 ranking 项。
- **统一多维度 profiling 协议**：同时报告质量 + 查询延迟 + 文档吞吐 + 峰值 VRAM + 索引体积 + 评分延迟，为部署导向研究树立可复用的评估范式。

## 关键术语表
- **Visual Document Retrieval (VDR)**：给定文本查询，从文档图像库中召回相关页的检索任务，直接入图避免 OCR 误差。
- **Single-vector vs multi-vector retriever**：前者每文档输出一个 dense vector（点积评分），后者输出 per-token vectors（MaxSim 评分）；后者质量高但存储与延迟代价大。
- **Late interaction (ColBERT-style)**：查询与文档的 per-token 向量做逐维 max-dot 聚合，精度高但评分慢两个数量级。
- **Cosine alignment distillation**：学生回归教师 embedding 的余弦相似度损失（$1-\langle \cdot,\cdot\rangle$），无需负样本或相关性标签。
- **Dynamic tiling**：按页宽高比选网格把文档图切成 $s\times s$ tile，保留局部高分辨同时控制总 token 数。
- **Matryoshka representation**：教师输出任意前缀仍是合法 embedding，支持截断维度以压缩索引而不重训。
- **One-cycle schedule**：学习率从基线升至峰值再降回的调度，用于蒸馏训练加速收敛。
- **ViDoRe**：Visual Document Retrieval 基准suite，分 v1/v2/v3 三档难度，覆盖 6 语言 8 专业域共 22 数据集。

## 可复现要素
- **数据集**：ViDoRe v1/v2/v3（公开）；训练文档 1.20 M（VisRAG、ColPali train、VDR-Multilingual、racineai/VDR_MEGA_2、Sujet-Finance-Vision-10k、FinHNQue，均公开可下载）；训练查询 1.49 M（NanoVDR 训练集 + MarianMT 五语翻译，公开）。
- **代码**：已开源 https://github.com/Ryenhails/NanoVDR（论文声明）。
- **权重**：DistilVDR-HiRes / Fast 与 8 B 教师 Qwen3-VL-Embedding-8B 均已公开（论文引用 Li et al., 2026a）。
- **关键超参**：教师输出 dim 4096；文档端 3 epoch、LR $1\times10^{-3}$、batch 256；查询端 15 epoch、LR $5\times10^{-4}$、batch 512；one-cycle、3% warmup；AdamW；Fast $T_{\max}=2$、HiRes $T_{\max}=6$；tile 尺寸 448；ModernBERT 上下文 8192。
- **硬件**：训练 $2\times$ H200（文档端与查询端各）；profiling 单 H200；教师预计算 99.5 H200-GPU-hours。
- **后端**：PyTorch 2.5、CUDA 12.6、transformers ≥ 4.46、bf16、flash-attention-2。
- **评估**：pytrec_eval、NDCG@5、B=8 吞吐、B=1 查询延迟（含 tokenize）、CPU 单线程 10K 评分。
