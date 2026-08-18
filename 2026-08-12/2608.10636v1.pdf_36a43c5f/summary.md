---
title: "DistilVDR: A Compact End-to-End Visual Document Retriever via Dual-Student Distillation"
source: https://arxiv.org/pdf/2608.10636v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:42:11"
---

# 论文速读：DistilVDR: A Compact End-to-End Visual Document Retriever via Dual-Student Distillation

## 一句话总结
论文提出 DistilVDR，一个仅 524M 参数的端到端单向量视觉文档检索系统，通过对称解耦的双学生余弦对齐蒸馏自单一 8B 视觉语言教师模型，在保持亚十亿基线最高检索质量的同时，将百万文档索引体积缩小约 15.6 倍、建库吞吐提升一个数量级，实现了质量与部署成本的最优权衡。

## 研究问题与动机
- **SOTA VDR 系统过于庞大**：当前最优检索器参数量达 2B–8B，单文档嵌入需 >16 GB VRAM，百万文档全库索引耗时数十 GPU 小时，难以在企业级场景中部署。
- **从 scratch 训练的小多向量模型代价高昂**：colSmol、ModernVBERT 等子十亿模型虽参数小，但依赖 per-token 存储与 MaxSim 晚期交互，索引膨胀一个数量级且评分延迟高两个数量级；若强行改为单向量则损失 10+ NDCG@5。
- **现有蒸馏路线未实现端到端紧凑化**：已知唯一 VDR 蒸馏工作 NanoVDR 仅蒸馏 70M 查询编码器，文档侧仍保留 2B 教师模型，索引与部署成本并未实质性降低。
- **缺乏统一的轻量单向量 VDR 方案**：亟需一种既能保留单向量检索的高效索引与点积打分优势，又能通过知识蒸馏逼近大模型质量的端到端架构与训练范式。

## 核心贡献（创新点）
- **双侧独立余弦对齐蒸馏框架**：将冻结的 8B 教师模型的文本查询与文档图像嵌入空间作为固定监督目标，两个学生编码器独立回归教师 embedding，无需相关性标签、难负样本或对比损失项；与已有蒸馏工作仅压缩查询侧或依赖对比/难负样本训练不同，本方法实现了真正端到端的双侧轻量化与训练解耦。
- **匹配 VDR 输入非对称性的 asymmetric encoder-only 架构**：文档塔采用 300M 视觉编码器（InternViT）+ 150M 文本骨干（ModernBERT）的混合设计处理图像切片，查询塔仅保留 70M 纯文本 DistilBERT；与追求模态对称的通用多模态嵌入器不同，该设计将容量精准分配给视觉密集的文档端，充分发挥编码器双向注意力的重编码能力。
- **基于宽高比自适应的动态分块与固定视觉 token 预算机制**：通过 InternVL-V2 规则为每页选择 aspect-ratio-matched 网格布局（上限 $T_{\max}$ 片 448
