---
title: "Map-Det3D-Metric-Feed-Forward-3D-Reconstruction-Prior-for-Mu"
source: https://arxiv.org/pdf/2608.12179v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 12:33:24"
---

# 论文速读：Map-Det3D: Metric Feed-Forward 3D Reconstruction Prior for Multi-view 3D Object Detection from Streaming Inputs

## 一句话总结
本文提出 Map-Det3D，一种仅依赖 RGB 视频流的在线多视角 3D 目标检测框架；通过将前馈度量级 3D 重建模型（FF3R）复用为几何骨干网，并在直接预测的 3D 空间中进行“Up-to-scale”检测头设计，有效克服了单目深度与绝对尺度模糊的固有缺陷，在 CA-1M 和零样本 ScanNetV2 上均取得 SOTA 性能。

## 研究问题与动机
- **单目 3D 检测的尺度与深度歧义**：从单张图像恢复绝对尺度和深度严重欠约束，依赖学习到的尺度先验在相机内参、运动模式或场景分布发生域偏移时极易失效。
- **传统 2D-to-3D Lifting 路径的脆弱性**：主流方法先在 2D 图像平面检测目标，再回归深度、尺寸和朝向，微小深度误差会被放大并主导 3D 空间中的 IoU 计算，导致域外表现急剧下降。
- **现有多视角/在线方法对传感器或离线假设的依赖**：多数可靠系统依赖 LiDAR/RGB-D 等主动深度传感器，或采用全局离线重建，难以满足轻量级消费级平台或因果在线流式感知的算力与延迟约束。
- **FF3R 模型的几何先验尚未被充分挖掘用于检测任务**：前馈 3D 重建模型已能鲁棒恢复相机位姿与稠密几何，并能输出解耦的度量尺度因子，但其强几何先验目前主要服务于场景级重建，尚未
