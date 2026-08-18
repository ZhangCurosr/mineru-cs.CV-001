---
title: "ProtoHGF-Net-Prototype-HyperGraph-Fusion-with-Intra-modal-Ca"
source: https://arxiv.org/pdf/2608.11595v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 12:38:09"
---

# 论文速读：ProtoHGF-Net-Prototype-HyperGraph-Fusion-with-Intra-modal-Ca

## 一句话总结
针对现有RGBT目标检测中全分辨率密集跨模态融合易引入背景干扰与模态错位的问题，本文提出ProtoHGF-Net，将交叉模态融合重构为原型级语义交互，并设计单模态教师掩码引导的预融合校准蒸馏（TM-Calib），在DroneVehicle、DVTOD和FLIR三个基准上均达到SOTA性能。

## 研究问题与动机
1. **密集全分辨率融合的 background 干扰**：现有方法假设所有空间位置与模态响应均参与跨模态交互，导致大量背景区域参与信息传播，引发不必要的背景耦合与不稳定的模态间干扰，削弱目标相关表征。
2. **预融合缺乏目标导向的特征校准**：现有蒸馏多依赖融合后的多模态教师，若教师特征本身含强背景响应，会将背景主导信息传递给student
