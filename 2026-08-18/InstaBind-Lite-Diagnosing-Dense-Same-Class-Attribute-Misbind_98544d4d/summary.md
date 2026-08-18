---
title: "InstaBind-Lite-Diagnosing-Dense-Same-Class-Attribute-Misbind"
source: https://arxiv.org/pdf/2608.16805v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:24:53"
---

# 论文速读：InstaBind-Lite-Diagnosing-Dense-Same-Class-Attribute-Misbind

## 一句话总结
本文形式化并提出了**稠密同类属性错绑（DSCAM）**这一被传统评测掩盖的失败模式，构建了高纯度诊断基准 **InstaBind-Lite**。通过受控同类组、有序实例标注与四层级问题设计，揭示了七款主流 LVLMs 在高分准确率下仍普遍存在从相邻实例直接复制属性的隐蔽错误，且局部定位与推理时提示仅对部分模型有效。

## 研究问题与动机
- **核心问题**：LVLMs 在密集同类场景中容易将图像中真实存在的属性错误地分配给另一个可见同类实例，而现有 VQA 准确率或对象幻觉指标无法区分“属性正确但归属错误”与“完全不存在的幻觉”。
- **通用 VQA/多模态基准盲点**：VQA、MMBench、MME 等仅评估最终答案对错，将错绑、随机猜测与识别失败统一归为错误，无法反映实例-属性对应可靠性。
- **幻觉评测盲点**：POPE、CHAIR、HallusionBench 等只验证预测内容是否在图像某处存在，不验证其是否归属于目标实体，导致“图中有依据但归属错”被漏检。
- **定位与组合性评测盲点**：RefCOCO、Flickr30K 以 IoU 为主，无法检测“定位正确但属性抄自邻居”的绑定失败；CLEVR、Winoground 等侧重语言组合敏感度，未建立自然密集场景中的定向实例间转移度量。

## 核心贡献（创新点）
1. **形式化 DSCAM 为独立失败类别**：将错误答案严格拆分为“无依据幻觉”、“属性识别失败”与“可见属性从另一实例复制”，解决现有评测中归属模糊的歧义问题。
2. **构建高纯度诊断基准 InstaBind-Lite**：包含 524 张图像、529 个受控同类组、1773 个实例与 9580 个
