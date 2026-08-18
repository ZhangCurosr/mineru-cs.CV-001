---
title: "Class Activation Mapping in Explainable Computer Vision: A Method-Centered Review of CNN, Transformer, and Foundation-Model-Era Visual Explanations"
source: https://arxiv.org/pdf/2608.12299v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 08:08:38"
---

# 论文速读：Class Activation Mapping in Explainable Computer Vision: A Method-Centered Review of CNN, Transformer, and Foundation-Model-Era Visual Explanations

## 一句话总结
本文系统综述了自2016年以来类激活映射（CAM）在可解释计算机视觉领域的方法演进，构建了涵盖CNN、Transformer与基础模型时代的八类方法论体系，并揭示了CAM研究从“单类分数定位”向“对比式、多层级、Token感知与先验驱动”的范式迁移，同时指出评估标准碎片化与因果忠实性缺失是当前核心瓶颈。

## 研究问题与动机
- **决策黑盒与高风险场景需求**：深度视觉模型在分类、分割、医疗、自动驾驶等任务中表现优异，但其分层表征导致决策逻辑不可解释；仅凭正确类别不足以建立信任，需可视化证据以验证模型是否依赖有效语义而非背景捷径或虚假关联。
- **CAM方法爆发但缺乏统一梳理**：自Grad-CAM将CAM从架构专用扩展为通用后验工具后，方法迅速分化至梯度优化、非梯度分数/消融、高分辨率融合、Transformer Token归因、因果去偏及CLIP/DINO/SAM等基础模型适配，亟需方法论层面的系统归类与演进脉络厘清。
- **评估协议碎片化阻碍横向对比**：不同论文在目标层选择、归一化方式、阈值策略、扰动基线与后处理（如CRF）上差异显著，导致Faithfulness、Localization、计算成本与人类信任等指标无法直接比较，形成“各说各话”的评估孤岛。
- **本文动机**：以“方法中心”为边界严格筛选57篇核心文献，明确各方法解决的瓶颈与遗留空白，提出标准化评估卡，为后续研究提供选型指南与可复现的实验报告规范。

## 核心贡献（创新点）
- **构建严格方法中心语料库与筛选协议**：基于布尔检索与IEEE/ACM/CVF/Springer/Elsevier等严格Venue过滤，排除纯应用型Grad-CAM引用与 surveys，最终纳入57篇真正生成、优化、评估或理论分析CAM的论文。
- **提出多维度CAM分类体系**：按归因机制（梯度/非梯度/混合）、架构依赖性（CNN/Transformer/基础模型）、评估目标（后验解释/弱监督定位分割/高分辨率/因果去偏）划分8大类，明确每类的核心机制、代表工作与适用边界。
- **刻画技术演进主线与权衡规律**：指出CAM已从“单一类分数定位”转向“对比式、概率化、Token感知与基础模型先验驱动”；揭示梯度法快但易饱和、非梯度法忠实但昂贵、高分辨率需配合语义目标否则放大噪声、基础模型方法依赖Prompt/先验来源等系统性权衡。
- **提出CAM标准化评估卡规范**：呼吁未来工作统一报告模型与目标层、归一化方式、阈值、前向/反向次数、是否使用CRF/SAM/其他先验、完整指标集合，以提升跨论文可比性。
- **明确因果与提示敏感性为下一代方向**：强调解释应从“画得更清晰”转向“解释正确的理由”，建议将因果干预、反事实编辑、Prompt/参考类敏感性测试纳入标准评估流程。

## 方法详解
- **通用CAM形式**：$L^c = h(\sum_k \alpha_k^c A^k)$，其中 $A^k$ 为选定层的特征图，$\alpha_k^c$ 为第k个通道/Token的重要性权重，$h$ 通常为ReLU等非负激活。各方法核心差异在于 $\alpha_k^c$ 的估计策略与 $A^k$ 的提取层级。
- **梯度类CAM**：Grad-CAM用目标分数对特征图的平均梯度加权；Grad-CAM++通过正偏导数组合增强多目标覆盖；Integrated Grad-CAM沿输入路径积分缓解梯度饱和；Relevance-CAM利用逐层相关性传播替代普通梯度以提升中层稳定性；LIFT-CAM基于DeepLIFT近似SHAP系数赋予理论可加性；Transformer归因（Chefer等）通过Deep Taylor Decomposition沿注意力与残差路径反向传播相关性，强调“注意力可视化≠解释”。
- **非梯度/扰动类CAM**：Score-CAM用激活图做掩码前向推理，以目标类置信度作权重；Ablation-CAM直接测量移除特征图的分数下降；Eigen-CAM取主成分实现无类特定解释；ReciproCAM通过空间扰动实现轻量快速归因；ScoreCAM++引入tanh门控强化高低优先级区分。
- **高分辨率与细粒度方法**：LayerCAM融合浅层空间细节与深层语义证据；Poly-CAM多尺度层融合；Finer-CAM将解释目标从“类分数”改为“目标类vs参考类logit差”，显著提升细粒度区分能力；F-CAM与Augmented CAM通过可训练解码器或数据增强实现超分。
- **Transformer与Token级方法**：TS-CAM耦合Token语义与自注意力；MCTformer引入多Class Token实现类特异性定位；CTI通过图内/图间Class Token注入提升一致性；Prompt-CAM学习类特定Prompt使注意力头揭示细粒度特征。
- **基础模型时代方法**：gScoreCAM用梯度选通道+Score-CAM加权加速CLIP归因；S2C将SAM分割先验注入CAM训练；DINO语义引导器基于自监督亲和图扩散CAM种子；DiffCAM基于特征分布差异而非决策边界梯度生成显著图；MetaCAM通过Top-k共识集成多方法提升鲁棒性。
- **因果与去偏方法**：CI-CAM通过因果干预解耦目标-上下文混淆；C-CAM建模医学图像中的类别-解剖因果链；Debiased-CAM训练多任务去偏模型以抑制模糊、色温、昼夜变化等扰动偏差。

## 实验与结果
- 本文为综述，不开展统一实验，而是从57篇文献中抽取兼容协议下的定量结果进行横向对比。
- **运行开销对比**：Grad-CAM/++因单次反向传播速度最快；Score-CAM在CLIP设定下需数千次前向传播显著变慢；gScoreCAM与ReciproCAM在保持定位质量的前提下将耗时降低约8倍至148倍。
- **忠实度（Faithfulness）趋势**：在ILSVRC2012/VGG16协议下，LayerCAM与Poly-CAM通过高分辨率融合在保持插入AUC的同时优化了删除AUC；ScoreCAM++通过门控显著降低Average Drop并提升信心保留；进步非单调，新方法未必全面优于旧方法。
- **定位精度（Localization）结果**：LIFT-CAM在ImageNet能量定位协议下将解释能量更紧凑地置于边界框内；TS-CAM在CUB-200-2011上取得大幅提升的Top-1定位分数；CI-CAM在CUB上Top-1定位达58.39%，ImageNet结果与基线相当；SEAM在PASCAL VOC上伪标签mIoU提升至55.41%（CRF前）。
- **核心结论
