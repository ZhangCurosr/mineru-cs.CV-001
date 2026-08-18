---
title: "Dynamic Context Adapters: Eficiently Infusing History into Vision-and-Language Models"
source: https://arxiv.org/pdf/2608.10525v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:02:17"
---

# 论文速读：Dynamic Context Adapters: Eficiently Infusing History into Vision-and-Language Models

## 一句话总结
本文提出动态上下文适配器（DCA），通过将变长历史视觉帧压缩为固定大小的可学习记忆向量，并以轻量交叉注意力逐层注入冻结的预训练VLM，在不膨胀输入序列与改动原架构的前提下，实现线性复杂度的长程时序上下文整合，在VLN任务上以3B小模型达到逼近7B基线的性能，同时显著降低FLOPs与显存占用。

## 研究问题与动机
- **静态VLM缺乏时序记忆**：现有预训练VLM（如LLaVA、Prismatic）仅处理单帧图像-文本对，无法在部分可观测（POMDP）的连续导航环境中维持跨步历史感知。
- **Token拼接引发计算爆炸**：直接将历史帧拼入LLM输入（如NaVid、UniNavid）会导致注意力复杂度呈二次方增长，输入长度受限且显存消耗急剧上升。
- **循环压缩损失细粒度时序结构**：基于RNN/LSTM的隐状态压缩（如RGB-Seq2Seq、RGB-CMA）虽控制序列长度，但长期序列下信息易衰减，无法保留关键时空细节。
- **外部地图依赖强泛化性差**：基于拓扑/语义地图的记忆方法（如HAMT、GridMM）需手工构建或依赖特定传感器先验，难以直接端到端融入通用VLM框架。

## 核心贡献（创新点）
1. **提出DCA双模块架构**：通过Memory Compression Module将任意长度历史帧动态压缩为固定尺寸$C \times d$记忆向量，从根本上消除输入序列膨胀瓶颈，与直接拼接方案在计算复杂度上存在本质差异。
2. **设计跨层轻量上下文适配机制**：Memory Integration Module在LLM每一层以交叉注意力形式注入压缩记忆，并引入可学习标量$\lambda$加权融合，无需修改原VLM结构即可实现多层时序条件化，区别于单层FiLM或全局Pooling融合。
3. **实现高效的计算-性能权衡**：将历史整合复杂度从$O(S \cdot t \cdot p)$降至$O(S \cdot C)$，在仅3B参数的PrismaticVLM上以>25% FLOPs节省与~13%显存节省，性能匹敌甚至超越7B拼接基线。
4. **验证选择性时序记忆机制**：通过注意力权重可视化证明压缩模块能智能聚焦关键历史帧（如目标地标），而非均匀压缩，有效避免循环模型的信息损耗并提升可解释性。

## 方法详解
- **基座选择**：采用轻量预训练VLM PrismaticVLM (phi-2+3b)，包含ViT-CLIP视觉编码器、Phi-2语言模型与多层跨模态投影，整体冻结保持 pretrained knowledge。
- **特征编码与降维**：当前帧$X_t$与历史帧$X_{1:t-1}$经ViT编码后，历史特征经网格池化算子$\mathcal{G}$将空间patch数从$P$压缩至$p$（$p \ll P$），得到$\mathbf{F}_{1:t-1} \in \mathbb{R}^{(t-1)\times p \times d}$。
- **动态上下文压缩（Memory Compression Module）**：初始化可学习向量$\mathbf{M}_{init} \in \mathbb{R}^{C \times d}$作为Query，历史特征投影为Key/Value：$Q_M = M_{init}W_Q,\ K_F = F W_K,\ V_F = F W_V$。通过交叉注意力输出压缩记忆$\mathbf{M}_{1:t-1} = \mathrm{Softmax}(Q_M K_F^T)V_F$，复杂度仅为$O(C \cdot p)$。
- **逐层上下文适配（Memory Integration Module）**：对LLM第$k$层输入$z_{k-1}$，将其投影为Query，压缩记忆投影为$K_M、V_M$，计算上下文增强输出$z_k^{context} = \mathrm{Softmax}(Q_{k-1} K_M^T)V_M$。
- **层间融合与输出**：最终层输出按可学习标量融合：$z_{k+1} \leftarrow z_{k+1} + \lambda z_{k+1}^{context}$。整个前向过程仅增加极少参数，保持原始VLM推理图不变，最终经Action Head解码下一步动作$a_t$。
- **训练目标**：采用标准next-token prediction损失进行端到端微调，压缩向量与适配器参数联合优化。

## 实验与结果
- **数据集与设置**：VLN-CE R2R Val-Unseen划分，RGB单模态输入，低层动作空间，评估指标包括SR、SPL、OSR、TL、NE。
- **导航性能**：DCA（3B）SR达25.3%，SPL达12.9
