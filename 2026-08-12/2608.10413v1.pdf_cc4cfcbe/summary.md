---
title: "DriveVLA-M0: Failure-Aware Memory Augmentation for Autonomous Driving"
source: https://arxiv.org/pdf/2608.10413v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:49:10"
field: "自动驾驶VLA规划"
keywords: ["端到端自动驾驶", "VLA模型", "测试时训练", "记忆增强", "故障感知", "检索增强生成", "参数高效微调"]
innovations: ["结构化检索：解耦静态道路与动态智能体的双分支Retrieve Model，替代传统VL特征检索", "故障感知记忆：基于PDM oracle自动识别失败案例并写入隐式记忆池，推理时按需触发纠偏", "解耦LoRA TTT：Map/Agent分支分别微调并路径融合评分，以26.44ms延迟实现情景特异性适配"]
benchmarks: ["NAVSIMv1 (Navtest)", "NAVSIMv2 (Navhard)"]
---

# 论文速读：DriveVLA-M0: Failure-Aware Memory Augmentation for Autonomous Driving

## 一句话总结
本文提出 DriveVLA-M0，一种面向自动驾驶的**故障感知记忆增强 VLA 框架**：通过构建隐式记忆池存储历史失败案例的结构化特征，并在线推理时利用解耦 LoRA 的测试时训练（TTT）机制注入知识，实现针对高风险场景的精准纠错，在 NAVSIMv1/v2 上均达到 SOTA。

---

## 研究问题与动机

1. **现有 VLA 模型在相似场景中反复犯错**：当前端到端 VLA 驾驶模型缺乏与历史失败经验关联的机制，导致对分布外或长尾场景的鲁棒性不足。
2. **基于视觉-语言特征的检索难以捕捉驾驶场景的结构特性**：已有记忆增强方法（如 MemoryVLA、EchoVLA）将中间 VL 特征直接作为检索键，但自动驾驶规划高度依赖**静态道路拓扑**和**动态交互智能体**两类结构信息，VL 空间检索易召回语义相似但结构不匹配的样本。
3. **离线重训练成本高昂且存在分布偏移**：使用失败样本进行后训练（offline fine-tuning）受限于固定数据混合，无法按场景进行针对性适配。
4. **人类认知启发的可迁移性**：人类通过关联当前情境与历史错误来预测失败概率并主动调整行为，这一机制尚未被充分引入 VLA 驾驶系统。

---

## 核心贡献（创新点）

1. **提出故障感知的隐式记忆增强框架**：显式构建记录失败案例的潜在记忆池，使模型能在推理阶段主动关联历史错误并触发纠正，与以往依靠静态参数或纯语言嵌入的方法形成本质区别。
2. **设计结构化检索机制**：训练专用 Retrieve Model，解耦静态道路结构与动态智能体交互，分别提取 map 和 agent 嵌入作为检索键，解决了现有方法仅依赖 VL 特征导致结构失配的问题。
3. **提出解耦 LoRA 测试时训练（TTT）注入策略**：将检索到的历史案例分别用于微调 Map（静态）和 Agent（动态）LoRA 分支，仅在触发条件下激活，实现零额外训练成本的情景特异性适配。
4. **验证了记忆扩展带来的免训练性能增益**：通过 Sim-Scale 合成数据将记忆池扩展至 10K 案例，未修改底座模型即可持续提升性能，证明框架具备良好的可扩展性。

---

## 方法详解

### 整体架构
DriveVLA-M0 分为两个阶段：**[M] 离线记忆生成（Memory Generation）** 和 **[I] 在线 TTT 推理（Inference with TTT）**，如图 2 所示。

### Base Model（底座模型）
- **VLM Backbone**：采用 InternVL3，输入前视图像 $I$ 和系统提示 $z$，提取 LLM 最后一层特征 $h^{-1} \in \mathbb{R}^{2800 \times 1536}$。
- **压缩模块（Compress Module）**：借鉴 Q-Former，使用可学习查询 $Q_{\text{cmp}} \in \mathbb{R}^{N \times D}$（$N{=}16, D{=}256$）经 cross-attention 压缩为紧凑的 $F_{\text{lang}} \in \mathbb{R}^{N \times D}$，压缩比达 1050×。
- **Action Decoder** 分两阶段：
  - **Trajectory Head**：将 ego 状态压缩为 $F_{\text{ego}}$，与 $F_{\text{lang}}$ 联合解码生成轨迹簇 $\hat{\mathbb{T}} \in \mathbb{R}^{M \times 8 \times 3}$。
  - **Score Head**：对轨迹簇重新编码并融合语言特征，输出 $K$ 个安全评分子项 $\hat{\mathbb{S}} \in \mathbb{R}^{M \times K}$，选择聚合得分最高的轨迹 $\hat{\tau}$ 作为最终输出。
- 训练损失：$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{traj}} + \mathcal{L}_{\text{score}}$，其中 $\mathcal{L}_{\text{traj}} = \min_i \|\hat{\tau}_i - \tau^*\|_1$，$\mathcal{L}_{\text{score}}$ 为基于 PDM oracle 得分的二元交叉熵。

### Retrieve Model（检索模型）
- 基于轻量级预训练 DINOv2 + LoRA 特征提取器，采用**解耦 LoRA 分支**并行提取：
  - **Map 分支**：捕获静态道路结构（道路边界、车道线等）→ 输出 $F_{\text{map}}$
  - **Agent 分支**：捕获动态智能体上下文（前方车辆等）→ 输出 $F_{\text{agent}}$
- 每个分支经 Transformer decoder + 可学习 query 聚合后，通过独立 head 映射为占用网格输出 $\hat{M}_{\text{map}}$、$\hat{M}_{\text{agent}}$。
- 训练损失：$\mathcal{L}_{\text{Retrieve}} = \text{BCE}(\hat{M}_{\text{map}}, M_{\text{map}}) + \alpha \cdot \text{BCE}(\hat{M}_{\text{agent}}, M_{\text{agent}})$，其中 $\alpha = 10$。

### Memory Generation（离线记忆构建）
- 用 PDM oracle scorer 对训练/外部数据中的基座模型预测轨迹 $\hat{\tau}$ 打分，得到聚合质量分数 $Q(\hat{\tau}) \in [0,1]$。
- 当 $Q(\hat{\tau}) < \beta$（$\beta = 0.5$）时，判定为失败案例，将其三元组 $(k_i, x_i, y_i)$ 写入隐式记忆池 $\mathbb{M}$：
  - $k_i = (F_{\text{map}}^{(i)}, F_{\text{agent}}^{(i)})$：检索键
  - $x_i = (F_{\text{lang}}^{(i)}, F_{\text{ego}}^{(i)}, \hat{\mathbb{T}}^{(i)})$：适配输入
  - $y_i = (\tau^{(i)}, \mathbb{S}^{(i)})$：监督标签
- 引入**余弦相似度去重机制**防止冗余条目，控制记忆池规模（Base 约 4K，Scale 约 10K）。

### Inference with TTT（在线推理与测试时训练）
1. **检索**：当前场景经 Retrieve Model 得到 $F_{\text{map}}^q, F_{\text{agent}}^q$，分别从 $\mathbb{M}$ 中检索 top-$k$（$k_1{=}9, k_2{=}3$，分层检索策略）相似案例。
2. **Trigger 门控**：若最高余弦相似度 $g = \mathbb{I}[\text{cos\_sim}(F, F^*) > \lambda]$（$\lambda = 0.9$）为 0，则跳过 TTT 直接输出基座轨迹；否则激活。
3. **解耦 LoRA TTT 微调**：
   - 每次推理时重新初始化 Map/Agent LoRA 权重
   - Map 检索案例 → 微调静态 LoRA 分支（优化 DAC、EP 等道路理解相关指标）
   - Agent 检索案例 → 微调动态 LoRA 分支（优化 NC、TTC 等动态避障相关指标）
   - 两个分支独立生成评分后按语义路径融合（静态分支负责道路合规分，动态分支负责碰撞相关分）
4. 最终通过 PDMS 风格聚合得分选择最优轨迹 $\hat{\tau}^*$ 输出。
5. TTT 超参：AdamW，学习率 $2 \times 10^{-4}$，3 步优化。

---

## 实验与结果

### 数据集与基准
- **NAVSIMv1（Navtest）**：非反应式开放环仿真基准，主要指标 PDMS（含 NC、DAC、EP、TTC、C 五个子项）。
- **NAVSIMv2（Navhard）**：新增伪闭环两阶段评估，指标扩展为 EPDMS（增加 DDC、TLC、LK、HC、EC）。

### 主要结果

**NAVSIMv1（Table 1）**：
| 方法 | PDMS↑ |
|---|---|
| DriveVLA-M0-Base | **92.3** |
| DriveVLA-M0-Scale | **94.1**（SOTA） |
| 次强基线 Centaur | 93.5 |
| 次强基线 DriveSuprim | 92.6 |
| Base Model（无记忆） | 91.0 |

DriveVLA-M0-Scale 在 NC（99.1）、DAC（98.1）、EP（90.2）、TTC（98.5）上全面领先，PDMS 较基座提升 **+3.1**。

**NAVSIMv2（Table 2）**：
| 方法 | EPDMS↑ |
|---|---|
| DriveVLA-M0-Base | **47.0**（SOTA） |
| 次强 GTRS-Dense | 45.3 |
| 次强 DriveSuprim | 44.7 |

在 Stage 1 和 Stage 2 均取得最高分，NC 和 DAC 等安全关键指标显著优于端到端方法。

**记忆扩展验证**：从 4K 扩至 10K 条目（使用 Sim-Scale 合成数据），**无需重新训练底座模型**，PDMS 从 92.3 提升至 94.1（+1.8）。

### 效率分析（Table 6，单卡 NVIDIA H20）
| 组件 | 耗时（ms） |
|---|---|
| Retrieve 前向 | 15.19 |
| 完整前向 | 30.79 |
| TTT 反向（Decoupled LoRA） | **26.44** |
| TTT 反向（Full Action Decoder） | 55.42 |

仅 26.44 ms 的 TTT 反向延迟，显著低于全量微调（约节省 52%）。

### 消融实验关键结论
- **检索策略（Table 3）**：Lang 检索（90.7）< Base（91.0）< Map-only（91.7）< Map+Agent（92.3），证实结构检索优于语言检索。
- **注入策略（Table 4）**：离线训练（91.2）< TTT Full（92.4）≈ TTT LoRA（92.3），验证在线场景特异性适配的有效性。
- **Trigger 阈值（Table 5）**：$\lambda = 0.9$ 最优（91.7），过低引入噪声，过高导致覆盖不足。

---

## 相关工作脉络

1. **End-to-End 自动驾驶基线**：TransFuser（TPAMI'23）、DiffusionDrive（CVPR'25）、VADv2（ICLR'26）、GoalFlow（CVPR'25）、Mimir（RAL'25）等端到端轨迹生成/评分方法，本文在其基础上引入记忆增强，在 NAVSIMv1/v2 上超越多数此类方法。
2. **VLA 驾驶模型**：DriveVLM（CVPR'24）、ReCogDrive（CVPR'26）、AutoVLA（NeurIPS'25）、DriveVLA-W0（ICLR'26）、MTDrive（arXiv'26）等，本文定位为在 VLA 范式内首次显式结合故障感知记忆与测试时适配。
3. **记忆增强代理/机器人**：MemoryVLA（ICLR'26）、EchoVLA（arXiv'25）、MemGen（ICLR'26）、JARVIS-1（NeurIPS'23）等，此类方法在具身智能中验证了记忆的价值，但主要依赖 VL 空间检索；本文指出其在驾驶场景中结构性失配的问题，并提出解耦方案。
4. **测试时训练（TTT）**：TENT（ICLR'21）、TTT++（NeurIPS'21）、LoRA-TTT（ICML'25）、Centaur（arXiv'25）等，Centaur 同样在自动驾驶中应用 TTT 但通过熵最小化而非记忆检索；本文强调通过显式外部记忆提供更有指导性的监督信号。
5. **轨迹预测中的记忆方法**：MANTRA（CVPR'20）、MemoNet（CVPR'22）等在运动预测中检索原型轨迹；本文将其思想扩展至 VLA 端到端驾驶规划，并引入故障感知筛选机制。
6. **最近失败驱动方法**：ELF-VLA（CVPR'26）、SoAD（TSMC'26）、BeyondDrive（ECCV'26）等方法也利用失败案例但依赖大量后训练；本文通过离线记忆+在线TTT实现了免后训练的轻量适配。

---

## 局限性与未来方向

1. **基座模型能力依赖**：记忆检索和 TTT 校正的效果取决于底座模型的基础规划能力，若基座模型在某一类场景完全无有效预测，记忆池中可能缺乏高质量检索样本。
2. **检索质量受限于双分支表征**：Map/Agent 解耦设计虽缓解了结构失配，但在复杂多智能体交互场景（如多车交汇、人车混行路口）中，静态与动态特征的耦合关系难以完全分离。
3. **Trigger 阈值需人工调参**：当前 $\lambda = 0.9$ 为固定超参，在不同驾驶域（城市/高速/极端天气）间可能需要重新校准，缺乏自适应触发机制。
4. **记忆容量上限**：虽通过去重控制了规模，但当场景复杂度剧增时，4K–10K 规模的记忆池可能不足以覆盖所有长尾故障模式。
5. **未验证真实闭环部署**：实验均在 NAVSIM 的开放环/伪闭环设定下进行，真实物理闭环中的延迟和传感器噪声对 TTT 有效性的影响尚待验证。

---

## 研究启发与可借鉴点

1. **结构化检索替代 VL 特征检索**：对于自动驾驶等强结构依赖任务，直接复用 VL 中间特征做检索容易失效；可借鉴其"任务专属检索编码器"的设计思路——为检索任务单独训练一个轻量编码器，监督信号来自任务关键结构属性（如占用网格、拓扑图），而非通用语义。
2. **解耦 TTT 路径融合**：将不同性质的知识（静态 vs 动态）分别注入模型的对应模块，并按语义路径聚合评分，避免跨类型知识干扰，该设计可迁移至多模态机器人操控等具有异构输入的任务。
3. **记忆扩展驱动免训练增益**：通过外部合成数据持续扩充记忆池而不改变基座，可作为模型 scaling law 研究的新视角——探索"记忆规模 vs 性能"的增益曲线，为低成本持续学习提供范式。
4. **故障感知筛选机制**：利用 oracle 评分（如 PDM）自动识别失败案例并写入记忆，这一"自动故障挖掘→记忆存储→在线利用"的闭环设计可推广至其他需要长尾场景覆盖的决策系统。
5. **分层检索策略**：先按粗粒度（地图结构）筛选候选集再按细粒度（动态智能体）精排，可有效过滤空场景噪声，这一策略对任何涉及稀疏动态目标的记忆检索系统均有参考价值。

---

## 关键术语表

**VLA（Vision-Language-Action Model）**：将视觉、语言和动作建模统一的端到端自动驾驶框架，通过大视觉语言模型提供场景语义推理并输出规划轨迹。

**PDM（Parting with Misconceptions Dynamics Model）/ PDMS**：NAVSIM 基准的评分系统，综合碰撞责任（NC）、道路合规（DAC）、向前进展（EP）、碰撞时间（TTC）和舒适性（C）等多个子指标计算驾驶质量得分。

**TTT（Test-Time Training）**：在推理时对模型参数进行少量梯度更新以适配当前测试样本的分布，而非依赖离线重训练。

**LoRA（Low-Rank Adaptation）**：冻结预训练模型权重，仅注入低秩增量矩阵进行微调的参数高效适配技术，本文将其扩展至测试时场景特异性更新。

**Retrieval Model**：本文专门训练的轻量检索编码器（基于 DINOv2+LoRA），解耦为 Map 和 Agent 两个分支，输出结构化场景嵌入用于记忆池查询。

**隐式记忆池（Latent Memory Pool）**：以向量嵌入形式存储历史失败案例的结构化记忆库，每个条目包含检索键、适配输入和监督标签三元组。

**Trigger Mechanism**：基于余弦相似度阈值的门控开关，仅当当前场景与记忆中某案例的结构相似度超过阈值时才激活 TTT，避免无关检索引入噪声。

**EPDMS（Extended PDM Score）**：NAVSIMv2 的扩展评分，在 PDMS 基础上增加 DDC、TLC、LK、HC、EC 等子指标，并在两阶段伪闭环设置下评估。

---

## 可复现要素

- **数据集**：NAVSIMv1（nuPlan + OpenScene）、NAVSIMv2，公开可用。
- **代码**：已开源，GitHub: https://github.com/ZebinX/DriveVLA-M0
- **底座模型**：InternVL3（开源），预训练使用 RecogDrive 整理的约 775K QA 对。
- **训练设备**：16 × NVIDIA H20 GPU（底座训练）；单张 H20（推理评估）。
- **关键超参**：
  - 底座学习率：$1 \times 10^{-4}$，AdamW
  - Retrieve Model $\alpha$（agent/map 权重比）：10
  - 故障阈值 $\beta$：0.5
  - Trigger 阈值 $\lambda$：0.9
  - TTT 学习率：$2 \times 10^{-4}$，AdamW，3 步
  - 记忆规模：Base 4K，Scale 10K
  - 检索数量：$k_1 = 9$（map），$k_2 = 3$（agent）
- **开源情况**：代码已公开，论文未明确声明权重是否开源。

---
