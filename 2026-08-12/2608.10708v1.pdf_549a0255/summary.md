---
title: "Self-Geometry: GT-Free and Plug-and-Play Test-Time Adaptation for Geometrically Consistent 3D Vision Foundation Models"
source: https://arxiv.org/pdf/2608.10708v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:53:28"
field: "3D 视觉重建与测试时适配"
keywords: ["test-time adaptation", "3D vision foundation models", "multi-view geometry", "epipolar geometry", "LoRA", "pseudo ground truth"]
innovations: ["提出 Self-Geometry 管线，以 2D 像素对应点为伪 GT 对预训练 VFM 施加显式多视图几何约束", "设计 GDO，联合 MVC Loss 与 EC Loss 并通过梯度解耦消除位姿监督冲突", "引入 FAN 视角采样与 LoRA 轻量化适配器，实现免 GT、免教师、即插即用的测试时适配"]
benchmarks: ["7Scenes", "ETH3D", "ScanNet++", "HiRoom"]
---

# 论文速读：Self-Geometry: GT-Free and Plug-and-Play Test-Time Adaptation for Geometrically Consistent 3D Vision Foundation Models

## 一句话总结
本文提出 **Self-Geometry**，一种无需地面真值（GT）、即插即用的测试时适配（TTA）管线，通过将外部特征匹配器提取的 2D 像素对应点作为伪 GT，对预训练 3D 视觉基础模型（VFMs）施加显式多视图几何约束，在不增加显著计算开销的前提下同步提升相机位姿估计与几何重建质量。

## 研究问题与动机
- **多视图几何不一致性**：主流 VFMs（如 VGGT、π³、DA3）在单次前向传播中预测深度、位姿和点图，但预训练阶段未显式 enforcing 多视图几何一致性（如束调整），导致输出存在内在不一致。
- **隐式自一致性方法的局限**：已有 TTA 方法（如 Free-Geometry）依赖模型输出间（点图/特征）的隐式自一致性监督，仅间接改善模型，在预训练 VFM 预测误差较大时提升有限（Fig. 2）。
- **现有基线方法的系统性缺陷**：TCO 依赖 GT 先验（位姿/内参/深度），Free-Geometry 依赖教师蒸馏，TTT3D/Test3R 等方法与特定架构绑定；本文在四维度（GT-free、Teacher-free、Plug-and-play、Explicit-Geometry）上全面超越（Tab. I）。
- **位姿-深度歧义性挑战**：重投影误差同时依赖位姿与深度，单一 MVC Loss 无法唯一确定准确位姿或深度，需额外引入深度无关的约束来消解歧义。

## 核心贡献（创新点）
1. **提出 Self-Geometry 管线**：首次将外部特征匹配器提取的 2D 像素对应点作为伪 GT，对预训练 VFM 施加显式多视图几何监督，区别于 Free-Geometry 依赖隐式特征一致性、TCO 依赖 GT 先验的两条路径。
2. **几何解耦优化（GDO）**：设计点-点 MVC Loss（联合监督位姿+深度）与点-线 EC Loss（深度无关，仅监督位姿），并通过梯度解耦（Gradient Disentanglement, Eq. 4）消除两损失在相机位姿上的梯度冲突，使二者互补而非对立。
3. **FAN 视角采样策略**：引入基于 SO(3) 测地线距离的场景尺度不变视角选择方法（GRV + ANS），替代 SE(3) 全位移度量，避免逐场景启发式归一化，同时降低预训练 VFM 全局注意力的二次计算开销。
4. **Lightweight TTA（LoRA 适配器）**：仅在 QKV 注意力权重中插入低秩适配器，冻结全部预训练参数，使每场景适配在单卡 RTX PRO 6000 上不超过两分钟（最多 40 输入视角，DA3-Giant），参数量仅增加 0.7%–3.4%。

## 方法详解
- **场景初始化**：在测试时一次性对每对输入视角运行 LightGlue [16] 提取 2D 像素对应点集 $\mathcal{M}$（Eq. 1），后续所有 TTA 迭代复用该集合。
- **伪对应点过滤**：每轮 TTA 迭代前，依次用 EC Loss 粗筛（剔除大位姿误差的误匹配）和 MVC Loss 细筛，联合提升对应点精度（Tab. V）。
- **MVC Loss（多视图一致性损失，Eq. 2）**：点-点约束，将源视角 $j$ 中像素经预测位姿 $\mathcal{P}$ 和深度 $\mathbf{D}_j$ 重投影到目标视角 $i$，惩罚与配对像素 $\mathbf{x}_i^m$ 的欧氏距离，联合监督位姿与深度。
- **EC Loss（对极一致性损失，Eq. 3）**：点-线约束，利用预测位姿导出的基础矩阵 $\mathbf{F}_{ij}$ 计算 Sampson 距离，仅监督相机位姿、与深度无关，有效消解 MVC Loss 的位姿-深度歧义。
- **梯度解耦（GD, Eq. 4）**：每轮迭代将 $\nabla \mathcal{L}_{\mathrm{mvc}}$ 正交投影至 $\nabla \mathcal{L}_{\mathrm{ec}}$ 的正交补空间，消除 42.4% 迭代中出现的梯度冲突（Fig. 5(b)），使 EC Loss 专攻位姿粗调、MVC Loss 专攻位姿+深度的精细微调。
- **FAN 视角采样**：用 SO(3) 测地线距离 $\theta_{ij}$（Eq. 5，仅相对旋转角）构建场景尺度不变的视角 bin；GRV（Eq. 6）选择使 active bin 熵最大的视角为目标视角，ANS 从各 bin 中均匀采样源视角，保障场景覆盖的几何丰富性。
- **Lightweight TTA**：每场景用 AdamW 优化 50 轮，LoRA rank=64、alpha=64、dropout=0.1，学习率 $5\times10^{-5}$，余弦调度 + 5% warmup；总损失包含 5 项：$\mathcal{L}_{\mathrm{mvc}}$、$\mathcal{L}_{\mathrm{ec}}$、光度一致性 $\mathcal{L}_{\mathrm{pc}}$、边缘感知深度平滑 $\mathcal{L}_{\mathrm{eds}}$、基准深度一致性 $\mathcal{L}_{\mathrm{bdc}}$；损失权重通过 Dynamic Weight Averaging（DWA）动态平衡；各损失经 Huber 鲁棒化；以 $\sqrt{\mathcal{L}_{\mathrm{ec}} \cdot \mathcal{L}_{\mathrm{mvc}}}$ 选最优 checkpoint。

## 实验与结果
- **数据集与模型**：4 个基准（7Scenes、ETH3D、ScanNet++、HiRoom）× 6 个预训练 VFM（VGGT、π³、DA3-G/L/B/S）。
- **主要结果——位姿估计（Tab. II）**：Self-Geometry 在 ETH3D 上将 VGGT 的 AUC@3 提升 **+37.3%**、AUC@30 提升 **+9.2%**；对 π³ 提升 AUC@3 **+8.3%**（TCO 在同一设置下崩溃至 -91.5%）。
- **主要结果——几何估计（Tab. III）**：在 HiRoom（高难度合成室内场景）上，DA3-Base 无位姿 F1 提升 **+85.2%**，DA3-Small 提升 **+70.4%**，显著提升欠拟合小模型的几何重建质量。
- **鲁棒性**：在所有 6 个 VFM × 4 个数据集的组合中均一致正向提升；相比 Free-Geometry（隐式监督）和 TCO（需 GT 先验），在多数 Mean 列取得最大增益。
- **少量退化说明**：VGGT+Ours 在 ScanNet++/HiRoom 部分列出现绝对值 6 pp 以内的小幅下降，Mean 仍为正（AUC@3 +3.3%，AUC@30 +1.4%）。
- **推理速度（Tab. VII）**：DA3-Giant 最重场景（40 视角，ETH3D）总适配时间 **1.65 分钟**；DA3-Small 仅需 0.49 分钟；LoRA 额外参数 0.7%–3.4%。

## 相关工作脉络
- **Free-Geometry [12]**：通过 LoRA 将全视角教师蒸馏至遮挡视角学生以强化跨视角特征一致性（隐式监督）；本文以显式几何约束取代隐式特征蒸馏，不依赖教师模型。
- **TCO [36]**：需外部 GT 先验（位姿/内参/深度）辅助适配；本文在无 GT 条件下仅用 2D 像素对应点完成相同任务，实现 GT-free。
- **Fin3R [30]**：在固定数据集上对 VGGT/MASt3R 等编码器做 LoRA 微调；本文仅需目标场景自身伪对应点，无需额外训练数据，且可适配任意新场景。
- **TTT3R / Test3R / Online3R / LoRA3D**：与特定架构（DUSt3R/MASt3R/CUT3R 等）绑定，迁移至其他 VFM 需重新设计；本文方法为纯 plug-and-play，对 6 种不同架构均有效。
- **SelfEvo [32]**：依赖自蒸馏（隐式监督）；本文用外部特征匹配器提供显式几何信号，突破教师-学生上界限制。
- **VGGT [1] / π³ [2] / DA3 [3]**：本文评测的三个 feed-forward VFM 代表当前 SOTA 零样本能力，均在预训练中未 enforcing 显式多视图几何一致性，正是本文改进对象。

## 局限性与未来方向
- **对外部特征匹配器的依赖**：在重复纹理、无纹理表面或大基线视差（重叠有限）场景下，LightGlue 匹配质量下降，导致几何监督信号减弱（S.VI 节自述）。
- **适配延迟未达实时**：尽管单场景 ≤2 分钟已较实用，仍远不能满足实时 SLAM/VR 等应用需求。
- **未来方向**：探索端到端内嵌特征匹配以消除外部依赖；结合神经辐射场（NeRF）/3D Gaussian Splatting 的后处理进一步细化重建；将适配过程加速至亚分钟级以支持在线部署。

## 研究启发与可借鉴点
- **"伪 GT + 显式几何约束"的设计范式**：将测试时提取的 2D 像素对应点直接作为监督信号，可迁移至其他 3D 视觉任务（如 SLAM、动态场景重建），为无监督/自监督测试时适配提供了新思路。
- **梯度解耦（Gradient Disentanglement）在多损失 TTA 中的通用价值**：当多个监督损失共享部分参数且可能产生梯度冲突时，正交投影方法是轻量有效的解耦手段，可推广至多任务 TTA 或领域自适应场景。
- **SO(3) 测地线视角采样的场景尺度不变性**：摒弃 SE(3) 平移分量后仅依赖旋转角，实现了免归一化的均匀场景覆盖策略，对任意尺度 indoor/outdoor 场景通用，值得在相机规划、主动感知任务中复用。
- **LoRA 适配器插入位置的通用设计**：仅修改 QKV 注意力权重、冻结全部原模型参数，以 <3.4% 额外参数实现显著性能提升，为其他 Foundation Model 的轻量化适配提供了参考配置。
- **Huber 鲁棒化 + DWA 动态加权 + 梯度裁剪的复合稳定性设计**：三层防护机制（Huber 抗异常值、DWA 防尺度失衡、clip 防梯度爆炸）使 TTA 在 diverse VFMs 上保持稳定收敛，可作为其他 TTA 系统的默认配置模板。

## 关键术语表
- **Vision Foundation Model (VFM)**：在大规模数据上预训练的端到端前馈模型（如 VGGT、DA3），可单次前向传播预测深度、相机位姿和点图，具备强零样本泛化能力。
- **Test-Time Adaptation (TTA)**：在测试阶段仅利用目标场景自身数据对预训练模型进行轻量适配，无需额外训练数据，以改善模型在特定分布上的性能。
- **Multi-View Consistency (MVC) Loss**：点-点重投影约束损失，惩罚源视角像素经预测位姿和深度重投影后与目标视角配对像素的偏差，联合监督位姿和深度。
- **Epipolar Consistency (EC) Loss**：点-线对极约束损失，利用 Sampson 距离衡量像素对与基础矩阵的对极几何一致性，仅监督位姿、深度无关，用于消解位姿-深度歧义。
- **Gradient Disentanglement (GD)**：将 MVC Loss 梯度正交投影到 EC Loss 梯度的补空间，消除两者在共享参数（相机位姿）上的梯度冲突，实现互补监督。
- **Frame Angular-Neighbor (FAN)**：基于 SO(3) 测地线距离的视角采样策略，通过几何丰富视角选择（GRV）和角度邻居采样（ANS）确保均匀的场景覆盖。
- **LightGlue**：轻量级局部特征匹配器，用于在测试时提取跨视角的 2D 像素对应点对合作为伪 GT 监督信号。
- **LoRA (Low-Rank Adaptation)**：低秩适配技术，通过在预训练模型权重中注入低秩分解矩阵，仅更新少量参数实现高效微调/测试时适配。

## 可复现要素
- **数据集**：7Scenes、ETH3D、ScanNet++、HiRoom（均为公开数据集）。
- **代码**：论文声明代码将在 https://github.com/CMLab-Korea/Self-Geometry 开源（尚未发布）。
- **权重**：使用官方预训练 VGGT、π³、DA3-G/L/B/S，未提供额外微调权重。
- **关键超参**：TTA 迭代 50 轮；LoRA rank=64、alpha=64、dropout=0.1；学习率 $5\times10^{-5}$，余弦调度 + 5% warmup；梯度裁剪 5.0；SO(3) bin 宽度 15°（9 bins）；DWA 温度 T=1.0；所有超参跨模型/数据集统一不变。
- **硬件**：单卡 NVIDIA RTX PRO 6000。
