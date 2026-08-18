---
title: "DON-T-DROP-THE-BATON-LONG-HORIZON-ROBOT-MANIPULATION-VIA-AGE"
source: https://arxiv.org/pdf/2608.16889v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:15:47"
field: "长 horizon 机器人操作"
keywords: ["long-horizon robot manipulation", "vision-language-action", "coding agent", "transition-aware memory", "frozen VLA", "RoboMemArena"]
innovations: ["以子任务为单位的分层探索，将探索成本从乘法(T^K)降至加法(T*K)", "三种转换感知记忆：调用/交接/前瞻转换的语言契约化表示与执行时检查"]
benchmarks: ["RoboMemArena"]
---

# 论文速读：DON-T-DROP-THE-BATON-LONG-HORIZON-ROBOT-MANIPULATION-VIA-AGE

## 一句话总结
BATON 提出了一种基于编码智能体的长 horizon 机器人操作框架，通过将子任务作为探索单位（探索成本从乘法降至加法）并引入三种转换感知记忆（调用/交接/前瞻转换），在冻结 VLA 上实现了端到端千步级家务任务的成功执行，无需任何参数更新。

## 研究问题与动机
1. **链式组合失败**：VLA/WAM 在单技能（开抽屉、拿罐子）上表现良好，但将其链式组合成千步任务时，中间阶段成功率约 50%，完整任务仅约 10%。
2. **误差累积无法修正**：端到端 VLA 过拟合演示数据的初始状态分布，链式组合时每个子任务从上一阶段的残差状态开始，协变量偏移累积，且成功-only 模仿未提供漂移检测信号。
3. **探索成本指数爆炸**：对 K 个子任务的整任务探索平均需要 $T^K$ 次尝试，且失败 episode 无法定位具体失败阶段。
4. **转换缺失**：VLA ACT 仅带退出条件（stop predicate）而无进入条件；子任务间状态耦合导致"局部成功但下游无法继承"的问题（如抽屉开启幅度足够但不足以下一个子任务的手伸入）。

## 核心贡献（创新点）
1. **首个冻结 VLA 上的长 horizon 编码智能体**：BATON 是第一个通过测试时探索在冻结 VLA 上端到端完成千步级操作任务的编码智能体，全程零参数更新。
2. **分层子任务探索（解决探索成本）**：将子任务作为探索单位，探索成本从 $T^K$ 降为 $T \cdot K$（加法而非乘法），且失败可精确定位到具体阶段。
3. **转换感知记忆（解决转换耦合）**：将三种转换（调用/交接/前瞻）显式建模为语言契约并存储在记忆中，使每个边界成为可检查、可修复的一等公民。

## 方法详解
**整体架构**：一个 LLM 编码智能体控制冻结的 $\pi_{0.5}$ VLA，使用解析基元（运动规划、夹爪控制）处理无接触段，仅在接触密集段调用 VLA ACT（封装为可重试工具，带任务条件 prompt 和停止谓词）。

**子任务分解与分层探索（解决❶）**：
- 给定长 horizon 指令，首先将其分解为有序子任务链：$\mathcal{T} = u_1 \triangleright \sigma_{1,2} \triangleright u_2 \triangleright \dots \triangleright \sigma_{K-1,K} \triangleright u_K$
- 每个未解决子任务通过 bootstrapping 循环独立探索（参考 Harness VLA [40]），但受腕部摄像头验证器约束
- 层次化组合：先组合相邻已解决子任务，逐步向外扩展；复合运行不引入新技能，仅验证边界

**转换感知记忆（解决❷）**，包含三种转换：

1. **调用转换（Invocation Transition）**：
   - 控制从智能体到 VLA 的边界，公式化为：$u_k = a_k \triangleright \sigma_k^{\text{act}} \triangleright v_k$
   - 引入 verifier agent：通过腕部摄像头视图确认场景就绪后才触发 VLA ACT
   - 条件示例："打开微波炉门"时两夹指须在把手两侧可见；放入篮子时夹爪两侧须在容器四边内可见
   - 存储为符号化感知查询而非坐标

2. **交接转换（Handoff Transition）**：
   - 解决前驱残差干扰后继进入状态的问题
   - 边界记录进入条件（entry condition）：后继子任务干净启动所需的腕部视图状态
   - verifier 在边界处用解析基元恢复状态（重定向夹爪、重新打开等）
   - 条件存储在边上（而非节点），因为是成对事实

3. **前瞻转换（Lookahead Transition）**：
   - 解决"本地成功但形态不可被后继继承"的问题
   - 边界记录执行约束：在当前后继下，当前子任务必须以何种形态结束
   - scheduler agent 在执行 $u_k$ 前读取剩余计划 $\mathcal{P}_{>k}$，收集约束并检索满足条件的竞争策略
   - 示例：拿番茄罐——紧跟微波炉放置时用侧面抓握；紧跟长距离运输倒出时用顶部深抓握

**记忆结构**：
- 节点（子任务）：存储失败条目和竞争粗轨迹，每条含调用转换的就绪条件
- 边（边界）：存储交接转换的进入条件和前瞻转换的执行约束，形成"交接契约" $\mathcal{C}_{k,k+1}$
- 竞争策略按需生成：首次成功单独入库，新约束排除 incumbent 时才重新探索并添加竞争者
- 契约写入失败后将违规策略标记并关联到转换，使经验可跨任务迁移

## 实验与结果
**数据集**：RoboMemArena [18]，26 个家务任务，平均超 1000 步、3-9 个验证阶段，分四类：transfering（4）、occlusion（11）、counting（7）、sequence（4）。68.9% 子任务依赖记忆。

**评估协议**：探索在 seed 50 上进行，评估在 seeds 51/52 上进行，报告两者平均。指标：TSR（任务成功率）和 CSR（累积成功率）。

**主要结果**：
| 方法 | 平均 TSR | 平均 CSR |
|------|----------|----------|
| FrameSamp+Modul（SoTA 报告） | 46.1% | 63.9% |
| Harness VLA（整任务探索） | 26.9% | 38.5% |
| **BATON（ours）** | **57.7%** | **78.8%** |

- BATON 相比最强 reported system（FrameSamp+Modul）提升 **11.6 个百分点 TSR** 和 **14.9 个百分点 CSR**
- 超越 ground-truth oracle（46.1% TSR / 64.8% CSR），说明纯回顾性记忆不足
- occlusion 类别提升最大（50.0 vs 39.1）
- 将 73% 完成阶段转化为完成任务，为表中最高比例
- transferring 类别落后 FrameSamp+Modul，但分析表明是 seed 51/52 上 task 19 的 BDDL 定义有缺陷导致

## 相关工作脉络
1. **长 horizon 操作（VLA 方向）**：$\pi_{0.5}$ [12]、HiF-VLA [29]、MemoryVLA [29]、MemER [32]、PrediMem [18]——训练增强策略或记忆，BATON 保持 VLA 冻结，通过测试时探索和语言记忆解决组合问题。
2. **Skill chaining / 边界训练**：HANDFUL [9]、Foresight Residual RL [21]——在边界处训练，每对新技能配对需支付训练成本；BATON 将边界经验编码为语言契约，一次学习、无限复用。
3. **技能库与记忆**：Voyager [37]、MEM [35]、HiMe [13]——Keystone on instruction/context，HiMe 甚至删除冲突策略；BATON 保留竞争策略，按剩余计划检索。
4. **任务导向 grasping**：固定语义 taxonomy [24]、FoundationGrasp [34]、TAMP [11,14]——需预定义谓词/值函数；BATON 零 taxonomy、零梯度，契约在失败时自修正。
5. **冻结 VLA 的编码智能体**：Harness VLA [40]——探索整任务，成本指数增长且无法区分边界失败；ASPIRE [23]——无学习策略的纯代码探索。BATON 将转换作为探索和学习单位。
6. **VLM 规划器**：Volo [4]、Hi Robot [30]、Cortex [25]——提升规划能力但未约束转换后状态；BATON 显式建模三种转换契约。

## 局限性与未来方向
1. **Transferring 类别性能下降**：benchmark 中 transferring 类别不如 FrameSamp+Modul，部分源于 seed 51/52 上 task 19 的 BDDL 定义缺陷，但也反映视觉相似容器映射可能仍有挑战。
2. **探索效率依赖子任务分解质量**：自动分解的质量影响后续探索效率，当前依赖 LLM 一次性 drafting，未探索主动修正分解的策略。
3. **腕部摄像头验证器的通用性**：verifier 的判断逻辑基于 hand-crafted 条件（如"两夹指在把手两侧"），对 novel 场景可能失效，且依赖高质量腕部视图。
4. **计算开销**：虽无需训练，但测试时多轮 exploration 涉及大量 planner token 消耗，千步任务的探索时间可能较长。
5. **未评估 real robot**：实验仅在仿真（RoboMemArena）中进行，sim-to-real gap 未知。

## 研究启发与可借鉴点
1. **转换作为一等公民**：将子任务边界显式建模为可检查的语言契约（而非训练副产品），为长 horizon 任务提供了可调试、可复用的模块化抽象，可迁移至任何需要 skill composition 的场景。
2. **分层探索的成本优化策略**：以子任务为单位的探索将乘法成本降至加法，且失败可归因——这一思路可推广至任何其他"复合任务探索"场景（如多智能体协作、分层强化学习）。
3. **竞争策略的按需存储**：记忆不预置策略枚举，而是在约束排除 incumbent 时才 re-exploration，兼顾了简洁性与表达力；可用于构建可进化的技能库。
4. **腕部摄像头作为进入条件的天然传感器**：利用近距离腕部视角验证就绪状态，比全局 RGB 更鲁棒，这一设计模式值得在需要精确接触前准备的任务中借鉴。
5. **Zero-shot 边界修复**：交接转换通过解析基元恢复进入状态而非重新学习策略，使修复代价极低且可立即迁移——这一"修复而非重学"原则在资源受限部署中极具价值。

## 关键术语表
**VLA (Vision-Language-Action)**：直接将视觉观测和语言指令映射到低层动作的端到端策略模型，通过大规模 web-scale 预训练获得泛化能力。
**WAM (World-Action Model)**：在 motor level 之上扩展预测性世界模型的架构，可同时输出动作和世界状态预测。
**Coding Agent**：通过编写程序调用感知/控制基元来控制机器人的 LLM 智能体，从执行反馈中迭代修订，无需训练。
**Harness VLA**：将冻结 VLA 封装为可重试基元（VLA ACT），配合任务条件 prompt 和 stop predicate，使编码智能体可在接触密集段调用 VLA。
**TSR (Task Success Rate)**：所有验证阶段均满足的任务成功率。
**CSR (Cumulative Success Rate)**：完成阶段占总阶段的比率，衡量链式执行的完整性。
**Handoff Contract**：以语言起草、执行时检查、失败时修订的边界契约，包含进入条件和执行约束。
**Invoker/Verifier/Scheduler**：BATON 中三种 agent role——explorer 负责探索，verifier 负责腕部视图就绪检查，scheduler 负责按剩余计划检索竞争策略。

## 可复现要素
- **数据集**：RoboMemArena [18]，论文已公开（arXiv:2605.10921）
- **代码/权重**：论文未提及开源代码或权重的具体链接，Harness VLA [40] 为底层依赖
- **关键超参**：探索预算——整任务 Harness VLA 每任务 20 episodes；VLA 模型为冻结 $\pi_{0.5}$；LLM 后端未明确指定（参考 Harness VLA 配置）
- **评估协议**：seed 50 探索，seeds 51/52 评估，两 seed 平均报告
