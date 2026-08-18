---
title: "DynaPPI: A large-scale dynamic protein dataset for AI-driven advances in protein interactomics"
source: https://arxiv.org/pdf/2608.10435v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:51:41"
field: "蛋白质相互作用与结构预测"
keywords: ["protein complex", "molecular dynamics", "diffusion model", "dataset", "enhanced sampling", "protein-protein interaction", "structural biology"]
innovations: ["提出面向蛋白复合体动态再结合的大规模 MD 轨迹数据集框架", "设计减少未结合构象偏差的三阶段起始状态构建流程"]
benchmarks: ["CAPRI", "Docking Benchmark v5.0", "Dynamic PDB"]
---

# 论文速读：DynaPPI: A large-scale dynamic protein dataset for AI-driven advances in protein interactomics

## 一句话总结
本文提出 **DynaPPI** 数据集，收录从解离蛋白链到结合态的分子动力学（MD）再结合轨迹，填补了现有资源在刻画多链复合体动态结合路径方面的空白；其设计旨在支撑扩散模型学习动态结合过程并预测未知复合体结构。

## 研究问题与动机
- **现有数据集偏静态或单链**：PDB、Dynamic PDB 等主要覆盖静态快照或单实体轨迹，无法刻画多链复合体从解离到结合的动态路径、中间态和非加和性涌现行为。
- **AI驱动结构生物学缺少动态训练数据**：扩散模型虽已用于蛋白骨架生成，但在预测未知多链复合体（complexes）方面仍缺乏时间分辨的基准轨迹数据。
- **结合过程的非加和性未被充分利用**：直接取 PDB 中的结合构象提取各链会引入界面重塑偏差，导致能量势垒被低估，无法真实反映天然未结合（apo）态的动力学。
- **动态轨迹与实验验证脱节**：需要弥合计算模拟与 cryo-EM、FRET 等实验观测之间的鸿沟，以推动药物设计、合成生物学等下游应用。

## 核心贡献（创新点）
- **提出首个面向蛋白复合体动态再结合的大规模 MD 轨迹数据集框架**：涵盖蛋白-蛋白/配体/核酸及分子内重排四类相互作用，与仅关注静态或单链的现有数据集形成本质区别。
- **设计了减少未结合构象偏差的三阶段起始状态构建流程**：通过 PDBe-KB 实验数据、短期 MD 弛豫或 AlphaFold3 单体预测来近似 apo 构象，再以随机旋转/平移实现 5–20 nm 初始分离，优于直接截取自结合 PDB 的做法。
- **将增强采样技术整合至复合体再结合模拟中**：结合伞形采样、元动力学（PLUMED + OpenMM）和 T-REMD，以提高跨越能量势垒的效率，区别于常规全原子 MD 的固定时长运行。
- **提供多模态 AI 友好的数据格式与完整标注体系**：包括 DCD/NetCDF 轨迹、HDF5 物性、contact map、自由能谱及 train/val/test 划分，支撑下游扩散模型的条件生成训练。

## 方法详解
- **数据源与筛选**：从 PDB 选取分辨率 ≤ 2.5 Å 的多链条目，补充 PDBe-KB、BioLiP 和 Docking Benchmark v5.0 以提高多样性；复合体限制为 2–4 条链、每条 ≤ 300 残基、总原子数 ≤ 50,000，排除跨膜和大组装体。
- **起始状态构建（三阶段）**：
  - Step 1 链提取：按链标识符解析并输出独立 PDB，检查缺口与碰撞，废弃 > 5% 未解析残基的结构。
  - Step 2 未结合态近似：优先从 PDBe-KB/UniProt 获取同源性 > 95% 的实验 apo 结构并做 TM-align 叠加；若无，则对每条链单独进行溶剂化、能量极小化（steepest descent + conjugate gradient，容差 2.39 kcal/mol·Å）及 NVT/NPT 各 1 ns 平衡后运行 5–10 ns 生产模拟，经 tICA + k-means 选取距结合态 RMSD > 1 Å 的主成分构象；链不完整时可用 AlphaFold3 单体预测补全。
  - Step 3 几何分离：以各链质心为基准施加均匀采样欧拉角及 5–20 nm 随机平移，初始速度按 300 K Maxwell-Boltzmann 分布赋值，校验最小距离 > 0.5 nm 后导出合并 PDB。
- **MD 模拟设置**：使用 OpenMM + AMBER ff14SB 力场，TIP3P 水盒 padding ≥ 1.5 nm，150 mM NaCl 中和；NVT（300 K，Langevin integrator，摩擦系数 1.0 ps⁻¹，2 fs 步长，SHAKE 约束 H）→ NPT（1 bar，Monte Carlo Barostat）各 1 ns 平衡后进入生产阶段，单副本时长 10 ns–1 ms。
- **增强采样与并行**：伞形采样沿质心距离反应坐标加偏置势；元动力学通过 PLUMED 插件实现；扩散限速阶段采用 T-REMD（300–350 K）跨越势垒；每个复合体 5–20 个不同随机种子的副本，基于实时结合判定（> 20 个链间接触）可提前终止收敛副本。
- **数据记录与标注**：原子坐标每 10 ps 保存、物性每 1 ns 保存（文档表 1 列出 position/velocity/force/energy/temperature/pressure/contact map/distance/SASA/hbond/salt bridge 等），轨迹存 DCD 或 NetCDF，属性存 HDF5，gzip 压缩；绑定事件时间戳、副本 ID、延长状态等均作为 metadata。
- **后处理与验证**：MDAnalysis 对齐至参考结合态去除周期边界伪影；tICA + k-means（PyEMMA）识别 unbound/transient/bound 态并构建 Markov 状态模型估算转变速率；能量漂移 < 1%，结合态 RMSD < 3 Å（vs PDB），MMPBSA 亲和力与文献值相关系数 > 0.7，DSSP 二级结构损失过滤 > 20% 的不稳定副本，目标成功率 > 80%。

## 实验与结果
- **数据集规模（目标/规划）**：首期可行性验证约 100–1,000 个复合体，试点 500 个（250 二聚体 + 250 三聚体），最终目标约 10,000 个；每复合体 5–20 副本，时长 10 ns–1 ms，总计约 10⁸–10¹¹ 帧；存储量 10–100 TB。
- **类别分布**：蛋白-蛋白 40%、蛋白-配体 30%、蛋白-核酸 20%、分子内重排 10%；序列冗余度 < 30%（MMseqs2 聚类）。
- **质量指标**：结合态 RMSD < 3 Å；能量漂移 < 1%；MMPBSA 亲和力与文献值相关性 > 0.7；二级结构保留率目标 > 80%。
- **计算资源**：试点阶段每机器 GPU 时约 50–200 h（NVIDIA A100 80 GB），预算约 9,000–15,000 美元（云算力定价参考 AWS p3.2xlarge $3/h）。
- **主要结论**：本文以数据构建与基准方案为主，暂未报告针对扩散模型训练的定量 AI 性能对比；数据集计划托管于 Zenodo/Dryad 并提供 DOI，划分 80/10/10 训练集。

## 相关工作脉络
- **PDB / Dynamic PDB**：静态结构库与单链动态数据集，侧重平衡态构象或单实体轨迹，本文聚焦多链再结合的完整动态路径以弥补其在结合动力学方面的缺失。
- **AlphaFold3 / AlphaFold-Multimer**：以序列为条件预测静态复合体结构，本文提供时间分辨的 MD 轨迹，旨在让扩散模型学习动态结合过程而非仅收敛到单一最低能量构象。
- **CAPRI / Docking Benchmark v5.0**：评估对接姿态准确性的经典基准，本文将其作为外部质量验证手段之一，但目标是生成动力学轨迹而不仅是静态 pose。
- **Umbrella Sampling / Metadynamics（AMBER/OpenMM 传统流程）**：本文的创新在于将这些增强采样方法系统性地集成到复合体再结合数据生产管线中，并规模化至 AI 训练可用体量。
- **PyEMMA / tICA / MSM 分析生态**：本文复用成熟的状态划分与转变速率估计工具链，并将其与 AI-ready 数据格式对接。

## 局限性与未来方向
- **当前主要为规划与试点阶段**：尚无大规模已发布数据集及下游模型训练结果，实际可用性待验证。
- **模拟时长与真实结合时标之间存在差距**：即使使用增强采样，10 ns–1 ms 仍可能难以覆盖慢速结合事件，部分复合体可能达不到收敛。
- **无膜蛋白与大组装体被排除**：限制了其在膜受体信号传导、超大复合物组装等场景的直接适用性。
- **增强采样偏置势的选取依赖反应坐标经验**：对复杂结合路径可能引入人为偏置，影响轨迹的物理真实性。
- **存储与分发成本较高**：10–100 TB 量级对开放获取和复用构成工程挑战。

## 研究启发与可借鉴点
- **未结合构象的近似策略可迁移**：三阶段起始态构建（实验 apo 查询 → 短 MD 弛豫 → AF3 预测）可作为其他动态蛋白数据集的标准预处理范式。
- **多副本 + 增强采样的规模化生产管线**：伞形采样/元动力学/T-REMD 的组合及实时终止策略，值得推广至配体结合、核酸-蛋白复合物等更广泛的生物大分子动态数据构建中。
- **多模态 AI 友好格式设计**：轨迹与物理量分离存储（DCD/NetCDF + HDF5）、pre-computed contact map 与 free energy profile 的配套输出，为扩散模型的条件生成训练提供了可直接复用的数据工程模板。
- **与扩散模型结合的潜在创新机会**：可将 DynaPPI 的时间分辨轨迹作为空间扩散模型的条件信号（如初始分离姿态 + 环境参数），探索"动态预训练 + 结构生成"的两阶段范式。
- **实验验证闭环思路**：用 MMPBSA、Markov 状态模型、CAPRI 等多维度指标相互校核，可为后续自建数据集建立质量控制 SOP。

## 关键术语表
- **DynaPPI**：本文提出的蛋白复合体动态再结合轨迹数据集，涵盖从解离链到结合态的 MD 模拟数据。
- **Molecular Dynamics (MD)**：通过数值求解牛顿方程模拟原子/分子随时间演化的计算方法。
- **Umbrella Sampling**：沿预设反应坐标施加偏置势以加速跨越自由能垒的增强采样技术。
- **Metadynamics**：在采样过程中逐步添加历史依赖偏置势以填充自由能阱、促进构象探索的方法。
- **Temperature Replica-Exchange MD (T-REMD)**：多个不同温度副本并行模拟并周期性交换，以改善构象空间采样效率的技术。
- **tICA (Time-lagged Independent Component Analysis)**：基于时间延迟协方差的降维方法，用于识别慢速动力学自由度。
- **Markov State Model (MSM)**：将连续构象空间离散化为状态并用马尔可夫转移矩阵描述动力学的建模框架。
- **MMPBSA**：基于分子力学与 Poisson-Boltzmann 表面积方法的结合自由能估算流程。

## 可复现要素
- **数据集**：计划公开，托管于 Zenodo 或 Dryad，提供 DOI；当前为规划/试点阶段，未给出已发布链接。
- **代码**：论文未明确开源，但使用 OpenMM、PLUMED、MDAnalysis、PyEMMA、Biopython、MMseqs2、PROPKA3、PDB2PQR、MODELLER、AlphaFold3 等开源工具。
- **关键超参**：时间步长 2 fs；NVT/NPT 各 1 ns，300 K、1 bar；摩擦系数 1.0 ps⁻¹；padding ≥ 1.5 nm；NaCl 150 mM；坐标采样间隔 10 ps，物性 1 ns；初始分离距离 5–20 nm；T-REMD 温度范围 300–350 K。
- **算力**：NVIDIA A100 80 GB，MPI 并行；试点 50–200 GPU·h/机器。
