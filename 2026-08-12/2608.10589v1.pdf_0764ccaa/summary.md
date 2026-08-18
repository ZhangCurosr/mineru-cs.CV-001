---
title: "π-SUB: A Physics-Informed Synthetic Underwater Benchmark Dataset for Underwater Image Enhancement"
source: https://arxiv.org/pdf/2608.10589v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:06:35"
field: "水下计算机视觉"
keywords: ["水下图像增强", "合成数据集", "物理感知模型", "Jaffe-McGlamery", "Jerlov水质类型", "超真实性", "泛化性评估"]
innovations: ["提出MUIF模型联合建模深度相关辐照度与生物分解吸收", "实现分离直接透射与后向散射衰减的完整物理参数化", "构建双轴评估协议验证超真实性和下游泛化性"]
benchmarks: ["π-SUB", "UIEB", "RUIE", "SQUID", "FishTrac", "OceanDark", "U45"]
---

# 论文速读：π-SUB: A Physics-Informed Synthetic Underwater Benchmark Dataset for Underwater Image Enhancement

## 一句话总结
本文提出 π-SUB，一个基于物理的水下图像增强合成基准数据集生成框架，通过在修改的 Jaffe–McGlamery 成像模型中引入深度相关下行辐照度、Jerlov 固有光学特性（IOPs）及生物光学效应，生成物理一致的成对合成水下图像，显著缩小了合成-真实域间隙，在超真实性和泛化性两方面均优于现有合成数据集。

## 研究问题与动机
- **无干净参考图像**：监督式水下图像增强（UIE）需要配对的高糊/清晰图像，但同一场景无水中的"干净参考"在物理上无法获取，现有数据集只能用增强算法输出或 CycleGAN 合成作为参考，引入了算法偏差。
- **已有合成数据集物理建模不足**：现有合成数据集普遍存在三大缺陷——忽略深度相关下行辐照度（假设表面光照恒定）、省略叶绿素吸收/CDOM/荧光等生物光学过程、未联合建模悬浮颗粒体积散射，导致合成图像分布与真实水下图像存在显著差异。
- **合成-真实域间隙未解决**：外观迁移类方法（如 WaterGAN、Syrea）依赖参考图像的分布，无法可控外推到未见光学条件；纯物理方法（如 SUIEB、PHISWID）则简化了衰减参数，缺乏生物学分辨率。
- **评估指标局限**：真实水下数据集缺乏物理有效的 ground truth，限制了训练和评估的物理可验证性。

## 核心贡献（创新点）
1. **提出 π-SUB 物理感知合成基准框架**：基于修改的 Jaffe–McGlamery 成像模型，整合 Jerlov 十类水质、相机深度、可见度、叶绿素浓度和光学衰减的完整参数化流程，生成物理一致的配对合成水下图像。
2. **改进的 Jaffe–McGlamery 成像模型（MUIF）**：首次联合引入深度相关下行辐照度 $E_d$、单独的直接透射与后向散射衰减系数 $\beta_d \neq \beta_b$、以及由生物成分分解的吸收系数 $a = a_w + a_{chl} + a_{cdom}$，实现可解析反演的确定性退化建模。
3. **增强真实感三阶段残余现象建模**：在确定性成像模型基础上，独立叠加悬浮颗粒物质（SP）、体积雾（Haze）和生物荧光（BE）三种可控制现象，形成多子集 benchmark 结构。
4. **双轴评估协议**：提出超真实性和泛化性两个维度的系统性评估，π-SUB 的全局 FID 为 95，较次优合成数据集 Syrea（176）降低 46%；在六个人工数据集上，跨四架构平均 UIQM 提升 9.46%（vs Syrea）和 4.18%（vs PHISWID）。

## 方法详解

### 框架整体流程（三个阶段）
- **输入**：参考图像源（Unreal Engine 渲染图或近零介质效应真实水下图）、物理水配置（Jerlov 水质类型 + 相机深度）、增强真实感配置（是否叠加 SP/Haze/BE）。
- **Stage I**：使用 Depth Anything V2 估计归一化深度图 $z_n \in [0,1]$，通过水类型相关的最大可见距离 $z_{max}$ 校准为度量场景范围 $z_s = z_{max}(1-z_n)$。
- **Stage II**：基于 MUIF 模型合成基线水下图像 $I_b$。
- **Stage III**：独立叠加悬浮颗粒、体积雾和生物荧光三个残余现象，生成 $I_h$、$I_{sp}$、$I_{bp}$ 及全叠加组合 $I_c$。

### 改进水下成像模型（MUIF）
经典 Jaffe–McGlamery 模型为 $I_b = J e^{-\beta_d z} + B_\infty(1-e^{-\beta_b z})$，本文扩展如下：

**① 分离直接透射与后向散射衰减系数**：
$$\beta_d(\lambda) = a(\lambda) + \delta b(\lambda), \quad \beta_b(\lambda) = a(\lambda) + b(\lambda) = c(\lambda)$$
其中 $\delta = 0.05$ 反映前向散射对直接路径衰减的贡献有限。

**② 深度相关下行辐照度**：
$$E_d(\lambda) = E_0 e^{-K_d(\lambda) \cdot d}$$
其中 $K_d$ 由 IOPs 推导（公式 3），$\mu_d = 0.89$（大洋型）或 $0.85$（沿海型），$G$ 为几何因子。垂直深度 $d$ 与水平范围 $z_s$ 不可互换，分别控制全局照明和空间衰减。

**③ 遮蔽光（Veiling Light）深度依赖**：
$$B_\infty(\lambda) = \frac{b_b(\lambda) \cdot E_d(\lambda)}{a(\lambda) + b(\lambda)}$$
其中后向散射系数 $b_b$ 由分子散射（一半）和颗粒后向散射（$\tilde{b}_{bp} \approx 0.018$）组成。

**④ 生物分解吸收**：
$$a(\lambda) = a_w(\lambda) + a_{chl}(\lambda) + a_{cdom}(\lambda)$$
叶绿素吸收：$a_{chl}(\lambda) = A_\lambda C_{chl}^{E_\lambda}$（Bricaud 幂律）；CDOM 与叶绿素耦合：$a_{cdom}(\lambda) = a_{chl}(440) \cdot M \cdot e^{-\alpha(\lambda-440)}$。

**⑤ 最大可见距离**：
$$z_{max} = \frac{-\ln \varepsilon}{\beta_{min}}, \quad \varepsilon \approx 0.01 \text{（相机阈值）}$$

**⑥ 最终 MUIF 公式**：
$$I_b(\lambda) = J \cdot E_d(\lambda) \cdot e^{-\beta_d(\lambda) z_s} + B_\infty(\lambda) \cdot (1 - e^{-\beta_b(\lambda) z_s})$$

### 增强真实感残余现象（Stage III）
- **悬浮颗粒（SP）**：各向异性高斯粒子程序化合成，透明度受深度图遮挡项调制：$I_{sp} = I_b + \lambda \sum_{i=1}^{N} \alpha_i \mathbf{c}_i$。
- **体积雾（Haze）**：多重散射近似：$I_h = I_b(1-\rho) + \mathbf{c}_{haze} \cdot \rho$，其中 $\rho = (1-z_n)^{1/s_h}$。
- **生物荧光（BE）**：非弹性荧光加法辐射：$\mathbf{F}_{em} = \phi_C \cdot a_\phi \cdot \mathbf{f}_{em}$，$\mathbf{f}_{em} = [0.97, 0.03, 0.00]^\top$（685nm 发射带映射），量子产率 $\phi_C$ 具有非单调深度剖面（表面光抑制→亚表层峰值→深处光子饥饿）。

### 数据集规模
117 张干净参考图像 × 10 种 Jerlov 水质类型，共 14,040 对合成水下-参考图像。

## 实验与结果

### 超真实性评估（Hyper-realism）
- **基准**：5 个真实数据集合并（UIEB、RUIE、SQUID、FishTrac、OceanDark）。
- **全局 FID**：π-SUB = **95**，Syrea = 176，SUIEB = 192，PHISWID = 227，SUID = 230；π-SUB 较 Syrea **降低 46%**。
- **逐聚类 FID**：在所有 7 个感知聚类中均最低。
- **OOD 率**：π-SUB 仅 **2.00%**，远低于 Syrea（12.23%）、PHISWID（4.02%）、SUIEB（19.26%）、SUID（27.33%）。
- **PCA 投影**：π-SUB 凸包与真实数据高度重叠，是唯一下覆盖全部 7 个感知域的合成数据集。

### 泛化性评估（Generalization）
- **4 个 UIE 架构**：Pix2Pix、FUnIE-GAN、Phaseformer、PUIE-Net。
- **6 个真实测试集**：UIEB、U45、RUIE、FishTrac、OceanDark、SQUID。
- **UIQM 提升**：跨 4 架构 × 6 数据集平均，较 Syrea 提升 **9.46%**，较 PHISWID 提升 **4.18%**。
- **NIQE 降低**：较 Syrea 降低 **23.98%**，较 PHISWID 降低 **48.78%**。
- **最强结果示例**：Pix2Pix@π-SUB 在 OceanDark 上 UIQM=3.3495，NIQE=3.4896；在 SQUID 上 NIQE=3.1016。
- **特征匹配任务**：在 SQUID Katzaa 序列上，Raw=23 个匹配，PHISWID=85，**π-SUB=157**（提升 6.8× vs Raw，1.8× vs PHISWID）。

### 消融实验
| 变体 | UIQM↑ | NIQE↓ |
|------|--------|--------|
| Base（仅 MUIF） | 2.8713 | 4.7283 |
| Base+Bio | 3.1138 | 3.7961 |
| Base+Haze | 3.0941 | 3.9254 |
| Base+SP | 2.9851 | 4.3382 |
| **Full（全部叠加）** | **3.1829** | **3.4129** |

残差现象之间互补而非冗余，Full 配置在所有六个基准上持续最优。

## 相关工作脉络
1. **Jaffe–McGlamery 经典模型**（McGlamery 1980；Jaffe 1990）：单散射假设下的水下成像基础，本文在其上扩展了深度相关辐照度和生物分解吸收。
2. **Akkaynak & Treibitz (Sea-thru, CVPR 2019)**：首次明确提出分离直接透射与后向散射衰减系数（$\beta_d \neq \beta_b$），本文沿用并扩展至 Jerlov IOPs 参数化。
3. **SUIEB**（Li et al., 2020）：将 Jaffe–McGlamery 扩展到 10 种 Jerlov 水质类型，但使用全局单一衰减系数且忽略深度相关辐照度和生物效应。
4. **SUID**（Hou et al., 2020）：启发式调参的 30 种退化效应组合，非 IOP 驱动，规模小（900 张），无物理可验证性。
5. **PHISWID**（Kaneko et al., 2026）：引入逐像素场景范围和海洋雪模型，但垂直辐照度固定于表面值，IOP 参数来自均匀采样而非实测 Jerlov 数据。
6. **Syrea**（Wen et al., ICRA 2023）：物理引导的风格迁移方法，从真实图像估计光学参数而非基于测量 IOPs，缺乏可控外推机制。
7. **RSUIGM**（Desai et al., 2024）：实现分离直接/后向散射系数并纳入垂直辐照度，但未建模生物光学效应。
8. **MUSE**（Li et al., 2025）：图形引擎渲染，但场景参数从图像统计推导而非物理测量，不参与本实验对比。

## 局限性与未来方向
- **荧光模型未独立验证**：叶绿素荧光模型基于 Zhai et al. 的辐射传输理论推导，尚未与实际原位测量数据对比验证，论文明确列为未来工作。
- **人工照明深海环境未覆盖**：当前数据集主要来自自然光照条件下的浅海至深海场景，强人工照明环境（如深海探测）尚未建模。
- **高浊度沿海水域覆盖有限**：尽管包含 Jerlov 9C 等高浊度类型，但极端浑浊条件仍与真实部署存在差距。
- **焦散（Caustics）未建模**：水面光线折射产生的焦散模式在现有真实基准中本就稀缺，π-SUB 也未包含此现象。
- **参考图像语义多样性受限**：约 10% 的真实参考图来自 UIEB/LSUI 的浅水精选样本，合成场景仍以 Unreal Engine 资产库为主，珊瑚物种和底栖生态的多样性可能不足。

## 研究启发与可借鉴点
1. **深度-范围解耦建模思路**：将垂直相机深度 $d$（控制全局照明 $E_d$）与水平场景范围 $z_s$（控制空间衰减/后向散射）明确分离，这一设计对雷达、大气成像等参与介质场景具有可迁移价值。
2. **物理可解释感知特征空间**：使用 L* 亮度、色度、暗通道均值、HSV 饱和度及 a*b* 色度坐标构建特征向量进行聚类分析，相比纯深度学习特征更具物理解释性，可作为域适应评估的通用范式。
3. **生物光学分辨率作为数据增强维度**：将叶绿素浓度 $C_{chl}$ 作为连续可控变量而非离散水质类型标签，实现了沿生产力梯度的场景渲染，这一思路可扩展到其他遥感/医学成像领域。
4. **残差现象独立可控叠加策略**：将确定性物理模型与三类可独立控制的残余现象（SP/Haze/BE）分离，既保证了基础模型的物理一致性，又保留了真实世界复杂性的覆盖，可作为合成数据设计的通用模板。
5. **双轴评估协议**：将"分布相似性（FID/OOD/PCA）"与"下游任务泛化性（UIQM/NIQE/特征匹配）"结合的双轴评估框架，可有效避免仅依赖分布指标导致的评估失真，值得推广至其他合成数据 benchmark 研究。

## 关键术语表
**Jerlov 水质类型**：根据海水光学清澈度划分的 10 种标准类型（5 种大洋型 I–III 和 5 种沿海型 1C–9C），每类具有独立的固有光学特性（IOPs）参数表。

**固有光学特性（IOPs）**：仅取决于水体介质本身的光学参数，包括吸收系数 $a(\lambda)$ 和散射系数 $b(\lambda)$，与光照和观测几何无关。

**视在光学特性（AOPs）**：同时依赖于水体介质和光照/观测几何的光学参数，如漫衰减系数 $K_d$、直接透射衰减 $\beta_d$、后向散射衰减 $\beta_b$ 和遮蔽光 $B_\infty$。

**深度相关下行辐照度（Depth-dependent Downwelling Irradiance）**：描述太阳辐射随水深指数衰减的过程，$E_d = E_0 e^{-K_d \cdot d}$，是本文区别于 prior work 的关键物理建模增量。

**生物荧光（Chlorophyll Fluorescence）**：浮游植物吸收蓝光/红光后在 685nm 附近再发射光子的非弹性过程，具有非单调深度剖面（光抑制→亚表层峰值→光子饥饿），无法用衰减系数建模。

**Fréchet Inception Distance（FID）**：衡量两个图像特征分布之间差异的指标，通过 Inception-v3 网络提取特征后计算 Fréchet 距离，越低表示越相似。

**Underwater Image Quality Measure（UIQM）**：无参考水下图像质量评估指标，综合色彩fulness（UICM）、锐度（UISM）和对比度（UIConM）三个分量。

**Natural Image Quality Evaluator（NIQE）**：基于自然场景统计的无参考图像质量指标，衡量增强结果与真实自然图像统计特性的偏差。

## 可复现要素
- **数据集**：π-SUB 数据集已公开，包含 14,040 对合成水下-参考图像，含完整物理元数据。
- **代码**：开源，GitHub 地址 https://github.com/airl-iisc/pi-SUB。
- **关键超参**：深度估计使用 Depth Anything V2；$\delta = 0.05$（前向散射系数）；后向散射颗粒分数 $\tilde{b}_{bp} \approx 0.018$；相机对比度阈值 $\varepsilon \approx 0.01$；UIQM 权重 $c_1=0.0282, c_2=0.2953, c_3=3.5753$；各 Jerlov 类型具体 IOP 参数及 $C_{chl}$、$(M, \alpha)$ 参数见补充材料。
