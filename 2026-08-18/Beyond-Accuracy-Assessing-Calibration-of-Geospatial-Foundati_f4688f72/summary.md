---
title: "Beyond-Accuracy-Assessing-Calibration-of-Geospatial-Foundati"
source: https://arxiv.org/pdf/2608.16614v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:54:04"
field: "遥感基础模型评测与可靠性"
keywords: ["geospatial foundation models", "calibration", "distribution shift", "uncertainty quantification", "earth observation", "benchmarking", "representation drift"]
innovations: ["首次系统评估GeoFM在分布偏移下的校准性，揭示干净数据排名无法迁移至偏移条件", "提出表征刚性假说，通过CKA关联EO预训练编码器的过置信偏移退化机制", "评估温度缩放/深度集成/SVGP在偏移下的校准极限，证明后处理无法充分恢复偏移校准"]
benchmarks: ["GEO-Bench-1", "PANGAEA", "REOBench", "EarthShift", "ADVANCE", "m-EuroSAT", "RESISC45", "So2Sat", "CloudSEN12", "DynamicEarthNet", "FLAIR#2", "Fields of the World", "SpaceNet2"]
---

# 论文速读：Beyond-Accuracy-Assessing-Calibration-of-Geospatial-Foundation-Models

## 一句话总结
本文系统评估了16个冻结的地球观测基础模型（GeoFM）编码器在物理模拟分布偏移下的校准性，发现干净数据上的准确率排名几乎无法迁移至受污染条件，且EO预训练编码器在偏移下比ImageNet预训练编码器表现出更严重的过置信（overconfidence）问题，揭示当前GeoFM评测生态中校准评估的全面缺失。

## 研究问题与动机
- **核心问题**：GeoFM当前在干净、全量数据基准测试上的准确率排名无法反映模型在实际部署条件下的可信行为，评测协议过于狭窄。
- **动机一**：Earth Observation（EO）领域的可信决策（如灾害监测、农业估产）要求模型不仅能正确预测，还需知道何时不确定——但当前所有主流评测体系（GEO-Bench、PANGAEA、REOBench、EarthShift）均只报告准确率，无一包含校准/不确定性指标。
- **动机二**：GeoFM被宣传为"单一编码器服务多传感器、多区域、多成像条件"，但其对近端协变量偏移（如云遮挡、传感器噪声、运动模糊）的校准敏感性从未被系统评估。
- **动机三**：EO预训练的"领域优势"假设仅在干净条件下成立；在分布偏移下，EO预训练是否赋予更好的校准鲁棒性尚不清楚，本文首次给出系统性回答。

## 核心贡献（创新点）
- **贡献一：首次建立GeoFM校准评测体系**——对16个冻结编码器、9个数据集、3种物理驱动的偏移类型和5个训练数据预算进行联合评估，填补了GeoFM评测中校准维度的空白。
- **贡献二：揭示排名不稳定性**——证明干净数据上的准确率和校准排名在偏移条件下系统性地丧失可迁移性（Kendall τ降至接近0），且偏移后的校准排名之间具有一定结构性（severity-3到severity-5的τ=0.72）。
- **贡献三：提出表征刚性假说**——通过CKA分析建立关联：EO预训练编码器的表征刚性更高（head层CKA≈0.82 vs. 自然图像编码器≈0.61），导致漂移-置信度解耦（ρ≈-0.07 vs. -0.17），进而产生更多"高漂移+高置信+错误"的预测（约2.4倍于对比组）。
- **贡献四：评估三种不确定性量化方法的极限**——温度缩放对偏移无改善；深度集成仅部分缓解且在严重偏移时可能适得其反；高斯过程探测仅在严重云层下减半ECE但使干净数据ECE增至3倍。

## 方法详解
- **编码器与探针设置**：16个冻结预训练编码器（7个GeoFM：Clay V1.5、Prithvi V2、DOFA、OLMo-Earth base/tiny、TerraMind、Panopticon、ViT-L DINOv3-Sat；9个通用编码器：ConvNeXt-L DINOv3、ResNet-18/50、Swin-T、ViT-B/ViT-L、MobileNetV3；1个RCF随机卷积基线）。所有编码器使用同一线性探针（ multinomial logistic regression via L-BFGS + ℓ₂正则）或FPN分割解码器，正则化强度从准确率最优选取，不在校准目标上调优。
- **校准度量**：主指标为Expected Calibration Error（ECE，B=15等宽bin）；辅指标包括NLL（负对数似然）、Brier Score、Signed ECE（衡量过/欠置信方向）、pixel-ECE（分割任务）。
- **偏移应力轴**：
  - **协变量偏移**：三种物理驱动的合成腐败——①云层/阴影（Perlin噪声合成，覆盖率约20%→全覆盖）；②Poisson-Gaussian传感器噪声（模拟CCD/CMOS退化）；③运动模糊（水平均匀核，k∈{3,5,9,15,21}像素），各含5级严重程度。
  - **训练数据预算**：5个比例{0.01, 0.10, 0.25, 0.50, 0.75}重训探针，测量Kendall τ排名稳定性。
- **表征漂移分析（CKA）**：在编码器中间层（4个等距层）提取激活矩阵，计算Clean↔Corrupted间的Centered Kernel Alignment（Eq. 9），同时追踪per-sample cosine相似度和参与率（participation ratio），以区分结构保持与逐样本漂移。
- **UQ方法**：①无校准基线（直接softmax）；②温度缩放（在干净校准集上拟合单一标量T）；③深度集成（M=5个独立探针，AdamW优化）；④稀疏变分高斯过程（SVGP）分类头，使用线性核。
- **选择性预测**：在90%覆盖率下测量选择性准确率及eAURC，检验置信度是否能识别需要拒绝的样本。

## 实验与结果
- **数据集**：分类4个（ADVANCE、m-EuroSAT、RESISC45、So2Sat）；分割5个（CloudSEN12、DynamicEarthNet、FLAIR#2、Fields of the World、SpaceNet2）。Calibration集统一为400样本，来自干净验证集。
- **关键数字**：
  - 干净条件下：GeoFM平均准确率0.847/ECE 0.034 vs. 通用编码器0.839/0.038，无显著差异。
  - 云层偏移severity 5：16编码器ECE跨度0.32~0.82；Clay和TerraMind最严重（≈0.71），Swin-Tiny最优（0.32）。
  - Poisson-Gaussian噪声severity 3：7/16编码器ECE>0.6，TerraMind达0.83。
  - 均值Top-1准确率从干净0.84骤降至云层severity 3的0.34、severity 5的0.15。
  - **排名转移**：干净准确率→偏移后准确率的Kendall τ在云层和运动模糊下保留约0.4，但在Poisson-Gaussian噪声下severity 3降至0.05、severity 5变为-0.13；干净准确率→偏移后ECE的τ在所有条件下≤0.3；干净ECE→偏移ECE的τ在云层severity 3仅0.07。
  - **领域条件效应**：GeoFM编码器在两种偏移下的sev-5/clean ECE比率分别为：运动模糊4.75×、Poisson-Gaussian 8.30×；通用编码器为3.47×和7.34×。
  - **数据预算**：准确率排名在1%数据下仍保持τ=0.75（vs. 75%），校准排名降至τ=0.28。
  - **缓解方法**：温度缩放对偏移后ECE几乎无效（EO均值0.400→0.403）；深度集成在severity 3改善但severity 5对部分编码器（如Prithvi-v2-300）反而恶化（ΔECE +0.027）；SVGP在严重云层下减半signed ECE，但干净数据ECE升至≈0.108（基线的3倍）。
  - **选择性预测**：90%覆盖率下的选择性准确率增益随偏移递减至零甚至为负；eAURC从干净0.038升至云层severity 5的0.201。

## 相关工作脉络
- **GeoFM评测审计**（Corley et al., 2026）：本文的延伸——该工作揭示了152篇GeoFM论文中准确率数字的不一致性（46例相差>10点），本文在此基础上进一步指出校准/鲁棒性维度完全缺失。
- **EarthShift**（Doerksen & Kerner, 2026）：真实EO偏移基准，评估8个GeoFM的跨地理/传感器/时间鲁棒性，仅报告准确率。本文结果与其一致（EO预训练无准确性鲁棒优势），并首次在该维度引入校准视角。
- **REOBench**（Li et al., 2025）：专为EO设计的鲁棒性基准，同样仅使用准确率指标。本文补充了校准维度并发现排名不稳定。
- **TARDIS**（Ekim et al., 2025）：基于内部激活的EO域OOD检测器，属于"检测偏移"而非"在校准偏移后仍可信"的范畴。本文论证softmax置信度在偏移下不可靠，暗示应转向基于表征的信号。
- **SHRUG-FM**（Gonzalez-Calabuig et al., 2026）：结合输入空间/嵌入空间/task-level不确定性的选择性预测框架，聚焦拒绝错误样本；本文结论与其互补——两者均发现置信度在偏移下失效，但SHRUG关注实际部署策略，本文聚焦校准机理。
- **Ovadia et al. **(2019)：证明温度缩放在偏移下失效；本文将这一结论扩展到GeoFM语境，并补充深度集成和SVGP的对比。
- **CKA方法**（Kornblith et al., 2019）：表征相似性度量；本文将其用于分析GeoFM偏移下的表征刚性，建立与校准失败的关联。

## 局限性与未来方向
- **合成偏移 ≠ 真实偏移**：使用的三种腐败是物理模拟的，与EarthShift等真实偏移基准互补而非替代；几何变换（旋转、缩放、平移）等未测试。
- **相关性≠因果性**：表征刚性假说基于单seed的跨模型相关分析（B1-B3）及相关CKA-ECE关联，需通过控制实验（固定架构/容量，仅变预训练数据）验证因果。
- **冻结探针的限制**：所有结果基于冻结骨干+线性探针，未经微调；全微调下的校准行为未测。虽然EarthShift暗示微调对OOV鲁棒性无显著影响，但校准维度的微调效应仍是开放问题。
- **分割解码器单一**：仅使用轻量FPN解码器，更复杂的上下文建模（如UperNet、DeepLab）下的校准行为未测。
- **领域归因的混杂因素**：EO与通用编码器在骨干架构、参数量、训练配方上不同，预训练数据的"augmentation exposure"（如自然图像编码器有Gaussian blur/photometric jitter增强）可能是部分解释，但无法在此分离。
- **未来方向**：① 探索不同于softmax置信度的偏移感知信号（如基于表征的OOD检测）；② 测试JEPA类预训练目标是否产生更鲁棒的表征；③ 在真实EO偏移基准上验证校准排名稳定性；④ 研究冻结vs.微调范式下的校准退化机制。

## 研究启发与可借鉴点
- **评测协议设计**：本研究的双轴压力测试框架（偏移严重程度×数据预算）值得复用到其他Foundation Model评测中——单一clean-data指标无法捕捉真实部署中的模型行为分化。
- **CKA表征分析的可迁移性**：将CKA、cosine drift、participation ratio三路指标联合用于分析模型在偏移下的表征退化，方法简洁且与校准失败直接关联，可迁移至其他视觉领域（如医学影像、自动驾驶）。
- **排名转移（rank transfer）作为鲁棒性度量**：使用Kendall τ量化"干净→偏移"的排名保持程度，比单一偏移条件下的数值更系统地刻画模型退化模式，可作为新的评测基准维度。
- **选择性预测的警示**：置信度驱动的拒绝策略在偏移条件下失效，提示下游工作应避免直接依赖softmax confidence做uncertainty-based abstention，转而探索基于embedding-space的判别信号。
- **团队结合机会**：若本团队涉及GeoFM微调或部署，可借鉴本文的三轴评估框架（准确率+校准+选择性能），作为模型卡片（model card）的标准报告项；同时，探索基于表征漂移的信号作为"何时拒绝预测"的替代方案。

## 关键术语表
- **Calibration（校准）**：模型预测置信度与经验准确率的一致性；校准良好的模型应满足"自信度为p的预测中，准确的比例约为p"。
- **Expected Calibration Error (ECE)**：将预测置信度分箱后，各箱内准确率与平均置信度的加权绝对差之和，是最常用的校准度量。
- **Distribution Shift（分布偏移）**：训练分布与测试分布不一致；本文聚焦covariate shift（输入分布改变但标签规则不变）。
- **Rank Transfer（排名转移）**：用Kendall τ衡量同一批模型在干净条件和偏移条件下的排名保持程度。
- **Centered Kernel Alignment (CKA)**：衡量两个激活矩阵间表征相似性的旋转不变度量，用于分析偏移下表征空间的几何结构变化。
- **Selective Prediction（选择性预测）**：允许模型对低置信度预测拒绝输出，交由人工或其他处理环节，以换取高覆盖率下的高准确率。
- **Signed ECE**：ECE的方向性版本（mean confidence − mean accuracy），正值表示过置信，负值表示欠置信。
- **Participation Ratio（参与率）**：表征空间有效维度的度量，衡量激活方差在多少个主成分方向上非平凡分布。

## 可复现要素
- **数据集**：9个公开数据集（ADVANCE、m-EuroSAT、RESISC45、So2Sat、CloudSEN12、DynamicEarthNet、FLAIR#2、Fields of the World、SpaceNet2），均已引用原始论文。
- **代码/权重**：论文未提及统一的开源代码仓库；16个编码器来自各自公开发布的预训练权重（如Clay、Prithvi V2、DINOv3、OLMo-Earth、TerraMind等均有公开访问）。
- **关键超参**：线性探针正则化强度从准确率最优选取（未在校准目标上调优）；温度缩放在400样本的干净校准集上拟合；深度集成M=5；SVGP使用线性核（初始尝试Matern核不稳定）；CKA在4个等距中间层测量；腐败使用确定性seed（base seed, image index）。
