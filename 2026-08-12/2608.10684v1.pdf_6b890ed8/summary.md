---
title: "Embedding Rotation Invariance for Provable Multi-Oriented Scene Text Recognition"
source: https://arxiv.org/pdf/2608.10684v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:52:25"
field: "场景文字识别"
keywords: ["Scene Text Recognition", "Rotation Invariance", "Equivariant Networks", "Cross-Attention", "Multi-Oriented Text"]
innovations: ["证明交叉注意力在固定查询下具有旋转不变性并据此设计 RITD 解码器", "提出 RELG 编码器：深层等变卷积配合渐进下采样与自注意力，兼顾细粒度保留与字间依赖建模", "端到端内嵌旋转不变性于 STR 架构，无需额外校正模块或旋转数据增强即可实现多方向鲁棒识别"]
benchmarks: ["Union14M-Benchmarks", "IC13", "SVT", "IC15", "SVTP", "CUTE80", "IIIT5k", "ASOT"]
---

# 论文速读：Embedding Rotation Invariance for Provable Multi-Oriented Scene Text Recognition

## 一句话总结
本文提出 RISTER，一种将旋转不变性内嵌到场景文字识别（STR）框架中的端到端网络：编码器采用旋转等变设计（RELG），解码器利用交叉注意力的旋转不变性（RITD），在无额外推理开销、无需旋转数据增强的前提下，实现了具有理论保证的多方向文字识别，并在多个基准上达到 SOTA。

## 研究问题与动机
1. **理论保证缺失**：现有方法（如 AON 多方向编码、ASTER/SVTRv2 校正模块）需显式感知文字朝向，若朝向估计出错则识别性能骤降。
2. **计算开销增加**：引入额外的朝向感知网络（如 TPS、极坐标变换）显著增加推理成本，降低效率。
3. **数据依赖严重**：旋转数据增强迫使模型为不同朝向分别学习特征，降低特征利用效率，且固定模型容量下会牺牲水平方向的精度。
4. **已有旋转等变网络不适配 STR**：纯卷积等变网络无法建模字间依赖；基于注意力的等变 Transformer 通常依赖浅层卷积 tokenization，导致早期下采样损失细粒度细节，难以直接复用为 STR 骨干。

## 核心贡献（创新点）
1. **首次在 STR 中系统嵌入旋转不变性**：与现有"显式估计朝向+校正"的思路不同，本文从架构层面保证旋转输入产生不变的输出，无需额外校正模块。
2. **证明交叉注意力的旋转不变性并提供理论推导**：在查询固定时，对源特征做任意空间旋转后，交叉注意力的输出保持不变；由此导出 RITD 的设计原则，这是以往 STR 文献中被忽视的性质。
3. **提出 RELG（旋转等变局部-全局提取网络）作为编码器**：与已有等变骨干（如纯 F-Conv 或浅层等变 ViT）的本质区别在于：采用深层等变卷积配合渐进下采样保留细粒度几何信息，同时引入自注意力建模字间依赖。
4. **端到端构建 RISTER 并在 14 个基准上取得 SOTA**：在通用多方向数据集上较第二强模型提升 4.0%；且推理 FLOPs、参数量、FPS 与标准 AR 方法相当，无额外开销。

## 方法详解
- **整体架构**：采用 Encoder–Decoder 范式。编码器 RELG 对输入图像 x 进行特征提取与序列关系建模，产出视觉表示 $F \in \mathbb{R}^{c \times h/8 \times w/8}$；解码器 RITD 自回归地将 F 映射为文本序列 y。
- **RELG（旋转等变编码器）**：
  - **Rot-E Local Extraction**：使用 F-Conv（基于傅里叶级数参数化的群等变卷积）构建 Local Block，以离散旋转群 $C_4$（对应 0°/90°/180°/270°  canonical angles）实现旋转等变局部特征提取；经过 3 个阶段渐进下采样（分辨率分别降至 1/2、1/4、1/8，通道数扩展至 96→192→384），最大程度上保留细粒度视觉细节。
  - **Rot-E Global Extraction**：对 $F_3$ 施加堆叠的多头自注意力 + MLP 构成的 Global Block，建模字间依赖并编码位置信息；已有工作表明自注意力层对旋转具有等变性。
- **RITD（旋转不变解码器）与交叉注意力不变性**：
  - **关键性质**：$\mathcal{C}(Q, R_\theta(F)) \equiv \mathcal{C}(Q, F)$，即在查询 Q 固定时，对源特征 F 做任意角度 θ 的空间旋转，交叉注意力的输出完全相同（在 canonical angles 精确成立，其他角度近似成立，证明见附录 A）。
  - **设计原则**：(1) 使用交叉注意力完成模态交互；(2) 注入交叉注意力的查询 token 在图像旋转下保持不变。
  - **解码过程**：时间步 t 时，先将已解码字符序列 $y_{0:t-1}$ 经 Embedding + MHSA + MLP 得到查询 $Q_{0:t-1}$；取 $Q_{t-1}$ 作为 query，F 作为 key/value 输入交叉注意力，经 MLP 后对 logits 取 argmax 得到 $y_t$。由于 Q 与输入图像旋转无关，每一步预测在旋转前后保持一致。
- **损失函数**：采用标准自回归损失（AR Loss），即负对数似然 $\min_\mathcal{W} \mathbb{E}_{(x,y)\sim\mathcal{D}}[-\sum_{t=1}^{L}\log p_\mathcal{W}(y_t|y_{<t}, x)]$。

## 实验与结果
- **数据集**：训练集 Union14M-Filter（3.2M 图像，来自 17 个数据集）；评测涵盖 14 个英文基准：6 个常见 STR 基准（IIIT5k、SVT、IC13、IC15、SVTP、CUTE80）+ 7 类 U14M-B  challenging 类别（Curved、Multi-Oriented、Artistic、Contextless、Salient、Multi-Words、General 400k）+ ASOT（3000 张多方向样本）。
- **消融验证旋转不变性**：RISTER 在旋转前后输出 logits 余弦相似度为 1（严格不变）；去除 Rot-E 编码器或 Rot-I 解码器后，输出随旋转角度显著变化（Table 2）。
- **编码器对比**：RELG 在 U14M-Multi-Oriented 上达到 96.6%，较第二强编码器 SVTR 提升 2%；在 0° 样本上与 SVTR 表现相当；去除自注意力（RELG†）后在 IC13/SVT/U14M-MO 分别下降 0.6%/1.9%/1.6%（Table 3）。
- **输入分辨率影响**：将分辨率从 32×128 提升至 128×128 可显著改善多方向样本识别，而水平样本精度基本不变；RISTER-S(128×128) 在 Mul-Ori 和 ASOT 上分别达 96.0% 和 93.3%，较 IGTR/SVTRv2 同分辨率分别高出 6.00%/2.95%（Table 5）。
- **与多方向识别器对比**：RISTER-S 在 Common 基准上与 SVTRv2 接近，在 Oriented 基准上较第二强方法提升 4.00%（Table 6）。
- **与 SOTA 对比**：RISTER-L 在 13 个基准中的 8 个取得最高精度（U14M-B: CUR 94.1/MO 96.6/ART 79.4/CTL 80.6/SAL 90.3；Common: IC13 98.9/SVT 98.1/IIIT 99.0/IC15 90.6），参数量 < 40M；RISTER-S 参数量 21.8M，在 U14M-B 平均准确率 97.2%，超越多数现有方法（Table 7）。
- **计算效率**：F-Conv 训练后参数通过循环移位固定，推理成本与标准卷积相同；RISTER-S 的 FLOPs (12.99×10⁹)、参数 (21.8M)、FPS (31.4) 与 NRTR 相当（Table 4）。

## 相关工作脉络
1. **AON [3]**：四方向编码 + 滤波门控融合，增加计算开销且不同方向特征空间语义可能不一致；本文从架构内生旋转不变性，无需多方向编码。
2. **SLOAN [5]**：笛卡尔→极坐标变换将旋转转化为平移，但极坐标变换造成空间畸变削弱特征提取；本文避免坐标变换，直接保证旋转等价性。
3. **ASTER [39] / ESIR [56] / SVTRv2 [11]**：基于 TPS 或特征重排的校正模块，引入额外计算且缺乏理论保证；本文无需任何校正模块。
4. **G-CNN [4] / F-Conv [49]**：群等变卷积，主要针对分类/低视觉任务；本文将其深度化并结合渐进下采样适配 STR 需求。
5. **LieTransformer [17] / Stand-Alone [32]**：等变自注意力 Transformer，依赖浅层卷积 tokenization，早期下采样损失细粒度细节；本文 RELG 通过深层等变卷积保留几何细节后再引入自注意力。
6. **PARSeq [1] / SVTRv2 [11]**：当前 STR SOTA 基准；本文在其基础上通过旋转不变性进一步提升多方向鲁棒性，且在通用基准上仍保持竞争力。

## 局限性与未来方向
- 理论证明在 canonical angles（0°/90°/180°/270°）精确成立，其他角度为近似不变。
- 当前使用离散旋转群 $C_4$，未覆盖任意连续角度的严格等变。
- 输入采用方形 128×128，对非方形长文本或数学公式识别场景的适配仍需探索（论文在 Conclusion 中明确提及）。
- 未讨论极端长序列（L > 25）下的表现。

## 研究启发与可借鉴点
1. **交叉注意力旋转不变性的普适性**：该性质可迁移至其他视觉-语言任务（如文档理解、公式识别），在查询端保持不变即可实现解码端的旋转鲁棒性。
2. **深层等变卷积 + 渐进下采样 + 自注意力的骨干设计**：RELG 证明了在 STR 任务中，仅靠浅层等变 ViT 不足以同时保持 0° 精度和多方向鲁棒性，深层等变卷积保留细节是必要条件。
3. **用架构不变性替代数据增强**：本文证明内嵌旋转不变性优于旋转数据增强——后者在固定容量下会牺牲水平样本精度；这一思路可扩展至其他对称性（如平移、缩放）。
4. **输入分辨率对多方向识别的关键作用**：方形高分辨率输入（128×128）对竖排文字的特征表达至关重要，值得在多方向 STR 研究中统一评估分辨率敏感性。
5. **与团队方向结合的机会**：若团队关注数学公式识别或长文本识别，可将 RISTER 的旋转不变解码器推广至非正方形输入场景，结合等变编码器处理任意方向的公式/乐谱。

## 关键术语表
- **RISTER**：Rotation-Invariant Scene TExt Recognition，本文提出的端到端旋转不变场景文字识别网络。
- **RELG**：Rotation-Equivariant Local–Global Extraction，本文提出的旋转等变局部-全局提取编码器骨干。
- **RITD**：Rotation-Invariant Text Decoder，基于交叉注意力旋转不变性设计的旋转不变文本解码器。
- **F-Conv**：Fourier-based equivariant convolution，基于傅里叶级数参数化的群等变卷积，用于实现旋转等变局部特征提取。
- **交叉注意力旋转不变性**：查询固定时，对源特征做空间旋转不改变交叉注意力输出，即 $\mathcal{C}(Q, R_\theta(F)) \equiv \mathcal{C}(Q, F)$。
- **Canonical angles**：规范朝向（0°、90°、180°、270°），本文离散旋转群 $C_4$ 对应的角度，不变性在此精确成立。
- **Rot-E / Rot-I**：分别为 Rotation-Equivariant（旋转等变）和 Rotation-Invariant（旋转不变）的缩写。
- **WAIC**：Word Accuracy Ignore Cases，文中采用的评测指标（大小写不敏感的单词级准确率）。

## 可复现要素
- **训练数据集**：Union14M-Filter（3.2M 图像，来自 17 个数据集）——论文未说明是否独立公开，Union14M 系列数据集通常可从原论文获取。
- **代码/权重**：模型基于 PaddlePaddle 实现，论文未明确提供开源链接（需关注作者主页或后续 release）。
- **关键超参**：输入分辨率 128×128；batch size 512；optimizer AdamW（weight decay 0.05）；learning rate 3.2×10⁻⁵（Tiny/Small）/ 1.6×10⁻⁵（Base/Large）；OneCycleLR scheduler，1.5 epoch 线性 warmup，共 20 epoch；最大文本长度 L=25；字符集大小 N=96（含字母、数字、标点及 start/end/pad token）；旋转数据增强**未使用**。
- **硬件**：单张 80GB A100 GPU。
- **评测指标**：WAIC（Word Accuracy Ignore Cases）。
