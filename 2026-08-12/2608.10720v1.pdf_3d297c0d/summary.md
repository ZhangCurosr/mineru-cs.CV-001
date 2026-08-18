---
title: "Ex-Omni-2D: Expressive Omni-Modal Dialogue Models with Native Visual Presence"
source: https://arxiv.org/pdf/2608.10720v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:56:01"
field: "多模态对话与生成"
keywords: ["全模态对话", "视觉Thought Plan", "流式视频生成", "多码本语音单元", "分阶段训练", "Prefix Streaming"]
innovations: ["结构化VTP将对话上下文转化为显式视觉意图规划", "原生多码本语音单元作为共享声学-时序接口解耦分路训练", "Prefix Streaming机制减少流式生成累积后期退化"]
benchmarks: ["CommonEval", "OmniCharacter", "VoiceBench"]
---

# 论文速读：Ex-Omni-2D: Expressive Omni-Modal Dialogue Models with Native Visual Presence

## 一句话总结
论文提出 Ex-Omni-2D，一个支持对话原生视频响应的全模态对话框架，通过结构化的 Visual Thought Plan (VTP) 和原生多码本语音单元接口，将对话语义规划与视听生成路径解耦训练，同时提供全序列 Teacher 和流式 Student 两种视频生成器，实现质量与效率的互补。

## 研究问题与动机
- **现有全模态对话系统缺乏视觉存在感**：当前系统能理解语音/文本输入并生成口语回复，但无法生成同步的视频形象响应，视觉上"无躯体化"。
- **视频驱动方法缺乏对话状态感知**：Talking-avatar 和音频驱动动画方法通常依赖已完成波形或外部提示，无法从对话状态推导响应特定的视觉意图。
- **联合音视频生成以提示为中心**：Joint audio-video 生成从 prompt 或参考信号创建内容，而非响应多模态用户查询，缺少对话条件机制。
- **大规模对齐数据难以获取**：query–text–speech–video 四模态对齐的对话数据规模有限，需要支持异构数据分阶段学习的框架设计。

## 核心贡献（创新点）
1. **提出对话原生视频响应的全模态对话框架**：将视觉行为规划从对话上下文推导，而非依赖手动提示，通过 VTP 显式建模场景、情感、动作风格与细节。
2. **引入原生多码本语音单元作为共享声学-时序接口**：用 16-codebook 语音单元同时驱动语音合成与视频帧对齐，避免波形后处理重编码，支持从异构数据分路训练。
3. **设计 Prefix Streaming 流式学生模型**：将全序列 Teacher 蒸馏为少数步块因果 Streaming Student，通过前缀连续潜变量传递减少累积后期退化。
4. **分阶段训练范式实现路径解耦**：四个训练阶段分别处理语音接口对齐、多模态响应适配、头像视频实现、流式蒸馏，各路径使用各自可用数据，无需大规模对话-视频对。

## 方法详解
- **多模态角色接地**：文本和语音输入通过 Qwen3-VL-2B 视觉塔和语言模型嵌入空间投影，参考图像分别输入对话模型空间和 Video Generator 的 3D VAE 获取外观参考潜变量。
- **视觉响应规划（VTP）**：LLM（Qwen3-8B）遵循结构化协议，先生成 `<thinking>` 块内的 VTP（五个字段：first_frame_scene、scene、emotion、movement_style、motion_description），再生成用户可见的 response。
- **共享声学接口**：Speech Generator（基于 Qwen3-TTS-0.6B）从响应隐状态生成 16-codebook 语音单元（12.5 Hz），同一单元既解码为语音波形，又通过轻量 adapter 投影为帧对齐的视频条件特征（25 FPS，每个声学帧重复两次）。
- **全序列 Teacher**：基于 Wan2.1-T2V-1.3B 和 OmniAvatar-1.3B LoRA，使用双向时间上下文，接收参考潜变量、帧对齐声学条件和 VTP 文本编码作为条件，通过 flow-matching 去噪。
- **Prefix Streaming Student**：采用 AnyFlow 的 flow-map 和 on-policy distillation，每个 DiT 窗口保持四个潜变量槽位：初始窗口包含参考潜变量+三个新生成潜变量；后续窗口将前一段最终干净潜变量作为 detached prefix，再加三个新潜变量，stop-gradient 防止漂移。
- **四阶段训练**：Stage 1 用 800K ASR + 1M TTS 样本训练 Speech Projector 和 Generator；Stage 2 用 InstructS2S-200K 和 OmniCharacter 适配 LLM 和 VTP-response 协议；Stage 3 用 140K SpeakerVid 片段训练全序列 Video Generator；Stage 4 将 Teacher 蒸馏为 Streaming Student（Phase I 学习 flow-map，Phase II on-policy distribution matching）。

## 实验与结果
- **数据集**：CommonEval（VoiceBench，200个语音查询）、OmniCharacter（400轮多轮对话）、SpeakerVid（视频生成数据）。
- **语音 QA**：AlpacaEval 4.28、CommonEval 3.71、BBH 58.70，仅次于 Qwen2.5-Omni。
- **多轮对话质量**：Fluency 3.812、Coherency 4.100、Consistency 3.902，三项平均分 3.938，优于 Qwen3-8B 基线 3.537。
- **音视频质量（Teacher，50步）**：SC 94.62、IQ 67.31、DD 72.00、Sync-C 4.95、SIM 0.417。
- **流式 Student（4步，四卡）**：SC 93.65、IQ 57.40、DD 32.00、Sync-C 4.00，E2E RTF 1.293，首段可播放视频 3.142秒后出现。
- **Prefix Streaming 消融**：相比无前缀变体，SC 从 92.85 提升至 93.65，IQ 从 55.18 提升至 57.40，Sync-C 从 3.69 提升至 3.90，后期主体退化斜率降低 21.4%。
- **VTP 消融**：移除响应特定 VTP 导致 SC 下降 1.04、Sync-C 下降 0.30；移除个性化参考语音导致 SIM 从 0.417 降至 0.015。

## 相关工作脉络
- **Omni-modal Dialogue（全模态对话）**：Qwen2.5-Omni、Mini-Omni2、SLAM-Omni 等系统生成口语回复但缺乏视觉存在；本文区别于它们在于生成参考图像条件化的同步视频响应。
- **Audio-driven Portrait Animation（音频驱动肖像动画）**：echomimic、OmniAvatar、StableAvatar 等以已完成波形为驱动条件，不直接从对话状态推导视觉意图；本文通过 VTP 实现对话驱动。
- **Joint Audio-Video Generation（联合音视频生成）**：UniVerse-1、UniAVGen 等以 prompt 或参考信号生成配对音视频；本文定位为对话条件响应而非提示生成。
- **Streaming Video Generation（流式视频生成）**：StreamAvatar 等探索实时人类头像生成；本文引入 Prefix Streaming 机制减少累积退化，与分阶段训练范式结合。

## 局限性与未来方向
- **语音相似度仍有提升空间**：生成语音与参考说话人的相似度（SIM）虽达 0.417，但仍有改进空间。
- **VTP 非独立充分控制信号**：VTP 提供高层语义指导而非独立视频控制，最终效果由 VTP 和声学条件联合决定，固定 Text CFG 和 Audio CFG 比例可能不适用于所有响应。
- **语言-视觉能力权衡**：共享自回归通道生成 VTP 导致 speech-QA 和推理分数下降（BBH 降低 2.40 分）。
- **Teacher 计算成本高**：50步双向去噪全序列生成速度慢（E2E RTF 26.917），不适合交互部署。
- **Streaming Student 非端到端实时**：首段视频 3.142 秒后出现，单请求 E2E RTF > 1，流式指增量输出而非端到端实时交互。
- **未来方向**： planner 隔离、自适应跨模态引导、因果视频骨干网络扩展、few-step 蒸馏优化。

## 研究启发与可借鉴点
1. **分阶段异构数据训练策略**：将对话、语音、视频路径分开训练，通过共享接口（VTP + 语音单元）在推理时重新连接，避免对大规模四模态对齐数据的依赖，适用于数据稀缺的多模态生成场景。
2. **Prefix Streaming 机制设计**：将前一段最终干净潜变量作为 detached prefix 注入当前窗口，通过 stop-gradient 防止漂移，有效缓解自回归流式生成的累积退化问题，可迁移至其他流式视频生成任务。
3. **多码本语音单元作为跨模态接口**：16-codebook 单元同时支持语音合成和帧对齐视频条件，固定 12.5 Hz 速率与 25 FPS 视频自然对齐，为多模态生成提供简洁的声学-时序桥梁。
4. **结构化 VTP 显式规划**：将视觉意图分解为五个可解释字段（场景、情感、动作风格等），通过受限解码确保规划完整性，比直接映射 LLM 隐状态更具可监督性和可解释性。

## 关键术语表
**Ex-Omni-2D**：一种支持对话原生视频响应的全模态对话框架，能生成文本、个性化语音和参考图像条件视频的协调响应。
**Visual Thought Plan (VTP)**：结构化的视觉意图描述，包含 first_frame_scene、scene、emotion、movement_style、motion_description 五个字段，将对话上下文转化为显式视觉引导。
**Native Multi-codebook Speech Units**：原生多码本语音单元，16 个码本组成的离散声学表示（12.5 Hz），同时驱动语音合成和帧对齐视频生成。
**Prefix Streaming**：流式生成机制，每个后续窗口以前一段最终干净潜变量作为 detached prefix，配合 stop-gradient 减少累积后期退化。
**Flow-matching**：流匹配参数化方法，去噪网络预测 ε - x₀ 目标，用于视频扩散模型训练。
**On-policy Distillation**：on-policy 分布匹配蒸馏，Streaming Student 在自生成的 chunk  rollout 上与 Teacher 进行分布匹配。
**Sync-C (SyncNet Confidence)**：唇音同步置信度指标，衡量嘴部运动与生成语音的时间对齐程度。
**E2E RTF (End-to-End Real-Time Factor)**：端到端实时因子，生成媒体时长与处理时长之比，< 1 表示快于实时。

## 可复现要素
- **数据集**：CommonEval（VoiceBench）、OmniCharacter、SpeakerVid；部分数据公开，部分需申请。
- **代码/权重**：项目页面 https://logo-cuhksz.github.io/Ex-Omni-2D；基础模型 Qwen3-8B、Qwen3-TTS-0.6B、Wan2.1-T2V-1.3B、OmniAvatar-1.3B LoRA 公开。
- **关键超参**：Teacher 50步去噪、Student 2/4/8步、Text CFG 3.5 / Audio CFG 8.5（Teacher）、Text/audio CFG 1.0/1.0（Student）、12.5 Hz 声学帧率、25 FPS 视频帧率、125帧 Teacher 训练窗口、121帧 Student 训练窗口。
