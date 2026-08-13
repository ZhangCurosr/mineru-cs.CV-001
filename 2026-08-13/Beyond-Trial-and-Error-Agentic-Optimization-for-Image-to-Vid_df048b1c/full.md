# Beyond Trial-and-Error: Agentic Optimization for Image-to-Video Adherence

Aman Tyagi<sup>1</sup> amanty@google.com

Hemanth Boinpally<sup>1</sup> brhemanth@google.com

Jonathan Chen<sup>1</sup> jonchen@google.com

Douglas Gebert<sup>1</sup> dgebert@google.com

<sup>2</sup>Steven Hickson<sup>2</sup> shickson@google.com

<sup>1</sup> Google Cloud <sup>2</sup> Google DeepMind

## Abstract

Modern black-box Image-to-Video (I2V) models offer powerful capabilities in automated content creation, yet their lack of fine-grained control and reliability presents significant challenges in professional workflows. Their inherent stochasticity causes minor variations in textual prompts or hyperparameters to yield drastically different outputs often necessitating inefficient, brute-force trial-and-error processes. To address these limitations, we introduce the “Agentic Self-Improvement" framework, which reframes video synthesis into a closed-loop, goal-directed optimization. Our framework systematically navigates the generation parameter space using a novel two-stage approach. In the first stage, an iterative prompt optimization loop uses a multimodal Large Language Model (mLLM) to refine the input prompt. This refinement implements two automated evaluations: Davidsonian Scene Graph (DSG) queries ensure semantic adherence, and Common Mistake Questions (CMQ) for artifact detection. At the second stage, we use Bayesian optimization to efficiently co-optimize stochastic seeds and CFG scales. This search is guided by a suite of quality metrics, including the novel Video-Text Adherence (VTA) score derived from the DSG and CMQ evaluations. Our framework significantly outperforms unguided search methods: in human preference studies, videos generated via our agentic approach were strongly preferred over baseline outputs, achieving win rates up to 69%. This work provides a practical and extensible methodology for enhancing the predictability and control of state-of-the-art video generation models, moving the field beyond speculative curiosities toward reliable, production-ready tools.

## Introduction

The advent of large-scale, black-box text and image-to-video (I2V) models has unlocked powerful capabilities in automated content creation, promising a new frontier for digital media and storytelling [1, 2]. However, a significant gap remains between the generative potential of these systems and the practical demands of production-quality video synthesis [3, 4]. For professional applications, the foremost requirement is precise adherence to creative briefs, where the generated video must faithfully reflect both a textual prompt and a given input image [5, 6, 7, 8]. The core challenge lies in the inherent stochasticity and sensitivity of current models; minor variations in hyperparameters, such as random seeds or classifier-free guidance (CFG) scales, can lead to dramatically different and often unpredictable outputs [9, 10]. This forces practitioners into a brute-force, trial-and-error workflow, generating a multitude of video variations in the hope that one serendipitously aligns with their intent—a process that is inefficient and computationally expensive. Bridging this gap requires moving beyond speculative generation towards a systematic framework for intelligently navigating the complex parameter space of these models.

To address these limitations, this paper introduces “Agentic Self-Improvement", a framework that re-architects the generation pipeline for I2V models from an open-loop, speculative process to a closed-loop, goal-directed optimization. Our framework employs a two-stage strategy that systematically targets the primary sources of variance: the textual prompt and the model’s hyperparameters. First, a prompt optimization stage uses a multimodal Large Language Model (mLLM) to iteratively refine the user’s initial prompt. To guide this refinement, the mLLM generates and scores against a structured set of evaluations: Davidsonian Scene Graph (DSG) queries [11] that probe semantic content, and Common Mistake Questions (CMQ) designed to detect typical generation artifacts. Second, a hyperparameter optimization stage uses Bayesian optimization to systematically search optimal random seeds and CFG scales. The entire process is guided by a suite of quality metrics and the novel Video-Text Adherence (VTA) score derived directly from the DSG and CMQ evaluations providing a quantitative measure of the output’s quality and alignment. The result is video content that is not only of high quality but is also aligned with the user’s specified intent.

Our work makes the following primary contributions:

• An “Agentic Self-Improvement" framework for systematic, goal-directed control of black-box I2V generation models.

• A prompt optimization method that uses an mLLM with automatically generated DSG and CMQ evaluations for iterative refinement.

• A Bayesian optimization to efficiently co-optimize stochastic seeds and CFG scales.

• A novel Video-Text Adherence (VTA) metric derived from structured DSG and CMQ evaluations to guide the optimization process and score final content.

## 2 Related Work

## 2.1 Advances in Text/Image-to-Video Generation

The field of video generation has seen a paradigm shift from Generative Adversarial Networks (GANs) [12] and Variational Autoencoders (VAEs) [13] to diffusion-based models [14]. Early methods often struggled with temporal coherence and were limited to lowresolution outputs [15]. The advent of diffusion models, particularly with the introduction of large-scale models such as Veo <sup>1</sup>, has marked a significant leap in generating high-fidelity, temporally consistent video sequences from textual prompts [16]. These models leverage vast datasets and sophisticated architectures, such as the Diffusion Transformer (DiT), to achieve unprecedented levels of realism and semantic accuracy [17].

More recently, the field has seen a rapid convergence towards unified autoregressive models and multimodal foundation models capable of natively processing interleaved text, image, and video tokens [18]. However, despite these improvements in raw fidelity, their black-box nature and reliance on massive, often uncurated, datasets present significant challenges in terms of controllability and reproducibility [19, 20].

## 2.2 Controllability in Video Synthesis

While text-to-video models have become increasingly powerful, relying solely on a textual prompt for control is often insufficient for professional applications that demand precise outputs [16]. This has spurred research into more fine-grained control mechanisms. One line of work focuses on incorporating additional conditioning inputs, such as a reference image for appearance control, to guide the generation process [21]. Other approaches leverage spatial controls like depth maps, edge maps, or human pose information to dictate the scene’s geometry and character motion [19]. A significant challenge in this domain is the disentanglement of content from motion, and achieving precise, multi-faceted control over the generated video’s content, style, and dynamics remains a key area of research [19, 22].

## 2.3 Automated Prompt Engineering

Prompt engineering or prompt optimization for diffusion models, has moved beyond manual tuning towards automated and systematic approaches. A significant body of work has explored gradient-based methods, which treat prompt embeddings as continuous vectors that can be optimized to maximize a differentiable reward function, a technique that has been shown to effectively discover “hard", interpretable prompts [23]. Another prominent direction involves the use of reinforcement learning (RL), where a policy network is trained to refine prompts based on feedback from a reward model, which can be designed to reflect human preferences or other desirable image characteristics [24]. Furthermore, researchers have leveraged large language models (LLMs) as optimizers themselves, employing them in black-box settings to iteratively rewrite and improve prompts based on the generated image outcomes, demonstrating strong performance without requiring direct access to the diffusion model’s gradients [11, 25, 26]. Building upon these efforts to improve textual alignment, our work advances the structured prompting methodology of [11] by adapting it for I2V/T2V synthesis, employing DSG to enforce a higher degree of text adherence and compositional consistency in the generated video.

## 2.4 Hyperparameter Optimization and Agentic AI

The performance of generative models is also critically dependent on a set of hyperparameters, such as the classifier-free guidance (CFG) scale and the initial random seed [27]. The conventional method for tuning these parameters is a brute-force grid search, which can be computationally expensive and often suboptimal. To address this, more systematic optimization techniques have to be explored. By building a probabilistic model of the objective function, Bayesian optimization can intelligently explore the parameter space to find optimal configurations with fewer evaluations [28]. Our work extends this approach by co-optimizing both deterministic hyperparameters and the stochastic random seed within a unified framework.

![](images/3b933f4e8cdf3fb9864ae3d161c2ac4d5fe4a4aa660b000503e63675059cea03.jpg)  
Figure 1: The two-stage architecture of our Agentic Self-Improvement framework. Top: Prompt Optimization. A prompt is iteratively improved by generating a video and scoring its alignment against a set of auto-generated DSG and CMQ questions. Bottom: Hyperparameter Optimization. Using the best prompt, a Bayesian search then finds optimal hyperparameters (CFGs, seeds) by maximizing a multi-objective function composed of quality metrics (UVQ, RAHF) and a Video-Text Adherence (VTA) score repurposed from the questions above.

The concept of “Agentic AI" represents a shift from passive, single-turn generative models to active, goal-directed systems that can autonomously plan, reason, and act to achieve a specified objective. In the context of content creation, agentic frameworks are being developed to automate complex, multi-step generative tasks [29]. These systems often employ a large language model as a central "agent" that can decompose a high-level goal into a sequence of smaller steps, execute those steps using various tools (such as generative models), and then evaluate the results to refine its plan [29, 30]. Our “Agentic Self-Improvement" framework is inspired by this paradigm. The term ’Agentic’ distinguishes our approach from standard generate-evaluate pipelines by aligning with formal definitions of autonomous LLM agents (dynamic perception, reasoning, action [31, 32]). Instead of a static metric, our agent autonomously constructs a prompt-specific perception rubric (DSG/CMQ), reasons over structural failures, and executes goal-directed actions across linguistic (prompt rewriting) and mathematical (Bayesian search) parameter spaces, conceptualizing the video generation process as a closed-loop optimization problem.

## 3 Methodology

The Agentic Self-Improvement framework operates as a closed-loop, goal-directed system that systematically optimizes the two primary inputs to a black-box I2V model: the textual prompt and its core hyperparameters. As depicted in Figure 1, the framework achieves this through two main subsystems—a Prompt Optimizer module and a Hyperparameter Optimizer module—which work in concert to reliably align the final video output with the user’s fiprecise creative intent.

![](images/865362f9373305b5e07e1edb1dc34fe12739e7f47aa2d67587c203b95a480df2.jpg)  
Figure 2: Example of our automated question generation process for a given input image and prompt “a group of women sitting on the steps of a building". Top (DSG Questions): Davidsonian Scene Graph [11] are generated to deconstruct the visual scene into semantic queries. Bottom (Common Mistakes): Common Mistake Questions are designed to target and preempt known failure modes of video generation models.

## 3.1 Prompt Optimization via Structured Evaluation

A primary challenge in video generation is the semantic gap between a user’s prompt and the data distributions on which foundation models are trained. To bridge this gap, our framework employs an iterative prompt optimization strategy that refines the initial text prompt into an unambiguous and effective directive for the I2V model.

## 3.1.1 Question Generation for Semantic Grounding:

The process begins by translating the input prompt and initial image frame(s) brief into a fiset of concrete, machine-verifiable questions. We use a multimodal Large Language Model (mLLM), such as Gemini, to generate two categories of evaluative queries:

Davidsonian Scene Graph (DSG) Questions: Leveraging the Davidsonian Scene Graph methodology [11] for fine-grained visual evaluation, these questions probe the fundamental semantic components of the desired scene. They deconstruct the prompt into verifiable queries about the presence of agents, their actions and states, object possession, and location in a graphical format. For instance, in Figure 2 (Top) for the prompt “A group of women sitting on the steps of a building" is broken down into verifiable questions such as,“Is the group of women present”, “Are the women sitting down?”, and “Are the women sitting on steps?”. This structured decomposition allows our framework to systematically check for the presence and correctness of each core semantic element.

Common Mistake Questions (CMQ): While DSG effectively covers semantic content, it may not capture common failure modes of video generation models. CMQs are designed to target these specific weaknesses, focusing on criteria such as object permanence, natural movement, and temporal consistency while also checking for visual glitches or artifacts. Continuing with the same example as above in Figure 2 (Bottom), a CMQ could be, “Are the anatomical features of the characters depicted correctly and consistently?” or“Are the facial features of the women stable and free from distortion or flickering?” These questions fiforce the evaluation to assess the dynamic and temporal qualities crucial for plausible video

synthesis.

## 3.1.2 Iterative Prompt Refinement Loop:

With the structured question graphs in place, the framework initiates a self-correction loop to find the optimal prompt:

• The I2V model generates an initial video using the current prompt.

• The generated video is then evaluated by the mLLM Visual Question Answering (VQA) system<sup>2</sup>, which provides ‘Yes/No’ answers to the full set of DSG and CMQ questions. The percentage of affirmative answers serves as a quantitative alignment score.

• Based on the score and, more specifically, the questions that were answered ‘no,’ the mLLM is prompted to revise the prompt to explicitly address the identified shortcomings.

• This process is repeated for a fixed number of iterations, with each step aiming to maximize the alignment score.

• The prompt that yields the highest alignment score is passed to the next stage for hyperparameter optimization.

## 3.1.3 mLLM VQA Reliability

Because this prompt refinement loop relies entirely on the mLLM functioning as an automated evaluator, we must first validate its reliability against human judgment. To validate this critical component, we conducted a study to measure the agreement between the answers provided by our selected VQA model, Gemini 2.5 Pro, and a human-annotated ground truth.

For this validation, we randomly sampled 100 video-question pairs from our experiments. This sample included 50 Davidsonian Scene Graph (DSG) and 50 Common Mistake Questions (CMQ). Two human annotators were tasked with watching each video and providing a “Yes/No" answer to the corresponding question. These human-provided labels served as our ground truth. We then posed the same questions to the Gemini 2.5 Pro model and compared its answers to the human ground truth. The results of this analysis are presented in Table 1.

Table 1: Accuracy of Gemini 2.5 Pro on the VQA task compared to human-annotated ground truth. The high level of agreement validates its use as an automated evaluator within our framework.
<table><tr><td>Question Type</td><td>Accuracy (%)</td></tr><tr><td>DSG Questions</td><td>92%</td></tr><tr><td>CMQ Questions</td><td>82%</td></tr><tr><td>Overall</td><td>87%</td></tr></table>

The model was exceptionally accurate on DSG questions (92%), which typically probe for more objective factual content (e.g., object presence). The accuracy on CMQ questions (82%) was lower, which is expected as these questions can sometimes involve more subjective assessments of motion or temporal consistency. The 18% CMQ error rate primarily stems from the mLLM occasionally confusing fast, natural motion with unnatural transitions, and being insensitive to certain criteria. This strong correlation with human judgment confirms that Gemini 2.5 Pro is a reliable and effective proxy for human evaluation for our specific, structured VQA task, thereby validating its use as the automated scoring engine within our agentic framework.

## 3.2 Hyperparameter Optimization via Bayesian Search

With an optimal prompt identified, the framework’s second stage automates hyperparameter tuning. We employ Bayesian optimization to efficiently search for the ideal configuration of the Classifier-Free Guidance (CFG) scales, which are explored within a range of [1 to 15], and the stochastic seed, replacing inefficient manual tuning. Because random seeds are non-smooth identifiers, where numerical proximity does not imply similar outputs, Vizier BO addresses this by explicitly representing the seed as an unordered categorical variable, handling this mixed space via its Gaussian Process Bandit algorithm. This co-optimization outperforms blind random sampling by: (1) learning the continuous marginal effects of CFG, evaluating new seeds only within high-yield CFG ranges; and (2) treating seed selection as a multi-armed bandit problem, where the default Upper Confidence Bound (UCB) acquisition function (with an exploration constant of $\sqrt { \beta } = 1 . 8$ [31, 34]) quickly discards ’bad’ seeds and reallocates budget to fine-tune the CFG of promising seeds, avoiding wasted trials. The search is guided by a multi-objective reward function executed by Vizier, a platform for black-box optimization. This search pushes the generative model towards higher quality outputs that exhibit superior prompt adherence with minimal visual artifacts yielding optimal results far more efficiently than a brute-force search would allow.

## 3.2.1 Automated Evaluation Suite

The efficacy of our agentic framework hinges upon a robust suite of automated metrics that collectively form the multi-objective reward function. This suite provides a holistic assessment of each generated video across three dimensions: perceptual quality, temporal coherence, and semantic alignment.

Perceptual Quality: We employ two models to assess visual quality. First, we use the RAHF (Rich Automated Human Feedback) model [35], a multimodal Transformer designed to simulate nuanced human quality judgments for images. For each video, our evaluation process involves sampling 10 frames at equidistant intervals. These frames are individually passed through the RAHF model to generate the overall quality score for each. To determine a single representative score for the entire video, we then select the minimum score from this set of 10 frame-level scores. Second, we use UVQ (Universal Video Quality) [36], a CNN-based architecture for general quality prediction. For this, we use the Mean Opinion Score (MOS) derived from the average of its content, distortion, and compression networks, as described in the original work.

Temporal Coherence: We implement two metrics to evaluate temporal and motion consistency. Shotcut Detection & Flow Continuity analyzes optical flow between frames to quantify motion and penalize abrupt, unintended scene changes. Concurrently, loop detection identifies and penalizes undesirable content repetition, defined as a two-second loop or two-second static freeze frame.

Semantic Alignment: Finally, to ensure adherence to the prompt we employ our novel Video-Text Adherence (VTA) metric. This metric repurposes the DSG and CMQ graphs generated during prompt optimization to directly score the video’s content against the detailed scene expectations and text. We formalize VTA to explicitly detail hierarchy handling, weighting, and aggregation. VTA evaluates a video by aggregating the mLLM’s binary responses (1=Yes, 0=No) across generated DSG and CMQ trees. To handle hierarchy (e.g., zeroing “Is the dog running?” if the parent node “Is there a dog $? ^ { \ast }$ is False), a child’s score is strictly masked by its parent $p ( i )$ . We define recursive node score $S _ { i } = \mathbb { I } ( \mathrm { m L L M } ( Q _ { i } , V ) =$ Yes) $S _ { p ( i ) }$ , with roots defaulting to $S _ { p ( \mathrm { r o o t } ) } = 1$ . VTA is the weighted average over all questions Q: $\begin{array} { r } { \mathrm { V T A } ( V ) = ( \sum w _ { i } ) ^ { - 1 } \sum w _ { i } S _ { i } } \end{array}$ . Using uniform weights $( w _ { i } = 1 )$ , this recursive design ensures foundational concepts (root nodes) naturally exert a mathematically higher penalty upon failure due to cascading zeroing effects on descendant nodes.

While this specific suite provides a comprehensive assessment, the framework is inherently modular. Any component can be augmented or replaced with other black-box evalua tors tailored to different quality dimensions.

## 4 Experiments

Table 2: Human preference evaluation comparing the top-ranked video selected by different automated metrics against a baseline. The candidate videos were generated from 10 and 100 trials of Bayesian optimization, respectively. For each metric, the single best video was chosen for the head-to-head comparison. The ‘Avg’ metric is a simple average of normalized RAHF, VTA and UVQ. We additionally evaluate the 100-run budget against a stronger ‘Bestof-Random’ baseline to isolate the efficacy of the Bayesian search from the final metric selection.
<table><tr><td rowspan="2">Top-1 Selection Metric</td><td colspan="3">10 Runs (vs. Random)</td><td colspan="3">100 Runs (vs. Random)</td><td colspan="3">100 Runs (vs. Best-of-Random)</td></tr><tr><td>Baseline Win (%)</td><td>Ours Win (%)</td><td>Tie (%)</td><td>Baseline Win (%)</td><td>Ours Win (%)</td><td>Tie (%)</td><td>Baseline Win (%)</td><td>Ours Win (%)</td><td>Tie (%)</td></tr><tr><td>RAHF</td><td>14</td><td>47</td><td>39</td><td>9</td><td>69</td><td>22</td><td>8</td><td>42</td><td>50</td></tr><tr><td>VTA</td><td>8</td><td>45</td><td>47</td><td>8</td><td>63</td><td>29</td><td>13</td><td>39</td><td>48</td></tr><tr><td>UVQ</td><td>11</td><td>42</td><td>47</td><td>10</td><td>60</td><td>30</td><td>13</td><td>38</td><td>49</td></tr><tr><td>Avg.</td><td>6</td><td>45</td><td>49</td><td>9</td><td>66</td><td>25</td><td>15</td><td>28</td><td>57</td></tr></table>

We validate our Agentic Self-Improvement framework through two primary experiments. First, a controlled ablation study isolates and evaluates the Prompt Optimizer module against the original V-Bench prompts[37]. Second, an end-to-end evaluation implements a Bayesian Search framework across the hyperparameters (seed, CFG scales, etc.), which approximates a typical un-guided workflow. For all experiments, we ground our analysis in the imageand-prompt-to-video tasks from V-Bench. Crucially, our agentic framework operates as a model-agnostic wrapper that fundamentally applies to any latent diffusion video model. We utilize Veo $2 . 0 ^ { 3 }$ as the experimental vehicle for this study, explicitly positioning it as a robust, representative baseline.

Our primary evaluation modality is a head-to-head human preference study. We prioritize this qualitative analysis because, as we detail in the Appendix (see section 7.1), standard automated metrics often lack the sensitivity to capture the nuanced improvements our framework provides. For the preference study, two expert annotators each evaluated 50 disjoint prompts (100 total) to maximize coverage, strictly calibrated on shared guidelines. They were presented with text-image pairs alongside the corresponding videos generated by our framework and the baseline, and instructed to select the video with superior realism and overall video quality or to declare a tie. We performed two-tailed binomial sign tests (excluding ties) and calculated 95% Wilson score CIs. Our win-rates are statistically significant: for N = 100, CIs range from [50.2%, 69.1%] (UVQ) to [59.4%, 77.2%] (RAHF), $p < 1 0 ^ { - 6 }$ ; for N = 10, CIs range from [32.8%, 51.8%] to [37.5%, 56.7%], $p < 0 . 0 0 0 0 3$

## 4.1 Prompt Optimization

To validate the impact of our prompt optimizer module, we conducted a controlled experiment using 100 image-prompt pairs randomly sampled from V-Bench’s image-to-video tasks. For each pair, we generated two videos while keeping all hyperparameters constant: one using the original V-Bench prompt and a second prompt generated by our optimization module after 10 iterations. These video pairs were then evaluated in a blind, head-to-head human preference study. In the comparison, videos generated from our optimized prompts were preferred by human evaluators 27% of the time, compared to 5% for the baseline, with the remaining resulting in a tie (68%). This indicates that our method provides a significant perceptual improvement in over a quarter of the cases.

## 4.2 Bayesian Search

Next, we evaluate the efficacy of our complete agentic framework, against a random search baseline. The goal is to measure whether our intelligent search finds superior videos within the same computational budget. For the same 100 image-prompt pairs from V-Bench, we conduct experiments with two distinct budgets: a 10-generation search and a more exhaustive 100-generation search. The primary baseline method involves randomly selecting one video from the 10 (or 100) videos generated with an unguided random search of hyperparameters. To rigorously isolate the contribution of the Bayesian search algorithm from the efficacy of our final metric-based candidate selection, we additionally evaluate the 100-generation budget against a stronger ’Best-of-Random’ baseline. This competitive baseline generates 100 candidates via standard random search, applies our artifact filters, and intelligently selects the Top-1 candidate using our proposed automated ranking metrics. Our method uses the Bayesian optimizer for the same number of iterations and then selects the best candidate.

To determine the single best output from our pipeline, we first apply a hard filter to discard videos with loops or shotcuts. From the remaining pool of candidates, we test four distinct automated ranking strategies to select the ‘top 1’ video. The selection is based on choosing the video with the highest score according to: (i) the UVQ quality model, (ii) our VTA text-adherence metric, (iii) the RAHF human-preference model, and (iv) a simple average of all three normalized scores. The human preference win rates for each of these four ranking strategies against the random baseline are presented for both the 10- and 100- generation in Table 2.

The results in Table 2 demonstrate that our agentic framework consistently produces videos that are preferred by human evaluators over the random search baseline. With a limited budget of 10 runs, our method achieves win rates of 42-47% vs. 14% for the baseline. The performance gap widens dramatically with 100-generation budget, where our framework’s win rate climbs to 60-69%. This trend, coupled with a significant reduction in ties, demonstrates that our search becomes more effective at identifying perceptibly superior videos as the computational budget grows. Notably, sorting by the RAHF score yields the highest human preference (69% win rate), suggesting it is the metric most aligned with human perception in this context.

Evaluating against the significantly stronger ’Best-of-Random’ baseline naturally narrows the performance gap compared to an unguided, randomly selected candidate. Because this new baseline intelligently filters and selects the best of 100 random videos, tie rates predictably increase (e.g., from 22% to 50% for RAHF) and our absolute win rate shifts from 69% to 42%. Crucially, however, our Bayesian Optimizer still decisively outperforms this highly competitive baseline across all metrics (e.g., 42% vs. 8% for RAHF). This confirms that our framework does not merely rely on post-hoc sorting; rather, the Bayesian search actively and effectively guides the generation process into higher-quality parameter regions.

## 5 Discussion

Our experimental results validate the hypothesis of this work: a systematic, closed-loop optimization process yields perceptibly superior results compared to a standard baseline. Our experiments on prompt optimization confirm that treating the prompt as a refinable target, rather than a fixed input, is a critical step for improving semantic alignment. Such a technique also helps the prompt to be model agnostic as different models are sensitive to different prompting techniques [38]. The subsequent end-to-end evaluation via Bayesian search demonstrates that an intelligent search of the hyperparameter space is substantially more effective at discovering human-preferred videos than an unguided, random search. Collectively, these findings show that by methodically addressing the two primary sources of variance—the prompt and the hyperparameters—our framework successfully transforms video generation from a speculative, trial-and-error method into a more predictable and controllable process.

The success of the prompt optimization highlights a powerful paradigm: using the advanced multimodal understanding of mLLMs as a direct feedback signal to steer the creative output of generative models. The core principle is a “generator-critic" loop where the I2V model acts as the generator and the mLLM functions as a sophisticated visual critic. By generating structured DSG and CMQ questions, we provide the mLLM with a detailed rubric for its critique, forcing it to verify the fine-grained semantic and temporal properties of the generated video. This moves beyond simplistic CLIP-score similarity and instead leverages the mLLM’s rich, compositional understanding of scenes, actions, and objects to guide the generation process. This approach is highly extensible, pointing toward future systems where mLLMs could provide free-form textual feedback or critique videos on more abstract qualities like narrative coherence or emotional tone, creating a powerful, self-correcting cycle for more controllable and aligned video synthesis. However, this mLLM-guided approach is naturally constrained by the current capabilities of video understanding. While state-of-the-art mLLMs excel at identifying objects and describing general scenes, their ability to comprehend nuanced motion and complex temporal dynamics is still an active area of research [39]. We anticipate that this capability will advance significantly with the increasing convergence of multimodal AI and the field of robotics [40]. The demand for embodied agents that can interpret and predict complex physical interactions from video will inevitably push the development of models with a more profound grasp of motion and causality. As these more sophisticated video-understanding mLLMs become available, they can be seamlessly integrated into our framework as more powerful ‘critics,’ enabling the optimization of not just semantic content, but also physically plausible dynamics and complex character actions.

Despite its effectiveness, we acknowledge several limitations that open avenues for future research. Our agentic framework, while more efficient than unguided manual iteration, stil carries a significant computational cost. Future work could explore more sample-efficient optimization algorithms or the distillation of the large mLLM critic into a smaller, specialized reward model to reduce inference overhead. Furthermore, our framework currently operates as a two-stage process, first optimizing the prompt and then the hyperparameters. A more advanced approach would be to explore the joint optimization of both prompts and parameters simultaneously which could uncover non-obvious interactions between prompt phrasing and hyperparameter sensitivity.

## 6 Conclusion

This work addressed the critical challenge of controlling black-box I2V models to achieve reliable, quality outputs that adhere to user intent. We introduced the Agentic Self-Improvement framework, a novel two-stage methodology that systematically reduces generative variance. Our approach first uses an mLLM-driven evaluation loop to optimize the textual prompt for semantic clarity and then employs Bayesian optimization to discover superior hyperparameters. Through human preference studies, we demonstrate that this structured, closed-loop process significantly outperforms standard, unguided search methods. By reframing video synthesis as a goal-directed optimization problem, this work provides a practical and extensible methodology for instilling predictability into state-of-the-art generative models. This shift moves I2V systems beyond speculative tools and toward robust, controllable engines required for demanding, real-world applications.

## References

[1] Joseph Cho, Fachrina Dewi Puspitasari, Sheng Zheng, Jingyao Zheng, Lik-Hang Lee, Tae-Ho Kim, Choong Seon Hong, and Chaoning Zhang. Sora as an agi world model? a complete survey on text-to-video generation. arXiv preprint arXiv:2403.05131, 2024.

[2] Rui Sun, Yumin Zhang, Tejal Shah, Jiahao Sun, Shuoying Zhang, Wenqi Li, Haoran Duan, Bo Wei, and Rajiv Ranjan. From sora what we can see: A survey of text-to-video generation. arXiv preprint arXiv:2405.10674, 2024.

[3] Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu, Guilin Liu, Andrew Tao, Jan Kautz, and Bryan Catanzaro. Video-to-video synthesis. arXiv preprint arXiv:1808.06601, 2018.

[4] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7310–7320, 2024.

[5] Jiale Cheng, Ruiliang Lyu, Xiaotao Gu, Xiao Liu, Jiazheng Xu, Yida Lu, Jiayan Teng, Zhuoyi Yang, Yuxiao Dong, Jie Tang, et al. Vpo: Aligning text-to-video generation models with prompt optimization. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 15636–15645, 2025.

[6] Xiangyu Peng, Zangwei Zheng, Chenhui Shen, Tom Young, Xinying Guo, Binluo Wang, Hang Xu, Hongxin Liu, Mingyan Jiang, Wenjun Li, Yuhui Wang, Anbang Ye, Gang Ren, Qianran Ma, Wanying Liang, Xiang Lian, Xiwen Wu, Yuting Zhong, Zhuangyan Li, Chaoyu Gong, Guojun Lei, Leijun Cheng, Limin Zhang, Minghao Li, Ruijie Zhang, Silan Hu, Shijie Huang, Xiaokang Wang, Yuanheng Zhao, Yuqi Wang, Ziang Wei, and Yang You. Open-sora 2.0: Training a commercial-level video generation model in \$200k. arXiv preprint arXiv:2503.09642, 2025.

[7] Yaosi Hu, Chong Luo, and Zhenzhong Chen. Make it move: controllable image-tovideo generation with text descriptions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18219–18228, 2022.

[8] Yatai Ji, Jiacheng Zhang, Jie Wu, Shilong Zhang, Shoufa Chen, Chongjian Ge, Peize Sun, Weifeng Chen, Wenqi Shao, Xuefeng Xiao, et al. Prompt-a-video: Prompt your video diffusion model via preference-aligned llm. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 18725–18735, 2025.

[9] Steven Bethard. We need to talk about random seeds. arXiv preprint arXiv:2210.13393, 2022.

[10] Dazhong Shen, Guanglu Song, Zeyue Xue, Fu-Yun Wang, and Yu Liu. Rethinking the spatial inconsistency in classifier-free diffusion guidance. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9370– 9379, 2024.

[11] Jaemin Cho, Yushi Hu, Jason Baldridge, Roopal Garg, Peter Anderson, Ranjay Krishna, Mohit Bansal, Jordi Pont-Tuset, and Su Wang. Davidsonian scene graph: Improving reliability in fine-grained evaluation for text-to-image generation. In International conference on learning representations, volume 2024, pages 15625–15645, 2024.

[12] Ian J Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014.

[13] Diederik P Kingma, Max Welling, et al. An introduction to variational autoencoders. Foundations and Trends® in Machine Learning, 12(4):307–392, 2019.

[14] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

[15] Carl Vondrick, Hamed Pirsiavash, and Antonio Torralba. Generating videos with scene dynamics. Advances in neural information processing systems, 29, 2016.

[16] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, et al. Video generation models as world simulators. OpenAI Blog, 1:8, 2024.

[17] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195– 4205, 2023.

[18] Hu Yu, Biao Gong, Hangjie Yuan, DanDan Zheng, Weilong Chai, Jingdong Chen, Kecheng Zheng, and Feng Zhao. Videomar: Autoregressive video generation with continuous tokens. Advances in neural information processing systems, 38:56928– 56958, 2026.

[19] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to textto-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3836–3847, 2023.

[20] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022.

[21] Weifeng Chen, Yatai Ji, Jie Wu, Hefeng Wu, Pan Xie, Jiashi Li, Xin Xia, Xuefeng Xiao, and Liang Lin. Control-a-video: Controllable text-to-video diffusion models with motion prior and reward feedback learning. arXiv preprint arXiv:2305.13840, 2023.

[22] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 22563–22575, 2023.

[23] Yuxin Wen, Neel Jain, John Kirchenbauer, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Hard prompts made easy: Gradient-based discrete optimization for prompt tuning and discovery. Advances in Neural Information Processing Systems, 36:51008– 51025, 2023.

[24] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. In International Conference on Learning Representations, volume 2024, pages 4965–4987, 2024.

[25] Yutong He, Alexander Robey, Naoki Murata, Yiding Jiang, Joshua Williams, George J Pappas, Hamed Hassani, Yuki Mitsufuji, Ruslan Salakhutdinov, and J Zico Kolter. Automated black-box prompt engineering for personalized text-to-image generation. arXiv preprint arXiv:2403.19103, 2(5), 2024.

[26] Yaru Hao, Zewen Chi, Li Dong, and Furu Wei. Optimizing prompts for text-to-image generation. Advances in Neural Information Processing Systems, 36:66923–66939, 2023.

[27] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

[28] Jasper Snoek, Hugo Larochelle, and Ryan P Adams. Practical bayesian optimization of machine learning algorithms. Advances in neural information processing systems, 25, 2012.

[29] Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Ehsan Azarnasab, Faisal Ahmed, Zicheng Liu, Ce Liu, Michael Zeng, and Lijuan Wang. Mm-react: Prompting chatgpt for multimodal reasoning and action. arXiv preprint arXiv:2303.11381, 2023.

[30] Minbin Huang, Yanxin Long, Xinchi Deng, Ruihang Chu, Jiangfeng Xiong, Xiaodan Liang, Hong Cheng, Qinglin Lu, and Wei Liu. Dialoggen: Multi-modal interactive dialogue system for multi-turn text-to-image generation. arXiv preprint arXiv:2403.08857, 2024.

[31] Z Xi, W Chen, X Guo, W He, Y Ding, B Hong, and T Gui. The rise and potential of large language model based agents: A survey: arxiv preprint. arXiv preprint arXiv:2309.07864, 2023.

[32] Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, et al. A survey on large language model based autonomous agents. Frontiers ofComputer Science, 18(6):186345, 2024.

[33] Google. Gemini 2.5 pro (gemini-2.5-pro-preview-06-05): Our most advanced intelligent ai model, 2025. URL https://developers.googleblog.com/en/ gemini-2-5-video-understanding/. Accessed on July 7th, 2025.

[34] Xingyou Song, Qiuyi Zhang, Chansoo Lee, Emily Fertig, Tzu-Kuo Huang, Lior Belenki, Greg Kochanski, Setareh Ariafar, Srinivas Vasudevan, Sagi Perel, et al. The vizier gaussian process bandit algorithm. arXiv preprint arXiv:2408.11527, 2024.

[35] Youwei Liang, Junfeng He, Gang Li, Peizhao Li, Arseniy Klimovskiy, Nicholas Carolan, Jiao Sun, Jordi Pont-Tuset, Sarah Young, Feng Yang, et al. Rich human feedback for text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19401–19411, 2024.

[36] Yilin Wang, Junjie Ke, Hossein Talebi, Joong Gon Yim, Neil Birkbeck, Balu Adsumilli, Peyman Milanfar, and Feng Yang. Rich features for perceptual quality assessment of ugc videos. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 13435–13444, 2021.

[37] Ziqi Huang, Fan Zhang, Xiaojie Xu, Yinan He, Jiashuo Yu, Ziyue Dong, Qianli Ma, Nattapol Chanpaisit, Chenyang Si, Yuming Jiang, Yaohui Wang, Xinyuan Chen, Ying-Cong Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. Vbench++: Comprehensive and versatile benchmark suite for video generative models. arXiv preprint arXiv:2411.13503, 2024.

[38] Bingjie Gao, Xinyu Gao, Xiaoxue Wu, Yujie Zhou, Yu Qiao, Li Niu, Xinyuan Chen, and Yaohui Wang. The devil is in the prompts: Retrieval-augmented prompt optimization for text-to-video generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 3173–3183, 2025.

[39] Zicheng Zhang, Ziheng Jia, Haoning Wu, Chunyi Li, Zijian Chen, Yingjie Zhou, Wei Sun, Xiaohong Liu, Xiongkuo Min, Weisi Lin, et al. Q-bench-video: Benchmark the video quality understanding of lmms. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 3229–3239, 2025.

[40] Gemini Robotics Team, Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Travis Armstrong, Ashwin Balakrishna, Robert Baruch, Maria Bauza, Michiel Blokzijl, et al. Gemini robotics: Bringing ai into the physical world. arXiv preprint arXiv:2503.20020, 2025.

[41] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Nattapol Chanpaisit, Xiaojie Xu, Qianli Ma, Ziyue Dong, Dian Zheng, Hongbo Liu, and Kai Zo. Vbench leaderboard. https://huggingface.co/spaces/Vchitect/VBench\_ Leaderboard, 2023. Accessed: 2025-08-05.

## 7 Appendix

## 7.1 Automated Metric Results

To provide a quantitative counterpart to our human preference studies, we evaluated our generated videos using the comprehensive suite of automated metrics from the V-Bench++ benchmark [37]. The detailed results for each quality dimension, broken down by run configuration and Top-1 selection metric, are presented in Table 3. These automated scores show minimal variation across the different conditions, which we posit is due to a lack of metric sensitivity to the subtle qualitative differences that drive human preference.

## 8 Prompts for Question Generation

Below are the detailed prompts provided to the mLLM for generating the structured DSG and CMQ question graphs used in our framework.

## 8.1 Prompt for Davidsonian Scene Graph (DSG) Questions

Analyze the provided text prompt and optional start/end frame images . Based on this information, generate a hierarchical (tree-like) set of questions following the principles of Davidsonian Scene Graphs (DSG). Identify key scene components (e.g., agents, actions, patients, locations, attributes, manners). For each component, formulate a specific yes/no question where a "yes" answer indicates correct depiction.

Table 3: Automated video quality metrics calculated using V-Bench++ [37] with standard deviation. The scores show minimal variation across different Top-1 selection methods and optimization run counts (10 vs. 100). This suggests a potential lack of sensitivity in current automated metrics to the subtle qualitative differences that drive human preference. This observation is further corroborated by the official V-Bench++ leaderboard, where many topperforming models exhibit closely clustered scores [41].
<table><tr><td>Run Configuration</td><td>Aesthetic Quality</td><td>Background Consistency</td><td>Dynamic Degree</td><td>Imaging Quality</td><td>Motion Smoothness</td><td>Subject Consistency</td></tr><tr><td colspan="7">From 100 Optimization Runs</td></tr><tr><td>100 Runs - Artifact</td><td> $0 . 6 1 6 \pm 0 . 0 8 0$ </td><td> $0 . 9 6 1 \pm 0 . 0 2 8$ </td><td>42.9% True</td><td> $6 9 . 3 5 1 \pm 8 . 4 6 0$ </td><td> $0 . 9 9 1 \pm 0 . 0 1 1$ </td><td> $0 . 9 4 8 \pm 0 . 0 6 4$ </td></tr><tr><td>100 Runs - Avg Score</td><td> $0 . 6 1 8 { \pm } 0 . 0 7 8$ </td><td> $0 . 9 6 5 \pm 0 . 0 2 2$ </td><td>25.5% True</td><td> $7 0 . 6 9 8 \pm 6 . 9 3 9$ </td><td> $0 . 9 9 2 { \scriptstyle \pm 0 . 0 0 9 }$ </td><td> $0 . 9 6 3 \pm 0 . 0 3 9$ </td></tr><tr><td>100 Runs - UVQ</td><td> $0 . 6 1 3 { \scriptstyle \pm 0 . 0 8 0 }$ </td><td> $0 . 9 6 5 \pm 0 . 0 2 3$ </td><td>25.5% True</td><td> $7 0 . 4 6 3 \pm 7 . 3 2 3$ </td><td> $0 . 9 9 2 { \scriptstyle \pm 0 . 0 0 9 }$ </td><td> $0 . 9 6 1 \pm 0 . 0 4 3$ </td></tr><tr><td>100 Runs - VTA</td><td> $0 . 6 1 6 \pm 0 . 0 7 9$ </td><td> $0 . 9 6 1 \pm 0 . 0 2 2$ </td><td>44.9% True</td><td> $7 0 . 1 3 5 \pm 7 . 9 4 3$ </td><td> $0 . 9 9 1 \pm 0 . 0 0 8$ </td><td> $0 . 9 5 7 { \scriptstyle \pm 0 . 0 4 0 }$ </td></tr><tr><td colspan="7">From 10 Optimization Runs</td></tr><tr><td>10 Runs - Artifact</td><td> $0 . 6 1 6 \pm 0 . 0 8 0$ </td><td> $0 . 9 6 6 \pm 0 . 0 2 4$ </td><td>34.3% True</td><td> $6 9 . 3 3 4 \pm 8 . 0 1 5$ </td><td> $0 . 9 9 2 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td> $0 . 9 6 1 \pm 0 . 0 4 1$ </td></tr><tr><td>10 Runs - Avg Score</td><td> $0 . 6 1 8 { \pm } 0 . 0 7 9$ </td><td> $0 . 9 6 5 \pm 0 . 0 2 1$ </td><td>32.3% True</td><td> $7 0 . 3 5 0 \pm 7 . 6 0 7$ </td><td> $0 . 9 9 2 { \scriptstyle \pm 0 . 0 1 1 }$ </td><td> $0 . 9 6 4 \pm 0 . 0 3 9$ </td></tr><tr><td>10 Runs - UVQ</td><td> $0 . 6 1 7 { \scriptstyle \pm 0 . 0 7 9 }$ </td><td> $0 . 9 6 6 \pm 0 . 0 2 1$ </td><td>30.3% True</td><td> $7 0 . 2 0 2 \pm 7 . 6 1 8$ </td><td> $0 . 9 9 1 \pm 0 . 0 1 1$ </td><td> $0 . 9 6 3 \pm 0 . 0 3 9$ </td></tr><tr><td>10 Runs - VTA</td><td> $0 . 6 1 5 { \pm } 0 . 0 8 0$ </td><td> $0 . 9 6 4 \pm 0 . 0 2 1$ </td><td>35.4% True</td><td> $7 0 . 2 9 6 \pm 7 . 4 0 3$ </td><td> $0 . 9 9 1 \pm 0 . 0 0 7$ </td><td> $0 . 9 6 0 { \scriptstyle \pm 0 . 0 3 9 }$ </td></tr><tr><td colspan="7">Baseline Runs</td></tr><tr><td>Initial Generation</td><td> $0 . 6 1 5 { \pm } 0 . 0 8 1$ </td><td> $0 . 9 6 8 \pm 0 . 0 2 5$ </td><td>23.0% True</td><td> $6 8 . 9 0 2 \pm 8 . 8 7 6$ </td><td> $0 . 9 9 4 \pm 0 . 0 0 4$ </td><td> $0 . 9 6 7 \pm 0 . 0 4 2$ </td></tr><tr><td>Prompt Opt. Only</td><td> $0 . 6 1 5 { \pm } 0 . 0 8 0$ </td><td> $0 . 9 6 8 \pm 0 . 0 2 3$ </td><td>27.0% True</td><td> $6 8 . 8 9 5 \pm 8 . 9 2 5$ </td><td> $0 . 9 9 4 \pm 0 . 0 0 5$ </td><td> $0 . 9 6 7 \pm 0 . 0 4 0$ </td></tr></table>

Organize these components and their questions into a JSON object.   
The JSON structure should have a main key as "scene\_graph",   
which is a list of root elements. Each element should have at   
least "id" (a unique identifier for the question node), "   
element\_type" (e.g., "agent", "action", "object", "location", "   
attribute"), "description" (a brief description of the element),   
"question" (the yes/no question), and "children" (a list of sub   
-elements, which can be empty).   
Example of an element structure:   
{   
"id": "agent\_red\_bird\_1",   
"element\_type": "agent",   
"description": "A red bird",   
"question": "Is there a red bird present?",   
"children": [   
{   
"id": "action\_bird\_flying\_1",   
"element\_type": "action",   
"description": "The red bird is flying",   
"question": "Is the red bird flying?",   
"children": []   
}   
]   
}   
Ensure the entire output is a single valid JSON object. Ensure that   
there are no more than 20 total questions.   
Initial Prompt: "[User's Initial Prompt]"

## 8.2 Prompt for Common Mistake Questions (CMQ)

Analyze the provided text prompt and optional start/end frame images   
. Based on this information, generate a hierarchical (tree-like)   
set of questions targeting potential common video generation   
mistakes that might occur when trying to generate a video for   
this prompt.   
Consider only the issues such as:   
- Unnatural Transitions   
- Anatomical Inconsistencies (missing/extra limbs/digits,   
distorted parts)   
- Environmental Inconsistencies (objects appearing/disappearing)   
- Illogical Changes within shots   
- Static Elements that should move   
- Extra characters and objects   
- Human characters speaking (they should never speak unless   
specified explicitly)   
- Inconsistent Detail/Texture   
Organize these components and their questions into a JSON object.   
The JSON structure should have a main key as "   
common\_mistakes\_graph", which is a list of root elements. Each   
element should have at least "id" (a unique identifier for the   
question node), "element\_type" (e.g., "mistake\_category", "   
specific\_mistake"), "description" (a brief description of the   
mistake/category), "question" (the yes/no question where "yes"   
indicates no mistake), and "children" (a list of sub-elements,   
which can be empty).   
Example of an element structure:   
{   
"id": "mistake\_cat\_transitions\_1",   
"element\_type": "mistake\_category",   
"description": "Unnatural Transitions",   
"question": "Are all transitions between shots smooth and   
logical?",   
"children": [   
{   
"id": "specific\_mistake\_jump\_cut\_1",   
"element\_type": "specific\_mistake",   
"description": "Potential jump cut between scene A and B   
",   
"question": "Is the transition from scene A to B free of   
jarring jump cuts?",   
"children": []   
}   
]   
}   
Ensure the entire output is a single, complete, and valid JSON   
object. Make sure all brackets and braces are correctly matched

and closed. Ensure that there are no more than 20 questions in the entire tree. All questions should be framed as a yes or no question such that "yes" is the correct answer (i.e., the mistake is avoided) and a "no" answer indicates a common mistake might be present.

Initial Prompt: "[User's Initial Prompt]"