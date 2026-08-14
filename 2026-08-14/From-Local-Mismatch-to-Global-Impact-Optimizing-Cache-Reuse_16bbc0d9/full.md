# From Local Mismatch to Global Impact: Optimizing Cache Reuse Policy for Efficient Diffusion

Xichen Ye<sup>1</sup>, Yifan Wu<sup>2</sup>, Zhikang Xie<sup>1</sup> Xiangyu Yue<sup>2</sup>, Cheng Jin<sup>1</sup>, Weizhong Zhang<sup>3,∗</sup>

<sup>1</sup>College of Computer Science and Artificial Intelligence, Fudan University <sup>2</sup>Faculty of Engineering, The Chinese University of Hong Kong <sup>3</sup>School of Data Science, Fudan University

## Abstract

Diffusion models have achieved dominant performance in visual generation but suffer from substantial inference overhead. While cache-based acceleration has emerged as a promising solution, existing policies rely on local similarity heuristics, which we identify as being significantly misaligned with final generation quality. This discrepancy stems from the non-uniform propagation and accumulation of errors along the denoising trajectory. To address this, we propose Global-Impact Cache (GCache). We first establish a rigorous theoretical characterization of the error propagation upper bound. Recognizing that this bound can be overly conservative for complex, highly non-convex diffusion models, we further reparameterize the propagation exponent with a Bernstein form and reformulate cache policy search as a bilevel optimization problem. In detail, GCache identifies an optimal reuse policy in the inner objective while aligning the error-weighting function with generation quality loss in the outer objective. This framework effectively reconciles theoretical rigor with empirical performance, learning to prioritize computation where it most impacts visual fidelity. Extensive experiments demonstrate that GCache consistently outperforms prior caching strategies on both video and image generation. Notably, on the state-of-the-art Wan2.1 video diffusion model, GCache maintains a 2.17× speedup while significantly enhancing generation quality, reducing LPIPS from 0.1095 to 0.0316.

## 1 Introduction

In recent years, diffusion models [1–4] have emerged as the dominant paradigm in visual generation, delivering high-fidelity and diverse outputs across various modalities, including images [5, 6] and videos [7, 8]. Despite these successes, diffusion models remain burdened by substantial inference overhead, stemming from the iterative nature of solving the underlying Ordinary Differential Equations (ODEs), which necessitates numerous model evaluations for a single sample. To mitigate this limitation, various acceleration mechanisms have been explored from multiple directions, including model-centric compression [9, 10], advanced sampling solvers [11–13], and, more recently, cache-based mechanisms [14–16].

Among current acceleration strategies, cache-based methods provide a practical approach to speeding up diffusion inference. Unlike model-centric techniques, caching avoids intensive retraining or distillation of model parameters and remains orthogonal to advanced sampling solvers, making it a highly lightweight solution for denoising acceleration. The primary objective is to establish an inference-time policy that identifies redundant intermediate residuals across adjacent timesteps to avoid repeated computations. While initial approaches [14, 17, 18] relied on uniform reuse schedules, they lacked the flexibility to adapt to varying residual dynamics across different timesteps. Consequently, recent works [15, 16] have shifted toward non-uniform strategies. These methods employ local similarity metrics to quantify the mismatch between target and cached residuals, triggering reuse only when the discrepancy is sufficiently small.

Despite these advancements, existing approaches remain limited by their reliance on local similarity metrics, such as the relative $\ell _ { 1 }$ distance. However, empirical evidence suggests that such locally-based strategies fail to accurately capture how individual cache reuse decisions affect final generation quality. As illustrated in Figure 1, large local discrepancies may correspond to only minor perceptual degradation: pronounced relative $\ell _ { 1 }$ spikes occur around step $\hat { 1 } 9 ^ { 2 }$ and increase sharply in the final denoising stages, yet result in only marginal increases in LPIPS. This mismatch reveals that local discrepancy alone is an unreliable proxy for global generation impact, indicating that effective cache reuse policies must account for both local reuse errors and their cumulative impact on final generation quality.

![](images/0de41aa789943af0eef9c3aca7a2b76427993c1926ce13510c21320f38fe8590.jpg)  
Figure 1: Comparison between local mismatch and global impact (lower is better). We conduct a series of independent experiments in which a cached residual is reused at exactly one specific timestep. For each timestep, the blue marker (Rel-L1) measures the local discrepancy between the ground-truth residual and the cached residual from the preceding step, while the red marker (LPIPS) reflects the resulting impact on final generation quality.

To this end, we first establish a theoretical characterization of the error propagation dynamics in cache-reused diffusion trajectories. This analysis provides a principled foundation by relating local discrepancies to a global cumulative error bound, offering a rigorous heuristic for policy optimization. While this analytical bound serves as a robust guide, its worst-case nature inherently introduces a pessimistic bias, as it does not fully account for the high non-convexity and intrinsic error-resilience of diffusion models. To bridge this gap and unlock the full potential of our theoretical framework, we propose Global-Impact Cache (GCache). In detail, we parameterize the propagation exponent in the theoretical bound as a Bernstein form and formulate the policy search as a bilevel optimization problem, where the inner objective identifies the optimal reuse policy that minimizes the current error bound for a given set of parameters, while the outer objective optimizes the propagation parameters by minimizing empirical generation loss. In this way, GCache learns an error-weighting function that better reflects empirical error propagation, leading to more informed cache reuse decisions and improved generation quality. Extensive experiments across multiple image and video generation models demonstrate that our method, GCache, consistently outperforms prior caching strategies. Overall, we summarize our contributions as follows:

• We identify a fundamental misalignment between local cache reuse errors and global generation quality, and provide a theoretical analysis that characterizes how cache reuse errors propagate during diffusion sampling, establishing an analytical upper bound on their global impact.

• We demonstrate that there is an estimative gap between this theoretical bound and empirical error behavior, observing that the bound’s worst-case assumptions can be conservatively biased in the context of highly non-convex diffusion dynamics.

• Motivated by this, we propose GCache, which reparameterizes the error propagation exponent with a Bernstein form and formulates cache reuse policy search as a bilevel optimization problem to better align theoretical estimates with empirical error behavior.

• We conduct extensive experiments on both image and video generation tasks, demonstrating that GCache consistently achieves superior speed–quality trade-offs compared to existing cache-based acceleration methods.

## 2 Preliminaries

In this section, we briefly review the background on diffusion models and cache reuse to establish notation and context. Further discussions of related work are provided in Appendix C.

## 2.1 Diffusion Models

Diffusion models [1, 2, 4] synthesize samples by gradually perturbing real data $\pmb { x } \sim p _ { \mathrm { d a t a } } ( \pmb { x } )$ into prior noise ${ \pmb n } \sim p _ { \mathrm { p r i o r } } ( { \pmb n } )$ (e.g., a standard Gaussian distribution ${ \mathcal { N } } ( \mathbf { 0 } , \mathbf { \bar { I } } ) )$ , subsequently learning to invert this process for sample generation.

Recent works [19] frequently adopt the Flow Matching paradigm [4]. Specifically, given a data-noise pair $( { \pmb x } , { \pmb n } )$ , a flow path is constructed as ${ \pmb x } _ { t } = ( 1 - t ) { \pmb x } + t { \pmb n }$ for $t \in [ 0 , 1 ]$ , which induces the conditional velocity field:

$$
{ \pmb v } _ { t } ( { \pmb x } _ { t } \mid { \pmb x } ) = \frac { \mathrm { d } } { \mathrm { d } t } \left[ ( 1 - t ) { \pmb x } + t { \pmb n } \right] = { \pmb n } - { \pmb x } .\tag{1}
$$

Since a specific $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ can result from various $( { \pmb x } , { \pmb n } )$ pairs, Flow Matching targets the marginal velocity field:

$$
{ \pmb v } ( { \pmb x } _ { t } , t ) : = \mathbb { E } _ { p _ { t } ( { \pmb x } | { \pmb x } _ { t } ) } \left[ { \pmb v } _ { t } ( { \pmb x } _ { t } \mid { \pmb x } ) \right] .\tag{2}
$$

A neural network ${ \pmb v } _ { \theta }$ is trained to approximate this marginal field by minimizing the conditional Flow Matching loss:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { C F M } } ( \theta ) = \mathbb { E } _ { t , \mathbf { x } , n } \left\| v _ { \theta } ( \mathbf { x } _ { t } , t ) - v _ { t } ( \mathbf { x } _ { t } \mid \mathbf { x } ) \right\| _ { 2 } ^ { 2 } . } \end{array}\tag{3}
$$

This objective is equivalent to minimizing the Flow Matching loss $\mathcal { L } _ { \mathrm { F M } } ( \theta ) = \mathbb { E } _ { t , p _ { t } ( \mathbf { x } _ { t } ) } | | \pmb { v } _ { \theta } ( \mathbf { x } _ { t } , t ) -$ ${ \mathbf { \boldsymbol { v } } } ( { \boldsymbol { x } } _ { t } , t ) \lVert _ { 2 } ^ { 2 }$ , effectively fitting the model to the true marginal velocity.

During inference, samples are generated by solving the corresponding ordinary differential equation (ODE):

$$
\frac { \mathrm { d } } { \mathrm { d } t } \pmb { x } _ { t } = \pmb { v } ( \pmb { x } _ { t } , t ) ,\tag{4}
$$

initialized at ${ \pmb x } _ { 1 } \sim p _ { \mathrm { p r i o r } }$ and integrated backward to $t = 0$ . The final sample is given by ${ \pmb x } _ { 0 } = { \pmb x } _ { 1 } -$ $\begin{array} { r } { \int _ { 0 } ^ { 1 } \pmb { v } ( \pmb { x } _ { \tau } , \tau ) \mathrm { d } \tau } \end{array}$ . In practice, this integral is approximated using numerical ODE solvers. For instance, the Euler method discretizes the trajectory over N timesteps $1 = t _ { N } > t _ { N - 1 } > \cdot \cdot \cdot > t _ { 2 } > t _ { 1 } = 0$ computing each step as:

$$
\pmb { x } _ { t _ { i } } = \pmb { x } _ { t _ { i + 1 } } - ( t _ { i } - t _ { i + 1 } ) \pmb { v } ( \pmb { x } _ { t _ { i + 1 } } , t _ { i + 1 } ) .\tag{5}
$$

## 2.2 Cache Reuse

The primary inference bottleneck in diffusion models arises from the repetitive model evaluations necessitated by the ODE solver. Cache reuse has emerged as a promising strategy to mitigate this overhead by leveraging the temporal redundancy of intermediate representations across sampling steps. A prevalent approach involves reusing residual mappings across specific model layers [14].

Formally, consider a feed-forward neural network (e.g., a Transformer [20]) decomposed as:

$$
v ( x _ { t } , t ) = ( f _ { \mathrm { o u t } } \circ f _ { \mathrm { m i d } } \circ f _ { \mathrm { i n } } ) ( x _ { t } , t ) ,\tag{6}
$$

where $f _ { \mathrm { m i d } }$ represents the intermediate blocks (e.g., core transformer layers), while $f _ { \mathrm { i n } }$ and $f _ { \mathrm { o u t } }$ denote the layers responsible for input embedding and output projection $( \mathrm { e . g . }$ , patchify and unpatchify), respectively. Let $\pmb { h } _ { t } : = f _ { \mathrm { i n } } ( \pmb { x } _ { t } )$ and $z _ { t } : = f _ { \mathrm { m i d } } ( h _ { t } )$ , the residual is then defined as

$$
\delta _ { t } : = z _ { t } - h _ { t } .\tag{7}
$$

During the sampling process, a cache reuse policy $\pmb { m } \in \{ 0 , 1 \} ^ { N }$ governs the use of a stored residual $\pmb { \delta } ^ { c }$ . At each timestep $t _ { n } ,$ the policy determines whether to approximate the state by reusing the cached residual $( m _ { t _ { i } } = 0 )$

$$
\begin{array} { r } { \boldsymbol { z } _ { t _ { i } } ^ { c } = \delta ^ { c } + \boldsymbol { h } _ { t _ { i } } , } \end{array}\tag{8}
$$

or to perform a full computation and update the cache $( m _ { t _ { i } } = 1 )$

$$
\delta ^ { c } \gets \delta _ { t _ { i } } .\tag{9}
$$

The objective of a cache reuse policy is to minimize the degradation of generation quality subject to a specific computational budget, which is typically defined by the total number of cache reuses, $\lVert \mathbf { m } \rVert _ { 0 }$

To develop effective caching policies, recent studies [15, 16] typically employ a local error metric, specifically the relative $\ell _ { 1 }$ distance, to determine whether to reuse the previously cached residual $\pmb { \delta } ^ { c }$ at timestep $t _ { i } { \mathrm { : } }$

$$
d ( \pmb { \delta } ^ { c } , \pmb { \delta } _ { t _ { i } } ) = \frac { \| \pmb { \delta } ^ { c } - \pmb { \delta } _ { t _ { i } } \| _ { 1 } } { \| \pmb { \delta } ^ { c } \| _ { 1 } } .\tag{10}
$$

However, as we demonstrate below, this local-only perspective fails to account for the error propagation inherent in the denoising process, ultimately resulting in sub-optimal caching decisions.

## 3 Method

In this section, we begin with preliminary experiments that reveal the error propagation behavior in the denoising process (Section 3.1). We then develop a theoretical upper bound that characterizes how local cache reuse errors propagate across timesteps (Section 3.2). Since directly optimizing cache reuse policies with this bound proves insufficient in practice, we further refine it via a bilevel optimization framework and propose Global-Impact Cache (GCache) with an efficient solution strategy (Section 3.3).

## 3.1 Motivation

To motivate our study, we first conduct controlled perturbation experiments on video generation. Specifically, we inject random noise of a fixed magnitude into intermediate feature maps at different denoising timesteps and track the log-scaled deviation $\| \pmb { x } _ { t } - \hat { \pmb { x } } _ { t } \|$ <sub>1</sub> over subsequent denoising steps, where $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ and xˆ denote the unperturbed and perturbed states. As illustrated in Figure 2, Perturbations introduced earlier are progressively amplified and lead to substantially larger final deviations than those injected later.

![](images/0d2b16c2a83274e71cd466f9da853943c1217398c4f24c897fbab3d7bda1159c.jpg)  
Figure 2: Characterization of error propagation across the denoising trajectory.

These findings reveal a cumulative and time-dependent error propagation effect, suggesting that an effective cache reuse policy should account for not only the local cache reuse error but also the error propagation inherent in the denoising process. Building on this observation, the following subsection establishes a formal relationship between local reuse discrepancies and their cumulative propagation.

## 3.2 Error Propagation of Cache Reuse

To rigorously quantify how local cache reuse errors accumulate across the denoising trajectory, we first establish a formal theoretical framework. Building on the empirical insights from Section 3.1, we aim to derive an upper bound for the final generation error that accounts for the time-dependent nature of error propagation. We first introduce the following regularity conditions, which are standard in the analysis of diffusion-based generative models [21].

Assumption 3.1 (Regularity and Approximation Conditions). For all denoising timesteps $t \in [ t _ { 1 } , t _ { N } ]$ we assume the following conditions hold:

(i) Bounded Approximation Error: The empirical velocity field ${ \pmb v } _ { \theta }$ approximates the ground-truth marginal velocity v with a uniform error bound $\eta \geq 0 , \mathrm { i . e . , } \| v ( x , \bar { t } ) - v _ { \theta } ( x , t ) \| _ { 1 } \leq \eta$ for all x and t.

(ii) Lipschitz Continuity: The output projection $f _ { \mathrm { o u t } }$ is $L _ { \mathrm { o u t } ^ { - 1 } }$ Lipschitz continuous, and the learned velocity field ${ \pmb v } _ { \pmb { \theta } }$ is $L _ { t }$ -Lipschitz continuous with respect to its first argument, $\mathrm { i . e . , } \parallel v _ { \theta } ( x , t ) -$ ${ v _ { \theta } } ( { y } , i ) \| _ { 1 } \le { L _ { t } } \| { x } - { y } \| _ { 1 }$ for all $x , y , t ,$ , where $L _ { t } \leq L$ for all t.

(iii) Bounded Velocity Dynamics: The total time derivative of the ground-truth velocity field along any trajectory $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ is bounded by $M , \mathrm { i . e . , \left\| \frac { d } { d t } \pmb { v } ( \pmb { x } _ { t } , t ) \right\| _ { 1 } \le M . }$

We begin by analyzing a scenario involving a single reuse event at timestep $t _ { i + 1 }$ . Specifically, we evaluate the final error relative to the ground-truth ODE trajectory under Euler discretization.

Theorem 3.2 (Global Error Bound under Cache Reuse). Under Assumption 3.1, let $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ be the groundtruth ODE state at time $t ,$ and let $\hat { \pmb x } _ { t } ^ { c }$ be the state generated by the Euler solver using the learned velocity $v _ { \theta } ,$ , incorporating a single cache reuse event. Specifically, suppose that at step $t _ { i + 1 } \to t _ { i } ,$ we reuse a cached residual $\delta _ { t _ { i + 1 } } ^ { c }$ with error $\epsilon _ { t _ { i + 1 } } ^ { c } = \delta _ { t _ { i + 1 } } - \delta _ { t _ { i + 1 } } ^ { c }$ . Given a sequence of timesteps $t _ { N } , t _ { N - 1 } , \ldots , t _ { 1 }$ with uniform step size $h ,$ the global error at thefinal timestep $t _ { 1 }$ is bounded by:

$$
\left\| \pmb { x } _ { t _ { 1 } } - \hat { \pmb { x } } _ { t _ { 1 } } ^ { c } \right\| _ { 1 } \leq \underbrace { h L _ { o u t } \left\| \epsilon _ { t _ { i + 1 } } ^ { c } \right\| _ { 1 } e ^ { ( i - 1 ) h L } } _ { C a c h e R e u s e E r r o r } + \underbrace { \left( \frac { \eta } { L } + \frac { h M } { 2 L } \right) \left( e ^ { ( N - 1 ) h L } - 1 \right) } _ { A p p r o x i m a t i o n \notin D i s c r e t i z a t i o n E r r o r } .\tag{11}
$$

Theorem 3.2 demonstrates that the final error can be decomposed into three distinct components: (i) the propagated error originating from cache reuse, (ii) the network approximation error stemming from the learned velocity, and (iii) the discretization error inherent to the ODE solver.

Notably, the cache reuse term is scaled by an exponential factor, indicating that local errors introduced by cache reuse are amplified by the system dynamics as the trajectory evolves. This suggests that the impact of cache reuse should not be judged solely by the magnitude of the local error $\pmb { \epsilon } _ { t _ { i + 1 } ^ { c } }$ , but rather by its “positional” impact: errors introduced earlier in the denoising process (larger i) may lead to significantly larger deviations in the final generated sample. This theoretical insight corroborates the empirical observations presented in Figure 2.

In practice, we are primarily concerned with the deviation of the cached trajectory from the baseline discretization rather than from the ideal ODE solution. Our objective is to enhance the efficiency of diffusion sampling while preserving its generative fidelity as faithfully as possible. To this end, we establish error propagation bounds for single- and multi-step cache reuse in the following theorems.

Theorem 3.3 (Error Propagation Bound under Single-Step Cache Reuse). Under Assumption 3.1, let xˆ be the state generated by the Euler solver using the learned velocity ${ \pmb v } _ { { \pmb \theta } } ,$ , and let $\hat { \pmb x } _ { t } ^ { c }$ be the state incorporating a single cache reuse event at step $t _ { i + 1 } \to t _ { i }$ . Suppose the reuse ofa cached residual $\delta _ { t _ { i + \cdot } } ^ { c }$ introduces an error $\epsilon _ { t _ { i + 1 } } ^ { c } = \delta _ { t _ { i + 1 } } - \delta _ { t _ { i + 1 } } ^ { c }$ . Given a sequence of timesteps $t _ { N } , t _ { N - 1 } , \ldots , t _ { 1 }$ with step sizes $h _ { t _ { n + 1 } } = t _ { n + 1 } - t _ { n } ;$ , the cumulative error at thefinal timestep $t _ { 1 }$ is bounded by:

$$
\left\| \hat { \pmb x } _ { t _ { 1 } } - \hat { \pmb x } _ { t _ { 1 } } ^ { c } \right\| _ { 1 } \leq \left\| \epsilon _ { t _ { i + 1 } } ^ { c } \right\| _ { 1 } e ^ { w _ { t _ { i + 1 } } } ,\tag{12}
$$

where the propagation exponent $w _ { t _ { i + 1 } }$ is defined as:

$$
w _ { t _ { i + 1 } } = \ln \left( h _ { t _ { i + 1 } } L _ { o u t } \right) + \sum _ { n = 1 } ^ { i - 1 } h _ { t _ { n + 1 } } L _ { t _ { n + 1 } } .\tag{13}
$$

Theorem 3.4 (Error Propagation Bound under Multi-Step Cache Reuse). Under Assumption 3.1, let $\hat { \mathbf { x } } _ { t }$ be the state generated by the Euler solver using the learned velocity $v _ { \theta } ,$ , and let $\hat { \mathbf { x } } _ { t } ^ { c }$ be the state incorporating multiple cache reuse events at every step $t _ { i + 1 }  t _ { i } f o r i \in \{ N - 1 , \ldots , 1 \}$ . Suppose each reuse of a cached residual introduces a local error $\epsilon _ { t _ { i + 1 } } ^ { c } = \delta _ { t _ { i + 1 } } - \delta _ { t _ { i + 1 } } ^ { c }$ . Given a sequence of timesteps $t _ { N } , t _ { N - 1 } , \ldots , t _ { 1 }$ with step sizes $h _ { t _ { i + 1 } } = t _ { i + 1 } - t _ { i } ,$ , the cumulative error at the final timestep $t _ { 1 }$ is bounded by:

$$
\left\| \hat { \pmb x } _ { t _ { 1 } } - \hat { \pmb x } _ { t _ { 1 } } ^ { c } \right\| _ { 1 } \leq \sum _ { i = 1 } ^ { N - 1 } \left\| \pmb \epsilon _ { t _ { i + 1 } } ^ { c } \right\| _ { 1 } e ^ { w _ { t _ { i + 1 } } } .\tag{14}
$$

where the propagation exponent $w _ { t _ { n + 1 } }$ is defined as:

$$
w _ { t _ { i + 1 } } = \ln \left( h _ { t _ { i + 1 } } L _ { o u t } \right) + \sum _ { j = 1 } ^ { i - 1 } h _ { t _ { j + 1 } } L _ { t _ { j + 1 } } .\tag{15}
$$

Theorem 3.3 and Theorem 3.4 provide an analytical characterization of how a cached trajectory deviates from the baseline discretization. Crucially, these bounds suggest that an optimal cache reuse policy m can be identified by minimizing the upper bound of the total propagation error:

$$
m ^ { \star } ( s ) \in \mathop { \arg \operatorname* { m i n } } _ { m \in \mathcal { C } } \sum _ { i = 1 } ^ { N - 1 } \left\| \epsilon _ { t _ { i + 1 } } ^ { c } \right\| _ { 1 } e ^ { w _ { t _ { i + 1 } } } ,\tag{16}
$$

![](images/29f2413e578625da05568066f560fb5e231c084747fe2db1ee3b89cbf965b8c7.jpg)  
(a) Comparison of different cache policies.

![](images/cff56d797a6b96cde7bf48e1f9b5ba73897b1ae604cad86b1a70cf97baef2bb8.jpg)  
(b) Comparison of error propagation estimations  
Figure 3: Analysis of cache policies and error estimation. (a) Policy comparison. Reconstruction error across the denoising process. The theoretical policy (orange) is overly pessimistic in early stages, leading to late-stage error spikes; GCache (green) balances computation to achieve the lowest final error. (b) Estimation fidelity. Comparison of error propagation profiles. The analytical bound (red) provides a conservative upper limit, while GCache’s optimized weighting (green) aligns tightly with the empirical ground truth (blue).

where ${ \mathcal { C } } = \{ m : \| m \| _ { 0 } = K \}$ denotes the feasible set of policies under a predefined budget K, representing the number of full-computation steps allowed throughout the denoising process.

Effectiveness of the Analytical Upper Bound. By directly minimizing the analytical upper bound in Equation (16), we derive a cache reuse policy that accounts for the cumulative impact of errors across the entire trajectory. As illustrated in Figure 3a, the policy optimized via this theoretical bound (Orange) consistently achieves a lower reconstruction error $\| \pmb { x } _ { t _ { 1 } } - \hat { \pmb { x } } _ { t _ { 1 } } \| _ { 1 }$ under a fixed computation budget $( K = 1 0 )$ at the final timestep compared to the state-of-the-art ERTACache [16] (Blue). This superiority stems from our bound’s ability to capture the long-term dependency of errors, whereas prior methods often rely on local heuristics that fail to account for the exponential amplification of early-stage perturbations.

## 3.3 Optimized Cache Reuse Policy

From Theoretical Bounds to Empirical Alignment. While the policy derived from Equation (16) effectively minimizes the cumulative error, it exhibits a distinct behavior of prioritizing full computation in early denoising stages and clustering cache reuse toward the end. To understand the underlying mechanics, we analyze the discrepancy between our analytical bound and the actual empirical error. As illustrated in Figure 3b, the analytical bound provides a conservative overestimation of the propagated error, particularly during the initial steps. This gap stems from the fact that the theoretical bound assumes a worst-case error growth, which does not fully account for the intrinsic error-resilience and non-linear dynamics of diffusion models. Motivated by this observation, we move beyond a conservative bound and introduce a bilevel optimization formulation. This approach adaptively “tightens” the error estimation by learning a weighting function that aligns our theoretical framework with empirical generation quality, ultimately yielding a more refined and effective cache reuse policy.

Bilevel Optimization Formulation. To address the gap of the analytical bound discussed above, we propose Global-Impact Cache (GCache). The key idea is to move beyond a conservative upper bound by optimizing a parameterized error-weighting function that better aligns with empirical results. Specifically, we parameterize the propagation exponent $w _ { t }$ using the d-th degree Bernstein polynomials as follows:

$$
w ( t ; s ) = \sum _ { \nu = 0 } ^ { d } s _ { \nu + 1 } { \binom { n } { \nu } } t ^ { \nu } ( 1 - t ) ^ { d - \nu } ,\tag{17}
$$

where $\pmb { \mathscr { s } } \in [ s _ { \operatorname* { m i n } } , s _ { \operatorname* { m a x } } ] ^ { d + 1 }$ denotes a vector of learnable parameters (coefficients). With this parameterization, the search for an optimal cache reuse policy can be formulated as a bilevel optimization

problem:

$$
\pmb { \mathscr { s } } = \underset { \pmb { \mathscr { s } } \in [ s _ { \operatorname* { m i n } } , s _ { \operatorname* { m a x } } ] ^ { d + 1 } } { \arg \operatorname* { m i n } } \mathcal { L } ( \pmb { m } ^ { \star } ( \pmb { s } ) ) ,\tag{18}
$$

$$
\mathrm { s . t . } \quad m ^ { \star } ( s ) = \underset { m \in \mathcal { C } } { \arg \operatorname* { m i n } } \sum _ { i = 1 } ^ { N - 1 } \left\| \epsilon _ { t _ { i + 1 } } ^ { c } \right\| _ { 1 } e ^ { w ( t _ { i + 1 } ; s ) } .\tag{19}
$$

Here, ${ \mathcal { C } } = \{ m : \| m \| _ { 0 } = K \}$ denotes the feasible set for the policy m under a predefined budget constraint K.

Intuitively, the inner objective (19) seeks an optimal cache reuse policy $m ^ { \star }$ that minimizes the current upper bound of the propagated error induced by the given polynomial parameters s; while the outer objective (18) optimizes the parameters s to minimize a generation quality metric ${ \mathcal { L } } \left( \mathbf { e . g . } \right.$ LPIPS [22]), effectively tuning the error-weighting function to better reflect empirical generation quality. However, standard first-order methods (e.g., SGD) are inapplicable to this problem, as neither objective is differentiable with respect to s or m. We therefore develop tailored strategies for both levels.

Inner Optimization via Dynamic Programming. The inner problem (19) can be cast as a constrained shortest path problem on a directed acyclic graph. We define a cost matrix $E \in \mathbb { R } ^ { N \times N }$ where each entry

$$
E _ { i , j } = \left\| \pmb { \delta } _ { t _ { i } } - \pmb { \delta } _ { t _ { j } } \right\| _ { 1 } e ^ { w ( t _ { j } ; \pmb { s } ) } ,\tag{20}
$$

quantifies the propagated error of reusing the residual from $t _ { i }$ at step $t _ { j }$ . Finding the optimal policy $m ^ { \star }$ is equivalent to finding a path from $t _ { N }$ to $t _ { 1 }$ that minimizes cumulative cost with exactly K refresh nodes. This admits an efficient Dynamic Programming (DP) solution with $O ( K N ^ { 2 } )$ complexity. Given $N \leq 1 0 0$ in modern schedulers, this overhead is negligible.

Outer Optimization via Bayesian Optimization. Since evaluating the outer objective $\mathcal { L } ( m ^ { \star } ( s ) )$ requires a full inference pass over the training set, it is a high-cost black-box function. We employ Bayesian Optimization (BO) to efficiently search the parameter space of s. BO models the objective $f _ { \mathrm { e v a l } } ( \pmb { s } ) = \mathcal { L } ( \pmb { m } ^ { \star } ( \pmb { s } ) )$ using a Gaussian Process (GP) surrogate. Let $\mathcal { D } _ { n } = \{ ( s _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ be the history of n evaluations. The GP provides a posterior predictive distribution $p ( \boldsymbol { y } _ { . } | \ s , S , Y ) \sim$ $\mathcal { N } ( \mu ( \dot { \boldsymbol { \mathscr { s } } } ) , \sigma ^ { 2 } ( \boldsymbol { \mathscr { s } } ) )$ , where the mean $\mu ( s )$ estimates performance and the variance $\scriptstyle { \dot { \sigma } } ^ { 2 } ( s )$ quantifies uncertainty:

$$
\mu ( s ) = k ( s , S ) \left( K + \sigma _ { \epsilon } ^ { 2 } { \bf I } \right) ^ { - 1 } Y ,\tag{21}
$$

$$
\sigma ^ { 2 } ( s ) = k ( { s } , { s } ) - k ( { s } , { S } ) \left( K + \sigma _ { \epsilon } ^ { 2 } { \bf I } \right) ^ { - 1 } k ( { s } , { S } ) ^ { \top } .\tag{22}
$$

Here, $\pmb { S } = [ \pmb { s } _ { 1 } , \pmb { s } _ { 2 } , \pmb { \cdot } \pmb { \cdot } \pmb { \cdot } , \pmb { s } _ { n } ] ^ { \top }$ are reviously sampled parameters and $\boldsymbol { Y } = [ y _ { 1 } , y _ { 2 } , \cdots , y _ { n } ] ^ { \top }$ are their corresponding empirical evaluations. We use the Lower Confidence Bound (LCB) [23] as the acquisition function:

$$
A _ { \mathrm { L C B } } ( \pmb { s } ; \mathcal { D } _ { n } ) = \mu ( \pmb { s } ) - \kappa _ { n } \sigma ( \pmb { s } ) ,\tag{23}
$$

where $\kappa _ { n }$ balances exploitation and exploration. This framework allows GCache to iteratively discover the optimal propagation exponent that best aligns the theoretical bound with empirical performance.

Detailed implementation specifics for Dynamic Programming and Bayesian Optimization are deferred to Appendix A.1 and Appendix A.2, respectively. Additionally, a thorough discussion regarding the selection of our optimization objective is provided in Appendix A.3.

## 4 Experiments

## 4.1 Experimental Settings

We conduct experiments on four representative DiT-based diffusion models to validate the generality and effectiveness of our method across both video and image generation. Specifically, we evaluate three video diffusion backbones, Open-Sora 1.2 [24], CogVideoX [25], and Wan 2.1 [26], as well as one strong text-to-image model, Flux-dev 1.0 [27]. Unless otherwise specified, all experiments are conducted on a single NVIDIA A800 80GB GPU, and results are reported by averaging over five random seeds. Additional experimental details are provided in Appendix D, respectively.

Table 1: Quantitative evaluation of efficiency and visual quality for video generation across different methods on three leading text-to-video diffusion models. Bold denotes the best performance under similar acceleration ratios. ↑ indicates higher is better, and ↓ indicates lower is better.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="2">Efficiency</td><td colspan="4">Visual Quality</td></tr><tr><td>Speedup↑</td><td>Latency (s)↓</td><td>VBench↑</td><td>LPIPS↓</td><td>SSIM↑</td><td>PSNR↑</td></tr><tr><td rowspan="10">Open-Sora 1.2 (51 frames, 480P)</td><td>Open-Sora 1.2 (T = 30)</td><td>1x</td><td>44.56</td><td>79.22%</td><td></td><td></td><td></td></tr><tr><td>∆-DiT [14]</td><td>1.03×</td><td></td><td>78.21%</td><td>0.5692</td><td>0.4811</td><td>11.91</td></tr><tr><td>T-GATE [28]</td><td>1.19×</td><td></td><td>77.61%</td><td>0.3495</td><td>0.6760</td><td>15.50</td></tr><tr><td>PAB-slow [18]</td><td>1.33×</td><td>33.40</td><td>77.64%</td><td>0.1471</td><td>0.8405</td><td>24.50</td></tr><tr><td>PAB-fast [18]</td><td>1.40×</td><td>31.85</td><td>76.95%</td><td>0.1743</td><td>0.8220</td><td>23.58</td></tr><tr><td>TeaCache-slow [15]</td><td>1.55×</td><td>28.78</td><td>79.28%</td><td>0.1316</td><td>0.8415</td><td>23.62</td></tr><tr><td>TeaCache-fast [15]</td><td>2.25×</td><td>19.84</td><td>78.48%</td><td>0.2511</td><td>0.7477</td><td>19.10</td></tr><tr><td>ERTACache-slow [16]</td><td>1.55×</td><td>28.75</td><td>79.36%</td><td>0.1006</td><td>0.8706</td><td>25.45</td></tr><tr><td>ERTACache-fast [16]</td><td>2.47×</td><td>18.04</td><td>78.64%</td><td>0.1659</td><td>0.8170</td><td>22.34</td></tr><tr><td>GCache-slow (K = 18) GCache-fast (K = 11)</td><td>1.56× 2.54×</td><td>28.54 17.48</td><td>79.48% 78.44%</td><td>0.0509 0.1363</td><td>0.9247 0.8428</td><td>31.52</td></tr><tr><td rowspan="10">CogVideoX-2B (48 frames, 480P)</td><td></td><td></td><td></td><td></td><td></td><td></td><td>24.92</td></tr><tr><td>CogVideoX-2B (T = 50)</td><td>1x</td><td>78.48</td><td>80.18%</td><td></td><td></td><td></td></tr><tr><td>∆-DiT [14]</td><td>1.26×</td><td>62.50</td><td>79.09%</td><td>0.4053</td><td>0.6126</td><td>16.15</td></tr><tr><td>PAB [18]</td><td>1.35×</td><td>57.98</td><td>79.76%</td><td>0.0860</td><td>0.8978</td><td>28.04</td></tr><tr><td>FasterCache [30]</td><td>1.62×</td><td>48.44</td><td>79.83%</td><td>0.0766</td><td>0.9066</td><td>28.93</td></tr><tr><td>TeaCache [15]</td><td>2.92×</td><td>26.88</td><td>79.00%</td><td>0.2057</td><td>0.7614</td><td>20.97</td></tr><tr><td>ERTACache-slow [16]</td><td>1.62×</td><td>48.44</td><td>79.30%</td><td>0.0368</td><td>0.9394</td><td>32.77</td></tr><tr><td>ERTACache-fast [16]</td><td>2.93×</td><td>26.78</td><td>78.79% 79.36%</td><td>0.1012</td><td>0.8702</td><td>26.44</td></tr><tr><td>GCache-slow  $( K = 3 1 )$ </td><td>1.62×</td><td>48.40</td><td>78.30%</td><td>0.0178</td><td>0.9647</td><td>37.53</td></tr><tr><td>GCache-fast (K = 17)</td><td>2.93×</td><td>26.76</td><td></td><td>0.0721</td><td>0.9042</td><td>29.14</td></tr><tr><td rowspan="6">Wan2.1-1.3B (81 frames, 480P)</td><td>Wan2.1-1.3B (T = 50)</td><td>1×</td><td>199</td><td>81.30%</td><td></td><td></td><td></td></tr><tr><td>TeaCache [15]</td><td>2.00×</td><td>99.5</td><td>76.04%</td><td>0.2913</td><td>0.5685</td><td>16.17</td></tr><tr><td>ProfilingDiT [29]</td><td>2.01×</td><td>99</td><td>76.15%</td><td>0.1256</td><td>0.7899</td><td>22.02</td></tr><tr><td>ERTACache [16]</td><td>2.17×</td><td>91.7</td><td>80.73%</td><td>0.1095</td><td>0.8200</td><td>23.77</td></tr><tr><td>GCache-slow (K = 24)</td><td>2.17×</td><td>91.6</td><td>80.81%</td><td>0.0316</td><td>0.9475</td><td>32.44</td></tr><tr><td>GCache-fast (K = 16)</td><td>3.01×</td><td>66.1</td><td>79.86%</td><td>0.0828</td><td>0.8854</td><td>22.06</td></tr></table>

Table 2: Quantitative evaluation of efficiency and visual quality for image generation across different methods on Flux-dev 1.0. Bold denotes the best performance under similar acceleration ratios. ↑ indicates higher is better, and ↓ indicates lower is better.
<table><tr><td rowspan="2">Method</td><td colspan="2">Efficiency</td><td colspan="3">Visual Quality</td></tr><tr><td>Speedup↑</td><td>Latency (s)↓</td><td>LPIPS↓</td><td>SSIM ↑</td><td>PSNR ↑</td></tr><tr><td>Flux-dev  $1 . 0 ( T = 3 0 )$ </td><td>1×</td><td>15.96</td><td></td><td></td><td></td></tr><tr><td>TeaCache [15]</td><td>2.84×</td><td>5.62</td><td>0.4427</td><td>0.7445</td><td>16.48</td></tr><tr><td>ERTACache [16]</td><td>2.87×</td><td>5.56</td><td>0.2658</td><td>0.7863</td><td>20.60</td></tr><tr><td>GCache-slow  $( K = 1 4 )$ </td><td>2.05×</td><td>7.78</td><td>0.0900</td><td>0.9129</td><td>28.50</td></tr><tr><td>GCache-fast (K = 10)</td><td>2.87×</td><td>5.56</td><td>0.1825</td><td>0.8423</td><td>23.76</td></tr></table>

Baselines. We compare against several recent diffusion acceleration baselines, including ∆-DiT [14], T-GATE [28], PAB [18], ProfilingDiT [29], FasterCache [30], TeaCache [15], and ERTACache [16].

Evaluation Metrics. Following prior work [15, 16], we evaluate video generation using the official 946 prompts provided by VBench [31], and image generation using the official 30K prompts from COCO [32]. For efficiency, we measure end-to-end inference latency, from prompt ingestion to the generation of the final frame, and report speedup relative to the corresponding base model. For quality, we adopt four widely used metrics: VBench [31], LPIPS [22], PSNR, and SSIM. More details are provided in Appendix D.2.

## 4.2 Main Results

Quantitative Evaluation. Across all evaluated settings, GCache consistently achieves state-of-the-art performance, delivering both faster inference and higher generation quality than existing cache-based acceleration methods. As shown in Table 1, GCache-Slow consistently outperforms the strongest baseline, ERTACache, across all video backbones, achieving higher VBench scores and reducing LPIPS by over 50% on average while maintaining comparable acceleration. Notably, on Wan 2.1, GCache-Slow reduces LPIPS from 0.1095 to 0.0316 under the same 2.17× speedup, while GCache-Fast further achieves a 3.01× speedup and still surpasses ERTACache in generation quality (0.0828 vs. 0.1095 LPIPS). Similarly, on Flux-dev 1.0 (Table 2), GCache achieves an LPIPS of 0.1825 under a 2.87× speedup setting, significantly outperforming prior methods. These results highlight a key distinction between GCache and prior cache-based methods. While methods such as ERTACache explicitly introduce additional error estimation and rectification mechanisms to compensate for reuseinduced approximation errors, GCache focuses directly on optimizing the cache reuse policy itself from a global-impact perspective. By allocating computation to the timesteps that most influence final generation quality, GCache achieves superior speed–quality trade-offs without introducing any additional inference-time computation.

![](images/cf90128e35af6e4a629a88d4ac0fe2fc16c8f6d2df6d656b37e117b68a44c732.jpg)  
Figure 4: Qualitative comparison of video generation results on Wan 2.1. GCache consistently preserves higher alignment with the original, while ERTACache exhibits noticeable object and motion misalignment. Red boxes highlight object misalignment, and blue boxes indicate motion misalignment. Best viewed when zoomed in.

Table 3: Ablation study on polynomial order d.
<table><tr><td>d</td><td>LPIPS↓</td><td>SSIM↑</td><td>PSNR↑</td></tr><tr><td>1</td><td>0.1114</td><td>0.8591</td><td>26.15</td></tr><tr><td>2</td><td>0.0733</td><td>0.9042</td><td>29.09</td></tr><tr><td>3</td><td>0.0721</td><td>0.9042</td><td>29.14</td></tr><tr><td>4</td><td>0.0733</td><td>0.9016</td><td>28.84</td></tr></table>

Table 4: Ablation study on the outer objective L.
<table><tr><td>Objective</td><td>LPIPS ↓</td><td>SSIM↑</td><td>PSNR ↑</td></tr><tr><td>LPIPS</td><td>0.0733</td><td>0.9016</td><td>28.84</td></tr><tr><td>SSIM</td><td>0.0736</td><td>0.9045</td><td>29.10</td></tr><tr><td>LPIPS+SSIM</td><td>0.0721</td><td>0.9042</td><td>29.14</td></tr></table>

Qualitative Comparison. Figure 4 shows video generation results on WAN 2.1. Obviously, ER-TACache suffers from motion and object misalignment under aggressive cache reuse, including misaligned motion dynamics (e.g., panda face motion) and object-level deviations (e.g., items on the table) from the original outputs. In contrast, GCache preserves consistent motion patterns and object semantics. Similarly, Figure 5 presents image generation results on Flux-dev 1.0. ERTACache exhibits clear semantic and spatial misalignment under the same prompts, such as generating four smoking stacks when the prompt specifies two, as well as malformed human structures and incorrect object relationships. By contrast, GCache maintains semantic correctness and spatial coherence, closely matching the original model outputs. Additional results and discussions are provided in Appendix F.

## 4.3 Ablation Studies and Empirical Analysis

Ablation Study on Poly Degree d. We study the effect of the polynomial degree d used in Eq. 17 for modeling the propagation exponent w . Specifically, we evaluate different choices of d under a fixed budget $K = 1 7$ on CogVideoX-2B. As shown in Table 3, the best performance is achieved at d = 3, yielding the lowest LPIPS and the highest SSIM and PSNR. Therefore, we set d = 3 as the default choice in all experiments.

Impact of the Outer Objective L. To investigate the sensitivity of GCache to the choice of the outer objective, we evaluate three loss formulations on CogVideoX-2B (K = 17): LPIPS, SSIM, and a hybrid LPIPS+SSIM objective. As summarized in Table 4, while individual losses focus on specific image attributes (LPIPS on perceptual features and SSIM on structural integrity), their combination (LPIPS+SSIM) yields the best overall performance across all metrics. Specifically, the hybrid objective achieves the lowest LPIPS (0.0721) and the highest PSNR (29.14), suggesting that a multi-faceted supervision signal is crucial for optimizing the reuse policy.

![](images/1edb56ea5de772b89ed74624496f7d6a1504bbb65da8d3018f4ad48278e8ca90.jpg)  
Figure 5: Qualitative comparison of image generation results on Flux-dev 1.0. Best viewed when zoomed in.

Additional Experimental Results. We provide more comprehensive evaluations and empirical studies in Appendix E. These include: (i) extensive robustness tests across diverse prompt distributions and spatial resolutions (Appendix E.1 and E.2); and (ii) a detailed validation of the learned policy’s effectiveness and the fidelity of our pre-computed error proxy (Appendix E.3 and E.4). These results further substantiate the stability and generalization of GCache across various scenarios.

## 5 Conclusion

In this paper, we first identify the misalignment between local reuse discrepancies and global generation error, establishing a formal theoretical characterization of error propagation dynamics in cache-based acceleration. We demonstrate that policies optimized strictly via conservative analytical bounds are often sub-optimal in practice. To bridge this gap, we introduce Global-Impact Cache (GCache), a framework that reformulates the policy search as a bilevel optimization problem. This approach effectively reconciles theoretical error control with empirical perceptual quality. Extensive evaluations across various image and video backbones show that GCache consistently outperforms prior caching strategies.

## References

[1] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020.

[2] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021.

[3] Prafulla Dhariwal and Alexander Quinn Nichol. Diffusion models beat gans on image synthesis. In Marc’Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and Jennifer Wortman Vaughan, editors, Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pages 8780–8794, 2021.

[4] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023.

[5] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pages 10674–10685. IEEE, 2022.

[6] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L. Denton, Seyed Kamyar Seyed Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, Jonathan Ho, David J. Fleet, and Mohammad Norouzi. Photorealistic text-to-image diffusion models with deep language understanding. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022.

[7] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, and Robin Rombach. Stable video diffusion: Scaling latent video diffusion models to large datasets. CoRR, abs/2311.15127, 2023.

[8] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 7310–7320. IEEE, 2024.

[9] Yefei He, Luping Liu, Jing Liu, Weijia Wu, Hong Zhou, and Bohan Zhuang. PTQD: accurate post-training quantization for diffusion models. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10-16, 2023, 2023.

[10] Thibault Castells, Hyoung-Kyu Song, Bo-Kyeong Kim, and Shinkook Choi. LD-Pruner: efficient pruning of latent diffusion models using task-agnostic insights. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, CVPRW 2024, Seattle, WA, USA, June 17-21, 2024, pages 821–830, 2024.

[11] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. DPM-Solver: a fast ODE solver for diffusion probabilistic model sampling in around 10 steps. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022.

[12] Qinsheng Zhang and Yongxin Chen. Fast sampling of diffusion models with exponential integrator. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023, 2023.

[13] Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, and Jiwen Lu. UniPC: a unified predictorcorrector framework for fast sampling of diffusion models. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10-16, 2023, 2023.

[14] Pengtao Chen, Mingzhu Shen, Peng Ye, Jianjian Cao, Chongjun Tu, Christos-Savvas Bouganis, Yiren Zhao, and Tao Chen. ∆-dit: A training-free acceleration method tailored for diffusion transformers. CoRR, abs/2406.01125, 2024.

[15] Feng Liu, Shiwei Zhang, Xiaofeng Wang, Yujie Wei, Haonan Qiu, Yuzhong Zhao, Yingya Zhang, Qixiang Ye, and Fang Wan. Timestep embedding tells: It’s time to cache for video diffusion model. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2025, Nashville, TN, USA, June 11-15, 2025, pages 7353–7363. Computer Vision Foundation / IEEE, 2025.

[16] Xurui Peng, Chenqian Yan, Hong Liu, Rui Ma, Fangmin Chen, XING WANG, Zhihua Wu, Songwei Liu, and Mingbao Lin. ERTACache: Error rectification and timesteps adjustment for efficient diffusion. In The Fourteenth International Conference on Learning Representations, 2026.

[17] Pratheba Selvaraju, Tianyu Ding, Tianyi Chen, Ilya Zharkov, and Luming Liang. FORA: fast-forward caching in diffusion transformer acceleration. CoRR, abs/2407.01425, 2024.

[18] Xuanlei Zhao, Xiaolong Jin, Kai Wang, and Yang You. Real-time video generation with pyramid attention broadcast. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025.

[19] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net, 2024.

[20] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pages 5998–6008, 2017.

[21] Hongrui Chen, Holden Lee, and Jianfeng Lu. Improved analysis of score-based generative modeling: User-friendly bounds under minimal smoothness assumptions. In Proceedings of the 40th International Conference on Machine Learning, ICML 2023, pages 4735–4763, 2023.

[22] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In 2018 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2018, Salt Lake City, UT, USA, June 18-22, 2018, pages 586–595. Computer Vision Foundation / IEEE Computer Society, 2018.

[23] Niranjan Srinivas, Andreas Krause, Sham M. Kakade, and Matthias W. Seeger. Gaussian process optimization in the bandit setting: No regret and experimental design. In Proceedings of the 27th International Conference on Machine Learning (ICML-10), June 21-24, 2010, Haifa, Israel, pages 1015–1022. Omnipress, 2010.

[24] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all. CoRR, abs/2412.20404, 2024.

[25] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, Da Yin, Yuxuan Zhang, Weihan Wang, Yean Cheng, Bin Xu, Xiaotao Gu, Yuxiao Dong, and Jie Tang. Cogvideox: Text-to-video diffusion models with an expert transformer. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025.

[26] Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Xiaofeng Meng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. CoRR, abs/2503.20314, 2025.

[27] Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024.

[28] Haozhe Liu, Wentian Zhang, Jinheng Xie, Francesco Faccio, Mengmeng Xu, Tao Xiang, Mike Zheng Shou, Juan-Manuel Pérez-Rúa, and Jürgen Schmidhuber. Faster diffusion through temporal attention decomposition. Trans. Mach. Learn. Res., 2025, 2025.

[29] Xuran Ma, Yexin Liu, Yaofu Liu, Xianfeng Wu, Mingzhe Zheng, Zihao Wang, Ser-Nam Lim, and Harry Yang. Model reveals what to cache: Profiling-based feature reuse for video diffusion models. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), pages 17150–17159, October 2025.

[30] Zhengyao Lv, Chenyang Si, Junhao Song, Zhenyu Yang, Yu Qiao, Ziwei Liu, and Kwan-Yee K. Wong. Fastercache: Training-free video diffusion model acceleration with high quality. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025.

[31] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. Vbench: Comprehensive benchmark suite for video generative models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 21807–21818. IEEE, 2024.

[32] Tsung-Yi Lin, Michael Maire, Serge J. Belongie, Lubomir Bourdev, Ross B. Girshick, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick. Microsoft COCO: common objects in context. In Computer Vision - ECCV 2014 - 13th European Conference, Zürich, Switzerland, September 6-12, 2014, Proceedings, Part V, pages 740–755, 2014.

[33] Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, JUN ZHOU, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[34] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In Nassir Navab, Joachim Hornegger, William M. Wells III, and Alejandro F. Frangi, editors, Medical Image Computing and Computer-Assisted Intervention - MICCAI 2015 - 18th International Conference Munich, Germany, October 5 - 9, 2015, Proceedings, Part III, volume 9351 of Lecture Notes in Computer Science, pages 234–241. Springer, 2015.

[35] William Peebles and Saining Xie. Scalable diffusion models with transformers. In IEEE/CVF International Conference on Computer Vision, ICCV 2023, Paris, France, October 1-6, 2023, pages 4172–4182. IEEE, 2023.

[36] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021.

[37] Xingyi Yang, Daquan Zhou, Jiashi Feng, and Xinchao Wang. Diffusion probabilistic model made slim. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 18-22, 2023, pages 22552–22562, 2023.

## Appendix

## Contents

A Implementation Details 15   
A.1 Inner Optimization via Dynamic Programming 15   
A.2 Outer Optimization via Bayesian Optimization 15   
A.3 Optimization Objective 16   
B Proofs 16   
B.1 Proof for Theorem 3.2 16   
B.2 Proof for Theorem 3.3 19   
B.3 Proof for Theorem 3.4 20   
C Related Work 21   
C.1 Diffusion Models . 21   
C.2 Diffusion Model Acceleration 22   
D Experimental Details 22   
D.1 Evaluation Prompts 22   
D.2 Evaluation Metrics 22   
D.3 Training Details 23   
D.4 Optimization Efficiency . 23   
D.5 Experimental Details for Main Paper Visualizations 24   
E Additional Experiment Results 24   
E.1 Robustness to Prompt Distribution Shifts . 24   
E.2 Generalization Across Resolutions 25   
E.3 Effectiveness of the Learned Policy . 25   
E.4 Validation of the Pre-computed Local Error Proxy 26   
F Additional Qualitative Comparison 26   
F.1 Image Generation 26   
F.2 Vedio Generation 27   
G Limitations 27   
H Broader Impact 27

## A Implementation Details

## A.1 Inner Optimization via Dynamic Programming

Algorithm 1 Dynamic Programming for Cache Reuse Mask Search   
Require: Error matrix $\boldsymbol { E } \in \mathbb { R } ^ { N \times N }$ , cache refresh budget K   
Ensure: Reuse mask $m \in \{ 0 , 1 \} ^ { N }$ with $\| m \| _ { 0 } = K$   
1: Define the cumulative propagated error: $\begin{array} { r } { \mathcal { E } _ { j , i } = \sum _ { \tau = j + 1 } ^ { i - 1 } E _ { j , \tau } , \quad 0 \leq j < i \leq N } \end{array}$   
2: Initialize DP table $\boldsymbol { d p } [ k , i ]$ with +∞ and parent table $p a r [ k , i ]$ with −1 for all $k , i$   
3: $d p [ 1 , 0 ] \gets 0$   
4: for $k = 2 , \ldots , K$ do   
5: for $i = k - 1 , \ldots , N - 1$ do   
6: $d p [ k , i ] \gets \operatorname* { m i n } _ { 0 \leq j < i } \left( d p [ k - 1 , j ] + \mathcal { E } _ { j , i } \right)$   
7: $p a r [ k , i ] \gets \arg$ min $\left( d p [ k - 1 , j ] + \mathcal { E } _ { j , i } \right)$   
0≤j<i   
8: end for   
9: end for   
10: $r _ { K } \gets \arg \operatorname* { m i n } _ { 0 \leq j \leq N - 1 } \left( d p [ K , j ] + \mathcal { E } _ { j , N } \right)$   
11: Construct m from $\{ r _ { 1 } , \ldots , r _ { K } \}$ by backtracking using par   
12: return m

Given the propagated error matrix E defined in Eq. 20, where each entry encodes the cost incurred by reusing a cached residual across timesteps, selecting an optimal cache reuse policy amounts to minimizing the cumulative propagation error under a fixed refresh budget.

Recall that a cache reuse policy can be characterized by a set of K refresh timesteps, which partition the denoising trajectory into K segments. Within each segment, intermediate timesteps reuse the most recent cached residual and accumulate the corresponding propagation costs. Consequently, the total error induced by a policy is the sum of propagation costs over all segments, which naturally leads to a shortest-path formulation.

To solve this problem efficiently, we adopt a dynamic programming approach. We define the DP state $d p [ k , i ]$ as the minimum accumulated propagation error when the k-th cache refresh occurs at timestep i. Transitioning from a previous refresh point $j < i$ to i incurs an additional cost given by the propagated error $\bar { \mathcal { E } } _ { j , i }$ , which aggregates reuse costs between these two refresh points. The objective is then to find K refresh points that minimize the total accumulated error while satisfying the constraint $\| m \| _ { 0 } = K$

Algorithm 1 summarizes the resulting dynamic programming procedure, which runs in $O ( K N ^ { 2 } )$ time. Since the number of denoising steps N is typically fewer than 100 in practice, the computational overhead of this optimization is totally negligible.

## A.2 Outer Optimization via Bayesian Optimization

The comprehensive optimization procedure for GCache is summarized in Algorithm 2. At each iteration $n + 1$ , we maintain a Gaussian Process (GP) surrogate to model the objective landscape over the parameter space s. The next candidate is identified by minimizing the Lower Confidence Bound (LCB) acquisition function, which strategically balances exploration and exploitation:

$$
\begin{array} { r } { \pmb { s } _ { n + 1 } = \arg \underset { \pmb { s } } { \operatorname* { m i n } } A _ { \mathrm { L C B } } ( \pmb { s } ; \mathcal { D } _ { n } ) . } \end{array}\tag{24}
$$

Conditioned on the proposed parameters $s _ { n + 1 }$ , the inner optimization level formulates the weighted error objective and identifies the optimal cache reuse policy $m ^ { \star } ( s _ { n + 1 } )$ . This is achieved by minimizing the accumulated propagation error subject to the budget constraint $\pmb { m } \in \mathcal { C } ( \mathrm { E q u a t i o n } \left( 1 6 \right) )$ ), leveraging the efficient dynamic programming solver detailed in Appendix $\mathrm { A . 1 }$ . Subsequently, we evaluate the empirical generation loss $y _ { n + 1 } = f _ { \mathrm { e v a l } } ( \pmb { \mathscr { s } } _ { n + 1 } ) = \mathscr { L } ( \pmb { m } ^ { \star } ( \pmb { \mathscr { s } } _ { n + 1 } ) )$ to augment the observation set $\mathcal { D } _ { n + 1 } \bar { \mathbf { \Omega } } = \mathcal { D } \cup \left\{ \left( \mathbf { s } _ { n + 1 } , y _ { n + 1 } \right) \right\}$ and refine the GP surrogate. This iterative cycle continues until the evaluation budget is exhausted, ultimately yielding an optimized propagation exponent that aligns the analytical error bound with empirical generation quality.

```latex
Algorithm 2 Bilevel Optimization for Cache Policy Search
Require: Initial observations $\mathcal { D } _ { 0 } = \{ ( \boldsymbol { s } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { m }$ , exploration parameter $\kappa _ { n } .$ , total optimization step T
1: for n = 0 to $T - 1$ do
2: Update GP surrogate using $\mathcal { D } _ { n }$
3: $\begin{array} { r } { \pmb { s } _ { n + 1 } = \arg \operatorname* { m i n } \mu ( \pmb { s } ) - \kappa _ { n } \sigma ( \pmb { s } ) } \end{array}$
s
4: $\begin{array} { r } { \pmb { m } ^ { \star } ( \pmb { s } _ { n + 1 } ) = \underset { \pmb { m } \neq \ b { \mathscr { o } } } { \arg \operatorname* { m i n } } \sum _ { i = 1 } ^ { N - 1 } \left. \epsilon _ { t _ { i + 1 } } ^ { c } \right. _ { 1 } e ^ { w ( t _ { i + 1 } ; \pmb { s } _ { n + 1 } ) } } \end{array}$
m∈C
5: $y _ { n + 1 } = \mathcal { L } ( m ^ { \star } ( s _ { n + 1 } ) )$
6: $\mathcal { D } _ { n + 1 } = \mathcal { D } _ { n } \cup \{ ( \pmb { s } _ { n + 1 } , y _ { n + 1 } ) \}$
7: end for
8: Update GP surrogate using $\mathcal { D } _ { T }$
9: s<sup>⋆</sup> = arg min µ(s)
10: $\begin{array} { r } { \pmb { m } ^ { \star } = \underset { \pmb { m } \in \mathcal { C } } { \arg \operatorname* { m i n } } \sum _ { i = 1 } ^ { N - 1 } \left. \epsilon _ { t _ { i + 1 } } ^ { c } \right. _ { 1 } e ^ { w ( t _ { i + 1 } ; \pmb { s } ^ { \star } ) } } \end{array}$
```

## A.3 Optimization Objective

In the outer objective of the bilevel optimization framework (Equation (18)), we define the loss function L to evaluate the empirical generation quality. Specifically, we utilize a joint objective that combines the Learned Perceptual Image Patch Similarity (LPIPS) [22] and the Structural Similarity Index Measure (SSIM):

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { L P I P S } } + ( 1 - \mathcal { L } _ { \mathrm { S S I M } } ) .\tag{25}
$$

The rationale for this dual-component objective is twofold:

• Semantic and Perceptual Fidelity: LPIPS leverages deep features from pre-trained networks (e.g., VGG or AlexNet) to quantify high-level perceptual similarity. In the context of cache-based acceleration, aggressive reuse can lead to “semantic drift” or the loss of fine-grained textures that traditional metrics often fail to capture. LPIPS ensures that the optimized caching policy preserves the overall visual intent and semantic coherence of the generated samples.

• Structural and Pixel-level Integrity: While LPIPS is effective for high-level perception, it can occasionally overlook local structural misalignments. By incorporating $1 - \mathcal { L } _ { \mathrm { S S I M } }$ , we explicitly penalize deviations in local luminance, contrast, and spatial structure. This term acts as a regularizer to ensure that the accelerated trajectory remains spatially faithful to the unperturbed baseline, preventing artifacts such as ghosting or structural blurring that may arise during the accumulation of propagation errors.

To empirically justify the rationale behind our hybrid optimization objective, we conduct an ablation study comparing various loss formulations, with detailed results provided in Section 4.3. The findings corroborate that the combination of LPIPS and SSIM achieves a superior balance between perceptual fidelity and structural consistency.

## B Proofs

## B.1 Proof for Theorem 3.2

Lemma B.1 (Single-Step Propagation Error). Under Assumption 3.1, let $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ denote the state at time t along the ground-truth ODE trajectory, and let $\hat { \mathbf { x } } _ { t }$ be the state generated by the Euler solver using the learned velocity ${ \pmb v } _ { \pmb { \theta } } .$ . For a backward step size $h _ { t _ { i + 1 } } = t _ { i + 1 } - t _ { i } > 0 ,$ , the error at time $t _ { i }$ is bounded by:

$$
\| { \pmb x } _ { t _ { i } } - \hat { { \pmb x } } _ { t _ { i } } \| _ { 1 } \leq \left( 1 + h _ { t _ { i + 1 } } L _ { t _ { i + 1 } } \right) \big \| { \pmb x } _ { t _ { i + 1 } } - \hat { { \pmb x } } _ { t _ { i + 1 } } \big \| _ { 1 } + h _ { t _ { i + 1 } } \eta + \frac { h _ { t _ { i + 1 } } ^ { 2 } } { 2 } M .\tag{26}
$$

Proof. By Taylor’s theorem, the ground-truth state $\mathbf { \mathcal { x } } _ { t _ { i } }$ can be expanded about $t _ { i + 1 }$ by:

$$
\begin{array} { l } { { \displaystyle { \pmb x } _ { t _ { i } } = { \pmb x } _ { t _ { i + 1 } } - ( t _ { i + 1 } - t _ { i } ) \left. \frac { \mathrm { d } { \pmb x } _ { t } } { \mathrm { d } t } \right. _ { t _ { i + 1 } } + \left. \frac { ( t _ { i + 1 } - t _ { i } ) ^ { 2 } } { 2 } \frac { \mathrm { d } ^ { 2 } { \pmb x } _ { t } } { \mathrm { d } t ^ { 2 } } \right. _ { \xi _ { i + 1 } } } } \\ { { \displaystyle ~ = { \pmb x } _ { t _ { i + 1 } } - h _ { t _ { i + 1 } } { \pmb v } ( { \pmb x } _ { t _ { i + 1 } } , t _ { i + 1 } ) + \frac { h _ { t _ { i + 1 } } ^ { 2 } } { 2 } \left. \frac { \mathrm { d } } { \mathrm { d } t } { \pmb v } ( { \pmb x } _ { t } , t ) \right. _ { \xi _ { i + 1 } } , } } \end{array}\tag{27}
$$

where $\xi _ { i + 1 } \in [ t _ { i } , t _ { i + 1 } ]$ is an intermediate time point arising from the Lagrange form of the remainder, and we use the relation $\begin{array} { r } { \frac { \mathrm { d } \pmb { x } _ { t } } { \mathrm { d } t } = \pmb { v } ( \pmb { x } _ { t } , t ) } \end{array}$

The Euler update for $\hat { \mathbf { x } } _ { t _ { i } }$ is defined as:

$$
\begin{array} { r } { \hat { \pmb { x } } _ { t _ { i } } = \hat { \pmb { x } } _ { t _ { i + 1 } } - h _ { t _ { i + 1 } } \pmb { v } _ { \theta } ( \hat { \pmb { x } } _ { t _ { i + 1 } } , t _ { i + 1 } ) . } \end{array}\tag{28}
$$

Subtracting the discrete update from the continuous Taylor expansion, we obtain:

$$
{ \boldsymbol { x } } _ { t _ { i } } - { \boldsymbol { \hat { x } } } _ { t _ { i } } = { \boldsymbol { x } } _ { t _ { i + 1 } } - { \boldsymbol { \hat { x } } } _ { t _ { i + 1 } } - h _ { t _ { i + 1 } } \left( { \boldsymbol { v } } ( { \boldsymbol { x } } _ { t _ { i + 1 } } , t _ { i + 1 } ) - { \boldsymbol { v } } _ { \theta } ( { \boldsymbol { \hat { x } } } _ { t _ { i + 1 } } , t _ { i + 1 } ) \right) + \frac { h _ { t _ { i + 1 } } ^ { 2 } } { 2 } \left. \frac { \mathrm { d } } { \mathrm { d } t } { \boldsymbol { v } } ( { \boldsymbol { x } } _ { t } , t ) \right| _ { \xi _ { i + 1 } } .\tag{29}
$$

To bound the difference in velocity terms, we decompose it into a propagation component and an approximation component:

$$
\begin{array} { r l } & { v ( x _ { t _ { i + 1 } } , t _ { i + 1 } ) - v _ { \theta } ( \hat { x } _ { t _ { i + 1 } } , t _ { i + 1 } ) = \underbrace { v ( x _ { t _ { i + 1 } } , t _ { i + 1 } ) - v _ { \theta } ( x _ { t _ { i + 1 } } , t _ { i + 1 } ) } _ { \mathrm { A p p r o x i m a t i o n ~ E r r o r } } } \\ & { \quad \quad \quad \quad \quad \quad \quad + \underbrace { v _ { \theta } ( x _ { t _ { i + 1 } } , t _ { i + 1 } ) - v _ { \theta } ( \hat { x } _ { t _ { i + 1 } } , t _ { i + 1 } ) } _ { \mathrm { P r o p a g a t e d ~ E r r o r } } . } \end{array}\tag{30}
$$

Taking the $\ell _ { 1 }$ -norm on both sides and applying the triangle inequality yields:

$$
\begin{array} { r l } & { \| \boldsymbol { x } _ { t _ { i } } - \boldsymbol { \hat { x } } _ { t _ { i } } \| _ { 1 } \leq \left\| \boldsymbol { x } _ { t _ { i + 1 } } - \boldsymbol { \hat { x } } _ { t _ { i + 1 } } \right\| _ { 1 } } \\ & { \qquad + \left. h _ { t _ { i + 1 } } \left\| \boldsymbol { v } ( \boldsymbol { x } _ { t _ { i + 1 } } , t _ { i + 1 } ) - \boldsymbol { v } _ { \theta } ( \boldsymbol { x } _ { t _ { i + 1 } } , t _ { i + 1 } ) \right\| _ { 1 } \right. } \\ & { \qquad + \left. h _ { t _ { i + 1 } } \left\| \boldsymbol { v } _ { \theta } ( \boldsymbol { x } _ { t _ { i + 1 } } , t _ { i + 1 } ) - \boldsymbol { v } _ { \theta } ( \boldsymbol { \hat { x } } _ { t _ { i + 1 } } , t _ { i + 1 } ) \right\| _ { 1 } \right. } \\ & { \qquad + \left. \displaystyle \frac { h _ { t _ { i + 1 } } ^ { 2 } } { 2 } \left\| \displaystyle \frac { \mathrm { d } } { \mathrm { d } t } \boldsymbol { v } ( \boldsymbol { x } _ { t } , t ) \right| _ { \xi _ { i + 1 } } \right\| _ { 1 } . } \end{array}\tag{31}
$$

Applying the Lipschitz continuity of $\pmb { v }$ (with constant $L _ { t _ { i + 1 } } )$ , the approximation bound $\eta ,$ and the total derivative bound M from Assumption 3.1, we have:

$$
\begin{array} { r l } & { \| { \boldsymbol x } _ { t _ { i } } - \hat { \boldsymbol x } _ { t _ { i } } \| _ { 1 } \leq \big \| { \boldsymbol x } _ { t _ { i + 1 } } - \hat { \boldsymbol x } _ { t _ { i + 1 } } \big \| _ { 1 } + h _ { t _ { i + 1 } } L _ { t _ { i + 1 } } \big \| { \boldsymbol x } _ { t _ { i + 1 } } - \hat { \boldsymbol x } _ { t _ { i + 1 } } \big \| _ { 1 } + h _ { t _ { i + 1 } } \eta + \displaystyle \frac { h _ { t _ { i + 1 } } ^ { 2 } } { 2 } M } \\ & { \qquad = \big ( 1 + h _ { t _ { i + 1 } } L _ { t _ { i + 1 } } \big ) \big \| { \boldsymbol x } _ { t _ { i + 1 } } - \hat { \boldsymbol x } _ { t _ { i + 1 } } \big \| _ { 1 } + h _ { t _ { i + 1 } } \eta + \displaystyle \frac { h _ { t _ { i + 1 } } ^ { 2 } } { 2 } M , } \end{array}\tag{32}
$$

which completes the proof.

□

Lemma B.2 (Single-Step Propagation Error under Cache Reuse). Under Assumption $3 . I ,$ let $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ denote the state at time t along the ground-truth ODE trajectory, and let $\hat { \mathbf { x } } _ { t }$ be the state generated by the Euler solver. Suppose that at time $t _ { i + 1 }$ , we reuse a cached residual $\delta _ { t _ { i + 1 } } ^ { c }$ such that the reuse error is $\epsilon _ { t _ { i + 1 } } ^ { c } = \delta _ { t _ { i + 1 } } - \delta _ { t _ { i + 1 } } ^ { c }$ . Let $\hat { \pmb { x } } _ { t _ { i } } ^ { c }$ be the state at $t _ { i }$ produced by an Euler step from $\hat { \pmb { x } } _ { t _ { i + } }$ using the cached velocity $\pmb { v } _ { \theta } ^ { c } .$ . For a backward step size $h _ { t _ { i + 1 } } = t _ { i + 1 } - t _ { i } > 0$ , the error at time $t _ { i }$ is bounded by:

$$
\left. \pmb { x } _ { t _ { i } } - \hat { \pmb { x } } _ { t _ { i } } ^ { c } \right. _ { 1 } \leq \left( 1 + h _ { t _ { i + 1 } } L _ { t _ { i + 1 } } \right) \left. \pmb { x } _ { t _ { i + 1 } } - \hat { \pmb { x } } _ { t _ { i + 1 } } \right. _ { 1 } + h _ { t _ { i + 1 } } \eta + h _ { t _ { i + 1 } } L _ { o u t } \left. \epsilon _ { t _ { i + 1 } } ^ { c } \right. _ { 1 } + \frac { h _ { t _ { i + 1 } } ^ { 2 } } { 2 } M .\tag{33}
$$

Proof. By Taylor’s theorem, the ground-truth state $\mathbf { \mathcal { x } } _ { t _ { i } }$ can be expanded about $t _ { i + 1 }$ by:

$$
{ \pmb x } _ { t _ { i } } = { \pmb x } _ { t _ { i + 1 } } - h _ { t _ { i + 1 } } { \pmb v } ( { \pmb x } _ { t _ { i + 1 } } , t _ { i + 1 } ) + \frac { h _ { t _ { i + 1 } } ^ { 2 } } { 2 } \left. \frac { \mathrm { d } } { \mathrm { d } t } { \pmb v } ( { \pmb x } _ { t } , t ) \right| _ { \xi _ { i + 1 } } ,\tag{34}
$$

where $\xi _ { i + 1 } \in [ t _ { i } , t _ { i + 1 } ]$ is an intermediate time point arising from the Lagrange form of the remainder, and we use the relation $\begin{array} { r } { \frac { \mathrm { d } \pmb { x } _ { t } } { \mathrm { d } t } = \pmb { v } ( \pmb { x } _ { t } , t ) } \end{array}$

Let ${ \pmb v } _ { \theta } ^ { c } ( { \pmb x } _ { t } , t )$ denote the learned velocity computed using the cached residual. The Euler update with cache reuse for $\hat { \pmb { x } } _ { t _ { i } }$ is defined as:

$$
\begin{array} { r } { \hat { \pmb { x } } _ { t _ { i } } ^ { c } = \hat { \pmb { x } } _ { t _ { i + 1 } } - h _ { t _ { i + 1 } } \pmb { v } _ { \theta } ^ { c } ( \hat { \pmb { x } } _ { t _ { i + 1 } } , t _ { i + 1 } ) . } \end{array}\tag{35}
$$

Subtracting the discrete update from the Taylor expansion, we have:

$$
\boldsymbol { x } _ { t _ { i } } - \boldsymbol { \hat { x } } _ { t _ { i } } ^ { c } = \boldsymbol { x } _ { t _ { i + 1 } } - \boldsymbol { \hat { x } } _ { t _ { i + 1 } } - h _ { t _ { i + 1 } } \left( \boldsymbol { v } ( \boldsymbol { x } _ { t _ { i + 1 } } , t _ { i + 1 } ) - \boldsymbol { v } _ { \theta } ^ { c } ( \boldsymbol { \hat { x } } _ { t _ { i + 1 } } , t _ { i + 1 } ) \right) + \frac { h _ { t _ { i + 1 } } ^ { 2 } } { 2 } \left. \frac { \mathrm { d } } { \mathrm { d } t } \boldsymbol { v } ( \boldsymbol { x } _ { t } , t ) \right| _ { \xi _ { i + 1 } } .\tag{36}
$$

We decompose the velocity difference into three components to isolate the cache reuse error:

$$
\begin{array} { r l } & { v ( x _ { t _ { i + 1 } } , t _ { i + 1 } ) - v _ { \theta } ^ { c } ( \hat { x } _ { t _ { i + 1 } } , t _ { i + 1 } ) = \underbrace { v ( x _ { t _ { i + 1 } } , t _ { i + 1 } ) - v _ { \theta } ( x _ { t _ { i + 1 } } , t _ { i + 1 } ) } _ { \mathrm { A p p r o x i m a t i o n ~ E r r o r } } } \\ & { ~ + \underbrace { v _ { \theta } ( x _ { t _ { i + 1 } } , t _ { i + 1 } ) - v _ { \theta } ( \hat { x } _ { t _ { i + 1 } } , t _ { i + 1 } ) } _ { \mathrm { P r o p a g a t e d ~ E r r o r } } } \\ & { ~ + \underbrace { v _ { \theta } ( \hat { x } _ { t _ { i + 1 } } , t _ { i + 1 } ) - v _ { \theta } ^ { c } ( \hat { x } _ { t _ { i + 1 } } , t _ { i + 1 } ) } _ { \mathrm { C a c h e ~ R e u s e ~ E r r o r } } . } \end{array}\tag{37}
$$

Taking the $\ell _ { 1 }$ -norm on both sides and applying the triangle inequality yields:

$$
\begin{array} { r l } & { \left\| x _ { t _ { i } } - \hat { x } _ { t _ { i } } ^ { c } \right\| _ { 1 } \leq \left\| x _ { t _ { i + 1 } } - \hat { x } _ { t _ { i + 1 } } \right\| _ { 1 } } \\ & { \qquad + h _ { t _ { i + 1 } } \left\| v ( x _ { t _ { i + 1 } } , t _ { i + 1 } ) - v _ { \theta } ( x _ { t _ { i + 1 } } , t _ { i + 1 } ) \right\| _ { 1 } } \\ & { \qquad + h _ { t _ { i + 1 } } \left\| v _ { \theta } ( x _ { t _ { i + 1 } } , t _ { i + 1 } ) - v _ { \theta } ( \hat { x } _ { t _ { i + 1 } } , t _ { i + 1 } ) \right\| _ { 1 } } \\ & { \qquad + h _ { t _ { i + 1 } } \left\| v _ { \theta } ( \hat { x } _ { t _ { i + 1 } } , t _ { i + 1 } ) - v _ { \theta } ^ { c } ( \hat { x } _ { t _ { i + 1 } } , t _ { i + 1 } ) \right\| _ { 1 } } \\ & { \qquad + \displaystyle \frac { h _ { t _ { i + 1 } } ^ { 2 } } { 2 } \left\| \frac { \mathrm { d } } { \mathrm { d } t } v ( x _ { t } , t ) \right\| _ { \xi _ { i + 1 } } \Bigg \| _ { 1 } . } \end{array}\tag{38}
$$

Given the model architecture ${ \pmb v } _ { \theta } ( { \pmb x } _ { t } , t ) = f _ { \mathrm { o u t } } ( { \pmb h } _ { t } + { \pmb \delta } _ { t } )$ , where $h _ { t }$ is the feature and $\delta _ { t }$ is the residual, the cache reuse error term is bounded by the Lipschitz constant $L _ { \mathrm { o u t } } \mathrm { : }$

$$
\begin{array} { r } { \left\| f _ { \mathrm { o u t } } ( h _ { t _ { i + 1 } } + \delta _ { t _ { i + 1 } } ) - f _ { \mathrm { o u t } } ( h _ { t _ { i + 1 } } + \delta _ { t _ { i + 1 } } ^ { c } ) \right\| _ { 1 } \leq L _ { \mathrm { o u t } } \left\| \delta _ { t _ { i + 1 } } - \delta _ { t _ { i + 1 } } ^ { c } \right\| _ { 1 } = L _ { \mathrm { o u t } } \left\| \epsilon _ { t _ { i + 1 } } ^ { c } \right\| _ { 1 } . } \end{array}\tag{39}
$$

Finally, substituting the Lipschitz constant $L _ { t _ { i + 1 } }$ , the approximation bound $\eta ,$ , and the total derivative bound M from Assumption 3.1, we obtain:

$$
\begin{array} { r l } & { \| \pmb { x } _ { t _ { i } } - \hat { \pmb { x } } _ { t _ { i } } \| _ { 1 } \leq \big \| \pmb { x } _ { t _ { i + 1 } } - \hat { \pmb { x } } _ { t _ { i + 1 } } \big \| _ { 1 } + h _ { t _ { i + 1 } } L _ { t _ { i + 1 } } \big \| \pmb { x } _ { t _ { i + 1 } } - \hat { \pmb { x } } _ { t _ { i + 1 } } \big \| _ { 1 } + h _ { t _ { i + 1 } } \eta } \\ & { \qquad + h _ { t _ { i + 1 } } L _ { \mathrm { o u t } } \left\| \epsilon _ { t _ { i + 1 } } ^ { c } \right\| _ { 1 } + \displaystyle \frac { h _ { t _ { i + 1 } } ^ { 2 } } { 2 } M } \\ & { \qquad = \big ( 1 + h _ { t _ { i + 1 } } L _ { t _ { i + 1 } } \big ) \left\| \pmb { x } _ { t _ { i + 1 } } - \hat { \pmb { x } } _ { t _ { i + 1 } } \right\| _ { 1 } + h _ { t _ { i + 1 } } \eta } \\ & { \qquad + h _ { t _ { i + 1 } } L _ { \mathrm { o u t } } \left\| \epsilon _ { t _ { i + 1 } } ^ { c } \right\| _ { 1 } + \displaystyle \frac { h _ { t _ { i + 1 } } ^ { 2 } } { 2 } M . } \end{array}\tag{40}
$$

This completes the proof.

Theorem 3.2 (Global Error Bound under Cache Reuse). Under Assumption 3.1, let $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ be the groundtruth ODE state at time $t ,$ and let $\hat { \pmb x } _ { t } ^ { c }$ be the state generated by the Euler solver using the learned velocity $v _ { \theta } ,$ , incorporating a single cache reuse event. Specifically, suppose that at step $t _ { i + 1 } \to t _ { i }$ we reuse a cached residual $\delta _ { t _ { i + 1 } } ^ { c }$ with error $\epsilon _ { t _ { i + 1 } } ^ { c } = \delta _ { t _ { i + 1 } } - \delta _ { t _ { i + 1 } } ^ { c }$ . Given a sequence of timesteps $t _ { N } , t _ { N - 1 } , \ldots , t _ { 1 }$ with uniform step size $h ,$ the global error at thefinal timestep $t _ { 1 }$ is bounded by:

$$
\left\| x _ { t _ { 1 } } - \hat { x } _ { t _ { 1 } } ^ { c } \right\| _ { 1 } \leq \underbrace { h L _ { o u t } \left\| \epsilon _ { t _ { i + 1 } } ^ { c } \right\| _ { 1 } } _ { C a c h e R e u s e E r r o r } + \underbrace { \left( \frac { \eta } { L } + \frac { h M } { 2 L } \right) \left( e ^ { ( N - 1 ) h L } - 1 \right) } _ { A p p r o x i m a t i o n \notin D i s c r e t i z a t i o n E r r o r } .\tag{41}
$$

Proof. We analyze the error accumulation by recursively applying the single-step error bounds. From Lemma B.1, the error at the final timestep $t _ { 1 }$ can be expressed in terms of the error at an intermediate step $t _ { i } { \mathrm { : } }$

$$
\big \| \pmb { x } _ { t _ { 1 } } - \hat { \pmb { x } } _ { t _ { 1 } } ^ { c } \big \| _ { 1 } \leq \big \| \pmb { x } _ { t _ { i } } - \hat { \pmb { x } } _ { t _ { i } } ^ { c } \big \| _ { 1 } ( 1 + h L ) ^ { i - 1 } + \bigg ( h \eta + \frac { h ^ { 2 } } { 2 } M \bigg ) \sum _ { n = 1 } ^ { i - 1 } ( 1 + h L ) ^ { n - 1 } .\tag{42}
$$

At the specific transition $t _ { i + 1 } \to t _ { i }$ where cache reuse occurs, Lemma B.2 provides the local bound:

$$
\big \| { \boldsymbol x } _ { t _ { i } } - \hat { { \boldsymbol x } } _ { t _ { i } } ^ { c } \big \| _ { 1 } \leq ( 1 + h L ) \left\| \boldsymbol x _ { t _ { i + 1 } } - \hat { { \boldsymbol x } } _ { t _ { i + 1 } } \right\| _ { 1 } + h { \eta } + h L _ { \mathrm { o u t } } \left\| \boldsymbol \epsilon _ { t _ { i + 1 } } ^ { c } \right\| _ { 1 } + \frac { h ^ { 2 } } { 2 } M .\tag{43}
$$

Continuing the recursion for the remaining steps from $t _ { i + 1 }$ back to the initial state $t _ { N }$ , we observe that for all other steps $n \in \{ i + 1 , \ldots , \bar { N } - 1 \bar  \}$ , the standard local error bound from Lemma B.1 applies. Combining these, the global error at $t _ { 1 }$ becomes:

$$
\begin{array} { r } { \left\| \boldsymbol { x } _ { t _ { 1 } } - \hat { \boldsymbol { x } } _ { t _ { 1 } } ^ { c } \right\| _ { 1 } \leq \left\| \boldsymbol { x } _ { t _ { N } } - \hat { \boldsymbol { x } } _ { t _ { N } } \right\| _ { 1 } \left( 1 + h L \right) ^ { N - 1 } + h L _ { \mathrm { o u t } } \left\| \epsilon _ { t _ { i + 1 } } ^ { c } \right\| _ { 1 } \left( 1 + h L \right) ^ { i - 1 } } \end{array}
$$

$$
+ \left( h \eta + \frac { h ^ { 2 } } { 2 } M \right) \sum _ { n = 1 } ^ { N - 1 } ( 1 + h L ) ^ { n - 1 } .\tag{44}
$$

Assuming the solver starts from the ground-truth initial noise, we have $\mathbf { x } _ { t _ { N } } = \hat { \mathbf { x } } _ { t _ { N } }$ and the first term vanishes. The remaining terms simplify to:

$$
\left\| { \boldsymbol x } _ { t _ { 1 } } - \hat { \boldsymbol x } _ { t _ { 1 } } ^ { c }  \right\| _ { 1 } \leq h L _ { \operatorname { o u t } } \left\| \epsilon _ { t _ { i + 1 } } ^ { c } \right\| _ { 1 } \left( 1 + h L \right) ^ { i - 1 } + \left( h \eta + \frac { h ^ { 2 } } { 2 } M \right) \sum _ { n = 1 } ^ { N - 1 } ( 1 + h L ) ^ { n - 1 } .\tag{45}
$$

Using the geometric series identity $\begin{array} { r } { \sum _ { j = 0 } ^ { k - 1 } r ^ { j } = \frac { r ^ { k } - 1 } { r - 1 } } \end{array}$ with $r = 1 + h L$ , the remaining terms simplifies to:

$$
\left\| x _ { t _ { 1 } } - \hat { x } _ { t _ { 1 } } ^ { c } \right\| _ { 1 } \leq h L _ { \mathrm { o u t } } \left\| \epsilon _ { t _ { i + 1 } } ^ { c } \right\| _ { 1 } \left( 1 + h L \right) ^ { i - 1 } + \left( h \eta + \frac { h ^ { 2 } } { 2 } M \right) \frac { 1 } { h L } \left( ( 1 + h L ) ^ { N - 1 } - 1 \right)
$$

$$
= h L _ { \mathrm { o u t } } \left\| \epsilon _ { t _ { i + 1 } } ^ { c } \right\| _ { 1 } ( 1 + h L ) ^ { i - 1 } + \left( \frac { \eta } { L } + \frac { h M } { 2 L } \right) \left( ( 1 + h L ) ^ { N - 1 } - 1 \right)\tag{46}
$$

Finally, employing the inequality $1 + a \le e ^ { a }$ , we arrive at the final bound:

$$
\left\| \boldsymbol { x } _ { t _ { 1 } } - \hat { \boldsymbol { x } } _ { t _ { 1 } } ^ { c } \right\| _ { 1 } \leq h L _ { \operatorname * { o u t } } \left\| \epsilon _ { t _ { i + 1 } } ^ { c } \right\| _ { 1 } e ^ { ( i - 1 ) h L } + \left( \frac { \eta } { L } + \frac { h M } { 2 L } \right) \left( e ^ { ( N - 1 ) h L } - 1 \right) .\tag{47}
$$

This completes the proof.

## B.2 Proof for Theorem 3.3

Theorem 3.3 (Error Propagation Bound under Single-Step Cache Reuse). Under Assumption 3.1, let xˆ be the state generated by the Euler solver using the learned velocity $v _ { \theta } ,$ , and let $\hat { \pmb x } _ { t } ^ { c }$ be the state incorporating a single cache reuse event at step $t _ { i + 1 } \to t _ { i }$ . Suppose the reuse ofa cached residual $\delta _ { t _ { i + } } ^ { c }$ introduces an error $\epsilon _ { t _ { i + 1 } } ^ { c } = \delta _ { t _ { i + 1 } } - \delta _ { t _ { i + 1 } } ^ { c }$ . Given a sequence of timesteps t<sub>N</sub>, $t _ { N - 1 } , \ldots , t _ { 1 }$ with step sizes $h _ { t _ { n + 1 } } = t _ { n + 1 } - t _ { n }$ , the cumulative error at the final timestep $t _ { 1 }$ is bounded by:

$$
\left\| \hat { \pmb x } _ { t _ { 1 } } - \hat { \pmb x } _ { t _ { 1 } } ^ { c } \right\| _ { 1 } \leq \left\| \epsilon _ { t _ { i + 1 } } ^ { c } \right\| _ { 1 } e ^ { w _ { t _ { i + 1 } } } ,\tag{48}
$$

where the propagation exponent $w _ { t _ { i + 1 } }$ is defined as:

$$
w _ { t _ { i + 1 } } = \ln \left( h _ { t _ { i + 1 } } L _ { o u t } \right) + \sum _ { n = 1 } ^ { i - 1 } h _ { t _ { n + 1 } } L _ { t _ { n + 1 } } .\tag{49}
$$

Proof. We first bound the error propagation for the steps following the cache reuse $( t _ { n }$ for $n < i )$ . The Euler updates for the standard and cached trajectories are:

$$
\begin{array} { r } { \hat { \pmb { x } } _ { t _ { n } } = \hat { \pmb { x } } _ { t _ { n + 1 } } - h _ { t _ { n + 1 } } \pmb { v } _ { \theta } \big ( \hat { \pmb { x } } _ { t _ { n + 1 } } , t _ { n + 1 } \big ) , } \end{array}\tag{50}
$$

$$
\begin{array} { r } { \hat { \pmb x } _ { t _ { n } } ^ { c } = \hat { \pmb x } _ { t _ { n + 1 } } ^ { c } - h _ { t _ { n + 1 } } \pmb v _ { \theta } \big ( \hat { \pmb x } _ { t _ { n + 1 } } ^ { c } , t _ { n + 1 } \big ) . } \end{array}\tag{51}
$$

Subtracting these updates and applying the triangle inequality:

$$
\begin{array} { r l } & { \left\| \hat { x } _ { t _ { n } } - \hat { x } _ { t _ { n } } ^ { c } \right\| _ { 1 } = \left\| \hat { x } _ { t _ { n + 1 } } - \hat { x } _ { t _ { n + 1 } } ^ { c } - h _ { t _ { n + 1 } } \left( v _ { \theta } ( \hat { x } _ { t _ { n + 1 } } , t _ { n + 1 } ) - v _ { \theta } ( \hat { x } _ { t _ { n + 1 } } ^ { c } , t _ { n + 1 } ) \right) \right\| _ { 1 } } \\ & { \qquad \leq \left\| \hat { x } _ { t _ { n + 1 } } - \hat { x } _ { t _ { n + 1 } } ^ { c } \right\| _ { 1 } + h _ { t _ { n + 1 } } \left\| v _ { \theta } ( \hat { x } _ { t _ { n + 1 } } , t _ { n + 1 } ) - v _ { \theta } ( \hat { x } _ { t _ { n + 1 } } ^ { c } , t _ { n + 1 } ) \right\| _ { 1 } } \\ & { \qquad \leq ( 1 + h _ { t _ { n + 1 } } L _ { t _ { n + 1 } } ) \left\| \hat { x } _ { t _ { n + 1 } } - \hat { x } _ { t _ { n + 1 } } ^ { c } \right\| _ { 1 } , } \end{array}\tag{52}
$$

where $L _ { t _ { n + 1 } }$ is the Lipschitz constant of ${ \pmb v } _ { \theta }$ . Applying this relation recursively from $t _ { 1 }$ back to $t _ { i }$ yields:

$$
\big \| \hat { \pmb x } _ { t _ { 1 } } - \hat { \pmb x } _ { t _ { 1 } } ^ { c } \big \| _ { 1 } \leq \big \| \hat { \pmb x } _ { t _ { i } } - \hat { \pmb x } _ { t _ { i } } ^ { c } \big \| _ { 1 } \prod _ { n = 1 } ^ { i - 1 } ( 1 + h _ { t _ { n + 1 } } L _ { t _ { n + 1 } } ) .\tag{53}
$$

Next, we bound the local error injected at step $t _ { i } . \ \mathrm { A t }$ this step, both trajectories originate from the same state $\hat { \pmb { x } } _ { t _ { i + 1 } }$ , but the cached trajectory uses the approximate velocity $\pmb { v } _ { \theta } ^ { c }$ . The updates are:

$$
\begin{array} { r } { \hat { \pmb { x } } _ { t _ { i } } = \hat { \pmb { x } } _ { t _ { i + 1 } } - h _ { t _ { i + 1 } } \pmb { v } _ { \theta } \big ( \hat { \pmb { x } } _ { t _ { i + 1 } } , t _ { i + 1 } \big ) , } \end{array}\tag{54}
$$

$$
\begin{array} { r } { \hat { \pmb { x } } _ { t _ { i } } ^ { c } = \hat { \pmb { x } } _ { t _ { i + 1 } } - h _ { t _ { i + 1 } } \pmb { v } _ { \theta } ^ { c } ( \hat { \pmb { x } } _ { t _ { i + 1 } } , t _ { i + 1 } ) . } \end{array}\tag{55}
$$

The norm of the difference is thus:

$$
\left\| \hat { \pmb { x } } _ { t _ { i } } - \hat { \pmb { x } } _ { t _ { i } } ^ { c } \right\| _ { 1 } = h _ { t _ { i + 1 } } \left\| \pmb { v } _ { \theta } ( \hat { \pmb { x } } _ { t _ { i + 1 } } , t _ { i + 1 } ) - \pmb { v } _ { \theta } ^ { c } ( \hat { \pmb { x } } _ { t _ { i + 1 } } , t _ { i + 1 } ) \right\| _ { 1 } .\tag{56}
$$

Using the model architecture ${ \pmb v } _ { \theta } ( { \pmb x } , t ) = f _ { \mathrm { o u t } } ( { \pmb h } + { \pmb \delta } )$ and the Lipschitz continuity of the output layer $f _ { \mathrm { o u t } }$ , we have:

$$
\begin{array} { r } { \left\| f _ { \mathrm { o u t } } ( h _ { t _ { i + 1 } } + \delta _ { t _ { i + 1 } } ) - f _ { \mathrm { o u t } } ( h _ { t _ { i + 1 } } + \delta _ { t _ { i + 1 } } ^ { c } ) \right\| _ { 1 } \leq L _ { \mathrm { o u t } } \left\| \epsilon _ { t _ { i + 1 } } ^ { c } \right\| _ { 1 } . } \end{array}\tag{57}
$$

Substituting this into the local error bound yields:

$$
\left\| \hat { \pmb x } _ { t _ { i } } - \hat { \pmb x } _ { t _ { i } } ^ { c } \right\| _ { 1 } \leq h _ { t _ { i + 1 } } L _ { \mathrm { o u t } } \left\| \epsilon _ { t _ { i + 1 } } ^ { c } \right\| _ { 1 } .\tag{58}
$$

Combining the propagation and local error bounds, and applying $1 + a \le e ^ { a } ;$

$$
\begin{array} { r l } & { \left\| \hat { \boldsymbol { x } } _ { t _ { 1 } } - \hat { \boldsymbol { x } } _ { t _ { 1 } } ^ { c } \right\| _ { 1 } \leq h _ { t _ { i + 1 } } L _ { \mathrm { o u t } } \left\| \epsilon _ { t _ { i + 1 } } ^ { c } \right\| _ { 1 } \displaystyle \prod _ { n = 1 } ^ { i - 1 } \exp ( h _ { t _ { n + 1 } } L _ { t _ { n + 1 } } ) } \\ & { \qquad = \left\| \epsilon _ { t _ { i + 1 } } ^ { c } \right\| _ { 1 } \exp \left( \ln ( h _ { t _ { i + 1 } } L _ { \mathrm { o u t } } ) + \displaystyle \sum _ { n = 1 } ^ { i - 1 } h _ { t _ { n + 1 } } L _ { t _ { n + 1 } } \right) . } \end{array}\tag{59}
$$

This completes the proof.

## B.3 Proof for Theorem 3.4

Theorem 3.4 (Error Propagation Bound under Multi-Step Cache Reuse). Under Assumption 3.1, let $\hat { \mathbf { x } } _ { t }$ be the state generated by the Euler solver using the learned velocity ${ \pmb v } _ { { \pmb \theta } ; }$ , and let $\hat { \pmb x } _ { t } ^ { c }$ be the state incorporating multiple cache reuse events at every step $t _ { n + 1 } \to t _ { n } f o r n \in \{ N - 1 , \ldots , 1 \}$ . Suppose each reuse ofa cached residual introduces a local error $\epsilon _ { t _ { n + 1 } } ^ { c } = \delta _ { t _ { n + 1 } } - \delta _ { t _ { n + 1 } } ^ { c }$ . Given a sequence of timesteps t<sub>N</sub>, $t _ { N - 1 } , \ldots , t _ { 1 }$ with step sizes $h _ { t _ { n + 1 } } = t _ { n + 1 } - t _ { n }$ , the cumulative error at the final timestep $t _ { 1 }$ is bounded by:

$$
\left\| \hat { \pmb x } _ { t _ { 1 } } - \hat { \pmb x } _ { t _ { 1 } } ^ { c } \right\| _ { 1 } \leq \sum _ { n = 1 } ^ { N - 1 } \left\| \epsilon _ { t _ { n + 1 } } ^ { c } \right\| _ { 1 } e ^ { w _ { t _ { n + 1 } } } .\tag{60}
$$

where the propagation exponent $w _ { t _ { n + 1 } }$ is defined as:

$$
w _ { t _ { n + 1 } } = \ln \left( h _ { t _ { n + 1 } } L _ { o u t } \right) + \sum _ { m = 1 } ^ { n - 1 } h _ { t _ { m + 1 } } L _ { t _ { m + 1 } } .\tag{61}
$$

Proof. We begin by analyzing the error transition for a single step $t _ { n + 1 } \to t _ { n }$ . The standard Euler update and the cached Euler update are given by:

$$
\begin{array} { r } { \hat { \pmb x } _ { t _ { n } } = \hat { \pmb x } _ { t _ { n + 1 } } - h _ { t _ { n + 1 } } \pmb v _ { \theta } ( \hat { \pmb x } _ { t _ { n + 1 } } , t _ { n + 1 } ) , } \end{array}\tag{62}
$$

$$
\begin{array} { r } { \hat { \pmb x } _ { t _ { n } } ^ { c } = \hat { \pmb x } _ { t _ { n + 1 } } ^ { c } - h _ { t _ { n + 1 } } \pmb v _ { \theta } ^ { c } ( \hat { \pmb x } _ { t _ { n + 1 } } ^ { c } , t _ { n + 1 } ) . } \end{array}\tag{63}
$$

Subtracting these updates, we have:

$$
\begin{array} { r } { \hat { x } _ { t _ { n } } - \hat { x } _ { t _ { n } } ^ { c } = \hat { x } _ { t _ { n + 1 } } - \hat { x } _ { t _ { n + 1 } } ^ { c } - h _ { t _ { n + 1 } } \left( v _ { \theta } ( \hat { x } _ { t _ { n + 1 } } , t _ { n + 1 } ) - v _ { \theta } ^ { c } ( \hat { x } _ { t _ { n + 1 } } ^ { c } , t _ { n + 1 } ) \right) . } \end{array}\tag{64}
$$

We decompose the velocity difference into two components to isolate the cache reuse error:

$$
\begin{array} { r l } & { v _ { \theta } ( \hat { x } _ { t _ { n + 1 } } , t _ { n + 1 } ) - v _ { \theta } ^ { c } ( \hat { x } _ { t _ { n + 1 } } ^ { c } , t _ { n + 1 } ) = \underbrace { v _ { \theta } ( \hat { x } _ { t _ { n + 1 } } , t _ { n + 1 } ) - v _ { \theta } ( \hat { x } _ { t _ { n + 1 } } ^ { c } , t _ { n + 1 } ) } _ { \mathrm { P r o p a g a t e d ~ E r r o r } } } \\ & { \quad \quad \quad \quad \quad + \underbrace { v _ { \theta } ( \hat { x } _ { t _ { n + 1 } } ^ { c } , t _ { n + 1 } ) - v _ { \theta } ^ { c } ( \hat { x } _ { t _ { n + 1 } } ^ { c } , t _ { n + 1 } ) } _ { \mathrm { C a c h e ~ R e u s e ~ E r r o r } } . } \end{array}\tag{65}
$$

Applying the Lipschitz constant $L _ { t _ { n + 1 } }$ for the velocity ${ \pmb v } _ { \theta }$ and $L _ { \mathrm { o u t } }$ for the output layer $f _ { \mathrm { o u t } }$ with the triangle inequality, we obtain the recurrence relation:

$$
\begin{array} { r l } & { \left\| \hat { \boldsymbol { x } } _ { t _ { n } } - \hat { \boldsymbol { x } } _ { t _ { n } } ^ { c } \right\| _ { 1 } \leq \left\| \hat { \boldsymbol { x } } _ { t _ { n + 1 } } - \hat { \boldsymbol { x } } _ { t _ { n + 1 } } ^ { c } \right\| _ { 1 } } \\ & { \qquad + h _ { t _ { n + 1 } } \left\| \boldsymbol { v } _ { \boldsymbol { \theta } } ( \hat { \boldsymbol { x } } _ { t _ { n + 1 } } , t _ { n + 1 } ) - \boldsymbol { v } _ { \boldsymbol { \theta } } ( \hat { \boldsymbol { x } } _ { t _ { n + 1 } } ^ { c } , t _ { n + 1 } ) \right\| _ { 1 } } \\ & { \qquad + h _ { t _ { n + 1 } } \left\| \boldsymbol { v } _ { \boldsymbol { \theta } } ( \hat { \boldsymbol { x } } _ { t _ { n + 1 } } ^ { c } , t _ { n + 1 } ) - \boldsymbol { v } _ { \boldsymbol { \theta } } ^ { c } ( \hat { \boldsymbol { x } } _ { t _ { n + 1 } } ^ { c } , t _ { n + 1 } ) \right\| _ { 1 } } \end{array}\tag{66}
$$

$$
\begin{array} { r } { \sum \left( 1 + h _ { t _ { n + 1 } } L _ { t _ { n + 1 } } \right) \left\| \hat { { \boldsymbol x } } _ { t _ { n + 1 } } - \hat { { \boldsymbol x } } _ { t _ { n + 1 } } ^ { c } \right\| _ { 1 } + h _ { t _ { n + 1 } } L _ { 0 0 t } \left\| \epsilon _ { t _ { n + 1 } } ^ { c } \right\| _ { 1 } . } \end{array}\tag{67}
$$

Applying this relation recursively from $t _ { 1 }$ back to $t _ { N }$ yields:

$$
\begin{array} { r l } { \displaystyle \left\| \hat { \boldsymbol { x } } _ { t _ { 1 } } - \hat { \boldsymbol { x } } _ { t _ { 1 } } ^ { c } \right\| _ { 1 } \leq \left\| \hat { \boldsymbol { x } } _ { t _ { N } } - \hat { \boldsymbol { x } } _ { t _ { N } } ^ { c } \right\| _ { 1 } \prod _ { n = 1 } ^ { N - 1 } \left( 1 + h _ { t _ { n + 1 } } L _ { t _ { n + 1 } } \right) } & { } \\ { \displaystyle + \sum _ { n = 1 } ^ { N - 1 } h _ { t _ { n + 1 } } L _ { \mathrm { o u t } } \left\| \epsilon _ { t _ { n + 1 } } ^ { c } \right\| _ { 1 } \prod _ { m = 1 } ^ { n - 1 } \left( 1 + h _ { t _ { m + 1 } } L _ { t _ { m + 1 } } \right) . } \end{array}\tag{68}
$$

Since both trajectories start from the same initial noise, we have $\hat { \mathbf { x } } _ { t _ { N } } = \hat { \mathbf { x } } _ { t _ { N } } ^ { c }$ and the first term vanishes,

$$
\left\| \hat { \pmb x } _ { t _ { 1 } } - \hat { \pmb x } _ { t _ { 1 } } ^ { c } \right\| _ { 1 } \leq \sum _ { n = 1 } ^ { N - 1 } h _ { t _ { n + 1 } } L _ { \mathrm { o u t } } \left\| \pmb \epsilon _ { t _ { n + 1 } } ^ { c } \right\| _ { 1 } \prod _ { m = 1 } ^ { n - 1 } \left( 1 + h _ { t _ { m + 1 } } L _ { t _ { m + 1 } } \right) .\tag{69}
$$

Finally, using the inequality $1 + a \le e ^ { a }$

$$
\left\| \hat { \pmb x } _ { t _ { 1 } } - \hat { \pmb x } _ { t _ { 1 } } ^ { c } \right\| _ { 1 } \leq \sum _ { n = 1 } ^ { N - 1 } \left\| \epsilon _ { t _ { n + 1 } } ^ { c } \right\| _ { 1 } \exp \left( \ln \left( h _ { t _ { n + 1 } } L _ { \mathrm { o u t } } \right) + \sum _ { m = 1 } ^ { n - 1 } h _ { t _ { m + 1 } } L _ { t _ { m + 1 } } \right) .\tag{70}
$$

This completes the proof.

## C Related Work

## C.1 Diffusion Models

In recent years, diffusion models [1–4] have emerged as a dominant paradigm in generative modeling, demonstrating remarkable performance across a wide range of visual generation tasks, including image [5, 6], video [7, 8], and multimodal synthesis [33]. Early diffusion models were primarily built upon lightweight U-Net architectures [34], which enabled stable training and strong generative fidelity through iterative denoising. Subsequent advances have significantly expanded model capacity by introducing transformer-based backbones, most notably Diffusion Transformers (DiTs) [35], which facilitate better scalability and have driven substantial progress in high-resolution and long-horizon generation, especially for video synthesis. Despite their strong generative capabilities, diffusion models remain computationally expensive at inference time. The sequential denoising process, combined with increasingly large backbones and longer generation horizons, results in slow sampling speed and high computational cost [14]. These limitations pose significant challenges for practical deployment in real-world scenarios, motivating extensive research on accelerating diffusion model inference.

## C.2 Diffusion Model Acceleration

A large body of work has investigated accelerating diffusion model inference from different perspectives. A representative line of research focuses on improving the sampling solvers, aiming to reduce the number of denoising steps without modifying the underlying model. Classic approaches, such as DDIM [36], DPM-Solver [11], UniPC [13], etc., have demonstrated that carefully designed numerical solvers can significantly accelerate sampling. As a result, modern high-fidelity image and video generation models (e.g., Wan [26], OpenSora [24], Flux-dev 1.0 [27]) already adopt these advanced general-purpose solvers in practice. Nevertheless, despite such solver-level optimizations, tens of denoising steps are still required for high-quality generation, leading to substantial inference cost. Another direction explores distillation-based acceleration, where a compact student model is trained to mimic a large teacher diffusion model using fewer steps. While such methods can achieve impressive speedups, they typically require expensive retraining and large-scale supervision. This limitation becomes particularly pronounced for video diffusion models, where both training and dis tillation incur prohibitive computational overhead [37]. Model-level optimization constitutes another important category, including quantization and pruning. Quantization methods such as PTQD [9] reduce numerical precision to lower computation and memory cost, while pruning-based approaches like LD-Pruner [10] remove redundant structures from diffusion models. Although effective, these techniques often involve nontrivial engineering effort, additional calibration, or task-specific retraining, and their performance can be sensitive to model architecture and deployment settings. More recently, cache-based acceleration has emerged as a promising and complementary direction [14–16]. These methods exploit temporal redundancy in the reverse diffusion process by reusing intermediate representations or residuals across adjacent timesteps, without modifying the solver or retraining the backbone model. As such, cache-based techniques are largely orthogonal to solver design and can be seamlessly applied on top of existing solver-based samplers (e.g., DDIM/UniPC), enabling further acceleration in a plug-and-play manner. By reusing computation from previous denoising steps, cache-based approaches can substantially reduce inference cost while preserving generation quality, which is particularly attractive for large-scale and high-fidelity image and video diffusion models.

## D Experimental Details

## D.1 Evaluation Prompts

For video generation, we follow prior work [15, 16] and evaluate all methods using the official 946 prompts provided by VBench [31]. These prompts cover a diverse set of content categories and motion patterns, and are designed to comprehensively assess video generation quality across multiple dimensions.

For image generation, we adopt the official COCO validation set [32] and use the first 30K text prompts as commonly done in recent diffusion acceleration studies [15, 16]. The prompt list is publicly available and released on Hugging Face to facilitate reproducibility. All compared methods are evaluated on the same prompt sets to ensure a fair comparison.

## D.2 Evaluation Metrics

We employ three evaluation metrics to assess generation quality and fidelity.

• VBench [31] serves as a holistic benchmarking framework for video generative models. It utilizes a hierarchical Evaluation Dimension Suite to disentangle the multifaceted nature of “video quality” into distinct, well-defined metrics, thereby enabling a granular and objective assessment of generative performance.

• LPIPS [22] quantifies perceptual similarity using deep feature representations, capturing subtle texture-level and semantic deviations.

• PSNR and SSIM measure pixel-level and structure-level fidelity, respectively, between outputs produced by the accelerated sampler and those generated by the corresponding base model.

## D.3 Training Details

In this section, we provide the technical details for the bilevel optimization process of GCache. The optimization is designed to be efficient while ensuring the generalizability of the learned propagation exponent.

Dataset and Sampling Strategy. We utilize a training set consisting of 512 prompts randomly sampled from the corresponding dataset. To ensure the robustness of the learned policy and avoid overfitting, we strictly ensure that the random seeds used during training are distinct from those used in the evaluation and testing phases. During each iteration of the outer optimization, we evaluate the current caching policy using a batch size of 32 prompts to obtain a stable estimate of the perceptual loss (LPIPS and SSIM).

Bayesian Optimization Configuration. We employ a Gaussian Process (GP) as the surrogate model for the outer objective. The optimization parameters and strategies are configured as follows:

• Search Space: The learnable coefficients s for the Bernstein polynomial are bounded within the range $[ s _ { \mathrm { m i n } } , s _ { \mathrm { m a x } } ] = [ 0 , 1 0 ]$ ]. For our experiments, we set the polynomial degree $d = 3 .$ , resulting in a 4-dimensional parameter space $( d \bar { + } 1 = 4 )$

• Structured Initialization: Rather than random sampling, we initialize the GP surrogate using 16 deterministic points to ensure comprehensive coverage of the search space. Specifically, we define two representative centers {0.25, 0.75} in the normalized parameter space for each dimension and generate the initial set via a Cartesian product across all 4 dimensions $( 2 ^ { 4 } = 1 6 )$ . These normalized coordinates are then linearly mapped to the actual range [0, 10]. This symmetric grid initialization ensures that the GP begins with a well-distributed understanding of the objective landscape across different quadrants.

• Optimization Steps: The total optimization budget is set to 500 steps.

• Exploration-Exploitation Trade-off: We utilize the Lower Confidence Bound (LCB) acquisition function: $A _ { \mathrm { L C B } } ( \mathbf { s } ) = \mu ( \mathbf { s } ) - \kappa \sigma ( \mathbf { s } )$ . The exploration parameter κ is scheduled to decay linearly from 2.576 to 1.0 over the course of training. This encourages the optimizer to prioritize global exploration in early iterations and transition towards local exploitation of identified high-quality regions in the later stages.

Efficient Error Pre-calculation. To minimize the computational overhead during the bilevel optimization, we implement an efficient evaluation strategy for the inner objective. Specifically, we pre-calculate the local approximation errors $\epsilon _ { i + 1 } ^ { c }$ across 128 representative samples before the start of the optimization. Since the inner objective (the Dynamic Programming solver) only requires the magnitude of these local errors to calculate the total propagated error $\sum \| \epsilon \| _ { 1 } e ^ { w ( t ; s ) }$ , pre-storing these values allows the optimization process to bypass redundant model evaluations. During each step of the bilevel search, we simply scale the stored error values by the updated propagation exponent e<sup>w(t;s)</sup>. This decoupling of error measurement from policy search significantly accelerates the optimization, reducing the search time from hours to minutes on a single GPU. Furthermore, to verify the fidelity of these pre-computed proxies, we evaluate the alignment between errors from the original and cache-reused trajectories in Appendix E.4. The results justify the use of pre-computed errors as a reliable and high-fidelity proxy for policy optimization.

## D.4 Optimization Efficiency

Table 5 reports the optimization time and associated budgets for GCache across different architectures. Experiments were performed using 4 × H100 GPUs for most models, with ${ \mathrm { ~ 8 ~ } } \times { \mathrm { ~ H 1 0 0 ~ } }$ GPUs reserved for Wan2.1-1.3B. Crucially, GCache demonstrates remarkable efficiency: even for the most computationally demanding backbones, the optimization is finalized in less than a day. This minimal overhead makes our approach highly scalable and suitable for the rapid deployment of new diffusion backbones.

Table 5: We report the budget K and the wall-clock time required. All policies were optimized in less than 24 hours, demonstrating the high efficiency of GCache.
<table><tr><td>Model</td><td>GPU Config</td><td>Budget (K)</td><td>Time Cost (hours)</td></tr><tr><td rowspan="2">OpenSora 1.2</td><td rowspan="2"> $4 \times \mathrm { H 1 0 0 }$ </td><td>18</td><td>17</td></tr><tr><td>11</td><td>12.5</td></tr><tr><td rowspan="2">CogVideoX-2B</td><td rowspan="2"> $4 \times \mathrm { H 1 0 0 }$ </td><td>31</td><td>7</td></tr><tr><td>17</td><td>5</td></tr><tr><td rowspan="2">Wan2.1-1.3B</td><td rowspan="2"> $8 \times \mathrm { H } 1 0 0$ </td><td>24</td><td>17.5</td></tr><tr><td>16</td><td>13.5</td></tr><tr><td rowspan="2">Flux-dev 1.0</td><td rowspan="2"> $4 \times \mathrm { H 1 0 0 }$ </td><td>14</td><td>6</td></tr><tr><td>10</td><td>5</td></tr></table>

Table 6: Cross-distribution generalization analysis. We evaluate GCache policies optimized on different prompt subsets (Static vs. Dynamic). The negligible performance gap across training distributions demonstrates the robust generalization of our learned policy.
<table><tr><td>Test Set</td><td>Train Source</td><td>LPIPS↓</td><td>SSIM↑</td><td>PSNR↑</td></tr><tr><td rowspan="3">Static</td><td>Mixed</td><td>0.0594</td><td>0.9122</td><td>30.25</td></tr><tr><td>Static</td><td>0.0595</td><td>0.9118</td><td>30.26</td></tr><tr><td>Dynamic</td><td>0.0611</td><td>0.9109</td><td>30.26</td></tr><tr><td rowspan="3">Dynamic</td><td>Mixed</td><td>0.0905</td><td>0.8982</td><td>28.04</td></tr><tr><td>Static</td><td>0.0905</td><td>0.8981</td><td>28.06</td></tr><tr><td>Dynamic</td><td>0.0921</td><td>0.8975</td><td>27.90</td></tr></table>

## D.5 Experimental Details for Main Paper Visualizations

In this section, we provide the specific experimental configurations used to generate the visualizations in the main manuscript:

• Figure 1: Results are evaluated using the Flux-dev 1.0 [27] backbone. Metrics are averaged over 32 randomly sampled prompts from the COCO [32] validation set.

• Figure 2: The error propagation analyses are derived from CogVideoX-2B [25], with results averaged across 32 representative prompts from the VBench [31] suite.

• Figure 3: The policy comparison study is conducted on Flux-dev 1.0 [27], utilizing 32 random prompts from the COCO [32] validation set for statistical consistency.

## E Additional Experiment Results

## E.1 Robustness to Prompt Distribution Shifts

To evaluate the generalization capability of GCache under distribution shifts, we investigate whether a policy optimized on a specific prompt characteristic (e.g., static scenes) can generalize to others (e.g., highly dynamic videos). Using VBench [31] as a base, we curate two distinct subsets:

• Static Prompts: Scenes with minimal temporal evolution, identified by keywords such as “in a still frame”, “a tranquil tableau”, “frozen in time”, “static view.”

• Dynamic Prompts: Motion-intensive sequences sampled from VBench’s motion-related dimensions, including human action, dynamic degree, motion smoothness, and subject consistency.

We optimize GCache on three training distributions: Static, Dynamic, and a Mixed (Random) set, and evaluate their cross-distribution performance. All of the experiments are conducted on CogVideoX-2B with K = 17.

As shown in Table $^ { 6 , }$ the performance variance across different training sources is remarkably marginal. For instance, a policy trained on static prompts performs almost identically to one trained on dynamic prompts when tested on dynamic sequences (0.0905 vs. 0.0921 LPIPS). This high degree of stability suggests that GCache captures fundamental structural redundancies within the diffusion process that are invariant to specific prompt semantics or motion levels, ensuring its robustness for diverse real-world applications.

Table 7: Zero-shot generalization across resolutions. We evaluate the GCache-fast policy (optimized at 1024 × 1024) on lower resolutions (512 and 256) without re-tuning. GCache-fast consistently outperforms ERTACache across all scales, demonstrating its robustness to spatial resolution shifts.
<table><tr><td>Resolution</td><td>Method</td><td>LPIPS↓</td><td>SSIM↑</td><td>PSNR↑</td></tr><tr><td rowspan="2">1024</td><td>ERTACache</td><td>0.2658</td><td>0.7863</td><td>20.60</td></tr><tr><td>GCache-fast</td><td>0.1825</td><td>0.8423</td><td>23.76</td></tr><tr><td rowspan="2">512</td><td>ERTACache</td><td>0.2359</td><td>0.7580</td><td>19.97</td></tr><tr><td>GCache-fast</td><td>0.1514</td><td>0.8344</td><td>23.47</td></tr><tr><td rowspan="2">256</td><td>ERTACache</td><td>0.2047</td><td>0.7209</td><td>19.70</td></tr><tr><td>GCache-fast</td><td>0.1558</td><td>0.7746</td><td>21.74</td></tr></table>

Table 8: Policy comparison under identical budget K. We compare GCache with ERTACache and its variant ERTACache\* (without the error rectification module). GCache consistently achieves superior performance without requiring any auxiliary corrector models.
<table><tr><td>Model</td><td>K</td><td>Method</td><td>LPIPS↓</td><td>SSIM↑</td><td>PSNR↑</td></tr><tr><td rowspan="3">OpenSora 1.2</td><td rowspan="3">12</td><td>ERTACache*</td><td>0.2198</td><td>0.7648</td><td>20.24</td></tr><tr><td>ERTACache</td><td>0.1659</td><td>0.8170</td><td>22.34</td></tr><tr><td>GCache</td><td>0.1188</td><td>0.8569</td><td>25.73</td></tr><tr><td rowspan="3">CogVideoX-2B</td><td rowspan="3">17</td><td>ERTACache*</td><td>0.1090</td><td>0.8687</td><td>26.54</td></tr><tr><td>ERTACache</td><td>0.1012</td><td>0.8702</td><td>26.44</td></tr><tr><td>GCache</td><td>0.0721</td><td>0.9042</td><td>29.14</td></tr></table>

## E.2 Generalization Across Resolutions

To assess the spatial scalability of GCache, we evaluate the transferability of a policy optimized at a fixed resolution to unseen spatial scales. Specifically, we apply the GCache-fast policy, originally optimized for 1024 × 1024 resolution on Flux-dev 1.0 [27], directly to 512 × 512 and 256 × 256 settings without any further re-tuning or adaptation.

As shown in Table 7, GCache-fast consistently outperforms the baseline ERTACache [16] by a significant margin across all tested resolutions. Notably, the performance gain remains robust even when the resolution is quadrupled (256 → 1024), underscoring that GCache captures resolutionagnostic redundancy patterns within the diffusion backbone. This zero-shot transfer capability is highly desirable for practical deployment, as it eliminates the need for resolution-specific optimization.

## E.3 Effectiveness of the Learned Policy

A key question is whether GCache’s performance gains stem from a superior reuse policy or simply from different computational budgets. Some existing methods, such as ERTACache [16], employ an auxiliary light-weight model to rectify the errors introduced by cache reuse. To ensure a fair comparison and isolate the impact of the policy itself, we evaluate GCache against two variants of ERTACache under identical budget K: (1) ERTACache, the full version with its Time Adjustment and Error Rectification modules; and (2) ERTACache\*, a stripped-down version that excludes these additional modules, relying solely on its default reuse policy.

As summarized in Table 8, GCache consistently outperforms both variants by a significant margin across multiple backbones. Notably, GCache achieves substantially better LPIPS and PSNR than the full ERTACache, despite not using any extra corrector models. For instance, on OpenSora 1.2, GCache improves PSNR from 22.34 to 25.73. These results demonstrate that GCache’s optimizationbased approach discovers a much more effective reuse trajectory, proving that a well-optimized policy can be more powerful than a sub-optimal policy supplemented by error correction.

![](images/cfc169b18b9769a626eeffacaeb2134711819707f1e3bbd18cca2964cd759d25.jpg)  
Figure 6: Validation of the pre-computed error proxy. We compare the L1 residual error $\| \delta ^ { c } - \delta _ { t _ { i } } \| _ { 1 }$ <sub>1</sub> during the denoising process on CogVideoX-2B $( K = 1 7 )$ . The Pre-compute (red) curve denotes the error calculated from the original trajectory, while the Cache (blue) curve represents the actual error in a trajectory with GCache reuse. The “sawtooth” peaks correspond to the K cache refresh points. The high degree of overlap between the two curves justifies the use of pre-computed error E as a high-fidelity and efficient proxy for optimization.

## E.4 Validation of the Pre-computed Local Error Proxy

As discussed in Section D.3, to circumvent the prohibitive computational overhead of generating online trajectories during optimization, GCache utilizes a pre-computed error matrix $\bar { E _ { i , j } } = \lVert \delta _ { t _ { i } } \bar { - }$ $\delta _ { t _ { j } } \| _ { 1 }$ <sub>1</sub> derived from the original (full-step) diffusion trajectory. However, a potential concern is trajectory drift: as cache reuse is introduced, the intermediate latent states may deviate from the original path, potentially rendering the pre-computed errors inaccurate.

To investigate the fidelity of this approximation, we conduct an empirical study on CogVideoX-2B $( K = 1 7 )$ . We compare the local residual error, defined as $\lVert \delta ^ { c } - \delta _ { t _ { i } } \rVert _ { 1 }$ , where $\dot { \delta _ { t _ { i } } }$ is the ground-truth residual at timestep $t _ { i }$ . We calculate this error under two settings: (1) the Original Trajectory, where $\pmb { \delta } ^ { c }$ is sampled from the full-step inference; and (2) the Cache-reused Trajectory, where $\pmb { \delta } ^ { c }$ is sampled from an actual inference process governed by our learned GCache policy.

The results are visualized in Figure 6. As shown, the error profiles of the two trajectories are remarkably aligned across the entire denoising process. The “sawtooth” pattern reflects the periodic refreshing of the cache, where the error peaks just before a cache update and drops to zero immediately after. Crucially, the mean and variance of the errors in the cache-reused trajectory (blue) closely track those of the pre-computed proxy (red), even in the late stages of denoising where drift is typically most pronounced. This high degree of alignment justifies the use of pre-computed errors as a reliable and efficient proxy for optimization, as it accurately reflects the error dynamics of the actual inference process.

## F Additional Qualitative Comparison

We provide additional qualitative results for both image and video generation.

## F.1 Image Generation

For image generation, Figure 7 presents additional qualitative comparisons on Flux-dev 1.0. Under the same acceleration setting, ERTACache introduces various visual degradations, including background misalignment, incorrect spatial relationships between objects and subjects, missing or distorted materials, and semantic artifacts such as erroneous objects (e.g., incorrect traffic signs) and physically implausible details (e.g., airplanes depicted mid-air with landing gear deployed). By contrast,

GCache remains highly faithful to the original model outputs, preserving semantic correctness, spatial coherence, and fine-grained visual details even under a 2.87× speedup setting.

## F.2 Vedio Generation

For video generation, we provide additional qualitative comparisons across three representative text-to-video diffusion models, including CogVideoX-2B (Figure 8), Open-Sora 1.2 (Figure 9), and Wan 2.1-1.3B (Figure 10). For each example, we uniformly sample six frames from the generated video sequence to visualize temporal consistency and structural fidelity over time. Within each group, we compare outputs from ERTACache, GCache, and the ground-truth full-step original generation. Across all three models and diverse prompts, GCache consistently produces videos that remain highly faithful to the ground-truth outputs, preserving semantic correctness, object structure, and temporal coherence throughout the sequence. The generated contents exhibit stable object appearances and natural motion transitions, with minimal degradation under accelerated sampling. In contrast, ERTACache frequently introduces noticeable visual artifacts and semantic inconsistencies. Typical failure cases include distorted object geometry (e.g., malformed cups, umbrellas, and sharks), incorrect semantic structures (e.g., unrealistic astronaut body configurations), and unstable object appearances across frames. These artifacts become more evident in dynamic scenes and complex compositions, indicating weaker preservation of global semantic information. Overall, the qualitative results further demonstrate that GCache achieves substantially better fidelity and temporal consistency under the same acceleration budget.

## G Limitations

Although GCache effectively optimizes cache policies, it currently operates under a fixed refresh budget and relies on pre-computed error proxies, which may not fully account for dynamic input complexity or extreme trajectory shifts. Future work could explore sample-adaptive scheduling and more diverse perceptual objectives to further enhance temporal consistency in highly dynamic videos.

## H Broader Impact

By substantially reducing the computational cost of large-scale diffusion models, GCache promotes environmental sustainability and democratizes access to state-of-the-art generative tools for users with limited hardware resources. While accelerated generation could potentially be misused for creating synthetic misinformation, we advocate for its deployment alongside robust safety filters and digital watermarking technologies to mitigate such risks.

![](images/943ab03b441efedd22984ad133bd92132f18cc6ffe7e1e56da3ab88639075892.jpg)  
Figure 7: Additional qualitative comparison results for image generation on Flux-dev 1.0. Best viewed zoomed in.

![](images/4d3a3a59cac8a3bed5678b6644723eee0d9e4502af39e3e5883fa22944d83793.jpg)  
(a) A cup and a couch.

![](images/7a22bbe6254879e4cc4a72e17d3b6a9ce56de2879f32da3d549e6aa08a608015.jpg)  
(b) An astronautflying in space, in super slow motion.

![](images/154724e995d7f93f4f1176539596f15ed817e7cbbc4ef159cdeefa7d7ec5770f.jpg)  
(c) Fireworks.

![](images/306f2727eac83bb3efd23db61b563b31e49bae7e16d3ccf83245af3edaf4e099.jpg)  
(d) A beautiful coastal beach in spring, waves lapping on sand, racking focus.  
Figure 8: Temporal consistency comparison on CogVideoX-2B. Each group visualizes six evenly spaced frames from the generated sequence.

![](images/3f77415dcc7dd806e79fe2c72130da96bc232184e1ab707fa0f4168fc9e6d2f2.jpg)  
(a) Gwen Stacy reading a book, in cyberpunk style.

![](images/e60a98e6f294c2754c63c0b2bb55ac762e3bae78f534b27c28aff0e7cd63970b.jpg)  
(b) a cell phone.

![](images/5e83f3a3b2bf7057cc2f8bc2393082f0c0272fa19b8fdbc1ce4294a2974d0b5a.jpg)  
(c) A couple informal evening wear going home get caught in a heavy downpour with umbrellas, in super slow motion.

![](images/b000c1df8b6affd370b7f9ddc67b2c95ec340e36e6f1d31b2de33b8fef0c8ae9.jpg)  
(d) a shark is swimming in the ocean, oil painting  
Figure 9: Temporal consistency comparison on Open-Sora 1.2. Each group visualizes six evenly spaced frames from the generated sequence.

![](images/3a5184e8de08976572202069bc5cb5ad23b8352b7f17f0e4718c8de61328cbcc.jpg)  
(a) A bear climbing a tree

![](images/7a1a587a1f5809f624f1133687412c9b78a655ee1d6be840b0fccb2bde891a5c.jpg)  
(b) A panda drinking coffee in a cafe in Paris, pan left.

![](images/2188b54e165618cfb91957c6e4d0ee7c4fa77f5907b4b422aa3c8d83a6dd5f82.jpg)  
(c) A tranquil tableau of bathroom.

![](images/37dacd4f889077477052f278fb0d6cbaceb930d978893ef9a620f87fb7fc156d.jpg)  
(d) Skis and a snowboard.  
Figure 10: Temporal consistency comparison on Wan 2.1-1.3B. Each group visualizes six evenly spaced frames from the generated sequence.