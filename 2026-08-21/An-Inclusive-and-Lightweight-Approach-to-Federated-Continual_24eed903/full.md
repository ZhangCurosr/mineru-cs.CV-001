# An Inclusive and Lightweight Approach to Federated Continual Learning for Cultural Heritage

Ioannis Theologitis<sup>1</sup>, Debin Meng<sup>2</sup>, Stylianos Eleftheriadis<sup>1</sup>, Vasileios Lolis<sup>1</sup>, Konstantinos Votis<sup>1</sup>

<sup>1</sup>Information Technologies Institute, Centre for Research and Technology Hellas, Thessaloniki, Greece

<sup>2</sup>School of Electronic Engineering and Computer Science, Queen Mary University of London, UK

{theolo, stylelev, vaslwlis, kvotis}@iti.gr

debin.meng@qmul.ac.uk

This is the accepted author manuscript of a paper accepted at the 2026 IEEE International Conference on Cyber Humanities (IEEE-CH 2026), Venice, Italy, September 7–9, 2026.

© 2026 IEEE. Personal use of this material is permitted. Permission from   
IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or   
redistribution to servers or lists, or reuse of any copyrighted component of this work in other works.

Abstract—Artificial intelligence can support cultural heritage and digital humanities through large-scale retrieval and analysis of digitized collections. However, cultural heritage data are often distributed across institutions, constrained by ownership and access restrictions, and continuously evolving over time. Federated Continual Learning (FCL) is well suited to this setting, as it enables models to learn from distributed and sequential data without sharing raw collections. In this paper, we propose FedCurv-DR, a lightweight, regularisation-based FCL strategy. The method accumulates parameter-importance estimates across clients and experiences to protect learned knowledge, while updating them only at fixed intervals to minimize communication and computation overhead. We evaluate FedCurv-DR in a continual learning scenario using the WikiArt image dataset for genre classification with evolving styles, reporting performance, energy, and fairness metrics. Our results show that FedCurv-DR reduces forgetting and balances performance, fairness, and energy efficiency for sustainable AI in cultural heritage.

Index Terms—federated continual learning, cultural heritage, wikiart, image classification, energy efficiency, fairness, privacy

## I. INTRODUCTION

AI is increasingly adopted across many domains, including cultural heritage [1]. However, cultural heritage data raise specific ethical and technical challenges. Collections are often distributed across institutions and continuously evolve through digitisation. Digitisation processes may also reproduce existing societal and cultural biases, including the under-representation or misrepresentation of minority groups. These biases can then be inherited or amplified by AI pipelines trained on such collections [2], [3]. Such bias can derive from unequal access to cultural heritage data, since ownership, copyright, licensing, and privacy constraints can limit the availability of collections for computational research and AI development [4]. At the same time, the unequal capacity of institutions to participate in digitisation and AI adoption can further reinforce this bias, as many cultural heritage organisations face limited funding, insufficient technical expertise, and resource-intensive digitisation requirements [5].

These challenges highlight the need for AI solutions that are green, trustworthy and inclusive - core principles of the Cyber Humanities vision, which calls for cultural heritage technologies that combine technical innovation with ethical governance and equitable participation [6], [7]. Federated Learning (FL) enables collaborative model training without sharing raw data [8], while Continual Learning (CL) enables models to adapt to sequentially arriving data without retraining from scratch. Their combination, Federated Continual Learning (FCL), can support green AI objectives and is well suited to cultural heritage settings [9]. As shown in Fig. 1, art institutions across the world can collaboratively train AI models on their private and evolving collections while respecting data sensitivity. However, to ensure equal participation in this process, energy efficiency and fairness must be considered when deploying such systems.

In this work, we propose a lightweight, exemplar-free FCL framework for cultural heritage image classification. We introduce FedCurv-DR, a regularisation-based strategy that periodically aggregates parameter-importance information across clients to adapt to new data, or experiences, while preserving performance on past data without relying on replay buffers or computationally intensive methods such as synthetic data generation. We evaluate FedCurv-DR on WikiArt, a large-scale public art dataset relevant to cultural heritage and digital humanities. By treating genre classification as the fixed objective and introducing artistic styles sequentially as new experiences, we model a realistic domain-incremental scenario where digitised cultural collections evolve over time.

The main contributions of this paper are:

• We propose FedCurv-DR, a lightweight regularisationbased FCL strategy suitable for cultural heritage.

• We evaluate the method on the public WikiArt image dataset for genre classification with style-based continual experiences.

• We report performance, energy, and fairness metrics to evaluate the sustainability of our method in accordance with Green AI principles [10].

![](images/ca965349985e4b8e930b25148fd1df178161efe04fbbeeb7213b25e2f6be207b.jpg)  
Fig. 1. Overview of the federated continual learning process. Art institutions across Europe collaborate on training a single global model for image classification on their evolving collections, without sharing private data. Institutions periodically send model updates, rather than raw data, to a server operated by a trusted coordinating party, which aggregates them into a shared global model.

## II. RELATED WORKS

## A. Federated Learning

In Federated Learning (FL), multiple clients collaboratively train a shared global model under the coordination of a central server, while keeping their data local and private. Instead of exchanging raw data, clients train models locally and periodically send model updates to the server for aggregation. The serverside aggregation algorithms used to combine these updates are commonly referred to as FL strategies. A representative baseline FL strategy is FedAvg [8], which constructs the global model by averaging client parameters weighted according to the sizes of the local datasets.

Despite its advantages for privacy preservation and scalability, FL introduces several key challenges. In real-world settings, client data are typically not independent and identically distributed (non-IID). In cultural heritage, for example, the number of artifacts may vary substantially between national museums and local galleries, while semantic distributions may differ across styles, periods, or art movements. This heterogeneity can cause local models to diverge during training, as each client optimizes toward its own data distribution, thereby hindering the convergence of the global model. This phenomenon is commonly referred to as inter-client drift [11]. Several FL strategies have been proposed to address this challenge under heterogeneous data conditions [12]–[16].

However, while many of these methods aim to improve performance under non-IID conditions, they may still favor larger or more representative clients, leading to models that perform better for dominant data distributions while neglecting under-represented participants. This can introduce or reinforce fairness concerns, particularly when performance disparities across clients are not explicitly reported. Several works have therefore investigated fairness in FL and proposed fairnessaware strategies to mitigate client-level bias [17]–[21].

## B. Continual Learning

Continual Learning (CL) addresses the problem of learning from a stream of tasks or data distributions over time, without retraining from scratch. A central challenge in this setting is catastrophic forgetting [22], where the model rapidly loses previously acquired knowledge when trained on new tasks. To mitigate this, several families of methods have been proposed. Regularization-based approaches such as Elastic Weight Consolidation (EWC) [22], Memory Aware Synapses (MAS) [23], and Synaptic Intelligence (SI) [24] estimate the importance of model parameters for past tasks and penalize changes to those parameters during subsequent training. Alternatively, replaybased methods [25], [26] maintain a buffer of past examples, allowing the model to rehearse previous knowledge while learning new information. Replay methods have been shown to outperform regularization-based methods in various continual learning scenarios [27], but they raise scalability and privacyrelated issues, as past data need to be stored.

## C. Federated Continual Learning

Federated Continual Learning (FCL) combines the challenges of Federated Learning and Continual Learning, requiring models to simultaneously cope with data heterogeneity across clients and the sequential arrival of new data or experiences. In this setting, both inter-client drift and catastrophic forgetting must be mitigated, making the learning process significantly more complex. Several approaches have been proposed to address these challenges [28]–[30].

Despite their effectiveness, many FCL methods introduce additional computational and communication overhead, for example through synthetic data generation [30]. These factors can significantly increase energy consumption, particularly in large-scale or resource-constrained deployments, and may even make such methods infeasible in practice. This is a critical consideration for an inclusive cultural heritage AI ecosystem. In contrast to distillation- or generation-based FCL methods, our approach does not require replay data, memory buffers, or computationally intensive mechanisms. Instead, it relies on parameter-importance regularization with reduced communication and computational overhead.

## III. METHODS

In order to achieve lightweight and exemplar-free federated continual learning, we enable continual learning during local training. This led us to explore FedCurv [14], combined with recent findings on the EWC strategy [31]. Our method reduces the communication and computation costs of the original FedCurv strategy and repurposes it to mitigate catastrophic forgetting.

## A. Elastic Weight Consolidation

Elastic Weight Consolidation (EWC) was proposed to mitigate catastrophic forgetting [22]. In this method, after training on each experience, a Fisher information matrix is calculated using the current training dataset. This matrix represents the importance of each model parameter for the corresponding experience. The diagonal Fisher importance for parameter $\theta _ { i }$ after experience t is approximated as

$$
F _ { t , i } = \frac { 1 } { | \mathcal { D } _ { t } | } \sum _ { ( \mathbf { x } , y ) \in \mathcal { D } _ { t } } \left( \frac { \partial } { \partial \theta _ { i } } \log p _ { \theta } ( y | \mathbf { x } ) \right) ^ { 2 } .\tag{1}
$$

In subsequent experiences, this matrix is used to compute a penalty term for the training loss. The EWC objective is defined as

$$
\mathcal { L } _ { \mathrm { E W C } } ( \pmb { \theta } ) = \mathcal { L } _ { \mathrm { t a s k } } ( \pmb { \theta } ; \mathcal { D } _ { t } ) + \frac { \lambda } { 2 } \sum _ { i } F _ { t - 1 , i } \left( \theta _ { i } - \theta _ { t - 1 , i } ^ { * } \right) ^ { 2 } ,\tag{2}
$$

where $\mathcal { L } _ { \mathrm { t a s k } }$ is the standard task loss, λ controls the strength of the regularization, and $\pmb { \theta } _ { t - 1 } ^ { * }$ denotes the model parameters after learning the previous experience. This term protects previously learned knowledge by penalizing changes to important model parameters, encouraging the optimizer to update less important parameters when learning future experiences.

In the original version of EWC, a different Fisher matrix is calculated for each experience, which can introduce scalability issues. Online EWC improves memory efficiency by aggregating Fisher matrices using a decay factor that controls the contribution of past experiences to the current penalty:

$$
\bar { F } _ { t , i } = \beta \bar { F } _ { t - 1 , i } + F _ { t , i } ,\tag{3}
$$

where $\bar { F } _ { t , i }$ is the accumulated importance estimate and $\beta \in$ $[ 0 , 1 ]$ is the decay factor.

## B. FedCurv

FedCurv [14] adapts regularization-based continual learning strategies, such as EWC, to the federated learning setting. However, it is designed to address client heterogeneity rather than catastrophic forgetting. FedCurv proposes summing Fisher matrices across clients after each training round into a single global Fisher matrix, which is then used to constrain local updates toward a global consensus and mitigate client drift.

Let S be the set of clients selected at communication round $r ,$ and let $F _ { k , i } ^ { r }$ denote the Fisher importance of parameter $\theta _ { i }$ computed by client k. The server aggregates the client importances as

$$
F _ { G , i } ^ { r } = \sum _ { k \in S _ { r } } F _ { k , i } ^ { r } .\tag{4}
$$

Clients use this global Fisher matrix during local training to apply the EWC penalty in (2). In this way, they avoid changing parameters that are important for other clients’ distributions, helping the model achieve stable convergence in non-IID settings.

## C. Logit Inversion

In recent work [31], the authors propose EWC-DR, an improvement to the Fisher matrix calculation in EWC. They observe that, when the model assigns high confidence to the correct class, the standard Fisher estimate in (1) may become small, leading to under-protection of parameters that are nevertheless important. To address this issue, they propose inverting the logits before computing the importance estimates. Let ${ \bf z } _ { \theta } ( { \bf x } )$ denote the logits of the model. The inverted logits are defined as $\tilde { \mathbf { z } } _ { \theta } ( \mathbf { x } ) = - \mathbf { z } _ { \theta } ( \mathbf { x } )$ and the corresponding predictive distribution is $\tilde { p } _ { \pmb { \theta } } ( y | \mathbf { x } ) = \mathrm { s o f t m a x } \left( \tilde { \mathbf { z } } _ { \pmb { \theta } } ( \mathbf { x } ) \right) _ { y }$ . The Fisher importance is then computed using this modified distribution:

$$
\tilde { F } _ { t , i } = \frac { 1 } { | \mathcal { D } _ { t } | } \sum _ { ( \mathbf { x } , y ) \in \mathcal { D } _ { t } } \left( \frac { \partial } { \partial \theta _ { i } } \log \tilde { p } _ { \theta } ( y | \mathbf { x } ) \right) ^ { 2 } .\tag{5}
$$

As a result, (5) assigns higher importance values to parameters associated with highly confident predictions.

## D. Our Method (FedCurv-DR)

We adapt FedCurv from a client-drift regularization method into a federated continual learning method. We make three main modifications. First, we incorporate the EWC-DR improvement into the calculation of parameter importances. Each client computes $\tilde { F } _ { k , i } ^ { r }$ using (5). Second, we modify the importance aggregation logic by accumulating importances not only across clients, but also across experiences. We also include a decay factor to reduce redundant protection of outdated experiences. The global importance estimate is updated as

$$
\bar { F } _ { G , i } ^ { r } = \beta \bar { F } _ { G , i } ^ { r - 1 } + \sum _ { k \in S _ { r } } \tilde { F } _ { k , i } ^ { r } ,\tag{6}
$$

where $\bar { F } _ { G , i } ^ { r }$ is the accumulated global importance of parameter $\theta _ { i }$ at round $r ,$ and $\beta$ is the decay factor. Third, we introduce an interval term I, following the example of [15], that controls how often importances are calculated and transmitted, reducing communication and computation costs.

The pseudocode for our method is shown in Alg. 1. It is worth noting that, for $\beta ~ = ~ 0$ and $I \ = \ 1$ , our method is equivalent to the aggregation logic of the original FedCurv.

```latex
Algorithm 1 FedCurv-DR
Input: rounds R, clients K, interval I, regularization
strength $\lambda ,$ decay factor $\beta ,$ datasets $\{ \mathcal { D } _ { k , e } \}$ , initial model
$\theta _ { G } ^ { 0 }$
Output: final global model $\pmb { \theta } _ { G } ^ { R } .$
Server Executes:
1: Initialize $\bar { \mathbf { F } } _ { G } ^ { 0 } \gets \mathbf { 0 }$
2: for each experience e: do
3: for $r = 1$ to R do
4: Select clients $S _ { r }$
5: Send $\theta _ { G } ^ { r - 1 }$ to clients in $S _ { r }$
6: for each client $k \in S _ { r }$ in parallel do
7: if $\bar { \mathbf { F } } _ { G }$ was updated in $r - 1$ round then
8: $\theta _ { K } ^ { r } , \bar { F } _ { K } ^ { r } \gets$ ClientUpdate $( \theta _ { G } ^ { r - 1 } , \bar { F } _ { G } )$
9: else
10: $\theta _ { K } ^ { r } , \bar { F } _ { K } ^ { r }$ ← ClientUpdate $( \theta _ { G } ^ { r - 1 } )$
11: end if
12: end for
13: $\begin{array} { r } { \pmb { \theta } _ { G } ^ { r }  \sum _ { k \in S _ { r } } \frac { | \mathcal { D } _ { k , e } | } { \sum _ { j \in S _ { r } } | \mathcal { D } _ { j , e } | } \pmb { \theta } _ { k } ^ { r } } \end{array}$
14: if r mod $I = 0 ^ { \circ }$ then
15: update $\bar { F } _ { G }$ using Eq. (6)
16: end if
17: end for
18: end for
ClientUpdate: $( \theta _ { G } ^ { R } , \bar { F } _ { G } )$
1: if $\bar { F } _ { G }$ was received then
2: $\bar { F } _ { K } ^ { r }  \bar { F } _ { G }$
3: end if
4: optimize Eq. (2) on local Data $D _ { k , e }$ to get updated local
model $\theta _ { k } ^ { r }$
5: if r mod $I = 0$ then
6: calculate new $F _ { k } ^ { r }$ using Eq. (5) and transmit to server.
7: end if
```

Exp.0: Realism Exp.1: Expressionism   
1000   
3000   
2500   
2000 1000   
1500   
Classes - Genres   
400 cityscape   
200 genre\_painting   
illustration   
0 2 0 2 landscape   
Exp.2: Symbolism Exp.3: Naive Art Primitivism nude\_painting   
portrait   
religiouspainting   
800 300 still\_life   
600   
200   
400   
100   
200   
0 1 2 0 1 2  
Fig. 2. Visualisation of the dataset partitioning with Dirichlet partitioner (α = 1) among three clients for each experience (style).

## IV. EVALUATION

## A. Dataset Preparation

Artistic styles evolve over time, introducing natural domain shifts in real-world visual data. This requires AI models to adapt to emerging styles while retaining previously learned knowledge, making it a practical continual learning problem. Motivated by this scenario, we evaluate our method on WikiArt, a large-scale digital art dataset spanning diverse artists, genres, and artistic styles. We use the refined version introduced by Tan et al. [32], accessed through the Hugging Face Datasets Hub. The dataset contains paintings from 11 genres, such as landscape and portrait, and 27 styles, such as Impressionism and Realism.

We define a genre classification task and construct a domain-incremental scenario by sequentially introducing artistic styles as experiences, while keeping the genre classification task fixed. WikiArt has a sparse style distribution across genres, with many styles missing certain genres. To create a clear domain-incremental scenario and ensure adequate samples per client and per experience, we drop two genres: Unknown Genre, due to vague semantics, and Abstract Painting, as it is mainly represented by a single style. We then select four styles that cover all nine remaining genres, forming four experiences as follows: Exp. 0: Realism, Exp. 1: Expressionism, Exp. 2: Symbolism, and Exp. 3: Naive Art Primitivism. The resulting subset contains approximately 20k images in total.

We partition each experience style among three clients using a Dirichlet distribution to create non-IID client partitions. The resulting distributions are shown in Fig. 2. We keep 20% of each partition as a local validation dataset.

## B. Experimental setup

We compare FedCurv-DR with the FedAvg baseline and the original FedCurv. We implement FedCurv with both EWC and EWC-DR importance estimates, and evaluate FedCurv-DR using two different intervals I. We use a pretrained EfficientNet-B0 model from torchvision as the backbone, with approximately 5.3 million parameters pretrained on ImageNet. For training, we use Adam optimizer with a learning rate of 0.0005, a batch size of 16, and no weight decay. Clients train for 2 local epochs per round, with 5 rounds per experience and 20 rounds in total. We set regularisation strength to $\lambda = 2 0 0$ . We use the same training settings for all methods and experiences to ensure a fair comparison. Our goal is not to achieve optimal performance, but rather to evaluate knowledge retention and convergence stability. All experiments are run in simulation on a single machine equipped with an 8GB NVIDIA RTX 4070 GPU. We use the Flower AI framework [33], combined with continual learning strategies from Avalanche [34] and the CodeCarbon framework [35] for energy tracking.

## C. Evaluation metrics

For performance evaluation, we use weighted average accuracy and report it both per experience and over the full stream. To quantify catastrophic forgetting mitigation, we use the Backward Transfer (BWT) [25] metric , defined as

$$
\mathrm { B W T } = \frac { 1 } { T - 1 } \sum _ { i = 1 } ^ { T - 1 } \left( A _ { T , i } - A _ { i , i } \right) ,\tag{7}
$$

<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Final Stream Acc.(%) ↑</td><td rowspan=1 colspan=1>BWT(pp) ↑</td><td rowspan=1 colspan=1>Disparity(pp) ↓</td><td rowspan=1 colspan=1>Energy(kWh) ↓</td></tr><tr><td rowspan=1 colspan=1>FedAvg</td><td rowspan=1 colspan=1>63.18</td><td rowspan=1 colspan=1>-11.30</td><td rowspan=1 colspan=1>7.22</td><td rowspan=1 colspan=1>0.0296</td></tr><tr><td rowspan=1 colspan=1>FedCurv(EWC)</td><td rowspan=1 colspan=1>63.99</td><td rowspan=1 colspan=1>-12.32</td><td rowspan=1 colspan=1>3.09</td><td rowspan=1 colspan=1>0.0448</td></tr><tr><td rowspan=1 colspan=1>FedCurv(EWC-DR)</td><td rowspan=1 colspan=1>65.46</td><td rowspan=1 colspan=1>-10.26</td><td rowspan=1 colspan=1>4.56</td><td rowspan=1 colspan=1>0.0458</td></tr><tr><td rowspan=1 colspan=1>FedCurv-DR(interval=2, beta=0.95)</td><td rowspan=1 colspan=1>66.92</td><td rowspan=1 colspan=1>-6.92</td><td rowspan=1 colspan=1>2.89</td><td rowspan=1 colspan=1>0.0385</td></tr><tr><td rowspan=1 colspan=1>FedCurv-DR(interval=5, beta=0.95)</td><td rowspan=1 colspan=1>65.86</td><td rowspan=1 colspan=1>-7.49</td><td rowspan=1 colspan=1>5.91</td><td rowspan=1 colspan=1>0.0334</td></tr></table>

![](images/6a69370ef9f95476ab372b09a0a65027c4864295d46dfcf89892925c5f64339d.jpg)  
(a)

![](images/082a55ff09e84e99b940745341e5b05baec3271e64f00fd1db2a1ae0f590f464.jpg)  
(b)

![](images/548163e1548887a9a843bdd5037b13e6c368b634a421ea8e77519e8d254f3c31.jpg)  
(c)  
Fig. 3. Comparison of federated continual learning strategies across communication rounds. (a) Full-stream accuracy evaluates retention over all learned experiences. (b) Current-experience accuracy measures performance on the active experience. (c) Accuracy disparity reports client-level performance differences as a proxy for fairness.

where $A _ { j , i }$ denotes the accuracy on experience i after learning experience $j ,$ and $T$ is the total number of experiences.

To quantify performance disparity across clients, we compute the standard deviation of client accuracies:

$$
{ \mathrm { D i s p a r i t y } } = { \sqrt { \frac { 1 } { | { \mathcal { C } } | - 1 } \sum _ { k \in { \mathcal { C } } } \left( A _ { k } - { \bar { A } } \right) ^ { 2 } } } ,\tag{8}
$$

where C is the set of clients, $A _ { k }$ is the accuracy of client $k ,$ and $\bar { A }$ is the average client accuracy. This follows the client-level disparity formulation suggested by [21]. Finally, we report the energy consumption and equivalent carbon emissions for each experiment using the CodeCarbon framework.

## V. RESULTS

## A. Performance

Fig. 3 and Table I show that FedCurv-DR mitigates catastrophic forgetting more effectively than FedAvg. FedAvg exhibits clear drops in full-stream accuracy after experience transitions and obtains a BWT of -11.30 pp. In contrast, FedCurv-DR with intervals of 2 and 5 rounds achieves higher final stream accuracy, 66.92% and 65.86%, compared with 63.18% for FedAvg, while also improving BWT to -6.92 pp and -7.49 pp, respectively. This indicates that the higher stream accuracy of FedCurv-DR is mainly due to better retention of previous experiences. Current-experience accuracy remains competitive, suggesting that the regularisation improves stability without severely limiting adaptation to new data.

## B. Fairness

We use client-level disparity as a proxy for fairness. FedAvg reaches a final disparity of 7.22 pp, while all FedCurv-based methods reduce this value. The lowest disparity is achieved by FedCurv-DR with interval 2, at 2.89 pp, followed by FedCurv(EWC) at 3.09 pp. Current results are not conclusive on fairness but they suggest that regularisation methods can support more balanced performance across heterogeneous clients.

## C. Energy Efficiency

Table I reports energy consumption. FedAvg has the lowest energy use, 0.0296 kWh, but also the weakest retention. FedCurv(EWC) and FedCurv(EWC-DR) require higher energy due to importances calculation, 0.0448 kWh and 0.0458 kWh. In contrast, the interval-based variants reduce this overhead: interval 2 consumes 0.0385 kWh, while interval 5 further reduces energy to 0.0334 kWh with only a small drop in final stream accuracy.

## VI. CONCLUSION AND FUTURE WORK

In this paper, we proposed a lightweight approach for inclusive and lightweight Federated Continual Learning. The results show that the proposed method mitigates catastrophic forgetting while reducing the energy footprint through intervalbased importance updates. We also observed improved clientlevel fairness in heterogeneous settings, an important property for bias mitigation in cultural heritage scenarios where institutions may differ in collection size, data quality, and style distribution. A limitation of this study is that the experiments use a public dataset synthetically partitioned across three simulated clients. This controlled setup allow us to isolate the effects of forgetting, client disparity, and energy consumption, but does not fully capture real institutional deployment constraints. FedCurv-DR is directly compatible with deployment-level privacy mechanisms; practical systems should combine it with secure aggregation [36] and differential privacy [37] to bound information leakage from model updates. In real institutional deployments, the FL server would require operation by a trusted neutral party, such as a national heritage agency or research infrastructure, with defined governance procedures covering leakage auditing, intellectual property attribution, and licensing of contributed collections, in alignment with European AI and cultural heritage data policies [38]. Future work will evaluate larger-scale and real distributed settings, together with ablation studies on I, β, and λ.

## ACKNOWLEDGMENT

This work is funded by the European Union’s Horizon Europe Research and Innovation Programme through the RAIDO project (Grant Agreement No. 101135800).

## REFERENCES

[1] M. Fiorucci, M. Khoroshiltseva, M. Pontil, A. Traviglia, A. Del Bue, and S. James, “Machine learning for cultural heritage: A survey,” vol. 133, pp. 102–108. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S0167865520300532

[2] A. Foka, G. Griffin, D. Ortiz Pablo, P. Rajkowska, and S. Badri, “Tracing the bias loop: AI, cultural heritage and bias-mitigating in practice,” vol. 40, no. 8, pp. 5835–5847. [Online]. Available: https://doi.org/10.1007/s00146-025-02349-z

[3] M. Zhitomirsky-Geffet, I. Kizhner, and S. Minster, “What do they make us see: a comparative study of cultural bias in online databases of two large museums,” vol. 79.

[4] M. Dis¸li, “Copyright and licencing for cultural heritage collections as data,” Journal of Open Humanities Data, vol. 11, 2025.

[5] European Parliamentary Research Service, “Artificial intelligence in the context of cultural heritage and museums,” European Parliament, Tech. Rep., 2023.

[6] G. Adorni and E. Bellini, “Towards a manifesto for cyber humanities: Paradigms, ethics, and prospects,” in 2025 IEEE International Conference on Cyber Humanities (IEEE-CH), pp. 1–8. [Online]. Available: https://ieeexplore.ieee.org/abstract/document/11279659

[7] E. Bellini, “Cyber humanities for heritage security,” vol. 68, no. 12, pp. 112–117. [Online]. Available: https://dl.acm.org/doi/10.1145/3735659

[8] H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y. Arcas, “Communication-Efficient Learning of Deep Networks from Decentralized Data,” Feb. 2017, arXiv:1602.05629 [cs]. [Online]. Available: http://arxiv.org/abs/1602.05629

[9] G. Tziolas et al., “Energy-efficient and privacy-preserving federated continual learning for cultural heritage preservation and digital humanities,” conference Name: 2025 IEEE International Conference on Cyber Humanities (IEEE-CH) (IEEE-CH2025) ISBN: 9798331514365 Publisher: IEEE. [Online]. Available: https://zenodo.org/records/18504685

[10] R. Schwartz, J. Dodge, N. A. Smith, and O. Etzioni, “Green ai,” Communications of the ACM, vol. 63, no. 12, pp. 54–63, 2020.

[11] I. Schoinas et al., “Federated Learning: Challenges, SoTA, Performance Improvements and Application Domains,” IEEE Open Journal of the Communications Society, vol. 5, pp. 5933–6017, 2024. [Online]. Available: https://ieeexplore.ieee.org/document/10677499/

[12] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith, “Federated Optimization in Heterogeneous Networks,” Apr. 2020, arXiv:1812.06127 [cs]. [Online]. Available: http://arxiv.org/abs/1812.06127

[13] T. Lin, L. Kong, S. U. Stich, and M. Jaggi, “Ensemble Distillation for Robust Model Fusion in Federated Learning,” Mar. 2021, arXiv:2006.07242 [cs]. [Online]. Available: http://arxiv.org/abs/2006.07242

[14] N. Shoham, T. Avidor, A. Keren, N. Israel, D. Benditkis, L. Mor-Yosef, and I. Zeitak, “Overcoming Forgetting in Federated Learning on Non-IID Data,” Oct. 2019, arXiv:1910.07796 [cs]. [Online]. Available: http://arxiv.org/abs/1910.07796

[15] X. Yao and L. Sun, “Continual Local Training For Better Initialization Of Federated Models,” in 2020 IEEE International Conference on Image Processing (ICIP). Abu Dhabi, United Arab Emirates: IEEE, Oct. 2020, pp. 1736–1740. [Online]. Available: https://ieeexplore.ieee.org/document/9190968/

[16] M. Asad, A. Moustafa, and T. Ito, “FedOpt: Towards Communication Efficiency and Privacy Preservation in Federated Learning,” Applied Sciences, vol. 10, no. 8, p. 2864, Jan. 2020, number: 8. [Online]. Available: https://www.mdpi.com/2076-3417/10/8/2864

[17] T. Li, M. Sanjabi, A. Beirami, and V. Smith, “Fair Resource Allocation in Federated Learning,” Feb. 2020, arXiv:1905.10497 [cs]. [Online]. Available: http://arxiv.org/abs/1905.10497

[18] Y. H. Ezzeldin, S. Yan, C. He, E. Ferrara, and S. Avestimehr, “FairFed: Enabling Group Fairness in Federated Learning,” Nov. 2022, arXiv:2110.00857 [cs]. [Online]. Available: http://arxiv.org/abs/2110.00857

[19] F. Zhang, D. Zhai, G. Bai, J. Jiang, Q. Ye, X. Ji, and X. Liu, “Towards fairness-aware and privacy-preserving enhanced collaborative learning for healthcare,” Nature Communications, vol. 16, no. 1, p. 2852, Mar. 2025. [Online]. Available: https://www.nature.com/articles/s41467-025- 58055-3

[20] K.-C. Lo, Y. He, Y. Jiang, and S. Parthasarathy, “FairWAG: Fairness-aware Weighted Aggregation for Graph Learning in a Federated Setting,” in Proceedings of the 5th ACM Conference on Equity and Access in Algorithms, Mechanisms, and Optimization, ser. EAAMO ’25. New York, NY, USA: Association for Computing Machinery, Aug. 2025, pp. 119–150. [Online]. Available: https://dl.acm.org/doi/10.1145/3757887.3763013

[21] F. Zhang, Z. Shuai, K. Kuang, F. Wu, Y. Zhuang, and J. Xiao, “Unified fair federated learning for digital healthcare,” Patterns, vol. 5, no. 1, p. 100907, Jan. 2024. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S2666389923003148

[22] J. Kirkpatrick et al., “Overcoming catastrophic forgetting in neural networks,” Proceedings of the National Academy of Sciences, vol. 114, no. 13, pp. 3521–3526, Mar. 2017, arXiv:1612.00796 [cs]. [Online]. Available: http://arxiv.org/abs/1612.00796

[23] R. Aljundi, F. Babiloni, M. Elhoseiny, M. Rohrbach, and T. Tuytelaars, “Memory Aware Synapses: Learning what (not) to forget,” Oct. 2018, arXiv:1711.09601 [cs]. [Online]. Available: http://arxiv.org/abs/1711.09601

[24] F. Zenke, B. Poole, and S. Ganguli, “Continual Learning Through Synaptic Intelligence,” Jun. 2017, arXiv:1703.04200 [cs]. [Online]. Available: http://arxiv.org/abs/1703.04200

[25] D. Lopez-Paz and M. Ranzato, “Gradient episodic memory for continual learning.” [Online]. Available: http://arxiv.org/abs/1706.08840

[26] D. Rolnick, A. Ahuja, J. Schwarz, T. Lillicrap, and G. Wayne, “Experience replay for continual learning,” in Advances in Neural Information Processing Systems, vol. 32. Curran Associates, Inc., 2019.

[27] G. M. v. d. Ven and A. S. Tolias, “Three scenarios for continual learning,” Apr. 2019, arXiv:1904.07734 [cs]. [Online]. Available: http://arxiv.org/abs/1904.07734

[28] J. Yoon, W. Jeong, G. Lee, E. Yang, and S. J. Hwang, “Federated Continual Learning with Weighted Inter-client Transfer,” Jun. 2021, arXiv:2003.03196 [cs]. [Online]. Available: http://arxiv.org/abs/2003.03196

[29] A. Usmanova, F. Portet, P. Lalanda, and G. Vega, “Federated Continual Learning through distillation in pervasive computing,” Jul. 2022, arXiv:2207.08181 [cs]. [Online]. Available: http://arxiv.org/abs/2207.08181

[30] J. Zhang, C. Chen, W. Zhuang, and L. Lv, “TARGET: Federated Class-Continual Learning via Exemplar-Free Distillation,” Aug. 2023, arXiv:2303.06937 [cs]. [Online]. Available: http://arxiv.org/abs/2303.06937

[31] X. Liu and X. Chang, “Elastic weight consolidation done right for continual learning,” version Number: 3. [Online]. Available: https://arxiv.org/abs/2603.18596

[32] W. R. Tan, C. S. Chan, H. Aguirre, and K. Tanaka, “Improved artgan for conditional synthesis of natural image and artwork,” IEEE Transactions on Image Processing, vol. 28, no. 1, pp. 394–409, 2019. [Online]. Available: https://doi.org/10.1109/TIP.2018.2866698

[33] D. J. Beutel et al., “FLOWER: A FRIENDLY FEDERATED LEARNING FRAMEWORK,” Nov. 2022. [Online]. Available: https://hal.science/hal-03601230

[34] V. Lomonaco et al., “Avalanche: an End-to-End Library for Continual Learning,” Apr. 2021, arXiv:2104.00405 [cs]. [Online]. Available: http://arxiv.org/abs/2104.00405

[35] B. Courty et al., “mlco2/codecarbon: v2.4.1,” May 2024. [Online]. Available: https://doi.org/10.5281/zenodo.11171501

[36] K. Bonawitz et al., “Practical Secure Aggregation for Federated Learning on User-Held Data,” Nov. 2016, arXiv:1611.04482 [cs]. [Online]. Available: http://arxiv.org/abs/1611.04482

[37] M. Abadi et al., “Deep learning with differential privacy,” in Proceedings of the 2016 ACM SIGSAC Conference on Computer

and Communications Security, pp. 308–318. [Online]. Available: http://arxiv.org/abs/1607.00133

[38] European Data Protection Supervisor, AI Act Regulation (EU) 2024/1689 – Regulation (EU) 2024/1689 of the European Parliament and of the Council of 13 June 2024 laying down harmonised rules on artificial intelligence and amending Regulations (EC) No 300/2008, (EU) No 167/2013, (EU) No 168/2013, (EU) 2018/858, (EU) 2018/1139 and (EU) 2019/2144 and Directives 2014/90/EU, (EU) 2016/797 and (EU) 2020/1828 (Artificial Intelligence Act) (Text with EEA relevance). Publications Office of the European Union, 2025.