# Cross-Domain Generalization in Machine Unlearning via Label-Conditioned Energy Magnitude Regularization

Syed Ali Ahmed<sup>1</sup> , Syed Bilal Ahsan<sup>1</sup> , and Muhammad Zaigham Zaheer<sup>2</sup>

<sup>1</sup> National University of Computer and Emerging Sciences, Karachi, Pakistan sayyidaliahmed1@gmail.com, syed.bilal.ahsan1@gmail.com

2 Mohamed bin Zayed University of Artificial Intelligence, Abu Dhabi, UAE zaigham.zaheer@mbzuai.ac.ae

Abstract. Machine unlearning removes the influence of specific data from a trained model. However, most methods treat the forgotten concept as isolated. In this paper, we study what happens to the rest of the model when a class is forgotten, using a label-conditioned energybased model (EBM) that assigns per-class energies, making the efect directly observable. We forget a class by raising the energy of its imagelabel pairs, training with a forget term, a retain anchor to the pretrained model, a global margin, and an energy regularizer that stops the energy magnitudes from growing without limit. A propagation term applies the same forget signal to retain samples, weighted by each sample’s DINOv2 similarity to the forget class, so forgetting reaches images that resemble it and leaves the rest untouched. We evaluate on two benchmark datasets: 1) On a subset of DomainNet across four visual domains, we forget tiger, lion, and scissors one at a time. Forgetting a class in the sketch domain also erases it from real, clipart, and painting, with forgetting error reaching 98% and 99% for lion and scissors, and the efect carrying over to the most similar class. 2) On CIFAR-10, we turn of the propagation term and forget each of the ten classes on its own. Forgetting is complete (100%), while the other nine classes retain 98.5% of their pre-unlearning accuracy on average.

Keywords: Machine unlearning · Energy-based models · Cross-domain generalization

## 1 Introduction

Modern classifiers are pretrained on large datasets comprising hundreds if not thousands of classes; however, in practice, these models are rarely used to their complete extent, as most tasks only require a subset of the classifier’s knowledge. In addition, these pretrained classifiers may also need to forget specific information that they have learned, due to a privacy request, a safety concern, or a correction to their training data. This process of completely forgetting a piece of information and its efects from a learned model is formally known as ‘machine unlearning’ [4].

Unlearning methods commonly perform isolated unlearning by identifying a target class, suppressing it, and preserving performance elsewhere [5, 8, 12, 30]. Such formulations typically treat the remaining data as a retain set, without explicitly accounting for semantic relationships between the target and particular retained classes or domains. In practice, however, the information being forgotten is rarely isolated. For instance, if a model is asked to forget ’tiger’ as it appears in hand-drawn sketches, the same object still exists in photographs, paintings, and clipart; forgetting the concept incompletely leaves it fully recoverable through these other domains. Moreover, ’tiger’ (regardless of the domain in which it occurs) is not an independent concept. It shares visual and semantic similarity with related classes such as ’lion’. Whether forgetting propagates to such related concepts, and how far it reaches, is a question that has gathered little attention.

Cheng et al. [6] establish that forgetting a specific class afects semantically related classes, but they treat this as a negative efect that needs to be actively suppressed to preserve retain-set performance. However, depending upon the scenario, suppression may not always be the right choice. In some applications, such as forgetting a person’s face for privacy reasons, some spread of forgetting to closely related identities may be preferable to prevent the model from indirectly reconstructing the forgotten identity through highly similar facial features [7]. Therefore, the deliberate and controlled propagation of forgetting signals across related domains or classes is an important yet largely overlooked aspect of machine unlearning.

In this paper, we study the propagation of forgetting using label-conditioned Energy Based Models (EBM) instead of a standard softmax classifier. Unlike a softmax classifier, an EBM assigns an explicit energy score to each class for a given input, and the lowest-energy class is the prediction. This lets us reshape the energy landscape around a forgotten class and observe how the energies of other classes and domains shift in response. We show that forgetting a class in one domain generalizes to that class in every other domain: erasing sketchtigers removes the tiger concept from real, clipart, and painting as well, with no domain-specific retraining. Forgetting also spills to visually similar classes, most strongly the nearest neighbour. This propagation is not solely an efect of our objective. It emerges from the shared representation in any unlearning process. We therefore focus on controlling how far this spread reaches, not only on showing that it occurs. We take advantage of it with a mechanism based on DINOv2 [25] feature similarity and PCA subspace [17] weighting that directs and scales the spread, so that the target class is forgotten across domains and the classes closest to it in feature space receive the strongest forgetting signal. The main contributions of this paper are summarized as follows:

– We introduce a label-conditioned energy-based model (EBM) with energy magnitude regularization to study class forgetting, giving each class an explicit, directly observable energy score.

– (Finding) We show that forgetting is not isolated, i.e., erasing a class in one domain removes it from all other domains as well, consistently across forget targets, and the efect further spills to the nearest visually similar classes.

– (Method) We introduce a DINOv2 similarity and PCA subspace weighting mechanism, together with a mask, that turns this incidental spread into a controllable one, scaling how far forgetting reaches and steering it toward chosen domains or similar classes.

## 2 Related Work

## 2.1 Machine Unlearning.

Machine unlearning aims to remove the influence of specific information from a trained model, ideally producing a model identical to one retrained from scratch on the retained data [27, 29, 36]. This field is broadly categorized into two foundational unlearning approaches, exact unlearning and approximate unlearning. Exact unlearning aims to ensure that the behavior of the updated model matches that of such a retrained model by producing the identical distributions [35]. Early work by Cao and Yang [4] described this goal in terms of recomputing model statistics as if the removed data had never been used. Later, Bourtoule et al. [3] proposed SISA, which employs a sharding technique that retrains only the subsets of data which contained forget samples, avoiding the need to retrain the entire model. Ginart et al. [14] introduced Q-k-means, which quantizes cluster centroids so forgetting can be done via metadata updates instead of recomputing the clustering. However, exact unlearning methods are generally based on linear regression, SVMs or leverage k-means clustering and do not scale to overparameterized neural nets. This limitation motivates the second broad category, approximate unlearning.

Approximate unlearning trades perfect forgetting for lower computational cost, achieving this via eficient parameter updates through gradient ascent on forget samples [16, 19]. Golatkar et al. [15] instead edit the weights directly, they ’scrub’ model parameters so that no test performed on the network can distinguish it from one that never saw the forgotten data, without requiring retraining the original training set.

## 2.2 Unlearning Across Domains.

A few recent works consider unlearning in settings where the same class appears across multiple domains but they approach it from diferent angles. Kawamura et al. [18] propose Approximate Domain Unlearning (ADU), for domain-level forgetting in vision-language models. Instead of removing specific classes, they target an entire domain and suppress its recognition across all classes, efectively removing domain-specific information from the model. Mishra et al. [23] keep the class as the unit of forgetting but wrap it around domains. Their training and data-free method removes a target class only from designated domains through a closed-form nullspace projection, while leaving that same class intact in the domains that are retained. Basak and Yin [1] use forgetting for a diferent purpose. They unlearn domain-specific features so that the remaining domain-agnostic features transfer better across domains. Sepahvand et al. [28] treat the forget set and a held-out validation set as two ’domains’ and apply gradient reversal to erase forget-set representations. In their case domain refers to distinct data distributions, not to visual domains such as sketch or photograph, and the method does not deal with forgetting across styles. None of these works, however, examine whether forgetting a class in one domain spreads to that same class in other domains or other semantically related classes. This is the question we study.

## 2.3 Energy-Based Models.

Energy-based models (EBMs) assign a scalar energy to each input and define an unnormalized distribution in which likely samples take low energy [20]. They have been applied to image generation [9,13,24,34], structured prediction [2,31], and out-of-distribution detection [22, 32]. Grathwohl et al. [11] showed that a standard classifier can be employed as an EBM, since its logits can be reinterpreted as a joint energy over inputs and labels (JEM); this energy is most often used generatively, producing samples through Langevin dynamics [10]. Li et al. [21] utilize EBMs for continual learning. They perform selective incremental updates by lowering the energy of an input at its true label and raising it at selected negatives, so that updates touch only the chosen classes and forgetting is avoided. Recent unlearning work builds directly on this view: DSDA [33] reinterprets a classifier’s logits as an energy function and runs Langevin sampling to synthesize surrogate training data, enabling source-free unlearning without the original dataset.

## 3 Method

In this paper, we propose reshaping the energy landscape of a pretrained, labelconditioned energy-based model to capture the forgetting of a class within a specific domain. The motivation builds on the standard EBM formulation proposed by LeCun et al. [20], which scores every (X, Y) pair with a single scalar value called energy where X is an input image and Y is the label associated with it. The idea is that a correct (X, Y ) pair should have lower energy compared to an incorrect or corrupted (X, Y ) pair. We adopt EBMs instead of a softmax classifier because forgetting, in our setting, means making specific image-label pairs unfavorable, and an energy gives one quantity per pair that we can push up to forget. This gives us one direct mechanism and one direct measurement built on the same quantity: we raise the energy of the forget cell while anchoring the rest of the model to its original behavior, and we later read the resulting energies to visualise forgetting on CIFAR-10, as shown in Figure 2.

The pipeline has two stages, shown in Figure 1. We first pretrain the EBM on all classes and domains with a contrastive energy loss, producing a reference model $E _ { 0 }$ (Section 3.2). We then freeze $E _ { 0 }$ and train a copy $E _ { \theta }$ to forget. The unlearning objective increases the energy of samples belonging to the forget set while anchoring retain energies to $E _ { 0 }$ . To enable this forgetting to propagate to other domains of the target class and to other similar classes, we add a directed propagation term: each retain sample is weighted by how strongly its DINOv2 features align with the forget cell’s feature subspace, and a similarityweighted forget signal is applied in proportion to that weight, though some propagation occurs even without this added term, since it also arises from the shared backbone alone (Section 3.4). A masking strategy is used alongside this to control the reach of the signal, either confining it to the forget class in other domains or letting it act across classes. The result is a model that no longer recognizes the forget set, with forgetting also propagating to related classes and domains, as we study in the subsequent sections.

![](images/2983b3c501b5b99dea1d634e1da046ae55aecc421abfc18a3ab17a374cd83c14.jpg)  
Fig. 1: Overview of our method. Stage 1 pretrains a label-conditioned EBM to obtain frozen reference $E _ { 0 }$ . Stage 2 trains a copy $E _ { \theta }$ to forget a single (class, domain) cell by combining four losses (forget, retain, margin, and energy regularization) with a DINOv2 similarity-weighted propagation mechanism. Scope masking controls whether the forget signal propagates across domains or semantically related classes, selectively forgetting target knowledge while preserving unrelated concepts.

## 3.1 Problem Formulation

Each image x is associated with a class label $c \in \{ 1 , \ldots , K \}$ and a domain label $d \in \{ 1 , \ldots , M \}$ . In our experiments, we set $K = 2 6$ and $M = 4$ . We randomly sample 26 of the 345 DomainNet classes and use four of its six domains, real, sketch, clipart, and painting. The energy model $E _ { \theta }$ assigns a scalar energy to an image x paired with a candidate label $y \in \{ 1 , \ldots , K \}$ , and is conditioned on the class label only; here $y$ is the query label whose energy we evaluate, not necessarily the true class c. The domain label is never given to the network and is used solely to define what we forget. A class is predicted by taking the candidate label of lowest energy,

$$
\hat { y } = \arg \operatorname* { m i n } _ { y \in \{ 1 , \ldots , K \} } E _ { \theta } ( x , y ) ,\tag{1}
$$

so the prediction $\hat { y }$ is correct when ${ \hat { y } } = c .$

We forget a single (class, domain) cell rather than a whole class or a whole domain. For a forget class $c ^ { \star }$ and forget domain $d ^ { \star }$ , the forget set is

$$
D _ { f } = \left\{ ( x , c ^ { \star } ) \mid \mathrm { c l a s s } ( x ) = c ^ { \star } , \mathrm { d o m a i n } ( x ) = d ^ { \star } \right\} ,\tag{2}
$$

and the retain set $D _ { r }$ is its complement, i.e., every remaining image-class pair in the collection. Taking $c ^ { \star } = \mathrm { t i g e r }$ and $d ^ { \star } = \mathrm { s k e t c h }$ , the goal is to stop the model recognizing sketch tigers while its behavior on tigers in real, clipart, painting, and on every non-tiger class, stays as it was. The forget class is not removed from the retain set entirely, i.e., images of tiger from the other domains remain in $D _ { r }$ . This overlap is intentional. It is what makes cross-domain forgetting measurable, and the propagation mechanism in Section 3.4 operates particularly on these retain samples.

## 3.2 Base architecture and pretraining

The energy model turns an (image x, label $y )$ pair into a single scalar score by combining an image feature with the label’s embedding. We use a ResNet-18 backbone to encode the image, followed by a linear projection, $f ( x ) =$ $\mathrm { p r o j } ( \mathrm { R e s N e t 1 8 } ( x ) ) \in \mathbb { R } ^ { d _ { e } }$ , while each class label is mapped to a vector $e ( y ) \in \mathbb { R } ^ { d _ { \epsilon } }$ through a learned embedding table. The two are multiplied elementwise and passed through a linear head to produce the scalar energy,

$$
E _ { \theta } ( x , y ) = w ^ { \top } \left( f ( x ) \odot e ( y ) \right) + b ,\tag{3}
$$

where w $\in \mathbb { R } ^ { d _ { e } }$ and $b \in \mathbb { R }$ are the learned weight vector and bias of the linear energy head, and $d _ { e } = 1 2 8$ in all experiments. The projection maps the image into a shared $d _ { e }$ -dimensional embedding space so that it can be combined with the label embedding $e ( y )$ , which lives in the same space. This score is unnormalized and specific to a single (image, label) pair, which is the quantity the unlearning objective later raises for the forget cell and holds fixed for the rest.

The backbone is initialized from ImageNet-pretrained weights, and we do not train all of it. During pretraining the last residual stage (layer4) is fine-tuned along with the projection, the label embedding, and the energy head, while the earlier layers are kept frozen. This keeps low-level features fixed and adapts only the higher-level representation to DomainNet. The number of trainable stages is a single setting we vary in later experiments, and a low-rank (LoRA) variant that freezes the convolutional weights and trains only rank-r adapters is available as an alternative.

We pretrain $E _ { \theta }$ on all 26 classes across the four domains with a supervised energy-contrast loss. For each image, the loss compares the energy of its correct label against N sampled incorrect labels and penalizes any case where the correct label is not lower by a margin $m _ { 0 }$ . Taking the hardest of the N negatives per image, the loss is

$$
{ \mathcal { L } } _ { \mathrm { p r e } } = { \frac { 1 } { | B | } } \sum _ { x \in B } \operatorname* { m a x } _ { j = 1 , \ldots , N } { \mathrm { R e L U } } \left( m _ { 0 } + E _ { \theta } ( x , c ) - E _ { \theta } ( x , c _ { j } ^ { - } ) \right) ,\tag{4}
$$

where c is the true class, $c _ { j } ^ { - } \neq$ c are the sampled negatives, and B is the minibatch. We use $N = 1 0$ and $m _ { 0 } = 1$ , optimize with Adam, and stop early on validation accuracy, with the remaining settings given in our implementation details. The model produced by this stage is the reference $E _ { 0 }$ . It is held frozen for the rest of the method.

## 3.3 Unlearning Objective

Unlearning trains a copy $E _ { \theta }$ , initialized from the pretrained reference $E _ { 0 } .$ , while $E _ { 0 }$ stays frozen. The objective moves the energy landscape in two directions at once. It raises the energy of the forget cell at its correct label until that label is no longer the lowest, so by Eq (1) the model settles on some diferent classes for those images and no longer recognizes them.

We design four loss functions for this task and every training step applies all of them together. The forget term raises the energy of each forget image at its correct label above the energies it assigns to the other labels. The retain term ties the rest of the model back to $E _ { 0 } . \mathrm { ~ A ~ }$ margin term enforces a consistent gap between forget and retain energies, and a small regularizer keeps the energy magnitudes bounded so the other terms cannot inflate them without limit. At each step we draw a forget minibatch from $D _ { f }$ and a retain minibatch from $D _ { r } ,$ evaluate the four terms on them, and update $E _ { \theta }$ by their weighted sum while $E _ { 0 }$ is held fixed. We describe each term below and then give the combined objective.

Forget term. The forget term acts on images from $D _ { f }$ , each associated with the forget class $c ^ { \star }$ . For each image, it requires the energy of the correct pair to exceed that of every other label by a margin m. The maximum selects the label for which this is hardest to satis $\mathrm { f y , }$ i.e., the label with the highest energy, and meeting that one constraint implies all the others are met as well:

$$
L _ { f } = \frac { 1 } { \vert B _ { f } \vert } \sum _ { x \in B _ { f } } \ \operatorname * { m a x } _ { y ^ { \prime } \neq c ^ { \star } } \ \mathrm { s o f t p l u s } \left( m + E _ { \theta } ( x , y ^ { \prime } ) - E _ { \theta } ( x , c ^ { \star } ) \right) .\tag{5}
$$

This mirrors the pretraining loss of Eq (4) with the sign reversed. Pretraining picks the lowest-energy negative to push the correct label to the bottom of the ranking, whereas the forget term picks the highest-energy competitor and drives the forget cell’s correct label to the top, so by Eq (1) the correct label becomes the label the model is least likely to output.

Retain anchor. The retain term is where $E _ { 0 }$ acts as an anchor. For each retain image it keeps the current energy close to the value $E _ { 0 }$ assigned, by penalizing the squared diference between the two. We divide by the average pretrained magnitude so the penalty does not depend on how large the energies have grown:

$$
L _ { r } = \frac { 1 } { \left| { \cal B } _ { r } \right| } \sum _ { x \in { \cal B } _ { r } } \left( E _ { \theta } ( x , c ) - E _ { 0 } ( x , c ) \right) ^ { 2 } \Bigg / \frac { 1 } { \left| { \cal B } _ { r } \right| } \sum _ { x \in { \cal B } _ { r } } \left| E _ { 0 } ( x , c ) \right| ,\tag{6}
$$

where c is the true class of x. Since $D _ { r }$ holds the forget class in the other domains, this term is also what keeps the model’s behavior on the forget class (‘tiger’ in our example) in real, clipart, and painting close to its pretrained state, until the mechanism of Section 3.4 deliberately relaxes it.

Global margin. The margin term adds a global separation between the two sets. It pushes forget energies to lie above retain energies by the same margin $m _ { ; }$ taken over a forget and a retain minibatch of equal size:

$$
L _ { m } = \frac { 1 } { | B _ { f } | } \sum _ { i = 1 } ^ { | B _ { f } | } \mathrm { s o f t p l u s } \left( m + E _ { \theta } ( x _ { r } ^ { ( i ) } , c _ { r } ^ { ( i ) } ) - E _ { \theta } ( x _ { f } ^ { ( i ) } , c _ { f } ^ { ( i ) } ) \right) ,\tag{7}
$$

where the i-th retain example $( x _ { r } ^ { ( i ) } , c _ { r } ^ { ( i ) } )$ and forget example $( x _ { f } ^ { ( i ) } , c _ { f } ^ { ( i ) } )$ are paired across the two equal-size minibatches $\left( \left| B _ { f } \right| = \left| B _ { r } \right| \right)$

Energy Magnitude Regularization. The final regularization term penalizes large energy magnitudes, discouraging the forget and margin objectives from driving the energies to arbitrarily large values:

$$
{ \cal L } _ { e } = \textstyle { \frac { 1 } { 2 } } \left( \frac { 1 } { | B _ { f } | } \sum _ { x _ { f } \in B _ { f } } E _ { \theta } ( x _ { f } , c _ { f } ) ^ { 2 } + \frac { 1 } { | B _ { r } | } \sum _ { x _ { r } \in B _ { r } } E _ { \theta } ( x _ { r } , c _ { r } ) ^ { 2 } \right) .\tag{8}
$$

Full objective. The four terms are combined by fixed weights,

$$
L = \lambda _ { f } L _ { f } + \lambda _ { r } L _ { r } + \lambda _ { m } L _ { m } + \lambda _ { e } L _ { e } ,\tag{9}
$$

where the weights $\lambda _ { f } , \lambda _ { r } , \lambda _ { m } , \lambda _ { e }$ balance the four terms, and the same margin m is shared by the forget, margin, and propagation terms. We report their values in the implementation details. Section 3.4 adds one further term that governs how far this forgetting propagates across domains.

## 3.4 Directed Propagation

We also design a mechanism that directs and amplifies how far forgetting spreads beyond the forget cell, i.e., how forgetting carries across domains. It applies a forgetting signal to retain samples, weighted by their resemblance to the forget cell, so samples that resemble the forget cell are pushed hardest while those with little in common are left almost untouched. A mask then fixes the scope of this push, either restricting it to the forget class in other domains or opening it to other classes. A part of this spread also occurs on its own, since the backbone and label embedding are shared.

Similarity weighting. We use the frozen DINOv2 image encoder to measure the similarity between each retain sample and the forget cell. Let $\phi ( x )$ be the L<sub>2</sub>-normalized DINOv2 feature of image x and $\mu$ the mean feature over the forget and retain sets. We then apply PCA to the centered forget-cell features and stack the top k principal directions as the rows of S. Each retain sample is then weighted by the fraction of its centered feature that lies in this forget subspace:

$$
\alpha ( x ) = \mathrm { c l i p } \left( \frac { \| S \left( \phi ( x ) - \mu \right) \| ^ { 2 } } { \| \phi ( x ) - \mu \| ^ { 2 } } , \ 0 , \ 1 \right) ,\tag{10}
$$

with $\mathrm { c l i p } ( z , 0 , 1 ) = \operatorname* { m i n } ( \operatorname* { m a x } ( z , 0 ) , 1 )$ . A weight near one means the sample looks much like the forget cell, and a weight near zero means it does not. We compute these weights once, before unlearning, and keep them fixed.

Weighted forget signal. This term applies the same forget-style push from Eq (5), but applies it to retain samples and scales it by the weight $\alpha ( x )$ . For a retain sample with true class $^ { c , }$ we pick one random incorrect label $c ^ { - }$ and push the correct-label energy $E _ { \theta } ( x , c )$ above $E _ { \theta } ( x , c ^ { - } )$ by the margin m:

$$
L _ { p } = \frac { 1 } { \left| { \cal B } _ { r } \right| } \sum _ { x \in { \cal B } _ { r } } \alpha ( x ) \mathrm { r e l u } \left( m + E _ { \theta } ( x , c ^ { - } ) - E _ { \theta } ( x , c ) \right) .\tag{11}
$$

The diference from Eq (5) is small: we use a single sampled negative and a plain hinge (relu) instead of a softplus over all labels. What matters is the weight $\alpha ( x )$ . A sample close to the forget cell (α near 1) gets a strong push, so the model begins to forget it the way it forgets the cell. A sample far from it (α near 0) gets almost no push, and the retain anchor of Eq (6) keeps its energy at the pretrained value.

Reach of the signal. A mask on the weights decides which samples are eligible. To allow unlearning to be propagated across other domains of the target class, we keep $\alpha ( x )$ only for retain samples whose class is the forget class and zero the rest, so the signal reaches tigers in real, clipart and painting but nothing else, and forgetting stays limited to the target class across domains. In the cross-class setting we keep the weights for all retain samples, so the signal also reaches other classes in proportion to their similarity and forgetting is allowed to spread to visually related categories. The two settings use the same weighting and difer only in which weights are set to zero. The propagation term is added to the objective of Section 3.3,

$$
L _ { \mathrm { t o t a l } } = L + \lambda _ { p } L _ { p } .\tag{12}
$$

Setting $\lambda _ { p } = 0$ does not switch propagation of because part of the spread comes from the shared backbone and not this term alone, its extent also depends on how much of the backbone adapts, set by the trainable stages or the LoRA rank.

Table 1: Unlearning on DomainNet across three forget targets, each forgotten in the sketch domain. F is forgetting error (100 acc, %, ↑) on the target cell $\mathrm { ( F / s k . ) }$ and on the same class in the other three domains $\left( \mathrm { F / o t h . } \right)$ ). Mem is retain accuracy before and after unlearning (↑).
<table><tr><td colspan="4">Target Method F/sk.↑F/oth.↑  $\scriptstyle { \mathrm { M e m } } _ { \mathrm { p r e } }$   $\operatorname { M e m } _ { \operatorname { u n l } } \uparrow$ </td></tr><tr><td>tiger</td><td>Retrain 19.9 NegGrad 100.0 Finetune 12.2 Ours 96.1</td><td>3.3 100.0 2.6 83.7</td><td>96.0 96.2 96.0 85.3 96.0 97.2 96.0 76.7</td></tr><tr><td>lion</td><td>Retrain 46.7 NegGrad 100.0 Finetune 18.2 Ours 99.4</td><td>4.1 100.0 1.9 98.5</td><td>95.9 96.1 95.9 70.0 95.9 97.5 95.9 80.9</td></tr><tr><td>scissors</td><td>Retrain 54.9 NegGrad 100.0 Finetune 36.6 Ours 100.0</td><td>4.5 100.0 1.4 99.2</td><td>96.1 97.2 96.1 72.4 96.1 96.3 96.1 77.2</td></tr></table>

## 4 Results

## 4.1 Experimental Setup

Datasets. We use a 26-class subset of DomainNet [26] across four domains: real, sketch, clipart, and painting. We forget three targets in turn: tiger, lion, and scissors, each in the sketch domain. The 25 retain classes span a range of DINOv2 similarity to the forget class, so degradation can be measured from highly similar down to unrelated classes. For isolated single-class forgetting, we additionally evaluate all ten classes of CIFAR-10.

Model. The label-conditioned EBM of Sec. 3.2 with a ResNet-18 backbone, used for both datasets.

Baselines. Retrain (from scratch on the retain set, gold reference), NegGrad (gradient ascent on the forget set), and Finetune (continued training on the retain set). On DomainNet we run our method with the cross-class (propagate) mask; on CIFAR-10 we run it in the isolated single-class setting.

Metrics. On DomainNet, we report post-unlearning top-1 accuracy. Forgetting is split into the target cell (tiger/sketch) and the same class in other domains. Retain-set accuracy, which we report as Mem, is measured over all retain classes before and after unlearning. The spread to other classes is measured as the Pearson r (and Spearman $\rho )$ between a retain class’s DINOv2 similarity to the forget class and its accuracy drop, reported in Table 3. On CIFAR-10, we report the forgetting rate, model utility (Mem after unlearning divided by Mem before), retain energy preservation (rank agreement of retain energies with $E _ { 0 } )$ , and the membership-inference AUC on forget samples, where 0.5 indicates the forgotten samples are indistinguishable from data the model never saw.

Table 2: Isolated unlearning on CIFAR-10, all ten classes as forget targets. F is forgetting error (100 − acc, %). Mem is reported before (pretrained) and after unlearning on the other nine classes; the pretrained model tops out near 70% on this backbone, so the near-unchanged Mem column reflects preserved utility, not degradation. Utility is the after/before ratio; retain energy preservation is the rank-order agreement of retain energies with the pretrained model (1.0 = identical); MIA is membership-inference AUC on forget samples (0.5 = indistinguishable).
<table><tr><td>Class</td><td>F↑  $\mathrm { M e m _ { p r e } }$ </td><td> $\scriptstyle { \mathrm { M e m } _ { \mathrm { u n l } } }$ </td><td></td><td></td><td>↑ Utility↑ Energy pres. ↑ MIA (→0.5)</td><td></td></tr><tr><td>airplane</td><td>100</td><td>69.3</td><td>67.8</td><td>97.9</td><td>0.862</td><td>0.489</td></tr><tr><td>automobile 100</td><td></td><td>69.3</td><td>66.4</td><td>95.7</td><td>0.872</td><td>0.496</td></tr><tr><td>bird</td><td>100</td><td>70.1</td><td>71.8</td><td>102.5</td><td>0.885</td><td>0.491</td></tr><tr><td>cat</td><td>100</td><td>71.0</td><td>72.5</td><td>102.1</td><td>0.864</td><td>0.490</td></tr><tr><td>deer</td><td>100</td><td>70.9</td><td>70.0</td><td>98.8</td><td>0.857</td><td>0.508</td></tr><tr><td>dog</td><td>100</td><td>70.6</td><td>70.4</td><td>99.8</td><td>0.813</td><td>0.484</td></tr><tr><td>frog</td><td>100</td><td>69.2</td><td>69.1</td><td>99.9</td><td>0.840</td><td>0.508</td></tr><tr><td>horse</td><td>100</td><td>69.6</td><td>67.8</td><td>97.4</td><td>0.839</td><td>0.498</td></tr><tr><td>ship</td><td>100</td><td>69.0</td><td>66.0</td><td>95.6</td><td>0.852</td><td>0.505</td></tr><tr><td>truck</td><td>100</td><td>68.7</td><td>65.6</td><td>95.5</td><td>0.853</td><td>0.491</td></tr><tr><td>mean</td><td>100</td><td>69.8</td><td>68.7</td><td>98.5</td><td>0.854</td><td>0.496</td></tr></table>

Table 3: Cross-class spillover (DomainNet). When a class is forgotten, degradation concentrates on its nearest visual neighbour. NN is the retain class with highest DI-NOv2 similarity to the forget class; “NN drop” is its accuracy drop after unlearning. r and ρ are the Pearson and Spearman correlation between a retain class’s similarity to the forget class and its accuracy drop, over all 25 retain classes.
<table><tr><td colspan="3">Forget target nearest neighbour (sim.) NN drop r ρ</td></tr><tr><td>tiger</td><td>lion (0.55)</td><td>-75% 0.76 0.44</td></tr><tr><td>scissors</td><td>pliers (0.78)</td><td>-46% 0.42 0.10</td></tr><tr><td>lion</td><td>tiger (0.56)</td><td>-9% 0.10 0.21</td></tr></table>

Implementation. Pretraining: 50 epochs, Adam, lr $1 0 ^ { - 4 } , N = 1 0 , m _ { 0 } = 1$ early stopping on validation accuracy. DomainNet unlearning: 1500 steps, batch 32, lr $1 0 ^ { - 4 }$ , weights $\lambda _ { f } ~ = ~ 1 , \lambda _ { r } ~ = ~ 1 5 , \lambda _ { m } ~ = ~ 1 , \lambda _ { e } ~ = ~ 1 0 ^ { - 3 }$ , margin $m \ : = \ : 5 ;$ propagation $\lambda _ { p } = 3 , k = 5$ , frozen DINOv2 ViT-B/14; 20% holdout for MIA. CIFAR-10 unlearning uses the same four-term objective without the propagation term $( \lambda _ { p } = 0 )$

## 4.2 Forgetting Generalizes Across Domains

Table 1 shows that our method drives F/oth. close to zero for lion and scissors and to ∼16% for tiger (98.5, 99.2, 83.7), meaning removing sketch tigers also removes them from real, clipart, and painting. Retrain and Finetune barely reduce the class in the other domains $\left( \mathrm { F / o t h . ~ < ~ 5 } \right)$ . Since the class still exists there, the model keeps recognizing it in sketch as well, so retraining on the retain set simply relearns it from the other domains and the deletion never takes place. NegGrad also forgets across domains, suggesting this efect comes from the shared backbone rather than our loss, but it is less selective and reduces retain accuracy (70–72 on lion and scissors) and it has no way to limit how far forgetting reaches. Our method keeps more of this accuracy while still forgetting across domains, at a moderate cost.

![](images/c27ecb3bf39b6ac8e559b37c79cb5c525c3b1ef73843911c936f79127c127284.jpg)  
Fig. 2: Energy-landscape reshaping under isolated unlearning across all ten CIFAR-10 classes. Bars show the mean energy at the true label for the forget class and the retain set, before and after unlearning. For every class the forget-class energy is lifted by roughly ten units (to ≈ +8.3), while the retain-set energy is essentially unchanged (≈−1.4 → −1.9). This is the mechanism behind the energy gaps: forgetting raises only the target’s energy and leaves the retain landscape intact.

## 4.3 Isolated Forgetting on CIFAR-10

On DomainNet, forgetting is allowed to spread across classes. Isolated forgetting is the opposite case where we remove a single class and leave the others unchanged despite them being highly similar to the target class/domain. We evaluate it on CIFAR-10 by applying the same unlearning procedure to each of the ten classes in turn.

Table 2 reports all ten runs. We achieve 100% forget error on all classes, while Mem stays close to the pretrained baseline of about 70% on CIFAR-10, within about one point on average. This gives a mean model utility of 98.5% and a minimum of 95.5%. The membership-inference AUC averages 0.496, indistinguishable from the chance value of 0.5, so forget-set membership cannot be recovered from the unlearned model.

Figure 2 reports the same ten runs in terms of energy. Before unlearning, the forget class and the retain set both have mean energies closer to mean energy −1.4. After unlearning, the forget-class energy rises to +8.3 while the retain energy barely moves, from −1.4 to −1.9. Because inference is arg min<sub>y</sub> $E _ { \theta } ( x , y )$ raising only the forget-class energy above the retain energies removes the forget class from the prediction and leaves the other decisions unchanged. The retain energy preservation in Table 2 is (≈ 0.85). This means the retain classes keep almost the same energy ordering after unlearning as before.

Table 4: What controls propagation (DomainNet, tiger/sketch). Backbone capacity and the propagation term each move the similarity–degradation correlation. As more of the backbone adapts, propagation (r) rises monotonically, and with the term of $( \lambda _ { p } = 0 )$ forgetting still spreads, confirming it is partly intrinsic to the shared backbone.
<table><tr><td>Variant trainable</td><td>F↑ Mem↑</td><td> $r \uparrow$   $\rho \uparrow$ </td></tr><tr><td colspan="3">Backbone capacity  $\left( \lambda _ { p } { = } 3 \right)$  Frozen (heads only) heads only 60.6 92.1</td></tr><tr><td>LoRA r8, layer4 157k LoRA r8, layer3+4 227k Full layer4 (Ours) 8.5M</td><td>66.6 89.3 62.7 87.0 96.1 77.9</td><td>0.535 0.083 0.642 0.319 0.76 0.44</td></tr><tr><td colspan="3">Propagation term (full capacity)</td></tr><tr><td> $\lambda _ { p } { = } 0 \ \mathrm { ( i s o l a t e d ) }$  8.5M</td><td>73.6 82.8</td><td>0.63 0.36</td></tr><tr><td> $\lambda _ { p } { = } 3 ~ ( \mathrm { O u r s } )$  8.5M</td><td>96.1 77.9</td><td>0.76 0.44</td></tr></table>

## 4.4 Propagation to Similar Classes

Forgetting spreads to other classes too, but the efect is small and afects mostly the nearest neighbour. The most afected retain class is always the one closest to the target in DINOv2 space. Lion drops 75% when tiger is forgotten, and pliers drops 46% when scissors is forgotten. Over all 25 retain classes the drop still correlates with similarity, with $r = 0 . 7 6$ for tiger, but most of that correlation comes from the single nearest class and the strength is not the same across targets. Figure 3 shows this for tiger, where classes above the similarity threshold lose a median of 28% and the rest lose 11%. Section 4.5 shows how the weighting term controls how far this spread reaches.

## 4.5 Controlling Propagation

We show in Table 4 that two factors can change how far forgetting spreads. The first is how much of the backbone we train. When the backbone is frozen and only the heads are updated, the similarity–degradation correlation is low $( r = 0 . 5 2 )$ Training small LoRA adapters raises it, and training the full last layer raises it to 0.76. The forgetting therefore spreads through the shared backbone, and it reaches further when more of the backbone is allowed to adapt. The second is the propagation term. Setting it from $\lambda _ { p } { = } 0$ to $\lambda _ { p } { = } 3$ raises the correlation from $r = 0 . 6 3$ to $r = 0 . 7 6$ and pushes the spread toward the classes most similar to the forget class.

![](images/26692106e8e08e65d48ae78812caf14a5f62218421ef63bdf3d229a2949448d0.jpg)  
Fig. 3: Cross-class spillover for tiger/sketch. Each point is a retain class’s accuracy drop after unlearning, grouped by DINOv2 similarity to tiger (threshold 0.25). Classes above the threshold lose a median of 28% against 11% for the rest, and lion, tiger’s nearest neighbour, degrades most. Bars mark group medians.

## 5 Conclusion

In this study, we examine how forgetting propagates when a model is made to drop a single class. A label-conditioned EBM made this easy to observe, since every class carries its own energy, letting us track how class-specific energies change rather than only verifying removal of the target. Forgetting is not fixed to where it is applied: removing a class from the sketch domain significantly drops accuracy on that class in real, clipart, and painting, though those samples are not in the forget set, and this held across all three targets. We also observe spillover to visually similar classes, strongest for nearest neighbours, though the magnitude varies across targets. Our approach controls how far this spreads, using DINOv2 similarity to determine where the forgetting signal is applied; disabling this component reduces the objective to isolated unlearning.

Limitations and future work. The spread to similar classes concentrates on the nearest neighbour and changes size with the target, so it is better read as a control to turn up or down than a precise setting. Removing a class also lowers accuracy slightly on retained classes, expected since shared features mean erasing one disturbs the others. Our experiments use one DomainNet subset and CIFAR-10 on small backbones, so they show the efect is real and steerable, not how far it can ultimately be pushed. Larger backbones, more datasets, and settings where controlled spread is the goal, such as removing a person’s identity together with its closest look-alikes, are the natural next steps.

## References

1. Basak, H., Yin, Z.: Forget more to learn more: Domain-specific feature unlearning for semi-supervised and unsupervised domain adaptation. In: European Conference on Computer Vision. pp. 130–148. Springer (2024)

2. Belanger, D., McCallum, A.: Structured prediction energy networks. In: Proceedings of the 33rd International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 48, pp. 983–992. PMLR (2016)

3. Bourtoule, L., Chandrasekaran, V., Choquette-Choo, C.A., Jia, H., Travers, A., Zhang, B., Lie, D., Papernot, N.: Machine unlearning. In: 2021 IEEE symposium on security and privacy (SP). pp. 141–159. IEEE (2021)

4. Cao, Y., Yang, J.: Towards making systems forget with machine unlearning. In: 2015 IEEE symposium on security and privacy. pp. 463–480. IEEE (2015)

5. Chen, M., Gao, W., Liu, G., Peng, K., Wang, C.: Boundary unlearning: Rapid forgetting of deep networks via shifting the decision boundary. In: 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 7766–7775. IEEE (2023)

6. Cheng, J., Liu, P., Li, Q., Zhang, C.: Machine unlearning under retain-forget entanglement (2026), https://arxiv.org/abs/2603.26569

7. Choi, D., Na, D.: Towards machine unlearning benchmarks: Forgetting the personal identities in facial recognition systems. arXiv preprint arXiv:2311.02240 (2023)

8. Chundawat, V.S., Tarun, A.K., Mandal, M., Kankanhalli, M.: Can bad teaching induce forgetting? unlearning in deep networks using an incompetent teacher. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 37, pp. 7210– 7217 (2023)

9. Du, Y., Li, S., Mordatch, I.: Compositional visual generation with energy based models. Advances in Neural Information Processing Systems 33, 6637–6647 (2020)

10. Du, Y., Mordatch, I.: Implicit generation and modeling with energy based models. In: Advances in Neural Information Processing Systems. vol. 32, pp. 3603–3613 (2019)

11. Duvenaud, D., Wang, J., Jacobsen, J., Swersky, K., Norouzi, M., Grathwohl, W.: Your classifier is secretly an energy based model and you should treat it like one. In: International Conference on Learning Representations (2020)

12. Fan, C., Liu, J., Zhang, Y., Wong, E., Wei, D., Liu, S.: Salun: Empowering machine unlearning via gradient-based weight saliency in both image classification and generation. In: International Conference on Learning Representations (2024)

13. Gao, R., Song, Y., Poole, B., Wu, Y.N., Kingma, D.P.: Learning energy-based models by difusion recovery likelihood. In: International Conference on Learning Representations (2021), https://openreview.net/forum?id=v\_1Soh8QUNc

14. Ginart, A., Guan, M., Valiant, G., Zou, J.Y.: Making ai forget you: Data deletion in machine learning. Advances in neural information processing systems 32 (2019)

15. Golatkar, A., Achille, A., Soatto, S.: Eternal sunshine of the spotless net: Selective forgetting in deep networks. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 9304–9312 (2020)

16. Jang, J., Yoon, D., Yang, S., Cha, S., Lee, M., Logeswaran, L., Seo, M.: Knowledge unlearning for mitigating privacy risks in language models. In: Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 14389–14408 (2023)

17. Jollife, I.T.: Principal Component Analysis. Springer, 2 edn. (2002)

18. Kawamura, K., Goto, Y., Yanagi, R., Kataoka, H., Irie, G.: Approximate domain unlearning for vision-language models. Advances in Neural Information Processing Systems 38, 21805–21833 (2025)

19. Kurmanji, M., Triantafillou, P., Hayes, J., Triantafillou, E.: Towards unbounded machine unlearning. Advances in neural information processing systems 36, 1957– 1987 (2023)

20. LeCun, Y., Chopra, S., Hadsell, R., Ranzato, M., Huang, F.J.: A Tutorial on Energy-Based Learning. MIT Press (2006)

21. Li, S., Du, Y., Van de Ven, G., Mordatch, I.: Energy-based models for continual learning. In: Conference on lifelong learning agents. pp. 1–22. PMLR (2022)

22. Liu, W., Wang, X., Owens, J., Li, Y.: Energy-based out-of-distribution detection. In: Advances in Neural Information Processing Systems. vol. 33, pp. 21464–21475 (2020)

23. Mishra, A., Nayak, G., Kumar, T., Shah, A., Bhattacharya, S., Foltin, M.: Selective, controlled and domain-agnostic unlearning in pretrained clip: A training- and datafree approach. arXiv preprint arXiv:2512.14113 (2025)

24. Nijkamp, E., Hill, M., Han, T., Zhu, S.C., Wu, Y.N.: On the anatomy of mcmcbased maximum likelihood learning of energy-based models. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 34, pp. 5272–5280 (2020)

25. Oquab, M., Darcet, T., Moutakanni, T., Vo, H.V., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba, W., Howes, R., Huang, P.Y., Li, S.W., Misra, I., Rabbat, M., Sharma, V., Synnaeve, G., Xu, H., Jegou, H., Mairal, J., Labatut, P., Joulin, A., Bojanowski, P.: Dinov2: Learning robust visual features without supervision. Transactions on Machine Learning Research (2024)

26. Peng, X., Bai, Q., Xia, X., Huang, Z., Saenko, K., Wang, B.: Moment matching for multi-source domain adaptation. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 1406–1415 (2019)

27. Sekhari, A., Acharya, J., Kamath, G., Suresh, A.T.: Remember what you want to forget: Algorithms for machine unlearning. Advances in Neural Information Processing Systems 34, 18075–18086 (2021)

28. Sepahvand, N., Triantafillou, E., Larochelle, H., Precup, D., Clark, J., Roy, D., Dziugaite, G.K.: Selective unlearning via representation erasure using domain adversarial training. In: International Conference on Learning Representations. vol. 2025, pp. 14918–14936 (2025)

29. Shaik, T., Tao, X., Xie, H., Li, L., Zhu, X., Li, Q.: Exploring the landscape of machine unlearning: A comprehensive survey and taxonomy. IEEE Transactions on Neural Networks and Learning Systems 36(7), 11676–11696 (2025)

30. Tarun, A.K., Chundawat, V.S., Mandal, M., Kankanhalli, M.: Fast yet efective machine unlearning. IEEE transactions on neural networks and learning systems 35(9), 13046–13055 (2024)

31. Tu, L., Gimpel, K.: Benchmarking approximate inference methods for neural structured prediction. In: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers). pp. 3313–3324. Association for Computational Linguistics (2019)

32. Wang, H., Liu, W., Bocchieri, A., Li, Y.: Can multi-label classification networks know what they don’t know? In: Advances in Neural Information Processing Systems. vol. 34, pp. 29074–29087 (2021)

33. Wang, X., Chen, C., Liu, W., Liao, X., Wang, F., Zheng, X.: Eficient sourcefree unlearning via energy-guided data synthesis and discrimination-aware multitask optimization. In: Forty-second International Conference on Machine Learning (2025)

34. Xie, J., Lu, Y., Zhu, S.C., Wu, Y.: A theory of generative convnet. In: International conference on machine learning. pp. 2635–2644. PMLR (2016)

35. Xu, H., Zhu, T., Zhang, L., Zhou, W., Yu, P.S.: Machine unlearning: A survey (2023), https://arxiv.org/abs/2306.03558

36. Xu, J., Wu, Z., Wang, C., Jia, X.: Machine unlearning: Solutions and challenges. IEEE Transactions on Emerging Topics in Computational Intelligence 8(3), 2150– 2168 (2024)