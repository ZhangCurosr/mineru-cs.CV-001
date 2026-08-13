# Distractor-Aware Video Object Segmentation

Andreas Robinson, Abdelrahman Eldesokey, and Michael Felsberg

Linkoping University, Sweden,¨ {firstname.lastname}@liu.se

Abstract. Semi-supervised video object segmentation is a challenging task that aims to segment a target throughout a video sequence given an initial mask at the first frame. Discriminative approaches have demonstrated competitive performance on this task at a sensible complexity. These approaches typically formulate the problem as a one-versus-one classification between the target and the background. However, in reality, a video sequence usually encompasses a target, background, and possibly other distracting objects. Those objects increase the risk of introducing false positives, especially if they share visual similarities with the target. Therefore, it is more effective to separate distractors from the background, and handle them independently.

We propose a one-versus-many scheme to address this situation by separating distractors into their own class. This separation allows imposing special attention to challenging regions that are most likely to degrade the performance. We demonstrate the prominence of this formulation by modifying the learning-whatto-learn [3] method to be distractor-aware. Our proposed approach sets a new state-of-the-art on the DAVIS 2017 val dataset, and improves over the baseline on the DAVIS 2017 test-dev benchmark by 4.6 percentage points.

## 1 Introduction

Semi-supervised video object segmentation (VOS) aims to segment a target throughout a video sequence, given an initial segmentation mask in the first frame. This task can be very challenging due to camera motion, occlusion, and background clutter. Several deep learning based methods have been proposed recently to address these challenges [3,8,9,11,12,15]. Among those, discriminative methods [3,11] have shown competitive performance at a reasonable computational cost, making them suitable for real-time applications, e.g., enhancing visual object tracking in crowded scenes, removing or replacing the background in video sequences or live conference calls, for privacy masking in surveillance videos, or as an attention mechanism in downstream vision tasks such as action recognition.

The majority of the discriminative approaches formulate the problem as a oneversus-one classification between the target and the background. Based on this, they attempt to construct a robust representation of the target that is as distinct as possible from the background. However, the background usually includes other objects that could be visually similar to the target. In this case, it might be challenging to find a good representation that discriminates the target from those distracting objects. This can usually lead to several false positive as the classifier is likely to fail given the underlying coarse representation between the target and the background. Figure 1 shows an example where a top-performing discriminative approach fails to discriminate between the target and other objects of the same type.

![](images/3e0c25d395eabff6e11a586cf84942cdf649df01ad9a67a772bfaaf1903717e0.jpg)  
Fig. 1. The impact of incorporating distractor-awareness into the baseline (LWL [3]). Our distractor-aware approach produces more accurate predictions than the baseline in highly ambiguous regions, where objects share visual similarities.

In this paper, we address this aforementioned challenge by reformulating the problem as a one-versus-many classification. We propose to separate the distracting objects from the background, and handle them as a distinct class. As a result, making the network aware of these distractors during training, promotes the learning of a robust representation of the target that is more discriminative against both the background and distractors.

We demonstrate the effectiveness of our approach by modifying the learning-whatto-learn (LWL) approach [3] to become distractor-aware. First, we modify the learning pipeline to incorporate information about the distractors. In case of videos with multiple objects, we initialize other objects in the scene as distractors. Second, to enhance the discriminative power of the network, we integrate high-resolution features to the target model. Finally, we introduce the use of adaptive refinement and upsampling [13] as a regularization to enforce local consistency at uncertain regions such as edges, and object-to-object boundaries.

Experiments show that our proposed framework sets a new state-of-the-art result on the DAVIS 2017 val dataset [10]. Moreover, we improve the results over the baseline on the DAVIS 2017 test-dev benchmark by a large margin. On the YouTube-VOS 2018 dataset [14], which is characterized by limited annotation accuracy at object-todistractor boundaries, our method still shows significant improvement over the baseline.

The remainder of the paper is organized as follows: we start by providing an overview of existing discriminative VOS approaches in the literature, distractor-awareness in other vision tasks, and existing upsampling and refinement methods in VOS. Next, we briefly describe the baseline approach, Learning-what-to-learn [3], followed by our proposed distractors modelling. Finally, we provide quantitative and qualitative results for our proposed approach in comparison with the baseline, and existing state-of-the-art methods as well as an ablation study.

## 2 Related Work

The video object segmentation (VOS) task can be tackled in a semi-supervised or an unsupervised manner, but we only consider the former in this paper. Semi-supervised approaches from the literature can usually be depicted as either generative or discriminative. Generative approaches [6,9] typically focus on constructing a robust model of the target of interest ignoring other objects in the scene. On the contrary, discriminative approaches [11,12] attempt to solve the task as a classification problem between the target an the background. A more recent method [3] follow an embedding approach to learn features that are as discriminative as possible. With the emergence of deep learning, the robustness of feature representations has significantly improved, boosting the performance of most variants of VOS approaches. However, the current top-performing semi-supervised VOS methods are mainly discriminative, taking into account both the target and the background when solving the task. Therefore, we focus on discriminative approaches in this paper.

Discriminative Video Object Segmentation Discriminative VOS methods were introduced quite recently. Yang et al. [15] proposed to build a target model from separate target and background feature embeddings, extracted from past images and target masks. For test frames, extracted features are matched against the two pretrained models of the target and the background. STM [9] incorporates a feature memory, encoding past images and segmentation masks. Similarly, Lu et al. [7] introduce an episodic graph memory unit to mine newly available features, and update the model accordingly. In contrast to Yang et al., both methods [9,7] produce the final target mask using a dedicated decoder network, from concatenated memory features and new image features. Seong et al. [12] extended STM [9] further with a soft Gaussian-shaped attention mask to limit confusion with distant objects. Robinson et al. [11] introduced the use of discriminative correlation filters to construct a target model that produces a coarse segmentation mask. This coarse mask is then enhanced and refined through a decoder network. Learning-what-to-learn [3] improved it further by learning to produce target embeddings that are more reliable for training the target model. All of these aforementioned papers adopt a one-vs-one classification between the target and the background, where other objects in the sequence are considered as background. In contrast, our approach reduces the likelihood of predicting false positives when some background objects share visual similarities with the target.

Distractor-Aware Modelling Distractor-aware modeling can be realized as a kind of hard-example mining when training a model. The general concept is to identify inputs that are more likely to confuse a given model and emphasis them during training, One of the earliest uses of this concept is found in the classical human detection algorithm, histograms of oriented gradients [4]. A more recent example [5] proposed marking flickering object detections in a video as hard-negatives, assigning them higher priority during training. A recent visual object tracking method [2] ranks target proposals based on their signal strength, where the strongest is assumed to be the target and the rest are distractors. The method tracks the complete scene state in a dense vector field over all spatial locations, using it to classify regions as either target, distractor or background. A similar Siamese-based approach [16] also classifies a target and any distractors through ranking of detection strengths. In both approaches, distractors are provided as hard examples during online training their respective tracker target models. Zhu et al. [16] also introduced distractor awareness to Siamese visual tracking networks as they noticed that the standard trackers posses imbalanced distribution of training data leading to less discriminative features. They propose an effective sampling strategy to make the model more discriminative towards semantic distractors. In this paper, we follow the strategy adopted by these approaches, and we introduce distractor-awareness to the task of video object segmentation.

Segmentation Mask Upsampling and Refinement Existing VOS CNNs employ different upsampling and refinement approaches to provide the final segmentation mask at the full resolution. RGMP and STM [8,9] employ two residual blocks with a bilinear interpolation in between for refinement and upsampling. Seong et al. [12] used a residual block followed by two refinement modules with skip connections. In contrast, Robinson et al.and Bhat et al.[11,3] replace the residual blocks with standard convolution, and employs bicubic instead of bilinear interpolation. All these approaches provide spatially independent prediction with no regularization to enforce local consistency, especially at uncertain regions such as edges and object boundaries. We employ the convex upsampler [13] to jointly upsample the final mask while enforcing spatial consistency.

## 3 Method

Ideally, in video object segmentation, it is desired to produce pixel-wise predictions y either as target T or background B. In a probabilistic sense, we are interested in maximizing the posterior probability for the target given an input embedding X:

$$
\begin{array} { l } { \displaystyle P ( Y = \mathcal { T } | X ) = \frac { P ( X , Y = \mathcal { T } ) } { P ( X ) } = \frac { P ( X , Y = \mathcal { T } ) } { P ( X , Y = \mathcal { T } ) + P ( X , Y = \mathcal { B } ) } } \\ { \displaystyle = \frac { 1 } { 1 + \frac { P ( X , Y = \mathcal { B } ) } { P ( X , Y = \mathcal { T } ) } } = \frac { 1 } { 1 + \frac { P ( X | Y = \mathcal { B } ) P ( Y = \mathcal { B } ) } { P ( X | Y = \mathcal { T } ) P ( Y = \mathcal { T } ) } } \ , } \end{array}\tag{1}
$$

The ratio in the denominator determines the posterior probability. If a pixel belongs to the target, the ratio becomes small and the posterior probability tends to 1. Contrarily, if a pixel belongs to the background and is quite distinct from the target, the ratio becomes large and the posterior probability goes to 0. However, if the target prior is large, and the background and target likelihoods are similar because X contains features of a distractor, the posterior can easily be larger than 0.5 and produce false positives. We propose splitting the non-target into two classes, background B and distractor D:

$$
\begin{array} { r } { P ( X , Y \neq \mathcal { T } ) = P ( X , Y = \mathcal { B } ) + P ( X , Y = \mathcal { D } ) \quad } \\ { = P ( X | Y = \mathcal { B } ) P ( Y = \mathcal { B } ) + P ( X | Y = \mathcal { D } ) P ( Y = \mathcal { D } ) ~ . } \end{array}\tag{2}
$$

Consequently, the ratio in (1) will have two terms in the enumerator as denoted by (2). This modification limits the occurrences of false positives as it models ambiguous pixels from the first frame and propagates them. As an example, if the likelihood of a certain pixel is similar between the target and the distractor at an intermediate frame, the propagated prior from previous frames will cause the ratio to be large, and the probability to drop. In the following sections, we will describe how to modify an existing baseline to be distractor-aware.

![](images/6b43c472e309106c3c294df3fd352d9c810f76ff00da6129046f2d778a651e6b.jpg)  
Fig. 2. An overview of our distractor-aware approach. We extend the baseline LWL to incorporate information about distractors throughout its mask encoder, target model, and segmentation decoder stages, and replace its upsampler with a joint refinement and upsampling approach, acting as local regularization.

## 3.1 Baseline Approach

We base our approach on the recently published method learning-what-to-learn [3] (LWL). In the LWL baseline, a target segmentation mask is encoded by a mask encoder into an multi-channel embedding at one sixteenth of the original image resolution. At the same time, a standard backbone encoder network (ResNet-50) is used to extract features from the whole image (see Figure 2). At the first frame, both the mask embeddings and the image features are utilized as training data for a target model $T _ { \theta } ( x ) = x * \theta$ that is trained with a few-shot learner. In subsequent video frames, this target model generates new multi-channel embeddings from the corresponding image features. A decoder network finally recovers segmentation masks at full resolution from these embeddings.

Eventually, the newly predicted masks and deep features are added to the training data, so that the target model can be tuned, adapting it to the changing target appearance. Note that multiple objects are tracked independently and individual target models do not interact with each other. In other words, the target model is not aware of any objects in the scene since everything other than the target is treated as background.

## 3.2 Introducing Distractor-Awareness

As describe in the related work section, there are several ways to detect distractors. Here, we consider other objects in the scene as potential distractors. More specifically, under the assumption that segmentation masks are binary, given a target masks $\mathbf { 1 } _ { t _ { i } }$ , we generate a distractor mask as:

$$
\mathbf { 1 } _ { d _ { i } } = \underset { j \neq i } { \cup } \mathbf { 1 } _ { t _ { j } } \ \forall j \in \mathbb { I } \ ,\tag{3}
$$

where $\mathbb { I } = \{ 1 . . . N \}$ is the set of target IDs in the current video sequence. Both the target and the distractor masks are set to the ground truth in the first frame, and updated with the previous predictions in later frames. To accommodate the distractor, we add a second input channel to the mask encoder and a second output channel to the segmentation decoder (see figure 2).

As the segmentation of frames progresses, we merge the decoded and upsampled target masks to form new distractors. However, in this case, the propagated masks are no longer binary and (3) needs to be replaced. For this, we develop a per-pixel winnertake-all (WTA) function.

Let $p _ { t _ { i } } ( x ) \in \mathbb { R } ^ { H \times W }$ be the target segment probability map, i.e. the network decoder output after a sigmoid activation, of the target with index i. Now let

$$
p _ { \operatorname* { m a x } } ( x ) = \operatorname* { s u p } _ { j } p _ { t _ { j } } ( x ) \forall j \in \mathbb { I } \ ,\tag{4}
$$

$$
p _ { \operatorname* { m i n } } ( x ) = \operatorname* { i n f } _ { j } p _ { t _ { j } } ( x ) \forall j \in \mathbb { I } \ ,\tag{5}
$$

merging the highest and lowest probabilities (per pixel), into $p _ { \mathrm { m a x } }$ and $p _ { \mathrm { m i n } }$

Now let, $L ( x ) \in ( \mathbb { I } \cup \{ 0 \} ) ^ { H \times W }$ be the map of merged segmentation labels (after softmax-aggregation, as introduced in [8]), with 0 the background label.

Also let

$$
\begin{array} { r } { \mathbf { 1 } _ { f } ( x ) = \left\{ \begin{array} { l l } { 1 \ } & { \mathrm { ~ i f ~ } L ( x ) > 0 , } \\ { 0 \ } & { \mathrm { ~ i f ~ } L ( x ) = 0 . } \end{array} \right. } \end{array}\tag{6}
$$

indicate regions with any foreground pixel, and

$$
\mathbf { 1 } _ { d _ { i } } ( x ) = \left\{ \begin{array} { l l } { \mathbf { 1 } _ { f } ( x ) } & { \mathrm { ~ i f ~ } L ( x ) \neq i , } \\ { 0 } & { \mathrm { ~ o t h e r w i s e ~ . ~ } } \end{array} \right.\tag{7}
$$

indicate distractors of target i. A new probability map of distractor i is generated as

$$
p _ { d _ { i } } ( x ) = \mathbf { 1 } _ { d _ { i } } ( x ) p _ { \operatorname* { m a x } } ( x ) + ( 1 - \mathbf { 1 } _ { f } ( x ) ) p _ { \operatorname* { m i n } } ( x ) .\tag{8}
$$

![](images/6eb807a3019c24536b9951aa507253dfc526a8b6548ff70d7a36ccdf6b07dc9b.jpg)  
Fig. 3. An overview of the refinement and upsampling module. The final decoder features are first projected into target and distractor logit maps. A weights estimation network then predicts a 5D tensor of weights in $3 \times 3$ windows around each pixel in the logit maps (shown in green), as well as interpolation data (shown in yellow). The weights are mapped to normalized probability vectors with the softmax function and then used refine the target and the distractor logits with weighted summation. The refined logits are finally upsampled to full resolution with the interpolation data.

In less formal terms, we let the the most certain predictions of target and background “win” in every pixel. $p _ { d _ { i } }$ takes information from every $p _ { t _ { j } }$ except that of target i.

In the ablation study, we compare the performance of this function to that of feeding back the decoded distractor to the few-shot learner without modification.

## 3.3 Joint Refinement and Upsampling for Local Consistency

The majority of existing VOS methods adopt a binary classification scheme between the foreground and the background. It is typically challenging to produce accurate predictions at uncertain regions such as target boundaries due to multi-modality along edges. This problem is aggravated when we introduce a new class for distractors, as the decision is now made between three classes instead of two. We tackle this problem by introducing two modifications to the baseline encoder and decoder, respectively.

First, we provide high-resolution feature maps from the backbone when training the target model in the few-shot learner. Unlike the baseline, we employ backbone features with a 1/8th of the full resolution instead of 1/16th. These higher-resolution features provide finer details to the target model, especially along edges, to help resolving ambiguities in uncertain regions. Second, we replace the baseline upsampler on the decoder output, with a joint refinement and upsampling unit based on the convex combination module of [13]. This is originally employed to upsample flow fields, while we adopt it to produce consistent selections between the target and distractors in logit space.

The proposed joint refinement/upsampling approach is illustrated in Figure 3. First, the output from the decoder is projected using two convolution layers to two channels resembling the likelihoods of the target and the distractor. Then, the likelihoods are unfolded into $3 \times 3$ patches around each pixel for computational efficiency. In parallel, we employ a weights estimation network that jointly perform likelihoods refinement, and mask upsampling. The former allows modifying the likelihoods of the target and the distractor based on their neighbors to enforce some local consistency, while the latter produces pixel values needed for upsampling.

For the the refinement, the weights estimation network predicts a local coefficients vector $\mathbf { c } _ { x }$ for a $3 \times 3$ window centered around each pixel $x \in X$

$$
{ \bf c } _ { x } = \left[ c _ { - 4 } , \ldots , c _ { - 1 } c _ { - 1 } , c _ { 0 } , c _ { 1 } , \ldots , c _ { 4 } \right] ,\tag{9}
$$

where $c _ { 0 }$ is the coefficient on top of x. However, these coefficients are not normalized, and to preserve likelihood ratios we first map $\mathbf { c } _ { x }$ to a normalized vector $\hat { \mathbf { c } } _ { x }$ using the softmax function. To modify the likelihoods, we subsequently apply the coefficients to the likelihoods using a weighted sum resembling convolution:

$$
P ( X | Y ) ^ { \prime } * \hat { \mathbf { c } } _ { x } [ x ] = \sum _ { m \in [ - 4 : 4 ] } P ( X | Y ) [ x - m ] \hat { \mathbf { c } } _ { x } [ m ] .\tag{10}
$$

For the upsampling, we need to predict pixel values for $4 \times 4$ patches around each pixel for a scaling of 4. Those values are also predicted using the weights-estimation network and are needed to produce the full resolution prediction. This combined refinementand-upsampling feature volume takes the shape of a local 3D tensor $\mathbf { A } _ { x } \in \mathbb { R } ^ { 4 \times 4 \times 9 }$ for each pixel $x \in X$

## 4 Experiments

In this section, we provide implementation details including the training procedure and loss function. We compare against state-of-the-art approaches on DAVIS 2017 [10], and YouTube-VOS 2018 [14]. In addition, we provide an extensive ablation analysis.

## 4.1 Implementation Details

Training procedure Similar to the baseline, our training setup imitates the segmentation processing during inference. Each training sample is a mini-video of four frames with one main target object to segment. The few-shot learner is provided the first frame to train a target model. The target model and the decoder then predicts segmentation masks from the subsequent three frames. The predictions in turn, are both used to update the target model and compute the network training loss.

To extend this procedure for distractors, we replicate the segmentation process for all other objects that are present in the ground-truth label map. These additional segments are then merged and provided as distractors to the main target. However, we set the network into evaluation mode when processing these targets to conserve memory.

We employ the same data augmentation and first two training phases as the baseline [3]. The proposed refinement and upsampling module is trained in a third phase for 6,000 iterations on DAVIS 2017, with learning rate $1 0 ^ { - 4 }$ , all other weights frozen.

Loss Function LWL was trained with the Lovasz-softmax loss [1], a differentiable´ relaxation of the Jaccard similarity measure, and we continue to do so here. However, in the SOTA experiments, we did initially not see any improvements with our method on YouTube-VOS. We hypothesize that this is due to large size-differences between objects in the same sequence in YouTube-VOS. To counter this, we split the loss into two terms, or a balanced loss:

$$
L = \operatorname { L o v a s z } ( T ) + w ( \hat { T } , \hat { D } ) \operatorname { L o v a s z } ( D )\tag{11}
$$

where $T$ and $D$ are the output batch from the decoder network, separated into target and distractor channels. $w ( \hat { T } , \hat { D } )$ reduces the influence of large objects as a function of the number of ground-truth target pixels $| \hat { T } |$ and ground-truth distractor pixels $| \hat { D } |$ across the training batch:

$$
w ( \hat { T } , \hat { D } ) = \operatorname* { m i n } ( | \hat { T } | / | \hat { D } | , 1 . 0 )\tag{12}
$$

In other words, when the distractors jointly occupy a larger region than the target, their influence on the loss is reduced.

Relaxed distractor loss Some sequences have only one target. This implies that no distractors exist, since we derive them from every other target. To not unnecessarily over-constrain the training, we partially disable the computation of the loss in training samples in these cases. Specifically, we require the distractor output to be zero in the area under the target, but allow it to take any value elsewhere.

We also train a variant with a “hard” loss, requiring that no distractor is output when no distractor is given, and compare them in the experiments.

## 4.2 State-of-the-art Comparisons

<table><tr><td>Method Name</td><td>DAVIS’17 val</td><td></td><td>DAVIS’17 test-dev</td><td></td><td></td><td>YouTube-VOS &#x27;18 valid</td><td></td><td></td><td></td></tr><tr><td></td><td>g J</td><td> $\mathcal { F }$ </td><td>g</td><td> $\mathcal { I }$ </td><td> $\mathcal { F }$ </td><td>g  $\mathcal { I } _ { \mathrm { s } }$ </td><td></td><td> $\mathcal { I } _ { \mathrm { u } }$ </td><td> $\mathcal { F } _ { \mathrm { s } }$   $\mathcal { F } _ { \mathrm { u } }$ </td></tr><tr><td>CFBI [15]</td><td>81.9 79.1</td><td>84.6</td><td>74.8 71.1</td><td>78.5</td><td>81.4</td><td>81.1</td><td>75.3</td><td>85.8</td><td>83.4</td></tr><tr><td>CFBI-MS [15]</td><td>83.3 80.5</td><td>86.0</td><td>77.5 73.8</td><td>81.1</td><td>82.7</td><td>82.2</td><td>76.9</td><td>86.8</td><td>85.0</td></tr><tr><td>KMN [12]</td><td>82.8 80.0</td><td>85.6</td><td>77.2 74.1</td><td></td><td>80.3 81.4</td><td>81.4</td><td>75.3</td><td>85.6</td><td>83.3</td></tr><tr><td>STM [9]</td><td>81.8 79.2</td><td>84.3</td><td>72.2</td><td>69.3</td><td>75.2</td><td>79.4 79.7</td><td>72.8</td><td>84.2</td><td>80.9</td></tr><tr><td>GMVOS [7]</td><td>82.8 80.2</td><td>85.2</td><td>-</td><td>-</td><td>-</td><td>80.2 80.7</td><td>74.0</td><td>85.1</td><td>80.9</td></tr><tr><td>LWL [3]</td><td>80.1 77.4</td><td>82.8</td><td>70.8</td><td>68.0</td><td>73.7</td><td>80.7 79.5</td><td>75.6</td><td>84.0</td><td>83.5</td></tr><tr><td>Ours</td><td>83.7 81.1</td><td>86.2</td><td>74.1</td><td>71.2</td><td>77.1</td><td>79.8 79.8</td><td>74.0</td><td>84.1</td><td>81.3</td></tr><tr><td>Ours (Balanced loss)</td><td>82.6 80.8</td><td>85.3</td><td>75.4</td><td>72.5</td><td>78.2</td><td>81.5 80.4</td><td>76.0</td><td>85.1</td><td>84.5</td></tr></table>

Table 1. Results on DAVIS 2017 and YouTube-VOS 2018, comparing our method to the LWL baseline and the state-of-the-art. The LWL scores are from our own run with the official code.

We compare against the most recent state-of-the-art approaches for video object segmentation: CFBI [15], STM [9], KMN [12], GMVOS [7], and the baseline LWL [3]. CFBI-MS is a multi-scale variant operating on three different scales. Our method in contrast, is single scale and thus more comparable to the plain CFBI. Running the official LWL implementation we noticed that our results do not coincide exactly with those reported in [3]. Therefore, we use these newly obtained scores to accurately compare the baseline to our method.

Table 1 summarizes the quantiative results on DAVIS 2017 and the YouTube-VOS 2018 validation split. On DAVIS val, our proposed approach without the balanced loss sets a new state-of-the-art on DAVIS val, while improving over the baseline on testdev by 3.6 percentage points. With the balanced loss however, test-dev surprisingly improves 4.6 percentage points, while lowering the gain on val to 2.5 percentage points. On the YouTube-VOS 2018 validation split, our approach improves the baseline by 0.8 percentage points, when trained with the balanced loss, while scoring similarly to other state-of-the-art approaches.

Some qualitative results are found in the supplement.

## 4.3 Ablation Study

<table><tr><td rowspan=1 colspan=1>ParametersL2D U B</td><td rowspan=1 colspan=1>DAVIS’17 valg   J   F</td><td rowspan=1 colspan=1>DAVIS’17 test-devg   J   F</td></tr><tr><td rowspan=3 colspan=1>√√√</td><td rowspan=1 colspan=1>80.177.482.8</td><td rowspan=1 colspan=1>70.868.0 73.7</td></tr><tr><td rowspan=1 colspan=1>80.777.9 83.4</td><td rowspan=1 colspan=1>71.067.7 74.2</td></tr><tr><td rowspan=1 colspan=1>80.778.283.3</td><td rowspan=1 colspan=1>71.768.7 74.8</td></tr><tr><td rowspan=2 colspan=1>√√      √</td><td rowspan=1 colspan=1>81.978.984.9</td><td rowspan=1 colspan=1>72.769.5 75.9</td></tr><tr><td rowspan=1 colspan=1>81.678.984.3</td><td rowspan=1 colspan=1>71.668.5 74.7</td></tr><tr><td rowspan=2 colspan=1>√ √√ V√</td><td rowspan=1 colspan=1>81.378.883.9</td><td rowspan=1 colspan=1>73.270.0 76.4</td></tr><tr><td rowspan=1 colspan=1>83.781.186.2</td><td rowspan=1 colspan=1>74.171.2 77.1</td></tr><tr><td rowspan=1 colspan=1>Hard lossNo DistractorsNo WTA</td><td rowspan=1 colspan=1>80.277.582.881.178.783.483.080.585.5</td><td rowspan=1 colspan=1>74.972.4 77.471.268.0 74.471.868.4 75.2</td></tr><tr><td rowspan=1 colspan=1>√VV1</td><td rowspan=1 colspan=1>82.680.885.3</td><td rowspan=1 colspan=1>75.472.5 78.2</td></tr></table>

Table 2. Ablation study on DAVIS 2017. See Section 4.3 for details. Interesting scores are printed in bold type.

In this section, we analyze the impact of different components of our method on the overall performance. We use the DAVIS 2017 dataset [10] for this purpose, where we include both val and test-dev splits for diversity. The results are shown in Table 2. Since LWL [3] is our baseline method, we reevaluate their official pretrained model. As mentioned above, there is a discrepancy between our obtained scores and the published ones. This could be due to several factors, e.g. hardware/software differences and driver variations. However, we use the scores that we obtained for a valid comparison.

We first enable or disable various combinations of these components: the highresolution features (L2), the distractor-awareness (D), the refinement/upsampling module (U). Enabling D alone causes a slight improvement. Adding U to this improves the test-dev split a further. We believe that the lack of high-resolution features prevents further gains. Similarly, introducing L2 alone, causes a marginal improvement on both splits and enabling U at the same time, shows no improvement on the valdiation set, and hurts the performance on the test-set. This can be explained by our argument in section 3.3: When the decision is made between two classes (target and background), there is no need for the upsampler to resolve ambiguities. As to why this combinatios hurts test-dev performance, is unclear. However, enabling all three (L2+D+U) at the same time, yields significantly better results over the baseline.

With L2, D and U included, we then separately replace the relaxed distractor loss with the “hard” loss (Hard loss). The results then drops back to the baseline for DAVIS validation split, but oddly improves on the test-dev split.

We also test a pair of inference-time modifications. First, we zero out the distractor masks so that the few shot learner do not use them. Second, we replace the WTA distractor-merging function with a simple pass-through; Distractors are directly routed from the decoder output back to the few-shot learner, bypassing the merging step. As one would expect, the results drop in both cases, proving that WTA is important. Surprisingly though, disabling the WTA function hits DAVIS test-dev split much harder than the validation split.

Finally, we add the balanced loss (B) to the L2+D+U variant. This clearly damages the DAVIS validation scores but improves the test-dev results. However, at the same time it also improves the YouTube-VOS results in Table 1.

These results suggest that the method works, but also reveal differences between the DAVIS dataset splits that we cannot explain at this time.

## 4.4 A Note on WTA vs. Softmax Aggregation

The softmax aggregation introduced in [8], and used to merge estimations of multiple targets, was introduced as a superior alternative to a winner-take-all approach. In our case however, we found that to merge distractors, WTA performs better. We hypothesize that lacking a dedicated background channel is beneficial to the refinement and upsampling module, as the decoder may output low activations on both target and distractors when the classification is uncertain.

## 4.5 Emergent distractors

An essential question is how our framework would behave when there is only one labeled object in the scene. Interestingly, the model learns to identify ambiguous regions. Figure 4 shows an example where our approach learns to identify the camel in the background without any explicit supervision (if columns 2+3 from the left). We attribute this to the relaxed distractor loss described in Section 4.1. To test this, we modify the loss to be “hard” (H). As suspected, this suppresses the behaviour greatly, while increasing the decoders’ certainty in both the target and background (columns 4+5).

RGB Image  
(R) Target  
(R) Distractor  
(H) Target  
(H) Distractor  
![](images/f5a2ebacb3f110d57d3a51298d66cd5740b26ad48e29df13af47bb85ab915a87.jpg)  
Fig. 4. Frames 40, 50 60 (from top to bottom) from the ’camel’ sequence, with target/distractor score maps. Our model can learn to identify distractors in case they were not explicitly provided, provided it is trained with the relaxed distractor loss. Red/yellow indicates positive loglikelihoods, blue/cyan negative. The colors represent the same value range in all columns. (R) and (H) indicate results from models trained with the relaxed and hard distractor losses, respectively.

## 5 Conclusion

We have proposed a distractor-aware, discriminative video object segmentation approach. In contrast to existing methods, our proposed method encodes distractors into a separate class, to exploit information about other objects in the scene that are likely to be confused with the target. Moreover, we propose the use of joint refinement and upsampling to regularize the likelihoods for highly uncertain regions with their neighborhoods. We demonstrated the effectiveness of our approach by modifying an existing state-of-the-art approach to be distractor-aware. Our modification sets a new state-ofthe-art on the DAVIS 2017 val dataset, while improving over the baseline with a remarkable margin on the DAVIS 2017 test-dev dataset. These results clearly indicate the efficacy of explicitly modelling distractors when solving video object segmentation.

## 6 Acknowledgements

This project was partially supported by the Wallenberg AI, Autonomous Systems and Software Program (WASP) funded by the Knut and Alice Wallenberg Foundation, the Excellence Center at Linkoping-Lund in Information Technology (ELLIIT), the¨ Swedish Research Council grant no. 2018-04673, and the Swedish Foundation for Strategic Research (SSF) project Symbicloud.

## References

1. Berman, M., Triki, A.R., Blaschko, M.B.: The lovasz-softmax loss: A tractable surrogate´ for the optimization of the intersection-over-union measure in neural networks. In: 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (June 2018)

2. Bhat, G., Danelljan, M., Van Gool, L., Timofte, R.: Know your surroundings: Exploiting scene information for object tracking. In: The European Conference on Computer Vision (ECCV) (August 2020)

3. Bhat, G., Lawin, F.J., Danelljan, M., Robinson, A., Felsberg, M., Van Gool, L., Timofte, R.: Learning what to learn for video object segmentation. In: The European Conference on Computer Vision (ECCV) (August 2020)

4. Dalal, N., Triggs, B.: Histograms of oriented gradients for human detection. In: 2005 IEEE computer society conference on computer vision and pattern recognition (CVPR’05)

5. Jin, S., RoyChowdhury, A., Jiang, H., Singh, A., Prasad, A., Chakraborty, D., Learned-Miller, E.: Unsupervised hard example mining from videos for improved object detection. In: The European Conference on Computer Vision (ECCV) (September 2018)

6. Johnander, J., Danelljan, M., Brissman, E., Khan, F.S., Felsberg, M.: A generative appearance model for end-to-end video object segmentation. In: 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (June 2019)

7. Lu, X., Wang, W., Martin, D., Zhou, T., Shen, J., Luc, V.G.: Video object segmentation with episodic graph memory networks. In: The European Conference on Computer Vision (ECCV) (August 2020)

8. Oh, S.W., Lee, J.Y., Sunkavalli, K., Kim, S.J.: Fast video object segmentation by referenceguided mask propagation. In: 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (June 2018)

9. Oh, S.W., Lee, J.Y., Xu, N., Kim, S.J.: Video object segmentation using space-time memory networks. In: 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (June 2019)

10. Pont-Tuset, J., Perazzi, F., Caelles, S., Arbelaez, P., Sorkine-Hornung, A., Van Gool, L.: The´ 2017 davis challenge on video object segmentation. arXiv:1704.00675 (2017)

11. Robinson, A., Lawin, F.J., Danelljan, M., Khan, F.S., Felsberg, M.: Learning fast and robust target models for video object segmentation. In: 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (June 2020)

12. Seong, H., Hyun, J., Kim, E.: Kernelized memory network for video object segmentation (August 2020)

13. Teed, Z., Deng, J.: Raft: Recurrent all-pairs field transforms for optical flow. In: The European Conference on Computer Vision (ECCV) (August 2020)

14. Xu, N., Yang, L., Fan, Y., Yue, D., Liang, Y., Yang, J., Huang, T.: Youtube-vos: A large-scale video object segmentation benchmark. arXiv preprint arXiv:1809.03327 (2018)

15. Yang, Z., Wei, Y., Yang, Y.: Collaborative video object segmentation by foregroundbackground integration. In: The European Conference on Computer Vision (ECCV). Springer (August 2020)

16. Zhu, Z., Wang, Q., Li, B., Wu, W., Yan, J., Hu, W.: Distractor-aware siamese networks for visual object tracking. In: The European Conference on Computer Vision (ECCV) (September 2018)

![](images/05a89894d1acc30138876167822255ad57a1648a48c3a12f0f886b50e988a814.jpg)  
Fig. 5. Qualitative results on the DAVIS val dataset. In the top row, the baseline confuses the background for the gun due to visual similarity, while our approach is successful. In the middle row, our approach produces a more complete segmentation of the green box, and achieves good separation between the background and the yellow leg. In the bottom row, our approach produces better segmentation for fine objects due to the use of high-resolution features.

## A Qualitative Results

Figure 5 shows some qualitative results comparing our approach to the LWL baseline on the DAVIS 2017 val dataset. The top row shows an example for how our approach is resilient against distractors in the background. While the baseline confuses an element in background with the gun since they have similar textures, our approach performs a more robust discrimination. On the second row, our approach produces a more accurate segmentation of all objects without bleeding to other intersecting objects. On the final row, the impact of incorporating high-resolution features is evident as our approach successfully detects fine objects.

## B Additional Implementation Details

In addition to adding a channel for the input and output distractors, we keep the architectures of the mask-encoder and few-shot learner fundamentally unchanged. However, the number of channels in the encoded mask representations is increased from 16 to 32 in the best model, to accommodate the distractors. In addition, it proved necessary to configure the few-shot learner to use five rather than three iterations of its internal optimizer, for our model to reach maximum performance. On the other hand, increasing the number of iterations for the baseline yielded no improvements.