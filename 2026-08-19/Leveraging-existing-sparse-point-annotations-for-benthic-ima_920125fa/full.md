# Leveraging existing sparse point annotations for benthic imagery dense segmentation

Cesar Borja<sup>1</sup> , Breck A. McCollum<sup>2</sup> , Jarrett E. Byrnes<sup>3</sup> , Kenneth P. Sebens<sup>4</sup> , and Ana C. Murillo<sup>1</sup>

<sup>1</sup> DIIS - I3A, Universidad de Zaragoza, Zaragoza, Spain {cborja,acm}@unizar.es

<sup>2</sup> Berklee College of Music, Boston, USA

bmccollum@berklee.edu

3 University of Massachusetts: Boston, Boston, USA Jarrett.Byrnes@umb.edu

<sup>4</sup> University of Washington, Seattle, USA sebens@uw.edu

Abstract. The health of marine ecosystems is a critical indicator of global environmental change, yet the physical constraints of underwater observation and the intrinsic challenges of processing marine imagery severely limit the scalability of systematic monitoring. While recent visual foundation models such as the Segment Anything Model (SAM) series show great promise, they still struggle with the fine-grained recognition required in these complex scenarios and still require expert supervision. Our work addresses this gap by bridging state-of-the-art foundation models with existing sparse supervision. Because historical benthic surveys are typically annotated with only a few sparse expert points per image, we utilize these legacy point-labels as visual prompts for SAM2. Our primary contribution is a novel mechanism to automatically identify which of these points are suitable, and which are actively harmful, when used for propagation. By filtering out unreliable points, we extract highquality pseudo-ground-truth masks capable of training more accurate, fine-grained semantic segmentation models. We demonstrate the efectiveness of our approach on public benthic data and introduce a new, challenging benchmark featuring real-world sparse expert annotations, paving the way for scalable ecological analysis.

Keywords: Automated benthic image analysis · Fine-grained semantic segmentation · Point-label propagation · Pseudo-label refinement

## 1 Introduction

Human-induced climate change has afected the distribution of numerous terrestrial, marine, and aquatic species around the globe [5, 12]. During the last 60 years, approximately 84% of all heat produced by global warming has been absorbed by the oceans [2, 7], leading to range shifts in marine species as waters warm [26]. To document and monitor these ecological range shifts, marine ecologists heavily rely on fixed-location photographic surveys of benthic environments. These visual “snapshots” capture crucial data on species composition over time [9], allowing researchers to count, measure, and identify species in the laboratory rather than enduring physically demanding and logistically expensive field conditions.

![](images/a96fe54bc8b73cbae85fd20d000517f3e0a0f3eb39b4c599f80aaae791072134.jpg)  
(a) Ideal mask

![](images/ff4d9432dd2bbe352b3b3ebb187053c0b4ff8e353d9edd290c871029cb764780.jpg)  
(b) Unsuitable point-label

![](images/b95a062a5f14ab75bb063b93d0100ffec5cd47b472fdee59fe5cf428deea0266.jpg)  
(c) Suitable point-label  
Fig. 1: Expanding sparse expert point-labels into dense masks is a common strategy to enable automated benthic image analysis. Since sparse point-label locations were usually placed random or uniformly, they often fall on ambiguous locations. Our framework automatically identifies and filters unsuitable points to significantly improve downstream segmentation performance. a) Ideal segmentation mask for class droebachiensis with two point-labels of the same class. b) Unsuitable point-label whose propagation bleeds across class boundaries, introducing severe label noise. c) Suitable point-label that propagates correctly.

While photographic data analysis is more eficient than many methods of collecting biodiversity data in the field, the process typically followed by ecology researchers to analyze these photos is very time consuming and requires a skilled researcher with extensive expertise in species identification. This huge manual efort is already frequently supported by the use of software tools to assist with annotation and visualization tasks. Broadly used platforms like CoralNet [3] have enabled a vast amount of annotations for benthic imagery databases. However, they do not contain dense, pixel-wise annotations, but rather rely on sparse point-level annotations. These annotations typically consist of tens to a few hundred randomly or uniformly sampled pixels per image.

Recent advances in computer vision, particularly visual foundation models, ofer a promising pathway to automate further benthic image analysis via semantic segmentation. However, even the top performing models, such as the SAM model family [6, 11, 20], sill struggle with fine grained and rare classes segmentation, precisely the case for most of the classes included in these datasets. Therefore, further supervision from experts to obtain more accurate and specific models. Since often the expert supervision is only sparse point labels, a common paradigm to train dense segmentation models on existing benthic data is pointlabel propagation. In this process, sparse expert point-labels are expanded into dense pseudo-ground-truth masks, leveraging superpixels [1,18] or more recently deep features like DINOv2 [19].

However, a critical flaw in current pipelines is the implicit assumption that all points are equally suitable for propagation. Because ecological survey points are historically sampled randomly or uniformly without regard for visual boundaries, many fall on class edges or ambiguous textures. Expanding these unsuitable seed points generates regions that bleed across object boundaries, overwriting adjacent classes. While the original point label may be correct, the resulting pseudo-mask is not; it injects severe label noise that degrades any downstream segmentation model trained on it. As illustrated in Fig. 1, our central hypothesis is that automatically distinguishing suitable from unsuitable points prior to propagation is critical for improving downstream segmentation performance in challenging real-world datasets.

While recent active learning approaches attempt to guide experts on where to place new annotations for optimal model prompting [4,19], we address a complementary problem: given the vast existing repositories of sparse annotations, which points can be trusted for propagation, and which parts of the resulting masks should be retained?

Overall, our contribution to automated analysis of benthic imagery is twofold:

– A Novel Framework for Point-Label Propagation. We identify and quantify an overlooked failure mode in label propagation: the expansion of correctly-labeled but visually ambiguous points into harmful pseudo-masks. To solve this, we introduce an approach based on two novel stages:

1. Pruning: A pre-expansion stage that automatically identifies and discards unsuitable points, preventing the generation of noisy masks.

2. Trimming: A post-processing stage that refines the surviving masks by cutting and discarding regions that spill or overlap into adjacent classes.

Our experiments demonstrate that these combined stages significantly improve the quality of dense masks generated from sparse labels.

– A New Benthic Segmentation Benchmark. We introduce a novel dataset derived from a long-term monitoring project, including expert point-labels placed at randomly selected locations. This dataset illustrates the real-world challenge of unsuitable propagation points. We validate our framework on this new data, as well as on established public datasets, demonstrating significant performance gains under randomly sampled point-labels.

To facilitate further research and provide the ecology community with improved analytical tools, our entire framework, including data, models, and code, is made publicly available. Our approach is directly extensible to other finegrained ecological image datasets facing similar annotation constraints.

## 2 Related Work

Sparse point-label propagation for benthic imagery. The CoralNet paradigm [3] established sparse expert points as the dominant annotation regime for benthic surveys. To train dense segmentation models from such labels, CoralSeg [1] and

PLAS [18] propagate points to dense masks via superpixels, enforcing label purity within each superpixel; D+NN [19] propagates with denoised DINOv2 [16] features and nearest-neighbor matching; and our own predecessor, SSeg [4], expands points with a hybrid SAM2-plus-superpixel scheme and actively guides where new points are placed. However, often the labeled point set is already provided with the ecological surveys, and the active point-selection paradigm of SSeg can not be applied, unless a new labeling job is performed. CoralSCOP [36] is a coral-specialized foundation segmenter, an alternative mask generator to SAM2. All of these treat the propagated region as trusted, or filter only at the whole-superpixel level; none detects and cuts cross-class drift within a grown mask, nor calibrates a removal threshold against points that are likely-wrong or unsuitable for propagation for other reasons. This is precisely one of the ideas developed in our work, proposing a novel point pruning stage before propagating the point-labels into segments.

Our pruning belongs to the instance-selection / label-noise filtering family and realizes the pruning principle of confident learning [15]. Approaches such as AUM [17] identify mislabeled examples through training-time margin statistics; we obtain a verified set of unsuitable points directly from held-out reference points, so the threshold is calibrated without training or tuning any model.

Pseudo-label reliability and refinement. In order to handle unreliable pseudolabels we find semi-supervised methods based on confidence thresholding, such as FixMatch [25] and its adaptive successors FlexMatch [34] and FreeMatch [28], or methods treating low-confidence regions separately U2PL [30]. Mask-refinement methods such as SAMRefiner [13] regrow cleaner masks with a foundation model. Our approach includes a refinement stage for trimming the obtained masks if necessary. It is inspired on the pseudo-mask refinement step in U2PL [30]: we partition pixels into reliable and unreliable and keep the unreliable ones out of the computations. The main diference of our refinement step is how we compute the reliability signal. Ours is based on embedding distance to a per-image class prototype, instead of the previously used model confidence. A parallel line of work tolerates the noise at training time instead of filtering it. Noise-robust losses [14, 29, 33, 35] bound the loss on mislabeled examples, assuming they produce a high loss. In our setting a mislabeled region is large, spatially coherent and internally consistent, so the model likely fits it fluently and at low loss, leaving nothing for a bounded loss to down-weight. We compare against Generalized Cross Entropy in Section 5.2

Prototype-based segmentation. Representing a class by the mean of its embeddings and classifying by nearest prototype is a classic non-parametric approach [24,27]. Our trimming step is exactly a per-image nearest-prototype classifier in the style of ProtoSeg [37] and PANet [27], but repurposed as a reliability filter on a foundation-model mask rather than as the primary segmenter.

Foundation model for image segmentation. We expand point-labels into dense segmentation masks by using each point as visual prompt for SAM2 [11, 20], which returns the most likely segmentation for the object where that point is contained. However, dense transformer features are known to carry artifacts: high-norm outlier tokens [8] and position-driven grid patterns [32] that mislead similarity-based propagation. These artifacts can cause patch-level drift, which motivates checking the reliability of the propagated masks. Therefore, we use a complementary vision model to describe small image patches, DINOv3 [23], which is the core ingredient in our verification steps, as detailed in Section 4.

## 3 Benthic datasets: a new long-term monitoring benchmark

Among the existing public benthic datasets we select two commonly used in recent works that contain a large number of high quality semantic segmentation annotations for a large variety of classes. In particular, we use data from Coralscapes [21], a densely-annotated coral-reef dataset of 39 classes whose images are extracted from underwater survey videos. The second is UCSD Mosaics [10], built from 16 large multi-species coral-reef mosaics with a resolution of 10K × 10K pixels and captured orthogonally, looking straight down at the seabed. CoralSeg [1] introduced a cropped version of this dataset by dividing the mosaics into $5 1 2 \times 5 1 2$ images, which we adopt in our work. Both of these datasets provide dense segmentation labels that allow us to evaluate the quality of the point-label propagation strategies considered in this work more accurately than with sparse point-labels frequently available in benthic surveys.

However, these datasets do not capture completely the challenges and limitations implicit to existing benthic imagery studies, such as those motivating this work. Therefore, we prepare and release a novel benchmark of data from a longterm monitoring project that includes images captured along large time spans (years). In addition, and actually what makes the tasks of label propagation more challenging, the point label annotation task was not explicitly guided by an endtask of segmentation. This means that the points were carefully annotated by marine biology experts, but these points were placed at random locations in the image. Figure 2 shows an example image from each of the three datasets.

The new presented benchmark consists of benthic imagery covering 24 foreground classes, 313 training and 78 test images, and about 180 expert pointlabels per image [22]. The annotations performed by expert biologists consist of 200 random points placed on each photo using CoralNet. Points falling on mobile fauna were excluded from the analysis.

The train/test split is performed randomly at the image level. All the information of the data used in our experiments is available in the project site <sup>5</sup>.

The photos included in this benchmark represent a sample of all photos taken that need to be analyzed for a larger, novel long-term monitoring project examining how the sessile invertebrate community on subtidal rock walls has changed over time. This larger study represents one of very few ongoing temperate long-term subtidal monitoring projects thus providing a rich and unparalleled data source for documenting change in this diverse and dynamic ecosystem. The whole project consists of vast amounts of data, with photos taken annually in the summer at multiple sites around the same location. Therefore, developing automated segmentation techniques is an extremely helpful tool to allow the adequate processing of all this survey data.

![](images/937752677e305e1eab001fe354bc59cc739e6865baea802c00cf2505a9962720.jpg)  
(a) Our dataset

![](images/ca740648acccf0a374c99ab5885c0cd7a40cb34f6313a76a07cae358807126dd.jpg)  
(b) Coralscapes

![](images/8688396ee1fba99a5050e750745b5f4d2254e8b4309dbf47889705ff1a420857.jpg)  
(c) UCSD  
Fig. 2: Visual comparison of the benthic imagery datasets: (a) Ours, (b) CoralScapes [21], and (c) UCSD Mosaics [1, 10].

The class inventory in this dataset is very challenging for existing generic models and appearance-based methods, as it contains near-identical colonial tunicate species, morphologically similar encrusting forms, and a heterogeneous mixed-turf category.

## 4 Segmentation-aware sparse point label propagation

## 4.1 Overview

We propose a method that given an image with a set of sparse labeled points, generates a dense semantic segmentation of the image. In particular, each point of the fixed, pre-defined, set of sparse point-labels is propagated into a segment using SAM2 (i.e., each point is provided as a visual prompt for the model). The output of SAM2 is the most likely object segment where that point is contained. All the masks provided by SAM2 are combined into the final image segmentation. However, because SAM2 is class-agnostic, a point’s specific spatial location can cause its expanded mask to bleed across semantic boundaries into regions of other classes. To detect and mitigate these semantic leaks, our method evaluates and refines the expansions through four sequential stages, as illustrated in Figure 3:

1. Propagation. Each given input labeled point is provided as a visual prompt for SAM2 [20], in order to propagate the initial sparse point-labels, resulting in one independent dense mask per point-label.

2. Pruning. With all the masks generated after propagating the input points, our pruning step identifies and removes point-labels, and their corresponding generated mask, that look unsuitable for propagation. Intuitively, this step filters out unreliable point-labels whose generated masks bleed into incorrect class regions. Section 4.2 details how this proposed step is implemented.

![](images/05385bbcf97b2498177489a784bd770a95a7ff83e0a02c0ac83e5b63895f6550.jpg)  
Fig. 3: Overview of our pipeline. Given an image with sparse point-labels, we propagate (1) each point-label i with SAM2. Using DINOv3 features, we calculate per-image class prototypes $\mu _ { c } ,$ and with them, we prune (2) the point-labels whose mask leak into other classes (i<sub>2</sub>), and trim (3) the patches that disagree with their mask’s class (p1, p4 and p5). The surviving masks are merged (4) into the final dense labels by ground-truth voting.

3. Trimming. While pruning discards unreliable point-labels and their propagated masks entirely, the remaining masks may still contain localized errors. We evaluate each mask internally to identify regions that visually disagree with its class. Rather than discarding the mask completely, we trim these inconsistent regions by excluding them from the mask, preserving only the reliable parts. Section 4.3 describes this proposed process in more detail.

4. Merging. The trimmed, independent masks are combined into a single dense pseudo-mask, resolving any overlapping class conflicts via the same groundtruth voting proposed in SSeg [4].

## 4.2 Pruning

This stage evaluates the initial set of sparse point-labels to identify and entirely remove those that are unsuitable for propagation with SAM2. To do this, we define a pruning score for each point-label and a calibrated threshold to decide which points to remove along with its expansion.

Pruning Score. Each image is encoded once through DINOv3 [23] into a grid of patch embeddings. Every patch of 16 × 16 pixels is represented by a single 1024- dimensional vector. Let $f _ { p }$ be this vector for patch p, which we L2-normalize. Since the grid is fixed, every point-label lies in exactly one patch.

For every class c present in the image, we define a local prototype $\mu _ { c }$ as the L2-normalized mean of the embeddings from the patches that contain class-c point-labels:

$$
\mu _ { c } = \frac { \sum _ { p \in P _ { c } } f _ { p } } { \left\| \sum _ { p \in P _ { c } } f _ { p } \right\| _ { 2 } } ,\tag{1}
$$

where $P _ { c }$ is that set of patches. Prototypes are per-image rather than global, so each one captures the appearance of its class under the illumination and blur of that specific scene.

Because $f _ { p }$ and $\mu _ { c }$ are unit vectors, their dot product equals their cosine similarity. Consider a point-label i of class $y _ { i }$ and its propagated mask $M _ { i }$ from stage (1). We look at the patches covered by $M _ { i }$ and assign each one to the class ypˆ of its nearest prototype:

$$
\hat { y } p = \arg \operatorname* { m a x } c ( f _ { p } \cdot \mu _ { c } ) .\tag{2}
$$

Since $M _ { i }$ is defined at the pixel level whereas patches are fixed $1 6 \times 1 6$ tiles, a patch at the mask boundary may fall only partially inside it. We count a patch as covered by $M _ { i }$ when more than half of its pixels lie within the mask.

A patch $p$ disagrees when ypˆ $\neq y _ { i }$ , i.e. it look more like another class than the one i assigns to it. The pruning score $S _ { p r u n e } ( i )$ is the fraction of the patches covered by $M _ { i }$ that disagree:

$$
S _ { p r u n e } ( i ) = { \frac { \# \ { \mathrm { p a t c h e s ~ o f ~ } } M _ { i } \ \mathrm { w i t h } \ \hat { y } _ { p } \neq y _ { i } } { \# \ { \mathrm { p a t c h e s ~ o f ~ } } M _ { i } } }\tag{3}
$$

A high $S _ { p r u n e } ( i )$ means that most of the region SAM2 would label $y _ { i }$ actually looks like other classes, marking point-label i as unsuitable for propagation.

Verification-calibrated threshold. Rather than fixing the pruning threshold by hand, we calibrate it from the current data, as is common in label-error detection [15], against point-labels that we can verify are unsuitable for propagation. To do so, we split each image’s point annotations into an expansion set (75%), the points we actually propagate, and a validation set (25%) held out only as witnesses. A point-label of the expansion set is verified unsuitable when its propagated mask $M _ { i }$ covers a validation point of a diferent class: a direct proof, against a real held-out label, that its expansion leaks into another class.

We gather the pruning scores of all verified-unsuitable point-labels across the training images,

$$
A _ { \mathrm { p o i n t } } = \{ S _ { \mathrm { p r u n e } } ( i ) \mid M _ { i } { \mathrm { ~ c o v e r s ~ a ~ v a l i d a t i o n ~ p o i n t } } \mathrm { { \mathrm { - l a b e l ~ o f ~ a ~ d i f f e r e n t ~ c l a s s } } } \} ,\tag{4}
$$

and set the threshold to their median, $\tau _ { \mathrm { p r u n e } } = \mathrm { m e d i a n } ( A _ { \mathrm { p o i n t } } )$ . A point-label i is then pruned when $S _ { \mathrm { p r u n e } } ( i ) \geq \tau _ { \mathrm { p r u n e } }$ , i.e. when it scores at least as poorly as a typical verified-unsuitable point. We use the median so the threshold reflects a typical verified-unsuitable point rather than the extremes. If $A _ { \mathrm { p o i n t } }$ is empty, i.e. no propagated mask in the training set covers a validation point-label of a diferent class, no verified evidence of harmful propagation exists and we prune nothing, recovering the keep-all baseline. In practice this does not occur on the datasets we evaluate.

Finally, we standardize scores per image so a single threshold is comparable across scenes. Points are then removed worst-first, in decreasing order of $S _ { p r u n e } ( i )$ , subject to a class guard: a point is pruned only while its class still retains more than one point-label in the image, so every class keeps at least its lowest-scoring point-label. The number of pruned points is thus not fixed but difers per image.

## 4.3 Trimming

Pruning discards whole point-labels, but a surviving mask can still over-extend locally into regions of other classes. Trimming refines each surviving mask by removing such regions patch by patch, reusing the DINOv3 features and prototypes from the previous stage.

For a patch p of a surviving mask of class $y _ { i }$ , we compare its cosine similarity to its own class prototype, $f _ { p } \cdot \mu _ { y _ { i } }$ , against its cosine similarity to the most similar class, $\operatorname* { m a x } _ { c \neq y _ { i } } ( f _ { p } \cdot \mu _ { c } )$ . The trimming score is the diference between the two:

$$
S _ { \mathrm { t r i m } } ( p ) = \operatorname* { m a x } _ { c \neq y _ { i } } ( f _ { p } \cdot \mu _ { c } ) - ( f _ { p } \cdot \mu _ { y _ { i } } )\tag{5}
$$

It is positive when the patch is more similar to another class than to its own, and grows the more it looks like that other class.

We calibrate the threshold as in pruning, now at patch granularity. We collect the trimming scores of the verified-unsuitable patches (those covering a validation point-label of a diferent class):

$A _ { \mathrm { p a t c h } } = \{ S _ { \mathrm { t r i m } } ( p ) \mid p$ covers a validation point-label of a different class}, (6) and take their median, $\tau _ { \mathrm { t r i m } } = \mathrm { m e d i a n } ( A _ { \mathrm { p a t c h } } )$ . A patch $p$ is trimmed when $S _ { \mathrm { t r i m } } ( p ) \geq \tau _ { \mathrm { t r i m } } .$ , excluding all its pixels from the mask.

## 5 Experiments

## 5.1 Experimental setup

Semantic Segmentation network training. To measure the quality of the generated pseudo-masks, we train a semantic segmentation model with them as supervision and evaluate the resulting model on the held-out test images. We train a SegFormer [31] architecture with a MiT-B2 encoder pretrained on ImageNet-1k. We optimize with SGD (momentum 0.9, weight decay 0.01) and layer-wise learning-rate decay (factor 0.9): the encoder’s top stage uses a learning rate of $7 \times 1 0 ^ { - 4 }$ , decayed by 0.9 per earlier stage, and the decode head uses $1 0 ^ { - 3 } .$ with a 10% linear warmup followed by cosine decay. We train with batch size 4 at $5 1 2 \times 5 1 2$ resolution (512 × 1024 for Coralscapes), for 160 epochs on our dataset and 60 on the larger Coralscapes and UCSD Mosaics, using standard augmentation (random horizontal flip, rotation up to $\pm 3 0 ^ { \circ }$ , and color jitter).

Datasets We evaluate on the three datasets described in more detail in Sec. 3: our benthic dataset (24 classes), Coralscapes [21] (39 classes), and UCSD Mosaics [1, 10] (34 classes). Our dataset provides a mean of 180 expert point-labels per image, over 313 training and 78 test images. Coralscapes and UCSD Mosaics are densely annotated, so we randomly sample 180 sparse point-labels per image from their training masks, matching the mean point density of our benthic dataset, to reproduce the weakly-supervised setting. We use the available train sets of 1683 and 3974 images, respectively, and evaluate on their corresponding test sets of 392 and 696 images.

Metrics We report standard mean Intersection-over-Union (mIoU) and mean per-class accuracy (mPA) over all pixels for Coralscapes and UCSD Mosaics, since they provide dense ground truth masks for test image. Since our dataset does only provide sparse point labels, we can only report mean per-class accuracy (mPA) evaluated on the available point annotations. Note that although mIoU is a more robust indicator for segmentation tasks, the trends observed are the same with both metrics. We exclude the background classes excluded from the mean in both cases.

## 5.2 Segmentation results with diferent propagation strategies

Our main evaluation compares SegFormer performance when trained with SAM2- propagated pseudo-masks under diferent strategies for handling unreliable propagated supervision. All strategies receive the same point-labels and identical supervision, so the comparison isolates the efect of the filtering strategy and reports the relative ranking of strategies in a weakly-supervised sparse-point regime, not absolute performance comparable to densely-supervised training. Table 1 summarizes the results of this experiment. Our strategy produces the largest gains on the two datasets with noisier labels, where it removes 34% (our benthic dataset) and 25% (Coralscapes) of the points.

The three datasets behave diferently under filtering. On our benthic dataset and Coralscapes, where labels are noisier, Ours clearly separates from keep all. UCSD Mosaics is a special case, where all methods instead cluster around 44 mIoU. Our hypothesis is that it presents a situation less prone to unsuitable point propagations, with high quality images and dense ground-truth masks with sharp boundaries, where sampled points rarely fall on ambiguous regions. The emergent drop rate supports this: our method removes only 11.7% of the pointlabels on UCSD Mosaics, against 24.6% on Coralscapes, indicating cleaner labels with fewer unsuitable propagations. Here, pruning is not relevant, but we do not penalize the baseline 44.41 mIoU as much as with other approaches to tackle noise (GCE).

The random baselines drop as many points as Ours, but at random. On the two noisier datasets they do not beat keep all, and on our benthic dataset they score well below it (22.6 and 23.5 vs. 27.5). This shows how pruning point-labels does not help by itself, what matters is which points are pruned. GCE keeps every label and down-weights the noisy ones in the loss, but it only matches mPA obtained with the baseline keep all on all three datasets, and even lowers at UCSD mIoU, showing that our pruning strategy results more efective.

Table 1: Filtering propagated labels by reliability across three datasets. All strategies propagate the same expert point-labels into dense pseudo-masks with SAM2 and train Segformer on them. Keep all trusts every propagated label; GCE also keeps all labels but trains with a noise-robust loss; the random baselines discard labels without any reliability estimate; Ours prunes unreliable points before propagation and trims unreliable regions afterwards. Our benthic dataset has no dense annotation and is evaluated only at its expert-labeled points. Mean over 5 seeds for our benthic dataset, a single seed for Coralscapes and UCSD Mosaics.
<table><tr><td rowspan="2">Strategy</td><td colspan="2">Our dataset</td><td colspan="2">Coralscapes [21]</td><td colspan="2">UCSD Mosaics [1, 10]</td><td rowspan="2"></td></tr><tr><td>mPA*</td><td>used%</td><td>mPA mIoU</td><td>used%</td><td>mPA mIoU</td><td>used%</td></tr><tr><td>None (keep all)</td><td>27.53</td><td>100</td><td>33.71</td><td>25.57</td><td>100</td><td>53.75 44.41</td><td>100</td></tr><tr><td>GCE [35]</td><td>27.86</td><td>100</td><td>33.54</td><td>25.52</td><td>100</td><td>53.87 41.66</td><td>100</td></tr><tr><td>Blind random drop †</td><td>22.60</td><td>66.1</td><td>33.76</td><td>25.39</td><td>75.4</td><td>54.77 45.12</td><td>88.3</td></tr><tr><td>Class-balanced random †</td><td>23.46</td><td>66.1</td><td>33.95</td><td>25.79</td><td>75.4</td><td>52.11 42.73</td><td>88.3</td></tr><tr><td>Ours</td><td>32.00</td><td>66.1</td><td>37.21 27.60</td><td></td><td>75.4</td><td>53.60 44.05</td><td>88.3</td></tr></table>

Only evaluated at the GT points. †Dropped points % matched to Ours.

## 5.3 Pruning and Trimming ablation

This ablation experiment, summarized in Table 2, isolates pruning and trimming stages, explained in Section 4, on our benthic dataset. Each of them is shown to contribute to improving the final result: trimming raises mPA from 27.5 to 29.8 and pruning to 31.0, suggesting that the significance of pruning bad points before propagation is higher than trimming bad regions afterwards, but both result complementary.

Table 2: Pruning and trimming ablation on our benthic dataset.
<table><tr><td>Method</td><td>mPA</td></tr><tr><td>keep all</td><td> $2 7 . 5 3 \pm 1 . 0$ </td></tr><tr><td>trim only</td><td> $2 9 . 8 2 \pm 0 . 9$ </td></tr><tr><td>prune only</td><td> $3 1 . 0 0 \pm 0 . 7$ </td></tr><tr><td>prune + trim (ours)</td><td> ${ \bf 3 2 . 0 0 \pm 1 . 1 }$ </td></tr></table>

## 5.4 Qualitative results

We complement the quantitative results with qualitative examples on our benthic dataset and Coralscapes. Figures 4 and 5 compares, for several images, the pseudo-mask obtained by propagating all points against Ours, with the pointlabels overlaid, showing how pruning and trimming remove unreliable expansions and mislabeled regions and, in some cases, recover the correct label for pixels that keep-all had mislabeled.

## 6 Conclusion

In this work, we addressed a critical bottleneck to enable scaling automated analysis of marine ecological monitoring data, by bridging the gap between sparse legacy annotations and modern dense semantic segmentation models. While visual foundation models like SAM2 ofer immense potential for automated image analysis, we demonstrated that their direct application to randomly sampled survey points frequently introduces harmful label noise due to boundary ambiguity. To overcome this, we introduced a novel point-label propagation framework that explicitly accounts for the suitability of visual prompts before and after expansion.

By integrating a pre-processing pruning stage to filter out unsuitable points for propagation, and a post-processing trimming stage to resolve class overlaps, our method significantly enhances the quality of generated pseudo-ground-truth masks. We validated these contributions on established public benchmarks and introduced a new challenging dataset featuring real-world sparse expert annotations. Our experiments confirm that automatically identifying and discarding unsuitable points mitigates the degradation of downstream segmentation models, leading to notable performance gains in fine-grained benthic semantic analysis.

These gains also come with limitations. The appearance-based signals rely on per-image class prototypes, so a class with few point-labels yields an unreliable prototype. Near-identical classes also remain hard to separate when appearance alone is not enough. Even so, this framework provides a robust and practical tool to unlock the value of vast, existing ecological image repositories without requiring additional, expensive human annotation efort. By releasing our code, models, and the new benchmark dataset, we aim to facilitate ongoing interdisciplinary collaboration between the computer vision and marine biology communities. Future work will explore extending these filtering mechanisms to other ecological domains with similar annotation constraints, further assisting on the computation of valuable metrics for ecological studies from complex environmental data.

## Acknowledgments

This work was supported by grants AIA2025-163563-C31 and PID2024-159284NB-I00, funded by MICIU/AEI /10.13039/501100011033 and ERDF\EU, and by DGA project T45\_23R. BAM and JEKB were partially supported by the Stone Living Lab and Woods Hole Sea Grant while working on data for this manuscript.

![](images/6b0f66e3ab81b0ef8607d2c01e1c13d50c05901cd49845594dd51fc4e30c7ac8.jpg)

![](images/9ec34b78392f512bb65c8d0e48edd96947f62c7f48c3400e22930810e9a14c11.jpg)

![](images/4295aa0fbd357f401ac4e8be357870e3b8fe7c4690f8065d1d102f8ac4a331d2.jpg)

![](images/19cce6be16df4fa7026f5ff25f8119e6ce8ac0ae0349fb1aa68ea49807fe9c2c.jpg)

![](images/12059d715b597414c1aee0d77f047e5e6088f3f6f9d697dcbfaf9ce7eb72c857.jpg)

![](images/0d282316f6ba6af10bd082503e09379e531c36052b92dbeb4a6d90e2b2599353.jpg)

![](images/73fcfc571a31bd83c09a19f357105da3059255061b71a17f52c1c04ae2314747.jpg)

![](images/b0b2fa6f3e38c38ecd64be5ffd62ba277153079c8b9899de63e2b0cfb14ff309.jpg)

![](images/eca1b0591b71416aa1225829f49873afd5b317af8ba2931013ff907f90d0c0ba.jpg)

![](images/7099c8ffe019a9b1fb2fe054adf82c285482937c8cda483cf42b8c28419f1ba9.jpg)  
(a) Image

![](images/53a8ffa24a488e10829bd2acd728c0157271f64e985d85dc5fa9726ca170b864.jpg)  
(b) Keep all

![](images/343878bce8c98dd4e32bdddcc5d8febd34d1312e1fd4a91a17fae6fa55395e80.jpg)  
(c) Ours  
Fig. 4: Qualitative examples on our benthic dataset. Each row shows, (a) the input image with the available the point-labels overlaid as class-colored dots (with yellow edge for visibility); (b) the masks obtained after propagating all points; (c) the masks obtained with our approach. (a) and (b) show all the point-labels while (c) only shows the points that survive our pruning. All examples show situations where expert correct point-labels get incorrectly covered by other class propagation. Red dashed boxes and arrows highlight some of the interesting cases. Pruning discards points whose expansion is unreliable and trimming removes mislabeled regions. In some cases removing an overextended mask is key to enable neighbouring mask supply the correct label, recovering pixels that keep-all had mislabeled (e.g., the example in row 1).

## References

1. Alonso, I., Yuval, M., Eyal, G., Treibitz, T., Murillo, A.C.: Coralseg: Learning coral segmentation from sparse annotations. Journal of Field Robotics 36(8), 1456–1477

<sup>ep</sup> <sup>the</sup> <sup>confidently-wrong,</sup> <sup>over-extended</sup> m<sub>ask</sub> <sub>that</sub> <sub>keep-</sub> <sup>y</sup> <sup>the</sup> <sup>regions</sup> <sup>of</sup> <sup>a</sup> <sup>surviving</sup> m<sub>ask</sub> <sub>that</sub> <sub>disagree</sub> <sub>with</sub> <sub>its</sub> <sub>class</sub>. <sub>As</sub> <sub>a</sub> <sub>result,</sub> <sub>ours</sub> <sub>often</sub> <sub>lea</sub> <sup>ask</sup> <sup>over-expands</sup> <sup>far</sup> <sup>beyond</sup> <sup>the</sup> <sup>object</sup>. <sup>Pruning</sup> <sup>discards</sup> <sup>the</sup> <sup>point</sup> <sup>when</sup> <sub>its</sub> <sub>whole</sub> <sub>expa</sub> <sup>uations</sup> <sup>our</sup> <sup>method</sup> <sup>corrects</sup>: <sup>expert-correct</sup> <sup>point-labels</sup> <sup>whose</sup> <sup>SA</sup>M<sub>2</sub> <sub>mask</sub> <sub>bleeds</sub> <sub>into</sub> <sub>a</sub> <sup>proach</sup>. <sup>(a)</sup> <sup>and</sup> <sup>(c)</sup> <sup>show</sup> <sup>all</sup> <sup>the</sup> <sup>point-labels</sup> w<sub>hile</sub> <sub>(d)</sub> <sub>only</sub> <sub>shows</sub> <sub>the</sub> <sub>points</sub> <sub>that</sub> <sub>surv</sub> <sup>))</sup><sub>)</sub> <sup>ow</sup> <sup>edge</sup> <sup>for</sup> <sup>visibility;</sup> <sup>(b</sup> <sup>the</sup> <sup>Ground</sup> <sup>truth;</sup> <sup>(c</sup> <sup>the</sup> m<sub>asks</sub> <sub>obtained</sub> <sub>after</sub> <sub>propagating</sub> <sub>al</sub> <sup>amples</sup> <sup>on</sup> <sup>Coralscapes</sup>. <sup>Each</sup> <sub>row</sub> <sub>shows,</sub> <sub>(a)</sub> <sub>the</sub> <sub>input</sub> <sub>image</sub> <sub>with</sub> <sub>the</sub> <sub>available</sub> <sub>the</sub> <sub>point-</sub>  
![](images/3a5a51738f44e54546e44061ca76b07a8c8d844844a9c21ce1bafc922e485a47.jpg)

(2019)

2. Barnett, T.P., Pierce, D.W., AchutaRao, K.M., Gleckler, P.J., Santer, B.D., Gregory, J.M., Washington, W.M.: Penetration of human-induced warming into the world’s oceans. Science 309(5732), 284–287 (2005). https://doi.org/10.1126/ science.1112418

3. Beijbom, O., Edmunds, P.J., Kline, D.I., Mitchell, B.G., Kriegman, D.: Automated annotation of coral reef survey images. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 1170–1177 (2012). https://doi.org/10. 1109/CVPR.2012.6247798

4. Borja, C., Plou, C., Martinez-Cantin, R., Murillo, A.C.: Sseg: Active sparse pointlabel augmentation for semantic segmentation. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. pp. 1424–1434 (2026)

5. Burrows, M.T., Schoeman, D.S., Buckley, L.B., Moore, P., Poloczanska, E.S., Brander, K.M., Brown, C., Bruno, J.F., Duarte, C.M., Halpern, B.S., Holding, J., Kappel, C.V., Kiessling, W., O’Connor, M.I., Pandolfi, J.M., Parmesan, C., Schwing, F.B., Sydeman, W.J., Richardson, A.J.: The pace of shifting climate in marine and terrestrial ecosystems. Science 334(6056), 652–655 (2011). https://doi.org/10.1126/science.1210288

6. Carion, N., Gustafson, L., Hu, Y.T., Debnath, S., Hu, R., Suris, D., Ryali, C., Alwala, K.V., Khedr, H., Huang, A., et al.: Sam 3: Segment anything with concepts. arXiv preprint arXiv:2511.16719 (2025)

7. Cheng, L., Abraham, J., Trenberth, K.E., Reagan, J., Zhang, H.M., Storto, A., von Schuckmann, K., et al.: Record high temperatures in the ocean in 2024. Advances in Atmospheric Sciences 42, 1092–1109 (2025). https://doi.org/10.1007/s00376- 025-4541-3

8. Darcet, T., Oquab, M., Mairal, J., Bojanowski, P.: Vision transformers need registers. In: International conference on learning representations (ICLR). vol. 2024, pp. 2632–2652 (2024)

9. Dunstan, P., Bax, N.: How far can marine species go? influence of population biology and larval movement on future range limits. Marine Ecology Progress Series 344, 15–28 (2007). https://doi.org/10.3354/meps06940

10. Edwards, C.B., Eynaud, Y., Williams, G.J., Pedersen, N.E., Zgliczynski, B.J., Gleason, A.C., Smith, J.E., Sandin, S.A.: Large-area imaging reveals biologically driven non-random spatial patterns of corals at a remote reef. Coral Reefs 36(4), 1291– 1305 (2017)

11. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., Dollár, P., Girshick, R.: Segment anything. In: IEEE/CVF International Conference on Computer Vision (ICCV) (2023)

12. Lawlor, J.A., Comte, L., Grenouillet, G., Lenoir, J., Baecher, J.A., Bandara, R.M.W.J., Bertrand, R., Chen, I.C., Diamond, S.E., Lancaster, L.T., Moore, N., Murienne, J., Oliveira, B.F., Pecl, G.T., Pinsky, M.L., Rolland, J., Rubenstein, M., Schefers, B.R., Thompson, L.M., van Amerom, B., Villalobos, F., Weiskopf, S.R., Sunday, J.: Mechanisms, detection and impacts of species redistributions under climate change. Nature Reviews Earth & Environment 5, 351–368 (2024). https://doi.org/10.1038/s43017-024-00527-z

13. Lin, Y., et al.: SAMRefiner: Taming segment anything model for universal mask refinement. In: International Conference on Learning Representations (ICLR) (2025), arXiv:2502.06756

14. Ma, X., Huang, H., Wang, Y., Romano, S., Erfani, S., Bailey, J.: Normalized loss functions for deep learning with noisy labels. In: International conference on machine learning. pp. 6543–6553. PMLR (2020)

15. Northcutt, C., Jiang, L., Chuang, I.: Confident learning: Estimating uncertainty in dataset labels. Journal of Artificial Intelligence Research 70, 1373–1411 (2021)

16. Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al.: DINOv2: learning robust visual features without supervision. Transactions on Machine Learning Research (TMLR) (2024), arXiv:2304.07193

17. Pleiss, G., Zhang, T., Elenberg, E.R., Weinberger, K.Q.: Identifying mislabeled data using the area under the margin ranking. In: Advances in Neural Information Processing Systems (NeurIPS) (2020), arXiv:2001.10528

18. Raine, S., Marchant, R., Kusy, B., Maire, F., Fischer, T.: Point label aware superpixels for multi-species segmentation of underwater imagery. IEEE Robotics and Automation Letters 7(3), 8291–8298 (2022). https://doi.org/10.1109/LRA. 2022.3187836

19. Raine, S., Marchant, R., Kusy, B., Maire, F., Sünderhauf, N., Fischer, T.: Humanin-the-loop segmentation of multi-species coral imagery. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). pp. 2723– 2732 (2024), arXiv:2404.09406

20. Ravi, N., Gabeur, V., Hu, Y.T., Hu, R., Ryali, C., Ma, T., Khedr, H., Rädle, R., Rolland, C., Gustafson, L., et al.: SAM 2: Segment anything in images and videos. In: International Conference on Learning Representations (ICLR) (2025)

21. Sauder, J., Domazetoski, V., Banc-Prandi, G., Perna, G., Meibom, A., Tuia, D.: The coralscapes dataset: Semantic scene understanding in coral reefs. In: 2025 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW). pp. 2136–2143 (2025). https://doi.org/10.1109/ICCVW69036.2025.00223

22. Sebens, K.P., Maney, E.J., McCollum, B.A.: Selected photo quadrats and sparse species annotations from 2009–2023 at subtidal rock walls around halfway rock, MA, USA ver 1. Environmental Data Initiative (2026)

23. Siméoni, O., et al.: Dinov3. arXiv preprint arXiv:2508.10104 (2025)

24. Snell, J., Swersky, K., Zemel, R.: Prototypical networks for few-shot learning. In: Advances in Neural Information Processing Systems (NeurIPS) (2017)

25. Sohn, K., Berthelot, D., Li, C.L., Zhang, Z., Carlini, N., Cubuk, E.D., Kurakin, A., Zhang, H., Rafel, C.: Fixmatch: Simplifying semi-supervised learning with consistency and confidence. In: Advances in Neural Information Processing Systems (NeurIPS) (2020)

26. Sunday, J.M., Pecl, G.T., Frusher, S., Hobday, A.J., Hill, N., Holbrook, N.J., Edgar, G.J., Stuart-Smith, R., Barrett, N., Wernberg, T., et al.: Species traits and climate velocity explain geographic range shifts in an ocean-warming hotspot. Ecology letters 18(9), 944–953 (2015)

27. Wang, K., Liew, J.H., Zou, Y., Zhou, D., Feng, J.: Panet: Few-shot image semantic segmentation with prototype alignment. In: IEEE/CVF International Conference on Computer Vision (ICCV). pp. 9197–9206 (2019)

28. Wang, Y., Chen, H., Heng, Q., Hou, W., Fan, Y., Wu, Z., Wang, J., Savvides, M., Shinozaki, T., Raj, B., Schiele, B., Xie, X.: Freematch: Self-adaptive thresholding for semi-supervised learning. In: International Conference on Learning Representations (ICLR) (2023)

29. Wang, Y., Ma, X., Chen, Z., Luo, Y., Yi, J., Bailey, J.: Symmetric cross entropy for robust learning with noisy labels. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 322–330 (2019)

30. Wang, Y., Wang, H., Shen, Y., Fei, J., Li, W., Jin, G., Wu, L., Zhao, R., Le, X.: Semi-supervised semantic segmentation using unreliable pseudo-labels. In:

IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 4248–4257 (2022)

31. Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J.M., Luo, P.: Segformer: Simple and eficient design for semantic segmentation with transformers. In: Advances in Neural Information Processing Systems (NeurIPS) (2021)

32. Yang, J., Luo, K.Z., Li, J., Deng, C., Guibas, L., Krishnan, D., Weinberger, K.Q., Tian, Y., Wang, Y.: Denoising vision transformers. In: European Conference on Computer Vision (ECCV). pp. 453–469 (2024), arXiv:2401.02957

33. Ye, X., Li, X., Liu, T., Sun, Y., Tong, W., et al.: Active negative loss functions for learning with noisy labels. Advances in Neural Information Processing Systems 36, 6917–6940 (2023)

34. Zhang, B., Wang, Y., Hou, W., Wu, H., Wang, J., Okumura, M., Shinozaki, T.: Flexmatch: Boosting semi-supervised learning with curriculum pseudo labeling. In: Advances in Neural Information Processing Systems (NeurIPS) (2021)

35. Zhang, Z., Sabuncu, M.: Generalized cross entropy loss for training deep neural networks with noisy labels. Advances in neural information processing systems 31 (2018)

36. Zheng, Z., Liang, H., Hua, B.S., Wong, Y.H., Ang, P., Chui, A.P.Y., Yeung, S.K.: Coralscop: Segment any coral image on this planet. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2024)

37. Zhou, T., Wang, W., Konukoglu, E., Van Gool, L.: Rethinking semantic segmentation: A prototype view. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 2582–2593 (2022)