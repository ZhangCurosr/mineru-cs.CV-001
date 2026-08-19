# Evaluation of AI-based Visual Crack Detection in Steel Bridges Using Probability of Detection

Andrii Kompanets<sup>a,b,∗</sup>, Finn Sherry<sup>b,c,∗</sup>, Remco Duits<sup>b,c</sup>, Davide Leonetti<sup>a,b</sup>, H.H. (Bert) Snijder<sup>a</sup>

<sup>a</sup>Eindhoven University ofTechnology, Department ofthe Built Environment, Eindhoven, The Netherlands <sup>b</sup>Eindhoven Artificial Intelligence Systems Institute, Eindhoven University ofTechnology, Eindhoven, The Netherlands <sup>c</sup>Eindhoven University ofTechnology, Department ofMathematics and Computer Science, Eindhoven, The Netherlands

## Abstract

Bridge structures are regularly inspected for structural damage such as cracks and corrosion in order to ensure pub lic safety and reduce maintenance costs. Much research has been done on automating this process using computer vision methods, which are often evaluated and compared using metrics such as intersection over union, mean aver age precision, etc. However, predicting the actual efectiveness of an inspection method within the field of structural engineering from these metrics remains challenging. To enable the systematic use of these increasingly popular methods in engineering practice, evaluating the performance of these methods in a way that is compatible with standard engineering approaches is therefore an urgent necessity. We present a new statistical evaluation framework to allow the comparison of computer vision methods with conventional visual inspection for crack detection in steel bridges. The framework is based on probability of detection curves and can account for the influence of image resolution. We apply this evaluation method to the real-world “Cracks in Steel Bridges” dataset [1], which contains annotated images of cracks in bridge structures. The quantification of the probability of detection and its uncertainty enables a practical assessment of the efect of automated methods for damage detection in structural reliability analyses. In turn, this enables the wide-spread use of automated (AI-based) damage detection in safety critical applications. This evaluation method provides evidence that the proposed computer vision approach approach is robust for the crack detection task and can have a high added value as an addition to conventional visual inspection methods.

Keywords: crack detection, steel bridges, nondestructive evaluation, probability of detection

## 1. Introduction

In Europe, bridges have aged considerably [2], and a large proportion of bridges are older than 50 years [3]. For example about 80 % of steel bridges and 20 % of concrete bridges in the Netherlands have less than a third of their design life remaining [4]. This poses an issue because, with time, the bridge structures tend to deteriorate due to various factors such as fatigue and corrosion. Prolonged exploitation of the bridge increases the chances of its structural failure or, in the worst case, complete collapse of the bridge. This explains the recent surge in research activities aimed at improving the monitoring and inspection of the structural health of bridges [5, 6]. Special attention has been paid to the improvement of deep learning-based methods for visual detection of fatigue cracks, as has been reviewed in [7], as the visual inspection is the simplest and most widely used method and fatigue cracks are the most threatening type of damage to the structural integrity of the bridge.

Conventional visual inspection is carried out by trained inspectors who visually examine the bridge structure. In some cases, special equipment, such as climbing ropes, elevator machines, magnifying lenses, and spotlights, is necessary to allow the inspector to get close to the bridge and to detect damage, such as cracks or corrosion, more accurately. Automated methods for visual bridge inspection have been proposed to improve and support conventional inspection methods, by increasing their speed, reducing their cost, and improving the consistency of these methods by minimising the chance of human error. These emerging methods usually involve a robotic system, e.g. an unmanned aerial vehicle or a crawling robot collecting images of the bridge structure. Those pictures are subsequently analysed by a Computer Vision (CV) algorithm.

Several reviews of articles studying automated visual inspections of damage in bridges have been published [8, 9]. From this we can see the general progress over time of CV methods for crack detection. Progress has been made by (1) using more advanced general-purpose CV algorithms, (2) tailoring those algorithms for specific tasks, e.g. crack detection in steel bridges, concrete, or asphalt (3) improving the quality and size of the datasets used to train the CV algorithms, and (4) improving the computing capabilities of the hardware used for the tasks. In order to deploy these methods in real-world scenarios, e.g. in supporting engineers during routine inspections of structures, however, their maturity must be properly evaluated.

In the race to develop better CV methods, they are often compared with each other using benchmark datasets. However, only a few studies have compared computer vision methods for automatic crack detection with conventional inspection methods, such as visual inspections performed by a trained inspector. Examples of such studies include [10], which applied the DeepCrack algorithm [11] to images of concrete bridges manually annotated by human inspectors. They found that while the accuracy, a standard metric used in the field of CV, was high (∼ 90 %), the real ability of the algorithm to accurately detect cracks was poor, since it incorrectly classified most surface irregularities as cracks.

In another case study reported in [12], a few concrete bridge piers were physically inspected by a human inspector and by an unmanned aerial vehicle with an automated crack detection system using CV methods. Diametrically opposite to the previous study, here the CV algorithm outperformed the human inspector and was able to detect cracks that were missed by the human. The use of a diferent CV algorithm may partially explain the divergence in the conclusions of [10] and [12].

Note that automated visual crack detection should not only be considered as a replacement for conventional inspection. In [13], the authors considered CV crack detection as a way to augment the performance of a human inspector, rather than to replace the human inspector. In this case study, an inspector was able to detect 90 % of cracks when assisted by CV and only 60 % without CV assistance.

To the best of our knowledge, up to now visual inspection and CV inspection have been compared only in case studies. Despite their relevance, case studies tend to be extremely time-consuming and they generalise poorly, as a consequence of small sample sizes and a high susceptibility to researcher bias. The fact that there are so few publications that compare CV methods to conventional visual inspection and that all existing results come from case studies represents a major gap in the field.

One of the main obstacles standing in the way of such comparisons is the fact that the metrics for evaluating conventional visual inspection are diferent from those typically used to evaluate CV algorithms. Conventional crack detection method performance is com monly quantified using a probability of detection (PoD) curve, which relates the probability of detecting a crack to its characteristic size [14, 15, 16]. PoD curves are used within a structural reliability assessment to quantify the beneficial efect of an inspection in terms of structural reliability, which is related to the probability of failure [17, 18, 19]. Without such a curve, it can be particularly dificult to judge the true performance of a CV method for inspection, compare it with manual visual inspection, determine whether it can be used in real applications, and consider it quantitatively in a structural reliability assessment.

Generally, the performance of CV methods for damage detection depends to a large degree on diferent inspection conditions, such as camera sensor and optics, distance from the camera to the damage, and other variables that afect the visibility of damage in the images, such as lighting conditions and incidence. To the best of our knowledge, there is no existing general methodology to model the dependence of the performance of CV methods for automated crack detection on these highly relevant additional variables.

## 1.1. Contributions

Given the aforementioned context, we identify the following three open questions in the field of automated visual inspection:

• How can an automated CV crack detection method be appropriately compared to conventional crack detection methods?

• Are CV methods for crack detection robust enough for deployment in real bridge inspection applications?

• How do camera specifications and the distance from the camera to the crack afect the performance of CV methods for automated crack detection?

The main contribution of this paper is to answer the aforementioned questions using the example of our CV method for crack detection in steel bridges, as follows:

• We introduce a framework for comparing the crack detection performance of CV methods to conventional inspection approaches, described in Sec. 5 and Algorithm (Alg.) 1. For this, we first adapt the use of PoD curves from conventional approaches to automated visual methods for crack detection in Sec. 4.

• We show, using our newly proposed comparison framework (Alg. 1), that our CV method can outperform existing estimates (Campbell et al. [20], DNVGL recommendation [16]) of conventional inspection performance for cracks of length less than 150 mm. For longer cracks, our CV method still outperforms the DNVGL recommendation [16].

• We analyse how the performance of our CV method is afected by the camera specifications and distance between camera and the crack. For this analysis, we use both a nonparametric (Sec. 6.1.1) and a parametric (Sec. 6.1.2) method. In both cases we observe that the PoD increases as the resolution improves. We show that an automated inspection, applying our CV method to images collected by a camera with instantaneous field of view (see Section 6) equal to $1 . 2 5 \times 1 0 ^ { - 4 }$ rad/pix from a distance of up to 3 metres, has a higher PoD than conventional visual inspection and has comparable performance for longer cracks.

## 1.2. Structure of the article

The paper is structured as follows. Section 2 describes the data that was used in this study. In Section 3, we present a CV method, based on a neural network, for the detection of cracks in images of steel bridges. Further, in Section 4 we propose a methodology to estimate the PoD curve of an automated visual crack detection approach. The PoD curve estimated for our CV method is then quantitatively compared with the conventional inspection method in Section 5. In Section 6, we propose two methods to explicitly take into account the efect of resolution on the probability of detection. Finally, Section 7 concludes the work.

## 2. Data

The analyses carried out in this paper are performed on images contained in the “Cracks in Steel Bridges” (CSB) [1] dataset, which has also been used in previous works [21, 22]. The dataset consists of 6163 images collected over the years during routine conventional bridge inspections. Some examples are shown in Fig. 1. The images have sizes ranging from a minimum of 600 × 450 pixels to a maximum of 4608 × 3456 pixels. The images capture various structures of diferent bridges with structural damage, such as fatigue cracks and/or corrosion damage. As the images were taken while the bridge structures were in use, often dust, dirt, oil or water stains, and even insects are visible. In some images, artificial marks made by inspectors are present, indicating i.a. the location of the crack tips on the surface and the crack length. Bridge structure surfaces are painted mainly in blue, brown, white, or black. In some cases, the paint is intentionally removed at the tips of the cracks by the inspector during routine inspections to reveal the true length of the crack. The images were captured at diferent angles relative to the captured structure surfaces, with diferent lighting conditions, and at diferent distances ranging from 0.5 to 5 metres. We have used these images first to train a CV method for automatic crack detection and then to assess the performance of this method using PoD estimation.

## 2.1. PoD estimation dataset

To determine the PoD curve of an inspection method, one needs to use a dataset containing cracks of known lengths. This information was not recorded for the available images. However, in some of the images, inspection markings made on the bridge surface indicating the length of the crack are visible. An example of such an image is shown in Fig. 1a. The dataset used for the PoD estimation consists of 239 manually selected images in which the crack length is identifiable from these markings. Notably, not all images with markings contain the crack length, cf. Fig. 1b. Such images were not included in the PoD estimation dataset.

For each image in the PoD estimation dataset, the digital crack length in pixels is additionally recorded. The digital crack length is determined by manually selecting points on the crack and connecting these by straight lines. For straight cracks, only the end points are used; for particularly curvy cracks, an additional point is added at the bend. The length of these lines in pixels is then computed using the Pythagorean theorem. Fig. 2a shows the digital crack length in pixels against the physical crack length in millimetres for each image; by dividing the digital length by the physical length, we obtain the resolution in pixels per millimetre, which is plotted against the physical crack length in Fig. 2b. In both figures, the marks correspond to images in which the crack was detected (hit, green) and missed (miss, red) by the crack detection CV method (see Sec. 3). In fact, the resolution is typically not uniform along the length of the crack if the camera is not facing straight onto the surface containing the crack. Therefore, we work with an approximate average resolution over a single crack.

![](images/1fe64247776e3493f3f595749f04a1fd9ca43c826bfb08e50e53bbd7ab389094.jpg)  
(a)

![](images/33530b920235de61f8124963b62c79e6e2c9c9c8a9f0cd4c271786141c74260b.jpg)  
(b)

![](images/41980f5c021bec0e2e753d94c5495558cf9236d747270846ffa225219b62881a.jpg)  
(c)  
Figure 1: Examples of images used in this study, taken from the “Cracks in Steel Bridges” dataset [1].

## 2.2. Dataset for neural network training

To train our CV method for crack detection, we follow the approach in [22]. The dataset described above was first manually annotated with bounding boxes to indicate cracks, and then split into train, validation, and test sets with 80/10/10 % of the images, respectively. All the images used for PoD estimation were assigned to the test set; the other images were randomly assigned to the train, validation, and test sets. This ensures that the PoD estimation is performed on images that were not seen by the neural network during training, thereby reducing the risk that overfitting afects the results.

## 3. CV method for crack detection

As a CV method for crack detection in images, we used the Faster-RCNN neural network [23], implemented in OpenMMLAB [24], which was also applied in [22]. We now give a brief overview of the method’s architecture.

The Faster-RCNN takes images as input and outputs a set of bounding boxes, each indicating a crack with a certain confidence level. The images are first resampled by the model using a bicubic interpolation method, so that their biggest size equals 1500 pixels. Subsequently, these images are encoded in feature maps with lower spatial dimensionality by the model’s backbone, which is based on a ResNext101 neural network [25] pretrained on the ImageNet dataset [26]. A set of regions is then generated. These regions have more extreme aspect ratios (0.33; 1; 3 compared to 0.5; 1; 2) and scales compared to the original Faster-RCNN [23], in order to account for the geometry of the cracks. Using the feature maps, the network assigns each region a score indicating how confident it is that the region contains a crack. Finally, the network removes unlikely and overlapping proposals, and refines the remaining regions and their confidence scores.

![](images/4633ccb1da54372dc524ec3e8b5c2f3405f4d251f04433f86b0158773c00fad1.jpg)

(a)  
![](images/67b5a680be42dce075c439964662f2fbbf1f2fe78a5c004dc72890762d6023fb.jpg)  
(b)  
Figure 2: Distribution of physical (mm) and digital (pix) crack lengths, and resolution over the dataset used for PoD estimation.

The model was trained using an AdamW optimiser [27] for 40 epochs with an initial learning rate of $8 \times 1 0 ^ { - 5 }$ and a cosine-annealing learning rate schedule [28] starting at epoch 20. In addition, the first 500 training iterations were used for the learning rate warm-up. A weight decay of $1 \times 1 0 ^ { - 4 }$ was applied for L2 regularisation. Data augmentation was performed during training by randomly flipping the images and the corresponding segmentations. For all other hyperparameters, we use the default values of the OpenMMLab implementation [24]. The neural network training was conducted using a single NVIDIA A100 40GB GPU.

Several examples of the output of the trained neural network are presented in Fig. 3, with indications of true positive, false negative, and false positive detections in green, red, and blue, respectively. Annotated ground truth bounding boxes are shown in yellow. Fig. 3a shows an example in which the network incorrectly identifies a dark line as a crack; this is the most common type of false positive.

We developed a set of rules to automatically classify bounding boxes as true positives or false positives by comparing the detections with ground-truth bounding boxes. The predicted boxes were considered in order of decreasing confidence and assigned to their best-overlapping ground-truth box. A prediction was scored as a true positive if it met any of the following three criteria: an intersection-over-union of at least 0.2, at least 80 % of the ground-truth box covered by the prediction, or at least 80 $\%$ of the predicted box covered by the ground-truth box (a prediction fully enclosed by the ground-truth box automatically satisfying the latter). This multiple-criterion rule accommodates the thin, elongated geometry of cracks, for which the intersection-over-union alone is overly strict. Each ground-truth box could be matched only once. Any remaining unmatched predictions were counted as false positives, and any unmatched ground-truth boxes as false negatives.

## 4. Probability of detection curve

In the field of bridge inspection, it is standard practice to evaluate an inspection method using a PoD curve, which shows the probability that this inspection method will detect some damage, as a function of the damage characteristic size. A large variety of inspection methods have been evaluated using PoD curves, including visual inspection [20], ultrasonic inspection, and liquid penetrant inspection [29]. There are established methodologies and standards for the estimation of PoD curves, most prominently MIL-HDBK-1823A [15]. The methods for statistical analysis for PoD data can also be found in [14, 30]. An example PoD curve can be seen in Fig. 6.

The first step when estimating the PoD curve of an inspection method is to collect multiple measurements in samples with a known characteristic size of the damage, often denoted by $^ { a , }$ in a realistic environment. Measurement data generally come in two forms. In the first form, often called $\ddot { \mathbf { \Omega } } \hat { a }$ vs $\boldsymbol { a } ^ { \prime \prime }$ , one measures a continuous real-valued signal ˆa. In the second form, called “hit/miss”, one only records whether the damage was detected (hit) or not detected (miss). In both cases, the ground truth size of the damage is also recorded. It is possible to transform a continuous detection signal ˆa into a hit/miss record by choosing a threshold $\hat { a } _ { \mathrm { t h } }$ , which means that the crack is considered to have been detected when $\hat { a } > \hat { a } _ { \mathrm { t h } }$

## 4.1. Methodology for PoD curve determination

The recommended way to determine the PoD curve from hit/miss data is to fit a Generalised Linear Model (GLM) [14]. Denoting the crack length by a (in millimetres) and the probability of detection by $p ,$ the GLM takes the form:

$$
g ( p ) = \beta _ { 0 } + \beta _ { 1 } h ( a )\tag{1}
$$

where $h ( a ) = a \mathrm { o r } h ( a ) = \log ( a )$ , depending on which option allows a better fit of the PoD model, $\beta _ { 0 }$ and $\beta _ { 1 }$ are the model parameters and $g$ is the link function. Using the logit link function, as is standard practice [14], we end up in the setting of logistic regression. Hence, the GLM in Eq. (1) becomes

$$
\log \left( \frac { p } { 1 - p } \right) = \beta _ { 0 } + \beta _ { 1 } h ( a ) .\tag{2}
$$

When solving for $p ,$ the probability of detection of a crack of length a is

$$
\operatorname { P o D } ( a ; \beta _ { 0 } , \beta _ { 1 } ) : = p = { \frac { \exp ( \beta _ { 0 } + \beta _ { 1 } h ( a ) ) } { 1 + \exp ( \beta _ { 0 } + \beta _ { 1 } h ( a ) ) } } .\tag{3}
$$

Parameters $\beta _ { 0 }$ and $\beta _ { 1 }$ are determined using the maximum likelihood method on a data set S of n independent samples

$$
S : = \{ ( a _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n } ,\tag{4}
$$

![](images/e3c822fea03b4b00a4af772f213d40418e1ef656c17afe6d323d420e711522da.jpg)  
(a)

![](images/1d9b0eef3a808a94174f5ab596638b5a3be4a04bc18e2064b4bf2178855c92ca.jpg)  
(b)

![](images/513d38c307534e9ea5db3399f5fb91d5d30af03d2034dcf1ee8400001592e75c.jpg)  
(c)  
Figure 3: Examples of the true positive (green), false negative (red), and false positive (blue) bounding boxes produced by the trained CV method. Annotated ground truth bounding boxes shown in yellow.

where $a _ { i }$ is the true length of the crack and $Y _ { i } \in \{ 0 , 1 \}$ indicates whether the crack is detected, $Y _ { i } = 1$ , or not detected , $Y _ { i } = 0$ . The likelihood of a pair of parameters $\beta _ { 0 } , \beta _ { 1 }$ is then given by

$$
\begin{array} { l } { \displaystyle { L ( \beta _ { 0 } , \beta _ { 1 } ; S ) } } \\ { \displaystyle { = \prod _ { i = 1 } ^ { n } \mathrm { P o D } ( a _ { i } ; \beta _ { 0 } , \beta _ { 1 } ) ^ { Y _ { i } } ( 1 - \mathrm { P o D } ( a _ { i } ; \beta _ { 0 } , \beta _ { 1 } ) ) ^ { 1 - Y _ { i } } . } } \end{array}\tag{5}
$$

Instead of optimising the likelihood directly, it is more convenient to maximise the log-likelihood, which is given by

$$
\begin{array} { l } { \displaystyle \ell ( \beta _ { 0 } , \beta _ { 1 } ; S ) : = \log ( L ( \beta _ { 0 } , \beta _ { 1 } ; S ) ) } \\ { \displaystyle = \sum _ { i = 1 } ^ { N } \left[ Y _ { i } \log ( \mathrm { P o D } ( a _ { i } ; \beta _ { 0 } , \beta _ { 1 } ) ) \right. } \\ { \displaystyle \qquad + \left. ( 1 - Y _ { i } ) \log ( 1 - \mathrm { P o D } ( a _ { i } ; \beta _ { 0 } , \beta _ { 1 } ) ) \right] } \\ { \displaystyle = \sum _ { i = 1 } ^ { N } \left[ Y _ { i } ( \beta _ { 0 } + \beta _ { 1 } h ( a _ { i } ) ) \right. } \\ { \displaystyle \qquad \left. - \log ( 1 + \exp ( \beta _ { 0 } + \beta _ { 1 } h ( a _ { i } ) ) ) \right] . } \end{array}\tag{6}
$$

We optimise Eq. (6) using the LogisticRegression method implemented in the Python library scikit-learn [31]. The PoD curve is then found by filling in the fitted parameters $\beta _ { 0 } , \beta _ { 1 }$ into Eq. (3); see Fig. 6 for some examples of PoD curves.

## 4.1.1. Confidence bounds

We use the statistics of maximum likelihood estimators to compute confidence bounds on the PoD curve. Under the assumption that the data S are generated according to Eq. (3) with some parameters $\beta _ { 0 } , \beta _ { 1 }$ , with corresponding maximum likelihood estimates $\hat { \beta } _ { 0 } , \hat { \beta } _ { 1 }$

Wilk’s likelihood ratio statistic, given by

$$
W ( \beta _ { 0 } , \beta _ { 1 } ; S ) : = - 2 ( \ell ( \beta _ { 0 } , \beta _ { 1 } ; S ) - \ell ( \hat { \beta } _ { 0 } , \hat { \beta } _ { 1 } ; S ) ) ,\tag{7}
$$

asymptotically follows a χ-squared distribution with two degrees of freedom, since the model has two free parameters. We then define the $9 5 \%$ confidence parameter region as $B ( S ) : = \{ ( \beta _ { 0 } , \beta _ { 1 } ) \in \mathbb { R } ^ { 2 } \ | \ W ( \beta _ { 0 } , \beta _ { 1 } ; S ) \le$ $\chi _ { 2 , 0 . 9 5 } ^ { 2 } \approx 5 . 9 9 \}$ . In words, this means that we include parameters $( \beta _ { 0 } , \beta _ { 1 } )$ in our confidence region B(S) if the probability of observing such a large value for Wilk’s likelihood ratio is at least 5 %, under the hypothesis that the data are distributed according to Eq. (3) with those parameters. Each pair of parameters $( \beta _ { 0 } , \beta _ { 1 } ) \in B ( S )$ defines a PoD curve; the confidence bounds are found by taking the envelope of all such PoD curves, i.e.

$$
\begin{array} { l l } { \displaystyle \mathrm { P o D } ^ { \mathrm { u p p e r } } ( a ; S ) = \operatorname* { m a x } _ { ( \beta _ { 0 } , \beta _ { 1 } ) \in B ( S ) } \frac { \exp ( \beta _ { 0 } + \beta _ { 1 } h ( a ) ) } { 1 + \exp ( \beta _ { 0 } + \beta _ { 1 } h ( a ) ) } , } \\ { \displaystyle \mathrm { P o D } ^ { \mathrm { l o w e r } } ( a ; S ) = \operatorname* { m i n } _ { ( \beta _ { 0 } , \beta _ { 1 } ) \in B ( S ) } \frac { \exp ( \beta _ { 0 } + \beta _ { 1 } h ( a ) ) } { 1 + \exp ( \beta _ { 0 } + \beta _ { 1 } h ( a ) ) } . } \end{array}\tag{and}
$$

(8)

## 4.2. Results ofPoD curve determination

The CV method previously described in Sec. 3 produces a detection confidence score between 0 and 1 for each image. Fig. 4 shows this score for every image in the dataset, as a function of the crack length. As in Fig. 3, the green, red, and blue marks correspond to true positives (hits), false negatives (misses), and false positives, respectively. Note that if a linear model were fitted to these data, the errors would not follow a normal distribution. This violates one of the basic assumptions behind the methods for computing PoD curves in a vs ˆa [15].

As a consequence, we choose a threshold to obtain hit/miss data, in such a way as to balance the probability of detection with false positive errors. During bridge inspection, it is particularly important not to miss cracks, as undetected cracks may compromise the operational safety of the bridge, whereas false positives produced by a CV algorithm can be easily removed by human verification. As can be seen in Fig. 5, at a very low value of $\hat { a } _ { \mathrm { t h } } ~ = ~ 0 . 0 1$ , false positive detections occur in 20 % of images, while false negative errors are minimised to 10%. Thus, for the experiments in this study, we fix $\hat { a } _ { \mathrm { t h } } = 0 . 0 1$ . Notably, a practicioner could choose a diferent threshold to change the tradeof between false positive and false negative rate to better suit the requirements of the application.

![](images/b1d99d2e9baf70218d61863543fa0ba1d0a20012952f02e267db6c8af7d86169.jpg)  
Figure 4: Detection confidence of our CV method on the PoD estimation dataset.

We fitted the PoD model on the PoD estimation dataset, following the method described in Sec. 4.1, using the transformation $h ( a ) = \log ( a )$ since this is commonly used and, as we show in Sec. 6.1.2, it provides the best fit of the model to the data (cf. Fig. 12). The maximum likelihood estimates of the parameters are $\beta _ { 0 } ~ = ~ - 1 . 8 9 0$ and $\beta _ { 1 } ~ = ~ 0 . 8 5 8$ , so that the fitted PoD curve is

$$
\mathrm { P o D } _ { \mathrm { C V } } ( a ) = \frac { \exp ( - 1 . 8 9 0 + 0 . 8 5 8 \log ( a ) ) } { 1 + \exp ( - 1 . 8 9 0 + 0 . 8 5 8 \log ( a ) ) } .\tag{9}
$$

As baselines, we use the PoD curve provided in the DNVGL recommendation [16], and the PoD curve estimated by Campbell et al. [20]. It should be noted that both curves correspond to conventional visual inspection. The PoD curve in the DNVGL recommendation is estimated based on both data and expert opinion [16], and is the accepted standard in the field. The PoD curve from the DNVGL recommendation is given by

$$
\mathrm { P o D } _ { \mathrm { D N V G L } } ( a ) = 1 - { \frac { 1 } { 1 + ( a / 3 7 . 1 5 ) ^ { 0 . 9 5 4 } } } .\tag{10}
$$

Campbell et al. [20] derived their PoD curve using a methodology similar to the one used in this study. The PoD curve from Campbell et al. uses the GLM given in Eq. (3), with $h ( a ) = a$ and parameters $\beta _ { 0 } = - 0 . 4 9 8$ and $\beta _ { 1 } = 0 . 0 1 9 4$ , so that

$$
\mathrm { P o D } _ { \mathrm { C a m p b e l l } } ( a ) = \frac { \exp ( - 0 . 4 9 8 + 0 . 0 1 9 4 a ) } { 1 + \exp ( - 0 . 4 9 8 + 0 . 0 1 9 4 a ) } .\tag{11}
$$

It also should be mentioned that in the study by Campbell et al, the PoD was evaluated from the inspection of cracks with length up to $5 \frac { 3 } { 8 }$ inches, which is less than 150 mm. The inspection was typically performed at the distance of an arm’s length [20].

The three PoD curves are shown in Fig. 6. We see that the PoD of our CV method is superior to conventional visual inspection for cracks up to about 150 mm in length. For longer cracks, the detection capability of our CV method is slightly worse. This may be explained by the fact that it could be dificult to observe smaller cracks for a human inspector and the results may be affected by the experience of the inspector or inspection conditions. However, if automated inspection were to be conducted, it would be easier to get a good image of a small crack with a good camera and the results of the CV method would be expected to be more consistent. At the same time, a human inspector rarely misses a relatively large crack (>300 mm), while a CV method may wrongly classify it as background noise given that the image would inevitably include a large portion of the structure. Alternatively, it is possible that the PoD curve determined by Campbell et al. is not valid for such large cracks, since these were not present in the data set used for fitting the curve.

![](images/d52ac9bfb940b3a7c9028ffc6eb51715222f1f8a94f296445de3b2c1d2740471.jpg)  
Figure 5: Efect of the decision threshold $\hat { a } _ { \mathrm { t h } }$ on the tradeof between false positive and false negative detection rates of our CV method.

![](images/5f7a4b690ce81349e5d486925aa963e1c8aa93c52c9b20faa1812cbac0115e68.jpg)  
Figure 6: PoD model (Eq. (3), black) with 95 % confidence intervals (gray) fitted to the hit/miss data generated by our CV method (cf. Fig. 4), compared to the PoD curves for conventional inspection methods by DNVGL (orange) and Campbell et al. (red). For better visualisation, the hit/miss data points are scattered around horizontal lines at 0 and 1.1.

## 5. Framework for comparison of PoD curves

We propose a framework to compare inspection methods based on their PoD curves using Monte Carlo simulation. The approach is modelled on a structural reliability assessment procedure used to estimate the crack size distribution of a structure at a given moment during its useful life [17, 18]. In this procedure, the crack size distribution is updated using the PoD curve when inspections are performed. In a structural reliability assessment, the crack size distribution is obtained from a probabilistic fracture mechanics model; we instead calculate the influence of inspections for specified prior crack length distributions.

## 5.1. Methodology for comparing PoD curves with Monte Carlo simulation

We now describe our framework for comparing PoD curves, summarised in Alg. 1, in detail. We first assume some prior log-normal distribution of the crack length. This represents the distribution of crack lengths in the structure before inspection. Next, we simulate an inspection event, which is based on the Bayes theorem. Hence, we numerically estimate the posterior crack distribution after inspection by sampling a crack length from the prior log-normal distribution. This crack size is used to determine the associated probability of detection using the PoD curve. We uniformly sample a random number between 0 and 1, and mark the crack as detected if this number is less than the probability of detection. If the sampled crack is missed by an inspection method, it is stored in the array of missed cracks, M. For each inspection method, we continue to sample cracks from the prior distribution until a statistically suficient number, namely 10<sup>6</sup>, of missed cracks is stored for each inspection method. This number is estimated empirically, so that when repeating the same experiment with the same inputs, but with diferent random instantiations, the deviations in performance are at most $1 0 ^ { - 4 }$ . For our automated CV inspection method, we use the maximum likelihood PoD curve estimated in this study, given by Eq. (9), and for conventional visual inspection, we use the PoD curves provided by the DNVGL recommendation (Eq. (10)) [16] and by Campbell et al. (Eq. (11)) [20].

```tcl
Algorithm 1 Monte Carlo estimation of posterior crack
length distributions after inspection
Require: Prior crack distribution $P _ { \mathrm { p r i o r } } ( \mu _ { c } , \mathrm { C o V } _ { c } )$
Require: Probability of detection curve for the inspec
tion method PoD<sub>CV</sub>, $\mathrm { P o D _ { C a m p b e l l } , }$ , PoD<sub>DNVGL</sub>
Require: Target number of missed cracks $N _ { \mathrm { t a r g e t } }$
1: Initialize empty lists: $\begin{array} { c c c c c } { { \bar { Z } } } & { {  } } & { { \emptyset , } } & { { { \mathcal M } _ { \mathrm { C V } } } } & { {  } } & { { \emptyset , } } \end{array}$
$\mathcal { M } _ { \mathrm { C a m p b e l l } }  \emptyset , \mathcal { M } _ { \mathrm { D N V G L } }  \emptyset$
2: while min $\{ | \mathcal { M } _ { \mathrm { C V } } | , | \mathcal { M } _ { \mathrm { C } }$ <sub>ampbell</sub>|<sub>,</sub> |M<sub>DNVGL</sub> $. | \} < \ N _ { \mathrm { t a r g e t } }$
do
3: Sample crack length $l \sim P _ { \mathrm { p r i o r } } ( \mu _ { c } , \mathbf { C o V } _ { c } )$
4: Append l to I.
5: Sample $u \sim \mathcal { U } ( 0 , 1 )$
6: for each method $j \in$ {CV Campbell DNVGL}
do
7: if $u > \mathrm { P o D } _ { j } ( l )$ then
8: Append l to $M _ { j }$
9: end if
10: end for
11: end while
12: for each PoD model $j \in \{ \mathrm { C V } ,$ Campbell DNVGL}
do
13: Fit a lognormal distribution to the missed-crack
sample set $M _ { j }$
14: Compute the normalised undetected crack
length $C _ { j }$ according to Eq. (12)
15: Compute the Kullback-Leibler divergence $\mathrm { K L } _ { j }$
according to Eq. (13)
16: end for
17: return Pre- and post-inspection crack distribution
histograms; C and KL metrics
```

We consider the prior distributions of crack length to be a log-normal distribution. Within the sensitivity study, we consider 12 distributions, varying the mean µ (37.15 mm; 117.51 mm; 371.72 mm) and the coefficient of variation CoV (0.25; 0.5; 1; 2). It should be noted that a random variable X is log-normally distributed with mean $\mu$ and coeficient of variation CoV if $X = \mathrm { e x p } ( \mu _ { N } + \sigma _ { N } Z )$ , where Z has a standard normal distribution, and $\mu _ { N } ~ = ~ \log ( \mu / \sqrt { 1 + \mathrm { C o V } ^ { 2 } } )$ and $\sigma _ { N } ^ { 2 } = \log ( 1 + \mathbf { C o } \mathbf { V } ^ { 2 } )$ The mean µ is selected based on the PoD provided by the DNVGL recommendation. Specifically, crack lengths of 37.15 mm, 117.51 mm, and 371.72 mm correspond to PoD 50 %, 75 %, 90 % in the PoD curve from the DNVGL recommendation, respectively. The coeficient of variation CoV is chosen to represent diferent levels of uncertainty in the prior distribution of the crack length, ranging from relatively concentrated crack populations $( \mathrm { C o V } = 0 . 2 5 )$ to highly dispersed populations $( \mathbf { C o V } \ : = \ : 2 )$ This allows for a sensitivity analysis of the inspection-method comparison framework with respect to the assumed prior crack size and variability.

![](images/354601bc8294d5a975168c46a33e15d1288f95e89c65da9b8205b737f6838ec2.jpg)  
(a) µ = 37 15 mm.

![](images/d5b2ad4040e3e668cce3fd33ace5157ed38941ce4ef244930c1c312c2e38cbac.jpg)  
(b) µ = 117 51 mm.

![](images/178853ea3880a46be5db90f50fca92b18eb872172a1d31551d3fa2dfd4cc9fc5.jpg)  
(c) µ = 371 72 mm.  
Figure 7: Results of computation of the normalised undetected crack length C for our CV method and conventional visual inspection provided by Campbell et al. [20] and by the DNVGL recommendation [16]

Finally, we quantify the efect of the inspection with two metrics: (a) the normalised undetected crack length C, and (b) the Kullback-Leibler divergence evaluated between prior and posterior crack distributions KL. The normalised undetected crack length is computed as:

$$
C = \frac { \sum _ { i \in { \mathcal { M } } } a _ { i } } { \sum _ { i \in { \mathcal { I } } } a _ { i } } ,\tag{12}
$$

where M is the set of missed cracks and I is the set of sampled cracks from the prior distribution. A value of $C = 1$ indicates that no crack has been detected. As C approaches 0, the total length of missed cracks decreases; better inspection methods will typically have a smaller C. However, C does not describe how the missed crack lengths are distributed. For example, the same value of C may result from missing many small cracks or from missing a smaller number of longer cracks, although these cases may have diferent impli cations for structural reliability.

The Kullback-Leibler (KL) divergence is computed as:

$$
\begin{array} { r l r } {  { \mathrm { K L } ( P _ { \mathrm { p o s t } } \| P _ { \mathrm { p r i o r } } ) } } \\ & { } & { = \int _ { 0 } ^ { \infty } P _ { \mathrm { p o s t } } ( a ) \log ( P _ { \mathrm { p o s t } } ( a ) / P _ { \mathrm { p r i o r } } ( a ) ) \mathrm { d } a . } \end{array}\tag{13}
$$

The KL divergence measures a generalised notion of distance between the distribution of crack lengths after inspection (posterior $P _ { \mathrm { p o s t } } )$ relative to the initial distribution of crack lengths (prior $P _ { \mathrm { p r i o r } } )$ The posterior $P _ { \mathrm { p o s t } }$ is determined by fitting a log-normal distribution on the sample of missed cracks. The KL divergence is a standard way of measuring the diference between prior and posterior distributions [32]. A higher value of KL would generally indicate a stronger efect of the inspection method and a shift of the crack distribution to the lower crack length. However, the KL divergence is computed between normalised probability distributions and therefore does not account for the total amount of missed crack damage. As a result, an inspection method may produce a large shift in the crack-size distribution while still leaving a substantial total crack length undetected.

For this reason, both metrics are required. The normalised undetected crack length C evaluates the residual amount of undetected damage, while the Kullback-Leibler divergence KL evaluates the change in the statistical distribution of the remaining cracks. Used together, they provide a more complete assessment of inspection performance than either metric alone.

## 5.2. Results of comparing PoD curves with Monte Carlo simulation

The quantitative results of our proposed Monte Carlo evaluation are presented visually in Fig. 7 and 8, and numerically in Table 1. It can be observed that for smaller cracks with $\mu \ : = \ : 3 7 . 1 5$ mm and $\mu = 1 1 7 . 5 1$ mm, the values of the normalised undetected crack length C are lower for the CV method than for either of the conventional inspection methods, regardless of the CoV of the prior distribution. This is in line with what is shown in Fig. 6, where the PoD for the CV method is higher than for Campbell et al. and DNVGL when detecting cracks up to approximately 150 mm in length. For the prior crack distribution with $\mu = 3 7 1 . 7 2$ , the normalised crack length C of the CV method lies between those of Campbell et al. and DNVGL. For short cracks, the CV method also produces a higher KL divergence, corresponding to a larger change in the shape of the crack length distribution. For long cracks, the visual inspection of Campbell et al. instead This again repeats the pattern from Fig. 6, where for cracks above 150 mm in length, the PoD curve of our CV method is between the PoD curves for manual visual inspection as proposed in [20, 16].

![](images/2f23b33c7b62b25a6708fd8aad241262eb7f7c3630358a8beb0b66704058dd7a.jpg)  
(a) µ = 37 15 mm.

![](images/c17a0bf036eeecfdaf64a7eda9ca530ed369bb64c906818886117ef528321fb7.jpg)  
(b) µ = 117 51 mm.

![](images/ab39913447b6dcdb6b49721bdc1191c70c72d21f6eb51f84850bbcc9d4f50724.jpg)  
(c) µ = 371 72 mm.  
Figure 8: Results of computation of the Kullback-Leibler divergence KL for our CV method and conventional visual inspection provided by Campbell et al. [20] and by the DNVGL recommendation [16]

Fig. 9 further visualises these results using histograms of the posterior crack distribution for a prior with $\mu = 1 1 7 . 5 1 ; \mathrm { C o V } = 1 \mathrm { a n d } \mu = 3 7 1 . 7 2 ; \mathrm { C o V } = 1$ The use of absolute histogram counts helps explain the need for both metrics. The reduction in counts reflects the amount of cracks that remains undetected, which is measured by C, whereas the shift in the distribution reflects the change in the posterior crack-size distribution, which is measured by KL. For instance, the CV method produces lower counts than the DNVGL recommendation, indicating a smaller amount of missed damage. At the same time, its posterior distribution remains closer to the prior, particularly in terms of mean crack length, which can lead to a lower KL. Therefore, the figure shows that C and KL capture complementary aspects of the inspection outcome.

Overall, we see that the proposed CV method fairly uniformly outperforms the conventional visual inspection according to the DNVGL baseline, while the inspection method by Campbell et al. tends to perform better than our CV method on long cracks. This is in line with the simulation results for the other prior distributions, which can be found in Appendix A.

## 6. Efect of resolution on PoD

To ensure the best inspection result, it is important not only to know the average performance of an inspection method, but also to understand the efect of diferent inspection variables on that performance. For automated CV crack detection, we expect that the resolution of the crack as visible in an image is the parameter that affects the result of inspections the most. The resolution of the cracks is designated as r, measured in pixels per millimetre (pix/mm).

The resolution of the damage depends on two variables: (a) the instantaneous field of view (IFOV) of the camera θ [rad/pixel], which defines the field of view of a single pixel on the camera, and (b) the distance between the camera and the damage d [mm]. As shown in Fig. 10, it is possible to derive the relation between the resolution of the damage r and the IFOV θ and the distance d:

$$
r = { \frac { 1 } { \theta \cdot d } } .\tag{14}
$$

To better illustrate the dependency of r on θ and $d ,$ we show the resolution as a heatmap in Fig. 11, where the three vertical lines indicate the IFOV of three cameras available on typical drones typically used for bridge inspection: a DJI Mavic 3 drone with inbuilt camera; a DJI Matrice drone with a DJI Zenmuse P1 camera; a Skydio drone X10 with 64 MP narrow camera. For comparison, a rough approximation of the IFOV of a human eye is also indicated in Fig. 11 based on [33].

Table 1: Results of computation of normalised undetected crack length C and Kullback-Leibler divergence KL for our CV method and conventional visual inspection with PoD curves provided by Campbell et al. [20] and by the DNVGL recommendation [16]
<table><tr><td>µ(mm)</td><td> $\operatorname { C o V }$ </td><td> $C _ { \mathrm { { C V } } }$ </td><td> $C _ { \mathrm { C a m p b e l l } }$ </td><td> $C _ { \mathrm { D N V G L } }$ </td><td> $\mathrm { K L } _ { \mathrm { C V } }$ </td><td> ${ \mathrm { K L } } _ { \mathrm { C a m p b e l l } }$ </td><td>KLDNVGL</td></tr><tr><td rowspan="4">37.15</td><td>0.25</td><td>0.227</td><td>0.434</td><td>0.493</td><td>0.013</td><td>0.005</td><td>0.007</td></tr><tr><td>0.5</td><td>0.221</td><td>0.406</td><td>0.474</td><td>0.045</td><td>0.020</td><td>0.022</td></tr><tr><td>1</td><td>0.203</td><td>0.340</td><td>0.428</td><td>0.115</td><td>0.055</td><td>0.055</td></tr><tr><td>2</td><td>0.173</td><td>0.256</td><td>0.355</td><td>0.189</td><td>0.080</td><td>0.083</td></tr><tr><td rowspan="4">117.51</td><td>0.25</td><td>0.099</td><td>0.141</td><td>0.247</td><td>0.018</td><td>0.104</td><td>0.015</td></tr><tr><td>0.5</td><td>0.097</td><td>0.135</td><td>0.240</td><td>0.064</td><td>0.286</td><td>0.052</td></tr><tr><td>1</td><td>0.091</td><td>0.123</td><td>0.220</td><td>0.175</td><td>0.447</td><td>0.128</td></tr><tr><td>2</td><td>0.081</td><td>0.103</td><td>0.186</td><td>0.315</td><td>0.413</td><td>0.198</td></tr><tr><td rowspan="4">371.72</td><td>0.25</td><td>0.039</td><td>0.003</td><td>0.099</td><td>0.020</td><td>1.103</td><td>0.022</td></tr><tr><td>0.5</td><td>0.039</td><td>0.007</td><td>0.098</td><td>0.074</td><td>2.134</td><td>0.078</td></tr><tr><td>1</td><td>0.037</td><td>0.017</td><td>0.092</td><td>0.218</td><td>2.278</td><td>0.211</td></tr><tr><td>2</td><td>0.034</td><td>0.024</td><td>0.082</td><td>0.435</td><td>1.591</td><td>0.364</td></tr></table>

![](images/af031bcfb81f03f1703867dd3df1d6bd706d2aa38fdc0f6149b68d45fcab1778.jpg)

![](images/9da66951cdfce0490813109839c06577f9a674e2f103a9f766a96358a9fc2cee.jpg)  
(c)

![](images/6e7f8ee1bbaf32a25b9edc0c481757b8b9416db87ed803e19f72d700d14a2093.jpg)

![](images/13f8795df6e47c7ec2a04f16fd972cc226d62005c78b09b14c0a36f72604de14.jpg)  
(d)  
Figure 9: Left: (unnormalised) histograms of prior and posterior crack distributions, obtained using Alg. 1. Right: cumulative distributions of crack lengths.

![](images/48ec8251bf424ca76aec752fce44451ff049d5b7a8e1d87f837355ded2420ac0.jpg)  
Figure 10: The resolution depends inversely on the distance of the camera to the surface d and the IFOV θ.

![](images/47fd8bba4bee2cc8e03d2296c95b6e6777411c36512a5952747615f165ad6e7b.jpg)  
Figure 11: The resolution as a function of the IFOV of the camera and the distance between the camera and the crack. The IFOV of cameras commonly used for inspection of bridges with drones is provided for reference.

## 6.1. Methodology for including resolution in PoD

We propose two methods to evaluate the efect of the resolution on the PoD: the first is based on PoD curves fitting hit/miss data which are binned depending on the resolution of the images, and the second makes use of a parametric PoD surface model.

## 6.1.1. Binning approach

To evaluate the efect of resolution on the PoD using a binning approach, we divide the images from the PoD estimation dataset into four resolution bins, with ranges of 0-1 pix/mm; 1-2 pix/mm; 2-4 pix/mm; 4-30 pix/mm. The boundaries between the bins are illustrated in Fig. 11. Next, we fit the PoD curve model from Eq. (3) for each resolution bin, using inspection outcomes on images only from that specific resolution bin. We assume that the resulting PoD curve for each bin is applicable for a crack resolution in the middle of each bin. According to [14], proper PoD estimation requires at least 60 samples; using only the PoD estimation dataset results in bins with too few samples. To ensure that at least 60 samples are available in each bin, we artificially increase the number of images by downsampling each of the 239 images from the PoD estimation dataset twice: by a factor of 2 and by a factor of 4. The downsampled images have the same physical length, but a smaller digital length, so that the resolution of each crack on the downsampled images is reduced. This is equivalent to taking a picture with a camera with less resolution.

## 6.1.2. Parametric approach

We also introduce a parametric approach to explictly include the influence of resolution in the formulation of the PoD. This parametric approach relates to the binning approach from Sec. 6.1.1 in the same way that the crack length PoD model in Eq. (3) relates to the classical binomial model [14]. Even in the case of a single covariate (only crack length), the binomial method has numerous deficiencies. First and foremost, it requires an arbitrary choice of bins, and this selection can have a large influence, e.g. on the confidence bounds. Additionally, the binomial model is ineficient with data since the data points only influence the PoD curve within their own bins. For these reasons, it has been generally recommended to forgo the binomial model in favour of parametric models [14]. To analyse the influence of resolution on PoD, we therefore extend our logistic model described in Sec. 4, by adding the resolution r in pix/mm as an additional independent variable:

$$
\log \left( \frac { p } { 1 - p } \right) = \beta _ { 0 } + \beta _ { 1 } h _ { a } ( a ) + \beta _ { 2 } h _ { r } ( r ) ,\tag{15}
$$

with $h _ { r } ( r ) = \log ( r )$ or $h _ { r } ( r ) = r ,$ so that

$$
\mathrm { P o D } ( a , r ; \beta _ { 0 } , \beta _ { 1 } , \beta _ { 2 } ) = \frac { \exp ( \beta _ { 0 } + \beta _ { 1 } h _ { a } ( a ) + \beta _ { 2 } h _ { r } ( r ) ) } { 1 + \exp ( \beta _ { 0 } + \beta _ { 1 } h _ { a } ( a ) + \beta _ { 2 } h _ { r } ( r ) ) } .\tag{16}
$$

Hence, instead of a PoD curve, the model in Eq. (16) gives a PoD surface, parametrised by the length of the crack a and the resolution r. Therefore, the dataset is now extended to include the resolution of the image:

$$
S : = \{ ( a _ { i } , r _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n } .\tag{17}
$$

Note that the PoD curve model in Eq. (3) is included in the PoD surface model in Eq. (16) by setting $\beta _ { 2 } ~ = ~ 0$ The PoD surface model therefore generalises the PoD curve model, enabling quantitative comparison of both models. The model parameters $\beta _ { 0 } , \beta _ { 1 }$ , and $\beta _ { 2 }$ are estimated using the maximum likelihood method. The loglikelihood is a simple modification of Eq. (6):

$$
\begin{array} { l } { \displaystyle \ell ( \beta _ { 0 } , \beta _ { 1 } , \beta _ { 2 } ; S ) } \\ { \displaystyle \quad = \sum _ { i = 1 } ^ { n } \big [ Y _ { i } ( \beta _ { 0 } + \beta _ { 1 } h _ { a } ( a _ { i } ) + \beta _ { 2 } h _ { r } ( r _ { i } ) ) } \\ { \displaystyle \qquad - \log ( 1 + \exp ( \beta _ { 0 } + \beta _ { 1 } h _ { a } ( a _ { i } ) + \beta _ { 2 } h _ { r } ( r _ { i } ) ) ) \big ] . } \end{array}\tag{18}
$$

Transformations $h _ { a }$ and $h _ { r }$ are selected using crossvalidation. First, 100 random shufles of the data are produced. Subsequently, each shufle is divided into 10 folds. For each of the folds, the logistic model in Eq. (16) is fitted to the remaining 9 of the 10 folds, and the log-likelihood is calculated on the left-out fold. This is repeated for each combination of $h _ { a } \ = \ 1 0 \mathrm { g }$ <sub>,</sub> id and $h _ { r } = \log .$ id, producing the empirical distributions shown in Fig. 12. Since $h _ { a } \ = \ \log \ = \ h _ { r }$ clearly produces the highest log-likelihood, we will continue with this choice of transformations.

Next, the model is fitted on all the data and confidence regions are computed using the (asymptotic) statistics of the maximum likelihood estimators. Specifically, under the assumption that the data is generated by our model in Eq. (16) with some parameters $\beta _ { 0 } , \beta _ { 1 } , \beta _ { 2 }$ , Wilk’s likelihood ratio statistic, given by

$$
\begin{array} { r l } {  { W ( \beta _ { 0 } , \beta _ { 1 } , \beta _ { 2 } ; S ) } \quad } & { } \\ & { : = - 2 ( \ell ( \beta _ { 0 } , \beta _ { 1 } , \beta _ { 2 } ; S ) - \ell ( \hat { \beta } _ { 0 } , \hat { \beta } _ { 1 } , \hat { \beta } _ { 2 } ; S ) ) , } \end{array}\tag{19}
$$

with $\hat { \beta } _ { 0 } , \hat { \beta } _ { 1 } , \hat { \beta } _ { 2 }$ the maximum likelihood estimates, asymptotically follows a χ-squared distribution with three degrees of freedom, since our model has three free parameters. Then we define our 95 % confidence region as $B ( S ) : = \{ ( \beta _ { 0 } , \beta _ { 1 } , \beta _ { 2 } ) \in \mathbb { R } ^ { 3 } \mid W ( \beta _ { 0 } , \beta _ { 1 } , \beta _ { 2 } ) \leq \chi _ { 3 , 0 . 9 5 } ^ { 2 } \approx$ 7 815}. In words this means that $( \beta _ { 0 } , \beta _ { 1 } , \beta _ { 2 } )$ is included in $B ( S )$ if the probability of observing such a large value of Wilk’s statistic is at least 5 %, under the hypothesis that the data is distributed according to Eq. (16) with those parameters.

![](images/0f615e8ed15f87ced89543716c74208a58e31b02bbcfe56058fbbb319055dcab.jpg)  
Figure 12: Distribution of log-likelihoods on validation sets determined using cross-validation for diferent choices of transformations $h _ { a }$ and $h _ { r }$

Since the model now produces a PoD surface instead of a PoD curve, we take constant resolution crosssections to visualise this surface in Fig. 14. Likewise, the confidence bounds are found by taking constant resolution cross-sections of the envelope of PoD surfaces generated by B(S):

$$
\begin{array} { r l } & { \mathrm { P o D } ^ { \mathrm { u p p e r } } ( a , r ; S ) } \\ & { \quad = \underset { ( \beta _ { 0 } , \beta _ { 1 } , \beta _ { 2 } ) \in B ( S ) } { \operatorname* { m a x } } \frac { \exp ( \beta _ { 0 } + \beta _ { 1 } h _ { a } ( a ) + \beta _ { 2 } h _ { r } ( r ) ) } { 1 + \exp ( \beta _ { 0 } + \beta _ { 1 } h _ { a } ( a ) + \beta _ { 2 } h _ { r } ( r ) ) } , \mathrm { ~ a n d } } \\ & { \mathrm { P o D } ^ { \mathrm { l o w e r } } ( a , r ; S ) } \\ & { \quad \quad = \underset { ( \beta _ { 0 } , \beta _ { 1 } , \beta _ { 2 } ) \in B ( S ) } { \operatorname* { m i n } } \frac { \exp ( \beta _ { 0 } + \beta _ { 1 } h _ { a } ( a ) + \beta _ { 2 } h _ { r } ( r ) ) } { 1 + \exp ( \beta _ { 0 } + \beta _ { 1 } h _ { a } ( a ) + \beta _ { 2 } h _ { r } ( r ) ) } . } \end{array}\tag{20}
$$

Finally, we want to assess whether extending the PoD model to include the influence of resolution in addition to crack length is a sensible choice. We define the following hypotheses:

$H _ { 0 } { \mathrm { : } }$ The probability of detection for a crack of length a at resolution r is given by the PoD model in Eq. (16) with parameters $\beta _ { 0 } , \beta _ { 1 } , 0 ,$ , for some $\beta _ { 0 } , \beta _ { 1 } \in \mathbb { R }$

$H _ { a } \colon$ The probability of detection for a crack of length a at resolution r is given by the PoD model in Eq. (16) with parameters $\beta _ { 0 } , \beta _ { 1 } , \beta _ { 2 }$ , for some $\beta _ { 0 } , \beta _ { 1 } , \beta _ { 2 } \in \mathbb { R }$

We define the log-likelihood ratio statistic

$$
\Lambda ( S ) : = - 2 ( \ell ( \hat { \beta } _ { 0 } , \hat { \beta } _ { 1 } , 0 ; S ) - \ell ( \tilde { \beta } _ { 0 } , \tilde { \beta } _ { 1 } , \tilde { \beta } _ { 2 } ; S ) ) ,\tag{21}
$$

with $\hat { \beta } _ { 0 } , \hat { \beta } _ { 1 }$ the maximum likelihood estimates for the parameters in the model which includes only the crack length and $\widetilde { \beta } _ { 0 } , \widetilde { \beta } _ { 1 } , \widetilde { \beta } _ { 2 }$ the maximum likelihood estimates in the model that includes both the crack length and the resolution. Then, under the null-hypothesis $H _ { 0 } ,$ , Λ(S) follows aχ-squared distribution with one degree of freedom, since the extended model has one free parameter more than the null model. We reject the null-hypothesis at a 5 % significance level if $\Lambda ( S ) > \chi _ { 1 . 0 . 9 5 } ^ { 2 } \approx 3 . 8 4 1$ , i.e. if the probability under the null-hypothesis of observing a statistic at least so large is less than 5 % for the considered dataset.

## 6.2. Results ofincluding resolution in PoD

## 6.2.1. Binning approach

The PoD curves are shown in Fig. 13, where we denote the bins by their central resolution, e.g. $r ~ = ~ 1 7$ pix/mm corresponds to the 4-30 pix/mm bin. The confidence bounds, which can be computed as explained in Sec. 4, have been omitted for visual clarity.<sup>1</sup> The PoD curve estimated on images of all resolutions, given by Eq. (9), aligns with the PoD curve for the 2-4 pix/mm bin. Fig. 13 additionally shows the PoD curves from Campbell et al. (Eq. (11)) [20] and the DNVGL recommmendation (Eq. (10)) [16]. As one might expect, the detection capabilities of our CV method improve as resolution increases. For moderate resolutions $r ~ > ~ 1 . 5$ pix/mm, the probability of detection is above the PoD curve from the DNVGL recommendation over all crack lengths, and above the PoD curve estimated by Campbell et al. for cracks up to about 100 mm in length.

Table 2: Parameters of the PoD curves fitted on each resolution bin
<table><tr><td> $r \left( \mathrm { p i x / m m } \right)$ </td><td> $\beta _ { 0 }$ </td><td> $\beta _ { 1 }$ </td></tr><tr><td>0-1</td><td>-2.022</td><td>0.451</td></tr><tr><td>1-2</td><td>-2.024</td><td>0.752</td></tr><tr><td>2-4</td><td>-1.977</td><td>0.833</td></tr><tr><td>4-30</td><td>-3.673</td><td>1.387</td></tr></table>

## 6.2.2. Parametric approach

We fitted the PoD surface as explained in Sec. 6.1.2. The maximum likelihood estimates of the parameters are $\beta _ { 0 } = - 3 . 4 4 6 , \beta _ { 1 } = 0 . 8 8 1$ , and $\beta _ { 2 } = 1 . 0 5 8 .$ , so that

$$
\begin{array} { l } { \displaystyle \mathrm { P o D } _ { \mathrm { C V } } ( a , r ) } \\ { \displaystyle = \frac { \exp ( - 3 . 4 4 6 + 0 . 8 8 1 \log ( a ) + 1 . 0 5 8 \log ( r ) ) } { 1 + \exp ( - 3 . 4 4 6 + 0 . 8 8 1 \log ( a ) + 1 . 0 5 8 \log ( r ) ) } . } \end{array}\tag{22}
$$

Fig. 14 shows the cross-sections for resolution $r \quad = { }$ 0 5 1 5 3 17 pix/mm, compared to the PoD curve from the DNVGL recommendation [16] and the PoD curve estimated by Campbell et al. [20]. The confidence bounds, which can be computed as explained in Sec. 6.1.2, have again been omitted for visual clar-$\mathrm { i t y . } ^ { 1 }$ In line with the results from the binning approach (cf. Fig. 13), the probability of detection increases not only as the crack becomes longer, but also as the resolution of the image improves.

We also use hypothesis testing to assess whether extending the PoD curve model to include the efect of resolution is supported by data. In our data, we observe that the log-likelihood ratio statistic, defined in Eq. (21), is $\Lambda ( S ) = 6 9 . 6 6 > 3 . 8 4 \approx \chi _ { 1 . 0 . 9 5 } ^ { 2 }$ , so that we reject the null-hypothesis $H _ { 0 }$ in favour of the alternative hypothesis $H _ { a }$ . In other words, we have found the improvement of the PoD surface model in Eq. (16) compared to the PoD curve model in Eq. (3) to be statistically significant at a 5 % significance level for the considered dataset.

## 6.2.3. Comparison ofapproaches

We now compare the binning approach to the parametric approach. In Fig. 15, the dotted lines give the PoD curves found using the binning method, and the solid lines give cross-sections of the parametric PoD surface, with resolution in the middle of each of the bins. We quantify the performance of the two approaches by computing the mean squared error (MSE), where the error is simply the diference between the observed outcome (hit = 1, miss = 0) and the probability of detection. We find that the binned approach gives an MSE of 0 097, while the parametric approach has an MSE of 0 101.

It is possible that the reduction in error of the binned approach compared to the parametric approach is due to the greater flexibility aforded by the binned approach. It must be noted, however, that this flexibility comes at the cost of the efective addition of 5 parameters, since it uses 4 PoD curves with 2 parameters each instead of 1 PoD surface with 3 parameters. These additional parameters can harm the robustness and generalisation of the model. Since the error reduction is minor (∼ 4 %), we therefore perform the remaining experiments with the PoD surface model.

## 6.3. Numerical evaluation of resolution dependence of crack detection performance

To further study the efect of the resolution on the inspection performance, we utilise the approach and the metrics proposed in Sec. 5. We take constant resolution, $r \ = \ 0 . 5 , 1 . 5 , 3 , 1 7$ pix/mm, cross-sections of the PoD surface given in Eq. (22). Such a cross-section can be interpreted to represent the PoD curve of the automated CV inspection method with a fixed camera and distance to the crack surface. We also again include the PoD curves from the DNVGL recommendation (Eq. (10)) [16] and determined by Campbell et al. (Eq. (11)) [20].

Tables 3 and 4 summarise how the normalised undetected crack length C and the Kullback-Leibler divergence KL, given by Eqs. (12) and (13), respectively, depend on the prior mean $\mu$ and coeficient of variation CoV. The same results are shown visually in Figs. 16 and 17. A clear trend can be seen that with the improvement of resolution, the PoD performance improves. Furthermore, the results show that for low crack resolution up to $r = 1 . 5$ pix/mm, which corresponds to inspection with the camera having IFOV $\theta = 1 . 2 5 { \times } 1 0 ^ { - 4 }$ rad/pix from a distance of 7 metres, the performance of our CV crack detection method is consistently worse than conventional visual inspection, regardless of the prior crack distribution. We can also conclude that for our CV method to outperform the conventional inspection the resolution of the images collected during inspection should be more than 3 pix/mm, which corresponds to distances closer than 3 metres for camera with IFOV $\theta = 1 . 2 5 \times 1 0 ^ { - 4 }$ rad/pix.

![](images/811aa64daf3ecd6590b3b04d7bdd1d9f215b5c28120d5049a0b45943f0034187.jpg)  
Figure 13: PoD curves calculated for 4 resolution bins, compared to the baselines of Campbell et al. [20] and DNVGL [16]. Note that confidence bounds (Eq. (8)) are omitted for visual clarity.

Note that the comparison framework could easily be extended to also take into account resolution by addi tionally sampling the resolution from some prior distribution. However, since the PoD curves by Campbell et al. [20] and from the DNVGL recommendation [16] do not include resolution, this would not be useful for this experiment.

## 7. Conclusions

In this paper, we proposed a new framework, namely Alg. 1, for the evaluation of computer vision methods for the detection of cracks using PoD curves (Fig. 6). Unlike standard evaluation metrics such as F1-score, intersection over union, mean average precision, etc., which are commonly used to evaluate the performance of computer vision methods, this approach allows the comparison of our computer vision method to conventional visual inspection of bridges performed by a human inspector. The typical evaluation results (histograms, posterior crack length distributions) of the approach, comparing our CV method to Campbell’s method [20] and the DNVGL recommendation [16], are shown in Fig. 9. The proposed evaluation and comparison provides a better insight into the true performance of the computer vision method and can be used to assess the reliability of the method in real applications. In general, the approach is suitable for determining PoD curves for any automated approach (AI-based) for damage detection. As our widely applicable evaluation method can be integrated into general structural reliability frameworks, we release the comparison code [34].

Based on existing PoD curves for conventional visual inspection, our computer vision method trained on a selected subset of images of the “Cracks in Steel Bridges” dataset [1], is capable of outperforming conventional visual inspection to detect cracks of length smaller than 150 mm. For larger cracks, the performance of our computer vision method lies between diferent estimates of conventional visual inspection performance, namely by Campbell et al. [20], and by the DNVGL recommendation [16]. These findings are also supported numerically, see Figs. 7 and 8. Moreover, the results are consistent across a wide range of prior crack length distributions, as can be seen in Appendix A. Hence, automated visual inspection with our computer vision and conventional visual inspection are complementary and give the best performance when applied in conjunction.

![](images/1a1f2968e07732c4d963ec45baf1d5c84479d6fe9162d39f311e1033ee6f4f18.jpg)

Figure 14: Cross-sections of the PoD surface (Eq. (16)) at 4 fixed resolutions (solid lines), compared to the baselines of Campbell et al. [20] and DNVGL [16]. Note that confidence bounds (Eq. (20)) are omitted for visual clarity.  
![](images/9fed3f2a4b5b8fad0e0c4217bde3666808c0980892198af00435d0aeaba3b8e0.jpg)  
Figure 15: Fixed resolution cross-sections of the PoD surface (Eq. (16)) (solid lines), and the PoD curves fitted on each resolution bin (dotted lines).

C  
![](images/17ea257249a39abbc9c586029e10f48c7395a1fc7254bc524ea4fc8f9934f3de.jpg)  
CoV  
(a) µ = 37<sub>.</sub>15 mm.

![](images/778866af7f29a24e39b92ebec129097b830a911654e4e55a30cea08ad2dcffce.jpg)  
CoV  
(b) µ = 117<sub>.</sub>51 mm.

![](images/aca4fc2d0f87e7c4570e7eae0cb674b3303218e22e1d8af23853c56c7b2dbdb8.jpg)  
CoV  
(c) = 371 72 mm.

Figure 16: Results of computation of normalised undetected crack length C for our CV method and conventional visual inspection provided by Campbell et al. [20] and by the DNVGL recommendation [16]  
![](images/d6f64cd264f4adc78c5cd813102ed35866366e65e65b0da4b97a6edf1ad2f2e7.jpg)  
CoV  
(a) µ = 37 15 mm.

![](images/ac5b39ae0df6fa0dbce5e6b1d4b1e7619ebde0ce858c8c05ec7c51d0a31143ca.jpg)  
CoV  
(b) µ = 117 51 mm.

![](images/c97afb95159fcbe6785e9303125b1b8e788aa142e751a6e1d25ea88ecea275e3.jpg)  
CoV  
(c) µ = 371 72 mm.  
Figure 17: Results of computation of the Kullback-Leibler divergence KL for our CV method and conventional visual inspection provided by Campbell et al. [20] and by the DNVGL recommendation [16]

Table 3: Results of computation of the normalised undetected crack length C for the CV method at diferent resolutions and conventional visual inspection with PoDs provided by Campbell et al. [20] and by the DNVGL recommendation [16]
<table><tr><td rowspan="2">µ(mm)</td><td rowspan="2">CoV</td><td colspan="4"> $C _ { \mathrm { { C V } } }$ </td><td rowspan="2"> $C _ { \mathrm { C a m p b e l l } }$ </td><td rowspan="2"> $C _ { \mathrm { D N V G L } }$ </td></tr><tr><td>r = 17</td><td>r = 3</td><td>r = 1.5</td><td> $r = 0 . 5$ </td></tr><tr><td rowspan="4">37.15</td><td>0.25</td><td>0.061</td><td>0.285</td><td>0.452</td><td>0.723</td><td>0.434</td><td>0.493</td></tr><tr><td>0.5</td><td>0.060</td><td>0.276</td><td>0.436</td><td>0.703</td><td>0.406</td><td>0.474</td></tr><tr><td>1</td><td>0.057</td><td>0.253</td><td>0.396</td><td>0.650</td><td>0.340</td><td>0.428</td></tr><tr><td>2</td><td>0.051</td><td>0.213</td><td>0.331</td><td>0.557</td><td>0.256</td><td>0.355</td></tr><tr><td rowspan="4">117.51</td><td>0.25</td><td>0.023</td><td>0.127</td><td>0.232</td><td>0.489</td><td>0.141</td><td>0.247</td></tr><tr><td>0.5</td><td>0.023</td><td>0.125</td><td>0.225</td><td>0.471</td><td>0.135</td><td>0.240</td></tr><tr><td>1</td><td>0.022</td><td>0.116</td><td>0.207</td><td>0.428</td><td>0.122</td><td>0.220</td></tr><tr><td>2</td><td>0.020</td><td>0.102</td><td>0.176</td><td>0.358</td><td>0.102</td><td>0.186</td></tr><tr><td rowspan="4">371.72</td><td>0.25</td><td>0.008</td><td>0.050</td><td>0.099</td><td>0.259</td><td>0.003</td><td>0.099</td></tr><tr><td>0.5</td><td>0.008</td><td>0.050</td><td>0.097</td><td>0.251</td><td>0.007</td><td>0.098</td></tr><tr><td>1</td><td>0.008</td><td>0.048</td><td>0.092</td><td>0.230</td><td>0.017</td><td>0.092</td></tr><tr><td>2</td><td>0.008</td><td>0.043</td><td>0.081</td><td>0.195</td><td>0.024</td><td>0.082</td></tr></table>

Table 4: Results of computation of the Kullback-Leibler divergence KL for the CV method at diferent resolutions and conventional visual inspection with PoDs provided by Campbell et al. [20] and by the DNVGL recommendation [16].
<table><tr><td rowspan="2">µ(mm)</td><td rowspan="2">CoV</td><td colspan="4"> $\mathrm { K L } _ { \mathrm { C V } }$ </td><td rowspan="2"> ${ \mathrm { K L } } _ { \mathrm { C a m p b e l l } }$ </td><td rowspan="2">KLDNVGL</td></tr><tr><td>r = 17</td><td>r = 3</td><td>r = 1.5</td><td> $r = 0 . 5$ </td></tr><tr><td rowspan="4">37.15</td><td>0.25</td><td>0.021</td><td>0.012</td><td>0.007</td><td>0.002</td><td>0.005</td><td>0.007</td></tr><tr><td>0.5</td><td>0.074</td><td>0.040</td><td>0.023</td><td>0.006</td><td>0.020</td><td>0.022</td></tr><tr><td>1</td><td>0.209</td><td>0.099</td><td>0.056</td><td>0.015</td><td>0.055</td><td>0.055</td></tr><tr><td>2</td><td>0.400</td><td>0.158</td><td>0.087</td><td>0.025</td><td>0.080</td><td>0.083</td></tr><tr><td rowspan="4">117.51</td><td>0.25</td><td>0.022</td><td>0.018</td><td>0.013</td><td>0.006</td><td>0.104</td><td>0.015</td></tr><tr><td>0.5</td><td>0.081</td><td>0.062</td><td>0.046</td><td>0.020</td><td>0.286</td><td>0.051</td></tr><tr><td>1</td><td>0.244</td><td>0.167</td><td>0.118</td><td>0.049</td><td>0.447</td><td>0.128</td></tr><tr><td>2</td><td>0.510</td><td>0.290</td><td>0.192</td><td>0.076</td><td>0.414</td><td>0.199</td></tr><tr><td rowspan="4">371.72</td><td>0.25</td><td>0.023</td><td>0.021</td><td>0.019</td><td>0.012</td><td>1.100</td><td>0.022</td></tr><tr><td>0.5</td><td>0.085</td><td>0.076</td><td>0.067</td><td>0.043</td><td>2.132</td><td>0.078</td></tr><tr><td>1</td><td>0.258</td><td>0.219</td><td>0.184</td><td>0.108</td><td>2.276</td><td>0.211</td></tr><tr><td>2</td><td>0.575</td><td>0.424</td><td>0.328</td><td>0.173</td><td>1.593</td><td>0.363</td></tr></table>

The resolution of the images plays an important role in the detectability of cracks by our computer vision method. We analysed this efect using two approaches: a nonparametric approach (Sec. 6.1.1) and a parametric one (Sec. 6.1.2). Using our proposed inspection comparison method (Alg. 1), we find that our computer vision method gives better performance than conventional visual inspection on the entire range of crack lengths, assuming that the images were collected by a camera with IFOV less than 0.0001 rad/pix and from a distance less than 3 m.

Overall, we have established a statistical evaluation framework to allow the comparison of CV crack detection methods with conventional inspection methods. Although beyond the scope of this article, it would be interesting to compare other CV methods using this evaluation approach in the future.

## Acknowledgments

The authors would like to thank the Dutch bridge infrastructure owners ProRail and Rijkswaterstaat, and the company Nebest, for their support in acquiring information on cracks in steel bridges. The research is primarily funded by the Eindhoven Artificial Intelligence Systems Institute (EAISI), and partly by the Dutch Foundation of Science NWO (Geometric Learning for Image Analysis, VI.C 202-031).

## References

[1] A. Kompanets, D. Leonetti, R. Duits, H. B. Snijder, Cracks in Steel Bridges (CSB) dataset, 4TU.ResearchData (2024). doi:10.4121/6162 a9b6-2a20-4600-8207-e9dcd53a264a.

[2] K. C. Brady, M. O’Reilly, L. Bevc, A. Znidaric,ˇ E. O’Brien, R. Jordan, Cost 345. procedures required for the assessment of highway structures. final report, Tech. rep., European Co-operation in the Field of Scientific and Technical Research, Brussels, Belgium (2009). URL http://cost345.zag.si/Reports/COST\_34 5\_Summary\_Document.pdf

[3] B. Bell, D1. 2 european railway bridge demography, Sustainable Bridges–Assessment for Future Trafic Demands and Longer Lives–a project within EU FP6 (2004). URL https://www.diva-portal.org/smash/get/ diva2:1330533/FULLTEXT01.pdf

[4] Staat van de infrastructuur rijkswaterstaat, Tech. rep., Rijkswaterstaat, accessed: 2024-03-06 (2022). URL https://www.rijksoverheid.nl/documente n/rapporten/2023/12/04/bijlage-3-staat-van -de-infrastructuur-2023

[5] F. Shahrivar, A. Sidiq, M. Mahmoodian, S. Jayasinghe, Z. Sun, S. Setunge, Ai-based bridge maintenance management: a comprehensive review, Artificial Intelligence Review 58 (5) (2025) 135. doi:10.1007/s10462-025 -11144-7.

[6] M. D. Isac, C. Cîmpean, D. L. Manea, The current status of structural monitoring: A bibliometric literature review, Buildings 15 (5) (2025) 739. doi:10.3390/buil dings15050739.

[7] L. Ali, F. Alnajjar, W. Khan, M. A. Serhani, H. Al Jassmi, Bibliometric analysis and review of deep learning-based crack detection literature published between 2010 and 2022, Buildings 12 (4) (2022) 432. doi:10.3390/buildings12040432.

[8] D. Amirkhani, M. S. Allili, L. Hebbache, N. Hammouche, J.-F. Lapointe, Visual concrete bridge defect classification and detection using deep learning: A systematic review, Transactions on Intelligent Transportation Systems 25 (9) (2024) 10483–10505. doi:10.110 9/TITS.2024.3365296.

[9] T. Panigati, M. Zini, D. Striccoli, P. F. Giordano, D. Tonelli, M. P. Limongelli, D. Zonta, Drone-based bridge inspections: Current practices and future directions, Automation in Construction 173 (2025) 106101. doi:10.1016/j.autcon.2025.106101.

[10] B. Wójcik, M. Zarski, Asesment of state-of-the-art<sup>˙</sup> methods for bridge inspection: case study, Archives of Civil Engineering 66 (4) (2020) 343–362. doi:10.244 25/ace.2020.135225.

[11] Y. Liu, J. Yao, X. Lu, R. Xie, L. Li, Deepcrack: A deep hierarchical feature learning architecture for crack segmentation, Neurocomputing 338 (2019) 139–153. doi: 10.1016/j.neucom.2019.01.036.

[12] I.-H. Kim, S. Yoon, J. H. Lee, S. Jung, S. Cho, H.-J. Jung, A comparative study of bridge inspection and condition assessment between manpower and a uas, Drones 6 (11) (2022). doi:10.3390/drones6110355.

[13] J. Kim, T. Ruan, C. A. Contreras, M. Chiou, An exploratory study on crack detection in concrete through human-robot collaboration, arXiv preprint (2025). doi: 10.48550/arXiv.2508.11404.

[14] L. Gandossi, C. Annis, Probability of Detection Curves: Statistical Best-Practices, ENIQ TGR Technical Document ENIQ Report No. 41; EUR 24429 EN, European Commission, Joint Research Centre (JRC), Institute for Energy, petten, The Netherlands (Jan. 2011). doi:10.2790/21826.

[15] Nondestructive evaluation system reliability assessment, Tech. Rep. MIL-HDBK-1823A, U.S. Department of Defense, Wright-Patterson Air Force Base, OH, handbook; supersedes MIL-HDBK-1823 (2004) (Apr. 2009). URL https://statistical-engineering.com/w p-content/uploads/2017/10/MIL-HDBK-1823A20 09.pdf

[16] Probabilistic methods for planning of inspection for fatigue cracks in ofshore structures, Recommended Practice DNVGL-RP-C210, DNV GL (Nov. 2015). URL https://www.dnv.com/energy/standards-g uidelines/dnv-rp-c210-probabilistic-metho ds-for-planning-of-inspection-for-fatigue -cracks-in-offshore-structures/

[17] D. Leonetti, J. Maljaars, H. B. Snijder, Influence of inspection on the safety of fatigue loaded welded cruciform steel joints – comparison of simplified and advanced crack growth models, in: H. B. Snijder, B. De Pauw, S. van Alphen, P. Mengeot (Eds.), IABSE CONGRESS GHENT 2021: Structural Engineering for Future Societal Needs, International Association for Bridge and Structural Engineering, 2021, pp. 1546– 1554. URL https://research.tue.nl/en/publication s/influence-of-inspection-on-the-safety-o f-fatigue-loaded-welded-cr/

[18] J. Maljaars, A. Vrouwenvelder, Probabilistic fatigue life updating accounting for inspections of multiple critical locations, International Journal of Fatigue (2014). doi: 10.1016/j.ijfatigue.2014.06.011.

[19] S. Hashemi, J. Maljaars, H. B. Snijder, Added value of regular in-service visual inspection to the fatigue reliability of structural details in steel bridges, in: 5th International Conference on Smart Monitoring, Assessment and Rehabilitation of Civil Structures, 2019, pp. 1–8. URL https://research.tue.nl/en/publication s/added-value-of-regular-in-service-visua l-inspection-to-the-fatigu/

[20] L. E. Campbell, L. R. Snyder, J. M. Whitehead, R. J. Connor, J. B. Lloyd, Probability of Detection Study for Visual Inspection of Steel Bridges: Volume 2—Full Project Report, Tech. rep., Indiana Dept. of Transportation (2019). doi:10.5703/1288284317104.

[21] A. Kompanets, R. Duits, G. Pai, D. Leonetti, H. B. Snijder, Loss function inversion for improved crack segmentation in steel bridges using a cnn framework, Automation in Construction 170 (2025) 105896. doi:ht tps://doi.org/10.1016/j.autcon.2024.105896.

[22] A. Kompanets, R. Duits, D. Leonetti, H. B. Snijder, Two-stage fatigue crack detection framework with crack-preserving downsampler, International Journal of Fatigue 109179 (2025) 1–15. doi:10.1016/j.ijfati gue.2025.109179.

[23] S. Ren, K. He, R. Girshick, J. Sun, Faster r-cnn: Towards real-time object detection with region proposal networks, Transactions on Pattern Analysis and Machine Intelligence 39 (6) (2017) 1137–1149. doi:10.1109/ TPAMI.2016.2577031.

[24] K. Chen, J. Wang, J. Pang, Y. Cao, Y. Xiong, X. Li, S. Sun, W. Feng, Z. Liu, J. Xu, Z. Zhang, D. Cheng, C. Zhu, T. Cheng, Q. Zhao, B. Li, X. Lu, R. Zhu, Y. Wu, J. Dai, J. Wang, J. Shi, W. Ouyang, C. C. Loy, D. Lin, MMDetection: Open mmlab detection toolbox and benchmark, arXiv preprint (2019). doi:10.48550 /arXiv.1906.07155.

[25] S. Xie, R. Girshick, P. Dollár, Z. Tu, K. He, Aggregated residual transformations for deep neural networks, in: Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, 2017, pp. 5987–5995. doi: 10.1109/CVPR.2017.634.

[26] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, L. Fei-Fei, Imagenet: A large-scale hierarchical image database, in: 2009 IEEE conference on computer vision and pattern recognition, IEEE, 2009, pp. 248–255. doi:https: //doi.org/10.1109/CVPR.2009.5206848.

[27] I. Loshchilov, F. Hutter, Decoupled Weight Decay Regularization, in: International Conference on Learning Representations (ICLR), 2019, pp. 1–18. URL https://openreview.net/forum?id=Bkg6Ri CqY7

[28] I. Loshchilov, F. Hutter, Sgdr: Stochastic Gradient Descent with Warm Restarts, in: International Conference on Learning Representations (ICLR), 2017, pp. 1–16. URL https://openreview.net/forum?id=Skq89S cxx

[29] T. D. Righiniotis, A comparative study of fatigue inspection methods, Journal of Constructional Steel Research 62 (4) (2006) 352–358. doi:10.1016/j.jcsr.2005. 08.008.

[30] G. A. Georgiou, Probability of detection (pod) curves, Recommended Practice 454, UK Health and Safety Executive (2006). URL https://webarchive.nationalarchives.go v.uk/ukgwa/20241208023035/https://www.hse. gov.uk/research/rrhtm/rr454.htm

[31] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, E. Duchesnay, Scikitlearn: Machine learning in Python, Journal of Machine Learning Research 12 (2011) 2825–2830. URL http://jmlr.org/papers/v12/pedregosa1 1a.html

[32] S. Kullback, R. A. Leibler, On Information and Sufficiency, The Annals of Mathematical Statistics 22 (1) (1951) 79 – 86. doi:10.1214/aoms/1177729694.

[33] P. K. Kaiser, Prospective evaluation of visual acuity assessment: a comparison of snellen versus ETDRS charts

in clinical practice (An AOS Thesis), Transactions of the American Ophthalmological Society 107 (2009) 311. URL https://europepmc.org/articles/PMC2814 576

[34] F. M. Sherry, finnsherry/PoDCurveComparison (Aug. 2026). doi:10.5281/zenodo.21771171. URL https://github.com/finnsherry/PoDCurve Comparison

Appendix A. Results of the crack update simulation

![](images/a1cb73e4b2ae49e2af4f3437b82c5675b22de4199e0db4b3eae791802b364f61.jpg)  
(a)

![](images/aaabcb80354a8701ad3eddc392725b33720d44656a1e355a4c47856404e2ccbf.jpg)  
(b)

Figure A.18: Histogram and CDF for prior crack-length distribution with $\mu = 3 7 . 1 5$ mm and $\mathrm { C o V } = 0 . 2 5$  
![](images/863e8309163dcfdd41298583d5033a51a155553c0d9bdbee8212bf8709fb6460.jpg)  
(a)

![](images/646ce18fa5e48dbcfdbcecb7ac98615bcc17903eb0c305393d6612f5db14165f.jpg)  
(b)

Figure A.19: Histogram and CDF for prior crack-length distribution with $\mu = 3 7 . 1 5$ mm and $\mathrm { C o V } = 0 . 5$  
![](images/ed299b7dc9f881eeb7250bd3d27340cd40ee8fe9271dc6f8896e734e0f48bdac.jpg)  
(a)

![](images/9601850817647c8e471b9292facd43a5bd89f16a4fc8befe987e6bbd43a32de5.jpg)  
(b)

Figure A.20: Histogram and CDF for prior crack-length distribution with $\mu = 3 7 . 1 5$ mm and $\mathrm { C o V } = 1$  
![](images/3bfcf49c2b1faf0d616ed78cece31ac0542ae221614e698b79076f56783809de.jpg)  
(a)

![](images/6f4ee79ed4adfe55563d0c3cca81a85ee8f2dc83397e92fa4e58d96f52c9a246.jpg)  
(b)  
Figure A.21: Histogram and CDF for prior crack-length distribution with $\mu = 3 7 . 1 5$ mm and $\mathbf { C o V } = 2 .$

![](images/045509bb9d9089176301849a2c033232308ef0c5924a91474c2d89c620102488.jpg)  
(a)

![](images/2fcbc51bb38f0ef1c06a335943f675af389cd13710c85d49288b14efb37d7c32.jpg)  
(b)

Figure A.22: Histogram and CDF for prior crack-length distribution with $\mu = 1 1 7 . 5 1$ mm and $\mathrm { C o V } = 0 . 2 5$  
![](images/5431c9907dce13dfab372cb32dc105ef91f771b48c04e09367be79d365890972.jpg)  
(a)

![](images/b9897d6a587ea31dd9256a8d0be943d874835ab4989d2909317c4de149a6dc80.jpg)  
(b)

Figure A.23: Histogram and CDF for prior crack-length distribution with $. \mu = 1 1 7 . 5 1$ mm and $\mathrm { C o V } = 0 . 5 $  
![](images/9b413d6f154995d89ad47b73a81cf13a5716c8e59bd76896e85a0c824e7e6e9c.jpg)  
(a)

![](images/e44e74214d1ed52d8412ef58f403adb26f457f4256322440ccd6a0c47a080523.jpg)  
(b)

Figure A.24: Histogram and CDF for prior crack-length distribution with $\mu = 1 1 7 . 5 1$ mm and $\mathrm { C o V } = 1$  
![](images/f78a6af10101f99c567eee74aafcb18291216bc230ae118192257de2d5214fc8.jpg)  
(a)

![](images/7d8b1db2aa585ec1974e881978202c12637fd9750f44cb60ae50a5c155218e67.jpg)  
(b)  
Figure A.25: Histogram and CDF for prior crack-length distribution with $\mu = 1 1 7 . 5 1$ mm and $\mathrm { C o V } = 2 .$

![](images/a013fe15bc87c26fc320d5248ecccbbccd9aebd47c3443b0a67cd803e8a9c2e1.jpg)  
(a)

![](images/938a9c63711e6696c3a833f12df93b58fbb198cc9905d7aa351832c834b41456.jpg)  
(b)

Figure A.26: Histogram and CDF for prior crack-length distribution with $\mu = 3 7 1 . 7 2$ mm and $\mathrm { C o V } = 0 . 2 5$  
![](images/e2ea9bccd8fc04a8bac30556d40ae887d3bb2a99997d8eade57d5a3ec00b8e61.jpg)  
(a)

![](images/1dbf002673e8f03afaefee235b95fe5b45e17c15dfd6811e5df4fd9c51dd0d93.jpg)  
(b)

Figure A.27: Histogram and CDF for prior crack-length distribution with $\mu = 3 7 1 . 7 2$ mm and $\mathrm { C o V } = 0 . 5 $  
![](images/91c282f2b3ba8c60278f1e29ea5b5c3f3b690a1e6cbd21edcc5b582c61b5d839.jpg)  
(a)

![](images/ff8d05072a552f3c08ca338f0a367148fc9dedbb79517451664ff8f9641f09fc.jpg)  
(b)

Figure A.28: Histogram and CDF for prior crack-length distribution with $\mu = 3 7 1 . 7 2$ mm and $\mathrm { C o V } = 1$  
![](images/33d163a745f7e0c9551695aff782994498c6be10f7cc620037d3b842000aee18.jpg)  
(a)

![](images/7563addda8d5563521fa7f51968ae28c1a57fbc3bffb17266bdd80ba55bdc5e8.jpg)  
(b)  
Figure A.29: Histogram and CDF for prior crack-length distribution with $\mu = 3 7 1 . 7 2$ mm and $\mathrm { C o V } = 2 .$