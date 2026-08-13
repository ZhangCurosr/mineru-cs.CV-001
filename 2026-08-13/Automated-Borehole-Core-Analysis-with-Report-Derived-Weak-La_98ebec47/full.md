# Automated Borehole Core Analysis with Report-Derived Weak Labels and Supervised Crack Segmentation

Usama Imdad<sup>a,e,∗</sup>, Ali Khan<sup>b</sup>, Luke Lu<sup>d</sup>, Zubair Khalid<sup>a,e</sup> and Arif Mahmood<sup>c</sup>

<sup>a</sup>Lahore University ofManagement Sciences, Pakistan

<sup>b</sup>Sydney Water, Australia

<sup>c</sup>Information Technology University, Pakistan

<sup>d</sup>Aurecon, Australia

<sup>e</sup>PI-Neuron, Australia

## A R T I C L E I N F O

Keywords:   
Borehole core   
defect spacing   
crack segmentation   
self-supervised learning   
bedding angle   
lithological color

## A BS T R AC T

Borehole archives commonly contain core tray photographs and corresponding digital log reports, but no native pixel-level crack annotations. We investigate two complementary approaches for extracting defect-spacing information from these archives. First, structured spacing categories recovered from the report text layer provide weak interval-level labels for classification. A DINO encoder trained on unlabeled core crops supplies domain-specific representations, and a manually verified subset is used to identify label inconsistencies. Second, we manually annotate 5,087 extracted core-row images and evaluate fully supervised crack-segmentation models. Our gated U-Net combines PiDiNet edge maps with Mask R-CNN masks through a learned spatial gating mechanism. This configuration achieves an F1 score of 0.860 and a crack-class IoU of 0.754, the highest result among the evaluated segmentation configurations. Deterministic post-processing converts predicted crack locations into defect-spacing categories. Separate rulebased branches estimate core-relative bedding angles and lithological color descriptors; their predictions agree with log-report references on 75.4% and 84.7% of 1,200 evaluated images, respectively. Because these references are extracted from existing reports, the reported values measure agreement with recorded geological observations rather than independent physical accuracy. The resulting framework combines report-derived weak supervision for spacing classification with fully supervised segmentation for image-based crack localization.

## 1. Introduction

Rock mass classification systems (RMCS), including RQD, RMR, and Q, summarize rock mass condition in indices that guide tunnel support selection, slope design, and excavation planning. These indices depend on the geometry, condition, and orientation of cracks in the rock (Alejano, 2025). We use cracks (discontinuities) consistently throughout this paper. We also retain the operator’s column heading, defect spacing, for the distance between successive natural cracks along the core. Defect spacing is a dominant term in these indices. Because the indices directly influence support decisions and costs, the quality of the underlying spacing measurements constrains the quality of the resulting engineering decisions.

In the archive considered here, a geologist lays out the recovered core, distinguishes natural cracks from drilling-induced cracks, and visually assigns a defect-spacing category based on professional judgement rather than measuring every crack-to-crack interval. This procedure is time-consuming and depends on subjective judgements about individual cracks and spacing categories. Consequently, defect spacing measurements show substantial interobserver variability and limited reproducibility (Alejano, 2025). Two geologists can derive diferent indices from the same core, and completed log reports are rarely revisited.

Automation must operate on the two artifact types routinely retained for each borehole. The first type comprises many core tray photographs (Figure 1, left). Each photograph contains three to five approximately one-metre core rows, with a yellow ruler providing the pixel-to-millimetre reference. The second type is a multi-page digital log report (Figure 1, right), produced by the operator’s logging software.

![](images/b138e4e80012d79d6db5ff291c5b948e0b9bb7ed31158f371ac8ed96de2208f6.jpg)  
Figure 1: The two input artifacts. Left: a core tray image containing the core images and the yellow ruler used to establish the pixel-to-millimetre scale. Right: the corresponding log report. Its “AVERAGE DEFECT SPACING (mm)” column contains the 10–3000 mm categories recovered as pseudo-labels. The artifacts are matched by borehole identifier and depth interval. Solid blocks redact client-identifying information.

The core images present additional challenges. White depth labels are attached to the rock surface, core loss creates gaps, scratches can resemble cracks, and dark lithologies can reduce crack contrast almost entirely. Each extracted core image has an extreme aspect ratio of approximately 400 × 5000 pixels. The target cracks are therefore thin, elongated structures within a very wide image. Most importantly, the archive contains no pixel-level ground truth. The log report records only a defect spacing category for each interval, so a conventional supervised formulation is not initially possible.

Existing work does not resolve this gap. CNN-based methods estimate RQD directly from core tray images (Alzubaidi, Mostaghimi, Si, Swietojanski and Armstrong, 2019, 2022b). Related pipelines segment cracks and recover bedding angles from core tray images or unwrapped core images (Liu, Zhu, Liu, Wang and Chen, 2025; Alzubaidi, Makuluni, Clark, Lie, Mostaghimi and Armstrong, 2022a; Zhou, Chang, Zheng, Zhu and Nie, 2023). These methods assume purpose-captured images and, in most cases, pixel-level annotations. Crack segmentation is well established for concrete, pavement, and masonry (Hamishebahar, Guan, So and Jo, 2022; Liu, Yao, Lu, Xie and Li, 2019; Kulkarni, Singh, Balakrishnan, Sharma, Devunuri and Korlapati, 2022), but these materials difer substantially from rock in texture and contrast. A method that converts this noisy, annotation-free archive into quantitative attributes is still missing. Section 2 reviews the relevant literature.

We study two complementary formulations. The first treats the structured defect-spacing categories in the log reports as weak interval-level supervision for classification; it is intended for settings in which pixel-level annotations are unavailable. The second uses polygon masks created in this study to train fully supervised crack-segmentation models, after which deterministic geometric post-processing derives spacing from the predicted crack positions. The two formulations therefore address diferent annotation regimes and should not be interpreted as one weakly supervised segmentation method. Parallel rule-based branches estimate core-relative bedding angles and lithologica color descriptors from the same extracted core images.

## Our contributions are:

1. We formulate defect-spacing analysis under two annotation regimes: report-derived weak supervision for spacing classification and fully supervised crack segmentation followed by geometric spacing computation.

2. We evaluate a gated U-Net configuration that combines PiDiNet edge maps and instance-segmentation masks and achieves the highest crack-segmentation result among the evaluated configurations. We also document a setting in which fusion with a sparse CrackCLIP input degrades performance.

3. We provide a deterministic post-processing procedure that converts ordered crack-mask components into physical spacing intervals using the image scale.

4. We evaluate auxiliary rule-based procedures for core-relative bedding-angle estimation and lithological color description against values extracted from the corresponding log reports.

## 2. Related Work

Rock mass classification and automated core logging. RMCS summarize rock quality for design and support decisions and remain central to geotechnical practice (Alejano, 2025). These systems rely on crack spacing, crack condition, and structural orientation. This dependence motivates the automated extraction of crack geometry and defect spacing from core images. CNN-based methods estimate RQD directly from core tray images and reduce the subjectivity of manual logging (Alzubaidi et al., 2019, 2022b).

Core images and bedding angle estimation. Core tray images and unwrapped core images support automated crack analysis with explicit geometric outputs. Liu et al. (2025) compare direct crack segmentation with an indirect method that first identifies core blocks and then fits crack points to a plane. Workflows based on unwrapped core images and Mask R-CNN provide accurate crack detection and bedding angle estimates (Alzubaidi et al., 2022a). Other studies estimate bedding angles from 3D digital core images through plane fitting (Zhou et al., 2023). These studies motivate our emphasis on interpretable intermediate outputs.

Digital rock and CT-based crack segmentation. Digital rock physics and CT workflows demonstrate that segmentation quality directly afects crack quantification. Deep CNNs designed for rock crack images improve robustness to visual clutter (Byun, Kim, Yoon, Kang and Song, 2021). Benchmarking studies also show that learning-based segmentation reduces user bias and improves downstream rock-physics estimates (Reinhardt, Jacob, Sadeghnejad, Cappuccio, Arnold, Frank, Enzmann and Kersten, 2022). Recent CT-based methods use deep networks to segment cracks and minerals in rock samples (He, Sadeghpour, Shi, Mishra and Roshankhah, 2024) or combine tomography with neural segmentation to extract crack networks (Caputo, Calcagni, Salerno, Mammoliti and Castellini, 2025). Engineering geology datasets further demonstrate that instance segmentation can automate crack identification at scale (Ji, Song, Zhang, Li, Xue and Chen, 2025).

Crack detection and segmentation in computer vision. Crack segmentation has progressed from classical edgebased methods to learned dense prediction (Hamishebahar et al., 2022). DeepCrack introduced multi-scale feature aggregation for this task (Liu et al., 2019). Transfer learning and CNN-based segmentation have also been applied to masonry (Dais, Bal, Smyrou and Sarhosis, 2021), while hierarchical feature aggregation improves pavement crack detection (Yang, Zhang, Yu, Prokhorov, Mei and Ling, 2019). Contrastive self-supervision and mixed-label strategies address label scarcity (Song, Yao, Tian, Guo, Muniyandi and An, 2024; Zhang, Xu, Zhu and Xie, 2024), and segmentation remains an efective baseline for industrial surface images (Joshi, Singh and Sharma, 2022). Architectures such as DepthCrackNet further improve segmentation accuracy (Saberironaghi and Ren, 2024). Vision-language methods, including CrackCLIP, extend weakly supervised segmentation through text prompts (Liang, Li, Yu and Wang, 2025; Radford, Kim, Hallacy, Ramesh, Goh, Agarwal, Sastry, Askell, Mishkin, Clark et al., 2021).

Datasets, benchmarks, and learning with limited labels. Data scarcity and heterogeneity remain important challenges. CrackSeg9k combines multiple datasets into a unified benchmark for cross-domain generalization (Kulkarni et al., 2022). Crack500 and CrackForest provide established crack detection baselines (Zhang, Yang, Zhang and Zhu, 2016; Shi, Cui, Qi, Meng and Chen, 2016), while CrackTree represents an earlier algorithmic approach (Zou, Cao, Li, Mao and Wang, 2012). Together, these datasets and methods inform the design of pipelines that tolerate label noise and domain shift.

Self-supervision and geometric priors. Self-supervised representation learning with DINO improves robustness when labels are sparse or noisy (Caron, Touvron, Misra, Jégou, Mairal, Bojanowski and Joulin, 2021). Edge and line detectors such as PiDiNet and MLSD provide geometric priors for fine crack boundaries and core segmentation (Su, Liu, Yu, Hu, Liao, Tian, Pietikäinen and Liu, 2021; Gu, Ko, Go, Lee, Lee and Shin, 2022). These tools motivate hybrid pipelines that combine classical geometry with learned representations for interpretable core analysis.

![](images/fc5de78568cf52c6f6137e1e1ac97538650fe5e0e5fb152e85c4919c0818793d.jpg)  
Figure 2: Core-row extraction. Top left: input core tray photograph. Top right: HSV mask used to detect the yellow ruler. Bottom left: candidate core-row boundaries. Bottom right: candidate row crops before manual quality control. Solid blocks redact client-identifying information. The first candidate contains the marker roll rather than core and is rejected during manual quality control.

## 3. Data and Preprocessing

## 3.1. Study Area and Data Provenance

The corpus contains hundreds of boreholes gathered from all over Australia. Confidentiality restrictions prevent us from identifying the operator and project. We have therefore removed all client-identifying fields from the figures (Figures 1 and 2).

Each borehole is associated with one digital log report and many core tray photographs. The reports store text and vector drawing commands, allowing the recorded values to be read without optical character recognition. The tray photographs are supplied as separate image files; although copies also appear inside the reports, only the separately supplied photographs are used for image analysis. Filenames encode the borehole identifier and depth interval, allowing each photograph to be aligned with the corresponding report interval.

The cored intervals are dominated by interbedded siltstone and sandstone with subordinate clay and span weathering grades from extremely to moderately weathered. Core was recovered using an HQ3 triple-tube system with a nominal 61 mm core diameter. Logged depths range from approximately 10 to 60 m. Across the corpus, multiple tray photographs per borehole yielded 5,087 retained core-row images after the exclusions described in Section 3.4.

Following the operator’s standard procedure, the logging geologist recorded defect spacing, bedding angle, weathering, and lithological color in structured log report columns. These include “AVERAGE DEFECT SPACING (mm)” with fixed 10–3000 mm categories, crack descriptions with bedding angles, and material descriptions with color. The recorded observations provide the reference for every evaluation in Section 5. No independent measurements of the core were available. Section 6.5 discusses the implications of this constraint.

## 3.2. Core Image Extraction

We extract core-row images in two stages. First, Algorithm 1 detects and rectifies the tray region below the yellow ruler. The tray image is converted to HSV color space and thresholded to isolate the ruler. Morphological operations clean the mask, and aligned ruler fragments are merged using vertical IoU and a convex hull. The fitted ruler orientation is then used to correct image rotation. Second, filename metadata provides the expected number of core rows. A linesegment detector (Gu et al., 2022) identifies the horizontal row boundaries, and the expected row count is used to select and order the resulting crops. All candidate crops are manually inspected; non-core regions, placeholder rows, and extraction failures are removed before downstream analysis.

Each core tray image contains three to five core images, and each core image spans approximately one metre of core. We use the ruler length to normalize scale and convert pixel distances to physical distances. This conversion is required for defect spacing categorization. Figure 2 shows the extraction stages.

```latex
Algorithm 1 Tray Region Extraction and Rectification
Require: RGB core tray image �
Ensure: Rectified image $I ^ { \prime } { , }$ , tray region �, and ruler mask �
1: $I ^ { \prime }  I$
2: Convert � to HSV space, $I _ { \mathrm { H S V } }  \mathrm { C o n v e r t T o H S V } ( I )$
3: Define yellow color bounds, $L \gets [ 2 0 , 1 0 0 , 1 0 0 ]$ and � ← [30, 255, 255]
4: Generate binary mask, � ← InRange $( I _ { \mathrm { H S V } } , L , U )$
5: Apply morphological closing on �
6: Fill internal holes using contours
7: Extract all external contours  from �
8: Find contour $c _ { \mathrm { m a x } } \in { \mathcal C }$ with maximum width $w _ { \mathrm { m a x } }$
9: if $\begin{array} { r } { w _ { \mathrm { m a x } } < \frac { 3 } { 4 } } \end{array}$ ⋅ image width then
10: Initialize set $S \gets \{ c _ { \mathrm { m a x } } \}$
11: for each $c \in c$ do
12: Compute vertical IoU between � and $c _ { \mathrm { m a x } }$
13: if IoU $> 0 . 3$ then
14: Add � to 
15: Merge contours in  with a convex hull to obtain $c _ { \mathrm { m e r g e d } }$
16: Set $c _ { \operatorname* { m a x } } \gets c _ { 1 }$ merged
17: Draw filled contour $c _ { \mathrm { m a x } }$ on a blank mask $R _ { \mathrm { m a s k } }$
18: if rotation correction is enabled then
19: Fit rotated rectangle � to $c _ { \mathrm { m a x } }$
20: Compute rotation matrix � from �
21: Apply afine warp to $R _ { \mathrm { m a s k } }$ using �
22: Update the region bounding box
23: Define � as the rectangular tray region below $c _ { \mathrm { m a x } }$
24: if rotation correction is enabled then
25: �<sup>′</sup> ← AffineWarp(�, �)
26: Crop � from the rectified image $I ^ { \prime }$
27: return �<sup>′</sup>, �, and �
```

## 3.3. Defect Spacing Label Extraction

To obtain the report-derived labels used in the classification experiments, we locate the “SPACING” column in each digital log report using the deterministic procedure in Algorithm 2. PyMuPDF reads words and their bounding boxes from the PDF text layer, while vector drawing commands provide the column rules; optical character recognition is not used. The vertical rules on either side of the heading define the column bounds, and each adjacent pair of rules defines a candidate region. Each region is rendered, converted to grayscale, and binarized to identify the graphical spacing patches. Patch heights are converted to report-depth intervals using the page scale and mapped to the operator’s spacing categories. The resulting report-derived labels are aligned with core-row depth intervals and used as weak supervision in the classification experiments.

## 3.4. Segmentation Dataset

We excluded tray positions in which no core was recovered and a white placeholder label appeared instead of rock. These images contain no crack evidence. Their bright, uniform surfaces would be scored as trivially correct background and would inflate the metrics without demonstrating an ability to segment rock. The remaining 5,087 core images form the dataset described below.

For crack segmentation, we manually annotated the 5,087 core-row images with polygonal masks. Each image has an extreme aspect ratio of approximately $4 0 0 \times 5 0 0 0$ pixels and contains thin, elongated cracks. Crack regions touching white labels on the core surface were not annotated and were therefore treated as background during training and evaluation. This choice avoids using ambiguous label boundaries as crack supervision, but a prediction on a real label-touching crack is counted as a false positive relative to the annotation. When adjacent cracks could not be visually separated, we drew one polygon around the complete cracked region. Annotations were saved in Labelme JSON format.

Algorithm 2 Defect Spacing Label Extraction   
Require: Page render $I _ { p } ,$ column rules �, text-layer words � with bounding boxes, top and bottom rules � and �,   
and page number �   
Ensure: Mapping from vertical regions to defect spacing labels   
1: Locate “SPACING” in the text-layer words and obtain its bounding box   
2: if the phrase is not found then   
3: Raise an error   
4: Identify the vertical rules to the left and right of the phrase   
5: Extract vertical rules within the horizontal and vertical column bounds   
6: Sort these vertical lines to define column regions   
7: Initialize a list of predefined defect spacing labels   
8: for each adjacent pair of vertical lines do   
9: Extract the sub-image bounded by �, �, and the two vertical rules   
10: Convert to grayscale and binarize to highlight black patches   
11: Crop side margins to remove noise   
12: Calculate the pixel-to-length scale from the ruler height   
13: Detect connected black regions (spacing patches)   
14: for each patch do   
15: Convert its pixel height to real-world units using scaling   
16: Assign the corresponding defect spacing label to the region   
17: Remove any overlapping regions   
18: return the mapping from physical regions to defect spacing labels

We randomly partitioned the boreholes into training, validation, and test sets using a 70/15/15 ratio. All core-row images and derived crops from a borehole were assigned to the same partition. We also created a split-based variant by dividing each core-row image into ten vertical cells. Table 1 reports the resulting image and cell counts.

Data-owner restrictions prevent disclosure of the exact aggregate and per-partition borehole counts; however, borehole identifiers were used internally as grouping keys, and no borehole contributed images to more than one partition.

## Table 1

Borehole-grouped partitions used for the crack-segmentation experiments. All core-row images and derived crops from one borehole remain in the same partition.
<table><tr><td>Dataset unit</td><td>Train</td><td>Validation</td><td>Test</td></tr><tr><td>Core-row images</td><td>3561</td><td>763</td><td>763</td></tr><tr><td>Core splits</td><td>35610</td><td>7630</td><td>7630</td></tr></table>

## 4. Methods

## 4.1. Defect Spacing Classification

Classical pipeline. We apply glare removal, Canny edge detection, and contour filtering to identify cracks. We then measure the crack-center positions and the distances between them. The method detects many cracks but also produces false positives under intensity variation and surface artifacts.

Supervised detection. As a preliminary experiment separate from the crack-segmentation dataset in Table 1, we train a YOLOv8-Large model on a 710-image bounding-box dataset for the six defect-spacing classes (10, 30, 100, 300, 1000, and 3000). This dataset contains 546 training images, 82 validation images, and 82 test images. Images are resized to $6 4 0 \times 6 4 0$ , and training runs for 100 epochs with a batch size of 32. Although the overall mAP is reasonable, Grad-CAM reveals attention to irrelevant image boundaries (Figure 4), indicating that the model learns spurious correlations.

![](images/994526a1dbfe277c1c3e4da6790af598954c7615c42ed6a183f327c31abd7d1e.jpg)  
Figure 3: End-to-end system overview. The two input artifacts are independent and are matched through the borehole identifier and depth interval encoded in each filename. The core tray image provides all image pixels. Extracted core images pass to a gated UNet for crack segmentation. Post-processing converts the masks into defect spacing categories, while parallel branches estimate bedding angles and dominant lithological colors. The log report provides the reference labels. Its PDF text layer supplies defect spacing pseudo-labels for the self-supervised method in Section 4.1 and the evaluation references used in Section 5.

![](images/ccdb8fd97bbcf91aa7e31e3be82afaf176e6cf11f27f72998ea16d59780ede57.jpg)

![](images/87e709ab276be01c4e33307988d7155ed18e1d79fee90b16210146935bce5ed6.jpg)  
Figure 4: Supervised detection results. The YOLO training curves appear on the left. The Grad-CAM visualization on the right shows attention to image boundaries rather than crack regions.

Self-supervised learning. We developed this method before pixel-level annotations became available. At that stage, the log reports provided only defect-spacing categories, so we posed the task as classification over core cells and evaluated representation learning without annotated crack pixels. We report this classification branch as a standalone experiment; its filtering procedure is not used to generate the manual masks for the later segmentation experiments.

Embedding-based label filtering. The report-derived categories (Section 3.3) are noisy because the logging geologist assigns a category to a depth interval by visual assessment rather than by measuring every crack-to-crack distance. Aligning an interval-level category with individual 100-mm core cells introduces additional ambiguity. For each of the six spacing classes, we manually verified 100 cells drawn only from the training boreholes and used their DINO embeddings as a reference pool. Each remaining training cell was compared with this pool using cosine similarity. If the majority class among its top-5 neighbors disagreed with its report-derived label, the cell was marked as embedding-inconsistent and excluded from classifier training; it was not given a replacement label. This procedure removed 5,723 of 35,610 training cells (16%).

Architecture. We train a DINO ViT-Tiny encoder from scratch using only unlabeled cells from the training boreholes and freeze it before training the defect-spacing classifiers (Caron et al., 2021). Dividing each core image into 10 vertical segments of fixed physical width provides a large pool of unlabeled samples. The DINO encoder maps each sample to a 192-dimensional embedding, which passes to a lightweight fully connected classifier. DINO uses the symmetrized cross-entropy between student and teacher views:

$$
\mathcal { L } _ { \mathrm { D I N O } } = \frac { 1 } { 2 } \left[ \mathcal { H } ( t _ { 1 } , s _ { 2 } ) + \mathcal { H } ( t _ { 2 } , s _ { 1 } ) \right] ,\tag{1}
$$

where $\textstyle { \mathcal { H } } ( t , s )$ is the cross-entropy between the sharpened teacher output and the student prediction. The teacher logits are centered and sharpened before softmax. An exponential moving average updates the teacher weights.

A flat six-class classifier trained on the DINO embeddings reaches 68.6% overall accuracy. This aggregate result hides a clear error pattern. The model reliably identifies the coarse, well-separated classes, achieving 93.8% for 3000 and 78.1% for 1000. In contrast, it confuses the adjacent fine classes, achieving 46.9% for 30 and 43.8% for 100. The errors therefore arise from confusion between neighboring bins rather than uniform task dificulty. This finding motivates a decomposed decision process.

We therefore introduce a hierarchical inference pipeline (Algorithm 3) that divides the six-class decision into three binary stages. Each extracted core image spans approximately 1 m and is divided into ten ordered cells of approximately 100 mm. The first stage labels each cell as cracked or uncracked. Cracked cells pass through classifiers for the 10, 30, and 100 mm categories, whereas uncracked cells are provisionally assigned to the 300 mm category. We then scan the cells in spatial order. Cells are processed in continuous depth order, and an uncracked run at the end of one core-row image is continued into the next depth-adjacent core-row image. Runs of fewer than three consecutive uncracked cells retain the 300 mm category, runs of three to ten uncracked cells are assigned to the 1000 mm category, and runs of eleven or more uncracked cells are assigned to the open-ended 3000 mm category. The coarser categories are therefore inferred from the lower bound provided by the length of an uncracked run. Because the crack position within a 100- mm cell is unknown, this run-length procedure provides an approximate category rather than an exact crack-to-crack distance.

## Table 2

Self-supervised defect spacing classification with DINO embeddings. The flat classifier predicts all six classes in one step. The hierarchical pipeline divides the same decision into three binary stages, each evaluated on the subset received from the previous stage.
<table><tr><td>Stage</td><td>Accuracy</td><td>F1</td></tr><tr><td>Flat six-class classifier</td><td>0.686</td><td></td></tr><tr><td>Stage 1: cracked vs. uncracked</td><td>0.964</td><td>0.964</td></tr><tr><td>Stage 2: 10 vs. all</td><td>0.946</td><td>0.946</td></tr><tr><td>Stage 3: 30 vs. 100</td><td>0.801</td><td>0.805</td></tr></table>

These results show that self-supervised representations support defect spacing classification without pixel-level annotations. This capability is valuable when such annotations are unavailable, as they were during this stage of the work. Once annotations became available, however, we adopted segmentation as the primary method because defect spacing is the distance between successive cracks. It is therefore a geometric property of crack positions along the core axis.

Classification requires the network to infer this geometry implicitly and return only a defect spacing category. The result omits the crack positions, cannot represent multiple spacing regimes within one split, and is limited by the split width. Segmentation instead identifies the cracks and measures the intervals directly. A disputed defect spacing estimate can therefore be traced to the mask that produced it. Classification also inherits the category resolution of the log report, whereas segmentation measures a continuous distance before assigning a category. We retain the selfsupervised method as an annotation-free alternative for defect-spacing classification.

Algorithm 3 Hierarchical Spacing Inference   
Require: Feature tensor � for cells ordered continuously across depth-adjacent core-row images   
Ensure: Predicted labels ${ \hat { y } } ,$ confidences $p$   
1: $r _ { 1 } $ CrackedUncrackedModel(�)   
2: $( p _ { 1 } , y _ { 1 } ) \gets \mathrm { S o f t m a x } ( r _ { 1 } ) ,$ arg max $( r _ { 1 } )$   
3: if any $y _ { 1 } = 0$ then   
4: $\mathbf { f } _ { 1 }  \mathbf { f } [ y _ { 1 } = 0 ]$   
5: $r _ { 2 } \gets \mathrm { T e n V s A l l M o d e l } ( \mathbf { f } _ { 1 } )$   
6: $( p _ { 2 } , y _ { 2 } ) \gets \mathrm { S o f t m a x } ( r _ { 2 } )$ , arg max $( r _ { 2 } )$   
7: if any $y _ { 2 } = 1$ then   
8: $\mathbf { f } _ { 2 }  \mathbf { f } _ { 1 } [ y _ { 2 } = 1 ]$   
9: $r _ { 3 } \gets$ ThirtyVsHundredModel $( \mathbf { f } _ { 2 } )$   
10: $( p _ { 3 } , y _ { 3 } ) \gets \mathrm { S o f t m a x } ( r _ { 3 } )$ , arg max $( r _ { 3 } )$   
11: $\mathbf { M a p } y _ { 3 }$ labels as $0  3 0$ and $1 $ 100   
12: $y _ { 2 } [ y _ { 2 } = 1 ]  y _ { 3 } , p _ { 2 } [ y _ { 2 } = 1 ]  p _ { 3 }$   
13: $y _ { 2 } [ y _ { 2 } = 0 ]  1 0$   
14: $y _ { 1 } [ y _ { 1 } = 0 ]  y _ { 2 } , p _ { 1 } [ y _ { 1 } = 0 ]  p _ { 2 }$   
15: $y _ { 1 } [ y _ { 1 } = 1 ]  3 0 0$   
16: $\hat { y } \gets y _ { 1 }$   
17: Identify maximal runs  of consecutive labels $\hat { y } = 3 0 0$ in the continuous depth order   
18: for each run $R _ { j } \in \mathcal { R }$ with length $r _ { j }$ do   
19: if $r _ { j } \geq 1 1$ then   
20: $\hat { y } [ R _ { j } ] \gets 3 0 0 0$   
21: else if $\dot { r _ { j } } \ge 3$ then   
22: $\hat { y } [ R _ { j } ] \gets 1 0 0 0$   
23: return $( p _ { 1 } , \hat { y } )$

## 4.2. Crack Segmentation

We evaluate YOLOv11-Nano, YOLOv11-Large, Mask R-CNN, and CrackCLIP on the annotated dataset (Liang et al., 2025; Radford et al., 2021). We resize the core images to $6 4 0 \times 6 4 0$ and use the borehole-grouped training, validation, and test partitions in Table 1. Both YOLOv11 variants are trained for 100 epochs with a batch size of 16 and their default augmentations. We compare full-image training with a split-based variant in which each core image is divided into 10 sub-images of 400 ×500 pixels. This division expands the training set from 3,561 to 35,610 samples. The full-image model outperforms the split-based model in both mAP@50 (0.76 versus 0.57) and precision (0.74 versus 0.58). The split crops therefore lose context needed to distinguish cracks from surface artifacts. All subsequent segmentation experiments use complete core images. Mask R-CNN is trained for 100 epochs with a batch size of 8 and an initial learning rate of $1 \times 1 0 ^ { - 4 }$ . Its augmentations include horizontal flips and brightness shifts.

Figure 6 shows representative YOLOv11 predictions. The detector identifies most distinct cracks with high confidence. However, it misses low-contrast cracks in dark lithologies and divides some continuous cracks into several detections. These errors motivate the fusion decoder below.

We combine multiple sources of evidence to improve segmentation. The following sections first describe a multiencoder UNet baseline and then present the gated decoder that supersedes it.

## 4.3. Multi-Encoder UNet Baseline

This baseline assigns one encoder to each of three inputs: the RGB core image, the PiDiNet edge map, and the instance-segmentation mask. All three encoders use ImageNet-pretrained ResNet34 weights. For each single-channel input, we initialize the first convolution with the channel-wise mean of the pretrained RGB kernels. The model fuses the three streams at all five encoder scales. At each scale, it concatenates the feature maps and reduces them to one vector through global average pooling. A two-layer MLP (3� → �∕4 → 3) then produces three softmax weights for their linear combination. A 512 → 1024 bottleneck connects the encoders to five decoder blocks. Each block upsamples through transposed convolution, concatenates the fused skip connection at the corresponding scale, and applies a double convolution. A final transposed convolution and 3 × 3 convolution produce the crack probability map.

![](images/4870d8c442a5205bd94eca8fc7ee38b9bc3e55cf9251688969a4f3605f239ed6.jpg)

![](images/b9e9f755b33e972eb31478d35ba0cfe9f87f4feaa08e5d23c943bc82d5ef0627.jpg)  
Input Image

![](images/3011e0c93a88fd92c5b7b6c0686767dff7c8ad7115cfccd7b99a3c6fddab9c37.jpg)  
Attention  
Figure 5: Self-supervised defect-spacing classification. Each core cell is resized, divided into 16 × 16 patches, and encoded by a ViT-Tiny backbone trained without labels using only cells from the training boreholes. The frozen 192-dimensional embedding passes to a lightweight MLP trained on the retained report-derived labels after embedding-based filtering. The lower panel shows a representative attention map.

The gated decoder difers in the granularity of its fusion. Global pooling reduces the baseline’s fusion weights to three scalars per core image. The model can determine that edges are generally more reliable than masks for a particular image, but it applies this decision uniformly to every pixel. It cannot prefer the edge map in one region and the mask in another, even though the sources fail at diferent locations within a core image.

## 4.4. Spatially Gated U-Net Fusion

The model receives two binary single-channel inputs: a segmentation mask � from Mask R-CNN or CrackCLIP and an edge map � from PiDiNet. The inputs are concatenated and passed through a sequential attention gate. Its three convolutions reduce the channel dimension from 2 to 16, from 16 to 8, and from 8 to 1. Batch normalization and ReLU follow the first two convolutions, and a sigmoid produces the one-channel gated representation supplied to the U-Net (Figure 8). This learned representation permits the relative contribution of the two input sources to vary spatially.

The gated output passes to an initial 32-channel ConvBlock and then through three encoder Down blocks with 64, 128, and 256 output channels. A 256-channel ConvBlock forms the bottom stage. Four decoder Up blocks produce 128, 64, 32, and 32 channels and combine the decoder representation with encoder skip features. A final 1 × 1 convolution produces one crack-logit map at the input spatial resolution. We train separate gated models for the Mask R-CNN and CrackCLIP segmentation inputs. All input and reference masks are resized to 512 × 512 pixels. Training uses Adam with a learning rate of 10<sup>−4</sup>, a batch size of 32, and 100 epochs. Flips and rotations are applied jointly to the segmentation mask, edge map, and reference mask. The objective is the unweighted sum of binary cross-entropy and soft Dice losses. We retain the checkpoint selected on the validation partition.

![](images/ac8602b06b9d344b35ef9ab64b36194ced81737eb6e319017a13cd18f6eb8224.jpg)  
Figure 6: YOLOv11 instance segmentation on test core images. Bounding boxes, confidence scores, and blue masks show the predicted cracks. The model reliably detects high-contrast cracks in brown lithologies (top images). Predictions are sparser on dark gray core (bottom images), where the model misses several visible cracks.

![](images/30ffaef588a9eb0a297d5fa94b9e9fae90220ea93a82bcbcfd29e990a648c93c.jpg)  
Figure 7: Spatially gated U-Net architecture. The attention gate combines a PiDiNet edge map and a segmentation mask into a one-channel representation. The encoder uses an initial 32-channel ConvBlock followed by Down blocks with 64, 128, and 256 channels. The decoder uses Up blocks with 128, 64, 32, and 32 channels; dashed arrows denote encoder–decoder skip connections. A final convolution produces the crack-logit map.

The edge map, segmentation mask, and attention map can each be inspected independently. These intermediate outputs make the pipeline more interpretable than single-encoder alternatives.

## 4.5. Post-processing for Spacing

We convert each segmentation mask into crack segments and broader cracked regions and order them along the core axis. For segment �, let $l _ { i }$ and $r _ { i }$ denote its left (start) and right (end) bounds, respectively. The gap between successive non-overlapping segments $i - 1$ and � is

$$
d _ { i } = l _ { i } - r _ { i - 1 } .\tag{2}
$$

We map these distances to fixed defect spacing ranges (Table 3) to produce structured annotations for the core interval. The ranges correspond to the geotechnical categories used in the log reports and therefore allow direct comparison with the recorded annotations. Figure 9 shows representative outputs for three core images.

![](images/f558908f781312aa720b35dff4ef1821602055fbb8eb517228d4dba035c3e032.jpg)  
Figure 8: Components of the spatially gated U-Net. The attention gate maps the concatenated two-channel input through convolutional outputs of 16, 8, and 1 channel, using batch normalization and ReLU after the first two convolutions and sigmoid after the last. ConvBlock applies two convolution–normalization– activation stages. Down applies max pooling before a ConvBlock, whereas Up applies twofold upsampling before a ConvBlock.

![](images/f5116839f4694dc22f7ee5c73269582fa26c2194ec52b69ba1be89e3870a2500.jpg)  
Figure 9: Defect spacing recovered from three core images. Predicted crack segments are green, and broader cracked regions are red. The gaps between consecutive segments define the cyan interval boundaries and their defect spacing categories. Dense groups of cracks merge into cracked regions and receive the finest categories (e.g., 10), whereas intact intervals between isolated cracks receive coarser categories (e.g., 300 or 1000).

## 4.6. Bedding Angle Estimation

We estimate bedding angles by detecting line segments with LSD, removing near-horizontal artifacts, clustering collinear segments, and fitting line models with PCA. Each detected segment $\ell _ { i } = ( x _ { i 0 } , y _ { i 0 } , x _ { i 1 } , y _ { i 1 } )$ has orientation � = arctan $2 ( y _ { i 1 } - y _ { i 0 } , x _ { i 1 } - x _ { i 0 } )$ . We suppress near-horizontal segments and form collinear clusters subject to angular, perpendicular-distance, and gap constraints. PCA then determines the dominant direction of each cluster.

We project the clustered trace points onto a cylindrical core model (Figure 10). A line orientation measured only in the 2D image plane is insuficient to determine the 3D bedding-plane normal. Therefore, we re-wrap the detected trace onto the photographed half of the cylindrical surface before fitting a plane. The long image coordinate � follows the longitudinal core axis, which we denote by �, whereas � spans the visible diameter and parameterizes the photographed half-circumference. For image coordinates (�, �), the cylindrical projection is

![](images/c6ac5dd4bcb23e679c38d09a302a9b18532b4744681db91e774724337e37377a.jpg)

Figure 10: Geometry of bedding-angle estimation. (a) The long image coordinate � follows core length �, while � spans the visible diameter. (b) Each trace point is projected onto the photographed half of the cylindrical surface: � determines �, (�, �) span the circular cross-section, and the longitudinal axis � points out of the page. (c) PCA fits a plane to the projected 3D trace points. The acute angle between its normal � and the longitudinal core axis � is the reported bedding angle. Fragmented core violates the single-cylinder assumption and causes the principal failure mode shown in Figure 11.  
![](images/a106c09104023f3db5ff65c672d87fbf5f9edebee66bafeecd75c2795ed3a1a3.jpg)  
Figure 11: Bedding angle estimates for four core images. The upper two images show brown lithologies with predicted angles of 40–50<sup>◦</sup>. The lower two show dark gray core with steeper bedding angles of 50–60<sup>◦</sup>. Estimates are stable for intact core but degrade under fragmentation (lower right), which violates the cylindrical geometry assumption.

$$
y = { \frac { u } { W } } L , \qquad \phi = { \frac { v } { H } } \pi - { \frac { \pi } { 2 } } , \qquad x = R \cos \phi , \qquad z = R \sin \phi .\tag{3}
$$

Thus, � is the longitudinal core-axis coordinate and (�, �) span the circular cross-section. If the fitted plane normal is $\mathbf { n } = ( a , b , c ) \mathrm { i n } ( x , y , z )$ coordinates, we define the bedding angle as the acute normal-to-axis angle

$$
\theta _ { \mathrm { b e d d i n g } } = \arctan \left( { \frac { \sqrt { a ^ { 2 } + c ^ { 2 } } } { | b | } } \right) .\tag{4}
$$

A plane perpendicular to the core axis has a normal parallel to � and gives $\theta _ { \mathrm { b e d d i n g } } = 0 ^ { \circ }$ , whereas more oblique planes give larger angles. We report bedding angles in the 10<sup>◦</sup> categories used in core logging. The log-report references use the same definition and resolution, which permits the comparison in Table 5.

## 4.7. Lithological Color Detection

We divide each core image into vertical core splits and convert them to LAB color space. The mean color summarizes each split, and the nearest reference assigns its semantic color. A mask removes near-white regions to prevent core loss from biasing the result. We retain only colors that occur in consecutive splits and order the final descriptors by a fixed geological precedence. Let $\bar { \mathbf { c } } _ { k }$ denote the mean LAB color of split �, let $\mathbf { r } _ { i }$ denote reference color

![](images/80ecb11e822e4a488f1eb76886958e49f9d1bf4ff8ee29373dd14fb0317dc47e.jpg)  
Figure 12: Lithological color descriptors for five core images. The method returns one dominant color (“gray” or “brown”) or a composite descriptor when two colors persist across consecutive core splits (“gray and dark gray,” “brown and dark gray,” or “brown and dark brown”). The near-white mask suppresses core-loss markers in the second image and excludes them from aggregation.

�, and let $\gamma _ { i }$ denote its semantic label. We assign

$$
i _ { k } ^ { \star } = \arg \operatorname* { m i n } _ { i } \left\| \bar { \mathbf { c } } _ { k } - \mathbf { r } _ { i } \right\| _ { 2 } , \qquad \hat { \gamma } _ { k } = \gamma _ { i _ { k } ^ { \star } } .\tag{5}
$$

We discard core splits with a high proportion of white pixels and retain only colors found in at least two consecutive splits. This procedure stabilizes the labels under illumination variation, surface artifacts, and core loss. Figure 12 shows representative descriptors.

## 5. Experiments and Results

We evaluate all segmentation models on the same 763 borehole-held-out test images. If an upstream model produces no detection for an image, we retain that image and represent the missing prediction by an empty mask. Probability outputs are thresholded at 0.5. True-positive, false-positive, and false-negative pixels are then accumulated over the complete test set for the crack class. We report crack-class intersection over union,

$$
\mathrm { I o U } _ { \mathrm { c r a c k } } = \frac { T P } { T P + F P + F N } ,\tag{6}
$$

precision $T P / ( T P + F P )$ , recall $T P / ( T P + F N )$ , and $F _ { 1 } = 2 T P / ( 2 T P + F P + F N )$ . Thus, the reported metrics describe one fixed binary operating point computed from aggregate test-set pixel counts.

The figures show representative outputs, and Tables 4, 5, and 6 summarize the quantitative results. The main text focuses on the strongest segmentation models and uses the remaining baselines as supporting evidence. The Mask R-CNN gated-fusion configuration outperforms the detector-only baselines and produces the highest crack-mask result among the evaluated models. The self-supervised classifier remains useful when pixel annotations are unavailable, bu its decisions are less spatially transparent. We therefore treat it as a complementary method.

The gated U-Net with Mask R-CNN input achieves the highest crack-class IoU and F1. Its binary masks preserve many thin crack boundaries while suppressing surface artifacts in the representative examples. The masks are subsequently passed to the deterministic spacing procedure illustrated in Figure 9; this figure is qualitative and does not constitute a downstream spacing-accuracy evaluation.

Bedding angle and lithological color. Both estimators are training-free and apply fixed geometric or colorimetric rules. No data is used for fitting, so every core image remains available for evaluation. We evaluate each method on 1,200 core images sampled randomly from the corpus. The parser used for defect spacing also extracts the corresponding bedding angle and color references from the log reports (Section 3.3). The results in Tables 5 and 6 therefore measure agreement with the log report, not correctness against an independent geological measurement. The references are subject to the logging variability discussed in Section 1. These results indicate how closely the methods reproduce a geologist’s recorded observations rather than their absolute physical accuracy.

Bedding angle reconstruction agrees with the recorded angle category for 75.4% of the core images (Table 5). The estimated angles also follow the dominant structural trends visible in the core (Figure 11). Lithological color agrees with the log report for 84.7% of the core images (Table 6 and Figure 12). This higher agreement is expected because color is a low-frequency property of the complete image and remains stable under LAB-space averaging. Bedding angle estimation instead depends on line evidence that fragmentation can destroy. Most color errors occur under extreme illumination or extensive core loss, whereas most bedding angle errors occur in fragmented core that violates the cylindrical assumption.

## Table 3

Mapping of distance ranges to defect spacing labels.

<table><tr><td>Distance (x)</td><td>Label</td></tr><tr><td> $x \leq 1 0$ </td><td>10</td></tr><tr><td> $1 0 < x \leq 3 0$ </td><td>30</td></tr><tr><td> $3 0 < x \le 1 0 0$ </td><td>100</td></tr><tr><td> $1 0 0 < x \leq 3 0 0$ </td><td>300</td></tr><tr><td> $3 0 0 < x \leq 1 0 0 0$ </td><td>1000</td></tr><tr><td> $1 0 0 0 < x$ </td><td>3000</td></tr></table>

Table 4  
Quantitative comparison of crack-segmentation models. Crack IoU, precision, recall, and F1 are computed pixel-wise for the crack class from the final binary masks.
<table><tr><td>Model</td><td>Crack loU</td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td>YOLOv11 Nano</td><td>0.2615</td><td>0.2745</td><td>0.8465</td><td>0.4146</td></tr><tr><td>YOLOv11 Large</td><td>0.2746</td><td>0.2967</td><td>0.7867</td><td>0.4309</td></tr><tr><td>Mask R-CNN</td><td>0.2717</td><td>0.2796</td><td>0.9056</td><td>0.4273</td></tr><tr><td>CrackCLIP</td><td>0.3000</td><td>0.4300</td><td>0.5000</td><td>0.4600</td></tr><tr><td>UNet ResNet34 (3 Encoders)</td><td>0.6394</td><td>0.6884</td><td>0.8998</td><td>0.7801</td></tr><tr><td>UNet Gate Fusion (CrackCLIP)</td><td>0.2149</td><td>0.3440</td><td>0.3639</td><td>0.3537</td></tr><tr><td>UNet Gate Fusion (Mask R-CNN)</td><td>0.7539</td><td>0.8160</td><td>0.9083</td><td>0.8597</td></tr></table>

## Table 5

Bedding angle estimation results using LSD-based line detection with cylindrical PCA fitting.
<table><tr><td>Method</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>F1 Score</td><td>Test Size</td></tr><tr><td>LSD + Cylindrical PCA</td><td>0.7538</td><td>0.7684</td><td>0.7421</td><td>0.7550</td><td>1200</td></tr></table>

## Table 6

Lithological color detection results using LAB-space nearest-reference classification.
<table><tr><td>Method</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>F1 Score</td><td>Test Size</td></tr><tr><td>LAB Nearest-Reference</td><td>0.8467</td><td>0.8523</td><td>0.8416</td><td>0.8469</td><td>1200</td></tr></table>

## 6. Discussion

The experiments evaluate two complementary approaches to defect-spacing analysis. The weakly supervised branch predicts report-derived spacing categories without crack masks, whereas the supervised branch first localizes cracks and then derives spacing through deterministic post-processing. The gated segmentation configuration achieves the highest crack-mask performance among the evaluated models. Figure 9 qualitatively illustrates the spacing output obtained from its predicted masks. We did not quantitatively evaluate the resulting interval distances or spacing categories against spacing derived from manual masks or against independent geological measurements; downstream spacing accuracy therefore remains to be established.

## 6.1. Comparison with prior work

Domain diferences limit direct comparison with existing crack segmentation methods. Most published benchmarks use concrete, pavement, or masonry rather than borehole core images. DeepCrack reports F1 scores of 0.86–0.87 on curated pavement datasets (Liu et al., 2019). CrackSeg9k shows that cross-domain generalization typically reduces segmentation quality by 10–20 percentage points (Kulkarni et al., 2022). Our gated fusion model achieves an F1 score of 0.860 (Table 4), which is comparable to these infrastructure crack benchmarks. The result is notable because the core images contain irregular illumination, surface labels, and thin cracks across an extreme aspect ratio. We compare F1 rather than IoU because published IoU values are averaged over classes, whereas ours is crack-class only. These measures are not interchangeable. CrackCLIP (Liang et al., 2025), which was designed for weakly supervised crack segmentation, achieves a crack-class IoU of 0.30 on our data. This result suggests that prompt-based vision-language models require adaptation to geological textures.

## 6.2. Comparison of Segmentation Architectures

Table 4 compares the individual segmentation models with two learned fusion architectures. Mask R-CNN alone achieves high recall (0.906) but low precision (0.280), resulting in a crack-class IoU of 0.272. The three-encoder U-Net described in Section 4.3, which combines the RGB image, PiDiNet edge map, and Mask R-CNN mask using imagelevel source weights, achieves an IoU of 0.639. The gated U-Net using PiDiNet and Mask R-CNN inputs achieves the highest observed result, with an IoU of 0.754, precision of 0.816, recall of 0.908, and F1 of 0.860.

The two fusion architectures difer in their inputs, encoder initialization, depth, capacity, and fusion mechanism. Their performance diference therefore does not isolate the causal efect of spatial gating. The comparison shows that the evaluated gated configuration performs best under the present protocol; a controlled ablation with matched inputs and capacity would be required to attribute the diference specifically to the gate.

## 6.3. When fusion degrades its input

The gate is not universally beneficial. Replacing Mask R-CNN with CrackCLIP as the mask source reduces crackclass IoU to 0.215 (Table 4). This result is lower than the 0.300 achieved by CrackCLIP alone, so fusion degrades its own input in this setting.

The attention gate learns a spatially varying representation from the segmentation and edge inputs; it does not explicitly encode agreement between them. With Mask R-CNN, the high-recall segmentation mask and the PiDiNet edge map provide complementary evidence, and the fused model raises precision to 0.816 while retaining a recall of 0.908. With CrackCLIP, recall falls from 0.500 for the standalone model to 0.364 after fusion. This result shows that the learned fusion configuration is sensitive to the quality and sparsity of its input masks. It does not, by itself, establish the internal gate values or an agreement-based failure mechanism; saved gate maps would be needed for that analysis.

## 6.4. Failure modes

Three recurring patterns limit crack segmentation. First, the model often misses hairline cracks narrower than three pixels. These cracks are under-represented in the annotations and can run parallel to surface scratches. Second, cracks separated by fewer than 5–10 pixels can merge into one predicted region, collapsing distinct spacing intervals. Third, dark brown or black lithologies can provide weak PiDiNet responses for low-contrast cracks, causing false negatives at the gate input. These segmentation errors can propagate into the number and length of intervals produced by post-processing, but that propagation is not quantified in the present study. Bedding angle estimation also fails when core is missing or heavily fragmented because these conditions violate the cylindrical geometry assumption. Future work should address these limitations through multi-scale annotation, contrast-adaptive preprocessing, and learned geometric priors.

## 6.5. Limitations

Three limitations bound how the reported numbers should be read.

First, the reference is the log report, not the rock. Every reference label comes from observations entered by a geologist into the logging software. This limitation applies to defect spacing, bedding angle, weathering, and lithological color. The reported accuracy therefore measures agreement with a geologist’s recorded judgment. As noted in the Introduction, this judgment has substantial inter-observer variability (Alejano, 2025). That variability is part of the ground truth and imposes an upper bound that cannot be quantified without independently re-logging the same core. Even perfect agreement would reproduce one geologist’s observations rather than independently measure the rock. The limitation does not invalidate relative model comparisons because all models use the same reference. However, the 75.4% agreement for bedding-angle categories should not be interpreted as independently measured physical accuracy. The segmentation reference also treats label-touching cracks as background, so the reported pixel metrics include this source of annotation noise.

Second, the corpus is narrow. All data are from Australia, cover a small number of rock types, and follow one image-acquisition protocol. Several pipeline components depend on this protocol. Core image extraction detects a particular yellow ruler, defect spacing extraction assumes one log report template, and the color references are calibrated to the observed lithologies. The methods are not inherently site-specific, but the reported results characterize only this corpus. Generalization to other operators, log report templates, acquisition protocols, and rock types remains untested.

Third, the learned-model results are single-run estimates. We do not report variation across random seeds or confidence intervals. The model rankings therefore describe the observed runs under the fixed borehole-grouped split; repeated training would be required to quantify their stability.

## 7. Conclusion

We presented a hybrid framework for borehole core analysis that uses report-derived weak labels for defectspacing classification and manually annotated masks for fully supervised crack segmentation. Among the evaluated segmentation configurations, the gated U-Net using PiDiNet and Mask R-CNN inputs achieved the highest crack-class IoU and F1. Deterministic post-processing converts the predicted crack geometry into spacing intervals, while separate rule-based procedures produce core-relative bedding-angle and lithological color descriptors. Future work will evaluate more diverse acquisition conditions and validate the outputs against independent geological measurements.

## Data availability

The borehole log reports and derived core images analyzed in this study are proprietary and cannot be released. The source code developed for this work is also not publicly available.

## References

Alejano, L.R., 2025. Rock mass classification systems: A useful rock mechanics tool, often misused. Rock Mechanics and Rock Engineering 58, 11147–11167.

Alzubaidi, F., Makuluni, P., Clark, S.R., Lie, J.E., Mostaghimi, P., Armstrong, R.T., 2022a. Automatic fracture detection and characterization from unwrapped drill-core images using mask r–cnn. Journal of Petroleum Science and Engineering 208, 109471.

Alzubaidi, F., Mostaghimi, P., Si, G., Swietojanski, P., Armstrong, R.T., 2019. Automated rock quality designation (rqd) estimation from digital images of drill cores using convolutional neural networks, in: Proceedings Future Mining 2019, Australasian Institute of Mining and Metallurgy. p. 57.

Alzubaidi, F., Mostaghimi, P., Si, G., Swietojanski, P., Armstrong, R.T., 2022b. Automated rock quality designation using convolutional neural networks. Rock mechanics and rock engineering 55, 3719–3734.

Byun, H., Kim, J., Yoon, D., Kang, I.S., Song, J.J., 2021. A deep convolutional neural network for rock fracture image segmentation. Earth science informatics 14, 1937–1951.

Caputo, A., Calcagni, M.T., Salerno, G., Mammoliti, E., Castellini, P., 2025. Measurement of fracture networks in rock sample by x-ray tomography, convolutional filtering and deep learning. Sensors 25, 4409.

Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., Joulin, A., 2021. Emerging properties in self-supervised vision transformers, in: 2021 IEEE/CVF international conference on computer vision (ICCV), IEEE. pp. 9630–9640.

Dais, D., Bal, I.E., Smyrou, E., Sarhosis, V., 2021. Automatic crack classification and segmentation on masonry surfaces using convolutional neural networks and transfer learning. Automation in Construction 125, 103606.

Gu, G., Ko, B., Go, S., Lee, S.H., Lee, J., Shin, M., 2022. Towards light-weight and real-time line segment detection, in: Proceedings of the AAAI Conference on Artificial Intelligence, pp. 726–734.

Hamishebahar, Y., Guan, H., So, S., Jo, J., 2022. A comprehensive review of deep learning-based crack detection approaches. Applied Sciences 12, 1374.

He, C., Sadeghpour, H., Shi, Y., Mishra, B., Roshankhah, S., 2024. Mapping distribution of fractures and minerals in rock samples using res-vgg-unet and threshold segmentation methods. Computers and Geotechnics 175, 106675.

Ji, Y., Song, S., Zhang, W., Li, Y., Xue, J., Chen, J., 2025. Automatic identification of rock fractures based on deep learning. Engineering Geology 345, 107874.

Joshi, D., Singh, T.P., Sharma, G., 2022. Automatic surface crack detection using segmentation-based deep-learning approach. Engineering Fracture Mechanics 268, 108467.

Kulkarni, S., Singh, S., Balakrishnan, D., Sharma, S., Devunuri, S., Korlapati, S.C.R., 2022. Crackseg9k: A collection and benchmark for crack segmentation datasets and frameworks, in: European conference on computer vision, Springer. pp. 179–195.

Liang, F., Li, Q., Yu, H., Wang, W., 2025. Crackclip: Adapting vision-language models for weakly supervised crack segmentation. Entropy 27, 127.

Liu, Y., Yao, J., Lu, X., Xie, R., Li, L., 2019. Deepcrack: A deep hierarchical feature learning architecture for crack segmentation. Neurocomputing 338, 139–153.

Liu, Y., Zhu, W., Liu, X., Wang, J., Chen, C., 2025. Core fracture identification and dip angle calculation using a deep learning model. Rock Mechanics and Rock Engineering 58, 1327–1345.

Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al., 2021. Learning transferable visual models from natural language supervision, in: International conference on machine learning, PmLR. pp. 8748–8763.

Reinhardt, M., Jacob, A., Sadeghnejad, S., Cappuccio, F., Arnold, P., Frank, S., Enzmann, F., Kersten, M., 2022. Benchmarking conventional and machine learning segmentation techniques for digital rock physics analysis of fractured rocks. Environmental Earth Sciences 81, 71.

Saberironaghi, A., Ren, J., 2024. Depthcracknet: A deep learning model for automatic pavement crack detection. Journal of imaging 10, 100.

Shi, Y., Cui, L., Qi, Z., Meng, F., Chen, Z., 2016. Automatic road crack detection using random structured forests. IEEE transactions on intelligent transportation systems 17, 3434–3445.

Song, Q., Yao, W., Tian, H., Guo, Y., Muniyandi, R.C., An, Y., 2024. Two-stage framework with improved u-net based on self-supervised contrastive learning for pavement crack segmentation. Expert Systems with Applications 238, 122406.

Su, Z., Liu, W., Yu, Z., Hu, D., Liao, Q., Tian, Q., Pietikäinen, M., Liu, L., 2021. Pixel diference networks for eficient edge detection, in: 2021 IEEE/CVF International Conference on Computer Vision (ICCV), IEEE. pp. 5097–5107.

Yang, F., Zhang, L., Yu, S., Prokhorov, D., Mei, X., Ling, H., 2019. Feature pyramid and hierarchical boosting network for pavement crack detection. IEEE transactions on intelligent transportation systems 21, 1525–1535.

Zhang, L., Yang, F., Zhang, Y.D., Zhu, Y., 2016. Road crack detection using deep convolutional neural network, in: 2016 IEEE Internationa Conference on Image Processing (ICIP), IEEE. pp. 3708–3712.

Zhang, S., Xu, H., Zhu, X., Xie, L., 2024. Automatic crack detection using weakly supervised semantic segmentation network and mixed-label training strategy. Foundations of Computing and Decision Sciences 49, 95–118.

Zhou, Y., Chang, D., Zheng, J., Zhu, D., Nie, X., 2023. A fast workflow for automatically extracting the apparent attitude of fractures in 3-d digita core images. Processes 11, 2517.

Zou, Q., Cao, Y., Li, Q., Mao, Q., Wang, S., 2012. Cracktree: Automatic crack detection from pavement images. Pattern Recognition Letters 33, 227–238.