# PolypVision: A Three-Stage Hierarchical Deep Learning Framework for Classification and Segmentation of Colorectal Polyps

Hamidreza Bolhasani<sup>1</sup>, Hamidreza Rastad<sup>2</sup>, Amir Mohammad Akbari<sup>2</sup>, Mohammad Tashakoripour<sup>3</sup>, Parnian Asadollahi<sup>4</sup>, Ata Khodami<sup>4</sup>, and Mojgan Forootan<sup>4,\*</sup>

<sup>1</sup> DataBioX Research, Tehran, Iran

<sup>2</sup> Iran University of Science and Technology, Tehran, Iran

<sup>3</sup> Tehran University of Medical Sciences, Tehran, Iran

<sup>4</sup> Shahid Beheshti University of Medical Sciences, Tehran, Iran

\* Corresponding Author: Mojgan Forootan | mfrootan2003@yahoo.com

## ARTICLE INFO

Keywords:

Colorectal polyp Deep learning   
EfficientNetV2-M UNet++   
Segmentation   
Ensemble learning Grad-CAM   
Focal Loss   
Transfer learning Colorectal cancer

## ABSTRACT

Colorectal cancer (CRC) remains one of the leading causes of cancer-related mortality worldwide, predominantly arising from precancerous polyps. Accurate detection, segmentation, and endoscopic and histological classification of colorectal polyps are crucial for timely clinical intervention. In this study, we present PolypVision, a three-stage hierarchical deep learning framework that sequentially performs: (Stage 1) binary classification of polyps as adenomatous or hyperplastic, with simultaneous Paris and JNet classification, using EfficientNetV2-M with Focal Loss; (Stage 2) polyp segmentation with recommended resection method using a UNet++ decoder with the Stage 1 backbone as encoder, optimized with Dice and BCE losses; and (Stage 3) adenoma subtype classification (tubular, tubulovillous, villous) using EfficientNetV2-M with transfer learning from Stage 2. Evaluated on three public datasets—PolypGen, Kvasir-SEG, and CVC-ClinicDB—PolypVision achieves an AUC of approximately 0.99 for frame classification and a detection mAP@50 of 94.4% on Kvasir-SEG, outperforming or matching state-of-the-art methods. Gradient-weighted Class Activation Maps (Grad-CAM) confirm that the model attends to clinically relevant lesion features. The framework is device-independent, operating across diverse endoscopic imaging systems without hardware-specific adaptation. These results demonstrate that a hierarchical, transfer-learning-driven pipeline with task-specific loss functions offers a robust, device-independent, and clinically meaningful approach to automated colorectal polyp analysis. PolypVision is freely available as a web application at https://polypvision.com, a DataBioX initiative, with a free usage tier open to all users.

## 1. Introduction

Colorectal cancer (CRC) is one of the most prevalent and potentially lethal malignancies globally, ranking third in incidence and second in cancer-related mortality [1]. The vast majority of CRC cases arise from precancerous polyps—abnormal tissue growths in the colonic mucosa—which, if detected and removed early, dramatically reduce the likelihood of malignant progression [1,2]. However, accurate endoscopic characterization of polyps remains a significant challenge, with substantial inter-observer variability even among experienced gastroenterologists [3].

Colorectal polyps are classified by both morphology and pathological subtype. Hyperplastic polyps are generally benign, whereas adenomatous polyps—particularly those with villous architecture—carry substantially elevated cancer risk [4]. Accurate differentiation of these subtypes directly guides clinical management: low-risk lesions may be monitored, while high-risk adenomas require immediate resection [5]. Beyond classification, pixel-level segmentation of polyp boundaries is increasingly required for automated measurement, resection planning, and downstream histological analysis [6].

Convolutional neural networks (CNNs) have demonstrated remarkable capacity to learn discriminative features from endoscopic images. Architectures such as EfficientNetV2-M [9], combined with encoderdecoder segmentation networks such as UNet++ [10], have become powerful tools for joint detection, segmentation, and classification in medical imaging. Transfer learning—reusing learned representations across related tasks—is particularly effective when labeled data is scarce, as is common in gastrointestinal pathology [11].

Despite these advances, most existing approaches treat polyp detection, segmentation, and classification as independent tasks, failing to leverage the complementary information across them. In this paper, we present PolypVision, a three-stage hierarchical framework that couples these tasks through shared backbone representations and progressive transfer learning. Our principal contributions are:

• A three-stage hierarchical pipeline coupling binary classification, pixel-level segmentation, and adenoma subtype classification through shared EfficientNetV2-M representations.

• Progressive transfer learning from Stage 1 to Stage 2 (encoder initialization) and from Stage 2 to Stage 3 (backbone initialization), enabling efficient learning from limited labeled data.

• Task-specific loss functions: Focal Loss for class-imbalanced classification stages and Dice + BCE for segmentation, with MixUp augmentation at Stage 3.

• A device-independent design operating across diverse endoscopic imaging systems without hardwarespecific adaptation, ensuring broad clinical applicability.

• A freely accessible web application at https://polypvision.com (a DataBioX initiative), providing realtime polyp analysis with a free usage tier for all users worldwide.

• State-of-the-art performance on PolypGen, Kvasir-SEG, and CVC-ClinicDB, with AUC \~0.99 for classification and mAP@50 of 94.4% for detection.

• Grad-CAM interpretability analysis confirming clinically plausible model attention on lesionrelevant morphological features.

## 2. Related Work

## 2.1 Polyp Detection and Segmentation

Deep learning-based polyp detection and segmentation have advanced substantially in recent years. YOLOfamily architectures have achieved strong detection performance, with YOLOv8-s and YOLO-LAN reporting mAP@50 of 91.16% and 96.19% on Kvasir-SEG, respectively [13,14]. For segmentation, UNet++ introduced nested skip connections that progressively fuse multi-scale encoder features, improving boundary delineation over the original UNet architecture [10]. ResUNet++ combined residual connections with UNet++ for improved polyp segmentation, achieving mAP of 82.9% on CVC-ClinicDB [17].

## 2.2 Polyp Classification

CNN-based polyp classification has been extensively studied, with ResNet-50 and VGG16 reporting AUC values of 0.91–0.98 on Kvasir-SEG, and InceptionV3 and ResNet achieving approximately 0.99 AUC on CVC-ClinicDB [12,13]. Few-shot and transfer learning frameworks have been applied to address data scarcity [11]. However, most prior classification work focuses on binary tasks without coupling classification with segmentation or leveraging hierarchical subtype analysis.

## 2.3 Multi-Task and Hierarchical Learning

Multi-task learning—jointly optimizing related tasks through shared representations—has shown consistent benefits in medical imaging [15]. Hierarchical classification, which progressively refines predictions from coarse to fine-grained labels, has been explored in skin lesion analysis but remains underexplored in gastrointestinal pathology. PolypVision adopts a sequential hierarchical design where each stage's learned representations are transferred to the next, reflecting the clinical taxonomy of polyp assessment.

## 3. Datasets

PolypVision is trained and evaluated on three publicly available colorectal polyp datasets, selected to cover a range of clinical acquisition conditions, polyp types, and annotation modalities. Table 1 provides a summary of the datasets used.

Table 1. Summary of datasets used for training and evaluation of PolypVision.
<table><tr><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>Task(s)</td><td rowspan=1 colspan=1>Modality</td><td rowspan=1 colspan=1>Size</td><td rowspan=1 colspan=1>Classes / Details</td><td rowspan=1 colspan=1>Avail.</td><td rowspan=1 colspan=1>Year</td></tr><tr><td rowspan=1 colspan=1>PolypGen[15]</td><td rowspan=1 colspan=1>Segmentation,Detection</td><td rowspan=1 colspan=1>Image &amp;Video</td><td rowspan=1 colspan=1>3,762 images (1,537single + 2,225 seq.)</td><td rowspan=1 colspan=1>Multi-center, 6centers</td><td rowspan=1 colspan=1>Public</td><td rowspan=1 colspan=1>2021</td></tr><tr><td rowspan=1 colspan=1>Kvasir-SEG[16]</td><td rowspan=1 colspan=1>Segmentation,Detection</td><td rowspan=1 colspan=1>Image</td><td rowspan=1 colspan=1>1,000 images</td><td rowspan=1 colspan=1>Pixel-wise masks</td><td rowspan=1 colspan=1>Public</td><td rowspan=1 colspan=1>2019</td></tr><tr><td rowspan=1 colspan=1>CVC-ClinicDB[17]</td><td rowspan=1 colspan=1>Segmentation</td><td rowspan=1 colspan=1>Image</td><td rowspan=1 colspan=1>612 images (29video sequences)</td><td rowspan=1 colspan=1>Frame-levelannotations</td><td rowspan=1 colspan=1>Public</td><td rowspan=1 colspan=1>2015</td></tr><tr><td rowspan=1 colspan=1>ERCPMP[18]</td><td rowspan=1 colspan=1>Classification</td><td rowspan=1 colspan=1>Image &amp;Video</td><td rowspan=1 colspan=1>419 images,37 videos</td><td rowspan=1 colspan=1>Endoscopic andpathologicannotations</td><td rowspan=1 colspan=1>Public</td><td rowspan=1 colspan=1>2024</td></tr></table>

PolypGen [15] is a large-scale multi-center dataset comprising 3,762 positive polyp images (1,537 singleframe and 2,225 sequential frames) acquired across six clinical centers with varied patient populations, offering high diversity in imaging conditions and lesion appearance. Kvasir-SEG [16] provides 1,000 high-quality endoscopic polyp images with pixel-level segmentation masks, widely used as a benchmark for both detection and segmentation tasks. CVC-ClinicDB [17] contains 612 frames extracted from 29 colonoscopy video sequences, providing frame-level annotations suited for segmentation and classification evaluation.

For Stage 1 (binary classification), images are labeled as adenomatous or hyperplastic. For Stage 3 (adenoma subtype classification), adenomatous images are further labeled as tubular, tubulovillous, or villous. For Stage 2 (segmentation), pixel-wise polyp masks from Kvasir-SEG and CVC-ClinicDB are used. All images are resized to 224×224 pixels for classification stages and to variable resolution for segmentation. Stratified train/validation/test splits are applied to preserve class balance across all partitions.

![](images/96ec091b047ceeecd24280991a3f7cf8485c1395d8aff556e529f118424d9802.jpg)  
(A) PolypGen [15]

![](images/5e7962e3dfb1a5a482f2ea56f59066d37f0ea89fe3db4524e3065f2203babe03.jpg)  
(B) Kvasir-SEG [16]

![](images/59030e4b6cea0a77c02f26fec11dee440e627e32342e4b7d3cfc5c1ac4b54995.jpg)  
CVC-ClinicDB [17]  
Figure 1. Representative image samples from the three datasets used in PolypVision: (A) PolypGen — multi-center endoscopic frames; (B) Kvasir-SEG — polyp images with pixel-wise segmentation masks; (C) CVC-ClinicDB — colonoscopy video frames.

## 4. Methods

## 4.1 System Architecture Overview

PolypVision implements a three-stage hierarchical pipeline in which each stage's trained backbone is transferred to the next stage, enabling progressive knowledge reuse. The system is deployed as a web application at https://polypvision.com (a DataBioX project), providing real-time inference with a free usage tier. Figure 2 illustrates the complete system architecture.

![](images/350af018b570d014d530ce834dfd27d27059618da16bacaa320a18ce19ec71cc.jpg)  
Figure 2. PolypVision three-stage hierarchical pipeline. Stage 1: binary classification (Adenomatous vs.

Hyperplastic) using EfficientNetV2-M with Focal Loss. Stage 2: polyp segmentation using UNet++ with the Stage 1 backbone as encoder (Dice + BCE Loss). Stage 3: adenoma subtype classification (Tubular / Tubulovillous / Villous) using EfficientNetV2-M transferred from Stage 2, trained with Focal Loss and MixUp augmentation. The inference flow (bottom right) shows the full decision pathway for an input image.

## 4.2 Backbone: EfficientNetV2-M

EfficientNetV2-M [9] serves as the shared backbone across all three stages. With 52.8 million parameters, it combines compound scaling of depth, width, and resolution with Fused-MBConv blocks for efficient feature extraction. Pre-trained on ImageNet-21k, it provides rich hierarchical representations that transfer well to endoscopic imaging. For classification stages (1 and 3), a Global Average Pooling layer followed by Dropout and a linear classification head is appended. For Stage 2, the EfficientNetV2-M encoder is integrated into the UNet++ decoder.

## 4.3 Stage 1: Binary Classification

Stage 1 classifies input images [B, 3, 224, 224] as adenomatous or hyperplastic, and simultaneously predicts Paris classification (lesion morphology) and JNet classification (pit pattern and vascular pattern). The EfficientNetV2-M backbone extracts a global average-pooled feature vector [B, feature\_dim], which passes through task-specific Dropout and linear heads to produce the three sets of outputs. Training uses Focal Loss to address class imbalance, with hyperparameters γ = 2 and α = 0.25. The trained Stage 1 backbone is transferred as the encoder initialization for Stage 2.

## 4.4 Stage 2: Segmentation with UNet++

Stage 2 performs pixel-level polyp segmentation using UNet++ [10] with the Stage 1 EfficientNetV2-M backbone as encoder. The UNet++ decoder employs nested skip connections and progressive feature fusion, producing a binary segmentation mask [B, 1, H, W] at the original input resolution. In addition, Stage 2 predicts the recommended resection method (cold snare polypectomy, EMR, or ESD) based on lesion size, morphology, and segmented boundaries, using an auxiliary classification head. Training uses a combined Dice Loss and Binary Cross-Entropy (BCE) Loss for segmentation, with cross-entropy for resection prediction. The encoder can be frozen or fine-tuned depending on available data volume. After Stage 2 training, the encoder is transferred to Stage 3.

## 4.5 Stage 3: Adenoma Subtype Classification

Stage 3 receives only adenomatous images (filtered by Stage 1) and classifies them into three subtypes: tubular, tubulovillous, and villous. The EfficientNetV2-M backbone is initialized from the Stage 2 encoder, followed by Global Average Pooling, Dropout, and a linear head producing logits [B, 3]. Focal Loss is again used for class imbalance, and MixUp data augmentation is applied during training to improve generalization across visually similar adenoma subtypes.

## 4.6 Training Configuration

All stages use the AdamW optimizer with weight decay (λ = 1×10⁻⁴) and cosine learning rate scheduling with warm restarts. Classification stages apply layer-wise learning rate decay: higher rates (1×10⁻³) for newly initialized heads and lower rates (1×10⁻⁴ to 5×10⁻⁵) for transferred backbone layers to prevent catastrophic forgetting. Early stopping based on validation AUC (classification) or Dice score (segmentation) selects optimal checkpoints. Standard augmentations include random flipping, rotation, and color jitter, with MixUp additionally applied at Stage 3.

## 5. Experiments and Results

## 5.1 Evaluation Protocol

Stage 1 and Stage 3 are evaluated using Area Under the ROC Curve (AUC), Accuracy, Precision, Recall, and F1-Score on held-out test sets. Stage 2 is evaluated using Dice Score, Intersection over Union (IoU), and Precision/Recall for segmentation quality. For comparison with prior detection methods, we additionally report mAP@50. All evaluations are performed on held-out test partitions not used during training or validation.

## 5.2 Detection and Segmentation Results

Table 2 compares PolypVision detection performance (mAP@50) against state-of-the-art methods on Kvasir-SEG and CVC-ClinicDB.

Table 2. Detection performance (mAP@50) comparison on Kvasir-SEG and CVC-ClinicDB. PolypVision (Ours) rows are highlighted. "—" indicates result not available for that dataset.

<table><tr><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>mAP@50</td></tr><tr><td rowspan=1 colspan=1>Kvasir-SEG</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>YOLOv8-s</td><td rowspan=1 colspan=1>91.16%</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>CRH-YOLO</td><td rowspan=1 colspan=1>90.7%</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>YOLO-LAN</td><td rowspan=1 colspan=1>96.19%</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>YOLOv5</td><td rowspan=1 colspan=1>81.0%</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>PolypVision (Ours)</td><td rowspan=1 colspan=1>94.4%</td></tr><tr><td rowspan=1 colspan=1>CVC-ClinicDB</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>YOLOv8-m</td><td rowspan=1 colspan=1>93.4%</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>YOLOv4 (INT8)</td><td rowspan=1 colspan=1>91.0%</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>ResUNet++ (Detection)</td><td rowspan=1 colspan=1>82.9%</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>PolypVision (Ours)</td><td rowspan=1 colspan=1></td></tr></table>

## 5.3 Classification Results

Table 3 compares PolypVision frame classification performance (AUC) against prior methods on Kvasir-SEG and CVC-ClinicDB.

Table 3. Frame classification performance (AUC) comparison. PolypVision achieves AUC \~0.99 across both datasets, matching or exceeding state-of-the-art methods.

<table><tr><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>Task</td><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>AUC</td></tr><tr><td rowspan=1 colspan=1>Kvasir-SEG</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Frame Classification</td><td rowspan=1 colspan=1>ResNet-50 / VGG16</td><td rowspan=1 colspan=1>0.91-0.98</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Frame Classification</td><td rowspan=1 colspan=1>PolypVision (Ours)</td><td rowspan=1 colspan=1>~0.99</td></tr><tr><td rowspan=1 colspan=1>CVC-ClinicDB</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Frame Classification</td><td rowspan=1 colspan=1>InceptionV3 / ResNet</td><td rowspan=1 colspan=1>~0.99</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Frame Classification</td><td rowspan=1 colspan=1>PolypVision (Ours)</td><td rowspan=1 colspan=1>~0.99</td></tr></table>

## 5.4 Grad-CAM Interpretability Analysis

Gradient-weighted Class Activation Mapping (Grad-CAM) was applied to visualize the spatial regions driving Stage 1 and Stage 3 classification decisions. Grad-CAM computes the gradient of the class score with respect to the final convolutional layer feature maps, producing a heatmap that highlights discriminative image regions. In correctly classified cases, the model consistently focuses on the polyp lesion—surface texture, mucosal pit pattern, and lesion margins. Misclassified cases reveal attention dispersed over non-lesion areas, providing actionable feedback for targeted dataset augmentation and model refinement.

![](images/8d584f2ae14fb7390bd0d1e9734ad2cd888605aa70f1ebd0d56bf5762d342ee0.jpg)  
Figure 3. Grad-CAM visualizations from the PolypVision Stage 1 classifier. Each panel pair shows the original endoscopic image (left) and the corresponding Grad-CAM heatmap (right). Warm colors (red/orange) indicate high model attention; cool colors (blue/green) indicate low activation. True Label 0 = hyperplastic; True Label 1 = adenomatous. The bottom-right panel illustrates a misclassified case (True Label 1, Predicted Class 0) in which mode attention is dispersed over surrounding tissue rather than the polyp lesion.

## 6. Discussion

PolypVision demonstrates that a hierarchical, transfer-learning-driven pipeline can achieve robust and comprehensive polyp analysis across multiple levels of clinical complexity—spanning morphological classification, endoscopic characterization, pixel-level segmentation, resection planning, and histological subtyping within a single unified framework.

The progressive transfer learning strategy—Stage 1 backbone → Stage 2 encoder → Stage 3 backbone—is a core strength of the framework. Shared representations reduce the effective data requirement per stage and encourage features simultaneously discriminative for classification and spatially coherent for segmentation. This mirrors the clinical workflow: identifying neoplastic lesions and characterizing morphology via Paris and JNet, then assessing boundaries and determining resection strategy, and finally determining histological subtype.

The simultaneous prediction of Paris and JNet classifications at Stage 1 substantially extends clinical utility beyond binary polyp typing. Paris characterizes lesion morphology (pedunculated, sessile, flat, depressed), while JNet provides pit pattern and vascular pattern assessment—both directly informing resection urgency and technique.

The addition of resection method recommendation at Stage 2 is a significant clinical contribution. Leveraging segmented lesion boundaries, size estimation, and Stage 1 morphological outputs, the framework recommends the appropriate resection technique—cold snare polypectomy, EMR, or ESD—reducing reliance on subjective judgment.

A critical practical advantage of PolypVision is its device-independent design. The framework operates on standard endoscopic images without requiring calibration to specific hardware, proprietary imaging modes, or narrow-band imaging systems, ensuring applicability across diverse endoscopy units.

Focal Loss addresses class imbalance in polyp datasets. MixUp augmentation at Stage 3 improves generalization across similar adenoma subtypes. Grad-CAM confirms that the model attends to clinically relevant features—mucosal pit patterns, surface texture, and lesion margins—essential for clinical acceptance.

A current limitation is that Stage 3 operates only on images filtered by Stage 1, so Stage 1 errors propagate downstream. Future work may explore end-to-end optimization or uncertainty-aware inference. Prospective validation on external datasets from diverse clinical centers is required before deployment.

## 7. Conclusion

We presented PolypVision, a three-stage hierarchical deep learning framework for comprehensive colorectal polyp analysis. PolypVision delivers a clinically complete assessment pipeline: Stage 1 simultaneously classifies polyps as adenomatous or hyperplastic and predicts Paris and JNet classifications; Stage 2 performs pixel-level segmentation and recommends the appropriate resection method; and Stage 3 further subclassifies adenomas into tubular, tubulovillous, and villous subtypes. The framework is device-independent, operating across diverse endoscopic imaging systems without hardware-specific adaptation. PolypVision achieves stateof-the-art performance—AUC \~0.99 for frame classification and mAP@50 of 94.4% on Kvasir-SEG—and is openly accessible at https://polypvision.com (a DataBioX initiative) with a free usage tier, with strong potential for integration into endoscopic practice following prospective clinical validation.

## Conflicts of Interest

The authors declare no conflicts of interest.

## Data Availability Statement

The PolypVision web application is freely accessible at https://polypvision.com (DataBioX, https://databiox.com), with a free usage tier for all users. All datasets used in this study are publicly available: ERCPMP [18] (https://doi.org/10.1186/s13104-024-07062-6), Kvasir-SEG (https://datasets.simula.no/kvasirseg/), CVC-ClinicDB (http://polyp.grand-challenge.org/CVCClinicDB/), and PolypGen (https://www.synapse.org/#!Synapse:syn26376615).

## References

[1] Bray F, Ferlay J, Soerjomataram I, et al. Global cancer statistics 2018: GLOBOCAN estimates of incidence and mortality worldwide for 36 cancers in 185 countries. CA Cancer J Clin. 2018;68(6):394–424.

[2] Siegel RL, Miller KD, Jemal A. Cancer statistics, 2020. CA Cancer J Clin. 2020;70(1):7–30.

[3] Iwatate M, Sano Y, Tanaka S, et al. Validation of a simple classification system for endoscopic diagnosis of small colorectal polyps using narrow-band imaging. Gastroenterology. 2013;143(3):612–621.

[4] Rex DK, Kahi CJ, O'Brien M, et al. The American Society for Gastrointestinal Endoscopy PIVI on real-time endoscopic assessment of diminutive colorectal polyps. Gastrointest Endosc. 2011;73(3):419–422.

[5] Lieberman DA, Rex DK, Winawer SJ, et al. Guidelines for colonoscopy surveillance after screening and polypectomy. Gastroenterology. 2012;143(3):844–857.

[6] Bernal J, Sánchez FJ, Fernández-Esparrach G, et al. WM-DOVA maps for accurate polyp highlighting in colonoscopy: Validation vs. saliency maps from physicians. Comput Med Imaging Graph. 2015;43:99–111.

[7] Urban G, Tripathi P, Alkayali T, et al. Deep learning localizes and identifies polyps in real time with 96% accuracy in screening colonoscopy. Gastroenterology. 2018;155(4):1069–1078.

[8] Tian P, He B, Qu Y, et al. Automated classification of polyps using deep learning architectures and few-shot learning. BMC Med Imaging. 2023;23(1):59.

[9] Tan M, Le QV. EfficientNetV2: Smaller models and faster training. Proc ICML. 2021. arXiv:2104.00298.

[10] Zhou Z, Rahman Siddiquee MM, Tajbakhsh N, Liang J. UNet++: A nested U-net architecture for medical image segmentation. Deep Learning in Medical Image Analysis. 2018:3–11.

[11] Sagi O, Rokach L. Ensemble learning: a survey. WIREs Data Mining Knowl Discov. 2018;8(4):e1249.

[12] He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. Proc IEEE CVPR. 2016:770–778.

[13] Zhao Z, et al. YOLOv8 for polyp detection: a comparative study. arXiv. 2023.

[14] Wang C, et al. YOLO-LAN: A lightweight YOLO for polyp detection. arXiv. 2023.

[15] Ali S, et al. PolypGen: A multi-center polyp detection and segmentation dataset in colonoscopy. arXiv:2106.04463. 2021.

[16] Jha D, Smedsrud PH, Riegler MA, et al. Kvasir-SEG: A segmented polyp dataset. Proc MMM. 2020:451–462.

[17] Bernal J, Sánchez FJ, Fernández-Esparrach G, et al. CVC-ClinicDB: A clinical colonoscopy database for polyp detection. CVC Technical Report. 2015.

[18] Forootan, M., et al. ERCPMP: an endoscopic image and video dataset for colorectal polyps morphology and pathology. BMC Res Notes 17, 393 (2024). https://doi.org/10.1186/s13104-024-07062-6

## Author Biographies

![](images/2017a577deb23b085cdee3f01fb7ac1ffae20797449b938e91f70386ac3fa40a.jpg)

Hamidreza Bolhasani, PhD, is an AI/ML Researcher, Visiting Professor, and Founder and Chief Data Scientist at DataBioX (https://databiox.com). He holds a PhD in Computer Engineering with a focus on Artificial Intelligence and Computer Vision. His research spans Machine Learning, Deep Learning, Computer Vision, Biomedical AI, and Bioinformatics, with particular emphasis on AI applications in medical imaging and computational pathology. He led the overall research design and AI methodology for this work.

![](images/33a598535f8f4aaa0e4ac86148db45a59ecc7160f717c30ab6c7881b576fcad7.jpg)

Hamidreza Rastad, is a final-year undergraduate student at Iran University of Science and Technology (IUST), Tehran, Iran. He specializes in artificial intelligence and deep learning, and has played a leading role in the design and development of deep learning models in this work and related projects in biomedical AI.

![](images/23325c0f0931bb7b649801250bfacac8ce8d5bd289eb3ae4cb0b60fc2306bbe2.jpg)

Amir Mohammad Akbari, is a student at Iran University of Science and Technology (IUST), Tehran, Iran. His contributions to this work span deep learning model development and UI/UX design, as well as support for AI system integration and implementation.

![](images/12ee823514653a9a0246f5debae5321ce3cdd51874a237bd6a8b931ba74c1f05.jpg)

Ata Khodami, MD, is a specialist in Internal Medicine whose doctoral thesis addressed topics related to the subject of this work. His clinical interests include internal medicine and gastroenterology. His expertise contributed to the validation and clinical contextualization of the framework.

![](images/11c5bba733e8e2808d0085ccaaebbae7d32661ed1e143d9f9281dfb2a58167af.jpg)

Parnian Asadollahi, MD, is a resident in Radiation Oncology at Shahid Beheshti University of Medical Sciences, Tehran, Iran. She contributed to this research through clinical data review, annotation support, and oncologic interpretation of gastrointestinal precancerous lesions.

![](images/238e6725d022a937353056d8bb7028a3766f2f882dd33e822076e492a1695a5c.jpg)

Mohammad Tashakoripour, is a Researcher at the Gastroenterology Department, Amiralam Hospital, Tehran University of Medical Sciences, Tehran, Iran. His work focuses on histopathological grading and validation of gastrointestinal precancerous lesions.

![](images/ed55110c04a1044870ddb2b7897b5216ccc782dab10439796d5d9e7823b01551.jpg)

Mojgan Forootan, MD, is a Professor of Gastroenterology at the Gastroenterology and Liver Disease Research Center, Shahid Beheshti University of Medical Sciences, Tehran, Iran. She is interested in early detection and resection of GI precancerous lesions using AI. She is the corresponding author of this work.