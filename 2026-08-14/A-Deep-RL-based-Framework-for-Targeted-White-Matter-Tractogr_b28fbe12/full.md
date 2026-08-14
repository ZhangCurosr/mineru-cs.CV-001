# A Deep RL based Framework for Targeted White Matter Tractography

A Thesis Submitted in Accordance with the requirements for the degree of

MTECH BY RESEARCH

By

Ankita Joshi

(S22041 )

![](images/f203249dab8c71321c2efe42c8d15e53ebb583988da20e65bb895b7b26e3c87c.jpg)

to the

SCHOOL OF COMPUTING AND ELECTRICAL ENGINEERING   
INDIAN INSTITUTE OF TECHNOLOGY MANDI December, 2025   
Copyright © Ankita Joshi   
Indian Institute of Technology Mandi   
175005 - INDIA   
All rights reserved   
2025

# Declaration by the Research Scholar

I hereby declare that the entire work embodied in this thesis entitled “A Deep RL based Framework for Targeted White Matter Tractography” is the result of investigations carried out by me in the School of Computing and Electrical Engineering, Indian Institute of Technology Mandi, Mandi, H.P., India, under the supervision of Dr. Aditya Nigam, Associate Professor, School of Computing and Electrical Engineering, Indian Institute of Technology, Mandi, Mandi, India.

I also declare that it has not been submitted elsewhere for the award of any degree or diploma. In keeping with general practice, due acknowledgments have been made wherever the work described is based on the findings of other investigators. Any omissions that might have occurred due to oversight or error in judgment are regretted.

Date: 09/12/2025

Place: IIT Mandi,

Himachal Pradesh, India

Ankita Joshi,

Enrollment No.: S22041

School of Computing and Electrical Engineering,

Indian Institute of Technology Mandi,

Mandi, Himachal Pradesh, India - 175075.

## Thesis Certificate

This is to certify that the thesis titled “A Deep RL based Framework for Targeted White Matter Tractography”, submitted by Ankita Joshi, Enrollment No. S22041, to Indian Institute of Technology Mandi, for the award of the degree of Master of Technology by research, is a bona fide record of the research work done by him under our guidance. The contents of this thesis, in full or in parts, have not been submitted to any other institute or university for the award of any degree or diploma.

Dr. Aditya Nigam (Guide)

Associate Professor

School of Computing and

Electrical Engineering,

Indian Institute of Technology Mandi,

Himachal Pradesh, India - 175005.

Dr. Arnav Bhavsar (Co-guide)

Professor

School of Computing and

Electrical Engineering,

Indian Institute of Technology Mandi,

Himachal Pradesh, India - 175075.

## Acknowledgements

I express my deepest gratitude to my supervisor, Dr. Aditya Nigam, for his continuous support, insightful feedback, and encouraging words throughout the course of this research. His unwavering support and guidance have been instrumental in this journey. I am also sincerely grateful to my co-supervisor, Prof. Arnav Bhavsar, for his support and valuable inputs that contributed significantly to shaping this work. I am also grateful to Dr. Ranjeet Ranjan Jha for his support and guidance which greatly helped me navigate the challenges of research with clarity and confidence.

I would like to thank Anoushkrit Goel for being a constant source of encouragement and belief. His unwavering support made this journey far more enriching. I am also thankful to Ashutosh Sharma for being a dedicated and inspiring teammate, making our collaboration both smooth and productive. I am deeply grateful to the faculty members of IIT Mandi, particularly Prof. Dileep AD and Dr. Erwin Fuhrer, for their inspiring teaching and for fostering a challenging yet intellectually enriching environment. I also wish to acknowledge Geetanjali Sharma for being a wonderful friend and a strong support system throughout my time at IIT Mandi. My heartfelt thanks also go to my labmates and friends — Sushovan Jena, Shilpa Chandra, Soma Ma’am, Abhishek Tandon, and Munish Daroch — for their camaraderie and support. Lastly, I am profoundly thankful to my family for their unconditional love, support, and belief in me, which have been the true foundation of all my endeavours.

\- Ankita Joshi

## Abstract

Fiber tractography is a computational technique that estimates the three-dimensional trajectories of white-matter fibers by utilizing directional information derived from difusion MRI. Building on this ability to reconstruct the brain’s structural pathways, tractography has become a crucial component of modern neuroimaging, enabling detailed, non-invasive mapping of structural connectivity and supporting a wide range of neurological research and clinical applications. However, despite its importance, tractography remains a challenging task due to the inherent complexity of white matter structure and its susceptibility to false positives, which can lead to the misrepresentation of critical pathways. To overcome these limitations, recent approaches have increasingly turned to deep learning, particularly supervised learning and reinforcement learning (RL). While supervised learning requires precisely annotated ground-truth fibers which are often unavailable or unreliable, RL ofers the advantage of learning policies without such annotations. Despite the benefits, RL methods still face challenges in terms of performance and generalization, motivating the need for novel strategies to enhance their efectiveness. In this thesis, we propose a hybrid framework that integrates reinforcement learning with supervised learning for refining RL policies, specifically tailored for tract-specific tractography. Notably, our framework does not rely on ground-truth fibers for training. Moreover, the tract-specific formulation bypasses the need for an explicit segmentation process, simplifying the overall pipeline. Our work includes two main contributions, each building upon the previous. First, we introduce a hybrid approach that combines reinforcement learning with supervised learning (specifically, GPT-based policy learning) to refine policies in a tract-specific context. Second, we propose a scalable framework for data-driven

multi-policy fusion, which leverages the complementary strengths of multiple RL policies to improve tractography performance and robustness.

We demonstrate the efectiveness of our framework through extensive validation on benchmark public datasets including TractoInferno, HCP, and ISMRM-2015, highlighting its ability to generalize across data sources and accurately reconstruct brain white matter tracts. We believe that these contributions represent significant advancements in the field of tractography, improving robustness, reliability, and accuracy while reducing dependence on ground-truth annotations.

## Contents

Abstract ii   
List of Figures viii   
Abbreviations x   
List of Tables x   
1 Introduction 1   
1.1 Artificial Intelligence in Medical Imaging 1   
1.1.1 Introduction to Brain Imaging . 1   
1.2 Difusion Magnetic Resonance Imaging 3   
1.2.1 Acquisition 4   
1.2.2 Preprocessing 5   
1.2.3 Modeling difusion 6   
1.2.3.1 Difusion Tensor Imaging 6   
1.2.3.2 Fiber Orientation Distribution Function 6   
1.3 White Matter Fiber Tractography 7   
1.3.1 Streamline generation . 10   
1.3.2 Challenges 10   
1.3.3 Related Work: Tractography Techniques 11   
1.3.3.1 Classical Methods 11   
1.3.3.2 Machine Learning Methods 13   
1.3.3.3 Tract-specific Tractography 14   
1.4 Thesis Motivation 14   
Contents v   
1.5 Thesis Objectives and Contributions 15   
1.6 Thesis Organization 15   
2 Preliminaries 17   
2.1 Reinforcement Learning 17   
2.1.1 Fundamentals of RL 18   
2.1.2 Deep Reinforcement Learning 23   
2.1.2.1 Deep Q Networks . 23   
2.1.2.2 Policy Gradients & Actor-Critic methods . 24   
2.1.2.3 Actor-Critic Algorithms 25   
2.2 Transformers for Sequence Modeling 28   
2.2.1 Self-Attention Mechanism 29   
2.2.2 Transformer Architecture 30   
2.2.2.1 Multi-head Self-attention 31   
2.2.2.2 Position-wise Feedforward Network 32   
2.2.3 Generative Pre-trained Transformer 33   
2.2.3.1 Unsupervised Pretraining 34   
2.2.3.2 Supervised Fine-tuning . 34   
3 Tract-specific Tractography using GPT-based RL Policy Refine  
ment 35   
3.1 Introduction 35   
3.2 Data Specifications and Pre-processing 36   
3.3 Proposed Method 37   
3.3.1 Tract-RLFormer Framework 38   
3.3.1.1 RL Environment Setup 38   
3.3.1.2 Mask Refinement Module 40   
3.3.2 RL Policy Learning 41   
3.3.3 Tract-RLFormer for Policy Refinement 42   
3.3.3.1 Training Data Generation for T-RLF 43   
3.3.3.2 Architecture of T-RLF 43   
3.3.3.3 Training Procedure of T-RLF 44   
3.3.4 Tract generation and cleaning 46   
3.4 Result and Performance Analysis 47   
3.4.1 Evaluation Parameters 48   
3.4.2 Comparative Analysis 48   
3.4.3 Generalization Performance Evaluation . 50   
3.4.4 Ablation Study 53   
3.5 Summary 55   
4 Data-driven RL Policy Fusion Framework with Multi-Critic fine  
tuning for Tract-specific Tractography 56   
4.1 Motivation 56   
4.2 Proposed TractRLFusion Framework 57   
4.2.1 Overview and Experimental Setup 57   
4.2.2 Tract-Specific RL Policy Training 58   
4.2.3 Episodic Data Selection 59   
4.2.4 FusionNet Architecture and Training 61   
4.2.5 Multi-critic Policy Fine-tuning 62   
4.2.6 Tract Generation and Cleaning 64   
4.3 Experiments and Results 65   
4.3.1 Comparative Study 65   
4.3.2 Ablation Study 69   
4.4 Summary 71   
5 Conclusion and Future Work 72   
List of Publications 75   
Bibliography 77

## List of Figures

1.1 Siemens Magnetom Verio 3T MRI Scanner [5]. 2   
1.2 Overview of Structural MRI, Functional MRI (fMRI), and Difusion   
MRI (dMRI) Modalities [12] . 3   
1.3 Illustration of (i) free isotropic movement in like CSF. (ii) depicts   
Anisotropic along the axon of the neuron, which happens in WM. 4   
1.4 Major White Matter tracts [27]. 8   
1.5 Tractography process 8   
1.6 Whole-brain tractogram and its segmentation into anatomically mean  
ingful white-matter tracts. Source: [28]. 9   
1.7 Illustration of common complex fiber configurations, including cross  
ing, branching, and fanning pathways. 11   
2.1 Illustration of a typical reinforcement learning setup 19   
2.2 Model-free RL algorithms (non-exhaustive) 23   
2.3 Illustration of actor-critic framework in RL 26   
2.4 Illustration of Transformer model architecture [60] . 30   
3.1 Overview of the proposed Tract-RLFormer framework. 39   
3.2 Mask Refinement Module for generating tract-specific masks 41   
3.3 Data Representation for Tract-RLForm 44   
3.4 Training Procedure of Tract-RLFOrmer 45   
3.5 Results of Qualitative Analysis of Tract-RLForemer and other trac  
tography methods . 52   
4.1 Illustration of TractRLFusion Framework 58   
4.2 Visual depiction of Episodic Data Selection . 60   
4.3 Illustration of Multi-Critic Policy Finetuning Module 63   
4.4 Visualization of Qualitative Comparison of TractRLFusion . 66

## List of Tables

2.1 Comparison of Supervised, Unsupervised, and Reinforcement Learning 18   
3.1 Details of the difusion datasets used in our experiments. 37   
3.2 Comparative Analysis of Tract-RLFormer with Supervised, Classical,   
and RL-based algorithms 49   
3.3 Generalization performance evaluation of CG and AF tracts 51   
3.4 Generalization performance evaluation of PYT and CC tracts 53   
3.5 Model architecture selection for Tract-RLFormer network . 54   
3.6 Ablation study of the efect of tw-stage training process and MRM   
masks on Tractography performance 54   
4.1 Summary of Features, Architecture, Training Data, and Hyperparam  
eters for training Tract-specific RL policies. 59   
4.2 Summary of FusionNet architecture, training, and tracking hyperpa  
rameters. 62   
4.3 Comparison of performance metrics (%) for AF and CC, showing gen  
eralization across HCP, TractoInferno(TtoI), and ISMRM datasets.   
Metrics (Dice, OL, OR) are averaged across Left/Right for AF, and   
across Oc/Pr\_Po for CC. 67   
4.4 Comparison of performance metrics (%) for CG and CST, showing   
generalization across HCP, TractoInferno(TtoI), and ISMRM datasets.   
Metrics (Dice, OL, OR) are averaged across Left/Right. . 68   
4.5 Ablation study of the EDS module highlighting the impact of diferent   
training data selection strategies . 70   
4.6 Ablation study of the MCPFT module 70

## Abbreviations

DWI Difusion Weighted Image Pg. 4   
MRI Magnetic Resonance Imaging Pg. 1   
CT Computed Tomography Pg. 1   
PET Positron Emission Tomography Pg. 1   
fMRI Functional Magnetic Resonance Imaging Pg. 2   
dMRI Difusion Magnetic Resonance Imaging Pg. 2   
HARDI High Angular Resolution Difusion Imaging Pg. 5   
DTI Difusion Tensor Imaging Pg. 4   
AD Axial Difusivity Pg. 6   
FA Fractional Anisotropy Pg. 6   
MD Mean Difusivity Pg. 6   
RD Radial Difusivity Pg. 6   
TBI Traumatic Brain Injury Pg. 6   
fODF Fiber Oriented Distribution Function Pg. 5   
CSD Constrained Spherical Deconvolution Pg. 4   
RL Reinforcement Learning Pg. 13   
DRL Deep Reinforcement Learning Pg. 13   
DDPG Deep Deterministic Policy Gradient Pg. 15   
TD3 Twin Delayed Deep Deterministic Policy Gradient Pg. 15   
SAC Soft Actor Critic Pg. 15   
GPT Generative Pretrained Transformer Pg. 15   
HCP Human Connectome Project Pg. 36   
ISMRM International Society for Magnetic Resonance in Medicine Pg. 36   
T-RLF Tract-RLFormer Pg. 37   
AF Arcuate Fasciculus Pg. 37   
CC Corpus Callosum Pg. 37   
CST Cortico-Spinal Tract Pg. 37   
PYT Pyramidal Tract Pg. 37   
OR Optical Radius Pg. 47   
EDS Episodic Data Selection Pg. 58   
MCPFT Multi-critic Policy Finetuning Pg. 58

## Chapter 1

## Introduction

## 1.1 Artificial Intelligence in Medical Imaging

Artificial intelligence has found widespread application in the medical domain, spanning drug discovery and delivery, diagnosis, treatment planning, robot-assisted surgery and guided interventions. Advances in machine learning and deep learning approaches have contributed to improved outcomes, greater eficiency, faster treatment pipelines, better planning, and reduced costs. Medical imaging, in particular, has become an essential part of modern healthcare, allowing doctors to study the structure and function of the human body without invasive procedures. Modalities such as MRI, CT, PET, ultrasound, and X-ray generate vast amounts of data, often far exceeding what can be feasibly analyzed through manual interpretation. In recent years, deep learning approaches have achieved remarkable success in image classification [1], detection [2], segmentation [3], and reconstruction [4], making them highly efective for managing these complexities. These learnable techniques provide datadriven insights that complement and augment clinical expertise, enabling a shift in the way clinicians and researchers interpret and make use of complex medical images.

## 1.1.1 Introduction to Brain Imaging

The human brain is a complex organ responsible for controlling a wide range of functions, including motor functions, sensory processing, cognitive operations, emotional regulation, and essential autonomic functions such as respiration and cardiovascular functions. To understand this complexity, brain imaging plays a crucial role, as it enables researchers and clinicians to visualize brain structure and function in detail. Such visualization is vital for diagnosing and studying neurological conditions, guiding treatments, and advancing neuroscience research.

![](images/123bba0caa2b1f67d701f09791076b7349f4ba3e6bc3c1882c9eff7550cb5495.jpg)  
Figure 1.1: Siemens Magnetom Verio 3T MRI Scanner [5].

Diferent imaging techniques, such as Magnetic Resonance Imaging (MRI) [6], Computed Tomography (CT) [7], and Positron Emission Tomography (PET) [8], provide complementary information about brain health, structure, and function. Among these, MRI is currently the most common imaging modality for brain tissue evaluation because of its excellent soft-tissue contrast and high spatial resolution [9]. CT scans, by contrast, are particularly important in emergency settings for rapid assessment of injuries and acute conditions such as stroke, though they ofer lower resolution compared to MRI and involve radiation exposure. PET scans, on the other hand, are used to visualize metabolic activity in the brain, aiding in the diagnosis of disorders like Alzheimer’s disease and Parkinson’s disease. However, PET also involves radiation exposure and is therefore less commonly used than functional MRI for studies of brain activity. Within MRI, two notable techniques are Difusion MRI (dMRI) and Functional MRI (fMRI). dMRI provides insights into the structural connections of the brain, particularly white matter, by measuring the difusion of water molecules in tissues. fMRI, in contrast, helps in understanding how diferent brain regions interact during specific tasks. It measures brain activity by detecting changes in blood flow associated with neural activation, often using Blood Oxygen Level Dependent (BOLD) contrast to map function. This makes fMRI particularly valuable for diagnosing conditions such as stroke, epilepsy, and brain tumors.

Over the past decade, brain imaging has undergone significant advancements, particularly through the integration of artificial intelligence (AI) and the development of multi-modal imaging [10, 11]. These innovations continue to enhance both clinical applications and research into the brain’s complex organization.

![](images/0ca92b7886f0b3cd074887e29736f4eca22a814d1096b0880e58b4783dc35608.jpg)  
Figure 1.2: Overview of Structural MRI, Functional MRI (fMRI), and Difusion MRI (dMRI) Modalities [12]

## 1.2 Difusion Magnetic Resonance Imaging

Difusion MRI (dMRI) is a non-invasive imaging technique that measures the movement of water molecules within biological tissues. Water difusion is not completely random but is constrained by cellular structures such as membranes, fibers, and organelles, and therefore varies across tissue types. In the brain, for example, cerebrospinal fluid (CSF) exhibits free isotropic difusion (Fig. 1.3 (i)), grey matter shows relatively isotropic difusion due to densely packed neuronal cell bodies, while white matter displays highly anisotropic difusion, with water preferentially moving along axonal fibers (Fig. 1.3 (ii)). Capturing these difusion patterns provides estimates of the underlying tissue microstructure.

![](images/f03dce4a6b2ca1655a9d952148029f1b3beb3719f99f3d5da3111bf97921d919.jpg)  
Figure 1.3: Illustration of (i) free isotropic movement in like CSF. (ii) depicts Anisotropic along the axon of the neuron, which happens in WM.

## 1.2.1 Acquisition

During dMRI acquisition, the scanner applies gradients to sensitize the MRI signal to water motion. This produces Difusion weighted images (DWIs), where contrast is weighted by the degree of water difusion. The degree of difusion sensitivity is determined by the b-value, while the gradient direction is defined by the b-vector. dMRI can be acquired at diferent b-values and across multiple directions. The data can be represented as a 4D array with dimensions $( \mathrm { X } , \mathrm { Y } , \mathrm { Z } , N _ { v o l u m e s } )$ , where $N _ { v o l u m e s }$ denotes the number of difusion-weighted images. Each volume is associated with a specific b-value and a corresponding b-vector. X, Y, and Z represent the spatial dimensions of the brain volume. The accompanying metadata includes b-values, b-vectors, voxel dimensions, and the afine transformation matrix.

Depending on the data (number of b-values and directions acquired), it can be further modeled to infer fiber orientation and other features using approaches such as Difusion Tensor Imaging (DTI) [13] or Constrained Spherical Deconvolution (CSD) [14], which are described in later sections.

## High Angular Resolution Difusion Imaging (HARDI):

HARDI is an acquisition technique in which difusion images are sampled at around 30 or more directions (up to 100) and at one or more b-values. With HARDI data, advanced modeling methods such as CSD can be applied to estimate the fiber Orientation Distribution Function (fODF). This provides richer angular coverage of the difusion signal compared to DTI, enabling the resolution of complex fiber structures that DTI cannot capture. HARDI thus enables the estimation of the fODF, ofering a more accurate description of fiber orientations within a voxel, including regions of fiber crossing, branching, and fanning. It has proven particularly valuable in advanced tractography and connectomics research [15, 16]. However, HARDI requires longer acquisition times and higher signal-to-noise ratios, which can be challenging in clinical settings.

## 1.2.2 Preprocessing

Preprocessing of difusion MRI (dMRI) data is an essential step prior to difusion modeling, tractography, and connectivity analyses. It is used to correct the subject motion, eddy current distortions and other artefacts present in the raw dMRI data.

• The preprocessing begins with converting the raw data from scanner-specific format DICOM (industry standard) to standard post-processing analysis formats such as NIfTI (Neuroimaging Informatics Technology Initiative) or MINC.

• This is followed by eddy currents and subject motion correction. Eddy currents are induced by rapid gradient switching during acquisition, and can cause geometrical distortions, scaling, and voxel misalignment in the final image. Tools such as FSL Eddy [17] are widely used to correct these distortions and align difusion-weighted images accurately.

• Bias field is a low frequency signal that causes homogenous tissues to have varying intensities in dMRI scans. These may arise from the scanner’s RF coil sensitivity or the subject’s anatomy. Correction techniques include filtering, surface fitting, segmentation, and histogram-based methods. N4ITK [18] is the current state-of-the-art correction technique, integrated into neuroimaging tools such as MRtrix [19], SimpleITK [20], ANTs [21], and FreeSurfer [22].

• Subsequently, brain extraction (or skull stripping) is performed to remove nonbrain tissues and isolate the brain volume for analysis.

• A final Quality control step is conducted to identify any residual artefacts and signal dropouts, using visual examination or automated inspection tools like SQUAD (Study-wise QUality Assessment for DMRI) [23] and DTIPrep [24] etc. Together, these preprocessing steps help ensure data reliability for subsequent difusion analysis and connectivity mapping.

## 1.2.3 Modeling difusion

The primary dMRI modeling techniques include:

## 1.2.3.1 Difusion Tensor Imaging

It is a modeling technique that characterizes water difusion in tissue as an ellipsoid by fitting a 3D Gaussian tensor. To estimate the difusion tensor, at least 6 noncollinear difusion-weighted gradient directions are required during MRI acquisition, typically acquired at a single b-value (i.e., a single-shell scheme).

From the fitted difusion tensor, we obtain three eigenvectors (principal difusion directions) and their corresponding eigenvalues $\left( \lambda _ { 1 } , \lambda _ { 2 } , \lambda _ { 3 } \right)$ (see Fig. 1.3 (ii)), which represent the square of magnitude of difusion along those directions. These values are used to derive DTI metrics such as Fractional Anisotropy (FA), Mean Difusivity (MD), Axial Difusivity (AD), and Radial Difusivity (RD) [13,25,26]. These metrics are widely used in research, particularly in the case of Traumatic Brain Injury (TBI). However, DTI assumes a single dominant fiber orientation per voxel, which limits its accuracy in regions with complex fiber crossings.

## 1.2.3.2 Fiber Orientation Distribution Function

The fiber orientation distribution function is a continuous function defined on the unit sphere that describes the probability distribution of fiber orientations within a voxel (a 3D pixel) of a difusion MRI image. This is crucial because older models like DTI struggled in areas where fibers cross, as they could only assign a single orientation to a voxel. Constrained Spherical Deconvolution [14] is a widely used technique to estimate fODF from the raw difusion-weighted signal. It computes the set of spherical harmonic coeficients (SHCs) of the fODF. Spherical Harmonics are essentially the angular basis functions defined on the surface of a sphere. The spherical harmonic coeficients (SHC) provide a compact parametric representation of the fODF. In regions of anisotropy such as white matter, the fODF exhibits distinct peaks, each corresponding to a dominant fiber orientation (local maxima of fODF) within the voxel. These principal directions are used as local features to guide the reconstruction of white matter pathways (tractography).

For a DWI image of shape $( X , Y , Z , N _ { v o l u m e s } )$ , the fODF is represented by its spherical harmonic (SH) coeficients with dimensions $( X , Y , Z , ( l { + } 1 ) ( l { + } 2 ) / 2 )$ Each voxel’s SH coeficients correspond to harmonics up to degree l (even integers for fODFs), which determine the angular resolution of the reconstructed distribution. Similarly, the fODF peaks are stored with dimensions $( X , Y , Z , 3 \times N _ { p e a k s } )$ , where $N _ { p e a k s }$ are the total number of peaks and each peak is represented by a 3D vector indicating its spatial direction.

## 1.3 White Matter Fiber Tractography

White matter plays a critical role in interconnecting cortical and subcortical regions of the brain. It constitutes roughly half of the brain’s volume and is composed of myelinated axons that form coherent fiber tracts. These tracts, shown in Fig. 1.4 ( Image taken from [27]), enable signal transmission between diferent brain regions and the spinal cord.

Tractography is the process of mapping these white matter pathways non-invasively using difusion MRI data. The fODF and its peaks, derived from dMRI data, represent the dominant fiber directions within each voxel and are often used as input to tractography algorithms, as shown in Fig. 1.5. Building on this local information, tractography estimates long-range fiber pathways that represent global structural

![](images/e251257166c8143261eb43478066b126119c5d24e44c7dadbca07fea63b4ddf5.jpg)  
Figure 1.4: Major White Matter tracts [27].

connectivity.

![](images/0feb4713d7ae038d9d5c236308236b36a45a5606f5af4f37ae8f9485a04db3b6.jpg)  
Figure 1.5: Tractography process

Tractography can be understood as tracking in 3D space (brain volume), constrained by a tracking mask and anatomy. Seed points are placed within this mask, and at each step the algorithm predicts the next direction based on local difusion information (eg. fiber orientation), scaled by a step size. Starting from each seed point, tracking continues until termination criteria are met, producing the reconstructed tractogram (Fig. 1.5). A tractogram can be described as a collection of fiber streamlines, where each streamline is a sequence of 3D points of variable length.

![](images/9121f0be7bb2ccf48002a8961384fb21230694ac5d5ae4798a10f11252942009.jpg)  
Figure 1.6: Whole-brain tractogram and its segmentation into anatomically meaningful whitematter tracts. Source: [28].

After producing a whole-brain tractogram, the next step is tractogram segmentation, which groups fibers that share similar anatomical trajectories or endpoints, producing coherent tracts such as the corticospinal tract, arcuate fasciculus, corpus callosum fibers, etc. displayed in Fig. 1.6 [28]. By extracting individual tracts, researchers and clinicians can analyze their microstructural properties, assess tractspecific damage or displacement, and relate structural changes to functional or behavioral outcomes. Segmentation process enables tractography to support precise diagnostics, surgical planning, and targeted neuroscientific investigations.

Building on this, Tractography is widely applied in neurosurgical planning, where it allows surgeons to visualize the structure of critical white matter pathways, such as corticospinal tract, and optic radiation, and plan surgical interventions that avoid or minimize damage to these fibers. It is also used in assessing damage from traumatic brain injury, which may cause fibers to stretch, shear or break down, resulting in reduced anisotropy and dirupted connectivity. It also aids in studying structural disruptions in neurological conditions such as Alzheimer’s and Parkinson’s disease, and is also widely used to explore altered brain connectivity patterns in tumors and psychiatric disorders.

## 1.3.1 Streamline generation

Tracking begins from a set of seed points, which can be defined within specific regions of interest (ROI) or across the entire white matter volume. From each seed, the algorithm iteratively propagates a streamline in small steps.

At each step, the algorithm predicts the next direction and propagates the streamline by a fixed step size, or directly estimates the next point, based on voxelwise difusion information derived from the chosen difusion model (e.g., difusion tensor, fODF, SHCs, or fODF peaks). Several stopping criteria are used to terminate a streamline. These typically include thresholds on fractional anisotropy (FA), maximum curvature angle, and anatomical boundaries. For example, tracking is commonly stopped when FA drops below a predefined limit, indicating regions of isotropic difusion, or when the angular deviation between successive steps exceeds a threshold, implying an improbable fiber turn. Fig. 1.5 shows the workflow of a typical tractography algorithm. It also includes a segmentation step, which is commonly performed to extract specific anatomical tracts from the whole-brain tractogram.

## 1.3.2 Challenges

Tractography has been described as an ill-posed problem due to the inability of difusion MRI to uniquely determine the underlying white-matter pathways [29]. This is because multiple anatomically distinct fiber configurations can produce similar voxel-wise difusion signals, making it impossible to distinguish between these cases purely at the voxel level. This uncertainty at the voxel level can propagate through tractography algorithms, leading to errors in resulting pathways, such as missing true connections (False negatives) or generating spurious streamlines (False positives) [30].

Moreover, regions with complex fiber architectures (e.g. crossing, branching, or fanning fibers), shown in Fig. 1.7 are very prone to errors (false positive and false negative connections). In these cases, algorithmic and methodological choices also play a crucial role. Deterministic tractography methods follow a single, most likely path at each point and can therefore fail to capture branching or crossing fibers, resulting in a large number of false negatives. Whereas probabilistic algorithms generate multiple potential fiber directions, which potentially lead to false positive connections. Deciding the optimal balance between sensitivity and specificity remains a persistent issue [31].

![](images/076eaf76e930219794c1f5a2e056be1323fe6bed7f4d7f8951f5e40a6a2fa54f.jpg)  
Figure 1.7: Illustration of common complex fiber configurations, including crossing, branching, and fanning pathways.

Furthermore, choices of angular threshold, step size, curvature constraints, fractional anisotropy (FA) and other parameters critically afect tracking results, leading to large inter-study variability. Finally, validation and reproducibility present ongoing challenges. The absence of a definitive in-vivo ground truth for human brain connectivity [32] makes it dificult to assess the accuracy of tractography results, highlighting the need for more reliable validation methods.

## 1.3.3 Related Work: Tractography Techniques

Tractography Techniques can be broadly classified into two main approaches: deterministic and probabilistic tracking. Deterministic methods [33, 34] follow the most likely difusion direction at each step, producing precise but potentially less robust reconstructions in areas of complex fiber orientations. In contrast, probabilistic methods [35, 36] sample from the distribution of possible orientations, generating multiple potential trajectories to account for uncertainty in the difusion model.

## 1.3.3.1 Classical Methods

As discussed in section 1.2.3, before applying any tractography algorithm, local difusion modeling techniques are first used to estimate fiber orientations within each voxel. Classical tractography methods heavily rely on such mathematical models.

Early tractography approaches were based on the DTI model, which estimates a single principal difusion direction per voxel [13]. Deterministic tractography methods using the DTI model, such as [33, 34] sufer from the “crossing fiber problem,” since DTI leads to missed connections in regions where multiple fiber bundles intersect. To overcome these limitations, probabilistic tractography techniques were developed, introducing uncertainty into the local orientation estimates. Methods such as [37, 38] generate multiple possible streamlines based on the probability distribution of fiber orientations, producing lesser False positives in complex regions. The iFOD2 algorithm [36] represents a second-order integration approach over the Orientation Distribution Functions (ODFs), improving angular accuracy and robustness to noise. Beyond local tracking, global tractography methods attempt to infer the entire set of streamlines simultaneously, optimizing for the configuration that best explains the measured DWI signal across the whole brain. Examples include Gibbs tracking [39], which formulates tractography as an energy minimization problem. Although this framework provides more globally consistent solutions, it is computationally expensive and thus less commonly used in routine applications.

Advances in difusion imaging have enabled the acquisition of HARDI data, making it possible to employ advanced difusion models like Q-ball Imaging [15] and Constrained Spherical Deconvolution [14]. These models estimate Orientation Distribution Functions (ODFs) that can resolve multiple fiber directions within a voxel, unlike DTI. As tractography techniques evolved, anatomical priors and tissue constraints were introduced to enhance biological plausibility. Anatomically Constrained Tractography (ACT) [40] implements this by using tissue-segmented anatomical priors to prevent streamlines from propagating into non-white matter regions. Particle Filtering Tractography (PFT) [41] further extends this approach by integrating partial volume estimates (PVEs) into the tracking process, guiding streamline propagation and termination based on the proportions of gray matter, white matter, and cerebrospinal fluid within each voxel. By incorporating anatomical constraints, ACT and PFT enhanced the biological plausibility of streamline reconstruction, bridging the gap between signal-driven and anatomically informed tractography. Even with the advent of machine learning and deep learning methods, classical tractography methods remain strong baselines. They are frequently used in ensemble frameworks for generating reference datasets or tract atlases [29, 42].

## 1.3.3.2 Machine Learning Methods

Early learning-based tractography methods focused on supervised classification of fiber orientations. Authors in [43] utilized a Random Forest classifier in a supervised setting to identify 25 distinct fiber bundles, leveraging data from the ISMRM 2015 Tractography Challenge dataset [29]. This approach demonstrated the feasibility of applying machine learning to white matter bundle segmentation, though it remained dependent on handcrafted features and labeled training data. Building on this foundation, research progressed toward regression-based models for streamline tracking. Learn to Track [44] proposed the use of a Gated Recurrent Unit (GRU) network to predict the next tracking direction directly from difusion-weighted signals resampled to 100 gradient directions. The recurrent design enabled modeling of temporal dependencies along streamlines. DeepTract [45] further advanced this paradigm by employing Gated Recurrent Unit (GRU) for both deterministic and probabilistic tractography. It outputs continuous orientation distribution functions (ODFs) at each step, allowing uncertainty-aware streamline propagation. In another work, [46] introduced a data-driven approach using a multilayer perceptron (MLP) that takes as input the local difusion signal from a $3 \times 3 \times 3$ voxel neighborhood, together with the last four tracking directions, thereby incorporating both spatial and temporal contextual information. Entrack [47] extended this concept through a probabilistic deep learning framework that predicts a full Fisher–von Mises distribution over possible tracking directions, allowing for uncertainty quantification and improved generalization across subjects. More recently, Deep Reinforcement Learning (DRL) methods such as Track-to-Learn [48,49] and TractOracle [50] have emerged, aiming to remove the dependence on ground-truth streamlines. These approaches frame tractography as a sequential decision-making problem, where an agent learns optimal tracking policies through interaction with difusion data, guided by anatomically informed reward signals.

## 1.3.3.3 Tract-specific Tractography

Tract-specific tractography aims to directly reconstruct the tract of interest, focusing on particular white matter pathways. Unlike whole-brain tractography, where tracking is conducted within the entire white matter (WM) mask followed by segmentation to obtain individual tracts, tract-specific methods restrict tracking to regions or orientations associated with a specific bundle. These methods are crucial for clinical surgical planning [51], and research focusing on the microstructural properties and precise geometry of a known white matter pathway.

TractSeg [28] uses Fully connected neural network (FCNN) to create a Tract Orientation Map (TOM) for a specific bundle, where each voxel contains the most likely fiber orientation for that particular bundle. Standard streamline tracking (deterministic or probabilistic) is then run only on this TOM (Wasserthal et al., 2019), bypassing the need for whole-brain tractography. On the other hand, Bundle-Specific Tractography (BST) [52] integrates anatomical and orientational prior knowledge, derived from a template or atlas of the target bundle, directly into a probabilistic tracking algorithm (such as iFOD2 [36] or related methods). This guidance is applied during the tracking process to enhance or suppress propagation in certain directions for a given bundle or tract.

## 1.4 Thesis Motivation

Classical deterministic and probabilistic tractography methods are highly sensitive to noise, depend heavily on difusion modeling assumptions, and often produce anatomically implausible or incomplete pathways. Supervised learning approaches, though more data-driven, rely on ground-truth streamlines that are themselves uncertain or unavailable, leading to propagation of labeling biases.

Deep Reinforcement Learning has recently emerged as a promising alternative by framing tractography as a sequential decision-making process, where an agent learns optimal tracking policies through interaction with difusion signals rather than supervision from predefined tracts. However, existing DRL-based tractography methods have immense scope for improvement and still face the persistent sensitivity–specificity trade-of, where agents either fail to fully capture the extent of the tract or erroneously extend into neighboring tracts or non-white matter regions (overreach).

This thesis addresses these challenges through two key contributions that improve the tractography performance of DRL policies and introduce a policy fusion approach to mitigate the sensitivity–specificity (overlap–overreach) trade-of. Furthermore, the method is designed to be tract-specific, directly reconstructing individual bundles of interest, thus eliminating the need for whole-brain tractography followed by segmentation.

## 1.5 Thesis Objectives and Contributions

The primary aim of this thesis is to address the challenges of applying deep reinforcement learning (DRL) to tractography by improving the performance and robustness of various RL-based tracking policies, such as TD3, SAC, and DDPG, and by fusing their complementary strengths to overcome existing limitations in fiber tractography. The main contributions of this research are summarized as follows:

• We present Tract-RLFormer, a hybrid framework for RL policy refinement in tractography, leveraging GPT-based modeling and deep RL.

• We introduce TractRLfusion, a novel RL policy fusion framework for tractography that combines multiple policies to achieve superior performance by balancing the specificity-sensitivity trade-of.

• The proposed tract-specific frameworks eliminate the need for a segmentation step. A Mask Refinement Module (MRM) generates the tracking regions of interest (masks). Furthermore, training is conducted without ground-truth annotations and achieve strong generalization performance.

## 1.6 Thesis Organization

This chapter provides an overview of the field of brain imaging, with a particular focus on difusion MRI-based white matter fiber tractography. It also discusses the motivation behind this thesis. The remainder of the thesis is organized as follows:

• Chapter 2: A review of the deep learning methods utilized in this work.

• Chapter 3: Introduction of a tract-specific framework that refines RL policies for tractography, thereby improving their tracking performance.

• Chapter 4: Presentation of a data-driven fusion strategy that integrates multiple RL policies to improve tractography outcomes.

• Chapter 5: Summary of the key contributions of this thesis and a discussion of potential directions for future work.

## Chapter 2

## Preliminaries

This chapter presents an overview of the deep learning and deep reinforcement learning foundations relevant to this thesis. We begin by introducing the major learning paradigms, followed by a detailed explanation of the reinforcement learning (RL) framework and the specific algorithms employed in this work. We then discuss sequence modeling and Transformer-based architectures. These preliminaries establish the conceptual basis for Chapters 3 and 4, where deep RL methods and GPT-based models are leveraged to advance the performance of RL algorithms for tractography.

The goal of Machine learning algorithms is to learn the mapping between input and output. There are several diferent ways to learn it, and these techniques are typically broadly classified into 3 paradigms, namely Supervised, Unsupervised and Reinforcement learning, as shown in Table 2.1. Supervised learning algorithms require labelled data (input- output examples) to train (mathematical) models. Unsupervised learning algorithms learn to identify hidden patterns, associations, or groupings within data without predefined target outputs. Reinforcement Learning algorithms learn by exploration.

## 2.1 Reinforcement Learning

In this work, tractography is performed within a reinforcement learning (RL) environment, following previous work [48]. The RL formulation of tractography, along with the modifications introduced for tract-specific tracking, are described in detail

<table><tr><td rowspan=1 colspan=1>Criteria</td><td rowspan=1 colspan=1>SupervisedLearning</td><td rowspan=1 colspan=1>UnsupervisedLearning</td><td rowspan=1 colspan=1>ReinforcementLearning</td></tr><tr><td rowspan=1 colspan=1>Type   ofData</td><td rowspan=1 colspan=1>Labeled data</td><td rowspan=1 colspan=1>Unlabeled data</td><td rowspan=1 colspan=1>Learns by interactionwith environment.</td></tr><tr><td rowspan=1 colspan=1>Goal</td><td rowspan=1 colspan=1>Predict outcomesbased on input-output mapping.</td><td rowspan=1 colspan=1>Discover    hiddenpatterns or group-ings in data.</td><td rowspan=1 colspan=1>Maximize cumulativerewards over time.</td></tr><tr><td rowspan=1 colspan=1>Examples</td><td rowspan=1 colspan=1>Image recognition,price prediction.</td><td rowspan=1 colspan=1>Customer segmen-tation, anomaly de-tection.</td><td rowspan=1 colspan=1>Self-driving      cars,game              play-ing(AlphaGo)</td></tr><tr><td rowspan=1 colspan=1>Challenges</td><td rowspan=1 colspan=1>Requires  labeleddata; less effectivefor complex tasks.</td><td rowspan=1 colspan=1>May produce lessaccurate   results;computationallyintensive.</td><td rowspan=1 colspan=1>Requires careful re-ward design; compu-tationally expensive.</td></tr></table>

Table 2.1: Comparison of Supervised, Unsupervised, and Reinforcement Learning

in Section 3.3.1.

Reinforcement Learning (RL) is inspired by how humans and animals learn through trial-and-error interactions with their environment. For example, a child learning to ride a bicycle tries various ways of pedaling and balancing, receives feedback (disbalancing, falling, or staying upright), and gradually improves by repeating actions that lead to success. This process of learning from interaction and feedback forms the basis of RL, where an agent observes the state(s) of the environment, takes actions(a), and receives rewards(r) from the environment for its actions. The agent’s objective is to learn a policy(π), which is a strategy to select actions that maximize cumulative rewards over time. The agent learns solely from its interactions with the environment and does not rely on external labeled data or “ground truth”.

## 2.1.1 Fundamentals of RL

RL is formulated as a Markov Decision Process (MDP), which is a mathematical framework for modeling sequential decision-making under uncertainty. It is defined by the tuple (S, A, P, R, γ). Here, S is the set of states, A is the set of actions, P refers to the transition probability matrix, R is the reward space and γ is the discount factor.

![](images/d7975b2052af59b1d1b1382bb784c130ca795dbfc1f0578989e96059b710f775.jpg)  
Figure 2.1: Illustration of a typical Reinforcement Learning (RL) setup: the agent interacts with the environment by taking actions, receives feedback in the form of rewards, and observes the resulting state to learn an optimal policy.

Formalizing the definitions of key components of RL:

• State (S): Represent current configuration of the environment (e.g. a robot’s position on a grid).

• Action (A): Decision taken by agent that transitions it from one state to another. Action space(A) can be both discrete (e.g. choosing left or right) or continuous (e.g. applying a steering angle or force value), depending on the problem at hand.

• Transition Probability $\left( P ( s ^ { ' } | s , a ) \right)$ : It provides probability of transitioning to state $s ^ { ' }$ from state s by taking an action a.

• Reward (R): $\mathrm { R } ( \mathrm { s } , \mathrm { a } , s ^ { \prime } )$ provides immediate feedback for transitioning from s to $s ^ { ' }$ via action a.

• Discount Factor (γ): Balances immediate and future rewards $( \gamma \in [ 0 , 1 ] )$ prioritizing short-term gains if closer to 0 and long-term if closer to 1.

The policy π is a function that maps a given input state to an action. The objective of MDP is to find an optimal policy (π<sup>∗</sup>) that maximizes the discounted

sum of rewards (return), denoted by:

$$
G _ { t } = r _ { t + 1 } + \gamma r _ { t + 2 } + \gamma ^ { 2 } r _ { t + 3 } + . . . .\tag{2.1.1}
$$

where $r _ { t }$ denotes the reward received at time step t, that the agent obtains after taking an action in a given state. A sequence of interactions between the agent and the environment, consisting of states, actions, rewards, and next states is called an episode or a trajectory. Formally, an episode is represented as: $\tau = ( s _ { 0 } , a _ { 0 } , r _ { 1 } , s _ { 1 } , a _ { 1 } , r _ { 2 } , \dots , s _ { T } )$ , where T is the terminal timestep (either fixed or determined by a stopping condition). Episodes usually terminate due to environmentdefined constraints, such as reaching a goal state, exceeding a time limit, or encountering a terminal condition (e.g., agent crashes or fails). For a T-length episode, return is defined as:

$$
G ( \tau ) = \sum _ { t = 0 } ^ { T - 1 } \gamma ^ { t } r _ { t + 1 }\tag{2.1.2}
$$

This formulation also includes unknown rewards from future timesteps. Since future rewards are uncertain (unknown due to stochastic transitions), return is a random variable that cannot be directly estimated or computed. Its value depends on the sequence of states and actions encountered, which are influenced by probabilities. Hence, the expected value of return is utilized to evaluate policies. This expectation is formalized through two functions that estimate the expected cumulative reward:

1. State-value function: Estimates the expected return starting from state s and following policy π.

$$
V ^ { \pi } ( s ) = \underset { \tau \sim \pi } { \mathbb { E } } \left[ G ( \tau ) \bigg | s _ { 0 } = s \right]\tag{2.1.3}
$$

Substituing the value of return from Eqn. 2.1.2

$$
V ^ { \pi } ( s ) = \mathbb { E } \left[ \sum _ { t = 0 } ^ { T - 1 } \gamma ^ { t } r _ { t + 1 } \Big | s _ { 0 } = s \right]\tag{2.1.4}
$$

2. Action-value function: Estimates expected return obtained by the agent,

starting from state s and performing action a and thereafter following policy $\pi$

$$
Q ^ { \pi } ( s , a ) = \mathbb { E } \left[ \sum _ { t = 0 } ^ { T - 1 } \gamma ^ { t } r _ { t + 1 } \bigg | s _ { 0 } = s , a _ { 0 } = a \right]\tag{2.1.5}
$$

The bellman equation decomposes the value functions into a recursive formulation of immediate rewards and discounted future values.

$$
V ^ { \pi } ( s ) = R ( s , a , s ^ { \prime } ) + \gamma V ^ { \pi } ( s ^ { \prime } )\tag{2.1.6}
$$

Here, $\mathrm { R } ( \mathrm { s } , \mathrm { a } , s ^ { \prime } )$ refers to the immediate reward for transitioning from state $\mathrm { ( s ) }$ to $\left( s ^ { ' } \right)$ by taking action a. $\mathrm { V } ( s ^ { \prime } )$ is the value for the subsequent state $s ^ { ' }$ . However, the environment can be stochastic meaning the next state $s ^ { ' }$ is uncertain (modeled by the transition probabilities P). Hence,

$$
V ^ { \pi } ( s ) = \sum _ { s ^ { \prime } } P ( s ^ { \prime } | s , a ) [ R ( s , a , s ^ { \prime } ) + \gamma V ^ { \pi } ( s ^ { \prime } ) ]\tag{2.1.7}
$$

Similarly, incorporating the stochastic nature of the policy $( \pi )$ in the Bellman equation gives us:

$$
V ^ { \pi } ( s ) = \sum _ { a } \pi ( a , s ) \sum _ { s ^ { \prime } } P ( s ^ { ' } | s , a ) [ R ( s , a , s ^ { ' } ) + \gamma V ^ { \pi } ( s ^ { ' } ) ]\tag{2.1.8}
$$

Similarly, the bellman equation for $\mathrm { Q }$ function states that:

$$
Q ^ { \pi } ( s , a ) = R ( s , a , s ^ { ' } ) + \gamma Q ^ { \pi } ( s ^ { ' } , a ^ { ' } )\tag{2.1.9}
$$

$$
Q ^ { \pi } ( s , a ) = \sum _ { a } \pi ( a , s ) \sum _ { s ^ { \prime } } P ( s ^ { \prime } | s , a ) [ R ( s , a , s ^ { \prime } ) + \gamma Q ^ { \pi } ( s ^ { \prime } , a ^ { \prime } ) ]\tag{2.1.10}
$$

We know that action a is not chosen by policy $\pi ,$ only $a ^ { ' }$ and subsequent actions are selected by $\pi$ therefore it can be rewritten as:

$$
Q ^ { \pi } ( s , a ) = \sum _ { s ^ { \prime } } P ( s ^ { ' } | s , a ) \left[ R ( s , a , s ^ { ' } ) + \gamma \sum _ { a ^ { \prime } } \pi ( a ^ { ' } , s ^ { ' } ) Q ^ { \pi } ( s ^ { ' } , a ^ { ' } ) \right]\tag{2.1.11}
$$

The optimal policy $( \pi ^ { * } )$ is the one that gives the maximum state value.

$$
V ^ { * } ( s ) = \operatorname* { m a x } _ { \pi } V ^ { \pi } ( s )\tag{2.1.12}
$$

To find this optimal policy, Bellman optimality equation is used. It helps to find optimal Value and Q function recursively. The Bellman equation for optimality of value function is:

$$
V ^ { * } ( s ) = \operatorname* { m a x } _ { a \in \mathcal { A } } \sum _ { s ^ { \prime } } P ( s ^ { \prime } | s , a ) \left[ R ( s , a , s ^ { \prime } ) + \gamma V ^ { * } ( s ^ { \prime } ) \right]\tag{2.1.13}
$$

Similarly, the bellman optimality equation for maximum $\mathrm { Q }$ value, corresponding to the optimal policy $( \pi ^ { * } )$ is:

$$
Q ^ { * } ( s , a ) = \sum _ { s ^ { \prime } } P ( s ^ { \prime } | s , a ) \left[ R ( s , a , s ^ { \prime } ) + \gamma \operatorname* { m a x } _ { a ^ { \prime } } Q ^ { * } ( s ^ { \prime } , a ^ { \prime } ) \right]\tag{2.1.14}
$$

By utilizing the optimal value function $( V ^ { * } ( s ^ { ' } ) )$ , Bellman Optimal Q function can be expressed as:

$$
Q ^ { * } ( s , a ) = \sum _ { s ^ { \prime } } P ( s ^ { \prime } | s , a ) \left[ R ( s , a , s ^ { \prime } ) + \gamma V ^ { * } ( s ^ { \prime } ) \right]\tag{2.1.15}
$$

The Bellman optimality equations provide a theoretical foundation for solving MDPs. In traditional tabular reinforcement learning methods, the action-value function $Q ( s , a )$ is stored in a Q-table, where each entry corresponds to a specific stateaction pair. However, these methods rely on dynamic programming techniques $( \mathrm { e . g . }$ 2 value iteration, policy iteration) that assume full knowledge of the environment’s dynamics, i.e., $P ( s ^ { \prime } | s , a )$ . Moreover, Q-tables are only practical for environments with small, discrete state-action spaces. These drawbacks are addressed by employing neural networks to approximate $Q ( s , a )$ , forming the basis for Deep Reinforcement Learning as discussed below.

## 2.1.2 Deep Reinforcement Learning

Deep Reinforcement Learning (Deep RL) combines reinforcement learning with deep learning, using neural networks to approximate policies, value functions, or models of the environment. This enables RL agents to operate in high-dimensional or continuous state spaces where classical tabular methods become infeasible.

In reinforcement learning, algorithms that require environment dynamics $\left( P ( s ^ { ' } | s , a ) \right)$ are classified as model-based methods. However, in real-world scenarios, these dynamics are often unknown or hard to model accurately, leading to the widespread use of model-free methods, which learn directly from interaction with the environment. Model-free RL learn to approximate value functions or policies. It is broadly divided into two paradigms, namely Value-Based(Q- learning) Methods and Policy-Based (Policy Optimization) Methods [53]. Some Deep RL algorithms simultaneously learn a policy and a value function, known as actor-critic methods. Fig. 2.2 shows a non-

![](images/28588742f5af6b8c87f1ce5252766d9031f3fa9cac1e7d97130d47f064f35ba7.jpg)  
Figure 2.2: Model-free RL algorithms (non-exhaustive)

exhaustive list of Model-free RL algorithms. The methods highlighted in blue are discussed in the following sections.

## 2.1.2.1 Deep Q Networks

Deep Q-Network(DQN) replaces the Q-table with a neural network that maps states to Q-values for all actions. It is a neural network that approximates the Q value. If the DQN is parametrized by $\theta ,$ the network is trained to find the optimal parameters $\theta ^ { * }$ , which will provide the optimal Q function $( Q ^ { * } ( s , a ) )$ ), which satisfies the Bellman optimality condition (Eqn 2.1.14). Optimal policy can be found hereafter.

DQN is trained to approximate the Q values for all possible actions for the given input state. The training data for DQN is obtained from replay bufer, which stores the agent’s $( s , a , r , s ^ { ' } )$ transitions over several episodes. The loss function is mean squared error between the target value (Optimal Q value) and predicted Q value (by DQN).

$$
L ( \theta ) = \frac { 1 } { K } \sum _ { i = 1 } ^ { K } ( y _ { i } - Q _ { \theta } ( s _ { i } , a _ { i } ) ) ^ { 2 }\tag{2.1.16}
$$

Here, $y _ { i }$ is the target value defined as $y _ { i } = r _ { i } + \operatorname* { m a x } _ { a ^ { \prime } } Q _ { \theta ^ { \prime } } ( s _ { i } , a _ { i } )$ . The target value is estimated using another neural network, which is a time delayed copy of the main network, known as target network (represented by $\theta ^ { ' } )$ in order to have target stationarity. The parameters $\theta ^ { ' }$ are copied from the main network parameters (θ) every C steps (where C is a fixed update frequency). This introduces a delay in the target calculation to stabilize the learning process.

## 2.1.2.2 Policy Gradients & Actor-Critic methods

Policy-based methods allow us to directly learn or optimize the policy, without requiring the computation of an explicit value function such as the optimal Q function. Unlike value-based methods, which typically struggle with continuous action spaces due to the dificulty of maximizing over an infinite set of actions, policy-based approaches can handle both discrete and continuous action spaces.

In policy gradient methods, we typically learn a stochastic policy $\pi _ { \boldsymbol { \theta } } ( a | \boldsymbol { s } )$ , where the policy network (parameterized by θ) outputs a probability distribution over possible actions for a given state. The policy is trained using data sampled from episodes generated by following the current policy $\pi _ { \theta }$ . Starting from an untrained policy, the agent gradually learns to select actions that yield higher rewards for a given state.

The core idea behind policy gradient methods is to adjust the policy parameters so as to increase the probability of actions that lead to higher returns, and decrease the probability of actions that lead to lower returns. The objective function is defined as the expected return under the policy:

$$
J ( \theta ) = E _ { \tau \sim \pi _ { \theta } } [ G ( \tau ) ]\tag{2.1.17}
$$

where $G ( \tau )$ is the total discounted return along trajectory $\tau .$ , with $\tau$ sampled according to the policy $\pi \ ( \mathrm { i . e . , } \ \tau \sim \pi _ { \theta } )$ . This objective is maximized via gradient ascent (as $\theta = \theta + \alpha \nabla _ { \theta } J ( \theta ) )$ where the policy gradient is given by:

$$
\nabla _ { \theta } J ( \theta ) = E _ { \tau \sim \pi _ { \theta } } \left[ \sum _ { t = 0 } ^ { T - 1 } \nabla _ { \theta } \log \pi _ { \theta } ( a _ { t } | s _ { t } ) G ( \tau ) \right]\tag{2.1.18}
$$

Here, log $\pi _ { \boldsymbol { \theta } } ( a _ { t } | \boldsymbol { s } _ { t } )$ is the log probability of taking action $a _ { t }$ in state $s _ { t }$ under the current policy. Refer to [54] for the details of policy gradient. In practice, this expectation is estimated using average over N trajectories (sampled on-policy using $\pi _ { \boldsymbol { \theta } } )$ , also known as Monte Carlo approximation:

$$
\nabla _ { \theta } J ( \theta ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left[ \sum _ { t = 0 } ^ { T - 1 } \nabla _ { \theta } \log \pi _ { \theta } ( a _ { t } | s _ { t } ) G ( \tau ) \right]\tag{2.1.19}
$$

This algorithm is commonly known as REINFORCE. While REINFORCE and similar policy gradient methods provide a foundation for policy optimization, they often sufer from high variance in gradient estimates. A simple yet efective improvement is the REINFORCE with baseline method, where a baseline value is subtracted from the return to reduce variance. Typically, the baseline is chosen as the value function $V _ { \phi } ( s _ { t } )$ , which is estimated by a separate neural network (Value network) parameterized by $\phi .$ . This leads to the following update:

$$
\nabla _ { \theta } J ( \theta ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left[ \sum _ { t = 0 } ^ { T - 1 } \nabla _ { \theta } \log \pi _ { \theta } ( a _ { t } | s _ { t } ) ( G ( \tau ) - V _ { \phi } ( s _ { t } ) \right]\tag{2.1.20}
$$

## 2.1.2.3 Actor-Critic Algorithms

Actor-Critic (AC) frameworks combine the strengths of both policy-based and valuebased methods to address the limitations of pure policy gradient algorithms such as

REINFORCE. This architecture (as shown in Fig. 2.3) uses two main components:

• Actor refers to a policy network that learns a parameterized policy $\left( \pi _ { \boldsymbol { \theta } } \right)$ to select action in each state.

• Critic is a value network (similar to REINFORCE with baseline) that estimates the expected return (for example, using the value function $V _ { \phi } ( s )$ or action-value function $Q _ { \phi } { \left( s , a \right) } )$ to guide the actor’s learning.

The actor is updated using policy gradients with critic value as the baseline (to reduce variance), similar to REINFORCE with baseline (Eqn. 2.1.21). But a major diference is that Actor Critic methods are online (updated after each timestep in episode) and therefore return $G ( \tau )$ in Eqn. 2.1.21 is replaced by the bootstrapped estimate as follows:

$$
\nabla _ { \theta } J ( \theta ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left[ \sum _ { t = 0 } ^ { T - 1 } \nabla _ { \theta } \log \pi _ { \theta } ( a _ { t } | s _ { t } ) ( r + \gamma V _ { \phi } ( s _ { t ^ { \prime } } ) - V _ { \phi } ( s _ { t } ) \right]\tag{2.1.21}
$$

Critic is updated using Temporal Diference (TD) error which is computed as the diference between the bootstrapped estimate and the current value prediction:

$$
J ( \phi ) = r + \gamma V _ { \phi } ( s _ { t ^ { \prime } } ) - V _ { \phi } ( s _ { t } )\tag{2.1.22}
$$

![](images/2e85b1601294598a7bb669112f8bd223890985957805ccdcb185db292ffc7e4b.jpg)  
Figure 2.3: Illustration of actor-critic framework in RL

Actor-critic methods eficiently handle continuous action spaces and provide more stable, lower-variance learning. They enable online, stepwise updates at every timestep, improving sample eficiency and convergence speed.

This section focuses on three prominent of-policy actor-critic algorithms: Deep Deterministic Policy Gradient (DDPG), Twin Delayed Deep Deterministic Policy Gradient (TD3), and Soft Actor-Critic (SAC). Each of these builds upon the limitations of the previous, progressively improving stability and sample eficiency.

(i) DDPG [55] extends Actor-Critic to continuous action spaces using deterministic policies and deep neural networks. It combines elements from DQN and deterministic policy gradients. Specifically, DDPG uses:

• A deterministic actor network (ϕ) and a critic network (θ).

• A target actor(parametrized by $\phi ^ { ' } )$ and target critic(parametrized by $\theta ^ { ' } )$ , which are slowly updated to stabilize training, similar to the target networks in DQN.

As the name suggests, DDPG learns a deterministic policy $( \mu )$ , therefore exploration noise is added to the action : $a = \mu _ { \phi } ( s ) + N$

Critic Loss: The critic is basically a DQN (value network) which predicts the Q value $( Q _ { \theta } ( s , a ) )$ . It is trained on transitions sampled from the replay bufer, to minimize the diference between actual and target $\mathrm { Q }$ value (refer eqn. 2.1.16). The target Q value $( y _ { i } )$ in eqn. 2.1.16 is obtained using the outputs from target actor $\left( \phi ^ { ' } \right)$ and target critic $( \boldsymbol { \theta } ^ { \prime } )$ networks as shown in Eqn. 2.1.2.3. The bellman target is modified to ensure stable convergence in actor-critic learning. Thus critic loss function becomes:

$$
L ( \theta ) = \frac { 1 } { K } \sum _ { i } \left( r _ { i } + \gamma Q _ { \theta ^ { \prime } } ( s _ { i } , \mu _ { \phi ^ { \prime } } ( s _ { i ^ { \prime } } ) ) - Q _ { \theta } ( s _ { i } , a _ { i } ) \right) ^ { 2 }\tag{2.1.23}
$$

Actor Loss: The actor is a policy network that learns the optimal policy by maximizing the following main objective function:

$$
J ( \phi ) = \frac { 1 } { K } \sum _ { i } Q _ { \theta } ( s _ { i } , a )\tag{2.1.24}
$$

(ii) TD3 [56] builds on DDPG and was introduced to address DDPG’s overestimation bias and instability through three key enhancements:

1. Clipped double Q learning: Uses two $\mathrm { Q }$ -networks(2 main critics: $\theta _ { 1 }$ and $\theta _ { 2 }$ , and 2 target critics: $\theta _ { 1 ^ { ' } }$ and $\theta _ { 2 ^ { ' } } )$ and takes the minimum(of target values $Q _ { \theta _ { 1 } ^ { \prime } } ( s ^ { \prime } , a ^ { \prime } )$ and $Q _ { \theta _ { 2 } ^ { \prime } } ( s ^ { ' } , a ^ { ' } ) )$ for target updates, to prevent overestimation bias.

2. Delayed policy updates: Updates the actor less frequently than the critics to stabilize training.

3. Target policy smoothing: Adds noise to the target actions to prevent overfitting.

(iii) Soft Actor-Critic (SAC) [57] is a state-of-the-art of-policy actor-critic algorithm that incorporates entropy regularization to promote exploration and improve learning stability. The actor (policy) objective includes an entropy term and the objective can be expressed as:

$$
J _ { \pi } ( \phi ) = \frac { 1 } { K } \sum _ { i } Q _ { \theta } ( s _ { i } , a ) - \alpha [ l o g \pi _ { \phi } ( a | s _ { i } ) ]\tag{2.1.25}
$$

Here, $l o g [ \pi _ { \phi } ( a | s _ { i } ) ]$ is the entropy term, where higher entropy corresponds to more randomness in action selection. The coeficient $\alpha ,$ known as the temperature parameter, controls the trade-of between maximizing expected reward and policy entropy, thereby balancing exploration and exploitation.

SAC employs a stochastic policy, where the actor outputs a probability distribution over actions rather than deterministic actions. To reduce overestimation bias in value estimation, SAC uses two Q-networks (twin critics), similar to TD3, but difers significantly in its use of entropy regularization and stochastic policies.

## 2.2 Transformers for Sequence Modeling

In this work, Transformer-based models are employed to learn tract-specific tracking policies. The GPT-based actor is trained on RL-generated trajectories, enabling it to refine and fuse the underlying RL policies for tractography by leveraging its strong pre-training and generalization capabilities, further detailed in Chapters 3 and 4.

Prior to transformer models, Recurrent Neural Network (RNN) [58] and Long short-term memory (LSTM) [59] networks were widely popular for sequential modeling. However, they processed input one timestep at a time, maintaining a hidden state to capture the information from previous timesteps in the sequence. This led to a major drawback of sequential computation to preserve the order of elements, which prevented the training parallelization. Other important issues included the dificulty in learning long range dependencies, and the vanishing or exploding gradient problem.

Transformers introduced in [60], removed the need for recurrent connections, by utilizing attention mechanisms to model relationships between all elements of the sequence as described below. They can model long-range dependencies, while also enabling parallelization for faster training.

## 2.2.1 Self-Attention Mechanism

Self attention computes a representation of each element or token with respect to all other tokens in the sequence. To achieve this, first, each input token is mapped to three d-dimensional vectors, Query $( q )$ , Key (k), and Value (v), through learnable linear transformations. For a given $i ^ { t h }$ token, say $t _ { i }$ with query vector q, attention scores are computed by comparing q with key vectors $k _ { t }$ of all tokens $t = 1 , 2 , . . . . , T$ in the sequence. The attention score between $q$ and $k _ { t }$ is computed using a scaled dot product:

$$
s _ { t } = \frac { q ^ { T } k _ { t } } { \sqrt { d } }\tag{2.2.26}
$$

The scores $\{ s _ { t } \} _ { t = 1 } ^ { T }$ are passed through a softmax function to obtain a distribution over all tokens in the sequence:

$$
a _ { t } = \frac { e x p ( s _ { t } ) } { \sum _ { l = 1 } ^ { T } e x p ( s _ { l } ) }\tag{2.2.27}
$$

This is also known as Scaled dot product Attention (SDPA). Finally, the resulting

representation of the token $t _ { i }$ is given by the attention $\left( \boldsymbol { a } _ { t } \right)$ weighted sum of its value vectors $\left( v _ { t } \right)$ . These representations preserve the sequential information by avoiding recurrence, and enable parallelization.

## 2.2.2 Transformer Architecture

The transformer follows an encoder-decoder framework. The encoder maps the input sequence into an intermediate representation, and the decoder generates an output sequence conditioned on the encoder’s output representations/embedding and the previously generated outputs.

![](images/57571b7e665ccec9fdf21c0d731f789dadccc8edc801fe089b7020fd9c700704.jpg)  
Figure 2.4: Illustration of Transformer model architecture [60]

The encoder is composed of N stacked layers, each consisting of a multi-head selfattention mechanism and a position-wise feedforward neural network, with residual connections followed by layer normalization (Add & Norm) applied after each sublayer, as shown in Figure 2.4.

Before entering the encoder stack, the input tokens are first passed through a learned input embedding layer to obtain dense vector representations. Since the Transformer architecture does not inherently capture the order of tokens, additional positional information is combined with the input embeddings. The positional encodings are either fixed (using sinusoidal functions) or learned during training. The sinusoidal positional encoding utilized in [60] is given as follows:

$$
P E _ { ( p o s , 2 i ) } = \sin { \left( \frac { p o s } { 1 0 0 0 0 ^ { \frac { 2 i } { d _ { \mathrm { m o d e l } } } } } \right) }\tag{2.2.28}
$$

$$
P E _ { ( p o s , 2 i + 1 ) } = \cos \left( { \frac { p o s } { 1 0 0 0 0 ^ { \frac { 2 i } { d _ { \mathrm { m o d e l } } } } } } \right)\tag{2.2.29}
$$

where pos denotes the position index, i denotes the dimension index, and $d _ { \mathrm { m o d e l } }$ is the dimensionality of the embeddings. The even dimensions use the sine function, and odd dimensions use the cosine function. This continuous and periodic encoding scheme enables the model to capture both absolute and relative positional relationships between tokens.

The resulting input representations, after adding input embeddings to positional encodings, are then fed into the encoder layers. The two core components of each encoder layer are:

## 2.2.2.1 Multi-head Self-attention

It runs multiple attention mechanisms or “heads” in parallel. Each head learns its own linear projections for queries, keys, and values. This allows for learning diferent types of relationships. The outputs of all heads are concatenated and passed through a final linear transformation to produce the output of the multihead attention module. This enables the model to jointly attend to information from diferent representation subspaces at diferent positions. Formally, the multi-

head attention is defined as:

$$
\mathrm { M u l t i H e a d } ( Q , K , V ) = \mathrm { C o n c a t } ( \mathrm { h e a d } _ { 1 } , \mathrm { h e a d } _ { 2 } , \dots , \mathrm { h e a d } _ { h } ) W ^ { O }\tag{2.2.30}
$$

where each attention head is computed as:

$$
\mathrm { h e a d } _ { i } = \mathrm { S D P A } ( Q W _ { i } ^ { Q } , K W _ { i } ^ { K } , V W _ { i } ^ { V } )\tag{2.2.31}
$$

Here:

$Q , K , V$ are the matrices of queries, keys, and values, respectively.

$W _ { i } ^ { Q } , W _ { i } ^ { K } , W _ { i } ^ { V }$ are learnable projection matrices for the i-th head.

$W ^ { O }$ is the learnable projection matrix applied after concatenating the outputs of all heads.

The SDPA function refers to scaled dot-product attention function (eq: 2.2.27).

## 2.2.2.2 Position-wise Feedforward Network

A shared feedforward neural network (FFNN) is used for each token(at each timestep) in the sequence. It captures the non-linear relationships in the data. Each FFNN consists of two dense layers with ReLU activation function, and the parameter weights are shared across all the timesteps in the sequence and diferent across the ‘N’ encoder blocks.

The decoder has a similar structure to the encoder, consisting of N stacked layers. Each decoder layer has three main sub-components: a masked multi-head self-attention mechanism, a multi-head cross-attention mechanism over the encoder outputs, and a position-wise feedforward network. As in the encoder, each sub-layer is followed by residual connections and layer normalization (Add & Norm).

The input to the decoder is the target sequence, passed through a learned embedding layer with positional information added. The masked self-attention ensures that each position in the decoder can only attend to earlier positions, maintaining the autoregressive property during generation. The cross-attention layer allows the decoder to attend to the encoder outputs, enabling it to condition its predictions on the encoded input sequence, as shown in Fig. 2.4.

## 2.2.3 Generative Pre-trained Transformer

The Generative Pre-trained Transformer (GPT) [61] is a decoder-only variant of the Transformer architecture, designed specifically for generative tasks. Unlike the original Transformer, which includes both encoder and decoder blocks (as shown in Fig. 2.4), GPT consists only the decoder stack and relies on causal (unidirectional) self-attention to autoregressively predict the next element in a sequence. This means that they generate outputs one element at a time by conditioning on previously generated elements. The model operates within a fixed context length, defined during training, which limits the number of previous tokens it can attend to when predicting the next token. In contrast, BERT (Bidirectional Encoder Representations from Transformers) [62] uses only the Transformer encoder and employs bidirectional selfattention to model full context, making it well-suited for discriminative tasks like classification, question answering, and token-level labeling.

GPT was originally introduced for language modeling, where the objective is to predict the next word (token) in a sentence given the preceding context. However, its autoregressive structure and flexible sequence modeling capabilities make it broadly applicable to a variety of sequential data domains, such as time-series forecasting [63], protein sequences [64], and code generation [65].

Subsequent versions such as GPT-2, GPT-3, and GPT-4 have demonstrated that increasing the model size (depth, width, and number of parameters), training data scale, and computational resources leads to substantial gains in generalization, reasoning ability, and few-shot learning. These models are pretrained on large unlabelled datasets and can be fine-tuned on smaller, task-specific datasets to improve performance on downstream applications.

While pretraining equips the model with a broad understanding of sequential patterns, fine-tuning helps specialize it for specific tasks, improving task performance, factual grounding, and alignment with human preferences. The pretraining and fine-tuning stages of GPT training are discussed in detail below:

## 2.2.3.1 Unsupervised Pretraining

GPT models are pretrained in an unsupervised manner using the Causal Language Modeling (CLM) objective, commonly known as next-token prediction. Training is conducted in an autoregressive manner on large-scale unlabelled sequential data, using the negative log-likelihood loss to predict the next token in a sequence, given all previous tokens. Thus given a sequence of tokens ${ \boldsymbol { x } } = ( x _ { 1 } , x _ { 2 } , . . . , x _ { T } )$ , the model is trained to minimize the following loss function:

$$
\mathcal { L } _ { \mathrm { C L M } } = - \sum _ { t = 1 } ^ { T } \log P ( x _ { t } \mid x _ { 1 } , . . . , x _ { t - 1 } )\tag{2.2.32}
$$

The masked self-attention mechanism ensures that, during training, each token can only attend to previous tokens in the sequence. This preserves the causal (leftto-right) structure required for generative modeling. This unsupervised pretraining approach allows the model to learn transferable representations and temporal dependencies, which can later be adapted to various downstream tasks through finetuning.

## 2.2.3.2 Supervised Fine-tuning

Fine-tuning is performed after pretraining to adapt the model to specific downstream tasks. This typically involves appending task-specific heads (e.g., linear layers) on top of the pretrained transformer and training the model on labeled data typically using supervised loss functions. Depending on the task and the size of the available dataset, either the entire model or only a subset of layers may be updated.

This study focuses on supervised fine-tuning (of GPT) for tractography, as detailed in Chapters 3 and 4.

## Chapter 3

## Tract-specific Tractography using GPT-based RL Policy Refinement

## 3.1 Introduction

Tractography is a complex sequential decision-making process. Recent research has increasingly leveraged machine learning and deep learning (DL) approaches to improve tractography accuracy. In particular, Deep Reinforcement Learning (DRL) has emerged as a promising direction, as it does not rely on unreliable or hard-toobtain ground truth data for training. However, there remains significant room for performance improvement.

## Problem Statement

To develop a tract-specific framework for white matter fiber tractography. The input is difusion MRI data, and the goal is to reconstruct white matter fiber trajectories belonging to a targeted anatomical tract. Each fiber is represented as an ordered sequence of 3D points $( x , y , z )$ that follows biologically plausible streamline propagation constraints.

Transformers have recently demonstrated outstanding performance across diverse domains, including language modeling [60], image recognition [66], time-series forecasting [63], and protein structure prediction [64]. Their strong ability to model long-range dependencies and contextual information makes them well-suited for sequence prediction tasks, such as mapping neural pathways in tractography. Building on these successes, Tract-RLFormer integrates the generalization, transfer learning, and autoregressive capabilities of transformers (e.g., GPT models) to enhance the performance of Deep RL policies for tractography.

## 3.2 Data Specifications and Pre-processing

Tractography involves using Difusion MRI data to generate white matter fiber tracts. Tractography datasets include difusion MRI data of one or more subjects and their corresponding white matter tracts.

In this study, we utilize 3 datasets, namely TractoInferno [42], Human Connectome Project (HCP) [67], and the ISMRM 2015 Tractography Challenge dataset [29] as described in Table 3.1. ISMRM is a synthetic phantom dataset consisting of reference tracts for 25 distinct white matter bundles. HCP provides in-vivo difusion MRI data from healthy adult subjects. TractoInferno is a synthetic benchmark dataset constructed using four tractography algorithms that provide reference tracts for 30 bundles for machine learning and evaluation tasks. For the HCP dataset, we use the reference white matter tracts released as part of [28], which provides reference tract segmentations of 72 bundles for 105 subjects of the HCP young adult dataset.

Difusion MRI undergoes a series of processing steps before it can be efectively used in tractography algorithms. This process typically consists of two main stages, as previously detailed in Sections 1.2.2 and 1.2.3:

1. Preprocessing, which includes denoising, motion and eddy current correction, and bias field correction.

2. Signal modeling and feature extraction, where fiber orientation distributions (fODFs) are estimated and local difusion features like fODF peaks are extracted as inputs to the tractography pipeline.

Table 3.1 summarizes the preprocessing steps that have already been applied to each dataset. The fODF and peaks computation was performed using scilpy [68]. Following preprocessing, fODFs are estimated using constrained spherical deconvolution (CSD) [14] (see Section 1.2.3.2). Each fODF is represented using 45 coeficients of 8th-order spherical harmonics (SH). From the resulting fODFs, five peaks are extracted per voxel, representing the dominant fiber directions. Table 3.1 also presents the acquisition details of the datasets, along with the number of subjects used for training and testing in this work.

Table 3.1: Details of the difusion datasets used in our experiments.
<table><tr><td>Dataset</td><td>Subjects</td><td>DWI data</td><td>Distortion Corrections</td></tr><tr><td>TractoInferno [42]</td><td>284a</td><td>b=1000  ${ \overline { { s / m m ^ { 2 } } } } ;$  resolution=1mm isometric</td><td>N4 bias field; eddy-current; head-motion</td></tr><tr><td>HCP [67]</td><td>1200b</td><td>b=1000/2000/3000  $s / m m ^ { 2 } ;$  270 di- rections; reso- lution=1.25mm isometric</td><td>EPI; eddy-current; subject-motion</td></tr><tr><td>ISMRM [29]</td><td>1c</td><td>b=1000  $\overline { { s / m m ^ { 2 } } } ;$  32 directions; resolu- tion=2mm isometric</td><td>Eddy currents; head mo- tion (by our preprocessing)</td></tr></table>

<sup>a</sup> Train IDs: 1030, 1079, 1119, 1180, 1198; Test IDs: 1160, 1078, 1061, 1159, 1171  
<sup>b</sup> Test IDs: 930449, 959574, 992774, 987983  
<sup>c</sup> Only one subject available; no ID needed

## 3.3 Proposed Method

Tract-RLFormer is an iterative policy learning framework for tract-specific generation, delineated as a five-step process (see Fig. 3.1). In this framework, we start by training an RL agent (TD3 [56]) to learn a policy by exploration (within the tracking mask) to generate a tract of interest. We call it as level-1 policy. Using this initial policy, the agent interacts with the (tracking) environment by taking actions (tracking steps). The agent’s experience (policy rollouts) is collected and sampled to train a refined version of the policy, by the T-RLF model, which learns in a data-driven manner through general pre-training and tract-specific fine-tuning. This study focuses on seven principal white matter (WM) tracts: Corpus Callosum (CC), left and right Pyramidal (PYT), Arcuate Fasciculus (AF), and Cingulum (CG) Tracts, shown in Fig. 1.4. The selection of these seven tracts is based on their clinical significance and frequent analysis as suggested in [49, 52]. To conduct such tract-specific training and generation, we first compute a tracking region of interest (mask) tailored for each tract using our Mask Refinement Module (MRM), described in 3.3.1.2. Following this, we proceed with the five sequential steps depicted in Fig. 3.1, detailed in subsequent subsections.

## 3.3.1 Tract-RLFormer Framework

## 3.3.1.1 RL Environment Setup

The reinforcement learning (RL) environment for tractography is designed to: (1) initialize seed points within the white matter mask to initiate tracking; (2) define the state at each voxel within the tracking region; (3) assign rewards and determine the next state based on the agent’s current state and actions; and (4) constrain the tracking process both spatially, within the brain mask and anatomically, according to plausible fiber length and bend. The RL environment for tractography was first proposed in [48], and since then it has been employed in subsequent works that employ Deep RL for tractography [49,50]. It is implemented using OpenAI Gym [53]. The diferent elements of the environment are:

• State: The state at a given voxel includes the 45 Spherical Harmonic Coeficients $( 8 ^ { t h }$ order) of that voxel and its six immediate neighbours, along with their white matter mask values. This is followed by appending last four tracking directions(actions) resulting in the final state vector of length $4 5 \times 7 + 7$ $+ 4 \times 3 = 3 3 4 .$

• Action: The action is a 3D vector $a _ { t }$ that determines the next tracking direction.

• Reward: It is the product between predicted action’s alignment (dot product) with the most aligned fODF peak and it’s alignment(dot product) with the last tracked streamline/fiber segment. It is computed as:

$$
r _ { t } = \left| \operatorname* { m a x } _ { \vec { p _ { i } } } \left( \vec { p _ { i } } \cdot \vec { a } _ { t } \right) \right| \times \left( \vec { a } _ { t } \cdot \vec { u } _ { t - 1 } \right)\tag{3.3.1}
$$

![](images/7a9b7b4dc9e4aa66f08f5864ae687895da68d93613d7620ff214f14187990380.jpg)  
Figure 3.1: Overview of the proposed Tract-RLFormer framework. (a) An RL agent $\scriptstyle ( \pi _ { \theta } )$ interacts with the environment (E) to learn an optimal level-1 policy $( \pi _ { \theta o p t } )$ . (b) This policy is used to generate tract-specific roll-outs, denoted as ’experience replay’. (c) and (d) illustrate the ofline, auto-regressive training of the proposed Tract-RLFormer $\phi ,$ referred to as T-RLF, over these rollouts. In (c), T-RLF undergoes general pre-training, while in (d) it is fine-tuned to learn an optimal tract-specific policy $( \pi _ { \phi o p t } )$ . (e) shows the testing phase, where T-RLF, which has learned the new level-2 policy $( \pi _ { \phi _ { o p t } } )$ , performs tracking in environment $E$ to produce the desired tract. Training and tracking steps are shown in yellow and orange backgrounds, respectively.

where, $\vec { p _ { i } }$ is the fODF peak, $\vec { a } _ { t }$ is the action taken at timestep t, and $\vec { u } _ { t - 1 }$ is the previously tracked step at timestep t − 1.

• Episode termination criteria: An episode terminates when the tracking exceeds the tracking mask, the fiber streamline length exceeds 200 mm, or the fiber curvature exceeds the maximum angle of $3 0 ^ { \circ }$ or ${ 6 0 } ^ { \circ }$ (depending on RL agent’s training algorithm).

## 3.3.1.2 Mask Refinement Module

In this study, we generate masks for 7 tracts: the Corpus Callosum (CC) and the left and right components of three bilateral tracts, namely the Pyramidal Tract (PYT), Arcuate Fasciculus (AF), and Cingulum (CG).

As discussed in Chapter 1, a tracking mask is required to constrain tractography to white-matter regions, and in our case, to a specific tract of interest. Instead of simply relying on an atlas-based mask, which can be inaccurate across subjects, we trained a neural network to generate subject-wise tract-specific masks that account for inter-subject anatomical variability for reliable tractography.

For a given tract, its reference streamlines are obtained from RecobundlesX Atlas [69]. The atlas reference tract is aligned using ANTs [21] to the given subject space, and then converted into a binary mask, having value of 1 for voxels from which atleast one reference streamline passes, and 0 for the rest. This mask is dilated by 5mm and input to MRM, which prunes the dilated mask to estimate the tract-specific mask. For each voxel in the dilated mask, the 45 SH coeficients and dilated mask value of the voxel and its 6 immediate neighbour voxels (resulting in a vector of size $4 5 \times 7 + 7 = 3 2 2 )$ are input to the MRM network, which outputs the probability of retaining that voxel in the predicted mask. Voxels with an output probability greater than 0.5 are kept in the predicted tract-specific mask, while those with lower probabilities are eliminated. Fig. 3.2 depicts the entire process of mask generation using MRM.

Architecture and Training details: The MRM is a Fully connected neural network comprising three hidden layers with 512, 256, and 128 neurons, respectively, and an input dimension of 322. Each layer employs a ReLU activation function and is followed by batch normalization and a dropout layer (0.5). The output layer has a sigmoid activation function, which outputs the probability of retaining voxel in the predicted mask. Training is performed in a voxel-wise manner, using Binary Cross Entropy as the loss function to compare the predicted mask value with the ground truth for each voxel. The model was trained on 50 subjects randomly selected from the TractoInferno dataset over 100 epochs. The resulting mask is then dilated by 1mm to produce the final refined tracking mask for the given subject. In Fig. 3.2, the apparent disconnected components in the various mask images are due to 2D slice visualization of the volumetric mask. The mask is connected in 3D space across adjacent slices. Visualization in diferent orientations confirms that the components are connected and anatomically consistent.

![](images/30600bfc6d4d8d8b77de72d54af1058ce58e8fb6efaed8edd5a9f455a5ab8d79.jpg)  
Figure 3.2: Mask Refinement Module for generating tract-specific masks. (1) The atlas reference tract is aligned to subject space. (2) Streamlines are converted into a binary threshold mask (3) The mask is dilated by 5 mm and pruned voxel-wise by the MRM. (4) Mask prediction uses as input the 45 SH coeficients together with the dilated mask value (0/1) for the voxel itself and its six neighbouring voxels. (5) The pruned mask is dilated by 1 mm to produce the final tract-specific mask.

## 3.3.2 RL Policy Learning

We utilize TD3 (Twin Delayed Deep Deterministic Policy Gradient) algorithm to train 7 tract-specific agents. It belongs to a class of Actor-Critic algorithms in RL (refer Section 2.1.2.3) and consists of an Actor and two Critic networks (along- with their time delayed target networks). We adopt the networks similar to [48], where actor and critic networks are both fully-connected neural networks with two ReLU activated hidden layers of 1024 neurons each. The actor has a 334 dimensional input layer and 3-neuron tanh activated output layer to predict the 3-D action, while the critic has a 337 dimensional input layer (corresponding to both 334-D state and 3-D action as input) and a single neuron tanh output layer predicting the Q-value for a given state-action pair. However, unlike [48], where RL agents are trained for whole-brain tractography, we perform tract-specific training to obtain one policy for each tract of interest. We train TD3 agent on tract-specific masks obtained from MRM. Each tract-specific RL agent is trained on five randomly selected training subjects from the TractoInferno dataset (1030, 1079, 1119, 1180, and 1198), for 50 batches (of 4096 episodes each) per subject, resulting in a total of 1,024,000 (250\*4096) training episodes.

We train the TD3 agent in 5 diferent instances of the environment specified by each subject’s distinct difusion data, fODF peaks, and tracking mask. Training is conducted at 7 seeds per voxel and a step size of 0.375mm, with fiber lengths between 20mm and 200mm. The maximum possible episode length is set to 530, corresponding to a 200 mm fiber and a 0.375 mm step size. Other hyper-parameters include: learning rate: 8.56e-06, Discount factor (γ): 0.776, and Exploration noise $\left( \sigma _ { t r a i n } \right)$ : 0.334.

## 3.3.3 Tract-RLFormer for Policy Refinement

RL agents learn a TD3 policy $( \pi _ { \theta o p t } )$ , which can be used to conduct tract-specific tractography. Using rollouts from this level-1 policy, Tract-RLFormer (T-RLF) learns a refined policy $( \pi _ { \phi o p t } )$ , by training on trajectories derived from the policy roll-outs of seven trained tract-specific TD3 agents. Each trajectory consists of sequence of state, action, and return-to-go of an episode. It is represented as $\tau = ( R _ { 0 } , s _ { 0 } , a _ { 0 } , R _ { 1 } , . . . , R _ { T } , s _ { T } , a _ { T } )$ , as shown in Fig. 3.3, where $R _ { t }$ is the scalar sum of rewards from timestep t to the episode’s end, $s _ { t }$ is the 334-dimensional state vector, and $a _ { t }$ is the 3-dimensional action as discussed in Section 3.3.1.1.

T-RLF is a GPT based model that is trained on these RL-derived trajectories in a two-stage process. (a) Initially, the Tract-RLFormer undergoes a generic, tractagnostic pre-training, (b) This is followed by fine-tuning for the downstream task of tract-specific generation. Together, these constitute the next three steps (out of five), namely training data generation (Fig. 3.1(b)) and the two-stage training process (Fig. 3.1(c, d)) of T-RLF. Each step of the refinement process is discussed in detail below:

## 3.3.3.1 Training Data Generation for T-RLF

For each tract, their trained RL (TD3) agent is used to initiate tracking within its designated masks (obtained from MRM) on the 5 randomly selected training subjects from TractoInferno dataset (Subject IDs: 1030, 1079, 1119, 1180, and 1198). As the policy traverses diferent voxels within the tracking mask, the corresponding trajectories for those transitions are saved as $< R ,$ s, a> tuple. Trajectories shorter than 47 transitions, corresponding to a fiber length of approximately 20 mm given the 1 mm isotropic voxel resolution of the TractoInferno dataset, are filtered out. From the remaining trajectories, 10,000 are selected per subject—comprising 5,000 of the longest trajectories and 5,000 chosen at random—resulting in a total of 50,000 trajectories per tract. This forms the tract-specific trajectories dataset, which is used for model finetuning. From the full set of 350,000 trajectories (7 tracts × 50,000 each), a subset of 150,000 is selected to create a mixed, tract-agnostic dataset, denoted as $\tau _ { m i x }$ . This subset includes 75,000 of the longest trajectories and 75,000 randomly selected ones, and is used for generic pre-training.

## 3.3.3.2 Architecture of T-RLF

The network architecture of the model is based on a decoder-only Transformer framework (GPT), consisting of 4 sequential decoder layers. It supports a maximum context length of 40 tokens, enabling it to process and generate sequences up to this length at a time. Each timestep processes 3 tokens one each for state, action and return-to-go. The model includes a dedicated learned embedding layer of 128 dimensions (as shown in Fig. 3.4) for each component of the trajectory: state (s), action (a), and return-to-go (R). This transforms each input timestep’s state, action and Return-to-go into vectors of dimension 128, which is combined with learned positional embeddings (of dimension 128) to retain order information. The maximum possible episode length $( m a x \_ e p \_ l e n )$ controls length of episode. It is set to 530 because the maximum length of a fiber is 200mm, equivalent to 530 steps for a

![](images/e54bea624d1febb263bd00c2a10a25b05f7381f1baef84c2203af451b6305d4d.jpg)  
Figure 3.3: Data Representation for T-RLF: Tract specific policy refinement using a trajectorybased approach in an RL agent’s experience space. The figure illustrates a k length fiber streamline f in human brain voxel space, represented as a trajectory $\tau = ( R _ { 0 } , s _ { 0 } , a _ { 0 } , R _ { 1 } , s _ { 1 } , a _ { 1 } , . . . , R _ { k } , s _ { k } , a _ { k } )$ Each point in the streamline corresponds to a state, action, and return-to-go tuple at time-step t.

TractoInferno subject (as 1 step corresponds to 0.375 mm). If an episode exceeds 530 timesteps, it is truncated to this length. Each decoder layer contains a masked multi-head self-attention mechanism with 1 attention head, followed by a positionwise feedforward neural network. Layer normalization and residual connections are employed around each sub-layer, like GPT. After processing through the decoder layers, the output is passed through an output embedding layer, from which we obtain the predicted action of dimension (3, 1). Dropout with a rate of 0.1 is applied to mitigate any overfitting during training.

## 3.3.3.3 Training Procedure of T-RLF

Tract-RLFormer model is trained on <state, action, return-to-go> trajectories sampled from TD3 agent (detailed in Section 3.3.3.1). It follows a two-stage training process: first, a general pre-training phase on a mixed tract dataset $\left( \tau _ { \mathrm { m i x } } \right)$ , followed by tract-specific fine-tuning on individual tract datasets $( \tau _ { i } )$

The first three decoder layers are pre-trained over 0.15 million mixed trajectories (taken from $\tau _ { m i x } )$ , containing a total of 30 million transitions for 30 iterations. Later the $4 ^ { t h }$ decoder layer is fine-tuned on the tract-specific trajectories bufer for 10 additional iterations. In each iteration, the model undergoes 10,000 training steps, each processing a batch\_ $s i z e \mathrm { = } 1 2 8$ number of K-length trajectories.

![](images/e8e426f3ebec23a2ab9598b09409d4664b87f2ef646c97ab6c05467bd9165a1d.jpg)  
Figure 3.4: T-RLF training: Visual representation of training Tract-RLFormer for action prediction at time-step t, using context information from K length fiber. The input sequence tuples $< R , s .$ $a >$ are causally masked from $a _ { t }$ onwards and processed through embedding layers $e m b _ { R } ,$ emb<sub>s</sub>, and em $\boldsymbol { { b } } _ { a } ,$ with a learnable positional encoding layer (PE). Embeddings are processed by L decoder blocks $( L = 3$ for pre-training, $L = 4$ for fine-tuning), incorporating Multi-Head Attention (MHA) and Multi-Layer Perceptron (MLP), to generate predicted action $\hat { a _ { t } }$

A batch of 128 tokens of $< R _ { t } , s , a > \mathrm { a r e }$ sampled from training data (τ) and stacked for a context length $( K = 4 0 )$ and fed as an input to the T-RLF. It passes through an embedding layer with 128 dimensions, and positional encoding is added, resulting in a $( 1 2 8 \mathrm { ~ x ~ } 1 2 0 \mathrm { ~ x ~ } 1 2 8 )$ matrix and is processed by the 4 decoder layers with causal masking (Fig. 3.4). The decoder output is mapped through an output embedding layer to predict the action. Unlike the TD3 agent, T-RLF does not interact with the environment during its training process. Instead, it is trained entirely in an ofline mode using only trajectory datasets $( \tau ^ { \prime } \mathrm { s } )$ . For the context length K, a 5-step loss (accounting for current and 2 steps in both forward and backward directions) is computed, aggregating the angular diference between predicted and actual action at each time-step.

$$
L = \sum _ { t = 2 } ^ { K - 2 } \left( \sum _ { i = - 2 } ^ { 2 } \cos ^ { - 1 } \left( \mathbf { a } _ { t + i } \cdot \hat { \mathbf { a } } _ { t + i } \right) \right)\tag{3.3.2}
$$

The learning of weights for $\pi _ { \phi o p t }$ is facilitated by this 5-step loss function, in order to generate more efective and robust actions. Here, $\vec { a } _ { t + i }$ and $\vec { \hat { a } } _ { t + i }$ are the true and predicted actions at $( t + i ) ^ { t h }$ timestep respectively.

Similar to [70], T-RLF training is conditioned to generate action $\left( \boldsymbol { a } _ { t } \right)$ using return $( R _ { t } )$ at each timestep. During inference, $R _ { t }$ is initialized to an expert return value or the longest trajectory return. In our case, the longest trajectory length is 292, and since the maximum possible reward at each timestep is 1, we initialize $R _ { t }$ to 300 (∼1x of expert return). This was experimentally verified among various values of $R _ { t }$ : 100, 200, 300, 500, and 600. For model training, we employed the AdamW optimizer, set with a learning rate of 1e-4 and a weight decay of 1e-4.

Importantly, all tract-specific T-RLF models share the same pre-trained backbone and difer only in their fine-tuning phase. This fine-tuning stage specializes the model for the downstream task of generating accurate and anatomically plausible streamlines for a specific white matter tract. The efectiveness of this two-stage training approach is evaluated in an ablation study, highlighting the contribution of both pre-training and fine-tuning to the model’s performance and generalization capabilities across diferent tracts in section: 3.4.4).

## 3.3.4 Tract generation and cleaning

The final step in our tractography pipeline involves using the trained T-RLF models to perform tracking, followed by cleaning the resulting tracts. Having learnt the refined policy $( \pi _ { \phi o p t } )$ , T-RLF can function autonomously as a tractography agent. While the T-RLF model was trained without environmental interaction, once trained (on RL trajectories), it serves as a refined tract-specific policy which can independently conduct tractography by interacting with the tracking environment (refer Section 3.2), similar to original RL policies.

• Tract generation: Tracking is executed within tract-specific masks obtained from MRM and is initialised with 7 seeds per voxel, and $R _ { t }$ is set to $R _ { 0 } = 3 0 0$ Tracking step size is (empirically selected) and is dataset-specific, 0.375mm for the TractoInferno, 0.468mm for HCP, and 0.75mm for the ISMRM dataset. At each step, the return-to-go $( R _ { t } )$ is reduced by the achieved reward and predicted action $\left( a _ { t } \right)$ , new state $\left( s _ { t } ^ { \prime } \right)$ , and $R _ { t }$ are appended to the context window to serve as input for the next prediction. This auto-regressive process is used by Tract-RLFormer to generate the fiber tract of interest.

• Tractogram cleaning: This is a post-processing step used to filter the resulting tracts. First, length-based cleaning is applied, where streamlines shorter than 20 mm are removed. Subsequent fine-grained filtering is performed using Fast Streamline Search (FSS) [71] to eliminate any extraneous fibers, by comparing the predicted tract with the atlas reference tracts (representing general anatomical structure). The tracts generated by T-RLF models are confined to the masks generated by the MRM module (tailored to each subject). This approach ensures that tract generation remains confined to regions proximate to the actual neural fibers of the subject, thus mitigating the risk of false positives. Consequently, we can perform a high radius search using FSS, without incurring a major risk of high overreach. This high radius search ensures that accurate fibers are not discarded based on minor deviations from atlas tracts.

## 3.4 Result and Performance Analysis

In this section, we present the outcomes of our evaluation of tract-specific T-RLF models under various experimental setups, including comparative analysis, generalization performance, and an ablation study. We trained TD3 and T-RLF models, on eight tracts — seven principal white matter tracts (refer Section 3.3.3.3) and Optical Radius (OR tract) (for analysis in 3.4.2) using five train subjects (id: 1030, 1079,

1119, 1180, and 1198) of the TractoInferno dataset and reported their performance on various test subjects across diferent datasets in subsequent subsections. Additionally, we assess their efectiveness relative to supervised approaches and classical tractography methods.

## 3.4.1 Evaluation Parameters

To evaluate the performance of tractography algorithms, we utilize three evaluation metrics, namely Dice, Overlap, and Overreach. These metrics are computed pairwise between the ground-truth tract and predicted tract in a voxel-wise manner. This means that TP refers to the number of true positive voxels, i.e number of voxels in groundtruth that are also present in predicted tract. Similarly, FP and FN refer to the number of False positive and False negative voxels respectively. Each metric serves a diferent purpose. Overlap $( O v L )$ computes the intersection of the ground truth and predicted tract. High value indicates better coverage of ground-truth area by predicted tract. It is given by:

$$
O v L = \frac { T P } { T P + F N }\tag{3.4.3}
$$

Overreach (OvR) indicates how much the generated tract exceeds the groundtruth, with lower scores suggesting better performance. It is given by:

$$
O v R = { \frac { F P } { T P + F N } }\tag{3.4.4}
$$

Dice score (D) assesses both the accurate coverage and the minimization of extraneous coverage beyond the ground truth area, where values near 1 signify a high similarity level. It is given by:

$$
D i c e = \frac { 2 . T P } { 2 . T P + F P + F N }\tag{3.4.5}
$$

## 3.4.2 Comparative Analysis

This section provides a comparative analysis of our model, T-RLF, against supervised learning, traditional tractography, and state-of-the-art (SOTA) reinforcement learning (RL) algorithms, using Dice scores to evaluate performance across three major white matter bundles: PYT, OR, and CC.

As presented in Table 3.2, all methods are tested on subject 1006 from the TractoInferno dataset (similar to [48, 49] for fair comparison). For the first and third tabular subparts of Table 3.2, the models are trained on ISMRM data. The second subpart does not involve training (classical methods). These 3 subparts are assessed using whole-brain tractography and segmentation [42, 49]. Additionally, the last subpart details the performance of our T-RLF and the TD3 model, where T-RLF was specifically trained on trajectories derived from the TD3 agent.

Table 3.2: Comparison of mean Dice scores for the OR, PYT, and CC tracts for subject 1006 from TractoInferno dataset. Supervised learning scores are from [42]; RL-based scores, with std. dev., are from [49]. The last 2 rows includes scores for T-RLF and TD3, evaluated using our tract-specific approach. The highest and second highest scores are highlighted in green and red, respectively. ‘\*’ denotes tract-specific setting for methods.
<table><tr><td rowspan=1 colspan=2>Algorithm</td><td rowspan=1 colspan=2>OR</td><td rowspan=1 colspan=2>PYT</td><td rowspan=1 colspan=1>CC</td></tr><tr><td rowspan=1 colspan=2>DET-SE</td><td rowspan=1 colspan=2>0.569</td><td rowspan=1 colspan=2>0.665</td><td rowspan=1 colspan=1>0.658</td></tr><tr><td rowspan=1 colspan=2>DET-Cosine</td><td rowspan=1 colspan=2>0.598</td><td rowspan=1 colspan=2>0.708</td><td rowspan=1 colspan=1>0.646</td></tr><tr><td rowspan=1 colspan=2>Prob-Sphere</td><td rowspan=1 colspan=2>0.599</td><td rowspan=1 colspan=2>0.695</td><td rowspan=3 colspan=1>0.6480.6680.614</td></tr><tr><td rowspan=1 colspan=2>Prob-Gaussian</td><td rowspan=1 colspan=2>0.542</td><td rowspan=1 colspan=2>0.723</td></tr><tr><td rowspan=1 colspan=2>Prob-Mixture</td><td rowspan=1 colspan=2>0.436</td><td rowspan=1 colspan=2>0.522</td></tr><tr><td rowspan=1 colspan=2>DET</td><td rowspan=1 colspan=2>0.516</td><td rowspan=1 colspan=2>0.475</td><td rowspan=3 colspan=1>0.3450.5900.827±0.008</td></tr><tr><td rowspan=1 colspan=2>PROB</td><td rowspan=1 colspan=2>0.549</td><td rowspan=1 colspan=2>0.740</td></tr><tr><td rowspan=1 colspan=2>PFT</td><td rowspan=1 colspan=2> $0 . 6 4 4 \pm 0 . 1 3 6$ </td><td rowspan=1 colspan=2>0.753 ± 0.010</td></tr><tr><td rowspan=1 colspan=2>VPG</td><td rowspan=1 colspan=2>0.369 ± 0.135</td><td rowspan=1 colspan=2>0.434 ± 0.128</td><td rowspan=2 colspan=1>0.428 ± 0.1820.222 ± 0.025</td></tr><tr><td rowspan=1 colspan=2>A2C</td><td rowspan=1 colspan=2>0.225 ± 0.108</td><td rowspan=1 colspan=2>0.323 ± 0.082</td></tr><tr><td rowspan=1 colspan=2>ACKTR</td><td rowspan=1 colspan=2>0.397 ± 0.171</td><td rowspan=1 colspan=2>0.559 ± 0.028</td><td rowspan=1 colspan=1>0.584 ± 0.054</td></tr><tr><td rowspan=1 colspan=2>TRPO</td><td rowspan=1 colspan=2>0.330 ± 0.154</td><td rowspan=1 colspan=2>0.498 ± 0.062</td><td rowspan=1 colspan=1>0.594 ± 0.048</td></tr><tr><td rowspan=1 colspan=2>PPO</td><td rowspan=1 colspan=2> $0 . 4 4 0 \pm 0 . 1 8 7$ </td><td rowspan=1 colspan=2>0.619 ± 0.042</td><td rowspan=1 colspan=1>0.650 ± 0.028</td></tr><tr><td rowspan=1 colspan=2>DDPG</td><td rowspan=1 colspan=2> $0 . 6 1 2 \pm 0 . 0 6 3$ </td><td rowspan=1 colspan=2> $0 . 6 3 0 \pm 0 . 0 4 5$ </td><td rowspan=1 colspan=1> $0 . 7 3 1 \pm 0 . 0 0 6$ </td></tr><tr><td rowspan=1 colspan=2>TD3</td><td rowspan=1 colspan=2> $0 . 5 5 5 \pm 0 . 0 9 7$ </td><td rowspan=1 colspan=2> $0 . 6 0 3 \pm 0 . 0 4 5$ </td><td rowspan=1 colspan=1> $0 . 6 8 8 \pm 0 . 0 3 5$ </td></tr><tr><td rowspan=1 colspan=2>SAC</td><td rowspan=1 colspan=2> $0 . 5 9 8 \pm 0 . 0 9 8$ </td><td rowspan=1 colspan=2> $0 . 6 5 8 \pm 0 . 0 2 8$ </td><td rowspan=1 colspan=1> $0 . 7 5 3 \pm 0 . 0 1 0$ </td></tr><tr><td rowspan=1 colspan=2>SAC Auto</td><td rowspan=1 colspan=2> $0 . 6 0 8 \pm 0 . 0 8 8$ </td><td rowspan=1 colspan=2>0.655 ± 0.032</td><td rowspan=1 colspan=1>0.747 ± 0.019</td></tr><tr><td rowspan=1 colspan=2>DET*</td><td rowspan=1 colspan=2>0.648</td><td rowspan=1 colspan=2>0.752</td><td rowspan=2 colspan=1>0.7130.731</td></tr><tr><td rowspan=2 colspan=2>PROB*</td><td rowspan=1 colspan=1>0.652</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>0.765</td></tr><tr><td rowspan=2 colspan=1>TD3*</td><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1>0.644</td><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1>0.764</td><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1>0.720</td></tr><tr><td rowspan=2 colspan=2>T-RLF (Ours)</td></tr><tr><td rowspan=1 colspan=1>0.673</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.772</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.738</td></tr></table>

Several insightful observations can be reported from the results of table 3.2. Comparison with SOTA Classical and RL methods:

In Table 3.2, our framework outperforms the state-of-the-art method (PFT) for

PYT and OR tracts, demonstrating its robustness in tract-specific tractography. Additionally, T-RLF shows comparable performance to state-of-the-art RL algorithms for CC tract.

## Comparison with TD3 agent:

The TD3 agent demonstrates markedly improved performance within our tractspecific generation framework. Testing of tract-specific TD3 on the ISMRM or HCP datasets cannot be conducted due to the absence of the evaluated tracts in these datasets. However, the enhancement in TD3’s performance in our tract-specific setting can be attributed to the training approach rather than dataset consistency. This is evidenced by TD3’s comparable or superior performance on diferent tracts across the ISMRM and HCP datasets, as detailed further in Tables 3.3, 3.4.

## Comparison with DET and PROB:

The Dice scores for DET and PROB improved for all tracts in the tract-specific setting, especially for CC, where DET increased by 106.67% (0.345 to 0.713) and PROB by 23.89% (0.590 to 0.731). The enhanced tracking performance of DET and PROB, despite not being trained, is indicative of the efectiveness of our tractspecific masks. Also, in the whole-brain setting, there is a huge diference between DET (0.475) and PROB (0.740) scores on the PYT tract (Table 3.2), whereas this gap is significantly smaller in the tract-specific setting (marked with ‘\*’), where the tract-specific performance of DET<sup>∗</sup> (0.752) and PROB<sup>∗</sup> (0.765) align closely with each other and with the T-RLF and TD3 methods. The consistency and stability observed for these classical methods are attributed to our tract-specific approach.

## 3.4.3 Generalization Performance Evaluation

In this section, we present the performance evaluation of our T-RLF model across three distinct datasets (Tables 3.3, 3.4), demonstrating its generalization performance. The results are averaged across test subjects for each dataset including five test subjects from the TractoInferno (TtoI) dataset (id: 1160, 1078, 1159, 1061, and 1171), four from the HCP dataset (id: 930449, 992774, 959574, and 987983), and one from the ISMRM phantom dataset.

A visual comparison across datasets and subjects is presented in Fig. 3.5(a).

We also compare the performance of T-RLF with classical algorithms, which were employed using tract-specific masks, and the tract-specific TD3 agent, from which the training data for T-RLF was derived (Fig. 3.5(b)).

A quantitative comparison of generalization performance is presented in Tables 3.3, and 3.4, for the left and right Cingulum, Arcuate Fasciculus, Pyramidal Tract and a part of Corpus Callosum tract. The analysis is performed across all three datasets, with each table including only the datasets that contain the corresponding tract(s). We observe that T-RLF model displays a notable generalization performance. Interestingly, the classical deterministic (DET) and probabilistic (PROB) methods exhibit slightly better performance than learnable methods in some cases.

Table 3.3: Performance metrics(in %) for the CG and AF tracts, trained on the TractoInferno dataset and tested across multiple datasets to evaluate generalization. Tracking is performed using our proposed tract-specific generation method. A dash (‘-’) indicates the absence of ground-truth tracts in the corresponding dataset, precluding evaluation.
<table><tr><td rowspan="3">ataa</td><td rowspan="3"></td><td colspan="4">Cingulum (CG)</td><td colspan="3">Arcuate Fasciculus(AF)</td></tr><tr><td colspan="2">Left</td><td colspan="2">Right</td><td>Left</td><td colspan="2">Right</td></tr><tr><td>Dice</td><td></td><td>OvL</td><td>Dice</td><td>OvL OvR</td><td>Dice OvL</td></tr><tr><td rowspan="3"></td><td>Algo. T-RLF</td><td>53.3 42.5</td><td>OvL OvR 16.6</td><td>Dice 45.6 33.7</td><td>OvR 13.9 61.8</td><td>51.2 13.4</td><td>41.8 27.9</td><td>OvR 5.40</td></tr><tr><td>TD3 DET</td><td>53.0 42.3 55.2 46.4</td><td>16.9 21..5</td><td>45.2 33.6 52.6 41.7</td><td>14.3 61.6 16.3 62.8</td><td>51.0 13.7 52.5 14.0</td><td>41.6 27.7 43.9 30.0</td><td>5.60 6.6</td></tr><tr><td>PROB PFT</td><td>57.6 67.3</td><td>51.8 27.9 55.2 7.8</td><td>56.1 45.5 59.6 45.4</td><td>16.6 6.4</td><td>65.5 57.9 18.3 71.3 71.9 29.7</td><td>47.4 33.7 69.9 71.6</td><td>8.6 33.3</td></tr><tr><td rowspan="3">TOl</td><td>T-RLF TD3</td><td>61.0 56.8 60.0 55.1</td><td>28.6 27.3</td><td>56.5 52.6 54.9 49.8</td><td>34.9 32.4</td><td>52.7 45.1 27.8 51.8 44.3 28.2</td><td>39.5 36.5 38.4 34.9</td><td>49.8 46.9</td></tr><tr><td>DET PROB</td><td>61.2 58.8 67.9 69.1</td><td>32.3 33.6</td><td>58.2 54.7 64.7 64.6</td><td>33.7 54.6 36.7 62.3</td><td>46.3 24.7 57.2 27.2</td><td>45.4 41.7 50.3 48.3</td><td>46.9 50.9</td></tr><tr><td>PFT T-RLF</td><td>55.9</td><td>48.8 25.4</td><td>54.5 51.3</td><td>38.6</td><td>62.8 60.1 31.5</td><td>53.9 62.8</td><td>88.0</td></tr><tr><td rowspan="5">S1T</td><td>TD3 DET</td><td>54.2 46.6 53.1 44.8</td><td>25.5 23.7</td><td>52.8 44.1 51.2 41.6</td><td>23.1 一 21.1 一</td><td>一</td><td>– 一 一</td><td>–</td></tr><tr><td>PROB</td><td>57.5 61.1</td><td>51.9 28.5</td><td>57.7 52.4</td><td>29.2</td><td>一</td><td>1</td><td></td></tr><tr><td>PFT</td><td>55.4</td><td>59.3 35.1 49.4 28.9</td><td>64.0 65.4 57.3 49.4</td><td>39.0 22.9</td><td>1</td><td>一</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>–</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

As previously mentioned in section 3.4.2, the consistency observed in Tables 3.3, 3.4 for the classical methods (DET and PROB) is due to our tract-specific approach. This improvement and stabilization may be attributed to the elimination of premature termination issues in narrow and deep WM regions, as described in [41], facilitated by the refined spatial exploration enabled by MRM in our tract-specific approach. It can be observed from Tables 3.2 and 3.4, that the performance of PFT declined in the tract-aware setting, dropping from 75% to 66.2% in the PYT and from 82% to 55% in the CC (refer Table 3.4). This decline can be attributed to use of Continuous Map Criterion (CMC) [41] as a stopping criterion for fiber tracking. The CMC terminates fiber tracking based on Partial Volume Estimate (PVE) maps, allowing tractography to continue until the streamline correctly stops in the gray matter. This approach may generate fibers beyond our tract-specific masks, leading to increased overreach (see Fig. 3.5(b)) and consequently lower Dice scores. Furthermore, fibers generated outside the tracking mask may be erroneous and subsequently filtered or cleaned via FSS, resulting in a lower OvL score.

![](images/13467de3e4f1e169e83a2d9e71244e8033847d20c319231820c318cd56a93834.jpg)  
Figure 3.5: Visual comparison of reconstructed tracts illustrating (a): Intra-dataset variability, Inter-dataset variability, and (b): Variability across tracts reconstructed by diferent algorithms. The depicted tracts include the left PYT, CG, and a part of CC. The algorithms evaluated in bottom section of figure are T-RLF (ours), TD3, and PFT.

Table 3.4: Results are presented for the left and right parts of PYT and a segment of CC on the TractoInferno (TtoI) dataset. Tracking for all algorithms is conducted using our proposed tractspecific generation method.
<table><tr><td rowspan="3">Dataset</td><td rowspan="3">Algo.</td><td colspan="3">Pyramidal Tract (PYT)</td><td rowspan="2">Corpus Callosum (CC)</td></tr><tr><td>Left</td><td></td><td>Right</td></tr><tr><td>Dice</td><td>OvL OvR</td><td>Dice OvL</td><td>OvR Dice OvL OvR</td></tr><tr><td rowspan="5">TtoI</td><td>T-RLF TD3</td><td>70.3 64.1 17.2</td><td>70.1 63.1</td><td>16.9</td><td>70.4 71.2 32.6</td></tr><tr><td>DET</td><td>69.4 62.5 15.9</td><td>69.2 61.4</td><td>15.8</td><td>68.1 64.8 26.1</td></tr><tr><td>PROB</td><td>72.7 79.3</td><td>38.8 70.3</td><td>76.2 40.7</td><td>70.1 72.6 35.8</td></tr><tr><td>PFT</td><td>77.6 79.5</td><td>25.3 74.8</td><td>72.5 21.3</td><td>72.6 76.4 36.7</td></tr><tr><td></td><td>66.2 55.7</td><td>12.4 65.9</td><td>57.8 17.4</td><td>54.9 51.2 36.1</td></tr></table>

Summarization: In summary, our results demonstrate that we surpass supervised methods (Table 3.2). Additionally, we consistently outperform the TD3 model (Tables 3.2- 3.4), which served as the basis for training T-RLF. Notably, our tractspecific setting not only improves TD3 performance but also the performance of classical methods like DET and PROB compared to the whole-brain setting. This suggests a promising new direction of data driven policy learning for tract specific fiber generation in limited ground truth scenarios that can naturally scale up efectively.

## 3.4.4 Ablation Study

In this section, we present an ablation study to determine the optimal configuration for our T-RLF model. We also examined the impact of two key components of our work: Mask Refinement Module (MRM) discussed in section 3.3.1.2, and the tract-specific policy fine-tuning as detailed in section 3.3.2.

Table 3.5 compares the performance of diferent configurations of the GPT architecture, varying the number of attention heads (n\_heads), context length (K), and embedding dimension (d). The models were evaluated based on their Dice scores for subject 1006 from the TractoInferno dataset, with scores averaged across the seven white matter tracts (see Section 3.2.1.2).

Table 3.5: Dice scores (in %) averaged over 7 tracts of subject 1006 from TractoInferno dataset, at diferent values of T-RLF parameters: number of attention heads (n\_heads), context length (K), and embedding dimension (d). Best score is in bold.
<table><tr><td rowspan=2 colspan=1></td><td rowspan=1 colspan=2>K = 20</td><td rowspan=1 colspan=2>K = 30</td><td rowspan=1 colspan=2>K = 40</td></tr><tr><td rowspan=1 colspan=1>d=128</td><td rowspan=1 colspan=1>d=512</td><td rowspan=1 colspan=1>d=128</td><td rowspan=1 colspan=1>d=512</td><td rowspan=1 colspan=1>d=128</td><td rowspan=1 colspan=1>d=512</td></tr><tr><td rowspan=2 colspan=1> $\overline { { n \_ h e a d s = 1 } }$  $n \_ h e a d s = 2$ </td><td rowspan=2 colspan=1>64.763.4</td><td rowspan=1 colspan=1>65.2</td><td rowspan=1 colspan=1>66.2</td><td rowspan=1 colspan=1>67.3</td><td rowspan=2 colspan=1>68.668.0</td><td rowspan=2 colspan=1>67.668.2</td></tr><tr><td rowspan=1 colspan=1>66.3</td><td rowspan=1 colspan=1>66.2</td><td rowspan=1 colspan=1>67.5</td></tr></table>

It can be seen that the highest Dice score is achieved with n\_heads = 1, K = 40, and d = 128, which was therefore selected for T-RLF. Additionally, the results highlight the importance of a larger context in tractography, demonstrating that a broader temporal receptive field can improve the model’s ability to generate accurate fiber tracts.

Table 3.6: Average performance metrics (in %) obtained using Tract-RLFormer highlight the impact of the MRM on test subjects from the TractoInferno dataset. The table also compares the performance of the pre-trained network with the fine-tuned network post MRM application, illustrating the efect of policy fine-tuning on the same dataset.
<table><tr><td rowspan="2">Tract</td><td colspan="2">Without MRM</td><td colspan="6">With MRM</td></tr><tr><td colspan="3"></td><td colspan="2">Pre-trained</td><td colspan="3">Fine-tuned</td></tr><tr><td></td><td>Dice</td><td>OvL</td><td>OvR Dice</td><td>OvL</td><td>OvR</td><td>Dice</td><td>OvL</td><td>OvR</td></tr><tr><td>PYT</td><td>44.1</td><td>30.4</td><td>5.6 65.9</td><td>55.1</td><td>11.8</td><td>70.3</td><td>67.2</td><td>24.7</td></tr><tr><td>CG</td><td>41.1</td><td>45.4</td><td>80.3 51.5</td><td>45.7</td><td>30.3</td><td>58.7</td><td>54.7</td><td>31.7</td></tr><tr><td>AF</td><td>34.2</td><td>34.4 65.1</td><td>45.8</td><td>39.5</td><td>36.1</td><td>46.1</td><td>40.8</td><td>38.8</td></tr><tr><td>CC</td><td>58.9</td><td>59.0 41.9</td><td>66.2</td><td>59.9</td><td>20.8</td><td>70.4</td><td>71.2</td><td>32.6</td></tr></table>

The analysis of the efect of tract-specific masks generated by our Mask Refinement Module (MRM) is presented in Table 3.6. The "Without MRM" scores are obtained by performing tractography within masks derived from converting Atlas streamlines into binary masks, as shown in Step 2 of Fig. 3.2. We observe that Atlas-derived tracking masks led to a significant overreach (OvR), extending beyond actual region of interest. This OvR was notably reduced after incorporating

MRM, leading to improved Dice and overlap metrics across all tracts.

We also present the efect of our two-stage training process, comparing the models obtained after both stages of mixed-tract pretraining and tract-specific finetuning. Table 3.6 reports the average scores across five test subjects from the TractoInferno dataset. It can be seen that fine-tuning the network specific to each tract allowed it to learn better and robust tract-specific difusion characteristics, resulting in additional improvements in the performance metrics.

## 3.5 Summary

Tract-RLFormer is a data driven RL policy refinement framework for tractography. It is a tract-specific framework that integrates supervised and reinforcement learning paradigms. A distinctive feature of our Tract-RLFormer is its ability to train within the reinforcement learning experience space, independent of ground truth fibers. The fine-tuning stage of T-RLF model focuses and refines its capabilities in generating the tracts of interest. This approach demonstrates its excellent generalization performance across various datasets as well as scalability. This framework has the potential to utilize data from any reinforcement learning agents trained in diverse neuroimaging environments. Moreover, our innovative tract-specific modeling approach simplifies the reconstruction process by directly generating the target tract, thus avoiding the complex and error-prone segmentation step.

Chapter 4

## Data-driven RL Policy Fusion Framework with Multi-Critic finetuning for Tract-specific Tractography

## 4.1 Motivation

In the previous chapter, we saw that Tract-RLFormer framework enables an efective data-driven RL policy refinement to directly reconstruct the tract of interest. Its GPT-based actor learns a refined policy entirely within the RL experience space, using mixed-tract pretraining and tract-specific fine-tuning to enable strong generalization across datasets and robust performance across diverse tract configurations. Despite these advantages, our findings and previous RL-based tractography studies [49] revealed that deterministic policies, such as TD3, tend to produce lower tract coverage and therefore higher false-negative rates, whereas stochastic policies, such as SAC, often generate erroneous pathways that result in increased false-positive connections. This trade-of is further discussed in Section 4.3.

Several studies [31,72] have emphasized this fundamental trade-of in the context of classical tractography: minimizing false negatives (Increasing Overlap) often leads to an increase in false positives (higher overreach), and vice versa.

This observation motivates the central idea of the present chapter: the potential of combining multiple diverse RL policies to achieve a more balanced and robust tractography behavior; a strategy that has shown promise in other domains, such as robotics and game-playing AI, where combining multiple policies has been demonstrated to improve generalization and robustness [55, 73].

Traditional RL policy ensembling strategies, such as decision-level aggregation (e.g., voting or averaging), often fail to leverage semantic context or state history. On the other hand, complex ensemble techniques, such as Ensemble Policy Gradient or hierarchical/hybrid ensembles, introduce significant system complexity due to the need for agent synchronization, parameter sharing, and joint optimization, making them dificult to implement, debug, and maintain. To sidestep these challenges, our TractRLFusion adopts a data-driven approach to fusion approach for combining multiple RL policies in tractography.

## 4.2 Proposed TractRLFusion Framework

TractRLFusion is a novel GPT-based multi-policy fusion framework (as shown in Fig. 4.1). It integrates the following key components:

• Episodic Data Selection (EDS) module is used to select tract-specific trajectories with anatomical precision, enabling data-driven fusion.

• Multi-Critic Policy Fine-Tuning (MCPFT) module is used for enhancing the policy fusion which enables robust, scalable, and generalizable tractography across diverse datasets.

## 4.2.1 Overview and Experimental Setup

We have utilized TractRLFusion framework to fuse three independently trained policies, namely TD3 [56], SAC [57], and DDPG [55], to combine their complementary strengths in tractography. While SAC learns in an exploration-driven manner (leading to high tract coverage at the cost of increased false positives), TD3 and DDPG lean towards conservative tracking (lower coverage with fewer false positives), further elaborated and shown in Section 4.3.1.

![](images/4cbbc9b52b639e6630c379308d47b569ec0f671212998b30e48cdc1f23f7671e.jpg)  
Figure 4.1: TractRLFusion: A generative framework where anatomically relevant trajectories with high expected returns from various RL $\mathrm { a g e n t s } ( \pi _ { \theta _ { i } } )$ are combined into a experience dataset (T) to train a generative model. The model leverages critic networks $\left( C ( \pi _ { \theta _ { i } } ) \right)$ to align streamline generation with reward estimates, producing highly plausible reconstructions.

TractRLFusion framework is composed of three key components, Episodic Data Selection (EDS), FusionNet, and Multi-Critic Policy Finetuning (MCPFT). To perform a data-driven fusion, we sample trajectories from trained RL policies, where Trajectories refer to sequences of $\mathrm { r e t u r n – t o – g o } ( r _ { t g } )$ –state(s)–action(a) obtained from an RL policy. A T-length trajectory is defined as: $\tau = ( r _ { t g _ { 0 } } , s _ { 0 } , a _ { 0 } , r _ { t g _ { 1 } } , . . . . r _ { t g _ { T } } , s _ { T } , a _ { T } )$

(i) Firstly EDS samples a mix of trajectories data from each of the 3 policies to train FusionNet on. (ii) Followed by 2 stage training of our GPT-based FusionNet on EDS data to learn the fusion policy. (iii) Finally FusionNet policy further undergoes refinement using MCPFT.

FusionNet learns a tract-specific policy for a given white matter tract and we train and test our policies in tract-specific masks obtained from our Mask Refinement Module (MRM) as described in section 3.3.1.2 [74]. Hence our proposed methods fall into the category of ROI-based tracking [28]. Among major tracts we focus on Occipital and pre/post central gyri of Corpus Callosum (CC), left and right Corticospinal Tract (CST), Arcuate Fasciculus (AF), and Cingulum (CG) tracts following prior works [49, 52] in order to have a fair comparison. Datasets used in this study are already detailed in Section 3.2.

## 4.2.2 Tract-Specific RL Policy Training

With an environment similar to [48] (Table 4.1), our training is conducted via exploration within tract-specific masks of 5 train subjects from TractoInferno dataset,

similar to previous chapter. The reward function is defined as the product of the action’s $\left( \vec { a } _ { t } \right)$ alignment with the most aligned fODF peak $( p )$ and its dot product with the previous tracking direction $\left( \vec { u } _ { t - 1 } \right)$

$$
r _ { t } = \left| \operatorname* { m a x } _ { \vec { p _ { i } } } \left( \vec { p _ { i } } \cdot \vec { a } _ { t } \right) \right| \times \left( \vec { a } _ { t } \cdot \vec { u } _ { t - 1 } \right)\tag{4.2.1}
$$

Episode termination conditions include (i) exceeding the maximum episode length of 530 (corresponding to max. fiber length=200mm), (ii) exiting tract mask, and (iii) angle > 60 degrees with the previous fiber segment. The following paragraphs explain the key components of Tract-RLFusion.
<table><tr><td rowspan=1 colspan=1>Category</td><td rowspan=1 colspan=1>Details</td></tr><tr><td rowspan=1 colspan=1>Architectures</td><td rowspan=1 colspan=1>- TD3/SAC/DDPG (Actor and Critics):- 3-layer ReLU-activated FCNNs- Neurons per layer: 1024</td></tr><tr><td rowspan=1 colspan=1>Training Data</td><td rowspan=1 colspan=1>- Train Subjects: 5 (IDs: 1030, 1079, 1119,1180, 1198) [42]</td></tr><tr><td rowspan=1 colspan=1>Hyperparameters</td><td rowspan=1 colspan=1>- Batches: 50 per subject- Batch size: 4096 transitions- Seeds: 7 per voxel- Step size: 0.375 mm</td></tr><tr><td rowspan=1 colspan=1>Policy Hyperparameters</td><td rowspan=1 colspan=1>- TD3: lr = 8.56e-6, action_std = 0.334, γ =0.776- SAC: lr = 3.7e-5, action_std = 0.4, γ = 0.89, $a l p h a = 0 . 0 7 6$ - DDPG: lr = 8.56e-6, action_std = 0.35, γ =0.5</td></tr></table>

Table 4.1: Summary of Features, Architecture, Training Data, and Hyperparameters for training Tract-specific RL policies.

## 4.2.3 Episodic Data Selection

It is the trajectory data selection process for the pre-training and fine-tuning stages of our FusionNet model (see Fig. 4.2). It involves following steps:

(i) Tracking is performed using each trained policy (TD3, SAC and DDPG) from the same batch of neighbouring initial seed points (sampled from tract mask of a given subject), resulting in streamlines $( x , y , z )$ and the corresponding $( r _ { t g } , s , a )$ trajectories (see Fig. 4.2(i)).

![](images/9150e7e8012b91b80108ad06e749fe131002fb06bcc4810572e8578f6c81ca78.jpg)  
Figure 4.2: Visual depiction of Episodic Data Selection. Row 1 illustrates within-policy trajectory selection using the MDF [75] constraint (Eq. 2) with atlas reference streamlines. Row 2 depicts across-policy selection via the Q-value constraint (Eq. 3), where expected returns are compared. For each region $R _ { i }$ , trajectories from the policy π with the highest expected return is selected to populate the dataset (T).

(ii) Trajectories shorter than 47 transitions (corresponding to approximately 20mm of fiber length for a 1mm<sup>3</sup> isometric voxel size and a 0.375mm step size) are filtered-out and the rest undergo batch-wise across-policy (Fig. 4.2 (ii)) selection and within-policy (Fig. 4.2 (iii)) selection. This threshold was chosen to exclude trajectories corresponding to anatomically implausible short streamlines.

(iii) Within-Policy trajectory selection: Fifteen reference streamlines are selected from the atlas tract using farthest streamline sampling [76]. Streamlines for each policy are filtered such that trajectories of streamlines having MDF (Mean Direct Flip) distance [75] less than 5mm with the reference streamlines are selected. This within-policy selection enables the selection of the best trajectories in terms of anatomical shape of any given tract.

(iv) Across policy selection selects one policy out of the three, based on the maximum expected Q-value of its MDF-filtered trajectories. These trajectories are selected for the finetuning dataset. Whereas, for pretraining dataset, Q-value based selection happens directly on unfiltered trajectories, as depicted in Fig. 4.2.

Neighbouring seeds are selected in each batch, enabling EDS to select best trajectories from all regions of the given tract. This is repeated for all tracts of the 5 train subjects, followed by length-based and random selection, similar to chapter 3 (3.3.3.1), resulting in a Pretraining dataset of 150000 trajectories and finetuning dataset of 50000 trajectories. It is important to note that the pretraining dataset is made from trajectories of all tracts, whereas finetuning data consists of tract-specific trajectories.

## 4.2.4 FusionNet Architecture and Training

Central to the TractRLFusion framework, FusionNet, described in Table 4.2, is a 4-layer GPT [61] that models RL trajectories as sequences which undergoes general pretraining and tract-specific fine-tuning on expert trajectories from EDS. Specifically, the first three layers are trained on mixed trajectories from the EDS pretraining data over 30 iterations, while the final layer is fine-tuned for 10 iterations using tractspecific trajectories from the EDS-Finetune data. This two-stage training strategy is designed to firstly capture broad trajectory patterns and then adapt to the finegrained tract-specific details. Similar to Chapter 3, we use a 5-step angular loss $\left( \mathcal { L } _ { d i s t _ { c o s } } \right)$ between predicted (aˆ) and actual (a) actions, defined as:

$$
\mathcal { L } _ { d i s t _ { c o s } } = \sum _ { t = 2 } ^ { K - 2 } \left( \sum _ { i = - 2 } ^ { 2 } \cos ^ { - 1 } \left( \mathbf { a } _ { t + i } \cdot \hat { \mathbf { a } } _ { t + i } \right) \right)\tag{4.2.2}
$$

where K denotes the context length of the FusionNet model.

<table><tr><td rowspan=1 colspan=1>Category</td><td rowspan=1 colspan=1>Details</td></tr><tr><td rowspan=1 colspan=1>Architecture</td><td rowspan=1 colspan=1>- GPT Layers: 4- Embedding Dim: 128- Attention Heads: 1- Context Length (lcontext): 40- Activation: ReLU- Dropout: 0.1</td></tr><tr><td rowspan=1 colspan=1>Training</td><td rowspan=1 colspan=1>- Optimizer: AdamW ; LR: 1e-4- Weight Decay: 1e-4- Batch Size: 128- Iters:30(General),10(Tract-specific)- Updates/iter: 10000- Warmup Steps: 10000</td></tr><tr><td rowspan=1 colspan=1>MCPFT</td><td rowspan=1 colspan=1>- LR: 1e-4 ; Batch Size: 512- Iters: 25- Updates/iter: 1000 (actor), 1 (critic)</td></tr><tr><td rowspan=1 colspan=1>Tracking</td><td rowspan=1 colspan=1>- Initial Return-to-Go: 300</td></tr></table>

Table 4.2: Summary of FusionNet architecture, training, and tracking hyperparameters.

The model was trained using the AdamW optimizer. Key hyperparameters, summarized in Table 4.2, were tuned to maximize Accumulated sum of rewards and Dice Scores. We observed that further training with the supervised loss (Eq. 4.2.2) saturates the performance, showing limited improvement.

## 4.2.5 Multi-critic Policy Fine-tuning

FusionNet policy is fine-tuned further using our multi-critic setup, known as Multicritic Policy Fine-tuning (MCPFT). Having learnt a fused policy by ofline training on EDS data, FusionNet is further trained in an actor-critic setup using the critic networks from original RL policies of SAC, DDPG, and TD3. Figure 4.3 shows the MCPFT module in detail.

![](images/71e11e409c3108991e423774648929c3dcb9a69c7c9c362fc973ef56cb4cbbe4.jpg)  
Figure 4.3: Visualization of MCPFT module: It further finetunes the FusionNet model using aggregated critic loss in addition to the original 5-step loss, interpreted as policy consistency loss.

In this setting, the critics update with their RL losses $\mathcal { L } _ { \mathrm { c r i t i c } , k }$ , while the actor (FusionNet) optimizes using $\mathcal { L } _ { \mathrm { a c t o r } }$ loss as defined below:

$$
\mathcal { L } _ { \mathrm { a c t o r } } = \mathcal { L } _ { d i s t _ { c o s } } + \sum _ { k = 1 } ^ { 3 } \mathcal { L } _ { \mathrm { c r i t i c } , k } , \quad \mathrm { w h e r e } ~ \mathcal { L } _ { \mathrm { c r i t i c } , k } = - \sum _ { t = 0 } ^ { K } Q _ { C ( \pi _ { k } ) } ( s _ { t } , \hat { a } _ { t } )\tag{4.2.3}
$$

where k indexes the three critic networks, K denotes the context length of FusionNet, and $Q _ { C ( \pi _ { k } ) }$ is the Q-value predicted by the $k ^ { t h }$ critic, conditioned on its policy $\pi _ { k }$ , for taking action $\hat { a } _ { t }$ at state $s _ { t }$

As discussed in Section 4.2.4, the performance of FusionNet saturates after 40 (pre-training and finetuning) training iterations. To further improve the FusionNet policy beyond this saturation point, it is natural to turn to the critics: unlike the 5- step loss, which primarily preserves previously learned behaviour, the critic networks can provide richer information about expected return across the state–action space. Using critic feedback for additional learning can therefore expose the FusionNet actor to value-based gradients that were not fully exploited during the initial training stages.

Moreover, using multiple critics to guide an actor is a common design choice in reinforcement learning. Multi-critic architectures are frequently employed to obtain more reliable value estimates, reduce over or under-estimation bias, or encourage more expressive behaviour [56, 57, 77]. Motivated by this general principle, we explored the integration of the three critics to improve the fused FusionNet policy.

However, critic-driven updates alone were insuficient to preserve the actor’s previously learned behaviour. To address this, we fine-tuned FusionNet using a combined objective that incorporates both our 5-step cosine-distance loss and the aggregated critic loss. Empirically, we observed that the $\mathcal { L } _ { d i s t _ { c o s } }$ component stabilizes training by retaining the actor’s prior policy structure, efectively anchoring the model to its pre-trained behaviour, while the aggregated Q-value term provides additional information to refine the policy based on value estimates from all three critics. This combination enables FusionNet to benefit from multi-critic feedback without destabilizing the actor, resulting in improved fine-tuning performance compared to using either component alone.

For fine-tuning under this objective (defined in 4.2.5), we trained the FusionNet model for 25 iterations, with each iteration consisting of 1,000 actor (FusionNet) updates and a single update for each critic. This update schedule intentionally delays critic updates, which we found to mitigate saturation efects caused by rapidly changing or unstable value estimates. A batch size of 512 trajectories was used throughout training.

## 4.2.6 Tract Generation and Cleaning

Using FusionNet policy, tracking is performed for diferent tracts within their respective tracking masks, by seeding at 7 seeds per voxel. Return-to-go is initialized $\left( \mathrm { a s \ } r _ { t g 0 } \right)$ to 300 for FusionNet, and dataset-specific step sizes (0.375mm for TractoIn-

ferno, 0.468mm for HCP, 0.75mm for ISMRM) are fixed similar to chapter 3. As a postprocessing step, tract filtering is performed using Fast Streamline Search [71] with atlas reference tracts [69], as elaborated previously in Section 3.3.4.

## 4.3 Experiments and Results

We evaluate our performance using five test subjects (IDs: 1160, 1078, 1061, 1159, 1171) from the TractoInferno dataset. Our generalization performance is assessed on 4 diverse subjects from the HCP [67] dataset (IDs: 930449, 959574, 992774, 987983); (reference tracts from [28]) and on the ISMRM dataset [29]. Our evaluation compares the Dice, Overlap (OL), and Overreach (OR) scores.

We have compared FusionNet against the following baselines.

• Classical methods DET [34] and PROB (iFOD1) [78].

• Deep RL policies [48, 49] SAC, TD3, and DDPG, employed to learn our fusion policy.

• Tract-specific method TractSeg [28] was evaluated within its own tract masks. For a fair comparison, FusionNet was also evaluated using the masks generated by TractSeg (presented as FusionNet<sup>∗</sup> in Tables 4.3 and 4.4).

• Policy Ensembles $\pi _ { m a x Q }$ and $\pi _ { A v g }$ that combine SAC, TD3, and DDPG at decision level by selecting the action with the max normalized Q-value, and average of the 3 actions, respectively, at each step.

## 4.3.1 Comparative Study

Tables 4.3 and 4.4 present a comprehensive comparison, while Fig. 4.4 shows a qualitative comparison among the best performing RL (SAC) and classical (PROB) algorithm. TractOracle [50] was excluded from our comparisons, as it is specifically designed for seeding from the white matter–gray matter (WM/GM) interface. Although the TractoInferno dataset does not explicitly annotate the corticospinal tract (CST), it does include the pyramidal tract (PYT), which contains the core components of the CST. Prior work by Dumais et al. [79] directly compared the PYT (as extracted using the RBx method in TractoInferno) with the CST (as extracted using the TractSeg method in the HCP dataset), demonstrating that the two tracts are anatomically and functionally equivalent for tract-level analyses. Accordingly, in our experiments we report CST results for the HCP and ISMRM datasets, and PYT results for the TractoInferno dataset.

![](images/c9194fafcf8473985db004eacbdd88f9cb45543b2bb5fd9a2b240c4d90779a80.jpg)  
Figure 4.4: Visualization of (a) right corticospinal tract, and (b) part of the left cingulum tract, from subject 992774 in the HCP dataset. (c) shows the occipital part of the corpus callosum from subject 1160 in the TractoInferno dataset; for comparison of proposed method with PROB and SAC (other top-performing algorithms considered in this study) and ground-truth tracts. Dashed circles highlight key regions for comparison.

Comparison of Dice, OL, and OR metrics together provides a more comprehensive assessment of performance rather than their individual comparison. A high Dice score indicates a favorable balance between OL and OR, with regards to tradeof between maximizing OL (capturing more true anatomy) and minimizing OR (reducing false positives).

Observations: Comparative results in Tables 4.3, 4.4 and qualitative results in Fig. 4.4 reveal the following notable insights:

• Our proposed FusionNet model consistently ranks best in terms of Dice score across all datasets and tracts.

Table 4.3: Comparison of performance metrics (%) for AF and CC, showing generalization across HCP, TractoInferno(TtoI), and ISMRM datasets. Metrics (Dice, OL, OR) are averaged across Left/Right for AF, and across Oc/Pr\_Po for CC.
<table><tr><td rowspan=2 colspan=1>Data</td><td rowspan=2 colspan=1>Algorithm</td><td rowspan=1 colspan=3>AF</td><td rowspan=1 colspan=1>CC</td></tr><tr><td rowspan=1 colspan=3>Dice ↑    OL↑      OR↓</td><td rowspan=1 colspan=1>Dice ↑    OL↑      OR↓</td></tr><tr><td rowspan=8 colspan=1>HOP</td><td rowspan=6 colspan=1>DETSACDDPGTD3πmaxQ $\pi _ { a v g }$ FusionNet</td><td rowspan=1 colspan=3>53.4±5.0  41.3±7.4  10.3±4.6</td><td rowspan=1 colspan=1>70.9±1.1  77.2±2.3  41.2±3.4</td></tr><tr><td rowspan=1 colspan=3>54.6±4.7 42.4±6.5 11.4±3.2</td><td rowspan=1 colspan=1>70.4±0.2 80.4±2.9 48.1±5.2</td></tr><tr><td rowspan=1 colspan=3>50.6±5.5 37.9±7.6 8.3±3.7</td><td rowspan=1 colspan=1>65.0±1.2 66.2±5.4 37.3±6.5</td></tr><tr><td rowspan=1 colspan=3>51.7±5.3 39.5±7.8 9.6±3.9</td><td rowspan=1 colspan=1>64.9±1.2 66.2±3.8 37.2±6.6</td></tr><tr><td rowspan=1 colspan=3>54.8±4.7  42.3±7.152.3±4.1  40.5±7.7 12.7±6.4</td><td rowspan=1 colspan=2>110.9±3.5</td><td rowspan=1 colspan=1>70.4±0.6 81.6±3.4 50.1±5.470.1±0.7 78.4±1.9 46.1±2.6</td></tr><tr><td rowspan=1 colspan=3>55.5±4.843.8±6.912.3±5.2</td><td rowspan=1 colspan=1>72.4±0.481.8±1.7  44.8±2.6</td></tr><tr><td rowspan=2 colspan=1>TractSegFusionNet*</td><td rowspan=1 colspan=3>70.9±1.5 59.5±2.5 8.1±0.9</td><td rowspan=1 colspan=1>76.8±2.2 68.5±3.4  11.7±1.3</td></tr><tr><td rowspan=1 colspan=3>74.1±3.666.6±6.313.0±2.2</td><td rowspan=1 colspan=1>77.4±1.878.6±2.823.9±3.1</td></tr><tr><td rowspan=7 colspan=1>TTI</td><td rowspan=7 colspan=1>DETSACDDPGTD3πmaxQ $\pi _ { a v g }$ FusionNet</td><td rowspan=1 colspan=3>50.0±6.5 44.0±3.1 35.8±9.1</td><td rowspan=1 colspan=1>49.7±4.8 39.5±5.2 19.6+8.7</td></tr><tr><td rowspan=1 colspan=3>52.3±8.9  52.4±4.752.1±9.6</td><td rowspan=1 colspan=1>54.6±4.6  48.3±8.6 28.6±8.2</td></tr><tr><td rowspan=3 colspan=3>46.3±7.0 39.3±2.5 28.6±9.945.3±8.3 39.6±4.2 37.6±9.352.7±8.5 51.9±4.4</td><td rowspan=1 colspan=1>49.6±5.6  40.4±8.0 22.1±8.9</td></tr><tr><td rowspan=1 colspan=1>48.1±6.2 38.3±7.4 20.9±5.3</td></tr><tr><td rowspan=1 colspan=1>47.3±9.5</td><td rowspan=1 colspan=1>30.7±8.8</td></tr><tr><td rowspan=1 colspan=2>52.1±7.5 51.9±7.6</td><td rowspan=1 colspan=1>49.5±8.7</td><td rowspan=1 colspan=1>53.3±7.2 47.3±8.6 31.5±5.8</td></tr><tr><td rowspan=1 colspan=3>53.2±8.951.3±4.3  39.1±9.3</td><td rowspan=1 colspan=1>56.5±3.250.5±6.128.8±6.6</td></tr></table>

• Note: PROB was one of the four tractography algorithms used to generate the ground-truth streamlines in TractoInferno, which inherently gives it an advantage in evaluations involving this dataset. In contrast, the reference streamlines for the HCP dataset in [28] were generated using iFOD2 [36], a close successor to PROB, which may cause PROB to systematically underperform on HCP. Therefore we report its performance only on the ISMRM dataset.

• Fusion policy (FusionNet) outperforms individual RL policies (SAC, TD3, DDPG) and their RL-Ensemble methods $\left( \pi _ { m a x Q } \right.$ and $\pi _ { a v g } )$ . Individual RL policies show a high OR-OL tradeof with SAC achieving highest OL with worst OR, and DDPG and TD3 achieving low OL and OR. The Ensemble’s Q-value approach yields suboptimal tracking, while FusionNet’s learned policy better balances OL-OR trade-ofs, resulting in the best Dice.

• FusionNet\* (FusionNet tested on TractSeg masks [28]) surpasses TractSeg across all HCP tracts, e.g., achieving a Dice score of 74.1 versus TractSeg’s 70.9 for AF, demonstrating superior tracking even within TractSeg’s masks.

Table 4.4: Comparison of performance metrics (%) for CG and CST, showing generalization across HCP, TractoInferno(TtoI), and ISMRM datasets. Metrics (Dice, OL, OR) are averaged across Left/Right.
<table><tr><td rowspan="2">Data</td><td rowspan="2">Algorithm</td><td colspan="3">CG</td><td colspan="3">CST/PYT</td></tr><tr><td colspan="3">Dice ↑ OL↑</td><td colspan="3">Dice ↑ OL↑ OR↓</td></tr><tr><td>HHO</td><td>DET SAC DDPG TD3 πmaxQ  $\pi _ { a v g }$  FusionNet TractSeg</td><td>53.9+2.4 56.2±1.9 48.0±3.4 49.2±2.8 56.3±2.5 56.0±1.4 56.6±2.0 76.8±3.7</td><td>44.1+3.3 47.3±3.6 35.8±3.6 37.5±3.5 47.5±3.8 46.6±2.4 47.2±3.1 73.1±5.8</td><td>OR↓ 18.9+2.8 20.2±3.7 12.8±1.8 14.7±2.7 20.7±3.3 19.6±2.9 19.2±2.9 17.2+3.4</td><td>62.9+1.7 67.7±0.7 66.1±0.4 55.7±1.2 67.9±1.1 69.2±0.8 69.4±1.3 80.8±1.7</td><td>51.8+4.8 60.8±0.7 60.4±4.3 46.2±5.4 59.5±1.8 63.7±0.7 62.5±2.2 73.2±3.1</td><td>11.7±3.8 17.4±3.2 20.4±6.3 15.8±6.1 14.9±4.3 17.6±3.7 17.0±7.1 7.8±2.3</td></tr><tr><td>TI</td><td>FusionNet* DET SAC DDPG TD3</td><td>77.0±1.7 59.7±5.5 63.1±5.6 59.9±6.9 57.5±6.1</td><td>86.7±3.7 56.8±7.1 70.9±7.9 55.7±7.2 52.4±6.3</td><td>38.2±6.5 33.0±9.6 55.9±2.9 30.4±9.7 29.9±7.4</td><td>82.2±1.1 71.5±4.8 73.0±4.4 69.2±4.3 69.6±5.4 62.1±6.5</td><td>79.8±3.2 77.8±9.0 70.3±9.2 60.7±8.7 15.9±6.3</td><td>14.4±3.9 39.8±8.5 22.0±8.6 14.3±5.8</td></tr><tr><td></td><td>πmaxQ  $\pi _ { a v g }$  FusionNet DET PROB SAC DDPG TD3</td><td>63.5±6.3 62.3±5.3 64.0±6.4 57.6 62.6 62.7 57.7</td><td>71.5±7.8 71.2±8.1 65.9±7.1 52.2 62.4 64.9</td><td>53.9±3.6 60.9±9.6 41.6±8.9 28.9 37.1 41.5</td><td>73.9±4.4 73.2±2.4 74.5±6.9 43.1 50.3 49.7 50.3</td><td>70.6±8.9 71.4±4.4 73.0±8.1 38.7 50.7</td><td>21.1±8.3 23.5±7.0 24.9±8.2 40.8 49.1 50.8</td></tr><tr><td>ISRM</td><td>πmaxQ  $\pi _ { a v g }$  FusionNet</td><td>52.2 62.8 62.8 63.9</td><td>52.9 43.2 65.1 59.5 65.4</td><td>30.0 22.3 41.6 30.1 39.1</td><td>36.2 36.9 50.2 46.7 52.6 54.3</td><td>31.4 31.6 49.8 45.4</td><td>39.9 39.0 48.8 49.1 51.2</td></tr></table>

• FusionNet\* and TractSeg achieve highest scores on HCP, as TractSeg was trained on HCP data to obtain those masks which were also involved in making reference streamlines of HCP. However, masks from [74] were available for all three datasets (TractoInferno, HCP, and ISMRM), preferred for a consistent comparison across these datasets. FusionNet’s superior performance within both masks demonstrates its robustness to variations in mask quality for tractspecific tractography.

• Spurious connections made by SAC and PROB in left CG and CC\_Oc encircled in red (Fig. 4.4(a,b)) show lower OR by our method, consistent with the scores in Tables 4.3 and 4.4.

• In Fig. 4.4(a), PROB struggles to reconstruct upper portion of CST right, at pyramidal decussation (intersection of left & right CST) compared to Fusion-Net.

Note: While simpler ensembling methods $( \pi _ { m a x Q }$ and $\pi _ { A v g } )$ achieve lower scores than FusionNet, more complex RL ensembling techniques often introduce significant stability and scalability challenges. TractRLFusion addresses this challenge by learning a robust fusion policy in a data-driven manner, which simplifies integration compared to traditional RL ensemble methods. Moreover, while we demonstrate our method using three policies (SAC, DDPG, and TD3) to handle the overlapoverreach trade-of (observed in these individual policies), the proposed framework is modular and scalable. It can be easily extended to include additional policies to further capture diverse tracking behaviors and improve generalization as evident in Fig. 4.4.

## 4.3.2 Ablation Study

This section presents an ablation study of the two major components of our work, namely, Episodic Data Selection (EDS) and Multi-Critic Policy Fine-Tuning (MCPFT).

EDS involves both within-policy and across-policy selection of training trajectories, as described in Section 4.2.3. While across-policy selection ensures a diverse mix of trajectories covering all three policies across diferent brain regions, we conduct an ablation to assess the contribution of within-policy selection. Hence we compared the full EDS method (which includes both selection steps) against a variant that uses only across-policy selection. These trajectory selection strategies are denoted as EDS and $E D S _ { a c r o s s - \pi } .$ , respectively, in Table 4.5. This comparison is conducted across both the pre-training (PT) and fine-tuning (FT) stages of FusionNet.

Table 4.5: Average performance (%) obtained using Fusion-Net on test subjects from the TtoI. The table compares diferent pretraining (PT) and finetuning (FT) strategies and highlights the impact of EDS.
<table><tr><td rowspan="2">Tract</td><td colspan="4">EDSacross—π (i) (PT &amp; FT)</td><td colspan="4">(ii) EDS (PT &amp; FT)</td><td colspan="4"> $\mathbf { \overline { { E D S _ { a c r o s s - \pi } P T } } }$ </td></tr><tr><td>Dice ↑</td><td>OL ↑</td><td></td><td>OR↓</td><td>Dice ↑</td><td>OL ↑</td><td></td><td>OR↓</td><td>(iii) Dice ↑</td><td>OL ↑</td><td></td><td>OR↓</td></tr><tr><td>PYT</td><td> $\overline { { 7 0 . 4 _ { - } ^ { + } 5 . 5 } }$ </td><td>67.3±6.6</td><td></td><td> $2 3 . 4 _ { - 8 . 4 } ^ { + }$ </td><td> $\overline { { 5 8 . 6 _ { - } ^ { + } 1 . 9 } }$ </td><td></td><td>45.7±3.1</td><td>8.9±5.2</td><td>70.9±5.1</td><td></td><td> $\overline { { 6 8 . 3 _ { - } ^ { + } 7 . 2 } }$ </td><td> $\overline { { 2 3 . 9 _ { - } ^ { + } 8 . 5 } }$ </td></tr><tr><td>CG</td><td> $5 9 . 2 _ { - } ^ { + } 7 . 6$ </td><td>58.6±6.3</td><td></td><td> $3 9 . 4 _ { - } ^ { + } 8 . 2 $ </td><td> $4 2 . 9 _ { - } ^ { + } 6 . 7$ </td><td></td><td>32.1±7.0</td><td>10.7±6.3</td><td>61.4±8.2</td><td></td><td> $5 9 . 5 _ { - } ^ { + } 6 . 6$ </td><td> $3 4 . 5 _ { - } ^ { + } 9 . 3 $ </td></tr><tr><td>AF</td><td> $4 7 . 3 _ { - } ^ { + } 8 . 7$ </td><td>39.7±8.1</td><td></td><td> $2 7 . 3 _ { - } ^ { + } 9 . 8 $ </td><td> $2 3 . 3 _ { - } ^ { + } 4 . 0$ </td><td></td><td></td><td>14.4±2.1 5.3±1.5</td><td> $4 8 . 7 _ { - } ^ { + } 8 . 8$ </td><td></td><td> $4 3 . 5 _ { - } ^ { + } 8 . 5$ </td><td> $3 8 . 9 _ { - } ^ { + } 9 . 4$ </td></tr><tr><td>CC</td><td> $6 7 . 7 _ { - } ^ { + } 4 . 9$ </td><td></td><td>74.1±5.3 44.2±6.9</td><td></td><td> $6 0 . 6 _ { - } ^ { + } 3 . 9$ </td><td></td><td></td><td>50.3±2.8 14.9±2.2</td><td> $6 8 . 6 _ { - } ^ { + } 4 . 0$ </td><td></td><td> $7 0 . 2 _ { - } ^ { + } 5 . 4$ </td><td> $3 6 . 4 _ { - } ^ { + } 7 . 2$ </td></tr></table>

Table 4.5 reports the average scores across five test subjects from the TractoInferno dataset under three trajectory selection configurations:

(i) $E D S _ { a c r o s s - \pi }$ (PT & FT): Both pre-training and fine-tuning trajectories are selected using only across-policy selection.

(ii) EDS (PT & FT): Both stages utilize the full Episodic Data Selection (EDS) method, incorporating both within-policy and across-policy trajectory selection.

(iii) $E D S _ { a c r o s s - \pi } \mathrm { P T }$ & EDS FT: Pre-training uses across-policy selection only, while fine-tuning applies the full EDS method. This setting corresponds to the proposed EDS configuration used in FusionNet.

Notably, in all 3 configurations, pre-training uses a mixture of trajectories from diferent tracts, whereas fine-tuning is performed on trajectories from the same tract. It can be observed that performing both pretraining and finetuning on EDS data yields the poorest performance. This is slightly improved when using $E D S _ { a c r o s s - \pi }$ for both stages. The best performance is achieved in case (iii), highlighting the efectiveness of our trajectory selection process.

Table 4.6: Average performance metrics (in %) obtained using Tract-RLFormer highlight the impact of the MCPFT on test subjects from the TractoInferno dataset.
<table><tr><td rowspan="2">Tract</td><td colspan="3">Without MCPFT</td><td colspan="3">With MCPFT</td></tr><tr><td>Dice ↑</td><td>OL ↑</td><td>OR↓</td><td>Dice ↑</td><td>OL ↑</td><td>OR↓</td></tr><tr><td>PYT</td><td>70.9±5.1</td><td>68.3±7.2</td><td>23.9±8.5</td><td> $\overline { { 7 4 . 5 _ { - } ^ { + } 6 . 9 } }$ </td><td>73.0±8.1</td><td> $\overline { { 2 4 . 9 _ { - } ^ { + } 8 . 2 } }$ </td></tr><tr><td>CG</td><td>61.4±8.2</td><td>59.5±6.6</td><td>34.5±9.3</td><td>64.0±6.4</td><td>65.9±7.1</td><td> $4 1 . 6 _ { - } ^ { + } 8 . 9$ </td></tr><tr><td>AF</td><td>48.7±8.8</td><td>43.5±8.5</td><td>38.9±9.4</td><td>53.2±8.9</td><td>51.3±4.3</td><td>39.1±9.3</td></tr><tr><td>CC</td><td> $6 8 . 6 _ { - } ^ { + } 4 . 0$ </td><td>70.2±5.4</td><td>36.4±7.2</td><td>72.2±3.2</td><td>73.2±6.1</td><td>32.3±6.6</td></tr></table>

Moreover, Table 4.6 shows that further finetuning of our FusionNet using MCPFT efectively reduces overreach (OR) while increasing Overlap (OL), a desirable outcome in tractography, leading to a higher Dice score.

## 4.4 Summary

TractRLFusion is a policy fusion framework for White Matter Tractography, allowing eficient and efective tracking owing to its ability to fuse complementary strengths of diferent actor-critic frameworks. The ablation experiments highlight the efectiveness of our two-stage data curation, where a strong baseline policy is first developed using a combination of broad and anatomically-aware data selection (EDS). This policy is then refined by the MCPFT process. TractRLFusion robustly generalizes well, being trained only on TractoInferno and inferred on HCP and ISMRM. Finally, TractRLFusion, powered by FusionNet, lays the groundwork for a foundational tractography model, with a potential to span additional datasets and incorporate diverse specialist policies, advancing the field of White Matter Tractography.

## Chapter 5

## Conclusion and Future Work

White matter fiber tractography is a core component of modern neuroimaging, enabling detailed mapping of structural brain connectivity that is foundational to both clinical decision-making and neuroscience research. This thesis contributes to advancing reinforcement-learning–based tractography by improving the performance, robustness, and tract-specific adaptability of RL tracking policies, by proposing deep learning based methods that demonstrate robust tractography and strong generalization.

We introduced two hybrid frameworks that leverage GPTs alongside reinforcement learning for tract-specific tractography: Tract-RLFormer, designed for RL policy refinement, and TractRLFusion, aimed at policy fusion. Central to both frameworks is the Mask Refinement Module (MRM), a tract-specific mask generator that facilitates targeted training and reliable white matter tracking.

In Tract-RLFormer, pretraining was conducted using a broad and diverse collection of trajectories across multiple tracts, while fine-tuning focused on a single tract of interest. This separation allowed the model to first acquire generalizable tracking knowledge and then specialize efectively. TractRLFusion preserved this two-stage training strategy, but enhanced tract-specificity by selecting fine-tuning trajectories based on their shape similarity to atlas reference fibers. This ensured more precise alignment with the geometry of the target tract and improved downstream tractography accuracy.

Across both frameworks, broad pretraining proved valuable for generalization, whereas tract-specific fine-tuning enabled focused adaptation to the intended bundle. Ablation studies, examining alternative data-selection strategies at each stage, further validated the efectiveness of this design. Quantitative and qualitative evaluations across previously unseen datasets consistently demonstrated robust generalization, improved anatomical plausibility, and reduced false positives, without relying on ground-truth data for training. These conclusions are consistent with insights from foundation-model research [80], where large-scale pretraining combined with targeted specialization yields strong performance across downstream tasks.

The major contributions of this thesis are summarized below:

• We introduced the Mask Refinement Module (MRM) to generate tract-specific masks, accounting for inter-subject variability, enabling anatomically precise targeted tractography.

• Tract-RLFormer successfully refined a base RL policy (TD3) and as a GPTbased general backbone for tract-specific agents, Tract-RLFormer also provided the architectural basis for the subsequent fusion framework, TractRL-Fusion.

• TractRLFusion, backed by Episodic Data Selection (EDS), FusionNet, and Multi-critic policy finetuning (MCPFT) demonstrated efective data-driven fusion of TD3, SAC, and DDPG policies, addressing the persistent challenge of False positive-False negative trade-of in tractography. Its scalable design can incorporate additional actor-critic policies, ofering a flexible mechanism to capture diverse tracking behaviors.

• Extensive experiments on seven major white-matter bundles across the TractoInferno, HCP, and ISMRM datasets confirmed the robustness, generalization capability, and practical utility of both proposed frameworks.

Future Directions: While this work focused on healthy participants, an important extension is to evaluate the proposed frameworks on diseased or lesion-bearing datasets. Such validation would further establish the clinical relevance of RL-based tracking agents.

Another promising direction is the systematic exploration of tractography hyperparameters, particularly tract-specific masks. Beyond model architecture, the choice of masks, seeding strategies, and stopping criteria substantially influences reconstruction quality. Improving or learning these components, potentially jointly with the RL policy, could yield more reliable and anatomically consistent tractograms.

Finally, although TractRLFusion demonstrated successful fusion of three widely used actor–critic policies, its architecture is inherently extensible. Incorporating additional policies may capture more diverse tracking strategies and further mitigate errors due to erroneous paths or missed fiber paths.

In summary, this thesis demonstrates that combining broad pretraining, tractspecific adaptation, and multi-policy fusion ofers a powerful and flexible approach for targeted white-matter tractography. We hope that these contributions will motivate further research at the intersection of reinforcement learning, generative modeling, and neuroimaging, ultimately improving the reliability and clinical utility of white matter fiber tractography.

# List of Publications

## Publications included in thesis

1 Ankita Joshi, Ashutosh Sharma, Anoushkrit Goel, Ranjeet Ranjan Jha, Chirag Ahuja, Arnav Bhavsar, Aditya Nigam, “TractRLFusion: A GPT-based Multi-Critic Policy Fusion Framework for Fiber Tractography” Accepted in 2026 IEEE 23<sup>rd</sup> International Symposium on Biomedical Imaging (ISBI-2026).

2 Ankita Joshi, Ashutosh Sharma, Anoushkrit Goel, Ranjeet Ranjan Jha, Chirag Kamal Ahuja, Arnav Bhavsar, Aditya Nigam, “Tract-RLFormer: A Tract-Specific RL policy based Decoder-only Transformer Network” in $2 7 ^ { t h }$ International Conference on Pattern Recognition (ICPR-2024).

## Others

3 Anoushkrit Goel, Simroop Singh, Ankita Joshi, Ranjeet Ranjan Jha, Chirag Ahuja, Aditya Nigam, Arnav Bhavsar, “TrackletGPT: A Language-like GPT Framework for White Matter Tract Segmentation” Accepted in 2026 IEEE $2 3 ^ { r d }$ International Symposium on Biomedical Imaging (ISBI-2026).

4 Anoushkrit Goel, Simroop Singh, Ankita Joshi, Ranjeet Ranjan Jha, Chirag Ahuja, Aditya Nigam, Arnav Bhavsar, “TractoGPT: A GPT Architecture

for White Matter Tract Segmentation” in 2025 IEEE $2 2 ^ { n d }$ International Symposium on Biomedical Imaging (ISBI-2025).

5 Anoushkrit Goel, Bipanjit Singh, Ankita Joshi, Ranjeet Ranjan Jha, Chirag Ahuja, Aditya Nigam, Arnav Bhavsar, “TractoEmbed: Modular Multilevel Embedding framework for white matter tract segmentation” in $2 7 ^ { t h }$ International Conference on Pattern Recognition (ICPR-2024).

6 Ranjeet Ranjan Jha, Hritik Gupta, S Pathak, Ankita Joshi, W Schneider, BVR Kumar, A Bhavsar, A Nigam, “PA-GAN: Parallel Attention Based GAN for Enhancement of fODF” in 2023 IEEE 20<sup>th</sup> International Symposium on Biomedical Imaging (ISBI-2023).

## Bibliography

[1] A. Krizhevsky, I. Sutskever, and G. E. Hinton, “Imagenet classification with deep convolutional neural networks,” Advances in neural information processing systems, vol. 25, 2012.

[2] S. Ren, K. He, R. Girshick, and J. Sun, “Faster r-cnn: Towards real-time object detection with region proposal networks,” Advances in neural information processing systems, vol. 28, 2015.

[3] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in International Conference on Medical image computing and computer-assisted intervention. Springer, 2015, pp. 234–241.

[4] C. Ledig, L. Theis, F. Huszár, J. Caballero, A. Cunningham, A. Acosta, A. Aitken, A. Tejani, J. Totz, Z. Wang et al., “Photo-realistic single image superresolution using a generative adversarial network,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 4681–4690.

[5] I. Block Imaging, “Siemens 3t mri scanners compared,” https://www. blockimaging.com/blog/siemens-3t-mri-scanners-compared, accessed: 2025-12- 07.

[6] R. H. Hashemi, W. G. Bradley, and C. J. Lisanti, MRI: the basics. Lippincott Williams & Wilkins, 2010.

[7] J. Hsieh, “Computed tomography: principles, design, artifacts, and recent advances,” 2003.

[8] M. E. Phelps, “Molecular imaging and its biological applications,” Eur J Nucl Med Mol Imaging, vol. 31, p. 1544, 2004.

[9] W.-D. Heiss, P. Raab, and H. Lanfermann, “Multimodality assessment of brain tumors and tumor recurrence,” Journal of Nuclear Medicine, vol. 52, no. 10, pp. 1585–1600, 2011.

[10] S. K. Zhou, H. Greenspan, C. Davatzikos, J. S. Duncan, B. Van Ginneken, A. Madabhushi, J. L. Prince, D. Rueckert, and R. M. Summers, “A review of deep learning in medical imaging: Imaging traits, technology trends, case studies with progress highlights, and future promises,” Proceedings of the IEEE, vol. 109, no. 5, pp. 820–838, 2021.

[11] T. Küstner, C. Qin, C. Sun, L. Ning, and C. M. Scannell, “The intelligent imaging revolution: artificial intelligence in mri and mrs acquisition and reconstruction,” Magnetic Resonance Materials in Physics, Biology and Medicine, vol. 37, no. 3, pp. 329–333, 2024.

[12] R. R. Jha, G. Jaswal, A. Nigam, and A. Bhavsar, “Advances and challenges in fmri and dti techniques,” Intelligent data security solutions for e-health applications, pp. 77–90, 2020.

[13] P. J. Basser, J. Mattiello, and D. LeBihan, “Mr difusion tensor spectroscopy and imaging,” Biophysical journal, vol. 66, no. 1, pp. 259–267, 1994.

[14] J.-D. Tournier, F. Calamante, and A. Connelly, “Robust determination of the fibre orientation distribution in difusion mri: non-negativity constrained superresolved spherical deconvolution,” Neuroimage, vol. 35, no. 4, pp. 1459–1472, 2007.

[15] D. S. Tuch, “Q-ball imaging,” Magnetic Resonance in Medicine: An Oficial Journal of the International Society for Magnetic Resonance in Medicine, vol. 52, no. 6, pp. 1358–1372, 2004.

[16] M. Descoteaux, E. Angelino, S. Fitzgibbons, and R. Deriche, “Regularized, fast, and robust analytical q-ball imaging,” Magnetic Resonance in Medicine: An Of-

ficial Journal of the International Society for Magnetic Resonance in Medicine, vol. 58, no. 3, pp. 497–510, 2007.

[17] J. L. Andersson and S. N. Sotiropoulos, “An integrated approach to correction for of-resonance efects and subject movement in difusion mr imaging,” Neuroimage, vol. 125, pp. 1063–1078, 2016.

[18] N. J. Tustison, B. B. Avants, P. A. Cook, Y. Zheng, A. Egan, P. A. Yushkevich, and J. C. Gee, “N4itk: improved n3 bias correction,” IEEE transactions on medical imaging, vol. 29, no. 6, pp. 1310–1320, 2010.

[19] J.-D. Tournier, R. Smith, D. Rafelt, R. Tabbara, T. Dhollander, M. Pietsch, D. Christiaens, B. Jeurissen, C.-H. Yeh, and A. Connelly, “Mrtrix3: A fast, flexible and open software framework for medical image processing and visualisation,” Neuroimage, vol. 202, p. 116137, 2019.

[20] B. C. Lowekamp, D. T. Chen, L. Ibáñez, and D. Blezek, “The design of simpleitk,” Frontiers in neuroinformatics, vol. 7, p. 45, 2013.

[21] B. B. Avants et al., “Advanced normalization tools,” Insight j, vol. 2, no. 365, pp. 1–35, 2009.

[22] B. Fischl, “Freesurfer,” Neuroimage, vol. 62, no. 2, pp. 774–781, 2012.

[23] M. Bastiani, M. Cottaar, S. P. Fitzgibbon, S. Suri, F. Alfaro-Almagro, S. N. Sotiropoulos, S. Jbabdi, and J. L. Andersson, “Automated quality control for within and between studies difusion mri data using a non-parametric framework for movement and distortion correction,” Neuroimage, vol. 184, pp. 801–812, 2019.

[24] I. Oguz, M. Farzinfar, J. Matsui, F. Budin, Z. Liu, G. Gerig, H. J. Johnson, and M. Styner, “Dtiprep: quality control of difusion-weighted images,” Frontiers in neuroinformatics, vol. 8, p. 4, 2014.

[25] P. J. Basser and C. Pierpaoli, “Microstructural and physiological features of tissues elucidated by quantitative-difusion-tensor mri,” Journal of magnetic resonance, vol. 213, no. 2, pp. 560–570, 2011.

[26] C. Pierpaoli and P. J. Basser, “Toward a quantitative assessment of difusion anisotropy,” Magnetic resonance in Medicine, vol. 36, no. 6, pp. 893–906, 1996.

[27] P. B. G. Alvez, “Inference of a human brain fiber bundle atlas from high angular resolution difusion imaging,” Ph.D. dissertation, Université Paris Sud-Paris XI, 2011.

[28] J. Wasserthal, P. Neher, and K. H. Maier-Hein, “Tractseg-fast and accurate white matter tract segmentation,” NeuroImage, vol. 183, pp. 239–253, 2018.

[29] K. H. Maier-Hein et al., “The challenge of mapping the human connectome based on difusion tractography,” Nature communications, vol. 8, no. 1, p. 1349, 2017.

[30] B. Jeurissen, M. Descoteaux, S. Mori, and A. Leemans, “Difusion mri fiber tractography of the brain,” NMR in Biomedicine, vol. 32, no. 4, p. e3785, 2019.

[31] K. G. Schilling, A. Daducci, K. Maier-Hein, C. Poupon, J.-C. Houde, V. Nath, A. W. Anderson, B. A. Landman, and M. Descoteaux, “Challenges in difusion mri tractography–lessons learned from international benchmark competitions,” Magnetic resonance imaging, vol. 57, pp. 194–209, 2019.

[32] P. Poulin, D. Jörgens, P.-M. Jodoin, and M. Descoteaux, “Tractography and machine learning: Current state and open challenges,” Magnetic resonance imaging, vol. 64, pp. 37–48, 2019.

[33] S. Mori, B. J. Crain, V. P. Chacko, and P. C. Van Zijl, “Three-dimensional tracking of axonal projections in the brain by magnetic resonance imaging,” Annals of Neurology: Oficial Journal of the American Neurological Association and the Child Neurology Society, vol. 45, no. 2, pp. 265–269, 1999.

[34] P. J. Basser, S. Pajevic, C. Pierpaoli, J. Duda, and A. Aldroubi, “In vivo fiber tractography using dt-mri data,” Magnetic resonance in medicine, vol. 44, no. 4, pp. 625–632, 2000.

[35] T. E. Behrens, M. W. Woolrich, M. Jenkinson, H. Johansen-Berg, R. G. Nunes, S. Clare, P. M. Matthews, J. M. Brady, and S. M. Smith, “Characterization and

propagation of uncertainty in difusion-weighted mr imaging,” Magnetic Resonance in Medicine: An Oficial Journal of the International Society for Magnetic Resonance in Medicine, vol. 50, no. 5, pp. 1077–1088, 2003.

[36] J. D. Tournier, F. Calamante, A. Connelly et al., “Improved probabilistic streamlines tractography by 2nd order integration over fibre orientation distributions,” in Proceedings of the international society for magnetic resonance in medicine, vol. 1670. Stockholm, 2010, p. 2010.

[37] G. J. Parker, H. A. Haroon, and C. A. Wheeler-Kingshott, “A framework for a streamline-based probabilistic index of connectivity (pico) using a structural interpretation of mri difusion measurements,” Journal of Magnetic Resonance Imaging: An Oficial Journal of the International Society for Magnetic Resonance in Medicine, vol. 18, no. 2, pp. 242–254, 2003.

[38] T. E. Behrens, H. Johansen-Berg, M. W. Woolrich, S. M. Smith, C. A. Wheeler-Kingshott, P. A. Boulby, G. J. Barker, E. Sillery, K. Sheehan, O. Ciccarelli et al., “Non-invasive mapping of connections between human thalamus and cortex using difusion imaging,” Nature neuroscience, vol. 6, no. 7, pp. 750–757, 2003.

[39] B. W. Kreher, I. Mader, and V. G. Kiselev, “Gibbs tracking: a novel approach for the reconstruction of neuronal pathways,” Magnetic Resonance in Medicine: An Oficial Journal of the International Society for Magnetic Resonance in Medicine, vol. 60, no. 4, pp. 953–963, 2008.

[40] R. E. Smith, J.-D. Tournier, F. Calamante, and A. Connelly, “Anatomicallyconstrained tractography: improved difusion mri streamlines tractography through efective use of anatomical information,” Neuroimage, vol. 62, no. 3, pp. 1924–1938, 2012.

[41] G. Girard, K. Whittingstall, R. Deriche, and M. Descoteaux, “Towards quantitative connectivity analysis: reducing tractography biases,” Neuroimage, vol. 98, pp. 266–278, 2014.

[42] P. Poulin et al., “Tractoinferno-a large-scale, open-source, multi-site database for machine learning dmri tractography,” Scientific Data, vol. 9, no. 1, p. 725, 2022.

[43] P. F. Neher, M. Götz, T. Norajitra, C. Weber, and K. H. Maier-Hein, “A machine learning based approach to fiber tractography using classifier voting,” in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2015, pp. 45–52.

[44] P. Poulin et al., “Learn to track: deep learning for tractography,” in MICCAI 2017: 20th International Conference, Quebec City, QC, Canada, September 11- 13, 2017, Proceedings, Part I 20. Springer, 2017, pp. 540–547.

[45] I. Benou and T. Riklin Raviv, “Deeptract: A probabilistic deep learning framework for white matter fiber tractography,” in MICCAI: Shenzhen, China, October 13–17, 2019. Springer, 2019, pp. 626–635.

[46] V. Wegmayr, G. Giuliari, S. Holdener, and J. Buhmann, “Data-driven fiber tractography with neural networks,” in 2018 IEEE 15th international symposium on biomedical imaging (ISBI 2018). IEEE, 2018, pp. 1030–1033.

[47] V. Wegmayr and J. M. Buhmann, “Entrack: Probabilistic spherical regression with entropy regularization for fiber tractography,” International Journal of Computer Vision, vol. 129, no. 3, pp. 656–680, 2021.

[48] A. Théberge et al., “Track-to-learn: A general framework for tractography with deep reinforcement learning,” MIA, vol. 72, p. 102093, 2021.

[49] A. Théberge, C. Desrosiers, A. Boré, M. Descoteaux, and P.-M. Jodoin, “What matters in reinforcement learning for tractography,” MIA, vol. 93, p. 103085, 2024.

[50] A. Théberge, M. Descoteaux, and P.-M. Jodoin, “Tractoracle: towards an anatomically-informed reward function for rl-based tractography,” in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2024, pp. 476–486.

[51] L. Bello, A. Gambini, A. Castellano, G. Carrabba, F. Acerbi, E. Fava, C. Giussani, M. Cadioli, V. Blasi, A. Casarotti et al., “Motor and language dti fiber tracking combined with intraoperative subcortical mapping for surgical removal of gliomas,” Neuroimage, vol. 39, no. 1, pp. 369–382, 2008.

[52] F. Rheault, E. St-Onge, J. Sidhu, K. Maier-Hein, N. Tzourio-Mazoyer, L. Petit, and M. Descoteaux, “Bundle-specific tractography with incorporated anatomical and orientational priors,” Neuroimage, vol. 186, pp. 382–398, 2019.

[53] G. Brockman, V. Cheung, L. Pettersson, J. Schneider, J. Schulman, J. Tang, and W. Zaremba, “Openai gym,” arXiv preprint arXiv:1606.01540, 2016.

[54] R. S. Sutton, D. McAllester, S. Singh, and Y. Mansour, “Policy gradient methods for reinforcement learning with function approximation,” Advances in neural information processing systems, vol. 12, 1999.

[55] T. Lillicrap, “Continuous control with deep reinforcement learning,” arXiv preprint arXiv:1509.02971, 2015.

[56] S. Fujimoto, H. Hoof, and D. Meger, “Addressing function approximation error in actor-critic methods,” in ICML. PMLR, 2018, pp. 1587–1596.

[57] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, “Soft actor-critic: Of-policy maximum entropy deep reinforcement learning with a stochastic actor,” in ICML. PMLR, 2018, pp. 1861–1870.

[58] D. E. Rumelhart, G. E. Hinton, R. J. Williams et al., “Learning internal representations by error propagation,” 1985.

[59] S. Hochreiter and J. Schmidhuber, “Long short-term memory,” Neural computation, vol. 9, no. 8, pp. 1735–1780, 1997.

[60] A. Vaswani et al., “Attention is all you need,” Advances in neural information processing systems, vol. 30, 2017.

[61] A. Radford, K. Narasimhan, T. Salimans, I. Sutskever et al., “Improving language understanding by generative pre-training,” 2018.

[62] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training of deep bidirectional transformers for language understanding,” in Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), 2019, pp. 4171–4186.

[63] H. Zhou, Zhang et al., “Informer: Beyond eficient transformer for long sequence time-series forecasting,” in Proceedings of the AAAI conference on artificial intelligence, vol. 35, no. 12, 2021, pp. 11 106–11 115.

[64] J. Jumper, R. Evans, A. Pritzel, T. Green et al., “Highly accurate protein structure prediction with alphafold,” nature, vol. 596, no. 7873, pp. 583–589, 2021.

[65] M. Chen, “Evaluating large language models trained on code,” arXiv preprint arXiv:2107.03374, 2021.

[66] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” arXiv preprint arXiv:2010.11929, 2020.

[67] D. C. Van Essen, K. Ugurbil, E. Auerbach, D. Barch, T. E. Behrens, R. Bucholz, A. Chang, L. Chen, M. Corbetta, S. W. Curtiss et al., “The human connectome project: a data acquisition perspective,” Neuroimage, vol. 62, no. 4, pp. 2222– 2231, 2012.

[68] “scilpy,” https://github.com/scilus/scilpy.

[69] F. Rheault, “Population average atlas for recobundlesx,” May 2023. [Online]. Available: https://doi.org/10.5281/zenodo.7950602

[70] L. Chen et al., “Decision transformer: Reinforcement learning via sequence modeling,” NeurIPS, vol. 34, pp. 15 084–15 097, 2021.

[71] E. St-Onge, E. Garyfallidis, and D. L. Collins, “Fast streamline search: An exact technique for difusion mri tractography,” Neuroinformatics, vol. 20, no. 4, pp. 1093–1104, 2022.

[72] K. Kamagata, C. Andica, W. Uchida, K. Takabayashi, Y. Saito, M. Lukies, A. Hagiwara, S. Fujita, T. Akashi, A. Wada et al., “Advancements in difusion mri tractography for neurosurgery,” Investigative Radiology, vol. 59, no. 1, pp. 13–25, 2024.

[73] T. V. Marinov, A. Agarwal, and M. Trofin, “Ofline imitation learning from multiple baselines with applications to compiler optimization,” arXiv preprint arXiv:2403.19462, 2024.

[74] A. Joshi, A. Sharma, A. Goel, R. R. Jha, C. K. Ahuja, A. Bhavsar, and A. Nigam, “Tract-rlformer: A tract-specific rl policy based decoder-only transformer network,” in International Conference on Pattern Recognition. Springer, 2024, pp. 258–275.

[75] E. Garyfallidis, M. Brett, M. M. Correia, G. B. Williams, and I. Nimmo-Smith, “Quickbundles, a method for tractography simplification,” Frontiers in neuroscience, vol. 6, p. 175, 2012.

[76] Y. Li, S. Ma, J. Zhao, Q. Li, and X. Sheng, “Farthest streamline sampling for the uniform distribution of forearm muscle fiber tracts from difusion tensor imaging,” arXiv preprint arXiv:2306.13969, 2023.

[77] S. Mysore, G. Cheng, Y. Zhao, K. Saenko, and M. Wu, “Multi-critic actor learning: Teaching rl policies to act with style,” in International Conference on Learning Representations, 2022.

[78] J.-D. Tournier, F. Calamante, and A. Connelly, “Mrtrix: difusion tractography in crossing fiber regions,” International journal of imaging systems and technology, vol. 22, no. 1, pp. 53–66, 2012.

[79] F. Dumais, J. H. Legarreta, C. Lemaire, P. Poulin, F. Rheault, L. Petit, M. Barakovic, S. Magon, M. Descoteaux, P.-M. Jodoin et al., “Fiesta: Autoencoders for accurate fiber segmentation in tractography,” NeuroImage, vol. 279, p. 120288, 2023.

## Bibliography

[80] R. Bommasani, “On the opportunities and risks of foundation models,” arXiv preprint arXiv:2108.07258, 2021.