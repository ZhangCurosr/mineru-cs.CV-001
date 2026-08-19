# Learning latent progression states from spatial heterogeneity in uterine histopathology

Qiming He<sup>1,5,+,\*</sup>, Yan Liu<sup>3,+</sup>, Shuang Ge<sup>2,4,+</sup>, Fan Yang<sup>2</sup>, Yuxiang Wang<sup>3</sup>, Ieng Man Zhang<sup>2</sup>, Jing Yang<sup>3</sup>, Zihao Jia<sup>1</sup>, Ajin Hu<sup>3</sup>, Yexing Zhang<sup>7</sup>, Zixiu Song<sup>3</sup>, Qiang Huang<sup>2</sup>, Xiaoya Zhao<sup>3</sup>, Zihan Wang<sup>2</sup>, Xianjing Zheng<sup>3</sup>, Yijun Zheng<sup>3</sup>, Liling Lin<sup>3</sup>, Shuxing Liu<sup>3</sup>, Bin Bao<sup>3</sup>, Yue Xie<sup>3</sup>, Tian Guan<sup>2,\*</sup>, Yonghong He<sup>2,5,6\*</sup>, and Congrong Liu<sup>3,\*</sup>

<sup>1</sup>Fuzhou University Afiliated Provincial Hospital, Interdisciplinary Institute for   
Medical Engineering, Fuzhou University, Fuzhou, China   
<sup>2</sup>Institute of Biopharmaceutical and Health Engineering, Tsinghua Shenzhen   
International Graduate School, Tsinghua University, Shenzhen, China   
<sup>3</sup>Department of Pathology, Third Hospital, School of Basic Medical Sciences, Peking   
University Health Science Center, Beijing, China   
<sup>4</sup>Peng Cheng Laboratory, Shenzhen, China   
<sup>5</sup>Medical Optical Technology R&D Center, Research Institute of Tsinghua, Pearl   
River Delta, Guangzhou, China   
<sup>6</sup>Jinfeng Laboratory, Chongqing, China   
<sup>7</sup>Jinan Inspur Data Technology Co., Ltd., Jinan, China   
<sup>+</sup>Qiming He, Yan Liu, and Shuang Ge contribute equally to this work.   
Corresponding authors: he-qm@fzu.edu.cn; guantian@sz.tsinghua.edu.cn;   
Congrong liu@163.com; heyh@sz.tsinghua.edu.cn

## Abstract

Tumor progression is accompanied by changes in architecture, morphology and microenvironmental organization, yet progression-associated heterogeneity is usually compressed into static diagnostic categories in histopathology. Here we present SpaTIE, a uterus-specific computa tional pathology framework that learns morphology-aware representations and organizes spatial histopathological heterogeneity into progression-associated tumor states. SpaTIE was developed using 10,426 uterine hematoxylin and eosin whole-slide images and evaluated in TCGA-UCEC and TCGA-UCS cohorts. The learned representations formed morphology manifolds, supported diagnostic, molecular and survival-related prediction tasks, and localized attention to informative tumor regions. Beyond supervised prediction, SpaTIE inferred tumor-state axes from crosssectional morphology without temporal or molecular supervision. These morphology-derived states were spatially coherent and showed associations with clinicopathological variables and survival outcomes, while not simply recapitulating staging or diagnostic labels. Integrative multi-omics analyses linked the inferred states to DNA methylation, somatic copy-number variation, mutation, RNA-seq and RPPA profiles, highlighting molecular programs related to chromatin regula tion, copy-number-associated structural variation, receptor tyrosine kinase signaling, cell adhesion, extracellular-matrix remodeling and metabolic adaptation. Progression-guided virtual perturbation further prioritized molecular features coupled to the morphology-derived state organization. Together, these findings suggest that uterine histopathology contains recoverable progressionassociated tumor-state information and establish SpaTIE as a framework for connecting spatia morphology with multi-omics-informed tumor-state discovery.

Keywords: Uterine Histopathology, Progression States, Computational Pathology, Foundation Model

## 1 Introduction

Tumor progression is not a discrete event but a continuum of coordinated changes in epithelial architecture, cellular morphology, stromal organization, immune contexture and tumor–microenvironment interactions [1, 2]. In uterine malignancies, such changes are particularly heterogeneous, spanning indolent endometrioid tumors, high-grade carcinomas and aggressive carcinosarcomas with biphasic epithelial–mesenchymal features [3, 4]. Although routine hematoxylin and eosin (H&E) histopathology remains the cornerstone of diagnosis, grading and clinical decision-making, it is usually interpreted through discrete categories such as histological subtype, tumor grade and stage. These categories are clinically useful, but they compress a continuum of morphological and biological variation into static diagnostic labels. Sequencing approaches can generate multi-omics data to characterize the molecular trajectories underlying tumor progression; however, these techniques are associated with substantially high costs. As a result, progression-associated tumor states and within-category heterogeneity may remain under-recognized.

A central but underexplored question is whether routine histopathology contains latent information about progression-associated tumor organization beyond these conventional labels. Tumors are spatially heterogeneous tissues in which morphologically distinct regions often coexist within the same specimen, including areas with preserved glandular diferentiation, solid growth, stromal reaction, necrosis, inflammatory infiltration and invasive or dediferentiated phenotypes [5, 6]. Rather than representing only sampling noise, such spatial heterogeneity may reflect coexisting tumor states distributed along a progression-associated morphological axis. This concept parallels trajectory inference in single-cell biology, where asynchronous cellular states sampled at one time point can be computationally ordered to recover latent biological processes [7, 8, 9]. Recent studies in spatial transcriptomics and single-cell genomics further suggest that tissue organization may preserve latent developmental or pathological state structure [10, 11]. However, whether analogous progression-associated tumor states can be extracted from routine histopathology images, without longitudinal sampling or molecular supervision, remains largely unresolved.

Recent advances in computational pathology and self-supervised representation learning have enabled large-scale extraction of morphological features from whole-slide images (WSIs) [12, 13, 14, 15, 16, 17]. Pathology foundation models can learn transferable tissue representations and support diverse downstream tasks, including diagnosis, molecular prediction and prognosis [18, 19, 20, 21, 22]. Nevertheless, most existing approaches still treat histopathology as a static prediction problem, optimizing representations for classification or regression rather than resolving continuous disease-state organization. In parallel, trajectory inference methods have been widely used in single-cell and spatial omics to reconstruct dynamic biological processes, but their application to routine histopathology remains limited [23, 24, 25, 26]. This gap motivates a trajectory-aware computational pathology framework that can transform spatial morphological heterogeneity into clinically and biologically interpretable progression-associated tumor states.

Here, we present SpaTIE, designed to identify latent progression-associated states from spatial morphological heterogeneity in uterine histopathology (Fig. 1a). We first formulated a theoretical basis showing how coexisting morphological states within histopathology images give rise to progressionassociated tumor-state organization (Sec. 2.1). Building on this rationale, SpaTIE learns uterusspecific morphological representations from 10,426 H&E WSIs and organizes cross-sectional morphology into continuous tumor-state axes without temporal or molecular supervision (Fig. 1b-d). We evaluated SpaTIE using large-scale uterine pathology data and independent public cohorts, covering endometrial carcinoma and uterine carcinosarcoma. SpaTIE identified spatially coherent progressionassociated organization, and the inferred states showed associations with clinicopathological variables, molecular alterations and survival outcomes (Fig. 1f,h). Recent foundation models have demonstrated that histopathological morphology can be aligned with spatial transcriptomic representations at scale [27]. Beyond clinicopathological associations, SpaTIE-derived states were linked to multi-omics variation across DNA methylation, somatic copy-number variation, mutation, transcriptomic and proteomic profiles (Fig. 1e,g,i). Trajectory-associated molecular analyses and progression-guided virtual perturbation further revealed candidate regulatory programs and pathways related to epithelial–mesenchymal transition, chromatin dysregulation, metabolic adaptation and microenvironmental remodeling (Fig. 1j). Together, these findings suggest that routine histopathology contains recoverable progressionassociated tumor-state information, providing a framework for trajectory-aware computational pathology and multi-omics-informed characterization of tumor states from H&E morphology.

![](images/70c30692cb891591278b4d09c61f5f95b851ad20ac1e538eb6759e89ec0d49eb.jpg)  
Figure 1: Overview of SpaTIE. a, H&E histopathology provides spatial morphology that may contain latent information. b, Study cohorts and data resources, including the PK3-Uter pretraining cohort and independent TCGA-UCEC and TCGA-UCS cohorts used for evaluation. c, SpaTIE performed uterus-specific representation learning and beat universal pathology foundation model (UPFM) across downstream clinical tasks. d, Mapping from static histopathology to progression states (pseudotime). e, Analysis workflow for linking SpaTIE-derived progression states to multi-omics features. f, Association between SpaTIE-derived progression states and fraction genome altered in TCGA-UCS. g, Summary of pathway-level associations with low- versus high-pseudotime groups across cohorts and omics modalities. h, Example survival stratification by pseudotime group in the TCGA-UCS high-FGA subgroup. i, Representative state-ordered heatmap of estrogen-response RNA-seq features in TCGA-UCS. j, Cross-omics virtual-perturbation schematic illustrating candidate relationships among epigenomic, transcriptomic and protein-level features.

## 2 Results

## 2.1 Conceptual foundation

Tumor progression is a continuous but spatially asynchronous process. Within a single histopathology specimen, tumor regions often exhibit substantial morphological diversity, including preserved glandular diferentiation, solid growth, stromal activation, necrosis, inflammatory infiltration, and invasive or dediferentiated phenotypes. Rather than representing only sampling variability, these coexisting patterns may reflect tumor states distributed along a progression-associated morphological continuum [2, 6]. We therefore reasoned that spatial heterogeneity in routine histopathology may preserve information about disease-state organization that is not fully captured by a single diagnostic label.

This concept is analogous to trajectory inference in single-cell biology, where asynchronously sampled cellular states can be computationally organized to recover latent biological processes [7, 8, 9]. We hypothesized that histopathological regions with similar progression-associated morphology should occupy nearby positions in a representation space and, across regions and patients, form a continuous tumor-state organization. Under this view, whole-slide histopathology images are spatial snapshots containing mixtures of coexisting tumor states, rather than single homogeneous disease categories.

To formalize this intuition, let $x \in \mathcal { X }$ denote a local histopathological observation, such as an image patch, and let $t \in \tau$ represent an unobserved progression-associated tumor state. We assume that tissue morphology arises from a stochastic generative process:

$$
x = g ( t , \epsilon ) , \quad \epsilon \sim p ( \epsilon ) ,\tag{1}
$$

where $g ( \cdot )$ describes progression-associated morphological variation and ϵ captures patient-specific variability, local microenvironmental diferences, and imaging noise. Under this formulation, histopathology specimens consist of spatially distributed observations sampled from multiple latent tumor states:

$$
p ( x ) = \int _ { T } p ( x | t ) p ( t ) d t .\tag{2}
$$

Recent advances in self-supervised representation learning suggest that morphology-aware encoders can preserve high-level tissue organization within latent embedding spaces [13, 14, 19]. Let $z _ { i }$ denote the latent representation of observation $x _ { i }$ . If progression-associated morphological similarity is preserved after representation learning, observations arising from nearby tumor states should remain locally close in the embedding space. This local organization can be characterized through a similarity kernel:

$$
K _ { i j } = \exp \left( - \frac { \| z _ { i } - z _ { j } \| ^ { 2 } } { \sigma ^ { 2 } } \right) ,\tag{3}
$$

which defines a neighborhood structure over tissue regions. Difusion- or graph-based trajectory inference methods can then be applied to organize tissue regions along low-dimensional state continua [28, 29]. Under this framework, latent progression-associated states emerge as geometric organization within morphology-aware representation spaces, rather than as explicit temporal measurements.

Motivated by this biological and computational rationale, we developed SpaTIE, a trajectoryaware computational pathology framework designed to recover progression-associated tumor states from uterine histopathology. SpaTIE combines uterus-specific representation learning with tumorstate inference to characterize morphology-associated organization across tissue regions and patients without requiring temporal sampling or molecular supervision.

## 2.2 Overview of SpaTIE

An overview of SpaTIE is illustrated in Fig. 1. Tumor progression is commonly studied using molecular profiling approaches, which can provide biologically informative state trajectories but remain dificult to apply at scale in routine pathology (Fig. 1a). In contrast, H&E histopathology is widely available and inherently reflects spatial morphological heterogeneity within tumor tissues. SpaTIE was designed based on the hypothesis that these spatially coexisting morphological states may preserve progressionassociated organization that can be recovered computationally. To support morphology-aware state modeling, we first constructed a large-scale uterine histopathology resource (Fig. 1b). Specifically, we collected 10,426 WSIs from Peking University Third Hospital, Beijing, China for uterus-specific selfsupervised pretraining (PK3-Uter), covering diverse uterine pathological subtypes and morphological patterns. In addition, TCGA-UCEC and TCGA-UCS cohorts were incorporated as independent external datasets for clinicopathological evaluation, survival analysis, and multi-omics association studies [3].

SpaTIE performs uterus-specific morphological representation learning using a pathology foundation model adapted through DINOv2-style self-supervised learning and LoRA-based eficient finetuning [30, 31] (Fig. 1c). Through this process, histopathological regions with similar morphology and progression-associated characteristics become locally organized within a shared latent embedding space. In feature-space comparisons, SpaTIE showed more eficient manifold construction for uterine disease types than a universal pathology foundation model (UPFM) pretrained on multi-organ and multi-disease datasets [32]. On TCGA-UCEC and TCGA-UCS, SpaTIE also improved or matched UPFM across downstream tasks covering morphological diagnosis, gene prediction, and survival prediction (Sec. 2.3).

Building upon the learned embedding space, SpaTIE organizes cross-sectional histopathology into latent progression-associated tumor-state axes without requiring temporal sampling or molecular supervision (Fig. 1d, Sec. 2.4). Morphology-aware embeddings are arranged into continuous manifolds, enabling slide-level and region-level characterization of tumor-state organization across tissues and patients. The inferred states are then linked to clinicopathological variables, molecular alterations, and spatial morphology patterns. To evaluate the biological and clinical relevance of morphologyderived tumor states, we analyzed their associations with histological subtype, tumor grade, genomic alteration burden, prognosis, and patient survival (Fig. 1f,h; Sec. 2.4). Beyond clinicopathologica analyses, we further integrated these states with multi-omics profiles, including DNA methylation, somatic copy-number variation, mutation, transcriptomic, and proteomic data (Fig. 1e,g,i; Sec. 2.5). Trajectory-aware molecular analyses and progression-guided virtual perturbation further identified candidate pathways and regulatory programs potentially associated with morphology-derived tumor-state organization in uterine tumors (Fig. 1j; Sec. 2.6).

## 2.3 SpaTIE learns a static morphology-aware representation space

Before inferring progression-associated states, we first examined whether SpaTIE could construct a biologically meaningful static representation space for uterine histopathology. We compared SpaTIE with UPFM and baseline representations on TCGA-UCEC and TCGA-UCS. UMAP visualization of patch-level embeddings showed that SpaTIE organized uterine histopathological patches into more structured and fine-grained latent manifolds than the baseline model, while maintaining local separation of morphologically distinct tissue regions (Fig. 2a). Compared with UPFM, SpaTIE preserved a similarly rich but more uterus-adapted embedding structure, suggesting that domain-adapted selfsupervised learning improved the representation of uterine-specific morphology.

We next quantified the quality of the static representation space using manifold structure quality (MSQ) and representation entropy (RE) (see details in Sec. 9). SpaTIE achieved the best overall performance across the two cohorts, with MSQ values of 45.833 and 35.225 and RE values of 3.958 and 3.910 on TCGA-UCEC and TCGA-UCS, respectively (Fig. 2b). These results indicate that SpaTIE improves both structural organization and information distribution in the latent histopathologica representation, providing a stable feature basis for subsequent tumor-state inference.

To further interpret the learned representation space, we inspected representative cluster-center patches derived from the SpaTIE embeddings (Fig. 2c). These cluster centers covered diverse tissue and tumor morphologies, including tumor epithelium, stromal regions, necrotic areas, inflammatory components, tissue edges, and morphologically heterogeneous tumor regions. SpaTIE not only sep arated broad tissue compartments but also captured finer variation within tumor regions, such as glandular architecture, solid growth, and poorly diferentiated patterns. This suggests that the static representation space learned by SpaTIE reflects biologically relevant morphological organization rather than only low-level visual similarity.

![](images/19eaccbcbbcee859935f34d0b17997d58d71d3bd7fa625e80e938c311d852ed9.jpg)

![](images/5bbadeaba925c8dadfe5803a903302d2de2ccfd71cb436b39dba824680e3c7b5.jpg)

![](images/23075b3caa37d10fe8c09dfe6159b047e3800beed6f8ec55b444de7fb23f80d2.jpg)

<table><tr><td rowspan="2">Metric</td><td colspan="3">TCGA-UCEC</td><td colspan="3">TCGA-UCS</td></tr><tr><td>SpaTIE</td><td>UPFM</td><td>Baseline</td><td>SpaTIE</td><td>UPFM</td><td>Baseline</td></tr><tr><td>MSQ</td><td>45.833</td><td>48.713</td><td>62.272</td><td>35.225</td><td>27.854</td><td>67.575</td></tr><tr><td>RE</td><td>3.958</td><td>3.904</td><td>3.552</td><td>3.910</td><td>3.90</td><td>3.501</td></tr></table>

![](images/4a9691e7f482391f1eb40d28ff6a2186f095f7c862dad4c49fb37ed12d50c57b.jpg)

![](images/8877102d4a443de9292e37df1b8b2bfc13fd7edfb5eb82262c7c35aae62bca34.jpg)

![](images/0e4183a00043a83286c05a321fd0d80bea333f7d0ad79b3be74ec12623105393.jpg)  
(a)

(b)  
![](images/5322e51acb04d66477b415b35aef04ee1512648d80f6055a02644734a34936ea.jpg)  
(c)

![](images/068f16af9e6cffd76029e3f2d7c4b473032ebe8061d9a5a5c653c4027ac50923.jpg)

![](images/609b82692f7fb247b1826c2b69bf29390b88f487e650acb05315913310587d62.jpg)

![](images/575c59c1689e59109cd15345117ab8af17d1c4e5a04d57ecfd4e1fd54fbfb4e6.jpg)

(d)  
![](images/20269fb5db0777df2cc17fc4db3155a34043941fd36d157c08c7280196bb348e.jpg)

(e)  
(f)  
![](images/f3d8b016fdb0cb91924f63858ab33f1ef64eaaa650354f3684c1ab39153decfb.jpg)  
(g)

![](images/5ef8a01b9345f1b83132e4c028d074886937f1e7fcc241b325964814b1ceb964.jpg)

![](images/f250dd47afbeac041f9dd3e1ff52e03fb9ee86ffaf9d0cdb2b60d40151eddeb9.jpg)  
(j)

(h)  
(i)  
![](images/efeeae8d4725dc800a5a10fc433a1553ebea6e6e35aedec1e63711cbd4e7911c.jpg)  
(k)

![](images/1345bdb396d75d05beada8d757ff19a2e72f1555d7b12e13bc52c6f97dfd7fc7.jpg)  
(l)

![](images/aca26d4148ffab7259013cf1aaa3706a46f9a5f2c00b570b53a8567b230b49cd.jpg)  
(m)

![](images/34f4d654ed7fb34018f912c1e33431626245a932205a5ec81ca1a007e6fc62bc.jpg)  
(n)

![](images/8bd22fbe26f3f4106d91ba78530dd3f8b3bed07ee9f4ddede38f0b627ed78796.jpg)  
(o)  
Figure 2: SpaTIE learns a static morphology-aware representation space for uterine histopathology. a, UMAP visualization of patch-level embeddings extracted by SpaTIE, UPFM, and Baseline on TCGA-UCEC and TCGA-UCS. b, Quantitative comparison of static latent-space organization. c, Examples of representative cluster-center patches identified from the SpaTIE latent space (TCGA-UCEC). d-f, Performance comparison of downstream prediction tasks on TCGA-UCEC using F1- score, accuracy, and ROC-AUC. g-i, Performance comparison of downstream prediction tasks on TCGA-UCS using F1-score, accuracy, and ROC-AUC. j-l, Representative attention maps and highattention histopathological regions from SpaTIE-based downstream prediction models for disease type, pten, and tp53 respectively (TCGA-UCEC). m-o, Representative attention maps and high-attention histopathological regions from SpaTIE-based downstream prediction models for morphology, tp53, and origin respectively (TCGA-UCS).

We then evaluated whether SpaTIE-derived static features retained clinically useful information through downstream prediction tasks (see details in Supplementary Sec. 1 and Supplementary Sec. 2). On TCGA-UCEC, SpaTIE achieved competitive performance across multiple tasks, including histological grade prediction, subtype classification, molecular alteration prediction, and survival-related prediction, as measured by F1-score, accuracy, and ROC-AUC (Fig. 2d-f). SpaTIE showed strong performance for grade-related tasks and mutation prediction of key endometrial cancer genes such as PTEN, TP53, and ARID1A, whereas tasks such as FIGO stage prediction remained more challenging. On TCGA-UCS, SpaTIE also achieved robust performance in morphology-related and molecular prediction tasks, including morphology classification, origin prediction, and TP53 mutation prediction (Fig. 2g-i). These results indicate that SpaTIE-derived static morphological features encode clinicopathological and molecular information relevant to uterine malignancies.

Finally, we visualized attention maps from SpaTIE-based downstream prediction models to assess whether model decisions were supported by histopathologically meaningful regions. Representative heatmaps showed that high-attention regions were concentrated in tumor-rich or morphologically informative areas rather than background tissue or irrelevant regions (Fig. 2j–o). High-attention patches further corresponded to regions with diagnostic or prognostic morphology, supporting the interpretability of SpaTIE-derived static features. Together, these analyses show that SpaTIE constructs a static morphology-aware representation space that is structurally coherent, biologically interpretable, and clinically informative.

## 2.4 SpaTIE maps static morphology to latent progression-associated tumor states

Having established a uterus-adapted morphology space, we next asked whether static histopathological features could be organized into latent progression-associated tumor states. We applied trajectoryaware inference to SpaTIE-derived embeddings and obtained continuous state axes without temporal labels or molecular supervision. UMAP visualization showed coherent latent state structures in both TCGA-UCEC and TCGA-UCS (Fig. 3a). We further examined the spatial distribution of latent states on representative WSIs. SpaTIE generated spatial state maps that were more locally coherent and morphologically consistent than those produced by UPFM and baseline models (Fig. 3b). Regions with similar histological appearance tended to share similar state values, whereas morphologically distinct regions showed diferent state levels. In contrast, the baseline model produced less organized spatial maps and frequently failed to distinguish tumor-rich regions from non-tumor or stromal regions. These spatial results suggest that the inferred states are linked to tissue-level morphology, rather than being only artifacts of dimensionality reduction.

We evaluated the inferred states using stability (STAB), spatial consistency (SPA), and feature smoothness (FS) (see details in Sec. 10). SpaTIE achieved consistently strong performance across these metrics, with STAB values of 0.997 and 0.994, SPA values of 0.919 and 0.893, and FS values of 0.040 and 0.042 on TCGA-UCEC and TCGA-UCS, respectively (Fig. 3c). These results indicate that the SpaTIE-derived axes are stable and locally coherent, supporting the presence of progressionassociated organization within the learned morphology manifold.

To assess the clinicopathological relevance of SpaTIE-derived states, we analyzed slide-level state values against available clinical annotations. In TCGA-UCEC, inferred states were associated with FIGO stage (Kruskal–Wallis $H = 1 5 . 3 5 7 $ $p = 1 . 5 4 \times 1 0 ^ { - 3 }$ $\mathrm { F D R } = 0 . 0 0 7 7 )$ , fraction genome altered ， $( \mathrm { F G A } ; \rho = - 0 . 1 8 1 , p = 1 . 7 6 \times 1 0 ^ { - 5 }$ $\mathrm { F D R } = 5 . 2 9 \times 1 0 ^ { - 5 } )$ , and prior malignancy status $( p = 0 . 0 0 7 8$ , FDR $= 0 . 0 2 5 9 )$ . Extended GDC clinical annotations further showed associations with tumor grade $( H =$ $3 5 . 1 4 2 , p = 1 . 1 4 \times 1 0 ^ { - 7 } , \mathrm { F D R } = 5 . 6 8 \times 1 0 ^ { - 6 } )$ , percent tumor invasion $( \rho = - 0 . 2 2 5 , p = 3 . 9 8 \times 1 0 ^ { - 7 }$ , FDR $= 9 . 9 6 \times 1 0 ^ { - 6 } )$ , and disease response $( H = 1 2 . 8 2 9$ $p = 0 . 0 0 1 6$ , FDR = 0.0238). By contrast, disease type, primary diagnosis, and morphology code were not significant after multiple-testing correction in TCGA-UCEC. In TCGA-UCS, the inferred state axis showed a positive association with FGA $\left( \rho = 0 . 2 1 9 , p = 3 . 7 7 \times 1 0 ^ { - 2 } , \mathrm { F D R } = 0 . 1 1 3 ; \mathrm { F i g . ~ 3 d } _ { } \right)$ ), consistent with a link between morphology-derived states and genomic alteration burden. By contrast, FIGO stage and ICD-10 anatomical classification were not significant in TCGA-UCS (FIGO: $p = 0 . 4 2 4$ , FDR = 0.566; ICD-10: $p = 0 . 2 9 2$ , FDR = 0.467). These results indicate that the inferred states capture selected tumor-burden and morphologyassociated signals, rather than broadly tracking every available clinical annotation (see details in Table S8).

(a)  
![](images/18f3c520325eefec7ca6d039fba0221c0cc6dd4063d7e0ccb6e2bc7492caf990.jpg)

![](images/d52710970e2230bd7b49f07237c0b80feb44a285b665f72190fbd194c189b10a.jpg)

![](images/065bdb7fe0c076d455c7599e4cae0beba75cb072679f7004c4e73c4881f00067.jpg)

![](images/875b59c786565dc25f672e2969ff70cc65a1473def6ecfe878fe202f60118a4e.jpg)

(b)  
![](images/5f429c765bca68e077da114b7e4ad7c005ac67d935e3221471f176521f83b156.jpg)

![](images/b10c2a58cd7a6e023cdc742b77f79b0be7e8deb04ff820266a4d5efd9ef92a26.jpg)

![](images/6ccfc19dfc20e6bb8e31622a6ae93a90a5e4f718e13678c78ae96522638682de.jpg)  
UCS: Fraction Genome Altered ρ = 0.219, p=3.77e-2

(c)  
Comparison of inferred state organization using STAB, SPA, and FS metrics  
(d)  
![](images/6337aa8cbaf2353301954afd262d92dd0f9dc66b5c9a0cb5205c80052465361f.jpg)

![](images/e4abc231336ad99c407c0e5a262b1952a6a8f6c8c62d00e679443bf6b8e4a472.jpg)

(e)  
K-M curve for DFS prediction on UCEC (FIGO IV) Log-rank p = 1.24e-2  
![](images/777ec9aadf6bd7b86ec17c6401fe8d3aa56a66b887a51d13649ed59f6f307ffd.jpg)

(f)  
K-M curve for OS prediction on UCS (FIGO IV) Log-rank p = 8.87e-3  
![](images/6c82b1df868bc8995657b0703f876b6399e3b9087b009c9ef5bf9e8af2618635.jpg)

(g)  
K-M curve for OS prediction on UCS (High fraction genome altered samples)  
![](images/34d5902c87bb0f9fcb803e04ae2666e265c5727bb25fd0cf8931c0cdfe4237c7.jpg)

(h)  
![](images/4e59fbb93e49698136af5c3e2896d87faec7b8f6e77a0e5a076ed5b2e9218e34.jpg)  
K-M curve for OS prediction on UCS (Prior malignancy = False)  
Figure 3: SpaTIE maps static morphological representations to latent progression-associated tumor states. (a) UMAP visualization of latent state organization inferred from SpaTIE, UPFM, and baseline embeddings on TCGA-UCEC and TCGA-UCS. (b) Representative spatial tumor-state maps generated by SpaTIE, UPFM, and baseline models. (c) Quantitative comparison of inferred state organization using STAB, SPA, and FS metrics. (d) Association between SpaTIE-derived states and fraction genome altered in TCGA-UCS. (e,f) Exploratory survival stratification within FIGO stage IV patients in TCGA-UCEC and TCGA-UCS. (g,h) Exploratory subgroup survival analyses in TCGA-UCS patients with high fraction genome altered or without prior malignancy.

We next evaluated prognostic associations. In TCGA-UCEC, slide-level survival analysis showed that the state axis was associated with overall survival (log-rank $p = 0 . 0 0 3 8 ;$ Cox $\mathrm { H R } = 0 . 6 7 7 .$ , 95% $\mathrm { C I ~ 0 . 5 3 6 - 0 . 8 5 5 , ~ } p = 0 . 0 0 1 0 )$ . Disease-free survival was less consistently associated in the full UCEC cohort, but an FIGO stage IV analysis showed separation of disease-free survival by state group $( n = 1 8 $ , log-rank $p = 0 . 0 1 2 4 ; \mathrm { F i g . ~ 3 e } )$ . In TCGA-UCS, several subgroup analyses showed prognostic separation. Within FIGO stage IV samples, state groups difered in overall survival $( n = 1 6$ , log-rank $p = 8 . 8 7 \times 1 0 ^ { - 3 } ; \mathrm { F i g . ~ 3 f } )$ . Among UCS patients with high FGA, the state axis also stratified overall survival $( n = 4 7 ,$ , log-rank $p = 1 . 6 8 \times 1 0 ^ { - 2 }$ in the analysis shown in Fig. 3g), consistent with the observed state–FGA correlation. A further nominal association was observed among UCS patients without prior malignancy $( n = 7 9$ , log-rank $p = 3 . 2 8 \times 1 0 ^ { - 2 } ;$ ; Fig. 3h). These subgroup signals should be interpreted as exploratory, particularly in UCS where sample sizes are modest. Taken together, SpaTIE maps uterine histopathology into spatially coherent latent states that are linked to selected clinicopathological variables, genomic alteration burden, and prognostic patterns in UCEC and UCS, without implying a universal or deterministic timeline of clinical progression.

(c)  
![](images/6a0359837add116bf4b9a64ac92943ef985c4769fd8b8f597af00e1c484c165c.jpg)

## 2.5 Progression-associated states are linked to pathway activity and multiomics features

UCEC / methylation / Estrogen Response rho=-0.187, p=2.09e-04, FDR=2.09e-03  
![](images/9cf8978b528746672534e2ffc71dad8cd60dcb90f3c59520d8926f033f149255.jpg)  
(f)  
Figure 4: Association between SpaTIE-derived progression-associated states and pathway or molecular features. (a) Violin plots comparing pathway scores between low- and high-state groups for significant pathway-level associations. (b) Summary of significant pathway associations across cohorts and omics modalities. (c) Representative scatter plots showing correlations between inferred states and pathway scores. (d) State-ordered heatmap of UCS RNA-seq estrogen-response pathway members. (e) State-ordered heatmap of UCEC methylation features in the estrogen-response pathway. (f) Representative feature-level associations between inferred states and individual molecular features.

We next asked whether morphology-derived states could be molecularly contextualized. SpaTIEderived state values were integrated with pathway scores and feature-level multi-omics profiles across

TCGA-UCEC and TCGA-UCS. For pathway-level analysis, patients were divided into low- and highstate groups according to the inferred state values, and pathway-score distributions were compared across DNA methylation, somatic copy-number variation (SCNV), mutation, RNA-seq, and RPPA modalities. A limited set of pathway programs difered between state groups (Fig. 4a,b). In TCGA-UCEC, associations were observed for PRC2/Polycomb-related methylation $( p = 0 . 0 1 9 )$ , MMR-related SCNV $( p = 0 . 0 2 2 )$ , AKT–mTOR RPPA signaling $( p = 0 . 0 2 5 )$ , and cell-cycle-related RPPA signaling $( p = 0 . 0 3 1 )$ . In TCGA-UCS, RNA-seq-based estrogen-response pathway scores difered between state groups $( p = 0 . 0 0 4 8 )$ . These results suggest that SpaTIE-derived states are linked to selected molecular programs rather than isolated image-derived variation.

We further examined continuous associations between inferred states and pathway scores. Representative examples showed correlations with multiple pathway-level activities, including UCEC methylation based estrogen-response scores $( \rho = - 0 . 1 8 7 , \ p = 2 . 0 9 \times 1 0 ^ { - 4 } )$ , UCEC methylation-based WNT/β- catenin scores $( \rho ~ = ~ 0 . 1 3 4 , ~ p ~ = ~ 8 . 0 6 \times 1 0 ^ { - 3 } )$ , UCEC RPPA AKT–mTOR scores $( \rho ~ = ~ - 0 . 1 3 7$ $p = 7 . 7 4 \times 1 0 ^ { - 3 } )$ , UCEC SCNV MMR scores $( \rho = - 0 . 0 9 1 , p = 4 . 2 9 \times 1 0 ^ { - 2 } )$ , and UCS RNA-seq estrogen-response scores $( \rho = - 0 . 3 3 0 , p = 1 . 2 1 \times 1 0 ^ { - 2 } ; \mathrm { F i g . 4 c } )$ . State-ordered heatmaps further showed gradual pathway-member variation along the inferred axis, including estrogen-response-related RNA-seq features in UCS and estrogen-response-related methylation features in UCEC (Fig. 4d,e). These observations provide pathway-level molecular context for the morphology-derived state organization.

At the feature level, SpaTIE-derived states were associated with selected molecular features across omics modalities (Fig. 4f and Supplementary Sec. 5). In TCGA-UCEC, the clearest corrected support was concentrated in methylation, SCNV, and RPPA layers; representative features overlapping with the virtual perturbation analysis included PRR22, RBM27, and LRRC42 for methylation, FAM81A, RNA5SP396, and GCNT3 for SCNV, and AXL, GAPDH, ERRFI1, CTNNA1, and SERPINE1 for RPPA. RNA-seq and mutation features in UCEC were interpreted more cautiously because top-ranked candidates had limited FDR support in the current analysis.

TCGA-UCS showed stronger nominal efect sizes for several individual features but weaker FDR support, consistent with the smaller cohort size. Representative UCS RNA-seq associations included UFD1L, FGF1, CCNG1, IL33, and FLNB, while UCS methylation, mutation, SCNV, and RPPA analyses highlighted additional candidate features with state-associated trends. Collectively, these pathwayand feature-level analyses indicate that morphology-derived states are accompanied by molecular variation across epigenetic, genomic, transcriptomic, and proteomic layers. We interpret these associations as molecular context for the inferred state axis, not as definitive evidence of single-feature molecular drivers.

## 2.6 Progression-guided virtual perturbation prioritizes candidate state-associated programs

To further evaluate whether molecular features were coupled to SpaTIE-derived state organization, we performed progression-guided virtual perturbation analysis across five omics modalities. For each molecular feature, we estimated a virtual perturbation score by measuring how strongly the association between the feature and state value was attenuated after removing the feature-associated component. This analysis was designed as a computational feature-prioritization strategy rather than experimental gene knockout or causal validation, following the emerging paradigm of AI-based virtual cells research, which emphasize in silico perturbation for biological hypothesis generation rather than direct causal inference [33].

In TCGA-UCEC, virtual perturbation highlighted candidate features across epigenetic, genomic, transcriptomic, and proteomic layers (Fig. 5a-e). The top methylation-associated perturbation feature was PRR22 (KO score = 0.312), whose association with state values was markedly reduced after virtual perturbation $( \rho$ from 0.338 to 0.026). Mutation-level analysis prioritized GPR84 (KO score = 2.307), while RNA-seq analysis identified RAB27B as the strongest transcriptomic candidate (KO score = $0 . 3 0 2 ; \rho$ from -0.302 to 0.0004). At the protein level, AXL showed the largest RPPA perturbation efect (KO score = 0.186; ρ from 0.189 to 0.003), suggesting a potential link between morphologyderived state organization and receptor tyrosine kinase or EMT-associated signaling. SCNV analysis prioritized FAM81A (KO score = 0.138; ρ from 0.144 to 0.006). Across top-ranked UCEC features, FDR-supported state associations were most evident in methylation, SCNV, and RPPA modalities, supporting these layers as relatively robust molecular anchors for the inferred state axis.

(a)  
UCEC / methylation: top virtual-KO site PRR22 KO score=0.3118. base=0.3378. after=0.0260  
![](images/d9f3bb6f4cad05b897c37a01de4ee2644f2d54a5095f996bf4abbafedb81bdb3.jpg)

![](images/39cb4b7879baba0b6e2d83c5bd33d532792dc22774c124782893e9f7a26e8617.jpg)  
(b)  
UCEC / rnaseg: top virtual-KO site RAB27B KO score=0.3019. base=0.3023. after=0.0004

![](images/9100f44ee774b15099efcbdeaa8156566693841e0194c13191570d84a331e11a.jpg)  
UCEC / mutation: top virtual-KO site GPR84 KO score=2.3068. base=2.3068, after=0.0000

(c)  
![](images/6e6ad10ddcbb95b9bd9870bd6dec855a84886f18113205d610bae197b118d537.jpg)  
UCEC / rppa: top virtual-KO site AXL KO score=0.1862. base=0.1895. after=0.0032

![](images/c42c825376eddf32a62a1b1c19941bfe6e49718cdac6fc80225097eead1be9e5.jpg)

![](images/6bba52fc546f3892625eb1d6cec951655724d0e2409fd1851ab60e902fdb6e72.jpg)

(d)  
![](images/a6abde1dd4e11311581b05c8dc3c4b408ec3c232f48e75423d8f91cf2014980c.jpg)

![](images/f857905e4e8d3bff30df8a9ed7f9d0a72bcd150859c7b0fc619cd48befd4b1b0.jpg)

(e)  
UCEC / scnv: top virtual-KO site FAM81A KO score=0.1380, base=0.1442, after=0.0062  
![](images/cfe4c2107ff033af51998bdaff76c96d60cdac4fe7a8a86a3abc3ed945db43ed.jpg)

![](images/6c9fe8478904af4ca090422c38f333c2d9da50a084744b0a7619b30d91cfb316.jpg)

(f)  
![](images/12684c16d664410b05c8d4f9d92a8600208ca1d5e3bdc445f9b830d438a124ca.jpg)  
UCS / methylation: top virtual-KO site PIK3C2G KO score=0.4283, base=0.4655, after=0.0372

(g)  
![](images/20e277f05ffa33fdffd7ab0140578680aeb745cb2d5cf5b4f46c7b07ae9040fa.jpg)  
UCS / mutation: top virtual-KO site TPTE KO score=1.5455. base=1.5455. after=0.0000

![](images/941e1cbb0bec2d3936c6d6f058f8e2cf08ff776644e54d2ef979cda4a070af48.jpg)  
(h)  
UCS / rppa: top virtual-KO site ARH KO score=0.3846. base=0.4088. after=0.0242

![](images/e69a90012400771daeff2bf8e76c49df6d8c93c539281b913108c0088538ef86.jpg)

![](images/84faa730fd8737cef15c23fd31986f2de1e2df75b17990fa1c88a991c626db56.jpg)

![](images/d0ddc07fc09dec2bd2e5cdd08ef793037364a96c5deaf10886ac6a457a7a51a2.jpg)

![](images/ddfc78150995ad8055d95012fd3c8ce6ce967f16e2a9b412095deabb5e8cfc65.jpg)

(i)  
![](images/391d8cddc8893b06271c4176569d7fd9e62fd5bbe0f218dff7faec83952ffe80.jpg)  
UCS / rnaseg: top virtual-KO site UFD1L KO score=0.4428. base=0.4922. after=0.0493

![](images/21cf926cc441d1e401e8583e0159cf6a1e937b3c62e60707c26ebdea967ce7da.jpg)

(k)  
UCS / scnv: top virtual-KO site 16q23.1 KO score=0.2744, base=0.2971, after=0.0227  
![](images/f03b1cf62b2d9c6e7c64757d7207472983f28f05ee93fc5f5009bc463e574a83.jpg)

![](images/d6679a1265351b832381b1d96f873b15662b81a70b5fc4a435bbc12cb2c9f6c6.jpg)

(l)  
![](images/d95bcfcedd392fa2db2c0b5542236f39b71d06adfa7f804fb3cdd116a5792229.jpg)  
Figure 5: Progression-guided virtual perturbation analysis. (a-e) Representative virtual perturbation results in TCGA-UCEC across methylation, mutation, RNA-seq, RPPA, and SCNV modalities. (f) Cross-omics schematic of UCEC candidate perturbation relationships. (g-k) Representative virtual perturbation results in TCGA-UCS across methylation, mutation, RNA-seq, RPPA, and SCNV modalities. (l) Cross-omics schematic of UCS candidate perturbation relationships.

In TCGA-UCS, virtual perturbation identified a distinct set of candidate features (Fig. 5g-k). The strongest methylation-associated candidate was PIK3C2G (KO score = 0.428), whose state association was reduced from $\rho = - 0 . 4 6 5$ to $\rho = 0 . 0 3 7$ after perturbation. Mutation-level analysis prioritized TPTE (KO score = 1.546), RNA-seq analysis prioritized UFD1L (KO score = 0.443; ρ from 0.492 to 0.049), RPPA analysis prioritized ARHI (KO score = 0.385; ρ from 0.409 to 0.024), and SCNV analysis prioritized chromosome region 16q23.1 (KO score = 0.274; ρ from -0.297 to 0.023). Although UCS perturbation scores were often large, individual feature-level FDR support was limited, likely reflecting the smaller sample size of the TCGA-UCS cohort. These UCS candidates were therefore interpreted as exploratory state-associated molecular hypotheses.

Cross-omics integration further suggested that progression-associated state organization reflects distributed molecular coupling rather than a single linear driver (Fig. 5f,l). In UCEC, top perturbation candidates did not form a strong same-gene DNA–RNA–protein chain, but instead suggested convergence from methylation and SCNV alterations toward transcriptomic and proteomic programs involving adhesion, signaling, metabolism, and extracellular remodeling. In UCS, several cross-layer relationships connected methylation or mutation candidates to RNA-level features, including same-gene methylation–RNA links such as MGC87042 and ROR2, as well as additional candidate links involving PRRX2, C11orf54, and OMA1. These patterns are compatible with the pronounced genomic and morphological heterogeneity of uterine carcinosarcoma.

Together, progression-guided virtual perturbation supports the utility of SpaTIE-derived states for molecular hypothesis generation (Supplementary Sec. 6). The top-ranked candidates point to possible biological themes including epigenetic regulation, copy-number-associated structural variation, receptor tyrosine kinase signaling, cell adhesion, extracellular-matrix remodeling, metabolic adaptation, and stress-response programs. Because the perturbation procedure is computational and based on bulk-level associations, these results should not be interpreted as direct causal evidence. Rather, they prioritize candidate molecular programs that may help explain how morphology-derived state organization is coupled to multi-omics tumor-state variation.

## 3 Discussion

In this study, we developed SpaTIE, a uterus-specific computational pathology framework that connects morphology-aware representation learning with progression-associated tumor-state inference. Rather than using histopathology only as an input for static diagnostic or molecular prediction, SpaTIE organizes spatially heterogeneous H&E morphology into latent state axes and then relates these axes to clinicopathological variables, survival outcomes, multi-omics profiles and computational perturbation analyses. Across TCGA-UCEC and TCGA-UCS, SpaTIE produced coherent morphology manifolds, supported a range of downstream prediction tasks, localized attention to histopathologically informative regions and revealed tumor-state organization with selected clinical and molecular associations. These findings suggest that uterine histopathology contains recoverable information about tumor-state heterogeneity that extends beyond conventional categorical labels.

A first contribution of SpaTIE is the construction of a uterus-adapted pathology representation. Recent pathology foundation models have demonstrated broad transferability across organs and tasks [13, 14, 19, 20]. However, uterine malignancies have distinctive morphological and biological spectra, ranging from low-grade endometrioid carcinoma to high-grade carcinoma and carcinosarcoma with epithelial–mesenchymal features [3, 4, 34]. By adapting representation learning to a large uterine histopathology resource, SpaTIE captured organ-relevant morphology and maintained competitive performance in diagnostic, molecular and survival-related prediction tasks. Importantly, the value of this adaptation was not limited to conventional task metrics; it also provided a more coherent substrate for downstream state inference.

The second contribution is conceptual. Most computational pathology pipelines treat each WSI as a static object to be assigned a diagnostic, prognostic or molecular label. SpaTIE instead treats tissue morphology as a spatial mixture of coexisting tumor states. This perspective is inspired by trajectory inference in single-cell biology, where asynchronously sampled cells are ordered to recover latent biological processes [7, 8, 23]. In histopathology, the inferred axes should not be interpreted as direct chronological timelines. Rather, they represent morphology-derived, progression-associated tumor-state organization embedded in cross-sectional tissue images. This distinction is important: the framework seeks to recover structured disease-state variation from spatial morphology, not to prove

the temporal history of an individual tumor.

The clinicopathological analyses support this restrained interpretation. SpaTIE-derived states were associated with selected variables related to tumor burden and disease state, including FIGO stage, FGA, tumor grade, tumor invasion, residual disease and disease response. The UCS association between the inferred state axis and FGA is particularly relevant because it links morphology-derived organization to genomic alteration burden. UCS and high-risk subgroup findings suggested potentia prognostic relevance in settings such as FIGO stage IV disease and high-FGA tumors. At the same time, the associations were not universal across all clinical variables or endpoints. SpaTIE-derived states should therefore be viewed as partially aligned with clinical progression and outcome, rather than as substitutes for established staging, grading or prognostic models.

A third contribution is the integration of morphology-derived tumor states with multi-omics data. Across DNA methylation, SCNV, mutation, RNA-seq and RPPA profiles, the inferred states were linked to selected molecular programs, including chromatin regulation, copy-number-associated structural variation, receptor tyrosine kinase signaling, cell adhesion, extracellular-matrix remodeling and metabolic adaptation. These analyses are best interpreted as molecular context for the image-derived state axes. The strongest feature-level support was observed in UCEC methylation, SCNV and RPPA analyses, whereas transcriptomic and mutation-level findings, particularly in UCS, were more exploratory. Nevertheless, the convergence of signals across multiple molecular layers suggests that morphology-derived state organization may reflect systems-level tumor biology rather than isolated visual variation.

Progression-guided virtual perturbation further extends this framework toward hypothesis generation. By estimating how strongly feature–state associations were attenuated after computationa perturbation, SpaTIE prioritized candidate molecular features coupled to the inferred morphologyderived state axes. This analysis is not equivalent to experimental gene knockout and should not be interpreted as causal validation. Its value is instead practical and exploratory: it ranks candidate features and cross-omics relationships that may help explain why particular molecular programs co-vary with morphology-derived states. In this sense, virtual perturbation provides a bridge from descriptive association analysis to experimentally testable hypotheses.

Several limitations should be acknowledged. First, SpaTIE infers state organization from crosssectional H&E cohorts rather than longitudinally sampled tumors. The inferred axes therefore cannot establish chronological tumor evolution, clonal ancestry or branching progression routes. Future studies incorporating longitudinal sampling, multi-region sequencing and spatial molecular profiling will be needed to determine how these morphology-derived states relate to true evolutionary trajectories [6, 11]. Second, although SpaTIE was trained on a large uterine histopathology resource and evaluated in TCGA cohorts, broader external validation is needed across institutions, scanners, staining protocols and patient populations. This is especially important for TCGA-UCS, where sample size is modest and several survival or multi-omics results should be considered exploratory. Third, the molecular analyses are based largely on bulk omics profiles. Bulk methylation, copy-number, transcriptomic and proteomic measurements can provide useful biological context, but they cannot precisely assign molecular programs to the same spatial regions that drive the histopathological state maps. Spatial transcriptomics, multiplex imaging and spatial proteomics could refine the interpretation of these states and help identify the cellular compartments contributing to each morphology-derived axis [10, 35, 36]. Fourth, the virtual perturbation analysis remains computational. It prioritizes candidate genes, proteins and genomic features, but it does not prove that these features drive the histological states. Wet-lab validation, such as CRISPR perturbation, organoid or cell-line models, and targeted functional assays, will be required to test whether the nominated candidates mechanistically regulate morphology-associated tumor-state programs.

In summary, SpaTIE provides a uterus-specific framework for connecting predictive pathology with tumor-state discovery. By combining organ-adapted pathology foundation modeling, spatially coherent state inference, multi-omics association analysis and computational perturbation, SpaTIE suggests a route for studying progression-associated tumor heterogeneity from routine histopathology while maintaining a clear boundary between image-derived state organization, molecular association and causal mechanism.

## 4 Methods

## 4.1 Ethics approval

This study was approved by the Medical Ethics Committee of Peking University Third Hospital (Approval No.: IRB00006761-M20250620). Publicly available TCGA-UCEC and TCGA-UCS were obtained from https://www.cancer.gov/ccg/research/genome-sequencing/tcga.

## 4.2 Study cohorts and data sources

## 4.2.1 PK3-Uter

To construct a uterus-specific pathology foundation model, we established a large-scale uterine histopathology dataset termed PK3-Uter from Peking University Third Hospital. The cohort included archival uterine histopathology WSIs. The dataset covered multiple uterine pathological entities and morphological patterns, including endometrioid carcinoma, serous carcinoma, clear cell carcinoma, carcinosarcoma, hyperplasia-related lesions, and other uterine pathological subtypes.

All slides were scanned using Shenzhen Shengqiang SQS-600P pathology scanner at 80× magnification (0.105 µm/pixel), stored in SDPC format. To facilitate large-scale self-supervised pretraining, tissue regions were first identified using foreground extraction and tissue masking. Tumor-rich regions were subsequently screened and extracted for downstream processing.

For self-supervised pretraining, WSIs were divided into non-overlapping image patches at 20× magnification. Each patch had a spatial size of 224 × 224 pixels. In total, the PK3-Uter contained 10,426 WSIs and approximately 62,288,077 image patches after quality control and tissue-region filtering.

## 4.2.2 TCGA-UCEC and TCGA-UCS

Two independent public uterine malignancy cohorts from The Cancer Genome Atlas were used for external evaluation and multi-omics analysis: uterine corpus endometrial carcinoma (TCGA-UCEC) and uterine carcinosarcoma (TCGA-UCS) [3, 4].

For TCGA-UCEC, we collected H&E whole-slide images together with clinicopathological annotations, including histological subtype, tumor grade, FIGO stage, ICD anatomical site code, mutation profiles, survival information, and molecular data. Similarly, TCGA-UCS included WSIs and corresponding clinical and molecular annotations for uterine carcinosarcoma samples. WSIs with severe artifacts, incomplete tissue regions, or insuficient tumor content were excluded from downstream analysis. After quality control, the final TCGA-UCEC cohort included 566 WSIs, whereas TCGA-UCS included 91 WSIs. For downstream prediction tasks, patient-level train-validation-test splitting was performed to avoid data leakage between slides originating from the same patient.

## 4.2.3 Multi-omics data acquisition and preprocessing

To investigate the molecular correlates of SpaTIE-derived progression-associated tumor states, we collected multiple molecular modalities from TCGA-UCEC and TCGA-UCS through the Genomic Data Commons portal.

The following molecular datasets were included:

• DNA methylation profiles;

• Somatic copy-number variation (SCNV) data;

• Somatic mutation profiles;

• RNA sequencing (RNA-seq) expression data;

• Reverse-phase protein array (RPPA) proteomic data.

DNA methylation data were processed as CpG-site-level beta values. SCNV data were represented using segmented copy-number regions. Mutation data were binarized at the gene level according to mutation occurrence. RNA-seq data were analyzed using normalized expression values from TCGA molecular data resources. RPPA data were processed using the standardized TCGA RPPA pipeline.

Samples lacking corresponding histopathology images or inferred tumor-state estimates were excluded from downstream molecular analyses. Multi-omics data were aligned with SpaTIE-derived states using TCGA patient identifiers. For statistical analyses, features with excessive missing values or extremely low variance were removed prior to downstream analysis. Detailed preprocessing procedures for each molecular modality are provided in Supplementary Methods.

## 4.3 Whole-slide image preprocessing and tumor-focused patch extraction

All WSIs were processed using OpenSdpc (https://github.com/WonderLandxD/opensdpc) or OpenSlide (https://openslide.org/) under a unified preprocessing pipeline. To reduce the influence of background regions and scanning artifacts, low-resolution tissue masking was first performed to identify valid tissue foreground regions. WSIs with severe blur, tissue folding, pen marks, excessive blank regions, or substantial scanning artifacts were excluded during quality control. Because SpaTIE focuses on morphology within uterine malignancies, tumor-rich regions were preferentially retained during downstream processing.

For patch extraction, WSIs were divided into non-overlapping image patches at 20× magnification. Each patch had a spatial resolution of 224 × 224 pixels. We used a tumor-focused patch extraction procedure (Supplementary Sec. 3) to build a dataset consisting of 24,318,109 patches for self-supervised pretraining. During pretraining, image augmentations included random horizontal and vertical flipping, random cropping, Gaussian blur, color jittering, and stain-related perturbation, following DINOv2-style self-supervised learning settings adapted for uterine histopathology [30]. These preprocessing procedures enabled construction of a large-scale uterine histopathology corpus suitable for morphology-aware representation learning.

## 4.4 Development of SpaTIE

SpaTIE consists of two major stages: uterus-specific self-supervised representation learning and latent progression-associated tumor-state inference. In the first stage, a general-purpose pathology foundation model was adapted to uterine histopathology using parameter-eficient self-supervised learning to construct a morphology-aware latent representation space (Supplementary Sec. 7). In the second stage, trajectory-aware manifold inference was applied to organize static morphological representations into continuous state axes, thereby identifying morphology-derived tumor-state continua from cross-sectional histopathology (Supplementary Sec. 8).

For uterus-specific representation learning, we adapted the universal pathology foundation model UNI [13] to the uterine pathology domain through continued self-supervised learning. UNI is based on a Vision Transformer Large architecture with 16×16 patch size (ViT-L/16), comprising approximately 303 million parameters pretrained on over 100 million pathology image patches using the DINOv2 framework [30, 37]. Although general pathology foundation models can capture transferable histopathological representations across tissues, many morphologic patterns in uterine malignancies are organ-specific, including glandular architecture, stromal remodeling, biphasic epithelial–mesenchyma structures, and heterogeneous tumor organization. Therefore, uterus-specific domain adaptation was performed to improve representation sensitivity to uterine tumor morphology.

To preserve general histopathological knowledge while enabling eficient uterine adaptation, we employed Low-Rank Adaptation (LoRA) [31]. Specifically, LoRA modules were inserted into the query and value projection layers of transformer attention blocks. For a pretrained weight matrix $W _ { 0 } \in \mathbb { R } ^ { m \times n }$ , the trainable update was parameterized as:

$$
W = W _ { 0 } + \Delta W = W _ { 0 } + B A ,\tag{4}
$$

where $B \in \mathbb { R } ^ { m \times r }$ and $A \in \mathbb { R } ^ { r \times n }$ are trainable low-rank matrices with rank $r \ll \operatorname* { m i n } ( m , n )$ . During training, pretrained parameters remained frozen while only LoRA parameters were optimized using the DINOv2 self-supervised objective. The model was trained for 50 epochs using the AdamW optimizer [38] with learning rate 1e − 4, and weight decay 0.998. Model selection was based on F1-score.

After self-supervised pretraining, SpaTIE was used as a frozen feature extractor for downstream analyses. For each image patch, the CLS token embedding from the final transformer layer was extracted as the patch-level feature representation:

$$
z _ { i } \in \mathbb { R } ^ { d } ,\tag{5}
$$

where d = 1024 denotes the embedding dimension. To obtain slide-level representations, patch-level embeddings from the same WSI were aggregated using [18]. Let $\{ z _ { 1 } , z _ { 2 } , \dots , z _ { N } \}$ denote patch-level embeddings from a WSI. Attention weights were computed as:

$$
a _ { i } = \frac { \exp { \left( w ^ { \top } \operatorname { t a n h } ( V z _ { i } ^ { \top } ) \right) } } { \sum _ { j = 1 } ^ { N } \exp { \left( w ^ { \top } \operatorname { t a n h } ( V z _ { j } ^ { \top } ) \right) } } ,\tag{6}
$$

where V and w are trainable parameters. The slide-level representation was then obtained as:

$$
Z = \sum _ { i = 1 } ^ { N } a _ { i } z _ { i } .\tag{7}
$$

To evaluate the efectiveness of uterus-specific representation learning, SpaTIE was compared with multiple baseline feature extractors, including the original universal pathology foundation model (UPFM, [32]), ImageNet-pretrained convolutional neural networks, ResNet-50-based pathology encoders, and other baseline histopathological representations. All baseline models were evaluated under identical preprocessing and downstream analysis settings whenever possible.

## 4.5 Latent progression-associated tumor-state inference

To infer latent tumor-state organization from histopathological morphology, we constructed lowdimensional manifolds from SpaTIE-derived slide-level embeddings. For each cohort, slide-level features were standardized and projected into a lower-dimensional latent space using principal component analysis (PCA), with the number of retained principal components selected to preserve at least 90% of total variance. Neighborhood graphs were subsequently constructed using the Scanpy framework [39]. Specifically, a k-nearest-neighbor (KNN) graph was computed based on Euclidean distances between slide-level embeddings:

$$
G = ( V , E ) ,\tag{8}
$$

where each node corresponds to a WSI and edges connect neighboring samples in the latent embedding space. Afinity weights between neighboring samples were calculated using Gaussian similarity kernels:

$$
K _ { i j } = \exp \left( - \frac { \| Z _ { i } - Z _ { j } \| ^ { 2 } } { \sigma ^ { 2 } } \right) ,\tag{9}
$$

where $Z _ { i }$ and $Z _ { j }$ denote slide-level embeddings and σ controls local neighborhood scaling. For visualization and manifold exploration, Uniform Manifold Approximation and Projection (UMAP) [40] and Leiden clustering [41] were applied to identify local morphology-associated communities within the embedding space.

Progression-associated tumor states were inferred using difusion pseudotime (DPT) [29], which estimates state relationships from difusion geometry on the manifold graph. In this study, DPT was used as a graph-ordering method for cross-sectional morphology rather than as evidence of chronological tumor evolution. Difusion operators were computed from the neighborhood afinity matrix:

$$
P = D ^ { - 1 } K ,\tag{10}
$$

where D is the diagonal degree matrix and K is the afinity matrix. Difusion distances were then used to estimate state relationships along the latent manifold. A root sample was selected according to lowstate morphology and manifold geometry. Difusion values were subsequently computed as geodesic distances from the root sample:

$$
t _ { i } = \mathrm { D P T } ( Z _ { i } ) ,\tag{11}
$$

where $t _ { i }$ denotes the latent state of sample i. Importantly, these inferred progression-associated states should not be interpreted as direct chronological timelines of tumor evolution. Instead, they represent morphology-derived latent state organization reconstructed from cross-sectional spatial heterogeneity.

To characterize spatial state organization within WSIs, both slide-level and patch-level progressionassociated states were estimated. Patch-level embeddings were projected into the same latent manifold space, and state values were assigned using neighborhood interpolation:

$$
t _ { p } = f ( z _ { p } ) ,\tag{12}
$$

where $z _ { p }$ denotes a patch-level embedding and $t _ { p }$ denotes its inferred state. Patch-level states were subsequently mapped back to original WSI coordinates to generate spatial state heatmaps. State quality was quantitatively evaluated using multiple unsupervised metrics describing latent manifold stability and spatial coherence. State stability (STAB) quantified robustness under repeated manifold reconstruction:

$$
\mathrm { S T A B } = \operatorname { c o r r } ( t ^ { ( 1 ) } , t ^ { ( 2 ) } ) ,\tag{13}
$$

where $t ^ { ( 1 ) }$ and $t ^ { ( 2 ) }$ denote states obtained from repeated analyses. Spatial coherence (SPA) measured local consistency of neighboring tissue regions:

$$
\mathrm { S P A } = \frac { 1 } { N } \sum _ { i } \frac { 1 } { | \mathcal { N } ( i ) | } \sum _ { j \in \mathcal { N } ( i ) } \exp ( - | t _ { i } - t _ { j } | ) ,\tag{14}
$$

where $\mathcal { N } ( i )$ denotes neighboring tissue patches. Feature smoothness (FS) quantified continuity of latent feature variation along state organization:

$$
\mathrm { F S } = 1 - \frac { 1 } { N } \sum _ { i } \| Z _ { i } - \bar { Z } _ { \mathcal { N } ( i ) } \| .\tag{15}
$$

## 4.6 Static representation evaluation and downstream tasks

To evaluate whether SpaTIE-derived static morphological representations retained clinically relevant information, multiple downstream prediction tasks were performed on TCGA-UCEC and TCGA-UCS cohorts. These tasks included histological subtype classification, tumor grade prediction, FIGO stage prediction, mutation prediction, molecular subtype prediction, and survival prediction. For each task, frozen SpaTIE-derived patch-level features were used as input to downstream MIL-based classifiers. Patient-level train-validation-test splitting was performed to avoid data leakage between slides originating from the same patient. The same downstream analysis pipeline was applied to all baseline feature extractors for fair comparison.

## 4.7 Progression states evaluation metrics

To quantitatively evaluate the quality of SpaTIE-derived static morphology representations and latent state organization, we introduced multiple unsupervised metrics describing latent manifold structure, information distribution, state robustness, spatial coherence, and feature continuity.

Manifold Structure Quality (MSQ). Manifold Structure Quality (MSQ) was designed to evaluate the intrinsic structural organization of the latent morphology manifold learned by SpaTIE. Specifically, principal component analysis (PCA) was first applied to the latent feature matrix:

$$
X = \{ z _ { 1 } , z _ { 2 } , \ldots , z _ { N } \} , \quad z _ { i } \in \mathbb { R } ^ { d } ,\tag{16}
$$

where $z _ { i }$ denotes the latent feature representation of the i-th histopathological region or WSI, N denotes the number of samples, and d denotes the feature dimension.

Let

$$
\lambda _ { 1 } , \lambda _ { 2 } , \ldots , \lambda _ { m }\tag{17}
$$

denote the explained variance ratios of PCA components, where m is the number of retained principal components. The intrinsic manifold dimensionality was estimated as:

$$
\mathrm { M S Q } = \operatorname* { m i n } \left\{ k : \sum _ { i = 1 } ^ { k } \lambda _ { i } \geq 0 . 9 \right\} .\tag{18}
$$

MSQ reflects the number of principal components required to explain 90% of the variance in the latent space. Lower MSQ values indicate that the latent manifold is more compact and structurally organized, suggesting that morphology-associated variation is encoded within a lower-dimensional and more coherent representation space.

Representation Entropy (RE). Representation Entropy (RE) was used to evaluate the information distribution and diversity of latent morphology representations. After PCA decomposition, normalized variance coeficients were computed as:

$$
p _ { i } = \frac { \lambda _ { i } } { \sum _ { j = 1 } ^ { m } \lambda _ { j } } ,\tag{19}
$$

where $\lambda _ { i }$ denotes the variance explained by the i-th principal component.

The entropy of the representation space was then calculated as:

$$
\mathrm { R E } = - \sum _ { i = 1 } ^ { m } p _ { i } \log ( p _ { i } + \epsilon ) ,\tag{20}
$$

where ϵ is a small constant for numerical stability.

RE measures how evenly information is distributed across latent dimensions. Higher RE values indicate richer and more diverse latent representations, whereas lower values suggest information collapse or overly concentrated latent organization.

Stability (STAB). Stability (STAB) was designed to evaluate the robustness of SpaTIE-derived state values under perturbation. Given an inferred state vector:

$$
t = \{ t _ { 1 } , t _ { 2 } , \ldots , t _ { N } \} ,\tag{21}
$$

where $t _ { i }$ denotes the latent state of the i-th patch or WSI, Gaussian noise was added to generate a perturbed state vector:

$$
\tilde { t } _ { i } = t _ { i } + \eta _ { i } , \quad \eta _ { i } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } ) .\tag{22}
$$

The robustness of state ordering was quantified using Spearman correlation:

$$
\mathrm { S T A B } = \rho _ { \mathrm { S p e a r m a n } } ( t , \tilde { t } ) ,\tag{23}
$$

where $\rho _ { \mathrm { S p e a r m a n } }$ denotes Spearman rank correlation.

Higher STAB values indicate that the inferred state organization is robust to perturbation and preserves stable relative ordering among samples.

Spatial Consistency (SPA). Spatial Consistency (SPA) was used to evaluate whether neighboring tissue regions within WSIs exhibit locally coherent state values. For each patch $i ,$ spatial neighbors were identified using a k-nearest-neighbor graph constructed from patch coordinates:

$$
\mathcal { N } _ { s } ( i ) = \{ j _ { 1 } , j _ { 2 } , . . . , j _ { k } \} ,\tag{24}
$$

where $\mathcal { N } _ { s } ( i )$ denotes the spatial neighborhood of patch i.

Spatial consistency was computed as:

$$
\mathrm { S P A } = \frac { 1 } { 1 + \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { 1 } { | N _ { s } ( i ) | } \sum _ { j \in { \mathcal N } _ { s } ( i ) } | t _ { i } - t _ { j } | } .\tag{25}
$$

SPA evaluates whether adjacent tissue regions exhibit similar state values. Higher SPA values indicate stronger local spatial coherence of state organization within tissue sections.

Feature Smoothness (FS). Feature Smoothness (FS) was designed to evaluate the continuity of latent morphology features along the inferred state axis. Let

$$
\{ z _ { ( 1 ) } , z _ { ( 2 ) } , \ldots , z _ { ( N ) } \}\tag{26}
$$

denote latent features sorted according to inferred state order:

$$
t _ { ( 1 ) } \leq t _ { ( 2 ) } \leq \cdot \cdot \cdot \leq t _ { ( N ) } .\tag{27}
$$

Feature variation between neighboring states was computed as:

$$
\Delta _ { i } = \lVert z _ { ( i + 1 ) } - z _ { ( i ) } \rVert _ { 2 } .\tag{28}
$$

Feature smoothness was then defined as:

$$
\mathrm { F S } = \frac { 1 } { 1 + \frac { 1 } { N - 1 } \sum _ { i = 1 } ^ { N - 1 } \Delta _ { i } } .\tag{29}
$$

FS measures whether latent morphology changes continuously along the inferred state organization. Higher FS values indicate smoother transitions of morphology-aware representations across inferred states, supporting the plausibility of continuous morphology-derived tumor-state organization.

## 4.8 Clinicopathological association and survival analysis

To investigate the clinical relevance of SpaTIE-derived progression-associated tumor states, we analyzed their associations with multiple clinicopathological variables. Group-wise state diferences were evaluated using Kruskal–Wallis tests followed by post hoc comparisons where appropriate. Correlations between state values and continuous clinicopathological variables were assessed using Spearman correlation analysis. For survival analysis, patients were stratified into state groups. Kaplan–Meier survival curves were generated and compared using log-rank tests. Cox proportional hazards regression was additionally performed to evaluate whether state values were associated with overall surviva independently of conventional clinicopathological variables:

$$
h ( t ) = h _ { 0 } ( t ) \exp ( \beta _ { 1 } x _ { 1 } + \cdot \cdot \cdot + \beta _ { n } x _ { n } ) .\tag{30}
$$

Hazard ratios (HRs) and 95% confidence intervals were reported.

## 4.9 Multi-omics association analysis

To investigate the molecular correlates of SpaTIE-derived progression-associated tumor states, we integrated state values with multiple molecular modalities, including DNA methylation, somatic copynumber variation (SCNV), mutation, RNA sequencing, and RPPA proteomic data from TCGA-UCEC and TCGA-UCS. Continuous molecular features were analyzed using Spearman correlation:

$$
\rho = { \mathrm { c o r r } } _ { \mathrm { S p e a r m a n } } ( x , t ) ,\tag{31}
$$

where x denotes the molecular feature and t denotes the latent state. For categorical mutation features, state distributions between mutated and non-mutated groups were compared using Mann–Whitney U tests.

To visualize state-associated molecular variation, samples were ordered according to state values and displayed as state-ordered heatmaps. Top-ranked molecular features were identified and were subsequently used for downstream enrichment analysis. Pathway enrichment analysis was performed using Enrichr [42] and KEGG pathway databases [43]. These analyses were designed to identify coordinated molecular programs associated with morphology-derived state organization.

## 4.10 Progression-guided virtual perturbation and pathway enrichment

To identify candidate molecular features potentially associated with state organization, we performed state-guided virtual perturbation analysis across methylation, SCNV, RNA-seq, and RPPA modalities. The concept is related to recent AI Virtual Cell frameworks, which advocate computational perturbation for biological hypothesis generation before experimental validation [33, 44]. For each candidate feature, a virtual perturbation score (KO score) was computed by evaluating the attenuation of feature–state association after computational removal of the feature-associated component:

$$
\mathrm { K O } ( g ) = | \rho ( t , g ) | - | \rho ( t _ { \mathrm { p e r t u r b e d } } , g ) | ,\tag{32}
$$

where $g$ denotes the molecular feature, t denotes the original state value, $t _ { \mathrm { p e r t u r b e d } }$ denotes the state value after virtual perturbation, and $\rho$ denotes the Spearman association. Features with larger KO scores were interpreted as having stronger coupling to latent state organization. Importantly, this analysis was designed as a computational hypothesis-generation strategy rather than direct causal inference or experimental knockout validation.

Top-ranked candidate genes and molecular features were further grouped into regulatory modules and subjected to pathway enrichment analysis using Enrichr and KEGG databases. These analyses were used to identify candidate biological programs potentially linked to morphology-derived states, including transcriptional regulation, chromatin remodeling, metabolic pathways, epithelial–mesenchyma organization, and signaling-related processes.

## 4.11 Statistical analysis

Continuous variables are presented as mean ± standard deviation unless otherwise specified. Group comparisons were performed using Mann–Whitney U tests or Kruskal–Wallis tests depending on the number of groups. Correlation analyses were performed using Spearman correlation coeficients. Survival analyses were conducted using Kaplan–Meier estimation, log-rank testing, and Cox proportional hazards models.

Unless otherwise specified, two-sided $p < 0 . 0 5$ was considered statistically significant. For highdimensional molecular analyses, multiple-testing correction was evaluated using the FDR procedure where appropriate. Given the exploratory nature of the multi-omics analyses, both nominal p values and FDR-adjusted statistics were reported and interpreted primarily in the context of state-associated molecular trends and candidate feature prioritization rather than definitive single-gene discovery. Al statistical tests were two-sided unless otherwise specified.

## 4.12 Computing hardware and software

Model training and downstream analyses were performed on a multi-node Linux computing platform equipped with NVIDIA A800 80GB GPUs. Deep learning models were implemented using PyTorch 1.13.0 under Python 3.9.13. Whole-slide image processing was performed using OpenSlide [45]. Manifold analysis and trajectory inference were performed using Scanpy [39], AnnData, and UMAP-related libraries. Statistical analyses were conducted using SciPy, Lifelines, Scikit-learn, and related Python scientific-computing packages.

All experiments were conducted under a unified software environment to ensure reproducibility. Source code and trained model weights will be publicly released at https://github.com/jiangnansss/UterPath-SpaTIE.

## Data availability

PK3-Uter is a private dataset protected by institutional ethics protocols and patient privacy regulations. Access to this dataset can be granted upon reasonable request to the corresponding authors, subject to approval by the relevant ethics committee and data use agreements. The TCGA-UCEC and TCGA-UCS datasets are publicly available through TCGA Data Portal (https://portal.gdc.cancer.gov). The pan-cancer multi-omics data, including gene expression, copy number variation, DNA methylation, and clinical annotations, were retrieved from the UCSC Xena Browser (https://xenabrowser.net/datapages/?cohort=TCG Cancer%20(PANCAN)).

## Code availability

All custom code and analysis pipelines developed for this study are publicly available at GitHub: https://github.com/jiangnansss/UterPath-SpaTIE.

## Supplementary information

See details in Supplementary Materials.

## Acknowledgments

This study was supported by National Natural Science Foundation of China (82473150), National Key Research and Development Program of China (2023YFC2705805), Beijing Municipal Health Commission Research Ward Excellence Clinical Research Program (BRWEP2024WO32240114), Natural Science Foundation of Fujian Province Project (2026J01310023), Clinical Cohort Construction Program of Peking University Third Hospital (BYSYZD2022015), and Fuzhou University Talent-Introduction Scientific Startup Fund (XRC-25119).

## Conflict of interest

All authors report no disclosures.

## Contributions

Conceptualization and methodology by Qiming He, Yan Liu, Shuang Ge, Tian Guan, Congrong Liu, and Yonghong He; data curation and formal analysis by Fan Yang, Yuxiang Wang, Jing Yang, Ajin Hu, Zixiu Song, Xiaoya Zhao, Xianjing Zheng, Yijun Zheng, Liling Lin, Shuxing Liu, Bin Bao, and Yue Xie; investigation and software by Fan Yang, Ieng Man Zhang, Zihao Jia, Yexing Zhang, Qiang Huang, and Zihan Wang; resources and funding acquisition by Qiming He, Congrong Liu, and Yonghong He; writing – original draft by Qiming He, Yan Liu, and Shuang Ge; writing – review & editing Congrong Liu, and Yonghong He; supervision by Tian Guan, Congrong Liu, and Yonghong He.

## Ethics Statement and Patient Consent

This retrospective study was approved by the Medical Ethics Committee of Peking University Third Hospital (Approval No. IRB00006761-M20250620; Ethical Review Approval Document No. (2025) Medical Ethics Review No. 577-02). The requirement for individual informed consent was waived by the Ethics Committee because the study involved the retrospective analysis of existing pathological slides and de-identified clinical data.

## References

[1] D. Hanahan, Hallmarks of cancer: New dimensions, Cancer Discovery 2022, 12, 1 31.

[2] M. Greaves, C. C. Maley, Clonal evolution in cancer, Nature 2012, 481, 7381 306.

[3] Cancer Genome Atlas Research Network, Integrated genomic characterization of endometrial carcinoma, Nature 2013, 497, 7447 67.

[4] A. D. Cherniack, et al., Integrated molecular characterization of uterine carcinosarcoma, Cancer Cell 2017, 31, 3 411.

[5] A. Heindl, et al., Relevance of spatial heterogeneity of immune infiltration for predicting risk of recurrence after endocrine therapy of er+ breast cancer, J Natl Cancer Inst 2018, 110, 2 166.

[6] A. Marusyk, V. Almendro, K. Polyak, Intra-tumour heterogeneity: a looking glass for cancer?, Nature Reviews Cancer 2012, 12, 5 323.

[7] C. Trapnell, et al., The dynamics and regulators of cell fate decisions are revealed by pseudotemporal ordering of single cells, Nature Biotechnology 2014, 32, 4 381.

[8] W. Saelens, R. Cannoodt, H. Todorov, Y. Saeys, A comparison of single-cell trajectory inference methods, Nature Biotechnology 2019, 37, 5 547.

[9] K. Street, et al., Slingshot: cell lineage and pseudotime inference for single-cell transcriptomics, BMC Genomics 2018, 19, 1 477.

[10] P. L. St˚ahl, F. Salm´en, S. Vickovic, et al., Visualization and analysis of gene expression in tissue sections by spatial transcriptomics, Science 2016, 353, 6294 78.

[11] S. K. Longo, et al., Integrating single-cell and spatial transcriptomics to elucidate intercellular tissue dynamics, Nature Reviews Genetics 2021, 22, 10 627.

[12] G. Campanella, et al., Clinical-grade computational pathology using weakly supervised deep learning on whole slide images, Nature Medicine 2019, 25, 8 1301.

[13] R. J. Chen, T. Ding, M. Y. Lu, et al., Towards a general-purpose foundation model for computational pathology, Nature Medicine 2024, 30 850.

[14] H. Xu, N. Usuyama, J. Bagga, et al., A whole-slide foundation model for digital pathology from real-world data, Nature 2024, 630, 8015 181.

[15] E. Vorontsov, et al., Virchow: A million-slide digital pathology foundation model, arXiv preprint arXiv:2309.07778 2024.

[16] Q. He, J. Li, T. Guan, Y. Ma, Z. Zhao, Y. Wang, H. Chen, Y. Xu, S. Ge, Y. Zhang, et al., Glopath: An entity-centric foundation model for glomerular lesion assessment and clinicopathological insights, Advanced Science 2026, e20580.

[17] L. Zhu, X. Ling, M. Ouyang, X. Liu, T. Guan, M. Fu, M. Zeng, Z. Cheng, F. Fu, Q. Huang, et al., Subspecialty-specific foundation model for intelligent gastrointestinal pathology, npj Digital Medicine 2026.

[18] M. Y. Lu, D. F. Williamson, T. Y. Chen, R. J. Chen, M. Barbieri, F. Mahmood, Data-eficient and weakly supervised computational pathology on whole-slide images, Nature biomedical engineering 2021, 5, 6 555.

[19] E. Vorontsov, et al., A foundation model for clinical-grade computational pathology and rare cancer detection, Nature Medicine 2024, 30 2924.

[20] T. Ding, S. J. Wagner, A. H. Song, R. J. Chen, M. Y. Lu, A. Zhang, A. J. Vaidya, G. Jaume, M. Shaban, A. Kim, et al., A multimodal whole-slide foundation model for pathology, Nature medicine 2025, 1–13.

[21] P. Neidlinger, O. S. El Nahhas, H. S. Muti, T. Lenz, M. Hofmeister, H. Brenner, M. van Treeck, R. Langer, B. Dislich, H. M. Behrens, et al., Benchmarking foundation models as feature extractors for weakly supervised computational pathology, Nature biomedical engineering 2025, 1–11.

[22] G. Campanella, S. Chen, M. Singh, R. Verma, S. Muehlstedt, J. Zeng, A. Stock, M. Croken, B. Veremis, A. Elmas, et al., A clinical benchmark of public self-supervised pathology foundation models, Nature Communications 2025, 16, 1 3640.

[23] V. Bergen, et al., Generalizing rna velocity to transient cell states through dynamical modeling, Nature Biotechnology 2020, 38, 12 1408.

[24] S. Deltadahl, J. Gilbey, C. Van Laer, N. Boeckx, M. P. Leers, T. Freeman, L. Aiken, T. Farren, M. Smith, M. Zeina, et al., Deep generative classification of blood cell morphology, Nature Machine Intelligence 2025, 7, 11 1791.

[25] Y. Liu, L. Cai, R. Rong, S. Wang, L. Jia, P. Quan, Q. Zhou, G. Xiao, Y. Xie, Image-based inference of tumor cell trajectories enables large-scale cancer progression analysis, Science Advances 2025, 11, 29 eadv9466.

[26] Z.-B. Qiu, J. Li, S. Dou, Q. Meng, M.-M. Wang, H.-J. Li, C. Zhang, H. Xie, B.-Y. Jiang, J.-T. Lin, et al., Quantifying early-stage lung adenocarcinoma progression with a radiomic trajectory, npj Digital Medicine 2025, 8, 1 664.

[27] W. Chen, P. Zhang, T. N. Tran, Y. Xiao, S. Li, V. V. Shah, H. Cheng, K. W. Brannan, K. Youker, L. Lai, et al., A visual–omics foundation model to bridge histopathology with spatial transcriptomics, Nature Methods 2025, 22, 7 1568.

[28] R. R. Coifman, S. Lafon, Geometric difusions as a tool for harmonic analysis and structure definition of data: Difusion maps, Proceedings of the National Academy of Sciences 2005, 102, 21 7426.

[29] L. Haghverdi, F. Buettner, F. J. Theis, Difusion pseudotime robustly reconstructs lineage branching, Nature Methods 2016, 13, 10 845.

[30] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, et al., Dinov2: Learning robust visual features without supervision, arXiv preprint arXiv:2304.07193 2023.

[31] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen, et al., Lora: Low-rank adaptation of large language models., Iclr 2022, 1, 2 3.

[32] R. J. Chen, T. Ding, M. Y. Lu, D. F. Williamson, G. Jaume, A. H. Song, B. Chen, A. Zhang, D. Shao, M. Shaban, et al., Towards a general-purpose foundation model for computational pathology, Nature medicine 2024, 30, 3 850.

[33] C. Bunne, Y. Roohani, Y. Rosen, A. Gupta, X. Zhang, M. Roed, T. Alexandrov, M. AlQuraishi, P. Brennan, D. B. Burkhardt, et al., How to build the virtual cell with artificial intelligence: Priorities and opportunities, Cell 2024, 187, 25 7045.

[34] E. J. Crosbie, S. J. Kitson, J. N. McAlpine, A. Mukhopadhyay, M. E. Powell, N. Singh, Endometrial cancer, The Lancet 2022, 399, 10333 1412.

[35] E. D. Shulman, E. M. Campagnolo, R. Lodha, Y. Chung, A. Stemmer, T. Cantore, B. Ru, T.-G. Chang, S. Biswas, S. R. Dhruba, et al., Ai-predicted spatial transcriptomics unlocks breast cancer biomarkers from pathology, Cell 2026.

[36] Y. Liu, Y. Dai, L. Wang, Spatial omics at the forefront: emerging technologies, analytical innovations, and clinical applications, Cancer cell 2025.

[37] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, et al., An image is worth 16x16 words: Transformers for image recognition at scale, arXiv preprint arXiv:2010.11929 2020.

[38] I. Loshchilov, F. Hutter, Decoupled weight decay regularization, arXiv preprint arXiv:1711.05101 2017.

[39] F. A. Wolf, P. Angerer, F. J. Theis, Scanpy: large-scale single-cell gene expression data analysis, Genome biology 2018, 19, 1 15.

[40] L. McInnes, J. Healy, J. Melville, Umap: Uniform manifold approximation and projection for dimension reduction, arXiv preprint arXiv:1802.03426 2018.

[41] V. A. Traag, L. Waltman, N. J. Van Eck, From louvain to leiden: guaranteeing well-connected communities, Scientific reports 2019, 9, 1 5233.

[42] M. V. Kuleshov, M. R. Jones, A. D. Rouillard, N. F. Fernandez, Q. Duan, Z. Wang, S. Koplev, S. L. Jenkins, K. M. Jagodnik, A. Lachmann, et al., Enrichr: a comprehensive gene set enrichment analysis web server 2016 update, Nucleic acids research 2016, 44, W1 W90.

[43] M. Kanehisa, M. Furumichi, Y. Sato, M. Ishiguro-Watanabe, M. Tanabe, Kegg: integrating viruses and cellular organisms, Nucleic acids research 2021, 49, D1 D545.

[44] L. Qian, Z. Dong, T. Guo, Grow ai virtual cells: three data pillars and closed-loop learning, Cell Research 2025, 35, 5 319.

[45] A. Goode, B. Gilbert, J. Harkes, D. Jukic, M. Satyanarayanan, Openslide: A vendor-neutral software foundation for digital pathology, Journal of pathology informatics 2013, 4, 1 27.

[46] Q. He, Y. Liu, F. Pan, H. Duan, J. Guan, Z. Liang, H. Zhong, X. Wang, Y. He, W. Huang, et al., Unsupervised domain adaptive tumor region recognition for ki67 automated assisted quantification, International Journal of Computer Assisted Radiology and Surgery 2023, 18, 4 629.

[47] Y. Liu, Q. He, H. Duan, H. Shi, A. Han, Y. He, Using sparse patch annotation for tumor segmentation in histopathological images, Sensors 2022, 22, 16 6053.

## Supplementary Material

## 1 Definition of downstream tasks

The definition of downstream tasks are shown as Table S1.

Table S1: Downstream prediction tasks in TCGA-UCEC and TCGA-UCS cohorts.
<table><tr><td>Task</td><td>TCGA- UCEC</td><td>TCGA- UCS</td><td>Definition &amp; Categories</td></tr><tr><td>grade</td><td>√</td><td></td><td>Neoplasm histologic grading (G1/G2/G3/High).</td></tr><tr><td>subtype</td><td>√</td><td></td><td>Molecular subtyping (CN_HIGH/CN_LOW/POLE/MSI).</td></tr><tr><td>FIGO</td><td>√</td><td>√</td><td>FIGO stage prediction (I/II/III/IV)</td></tr><tr><td>Disease_type</td><td>√</td><td></td><td>Tumor histological subtyping (Endometrioid Endometrial Adenocarci- noma, Mixed Serous and Endometrioid Carcinoma, Serous Endometrial</td></tr><tr><td>arid1a</td><td>√</td><td></td><td>Adenocarcinoma). ARID1A mutation status prediction (Negative/Positive).</td></tr><tr><td>kmt2d</td><td>√</td><td></td><td>KMT2D mutation status prediction (Negative/Positive).</td></tr><tr><td>muc16</td><td>√</td><td></td><td>MUC16 mutation status prediction (Negative/Positive).</td></tr><tr><td>pik3a</td><td>√</td><td>√</td><td>PIK3CA mutation status prediction (Negative/Positive).</td></tr><tr><td>pten</td><td>√</td><td></td><td>PTEN mutation status prediction (Negative/Positive).</td></tr><tr><td>tp53</td><td>√</td><td>√</td><td>TP53 mutation status prediction (Negative/Positive).</td></tr><tr><td>ttn</td><td>√</td><td></td><td>TTN mutation status prediction (Negative/Positive).</td></tr><tr><td>3_year_survival</td><td>√</td><td>√</td><td>Three-year survival prediction (Dead/Alive).</td></tr><tr><td>5_year_survival</td><td>√</td><td>√</td><td>Five-year survival prediction (Dead/Alive).</td></tr><tr><td>morphology</td><td>√</td><td></td><td>Distinguishing between UCEC and UCS (UCEC/UCS).</td></tr><tr><td>ppp2r1a</td><td></td><td>√</td><td>PPP2R1A mutation status prediction (Negative/Positive).</td></tr><tr><td>fbxw7</td><td></td><td>√</td><td>FBXW7 mutation status prediction (Negative/Positive).</td></tr><tr><td>origin</td><td></td><td>√</td><td>Distinguishing between primary and metastatic UCS (Pri- mary/Metastatic).</td></tr></table>

## 2 Detailed results of downstream tasks

The detailed results of donwstream tasks are shown in Table S2-S7.

Table S2: Performance of all the models on TCGA-UCEC (F1-score, mean ± std). Bold indicates the optimal result for the current task.
<table><tr><td>Task</td><td>Baseline</td><td>UPFM</td><td>SpaTIE</td></tr><tr><td>disease_type</td><td>0.813±0.013</td><td>0.882±0.029</td><td>0.880±0.002</td></tr><tr><td>subtype</td><td>0.494±0.005</td><td>0.592±0.008</td><td>0.593±0.015</td></tr><tr><td>figo</td><td>0.352±0.010</td><td>0.453±0.007</td><td>0.446±0.018</td></tr><tr><td>grade</td><td>0.640±0.004</td><td>0.688±0.019</td><td>0.685±0.026</td></tr><tr><td>pten</td><td>0.739±0.032</td><td>0.842±0.015</td><td>0.846±0.017</td></tr><tr><td>pik3a</td><td>0.577±0.009</td><td>0.585±0.017</td><td>0.579±0.003</td></tr><tr><td>arid1a</td><td>0.623±0.007</td><td>0.701±0.003</td><td>0.711±0.006</td></tr><tr><td>muc16</td><td>0.604±0.018</td><td>0.666±0.003</td><td>0.659±0.026</td></tr><tr><td>ttn</td><td>0.588±0.006</td><td>0.701±0.012</td><td>0.710±0.017</td></tr><tr><td>kmt2d</td><td>0.579±0.003</td><td>0.642±0.016</td><td>0.627±0.003</td></tr><tr><td>tp53</td><td>0.722±0.010</td><td>0.797±0.019</td><td>0.809±0.005</td></tr><tr><td>3-years</td><td>0.760±0.004</td><td>0.758±0.027</td><td>0.765±0.002</td></tr><tr><td>5-years</td><td>0.674±0.015</td><td>0.742±0.007</td><td>0.742±0.002</td></tr></table>

## 3 Tumor-focused patch extraction

To improve pretraining eficiency and reduce background-induced noise, we performed a tumor-focused patch extraction procedure before self-supervised pretraining. Directly extracting patches from all tissue regions yielded more than 60 million candidate patches, many of which corresponded to normal tissue, stromal regions, or low-information areas unrelated to tumor morphology. To concentrate the

Table S3: Performance of all the models on TCGA-UCEC (Accuracy, mean ± std).
<table><tr><td>Task</td><td>Baseline</td><td>UPFM</td><td>SpaTIE</td></tr><tr><td>disease_type</td><td> $0 . 8 6 0 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td> $0 . 9 0 8 { \pm } 0 . 0 2 8$ </td><td> $\mathbf { 0 . 9 1 1 { \pm } 0 . 0 0 3 }$ </td></tr><tr><td>subtype</td><td> $0 . 5 3 9 { \pm } 0 . 0 0 4$ </td><td> $0 . 6 2 9 { \pm } 0 . 0 2 2$ </td><td> $\mathbf { 0 . 6 3 0 { \pm 0 . 0 1 1 } }$ </td></tr><tr><td> $\mathrm { f g o }$ </td><td> $0 . 6 2 0 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $0 . 6 6 8 { \pm } 0 . 0 2 3$ </td><td> $\mathbf { 0 . 6 7 8 { \scriptstyle \pm 0 . 0 0 2 } }$ </td></tr><tr><td>grade</td><td> $0 . 7 0 6 { \pm } 0 . 0 0 8$ </td><td> $0 . 7 4 6 { \pm } 0 . 0 1 2$ </td><td> $\mathbf { 0 . 7 5 4 } \pm \mathbf { 0 . 0 1 6 }$ </td></tr><tr><td> $\mathrm { p t e n }$ </td><td> $0 . 7 7 4 { \pm } 0 . 0 3 0$ </td><td> $0 . 8 5 8 { \pm } 0 . 0 0 5$ </td><td> $\mathbf { 0 . 8 6 5 { \pm } 0 . 0 0 2 }$ </td></tr><tr><td> $\mathrm { \ p i k 3 a }$ </td><td> $0 . 5 7 7 { \scriptstyle \pm 0 . 0 0 9 }$ </td><td> $0 . 5 8 7 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td> $\mathbf { 0 . 6 0 2 { \scriptstyle \pm 0 . 0 0 3 } }$ </td></tr><tr><td> $\mathrm { a r i d l a }$ </td><td> $0 . 6 2 4 { \pm } 0 . 0 0 3$ </td><td> $0 . 7 0 2 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $\mathbf { 0 . 7 1 1 { \pm } 0 . 0 0 5 }$ </td></tr><tr><td> $\mathrm { \ m u c 1 6 }$ </td><td> $0 . 7 1 3 { \pm } 0 . 0 0 2$ </td><td> $\mathbf { 0 . 7 6 9 { \scriptstyle \pm 0 . 0 1 6 } }$ </td><td> $0 . 7 4 9 { \pm } 0 . 0 0 3$ </td></tr><tr><td>ttn</td><td> $0 . 6 1 1 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $0 . 7 1 7 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $\mathbf { 0 . 7 3 0 { \pm } 0 . 0 1 8 }$ </td></tr><tr><td>kmt2d</td><td> $0 . 6 9 8 { \pm } 0 . 0 0 2$ </td><td> $0 . 7 3 2 { \pm } 0 . 0 1 0$ </td><td> $\mathbf { 0 . 7 4 1 { \pm } 0 . 0 1 3 }$ </td></tr><tr><td> $\mathrm { t p 5 3 }$ </td><td> $0 . 7 3 9 { \pm } 0 . 0 1 4$ </td><td> $0 . 8 1 8 { \pm } 0 . 0 0 3$ </td><td> $\mathbf { 0 . 8 2 5 { \pm 0 . 0 2 4 } }$ </td></tr><tr><td> $3 \mathrm { - y e a r s }$ </td><td> $0 . 7 6 9 { \pm } 0 . 0 1 4$ </td><td> $0 . 7 7 0 { \scriptstyle \pm 0 . 0 3 1 }$ </td><td> $\mathbf { 0 . 7 7 4 { \pm 0 . 0 0 5 } }$ </td></tr><tr><td> $5 \mathrm { - y e a r s }$ </td><td> $0 . 7 9 8 { \pm } 0 . 0 0 7$ </td><td> $0 . 8 3 2 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $\mathbf { 0 . 8 3 3 \pm 0 . 0 1 4 }$ </td></tr></table>

Table S4: Performance of all the models on TCGA-UCEC (ROC-AUC, mean ± std).
<table><tr><td>Task</td><td>Baseline</td><td>UPFM</td><td>SpaTIE</td></tr><tr><td>disease_type</td><td> $0 . 8 9 3 { \scriptstyle \pm 0 . 0 2 6 }$ </td><td> $0 . 9 3 4 { \pm } 0 . 0 2 9$ </td><td> $\mathbf { 0 . 9 3 8 { \pm 0 . 0 2 5 } }$ </td></tr><tr><td>subtype</td><td> $0 . 7 2 9 { \pm } 0 . 0 0 8$ </td><td> $0 . 7 9 8 { \pm } 0 . 0 0 5$ </td><td> $\mathbf { 0 . 7 9 9 } { \pm } \mathbf { 0 . 0 1 4 }$ </td></tr><tr><td>figo</td><td> $0 . 6 1 3 { \pm } 0 . 0 0 2$ </td><td> $0 . 6 8 4 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td> $\mathbf { 0 . 6 9 3 { \scriptstyle \pm 0 . 0 0 4 } }$ </td></tr><tr><td>grade</td><td> $0 . 8 3 7 { \pm } 0 . 0 0 8$ </td><td> $\mathbf { 0 . 8 6 6 { \pm 0 . 0 3 0 } }$ </td><td> $0 . 8 6 5 { \pm } 0 . 0 0 9$ </td></tr><tr><td>pten</td><td> $0 . 8 0 1 { \scriptstyle \pm 0 . 0 2 7 }$ </td><td> $\mathbf { 0 . 8 9 0 { \pm } 0 . 0 0 2 }$ </td><td> $0 . 8 8 9 { \pm } 0 . 0 0 6$ </td></tr><tr><td>pik3a</td><td> $0 . 5 8 7 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td> $0 . 5 9 2 { \scriptstyle \pm 0 . 0 2 2 }$ </td><td> $\mathbf { 0 . 6 0 2 } { \pm } \mathbf { 0 . 0 0 5 }$ </td></tr><tr><td>arid1a</td><td> $0 . 6 5 1 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td> $0 . 7 4 8 { \pm } 0 . 0 0 5$ </td><td> $\mathbf { 0 . 7 5 1 { \pm } 0 . 0 0 7 }$ </td></tr><tr><td>muc16</td><td> $0 . 6 3 3 { \pm } 0 . 0 2 8$ </td><td> $0 . 7 0 4 { \scriptstyle \pm 0 . 0 2 4 }$ </td><td> $\mathbf { 0 . 7 3 6 { \pm 0 . 0 0 8 } }$ </td></tr><tr><td>ttn</td><td> $0 . 5 9 6 { \pm } 0 . 0 0 8$ </td><td> $\mathbf { 0 . 7 0 5 { \scriptstyle \pm 0 . 0 0 9 } }$ </td><td> $\mathbf { 0 . 7 0 5 { \scriptstyle \pm 0 . 0 0 9 } }$ </td></tr><tr><td>kmt2d</td><td> $0 . 6 3 3 { \pm } 0 . 0 0 4$ </td><td> $0 . 6 7 0 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $\mathbf { 0 . 6 8 3 { \pm } 0 . 0 0 5 }$ </td></tr><tr><td> $\mathrm { t p 5 3 }$ </td><td> $0 . 7 7 6 { \pm } 0 . 0 0 3$ </td><td> $0 . 8 3 7 { \pm } 0 . 0 1 6$ </td><td> $\mathbf { 0 . 8 4 4 { \scriptstyle \pm 0 . 0 1 5 } }$ </td></tr><tr><td>3-years</td><td> $0 . 8 0 5 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td> $0 . 8 2 1 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td> $\mathbf { 0 . 8 2 4 } 2 \mathbf { 0 . 0 2 6 }$ </td></tr><tr><td>5-years</td><td> $0 . 7 7 6 { \pm } 0 . 0 1 8$ </td><td> $0 . 8 4 5 { \scriptstyle \pm 0 . 0 2 4 }$ </td><td> $\mathbf { 0 . 8 4 6 { \pm 0 . 0 2 4 } }$ </td></tr></table>

Table S5: Performance of all the models on TCGA-UCS (F1-score, mean ± std).
<table><tr><td>Task</td><td>Baseline</td><td>UPFM</td><td>SpaTIE</td></tr><tr><td>morphology</td><td> $0 . 8 4 4 { \pm } 0 . 0 0 2$ </td><td> $0 . 7 8 8 { \pm } 0 . 0 0 2$ </td><td> $\mathbf { 0 . 8 5 4 \pm 0 . 0 2 0 }$ </td></tr><tr><td>origin</td><td> $0 . 7 5 0 { \pm } 0 . 0 1 0$ </td><td> $0 . 7 6 3 { \pm } 0 . 0 2 8$ </td><td> $\mathbf { 0 . 8 0 0 { \pm } 0 . 0 0 5 }$ </td></tr><tr><td>figo</td><td> $\mathbf { 0 . 5 8 9 { \pm 0 . 0 1 9 } }$ </td><td> $0 . 5 2 3 { \pm } 0 . 0 0 5$ </td><td> $0 . 5 5 9 { \pm } 0 . 0 0 2$ </td></tr><tr><td>fbxw7</td><td> $0 . 7 1 1 { \pm } 0 . 0 0 6$ </td><td> $0 . 6 6 5 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $\mathbf { 0 . 7 3 5 { \pm } 0 . 0 0 7 }$ </td></tr><tr><td> $\mathrm { \ p i k 3 a }$ </td><td> $0 . 6 7 6 { \scriptstyle \pm 0 . 0 3 1 }$ </td><td> $0 . 7 4 3 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td> $\mathbf { 0 . 7 4 6 { \scriptstyle \pm 0 . 0 0 4 } }$ </td></tr><tr><td> $\mathrm { p p p 2 r 1 a }$ </td><td> $0 . 5 7 9 { \pm } 0 . 0 1 2$ </td><td> $\mathbf { 0 . 6 4 2 { \scriptstyle \pm 0 . 0 0 7 } }$ </td><td> $0 . 6 2 7 { \scriptstyle \pm 0 . 0 0 7 }$ </td></tr><tr><td> $\mathrm { t p 5 3 }$ </td><td> $0 . 7 6 8 { \pm } 0 . 0 0 2$ </td><td> $0 . 8 3 8 { \pm } 0 . 0 0 4$ </td><td> $\mathbf { 0 . 8 4 0 { \scriptstyle \pm 0 . 0 1 7 } }$ </td></tr></table>

Table S6: Performance of all the models on TCGA-UCS $( \mathrm { A c c u r a c y } , \mathrm { m e a n } \pm \mathrm { s t d } )$
<table><tr><td>Task</td><td>Baseline</td><td>UPFM</td><td>SpaTIE</td></tr><tr><td>morphology</td><td> $0 . 9 0 0 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td> $0 . 8 8 0 { \pm } 0 . 0 0 9$ </td><td> $\mathbf { 0 . 9 2 6 { \pm } 0 . 0 0 3 }$ </td></tr><tr><td>origin</td><td> $0 . 7 8 1 { \pm } 0 . 0 0 5$ </td><td> $0 . 7 9 2 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $\mathbf { 0 . 8 2 5 { \pm 0 . 0 1 2 } }$ </td></tr><tr><td>figo</td><td> $0 . 6 0 5 { \scriptstyle \pm 0 . 0 2 4 }$ </td><td> $0 . 5 9 4 { \pm } 0 . 0 0 3$ </td><td> $\mathbf { 0 . 6 2 7 { \pm 0 . 0 2 8 } }$ </td></tr><tr><td>fbxw7</td><td> $0 . 7 2 4 { \pm } 0 . 0 2 1$ </td><td> $0 . 7 5 6 { \pm } 0 . 0 0 2$ </td><td> $\mathbf { 0 . 7 5 9 { \pm 0 . 0 0 9 } }$ </td></tr><tr><td>pik3a</td><td> $0 . 7 2 6 { \pm } 0 . 0 1 4$ </td><td> $0 . 7 5 9 { \pm } 0 . 0 0 8$ </td><td> $\mathbf { 0 . 7 6 7 { \scriptstyle \pm 0 . 0 1 8 } }$ </td></tr><tr><td> $\mathrm { p p p 2 r 1 a }$ </td><td> $0 . 6 9 8 { \pm } 0 . 0 2 5$ </td><td> $0 . 7 3 2 { \pm } 0 . 0 0 8$ </td><td> $\mathbf { 0 . 7 4 1 { \pm } 0 . 0 1 3 }$ </td></tr><tr><td> $\mathrm { t p 5 3 }$ </td><td> $0 . 9 2 3 { \pm } 0 . 0 2 1$ </td><td> $0 . 9 4 4 { \pm } 0 . 0 1 0$ </td><td> $\mathbf { 0 . 9 4 5 { \pm 0 . 0 0 2 } }$ </td></tr></table>

Table S7: Performance of all the models on TCGA-UCS (ROC-AUC, mean ± std).
<table><tr><td>Task</td><td>Baseline</td><td>UPFM</td><td>SpaTIE</td></tr><tr><td>morphology</td><td>0.924±0.030</td><td> $\mathbf { 0 . 9 3 5 { \pm 0 . 0 0 5 } }$ </td><td>0.905±0.025</td></tr><tr><td>origin</td><td>0.783±0.019</td><td> $\mathbf { 0 . 8 2 2 { \scriptstyle \pm 0 . 0 0 4 } }$ </td><td> $0 . 8 2 1 { \scriptstyle \pm 0 . 0 2 7 }$ </td></tr><tr><td>figo</td><td>0.734±0.025</td><td> $0 . 7 1 0 { \pm } 0 . 0 1 3$ </td><td> $\mathbf { 0 . 7 1 2 { \scriptstyle \pm 0 . 0 1 0 } }$ </td></tr><tr><td>fbxw7</td><td>0.698±0.026</td><td> $\mathbf { 0 . 7 3 0 { \pm } 0 . 0 0 2 }$ </td><td> $0 . 7 1 3 { \pm } 0 . 0 0 4$ </td></tr><tr><td>pik3a</td><td>0.708±0.005</td><td> $0 . 7 8 0 { \pm } 0 . 0 1 9$ </td><td> $\mathbf { 0 . 8 2 3 { \pm } 0 . 0 2 3 }$ </td></tr><tr><td>ppp2r1a</td><td>0.633±0.007</td><td> $0 . 6 7 0 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $\mathbf { 0 . 6 8 3 { \pm 0 . 0 2 6 } }$ </td></tr><tr><td>tp53</td><td> $0 . 8 1 8 { \pm } 0 . 0 0 4$ </td><td> $\mathbf { 0 . 9 5 2 } { \pm } \mathbf { 0 . 0 0 5 }$ </td><td> $0 . 9 5 0 { \pm } 0 . 0 1 3$ </td></tr></table>

pretraining corpus on diagnostically relevant regions, we additionally annotated 10,000 tumor patches and 10,000 non-tumor patches from the uterine pathology slides and trained a binary tumor-region classifier.

Previous tumor region segmentation methods have provided insights into the extraction of key patches [46, 47]. Specifically, we initialized a ViT-L backbone with UNI weights [13] and fine-tuned the model for tumor-versus-non-tumor classification using standard cross-entropy loss. The trained classifier was then applied to the full patch pool, and only patches predicted as tumor regions were retained for subsequent self-supervised pretraining. This procedure reduced the number of pretraining patches from over 60 million to approximately 20 million, substantially increasing the tumor-content ratio and improving the signal-to-noise characteristics of the pretraining dataset.

Table S8: Associations between SpaTIE-derived state values and clinicopathological variables in TCGA-UCEC and TCGA-UCS cohorts.
<table><tr><td>Cohort Clinical variable</td><td></td><td>Statistical value</td><td>p value</td><td>FDR</td></tr><tr><td colspan="5">TCGA-UCEC</td></tr><tr><td>UCEC UCEC</td><td>Fraction Genome Altered (FGA) Diagnosis Age</td><td>ρ= −0.181 ρ= −0.071</td><td> $\overline { { 1 . 7 6 \times 1 0 ^ { - 5 } } }$ </td><td> $5 . 2 9 \times 1 0 ^ { - 5 }$  0.1507</td></tr><tr><td>UCEC UCEC</td><td>Birth from Initial Pathologic Diagnosis Date</td><td>ρ = 0.069 ρ= −0.051</td><td>0.0930 0.1004</td><td>0.1507 0.2920</td></tr><tr><td>UCEC</td><td>TMB (nonsynonymous) Mutation Count</td><td>ρ = −0.041</td><td>0.2433 0.3442</td><td>0.3442</td></tr><tr><td>UCEC</td><td>FIGO Major Stage</td><td>H = 15.357</td><td>0.0015</td><td>0.0077</td></tr><tr><td>UCEC</td><td></td><td>U = 13936.0</td><td>0.0078</td><td>0.0259</td></tr><tr><td>UCEC</td><td>Prior Malignancy</td><td>U = 23647.0</td><td>0.0262</td><td>0.0656</td></tr><tr><td></td><td>Patient Vital Status</td><td>U = 2334.0</td><td></td><td></td></tr><tr><td>UCEC</td><td>Ethnicity Category</td><td>H = 8.671</td><td>0.0355</td><td>0.0709</td></tr><tr><td>UCEC</td><td>Race Category</td><td></td><td>0.0699</td><td>0.1164</td></tr><tr><td>UCEC</td><td>Morphology</td><td>H = 5.093</td><td>0.1651</td><td>0.2064</td></tr><tr><td>UCEC</td><td>Disease Type</td><td>H = 2.700</td><td>0.2593</td><td>0.2593</td></tr><tr><td>UCEC</td><td>Primary Diagnosis</td><td>H = 2.700</td><td>0.2593</td><td>0.2593</td></tr><tr><td>TCGA-UCS</td><td></td><td></td><td></td><td></td></tr><tr><td>UCS</td><td>Year of Diagnosis</td><td>ρ = −0.241</td><td>0.0215</td><td></td></tr><tr><td>UCS</td><td>Fraction Genome Altered (FGA)</td><td>ρ= 0.219</td><td>0.0377</td><td>0.1130</td></tr><tr><td>UCS</td><td></td><td>ρ = 0.133</td><td>0.2091</td><td>0.1130</td></tr><tr><td>UCS</td><td>Mutation Count</td><td>ρ = 0.133</td><td>0.2100</td><td>0.3015</td></tr><tr><td>UCS</td><td>TMB (nonsynonymous)</td><td>ρ= 0.121</td><td>0.2533</td><td>0.3015</td></tr><tr><td>UCS</td><td>Birth from Initial Pathologic Diagnosis Date</td><td></td><td></td><td>0.3015</td></tr><tr><td></td><td>Diagnosis Age</td><td>ρ = −0.109</td><td>0.3015</td><td>0.3015</td></tr><tr><td>UCS</td><td>Morphology</td><td>U = 916.0</td><td>0.0047</td><td>0.0379</td></tr><tr><td>UCS</td><td>Race Category</td><td>H = 8.847</td><td>0.0120</td><td>0.0480</td></tr><tr><td>UCS</td><td>Disease Type</td><td>H = 2.706</td><td>0.2584</td><td>0.4669</td></tr><tr><td>UCS</td><td>Primary Diagnosis</td><td>H = 2.706</td><td>0.2584</td><td>0.4669</td></tr><tr><td>UCS</td><td>FIGO Major Stage</td><td>H = 2.795</td><td>0.4243</td><td>0.5657</td></tr><tr><td>UCS</td><td>Prior Malignancy</td><td>U = 515.0</td><td>0.6347</td><td>0.7254</td></tr><tr><td>UCS</td><td>Patient Vital Status</td><td>U = 991.0</td><td>0.8600</td><td>0.8600</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

## 4 Details of association between SpaTIE-derived progression states and clinical features

The details of association between SpaTIE-derived progression states and clinical features are shown in Table S8.

## 5 Details of association between SpaTIE-derived progression states and multi-omics tumor programs

Table S9: Pathway-level associations between SpaTIE-derived progression states and multi-omics pathway scores. Pathway scores were compared between low- and high-state groups. These results are used as pathway-level context and should not be interpreted as evidence that all member features are individually significant.
<table><tr><td>Cohort</td><td>Omics</td><td>Pathway</td><td>p value</td><td>n</td><td>Higher group</td></tr><tr><td>UCS</td><td>RNASeq</td><td>Estrogen_Response</td><td>0.0048</td><td>57</td><td>low-state</td></tr><tr><td>UCEC</td><td>Methylation</td><td>PRC2_Polycomb</td><td>0.0189</td><td>389</td><td>high-state</td></tr><tr><td>UCEC</td><td>SCNV</td><td>MMR_CNV</td><td>0.0222</td><td>496</td><td>low-state</td></tr><tr><td>UCEC</td><td>RPPA</td><td>AKT_mTOR</td><td>0.0249</td><td>405</td><td>low-state</td></tr><tr><td>UCEC</td><td>RPPA</td><td>Cell_Cycle</td><td>0.0310</td><td>405</td><td>low-state</td></tr></table>

We re-evaluated the association between SpaTIE-derived progression states and multi-omics profiles using DNA methylation, somatic copy-number variation (SCNV), mutation, RNA sequencing (RNASeq), and RPPA data from TCGA-UCEC and TCGA-UCS. Because this analysis involved many molecular features and the UCS cohort was comparatively small, we emphasized signals that were supported after multiple-testing correction or were reproduced at pathway level. Nominal associations with FDR > 0.05 are reported only as exploratory observations.

At the pathway level, only a limited number of pathway-score diferences were detected between low- and high-state groups (Table S9). In TCGA-UCEC, significant nominal diferences were observed for methylation-based PRC2/Polycomb scores, SCNV-based MMR-related scores, and RPPA-based AKT–mTOR and cell-cycle scores. In TCGA-UCS, RNASeq-based estrogen-response scores difered between state groups. These pathway-level findings support the interpretation that the morphologyderived state axis is associated with selected molecular programs rather than with a broad, uniformly significant multi-omics shift.

Feature-level analyses showed the clearest corrected support in TCGA-UCEC. The strongest FDRsupported signals were concentrated in DNA methylation, SCNV, and RPPA layers, whereas mutation and RNASeq features were mainly nominal in the current analysis. Representative UCEC features that were also prioritized by virtual perturbation included methylation features such as PRR22, RBM27, and LRRC42; SCNV features such as FAM81A, RNA5SP396, and GCNT3; and RPPA features such as AXL, GAPDH, ERRFI1, CTNNA1, and SERPINE1 (Table S10). These results suggest that the UCEC state axis has measurable molecular correlates, especially in epigenetic, copy-number, and protein-level measurements.

In TCGA-UCS, several individual features showed relatively large efect sizes, including RNASeq, methylation, RPPA, mutation, and cytoband-level SCNV candidates. However, the top UCS featurelevel associations did not show FDR support in the current sample. Therefore, UCS molecular results are retained as exploratory candidates for hypothesis generation, not as validated molecular correlates of the morphology-derived state axis. Overall, the multi-omics analyses support a cautious interpretation: SpaTIE-derived states are accompanied by selected molecular variation, with stronger statistical support in UCEC than in UCS.

## 6 Details of progression-guided virtual perturbation analysis

Progression-guided virtual perturbation was used as a computational feature-prioritization procedure. For each molecular feature, we estimated how much the feature–state association was attenuated after removing the feature-associated component from the inferred state values. The resulting score should not be interpreted as an experimental knockout efect or as causal evidence. It is better viewed as a ranking statistic that highlights features whose measured variation is tightly coupled to the morphology-derived state axis.

The updated perturbation results are summarized in Table S10. In TCGA-UCEC, the topranked features were PRR22 for methylation, GPR84 for mutation, FAM81A for SCNV, RAB27B for RNASeq, and AXL for RPPA. Among the top 20 perturbation hits, FDR-supported state associations were present for methylation (20/20), SCNV (3/20), and RPPA (5/20), but not for mutation or RNASeq. Accordingly, we place more weight on UCEC methylation, SCNV, and RPPA findings, and treat UCEC mutation and RNASeq candidates as exploratory.

In TCGA-UCS, the top-ranked perturbation features were PIK3C2G for methylation, TPTE for mutation, $1 6 \mathrm { q 2 3 . 1 }$ for SCNV, UFD1L for RNASeq, and ARHI for RPPA. Although the perturbation scores were sometimes larger than those observed in UCEC, none of the top 20 features in the UCS modalities had feature-level FDR support. This pattern is consistent with limited statistical power and greater uncertainty in the UCS cohort. We therefore describe UCS perturbation results as exploratory molecular leads.

Table S10: Overview of progression-guided virtual perturbation results. The last column reports how many of the top 20 perturbation-ranked features also showed FDR-supported association with SpaTIEderived progression states in the feature-level analysis.
<table><tr><td>Cohort</td><td>Omics</td><td>Top feature</td><td>KO score</td><td>Base ρ</td><td>p value</td><td>FDR-supported top 20</td></tr><tr><td>UCEC</td><td>Methylation</td><td>PRR22</td><td>0.312</td><td>0.338</td><td> $\overline { { 7 . 7 5 \times 1 0 ^ { - 1 2 } } }$ </td><td>20</td></tr><tr><td>UCEC</td><td>Mutation</td><td>GPR84</td><td>2.307</td><td>-0.176</td><td> $5 . 8 1 \times 1 0 ^ { - 3 }$ </td><td>0</td></tr><tr><td>UCEC</td><td>SCNV</td><td>FAM81A</td><td>0.138</td><td>0.144</td><td> $1 . 2 8 \times 1 0 ^ { - 3 }$ </td><td>3</td></tr><tr><td>UCEC</td><td>RNASeq</td><td>RAB27B</td><td>0.302</td><td>-0.302</td><td> $1 . 7 0 \times 1 0 ^ { - 4 }$ </td><td>0</td></tr><tr><td>UCEC</td><td>RPPA</td><td>AXL</td><td>0.186</td><td>0.189</td><td> $1 . 2 5 \times 1 0 ^ { - 4 }$ </td><td>5</td></tr><tr><td>UCS</td><td>Methylation</td><td>PIK3C2G</td><td>0.428</td><td>-0.465</td><td> $2 . 6 4 \times 1 0 ^ { - 4 }$ </td><td>0</td></tr><tr><td>UCS</td><td>Mutation</td><td>TPTE</td><td>1.546</td><td>-0.339</td><td> $9 . 8 8 \times 1 0 ^ { - 3 }$ </td><td>0</td></tr><tr><td>UCS</td><td>SCNV</td><td>16q23.1</td><td>0.274</td><td>-0.297</td><td> $2 . 6 2 \times 1 0 ^ { - 2 }$ </td><td>0</td></tr><tr><td>UCS</td><td>RNASeq</td><td>UFD1L</td><td>0.443</td><td>0.492</td><td> $1 . 0 1 \times 1 0 ^ { - 4 }$ </td><td>0</td></tr><tr><td>UCS</td><td>RPPA</td><td>ARHI</td><td>0.385</td><td>0.409</td><td> $3 . 9 2 \times 1 0 ^ { - 3 }$ </td><td>0</td></tr></table>

Cross-omics inspection did not reveal a robust same-gene chain spanning DNA, RNA, and protein layers. In UCEC, the most consistent support came from multiple FDR-supported methylation, SCNV, and RPPA candidates rather than from a single linear molecular cascade. In UCS, two same-gene methylation–RNA candidates, MGC87042 and ROR2, appeared among the top 20 perturbation hits across layers, but neither had feature-level FDR support. Several cross-layer correlations were observed among top-ranked candidates, especially between methylation or mutation and RNASeq features in UCS, and between RNASeq and RPPA features in UCEC. Because these edges are based on bulk Spearman correlations, we interpret them only as candidate inter-layer relationships.

Taken together, virtual perturbation provides a ranked set of molecular hypotheses linked to SpaTIE-derived state organization. The most statistically supported evidence in the current data comes from UCEC methylation, SCNV, and RPPA features. UCS perturbation results and all features without FDR support should be considered exploratory and require independent validation or experimental follow-up.

## 7 Detailed Pretraining Methodology for SpaTIE

## 7.1 Model initialization

We selected UNI [32] as the initialization for our domain-specific pretraining. UNI is a generalpurpose computational pathology foundation model built upon the Vision Transformer with Large architecture and $1 6 \times 1 6$ patch size (ViT-L/16). It comprises approximately 303 million parameters and was pretrained on the Mass-100K dataset, which contains over 100 million tissue patches extracted from 100,426 diagnostic WSIs across 20 major organ types, encompassing normal tissue, cancerous tissue, and other pathological conditions [32]. The pretraining of UNI employed the DINOv2 selfsupervised learning framework [30], which leverages a student-teacher distillation paradigm to learn robust visual representations without manual annotations. By initializing from UNI, SpaTIE inherits general histopathological feature extraction capabilities that have been demonstrated to generalize across 33 diverse anatomic pathology tasks.

## 7.2 Vision Transformer Architecture and Self-Attention Mechanism

SpaTIE retains the ViT-L/16 architecture from UNI. The model processes an input image $\textbf { x } \in$ $\mathbb { R } ^ { H \times W \times C }$ by dividing it into a sequence of flattened non-overlapping patches $\mathbf { x } _ { p } \in \mathbb { R } ^ { N \times ( P ^ { 2 } \cdot C ) }$ , where $P = 1 6$ is the patch size, $C = 3$ denotes the RGB channels, and $N = H W { \bar { / } } P ^ { 2 }$ is the total number of patches. Each patch is linearly projected via a trainable matrix $\mathbf { E } \in \mathbb { R } ^ { ( P ^ { 2 } \cdot C ) \times D }$ to obtain patch embeddings, where $D = 1 0 2 4$ is the embedding dimension for ViT-L. A learnable class token $\mathbf { x } _ { \mathrm { c l s } }$ and sinusoidal positional embeddings $\mathbf { E } _ { \mathrm { p o s } } \in \mathbb { R } ^ { ( \sim 1 ) \times D }$ are added to form the input sequence $\begin{array} { r } { \mathbf { z } _ { 0 } = [ \mathbf { x } _ { \mathrm { c l s } } ; \mathbf { x } _ { p } ^ { 1 } \mathbf { E } ; \ldots ; \mathbf { x } _ { p } ^ { N } \mathbf { E } ] + \mathbf { E } _ { \mathrm { p o s } } , } \end{array}$

The sequence $\mathbf { z } _ { 0 }$ is fed through $L = 2 4$ identical transformer encoder blocks. Each block l consists of Layer Normalization (LN), multi-head self-attention (MSA), and a feed-forward network (FFN), with residual connections:

$$
\begin{array} { r } { \mathbf { z } _ { l } ^ { \prime } = \mathrm { M S A } ( \mathrm { L N } ( \mathbf { z } _ { l - 1 } ) ) + \mathbf { z } _ { l - 1 } , } \end{array}\tag{S1}
$$

$$
\mathbf z _ { l } = \mathrm { F F N } ( \mathrm { L N } ( \mathbf z _ { l } ^ { \prime } ) ) + \mathbf z _ { l } ^ { \prime } .\tag{S2}
$$

The multi-head self-attention mechanism is the core computational primitive of the transformer. For an input sequence $\mathbf { Z } \in \mathbb { R } ^ { ( N + 1 ) \times D }$ , the model first projects it into query, key, and value matrices:

$$
\mathbf { Q } = \mathbf { Z } \mathbf { W } ^ { Q } , \mathbf { K } = \mathbf { Z } \mathbf { W } ^ { K } , \mathbf { V } = \mathbf { Z } \mathbf { W } ^ { V } ,\tag{S3}
$$

where $\mathbf { W } ^ { Q } , \mathbf { W } ^ { K } , \mathbf { W } ^ { V } \in \mathbb { R } ^ { D \times D }$ are learnable projection matrices. For multi-head attention with h heads, these projections are partitioned into h groups, each with dimension $d _ { k } = D / h$ . The scaled dot-product attention for each head i is computed as:

$$
{ \mathrm { h e a d } } _ { i } = { \mathrm { A t t e n t i o n } } ( \mathbf { Q } _ { i } , \mathbf { K } _ { i } , \mathbf { V } _ { i } ) = { \mathrm { s o f t m a x } } \left( { \frac { \mathbf { Q } _ { i } \mathbf { K } _ { i } ^ { \top } } { \sqrt { d _ { k } } } } \right) \mathbf { V } _ { i } .\tag{S4}
$$

The outputs of all heads are concatenated and linearly projected:

$$
\mathrm { M S A } ( \mathbf { Z } ) = \mathrm { C o n c a t } ( \mathrm { h e a d } _ { 1 } , \dots , \mathrm { h e a d } _ { h } ) \mathbf { W } ^ { O } ,\tag{S5}
$$

where $\mathbf { W } ^ { O } \in \mathbb { R } ^ { D \times D }$ . The self-attention mechanism enables each patch token to attend to all other tokens, capturing long-range spatial dependencies and contextual relationships critical for histopathological pattern recognition.

## 7.3 Parameter-Eficient Fine-Tuning with LoRA

To adapt the pretrained UNI model to the uterine pathology domain without catastrophic forgetting of general histopathological knowledge, we employed Low-Rank Adaptation (LoRA) [31], a parametereficient fine-tuning (PEFT) technique. LoRA is grounded in the hypothesis that the intrinsic dimensionality of fine-tuning adaptations is low, and thus the weight updates can be efectively approximated by low-rank matrices.

For each linear layer with pretrained weight matrix $\mathbf { W } _ { 0 } \in \mathbb { R } ^ { m \times n }$ in the transformer, LoRA introduces a trainable low-rank decomposition to approximate the full fine-tuning update ∆W:

$$
\mathbf { W } = \mathbf { W } _ { 0 } + \Delta \mathbf { W } = \mathbf { W } _ { 0 } + \mathbf { B } \mathbf { A } ,\tag{S6}
$$

where $\mathbf { B } \in \mathbb { R } ^ { m \times r }$ and $\mathbf { A } \in \mathbb { R } ^ { r \times n }$ are the trainable low-rank matrices, and the rank $r \ll \operatorname* { m i n } ( m , n )$ is a hyperparameter. During training, the original pretrained weights $\mathbf { W } _ { 0 }$ remain frozen, and only A and B are updated via gradient descent. This reduces the number of trainable parameters from mn to $r ( m + n )$ , achieving substantial memory and computational savings. Following standard practice [31], we applied LoRA to the query $( \mathbf { W } ^ { Q } )$ and value $( \mathbf { W } ^ { V } )$ projection matrices in all $L = 2 4$ self-attention layers. The matrices A were initialized with random Gaussian values and B with zeros, such that $\Delta \mathbf { W } = \mathbf { B } \mathbf { A } = \mathbf { 0 }$ at initialization, ensuring that the adaptation starts from the pretrained UNI representation.

## 7.4 DINOv2 Self-Supervised Pretraining Objective

We continued pretraining the LoRA-adapted UNI model using the DINOv2 self-supervised learning framework [30], which extends the original DINO approach with improved training stability and representation quality. DINOv2 employs a student-teacher self-distillation architecture where the student network processes augmented views of an input image and learns to predict the output distribution of a momentum-updated teacher network.

Teacher-Student Architecture. Let $\theta _ { s }$ and $\theta _ { t }$ denote the parameters of the student and teacher networks, respectively. The student network is updated via backpropagation, while the teacher parameters are computed as an exponential moving average (EMA) of the student parameters:

$$
\theta _ { t }  \lambda \theta _ { t } + ( 1 - \lambda ) \theta _ { s } ,\tag{S7}
$$

where $\lambda \in [ 0 , 1 )$ is the EMA decay rate. Both networks share the same $\mathrm { V i T - L } / 1 6$ architecture with LoRA-adapted attention layers. During training, multiple augmented views are generated from each image: a set of global views (covering a large portion of the image) and local views (covering smaller regions). Only the global views are passed through the teacher network, while all views are processed by the student network.

DINO Loss (Image-Level Self-Distillation). The DINO loss enforces consistency between the student and teacher outputs at the image level using the class token (cls) representations. Let $g _ { s } ( \mathbf { x } )$ and $g _ { t } ( \mathbf { x } )$ denote the output logits of the student and teacher, respectively, obtained by projecting the cls token features through a learnable projection head followed by a softmax normalization with temperature scaling. The DINO loss is formulated as the cross-entropy between the teacher’s target distribution and the student’s predicted distribution:

$$
\mathcal { L } _ { \mathrm { D I N O } } = - \sum _ { c = 1 } ^ { C } p _ { t } ^ { c } ( \mathbf { x } ) \log p _ { s } ^ { c } ( \mathbf { x } ) ,\tag{S8}
$$

where $\begin{array} { r } { p _ { s } ^ { c } ( \mathbf { x } ) = \frac { \exp ( g _ { s } ^ { c } ( \mathbf { x } ) / \tau _ { s } ) } { \sum _ { j = 1 } ^ { C } \exp ( g _ { s } ^ { j } ( \mathbf { x } ) / \tau _ { s } ) } } \end{array}$ and $\begin{array} { r } { p _ { t } ^ { c } ( \mathbf { x } ) = \frac { \exp ( g _ { t } ^ { c } ( \mathbf { x } ) / \tau _ { t } ) } { \sum _ { i = 1 } ^ { C } \exp ( g _ { t } ^ { j } ( \mathbf { x } ) / \tau _ { t } ) } } \end{array}$ are the probability distributions of the student and teacher, respectively, C is the dimension of the projection head output, and $\tau _ { s } , \ \tau _ { t }$ are temperature parameters. The teacher outputs are typically centered with a running mean to prevent collapse.

iBOT Loss (Patch-Level Masked Modeling). DINOv2 integrates the iBOT objective to enhance spatial consistency and local feature learning. Random patches in the student input are masked, and the student is trained to predict the teacher’s output distribution for these masked patches using their unmasked context. Let M denote the set of masked patch indices. For each masked patch $i \in \mathcal { M }$ , the iBOT loss is:

$$
\mathcal { L } _ { \mathrm { i B O T } } = - \sum _ { i \in \mathcal { M } } \sum _ { c = 1 } ^ { C } p _ { t } ^ { c , i } ( \mathbf { x } ) \log p _ { s } ^ { c , i } ( \mathbf { x } ) ,\tag{S9}
$$

where $p _ { s } ^ { c , i }$ and $p _ { t } ^ { c , i }$ are the student and teacher probability distributions for patch i, respectively.

KoLeo Loss. DINOv2 additionally employs the KoLeo divergence loss to encourage uniformity in the representation space and prevent representation collapse. For a batch of B samples with normalized features $\{ \mathbf { f } _ { i } \} _ { i = 1 } ^ { B }$ , the KoLeo loss is:

$$
\mathcal { L } _ { \mathrm { K o L e o } } = - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \log \operatorname* { m i n } _ { j \neq i } \| { \bf f } _ { i } - { \bf f } _ { j } \| _ { 2 } .\tag{S10}
$$

Total Training Objective. The overall DINOv2 loss combines the above components:

$$
\mathcal { L } _ { \mathrm { D I N O v 2 } } = \mathcal { L } _ { \mathrm { D I N O } } + \lambda _ { \mathrm { i B O T } } \mathcal { L } _ { \mathrm { i B O T } } + \lambda _ { \mathrm { K o L e o } } \mathcal { L } _ { \mathrm { K o L e o } } ,\tag{S11}
$$

where $\lambda _ { \mathrm { i B O T } }$ and $\lambda _ { \mathrm { K o L e o } }$ are weighting hyperparameters. Through continued pretraining with this objective on uterine pathology data, the LoRA-adapted parameters progressively shift the model’s feature space toward uterine-specific morphological patterns while the frozen UNI backbone preserves general histopathological priors, yielding the uterine pathology foundation model SpaTIE.

## 8 Detailed Methodology for Pseudo-temporal Mapping

## 8.1 Feature Sampling and Dimensionality Reduction

Pseudo-temporal analysis was conducted on slide-level features extracted by SpaTIE. To ensure computational tractability while preserving population diversity, we randomly subsampled 1% of the full cohort without replacement, yielding a representative subset of slides. Each slide was represented by a D = 1024-dimensional feature vector $\mathbf { f } _ { i } \in \mathbb { R } ^ { 1 0 2 4 }$ obtained from the frozen SpaTIE backbone. The feature matrix $\mathbf { X } \in \mathbb { R } ^ { N \times 1 0 2 4 }$ was assembled by stacking all subsampled slide features, where N is the number of selected slides.

Due to the high dimensionality and potential redundancy in the feature space, we applied principal component analysis (PCA) as a preprocessing step. PCA seeks an orthogonal linear transformation that maps the data to a new coordinate system such that the greatest variance lies on the first coordinate (principal component), the second greatest variance on the second coordinate, and so on. Formally, given the centered data matrix $\tilde { \mathbf { X } } = \mathbf { X } - \bar { \mathbf { X } }$ , we solve the eigenvalue decomposition of the covariance matrix:

$$
\mathbf { C } = \frac { 1 } { N - 1 } \tilde { \mathbf { X } } ^ { \top } \tilde { \mathbf { X } } = \mathbf { U } \mathbf { A } \mathbf { U } ^ { \top } ,\tag{S12}
$$

where $\textbf { U } \in \ \mathbb { R } ^ { 1 0 2 4 \times 1 0 2 4 }$ contains the orthonormal eigenvectors and $\textbf { \Lambda } = \mathrm { d i a g } ( \lambda _ { 1 } , \ldots , \lambda _ { 1 0 2 4 } )$ with $\lambda _ { 1 } \geq \lambda _ { 2 } \geq \cdot \cdot \cdot \geq \lambda _ { 1 0 2 4 } \geq 0$ are the eigenvalues representing the variance explained by each component. Rather than fixing the number of components a priori, we retained the minimal set of principal components explaining 90% of the cumulative variance:

$$
d = \operatorname* { m i n } \left\{ k : \frac { \sum _ { i = 1 } ^ { k } \lambda _ { i } } { \sum _ { i = 1 } ^ { 1 0 2 4 } \lambda _ { i } } \geq 0 . 9 \right\} .\tag{S13}
$$

The projected data $\mathbf { X } _ { \mathrm { p c a } } = \tilde { \mathbf { X } } \mathbf { U } _ { : , 1 : d } \in \mathbb { R } ^ { N \times d }$ served as the input for subsequent graph-based trajectory inference.

## 8.2 K-Nearest Neighbor Graph Construction

To capture the local manifold structure of the histopathological state space, we constructed an undirected k-nearest neighbor (k-NN) graph. For each slide i, we identified its $k = 1 0$ nearest neighbors $\mathcal { N } _ { k } ( i )$ based on Euclidean distance in the PCA-reduced space. The adjacency matrix $\mathbf { A } \in \mathbb { R } ^ { N \times N }$ was defined as:

$$
A _ { i j } = { \left\{ \begin{array} { l l } { 1 } & { { \mathrm { i f ~ } } j \in { \mathcal { N } } _ { k } ( i ) { \mathrm { ~ o r ~ } } i \in { \mathcal { N } } _ { k } ( j ) , } \\ { 0 } & { { \mathrm { o t h e r w i s e . } } } \end{array} \right. }\tag{S14}
$$

The degree matrix $\mathbf { D } = \mathrm { d i a g } ( d _ { 1 } , \dots , d _ { N } )$ with $\begin{array} { r } { d _ { i } = \sum _ { j } A _ { i j } } \end{array}$ was computed for subsequent graph operations.

## 8.3 Leiden Community Detection

To identify biologically coherent histopathological subgroups, we applied the Leiden algorithm for community detection on the k-NN graph. The Leiden algorithm optimizes the modularity-like quality function known as the Constant Potts Model (CPM), which for resolution parameter γ is defined as:

$$
{ \mathcal { H } } ( \sigma ) = - \sum _ { i j } { \left( A _ { i j } - \gamma \right) } \delta ( \sigma _ { i } , \sigma _ { j } ) ,\tag{S15}
$$

where $\sigma _ { i } \in \{ 1 , \ldots , K \}$ denotes the cluster assignment of node i, and $\delta ( \cdot , \cdot )$ is the Kronecker delta. The algorithm proceeds through three phases: (1) local moving of nodes to maximize $\mathcal { H } ( \sigma )$ ; (2) refinement of the partition to guarantee that all clusters are connected and well-separated; and (3) aggregation of the network based on the refined partition. We set $\gamma = 1 . 0$ to obtain a moderate number of clusters balancing granularity and statistical power. The resulting partition $\mathcal { P } = \{ C _ { 1 } , \ldots , C _ { K } \}$ was stored as categorical annotations in the AnnData object.

## 8.4 Partition-Based Graph Abstraction (PAGA)

To infer a coarse-grained representation of the developmental trajectory while preserving biological meaningfulness, we employed PAGA. PAGA constructs a graph where each node represents a Leiden cluster and edges encode the statistical significance of inter-cluster connectivity. For two clusters $C _ { \alpha }$ and $C _ { \beta }$ , the connectivity score is computed as:

$$
w _ { \alpha \beta } = \frac { \sum _ { i \in C _ { \alpha } , j \in C _ { \beta } } A _ { i j } + \sum _ { i \in C _ { \beta } , j \in C _ { \alpha } } A _ { i j } } { \lvert C _ { \alpha } \rvert \cdot \lvert C _ { \beta } \rvert } ,\tag{S16}
$$

normalized by the expected connectivity under a random null model. The statistical significance is assessed via a one-sample test comparing observed to expected connectivities. The PAGA graph $\mathcal { G } _ { \mathrm { P A G A } } = ( \mathcal { V } _ { \mathrm { P A G A } } , \mathcal { E } _ { \mathrm { P A G A } } )$ provides a topologically meaningful backbone for trajectory inference, with edges weighted by $- \log _ { 1 0 } ( p _ { \alpha \beta } )$ , where $p _ { \alpha \beta }$ is the p-value of the connectivity test.

## 8.5 Difusion Pseudotime (DPT) Computation

To establish a quantitative ordering of histopathological progression, we computed difusion pseudotime using the random-walk normalized graph Laplacian. The random-walk Laplacian is defined as:

$$
{ \bf L } _ { \mathrm { r w } } = { \bf I } - { \bf D } ^ { - 1 } { \bf A } ,\tag{S17}
$$

where I is the identity matrix and $\mathbf { D } ^ { - 1 } \mathbf { A }$ is the random-walk transition matrix. The eigendecomposi tion of $\mathbf { L } _ { \mathrm { r w } }$ yields:

$$
\mathbf { L } _ { \mathrm { r w } } \psi _ { i } = \lambda _ { i } \psi _ { i } , \quad i = 1 , \ldots , N ,\tag{S18}
$$

with eigenvalues $0 = \lambda _ { 1 } < \lambda _ { 2 } \leq \cdot \cdot \cdot \leq \lambda _ { N }$ and corresponding eigenvectors $\psi _ { i }$ . The eigenvector $\psi _ { 1 }$ corresponding to $\lambda _ { 1 } = 0$ represents the stationary distribution and is excluded from the difusion map. The difusion map coordinates for slide $j$ are given by the top $n _ { \mathrm { d c s } } = 1 0$ non-trivial eigenvectors:

$$
\Psi _ { j } = \left[ \psi _ { 2 } ( j ) , \psi _ { 3 } ( j ) , \ldots , \psi _ { 1 1 } ( j ) \right] ^ { \top } \in \mathbb { R } ^ { 1 0 } .\tag{S19}
$$

To orient the trajectory, we designated a reference slide with minimal histopathological abnormality (typically a low-grade or normal-adjacent sample) as the root state $\mathbf { x } _ { \mathrm { r o o t } }$ and set its index in the AnnData object via adata.uns[‘iroot’]. The difusion pseudotime of slide $j$ relative to the root is then computed as the Euclidean distance in the difusion map space:

$$
\mathrm { D P T } ( j ) = \| \Psi _ { j } - \Psi _ { \mathrm { r o o t } } \| _ { 2 } = \sqrt { \sum _ { k = 2 } ^ { 1 1 } \left( \psi _ { k } ( j ) - \psi _ { k } ( \mathrm { r o o t } ) \right) ^ { 2 } } .\tag{S20}
$$

This metric quantifies the expected time for a random walk to travel from the root to slide $j$ on the data manifold, providing a biologically interpretable measure of progression distance. Slides with higher DPT values are inferred to represent more advanced histopathological states relative to the designated root.

## 9 Evaluation metrics for Static Feature Space Validation

To complement the UMAP-based visualization, we computed three quantitative metrics that capture distinct geometric and information-theoretic properties of the learned feature spaces.

## 9.1 Manifold Structure Quality

Manifold structure quality measures the intrinsic dimensionality of the data distribution in the feature space, providing insight into how compactly and eficiently the model encodes histopathological information. We estimate this via the cumulative explained variance ratio of principal component analysis. Given centered features F<sup>˜</sup> , the covariance matrix is $\begin{array} { r } { \mathbf { C } = \frac { 1 } { n - 1 } \tilde { \mathbf { F } } ^ { \top } \tilde { \mathbf { F } } } \end{array}$ , with eigendecomposition $\mathbf { C } = \mathbf { U } \mathbf { A } \mathbf { U } ^ { \top }$ where $\pmb { \Lambda } = \mathrm { d i a g } ( \lambda _ { 1 } , \ldots , \lambda _ { d } )$ and $\lambda _ { 1 } \geq \lambda _ { 2 } \geq \cdot \cdot \cdot \geq \lambda _ { d } \geq 0$ . The explained variance ratio of the k-th component is $r _ { k } = \lambda _ { k } / \textstyle \sum _ { i = 1 } ^ { d } \lambda _ { i }$ . The intrinsic dimensionality is defined as the minimal number of components required to explain 90% of total variance:

$$
d _ { \mathrm { i n t r i n s i c } } = \operatorname* { m i n } \left\{ k : \sum _ { i = 1 } ^ { k } r _ { i } \geq 0 . 9 \right\} .\tag{S21}
$$

A lower intrinsic dimensionality indicates that the model has learned a more compact, structured representation where most information is concentrated in a few dominant directions, whereas a higher value suggests dispersed, less organized feature encoding.

## 9.2 Representation Entropy

Representation entropy quantifies the uniformity of variance distribution across the principal components of the feature space, serving as a measure of dimensional utilization. Let $\mathbf { r } = [ r _ { 1 } , \ldots , r _ { d } ] ^ { \top }$ denote the vector of explained variance ratios, which can be interpreted as a discrete probability distribution over the principal axes. The representation entropy is defined as the Shannon entropy of this distribution:

$$
H _ { \mathrm { r e p } } = - \sum _ { i = 1 } ^ { d } r _ { i } \log ( r _ { i } + \epsilon ) ,\tag{S22}
$$

where $\epsilon \ : = \ : 1 0 ^ { - 1 2 }$ is a small constant to ensure numerical stability. The entropy is bounded as $0 \leq H _ { \mathrm { r e p } } \leq \log ( d )$ , with the maximum achieved when variance is uniformly distributed across all components $( r _ { i } = 1 / d$ for all i). High entropy indicates that the model utilizes a broad spectrum of feature dimensions, potentially capturing diverse histomorphological patterns, whereas low entropy suggests that information is concentrated in a few dominant directions, which may limit representational richness.

## 10 Evaluation metrics for Pseudo-temporal Performance Evaluation

To rigorously assess the quality of pseudo-temporal trajectories extracted from diferent feature spaces, we designed five complementary slide-level metrics. Each metric targets a distinct aspect of biological plausibility: local coherence, robustness, spatial organization, morphological scaling, and feature regularity along the inferred progression axis. All metrics were computed on a per-slide basis using 1% randomly subsampled patches and subsequently aggregated across the cohort.

## 10.1 Trajectory Quality (Local Neighbor Consistency)

Trajectory quality measures whether patches that are close in the learned feature space exhibit similar pseudo-temporal values, indicating that the trajectory respects the intrinsic geometry of the histomorphological manifold. For each patch i, we identify its k nearest neighbors $\mathcal { N } _ { k } ( i )$ in feature space using a cKDTree with Euclidean distance. The metric is defined as the inverse of the mean absolute pseudo-temporal diference between each patch and its feature-space neighbors:

$$
\mathcal { Q } _ { \mathrm { t r a j } } = \frac { 1 } { 1 + \frac { 1 } { N k } \sum _ { i = 1 } ^ { N } \sum _ { j \in \mathcal { N } _ { k } ( i ) } \left| t _ { i } - t _ { j } \right| } ,\tag{S23}
$$

where $t _ { i }$ denotes the pseudo-time of patch i, N is the number of patches, and $k = 5$ is the neighborhood size. Values approaching 1 indicate that the pseudo-temporal ordering is smooth and coherent with respect to the feature manifold structure.

## 10.2 Stability (Perturbation Robustness)

Stability quantifies the robustness of the pseudo-temporal ordering against small perturbations, a critical property for reliable downstream analysis. We add Gaussian noise $\mathbf { \epsilon } \sim \mathcal { N } ( \mathbf { 0 } , 1 0 ^ { - 3 } \mathbf { I } )$ to the pseudo-time vector and recompute the ordering. The stability metric is defined as the Spearman rank correlation coeficient between the original and perturbed pseudo-times:

$$
S _ { \mathrm { s t a b } } = \rho _ { \mathrm { S p e a r m a n } } ( \mathbf { t } , \mathbf { t } + \epsilon ) = \frac { \mathrm { C o v } ( R ( \mathbf { t } ) , R ( \mathbf { t } + \epsilon ) ) } { \sigma _ { R ( \mathbf { t } ) } \cdot \sigma _ { R ( \mathbf { t } + \epsilon ) } } ,\tag{S24}
$$

where $R ( \cdot )$ denotes the rank transformation. High stability $( S _ { \mathrm { s t a b } } \approx 1 )$ indicates that the pseudotemporal landscape is structurally robust and not dominated by numerical instabilities or overfitting artifacts.

## 10.3 Spatial Consistency

Spatial consistency evaluates whether the pseudo-temporal progression respects the physical spatial organization of tissue, reflecting the biological reality that histopathological changes often exhibit spatial propagation patterns $\left( \mathrm { e . g . } \right.$ ., from tumor center to invasive margin). For each patch i, we identify its k nearest neighbors $\mathcal { N } _ { k } ^ { \mathrm { s p a t } } ( i )$ in physical coordinate space using a cKDTree. The metric is computed as:

$$
\mathcal C _ { \mathrm { s p a t } } = \frac { 1 } { 1 + \frac { 1 } { N k } \sum _ { i = 1 } ^ { N } \sum _ { j \in \mathcal N _ { k } ^ { \mathrm { s p a t } } ( i ) } \left| t _ { i } - t _ { j } \right| } .\tag{S25}
$$

High spatial consistency suggests that the inferred trajectory captures genuine spatial propagation of pathological states rather than random noise.

## 10.4 Morphological Continuity

Morphological continuity tests whether the rate of morphological change (measured by feature-space distance) is proportional to the rate of pseudo-temporal change, a fundamental requirement for biologically meaningful trajectories. For each patch i and its feature-space neighbor $j \in \mathcal { N } _ { k } ( i )$ , we compute the ratio of pseudo-temporal distance to morphological distance:

$$
\mathcal { R } _ { i j } = \frac { \left| t _ { i } - t _ { j } \right| } { \| \mathbf { f } _ { i } - \mathbf { f } _ { j } \| _ { 2 } + 1 0 ^ { - 6 } } .\tag{S26}
$$

The morphological continuity metric is then defined as:

$$
\mathcal { C } _ { \mathrm { m o r p h } } = \frac { 1 } { 1 + \frac { 1 } { N k } \sum _ { i = 1 } ^ { N } \sum _ { j \in \mathcal { N } _ { k } ( i ) } \mathcal { R } _ { i j } } .\tag{S27}
$$

This metric penalizes abrupt pseudo-temporal transitions between morphologically similar patches (small denominator, large ratio) or excessively slow progression across large morphological gaps. Values near 1 indicate that the trajectory advances at a pace commensurate with observable morphological changes.

## 10.5 Feature Smoothness

Feature smoothness measures the total variation of feature vectors along the pseudo-temporal ordering, assessing whether the feature manifold exhibits gradual transitions rather than discrete jumps when traversed according to inferred progression time. We sort patches by pseudo-time to obtain the ordered sequence $\{ \mathbf { f } _ { ( 1 ) } , \mathbf { f } _ { ( 2 ) } , \ldots , \mathbf { f } _ { ( N ) } \}$ , where (i) denotes the index of the patch with the i-th smallest pseudotime. The metric is computed as:

$$
\mathcal { F } _ { \mathrm { s m o o t h } } = \frac { 1 } { 1 + \frac { 1 } { N - 1 } \sum _ { i = 1 } ^ { N - 1 } \| \mathbf { f } _ { ( i + 1 ) } - \mathbf { f } _ { ( i ) } \| _ { 2 } } .\tag{S28}
$$

High feature smoothness indicates that the pseudo-temporal path traverses regions of the feature space where morphological characteristics evolve gradually, consistent with continuous biological processes such as tumor progression or tissue diferentiation.