# CytoFormer: A Molecularly Supervised Cell Foundation Model for Histopathology Cell Classification

Jialu Yao<sup>1,2,∗</sup>, Songhao Li<sup>1,2,∗</sup>, Alina Yu<sup>3</sup>, Zhi Huang<sup>1,4,#</sup>

<sup>1</sup> Department of Pathology and Laboratory Medicine, University of Pennsylvania, Philadelphia, PA, USA

<sup>2</sup> Department of Electrical and Systems Engineering, University of Pennsylvania, Philadelphia, PA, USA

<sup>3</sup> Germantown Friends School, Philadelphia, PA, USA

<sup>4</sup> Department of Biostatistics, Epidemiology & Informatics, University of Pennsylvania, Philadelphia, PA, USA

\* Equal contribution

<sup>#</sup> Correspondence: Zhi Huang (zhi.huang@pennmedicine.upenn.edu)

## Abstract

Identifying cell types directly from routine haematoxylin and eosin (H&E) histology would enable single-cell analysis at scale, but training such models has relied on manual pathologist annotations, which are slow, expensive and unreliable for many cell types. We instead supervise morphology with molecules. Imaging-based spatial transcriptomics profiles individual cells in situ on a section that can afterwards be stained with H&E, so that molecular identity and morphology are observed for the same physical cell. We assembled 81 such paired Xenium sections spanning 16 organs, derived per-cell labels by clustering, marker-gene annotation, organ-wise human review and quality control, and mapped them onto the cell types commonly reported in each organ. This yielded 15.4 million cells, each with a paired H&E image patch and one of 23 cell types, on which we trained CytoFormer, a cell foundation model with a multi-task, per-organ classification head. On spatially held-out tissue CytoFormer reached an accuracy of 0.85 and a macro-F1 of 0.78 across all 16 organs, and its predictions reproduced the tissue architecture of an entire held-out section. The representation also transfers: with the encoder frozen, a linear head on CytoFormer features performed better than six pathology foundation models on four expert-annotated benchmarks, including on organs and cell types that were not part of pretraining. Finally, in an interactive active-learning setting, CytoFormer’s embeddings are markedly more label-efficient than existing pathology foundation models, detecting normal epithelium amid look-alike tumour with an F1 of 0.82 from only a few annotations and leading the strongest baseline by 0.13 in F1. CytoFormer turns paired H&E and spatial transcriptomics into a reusable, label-efficient representation for cell-level analysis of routine histology. Code and model checkpoints are available at https://github.com/zhihuanglab/CytoFormer.

## 1 Introduction

Haematoxylin and eosin (H&E) staining is the basis of routine pathology diagnosis. Much of what a pathologist reads from an H&E section is cellular: which cell types are present, in what proportion, and how they are arranged. Tumour cellularity, the density of tumour-infiltrating lymphocytes, the abundance of macrophages and plasma cells, and the state of the surrounding stroma and vasculature all carry diagnostic and prognostic weight. All of them can be read from the slide. In practice they are estimated by eye and reported semi-quantitatively, which is subjective and varies between observers. Typing every nucleus turns these judgements into counts, which are reproducible and can be compared across slides and cohorts.

Progress on this task has been limited by supervision rather than by architecture. In the past, public cell-classification resources were built from manual annotation by pathologists. Such annotation is slow and expensive, and for many cell types it is also unreliable: a macrophage, a fibroblast and a poorly differentiated tumour cell can be difficult to tell apart within a single H&E nucleus. The largest expert-annotated datasets therefore contain on the order of $\dot { 1 } 0 ^ { 5 }$ nuclei and only a handful of coarse categories, as in Pan-Nuke [1], CoNSeP [2] and MoNuSAC [3], and their class definitions are not mutually consistent. Recently, pathology foundation models have advanced rapidly [4–10], and are widely used for downstream diagnostic tasks [11], but they are pretrained and evaluated on patches and whole slides. Their supervision and representations can overlook cell-level features, so repurposing them for cell classification leaves the labelling bottleneck untouched.

Spatial transcriptomics offers a way around this bottleneck. Imaging-based platforms such as Xenium measure hundreds to thousands of transcripts in situ at subcellular resolution and segment individual cells. The same section can then be stained with H&E and imaged [12]. Molecular identity and morphology are therefore available for the same physical cell, and the molecular channel can supervise the morphological one. Labels for H&E can therefore be obtained without a pathologist drawing them, as long as the two images are registered and the molecular calls are checked. It also scales: a single Xenium section yields $1 0 ^ { 5 }$ to 10<sup>6</sup> typed cells, far more than a manually annotated cohort.

These labels are derived through a relatively comprehensive pipeline. Cell identity must be inferred from unsupervised clustering and marker interpretation, which is panel-dependent and error-prone. Segmentation errors and transcript spill-over generate doublets and ambient contamination [13]. Artefact clusters are often tight and internally consistent, so they survive naive quality filters. Only a subset of cells can be confidently mapped onto a morphology after registration. A label set also has to be chosen, since the clusters found in one section do not by themselves define a taxonomy that holds across organs. Histology and spatial transcriptomics have been linked before, but existing models work at the level of spots or patches and predict or align gene expression [14, 15], rather than using the transcriptome as supervision for the identity of individual cells.

Here we present CytoFormer, a cell foundation model learned from paired H&E and spatial transcriptomics. We curated 81 Xenium sections spanning 16 organs. Per-cell reference labels were derived by clustering, marker-based cluster annotation, cluster-level human quality control and a per-cell confidence gate, and were then mapped onto the cell types commonly reported in each organ, giving 23 classes. Registering each section to its own H&E image gave 15.4 million cells with a paired 56 µm H&E patch. On these we trained a cell encoder with a multi-task, per-organ classification head. CytoFormer classifies cell types on held-out sections across all 16 organs, and its representation transfers. With the encoder frozen, a linear head on CytoFormer features performs better than six pathology foundation models on four expertannotated benchmarks. The advantage is largest where supervision is scarce, and in an interactive activelearning setting CytoFormer detects normal epithelium amid look-alike tumour from only a few hundred annotations.

## 2 Results

## 2.1 Curate a pan-tissue single-cell H&E dataset from paired spatial transcriptomics

To train a cell-level model without manual annotation, we built a single-cell H&E dataset in which every cell is typed by its own transcriptome. Imaging-based spatial transcriptomics profiles single cells in situ, and the same section can afterwards be stained with H&E and imaged. Each physical cell is therefore observed twice: as morphology on the H&E image (Figure 1a), and as an expression profile in the Xenium cell map (Figure 1b). We assembled public human Xenium experiments that carry a paired H&E image of the same section, drawn from the 10x Genomics data portal and from HEST-1k [16]. We then assigned a cell type to every cell in three steps: unsupervised clustering of the expression profiles, annotation of each cluster from its marker genes, and organ-wise human review of every cluster (Figure 1c). Each type was then transferred to the matched H&E morphology. After quality control, per-cell confidence filtering and registration, the dataset contains 15,422,352 cells from 81 sections and 16 organs. Every cell carries a 56 µm H&E patch, a cell type and an organ label. Of the sections, 54 come from the 10x Genomics portal and

![](images/65fa5a93132047e53829a9573e83b9369d4ec5dc3d1e6f3b8d14cb36a4ffd570.jpg)

![](images/67e5269efcdb2975e6f2f7d02079cbd4d391bde70aa2423dc73ae77f84a5b63d.jpg)

![](images/ed286a39c4900a0741f268dc3223d8bb8861e3f8be49fc9e739dc00bf0e66026.jpg)

![](images/0acb9eb6b66cd445d9487e779db9638a576d90e9c2afc18c589dee6cf2da083c.jpg)

h Training  
![](images/5ad4ed41af3014ad8204fcdb4b90f9a060898a985adbab5b2fc64ed66467c8c6.jpg)

i Inference  
![](images/93eadef6865f9862524b413eaab2944854c985e0ea811c73df000c44affeaa9f.jpg)

Figure 1: A pan-tissue single-cell H&E dataset built from paired spatial transcriptomics. (a) H&E image of a tissue section that was also profiled with Xenium. (b) Xenium cell map of the same section. (c) Cell types are derived by clustering, marker-gene annotation and human review, and transferred to the paired H&E morphology. (d) Cell types recovered in each organ, coloured and labelled by the number of cells. (e) Cells per organ, split into training (dark) and test (light) cells. (f) Cells per cell type, split into training and test cells; both axes are logarithmic. (g) Overall training and test split. (h) During training, a 56 × 56 µm crop around each nucleus is encoded by CytoFormer and classified by a per-organ head. (i) At inference the encoder is frozen and the prediction is made by the head of the specified organ.

contribute 12,078,610 cells (78.3%); the other 27 come from HEST-1k and contribute 3,343,742 cells (21.7%).

The cell types split into common classes and organ-specific classes (Figure 1d). Seven common classes are shared by most organs: Tumour, Stroma, Epithelium, Lymphocyte, Endothelium, Macrophage and Plasma cell. Together they cover 97.5% of the dataset. Endothelium is the only class found in all 16 organs. Stroma appears in 15 organs, and Lymphocyte and Macrophage in 13 each. The other 16 classes are rare, and each appears in a single tissue: Hepatocyte and Cholangiocyte in liver, Acinar, Ductal and Islet in pancreas, Renal tubule in kidney, Microglia in brain, Squamous epithelium in lymph node, Chondrocyte in lung, Cardiomyocyte in heart, Erythroid, Neutrophil and Myeloid progenitor in bone marrow, Bone cell in bone, Nerve in prostate, and melanocytic in skin. The cell types were chosen per organ rather than globally: for each tissue we used the types that are commonly reported in it. Each organ therefore has its own list, with a median of seven cell types and up to nine in pancreas.

The dataset covers 16 organs: breast, lung, ovary, lymph node, colon, tonsil, cervix, brain, kidney, pancreas, liver, skin, prostate, bone marrow, heart and bone (Figure 1e). Breast is the largest, with 5,546,376 cells (36.0%) from 23 sections. Lung is the second largest, with 3,592,765 cells (23.3%) from 28 sections. Together they hold 59.3% of all cells, and they are the only two organs with many comparable sections. The other 14 organs are each represented by one to five sections, and range from 1,405,609 cells in ovary to 6,928 cells in bone.

The cell counts are long-tailed and span more than three orders of magnitude (Figure 1f). Tumour is the largest class, with 4,408,081 cells (28.6%). Stroma is the second largest, with 3,139,069 cells (20.4%). The two together make up almost half of the dataset. The remaining common classes hold between 329,965 cells (Plasma cell) and 2,141,342 cells (Epithelium). The organ-specific classes are much smaller, from 114,733 Hepatocytes down to 1,042 Nerve cells and 945 melanocytic cells. This imbalance comes from sampling whole tissue sections. We handle it during training rather than by resampling the dataset.

Neighbouring cells share tissue context, so we split training and test cells by space rather than at random (Figure 1g). For breast and lung we held out whole sections, 7 of 23 and 7 of 28. The other 14 organs have only one to five sections each. For these we cut every section into spatial blocks and held out dispersed blocks. We then discarded a 112 µm guard band along each train–test boundary, so that no test cell shares a field of view with a training cell. This gives 12,111,769 training cells (79%), 2,944,356 test cells (19%) and 366,227 guard-band cells (2.4%). Every cell type is present in both splits.

## 2.2 Train a cell foundation model across 16 organs

CytoFormer classifies one cell at a time. For every nucleus we crop a 56 × 56 µm field of view centred on its centroid and resize it to 224 × 224 pixels (Figure 1h). The crop keeps the nucleus at diagnostic resolution and still shows its immediate neighbours, which is the trade-off examined in Section 2.4. The image encoder is a ViT-giant transformer that outputs a 1536-dimensional embedding per cell. It is initialised from UNI2-h [4] and fine-tuned end-to-end, and its output is layer-normalised before classification.

A cell type does not look the same in every organ. A macrophage in lung alveoli, a Kupffer cell in liver sinusoids and a tumour-associated macrophage in breast stroma share a lineage but differ in shape and in the tissue around them. Many cell types are also organ-specific, such as hepatocytes in liver or islet cells in pancreas. One classifier shared by all organs would have to cover every appearance at once and rule out

the types that cannot occur.

We therefore learn one linear classifier per organ on top of a single shared encoder (Figure 1h). Head sizes range from three to nine classes, and the organ identifier selects which head is applied. In a clinical setting the organ is usually known, so it is available at inference (Figure 1i). The heads are organ-specific, but the encoder is not. It is trained jointly on all 16 organs, so the 16 classifiers are not independent models but share one representation. Features that recur in every tissue, such as the small round nucleus of a lymphocyte or the flat profile of a vessel lining, are therefore learned from all organs together, and each head only has to separate the cell types of its own organ.

The encoder and all 16 heads were trained jointly on the 12,111,769 training cells, and we kept the checkpoint with the best macro-F1 on the held-out split. Training settings and hyperparameters are described in Methods.

## 2.3 CytoFormer recovers cell types and tissue architecture on held-out sections

We first evaluated CytoFormer on the spatially held-out split, which contains 2,944,356 cells from all 16 organs. The model reached an accuracy of 0.846 and a macro-F1 of 0.779. Per-class F1 varied widely (Figure 2a). Cell types with a distinctive morphology were recognised almost perfectly: Hepatocyte reached 98.1, Renal tubule 97.8, Squamous epithelium 95.7, Islet 94.8, Cholangiocyte 93.7, tumour 92.8 and Acinar 90.3. The five most abundant classes were also recovered reliably, with tumour at 92.8, Lymphocyte at 85.3, Epithelium at 83.9, Endothelium at 82.6 and Stroma at 83.0. The weakest classes were Chondrocyte (37.6), Nerve (46.4), Myeloid progenitor (46.7), Bone cell (56.8), Microglia (60.1) and Macrophage (62.4).

Accuracy was driven by morphological distinctiveness rather than by how many cells a class contributes. Several very rare classes were classified well, including melanocytic at 85.0 from only 61 test cells, Cholangiocyte at 93.7 from 406, Erythroid at 86.1 from 705 and Islet at 94.8 from 863. Conversely, two of the weakest classes are not rare at all: Macrophage has 228,718 test cells and reaches 62.4, and Chondrocyte has 12,508 and reaches 37.6. The classes that fail are those whose nuclei resemble another class in the same tissue, in particular the mononuclear cells, where macrophages and plasma cells are drawn towards lymphocytes.

Performance by organ followed the same pattern (Figure 2b). Macro-F1 was highest in organs built from a few visually distinct compartments, reaching 88.1 in pancreas, 87.9 in liver, 85.3 in kidney, 83.6 in brain and 82.9 in colon. It was lowest in the organs with the fewest cells or the most confusable stromal and immune mixtures, at 59.0 in bone, 60.8 in lung, 66.2 in cervix and 66.7 in heart. Accuracy was considerably higher than macro-F1 wherever one class dominates the organ, for example 86.0 against 67.7 in breast, 91.5 against 74.6 in prostate and 77.0 against 60.8 in lung. Per-organ and per-class accuracy, precision and recall are reported in Extended Data Figure 1.

To show what the predictions look like in practice, we applied the model to a complete breast section that was held out from training. Figure 2c shows the raw H&E, Figure 2d the spatial-transcriptomic reference and Figure 2e the CytoFormer prediction for the same region. All 124,682 cells of the section were classified, with an accuracy of 79.3%. Comparing the two maps in Figure 2d,e shows that the predicted tissue architecture matches the reference. The same ducts, the same stromal bands and the same immune aggregates appear in the same places, and the visible disagreements are at the level of individual cells rather than of structures. Figure 2f gives the confusion matrix for this same section, and shows where the model is reliable and where it is not. Tumour, Lymphocyte, Stroma and Endothelium were recovered at 94.8%, 86.4%, 80.7% and 80.2% of their true cells. Most of the error mass sits in the mononuclear compartment, with 44.6% of macrophages and 63.5% of plasma cells predicted as lymphocytes, and a further 25.8% of epithelial cells predicted as tumour. These are the distinctions that motivated merging cell types in the first place, and they remain the limiting factor rather than a diffuse loss of accuracy.

a  
![](images/ad2744946e67c5f6879db389804e54ee7d84caef7f229c5ead00c7f34c5f953a.jpg)  
b

![](images/c09da39da31662f8847f0225202bcec80e6f00742c4a10c40ae619204ed3bc2a.jpg)

c  
![](images/a7a9122bedd10c75b2bdab50deb2012e5cbfdb6032b72d4352cd098157896023.jpg)

d  
![](images/227b758c2d9cacbe022301c0523c8256824e96fe9e9c345cd41d68ed9849b466.jpg)

e  
![](images/b78ac65153deeef1f25b662562643ed15e70eea6e36736db3d48b179529ee892.jpg)

![](images/b88fad9918d5ec3674e7120ae8d56e14198d535a5077a9b96d143f2002f716ec.jpg)

g  
![](images/e564bacc93f21120055260ea1758d1c70287afe4dd171d3e4c99d231f118f8d9.jpg)

h  
![](images/ff24c3ffb499f6fb9c42d615387afa49382e5c8c2250259f740a706d7366f901.jpg)

i  
![](images/2839430185f1f08303dcb23377b78bb2950a3fa2a429db83ef9fd911bd18d2cb.jpg)  
Figure 2: CytoFormer classifies cell types on held-out H&E tissue across 16 organs. (a) F1 per cell type and (b) macro-F1 per organ on the spatially held-out test split. (c–e) A breast section held out in its entirety: (c) raw H&E, (d) the cell-type reference derived from the paired spatial transcriptomics and (e) the CytoFormer prediction. (f) Confusion matrix for that section. (g) Crop field-of-view ablation: one fixed set of cells is cropped at each field of view and resized to $2 2 4 \times 2 2 4$ pixels. (h) Macro-F1 and (i) accuracy at each crop field of view. Error bars represent 95% confidence intervals from bootstrap resampling over test cells. Statistical significance against the 56 µm setting was assessed using a paired two-sided bootstrap test: ${ } ^ { * } p \ < \ 0 . 0 5 ,$ $^ { * * } p < 0 . 0 1 , { } ^ { * * * } p < 0 . 0 0 1$ . A bracket is drawn only between the 56 µm setting and the second-best setting and gives their exact p-value; the other settings are marked with asterisks only.

## 2.4 Ablation study on the crop field of view

Cell classification on H&E requires both the nucleus itself and enough of its neighbourhood to place it in context, and at a fixed model input size these two demands are in direct conflict: a larger crop adds context but lowers the effective resolution. We tested this trade-off directly by fixing one set of cells (500k training and 100k test cells stratified over 10 organs and 17 cell types) and cropping the same cells at 28, 56, 112 and 224 µm, always resized to 224 pixels, so that the crop window is the only variable (Figure 2g).

A 56 µm crop was the best setting (macro-F1 79.1, 83.3, 82.5 and 77.1 and accuracy 86.1, 87.2, 86.6 and 83.6 for 28, 56, 112 and 224 µm; Figure 2h,i). Both metrics dropped at the two extremes. Under a paired bootstrap over the 100k shared test cells, 56 µm was better than 28 and 224 µm $( p < 0 . 0 0 1 )$ ) and better than 112 µm $( p = 0 . 0 1 0 )$ . A 28 µm crop is close to the nucleus alone and loses the tissue context that separates, for example, stroma from endothelium; a 224 µm crop contains a whole neighbourhood but at 1 µm per pixel the target nucleus is no longer resolved. A 56 µm window, corresponding to 0.25 µm per pixel at 224 pixels, keeps the nucleus at diagnostic resolution while still including its immediate neighbours, and we adopt it throughout.

## 2.5 Linear fine-tuning on public cell-classification benchmarks

We next tested how well CytoFormer performs under linear fine-tuning. The encoder was frozen and only a linear head on top of it was trained (Figure 3a). We compared it with six pathology foundation models used in the same way: UNI2-h [4], Virchow2 [10], MUSK [7], CONCH [5], PathGen [17] and PLIP [6]. The evaluation used four datasets annotated by pathologists and independent of our training data: PanNuke [1] (186,836 cells, 19 organs, 4 classes), CoNSeP [2] (23,400 cells, colon, 6 classes), MoNuSAC [3] (46,901 cells, 4 organs, 4 classes) and PUMA [18] (95,389 cells, melanoma skin, 9 classes). All backbones were evaluated on the same cells, with the same 56 µm crops and the same class-balanced linear head.

Each dataset was evaluated organ by organ, giving 25 settings in total: 19 organs in PanNuke, four in MoNuSAC, and one each in CoNSeP and PUMA. CytoFormer gave the best representation in 24 of these 25 settings, and led on every dataset. On CoNSeP, which has six classes in colon, CytoFormer had the highest F1 on all six (Figure 3b). Against the second-best method of each class, it reached 97.0 against 94.8 for UNI2-h on dysplastic malignant epithelium, 87.4 against 76.1 for Virchow2 on healthy epithelium, 81.5 against 63.3 for PathGen on fibroblasts, 79.2 against 57.6 for MUSK on inflammatory cells, 77.4 against 68.2 for PathGen on muscle and 65.9 against 32.7 for PLIP on endothelial cells, in every case with $p < 0 . 0 0 1$ . At the organ level it reached a macro-F1 of 81.4 against 59.9 for MUSK $( p < 0 . 0 0 1 )$ . On PUMA, which has nine classes in melanoma skin, CytoFormer was highest on eight of them (Figure 3c). Against the second-best method of each class, it reached 87.0 against 80.0 for Virchow2 on tumour, 78.6 against 54.0 for MUSK on lymphocytes, 68.8 against 37.5 for MUSK on neutrophils, 48.6 against 26.3 for Virchow2 on endothelium, 47.6 against 28.2 for MUSK on histiocytes, 47.6 against 31.9 for Virchow2 on plasma cells, 41.3 against 24.8 for MUSK on melanophages and 38.4 against 19.5 for MUSK on stroma, again all with $p < 0 . 0 0 1$ . The one class on which CytoFormer did not lead is epithelium, where it still ranked among the top methods. At the organ level CytoFormer reached 58.7 against 39.7 for Virchow2 $( p < 0 . 0 0 1 )$ .

On MoNuSAC, CytoFormer led in three of the four organs, with 75.4 against 65.9 for UNI2-h in breast, 87.4 against 82.1 for MUSK in kidney and 90.9 against 79.4 for UNI2-h in lung, all with $p < 0 . 0 0 1$ (Figure 3d). The one organ in which CytoFormer did not lead is prostate, where it still ranked among the top methods. Averaged over the four organs it reached 83.5 against 78.6 for UNI2-h $( p < 0 . 0 0 1 )$ . On PanNuke, averaged over its 19 organs, it was highest on all four classes, with 89.7 against 85.1 for Virchow2 on neoplastic cells, 87.1 against 81.5 for Virchow2 on epithelial cells, 76.7 against 68.6 for MUSK on connective cells and 64.8 against 52.8 for MUSK on inflammatory cells, and averaged 79.3 against 70.3 for MUSK, all with $p < 0 . 0 0 1$ (Figure 3e). Across all 25 settings CytoFormer averaged 79.2 macro-F1, against 71.4 for the second-best method of each setting.

CytoFormer therefore performed well on all four datasets, at both the class and the organ level. The advantage was not restricted to the organs and cell types it had been trained on, and CytoFormer performed just as well on those it had never seen. Nine of PanNuke’s 19 organs are not among the organs seen during

dysplastic malignant epithelia

![](images/02fa833fa61153f7ce4f56a64339889cc1e487540f0f4f78e8b3fd0de51e6631.jpg)

## CoNSeP / colonb

## MoNuSACd

![](images/298de0df05cb890d65d2a35ffdf2cd6598c0ff174b070e3f6cfe9217bbade885.jpg)

![](images/6a15b6efb60548ee27055ae81e9cbbed91f822fdd260b903d6eef2f3ab6a8fd2.jpg)

![](images/8b81748dcacd98ffbd58f429659b3890c94c9a76a29c60d52b4c8fb664d40a73.jpg)

![](images/57cd77962ad66e619aba336f8ef475307391f7fc46fa6a2db678e4e57f5b8f06.jpg)

![](images/9a11d603471dfe59151ba8c31ca2c2ed05d70055aa8d5550747e7c3b8ed1b0ba.jpg)

![](images/b0cc2fdb1130d8658395731a3ac91a169686f1fd50f3af2061f47d698c70f978.jpg)

![](images/c68597815d71d5e43c58d8866b92020a5c26e9f59b1d3beae902fa9b997e42c9.jpg)

![](images/113ff407708080cf6f3c051eefd9100d9421c415839360cfa3984b896f2af632.jpg)

## PUMA / skinc

![](images/9f8d0362794b01829895292dc4b5fd5a5a983416c1feab11c6aaa9854e0f27ca.jpg)

![](images/04fbcea673c11d510b8dc790124b430098d0a49eb7e1e99d852a2b581ae8d5d9.jpg)

![](images/c90dd0423276c0a340be157185d8ad9622ad4c15e8caa91db950ca8286231177.jpg)

![](images/e08dd7bf9034f563ad9133103902d27d78bc2b6d4de9a8868bca3e3a962333c7.jpg)

![](images/0e5c931055ec19ed7cbfefac6bb138efab5a268afe9607a9dc258a5432b98405.jpg)

![](images/5b402201d6be59a9c19237340d74326271fb2a26dd245aa83f900b4c8a185ba6.jpg)

![](images/5f883710fda9ad5469f96861f7664972c9b3c931b4bc9b782fe62522cb4e32d1.jpg)

## PanNuke - 19 organse

![](images/a50162a077ec14c810f7ee3c1b658278e5da8cfa5dcea4a1776985a78b09474f.jpg)

![](images/a99e8524de82ed4e12c2776ba9167e1690c24e14c00bc68c691261a900535db0.jpg)

![](images/8878fcfd8df48c65aa53698afb409f5be18aede7df21c3cd9625b97bcbb630a8.jpg)

![](images/012afe041324832d0057b0ee8b7ab16b867a86008af1819d1300bb0e9f650f02.jpg)

![](images/6785093b46357f849b02444193d14f75dfc6a478ab639f9fa699e4a91f612996.jpg)

![](images/b49c135d9217d68f0ac871b0057d4470ceced1d51422897966f8db709aadf389.jpg)

![](images/95ac1ed2046c9178e7eb5ca6a44aae499496eedd7163312f3c91ec0ee2448549.jpg)

Figure 3: Linear fine-tuning of CytoFormer and six pathology foundation models on four public benchmarks. (a) Linear fine-tuning setup. A 56 µm crop around each nucleus is encoded by the frozen backbone (snowflake) and classified by a linear head, which is the only trainable part (flame). (b) F1 per cell type on CoNSeP (colon). (c) F1 per cell type on PUMA (melanoma skin). (d) Macro-F1 per organ on MoNuSAC and their average. (e) F1 per cell type on PanNuke and the overall average, each averaged over the 19 organs. Error bars represent 95% confidence intervals from bootstrap resampling over test cells. Statistical significance was assessed using a paired two-sided bootstrap test: $^ { * } p < 0 . 0 \bar { 5 } , ^ { * * } \bar { p } < 0 . 0 1 , ^ { * * * } p < 0 . 0 0 1$ . A bracket is drawn only between CytoFormer and the second-best method and gives their exact p-value; the remaining methods are marked with asterisks only.

pretraining, among them bladder, oesophagus, stomach, thyroid, testis and adrenal gland, and CytoFormer led on every one of them, by the same margin as on the organs it had seen. It also led on cell types that its own label space never contained as separate classes, such as melanophages in PUMA. This indicates that what the model has learned is generic cell-level information, the appearance of the nucleus together with the tissue immediately around it, rather than a fixed set of organ-specific categories.

## 2.6 Full fine-tuning on public cell-classification benchmarks

We then repeated the comparison with every backbone fine-tuned end-to-end (Figure 4a). This evaluation asks a practical question: when choosing a pretrained model for H&E cell classification, which backbone provides the strongest starting point and can achieve the best performance after full fine-tuning? The datasets, the cells, the crops and the training recipe were the same for all methods, so the only difference remains the backbone the model starts from.

We fully fine-tuned every model on each benchmark. CytoFormer continued to perform very well. On CoNSeP it had the highest F1 on four of the six classes (Figure 4b), with 97.7 against 97.1 for UNI2-h on dysplastic malignant epithelium $( p = 0 . 0 0 4 )$ , 92.3 against 90.2 for UNI2-h on healthy epithelium $( p =$ 0.056), 81.0 against 80.5 for PathGen on muscle $( p = 0 . 6 3 6 )$ and 66.0 against 45.1 for MUSK on endothelial cells $( p < 0 . 0 0 1 ) ;$ on fibroblasts and inflammatory cells CytoFormer was not the highest, but in both cases the difference from the best method was not significant $( p = 0 . 3 2 2$ and $p \ = \ 0 . 1 6 6 )$ . At the organ level CytoFormer reached a macro-F1 of 83.9 against 78.5 for UNI2-h $( p < 0 . 0 0 1 )$ . On PUMA it had the highest F1 on five of the nine classes (Figure 4c), among them epithelium at 85.8 against 84.8 and plasma cells at 65.6 against 64.7. On the remaining four classes CytoFormer was not the highest; on three of them the difference from the best method was not significant, and on tumour it stayed among the top methods. At the organ level CytoFormer reached 69.1 against 67.8 for UNI2-h $( p = 0 . 0 0 8 )$

On MoNuSAC, CytoFormer had the highest macro-F1 in breast (98.7 against 98.2 for UNI2-h, $p = 0 . 5 2 8 )$ and kidney (92.6 against 89.6 for UNI2-h, $p = 0 . 0 2 8 )$ . In lung it was not the highest, but the difference from the best method was not significant $( p = 0 . 0 8 2 )$ , and in prostate it stayed among the top methods. Averaged over the four organs CytoFormer reached 95.2, with no significant difference from the best method $( p =$ 0.732; Figure 4d). On PanNuke, averaged over its 19 organs, CytoFormer was highest on all four classes, with 91.6 against 91.1 for UNI2-h on neoplastic cells $( p = 0 . 0 1 6 )$ , 92.0 against 91.9 for Virchow2 on epithelial cells $( p = 0 . 7 9 0 )$ , 80.6 against 79.0 for UNI2-h on connective cells $( p < 0 . 0 0 1 )$ and 70.8 against 68.6 for UNI2- h on inflammatory cells $( p = 0 . 0 1 0 )$ , and averaged 83.5 against 82.2 for Virchow2 $( p = 0 . 0 0 2 ,$ ; Figure 4e).

CytoFormer therefore still performed well under full fine-tuning on every dataset. It stayed clearly ahead on the difficult classes, such as endothelial cells in CoNSeP and connective cells in PanNuke, even after every backbone had been trained on the target labels. This confirms that pretraining on our dataset is effective, and that CytoFormer is a strong starting point for cell classification on H&E.

![](images/9db6211af66cd3bfa7990d4eb20f513ae8473008c367beb91941c649a33352ba.jpg)

## <sub>CoNSeP</sub> <sub>/</sub> <sub>colon</sub>b

## MoNuSAC<sup>d</sup>

![](images/32ccb13cc4135d2e9458d8284f1f46658c252387a800dc69149871781710ca04.jpg)

![](images/d78f96ce258afe2e0973d5e241488f70456284744150d7660762eed724601dd8.jpg)

![](images/ce1955b000f0e8d71caa1564e497b8ea634f7f8357354f2bfb0840341ad6a522.jpg)

![](images/b4b98d1d0235d13298a9df3a2d0b886ec2c8795966455d417048fd839c4a46e2.jpg)

![](images/9ca6a2c036ad3452d60e5f9001f5ebecc1afca45c6f4fd66e1489f783fa9c6fc.jpg)

![](images/11b01b741f0476139feb32080772fc48ff75f1b9f05d9afff5b5be88f75e47cb.jpg)

![](images/7d28df03e7e0f139185d5e4fee612a6e27d076b4b236b715b3ca916e119def8b.jpg)

![](images/cecb34e60a545b590f43071e93792f4fc8f1c11d268092571c7acf547606b539.jpg)

## PUMA / skin<sup>c</sup>

![](images/fa847ab3625d9420ee368c39c87afa4309600580cd43f3425eaa322163d0fbaf.jpg)

![](images/262ae3dc1e3ad7d30c0e785d68d5b93689b8eb89f5fa95cf694b37cc676205f4.jpg)

![](images/69bd8358b2d72c79576376fbfddcaffa39383099e8eba5ab9a23c074e810d66e.jpg)

## PanNuke - 19 organs

![](images/19482ad8c6ffeef7ea4f8519f4a0947535e7f08e66a2ece52c8ae4a37e295661.jpg)

![](images/9ade7d668d40b32f1835fee4345f5a80cb5154f25a54c5e5336c05787d013b2d.jpg)

![](images/764ec620f02e78972516ac60b9a6197b36379c4433a242a63c53246d12a675fd.jpg)

![](images/16f9e5e5bdde65aff2a8ceefe04bd6244db680351c5cc19912e521c9a1e90c24.jpg)

![](images/9d128912c98f716e590836198d3e562044033abca92b105bcc3e9594fa2d15da.jpg)

![](images/2dcad0e1d7d4820e8abe42e1341cd525f1be865b0d9a954ec2d71041cbb0c8b2.jpg)

![](images/da88b7878d31940d7d1f293adfa43b59912d7f2264cbb90188462d87a0f2205b.jpg)

![](images/85e322f1334457e2033d855873ca645e58a20c28da3284e6e462015468c87921.jpg)

![](images/3ad726133b7fa2e875f35ffff908ccb42bc48c2f61ced21823739542f3452ab1.jpg)

![](images/52840f6b7a3a4087260802550809feac9c1b74b00f2ff50eb52c2a9abb5d5fd4.jpg)

![](images/a208e8b225a060e0a32d264ba5b2be0a5c35f217d29fb9895211d86ac3182ec0.jpg)

Figure 4: Full fine-tuning of CytoFormer and six pathology foundation models on four public benchmarks. (a) Full fine-tuning setup. A 56 µm crop around each nucleus is encoded by the backbone and classified by a head, and both are trained end-to-end (flames). (b) F1 per cell type on CoNSeP (colon). (c) F1 per cell type on PUMA (melanoma skin). (d) Macro-F1 per organ on MoNuSAC and their average. (e) F1 per cell type on PanNuke and the overall average, each averaged over the 19 organs. Error bars represent 95% confidence intervals from bootstrap resampling over test cells. Statistical significance was assessed using a paired two-sided bootstrap test: $^ { * } \bar { p } < 0 . 0 \bar { 5 } , ^ { * * } \bar { p } < 0 . 0 1 , ^ { * * * } p < 0 . 0 0 1$ . A bracket is drawn only between CytoFormer and the second-best method and gives their exact p-value; the remaining methods are marked with asterisks only.

## 2.7 CytoFormer provides more label-efficient representations for interactive cell annotation

A practical foundation model should let a pathologist obtain accurate cell typing from only a handful of annotations. We therefore benchmarked CytoFormer against existing pathology foundation models as the representation for interactive, active-learning cell annotation on our TissueLab platform [19], focusing on label efficiency.

To evaluate it, we used a colorectal cancer VisiumHD section (Figure 5a) and framed the task as detecting normal epithelium, which is hard because benign crypt epithelium and adenocarcinoma look alike on H&E and share most marker genes. We derived per-cell ground truth from the paired VisiumHD transcriptome together with the crypt structure (Figure 5b). For every foundation model baseline we trained an identical logistic-regression head on the same user annotations and evaluated normal-epithelium detection, Epithelium versus all other cells, across the section (10,627 positive and 151,075 negative ground-truth cells), so that the embedding was the only variable.

CytoFormer detected normal epithelium most accurately, reaching an F1 of 0.82 and confining the Ep ithelium prediction to the normal crypt region (Figure 5c). All other foundation-model baselines scored markedly lower, with UNI2-h the strongest at F1 0.69, followed by Virchow2 at 0.67, MUSK at 0.49, Path-Gen at 0.47, and CONCH and PLIP at 0.44, so that CytoFormer led the best baseline by about 0.13 in F1. Zooming into a region of interest at the benign–malignant boundary (Figure 5d, with ground truth in Figure 5e) made this difference concrete. The advantage was largest in the low-annotation regime that active learning operates in. Across the annotation schedule, the CytoFormer learning curve rose fastest and highest (Figure 5f and Extended Data Figure 2), reaching most of its final F1 within the first ∼200 annotations, whereas the baselines plateaued well below it and did not close the gap even at the largest annotation budget. Per-model overlays within this region (Figure 5g) confirmed that CytoFormer placed Epithelium precisely on the normal crypts while the baselines scattered it across malignant and non-epithelial cells. Because every model used an identical classifier and differed only in its embedding, this advantage is attributable to the representation, showing that CytoFormer yields linearly separable, label-efficient cell features that make detecting subtle cell states such as normal epithelium amid look-alike tumour practical from only a few annotations.

## 3 Discussion

We have presented CytoFormer, a cell foundation model trained on H&E cells whose identity comes from paired spatial transcriptomics. Reference labels were derived from Xenium by clustering, marker-gene annotation and human review, and were then transferred to the matched H&E morphology. This gave 15.4 million cells from 81 sections and 16 organs without any manual annotation of images. On spatially held-out tissue the model reached an accuracy of 0.846 and a macro-F1 of 0.779 across all 16 organs, and when applied to a whole section its predictions reproduced the tissue architecture of the reference. The representation also transfers. Under linear fine-tuning it performed very well on four public benchmarks, including on organs and cell types that were not part of pretraining, and it continued to perform well when

![](images/7c45bc59f3f9545163202e7e4a586e73610a5a0248768786333a3d3409813040.jpg)

![](images/0fa66fa73d025c9fd1166af50ca39a8749f05b812962cfc2e1d23f23f749b16b.jpg)

Figure 5: CytoFormer embeddings enable label-efficient active learning for cell classification on VisiumHD. For every foundation model an identical logistic-regression head is trained on the same user annotations and evaluated on the same held-out cells, so that the embedding is the only variable. (a) H&E whole-slide image of the section. (b) VisiumHD ground-truth overlay, with Negative control shown in grey, Tumor in red and Epithelium in green. (c) Active-learning prediction from CytoFormer (ours), with the Epithelium prediction confined to the normal crypt region. (d) H&E of a zoomed region of interest (ROI) spanning the benign–malignant boundary. (e) Ground truth of the zoomed region of interest spanning the benign–malignant boundary. (f) Active-learning learning curves (F1 for detecting normal Epithelium versus all other cells) as a function of the number of annotated cells, averaged over five random subsets; CytoFormer (bold) is highest throughout and improves most with additional annotations. (g) Per-model prediction overlays within the ROI: CytoFormer places Epithelium precisely on the normal crypts, whereas frozen foundation models over-call epithelium across malignant and non-epithelial cells.

every model was fully fine-tuned.

Several limitations should be noted. First, our label space covers the cell types that are commonly reported in each organ, which are the ones used in routine practice, and does not go below that level. Finer subtypes are therefore not separated, for example T and B cells, which are grouped as Lymphocyte, or the states of macrophages, even though the transcriptome that produced the labels often carries that information. Second, this work uses Xenium alone as the source of cell types. In future work we will extend it to other paired modalities, such as spatial proteomics [20] and Visium HD [21]. Third, we have used CytoFormer only to assign a type to each cell, but its output is a map of every cell in a slide, and that map is itself a measurement. Cell composition and the spatial arrangement of cells carry diagnostic and prognostic information, and in future work we will use the counts, densities and neighbourhoods that the model produces as features for clinical tasks such as diagnosis, prognosis and the prediction of treatment response [19].

In summary, CytoFormer shows that paired spatial transcriptomics can replace manual annotation as the source of cell-level supervision on H&E, and that a cell foundation model trained in this way transfers to organs and cell types it has never seen. The model turns any routine H&E slide into a map of typed cells, and provides a cell-level representation that downstream analyses can build on.

## 4 Methods

## 4.1 Dataset introduction

Data sources. We went through the human Xenium experiments available from two sources: the 10x Genomics public dataset portal, and HEST-1k [16], which collects spatial transcriptomics datasets deposited elsewhere. We kept the experiments in which the same tissue section had also been imaged with H&E, and removed samples that appear in both sources. After removing sections without an H&E image or without a usable image-to-transcriptome registration, 81 sections from 16 organs entered training: bone, bone marrow, brain, breast, cervix, colon, heart, kidney, liver, lung, lymph node, ovary, pancreas, prostate, skin and tonsil. Of these, 54 come from the 10x Genomics portal and 27 from HEST-1k deposits. Gene panels range from targeted about 300-plex to the 5,000-plex Prime panel, and sections include healthy, inflammatory and malignant tissue.

Reference cell typing. Each section was processed independently in Scanpy [22]: counts were normalised per cell and log(1 + x) transformed, highly variable genes selected, a PCA computed, and a k-nearestneighbour graph clustered with Leiden [23] at a resolution of 0.8. Sections sharing a gene panel within an organ were integrated before clustering; single sections were clustered on their own without scaling, so that absolute expression levels remain interpretable. For every cluster we computed ranked marker genes

a  
![](images/741dd0779e8fb231036a582be87e41cda1eb143204e605dc3a2f2d5cb0d52002.jpg)  
b

![](images/70215cbec1f77718e2bcb8f7fc9932fdae2363e3c87265feeb6dce9ab01c9401.jpg)

c  
![](images/d64ed2730204a0f0763cd4e07ab2be28e6091540b2dfd1798978ace7ceb9f05a.jpg)  
d

![](images/9bbe9624759324773ba8716d0eb9584730b9f5283a2e08adb92d8b964faa44c0.jpg)

e  
![](images/a597acf95405d03c091c9ed443fec1584e8abd67cf7ebf0aac6aed5fea22f5aa.jpg)  
f

![](images/f4e3ebc5164620d9e02cb5e49ab5d761e16066b4123e31c5e9318ce4ced61841.jpg)

Extended Data Figure 1: Additional in-distribution metrics on the held-out Xenium test split. Complementary metrics to Figure 2a,b, all computed on the spatially held-out split (2,944,356 cells). (a) Accuracy per organ. (b) Accuracy per cell type, which for a single class equals its recall and is therefore identical to panel (f). (c) Macro-precision per organ. (d) Precision per cell type. (e) Macro-recall per organ. (f) Recall per cell type. Error bars represent 95% confidence intervals from bootstrap resampling over test cells.

<table><tr><td colspan="3"></td><td></td><td>●</td><td>●</td><td>●</td><td>●</td><td>●</td></tr><tr><td>Iter.</td><td>Annotations</td><td>CytoFormer</td><td>UNI2</td><td>Virchow2</td><td>CONCH</td><td>PathGen</td><td>PLIP</td><td>MUSK</td></tr><tr><td>1</td><td>10</td><td>0.246</td><td>0.174</td><td>0.161</td><td>0.139</td><td>0.102</td><td>0.130</td><td>0.155</td></tr><tr><td>2</td><td>20</td><td>0.453</td><td>0.395</td><td>0.346</td><td>0.291</td><td>0.234</td><td>0.237</td><td>0.299</td></tr><tr><td>3</td><td>30</td><td>0.609</td><td>0.493</td><td>0.395</td><td>0.315</td><td>0.208</td><td>0.226</td><td>0.289</td></tr><tr><td>4</td><td>50</td><td>0.630</td><td>0.474</td><td>0.412</td><td>0.287</td><td>0.211</td><td>0.214</td><td>0.267</td></tr><tr><td>5</td><td>75</td><td>0.706</td><td>0.601</td><td>0.512</td><td>0.361</td><td>0.266</td><td>0.286</td><td>0.335</td></tr><tr><td>6</td><td>100</td><td>0.709</td><td>0.616</td><td>0.573</td><td>0.394</td><td>0.284</td><td>0.317</td><td>0.331</td></tr><tr><td>7</td><td>150</td><td>0.773</td><td>0.669</td><td>0.651</td><td>0.424</td><td>0.373</td><td>0.368</td><td>0.444</td></tr><tr><td>8</td><td>200</td><td>0.797</td><td>0.672</td><td>0.656</td><td>0.421</td><td>0.407</td><td>0.387</td><td>0.478</td></tr><tr><td>9</td><td>300</td><td>0.806</td><td>0.665</td><td>0.653</td><td>0.422</td><td>0.408</td><td>0.385</td><td>0.464</td></tr><tr><td>10</td><td>500</td><td>0.815</td><td>0.675</td><td>0.671</td><td>0.451</td><td>0.415</td><td>0.401</td><td>0.480</td></tr><tr><td>11</td><td>800</td><td>0.818</td><td>0.678</td><td>0.674</td><td>0.453</td><td>0.428</td><td>0.425</td><td>0.483</td></tr><tr><td>12</td><td>1,100</td><td>0.818</td><td>0.681</td><td>0.677</td><td>0.445</td><td>0.436</td><td>0.421</td><td>0.488</td></tr><tr><td>13</td><td>1,844</td><td>0.821</td><td>0.685</td><td>0.670</td><td>0.444</td><td>0.468</td><td>0.444</td><td>0.494</td></tr></table>

Extended Data Figure 2: Epithelium detection F1 across active-learning iterations. Epithelium vs all others; 10,627 positive / 151,075 negative cells; same annotations, schedule and 5 seeds for every model. F1 for the Epithelium class, mean of 5 seeds. Bold = best in row.

by Wilcoxon rank-sum test and annotated the cluster from the top 30 markers together with the mean expression and per cent positivity of canonical lineage markers.

Quality control. Two orthogonal filters were applied. At the cluster level, every cluster of every organ was reviewed by hand: whether the cell types recovered and their proportions match what the tissue should contain, what the top markers of the cluster are, and how its absolute expression compares with competing lineages. Each cluster was then kept, relabelled, or removed. We removed the clusters that were artefacts, doublets or of low quality, and flagged the clusters that are real but ambiguous, which are excluded from training. At the cell level, we computed a confidence for every cell, which combines two scores, a neighbour purity and a marker margin: the first asks whether a cell really belongs to the cluster it was assigned to, the second whether that cluster’s markers are stronger in the cell than those of the most similar other cluster. Neighbour purity is the fraction of a cell’s k = 50 nearest neighbours in expression PCA space that belong to its own Leiden cluster. Marker margin is the mean expression of its own cluster’s marker set minus that of the strongest competing cluster’s marker set, and is then normalised to [0, 1]. The confidence of a cell is the average of the two, and we kept only the cells with a confidence above 0.6 for training. The intention is to drop the ambiguous cells that sit at the border between clusters to reduce the label noise.

Cell-type taxonomy. For each organ we selected the cell types that are commonly reported in that tissue, and mapped the cluster-level annotations onto them, giving 23 categories in total. Seven pan-tissue classes (tumour, Stroma, Epithelium, Lymphocyte, Endothelium, Macrophage, Plasma cell) cover about 97% of cells; the other 16 are organ-specific (Hepatocyte, Cholangiocyte, Acinar, Ductal, Islet, Renal tubule, Microglia, Chondrocyte, Squamous epithelium, Cardiomyocyte, Erythroid, Neutrophil, Myeloid progenitor, Bone cell, Nerve and melanocytic). The subtypes found by clustering were then merged into these predefined categories, for example T and B cells into Lymphocyte.

H&E registration and patch extraction. Cell coordinates were mapped into H&E pixel space using the registration supplied with each section, and we checked the mapping by overlaying nuclear boundaries from the DAPI channel on the H&E image. For each cell a 56 µm window centred on the nucleus centroid was cropped from the finest pyramid level and resized to 224 × 224 pixels. Crops falling outside tissue were removed with a tissue mask that also excludes tears and folds, and out-of-focus crops were removed by thresholding the Laplacian variance of the resized patch at 10, a cut-off we set by visually inspecting the discarded patches.

Training and testing split. For breast and lung, where many comparable sections exist, whole sections were held out (7 of 23 and 7 of 28). For the remaining 14 organs, training and test cells were separated by location within the section rather than at random: the section was recursively bisected at the median of the cell distribution into 16 KD-tree blocks [24], and dispersed blocks totalling about 20% of cells were assigned to test. Because every cell is represented by a 56 µm crop, two cells whose centroids lie closer than 56 µm share image content, so a test cell next to a training cell would already have been partly seen during training. We therefore discarded a guard band at every block boundary: any cell whose nearest neighbour of the opposite split lay within 112 µm was removed from both splits. The threshold is twice the crop width, so every retained training cell and every retained test cell are at least one full crop apart and their fields of view cannot overlap. This removed 366,227 cells, 2.4% of the dataset. The resulting split contains 12,111,769 training, 2,944,356 test and 366,227 guard-band cells.

## 4.2 Model architecture and training

CytoFormer is a shared image encoder followed by a multi-task, per-organ classification head. The encoder is a ViT-giant transformer [25] that takes a 224 × 224 input with a patch size of 14 and returns a 1536- dimensional embedding. It is initialised from UNI2-h [4] and fine-tuned end-to-end, and its pooled class token is layer-normalised to give one embedding per cell.

The head holds 16 linear classifiers, one per organ, each mapping the embedding to the cell types of that organ alone. Head sizes range from three to nine classes, which gives 103 outputs in total. At each forward pass the organ identifier sends every cell to the head of its organ.

We trained with AdamW [26] (weight decay $1 0 ^ { - 2 } )$ , using separate learning rates for the backbone $( 2 \times 1 0 ^ { - 5 } )$ and the heads $( 1 0 ^ { - 3 } )$ , one epoch of linear warm-up followed by cosine decay, gradient clipping at 1.0, and a batch size of 256 in bfloat $^ { 1 6 , }$ under distributed data-parallel training on four NVIDIA B200 GPUs. Dropout of 0.25 is applied to the embedding before the heads. Augmentation consisted of random horizontal and vertical flips, random rotation up to 180<sup>◦</sup>, and colour jitter with brightness, contrast and saturation 0.1 and hue 0.02. Training ran for up to 10 epochs with early stopping on held-out macro-F1 (patience 5), evaluated at the end of each epoch on a capped subsample of the test split for speed. We kept the checkpoint with the best macro-F1, which occurred at epoch 5, and re-evaluated it on the full test split.

## 4.3 Field-of-view ablation

To isolate the effect of the crop window we sampled one fixed set of cells (500,000 for training and 100,000 for testing, stratified by organ and cell type over 10 organs and 17 cell types) and cropped that same set at 28, 56, 112 and 224 µm, each resized to 224 pixels. Every other element of the recipe was held constant, so the crop window, and therefore the effective resolution, is the only variable. Models were compared by macro-F1 over the 17 classes.

## 4.4 Public benchmarks

We benchmarked against six pathology foundation models used as feature extractors (UNI2-h, Virchow2, MUSK, CONCH, PathGen and PLIP) on four expert-annotated datasets: PanNuke [1] (186,836 cells, 19 organs, 4 classes), CoNSeP [2] (23,400 cells, colon, 6 classes), MoNuSAC [3] (46,901 cells, 4 organs, 4 classes) and PUMA [18] (95,389 cells, melanoma skin, 9 classes). Categories that do not correspond to a definite cell identity were removed before the benchmark was built: Dead cells in PanNuke, Ambiguous in $\mathsf { M o N u S A C } ,$ apoptosis in PUMA and the unspecified other class in CoNSeP. This leaves four, four, nine and six classes respectively. Each dataset’s own split was used where defined (PanNuke folds 1–2 for training and fold 3 for testing; the CoNSeP and MoNuSAC official splits) and PUMA was split by region of interest (about 20% held out). Every cell was cropped identically at 56 µm and resized to 224 pixels.

In the linear fine-tuning regime, features were extracted with the frozen backbone, standardised with training statistics, and a single linear layer was trained per (dataset, organ) with class-balanced crossentropy (AdamW [26], learning rate $1 0 ^ { - 3 }$ weight decay $\mathrm { \dot { 1 } 0 ^ { - 4 } } ,$ , 300 full-batch steps). In the fine-tuning regime the same backbone and head were trained end-to-end for 10 epochs with backbone learning rate $2 \times 1 0 ^ { - 5 }$ and head learning rate $1 0 ^ { - 3 }$ , cosine schedule with warm-up, bfloat16, and early stopping on test macro-F1 with patience 3.

## 4.5 Statistical analysis

Statistical significance between methods was assessed using a paired two-sided bootstrap test [27] over the test cells, with 1,000 resamples and a single set of resampling indices shared by all methods. Significance levels are denoted $^ { * } p < 0 . { \bar { 0 } } 5 , ^ { * * } p < 0 . 0 1$ and $^ { * * * } p < 0 . 0 \bar { 0 } 1$ . Confidence intervals (95% CI) were computed by bootstrap resampling over test cells.

## 4.6 Label-efficiency benchmark

We benchmarked CytoFormer against six pathology foundation models used as feature extractors, UNI2-h, Virchow2, CONCH, PathGen, PLIP and MUSK, on interactive normal-epithelium detection in a colorectal cancer VisiumHD section.

We assigned each segmented nucleus a cell type from the paired VisiumHD transcriptome, aggregating the 2 µm-bin expression that fell within its contour and scoring curated marker sets, and kept only confident calls that met total-UMI and score-margin thresholds. Because benign and malignant epithelium express the same markers, we further resolved the benign versus malignant axis from crypt structure. Epithelial cells lying within normal crypts were labelled normal Epithelium, epithelial cells elsewhere were labelled Tumor, and all non-epithelial cells were labelled Negative control. This yielded 161,702 confidently typed cells, of which $1 0 , 6 2 7$ were normal Epithelium and 151,075 were other cells.

For every cell we cropped a $5 6 \mu \mathrm m$ H&E patch centred on its nucleus, resized it to $2 2 4 \times 2 2 4$ pixels and encoded it with each foundation model to obtain a per-cell embedding. On top of each embedding we trained a multinomial logistic-regression head with balanced class weights and $C = 1$ on the user annotations and predicted the three classes for every cell. All models shared the same annotations, the same cells and the same head, so the embedding was the only variable.

We scored normal-epithelium detection as the F1 of the Epithelium class, one versus all other cells, over all 161,702 confidently typed cells. Because the goal is to measure how accurately a small set of annotations propagates across the slide under active learning rather than generalization to unseen data, we evaluated on the whole slide including the annotated cells, which form a small fraction of the evaluation set and are identical across models. To trace label efficiency we retrained on nested random subsets of the annotations of increasing size, from 10 up to 1,844, and averaged over five fixed random seeds (0 to 4); at each annotation budget the same seed selected the same subset for every model, so the comparison is matched and reproducible.

## Author Contributions

J.Y., S.L. developed the methodology. J.Y., S.L., A.Y. performed the experiments. J.Y., S.L., A.Y., Z.H. wrote the manuscript. Z.H. conceived and supervised the study. All authors discussed the results and approved the final manuscript.

## Acknowledgements

This project was supported by startup funding from the Perelman School of Medicine, University of Pennsylvania (Z.H.) and the Abramson Cancer Center Pilot Award (Z.H.).

## Code Availability

Code is available at https://github.com/zhihuanglab/CytoFormer. Model checkpoint is available at https://huggingface.co/zhihuanglab/CytoFormer.

## Conflict of Interests

None declared.

## References

1. Gamper, J., Alemi Koohbanani, N., Benet, K., Khuram, A. & Rajpoot, N. PanNuke: An Open Pan-Cancer Histology Dataset for Nuclei Instance Segmentation and Classification in European Congress on Digital Pathology (ECDP) (2019), 11–19.

2. Graham, S., Vu, Q. D., Raza, S. E. A., Azam, A., Tsang, Y. W., Kwak, J. T. & Rajpoot, N. Hover-Net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images. Medical Image Analysis 58, 101563 (2019).

3. Verma, R., Kumar, N., Patil, A., Kurian, N. C., Rane, S., Graham, S., Vu, Q. D., Zwager, M., Raza, S. E. A., Rajpoot, N., et al. MoNuSAC2020: A Multi-Organ Nuclei Segmentation and Classification Challenge. IEEE Transactions on Medical Imaging 40, 3413–3423 (2021).

4. Chen, R. J., Ding, T., Lu, M. Y., Williamson, D. F. K., Jaume, G., Song, A. H., Chen, B., Zhang, A., Shao, D., Shaban, M., Williams, M., Oldenburg, L., Weishaupt, L. L., Wang, J. J., Vaidya, A., Le, L. P., Gerber, G., Sahai, S., Williams, W. & Mahmood, F. Towards a general-purpose foundation model for computational pathology. en. Nat. Med. 30, 850–862 (Mar. 2024).

5. Lu, M. Y., Chen, B., Williamson, D. F. K., Chen, R. J., Liang, I., Ding, T., Jaume, G., Odintsov, I., Le, L. P., Gerber, G., Parwani, A. V., Zhang, A. & Mahmood, F. A visual-language foundation model for computational pathology. en. Nat. Med. 30, 863–874 (Mar. 2024).

6. Huang, Z., Bianchi, F., Yuksekgonul, M., Montine, T. J. & Zou, J. A visual–language foundation model for pathology image analysis using medical Twitter. en. Nat. Med. 29, 2307–2316 (Sept. 2023).

7. Xiang, J., Wang, X., Zhang, X., Xi, Y., Eweje, F., Chen, Y., Li, Y., Bergstrom, C., Gopaulchan, M., Kim, T., Yu, K.-H., Willens, S., Olguin, F. M., Nirschl, J. J., Neal, J., Diehn, M., Yang, S. & Li, R. A vision–language foundation model for precision oncology. en. Nature 638, 769–778 (Feb. 2025).

8. Wang, X., Zhao, J., Marostica, E., Yuan, W., Jin, J., Zhang, J., Li, R., Tang, H., Wang, K., Li, Y., Wang, F., Peng, Y., Zhu, J., Zhang, J., Jackson, C. R., Zhang, J., Dillon, D., Lin, N. U., Sholl, L., Denize, T., Meredith, D., Ligon, K. L., Signoretti, S., Ogino, S., Golden, J. A., Nasrallah, M. P., Han, X., Yang, S. & Yu, K.-H. A pathology foundation model for cancer diagnosis and prognosis prediction. en. Nature 634, 970–978 (Oct. 2024).

9. Xu, H., Usuyama, N., Bagga, J., Zhang, S., Rao, R., Naumann, T., Wong, C., Gero, Z., GonzÃ ˛alez, J., Gu, Y., Xu, Y., Wei, M., Wang, W., Ma, S., Wei, F., Yang, J., Li, C., Gao, J., Rosemon, J., Bower, T., Lee, S., Weerasinghe, R., Wright, B. J., Robicsek, A., Piening, B., Bifulco, C., Wang, S. & Poon, H. A whole-slide foundation model for digital pathology from real-world data. en. Nature 630, 181–188 (June 2024).

10. Zimmermann, E., Vorontsov, E., Viret, J., Casson, A., Zelechowski, M., Shaikovski, G., Tenenholtz, N., Hall, J., Klimstra, D., Yousfi, R., Fuchs, T., Fusi, N., Liu, S. & Severson, K. Virchow2: Scaling Self-Supervised Mixed Magnification Models in Pathology. arXiv preprint arXiv:2408.00738 (2024).

11. Yao, J., Li, S., Liang, P., Xu, X., Elder, D. & Huang, Z. Melan-Dx: a knowledge-enhanced vision-language framework improves differential diagnosis of melanocytic neoplasm pathology. npj Digital Medicine 9, 171 (2026).

12. Janesick, A., Shelansky, R., Gottscho, A. D., Wagner, F., Williams, S. R., Rouault, M., Beliakoff, G., Morrison, C. A., Oliveira, M. F., Sicherman, J. T., Kohlway, A., Abousoud, J., Drennon, T. Y., Mohabbat, S. H., 10x Development Teams & Taylor, S. E. B. High resolution mapping of the tumor microenvironment using integrated single-cell, spatial and in situ analysis. Nature Communications 14, 8353 (2023).

13. Marco Salas, S., Kuemmerle, L. B., Mattsson-Langseth, C., Tismeyer, S., Avenel, C., Hu, T., Rehman, H., Grillo, M., Czarnewski, P., Helgadottir, S., Tiklova, K., Andersson, A., Rafati, N., Chatzinikolaou, M., Theis, F. J., Luecken, M. D., WÃd’hlby, C., Ishaque, N. & Nilsson, M. Optimizing Xenium In Situ data utility by quality assessment and best-practice analysis workflows. en. Nat. Methods 22, 813–823 (Apr. 2025).

14. Chen, W., Zhang, P., Tran, T. N., Xiao, Y., Li, S., Shah, V. V., Cheng, H., Brannan, K. W., Youker, K., Lai, L., Fang, L., Yang, Y., Le, N.-T., Abe, J.-i., Chen, S.-H., Ma, Q., Chen, K., Song, Q., Cooke, J. P. & Wang, G. A visual–omics foundation model to bridge histopathology with spatial transcriptomics. en. Nat. Methods 22, 1568–1582 (July 2025).

15. Huang, T., Liu, T., Babadi, M., Ying, R. & Jin, W. STPath: a generative foundation model for integrating spatial transcriptomics and whole-slide images. en. NPJ Digit. Med. 8, 659 (Nov. 2025).

16. Jaume, G., Doucet, P., Song, A. H., Lu, M. Y., Almagro-Pérez, C., Wagner, S. J., Vaidya, A. J., Chen, R. J., Williamson, D. F. K., Kim, A. & Mahmood, F. HEST-1k: A Dataset for Spatial Transcriptomics and Histology Image Analysis in Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track (2024). https://arxiv.org/abs/2406.16192.

17. Sun, Y., Zhang, Y., Si, Y., Zhu, C., Zhang, K., Shui, Z., Li, J., Gong, X., Lyu, X., Lin, T. & Yang, L. PathGen-1.6M: 1.6 Million Pathology Image-text Pairs Generation through Multi-agent Collaboration in International Conference on Learning Representations (ICLR) (2025). https://openreview.net/forum?id= rFpZnn11gj.

18. Schuiveling, M., Liu, H., Eek, D., Breimer, G. E., Suijkerbuijk, K. P. M., Blokx, W. A. M. & Veta, M. A novel dataset for nuclei and tissue segmentation in melanoma with baseline nuclei segmentation and tissue segmentation benchmarks. GigaScience 14, giaf011 (2025).

19. Li, S., Xu, J., Bao, T., Liu, Y., Liu, Y., Liu, Y., Wang, L., Lei, W., Wang, S., Xu, Y., Cui, Y., Yao, J., Koga, S. & Huang, Z. A co-evolving agentic AI system for medical imaging analysis. arXiv [cs.CV] (Sept. 2025).

20. Shaban, M., Chang, Y., Qiu, H., Yeo, Y. Y., Song, A. H., Jaume, G., Wang, Y., Weishaupt, L. L., Ding, T., Vaidya, A., et al. A foundation model for spatial proteomics. arXiv preprint arXiv:2506.03373 (2025).

21. Oliveira, M. F. d., Romero, J. P., Chung, M., Williams, S. R., Gottscho, A. D., Gupta, A., Pilipauskas, S. E., Mohabbat, S., Raman, N., Sukovich, D. J., Patterson, D. M., Visium HD Development Team & Taylor, S. E. B. High-definition spatial transcriptomic profiling of immune cell populations in colorectal cancer. Nature Genetics 57, 1512–1523 (2025).

22. Wolf, F. A., Angerer, P. & Theis, F. J. SCANPY: large-scale single-cell gene expression data analysis. Genome Biology 19, 15 (2018).

23. Traag, V. A., Waltman, L. & van Eck, N. J. From Louvain to Leiden: guaranteeing well-connected communities. Scientific Reports 9, 5233 (2019).

24. Bentley, J. L. Multidimensional binary search trees used for associative searching. Communications of the ACM 18, 509–517 (1975).

25. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J. & Houlsby, N. An Image is Worth 16x16 Words: Transformersfor Image Recognition at Scale in International Conference on Learning Representations (ICLR) (2021).

26. Loshchilov, I. & Hutter, F. Decoupled Weight Decay Regularization in International Conference on Learning Representations (ICLR) (2019).

27. Efron, B. Bootstrap Methods: Another Look at the Jackknife. The Annals of Statistics 7, 1–26 (1979).