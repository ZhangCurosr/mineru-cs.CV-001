# AmalthAI: An Open-Source Computer Vision Platform for Cultural Heritage

Christos Chatzisavvas<sup>1,2</sup> , Stelios Alvanos<sup>1,2</sup> , Efstratios Politis<sup>1,2</sup> , Panagiotis Rigas<sup>3,4</sup> , Thomas Pappas<sup>2</sup> , Ioannis Giannoukos<sup>2</sup> , Nikolaos Mitianoudis<sup>1,2</sup> , Agata Ulanowska<sup>5</sup> , Katarzyna Żebrowska<sup>5</sup> , Nazarij Buławka<sup>5</sup> , Christina Margariti<sup>2</sup> , George Pavlidis<sup>2</sup> , Chairi Kiourt<sup>2</sup> Anestis Koutsoudis<sup>2</sup> , Vassilis Katsouros<sup>2</sup> , and George Ioannakis<sup>2</sup>

<sup>1</sup> Department of Electrical and Computer Engineering, Democritus University of Thrace, Greece

2 Institute for Language and Speech Processing, Athena Research Center, Greece

3 Department of Informatics & Telecommunications, National and Kapodistrian University of Athens, Greece

<sup>4</sup> Archimedes, Athena Research Center, Greece

Faculty of Archaeology, University of Warsaw, Poland

Abstract. Computer vision (CV) and machine learning (ML) ofer new tools for cultural heritage (CH) artifact analysis, but the CV/ML pipeline remains largely inaccessible to CH domain experts, who lack the background to configure, train, or assess models. We present AmalthAI, an open-source CV platform that bridges this gap, enabling non-ML CH experts to independently produce and validate archaeologically meaningful findings. The interface covers dataset management, training, and inference for classification, segmentation, and object detection, with Kubeflow and Katib handling scalable training and hyperparameter search. Grad-CAM localizes the image region behind a prediction, and a visionlanguage model (VLM) adds a text description of it for expert review. Since archaeological data is often state-owned or rights-encumbered and cannot leave institutional custody, AmalthAI’s self-hostable deployment ensures sensitive data is kept within premises. We test the platform on an archaeological use case built on a custom dataset of clay textile imprints, where CH experts trained and validated segmentation, and classification models for hypothesis testing. We provide the implementation code at https://github.com/TEXTaiLES/AmalthAI.

Keywords: Cultural Heritage · Computer Vision · Machine Learning Platform · MLOps · Archaeology · Human-Centered AI · Data Sovereignty

## 1 Introduction

Cultural heritage research increasingly depends on automated image analysis, from architectural classification [32] to textile pattern recognition [38], a shift enabled by advances in deep learning and mature software libraries [1, 35]. The same holds for semantic segmentation of cultural heritage imagery [37], a task CH researchers can define on archaeological grounds. Building computer vision applications, however, still requires expertise in dataset preparation, model configuration, parameter tuning, and software engineering, a barrier for CH researchers whose primary expertise lies outside computer science.

CH research needs both a no-code interface and infrastructure controlled by the lab itself, yet existing platforms split the two. The ones open enough to selfhost tend to target users already fluent in machine learning pipelines. The ones built for no-code accessibility tend to keep training and dataset management on vendor cloud infrastructure by default, with self-hosting locked behind enterprise contracts most CH research groups cannot aford or negotiate. Neither kind of platform closes the loop back to the domain expert. Evaluation often stops at accuracy metrics, with little support for validating whether a prediction is correct for the right reasons. Prior work [3] addressed a binary segmentation task with a single-user architecture. Here we extend this to three computer vision tasks with multiclass support, a multi-user identity-federated architecture, and an interpretability loop that lets domain experts interrogate model predictions directly.

We present AmalthAI, a web-based machine learning platform that closes both gaps at once. A unified environment with an intuitive graphical interface lets users manage datasets, configure training sessions, evaluate trained models, and perform inference without interacting directly with source code, while the platform’s self-hostable, multi-user architecture keeps restricted artifact data under institutional control rather than transferring it to external services. To close the validation gap, AmalthAI integrates Grad-CAM attention maps [43], and a vision-language model (VLM) assistant that explains individual predictions in plain language, letting domain experts judge whether the model’s highlighted regions and generated descriptions align with domain-relevant visual evidence rather than relying on accuracy alone. The platform supports three application domains, image classification, semantic segmentation, and object detection, alongside configurable training options such as hyperparameter tuning and data augmentation for controlled experimentation. We demonstrate AmalthAI on a custom experimental archaeological dataset, where sample localization and classification are combined with CH expert-supported hypothesis testing to produce archaeologically informative results. This case study shows how domain experts can develop, train, and validate computer vision models while substantially reducing the technical expertise the workflow demands.

The primary contributions of this work are summarized as follows:

• A Cultural Heritage-Oriented Machine Learning Platform: An opensource platform that unifies dataset management, model training, evaluation, and inference for computer vision tasks within a single application that makes these capabilities accessible to domain experts without a machine learning background, supporting three computer vision modalities.

• Scalable and Modular Multi-User Architecture: A dockerized, multiuser architecture which allows on-premise standalone deployment, integration with external identities and data services, and expansion to additional models.

• Experimental Validation: The platform is evaluated on a real-world archaeological computer vision case study, demonstrating its practical efectiveness and usability across visual analysis tasks leading to meaningful findings using hypothesis testing.

• Explainability, Interpretability and Expert Analysis: A system combining Grad-CAM attention maps with VLM-generated prediction explanations, letting domain experts inspect whether model attention aligns with the archaeologically relevant region rather than relying on accuracy alone.

## 2 Related Work

Machine Learning Platforms. Integrated computer vision platforms have emerged to simplify the end-to-end development lifecycle by combining dataset management, annotation, model training, evaluation, and deployment within a unified environment. Roboflow [10] provides one of the most comprehensive ecosystems, ofering dataset versioning, annotation, preprocessing, augmentation, training, and hosted inference for a wide range of vision tasks. Intel Geti [17] emphasizes AI-assisted annotation and active learning, enabling iterative dataset refinement while supporting no-code model training, evaluation, and deployment. Another notable example is the Ultralytics Platform [48], which introduced a unified workflow centered on the YOLO ecosystem, integrating annotation, cloud-based training, evaluation, inference, and deployment into a streamlined user interface. While these platforms significantly reduce the engineering efort required to develop computer vision applications, they are predominantly proprietary and cloud-oriented, limiting transparency, extensibility, and self-hosted deployment.

In cultural heritage, this limitation is particularly important because many workflows require image collections, annotations and metadata to be stored in external databases or cloud services. For restricted artifact data, this can make the use of such platforms questionable or even incompatible with institutional, legal or collection specific distribution policies. In contrast, our work presents an open-source, modular platform that ofers comparable end-to-end functionality while allowing researchers and practitioners to customize, extend, and deploy the entire pipeline within their own infrastructure, thereby facilitating reproducible research and collaborative development.

Machine Learning Operations (MLOps). MLOps have become an essential component of modern AI systems, enabling reproducible experimentation, automated training, scalable deployment, and lifecycle management of machine learning models across heterogeneous computing environments [22]. Several opensource frameworks have been proposed to streamline these processes. MLflow [52] provides a modular platform for experiment tracking, model versioning, artifact management, and deployment, making it a popular choice for reproducible machine learning workflows. Kubeflow [23] extends these capabilities by leveraging Kubernetes to orchestrate scalable machine learning pipelines, supporting distributed training, workflow automation, and resource management for production environments. In particular, Kubeflow integrates Katib, which enables automated hyperparameter optimization through search strategies such as random search, grid search, and Bayesian optimization. ClearML [6] has emerged as a comprehensive open-source MLOps platform that combines experiment tracking, dataset and model versioning, pipeline orchestration, remote execution, and resource management within a unified framework, making it particularly attractive for computer vision applications.

Our platform incorporates MLOps principles to support the complete lifecycle of vision models, from experiment management and automated training to deployment and inference. By combining computer vision workflow management with scalable MLOps functionality, AmalthAI facilitates reproducible experimentation, eficient resource utilization, and streamlined deployment while remaining fully open-source and extensible.

Use Cases in Cultural Heritage. Computer vision and deep learning have become increasingly important in CH research, enabling the automated analysis of large image collections that would otherwise require extensive manual efort [14]. Recent studies have applied image classification, object detection, and semantic segmentation to tasks such as artifact categorization [50], architectural style recognition [51], material identification [36], and damage assessment [12]. These methods have been successfully employed across diverse CH assets, including textiles [45], ceramics [26], and archaeological monuments [32]. However, most existing works focus on developing task-specific models [5, 18] and require substantial expertise in machine learning frameworks and programming, limiting their accessibility to domain experts. This highlights the need for integrated and user-friendly platforms that facilitate the adoption of computer vision techniques in CH research.

## 3 The AmalthAI Platform Architecture

## 3.1 Architectural Overview

Following the modularity principles of the framework presented in [3], AmalthAI has been developed as a fully dockerized machine learning platform that supports the complete workflow from dataset management to model deployment. Containerization ensures reproducible execution environments, simplifies deployment, and enables scalable operation across heterogeneous computing infrastructures. The platform follows a modular architecture that facilitates the integration of new computer vision tasks, machine learning models, and backend services while maintaining a consistent user experience. Fig. 1 illustrates the overall architecture of the system.

![](images/ec0946a6f0c8dee582e11c61c50e0908bfdf69e605b7f5719f17593d03e0a78f.jpg)  
Fig. 1: Overview of the AmalthAI architecture and CH computer vision workflow. The platform exposes three supported task modalities (classification, segmentation, and object detection) within a unified interface. A CH expert interacts with the system through three main stages: data management, where image datasets are imported or accessed from a database; model training, where hyperparameter tuning, training, and model selection are handled through an accessible frontend and backend execution layer; and evaluation and inference, where task-specific metrics and predictions are returned for expert review and analysis.

## 3.2 User Interface

The frontend is implemented using the Flask web framework [34] and serves as the primary interface between users and the machine learning pipeline. Its design abstracts the underlying implementation details, allowing users to configure experiments, manage datasets, launch training jobs, and perform inference without interacting with source code or command-line tools. The configuration interface provides contextual descriptions of the available options, helping users understand the purpose of each parameter before launching an experiment. It also provides recommendations on the typical use cases and trade-ofs of the available model architectures, assisting users in selecting appropriate models fiaccording to their task requirements and computational constraints. Similarly, the dataset management interface provides guidance on the expected dataset structure and organization.

For more detailed guidance, the platform provides direct links to the accompanying documentation<sup>1</sup>. The documentation contains recommendations for configuring training parameters, including suggested values and considerations for diferent experimental settings. It also provides dataset size requirements and guidelines regarding recommended training, validation, and test splits to assist with reliable model development. Throughout the platform, interactive visual components are used to simplify navigation and provide immediate feedback on dataset management, training progress, model evaluation, and inference results.

## 3.3 MLOps & Model Parameterization

Training. The training pipeline is built upon Kubernetes-based MLOps, with Kubeflow orchestrating the execution of training jobs and Katib performing automated hyperparameter optimization through parallel trial execution. User selections from the frontend are translated into experiment configurations that are submitted to the Kubernetes cluster for execution. In addition to automated optimization, the platform allows users to define custom search intervals for selected hyperparameters, providing greater control over the optimization process. Data augmentation policies can also be configured through the user interface, enabling comparative experiments under diferent preprocessing strategies.

AmalthAI currently supports three computer vision tasks: Image Classification, Semantic Segmentation, and Object Detection, with multi-class learning available across all tasks. Training from scratch is supported for all three tasks, while image classification and object detection additionally support finetuning of pretrained models. The training strategy can be configured directly through the user interface. For semantic segmentation, the platform integrates widely adopted architectures, including U-Net [39], DeepLabV3+ [4], and PSP-Net [53]. Object detection is supported through the Ultralytics YOLO family of models [19], specifically YOLOv8, YOLO11, and YOLO26, selected for their favorable balance between accuracy, computational eficiency, and inference speed. For image classification, the platform includes several established convolutional neural network architectures, such as ResNet [16], EficientNet [46], MobileNetV2 [42], ShufleNetV2 [29], and ConvNeXt [27]. The model repository has been designed to be easily extensible, allowing additional classification architectures available through the TorchVision library to be incorporated with minimal implementation efort.

Each training experiment may consist of multiple parallel training trials executed by Katib, where each trial evaluates a diferent combination of hyperparameters. Once all trials have been completed, the platform automatically selects and stores only the best-performing model checkpoint along with its associated configuration and metadata. By default, the selection criterion is the primary evaluation metric of the corresponding task: mean Intersection over Union (mIoU) [28] for Semantic Segmentation, mAP@50–95 [25] for Object Detection, and Accuracy [40] for Image Classification. However, the optimization objective is configurable and may be adapted to alternative criteria, such as minimizing the training or validation loss, depending on the experimental requirements.

![](images/790a4784cca27eb59c3589efa1de925369209aac37d2e62055f698214aa9837c.jpg)  
Fig. 2: Model training configuration interface. The platform enables intuitive selection of model architectures, hyperparameter tuning, and data augmentations without interaction with underlying code.

Inference. The inference module allows users to deploy previously trained models directly through the web interface. Users may submit either a single image or multiple images simultaneously for batch inference. For every inference request, the platform instantiates a temporary Docker container using the same software environment employed during training. The container performs the requested predictions and is automatically discarded upon completion (Fig. 3). This stateless execution model ensures consistency between training and inference, while maintaining eficient utilization of computational resources. Depending on the selected task, the platform returns image classification predictions, object detections, or semantic segmentation masks, together with their corresponding confidence scores. The prediction results are presented alongside the original input images through the web interface. Additionally, for the image classification task and to support model explainability, the platform also returns Grad-CAM [43] visualizations alongside the predicted class probabilities. These heatmaps highlight the regions of the input image that most strongly influenced the model’s decision, enabling users to better understand which visual features the model focused on when producing its prediction.

![](images/1850db974635664072bbf28566671c72228516c74e576b833ad4dff112446ebe.jpg)  
Fig. 3: Stateless inference. For each request, the platform instantiates a temporary Docker container from the same software environment used during training, loads the selected trained model, performs the prediction, returns the results to the interface, and discards the container on completion.

## 3.4 Vision-Language Assistant

In addition to its machine learning capabilities, AmalthAI integrates a Vision-Language Model (VLM), specifically Qwen2-VL-2B-Instruct [49], to support explainable decision-making. The VLM takes as input both artifact images and the corresponding Grad-CAM [43] visualizations to generate natural-language explanations of classifier decisions, particularly in cases of misclassification. Rather than replacing domain expertise, it functions as a decision-support tool, providing evidence-based observations, suggesting plausible interpretations, and indicating uncertainty when visual evidence is insuficient. Representative visual examples together with the corresponding prompts are included in the supplementary material. The pipeline is designed in a way that requires domain knowledge rather than ML or VLM engineering expertise. Users are not expected to manually design system prompts; AmalthAI maintains templates while the user provides a small set of dataset specific knowledge during configuration which is then used by the platform to adapt the prompt.

## 3.5 User Management

AmalthAI supports concurrent, multi-user operation within a shared deployment. In a standalone setting, user authentication is managed by the platform itself. Alternatively, authentication can be delegated to a third-party identity provider, making AmalthAI compatible with established identity and accessmanagement systems, such as Directus [9], Keycloak [20], or Authentik [2], and, more broadly, with any service implementing the OAuth 2.0 [15] and OpenID Connect [41] standards. This enables seamless integration with organizational single sign-on and easy interconnection with other services.

In both cases, AmalthAI associates authenticated users with a stable, unique identifier, valid across every resource and reference they create within the system. Therefore, all uploaded datasets, training experiments, model checkpoints and inference results are isolated per user, while the underlying computational and storage resources remain shared. This ensures a clean, privacy-respecting workflow, while enabling efective collaboration and organization across users.

Since authentication is decoupled from the rest of the platform, and per-user identifiers are persistent across the system, local user identities can be trivially reconciled with external accounts when such a service is used, contributing to the modular nature of the platform.

## 3.6 Data Storage & Deployment

Data management follows a similarly modular pattern: the entire lifecycle of all user assets can be self-managed by the tool or delegated to an external data service through an abstracted storage layer that provides data persistence and external service connectivity.

By default, the platform is fully self-contained: datasets, training experiments and configurations, trained model checkpoints and inference inputs/outputs are persisted to local storage on the machine AmalthAI is deployed on, allowing a completely standalone deployment without external dependencies (Fig. 4a). This is particularly important for archaeological and cultural heritage datasets that cannot be transferred to external databases or cloud services due to ownership or distribution restrictions.

At the same time, the tool’s architecture supports complete asset management via an external service when a larger infrastructure is needed. When an external data service is configured (Fig. 4b), AmalthAI treats its local working directories as a cache over that service rather than as the primary copy, retrieving assets required for a training or inference job on demand into local storage. Transfers are content-addressed to avoid redundant data movement, and a local index of all available assets is maintained and synchronized both ways with the remote service. In case the external service is momentarily unavailable, the system falls back to the locally cached state, and automatically re-synchronizes the moment it re-establishes a connection to the network.

This yields the benefits of decoupled, synchronized storage management: end users and infrastructure operators may adopt any external storage service, or none at all, while still obtaining two-way synchronization, durable data persistence and automatic caching with local fallback.

The platform is distributed as a collection of Docker images, ensuring reproducible deployment across heterogeneous infrastructures. In a standalone configuration, AmalthAI runs as a self-suficient unit on a single host. In a larger deployment, the compute layer and the data storage service operate as independent, separately configurable units that communicate over the network. Scalable training and inference are handled by the Kubernetes-in-Docker cluster, with data management able to be configured in several ways (e.g. a relational database for structured metadata combined with an S3-compatible object store for large binary data, connected via REST API), or left to the default local storage solution. This separation enables AmalthAI to scale from a single-machine installation to a component of a shared, multi-platform ecosystem with minimal configuration.

![](images/2ef5ba70101ec7e60a40c07f357861bb3bec79a7a8b3d71bf25d4d6e1318cad2.jpg)  
Fig. 4: AmalthAI’s modular data layer. (a) Standalone: assets are read from and written to local storage through the storage-abstraction layer; no external service is required. (b) With an external service, the local directory becomes a cache: assets are rehydrated on first use and new ones pushed back in two-way, content-addressed synchronization, with automatic fallback to the cache when the service is ofline.

## 4 Workflow Validation Through an Archaeological Case Study

Experimental archaeology and machine learning have been intertwined in research on cereal growing conditions [31], use-wear analysis [13, 44], lithic microdebitage [11], and can be considered a promising direction in the studies of archaeological textile imprints. The approach ofers possibility of filling the gap of the limited amount of archaeological remains of the textile imprints by creating data in the controlled conditions. To ground AmalthAI in a realistic cultural heritage workflow, we use an experimental archaeology case study concerning textile imprints on clay. The relevance of such imprints lies in the indirect evidence that may be left of the original textile, such as raw materials, textile structure, and production techniques. The dataset was produced from controlled textile samples impressed on clay and documented through close-range and microscopic imaging (Dino-Lite) and was curated and annotated by CH experts. Indicatory samples of the experimental dataset can be seen in Fig. 5.

## 4.1 Hypothesis Testing in Archaeology

Hypothesis testing refers to the process of evaluating an expert-formulated proposition through observable evidence [30,33]. In this case study, the central question regarding the examined use case is the following: Can clay imprints preserve archaeologically meaningful visual information from the original textile, and, if so, can this information be recovered through computer vision models? This question is particularly relevant in textile archaeology, where preserved textile artifacts are uncommon, fragile, and often dificult to examine directly [8].

![](images/5b23c232a7eb4977ec275acaee15da6008a647e1821398efb20554ec4d714498.jpg)  
Fig. 5: Indicatory samples from the experimental textile imprint dataset. Samples have been selected to represent the variety and complexity of the dataset across diferent clay textures, lighting conditions and camera settings.

The hypothesis regarding this case study is formulated as follows: textile imprints on clay preserve visual features that are informative of the original textile’s raw material and production technique, and these features can be exploited by machine learning models trained on controlled experimental data. The hypothesis is considered supported if the model can efectively classify the experimental data according to these attributes and that misclassifications can be attributed to the structural similarities between the corresponding textile classes.

## 4.2 Use Case Workflow and Task Formulation

As a computer vision task, the use case can be viewed as a binary segmentation problem, with the purpose of precisely isolating the area of the textile imprint, and two classification problems, where each image can be classified according to production technique and raw material. The experiments for both tasks were conducted by CH experts with minimal technical knowledge in machine learning. No expert guidance was provided directly throughout the process besides the platform’s documentation and implemented information. The final results were evaluated by both machine learning and CH experts to confirm expected model behavior and interpretable results.

Experimental Setup. For classification, the dataset was split into classes according to production technique and raw material before being uploaded and processed by AmalthAI, resulting in 1927 images for material classification and 1837 images for technique classification, with only a few samples outside the defined classes. For segmentation, a curated subset of 387 images was used, with masks manually annotated by CH experts using the CVAT annotator [7] integrated with SAM [21]. Across all tasks, training was configured to include all models available for the corresponding modality, while task-specific hyperparameter ranges and augmentations were selected through the platform, as summarized in Tab. 1. The best-performing models were saved and subsequently used for inference. Given the experimental nature of the data, the training process was repeated with diferent training/test splits, with all splits producing similar results, as documented in Tab. 2.

Table 1: Training configuration used for the textile-imprint classification and segmentation experiments.
<table><tr><td>Task</td><td>Dataset size</td><td>Learning rate</td><td>Batch size</td><td>Epochs</td></tr><tr><td>Classification</td><td>1927 / 1837</td><td>0.001-0.1</td><td>4-16</td><td>30-50</td></tr><tr><td>Segmentation</td><td>387</td><td>0.001-0.1</td><td>8-16</td><td>20-30</td></tr></table>

## 4.3 Results and Workflow Validation

The results displayed in Tab. 2, together with feedback from the end users, suggest that the workflow is practical for standard computer vision tasks in an archaeological setting. According to the CH experts, the resulting scores, as well as inference visualizations, are consistent with the properties, size and visual complexity of the dataset. The main observation is operational: AmalthAI allows a CH expert to move from a structured image dataset to a trained, reusable model without directly managing code, dependencies or backend execution. The experiments also illustrate the value of a reproducible workflow for cultural heritage data analysis. Since the samples, imaging conditions, labels, training configuration, and trained models can be organized within a single platform, the resulting workflow is easier to inspect, repeat, and compare across dataset splits or task formulations. We note that the provided model results are intended as workflow-validation evidence rather than as a benchmark claim.

Classification results. The aforementioned classification experiments demonstrate that AmalthAI can support parallel evaluation of multiple standard image classification architectures. The repeated train/test splits produced similar scores, suggesting that the observed performance was not dependent on a single favorable data partition. From a workflow perspective, the important result is that the platform can configure, execute, compare, register, and reuse classification models for more than one archaeological label set.

Table 2: Performance results of the AmalthAI workflow on the experimental textileimprint use case. Classification is reported as accuracy for material and productiontechnique prediction, while segmentation is reported as mIoU for imprint-region delineation. Values are reported as mean ± standard deviation over 5 experimental splits.  
(a) Classification results.
<table><tr><td>Model</td><td> $\overline { { \mathrm { M a t e r i a l ~ A c c . ~ } ( \% ) } }$ </td><td> $\overline { { \mathrm { T e c h n i q u e ~ A c c . ~ } ( \% ) } }$ </td></tr><tr><td>ResNet18 [16]</td><td> $\overline { { 7 4 . 8 8 \pm 1 . 0 4 } }$ </td><td> $\overline { { 8 3 . 0 9 \pm 2 . 0 9 } }$ </td></tr><tr><td>EfficientNetB0 [46]</td><td> ${ \bf 7 6 . 3 2 \pm 1 . 2 1 }$ </td><td> $\mathbf { 8 3 . 6 8 \pm 1 . 8 9 }$ </td></tr><tr><td>MobileNetV2 [42]</td><td> $7 4 . 3 2 \pm 1 . 3 3$ </td><td> $8 2 . 8 9 \pm 2 . 3 7$ </td></tr><tr><td>ShuffleNetV2 [29]</td><td> $7 2 . 2 8 \pm 1 . 9 8$ </td><td> $8 0 . 1 0 \pm 2 . 6 6$ </td></tr></table>

(b) Segmentation results.
<table><tr><td>Model</td><td>mIoU (%)</td></tr><tr><td>UNet [39]</td><td> $\overline { { 9 0 . 0 2 \pm 1 . 7 9 } }$ </td></tr><tr><td>DeeplabV3+ [4]</td><td> ${ \bf 9 0 . 4 7 \pm 1 . 7 8 }$ </td></tr><tr><td>PSPNet [53]</td><td> $8 4 . 2 0 \pm 3 . 6 0$ </td></tr></table>

Segmentation results. The segmentation task has a diferent role from the classification experiments. While it does not directly infer raw material or production technique, it provides a precise localization layer for the imprint region. This is important because the textile imprint images contain substantial non-diagnostic visual information, including clay texture, lighting variation, and background surface irregularities. A reliable segmentation model therefore supports a more controlled analytical workflow by separating the impressed textile trace from its surrounding matrix. The numerical results indicate that this localization step can be reproduced consistently, despite the limited number of manually annotated masks. A set of qualitative segmentation results can be viewed in the supplementary material.

Hypothesis-testing evidence. The classification inference results in Tab. 3 provide evidence relevant to the hypothesis formulated in Sec. 4. The trained models classify the textile-imprint samples according to both raw material and production technique with meaningful overall performance, reaching 77.53% accuracy for material classification and 85.57% accuracy for technique classification. These results indicate that the imprints contain recoverable visual information associated with the archaeological labels.

The class-level results are also informative from a cultural heritage perspective. Misclassifications are not distributed uniformly across all classes: nettle shows lower material classification accuracy, while splicing is the most dificult production-technique class. According to CH experts, these are also among the cases that are more dificult to distinguish through visual inspection, especially when fibres have been processed in similar ways [47]. This correspondence is consistent with the models exploiting label relevant visual properties, although it does not by itself exclude reliance on dataset specific cues.

Taken together, the results are consistent with the hypothesis that experimentally produced clay imprints can preserve label-relevant information from textile samples. They do not imply that every imprint preserves the same level of information as the original textile, nor that automated classification can replace expert analysis. Rather, within this controlled dataset, the results suggest that textile-imprint imagery can support computational analysis of raw material and production technique, and that AmalthAI can provide an accessible workflow for testing such archaeological propositions.

Table 3: Inference results from the classification tasks used for hypothesis testing. The models used for the experiment were the best performing models from each task (EficientNetB0 [46]) as documented in Tab. 2a.
<table><tr><td colspan="4">Material</td><td colspan="3">Technique</td></tr><tr><td>Class</td><td></td><td>Correct / Total Acc. (%)</td><td></td><td>Class</td><td>Correct</td><td>Total Acc. (%)</td></tr><tr><td>Flax</td><td>195/ 267</td><td>73.03</td><td>Drilling</td><td>644</td><td>727</td><td>88.58</td></tr><tr><td>Nettle</td><td>385 / 560</td><td>68.75</td><td></td><td>Spinning</td><td>691/ 761</td><td>90.80</td></tr><tr><td>Lime Bast</td><td>433 / 550</td><td></td><td>78.73</td><td>Splicing</td><td>237 / 349</td><td>67.91</td></tr><tr><td>Wool</td><td>481 / 550</td><td>87.45</td><td></td><td></td><td></td><td></td></tr><tr><td>Overall</td><td>1494</td><td>1927</td><td>77.53</td><td>Overall</td><td>1572  / 1837</td><td>85.57</td></tr></table>

Limitations and Future Work. AmalthAI is designed to make computer vision workflows more accessible to cultural heritage experts by providing a streamlined interface for dataset handling, model training, evaluation, and inference. While this design reduces the amount of technical configuration required from the user, it also limits the degree of control available through the interface. Although the platform exposes several commonly used training parameters, more advanced options, such as custom loss functions, optimizer selection, scheduler configuration, and other low-level training settings, are currently not accessible to end users. Furthermore, AmalthAI currently requires all datasets to be strictly pre-partitioned by the user prior to upload, without internal validation of the data distribution.

Future work will focus on extending the platform’s functionality while preserving its ease of use. To address the vulnerability of non-expert users to overfitting or poor generalization due to inadequate data preparation, automated data sanity checks will be incorporated during the upload phase. This update will provide in-app guidance to flag small dataset sizes, warn against severe class imbalances, and evaluate user-provided split ratios before training begins. Another promising direction is the introduction of role-based access control, enabling advanced users to access more sophisticated training configurations without increasing complexity for non-expert users. Finally, the current platform is limited to 2D RGB imagery. Therefore, an important enhancement is the support for multispectral imagery, as spectral bands beyond the visible range can reveal material properties and features that are not detectable in conventional RGB images, providing richer and more informative input for AI models [24].

## 5 Conclusion

We presented AmalthAI, an open-source computer vision platform designed for use in cultural heritage machine learning workflows. The standard machine learning pipeline, which includes data handling, task selection, training configuration, and inference, is served as a unified workflow through an intuitive web interface and contextual guidance, making it approachable to experts unfamiliar with machine learning. The platform’s usability has been validated through a real-world case study, establishing it as a useful tool for efective data analysis in the field. By supporting three major computer vision tasks, individual user management and modular data storage, hostable on both internal and external premises, it ensures a practical and scalable bridge between established computer vision methods and cultural heritage research.

## Acknowledgments

This work has been partially supported by the European Union’s Horizon Europe research and innovation programme under Grant Agreement No. 101158328 (TEXTaiLES), and by the European High-Performance Computing Joint Undertaking (JU) under Grant Agreement No. 101234269 for the Pharos AI Factory project, as well as by the Greek Ministry of Digital Governance and Artificial Intelligence. Additional support was provided by project MIS 5154714 of the National Recovery and Resilience Plan Greece 2.0, funded by the European Union under the NextGenerationEU Programme.

The dataset was produced by archaeologists from the Faculty of Archaeology, University of Warsaw, involved in the Textile Digitisation Tools and Methods for Cultural Heritage (TEXTaiLES) and Exploring Textile Imprints on Clay from the 3rd and the 2nd Millennia BCE: Advancing Cutting-Edge Research and Documentation Protocols with Case Studies of Diverse Textile Consumption Contexts (ExplorTIC) projects, in close collaboration with the Biskupin Archaeological Museum, Poland.

## References

1. Abadi, M., Barham, P., Chen, J., Chen, Z., Davis, A., Dean, J., Devin, M., Ghemawat, S., Irving, G., Isard, M., et al.: TensorFlow: a system for Large-Scale machine learning. In: 12th USENIX symposium on operating systems design and implementation (OSDI 16). pp. 265–283 (2016)

2. Authentik Security Inc.: authentik: The authentication glue you need. https:// goauthentik.io (2026), accessed: Jun. 28, 2026

3. Chatzisavvas, C., Pappas, T., Rigas, P., Mitianoudis, N., Pavlidis, G., Kiourt, C., Koutsoudis, A., Katsouros, V., Ioannakis, G.: Towards an easy-to-use machine learning framework for cultural heritage scientists. In: Computer Applications and Quantitative Methods in Archaeology Conference (CAA2025). Athens, Greece (May 2026). https://doi.org/10.5281/zenodo.20048428

4. Chen, L.C., Zhu, Y., Papandreou, G., Schrof, F., Adam, H.: Encoder-decoder with atrous separable convolution for semantic image segmentation. In: Proceedings of the European conference on computer vision (ECCV). pp. 801–818 (2018)

5. Chen, M., Yu, L., Zhi, C., Sun, R., Zhu, S., Gao, Z., Ke, Z., Zhu, M., Zhang, Y.: Improved faster R-CNN for fabric defect detection based on Gabor filter with Genetic Algorithm optimization. Computers in Industry 134, 103551 (2022)

6. ClearML: Clearml - your entire mlops stack in one open-source tool (2024), https: //clear.ml/, accessed: Jun. 28, 2026

7. CVAT.ai Corporation: Computer Vision Annotation Tool (CVAT) (Nov 2023), https://github.com/cvat-ai/cvat, accessed: Jun. 28, 2026

8. Cybulska, M.: Archaeological textiles–a need for new methods of analysis and reconstruction. Fibres and Textiles in Eastern Europe 15, 64–65 (01 2007)

9. Directus: Directus: The collaborative backend and headless cms. https:// directus.com (2026), accessed: Jun. 29, 2026

10. Dwyer, B., Nelson, J., Hansen, T., et al.: Roboflow (version 1.0) (2026), https: //roboflow.com, accessed: Jun. 25, 2026

11. Eberl, M., Bell, C.S., Spencer-Smith, J., Raj, M., Sarubbi, A., Johnson, P.S., Rieth, A.E., Chaudhry, U., Aguila, R.E., McBride, M.: Machine learning–based identification of lithic microdebitage. Advances in Archaeological Practice 11(2), 152–163 (2023)

12. ElBehairy, A., El-Nasr, N.A.A., Grimberg, P., Said, L.A.: A comprehensive review of deep learning methods in damage classification, detection, and segmentation of cultural heritage sites. Journal of Cultural Heritage 78, 228–237 (2026)

13. Eleftheriadou, A., McPherron, S.P., Marreiros, J.: Machine learning applications in use-wear analysis: A critical review. Journal of Computer Applications in Archaeology (Jun 2025). https://doi.org/10.5334/jcaa.190

14. Fiorucci, M., Khoroshiltseva, M., Pontil, M., Traviglia, A., Del Bue, A., James, S.: Machine learning for cultural heritage: A survey. Pattern Recognition Letters 133, 102–108 (2020)

15. Hardt, D.: The OAuth 2.0 Authorization Framework. RFC 6749 (Oct 2012). https: //doi.org/10.17487/RFC6749, https://www.rfc-editor.org/info/rfc6749/

16. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 770–778 (2016)

17. Intel: Intel Geti (2023), https://github.com/open- edge- platform/geti, accessed: Jun. 25, 2026

18. Jing, J., Wang, Z., Rätsch, M., Zhang, H.: Mobile-Unet: An eficient convolutional neural network for fabric defect detection. Textile research journal 92(1-2), 30–42 (2022)

19. Jocher, G., Qiu, J., Liu, M., Lyu, S., Akyon, F.C., Kalfaoglu, M.E.: Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models. arXiv preprint arXiv:2606.03748 (2026)

20. Keycloak: Keycloak: Open source identity and access management. https://www. keycloak.org (2026), accessed: Jun. 28, 2026

21. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., et al.: Segment anything. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 4015–4026 (2023)

22. Kreuzberger, D., Kühl, N., Hirschl, S.: Machine learning operations (MLops): Overview, definition, and architecture. IEEE access 11, 31866–31879 (2023)

23. Kubeflow: A machine learning toolkit for Kubernetes (2021), https://www. kubeflow.org/, accessed: Jun. 28, 2026

24. Liang, H.: Advances in multispectral and hyperspectral imaging for archaeology and art conservation. Applied Physics A 106, 309 – 323 (2011)

25. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft coco: Common objects in context. In: European conference on computer vision. pp. 740–755. Springer (2014)

26. Ling, Z., Delnevo, G., Salomoni, P., Mirri, S.: Findings on machine learning for identification of archaeological ceramics: A systematic literature review. IEEE Access 12, 100167–100185 (2024)

27. Liu, Z., Mao, H., Wu, C.Y., Feichtenhofer, C., Darrell, T., Xie, S.: A convnet for the 2020s. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2022)

28. Long, J., Shelhamer, E., Darrell, T.: Fully convolutional networks for semantic segmentation. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 3431–3440 (2015)

29. Ma, N., Zhang, X., Zheng, H.T., Sun, J.: Shuflenet v2: Practical guidelines for eficient CNN architecture design. In: Proceedings of the European conference on computer vision (ECCV). pp. 116–131 (2018)

30. Neyman, J., Pearson, E.S.: Ix. on the problem of the most eficient tests of statistical hypotheses. Philosophical Transactions of the Royal Society of London, Series A: Containing Papers of a Mathematical or Physical Character 231(694-706), 289– 337 (02 1933)

31. Orengo, H., Berganzo-Besga, I., Esmoris, J., Lumbreras, F., Aliende, P., Wallace, M., Livarda, A.: High-performance 3D morphometrics via deep learning and tabular foundation models: a case study on complex cereal grain classification. Journal of Archaeological Science 191, 106607 (2026)

32. Ottoni, A.L.C., Ottoni, L.T.C.: A deep learning approach for cultural heritage building classification using transfer learning and data augmentation. Journal of Cultural Heritage 74, 214–224 (2025)

33. Outram, A.K.: Introduction to experimental archaeology. World Archaeology 40(1), 1–6 (2008)

34. Pallets: Flask (version 3.1.0). https://palletsprojects.com (2024), accessed: Jun. 28, 2026

35. Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., et al.: Pytorch: An imperative style, highperformance deep learning library. Advances in neural information processing systems 32 (2019)

36. Pei, H., Zhang, C., Zhang, X., Liu, X., Ma, Y.: Recognizing materials in cultural relic images using computer vision and attention mechanism. Expert systems with applications 239, 122399 (2024)

37. Ragusa, F., Di Mauro, D., Palermo, A., Furnari, A., Farinella, G.M.: Semantic object segmentation in cultural sites using real and synthetic data. In: 2020 25th International Conference on Pattern Recognition (ICPR). pp. 1964–1971. IEEE (2021)

38. Rei, L., Mladenić, D., Dorozynski, M., Rottensteiner, F., Schleider, T., Troncy, R., Lozano, J.S., Salvatella, M.G.: Multimodal metadata assignment for cultural heritage artifacts. arXiv preprint arXiv:2406.00423 (2024)

39. Ronneberger, O., Fischer, P., Brox, T.: U-net: Convolutional networks for biomedical image segmentation. In: International Conference on Medical image computing and computer-assisted intervention. pp. 234–241. Springer (2015)

40. Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M., et al.: Imagenet large scale visual recognition challenge. International journal of computer vision 115(3), 211–252 (2015)

41. Sakimura, N., Bradley, J., Jones, M., De Medeiros, B., Mortimore, C.: Openid connect core 1.0 incorporating errata set 1. The OpenID Foundation, specification 335 (2014)

42. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., Chen, L.C.: Mobilenetv2: Inverted residuals and linear bottlenecks. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 4510–4520 (2018)

43. Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., Batra, D.: Gradcam: visual explanations from deep networks via gradient-based localization. International journal of computer vision 128(2), 336–359 (2020)

44. Sferrazza, P.: Archaeological and experimental lithic microwear classification through 2D textural analysis and machine learning. Journal of Archaeological Method and Theory 32(1), 31 (2025)

45. Sha, S., Li, Y., Wei, W., Liu, Y., Chi, C., Jiang, X., Deng, Z., Luo, L.: Image classification and restoration of ancient textiles based on convolutional neural network. International Journal of Computational Intelligence Systems 17(1), 11 (2024)

46. Tan, M., Le, Q.: Eficientnet: Rethinking model scaling for convolutional neural networks. In: International conference on machine learning. pp. 6105–6114. PMLR (2019)

47. Ulanowska, A.: Why not wool? Evidence for raw materials and technical uses of textiles based on imprints on the undersides of clay sealings from Bronze Age Greece. In: Banck-Burgess, J., Marinova, E., Mischka, D. (eds.) THE SIGNIFICANCE OF ARCHAEOLOGICAL TEXTILES Papers of the Interna-tional Online Conference 24th–25th February 2021. THEFBO, Volume II, pp. 165–177. No. 28 in Forschungen und Berichte zur Archäologie in Baden-Württemberg, Reichert, Wiesbaden (2023)

48. Ultralytics: Ultralytics Platform (2026), https://platform.ultralytics.com/, accessed: Jun. 25, 2026

49. Wang, P., Bai, S., Tan, S., Wang, S., Fan, Z., Bai, J., Chen, K., Liu, X., Wang, J., Ge, W., et al.: Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191 (2024)

50. Winterbottom, T., Leone, A., Al Moubayed, N.: A deep learning approach to fight illicit traficking of antiquities using artefact instance classification. Scientific Reports 12(1), 13468 (2022)

51. Yoshimura, Y., Cai, B., Wang, Z., Ratti, C.: Deep learning architect: classification for architectural design through the eye of artificial intelligence. In: International Conference on Computers in Urban Planning and Urban Management. pp. 249– 265. Springer (2019)

52. Zaharia, M.A., Chen, A., Davidson, A., Ghodsi, A., Hong, S.A., Konwinski, A., Murching, S., Nykodym, T., Ogilvie, P., Parkhe, M., Xie, F., Zumar, C.: Accelerating the Machine Learning Lifecycle with MLflow. IEEE Data Eng. Bull. 41, 39–45 (2018)

53. Zhao, H., Shi, J., Qi, X., Wang, X., Jia, J.: Pyramid scene parsing network. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 2881–2890 (2017)

## A Supplementary Material

## A.1 Hardware Requirements & Technical Background

Defining strict minimum hardware requirements is inherently dificult, as computational demands depend on user-defined configurations, including the selected computer vision models, Vision-Language Model (VLM), batch sizes, dataset sizes, and the number of concurrent users. All experiments and inference runs presented in Sec. 4, including VLM inference tasks, were executed on a workstation equipped with a single NVIDIA RTX 5090 GPU and 64 GB of system memory (RAM). The available computational resources were suficient to execute computer vision training and inference tasks alongside the VLM pipeline for a single user. We expect single-user sequential tasks to be feasible under more constrained setups, e.g. a GPU with 16 GB of VRAM and 32 GB of system memory. While the platform itself remains hardware-agnostic and requires minimal resources for the interface and orchestration layers, several of its underlying components natively support multi-GPU execution, allowing deployments to scale across multiple GPU-enabled worker nodes.

For deployments serving multiple users, a distributed GPU allocation strategy is recommended. In particular, dedicating a single GPU to the VLM service while scheduling computer vision training and inference workloads across separate GPU workers enables concurrent execution while preventing resource bottlenecks. As workload orchestration is handled by Kubernetes, additional GPUenabled worker nodes can be provisioned to accommodate increasing computational demands and larger numbers of simultaneous users. The same hardwareagnostic scaling is applied to storage too: training and inference are executed on the local node, therefore the maximum practical dataset size is dictated by that node’s storage capabilities rather than by AmalthAI itself.

While using the platform requires no technical background, deployment and setup assume mild familiarity with the command line and containerization. The provided documentation aims to ofer a streamlined and easy process for basic setup, while deeper configuration, such as changing the available ML models or a custom data source configuration, requires the corresponding domain expertise.

## A.2 Visual Language Model Configuration and Outputs

Beyond the three core computer vision tasks, AmalthAI provides a Vision-Language Model (VLM) tool for interpreting classification results. For the use case described above, users can inspect individual misclassified samples by prompt ing the VLM with the original raw microscope image, the corresponding Grad-CAM visualization, which is generated from lower-resolution feature maps during inference, and the classifier’s prediction metadata, including the ground-truth label, predicted label, prediction confidence, and class probability distribution. The VLM is guided by a dedicated system prompt (Fig. 6), which instructs it to explain why the classifier may have favored the predicted class over the groundtruth class based solely on the provided visual evidence, without re-evaluating the prediction. To encourage consistent reasoning and a standardized JSON response, the prompt is further augmented with four representative in-context examples extracted from the classification inference stage (Figs. 7–10).

![](images/5aeda7e3dc09d750f739b9d3690eed64e607f9949c22fd6e7ac297f37fd08ab3.jpg)  
Fig. 6: System prompt for controlled visual explanations, separating raw-image description from Grad-CAM attention while preventing class labels, correctness judgments, and unsupported interpretations.

![](images/856b09c1e56a7a954c2114b7feeefb4333cb419016e179a462ad6229299e994e.jpg)  
Fig. 7: Example of a visual explanation pipeline for material-related errors. The model processes a raw microscope image (a), its Grad-CAM heatmap (b), and prediction metadata to produce a structured JSON explanation of a misclassified material.

![](images/94b7af9c2b1992485e7611e44eee6832b29b8900938bebccfc5e35a60701053e.jpg)  
Fig. 8: Example of a visual explanation pipeline for material-related errors. The model processes a raw microscope image (a), its Grad-CAM heatmap (b), and prediction metadata to produce a structured JSON explanation of a misclassified material.

![](images/6b3ed1819fe7fbfdbec60acf554816e8418582213c70baf58da1da9a9dcd7ca1.jpg)  
Fig. 9: Example of a visual explanation pipeline for errors in technique recognition. A raw microscope image (a), its corresponding Grad-CAM visualization (b), and classification outputs are combined to generate a structured JSON explanation of a techniquelevel misclassification.

![](images/63e014c359beb74f09f56c77ee58a2bd3edafca3e347f1cc4eeb3a84e22e11ed.jpg)  
Fig. 10: Example of a visual explanation pipeline for errors in technique recognition. A raw microscope image (a), its corresponding Grad-CAM visualization (b), and classification outputs are combined to generate a structured JSON explanation of a techniquelevel misclassification.

## A.3 Platform User Interface Visualizations

Fig. 11 summarizes the main AmalthAI user-interface visualizations. The dataset interface supports inspection of uploaded data splits and dataset attributes, the inference interface presents model outputs and segmentation masks, and the AI Assistant page provides interaction with a VLM that supports visual-language inspection of classification results.

![](images/be29d26315d71583a922fa7d78d3108cc5f03d18ef494053dc1c7b9de66dade8.jpg)  
(a) Dataset information interface.

![](images/4ea2cc1cc3e1bc654cd938429d58baf8af52dda6f8609170ae14408c2f34a4a1.jpg)  
(b) Inference interface.

![](images/c412f4809f9dacecc8bf90ab36d6226f8eef2e917453f7a41be335be0dbcd447.jpg)  
(c) VLM interface.  
Fig. 11: Platform user-interface visualizations in AmalthAI. The panels show the dataset information interface, the inference interface, and the AI assistant interface.

## A.4 Qualitative Segmentation Results

Fig. 12 presents qualitative segmentation results for the textile-imprint dataset. These examples complement the quantitative segmentation results by illustrating how the learned masks can support downstream expert inspection of the relevant imprint area.

Image  
Ground Truth  
![](images/744dc72ed8ef99cb0a8cf844a7faf9b4e83c60271ab03aa2e090064d1391fae4.jpg)  
Inference Result  
Image  
Ground Truth  
Inference Result  
Fig. 12: Qualitative segmentation inference results. The inference was performed by the best performing model across the experimental textile imprint dataset. Predicted segmentation masks are colored in red.