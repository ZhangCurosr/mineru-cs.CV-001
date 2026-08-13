# Better Slots, Better Worlds: Representation Quality & Robustness in Object-Centric World Models

Shukrullo Nazirjonov<sup>1</sup>, Sai Prasanna <sup>1,2,†</sup>, Anna Manasyan <sup>1,2,†</sup>, Georg Martius 1,2

{shukrullo.nazirjonov, sai.raman, anna.manasyan}@uni-tuebingen.de

<sup>1</sup>University of Tuebingen

<sup>2</sup>Max Planck Institute for Intelligent Systems

<sup>†</sup> Equal contribution

## Abstract

Learning world models from offline trajectories enables agents to accomplish different tasks through planning. Object-centric (OC) representations, which decompose a scene into a set of slots that bind to its objects, have been proposed as an inductive bias for world models that are more sample-efficient and generalize better. Yet prior objectcentric world models (OCWMs) take the slot encoder as given and evaluate only indistribution, leaving open whether the object-centric bias actually delivers for planning and what within the OCWM drives it. We conduct a controlled study of OCWMs for visual model-predictive control along two axes: object-centric representation quality and generalization under distribution shift relative to scene-centric models. We find that (i) planning success correlates positively with unsupervised slot-quality metrics (FG-ARI, mBO), though the gains saturate at high slot quality; (ii) with well-bound slots, the auxiliary proprioception inputs and masking inductive bias that prior methods relied on become unnecessary; and (iii) an OCWM with well-bound slots plans more robustly under unseen distribution shifts than the end-to-end trained scene-centric LeWM, while DINO-WM, built on similar frozen pretrained features, remains comparably robust — suggesting that pretrained visual representations are an important contributor to robustness.

## 1 Introduction

A central promise of world models (Ha & Schmidhuber, 2018) is that an agent can learn its environment’s dynamics from offline trajectories, and then plan toward arbitrary goals at test time (Zhou et al., 2025; Maes et al., 2026b). Since offline action-conditioned trajectories are costly to collect, the representation a world model uses matters for sample efficiency and generalization. Prior work spans a spectrum of representations: scene-centric models predict over DINOv2 patch features (Oquab et al., 2024) (DINO-WM) or a single global latent (LeWM), while object-centric models predict over a set of object slots (C-JEPA) (Nam et al., 2026). Object-centric representations are appealing in principle: by factorizing a scene to match its compositional, causal structure (Locatello et al., 2020; Schölkopf et al., 2021), they should allow a world model to generalize under distribution shifts. Two assumptions behind this appeal, however, are untested. First, the slot encoder i trained separately from the world model and selected using unsupervised slot-quality metrics (FG-ARI (Greff et al., 2019) and mBO (Pont-Tuset et al., 2017)) that reward clean object masks rather than planning success; whether better-scoring slots yield better downstream planning is therefore unknown. Second, whether the promised generalization holds for planning under distribution shift is untested.

We address both with a controlled study of OCWMs for visual model-predictive control on 2D PushT and 3D OGBench-Cube, along two axes: for quality, we train world models on SlotContrast (Manasyan et al., 2025) checkpoints of varying slot quality and measure planning success; for generalization, we compare our OCWM against scene-centric world models (DINO-WM, LeWM) under matched conditions. Our findings are:

• Quality. Planning success correlates positively with unsupervised slot-quality metrics (FG-ARI, mBO), with gains saturating at high quality; with well-bound slots, the OCWM no longer needs the auxiliary proprioception inputs and masking inductive bias that prior methods relied on.

• Robustness. Under unseen distribution shifts, the OCWM degrades the least, while DINO-WM remains similarly robust and LeWM degrades substantially; together, these results suggest that planning over frozen pretrained visual representations is associated with improved robustness.

## 2 Related Work

OCWMs such as OC-STORM and C-JEPA (Zhang et al., 2025; Nam et al., 2026; Spieler et al., 2026) are evaluated only in-distribution. While Wang et al. (2026) evaluate their Dyn-O OCWM on held-out Procgen levels, they report only rollout visual fidelity rather than control metrics (returns or planning performance). Outside of world modeling, Dittadi et al. (2022) provide a controlled study of OOD generalization for slot representations on downstream property prediction, and Yoon et al. (2023) examine pre-trained OC representations for model-free RL. Our study fills this gap for OCWMs by sweeping slot quality with a fixed dynamics model and evaluating per-object visual, frame-level, and dynamics shifts against scene-centric WMs under matched conditions. Related works on object-centric learning and world modeling are summarized in Appendix D.

## 3 Experiments and Results

We investigate three research questions: (1) Does slot quality impact planning success? (2) Do better slots allow sidestepping redundant auxiliary inputs and additional training objectives? (3) Do object-centric representations plan more robustly under distribution shifts than scene-centric ones?

## 3.1 Setup

Our setup builds on C-JEPA (Nam et al., 2026), an object-centric world model with a non-causal variant (OC-JEPA) and a causal one (C-JEPA); both encode each frame into object slots with VideoSAUR (Zadaianchuk et al., 2023) and predict future slots with a transformer dynamics module. C-JEPA augments OC-JEPA with a causal inductive bias: at each step, it partitions the slots into masked and unmasked subsets and predicts the masked slots from the unmasked ones, encouraging the model to capture inter-object interactions rather than relying on self-dynamics.

We adopt this framework for visual model-predictive control with several improvements: we replace VideoSAUR with SlotContrast (Manasyan et al., 2025), which provides stronger temporal consistency and eliminates the need for Hungarian matching across slots and we update DINOv2 with DINOv3 as the feature extractor. Unless noted otherwise, our default world model, SlotContrast-WM, is the non-causal OC-JEPA backbone with neither the slot-masking objective nor the auxiliary proprioception token. Extended details are in Appendix A.2.

Environments and Baselines We evaluate our approach on the 2D PushT and 3D OGBench-Cube environments. We compare SlotContrast-WM against two scene-centric baselines: DINO-WM, which utilizes frozen DINOv2 patch tokens, and LeWM, which relies on the global CLS token of an end-to-end trained ViT (Dosovitskiy et al., 2021). Extended hyperparameters and planning configurations are in Appendix A.3.

Metrics We assess slot quality with two unsupervised metrics: the Foreground Adjusted Rand Index (FG-ARI) (Greff et al., 2019), measuring how well objects are separated into slots, and mean Best

Overlap (mBO) (Pont-Tuset et al., 2017), an IoU-based segmentation score that assesses mask sharpness. We use their video variants because, for world modeling, temporally consistent slot identity matters more than per-frame segmentation quality. For planning, we report task success rate, with criteria detailed in Appendix A.1.

## 3.2 Does slot quality impact planning success?

C-JEPA, the OCWM we build on, encodes frames with a VideoSAUR encoder whose slots are poorly bound, fragmenting objects and bleeding them into the background (Figure 1, right), yet C-JEPA reports strong planning performance with it. If the object-centric inductive bias is what aids planning, a representation that so visibly fails to isolate objects ought to plan poorly, raising a more basic question: does unsupervised slot quality bear on planning success at all? To test this, we hold the dynamics model and planner fixed and train SlotContrast-WM on intermediate SlotContrast checkpoints spanning a range of slot quality, measuring downstream planning success.

![](images/e553c0980807d1ba7ac2fe44a3ed96c8f7304a34df06ad1a7bb53591c94efe14.jpg)  
Figure 1: Slot decomposition on PushT. Left: SlotContrast binds each object (agent, T-block, goal) to a dedicated slot. Right: VideoSAUR fragments objects across slots and mixes them with the background.

As shown in Figure 2, downstream planning performance closely tracks the emergence of object separation. On OGBench-Cube, however, the same sweep saturates early (Appendix B.1, Figure 5).

![](images/970a300e3256a4cdd463b675a8e8905f711245929b6e320bd51cecaa641d8ddb.jpg)  
Figure 2: Across SlotContrast checkpoints (orange), planning success rate above the random-policy baseline (2%) is positively correlated with video slot-quality metrics: FG-ARI (left, Pearson $r =$ 0.96) and mBO (right, $r = 0 . 9 4 )$ . VideoSAUR (blue diamond) uses the same minimal configuration (no slot masking or proprioception).

Takeaway 1: Planning success is positively correlated with slot quality with no change to the dynamics model, but the gains saturate: at high slot quality, where the metrics themselves plateau, better-bound slots yield little additional planning success.

## 3.3 Do better slots sidestep auxiliary inputs and training objectives?

Why does VideoSAUR plan well under C-JEPA despite its poorly bound slots (Figure 1)? We hypothesize that the auxiliary mechanisms of masked-slot history prediction objective and the auxiliary proprioception token, which C-JEPA inherits, do not add predictive power but compensate for weak slots. We test this by sweeping num\_masked\_slots ∈ {0, 1, 2}, the number of slots masked at each training step and predicted from the rest, with and without proprioception on a high-quality (SlotContrast) and a low-quality (VideoSAUR) encoder (further visualizations in Figure 4a).

Without proprioception or masking, SlotContrast-WM reaches 84.7% (±1.9) SR, and adding the proprioception token barely helps (+0.6 pp). This minimal configuration already matches, within seed variance, the 85.3% (±3.4) of the full C-JEPA recipe (VideoSAUR with proprioception and masking) and exceeds VideoSAUR (74.7%) by 10 pp. Without proprioception, masking monotonically degrades both encoders (Appendix B, Table 3, −prop rows), and the same ordering holds on OGBench-Cube. Masking helps only when paired with proprioception on the weak encoder (73.3 → 85.3%), suggesting it leans on a proprioceptive shortcut rather than exploiting visual object interactions.

Takeaway 2: In our manipulation tasks, a sufficiently object-centric representation makes both slot history masking and proprioception unnecessary: they only compensate for weak representations.

## 3.4 Do object-centric representations allow robustness to distribution shifts?

We evaluate world-model robustness under visual and dynamics shifts (variations visualized in Ap pendix C). Under object-level appearance shifts, SlotContrast-WM retains the highest success rates, and DINO-WM (patch features) degrades only moderately, whereas LeWM (CLS token of an endto-end trained ViT) collapses (Figure 3). DINO-WM plans over frozen pretrained DINO features directly, while SlotContrast-WM plans over object slots that Slot Attention extracts from the same features; the shared pretrained foundation appears to be the main source of robustness to appearance shifts. Frame-level perturbations such as a changed background color are difficult for all models and most damaging for LeWM, whose global CLS token they fundamentally alter. On OGBench-Cube, SlotContrast-WM and DINO-WM stay close to their in-distribution success across all shifts, while LeWM degrades under scene-level shifts (Appendix C.4, Figure 8). Notably, every model fails under geometric variations, since changing an object’s shape alters its contact dynamics.

![](images/c16e26b14f34b8acd9894dcaf8611e38374fbe1867cf6d40416c1d229434f8f8.jpg)  
Figure 3: Planning success under distribution shift on PushT. OCWM (SlotContrast-WM) versus scene-centric baselines (DINO-WM, LeWM).

Takeaway 3: World models planning over frozen pretrained features — object-centric or not — tolerate appearance andframe-level shiftsfar better than the end-to-end trained LeWM.

## 4 Conclusion

Our controlled study isolates what makes object-centric world models work for planning: representation quality is the dominant factor — planning success closely tracks slot quality and saturates once slots are well-bound, at which point the auxiliary proprioception input and slot-masking objective become unnecessary. Given good slots, the resulting OCWM is also the most robust model in our study, though DINO-WM, built on similar frozen pretrained features, retains much of this robustness while the end-to-end trained LeWM degrades severely. In short, better slots enable stronger objectcentric planning, while frozen pretrained visual representations appear to be an important ingredient for robustness under distribution shift.

Limitations & future work. Unsupervised slot metrics are less informative when task-relevant objects are small relative to the scene (e.g., OGBench-Cube), motivating task-aware quality measures and end-to-end encoder–world-model training. Future work should test these findings in more diverse environments—more objects, varied scales, richer dynamics—to probe OCWM’s robustness and compositional generalization.

## References

Görkay Aydemir, Weidi Xie, and Fatma Güney. Self-supervised object-centric learning for videos. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (eds.), Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper\_files/paper/2023/hash/ 67b0e7c7c2a5780aeefe3b79caac106e-Abstract-Conference.html.

Christopher P. Burgess, Loïc Matthey, Nicholas Watters, Rishabh Kabra, Irina Higgins, Matthew M. Botvinick, and Alexander Lerchner. Monet: Unsupervised scene decomposition and representation. CoRR, abs/1901.11390, 2019. URL http://arxiv.org/abs/1901.11390.

Andrea Dittadi, Samuele S. Papa, Michele De Vita, Bernhard Schölkopf, Ole Winther, and Francesco Locatello. Generalization and robustness implications in object-centric learning. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvári, Gang Niu, and Sivan Sabato (eds.), International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, Proceedings of Machine Learning Research, pp. 5221–5285. PMLR, 2022. URL https://proceedings.mlr.press/v162/dittadi22a.html.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. URL https://openreview.net/forum? id=YicbFdNTTy.

Gamaleldin F. Elsayed, Aravindh Mahendran, Sjoerd van Steenkiste, Klaus Greff, Michael C. Mozer, and Thomas Kipf. Savi++: Towards end-to-end object-centric learning from real-world videos. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh (eds.), Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. URL http://papers.nips.cc/paper\_files/paper/2022/hash/ ba1a6ba05319e410f0673f8477a871e3-Abstract-Conference.html.

Fan Feng, Phillip Lippe, and Sara Magliacane. Learning interactive world model for object-centric reinforcement learning. CoRR, abs/2511.02225, 2025. DOI: 10.48550/ARXIV.2511.02225. URL https://doi.org/10.48550/arXiv.2511.02225.

Stefano Ferraro, Pietro Mazzaglia, Tim Verbelen, and Bart Dhoedt. FOCUS: object-centric world models for robotic manipulation. Frontiers Neurorobotics, 19, 2025. DOI: 10.3389/FNBOT.2025. 1585386. URL https://doi.org/10.3389/fnbot.2025.1585386.

Klaus Greff, Raphaël Lopez Kaufman, Rishabh Kabra, Nick Watters, Chris Burgess, Daniel Zoran, Loic Matthey, Matthew M. Botvinick, and Alexander Lerchner. Multi-object representation learning with iterative variational inference. In Kamalika Chaudhuri and Ruslan Salakhutdinov (eds.), Proceedings of the 36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA, Proceedings of Machine Learning Research, pp. 2424–2433. PMLR, 2019. URL http://proceedings.mlr.press/v97/greff19a.html.

David Ha and Jürgen Schmidhuber. Recurrent world models facilitate policy evolution. In Samy Bengio, Hanna M. Wallach, Hugo Larochelle, Kristen Grauman, Nicolò Cesa-Bianchi, and Ro man Garnett (eds.), Advances in Neural Information Processing Systems 31: Annual Confer ence on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montréal, Canada, pp. 2455–2467, 2018. URL https://proceedings.neurips.cc/ paper/2018/hash/2de5d16682c3c35007e4e92982f1a2ba-Abstract.html.

Thomas Kipf, Gamaleldin Fathy Elsayed, Aravindh Mahendran, Austin Stone, Sara Sabour, Georg Heigold, Rico Jonschkowski, Alexey Dosovitskiy, and Klaus Greff. Conditional object-centric learning from video. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022. URL https://openreview. net/forum?id=aD7uesX1GF\_.

Thomas N. Kipf, Elise van der Pol, and Max Welling. Contrastive learning of structured world models. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net, 2020. URL https://openreview.net forum?id=H1gax6VtDB.

Francesco Locatello, Dirk Weissenborn, Thomas Unterthiner, Aravindh Mahendran, Georg Heigold, Jakob Uszkoreit, Alexey Dosovitskiy, and Thomas Kipf. Object-centric learning with slot attention. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin (eds.), Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020. URL https://proceedings.neurips.cc/paper/2020/hash/ 8511df98c02ab60aea1b2356c013bc0f-Abstract.html.

Lucas Maes, Quentin Le Lidec, Dan Haramati, Nassim Massaudi, Damien Scieur, Yann LeCun, and Randall Balestriero. stable-worldmodel-v1: Reproducible world modeling research and evaluation, 2026a. URL https://arxiv.org/abs/2602.08968.

Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, and Randall Balestriero. Leworldmodel: Stable end-to-end joint-embedding predictive architecture from pixels. CoRR, abs/2603.19312, 2026b. DOI: 10.48550/ARXIV.2603.19312. URL https://doi.org/10. 48550/arXiv.2603.19312.

Anna Manasyan, Maximilian Seitzer, Filip Radovic, Georg Martius, and Andrii Zadaianchuk. Temporally consistent object-centric learning by contrasting slots. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2025, Nashville, TN, USA, June 11-15, 2025, pp. 5401–5411. Computer Vision Foundation / IEEE, 2025. DOI: 10.1109/CVPR52734.2025.00508. URL https://openaccess.thecvf.com/content/CVPR2025/html/ Manasyan\_Temporally\_Consistent\_Object-Centric\_Learning\_by\_ Contrasting\_Slots\_CVPR\_2025\_paper.html.

Malte Mosbach, Jan Niklas Ewertz, Angel Villar-Corrales, and Sven Behnke. SOLD: slot objectcentric latent dynamics models for relational manipulation learning from pixels. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu (eds.), Forty-second International Conference on Machine Learning, ICML 2025, Vancouver, BC, Canada, July 13-19, 2025, Proceedings of Machine Learning Research. PMLR / OpenReview.net, 2025. URL https://proceedings.mlr.press/ v267/mosbach25a.html.

Heejeong Nam, Quentin Le Lidec, Lucas Maes, Yann LeCun, and Randall Balestriero. Causaljepa: Learning world models through object-level latent interventions, 2026. URL https: //doi.org/10.48550/arXiv.2602.11389.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jégou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024, 2024. URL https: //openreview.net/forum?id=a68SUt6zFt.

Seohong Park, Kevin Frans, Benjamin Eysenbach, and Sergey Levine. Ogbench: Benchmarking offline goal-conditioned RL. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025. URL https: //openreview.net/forum?id=M992mjgKzI.

Jordi Pont-Tuset, Pablo Arbeláez, Jonathan T. Barron, Ferran Marqués, and Jitendra Malik. Multiscale combinatorial grouping for image segmentation and object proposal generation. IEEE Trans. Pattern Anal. Mach. Intell., 39(1):128–140, 2017. DOI: 10.1109/TPAMI.2016.2537320. URL https://doi.org/10.1109/TPAMI.2016.2537320.

Reuven Rubinstein. The cross-entropy method for combinatorial and continuous optimization. Methodology And Computing In Applied Probability, 1(2):127–190, sept 1999. ISSN 1573-7713. DOI: 10.1023/a:1010091220143. URL http://dx.doi.org/10.1023/A: 1010091220143.

Mehdi S. M. Sajjadi, Daniel Duckworth, Aravindh Mahendran, Sjoerd van Steenkiste, Filip Pavetic, Mario Lucic, Leonidas J. Guibas, Klaus Greff, and Thomas Kipf. Object scene representation transformer, 2022. URL http://papers.nips.cc/paper\_files/paper/2022/ hash/3dc83fcfa4d13e30070bd4b230c38cfe-Abstract-Conference.html.

Bernhard Schölkopf, Francesco Locatello, Stefan Bauer, Nan Rosemary Ke, Nal Kalchbrenner, Anirudh Goyal, and Yoshua Bengio. Toward causal representation learning. Proc. IEEE, 109(5): 612–634, 2021. DOI: 10.1109/JPROC.2021.3058954. URL https://doi.org/10.1109/ JPROC.2021.3058954.

Maximilian Seitzer, Max Horn, Andrii Zadaianchuk, Dominik Zietlow, Tianjun Xiao, Carl-Johann Simon-Gabriel, Tong He, Zheng Zhang, Bernhard Schölkopf, Thomas Brox, and Francesco Locatello. Bridging the gap to real-world object-centric learning. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https://openreview.net/forum?id=b9tUk-f\_aG.

Oriane Siméoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timothée Darcet, Théo Moutakanni, Leonel Sentana, Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Hervé Jégou, Patrick Labatut, and Piotr Bojanowski. Dinov3, 2025. URL https://arxiv.org/ abs/2508.10104.

Gautam Singh, Yi-Fu Wu, and Sungjin Ahn. Simple unsupervised object-centric learning for complex and naturalistic videos. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh (eds.), Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. URL http://papers.nips.cc/paper\_files/paper/2022/hash/ 735c847a07bf6dd4486ca1ace242a88c-Abstract-Conference.html.

Jonathan Spieler, Angel Villar-Corrales, and Sven Behnke. Slot-mpc: Goal-conditioned model predictive control with object-centric representations. arXiv preprint arXiv: 2605.14937, 2026.

Rishi Veerapaneni, John D. Co-Reyes, Michael Chang, Michael Janner, Chelsea Finn, Jiajun Wu, Joshua B. Tenenbaum, and Sergey Levine. Entity abstraction in visual model-based reinforcement learning. In Leslie Pack Kaelbling, Danica Kragic, and Komei Sugiura (eds.), 3rd Annual Conference on Robot Learning, CoRL 2019, Osaka, Japan, October 30 - November 1, 2019, Proceedings, Proceedings of Machine Learning Research, pp. 1439–1456. PMLR, 2019. URL http://proceedings.mlr.press/v100/veerapaneni20a.html.

Zizhao Wang, Kaixin Wang, Li Zhao, Peter Stone, and Jiang Bian. Dyn-o: Building structured world models with object-centric representations. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2026. URL https://openreview.net/forum? id=b2u1yrTwFK.

Ziyi Wu, Nikita Dvornik, Klaus Greff, Thomas Kipf, and Animesh Garg. Slotformer: Unsupervised visual dynamics simulation with object-centric models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https://openreview.net/forum?id=TFbwV6I0VLg.

Jaesik Yoon, Yi-Fu Wu, Heechul Bae, and Sungjin Ahn. An investigation into pre-training object-centric representations for reinforcement learning. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, Proceedings of Machine Learning Research, pp. 40147–40174. PMLR, 2023. URL https://proceedings.mlr.press/v202/yoon23c.html.

Andrii Zadaianchuk, Maximilian Seitzer, and Georg Martius. Object-centric learning for real-world videos by predicting temporal feature similarities. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (eds.), Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper\_files/paper/2023/hash/ c1fdec0d7ea1affa15bd09dd0fd3af05-Abstract-Conference.html.

Weipu Zhang, Adam Jelley, Trevor McInroe, and Amos J. Storkey. Objects matter: object-centric world models improve reinforcement learning in visually complex environments, 2025. URL https://doi.org/10.48550/arXiv.2501.16443.

Gaoyue Zhou, Hengkai Pan, Yann LeCun, and Lerrel Pinto. DINO-WM: world models on pretrained visual features enable zero-shot planning, 2025. URL https://proceedings.mlr. press/v267/zhou25t.html.

## A Experiment Details

## A.1 Environments and Datasets

We evaluate our framework across two distinct physical manipulation domains.

PushT A 2D planar manipulation task where a circular agent must push a T-shaped block to align with a specific target pose.

• Dataset: The offline dataset consists of 18,500 expert demonstrations augmented with varying levels of action noise. This dataset is adopted from Maes et al. (2026b), utilizing the original generation protocol introduced by Zhou et al. (2025).

• Success Criterion: For goal-conditioned planning, an episode is marked successful if the combined spatial translation error of both the agent and the block is less than 20 pixels, and the block’s angular orientation error is less than $\pi / 9$ radians (≈ 20<sup>◦</sup>) relative to the goal configuration. This follows the evaluation protocols established in prior work (Zhou et al., 2025; Maes et al., 2026b).

OGBench-Cube A 3D object manipulation environment introduced by Park et al. (2025). The workspace features a simulated UR5e robotic arm equipped with a Robotiq 2F-85 adaptive gripper, tasked with interacting with a colored cube to achieve a spatial goal configuration.

• Dataset: The offline dataset contains 10,000 heuristic-generated demonstrations (200 transitions per episode), adopted from Maes et al. (2026b).

• Success Criterion: An episode is considered successful if the Euclidean distance between the cube’s final position and its goal target position is less than 4 cm. In contrast to the PushT, this environment does not enforce kinematic constraints on the final pose of the agent (robotic end effector), making the random policy achieve high planning success rate. This setup follows the baseline methods and is a known limitation of the environment evaluation.

Both environments are implemented in stable-worldmodel (Maes et al., 2026a) and we utilize this framework for all evaluations throughout this paper.

## A.2 Object-Centric Encoders

SlotContrast. SlotContrast (Manasyan et al., 2025) is a video object-centric learning framework designed to discover temporally consistent object representations from videos. The model comprises three main components: (1) a pre-trained self-supervised dense feature encoder, such as DINOv2 (Oquab et al., 2024), which extracts rich spatial features from each frame; (2) a Recurrent Slot Attention (Locatello et al., 2020; Kipf et al., 2022) module that groups these features into object-centric slots while modeling their temporal evolution across frames; and (3) a decoder that reconstruct the dense self-supervised features from the slot representations. Training is guided by two complementary objectives. A feature reconstruction loss encourages the slots to capture the information necessary to reconstruct the encoder features, while a temporal consistency loss contrasts slot rep resentations across consecutive frames. This temporal contrastive objective promotes stable slot assignments and enables the model to learn object representations that remain consistent over time, leading to improved object discovery and tracking in dynamic video scenes.

Our setup. We adopt SlotContrast as our object-centric backbone, replacing the original DINOv2 encoder with the more recent DINOv3 (Siméoni et al., 2025), which provides stronger dense features. We otherwise follow the original training objectives. The full list of hyperparameters for PushT and OGBench-Cube is given in Table 1.

VideoSAUR. VideoSAUR (Zadaianchuk et al., 2023) is another object-centric video representation learning framework that combines DINO-based semantic features with Slot Attention for object discovery. Its key contribution is a temporal feature similarity objective that exploits motion bias to promote object disentanglement by preserving semantic and temporal relationships among DINO patch embeddings. This objective can be used either alongside a reconstruction loss or as a standalone training signal. However, because VideoSAUR does not explicitly constrain slot identities to remain temporally consistent, slot correspondences across frames may need to be recovered post hoc using Hungarian matching or similar methods.

Table 1: Hyperparameters of SlotContrast Model on PushT and OGBench-Cube.
<table><tr><td>Hyperparameter</td><td>PushT</td><td>OGBench</td></tr><tr><td>Training Steps</td><td>100k</td><td>100k</td></tr><tr><td>Batch Size</td><td>128</td><td>128</td></tr><tr><td>Training Segment Length</td><td>4</td><td>4</td></tr><tr><td>Learning Rate Warmup Steps</td><td>2500</td><td>2500</td></tr><tr><td>Optimizer</td><td>Adam</td><td>Adam</td></tr><tr><td>Peak Learning Rate</td><td>0.0004</td><td>0.0004</td></tr><tr><td>ViT Architecture</td><td>DINOv3 Small</td><td>DINOv3 Small</td></tr><tr><td>Initialization</td><td>FixedLearnedInit</td><td>FixedLearnedInit</td></tr><tr><td>Patch Size</td><td>16</td><td>16</td></tr><tr><td>Feature Dimension  $( D _ { \mathrm { f e a t } } )$ </td><td>384</td><td>384</td></tr><tr><td>Gradient Norm Clipping</td><td>0.05</td><td>0.05</td></tr><tr><td>Image Specifications</td><td></td><td></td></tr><tr><td>Image / Crop Size</td><td>256</td><td>256</td></tr><tr><td>Cropping Strategy</td><td>Full</td><td>Full</td></tr><tr><td>Augmentations</td><td>Rand. Horizontal Flip</td><td>Rand. Horizontal Flip</td></tr><tr><td>Image Tokens</td><td>256</td><td>256</td></tr><tr><td>Slot Attention</td><td></td><td></td></tr><tr><td>Slots</td><td>4</td><td>3</td></tr><tr><td>Iterations (first / other frames)</td><td>3/2</td><td>3/2</td></tr><tr><td>Slot Dimension  $( D _ { \mathrm { s l o t s } } )$ </td><td>128</td><td>128</td></tr><tr><td>Predictor</td><td></td><td></td></tr><tr><td>Type</td><td>Transformer</td><td>Transformer</td></tr><tr><td>Layers</td><td>1</td><td>1</td></tr><tr><td>Heads</td><td>4</td><td>4</td></tr><tr><td>Decoder</td><td></td><td></td></tr><tr><td>Type</td><td>MLP</td><td>MLP</td></tr><tr><td>Loss Parameters</td><td></td><td></td></tr><tr><td>Softmax Temperature (τ)</td><td>0.1</td><td>0.01</td></tr><tr><td></td><td></td><td></td></tr><tr><td>Slot-Slot Contrast Weight (α)</td><td>0.01</td><td>0.001</td></tr></table>

Qualitative Mask Analysis. To visually validate the differences in slot quality between the evaluated encoders, we extract and render the learned slot masks over an episode. As shown in Figure 4a, SlotContrast successfully isolates the agent, the T-block, and the background into distinct, temporally consistent slots in the PushT environment. In contrast, VideoSAUR suffers from severe feature entanglement, partitioning the space into Voronoi-like regions where object identities bleed across slot boundaries (a known limitation of Slot Attention-based methods (Sajjadi et al., 2022)). Figure 4b further validates SlotContrast’s object-centricity in 3D settings, demonstrating clean isolation and tracking of the robotic arm, the gripper, and the cube, even in contact events when the end-effector grasps the cube.

## A.3 Visual Model-Based Planning

All world models are evaluated under the same Cross-Entropy Method (CEM) (Rubinstein, 1999) planner budget and identical context history.

Planner Cost Functions The CEM planner optimizes action sequences by minimizing the distance between the model’s predicted latent state $\hat { Z }$ and the target goal state $Z _ { \mathrm { g o a l } }$ . While all models utilize an $L _ { 2 }$ norm formulation, the structure of the representation dictates exactly how this cost is computed:

• LeWM: Evaluates the distance between the single, global [CLS] vectors:

$$
J = | | \hat { z } ^ { \mathrm { c l s } } - z _ { \mathrm { g o a l } } ^ { \mathrm { c l s } } | | _ { 2 } ^ { 2 }
$$

• DINO-WM: Evaluates the mean squared error across the rigid spatial grid of P patches:

$$
J = \frac { 1 } { P } \sum _ { p = 1 } ^ { P } | | \hat { z } _ { p } - z _ { p , \mathrm { g o a l } } | | _ { 2 } ^ { 2 }
$$

• SlotContrast-WM (Ours): Because SlotContrast enforces strict temporal consistency, slot identities are preserved across time. This allows us to compute a direct, one-to-one distance across the K slots without any heuristic matching:

$$
J = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } | | \hat { z } _ { k } - z _ { k , \mathrm { g o a l } } | | _ { 2 } ^ { 2 }
$$

• C-JEPA (VideoSAUR): VideoSAUR does not guarantee temporal consistency. Therefore, evaluating the cost requires solving a bipartite matching problem via the Hungarian algorithm at every single planning step to align predicted slots to goal slots, where Π is the set of all possible permu tations:

$$
J = \operatorname* { m i n } _ { \pi \in \Pi } \frac { 1 } { K } \sum _ { k = 1 } ^ { K } | | \hat { z } _ { k } - z _ { \pi ( k ) , \mathrm { g o a l } } | | _ { 2 } ^ { 2 }
$$

Where specified, this visual cost is linearly combined with an auxiliary proprioceptive cost, scaled by proprio\_cost\_weight $( W \in \{ 0 , 1 \} )$ , detailed further in Appendix B.2.

All reported planning results are averaged over three random seeds. The fixed hyperparameters used for the CEM optimization per environment are detailed in Table 2.

![](images/7f306e2745abd50fe09ed7bdc00892894b66184ae6d39ab72bf885d92e171c21.jpg)

![](images/3c5f9b798dcb63cfa93cced131b939ba6c6ca1af98648bf64759ef1549972472.jpg)

![](images/801c4281c33b20a1f1e6aeeda7711268cd33fccdf95c78a505978be84166dfb1.jpg)

![](images/16487bb3e5f954f5a08c114e3178bd8f7b6bdf005186a696d525bb7d9919c2db.jpg)

![](images/952d5ba87740bfb1806d6c024ebba68a0b242d3d865d89a297e662b8d17e6937.jpg)

![](images/031eb8c69f94c7f754c2cc46255d7c7a6a9dbd688de02d05c71095460405e4bb.jpg)

![](images/0b8b1c5ed8118c4ba501f81c1b66f4a37566bf5d1ccc82584bd44b13b1bbbf75.jpg)

![](images/ae4e4d714de837acb44df66c5c2f1026d5f3b8846b85c0aeaa7e3716c6d0c1b0.jpg)

![](images/7e71aa8b0b11f29c579267c3ce1dd99d6321faee0043cad23afe6e6cea72382c.jpg)

![](images/710e83d1b6f4e96a470fd87c33d5c9c871c2f749a2e5b9e9fc9d6f08f1a07db5.jpg)

![](images/083b3b602596bf7da5cbcf2bf0104425fe75c3622a453f8043c3c94815b6e945.jpg)

![](images/428526fe07fdb58dc2cb7ec41288b23235bbf782c1d9bb2f0f4f7a0203ce2aa6.jpg)

![](images/553cf15fba8e72134af345305f03bbe2d590422e0e24983fd06e752c65e2e0f8.jpg)

![](images/ef0f6d6054ffd24af30c0532700b812afc3882ae7ba53579f52164ac6079da0d.jpg)

![](images/37b9305209a6de8614cf4bf078b40c1f82c056afa7abc6f4774808a04a0b5729.jpg)

![](images/6603efd0c8a1bbb666a784450a6b0e672b7d0b35288b482fd0a55baa287c48af.jpg)  
(a) PushT Slot Masks: SlotContrast accurately isolates the physical objects (agent, T-block) from the background. VideoSAUR produces entangled partitions where object features bleed across slots.

t=0  
t=20  
t=40  
t=60  
![](images/e287cc9458ea7213366fa99663966bc6813dde2f0c1cc4223f6a06994e4ae096.jpg)  
t=80  
(b) OGBench-Cube Slot Masks: SlotContrast successfully discovers and tracks 3D components (robotic arm and its shadow, cube, background) with temporal consistency over the episode. Notably, robotic arm and its shadow are assigned to the same slot.  
Figure 4: Qualitative comparison of learned object-centric representations over 80 timesteps.

Table 2: Cross-Entropy Method (CEM) planning parameters used across all evaluated world models.
<table><tr><td>Parameter</td><td>PushT</td><td>OGBench-Cube</td></tr><tr><td>Environment Steps Horizon (H)</td><td>25</td><td>25</td></tr><tr><td>Number of evaluated goals</td><td>50</td><td>50</td></tr><tr><td>Frameskip</td><td>5</td><td>5</td></tr><tr><td>Latent Planning Horizon (H, frameskipped)</td><td>5</td><td>5</td></tr><tr><td>CEM Samples  $( N _ { \mathrm { s a m p l e s } } )$ </td><td>300</td><td>300</td></tr><tr><td>CEM Iterations  $( N _ { \mathrm { i t e r } } )$ </td><td>30</td><td>30</td></tr><tr><td>Top Elites (K)</td><td>30</td><td>30</td></tr><tr><td>Initial Sampling Variance</td><td>1.0</td><td>1.0</td></tr><tr><td>MPC Execution Steps (frameskipped)</td><td>5</td><td>5</td></tr></table>

## B Analyses on Encoder Quality

## B.1 Slot Quality vs. Planning on OGBench-Cube

Figure 5 repeats the slot-quality sweep of the main text (Figure 2) on OGBench-Cube. Unlike PushT, the metrics start high and the relationship saturates early: FG-ARI and mBO are aggregated over the whole scene, where the large, easily segmented robot arm dominates the score while the small, task-relevant cube contributes little.

![](images/c5cca3f33ecc4266655021ba85c2733dcf5a2f40d92b96cf6f4e6fed3db72eac.jpg)  
Figure 5: Slot quality vs. planning success on OGBench-Cube. Success rate is reported relative to the random-policy baseline (48%)

## B.2 Role of Slot Masking and Proprioception Token

We sweep num\_masked\_slot $\mathsf { s } \in \{ 0 , 1 , 2 \}$ crossed with use\_proprio ∈ {true, false}, separately for the two encoder families. Each cell trains a C-JEPA WM with respective object-centric encoder(SlotContrast, VideoSAUR) on the same expert dataset with the same hyperparameters.

The same ordering holds on OGBench-Cube. With the SlotContrast-WM and no proprioception, masking provides no clear benefit in planning performance: nms=0 reaches $7 2 . 7 \pm 3 . 4 \%$ SR versus $7 0 . 0 \pm 2 . 8 \%$ for nms=1.

Table 3: Planning SR (%) on PushT evaluating num\_masked\_slots and proprioception across strong (SlotContrast) and weak (VideoSAUR) encoders. Without proprioception (−prop), masking strictly degrades planning for both encoders.
<table><tr><td>Encoder</td><td>Proprio</td><td>nms=0</td><td>nms=1</td><td>nms=2</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SlotContrast</td><td>+prop</td><td> ${ \bf 8 5 . 3 \pm 3 . 4 }$ </td><td> $8 2 . 7 \pm 1 . 9$ </td><td>82.7 ±3.4</td></tr><tr><td>SlotContrast</td><td>-prop</td><td> ${ \mathbf 8 4 . 7 \pm 1 . 9 }$ </td><td> $7 2 . 0 \pm 3 . 3$ </td><td>34.0 ±6.5</td></tr><tr><td>VideoSAUR</td><td>+prop</td><td> $7 3 . 3 \pm 5 . 7$ </td><td> ${ \bf 8 5 . 3 \pm 3 . 4 }$ </td><td>83.3 ±4.1</td></tr><tr><td>VideoSAUR</td><td>-prop</td><td> $7 4 . 7 \pm 1 . 9$ </td><td> $5 2 . 0 \pm 5 . 9$ </td><td> $4 9 . 3 \pm 6 . 6$ </td></tr></table>

## C More Visualizations and Robustness Details

## C.1 OOD Variation Groups

Each world model is trained on a single in-distribution rendering and evaluated on a fixed grid of unseen variations, changing one factor at a time. We use the same grouping as in the main-text figures. PushT variations are grouped into appearance (object and background colour), scale (object size), and object shape. OGBench-Cube variations are grouped into object-level (cube and agent) and scene-level (floor, camera, and lighting). The exact factors and values are listed in Tables 4 and 5.

## C.2 PushT Variation Suite

Figure 6 shows one representative variation per factor; Table 4 lists all values.  
![](images/07833cd8a5e2a9110e46857d10a8f66a257a893f5da67c8143e564bf4667cfd9.jpg)  
Figure 6: PushT OOD variation visualizations baseline plus one representative variation per factor. All tiles share the same fixed initial layout. Full value list is available in Table 4.

## C.3 OGBench-Cube Variation Suite

Figure 7 shows one representative variation per factor; Table 5 lists all values.

## C.4 OGBench-Cube Robustness Results

Figure 8 reports zero-shot planning success under the OGBench-Cube distribution shifts, complementing the PushT results in the main text (Figure 3). SlotContrast-WM and DINO-WM behave similarly in this environment, both staying close to their in-distribution success across all shifts. LeWM is comparatively robust to object-level variations (cube and agent color, cube size) but degrades under scene-level shifts—background color and camera angle bring it close to the 48% random-policy baseline.

Table 4: PushT OOD variation values. “In-dist.” is the training-distribution default.
<table><tr><td>Group</td><td>Factor</td><td>In-dist. value</td><td>OOD values</td></tr><tr><td>appearance</td><td>agent.color block.color</td><td>blue slate gray</td><td>red, yellow, black red, blue, black</td></tr><tr><td></td><td>goal.color background.color distractor.color</td><td>light green white off</td><td>red, blue, black red, blue, black gray, magenta</td></tr><tr><td>scale</td><td></td><td></td><td></td></tr><tr><td></td><td>agent.scale block.scale</td><td>40 40</td><td>20,60 20,60</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>object shape</td><td>agent.shape</td><td>circle</td><td>L, square, plus</td></tr></table>

![](images/0900a7b0cddc73631731525f6268d3db9455e47ed6624b938253279c36a24883.jpg)  
Figure 7: OGBench-Cube OOD variation visualizations: baseline and one representative variation per factor. Full value list is available in Table 5.

![](images/9c2448e86049ded2c9c4bb63db3cea868252eeb31c57a59ab6758b9d32ae0a5a.jpg)  
Figure 8: Zero-shot planning success rate (%) on OGBench-Cube under object-level and scene-level distribution shifts, for SlotContrast-WM, DINO-WM, and LeWM. The random-policy baseline in this environment achieves 48%.

## D Extended Related Work

Unsupervised object-centric learning. Unsupervised object-centric learning (Burgess et al., 2019; Greff et al., 2019; Locatello et al., 2020) aims to decompose a scene into a small set of latent representations, commonly referred to as slots, where each slot captures a distinct object or entity. These representations are typically learned in a self-supervised manner through reconstructionbased objectives. Slot Attention (Locatello et al., 2020) introduced an iterative attention mechanism in which slots compete to explain different regions of an image, enabling effective unsupervised object binding. Subsequent works extended this to video by adding temporal consistency and object dynamics — SAVi, SAVi++, STEVE (Kipf et al., 2022; Elsayed et al., 2022; Singh et al., 2022). More recent approaches — DINOSAUR (Seitzer et al., 2023) and the video extensions VideoSAUR, SOLV, and SlotContrast (Zadaianchuk et al., 2023; Aydemir et al., 2023; Manasyan et al., 2025) — combine slot-based architectures with pretrained vision foundation models such as DINO to improve robustness and scalability on real-world data.

Table 5: OGBench-Cube OOD variation values. “In-dist.” is the training-distribution default.
<table><tr><td>Group</td><td>Factor</td><td>In-dist. value</td><td>OOD values</td></tr><tr><td rowspan="4">object-level</td><td></td><td>crimson</td><td rowspan="4">red, blue, yellow 0.012, 0.028</td></tr><tr><td>cube.color</td><td>0.020 m</td></tr><tr><td>cube.size agent.color</td><td>purple</td></tr><tr><td></td><td>red, black</td></tr><tr><td>scene-level</td><td>floor.color camera.angle_delta</td><td>slate (0°, 0°)</td><td>brown, magenta yaw ±5°, pitch ±5° 0.3,0.95</td></tr></table>

Object-centric world modeling. Early OCWMs paired slot or entity encoders with relational dynamics models, including C-SWM (Kipf et al., 2020) and OP3 (Veerapaneni et al., 2019), predominantly evaluated on synthetic physics. SlotFormer (Wu et al., 2023) replaced graph-net dynamics with an autoregressive Transformer over slots, scaling to richer video datasets. More recent OCWMs target control: FOCUS (Ferraro et al., 2025) and SOLD (Mosbach et al., 2025) integrate slots into Dreamer-style model-based RL; Slot-MPC (Spieler et al., 2026) performs sampling-based MPC in slot space; Dyn-O (Wang et al., 2026) adds SAM-supervised slots for Procgen video prediction; OC-STORM (Zhang et al., 2025) pairs SAM-derived object features with a transformer world model for visually complex domains; and Feng et al. (2025) factorize representations hierarchically for policy learning.