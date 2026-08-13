# DIALOGUE-AWARE VIDEO-TO-MUSIC GENERATION USING PUBLIC DOMAIN FILM COLLECTIONS

Haven Kim<sup>1</sup> Zachary Novack<sup>1</sup> Juian McAuley<sup>1</sup> Hao-Wen Dong<sup>2</sup>

<sup>1</sup>University of California San Diego <sup>2</sup>University of Michigan

## ABSTRACT

Video-to-music generation has drawn growing interest for its role in conveying the emotion of visual media, including film. Progress in the field, however, is hampered by a reproducibility gap: models are often trained on crawled corpora referenced through YouTube URLs that may be deleted, with the underlying data often difficult and time-consuming to retrieve. To address this, we introduce the Open Screen Soundtrack Library version 2 (OSSL-v2), a self-hosted corpus of 34,343 video clips totaling 246.4 hours, sourced from public-domain films. Unlike crawled corpora, OSSL-v2 is reproducible (i.e., not subject to link rot) and copyrightconscious, yet still large enough to train functional video-tomusic models. We then use this film-domain corpus to study dialogue as a conditioning signal for video-to-music generation, motivated by the close temporal coupling between film music and on-screen speech. Specifically, we augment existing models’ video cross-attention with a time axis and modulate it frame-by-frame with the dialogue track. Evaluated on both public-domain and commercial films, our approach shows improvement over the state-of-the-art baselines. The dataset is available at https://huggingface.co/ datasets/McAuley-Lab/OSSL-v2.

Index Terms— Video-to-Music Generation, Music Generation, Audio Generation

## 1. INTRODUCTION

Music is central to how video communicates: it conveys emotion, establishes mood, and shapes a viewer’s interpretation of a scene [1, 2]. Video-to-music generation systems, which automatically generate music that matches the emotional content of a video [3, 4, 5, 6], have therefore attracted growing interest and remain an important direction for research [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]. However, one problem stands out: the field lacks a common and durable training dataset that can serve as a benchmark.

The absence of such a dataset makes it difficult to measure progress in video-to-music generation, as most systems are trained on web-scraped corpora, often distributed as lists of YouTube or TikTok URLs [17, 18, 19, 20, 9], for two reasons.

First, the data is impermanent: videos are often removed or made private, meaning that models may be trained and evaluated on different, partially unavailable data, making faithful reproduction difficult. Second, the rate-limiting imposed by the host platforms substantially slows down obtaining the dataset. We observed this when attempting to retrieve a publicly released video-to-music benchmark [9] distributed as a list of YouTube IDs. We found that more than 10% of its 300 benchmark videos were no longer accessible because they had been deleted or made private. Notably, this erosion occurred within only about two years of the dataset’s initial release, and the inaccessible fraction can be expected to increase over time. Furthermore, the rate limiting from YouTube makes it time-consuming to re-crawl the full corpus of roughly 360k clips. Although a recent effort distributes self-hosted videomusic pairs [21], its scale remains limited: the dataset contains roughly 36.5 hours across 736 clips, making it difficult to use as a training and evaluation benchmark.

We build upon this self-hosted philosophy at scale by introducing the Open Screen Soundtrack Library version 2 (OSSL-v2), a corpus of 34,343 video clips totaling approximately 246 hours, sourced from public-domain films. Because the dataset is self-hosted rather than referenced by URL, OSSL-v2 is not subject to deletion from third-party platforms and does not require time-consuming web scraping. At the same time, it is large enough to train competitive video-to-music generation models. OSSL-v2 therefore enables different research groups to train and evaluate on exactly the same data, making it suitable as a fixed benchmark for fair comparison.

Beyond data, we show that video-to-music generation may be missing a key input, particularly in the film domain: dialogue. In film, dialogue provides time-local cues about character interaction, narrative emphasis, and emotional density [22], and film music is often shaped around spoken lines [23, 24] — we observe that per-frame dialogue and music loudness (LUFS) are negatively correlated (Pearson r= − 0.11; Fig. 1) in OSSL-v2 as well. These interactions are inherently local in time, and may be difficult to capture with global conditioning alone.

To incorporate dialogue into video-to-music generation, we adapt existing open-source models [9, 12, 16] to use the speech track’s acoustic envelope (e.g., loudness, energy) as a time-aligned conditioning signal during music generation. Because dialogue occurs at specific moments in a scene, the model must preserve when visual events and spoken lines occur relative to the generated audio. We therefore modify the video-conditioning pathway so that visual features retain temporal identity, the cross-attention memory can be aligned with the audio timeline, and dialogue can modulate the conditioning signal frame by frame. Trained and evaluated on OSSLv2, this dialogue-aware formulation improves over state-ofthe-art baselines on both public-domain and commercial film evaluation sets [21], especially in terms of temporal fidelity, i.e., similarity to the ground truth.

![](images/24aec427e12bce50011885f16c1ad0e9972f69f59daeb5393142280b1366a3f5.jpg)  
Fig. 1: Binned ialogue vs. music loudness (LUFS) on OSSL-$\mathbf { v } 2 \left( r = - 0 . 1 1 \right)$

Table 1: Comparison of paired video-music dataset.
<table><tr><td>Dataset</td><td>Self- Hosted</td><td>Video Content</td><td>Length (Hours)</td></tr><tr><td>HIMV-200K [17]</td><td>X</td><td>Music Video, User-Generated Video</td><td></td></tr><tr><td>URMP [25]</td><td>X</td><td>Music Performance</td><td>33.5</td></tr><tr><td>TikTok [18]</td><td>X</td><td>Dance Video</td><td>1.5</td></tr><tr><td>SymMV [19]</td><td>X</td><td>Music Video</td><td>76.5</td></tr><tr><td>MuVi-Sync [26]</td><td>X</td><td>Music Video</td><td></td></tr><tr><td>BGM909 [20]</td><td>X</td><td>Music Video</td><td></td></tr><tr><td>VidMuse [9]</td><td>X</td><td>Music Video, Advertisements, Trailer</td><td>18k</td></tr><tr><td>OSSL [21]</td><td></td><td>Films</td><td>36.5</td></tr><tr><td>OSSL-v2 (ours)</td><td></td><td>Films</td><td>246.4</td></tr></table>

## 2. RELATED WORK

In this section, we review prior work on video-to-music generation and their datasets, with an emphasis on music audio generation rather than symbolic music.

Early work in modern video-to-music generation systems uses large-scale web music-video corpora and autoregressive modeling over semantic acoustic tokens to generate music [8]. Subsequent work has advanced such systems along several dimensions. One direction improves visual representation learning and model architecture, including long- and shortterm visual modeling [9], hierarchical attention [12], dynamic motion or rhythmic modeling [27, 16], a rhythm-aware adaptor [28, 29, 30], parameter-efficient semantics [28], storyboard guidance [30], and semantic planning [31]. A second direction revisits training objectives and supervision by using beat-alignment objectives [10] or leveraging unpaired data, therefore relaxing paired supervision [11, 15]. Another direction introduces domain specialization, such as Chinese-style music generation [14]. Finally, film-oriented approaches have been explored via composition-style transfer [32], a video-adapter to text-to-music generation model [21], integration of different pipelines within a production-oriented framework [33].

Beyond video-only conditioning, another line of work treats video-to-music generation as part of broader multimodal music generation, enabling music creation from images, videos, text, or a combination of modalities [34, 35, 36].

On the other hand, a number of datasets have been developed for video-to-music generation tasks. These span various types of video content, including music videos [17, 19, 29, 20, 9], musical performance recordings [25], and user-generated content [17]. The most closely related work is the Open Screen Soundtrack Library [21], which is a relatively small-scale dataset comprised of music-movie clip pairs sourced from public domain films. In short, most of the video-to-music datasets require a separate web-scraping procedure (i.e., not self-hosted), which makes the dataset prone to deletion and time-consuming to download, or its scale is not suitable for training purposes. Table 1 compares our dataset against publicly available paired video–music datasets. Ours is among the few self-hosted video–music datasets, while also offering a relatively large scale.

## 3. OPEN SCREEN SOUNDTRACK LIBRARY VERSION 2

Dataset Construction. Our video-music dataset, Open Screen Soundtrack Library Version 2 (OSSL-v2), is constructed from 1,886 public domain films downloaded from YouTube. The dataset is self-hosted, meaning that readers do not need to undergo a separate download process such as web scraping. Our dataset construction process consists of two main components: source separation and event detection.

In the first step, in order to extract music from each movie clip’s audio track, we apply an open source cinematic source separation model [37], which offers a high-quality processing option that requires three times longer than the default option. We select the high-quality option because our objective is to create a music-movie clip dataset with the highest possible quality. In the second step, we employ an event detection model to estimate the probability distribution of event types in source-separated musical tracks. This step is essential because source-separated music, even when using a high-quality option, often contains non-musical events. To address this, we use an open-source automatic event detection model [38], from which we manually identify 157 out of 527 categories as musical events (e.g., “trance music”). We define the music probability as the sum of probabilities for the 157 musical events, and the non-music probability as the sum of probabilities for the 370 non-musical events, to source-separated music. We extract segments where the music probability exceeds the non-music probability for at least 10 consecutive seconds. However, this fails to filter out cases where both musical and non-musical events are prominent (e.g., music probability of 0.8 and non-music probability of 0.7). Therefore, we apply an additional filter to exclude cases where the non-music probability exceeds a threshold, which we empirically set to 0.05.

![](images/57a9a3c805dc995b81a32fba55ba0230b1df0cf8abceafac90e516c451ff0822.jpg)  
Fig. 2: Overview of the proposed modification to GVMGen. Grayscale blocks correspond to the original GVMGen model, while colored blocks represent the added components.

This results in 34,343 movie clips paired with music, totaling 246.4 hours, with an average length of 28.6 seconds.

## 4. DIALOGUE-AWARE VIDEO-TO-MUSIC GENERATION

In film, dialogue provides important cues about character interaction, narrative emphasis, and emotional intensity [22]. These cues occur at specific moments in time, so a video-tomusic model should be able to condition the generated score on when dialogue appears and how it aligns with the visual content, rather than relying only on global video-level information. To incoporate this information, we introduce a lightweight dialogue-conditioned module that can be inserted into pen-source video-to-music generation models.

Common Design Let the video conditioning of a backbone be a sequence of vectors $V = [ v _ { 1 } , \ldots , v _ { N } ]$ , where $v _ { i } \in$ $\mathbb { R } ^ { d }$ corresponds to video frame i. For each clip, we isolate the dialogue using the same source separation model used for dataset construction [37] and encode it with a small 1-D convolutional layer, yielding a per-frame low-level acoustic representation of dialogue (e.g., loudness, energy) $u _ { i } \in \mathbb { R } ^ { c }$ temporally aligned to V. A two-layer MLP ϕ maps each dialogue representation to Feature-wise Linear Modulation (FiLM) [39] parameters $( \gamma _ { i } , \beta _ { i } ) = \phi ( u _ { i } )$ , which modulate the corresponding video-conditioning vector,

$$
\tilde { v } _ { i } = ( 1 + \gamma _ { i } ) \odot v _ { i } + \beta _ { i } ,\tag{1}
$$

where ⊙ denotes element-wise multiplication. The final layer of ϕ is zero-initialized, so that the module is initialized as an identity function at the start of training. How this condition is met differs across the backbones.

VidMuse [9] is a MusicGen [40]-based autoregressive model that encodes video with a local branch, co-segmented with the target audio. Because it embeds each frame with CLIP [41] along the time axis and adds a learned positional embedding, its cross-attention conditioning sequence is already temporarily order. We therefore apply Equation 1 directly to the sequence.

GVMGen [12] is a MusicGen [40]-based autoregressive Transformer that predicts discrete audio tokens conditioned on video through cross-attention. In the original model, each frame’s CLIP [41] patch tokens are compressed by a Q-Former [42] and then mean-pooled to a single vector per frame, and the resulting per-frame vectors are used as cross-attention conditioning without positional embeddings. Therefore, Applying Equation 1 directly to this memory would collapse to a global effect, because the memory has no time axis. We therefore restore one with two further zero-initialized components. First, we add a learned perframe positional embeddings to the Q-Former outputs before FiLM modulation (Equation 1), giving each frame a distinct positional identity. Second, we enable a fixed sinusoidal positional embedding on the cross-attention keys and values, allowing the audio-to-video attention to use this temporal identity.

Diff-V2M [16] is a Stable Audio Open [43]-based latent diffusion model whose visual semantic features condition the generator through cross-attention on a temporal grid. We apply Equation 1 to the per-frame semantic features, whith the FiLM module initialized as an identity function.

## 5. EXPERIMENTS

The goal of our experiments are two-fold. First, we benchmark video-to-music generation tasks by comparing existing video-to-music generation models trained and evaluated on identical data, using OSSL-v2. Second, we assess the effectivenss of the proposed dialogue adapter when applied to these models.

Compared Models. We comprehensively benchmark publicly available video-to-music generation models that learn from paired video-music data, namely VidMuse [9], GVMGen [12], and Diff-V2M [16], together with the variants obtained by augmenting each model with the dialogue adapter described in Section 4. We exclude Sonique [11] although it is an open-source video-to-music generation model, as it is designed to exploit unpaired data, making it incomparable under our protocol. For every model we adopt the exact hyperparameter settings recommended in the corresponding public repository, including the number of training steps, batch size, learning rate, etc.; where implementation details are missing from the released code, we follow the descriptions in the respective papers. One gap is the training segment length. Although all these models are originally trained on 30-second segments, we set the training length to 10 seconds, since the shortest clips in OSSL-v2 are 10 seconds long, and randomly crop a 10-second window from clips that exceed this length. At inference time, however, each model is required to generate a soundtrack matching the full duration of the input video clip.

Dataset We partition OSSL-v2 into training and test splits with a 9:1 ratio, and all models are trained and evaluated on exactly the same splits. We report results on the test set as well as OES-Com [21], a set of one hundred 30-second movie clips drawn from commercial films. The former reflects how well each model fits the training distribution, whereas the latter assesses generalization to out-of-distribution data. In addition, we evaluate on low-residual-music (LRM) subsets of both evaluation sets to assess whether any benefit from dialogue conditioning is merely an artifact of residual music leaking into the separated dialogue stem under imperfect source separation. Specifically, we run an audio event detection model [38] on each clip’s separated dialogue stem and retain only those clips whose summed probabilities over the 157 music event labels fall below 0.05 (i.e., the average probability is less than ≈ 0.0003). This yields 878 clips from the OSSL-v2 test set and 20 clips from OES-Com. If the dialogue adapter yields comparable improvements on these lowresidual-music subsets and on the full evaluation sets, we can conclude that leftover music in the separated dialogue stem contributes little to the observed gains.

Evaluation Metrics Following prior work [21], we evaluate generation quality along three axes. We measure distributional fidelity, i.e., how closely the distribution of generated music matches that of the reference music, using FAD [44] and Precision [45], as well as paired fidelity, i.e., how semantically faithful each generated track is to its corresponding ground-truth music, using CLAP [46] similarity and the PaSST [47]-based KL divergence between the predicted audio-event distributions of the generated and reference tracks. Finally, we measure diversity using Recall [45]. Following prior work [21], the reference-based metrics (FAD, Precision, and Recall) are computed against a reference music distribution drawn from 5,000 commercial film soundtracks that we privately crawled <sup>1</sup>

## 6. RESULTS

Table 2 reports the objective evaluation results for all compared models on both the public-domain test set and the commercial OES-Com set.

Benchmarking video-to-music models. On the publicdomain OSSL-v2 test set, GVMGen achieves the strongest overall performance, followed by VidMuse and Diff-V2M. Diff-V2M is a notable outlier in distributional fidelity: its Precision collapses to 0.00 and its FAD reaches a substantially higher value than the other models, indicating that its generated samples fall almost entirely outside the reference distribution. On the out-of-distribution OES-Com set, Diff-V2M attains the strongest paired fidelity, with the lowest KL and CLAP similarity on par with GVMGen. However, it again attains a Precision of 0.00, in contrast to the non-trivial Precision achieved by VidMuse (0.17) and GVMGen (0.53). We interpret this as evidence that Diff-V2M captures coarse semantic correspondences between video and music but fails to generate samples that conform to the ground-truth distribution. Taken together, these results establish GVMGen as the strongest model under our evaluation protocol.

Effect of the dialogue adapter. We next examine the impact of augmenting each backbone with the proposed dialogue adapter (the “+Dialogue” rows in Table 2). The most consistent gains appear in paired fidelity. For VidMuse, the adapter sharply reduces KL divergence on both evaluation sets (OES-Com: 1.47→0.85; OSSL-v2: 0.94→0.82) while nudging CLAP similarity upward (OES-Com: 0.19→0.23; OSSL-v2: 0.35→0.36); Diff-V2M shows the same qualitative pattern, with CLAP rising on both sets (OES-Com: 0.23→0.25; OSSL-v2: 0.26→0.27) and KL improving on OSSL-v2 (0.75→0.68). GVMGen is a partial exception: its paired fidelity improves on the out-of-distribution OES-Com set (KL: 0.97→0.87) but degrades on the in-distribution OSSL-v2 set (CLAP: 0.43→0.39; KL: 0.72→0.73). Notably, the paired-fidelity gains are generally more pronounced on OES-Com than on OSSL-v2, suggesting that dialogue conditioning acts as a regularizer that generalizes beyond the training domain. In contrast, the adapter has no consistent effect on distributional fidelity: both Precision and FAD move in either direction across models and sets, indicating that dialogue conditioning does not systematically reshape the generated distribution itself.

The same behaviors persist on the low-residual-music (LRM) subsets, where the separated dialogue stem is almost free of leaked music. Across all three backbones, the direction and rough magnitude of every effect above are preserved on LRM—VidMuse’s KL still drops sharply (OES-Com: 1.58→0.89; OSSL-v2: 0.89→0.80), GVMGen still improves on the OOD set (KL: 0.94→0.77) while regressing on OSSLv2, and its in-distribution paired fidelity still declines (CLAP: 0.43→0.39). Because these behaviors remain when residual music is minimal, we attribute them to the dialogue signal itself rather than to music leaking into the dialogue stem.

Table 2: Objective evaluation on the OSSL-v2 public-domain test set and the commercial OES-Com set, on the full evaluation sets (All) and on the low-residual-music subsets (LRM: clips whose separated dialogue stem has summed probability < 0.05 over the 157 PANNs music-event labels, i.e. average probability < 0.0003; 20/100 OES-Com clips and 878/3,332 OSSL-v2 clips). ↑ / ↓ indicate that higher / lower is better.
<table><tr><td></td><td colspan="5">OES-Com (Al1, n=100)</td><td colspan="5">OES-Com (LRM, n=20)</td><td colspan="5">OSSL-v2 (All, n=3,332)</td><td colspan="5">OSSL-v2 (LRM, n=878)</td></tr><tr><td></td><td>Sim↑</td><td>KL↓</td><td>P↑</td><td>FAD↓</td><td>R↑</td><td>Sim↑</td><td>KL↓</td><td>P↑</td><td>FAD↓</td><td>R↑</td><td>Sim↑</td><td>KL↓</td><td>P↑</td><td>FAD↓</td><td>R↑</td><td>Sim↑</td><td>KL↓</td><td>P↑</td><td>FAD↓</td><td>R↑</td></tr><tr><td>VidMuse [9]</td><td>0.19</td><td>1.47</td><td>0.17</td><td>93.64</td><td>0.00</td><td>0.19</td><td>1.58</td><td>0.10</td><td>109.04</td><td>0.09</td><td>0.35</td><td>0.94</td><td>0.38</td><td>65.57</td><td>0.00</td><td>0.34</td><td>0.89</td><td>0.43</td><td></td><td>0.00</td></tr><tr><td>(+Dialogue)</td><td>0.23</td><td>0.85</td><td>0.04</td><td>97.70</td><td>0.03</td><td>0.22</td><td>0.89</td><td>0.10</td><td>113.24</td><td>0.00</td><td>0.36</td><td>0.82</td><td>0.32</td><td>66.12</td><td>0.00</td><td>0.35</td><td>0.80</td><td>0.45</td><td>64.04 66.89</td><td>0.00</td></tr><tr><td>GVMGen [12]</td><td>0.23</td><td>0.97</td><td>0.53</td><td>73.40</td><td>0.17</td><td>0.23</td><td>0.94</td><td>0.55</td><td>89.81</td><td>0.03</td><td>0.43</td><td>0.72</td><td>0.56</td><td>49.05</td><td>0.22</td><td>0.43</td><td>0.72</td><td>0.56</td><td>48.98</td><td>0.25</td></tr><tr><td>(+Dialogue)</td><td>0.22</td><td>0.87</td><td>0.57</td><td>77.29</td><td>0.01</td><td>0.24</td><td>0.77</td><td>0.50</td><td>90.17</td><td>0.04</td><td>0.39</td><td>0.73</td><td>0.42</td><td>56.16</td><td>0.11</td><td>0.39</td><td>0.73</td><td>0.44</td><td>55.98</td><td>0.13</td></tr><tr><td>Diff-V2M [16]</td><td>0.23</td><td>0.62</td><td>0.00</td><td>129.30</td><td>0.00</td><td>0.21</td><td>0.64</td><td>0.00</td><td>137.38</td><td>0.00</td><td>0.26</td><td>0.75</td><td>0.00</td><td>119.68</td><td>0.00</td><td>0.27</td><td>0.73</td><td>0.00</td><td>120.22</td><td>0.00</td></tr><tr><td>(+Dialogue)</td><td>0.25</td><td>0.63</td><td>0.00</td><td>130.72</td><td>0.00</td><td>0.23</td><td>0.68</td><td>0.00</td><td>137.45</td><td>0.00</td><td>0.27</td><td>0.68</td><td>0.00</td><td>118.87</td><td>0.00</td><td>0.28</td><td>0.64</td><td>0.00</td><td>119.41</td><td>0.00</td></tr></table>

## 7. CONCLUSION

We introduced OSSL-v2, a large-scale, self-hosted corpus of music-film clip pairs sourced from public-domain films. Because the dataset is free from link rot and does not require separate web scraping, our dataset is suitable as a durable benchmark for the field. Beyond the dataset, we showed that dialogue is an informative conditioning signal for music generation for film clips. By augmenting video-to-music generation models with frame-by-frame dialogue adapters along time axis, we obtain improvements over the corresponding baselines. We hope that OSSL-v2 lowers the barrier to reproducible research and fair comparison in video-to-music generation, and thereby supports the continued development of the filed, while our dialogue-aware experiments point to one promising direction for future work.

## 8. ACKNOWLEDGEMENT

This work is partially supported by the NVIDIA Academic Grant Program under the project titled “Teaser Generation for Long Documentaries and Educational Videos”.

## 9. REFERENCES

[1] Ha Thi Phuong Thao, BT Balamurali, Gemma Roig, and Dorien Herremans, “Attendaffectnet–emotion prediction of movie viewers using multimodal fusion with self-attention,” Sensors, vol. 21, no. 24, pp. 8356, 2021.

[2] Phoebe Chua, Dimos Makris, Dorien Herremans, Gemma Roig, and Kat Agres, “Predicting emotion from music videos: exploring the relative contribution of visual and auditory information to affective responses,” arXiv preprint arXiv:2202.10453, 2022.

[3] Marilyn G Boltz, “The cognitive processing of film and musical soundtracks,” Memory & Cognition, vol. 32, no. 7, pp. 1194–1205, 2004.

[4] Ha Thi Phuong Thao, Dorien Herremans, and Gemma Roig, “Multimodal deep models for predicting affective responses evoked by movies.,” in ICCV Workshops, 2019, pp. 1618–1627.

[5] Ann-Kristin Herget, “On music’s potential to convey meaning in film: A systematic review of empirical evidence,” Psychology ofMusic, vol. 49, no. 1, pp. 21–49, 2021.

[6] Minz Won, Justin Salamon, Nicholas J Bryan, Gautham J Mysore, and Xavier Serra, “Emotion embedding spaces for matching music to stories,” arXiv preprint arXiv:2111.13468, 2021.

[7] Chuang Gan, Deng Huang, Peihao Chen, Joshua B Tenenbaum, and Antonio Torralba, “Foley music: Learning to generate music from videos,” in Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XI 16. Springer, 2020, pp. 758–775.

[8] Kun Su, Judith Yue Li, Qingqing Huang, Dima Kuzmin, Joonseok Lee, Chris Donahue, Fei Sha, Aren Jansen, Yu Wang, Mauro Verzetti, et al., “V2meow: Meowing to the visual beat via video-to-music generation,” in Proceedings of the AAAI Conference on Artificial Intelligence, 2024, vol. 38, pp. 4952–4960.

[9] Zeyue Tian, Zhaoyang Liu, Ruibin Yuan, Jiahao Pan, Qifeng Liu, Xu Tan, Qifeng Chen, Wei Xue, and Yike Guo, “Vidmuse: A simple video-to-music generation framework with long-short-term modeling,” 2025.

[10] Yan-Bo Lin, Yu Tian, Linjie Yang, Gedas Bertasius, and Heng Wang, “Vmas: Video-to-music generation via semantic alignment in web music videos,” in 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). IEEE, 2025, pp. 1155–1165.

[11] Liqian Zhang and Magdalena Fuentes, “Sonique: Video background music generation using unpaired audiovisual data,” in ICASSP 2025-2025 IEEE International

Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2025, pp. 1–5.

[12] Heda Zuo, Weitao You, Junxian Wu, Shihong Ren, Pei Chen, Mingxu Zhou, Yujia Lu, and Lingyun Sun, “Gvmgen: A general video-to-music generation model with hierarchical attentions,” arXiv preprint arXiv:2501.09972, 2025.

[13] Zhaokai Wang, Chenxi Bao, Le Zhuo, Jingrui Han, Yang Yue, Yihong Tang, Victor Shea-Jay Huang, and Yue Liao, “Vision-to-music generation: A survey,” arXiv preprint arXiv:2503.21254, 2025.

[14] Moxi Cao, Jiaxiang Zheng, and Chongbin Zhang, “Aibased chinese-style music generation from video content: a study on cross-modal analysis and generation methods,” EURASIP Journal on Audio, Speech, and Music Processing, vol. 2025, no. 1, pp. 8, 2025.

[15] Yan-Bo Lin, Jonah Casebeer, Long Mai, Aniruddha Mahapatra, Gedas Bertasius, and Nicholas J Bryan, “V2mzero: Zero-pair time-aligned video-to-music generation,” arXiv preprint arXiv:2603.11042, 2026.

[16] Shulei Ji, Zihao Wang, Jiaxing Yu, Xiangyuan Yang, Shuyu Li, Songruoyao Wu, and Kejun Zhang, “Diffv2m: A hierarchical conditional diffusion model with explicit rhythmic modeling for video-to-music generation,” in Proceedings ofthe AAAI Conference on Artificial Intelligence, 2026, vol. 40, pp. 22219–22227.

[17] Sungeun Hong, Woobin Im, and Hyun S. Yang, “Content-based video-music retrieval using soft intramodal structure constraint,” arXiv:1704.06761, 2017.

[18] Ye Zhu, Kyle Olszewski, Yu Wu, Panos Achlioptas, Menglei Chai, Yan Yan, and Sergey Tulyakov, “Quantized gan for complex music generation from dance videos,” 2022.

[19] Le Zhuo, Zhaokai Wang, Baisen Wang, Yue Liao, Chenxi Bao, Stanley Peng, Songhao Han, Aixi Zhang, Fei Fang, and Si Liu, “Video background music generation: Dataset, method and evaluation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 15637–15647.

[20] Sizhe Li, Yiming Qin, Minghang Zheng, Xin Jin, and Yang Liu, “Diff-bgm: A diffusion model for video background music generation,” arXiv:1704.06761, 2024.

[21] Haven Kim, Zachary Novack, Weihan Xu, Julian McAuley, and Hao-Wen Dong, “Video-guided textto-music generation using public domain movie collections,” arXiv preprint arXiv:2506.12573, 2025.

[22] Rainer Banse and Klaus R Scherer, “Acoustic profiles in vocal emotion expression.,” Journal of personality and social psychology, vol. 70, no. 3, pp. 614, 1996.

[23] Matteo Torcoli, Alex Freke-Morin, Jouni Paulus, Christian Simon, and Ben Shirley, “Background ducking to produce esthetically pleasing audio for tv with clear speech,” in Audio Engineering Society Convention 146. Audio Engineering Society, 2019.

[24] Tim Addy, Scott Norcross, Mike Ward, Pau Arumi, and Heidi-Maria Lehtonen, “Improving dialogue intelligibility in streaming media,” in Proceedings of the 5th Mile-High Video Conference, 2026, pp. 34–35.

[25] Bochen Li, Xinzhao Liu, Karthik Dinesh, Zhiyao Duan, and Gaurav Sharma, “Creating a multitrack classical music performance dataset for multimodal music analysis: Challenges, insights, and applications,” IEEE Transactions on Multimedia, vol. 21, no. 2, pp. 522–535, Feb. 2019.

[26] Jaeyong Kang, Soujanya Poria, and Dorien Herremans, “Video2music: Suitable music generation from videos using an affective multimodal transformer model,” Expert Systems with Applications, vol. 249, pp. 123640, Sept. 2024.

[27] Xiaohao Liu, Teng Tu, Yunshan Ma, and Tat-Seng Chua, “Extending visual dynamics for video-to-music generation,” arXiv:2504.07594, 2025.

[28] Sifei Li, Binxin Yang, Chunji Yin, Chong Sun, Yuxin Zhang, Weiming Dong, and Chen Li, “Vidmusician: Video-to-music generation with semanticrhythmic alignment via hierarchical visual features,” arXiv:2412.06296, 2024.

[29] Ruiqi Li, Siqi Zheng, Xize Cheng, Ziang Zhang, Shengpeng Ji, and Zhou Zhao, “Muvi: Video-to-music generation with semantic alignment and rhythmic synchronization,” arXiv:2410.12957, 2024.

[30] Xinyi Tong, Yiran Zhu, Jishang Chen, Chunru Zhan, Tianle Wang, Sirui Zhang, Nian Liu, Tiezheng Ge, Duo Xu, Xin Jin, Feng Yu, and Song-Chun Zhu, “Video echoed in music: Semantic, temporal, and rhythmic alignment for video-to-music generation,” arXiv:2511.09585, 2025.

[31] Vaibhavi Lokegaonkar, Aryan Vijay Bhosale, Vishnu Raj, Gouthaman KV, Ramani Duraiswami, Lie Lu, Sreyan Ghosh, and Dinesh Manocha, “Video-robin: Autoregressive diffusion planning for intent-grounded video-to-music generation,” arXiv:2604.17656, 2026.

[32] F. Qi, L. Ni, and C. Xu, “Harmonizing pixels and melodies: Maestro-guided film score generation and composition style transfer,” arXiv:2411.07539, 2024.

[33] Zhifeng Xie, Qile He, Youjia Zhu, Qiwei He, and Mengtian Li, “Filmcomposer: Llm-driven music production for silent film clips,” arXiv:2503.08147, 2025.

[34] Shansong Liu, Atin Sakkeer Hussain, Qilong Wu, Chenshuo Sun, and Ying Shan, “M<sup>2</sup>ugen: Multi-modal music understanding and generation with the power of large language models,” arXiv:2311.11255, 2024.

[35] Shansong Liu, Atin Sakkeer Hussain, Qilong Wu, Chenshuo Sun, and Ying Shan, “Mumu-llama: Multi-modal music understanding and generation via large language models,” arXiv:2412.06660, 2024.

[36] Baisen Wang, Le Zhuo, Zhaokai Wang, Chenxi Bao, Wu Chengjing, Xuecheng Nie, Jiao Dai, Jizhong Han, Yue Liao, and Si Liu, “Multimodal music generation with explicit bridges and retrieval augmentation,” arXiv preprint arXiv:2412.09428, 2024.

[37] Roman Solovyev, Alexander Stempkovskiy, and Tatiana Habruseva, “Benchmarks and leaderboards for sound demixing tasks,” arXiv:2305.07489, 2023.

[38] Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, and Mark D Plumbley, “Panns: Largescale pretrained audio neural networks for audio pattern recognition,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 28, pp. 2880– 2894, 2020.

[39] Ethan Perez, Florian Strub, Harm De Vries, Vincent Dumoulin, and Aaron Courville, “Film: Visual reasoning with a general conditioning layer,” in Proceedings ofthe AAAI conference on artificial intelligence, 2018, vol. 32.

[40] Jade Copet, Felix Kreuk, Itai Gat, Tal Remez, David Kant, Gabriel Synnaeve, Yossi Adi, and Alexandre Defossez, “Simple and controllable music generation,”´ Advances in neural information processing systems, vol. 36, pp. 47704–47720, 2023.

[41] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PmLR, 2021, pp. 8748–8763.

[42] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi, “Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models,” in International conference on machine learning. PMLR, 2023, pp. 19730–19742.

[43] Zach Evans, Julian D Parker, CJ Carr, Zack Zukowski, Josiah Taylor, and Jordi Pons, “Stable audio open,” in ICASSP 2025-2025 IEEE International Conference

on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2025, pp. 1–5.

[44] Kevin Kilgour, Mauricio Zuluaga, Dominik Roblek, and Matthew Sharifi, “Fr\’echet audio distance: A metric for evaluating music enhancement algorithms,” arXiv preprint arXiv:1812.08466, 2018.

[45] Muhammad Ferjad Naeem, Seong Joon Oh, Youngjung Uh, Yunjey Choi, and Jaejun Yoo, “Reliable fidelity and diversity metrics for generative models,” in International conference on machine learning. PMLR, 2020, pp. 7176–7185.

[46] Yusong Wu, Ke Chen, Tianyu Zhang, Yuchen Hui, Taylor Berg-Kirkpatrick, and Shlomo Dubnov, “Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation,” in ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2023, pp. 1–5.

[47] Khaled Koutini, Jan Schluter, Hamid Eghbal-Zadeh, and¨ Gerhard Widmer, “Efficient training of audio transformers with patchout,” arXiv preprint arXiv:2110.05069, 2021.