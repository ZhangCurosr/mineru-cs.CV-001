# InstaBind-Lite: Diagnosing Dense Same-Class Attribute Misbinding in Large Vision-Language Models

Yuanzhi Xu, Qian Gao, Jun Fan, Guohui Ding, Zhenyu Yang, Yuteng Xiao, and Sixue Lin

Abstract—Large vision-language models can recognize the objects and attributes in a crowded scene yet assign an attribute to the wrong same-class instance. Generic visual-questionanswering accuracy marks the response as wrong, while objecthallucination metrics may regard both the object and attribute as image-supported; neither reveals the transfer. This study formalizes this blind spot as Dense Same-Class Attribute Misbinding and presents InstaBind-Lite, a controlled benchmark that makes it directly measurable. Its 524 images contain 529 curated groups of 3–6 same-class entities, 1773 boxed instances, ordered neighbors, distinguishable color-like attributes, and four complementary question levels, yielding 9580 deterministically evaluated questions. Unlike existing protocols, source-instance annotations separate unsupported generation and recognition failure from an attribute copied from another visible entity. Binding-specific metrics further quantify transfer frequency, adjacency, ordinal distance, and intervention effects. Across five open-source and two commercial/API models, the opensource systems average 19.84% Misbinding Rate and the API systems 7.55%; these errors are hidden by aggregate accuracy. Among identifiable transfers, 80.70% and 81.51%, respectively, originate from adjacent instances. Localization and instance-first interventions help selected models but are not universal remedies. InstaBind-Litetherefore turns previously undifferentiated wrong answers into source-identifiable failure categories and tests a reliability dimension that conventional benchmarks cannot determine: whether a model knows not only what is visible, but which instance owns each attribute.

Index Terms—Large vision-language models, visual hallucination, visual grounding, attribute binding, benchmark.

## I. INTRODUCTION

Large vision-language models (LVLMs) now support image search, visual assistance, traffic analysis, retail inspection, and robotic interaction [1]–[5]. Their errors, however, have different causes and risks: a model may invent absent content, miss an object, misread an attribute or relation, or attach a correctly perceived attribute to the wrong entity. Existing benchmarks often collapse these outcomes into the same incorrect answer, so improved aggregate accuracy does not establish reliable instance–attribute correspondence [6]–[9]. A correct color attached to the wrong vehicle, person, product, or container can still produce unsafe entity-specific retrieval or action.

Public incidents outside LVLM benchmarking illustrate the stakes of visual attribution failure. A traffic-recognition system in Ningbo reportedly treated a face printed on a bus advertisement as a jaywalking pedestrian [10]; the U.S. National Transportation Safety Board found that the developmental automated-driving system in the 2018 Tempe crash repeatedly changed its classification of a pedestrian before the fatal collision [11]. Neither incident establishes DSCAM or evaluates an LVLM, but both show why visually supported content is insufficient when identity, location, and attributes must remain attached to the correct entity. The analogous LVLM error can direct a user to the wrong person, product, vehicle, or waste stream.

Generic visual-question-answering benchmarks collapse these cases into the same zero-accuracy event, whereas objecthallucination benchmarks mainly test whether predicted content is supported somewhere in the image [7]–[9], [12]. They therefore cannot determine whether a multimedia system preserves which entity has which property.

This study defines the hidden failure as Dense Same-Class Attribute Misbinding (DSCAM): a wrong answer copied from another visible same-class instance. The hypothesis is structural: under same-class competition, transfers should disproportionately originate from neighbors rather than uniformly random sources. Identifying that source separates binding failure from recognition error and gives grounding, decoding, and training methods a specific repair target.

Existing resources lack the joint annotations needed to test this hypothesis. General VQA does not guarantee inspectable same-class groups with distinguishable attributes; grounding evaluates localization but not which neighbor supplied a wrong attribute; and compositionality benchmarks do not measure directed instance-to-instance transfer in natural dense scenes. Their standard scores therefore cannot recover DSCAM prevalence, source, or distance.

InstaBind-Litesupplies that missing capability through ordered same-class groups, visible color-like attributes, boxes, and neighbor links. Four levels probe position-to-attribute, attribute-to-position, relation-mediated binding, and proposition verification. A controlled vocabulary and source annotations support deterministic parsing into adjacent or nonadjacent misbinding, recognition failure, and out-of-set hallucination. Across seven LVLMs, this design reveals measurable MBR even at high aggregate accuracy and shows that transfers overwhelmingly follow neighbor structure.

The contributions are threefold:

• Dense Same-Class Attribute Misbindingis formalized as a distinct failure category, resolving the evaluation ambiguity between unsupported hallucination, attribute recognition failure, and a visible attribute transferred to the wrong same-class instance.

• InstaBind-Liteis constructed as a high-purity diagnostic benchmark with 524 images, 1773 instances, and 9580 questions. Its controlled groups and source-instance annotations make wrong-instance binding measurable in a way that aggregate VQA and object-presence scores do not.

• Binding-oriented metrics and controlled interventions are introduced. MBR identifies visible wrong-instance transfers, A-MBR and ordinal Distance-MBR locate their spatial source, while crop/context localization and instancefirst prompting test whether reducing visual competition or explicitly enumerating instance–attribute pairs mitigates the failure.

## II. RELATED WORK

Table I separates benchmark breadth from diagnostic observability. The comparison does not imply that a focused benchmark is generally superior to broad evaluation; it identifies which protocols can trace a wrong attribute to a competing same-class source instance.

## A. LVLM Hallucination Evaluation

CHAIR identifies absent objects in captions [12]; POPE tests object existence [7]; AMBER provides judge-free multidimensional analysis [8]; and HallusionBench couples language hallucination with visual illusion [9]. Detection and decoding methods likewise target unsupported generation [13], [14]. DSCAM is different: both class and attribute are imagesupported, but their instance correspondence is false. Presencebased protocols either miss this event or count it only as an unspecified error.

B. Visual Question Answering and General Multimodal Benchmarks

VQA, VQA v2, and Visual7W evaluate image-conditioned answers and reduce language bias or add region grounding [6], [15], [16]. COCO, Visual Genome, GQA, and VAW provide categories, boxes, scene graphs, relations, and attributes [17]– [20]; MME, SEED-Bench, MM-Vet, and MMBench broaden capability coverage [21]–[24]. Their standard protocols do not jointly control same-class density, attribute uniqueness, ordered neighbors, and wrong-source attribution. An adjacent transfer thus receives the same score as an absent-color guess. InstaBind-Litetrades breadth for controlled evidence that separates these risks.

## C. Visual Grounding and Referring Expressions

Flickr30K Entities, ReferItGame, and RefCOCO associate expressions with image regions, while MAttNet decomposes references into subject, location, and relation modules [25]– [28]. Their output is typically a region scored by localization overlap. DSCAM instead tests attribution after or alongside selection: a model may locate the target yet answer with its neighbor’s attribute. InstaBind-Litetherefore retains boxes for interventions but adds ordered attributes and source-instance error labels that overlap alone cannot provide.

## D. Compositionality and Attribute–Relation Binding

CLEVR studies compositional reasoning in synthetic scenes [29]; Winoground contrasts image–caption pairs with shared words but different compositions [30]; and ARO, VL-CheckList, CREPE, and SugarCrepe test attributes, relations, and word order [31]–[34]. Winoground’s minimalpair principle motivates separating component recognition from composition in InstaBind-Lite: the same-class set and vocabulary remain fixed while the queried instance changes. ARO’s relation sensitivity motivates L3 and ordinal-distance analysis. Thus, natural-image composition becomes a directed transfer from target i to source j, and MBR connects failure to a competing entity rather than only an incorrect global match.

## E. Binding Mechanisms and Multi-Subject Misbinding

Mechanistic studies examine object–reference binding and its distribution across vision encoders and language backbones [35], [36]; MultiBind studies cross-subject attribute transfer in image generation [37]. This work asks the complementary behavioral question of whether an LVLM reading a natural image assigns an observed attribute to the correct same-class instance. Spatial signatures and interventions can guide, but do not replace, mechanistic analysis.

## III. PROBLEM DEFINITION

Let an image contain a same-class group

$$
G = \{ e _ { 1 } , e _ { 2 } , \ldots , e _ { n } \} ,\tag{1}
$$

where all entities share class c and have a spatial order, $\mathrm { e . g . }$ ., left-to-right. Each entity $e _ { i }$ has an attribute value $a _ { i }$ for an attribute type such as color or upper-clothing color.

A question specifies a target entity $e _ { t }$ directly or indirectly. In L1, the target is specified by position and the answer is its attribute. In L2, the attribute is specified and the answer is the target position. In L3, the target is obtained through a local relation such as left or right neighbor. In L4, the model verifies an instance-attribute proposition. A model prediction is correct if it matches the canonical answer after answer normalization.

If the prediction is incorrect but matches the attribute or position of another same-class entity $e _ { j }$ , the error is classified as a misbinding:

$$
\hat { a } = a _ { j } , \quad j \neq t .\tag{2}
$$

The instance distance of a misbinding is defined by the absolute order difference:

$$
d ( t , j ) = | \mathrm { o r d e r } ( e _ { t } ) - \mathrm { o r d e r } ( e _ { j } ) | .\tag{3}
$$

An adjacent misbinding occurs when $d ( t , j ) = 1$

This definition intentionally separates misbinding from outof-set hallucination. If a model predicts an attribute that does not appear in the same-class group, the error is counted as outof-set hallucination rather than misbinding. This distinction tests whether a wrong answer is grounded in the image but attached to the wrong instance.

The key diagnostic question is therefore not only whether the model answers correctly, but what kind of wrong answer it gives. A random attribute error, an out-of-image hallucination, and a neighbor’s attribute are all incorrect under accuracy, but they imply different failure mechanisms. Dense Same-Class Attribute Misbindingfocuses on the last case.

TABLE I  
COMPARISON BY DIAGNOSTIC OBJECTIVE. “PARTIAL” INDICATES THAT SAME-CLASS INSTANCES MAY OCCUR NATURALLY, BUT ARE NOT JOINTLY CONTROLLED WITH SOURCE-INSTANCE ATTRIBUTION FOR MISBINDING ANALYSIS.
<table><tr><td>Evaluation family</td><td>Primary diagnostic target</td><td>Controlled same-class groups</td><td>Wrong-source attribution</td><td>Spatial misbinding metrics</td></tr><tr><td>Object hallucination</td><td>Unsupported objects or attributes</td><td>No</td><td>No</td><td>No</td></tr><tr><td>General VQA / multimodal</td><td>End-answer correctness and broad ca- Partial pability</td><td></td><td>No</td><td>No</td></tr><tr><td>Referring / grounding</td><td>Phrase-to-region localization</td><td>Partial</td><td>No</td><td>No</td></tr><tr><td>Compositionality</td><td>Attribute-object and relation sensitiv- Partial ity</td><td></td><td>No</td><td>No</td></tr><tr><td>InstaBind-Lite</td><td>Same-class instance-attribute binding</td><td>Yes</td><td>Yes</td><td>Yes</td></tr></table>

TABLE II  
DATASET STATISTICS OF INSTABIND-LITE V0.4.
<table><tr><td>Item</td><td>Value</td></tr><tr><td>Images</td><td>524</td></tr><tr><td>Same-class groups</td><td>529</td></tr><tr><td>Instances</td><td>1773</td></tr><tr><td>Questions</td><td>9580</td></tr><tr><td>Person groups</td><td>36.86%</td></tr><tr><td>Non-person groups</td><td>63.14%</td></tr><tr><td>Group size 3/4/5/6</td><td>391  / 98 / 32 / 8</td></tr><tr><td>Question L1/L2/L3/L4</td><td>1773 /  1773 /  2488 /  3546</td></tr><tr><td>Sources: COCO/GQA/VAW</td><td>205  /  21  /  4</td></tr><tr><td>Sources: web/self-shot</td><td>277  /  17</td></tr></table>

## IV. INSTABIND-LITEDATASET

InstaBind-Liteis a high-purity diagnostic benchmark rather than a general-purpose VQA dataset. Each image is curated to contain 3–6 same-class entities with clear order, visible attributes, and low ambiguity, making targets and neighbors human-inspectable. Table II summarizes the 524 images, 529 groups, 1773 instances, and 9580 questions.

## A. Data Sources

Images come from COCO [17], GQA [19], VAW [20], manually selected open-license web images, and self-shot images. Manual verification removes incomplete groups, severe occlusion, strong reflection, tiny objects, and uncertain attributes. Source metadata and original licenses remain separate from benchmark annotations.

## B. Annotation Schema

Each group records its image, class, spatial order, boxes, attributes, and neighbor links. Boxes support quality control and interventions; questions use natural-language positions and relations. Person questions use upper-clothing color, and transparent or multicolor labels are retained only when unambiguous.

## C. Attribute Scope and Design Trade-off

The benchmark focuses on color and upper-clothing color: local attributes shared across classes and expressible with a compact vocabulary. This enables source attribution without an LLM judge; L2 additionally requires within-group attribute uniqueness. The scope is a controlled slice of binding and does not assume unchanged rates for actions, materials, textures, shapes, or states.

## D. Question Levels

The benchmark generates four levels of questions:

• L1: position-to-attribute, e.g., “What color is the leftmost car?”

• L2: attribute-to-position, e.g., “Where is the red car?”

• L3: relation-interference, e.g., “What color is the car to the right of the red car?”

• L4: instance verification, e.g., “Is the middle car red?”

L2 questions require the queried attribute to be unique within the same-class group.

Together, L1 tests position-conditioned reading, L2 reverse binding, L3 local relational interference, and L4 proposition verification, separating perception errors from instance-level transfer.

## V. METRICS

## A. Accuracy

Accuracy measures whether the normalized model prediction matches the canonical answer. It provides the standard task-level score but does not explain how wrong answers relate to the image content.

## B. Misbinding and Adjacency

Let Q, W, M, and A denote all questions, wrong answers, misbindings, and adjacent misbindings. The three rates are

$$
\mathrm { M B R } = { \frac { | { \mathcal { M } } | } { | { \mathcal { Q } } | } } , \quad \mathrm { E r r - M B R } = { \frac { | { \mathcal { M } } | } { | { \mathcal { W } } | } } , \quad \mathrm { A \cdot M B R } = { \frac { | { \mathcal { A } } | } { | { \mathcal { M } } | } } .\tag{4}
$$

MBR measures transfer frequency, error-conditioned MBR its share among wrong answers, and A-MBR whether transfers follow within-group neighbor structure.

## C. Out-of-Set Hallucination

Out-of-set hallucination measures predictions that do not match any same-class instance attribute or position. This separates ungrounded attributes from in-image but misbound attributes. A model can therefore have a low object hallucination profile while still exhibiting substantial instance-level misbinding.

## D. Distance-MBR and Confusion Matrix

Distance-MBR analyzes the distribution of misbinding over the ordinal distance $d ( t , j )$ defined in Section III. Thus, distance 1 denotes adjacent instances in the annotated leftto-right order; it is not a Euclidean pixel-distance bin. This choice is robust to image scale and uneven object spacing and directly tests local instance competition, but it cannot determine whether ordinal adjacency or physical separation is the dominant cause. Misbinding confusion matrices record how often the target instance index i is confused with source instance index j.

## E. Intervention Gap

Full-image evaluation is compared with localized variants, including crop oracle and context crop. The binding gap is the accuracy difference between an intervention setting and the full-image setting on the same underlying target-question subset. Because crop views require a localized reformulation of the query, these are joint visual-and-query interventions rather than image-only ablations. A positive gap suggests that reducing same-class visual competition helps the model, while a small or negative gap indicates that localization is insufficient or that the model relies on broader context.

## VI. EXPERIMENTS

## A. Models and Inference Protocol

The evaluation covers five open-source LVLMs and two commercial/API LVLMs: Qwen2.5-VL-7B [38], InternVL3- 8B [39], LLaVA-1.5-7B [40], [41], LLaVA-OneVision-7B [42], MiniCPM-V-2.6 [43], Gemini-3.5-Flash [44], and Qwen3-VL-Plus [45]. Every model receives the same question set, answer constraints, and normalized parser. Deterministic or low-temperature decoding is used whenever supported, and prompts request short answers so that evaluation reflects visual binding rather than generation style.

## B. Main Results: What Aggregate Accuracy Conceals

Table III demonstrates the diagnostic information added by InstaBind-Lite. Aggregate accuracy identifies Qwen3-VL-Plus and Gemini-3.5-Flash as the strongest evaluated systems, at 81.52% and 80.13%, but it cannot state why their remaining answers are wrong. MBR reveals that 9.31% and 5.79% of all questions, respectively, are answered with information belonging to another visible same-class instance. Thus, high general accuracy does not imply reliable instance–attribute correspondence.

The distinction is larger for open-source models. Their MBR ranges from 13.65% to 34.01%; for LLaVA-1.5, 74.54% of all wrong answers are identifiable wrong-instance transfers rather than arbitrary mistakes. Moreover, MBR exceeds outof-set hallucination in six of seven models. A conventional object-presence evaluation would therefore underdescribe a substantial error component: the predicted attribute often exists in the scene and is visually supported, but its ownership is wrong. These results do not negate improvements measured by general benchmarks; they show that claims of visual reliability require a complementary instance-binding test.

The second finding concerns error geometry. A-MBR ranges from 76.97% to 84.65% across all seven models. Even the two API systems retain A-MBRs of 78.38% and 84.64%, despite their lower total MBR. Stronger models reduce the frequency of misbinding, but the residual failures preserve the same local signature. This consistency across architectures and capability levels is evidence against uniform attribute guessing and supports competition between neighboring sameclass representations.

## C. Statistical Stability

Statistical stability is evaluated with 1000 image-cluster bootstrap resamples. Images, rather than individual questions, are sampled with replacement, and all questions from each selected image remain together. This prevents the many templates derived from one scene from being treated as independent evidence. Table IV shows narrow MBR intervals relative to the cross-model differences. LLaVA-1.5 remains distinctly high at 34.01% [32.70, 35.24], whereas Gemini-3.5-Flash remains distinctly low at 5.79% [5.02, 6.53]; neither result is explained by a small set of unusual images. More importantly, every A-MBR interval remains high: even Gemini’s lower bound is 73.93%, and Qwen3-VL-Plus reaches 84.64% [81.34, 87.99]. Total error frequency is model-dependent, but neighbor concentration is stable under image-level sampling variation. The bootstrap therefore supports two separate conclusions: model strength changes how often misbinding occurs, while the benchmark consistently reveals where the transferred attribute comes from.

## D. Parser Reliability

A stratified audit covers 200 outputs from the five opensource models, with 40 outputs per model. It includes 87 full-image and 113 intervention outputs, all four question levels, and five outcome strata: 50 correct answers, 50 adjacent misbindings, 35 non-adjacent misbindings, 35 out-ofset hallucinations, and 30 attribute-recognition failures. Human judgment agrees with the normalized parser on all 200 outputs. The point estimate is therefore 100%, with a 95% Wilson interval of approximately [98.1%, 100%]. This finite audit does not prove that parsing is error-free outside the sample, but it rules out a parser disagreement rate large enough to explain the reported MBR and A-MBR patterns and verifies that the error taxonomy can be applied reproducibly across answer formats and intervention settings.

## VII. QUALITATIVE ANALYSIS AND PRACTICAL SIGNIFICANCE

Figure 4 shows six adjacent transfers across bags, basins, bottles, cars, cups, and umbrellas. Each full-image prediction is the attribute of the annotated distance-1 source, while crop and context-crop recover the target value. An identifiable adjacent source plus localized recovery is more consistent with same-class competition than arbitrary color guessing, although the joint image-and-query intervention does not establish an internal causal mechanism. Complete questions, ordered attribute sequences, and outputs are included in the release metadata.

TABLE III  
FULL-IMAGE PERFORMANCE AND MISBINDING METRICS ON INSTABIND-LITE.
<table><tr><td>Model</td><td>Type</td><td>Acc.</td><td>MBR</td><td>Err-MBR</td><td>A-MBR</td><td>Out-set</td><td>Invalid</td></tr><tr><td>Qwen2.5-VL-7B</td><td>Open</td><td>75.89</td><td>13.65</td><td>56.62</td><td>82.65</td><td>7.07</td><td>0.0</td></tr><tr><td>InternVL3-8B</td><td>Open</td><td>76.69</td><td>14.01</td><td>60.1</td><td>84.65</td><td>7.36</td><td>0.0</td></tr><tr><td>LLaVA-1.5-7B</td><td>Open</td><td>54.37</td><td>34.01</td><td>74.54</td><td>78.7</td><td>7.89</td><td>0.0</td></tr><tr><td>LLaVA-OneVision-7B</td><td>Open</td><td>72.3</td><td>18.58</td><td>67.07</td><td>76.97</td><td>6.88</td><td>0.0</td></tr><tr><td>MiniCPM-V-2.6</td><td>Open</td><td>71.18</td><td>18.96</td><td>65.77</td><td>80.51</td><td>7.13</td><td>0.0</td></tr><tr><td>Gemini-3.5-Flash</td><td>API</td><td>80.13</td><td>5.79</td><td>29.15</td><td>78.38</td><td>9.66</td><td>1.42</td></tr><tr><td>Qwen3-VL-Plus</td><td>API</td><td>81.52</td><td>9.31</td><td>50.4</td><td>84.64</td><td>7.53</td><td>0.02</td></tr></table>

Full-image performance across seven LVLMs  
![](images/04ae4c18ddb6dc148290c83ca987adddb23823bb88f95d2625718474f0f38072.jpg)  
Fig. 1. Full-image accuracy, MBR, and out-of-set hallucination across seven LVLMs. Aggregate accuracy and object-level hallucination do not reveal the visible wrong-instance transfers measured by MBR.

TABLE IV  
IMAGE-CLUSTER BOOTSTRAP 95% CONFIDENCE INTERVALS FOR FULL-IMAGE EVALUATION. ALL INTERVALS ARE COMPUTED BY RESAMPLING IMAGES WITH REPLACEMENT, PRESERVING ALL QUESTIONS FROM EACH SAMPLED IMAGE.
<table><tr><td>Model</td><td>Accuracy</td><td>MBR</td><td></td><td>A-MBR</td></tr><tr><td>Qwen2.5-VL-7B</td><td>75.89 [74.50, 77.50] 13.65 [12.50, 14.83] 82.65 [80.08, 85.36]</td><td></td><td></td><td></td></tr><tr><td>InternVL3-8B</td><td></td><td></td><td></td><td>76.69 [75.28, 78.12]14.01 [12.94, 15.03] 84.65 [82.33, 86.97]</td></tr><tr><td>LLaVA-1.5-7B</td><td></td><td></td><td></td><td>54.37 [53.05, 55.78] 34.01 [32.70, 35.24] 78.70 [77.15, 80.54]</td></tr><tr><td>LLaVA-OneVision-7B 72.30 [70.80, 73.73] 18.58 [17.41, 19.85] 76.97 [75.03, 78.95]</td><td></td><td></td><td></td><td></td></tr><tr><td>MiniCPM-V-2.6</td><td>71.18 [69.69, 72.70]</td><td></td><td></td><td>18.96 [17.77, 20.13]80.51 [78.66, 82.38]</td></tr><tr><td>Gemini-3.5-Flash</td><td>80.13 [78.57, 81.54]</td><td>5.79 [5.02, 6.53]</td><td></td><td>78.38 [73.93, 82.85]</td></tr><tr><td>Qwen3-VL-Plus</td><td>81.52 [80.17, 82.96]</td><td>9.31 [8.36, 10.29]</td><td></td><td>84.64 [81.34, 87.99]</td></tr></table>

Such transfers can return the wrong belonging in assistive search, select the wrong inventory or robot-picking item, or retrieve the wrong vehicle in traffic analysis. These are risk mappings rather than measured deployment outcomes, but they show why an in-image answer is not necessarily safe: object presence checks cannot flag a wrong owner when every class and color is visible.

## VIII. INTERVENTION ANALYSIS

## A. Protocol

Interventions are applied to the 4261 L1/L3 questions with a single defined target and attribute. All intervention images preserve aspect ratio. A source image is downsampled only when its longer side exceeds 1800 pixels, with the target box scaled by the same factor. Crop oracle uses the annotated target box with horizontal and vertical padding equal to 15% of target width and height, clipped to image boundaries. Context crop uses 50% padding to retain more local context. Because an original expression such as “third from the left” is no longer meaningful after cropping, the direct crop prompt asks for the depicted instance’s attribute and the context prompt asks for the attribute of the instance closest to the crop center.

“Oracle” denotes access to the ground-truth target box, not perfect pixel isolation. A 15% crop can retain fragments of an overlapping neighbor, while the 50% context crop intentionally preserves nearby entities and scene cues. The comparison is therefore a joint visual-and-query localization test: it asks whether making the target easier to isolate reduces sourceattributable misbinding.

Distance-MBR distribution  
Full-image error decomposition  
![](images/e4a8668b4d918e6a2824f865b2c644ae8792be69eb28e5fd95bd61b484faea88.jpg)

Fig. 2. Full-image error decomposition. Adjacent misbinding forms a substantial error component in open-source models and remains structurally visible in API models.  
![](images/82fda82e68cf791637b518a3933e907aa6205d61c53537be2abed4306fdfad28.jpg)  
Fig. 3. Ordinal Distance-MBR. Most identifiable transfers originate from the immediately adjacent same-class instance (d = 1), while errors rapidly decline for more distant instances.

Table V separates models whose errors are sensitive to visual competition from those that need broader context. LLaVA-1.5 gains 20.14 accuracy points and reduces MBR by 19.12 points under crop oracle; MiniCPM-V-2.6 gains 9.22 accuracy points and reduces MBR by 10.57 points. LLaVA-OneVision also reduces MBR by 4.95 points. These paired changes are stronger evidence of competition-sensitive binding than accuracy alone: the intervention removes competing instances, and the specific error component attributed to those instances falls.

The response is not universal. Gemini-3.5-Flash and Qwen3-VL-Plus begin with lower full-image MBR but increase under localized views. Proprietary preprocessing prevents separation of global-context removal from crop-induced scale or distribution shift. The negative gaps therefore do not imply that visual competition is beneficial; they show that target isolation is not a universally safe mitigation. This model-dependent pattern is another diagnostic advantage of InstaBind-Lite: a proposed intervention can be evaluated on the failure type it is intended to repair, rather than judged only by aggregate accuracy.

The image-cluster intervals in Table VI reinforce this interpretation. MBR reduction is reliably positive for LLaVA-

![](images/8617bf00e9f6a63961a34b91805f483503e8ff1cfb43bdd3e06edfc0714f37f7.jpg)  
Fig. 4. Six manually selected, intervention-confirmed adjacent misbindings. The queried target is outlined in red and the source of the full-image answer in orange. In every panel, the full-image prediction is a visible attribute of the adjacent source instance, whereas both crop and context-crop evaluation recover the gold answer. The first case is from LLaVA-1.5 and the remaining cases are from InternVL3.

TABLE V  
INTERVENTION RESULTS ON THE L1/L3 SUBSET. CROP AND CONTEXT CROPS ARE COMPARED AGAINST FULL IMAGES.
<table><tr><td>Model</td><td>Full Acc.</td><td>Full MBR</td><td>Crop Acc.</td><td>Crop MBR</td><td>Ctx Acc.</td><td>Ctx MBR</td></tr><tr><td>Qwen2.5-VL-7B</td><td>69.56</td><td>14.57</td><td>69.44</td><td>12.72</td><td>73.36</td><td>11.24</td></tr><tr><td>InternVL3-8B</td><td>69.87</td><td>13.59</td><td>66.7</td><td>13.42</td><td>70.52</td><td>12.27</td></tr><tr><td>LLaVA-1.5-7B</td><td>47.92</td><td>34.33</td><td>68.06</td><td>15.21</td><td>66.86</td><td>18.21</td></tr><tr><td>LLaVA-OneVision-7B</td><td>63.98</td><td>20.56</td><td>69.56</td><td>15.61</td><td>69.54</td><td>15.84</td></tr><tr><td>MiniCPM-V-2.6</td><td>62.22</td><td>21.76</td><td>71.44</td><td>11.19</td><td>72.8</td><td>11.92</td></tr><tr><td>Gemini-3.5-Flash</td><td>71.46</td><td>7.16</td><td>71.53</td><td>8.97</td><td>70.55</td><td>9.18</td></tr><tr><td>Qwen3-VL-Plus</td><td>74.23</td><td>8.89</td><td>71.84</td><td>10.02</td><td>72.64</td><td>11.24</td></tr></table>

TABLE VI

IMAGE-CLUSTER BOOTSTRAP 95% CONFIDENCE INTERVALS FOR MBR REDUCTION ON L1/L3. POSITIVE VALUES INDICATE THAT THE INTERVENTION REDUCES MBR RELATIVE TO FULL-IMAGE EVALUATION.
<table><tr><td>Model</td><td>Crop</td><td>Context</td></tr><tr><td>Qwen2.5-VL</td><td>+1.85 [-0.12, +3.69]</td><td>+3.33 [+1.38, +5.18]</td></tr><tr><td>InternVL3</td><td>+0.16 [-1.68, +2.06]</td><td>+1.31 [-0.38, +2.93]</td></tr><tr><td>LLaVA-1.5</td><td>+19.13 [+16.73, +21.41]</td><td>+16.12 [+13.62, +18.40]</td></tr><tr><td>LLaVA-OV</td><td>+4.95 [+2.89, +6.94]</td><td>+4.72 [+2.73, +6.64]</td></tr><tr><td>MiniCPM-V</td><td>+10.56 [+8.64, +12.45]</td><td>+9.83 [+7.87, +11.78]</td></tr><tr><td>Gemini</td><td>-1.81 [-3.18, -0.23]</td><td>-2.02 [-3.52, -0.54]</td></tr><tr><td>Qwen3-VL+</td><td>-1.13 [-2.74, +0.35]</td><td>-2.35 [-3.87, -0.88]</td></tr></table>

1.5, LLaVA-OneVision, and MiniCPM-V, while Qwen2.5- VL’s context crop is positive but its crop interval crosses zero. InternVL3’s intervals also cross zero, and the API-model intervals are zero-crossing or negative. Localization therefore identifies a substantial same-class competition component in selected open-source models, but not a single mechanism shared equally by every system.

## B. Lightweight Inference-Time Mitigation

Instance-first prompting explicitly externalizes an intermediate binding map: the model first enumerates same-class entities from left to right with their attributes and then answers the original question. No training or architectural access is required. InternVL3 improves accuracy by 1.54 points while reducing MBR by 4.36 points, and MiniCPM-V improves by 2.17 points while reducing MBR by 3.46 points. These paired gains indicate that explicit decomposition can repair a subset of binding errors.

Other models expose a different trade-off. Qwen2.5-VL reduces MBR slightly while losing accuracy, whereas LLaVA-1.5 loses 22.58 accuracy points and increases MBR under the longer prompt. A standard score would report only improvement or degradation; the binding-specific evaluation reveals whether the change actually repairs wrong-instance transfer. The mixed outcome positions InstaBind-Liteas a testbed for future grounding-aware decoding, contrastive neighbor suppression, or targeted fine-tuning: mitigation should lower MBR without sacrificing general recognition and instruction following.

Intervention effect on L1/L3 misbinding  
![](images/63b9cdec4f27d45a4f4a4812c83b752c3ed2d82b90e2e794ff6c794c982edf69.jpg)  
Fig. 5. Intervention effect on L1/L3 MBR. Localization sharply reduces source-attributable misbinding for several open-source models but can remove useful context or shift the input distribution for stronger API systems.

TABLE VII  
INSTANCE-FIRST PROMPTING ON OPEN-SOURCE LVLMS.
<table><tr><td>Model</td><td>Full Acc.</td><td>IF Acc.</td><td>Acc. Gap</td><td>Full MBR</td><td>IF MBR</td><td>MBR Gap</td></tr><tr><td>Qwen2.5-VL-7B</td><td>75.89</td><td>73.19</td><td>-2.69</td><td>13.65</td><td>12.76</td><td>-0.9</td></tr><tr><td>InternVL3-8B</td><td>76.69</td><td>78.24</td><td>1.54</td><td>14.01</td><td>9.65</td><td>-4.36</td></tr><tr><td>LLaVA-1.5-7B</td><td>54.37</td><td>31.8</td><td>-22.58</td><td>34.01</td><td>35.16</td><td>1.15</td></tr><tr><td>LLaVA-OneVision-7B</td><td>72.3</td><td>70.66</td><td>-1.64</td><td>18.58</td><td>19.83</td><td>1.25</td></tr><tr><td>MiniCPM-V-2.6</td><td>71.18</td><td>73.35</td><td>2.17</td><td>18.96</td><td>15.5</td><td>-3.46</td></tr></table>

## IX. DISCUSSION

## A. Why a Source-Aware Benchmark Is Necessary

Standard accuracy answers whether a response is correct; object-hallucination measures answer whether predicted content is supported; grounding measures answer whether a referent is localized. None of these quantities alone answers whether a visible attribute was assigned to its owner. InstaBind-Liteadds that missing variable by recording target and competing source instances. The resulting decomposition is operationally important: unsupported generation calls for hallucination suppression, recognition failure calls for stronger visual features, and adjacent transfer calls for better instance separation or binding. Collapsing them into one error rate obscures which component a new model or mitigation actually improves.

The seven-model results illustrate the consequence. API models lead aggregate accuracy, yet retain measurable MBR and high A-MBR. Six of seven models produce wronginstance transfers more often than out-of-set hallucinations.

A model can therefore appear strong under broad capability and object-presence tests while remaining unreliable for entity-specific questions. InstaBind-Litedoes not replace those benchmarks; it prevents their scores from being interpreted as evidence of instance-grounded reliability without a direct binding test.

## B. Adjacent Structure and Mechanistic Implications

Across architectures, most identifiable transfers originate from ordinal distance 1. The bootstrap analysis shows that this neighbor concentration persists under image-level resampling even when total MBR differs substantially. This result supports local same-class competition rather than uniform answer noise and motivates mechanisms that explicitly contrast adjacent instances. The current ordinal analysis does not establish a causal effect of Euclidean separation: adjacent indices can have unequal pixel gaps, object scales, and overlaps. Normalized center distance, boundary distance, and overlap are therefore required to distinguish order-local competition from physical proximity.

## C. Implications for Multimedia Systems

The cases in Fig. 4 map this distinction to multimedia operations: attributes cue assistive identification, media and catalog search, robotic selection, and traffic retrieval. Real visual evidence can still identify the wrong entity, producing wrong-item retrieval or action rather than merely an awkward caption. Source-aware evaluation is therefore required before treating an LVLM as reliable for entity-specific decisions.

## D. Diagnostic Scope and Mitigation

The controlled color domain enables deterministic source attribution and exposes a clean binding signal. It does not imply that actions, materials, states, or part attributes will exhibit identical rates. Those semantics introduce temporal and annotation ambiguity and require adapted question templates and judges. Similarly, crop and instance-first prompting are probes rather than a complete solution. Their model-dependent effects point toward two research directions: binding-aware training that contrasts neighboring same-class instances, and inference methods that preserve global context while explicitly grounding the target. The benchmark provides the boxes, neighbor structure, and error labels needed to evaluate whether such methods repair binding rather than merely shift aggregate accuracy.

## X. DATA AVAILABILITY AND ETHICAL CONSIDERATIONS

InstaBind-Liteis intended to be released with annotations, generated questions, evaluation scripts, parser code, and model-output metadata needed to reproduce the reported tables and figures. For images originating from established datasets, release procedures will follow the original terms and provide source identifiers when direct redistribution is not appropriate. For manually collected web images, redistribution will be limited to files with recoverable license evidence permitting such use; otherwise, the public package will provide derived annotations and exclude the image file. Self-shot images can be released directly by the authors.

The benchmark contains images of people only for visual attribute binding evaluation. Identity, demographic, and sensitive-attribute questions are excluded; person questions concern only visible upper-clothing color. Manual review excludes potentially offensive, private, or ambiguous depictions. The dataset is designed for diagnostic evaluation of LVLM behavior and should not be used for surveillance, identity inference, or demographic profiling.

## XI. CONCLUSION

This study formalizes Dense Same-Class Attribute Misbindingas a source-identifiable failure in which an LVLM predicts an attribute present in the image but assigns it to the wrong same-class instance. InstaBind-Liteaddresses a gap left by aggregate VQA, object-hallucination, grounding, and compositionality evaluation: its controlled groups, ordered instance annotations, and four question levels reveal not only that an answer is wrong, but whether its evidence was copied from a specific visible neighbor.

Evaluation across seven LVLMs shows that instance-level reliability cannot be inferred from general accuracy alone. Open-source models average 19.84% MBR and API models 7.55%, while both groups retain approximately 81% adjacent concentration among identifiable transfers. Image-cluster confidence intervals, a stratified parser audit, real-image cases, and controlled localization and prompting interventions support the same conclusion: DSCAM is a reproducible, spatially structured error category, and mitigation must reduce wronginstance transfer without discarding useful context or general capability.

## A. Future Work

Future work will turn InstaBind-Liteinto a controlled modeldevelopment loop. Leakage-safe train, validation, and heldout test splits will support comparisons of box-conditioned visual tokens, coordinate embeddings, neighbor-contrastive losses, and ordered instance–attribute supervision. Instancefirst prompting and neighbor-contrastive decoding will remain inference-time baselines. All variants will use the same frozen test set and parser; success requires lower MBR and Err-MBR, stable or higher accuracy, and no increase in out-of-set hallucination or invalid answers. A claimed spatial repair must also reduce A-MBR or Distance-MBR, with image-cluster bootstrap intervals, paired image-level tests, and stratification by class, level, group size, position, and source distance.

Expansion will balance underrepresented classes and extend attributes to material, state, action, part, depth, and temporal identity using auditable evaluation. Ordinal distance will be complemented by normalized pixel distance, overlap, depth, and occlusion; parser audits will include API outputs and broader strata. Successful mitigations will finally be tested on newly collected and cross-dataset images to establish transfer beyond the controlled color slice.

## REFERENCES

[1] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, “Learning transferable visual models from natural language supervision,” in International Conference on Machine Learning, 2021, pp. 8748–8763.

[2] J. Li, D. Li, C. Xiong, and S. Hoi, “Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation,” in International Conference on Machine Learning, 2022, pp. 12 888– 12 900.

[3] J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds et al., “Flamingo: A visual language model for few-shot learning,” in Advances in Neural Information Processing Systems, vol. 35, 2022, pp. 23 716–23 736.

[4] J. Li, D. Li, S. Savarese, and S. Hoi, “Blip-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models,” in International Conference on Machine Learning, 2023.

[5] W. Dai, J. Li, D. Li, A. M. H. Tiong, J. Zhao, W. Wang, B. Li, P. Fung, and S. C. H. Hoi, “Instructblip: Towards general-purpose vision-language models with instruction tuning,” in Advances in Neural Information Processing Systems, vol. 36, 2023.

[6] S. Antol, A. Agrawal, J. Lu, M. Mitchell, D. Batra, C. L. Zitnick, and D. Parikh, “Vqa: Visual question answering,” in Proceedings ofthe IEEE International Conference on Computer Vision, 2015.

[7] Y. Li, Y. Du, K. Zhou, J. Wang, X. Zhao, and J.-R. Wen, “Evaluating object hallucination in large vision-language models,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. Singapore: Association for Computational Linguistics, 2023, pp. 292–305.

[8] J. Wang, Y. Wang, G. Xu, J. Zhang, Y. Gu, H. Jia, J. Wang, H. Xu, M. Yan, J. Zhang, and J. Sang, “Amber: An llm-free multidimensional benchmark for mllms hallucination evaluation,” arXiv preprint arXiv:2311.07397, 2023.

[9] T. Guan, F. Liu, X. Wu, R. Xian, Z. Li, X. Liu, X. Wang, L. Chen, F. Huang, Y. Yacoob, D. Manocha, and T. Zhou, “Hallusionbench: An advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

[10] The Guardian, “Chinese facial recognition system mistakes bus advert for jaywalker,” The Guardian, November 22, 2018, 2018, accessed: August 6, 2026. [Online]. Available: https://www.theguardian.com/world/2018/nov/22/ chinese-facial-recognition-system-mistakes-bus-advert-for-jaywalker

[11] National Transportation Safety Board, “Collision between vehicle controlled by developmental automated driving system and pedestrian, tempe, arizona, march 18, 2018,” National Transportation Safety Board, Tech. Rep. HAR-19/03, 2019, accessed: August 6, 2026. [Online]. Available: https://www.ntsb.gov/investigations/ AccidentReports/Reports/HAR1903.pdf

[12] A. Rohrbach, L. A. Hendricks, K. Burns, T. Darrell, and K. Saenko, “Object hallucination in image captioning,” in Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing. Brussels, Belgium: Association for Computational Linguistics, 2018, pp. 4035–4045.

[13] A. Gunjal, J. Yin, and E. Bas, “Detecting and preventing hallucinations in large vision language models,” in Proceedings ofthe AAAI Conference on Artificial Intelligence, vol. 38, no. 16, 2024, pp. 18 135–18 143.

[14] Q. Huang, X. Dong, P. Zhang, B. Wang, C. He, J. Wang, D. Lin, W. Zhang, and N. Yu, “Opera: Alleviating hallucination in multimodal large language models via over-trust penalty and retrospectionallocation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 13 418–13 427.

[15] Y. Goyal, T. Khot, D. Summers-Stay, D. Batra, and D. Parikh, “Making the v in vqa matter: Elevating the role of image understanding in visual question answering,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017, pp. 6904–6913.

[16] Y. Zhu, O. Groth, M. Bernstein, and L. Fei-Fei, “Visual7w: Grounded question answering in images,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016, pp. 4995–5004.

[17] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, and C. L. Zitnick, “Microsoft coco: Common objects in´ context,” in Proceedings of the European Conference on Computer Vision, 2014, pp. 740–755.

[18] R. Krishna, Y. Zhu, O. Groth, J. Johnson, K. Hata, J. Kravitz, S. Chen, Y. Kalantidis, L.-J. Li, D. A. Shamma, M. S. Bernstein, and L. Fei-Fei, “Visual genome: Connecting language and vision using crowdsourced dense image annotations,” International Journal of Computer Vision, vol. 123, no. 1, pp. 32–73, 2017.

[19] D. A. Hudson and C. D. Manning, “Gqa: A new dataset for real-world visual reasoning and compositional question answering,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 6700–6709.

[20] K. Pham, K. Kafle, Z. Lin, Z. Ding, S. Cohen, Q. Tran, and A. Shrivastava, “Learning to predict visual attributes in the wild,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 13 018–13 028.

[21] C. Fu, P. Chen, Y. Shen, Y. Qin, M. Zhang, X. Lin, Z. Qiu, W. Lin, J. Yang, X. Zheng, K. Li, X. Sun, and R. Ji, “Mme: A comprehensive evaluation benchmark for multimodal large language models,” arXiv preprint arXiv:2306.13394, 2023.

[22] B. Li, R. Wang, G. Wang, Y. Ge, Y. Ge, and Y. Shan, “Seed-bench: Benchmarking multimodal llms with generative comprehension,” arXiv preprint arXiv:2307.16125, 2023.

[23] W. Yu, Z. Yang, L. Li, J. Wang, K. Lin, Z. Liu, X. Wang, and L. Wang, “Mm-vet: Evaluating large multimodal models for integrated capabilities,” in International Conference on Machine Learning, 2024.

[24] Y. Liu, H. Duan, Y. Zhang, B. Li, S. Zhang, W. Zhao, Y. Yuan, J. Wang, C. He, Z. Liu, K. Chen, and D. Lin, “Mmbench: Is your multimodal model an all-around player?” in Proceedings of the European Conference on Computer Vision, 2024, pp. 216–233.

[25] B. A. Plummer, L. Wang, C. M. Cervantes, J. C. Caicedo, J. Hockenmaier, and S. Lazebnik, “Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models,” International Journal of Computer Vision, vol. 123, no. 1, pp. 74–93, 2017.

[26] S. Kazemzadeh, V. Ordonez, M. Matten, and T. Berg, “Referitgame: Referring to objects in photographs of natural scenes,” in Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing. Doha, Qatar: Association for Computational Linguistics, 2014, pp. 787–798.

[27] L. Yu, P. Poirson, S. Yang, A. C. Berg, and T. L. Berg, “Modeling context in referring expressions,” in Proceedings of the European Conference on Computer Vision, 2016, pp. 69–85.

[28] L. Yu, Z. Lin, X. Shen, J. Yang, X. Lu, M. Bansal, and T. L. Berg, “Mattnet: Modular attention network for referring expression comprehension,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 1307–1315.

[29] J. Johnson, B. Hariharan, L. van der Maaten, L. Fei-Fei, C. L. Zitnick, and R. Girshick, “Clevr: A diagnostic dataset for compositional language and elementary visual reasoning,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017.

[30] T. Thrush, R. Jiang, M. Bartolo, A. Singh, A. Williams, D. Kiela, and C. Ross, “Winoground: Probing vision and language models for visiolinguistic compositionality,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 5238–5248.

[31] M. Yuksekgonul, F. Bianchi, P. Kalluri, D. Jurafsky, and J. Zou, “When and why vision-language models behave like bags-of-words, and what to do about it?” in International Conference on Learning Representations, 2023.

[32] T. Zhao, T. Zhang, M. Zhu, H. Shen, K. Lee, X. Lu, and J. Yin, “An explainable toolbox for evaluating pre-trained vision-language models,” in Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing: System Demonstrations. Abu Dhabi, UAE: Association for Computational Linguistics, 2022, pp. 30–37.

[33] Z. Ma, J. Hong, M. O. Gul, M. Gandhi, I. Gao, and R. Krishna, “Crepe: Can vision-language foundation models reason compositionally?” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 10 910–10 921.

[34] C.-Y. Hsieh, J. Zhang, Z. Ma, A. Kembhavi, and R. Krishna, “Sugarcrepe: Fixing hackable benchmarks for vision-language compositionality,” in Advances in Neural Information Processing Systems, vol. 36, 2023.

[35] D. Saravanan, M. Tapaswi, and V. Gandhi, “Investigating mechanisms for in-context vision language binding,” in 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2025, pp. 4852–4856.

[36] K. Cui, N. Prakash, S. Messica, A. Raina, D. Bau, A. Torralba, and T. R. Shaham, “The dual mechanisms of spatial variable binding in vision-language models,” arXiv preprint arXiv:2603.22278, 2026.

[37] W. Tian, H. Mao, Z. Liu, L. Zhang, Q. Liu, J. Wu, and L. Wang, “Multibind: A benchmark for attribute misbinding in multi-subject generation,” arXiv preprint arXiv:2603.21937, 2026.

[38] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang, H. Zhong, Y. Zhu, M.-H. Yang, Z. Li, J. Wan, P. Wang, W. Ding, Z. Fu, Y. Xu, J. Ye, X. Zhang, T. Xie, Z. Cheng, H. Zhang, Z. Yang, H. Xu, and J. Lin, “Qwen2.5-vl technical report,” arXiv preprint arXiv:2502.13923, 2025.

[39] J. Zhu, W. Wang, Z. Chen, Z. Liu, S. Ye, L. Gu, H. Tian, Y. Duan, W. Su, J. Shao, Z. Gao, E. Cui, X. Wang, Y. Cao, Y. Liu, X. Wei, H. Zhang, H. Wang, W. Xu, H. Li, J. Wang, N. Deng, S. Li, Y. He, T. Jiang, J. Luo, Y. Wang, C. He, B. Shi, X. Zhang, W. Shao, J. He, Y. Xiong, W. Qu, P. Sun, P. Jiao, H. Lv, L. Wu, K. Zhang, H. Deng, J. Ge, K. Chen, L. Wang, M. Dou, L. Lu, X. Zhu, T. Lu, D. Lin, Y. Qiao, J. Dai, and W. Wang, “Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models,” arXiv preprint arXiv:2504.10479, 2025.

[40] H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” in Advances in Neural Information Processing Systems, vol. 36, 2023, pp. 34 892–34 916.

[41] H. Liu, C. Li, Y. Li, and Y. J. Lee, “Improved baselines with visual instruction tuning,” arXiv preprint arXiv:2310.03744, 2023.

[42] B. Li, Y. Zhang, D. Guo, R. Zhang, F. Li, H. Zhang, K. Zhang, P. Zhang, Y. Li, Z. Liu, and C. Li, “Llava-onevision: Easy visual task transfer,” arXiv preprint arXiv:2408.03326, 2024.

[43] Y. Yao, T. Yu, A. Zhang, C. Wang, J. Cui, H. Zhu, T. Cai, H. Li, W. Zhao, Z. He et al., “Minicpm-v: A gpt-4v level mllm on your phone,” arXiv preprint arXiv:2408.01800, 2024.

[44] Google DeepMind, “Gemini 3.5 flash model card,” https://deepmind. google/models/model-cards/gemini-3-5-flash/, 2026, accessed: 2026- 07-17.

[45] Qwen Cloud, “Qwen3-vl-plus,” https://www.qwencloud.com/models/ qwen3-vl-plus, 2026, accessed: 2026-07-17.