# FormStruct-Bench: A Hierarchical and Diagnostic Benchmark for Table-Form Document Structure Recognition

Lujie Ban<sup>1</sup>, Jiangtao Zhu<sup>1</sup>, Yuanheng Yu<sup>2</sup>, Jiasheng Shi<sup>1</sup>, Chenhao Ma<sup>∗1</sup>

<sup>1</sup> The Chinese University of Hong Kong, Shenzhen,<sup>2</sup> University of Washington Shenzhen, China

{lujieban,jiangtaozhu}@link.cuhk.edu.cn, yuanhy@uw.edu, {shijiasheng,machenhao}@cuhk.edu.cn

## Abstract

Transforming table-form documents into machine-processable record requires recovering not only their visible content but also the multilevel structure that organizes it. However, existing benchmarks evaluate either holistic document outputs or conventional table grids, and their aggregate scores provide little insight into where structural failures occur. We introduce FormStruct-Bench, a hierarchical and diagnostic benchmark that evaluates table-form document structure recognition at both the document level and progressively finer component levels, allowing aggregate performance to be traced back to specific structural failure modes. To construct auditable ground truth at scale, we annotate 70 reusable templates and expand them into 7,000 verified instances through a provenance-preserving Director–Artist–Verifier pipeline; all 1,100 instances in the template-disjoint test set additionally receive human review. Our evaluation protocol uses five primary metrics and three structure-specific diagnostics across page, schema, and component levels, together with slices over dificulty, structural constraints, and visual degradation. Across 14 API-hosted and locally deployable systems plus two SFT variants, the best document-level score reaches 83.85%, whereas the best reported fine-grained structural score remains below 18%. These results reveal a pronounced gap between reading document content and recovering the hierarchy and regional organization required for reliable table-form understanding.

## CCS Concepts

• Applied computing → Document analysis; • Computing methodologies → Artificial intelligence.

## Keywords

table-form documents, document structure recognition, form understanding, diagnostic evaluation, multimodal benchmarking

## ACM Reference Format:

Lujie Ban<sup>1</sup>, Jiangtao Zhu<sup>1</sup>, Yuanheng Yu<sup>2</sup>, Jiasheng Shi<sup>1</sup>, Chenhao Ma<sup>∗1</sup>. 2027. FormStruct-Bench: A Hierarchical and Diagnostic Benchmark for Table-Form Document Structure Recognition. In Proceedings of 33rd ACM

SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’27). ACM, New York, NY, USA, 20 pages. https://doi.org/XXXXXXX.XXXXXXX

## 1 Introduction

Documents are a primary carrier of real-world information and an essential source of data for automated information processing [3, 39]. Among them, forms constitute a particularly important document class, providing standardized interfaces through which organizations collect, validate, and exchange information [16, 31]. Table-form documents, such as application, tax, and medical-intake forms, organize such information through table-like layouts and are central to record ingestion [15, 31], compliance auditing [4], and automated workflow execution [7, 34]. Their information is encoded not only in visible text but also in structure: a page is divided into semantic sections, a section may carry its own local grid, and checkboxes and other widgets take their meaning from the fields and labels they are linked to (Figure 1) [1, 18]. Consequently, transforming a table-form document into reliable, machine-processable records requires recovering these structures rather than merely reading the visible text.

![](images/51273e4271e5dfa2142b415742ac7fe663872b78a8964a649c586758ace47236.jpg)  
Figure 1: An example table-form document with its structural annotation: semantic regions, a region-local grid, field and widget groups, and relation edges.

This transformation is a central goal of document understanding [3, 39]. With the rise of LLMs, document understanding has advanced rapidly from localized text and field extraction toward end-to-end document parsing and structured generation, with models jointly reasoning over visual, textual, and layout signals [14, 17, 30, 37]. In parallel, Table Structure Recognition (TSR) has focused on the precise reconstruction of conventional table layouts, including row–column topology, cell spans, and explicit grid or markup representations [8, 32, 33, 45].

However, table-form documents fall between these two lines of research. Document understanding extracts fields and answers questions over forms but rarely requires faithful reconstruction of the underlying structure, whereas TSR reconstructs structure precisely but confines its output to a single row–column grid, leaving fields, widgets, and their semantic relations outside its scope. Recognizing table-form documents demands both: the geometric precision of TSR, applied to the hierarchical and relational organi zation studied in document understanding. We term this problem table-form document structure recognition (Figure 1). This broader recognition target also creates a distinct evaluation challenge. Existing benchmarks typically evaluate either document-level understanding through question answering, information extraction, and holistic document parsing [22, 25, 31], or conventional table recon struction through canonical grid and markup recovery [8, 32, 33, 45], but neither benchmark family fully captures the joint structural requirements of table-form documents, as summarized in Table 1. Current benchmarks therefore do not provide an adequate basis for evaluating table-form document structure recognition.

Table 1: Comparison of structural targets and diagnostic capabilities. △ denotes partial or indirect support.
<table><tr><td>Benchmark</td><td>Grid structure</td><td>Full-page layout</td><td>Form elements</td><td>Structural relations</td><td>Component Diagnostic metrics</td><td>slices</td></tr><tr><td>SciTSR [8]</td><td>√</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>PubTabNet [45]</td><td>√</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>PubTables-1M [32]</td><td>√</td><td>Δ</td><td>×</td><td>×</td><td>Δ</td><td>×</td></tr><tr><td>FUNSD [16]</td><td>×</td><td>√</td><td>V</td><td>√</td><td>△</td><td>×</td></tr><tr><td>DocILE [31]</td><td>△</td><td>√</td><td>√</td><td>△</td><td>△</td><td>×</td></tr><tr><td>SRFUND [21]</td><td>△</td><td>√</td><td>√</td><td>√</td><td>√</td><td>×</td></tr><tr><td>Image2Struct [30]</td><td>△</td><td>√</td><td>Δ</td><td>△</td><td>×</td><td>×</td></tr><tr><td>OmniDocBench [25]</td><td>√</td><td>√</td><td>×</td><td>×</td><td>Δ</td><td>△</td></tr><tr><td>VAREX [5]</td><td>×</td><td>×</td><td>√</td><td>×</td><td>√</td><td>√</td></tr><tr><td>FormStruct-Bench</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

More specifically, the missing component is diagnostic evaluation: current benchmarks often report whether a predicted structure matches the target, but not the root cause of the failure. For example, a visa or medical form may place personal information, eligibility questions, checkboxes, and signature fields in diferent regions of the same page; the value of one field is often determined not only by nearby text, but also by its section, option group, and relation to distant labels. Therefore, current evaluations make it dificult to answer a question Q : when a model fails on a table-form document, which part of the structure caused the failure? Answering this question requires fine-grained structural annotations, which are dificult to obtain reliably from automatic LLM labeling, while instance-level human annotation is dificult to scale.

To answer question Q , we introduce FormStruct-Bench, a hierarchical and diagnostic benchmark for table-form document structure recognition. The key idea is to evaluate table-form documents as structured pages rather than as a single flattened table or a bag of extracted fields. Each sample is represented through a hierarchy of page regions, region-local grids, field and widget groups, and relation edges (Figure 1), which allows evaluation to isolate diferent structural failure modes. This representation also raises a data-construction requirement: the labels must remain consistent, traceable, and verifiable across document instances. To make the labels auditable, FormStruct-Bench decouples annotation from generation: human annotators construct reusable structural templates, and a multi-agent generation pipeline expands each template into verified instances while preserving provenance for the source template, constraints, visual parameters, and verification outcomes.

This design couples the reliability of human annotation with the scalability of automatic generation.

The same hierarchical representation also defines the evaluation protocol. Instead of reducing each prediction to a single table-level similarity score, FormStruct-Bench assesses whether a model recovers each major structural component, scoring predictions at the document level and at progressively finer component levels so that an overall score can be traced back to specific structural failures. It then stratifies these component scores by dificulty, structural constraints, and visual perturbations to expose where each model’s capability boundary lies.

Our contributions are summarized as follows:

• We construct FormStruct-Bench, a benchmark for table-form document structure recognition, using 70 reusable human-annotated templates and 7,000 verified generated instances with auditable provenance.

• We provide a diagnostic evaluation protocol that combines documentlevel scoring with finer component-level analysis.

• We benchmark 14 API-hosted and locally deployable systems plus two SFT variants, revealing a substantial gap between content recovery and structural recognition: the best document-level score exceeds 80%, while all reported fine-grained structural scores remain below 18%, with further degradation on relationdense, grid-heavy, and visually corrupted cases.

## 2 Dataset Construction

We construct FormStruct-Bench by converting human-annotated table-form documents into reusable structural templates and expanding each template through a Director–Artist–Verifier pipeline (Figure 2). This design scales instance generation while preserving exact labels, diagnostic constraints, and provenance.

## 2.1 Table-Form Document Structure Recognition

We define the prediction target of table-form document structure recognition as follows.

Definition 1 (Table-Form Document Structure Recognition). Given a table-form document image �, the task is to predict

$$
\hat { S } = f ( I ) , \qquad S = \big ( \mathcal { R } , \{ \mathcal { G } _ { r } \} _ { r \in \mathcal { R } } , \{ \mathcal { F } _ { r } \} _ { r \in \mathcal { R } } , \{ \mathcal { W } _ { f } \} _ { f \in \mathcal { F } } , \mathcal { E } \big ) ,\tag{1}
$$

where R denotes typed semantic regions, $\mathcal { G } _ { r }$ the (possibly empty) local grid of region $r \in \mathcal { R } , \mathcal { F } _ { r }$ the fillable fields of region � with $\textstyle { \mathcal { F } } = \bigcup _ { r \in { \mathcal { R } } } { \mathcal { F } } _ { r } , { \mathcal { W } } _ { f }$ the widget groups attached to field $f \in { \mathcal { F } } _ { : }$ , and E directed, typed relations among these elements $( \mathrm { e . g . }$ , key-to-field links).

This nesting makes S a hierarchical page structure rather than the single canonical grid of conventional TSR: regions carry local grids and fields, and fields carry widget groups. The evaluation protocol in Section 4 follows the same hierarchy: local grids are matched only under paired parent regions, and widget groups only under matching parent fields.

## 2.2 Template Collection and Annotation

Form-like tables combine local grids, fillable fields, selection controls, signature areas, and relations across non-adjacent regions.

![](images/a5b95914bf6386690e6e773dda5f94ff290cbeac7b0882b6ffe634f48c870996.jpg)  
Figure 2: Overview of the FormStruct-Bench dataset construction pipeline.

Because labeling every filled instance is costly, we instead collect reusable templates from web sources and 2 datasets including XFUND [41] and Common Crawl [10]. We retain candidates that combine table-like spatial organization with form-specific elements and whose stable layouts annotators can recover unambiguously; filled values are removed without altering the layout.

Each document is rasterized at normalized resolution and orientation, then annotated as a structural template. The annotation captures table boundaries, cells and spans, visible text, semantic regions, region-local grids, fillable fields, widgets, and structural relations. Rather than imposing a page-wide grid, annotators mark grids locally and represent checkboxes, radio buttons, character boxes, blank lines, and signature areas as fields or widgets. They record observable relations, including key-to-field links, field-towidget links, section membership, and line-item grouping; constraint tags, counts, and dificulty levels are instead derived from the exported metadata.

## 2.3 Multi-agent Instance Generation

The three agents separate content planning, rendering, and quality control. The Director samples field values, layout parameters, and visual degradation, and assigns each instance a target constraint profile. It can therefore populate specific diagnostic slices deliberately; the dificulty tier is derived from the assigned structural and context constraints rather than labeled post hoc. The resulting metadata fully specifies the rendering job.

Using only this metadata, the Artist renders the image without access to the original document. Geometry and annotations are computed at render time, yielding exact ground truth for typed regions, region-local grid topology, fields, and widget states. The Verifier uses few-shot in-context learning to instruct an LLM to infer the corresponding structured representation from the rendered image. The inferred result is subsequently validated against the reference answer, with any detected inconsistency returned to the Artist for iterative refinement and re-rendering.

## 2.4 Verification and Provenance

Automated verification checks four aspects: whether region types, local grids, and widget states match the metadata; whether OCR recovers the assigned values from fillable regions; whether border visibility, contrast, and skew remain within their assigned ranges;

and whether annotations have valid coordinates, field types, and relation targets. Failed samples trigger a structured report and are re-generated until they pass or exhaust the retry budget, after which they are discarded and logged.

Every test sample that passes these checks additionally receives human review to ensure that its regions, relations, and widget states are unambiguously supported by visible evidence. Reviewers mark a sample as accepted, corrected, or discarded; corrected samples are re-verified, and unresolved samples are excluded.

Passing samples receive a provenance record containing the source template, generation metadata, constraint tags, and, for test samples, the review outcome. Instances retain their template’s structural skeleton while varying filled values and visual parameters. Consequently, the test set can be partitioned by recorded constraint tags without further annotation, enabling constraint-level diagnostic evaluation.

## 3 Data Statistics

Reporting only sample counts and coarse category distributions tells a researcher little about whether a benchmark will expose the failure modes they care about. FormStruct-Bench organizes its statistics around three questions: Q1. what structural configurations the templates cover; Q2. which constraint combinations appear in the generated instances; and Q3. whether the test split samples those combinations in suficient numbers to support per-constraint diagnostic evaluation. Our dataset is publicly available at https:// huggingface.co/datasets/D2I-CUHK-Shenzhen/FormStruct-Bench.

## 3.1 Template-level Statistics

Q1 concerns domain breadth and structural complexity. Figure 3 shows that the 70 templates span eight named application domains, one residual unclassified template, and 20 fine-grained labels. Government and immigration is the largest coarse domain (14 templates, 20.0%), while no fine-grained use case exceeds 9 templates (12.9%). The calibrated dificulty distribution of11/24/24/11 templates across L1–L4 further balances routine and challenging structures, reducing dependence on any single domain or layout convention.

Table 2 shows that templates contain 1–7 semantic sections and 1–9 table regions; the median template has 4 semantic sections, 4 table regions, 23.5 label-field units, 16 selection fields, and 79 structural relation edges. Thus, FormStruct-Bench targets page-level form layouts whose local grids, fillable regions, selection controls, and cross-region relations jointly define the recognition target, rather than isolated data tables.

![](images/b9ca1747dcbf008edb53967c7d66c6590cd41ba421f5cf6200e6806c8e37ec0c.jpg)  
Figure 3: Template distribution across coarse and fine domains.

Table 2: Template-level structural statistics.
<table><tr><td>Feature</td><td>Min</td><td>Median</td><td>Mean</td><td>Max</td></tr><tr><td>Semantic sections</td><td>1</td><td>4</td><td>3.9</td><td>7</td></tr><tr><td>Table regions</td><td>1</td><td>4</td><td>4.6</td><td>9</td></tr><tr><td>Field groups</td><td>0</td><td>4</td><td>5.2</td><td>25</td></tr><tr><td>Label-field units</td><td>2</td><td>23.5</td><td>25.0</td><td>59</td></tr><tr><td>Multi-field units</td><td>0</td><td>4</td><td>5.7</td><td>23</td></tr><tr><td>Selection fields</td><td>0</td><td>16</td><td>19.0</td><td>69</td></tr><tr><td>Structural relation edges</td><td>38</td><td>79</td><td>88.1</td><td>204</td></tr><tr><td>Maximum hierarchy depth</td><td>3</td><td>5</td><td>4.7</td><td>8</td></tr></table>

Table 3 covers seven languages plus bilingual Chinese–English templates across three script families. Japanese, English, and Chi nese account for 74.3% of the pool; Arabic contributes right-to-left layouts, while German, Spanish, and Portuguese broaden the Latinscript coverage. This mix helps separate structural-recognition fail ures from assumptions tied to a single script or writing direction.

Table 3: Template language, script, and writing direction.
<table><tr><td>Language</td><td>Script</td><td>Direction</td><td>Templates</td><td>%</td></tr><tr><td>Arabic</td><td>Arabic</td><td>RTL</td><td>8</td><td>11.43</td></tr><tr><td>Chinese</td><td>CJK</td><td>LTR</td><td>11</td><td>15.71</td></tr><tr><td>Chinese-English</td><td>CJK + Latin</td><td>LTR</td><td>2</td><td>2.86</td></tr><tr><td>English</td><td>Latin</td><td>LTR</td><td>19</td><td>27.14</td></tr><tr><td>German</td><td>Latin</td><td>LTR</td><td>2</td><td>2.86</td></tr><tr><td>Japanese</td><td>CJK</td><td>LTR</td><td>22</td><td>31.43</td></tr><tr><td>Portuguese</td><td>Latin</td><td>LTR</td><td>3</td><td>4.29</td></tr><tr><td>Spanish</td><td>Latin</td><td>LTR</td><td>3</td><td>4.29</td></tr><tr><td>Total</td><td></td><td></td><td>70</td><td>100.00</td></tr></table>

![](images/46d5f4f483dde077cb8955c5b23d85fa80fa194eee1cee5a5bc2306f1b2ddc2b.jpg)

Figure 4: Distribution of instance-level semantic structure, answer scale, content composition, and visual properties. Table 4: Within-template instance diversity measured by pairwise diferences.
<table><tr><td>Metric</td><td>Coverage</td><td>Median</td><td>IQR</td></tr><tr><td>Schema disagreement probability</td><td>58.6%</td><td>0.746</td><td>[0.504, 0.849]</td></tr><tr><td>Optional-path Hamming distance</td><td>58.6%</td><td>0.440</td><td>[0.325, 0.471]</td></tr><tr><td>Value-set Jaccard distance</td><td>100.0%</td><td>0.775</td><td>[0.726, 0.831]</td></tr><tr><td>Shared-field exact-value difference</td><td>100.0%</td><td>0.673</td><td>[0.599, 0.727]</td></tr><tr><td>Robust visual-feature distance</td><td>100.0%</td><td>0.094</td><td>[0.076, 0.119]</td></tr><tr><td>Full-answer uniqueness ratio</td><td>97.1%</td><td>1.000</td><td>[1.000, 1.000]</td></tr></table>

## 3.2 Instance-level Statistics

Q2 examines variation among the 7,000 generated instances. Figure 4 summarizes 20 attributes spanning semantic structure, answer scale, content, and appearance: the median instance contains 32 leaf fields, 8 semantic groups, 41.5 tree edges, 28 active answer paths, and 352 answer characters. Numeric values occupy a median 34% of populated fields, while the median non-white-pixel ratio and edge density are 21% and 0.09. The broad distribution tails show that FormStruct-Bench varies structural load, response length, value composition, and rendering conditions rather than producing near-duplicate instances.

Table 4 isolates variation among instances from the same template. For the 58.6% of templates with variable schemas, median schema disagreement and optional-path distance reach 0.746 and 0.440; across all templates, median value-set and shared-field diferences are 0.775 and 0.673. Visual variation is also universal (median distance 0.094), and 97.1% of templates have fully unique answers, with a dataset-level uniqueness ratio of 0.989. These row-wise metrics confirm that repeated generation changes structure, content, and appearance, not only field values.

Table 5: Dificulty tier distribution across the full corpus and test split.
<table><tr><td>Tier</td><td>Name</td><td> $D _ { \mathrm { m a i n } }$  Range</td><td>Corpus (%)</td><td>Test (%)</td><td>Mean  $S _ { \mathrm { f o r m } }$ </td><td>Mean  $C _ { \mathrm { c o n t e x t } }$ </td></tr><tr><td>L1</td><td>Easy</td><td>[0,0.6927)</td><td>15.71</td><td>9.09</td><td>0.3747</td><td>0.2511</td></tr><tr><td>L2</td><td>Medium</td><td>[0.6927, 0.8220)</td><td>34.29</td><td>36.36</td><td>0.4460</td><td>0.3000</td></tr><tr><td>L3</td><td>Hard</td><td>[0.8220, 1.0273)</td><td>34.29</td><td>36.36</td><td>0.5601</td><td>0.3590</td></tr><tr><td>L4</td><td>Expert</td><td>[1.0273, 2.0000]</td><td>15.71</td><td>18.18</td><td>0.7452</td><td>0.3720</td></tr></table>

Table 5 shows that L2 and L3 each account for 34.29% of the corpus and 36.36% of the test split, while L1 and L4 retain the lowand high-dificulty extremes. Mean form-structure and contextcomplexity scores rise from 0.3747/0.2511 at L1 to 0.7452/0.3720 at L4, supporting the intended tier progression. Because the Director derives these tiers from assigned constraints, each sample carries its dificulty label without additional annotation.

## 3.3 Test Split Design and Distributional Coverage

Q3 asks how a template-disjoint partition balances leakage prevention with evaluation coverage. As shown in Figure 5(a), we assign entire templates, rather than individual instances, to the training, validation, and test splits. The resulting 49/10/11-template partition corresponds to 4,900/1,000/1,100 instances, or 70.0%/14.3%/15.7% of the corpus, because every template contributes 100 instances. This construction prevents the same layout from appearing in both training and evaluation. All 1,100 test instances pass automated verification and additionally receive human review.

![](images/7c053fa8cd1cd41589091e77243b4a296b4751e877a741aaa95734eecd33314b.jpg)  
Figure 5: Template-disjoint split composition and distributional coverage.

The template-disjoint split favors leakage control over proportional domain matching. Figure 5(b) shows that government and immigration and business and operations account for 26.53% and 16.33% of the training templates, respectively, but have no test templates. Conversely, finance, banking, and utilities increases from

12.24% in training to 36.36% in test, while education and student services increases from 8.16% to 27.27%. The maximum train–test gap is therefore 26.53 percentage points. Since the test split contains only 11 templates, each template changes a domain share by 9.09 percentage points; aggregate test results should thus be interpreted as performance on the represented domains rather than as corpus-wide domain estimates.

Figure 5(c) reveals a complementary concentration in structural scale. The training split spans one to eight regions per template, whereas the test split contains only four to six; six of its eleven templates (54.55%) contain five regions. The resulting Jensen–Shannon divergence of 0.355 indicates a clear separation between the two discrete distributions. Consequently, the test split provides a fully reviewed, template-leakage-free diagnostic subset for mid-range region complexity, but it does not measure generalization to the low- and high-region-count tails. We therefore bound domain- and complexity-level conclusions to the coverage visible in Figure 5 rather than claiming distributional representativeness.

## 4 Evaluation Protocol

FormStruct-Bench evaluates predictions at three complementary levels: page-level metrics measure schema and content recovery, path-level metrics test hierarchical field binding, and component metrics assess regions and line-item groups. Three structure-specific diagnostics additionally evaluate region-local grids, widget groups, and typed relations. We report the levels separately to distinguish recognition, binding, localization, and organization errors.

## 4.1 Page-level Metrics

Let �<sup>ˆ</sup> and �<sup>∗</sup> denote the predicted and ground-truth answer JSON objects for a page.

Schema Normalized Tree Edit Similarity (Schema-nTED). Schema-nTED compares schema trees � (�) that retain field names, container types, and hierarchy while discarding values:

$$
\mathrm { S c h e m a - n T E D } = 1 - \frac { d _ { \mathrm { A P T E D } } \Big ( T ( \hat { A } ) , T ( A ^ { * } ) \Big ) } { | T ( \hat { A } ) | + | T ( A ^ { * } ) | } .\tag{2}
$$

It equals 1 only when the predicted and ground-truth schemas match, independently of recognized values.

Value Normalized Edit Similarity (Value-nED). Let $\hat { V }$ and $V ^ { * }$ be the multisets of non-empty leaf values in �<sup>ˆ</sup> and �<sup>∗</sup>. Value-nED uses normalized Levenshtein similarity � and maximum-weight one-to-one matching:

$$
\operatorname { V a l u e - n E D } = \frac { \operatorname* { m a x } _ { M \in \mathcal { B } ( \hat { V } , V ^ { * } ) } \sum _ { ( \hat { v } , v ^ { * } ) \in M } s \big ( \hat { v } , v ^ { * } \big ) } { \operatorname* { m a x } ( | \hat { V } | , | V ^ { * } | ) } ,\tag{3}
$$

where $\mathcal { B } ( \hat { V } , V ^ { * } )$ denotes the set of bipartite matchings. Unmatched values score zero, penalizing missing and spurious content independently of field paths.

## 4.2 Component-level Metrics

Table Structure Recognition Path Accuracy (TSR-path). Let $P ( A )$ be the set of leaf field-path–value pairs obtained by flattening an answer tree. TSR-path counts a ground-truth leaf only when its

complete path and value are reproduced exactly:

$$
\mathrm { T S R - p a t h } = \frac { \sum _ { ( p , v ) \in P ( A ^ { * } ) } \mathbb { 1 } \left[ ( p , v ) \in P ( \hat { A } ) \right] } { | P ( A ^ { * } ) | } .\tag{4}
$$

It is therefore strict ground-truth leaf recall rather than a precisionaware F1 score.

Region Detection F1 at IoU 0.5 (R-F1@0.5). Let $\hat { R }$ and $R ^ { * }$ denote the predicted and ground-truth region sets, and let $M _ { R }$ be the maximum one-to-one matching over pairs with the same semantic type and box IoU of at least 0.5. We compute

$$
\mathrm { R - F } 1 @ 0 . 5 = \frac { 2 | \mathcal { M } _ { R } | } { | \hat { R } | + | R ^ { * } | } .\tag{5}
$$

Thus, a region requires both correct type and location.

Line-item-group F1 (LIG-F1). Let $\hat { G }$ and $G ^ { * }$ denote predicted and ground-truth line-item-group boxes, and let $M _ { G }$ be their maximum one-to-one matching at an IoU threshold of 0.5. LIG-F1 is

$$
\mathrm { L I G - F } 1 = \frac { 2 | { \cal M } _ { G } | } { | \hat { G } | + | G ^ { * } | } .\tag{6}
$$

It measures whether repeated line-item blocks are localized as coherent groups.

Local-grid GriTS Topology $\mathrm { ( L G \mathrm { - } G r i T S _ { T o p } ) } .$ . Let $\hat { g }$ and $\mathcal { G } ^ { * }$ be the predicted and ground-truth region-local grids. We match grids oneto-one only when their parent regions are paired by $M _ { R } ,$ using $s _ { \mathrm { T o p } } = \mathrm { G r i T S } _ { \mathrm { T o p } } \ [ 3 3 ]$ as the pair weight. For the maximum-weight matching $\mathcal { M } _ { \mathrm { L G } }$

$$
\mathrm { L G - G r i T S } _ { \mathrm { T o p } } = \frac { 2 \sum _ { ( \hat { g } , g ^ { * } ) \in { \mathcal M } _ { \mathrm { L G } } } s _ { \mathrm { T o p } } ( \hat { g } , g ^ { * } ) } { | \hat { g } | + | { \mathcal G } ^ { * } | } .\tag{7}
$$

This score isolates row/column topology and span recovery within local grids.

Widget Grouping F1 (WG-F1). Let $\hat { \mathcal W }$ and $\mathbf { \mathcal { W } } ^ { * }$ denote widget groups and $M _ { W }$ their maximum valid one-to-one matching. A valid group match requires the same parent field, widget family, option membership, and observable selection states after member alignment. We compute

$$
\mathrm { W G - F } \imath = \frac { 2 | M _ { W } | } { | \hat { \mathcal { W } } | + | \mathcal { W } ^ { * } | } .\tag{8}
$$

Typed Relation F1 (Rel-F1). Let $\hat { \varepsilon }$ and $\varepsilon ^ { * }$ be the predicted and ground-truth directed, typed relation-edge sets. After aligning their endpoint objects, $M _ { E }$ contains edges whose source, target, direction, and relation type all agree. Rel-F1 is

$$
\mathrm { R e l - F } 1 = \frac { 2 \vert M _ { E } \vert } { \vert \hat { \mathcal { E } } \vert + \vert \mathcal { E } ^ { * } \vert } .\tag{9}
$$

## 4.3 Aggregation and Reporting Protocol

All metrics are macro-averaged over pages. Missing, unparsable, or schema-invalid predictions receive zero on applicable metrics, whereas pages without the corresponding ground-truth component are excluded only from that component-level average. We report valid-output rate separately and use valid-only scores solely as coverage diagnostics. Detailed tree construction, normalization, matching, and edge-case rules are provided in Section A.5.

## 5 Experiments

We evaluate 14 API-hosted and locally deployable systems plus two SFT variants to characterize the current state of table-form document structure recognition. We first report overall performance under the hierarchical evaluation protocol, and then analyze how model behavior varies across dificulty levels, structural constraints, visual degradations, and evaluation criteria. These analyses are designed to expose which structural capabilities remain unresolved and where document-level recovery ceases to be a reliable indicator of fine-grained structural correctness. Our code is publicly available at <sup>§</sup> https://github.com/D2I-CUHKSZ/FormStruct-Bench.

## 5.1 Baselines

We compare six API-hosted VLMs with eight locally deployable systems spanning open-weight VLMs, document/OCR parsers, and TSR pipelines, and additionally report SFT variants of Qwen3.5- 9B and Qwen3.6-35B-A3B. This grouping contrasts vendor-hosted capability with reproducible local alternatives; model roles and schema mappings are detailed in Section A.1.

## 5.2 Implementation Protocol

All systems are evaluated on the same page images and normalized to the FormStruct-Bench JSON schema before scoring. VLMs share the same task definition and use deterministic decoding when available, while local pipelines follow their released inference procedures; parsing, repair, and conversion rules are fixed across systems (Section A.3). The prompt is provided in Section A.12.

## 5.3 Overall Performance and Cross-level Gap

Finding 1 : Under our hierarchical evaluation framework, current systems recover document content substantially better than the structures that organize it. As summarized in Table $^ { 6 , }$ the best document-level scores reach 83.85 Value-nED and 57.45 Schema-nTED, whereas the maxima on the primary structure metrics are only 17.91 TSR-path, 12.49 R-F1@0.5, and 14.21 LIG-F1. The structure-specific diagnostics remain lower still, peaking at 5.56 $\mathrm { L G } \mathrm { - G r i T S _ { T o p } }$ , 3.94 WG-F1, and 2.81 Rel-F1. This cross-level gap answers our first question: existing systems can read form content, but they still struggle to bind it to the correct hierarchy, region, local grid, widget group, and relation endpoint.

Finding 2 : SFT improves content recovery, region localization, and grouping, but does not uniformly improve all structural capabilities. Qwen3.6-35B-A3B-SFT achieves the best Schema-nTED (57.45), R-F1@0.5 (12.49), LIG-F1 (14.21), and WG-F1 (3.94), while Qwen3.5-9B-SFT leads Value-nED (83.85). However, relative to its base model, Qwen3.6-35B-A3B-SFT drops from 17.91 to 16.84 on TSR-path and from 5.56 to 1.00 on $\mathrm { L G - G r i T S _ { T o p } ; }$ its 2.24 Rel-F1 also remains below Seed 2.1 Pro’s 2.81. The uneven gains indicate that supervised adaptation strengthens selected output components rather than resolving the broader coupling problem among document reading, spatial grounding, topology reconstruction, and relation binding.

## 5.4 Dificulty Analysis

Finding 3 : Increasing structural dificulty widens the gap between content recovery and spatial grounding. As shown in Figure 6, across the L1–L4 tiers, document-level value recovery is generally more stable than region-level structural localization. GPT-5.5 and Seed 2.1 Pro maintain comparatively stable Value-nED, whereas Seed 2.1 Pro’s R-F1@0.5 decreases from 13.76 at L2 to 7.23 at L4, a 47.5% relative reduction. Qwen3.6-35B-A3B exhibits degradation at both levels, but its region score drops more sharply: Value-nED decreases from 73.85 at L1 to 58.37 at L4, while R-F1@0.5 falls from 7.09 to 4.07, corresponding to relative reductions of 21.0% and 42.6%, respectively. This asymmetric degradation suggests that current multimodal models can continue recovering plausible field content from global semantic cues as structural and contextual complexity increases, but struggle to preserve precise region iden tities and bind that content to the correct local organization. The near-flat R-F1@0.5 trajectory of GPT-5.5 should not be interpreted as robustness, since the model already operates close to the metric floor.

Table 6: Main results on the five primary metrics and three structure-specific diagnostics of FormStruct-Bench. All values are percentages. Bold marks the best reported result in each column.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Model Size</td><td colspan="2">Document-level Metrics</td><td colspan="3">Primary Structure Metrics</td><td colspan="3">Structure-specific Diagnostics</td></tr><tr><td>Schema-nTED ↑</td><td>Value-nED ↑</td><td>TSR-path↑</td><td>R-F1@0.5 ↑</td><td>LIG-F1 ↑</td><td> $\mathrm { L G - G r i T S _ { T o p } } \uparrow$ </td><td>WG-F1 ↑</td><td>Rel-F1 ↑</td></tr><tr><td>API-hosted</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-5.5 [24]</td><td>Undisc.</td><td>54.15</td><td>80.98</td><td>17.15</td><td>1.74</td><td>2.07</td><td>0.53</td><td>0.02</td><td>0.09</td></tr><tr><td>AClaude Sonnet 5 [2]</td><td>Undisc.</td><td>52.22</td><td>75.42</td><td>13.77</td><td>1.75</td><td>0.52</td><td>1.77</td><td>0.02</td><td>0.13</td></tr><tr><td>Gemini 3.5 Flash [12]</td><td>Undisc.</td><td>48.59</td><td>66.80</td><td>14.55</td><td>8.19</td><td>4.21</td><td>4.26</td><td>1.99</td><td>1.87</td></tr><tr><td>Qwen3.7-Plus [27]</td><td>Undisc.</td><td>34.01</td><td>74.02</td><td>9.75</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>Kimi2.6 [23]</td><td>Undisc.</td><td>48.20</td><td>76.63</td><td>12.96</td><td>0.47</td><td>0.19</td><td>0.00</td><td>0.05</td><td>0.09</td></tr><tr><td>Seed 2.1 Pro [6]</td><td>Undisc.</td><td>52.81</td><td>77.45</td><td>15.90</td><td>10.77</td><td>12.10</td><td>5.43</td><td>0.86</td><td>2.81</td></tr><tr><td>Locally deployable</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3.6-35B-A3B [29]</td><td>35B / 3B act.</td><td>48.46</td><td>66.93</td><td>17.91</td><td>5.95</td><td>7.65</td><td>5.56</td><td>0.40</td><td>0.00</td></tr><tr><td>Qwen3.6-35B-A3B-SFT [29]</td><td>35B / 3B act.</td><td>57.45</td><td>77.83</td><td>16.84</td><td>12.49</td><td>14.21</td><td>1.00</td><td>3.94</td><td>2.24</td></tr><tr><td> DeepSeek-VL-2 [38]</td><td>27B</td><td>5.89</td><td>5.29</td><td>0.00</td><td>0.00</td><td>0.01</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>Qwen3.5-9B [28]</td><td>9B</td><td>45.58</td><td>65.73</td><td>12.58</td><td>2.03</td><td>1.84</td><td>1.23</td><td>0.01</td><td>0.00</td></tr><tr><td>Qwen3.5-9B-SFT [28]</td><td>9B</td><td>56.98</td><td>83.85</td><td>17.78</td><td>10.81</td><td>9.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>GLM-4.6V-Flash [43]</td><td>9B</td><td>30.41</td><td>47.88</td><td>4.92</td><td>5.59</td><td>3.66</td><td>1.16</td><td>0.05</td><td>0.00</td></tr><tr><td>Step3-VL-10B [13]</td><td>10B</td><td>26.20</td><td>59.48</td><td>0.04</td><td>0.57</td><td>1.05</td><td>0.00</td><td>0.03</td><td>0.00</td></tr><tr><td>InternVL3.5-8B [36]</td><td>8B</td><td>30.34</td><td>57.03</td><td>0.01</td><td>1.05</td><td>0.00</td><td>0.00</td><td>0.01</td><td>0.00</td></tr><tr><td> MinerU 2.5 Pro [35]</td><td>1.2B</td><td>4.53</td><td>10.97</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>PaddleOCR-VL-1.6 [44]</td><td>0.9B</td><td>2.97</td><td>21.73</td><td>0.00</td><td>3.68</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr></table>

![](images/85c6530f9a30bbe68168223aab5ddce8b0f6e4e1c119afb57de478978aa61462.jpg)  
Figure 6: Performance across Dificulty L1–L4.

## 5.5 Visual Degradation Robustness

Finding 4 : Visual degradation leaves the page-level capability hierarchy largely intact but consistently weakens exact hierarchical binding, while region localization and grouping exhibit model-specific responses. Visual degradation exposes

![](images/fd186a32e0ebaf1083bc2d6cfb3146abc23b6ac6ae861ccce66d43590203d8a0.jpg)  
(a) Degraded page scores

![](images/5bf91d64bd7f5ab8748c39d71e58432f538c94e6c7ffd6e826b81dfb121b77dd.jpg)  
(b) Page ∆ by degradation

![](images/1fc038f62cd1171d9d6bb3e1b367bc5e18f209e49f3cbe29ccc0e144cd1454af.jpg)

![](images/a972cb4bd16e056469a1659244440c2a713072bd9f4e6009a0555bbc75148d18.jpg)  
(c) TSR-path ∆ by severity  
(d) Region ∆ by model  
Figure 7: Visual-degradation robustness: (a) degraded pagelevel scores; (b) page-level changes by degradation type; (c)TSR-path changes by severity; and (d) region-level changes by model.

a clear capability boundary in current multimodal models. Their page-level extraction can remain relatively stable because visible values and document schemas can be recovered from redundant textual and global semantic cues. In contrast, exact hierarchical binding depends more heavily on fragile local evidence, such as boundaries, alignment, spacing, and the visual correspondence between labels and fields. Region localization and line-item grouping are also not afected uniformly, indicating that detecting an approximate document region and preserving the internal grouping of its elements are only partially coupled capabilities. These results suggest that current models lack a visual-structural representation that remains consistently aligned across semantic extraction, hierarchical binding, localization, and grouping when document appearance changes.

Figure 7 supports this boundary. The page-level ranking remains largely unchanged under degradation (Figure 7(a)), and stronger models show only limited degraded–clean changes (Figure 7(b)). In contrast, TSR-path decreases under every degradation type and severity level, with losses of approximately 3.8–6.3 points (Figure 7(c)), showing that exact value-to-path binding is consistently more fragile than page-level extraction. Region-level efects are less uniform (Figure 7(d)): some models preserve region localization while losing line-item grouping, indicating that localization and grouping remain only partially coupled. Overall, visual degradation primarily destabilizes fine-grained structural binding rather than coarse document semantics.

## 5.6 In-Domain Adaptation & Cross-Dataset Transfer

Finding 5 : SFT on synthetic table-form documents improves real-world structure recognition across both model scales, covering content recovery, schema reconstruction, and hierarchical value binding. To test whether the generated data provides transferable supervision rather than merely fitting the synthetic distribution, we fine-tune two model scales on synthetic table-form documents and evaluate them on both synthetic and real-world data. As shown in Figure 8, SFT improves all three metrics on real documents for both models, covering content recovery, schema reconstruction, and hierarchical value binding. The gains in Schema-nTED and TSR-path further show that the models learn structural organization and field-path alignment, not only surface-text recognition. These results indicate that the synthetic data captures transferable structural regularities and can efectively support real-world table-form structure recognition.

## 6 Related Work

Table structure recognition. Existing TSR benchmarks mainly study regular or scientific tables under a global-grid view. Table-Bank and SciTSR cover table detection and scientific-table structure recognition [8, 19], while PubTabNet introduces TEDS to measure tree-edit similarity over HTML-like tables [45]. PubTables-1M scales scientific-document table extraction with canonicalized annotations [32], and GriTS evaluates topology, content, and location similarity [33]. These benchmarks omit form-specific elements such as fillable fields, option groups, local subgrids, and non-local relations; consequently, table-level scores can obscure failures in localization, span recovery, widget grouping, reading order, and local-grid reconstruction.

Document AI and form-like documents. Full-page layout and parsing benchmarks, including PubLayNet, DocBank, DocLayNet, and OmniDocBench [20, 25, 26, 46], are supported by general documentunderstanding systems such as the LayoutLM family, LayoutXLM, Donut, and MinerU [14, 17, 35, 39, 40, 42]. Form and receipt datasets such as FUNSD, SROIE, and DocILE instead focus on key–value and business-information extraction [15, 16, 31]. SRFUND directly benchmarks multi-granularity hierarchical form reconstruction [21], while FormStruct-Bench emphasizes explicit table-form structures and diagnostic evaluation. VAREX closely overlaps with our template-based synthesis, deterministic ground truth, and qualityassurance workflow [5], but targets schema-based value extraction rather than structure reconstruction. These tasks may recover correct values without reconstructing the local grids, widget groups, section membership, and relations that make forms machine-consumable. FormStruct-Bench therefore complements them with an explicit hierarchy over regions, local grids, fields, wid gets, and relations, evaluated through component-level diagnostics rather than a single table or extraction score.

![](images/b6438056cf8ef5e0a72ac1ee26203b4d3e739e5ca87aa19c67482ef59ae1adc1.jpg)  
Figure 8: Models fine-tuned on synthetic table-form documents achieve consistent gains on real-world data.

## 7 Conclusion

FormStruct-Bench reframes table-form document structure recognition as hierarchical recovery over semantic regions, local grids, fields, widgets, and typed relations. Its 70 templates and 7,000 verified instances support five complementary metrics and diagnostic slices over dificulty, structural constraints, and visual degradation. Across 14 systems and two SFT variants, document-level performance reaches 83.85%, yet the strongest fine-grained structural score remains below 18%, with no system dominating all components. Within our template-disjoint test coverage, this gap shows that recovering content does not imply recovering the hierarchy required for reliable table-form understanding. FormStruct-Bench thus measures progress at the level of structural failure rather than aggregate output similarity.

## References

[1] Milan Aggarwal, Mausoom Sarkar, Hiresh Gupta, and Balaji Krishnamurthy. 2021. Multi-Modal Association based Grouping for Form Structure Extraction. arXiv:2107.04396 [cs.CV] https://arxiv.org/abs/2107.04396

[2] Anthropic. 2026. Claude Sonnet 5. Accessed: 2026-07-07. https://platform.claude. com/docs/en/about-claude/models/whats-new-sonnet-5

[3] Srikar Appalaraju, Bhavan Jasani, Bhargava Urala Kota, Yusheng Xie, and R. Manmatha. 2021. DocFormer: End-to-End Transformer for Document Understanding. In Proceedings of the IEEE/CVF International Conference on Computer Vision. IEEE, Montreal, QC, Canada, 973–983. doi:10.1109/ICCV48922.2021.00103

[4] Deniz Appelbaum, Alexander Kogan, and Miklos A. Vasarhelyi. 2017. Big Data and Analytics in the Modern Audit Engagement: Research Needs. Auditing: A Journal ofPractice & Theory 36, 4 (2017), 1–27. doi:10.2308/ajpt-51684

[5] Udi Barzelay, Ophir Azulai, Inbar Shapira, Idan Friedman, Foad Abo Dahood, Madison Lee, and Abraham Daniels. 2026. VAREX: A Benchmark for Multi-Moda Structured Extraction from Documents. arXiv:2603.15118 [cs.CV] doi:10.48550 arXiv.2603.15118

[6] ByteDance Seed. 2026. Seed2.1 Model Card. Accessed: 2026-07-07. https://lf3-static.bytednsdoc.com/obj/eden-cn/lapzild-tss/ljhwZthlaukjlkulzlp/ seed2.1/Seed2\_1\_Model\_Card.pdf

[7] Tathagata Chakraborti, Vatche Isahagian, Rania Khalaf, Yasaman Khazaeni, Vinod Muthusamy, Yara Rizk, and Merve Unuvar. 2020. From Robotic Process Automation to Intelligent Process Automation. arXiv:2007.13257 [cs.AI] https://arxiv.org/abs/2007.13257

[8] Zewen Chi, Heyan Huang, Heng-Da Xu, Houjin Yu, Wanxuan Yin, and Xian-Ling Mao. 2019. Complicated Table Structure Recognition. arXiv:1908.04729 [cs.IR] https://arxiv.org/abs/1908.04729

[9] Common Crawl Foundation. 2024. Common Crawl Terms of Use. Terms for use of Common Crawl services and crawled content.. https://commoncrawl.org/termsof-use

[10] Common Crawl. 2026. Common Crawl Dataset. https://commoncrawl.org/

[11] doc-analysis. 2022. XFUND: Repository License. CC BY-NC-SA 4.0 license statement.. https://github.com/doc-analysis/XFUND

[12] Google AI for Developers. 2026. Gemini API Models. Accessed: 2026-07-07. https://ai.google.dev/gemini-api/docs/models

[13] Ailin Huang, Chengyuan Yao, Chunrui Han, Fanqi Wan, Hangyu Guo, Haoran Lv, Hongyu Zhou, Jia Wang, Jian Zhou, Jianjian Sun, Jingcheng Hu, Kangheng Lin, Liang Zhao, Mitt Huang, Song Yuan, Wenwen Qu, Xiangfeng Wang, Yanlin Lai, Yingxiu Zhao, Yinmin Zhang, Yukang Shi, Yuyang Chen, Zejia Weng, Ziyang Meng, Ang Li, Aobo Kong, Bo Dong, Changyi Wan, David Wang, Di Qi, Dingming Li, En Yu, Guopeng Li, Haiquan Yin, Han Zhou, Hanshan Zhang, Haolong Yan, Hebin Zhou, Hongbo Peng, Jiaran Zhang, Jiashu Lv, Jiayi Fu, Jie Cheng, Jie Zhou, Jisheng Yin, Jingjing Xie, Jingwei Wu, Jun Zhang, Junfeng Liu, Kaijun Tan, Kaiwen Yan, Liangyu Chen, Lina Chen, Mingliang Li, Qian Zhao, Quan Sun, Shaoliang Pang, Shengjie Fan, Shijie Shang, Siyuan Zhang, Tianhao You, Wei Ji, Wuxun Xie, Xiaobo Yang, Xiaojie Hou, Xiaoran Jiao, Xiaoxiao Ren, Xiangwen Kong, Xin Huang, Xin Wu, Xing Chen, Xinran Wang, Xuelin Zhang, Yana Wei, Yang Li, Yanming Xu, Yeqing Shen, Yuang Peng, Yue Peng, Yu Zhou, Yusheng Li, Yuxiang Yang, Yuyang Zhang, Zhe Xie, Zhewei Huang, Zhenyi Lu, Zhimin Fan, Zihui Cheng, Daxin Jiang, Qi Han, Xiangyu Zhang, Yibo Zhu, and Zheng Ge. 2026. STEP3-VL-10B Technical Report. arXiv:2601.09668 [cs.CV] doi:10.48550 arXiv.2601.09668

[14] Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and Furu Wei. 2022. LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking. In Proceedings ofthe 30th ACM International Conference on Multimedia. ACM, New York, NY. USA, 4083-4091, doi:10.1145/3503161.3548112

[15] Zheng Huang, Kai Chen, Jianhua He, Xiang Bai, Dimosthenis Karatzas, Shjian Lu, and C. V. Jawahar. 2021. ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction. arXiv:2103.10213 [cs.AI] doi:10.1109/ICDAR.2019.00244

[16] Guillaume Jaume, Hazim Kemal Ekenel, and Jean-Philippe Thiran. 2019. FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents. arXiv:1905.13538 [cs.IR] https://arxiv.org/abs/1905.13538

[17] Geewook Kim, Teakgyu Hong, Moonbin Yim, Jeongyeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. 2022. OCR-free Document Understanding Transformer. arXiv:2111.15664 [cs.LG] https://arxiv.org/abs/2111.15664

[18] Chen-Yu Lee, Chun-Liang Li, Timothy Dozat, Vincent Perot, Guolong Su, Nan Hua, Joshua Ainslie, Renshen Wang, Yasuhisa Fujii, and Tomas Pfister. 2022. Form-Net: Structural Encoding beyond Sequential Modeling in Form Document Information Extraction. In Proceedings ofthe 60th Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, Dublin, Ireland, 3735–3754. doi:10.18653/v1/2022.acl-long.260

[19] Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, Ming Zhou, and Zhoujun Li. 2020. TableBank: A Benchmark Dataset for Table Detection and Recognition. arXiv:1903.01949 [cs.CV] https://arxiv.org/abs/1903.01949

[20] Minghao Li, Yiheng Xu, Lei Cui, Shaohan Huang, Furu Wei, Zhoujun Li, and Ming Zhou. 2020. DocBank: A Benchmark Dataset for Document Layout Analysis.

arXiv:2006.01038 [cs.CL] https://arxiv.org/abs/2006.01038

[21] Jiefeng Ma, Yan Wang, Chenyu Liu, Jun Du, Yu Hu, Zhenrong Zhang, Pengfei Hu, Qing Wang, and Jianshu Zhang. 2024. SRFUND: A Multi-Granularity Hierarchical Structure Reconstruction Benchmark in Form Understanding. In Advances in Neural Information Processing Systems, Vol. 37. Curran Associates, Inc., Red Hook, NY, USA, 112411–112432. doi:10.52202/079017-3571

[22] Minesh Mathew, Dimosthenis Karatzas, and C. V. Jawahar. 2021. DocVQA: A Dataset for VQA on Document Images. In Proceedings ofthe IEEE/CVF Winter Conference on Applications of Computer Vision. IEEE, Online, 2200–2209. doi:10. 1109/WACV48630.2021.00225

[23] Moonshot AI. 2026. Kimi. Accessed: 2026-07-07. https://www.kimi.com/

[24] OpenAI. 2026. OpenAI API Models. Accessed: 2026-07-07. https://developers. openai.com/api/docs/models

[25] Linke Ouyang, Yuan Qu, Hongbin Zhou, Jiawei Zhu, Rui Zhang, Qunshu Lin, Bin Wang, Zhiyuan Zhao, Man Jiang, Xiaomeng Zhao, Jin Shi, Fan Wu, Pei Chu, Minghao Liu, Zhenxiang Li, Chao Xu, Bo Zhang, Botian Shi, Zhongying Tu, and Conghui He. 2025. OmniDocBench: Benchmarking Diverse PDF Document Parsing with Comprehensive Annotations. arXiv:2412.07626 [cs.CV] https: //arxiv.org/abs/2412.07626

[26] Birgit Pfitzmann, Christoph Auer, Michele Dolfi, Ahmed S. Nassar, and Peter Staar. 2022. DocLayNet: A Large Human-Annotated Dataset for Document-Layout Segmentation. In Proceedings ofthe 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. ACM, New York, NY, USA, 3743–3751. doi:10.1145/3534678.3539043

[27] Qwen Team. 2026. Qwen: Oficial Models and Chat Service. Accessed: 2026-07-07. https://qwen.ai/

[28] Qwen Team. 2026. Qwen3.5: Towards Native Multimodal Agents. https://qwen. ai/blog?id=qwen3.5

[29] Qwen Team. 2026. Qwen3.6-35B-A3B: Agentic Coding Power, Now Open to All. https://qwen.ai/blog?id=qwen3.6-35b-a3b

[30] Josselin Somerville Roberts, Tony Lee, Chi Heem Wong, Michihiro Yasunaga, Yifan Mai, and Percy Liang. 2024. Image2Struct: Benchmarking Structure Extraction for Vision-Language Models. NeurIPS 2024. arXiv:2410.22456 [cs.CV] https://arxiv.org/abs/2410.22456

[31] Štěpán Šimsa, Milan Šulc, Michal Uřičář, Yash Patel, Ahmed Hamdi, Matěj Kocián, Matyáš Skalický, Jiří Matas, Antoine Doucet, Mickaël Coustaty, and Dimosthenis Karatzas. 2023. DocILE Benchmark for Document Information Localization and Extraction. arXiv:2302.05658 [cs.CL] https://arxiv.org/abs/2302.05658

[32] Brandon Smock, Rohith Pesala, and Robin Abraham. 2021. PubTables-1M: Towards comprehensive table extraction from unstructured documents. arXiv:2110.00061 [cs.LG] https://arxiv.org/abs/2110.00061

[33] Brandon Smock, Rohith Pesala, and Robin Abraham. 2023. GriTS: Grid table similarity metric for table structure recognition. arXiv:2203.12555 [cs.LG] https: //arxiv.org/abs/2203.12555

[34] Wil M. P. van der Aalst, Martin Bichler, and Armin Heinzl. 2018. Robotic Process Automation. Business & Information Systems Engineering 60, 4 (2018), 269–272. doi:10.1007/s12599-018-0542-4

[35] Bin Wang, Chao Xu, Xiaomeng Zhao, Linke Ouyang, Fan Wu, Zhiyuan Zhao, Rui Xu, Kaiwen Liu, Yuan Qu, Fukai Shang, Bo Zhang, Liqun Wei, Zhihao Sui, Wei Li, Botian Shi, Yu Qiao, Dahua Lin, and Conghui He. 2024. MinerU: An Open-Source Solution for Precise Document Content Extraction. arXiv:2409.18839 [cs.CV] https://arxiv.org/abs/2409.18839

[36] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, Zhaokai Wang, Zhe Chen, Hongjie Zhang, Ganlin Yang, Haomin Wang, Qi Wei, Jinhui Yin, Wenhao Li, Erfei Cui, Guanzhou Chen, Zichen Ding, Changyao Tian, Zhenyu Wu, Jingjing Xie, Zehao Li, Bowen Yang, Yuchen Duan, Xuehui Wang, Zhi Hou, Haoran Hao, Tianyi Zhang, Songze Li, Xiangyu Zhao, Haodong Duan, Nianchen Deng, Bin Fu, Yinan He, Yi Wang, Conghui He, Botian Shi, Junjun He, Yingtong Xiong, Han Lv, Lijun Wu, Wenqi Shao, Kaipeng Zhang, Huipeng Deng, Biqing Qi, Jiaye Ge, Qipeng Guo, Wenwei Zhang, Songyang Zhang, Maosong Cao, Junyao Lin, Kexian Tang, Jianfei Gao, Haian Huang, Yuzhe Gu, Chengqi Lyu, Huanze Tang, Rui Wang, Haijun Lv, Wanli Ouyang, Limin Wang, Min Dou, Xizhou Zhu, Tong Lu, Dahua Lin, Jifeng Dai, Weijie Su, Bowen Zhou, Kai Chen, Yu Qiao, Wenhai Wang, and Gen Luo. 2025. InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Eficiency. arXiv:2508.18265 [cs.CV] doi:10. 48550/arXiv.2508.18265

[37] Yandi Wang, Libin Zhan, Ziwei Huang, Tiancheng Luo, YuxuanJiang, Wang Dong, Leilei Gan, and Jun Chen. 2026. From Recognition to Reasoning: Benchmarking and Enhancing MLLMs on Real-World Receipt Document Understanding. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, San Diego, California, United States, 46007–46024. https://aclanthology.org/2026. acl-long.2135/

[38] Haoran Wei, Yaofeng Sun, and Yukun Li. 2026. DeepSeek-OCR 2: Visual Causal Flow. arXiv:2601.20552 [cs.CV] doi:10.48550/arXiv.2601.20552

[39] Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, and Ming Zhou. 2020. LayoutLM: Pre-training of Text and Layout for Document Image Understanding. In Proceedings ofthe 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, New York, NY, USA, 1192–1200. doi:10.1145/ 3394486.3403172

[40] Yiheng Xu, Tengchao Lv, Lei Cui, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, and Furu Wei. 2021. LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding. arXiv:2104.08836 [cs.CL] https://arxiv. org/abs/2104.08836

[41] Yiheng Xu, Tengchao Lv, Lei Cui, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, and Furu Wei. 2022. XFUND: A Benchmark Dataset for Multilingual Visually Rich Form Understanding. In Findings of the Association for Computational Linguistics: ACL 2022. Association for Computational Linguistics, Dublin, Ireland, 3214–3224. doi:10.18653/v1/2022.findings-acl.253

[42] Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Wanxiang Che, Min Zhang, and Lidong Zhou. 2021. LayoutLMv2: Multi-modal Pre-training for Visually-rich Document Understanding. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing. Association for Computational Linguistics, Online, 2579–2591. doi:10.18653/v1/2021.acl-long.201

[43] Z.ai. 2025. GLM-4.6V Model Card. Accessed: 2026-07-07. https://huggingface. co/zai-org/GLM-4.6V

[44] Zelun Zhang, Hongen Liu, Suyin Liang, Yubo Zhang, Yiqing Xiang, Jiaxuan Liu, Ting Sun, Manhui Lin, Yue Zhang, Changda Zhou, Tingquan Gao, Cheng Cui, Yi Liu, Dianhai Yu, and Yanjun Ma. 2026. PaddleOCR-VL-1.6: Expanding the Frontier of Document Parsing with Under-Optimized Region Refinement and Progressive Post-Training. arXiv:2606.03264 [cs.CV] https://arxiv.org/abs/2606.03264

[45] Xu Zhong, Elaheh ShafieiBavani, and Antonio Jimeno Yepes. 2020. Image-based table recognition: data, model, and evaluation. arXiv:1911.10683 [cs.CV] https: //arxiv.org/abs/1911.10683

[46] Xu Zhong, Jianbin Tang, and Antonio Jimeno Yepes. 2019. PubLayNet: largest dataset ever for document layout analysis. arXiv:1908.07836 [cs.CL] https: //arxiv.org/abs/1908.07836

## A Additional Experimental Details

## A.1 Baseline Systems

The API-hosted group comprises GPT-5.5 [24], Claude Sonnet 5 [2], Gemini 3.5 Flash [12], Qwen3.7-Plus [27], Kimi2.6 [23], and Seed 2.1 Pro [6]. These systems directly generate the complete target JSON and represent vendor-hosted multimodal models available only through remote inference services.

The locally deployable VLMs include Qwen3.6-35B-A3B [29], DeepSeek-VL-2 [38], Qwen3.5-9B [28], and GLM-4.6V-Flash [43]. We additionally evaluate Step3-VL-10B [13] and InternVL3.5-8B [36]. The locally deployable document/OCR pipelines are MinerU 2.5 Pro [35] and PaddleOCR-VL-1.6. The VLMs are prompted to generate the target schema, whereas the document/OCR pipelines retain their released output interfaces. For the latter, only supported predictions are mapped into the closest compatible subset of the FormStruct-Bench schema; components not produced by a pipeline remain absent after conversion.

## A.2 Experimental Environment

Hardware. Local inference was conducted on a server with two NVIDIA A100-SXM4 GPUs (80 GB each; 81,920 MiB reported per device) using driver 550.90.07, with Multi-Instance GPU (MIG) disabled. The preserved host has two 48-core AMD EPYC 7A23 CPUs (96 physical cores and 192 logical CPUs) and approximately 1 TiB of RAM, and runs Debian GNU/Linux 13 user space on a Linux 5.15 kernel. The surviving inference logs corroborate the GPU model and driver for the formal runs; CPU, memory, and operating-system details were not stored in per-run manifests and are reported from the preserved host snapshot.

Software. Table 7 summarizes the package versions that can be verified from preserved environments and formal server logs. The NVIDIA driver reports CUDA 12.4 compatibility, whereas the main vLLM stack uses CUDA 12.8 wheels and the Paddle stack uses CUDA 12.6 wheels; we therefore report these layers separately rather than assigning a single CUDA version to all runs.

Table 7: Verified software environments for locally deployable baselines.
<table><tr><td>Stack</td><td>Verified environment</td></tr><tr><td>vLLM</td><td>Python 3.12.8; PyTorch 2.10.0+cu128; Transformers 4.57.6; vLLM 0.18.0.</td></tr><tr><td>MinerU</td><td>MinerU-VL-Utils 1.0.5 with its in-process vLLM inference engine.</td></tr><tr><td>PaddleOCR-VL</td><td>SGLang 0.5.2; PyTorch 2.8.0; Transformers 4.56.1; PaddlePaddle-GPU 3.2.1; PaddleOCR 3.7.0; PaddleX 3.7.2.</td></tr></table>

## A.3 Inference and Output Normalization

Each VLM receives the original page image and the same instruction to return a JSON object containing a semantic answer tree, regions, widgets, local grids, cells, and relations. The shared pre diction prompt is reproduced verbatim in Section A.12. We use deterministic or near-deterministic decoding whenever the API or inference backend exposes the corresponding control. Locally deployable document/OCR and TSR systems are run with their released image-to-table or document-parsing procedures.

Table 8: Hyperparameters used for FormStruct-Bench LoRAbased SFT.
<table><tr><td>Hyperparameter</td><td>Qwen3.5-9B</td><td>Qwen3.6-35B-A3B</td></tr><tr><td>Optimization method</td><td>LoRA SFT</td><td>LoRA SFT</td></tr><tr><td>Epochs / optimizer steps</td><td>1 / 307</td><td>1 / 307</td></tr><tr><td>Peak learning rate</td><td> $5 \times { 1 0 } ^ { - 5 }$ </td><td> $5 \times { 1 0 } ^ { - 5 }$ </td></tr><tr><td>Optimizer</td><td>Fused AdamW</td><td>Fused AdamW</td></tr><tr><td>Weight decay</td><td>0</td><td>0</td></tr><tr><td>LR schedule</td><td>Cosine, 3% warmup</td><td>Cosine, 3% warmup</td></tr><tr><td>Per-GPU batch size</td><td>2</td><td>1</td></tr><tr><td>Gradient accumulation</td><td>4</td><td>8</td></tr><tr><td>Effective global batch size</td><td>16</td><td>16</td></tr><tr><td>Maximum sequence length</td><td>12,288</td><td>14,336</td></tr><tr><td>Maximum image pixels</td><td>1,048,576</td><td>1,048,576</td></tr><tr><td>LoRA rank r</td><td>16</td><td>16</td></tr><tr><td>LoRA scaling α</td><td>32</td><td>32</td></tr><tr><td>LoRA dropout</td><td>0.05</td><td>0.05</td></tr><tr><td>LoRA bias</td><td>None</td><td>None</td></tr><tr><td>Random seed</td><td>42</td><td>42</td></tr></table>

Predictions are processed by the same task-agnostic JSON parser and schema-conversion pipeline. The parser may repair invalid JSON syntax, but it does not infer or manually complete missing fields; any field absent after parsing remains missing during evaluation. Native pipeline outputs are then converted into their compatible FormStruct-Bench components before the documentand component-level metrics are computed.

## A.4 Supervised Fine-Tuning Details

We performed parameter-eficient supervised fine-tuning (SFT) on the FormStruct-Bench training split using LoRA. Each SFT example consisted of a rendered form image, a task instruction, and a structured JSON target. Only assistant-side JSON tokens contributed to the language-modeling loss; image and instruction tokens were masked. Both models were trained for one epoch, corresponding to 307 optimizer updates. Validation was performed at step 300. After training, the LoRA adapters were merged into their respective base models in bfloat16 precision using safe weight merging.

## A.5 Metric Computation and Edge Cases

Schema-nTED.. Each answer is converted into a schema tree � (�) that retains field names, container types (object, array, or scalar), and hierarchy while discarding field values. Object keys are stripped and sorted so that serialized object order does not afect the score, whereas array order is preserved. In the APTED distance used by Equation (2), insertion and deletion cost 1; renaming costs 0 for identical node labels and 2 otherwise.

Value-nED.. Before extracting the multisets of non-empty leaf values, we apply Unicode NFKC, lowercasing, whitespace collapsing, and removal of spaces or commas between digits. Pairwise

similarity is

$$
s ( \hat { v } , v ^ { * } ) = 1 - \frac { d _ { \mathrm { L e v } } ( \hat { v } , v ^ { * } ) } { \operatorname* { m a x } ( | \hat { v } | , | v ^ { * } | ) } .\tag{10}
$$

The maximum-weight bipartite matching in Equation (3) is oneto-one; unmatched predicted or ground-truth values contribute zero.

TSR-path. We recursively flatten each answer tree into leaf fieldpath–value pairs. Strings are stripped and repeated whitespace is collapsed, but no fuzzy matching, case folding, or Unicode normalization is applied. Extra predicted fields do not change the denominator in Equation (4); the metric is therefore strict ground-truth field recall.

Region and line-item-group matching. For R-F1@0.5, a region pair is a valid matching candidate only when its semantic types agree and box IoU is at least 0.5. We then compute the maximumcardinality one-to-one matching $M _ { R }$ used in Equation (5). For LIG-F1, region types are not involved: predicted and ground-truth lineitem-group boxes are matched one-to-one whenever their IoU is at least 0.5, producing $M _ { G }$ in Equation (6).

Region-local grid topology. For $\mathrm { L G } { \mathrm { - G r i T S _ { \mathrm { T o p } } } }$ , each local grid is evaluated only inside its associated semantic region. Widgets are removed before grid construction. Following GriTS [33], a cell with top-left grid position $( \rho , \theta )$ , rowspan $\alpha ,$ and colspan $\beta$ is repeated at every grid position $( i , j )$ it occupies and represented there by the topology box

$$
\begin{array} { r } { b _ { i , j } ^ { \mathrm { T o p } } = [ \theta - j , \rho - i , \theta - j + \beta , \rho - i + \alpha ] . } \end{array}\tag{11}
$$

The oficial factored two-dimensional most-similar-substructure alignment compares these boxes with IoU, yielding $s _ { \mathrm { T o p } } \in \left[ 0 , 1 \right]$ We use these scores as weights in a maximum-weight one-to-one matching between predicted and ground-truth grids whose parent regions are already paired by $M _ { R } .$ The denominator in Equation (7) therefore penalizes omitted and spurious whole grids, while the GriTS term penalizes incorrect rows, columns, and spans within a matched grid. A malformed grid with non-positive spans, conflicting cell occupancy, or no valid cells receives zero.

Widget-group matching. A canonical widget group consists of one parent field and all widgets attached to it by field-to-widget relations; explicit group identifiers, when available, must induce the same partition. Raw prediction identifiers are not compared. Parent fields are aligned by their canonical schema paths within matched parent regions, after which member widgets are matched one-to-one at IoU ≥ 0.5. A member match requires the same widget type and, when a ground-truth state is defined, the same selected/unselected state. A widget group contributes to $M _ { W }$ only when the parent fields correspond and all members admit a bijection, so a missing, extra, mistyped, or state-inconsistent member makes the group incorrect. Visible label text is excluded to keep WG-F1 focused on grouping rather than OCR.

Typed-relation matching. Each relation is canonicalized as a directed triple $( u , \ell , v )$ , where ℓ is one of the released relation types, including key-to-field, field-to-widget, section-membership, and line-item-membership links. We construct a partial endpoint map $\phi \colon$ regions use ${ \cal { M } } _ { R } ;$ fields use exact canonical schema paths within matched regions; widgets use the member matches from WG-F1; local cells use the alignment induced by $\mathcal { M } _ { \mathrm { L G } } ;$ and line-item groups use $M _ { G } . \mathrm { A }$ predicted edge $( \hat { u } , \hat { \ell } , \hat { v } )$ is in $M _ { E }$ only if both endpoints are mapped and $( \phi ( \hat { u } ) , \hat { \ell } , \phi ( \hat { v } ) ) \in \mathcal { E } ^ { * }$ . Edge direction and type must match exactly; edges with an unmatched endpoint remain false positives, and duplicate triples are collapsed before scoring.

Aggregation and invalid outputs. Metrics are macro-averaged over pages. Missing, unparsable, or schema-invalid predictions receive zero for every applicable metric. If a ground-truth component exists but is omitted from the prediction, its component score is zero; if the ground-truth component itself is absent, that page is excluded only from the corresponding component-level average. In particular, pages without a ground-truth line-item group are marked not applicable rather than counted as perfect negatives. We report valid-output rate separately, and any score conditioned on valid outputs is treated only as a coverage diagnostic.

## A.6 Qualitative Analysis of Relation Errors

Figure 9 separates relation-semantic errors from failures to localize or align their endpoint objects. In the English example, the model links Date available to start work to the correct value box and preserves the edge direction, but predicts parent-child instead of the ground-truth key-value type. This type substitution makes the predicted edge a false positive and leaves the corresponding ground-truth edge unmatched. At the page level, 11 of 82 predicted relations match 95 ground-truth relations, yielding 12.43% Rel-F1; after restricting the comparison to endpoint-alignable relations, the same 11 matches are scored against 19 predictions and 21 groundtruth relations, yielding 55.00% aligned-endpoint F1.

The Japanese example isolates a complementary failure. The model identifies both the Gender field and the selected Female widget and predicts the field-widget type, but reverses the directed edge from Gender → Female to Female → Gender. The matched, predicted, and ground-truth relation counts are respectively 0, 78, and 75 on the full page, and 0, 4, and 2 within the endpoint-alignable subset, producing 0.00% under both scores. Together, these examples show that low Rel-F1 is not explained solely by failed endpoint localization: even when the relevant objects are successfully aligned, a model can still confuse the relation ontology or its direction. The cases are illustrative and are not used to estimate error prevalence.

## A.7 External Structural Representativeness and Diagnostic Utility

Real-template grounding and leakage-controlled reference. All 7,000 instances in FormStruct-Bench are derived from 70 selected, real, manually structured form templates; synthesis changes field content, schema realization, and visual conditions while remaining anchored to a human-created layout skeleton. We evaluate how well these skeletons align with an external real-form reference using SRFUND [21]. Because pages are not independent layout samples, we cluster SRFUND’s 1,592 pages using rotation-aware image hashes, OCR shingles, normalized entity-box occupancy, aspect ratio, and entity-count agreement, obtaining 1,566 layout clusters. Cross-dataset provenance and visual/OCR matching identify 36 confirmed and one corroborated overlap; excluding all 37 afected clusters leaves 1,529 non-overlapping reference clusters.

![](images/c72782efb1de4577ec7f1e6dd36d048243dfdbb512c2213a93a76d80a640d248.jpg)  
Figure 9: Representative Rel-F1 errors after successful endpoint alignment. Red boxes and bottom insets identify the evaluated relations. Aligned-endpoint F1 is Rel-F1 computed only over edges whose two endpoints can be aligned; its exact type and direction requirements remain unchanged.

Thus neither repeated pages nor source overlap can inflate the comparison.

Real–Real calibrated support. Let T denote FormStruct templates and R denote deduplicated SRFUND layout clusters. We compute a group-balanced Gower distance $d _ { G }$ that first averages correlated features within hierarchy, relation, layout, language, entitycomposition, and table groups, and then weights the groups equally. Direct-only comparisons retain aligned language, hierarchy, untyped relation, and normalized bounding-box features; the mapped analysis additionally includes explicitly marked conditional mappings for entity composition, table structure, script, and spatial density. In each of 1,000 seeded rounds, two disjoint, size-matched SRFUND sets form the Real–Real baseline, and � is the 95th percentile of Real-B-to-Real-A nearest-neighbor distances. We report

$$
\mathrm { T e m p l a t e S u p p o r t } @ \tau = \frac { 1 } { | \mathcal { T } | } \sum _ { t \in \mathcal { T } } \mathbf { 1 } \left[ \operatorname* { m i n } _ { r \in \mathcal { R } } d _ { G } ( t , r ) \leq \tau \right] ,\tag{12}
$$

which measures how much of the benchmark template set lies within typical real-layout support rather than whether both corpora have identical sampling frequencies.

The shared-language comparison is the most conservative evidence because it removes unsupported language categories and prioritizes directly aligned features. Under this setting, 61.7% of templates fall within Real–Real calibrated support, rising to 78.3% when the calibration preserves language composition. For the complementary reference-to-template direction, we define

$$
\mathrm { R e f e r e n c e C o v e r a g e } @ \tau = \frac { 1 } { | \mathcal { R } | } \sum _ { r \in \mathcal { R } } 1 \left[ \operatorname* { m i n } _ { t \in \mathcal { T } } d _ { G } ( r , t ) \le \tau \right] .\tag{13}
$$

Using the global mapped-feature threshold, 876 of 1,529 SRFUND clusters (57.3%) and 806 of 1,137 shared-language clusters (70.9%) have a nearby FormStruct template. FormStruct therefore covers a substantial portion of independently collected real layouts without merely reproducing SRFUND’s composition. The remaining distance above the Real–Real baseline is expected from a benchmark that deliberately emphasizes denser structures: compared with SR-FUND, FormStruct contains more entities and relation links while retaining similar median hierarchy depth, relation density, and normalized entity-box area. We consequently characterize FormStruct-Bench as coverage-oriented rather than prevalence-matched.

Advantages over the external reference. The comparison above evaluates only the intersection of the two annotation schemas. Within that shared space, the real-template grounding and leakagecontrolled calibration establish that FormStruct is not an unconstrained synthetic-layout collection. Beyond it, FormStruct-Bench explicitly annotates semantic regions, region-local grids and spans, widget types and states, line-item groups, and typed structural relations, none of which has a directly equivalent SRFUND annotation. This combination provides both external grounding and finer diagnostic resolution. Consistent with this purpose, Table 6 shows that no evaluated system dominates the five primary metrics and that strong value recovery can coexist with near-zero region or grouping scores; the added structural supervision therefore exposes failure modes hidden by document-level extraction alone. This evidence supports the benchmark’s diagnostic utility, while claims about synthetic-data augmentation or real-world training gains require a separate transfer experiment. The external comparison is also deliberately scoped to template-level shared structure and does not assert that content, missingness, handwriting, or visual-perturbation frequencies of all generated instances match a real-world population.

Table 9: Real–Real calibrated structural support against non-overlapping SRFUND clusters. T–R is the mean template-to-real nearest-neighbor distance; R–R is its size-matched real-to-real baseline. Support is defined in Equation (12). Higher support and a distance ratio closer to one indicate stronger alignment.
<table><tr><td>Scope</td><td>Features / calibration</td><td>Matched  $n _ { T } / n _ { R }$ </td><td>T-R</td><td>R-R median [95% interval]</td><td>Ratio</td><td>Support (%) [95% interval]</td></tr><tr><td>All comparable</td><td>Direct only</td><td>70/70</td><td>0.127</td><td>0.044 [0.040, 0.050]</td><td>2.89</td><td>45.7 [31.4, 65.7]</td></tr><tr><td>Shared languages</td><td>Direct only</td><td>60/60</td><td>0.074</td><td>0.046 [0.040, 0.052]</td><td>1.61</td><td>61.7 [45.0, 88.3]</td></tr><tr><td>Shared languages</td><td>Direct, language-stratified</td><td>60/60</td><td>0.063</td><td>0.042 [0.038, 0.048]</td><td>1.50</td><td>78.3 [60.0, 91.7]</td></tr><tr><td>Shared languages</td><td>Direct + conditional mappings</td><td>60/60</td><td>0.084</td><td>0.053 [0.047, 0.061]</td><td>1.60</td><td>70.0 [43.3, 93.3]</td></tr></table>

## A.8 Evaluation Sensitivity and Error Analysis

For each metric � and predefined constraint tag �, we summarize the descriptive performance gap as

$$
\Delta _ { q } ( c ) = \mathbb { E } [ q \mid \neg c ] - \mathbb { E } [ q \mid c ] .\tag{14}
$$

A positive value indicates lower performance on tagged pages and is interpreted descriptively rather than causally.

![](images/54dffcf96e4f0eda48815e7a3ea633267885d948d90dd55e0b4b1cdf73e0fe91.jpg)  
(c) Across-model SD (%)

![](images/8d8b983b83986af072f86889a20659b6a26ad26531120776acd5f6ab3d75fb6d.jpg)  
(d) Local-grid spread

![](images/571a2f61cb42219b7c23016dbf3842d7af205d89f6f4e7442b984673a467521d.jpg)

![](images/02a7a5351b35a79d1314f5205124d8014b00c445c37884214477967531f11433.jpg)  
Figure 10: Evaluation sensitivity of API-hosted and local models.

Relative to relaxed scoring, full criteria reduce representative API and local scores by 58.1–63.4% and 42.0–51.8%, respectively (Figure 10). Relation penalties are stable across groups, whereas local-grid efects vary by model.

Table 10 identifies dense relations and local grids as the largest mean losses (11.5/10.8 points), followed by widgets (6.3). Mixedlayout and line-item efects are inconsistent, localizing errors to field binding and local organization.

Table 10: TSR-path loss (%) by constraint for four local VLMs.
<table><tr><td>Constraint</td><td>Min</td><td>Mean</td><td>Max</td></tr><tr><td>Dense key-field relations</td><td>7.2</td><td>11.5</td><td>15.7</td></tr><tr><td>Region-local grids</td><td>3.5</td><td>10.8</td><td>17.6</td></tr><tr><td>Widget grouping</td><td>4.5</td><td>6.3</td><td>8.3</td></tr><tr><td>Mixed layout</td><td>-1.2</td><td>3.3</td><td>13.0</td></tr><tr><td>Line-item groups</td><td>-5.0</td><td>-0.9</td><td>1.8</td></tr></table>

Table 11: Agreement between initial template annotations and third-annotator verification.
<table><tr><td>Target</td><td>Agreement statistic</td><td>Result</td></tr><tr><td>Region boundaries</td><td>Mean IoU; object  $F _ { 1 }$  at IoU  $\geq 0 . 5$ </td><td>0.971,0.903</td></tr><tr><td>Region types</td><td>Cohen&#x27;s κ on matched regions</td><td>1.000</td></tr><tr><td>Local-grid topology</td><td>Cell/span topology  $F _ { 1 }$ </td><td>1.000</td></tr><tr><td>Widget types and states Matched-object</td><td> $F _ { 1 }$ </td><td>1.000</td></tr><tr><td>Structural relations</td><td>Typed-relation  $F _ { 1 }$ </td><td>0.993</td></tr></table>

## A.9 Annotation Agreement and Review Outcomes

Annotation and verification protocol. The 70 templates were divided evenly between two annotators, each of whom annotated 35 templates (50%). A third annotator reviewed all completed annotations for consistency and correctness. Because the annotation contains both categorical and structured objects, we do not collapse agreement into a single statistic. Categorical decisions are measured with Cohen’s $\kappa ,$ whereas spatial, topological, and relational objects are compared after matching their identities and boundaries. Table 11 reports the category-specific statistics obtained during verification. Spatial objects were aligned using maximum-IoU one-to-one bipartite matching, with a pair considered valid when its bounding-box IoU was at least 0.5. Unmatched objects were treated as missing or additional annotations, while region types were compared only after geometric matching.

Accepted, corrected, and discarded samples. All 1,100 retained test instances receive human review after automated verification. Reviewers assign one primary outcome: accepted when the image and annotations require no change, corrected when a localized error can be repaired and the sample can pass re-verification, or discarded when the visible evidence remains ambiguous or the error cannot be repaired reliably. Table 12 reports one terminal human-review outcome for each of the 1,100 retained test instances. Each instance is counted once; correction and re-verification do not create additional review entries.

Table 12: Human-review outcomes for test-set construction.
<table><tr><td>Primary outcome</td><td>Count</td><td>Rate</td></tr><tr><td>Accepted</td><td>600</td><td>54.5%</td></tr><tr><td>Corrected and re-verified</td><td>500</td><td>45.5%</td></tr><tr><td>Discarded</td><td>0</td><td>0.0%</td></tr><tr><td>Total reviewed candidates</td><td>1,100</td><td>100%</td></tr></table>

Table 13: Retry distribution and terminal outcomes of automated verification.
<table><tr><td>Terminal path</td><td>Count</td><td>Rate</td></tr><tr><td>Passed without retry</td><td>6,801</td><td>97.16%</td></tr><tr><td>Passed after one retry</td><td>199</td><td>2.84%</td></tr><tr><td>Passed after two retries</td><td>0</td><td>0.00%</td></tr><tr><td>Passed after three or more retries</td><td>0</td><td>0.00%</td></tr><tr><td>Retry budget exhausted; discarded</td><td>0</td><td>0.00%</td></tr><tr><td>Total generation candidates</td><td>7,000</td><td>100.00%</td></tr></table>

A second reviewer independently assessed all 1,100 test instances (100%) for agreement measurement.

## A.10 Generation Retries and Filtering

A failed automated check produces a structured failure report and returns the candidate to the Director–Artist pipeline. Generation stops when the candidate passes all checks or reaches the maximum retry budget of three retries, in which case it is discarded. To make this filtering stage auditable, Table 13 reports the first-pass yield and terminal retry distribution over the 7,000 logical generation candidates initiated for dataset construction. Repeated attempts for the same target instance remain part of one candidate, and a retry is counted whenever a failed automated verification triggers a new Director–Artist regeneration followed by another complete verification cycle. Human corrections are tracked separately from automated generation retries.

## A.11 Source Licensing and Privacy

Sources and permissions. The reusable templates originate from public web documents and the XFUND and Common Crawl source pools described in Section 2. For each retained template, the release manifest will record the source collection, source URL or upstream identifier, retrieval date, applicable license or terms of use, permitted redistribution scope, and the transformation applied by our pipeline. Table 14 separates upstream permissions from the license assigned to newly generated benchmark artifacts; inclusion in Common Crawl is not itself treated as permission to redistribute the underlying page.

Privacy controls. Original filled values are removed before a source document is converted into a reusable template, and released instances are rendered from newly sampled field values rather than copying those removed values. Before release, we applied and documented a manual PII-screening procedure for residual sensitive content, including names, contact details, identifiers, account numbers, signatures, faces, hidden metadata, and identifying visual elements such as titles, logos, seals, and annotations. Templates containing sensitive structural metadata, such as field keys or semantic labels, were excluded from release, while sensitive filled values and removable identifying visual elements were deleted or redacted. Provenance information is released only through de-identified or access-controlled source identifiers to preserve auditability without re-exposing personal information. We also document the access, retention, and deletion controls applied during annotation.

Table 14: Source and release licensing disclosure.
<table><tr><td>Source pool</td><td>Material retained</td><td>License or permission</td></tr><tr><td>Public web documents</td><td>22 de-identified blank templates with field labels and manual layout annotations</td><td>Mixed source terms: copyright/permission- controlled. Source URLs and terms are recorded</td></tr><tr><td>XFUND [41]</td><td>42 template images with derivative annotations</td><td>CC BY-NC-SA 4.0 [11]; derivative materials follow non-commercial/share-</td></tr><tr><td>Common Crawl [10]</td><td>6 de-identified blank templates with annotations and provenance</td><td>alike constraints. Used under Common Crawl terms [9]; retained materials are de-identified derived templates with</td></tr><tr><td>Total</td><td>70 templates</td><td>Provenance records include source URLs or dataset versions, retrieval metadata,</td></tr></table>

## A.12 Shared VLM Prediction Prompt

All evaluated VLMs use the following prediction prompt. For readability, the shaded bars segment the prompt; their numbering is typographic and is not part of the model input. The wording and ordering remain unchanged

## 1. Task and output contract

You are extracting the semantic answer tree and the minimal visible form structure needed for FormStruct-Bench evaluation.

Return only strict JSON. Do not include markdown fences, prose, comments, or explanations. Do not include thinking tags such as <think> or </think>. Do not output hidden reasoning.

## 2. JSON validity requirements:

• The response must be parseable by a standard JSON parser.

• Every object member must be a "key": value pair. Do not put a bare string inside an object.

• Escape quotation marks and line breaks inside strings.

• Close every object and array that you open.

## 3. Required top-level schema:

```jsonl
{
"answer": {
"visible form title or top-level section": {
"visible field label": "visible filled value",
"nested visible section": {
"visible field label": "visible filled value"
}
}
},
"regions": [
{"id": "r1", "type": "title|section|field|value|text|widget|table|other", "bbox": [x1, y1, x2, y2], "text": "visible label or text"}
],
"widgets": [
{"id": "w1", "type": "checkbox|radio|input|signature|other", "bbox": [x1, y1, x2, y2], "label": "visible option or field label",
↩→ "selected": true}
],
"local_grids": [
{"id": "g1", "region_id": "r_table", "cells": [{"id": "c1", "row": 0, "col": 0, "rowspan": 1, "colspan": 1, "bbox": [x1, y1, x2, y2],
↩→ "text": "visible cell text"}]}
],
"cells": [
{"id": "c1", "row": 0, "col": 0, "rowspan": 1, "colspan": 1, "bbox": [x1, y1, x2, y2], "text": "visible cell text"}
],
"relations": []
}
```

## 4. Answer rules:

• Preserve visible labels as keys as closely as possible, including the original language/script.

• Preserve nested section/field hierarchy.

• Use strings for filled values. Use objects for nested sections and arrays for repeated line-item rows.

• For selected checkboxes/radio buttons, output the selected option label or value in the corresponding answer field.

• If a visible field is blank, use an empty string.

• If a section contains a paragraph/comment without a clear field label, represent it as {"value": "the visible text"} rather than as a bare string.

## 5. Structure rules:

• Use pixel coordinates relative to the input image: [left, top, right, bottom].

• Include visible titles, section headers, field labels, filled value boxes/text areas, and checkbox/radio/input widgets as regions.

• Each region must have id, type, bbox, and text.

• Include widgets separately in "widgets" with selected=true/false when visually determinable.

• Use stable ids.

• Add local\_grids/cells only for table-like repeated rows when row/column positions are visually clear. Each cell must have row, col, rowspan, colspan, bbox, and text.

• Set "relations": [] unless a few relation edges are obvious and short.

• Do not invent unreadable text or uncertain boxes. Omit items that are not visually supported.

Output a single JSON object with all top-level keys shown above. Empty arrays are allowed when no such structure is visible

## A.13 Director Prompt for Multi-agent Form Generation

The Director agent converts structured field metadata into executable form-filling actions for the multi-agent generation pipeline. Given a field\_key, semantic\_key, and data type for each field, it generates semantically plausible content while preserving the original field\_key for downstream alignment with the template. The resulting JSON action array is passed to the Artist agent for rendering. For readability, the shaded bars segment the prompt; their numbering is typographic and is not part of the model input. The wording and ordering remain unchanged.

## 1. Task and input contract

You are generating form-filling ACTIONS.

You will receive a list of semantic fields that need content for text boxes. Each input item includes field\_key, data\_type, and semantic\_key.

## 2. Field-key preservation rules

• field\_key is the exact output key chosen by the planner.

• Output actions must use the same field\_key string.

• Do not rename, translate, normalize, or deduplicate field\_key.

• If the printed label is missing or unusable, the planner may fall back to semantic\_key, but the emitted action must still preserve the provided field\_key.

## 3. Output format and action schema

• Output only a JSON array of actions, with no extra text or markdown.

• For each input item, output exactly one action in the same order.

• Do not invent extra actions or checkbox actions.

• The action\_type must match data\_type: number fields use number, date fields use date, and text fields use write\_text.

## 4. Semantic value generation rules

• Use both field\_key and semantic\_key when generating content.

• Preserve field\_key as the output key, while using semantic\_key to infer the intended field meaning.

• If field\_key is ambiguous, generic, abbreviated, or not human-friendly, rely on semantic\_key to choose the content.

• Generate realistic values related to the intended field meaning.

## 5. Localized date and partial-date rules

Date values should follow the language and form style implied by field\_key and semantic\_key. Full-date fields use localized formats, while explicit date-component fields such as year, month, or day must contain only the corresponding component. Split-date fields distribute one logical date across sibling fields instead of repeating the full date.

## 6. Anti-copying and self-check rules

• Do not copy field\_key as the generated content.

• Do not copy semantic\_key as content unless it is genuinely the intended filled value.

• Generic option labels must be expanded into more specific plausible values.

• Before returning the final JSON array, silently check that each action has the correct type, plausible localized content, valid partial-date formatting, and no copied label content.

## 7. Output action schema

```json
[
{
"field_key": "EXACT_FIELD_KEY_FROM_INPUT",
"action_type": "write_text|number|date",
"content": "GENERATED_FIELD_VALUE",
"bbox": [x1, x2, y1, y2],
"semantic_key": "EXACT_SEMANTIC_KEY_FROM_INPUT"
}
]
```

## A.14 Artist Prompt for Multi-agent Form Generation

The Artist agent converts the Director actions into a realistic filled form image. It first uses a deterministic CV renderer to place the generated actions onto the template according to the layout metadata. The resulting draft image is then passed to an image-editing mode for constrained visual naturalization. The prompt below is used in the second stage

## 1. Task and input contract

You are editing an already-filled form draft image.

The input image is a deterministic draft produced from structured form-filling actions. It already contains all required filled values, repeated field occurrences, checkbox selections, radio selections, circled options, signatures, dates, numbers, and free-text entries in their intended locations.

You will also receive a reference answer JSON containing the exact content that must be preserved.

## 2. Artist role and scope

• Your task is to make the filled form look realistic and naturally completed.

• Do not generate new semantic content.

• Do not decide which fields should be filled.

• Do not change the meaning, value, order, or location of any filled content.

• Use the draft image as the positional ground truth.

• Use the reference answer JSON as the textual ground truth.

## 3. Filled-content preservation rules

• Preserve every filled value exactly as shown in the draft and as specified in the reference answer JSON.

• Do not add, remove, rewrite, paraphrase, translate, substitute, merge, or reorder any filled value.

• Preserve repeated field occurrences separately and keep their original top-to-bottom order.

• Preserve all dates, numbers, emails, addresses, names, signatures, free-text responses, checkbox selections, radio selections, and circled options.

• If there is any ambiguity, preserve the draft content and placement exactly.

## 4. Template preservation rules

• Treat all fixed printed template content as locked background.

• Do not redraw, rewrite, translate, paraphrase, stylize, or alter printed labels, instructions, table lines, borders, logos, seals, stamps, URLs, organization names, footer text, or boilerplate.

• Do not redesign the form layout.

• Keep page geometry, spacing, table structure, alignment, and scan-like document appearance unchanged.

• Keep filled content within the same semantic region of the page.

## 5. Visual naturalization rules

• Only naturalize user-entered content.

• Make the filled entries look like realistic form entries, such as handwritten text, typed text, mild pen-pressure variation, slight ink variation, natura baseline variation, and subtle scan noise.

• Keep all filled content crisp and readable, especially in dense cells, narrow fields, tables, date boxes, and ID or phone-number fields.

• Improve realism without changing the actual filled information.

• Do not sacrifice legibility for style.

## 6. Option-mark preservation rules

• Preserve checkbox, radio, and circled-option states exactly as shown in the draft.

• Do not add new option marks.

• Do not remove existing option marks

• Do not change selected options into unselected options.

• Do not change unselected options into selected options.

• Do not move, thicken, thin, blur, redraw, simplify, or reinterpret option marks.

## 7. Critical self-check rules

• The final image must still exactly match the reference answer JSON.

• Before producing the final edited image, silently check that every filled value, repeated occurrence, and option state is preserved.

• When visual realism conflicts with textual or positional fidelity, fidelity wins.

## A.15 Verifier Prompt for Multi-agent Form Generation

The Verifier agent closes the generation loop by independently re-extracting the filled content from each stylized form image. Given a target generated image and a few same-class examples with standard answer.json outputs, the Verifier produces an answer\_verifier.json using the same schema as the reference answer. This extracted JSON is compared field-by-field against the original answer.json. If any required field is missing, mismatched, or incorrectly selected, the sample is marked as failed and returned to the Artist stage for regeneration

## 1. Task and input contract

You are an expert form-layout analysis and key-value extraction assistant.

You will receive full form images plus a few same-class examples with their standard answer.json outputs. Each example contains one same-class form image and its standard structured answer. You will then receive one target generated image to extract.

## 2. Validation role

• Your output will be used for automatic field-level validation.

• The extracted JSON will be compared against the original reference answer.json.

• Missing fields, mismatched values, incorrect option states, or invalid structure may cause the target image to be rejected and regenerated by the Artist stage.

• Your role is not to repair the image or explain errors.

• Your role is to faithfully extract what is visible in the target image.

## 3. Few-shot schema learning rules

• Use the example images and standard answer.json files to learn the form class’s JSON hierarchy.

• Preserve the same field names, nesting levels, array structure, list order, option representation, and value style whenever the target form uses the same fields.

• The examples define the output schema; the target image defines the actual field values.

• Do not copy concrete values from the examples into the target output unless the same value is visibly present in the target image.

## 4. Target extraction rules

• Extract a complete structured key-value JSON object from the target image.

• Traverse the entire image from the header or top row to the bottom.

• Do not skip any visible region.

• Treat label-like or title-like text as keys.

• Match each key to its value using nearby layout, alignment, and semantic relationship.

• Capture all logical keys through the end of the image.

## 5. Value fidelity rules

• Output every nested key-value relationship.

• Values may be strings, nested JSON objects, or arrays.

• Dates must be complete.

• Preserve date ranges when present.

• Preserve visible details such as numbers, emails, addresses, dates, punctuation, capitalization, and spacing as much as possible

• Re-check high-risk values such as phone numbers, IDs, emails, dates, amounts, percentages, and handwritten free-text fields digit by digit or character by character.

• If a field is present but blank or unreadable, output an empty string instead of deleting the field.

## 6. Option-state extraction rules

• For checkboxes, radio buttons, dropdowns, and option groups, output the option text actually selected in the target image.

• Do not infer selected options from row order, field defaults, or the few-shot examples.

• Do not output unselected options as selected values.

• Do not change a selected option into a nearby unselected option.

• Do not merge multiple options unless the target image visibly selects multiple options.

## 7. Output format rules

• Return exactly one JSON object.

• The response must start with { and end with }.

• The JSON must be valid and fully closed.

• Do not include explanations, Markdown, comments, confidence scores, code fences, error labels, or extra wrapper keys.

• Return the final answer\_verifier.json for the target image only.

## A.16 Hierarchical Annotation Example and Visual Encoding

Figure 11 illustrates how a table-form page is decomposed into nested semantic regions and fine-grained structural components. Colored boxes distinguish sections, table regions, keys, values, cells, and line-item groups, while orange connectors visualize group-level associations among repeated entries. The accompanying legend defines the annotation colors used in the overlay.

![](images/78abf8e10b3569adace1ac5d93df48d400bc77570c2399bd3f54dbaa2293bcb8.jpg)  
Figure 11: Example annotation diagram.