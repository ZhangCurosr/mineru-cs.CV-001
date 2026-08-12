# Share First, Route What Remains:

# A Unified Framework for Token-Adaptive MoE Computation

Gongli Zhang , Zhulin Liu <sup>\*</sup>, and C. L. Philip Chen

Guangdong Provincial Key Laboratory of Computational AI Models and Cognitive Intelligence,

School of Computer Science and Engineering, South China University of Technology, Guangzhou 510006, China Pazhou Lab, Guangzhou 510335, China

Engineering Research Center of the Ministry of Education on Health Intelligent Perception and Paralleled Digital-Human, Guangzhou 510641, China

Corresponding author: Zhulin Liu.

Emails: csglzhang@mail.scut.edu.cn, liuzhl@scut.edu.cn, philip.chen@ieee.org

Abstract—Mixture-of-experts (MoE) models have recently moved beyond routing a fixed number of complete experts. Shared-expert designs preserve reusable knowledge, fine-grained methods vary computation within experts, and dynamic routers adapt the number of active experts. Yet these decisions are usually made independently, overlooking a basic dependency: extracting reusable computation changes both what remains and how much expert capacity the remainder needs. We study this dependency by decomposing sparsely upcycled feed-forward experts into keyvalue channels. Co-activated experts align at a subset of value positions; removing these positions changes expert preference; and greater shared coverage is associated with lower residual expert demand. These observations lead to one principle: share first, then route what remains. We instantiate it in UniF-MoE, a unified framework for token-adaptive MoE computation. Each expert is partitioned into aligned blocks. A shared-demand score sets the shared block count and pathway weight, key prototypes select the shared content, and the complementary demand determines the residual expert count through cumulative routing mass. A Gram regularizer separates and normalizes router embeddings, promoting diverse routing directions, sparse expert overlap, and a simple routing geometry. Experiments on DomainBed and GLUE show that this unified design improves predictive performance over representative static and dynamic MoEs while reducing activated computation, inference latency, and memory. Code is available at https://github.com/existence0420/UniF-MoE.

Index Terms—mixture of experts, unified framework, tokenadaptive computation, shared, residual

## I. INTRODUCTION

Mixture-of-experts (MoE) models increase parameter capacity without evaluating every parameter for every token [1]. A router sends each token to a small set of feed-forward networks (FFNs), or experts, so capacity can grow while execution remains sparse [2], [3]. Conventional top-k routing answers one question: which experts should process this token?

This direct formulation hides two allocation decisions. It treats a complete expert as the unit of computation, even when selected experts repeat common responses, and it gives every token the same expert count, even when their semantic requirements differ. The result can be redundant computation for simple tokens and insufficient capacity for harder ones.

Recent work relaxes this formulation along three complementary directions. Shared experts capture common knowledge [4], [5]; modular, nested or slimmable experts vary computation within an expert [6]–[8]; and dynamic routers vary the number of active experts [9]–[12]. Other methods encourage experts to remain distinct [13], [14]. These advances are usually developed as separate mechanisms, but their decisions are not independent. Once reusable computation is extracted, the residual content changes, the best experts can change, and the capacity needed from them can change as well. A router that decides these quantities separately can therefore repeat shared work or allocate expert capacity for a response that has already been partly computed.

To expose this dependency, we delve into the internal structure of the FFN. An expert can be written as a sum of hidden channels, each defined by an up-projection key and a down-projection value [15]. In a sparsely upcycled MoE [16], experts start from the same pretrained FFN, so matching channel positions remain comparable across experts. This lets us test whether co-activated experts produce reusable responses at aligned positions and what happens to routing after those responses are separated.

Our analysis gives a consistent answer. Co-activated experts are neither wholly redundant nor wholly distinct: they align at a subset of value positions, and frequently co-activated pairs tend to align more. Removing these positions substantially changes the preferred experts. The number of residual experts needed to recover the original output also varies across tokens and decreases as shared coverage grows. Shared modeling, fine-grained computation, and dynamic routing are therefore three stages of one allocation problem, not three parallel knobs. The reusable response should be identified first; routing should then act on what remains.

This principle leads to UniF-MoE, a unified framework for token-adaptive MoE computation. Each layer contains one shared expert and K residual experts, all divided into aligned blocks. For each token, a shared-demand score first determines how much computation belongs to the shared pathway and how strongly that pathway contributes. Key prototypes then select which blocks provide that reusable content. Only the complementary blocks are routed onward, and their demand determines how many residual experts are activated by cumulative routing mass. A single router thus coordinates three decisions in functional order: how much to share, what to share, and how many experts the remainder needs. A Gram constraint shapes its embeddings into distinct, normalized directions, encouraging diverse expert roles and sparse overlap through a simple routing geometry.

Our contributions are summarized as follows:

• We reveal a routing-conditioned dependency between reusable and token-specific computation inside sparsely upcycled experts. This turns shared modeling, finegrained computation, and dynamic routing into one ordered principle: share first, then route what remains.

• We instantiate this principle in UniF-MoE, which couples shared width, shared content, and residual expert count through one token-dependent budget. The resulting process coordinates intra-expert and inter-expert sparsity instead of deciding them independently.

• Across vision and language benchmarks, UniF-MoE achieves a stronger accuracy–efficiency trade-off than representative MoEs. Ablations and routing analyses further show that its gains arise from coordinating shared reuse with residual specialization.

## II. MOTIVATION

## A. Preliminaries

Let $\mathbf { x } \in \mathbb { R } ^ { 1 \times d }$ be one input token to a sparse MoE layer with K experts. The router stores one embedding per expert:

$$
\mathbf { W } _ { g } = [ \mathbf { W } _ { 1 } , \mathbf { W } _ { 2 } , \ldots , \mathbf { W } _ { K } ] \in \mathbb { R } ^ { d \times K } ,\tag{1}
$$

where $\mathbf { W } _ { j } ~ \in ~ \mathbb { R } ^ { d \times 1 }$ is the embedding of expert j. The router produces an affinity distribution and selects its k largest entries:

$$
\mathbf { s } ( \mathbf { x } ) = \mathrm { s o f t m a x } ( \mathbf { x W } _ { g } ) \in \mathbb { R } ^ { 1 \times K } ,\tag{2}
$$

$$
\begin{array} { r } { \mathcal { T } _ { k } ( \mathbf { x } ) = \mathrm { T o p K } ( \mathbf { s } ( \mathbf { x } ) , k ) , } \end{array}\tag{3}
$$

where TopK returns the index set of the selected entries.

Expert j is an FFN with up-projection ${ \bf K } _ { j } \in \mathbb { R } ^ { d \times H }$ and down-projection $\mathbf { V } _ { j } \in \mathbb { R } ^ { H \times d }$ . The intermediate width H is also the number of hidden channels. Its output can be written as

$$
E _ { j } ( \mathbf { x } ) = \sum _ { h = 1 } ^ { H } \mathrm { G e L U } ( \mathbf { x } \mathbf { K } _ { j } [ : , h ] ) \mathbf { V } _ { j } [ h , : ] .\tag{4}
$$

The column $\mathbf { K } _ { j } [ : , h ]$ is the key of channel h: its inner product with x determines the channel activation. The row $\mathbf { V } _ { j } [ h , : ]$ is the value: it determines the direction written to the output. We omit bias terms for clarity. A conventional top-k MoE returns

$$
\mathbf { y } = \sum _ { j \in \mathcal { T } _ { k } ( \mathbf { x } ) } s _ { j } ( \mathbf { x } ) E _ { j } ( \mathbf { x } ) .\tag{5}
$$

Here, $s _ { j } ( \mathbf { x } )$ denotes entry j of s(x). Equation (5) routes complete experts. Equation (4), however, shows that each expert is itself a sum of channel responses. This internal structure provides a natural place to separate shared and residual computation.

![](images/df96f72dce0bea8bf822f8cb9a2c8bf387b9ffd12ec3fb88db28b3ad303d1831.jpg)

(a) Shared ratio and co-activation frequency  
![](images/6637ed878d3354faebacd05d4215854f3fdd049728af6e7562474f86efd12eea.jpg)

![](images/0ef65a7a40e1e7d46f4c9d66dc26f85ad53b5fe61bbf99bb2a774cbfe4084275.jpg)  
(b) Router rank transition  
(c) Functional-demand CDF  
Fig. 1. Diagnosing the MoE module in Transformer layer 10 trained on TerraIncognita. (a) Value alignment and co-activation for each expert pair. (b) Expert-rank transitions after the aligned positions are extracted. (c) Pairbalanced cumulative distribution function (CDF) $F _ { \varepsilon } ( m ) = \mathrm { P r } [ k _ { \varepsilon } ( \mathbf { x } ) \leq m ]$ at $\varepsilon \ : = \ : 0 . 0 5$ for the five pairs with the lowest and highest shared ratios. Shading shows the standard error across pairs.

## B. Observations

a) Diagnostic setup.: We analyze layer 10 of a standard top-2 MoE model trained on TerraIncognita [17]. The layer contains 6 experts with 1536 hidden channels each. All experts are copied from the same pretrained Transformer [18] FFN before fine-tuning. Channel index h therefore refers to the same initial position across experts, which makes same-index values directly comparable. For expert pair (i, j), we define

$$
\begin{array} { r l } & { \quad c _ { i j h } = \cfrac { \mathbf { V } _ { i } [ h , : ] \mathbf { V } _ { j } [ h , : ] ^ { \top } } { \| \mathbf { V } _ { i } [ h , : ] \| _ { 2 } \| \| \mathbf { V } _ { j } [ h , : ] \| _ { 2 } } , } \\ & { \quad S _ { i j } ( \delta ) = \{ h \in \{ 1 , . . . , H \} : c _ { i j h } \geq \delta \} , } \\ & { \quad \rho _ { i j } ( \delta ) = \cfrac { | S _ { i j } ( \delta ) | } { H } . } \end{array}\tag{6}
$$

We use $\delta = 0 . 9 8$ . The set $\boldsymbol { S } _ { i j }$ contains positions where the two experts write in nearly the same direction, and $\rho _ { i j }$ is the fraction of such positions. We treat this set as a weight-space estimate of the pair’s reusable response. Its complement $\mathcal { R } _ { i j } =$ $\{ 1 , \ldots , H \} \setminus S _ { i j }$ contains the residual positions.

b) Observation 1: co-activation identifies reusable responses.: Figure 1(a) compares $\rho _ { i j }$ with the fraction of tokens routed to pair (i, j). Their Pearson correlation is 0.697. Frequently co-activated pairs align at up to roughly 80% of their value positions, while some rarely selected pairs align at less than 2%. The aligned positions are therefore not spread uniformly across expert pairs. They concentrate among the pairs that the router most often asks to work together, which makes them a useful target for shared computation.

c) Observation 2: residual computation needs a new routing decision.: Similarity alone does not show that the aligned positions should be separated from routing. We therefore freeze the experts, extract $\boldsymbol { S } _ { i j }$ for each token’s original top-2 pair, and train a new top-2 router on the residual positions. Figure 1(b) compares the two expert rankings. Only 5.7% of tokens keep the same top-2 set, and the mean diagonal mass is 26.9%. The original router scores the complete response, not the residual alone. Once the shared response is handled, different experts often become better choices.

d) Observation 3: residual demand varies across tokens.: We next ask how many residual experts are needed after the shared response has been separated. For the active pair $( i , j )$ let $\widetilde { s } _ { e } ( \mathbf { x } ) = s _ { e } ( \mathbf { x } ) / ( s _ { i } ( \mathbf { x } ) + s _ { j } ( \mathbf { x } ) )$ be the normalized original gate weight. We split each expert output into $E _ { e } ^ { S _ { i j } }$ and $\bar { E } _ { e } ^ { \mathcal { R } _ { i j } }$ which retain the shared and residual positions, respectively. Merging the shared responses once gives

$$
{ \bf y } _ { i j } ^ { \mathrm { o r i g } } ( { \bf x } ) = \sum _ { e \in \{ i , j \} } \widetilde { s } _ { e } ( { \bf x } ) E _ { e } ( { \bf x } ) ,\tag{7}
$$

$$
\overline { { \mathbf { S } } } _ { i j } ( \mathbf { x } ) = \sum _ { e \in \{ i , j \} } \widetilde { s } _ { e } ( \mathbf { x } ) E _ { e } ^ { S _ { i j } } ( \mathbf { x } ) ,\tag{8}
$$

$$
\mathbf { r } _ { i j } ( \mathbf { x } ) = \mathbf { y } _ { i j } ^ { \mathrm { o r i g } } ( \mathbf { x } ) - \overline { { \mathbf { S } } } _ { i j } ( \mathbf { x } ) .\tag{9}
$$

The residual target is ${ \bf r } _ { i j } ( { \bf x } )$ . Let $q _ { 1 } ( \mathbf { x } ) , \ldots , q _ { K } ( \mathbf { x } )$ be the expert order produced by the residual router. For a prefix of length $m \in \{ 1 , \ldots , K \}$ , we stack its residual outputs:

$$
\mathbf { R } _ { m } ( \mathbf { x } ) = \left[ \begin{array} { c } { E _ { q _ { 1 } ( \mathbf { x } ) } ^ { \mathcal { R } _ { i j } } ( \mathbf { x } ) } \\ { \vdots } \\ { E _ { q _ { m } ( \mathbf { x } ) } ^ { \mathcal { R } _ { i j } } ( \mathbf { x } ) } \end{array} \right] \in \mathbb { R } ^ { m \times d } .\tag{10}
$$

We then use ridge-stabilized least squares to measure the best reconstruction available in the row span of $\mathbf { R } _ { m } ( \mathbf { x } )$

$$
\mathbf { a } _ { m } ^ { \star } ( \mathbf { x } ) = \arg \operatorname* { m i n } _ { \mathbf { a } \in \mathbb { R } ^ { m } } \left\| \mathbf { a } ^ { \top } \mathbf { R } _ { m } ( \mathbf { x } ) - \mathbf { r } _ { i j } ( \mathbf { x } ) \right\| _ { 2 } ^ { 2 }
$$

$$
+ \gamma _ { m } ( \mathbf { x } ) \Vert \mathbf { a } \Vert _ { 2 } ^ { 2 } ,\tag{11}
$$

$$
\widehat { \mathbf { y } } _ { m } ( \mathbf { x } ) = \overline { { \mathbf { S } } } _ { i j } ( \mathbf { x } ) + ( \mathbf { a } _ { m } ^ { \star } ( \mathbf { x } ) ) ^ { \top } \mathbf { R } _ { m } ( \mathbf { x } ) ,\tag{12}
$$

$$
e _ { m } ( \mathbf { x } ) = \frac { \| \widehat { \mathbf { y } } _ { m } ( \mathbf { x } ) - \mathbf { y } _ { i j } ^ { \mathrm { o r i g } } ( \mathbf { x } ) \| _ { 2 } } { \operatorname* { m a x } \{ \| \mathbf { y } _ { i j } ^ { \mathrm { o r i g } } ( \mathbf { x } ) \| _ { 2 } , \eta \} } ,\tag{13}
$$

$$
\begin{array} { r } { \mathcal { A } _ { \varepsilon } ( \mathbf { x } ) = \left\{ m \in \left\{ 1 , \dots , K \right\} : e _ { m } ( \mathbf { x } ) \leq \varepsilon \right\} , } \end{array}\tag{14}
$$

$$
k _ { \varepsilon } ( \mathbf x ) = \operatorname* { m i n } \left( \mathcal { A } _ { \varepsilon } ( \mathbf x ) \cup \{ K + 1 \} \right) .\tag{15}
$$

We set $\gamma _ { m } ( { \bf x } ) ~ = ~ 1 0 ^ { - 6 } \mathrm { t r } ( { \bf R } _ { m } ( { \bf x } ) { \bf R } _ { m } ( { \bf x } ) ^ { \top } ) / m$ and $\eta \quad =$ $1 0 ^ { - 1 2 } . ~ \varepsilon { \mathrm { ~ i s ~ } }$ the fidelity tolerance, and $\mathcal { A } _ { \varepsilon } ( \mathbf { x } )$ is the set of prefixes that attain it. The sentinel K + 1 records that no available prefix reaches the target. Thus, $k _ { \varepsilon } ( \mathbf { x } )$ measures how quickly the ordered residual span recovers the original output. Figure 1(c) plots its CDF with ε set to 0.05. At $m = 2 .$ , the pair-balanced success rates are 56.3% for high-shared pairs and 11.8% for low-shared pairs; their mean $k _ { \varepsilon }$ values are 2.70 and 4.61. Across pairs, shared ratio and mean demand correlate at −0.673 (bootstrap 95% CI [−0.899, −0.315]), and the trend persists under uniform shared averaging and across all four held-out environments $\mathrm { a t ~ } \varepsilon \in \ \{ 0 . 1 0 , 0 . 2 0 \}$ . Shared coverage is therefore related to the full distribution of residual demand.

## C. Design Implications

These observations show that the key question is not whether shared modeling, fine-grained computation, and dynamic routing coexist, but in what order they act. Shared modeling must first determine how much computation is reusable. Fine-grained computation must then identify which content provides that response. Only after these choices should dynamic routing decide how many experts the remainder needs. Reversing or separating this order would route a response that has already changed. The next section implements this principle as share first, then route what remains.

## III. METHOD

Figure 2 compares UniF-MoE with conventional top-k MoE. Rather than attaching independent controllers for shared computation, internal selection, and expert count, UniF-MoE makes these decisions sequentially under one token-dependent budget.

## A. Blockwise Shared-Residual Partitioning

A UniF-MoE layer contains one shared expert $E _ { \mathrm { s h r } }$ and K residual experts $\{ E _ { 1 } , \ldots , E _ { K } \}$ . Each expert has intermediate width H and is divided into B aligned blocks of width $M = H / B$ . Block b contains positions $\mathcal { H } _ { b } = \{ ( b - 1 ) M +$ $1 , \ldots , b M \}$ and produces

$$
E _ { j } ^ { ( b ) } ( \mathbf { x } ) = \sum _ { h = ( b - 1 ) M + 1 } ^ { b M } \mathrm { G e L U } ( \mathbf { x K } _ { j } [ : , h ] ) \mathbf { V } _ { j } [ h , : ] .\tag{16}
$$

Thus, $\begin{array} { r } { E _ { j } ( { \bf x } ) ~ = ~ \sum _ { b = 1 } ^ { B } E _ { j } ^ { ( b ) } ( { \bf x } ) } \end{array}$ . All experts are initialized from the same dense FFN, so their block boundaries are aligned at the start of training. For each token, the shared expert executes one subset of blocks, and the residual experts execute the complementary subset. Here, “residual” refers to the channel positions left after shared-block selection, not to the Transformer residual connection. No hidden position is assigned to both pathways for the same token.

## B. Token-Adaptive Shared Modeling

a) Shared demand.: The standard router $\mathbf { W } _ { g }$ contains the embeddings of K residual experts. We add one learnable column $\mathbf { W } _ { \mathrm { s h r } }$ for the shared expert:

$$
\mathbf { W } _ { g } ^ { \star } = [ \mathbf { W } _ { \mathrm { s h r } } , \mathbf { W } _ { g } ] \in \mathbb { R } ^ { d \times ( K + 1 ) } .\tag{17}
$$

This column produces the shared-demand score

$$
\alpha ( { \bf x } ) = \tau + ( 1 - 2 \tau ) \sigma ( { \bf x W } _ { \mathrm { s h r } } ) , \qquad \tau = \frac { B - 1 } { B ^ { 2 } } ,\tag{18}
$$

where $\sigma ( \cdot )$ is the sigmoid function. We interpret $\alpha ( \mathbf { x } )$ as the token’s demand for shared computation. It controls both the

![](images/5cf06c4bf8eb0cff12b1ce998f1df75c9c6083131cee5120fa702fd3e1bec0f5.jpg)  
Fig. 2. Conventional top-k MoE selects complete experts in one step. UniF-MoE first executes token-selected blocks once in a shared expert, then routes only the complementary blocks to a token-dependent number of residual experts. Colored blocks are active.

number of shared blocks and the mixture weight of the shared pathway. The shared block count is

$$
b ( \mathbf { x } ) = \operatorname { r o u n d } ( B \alpha ( \mathbf { x } ) ) .\tag{19}
$$

Because $\sigma ( z ) \in ( 0 , 1 )$ , the chosen bound gives $\tau < \alpha ( { \bf x } ) <$ $1 - \tau$ . For $B \ \geq \ 2$ , rounding therefore guarantees $b ( \mathbf { x } ) \in$ $\{ 1 , \ldots , B - 1 \} ;$ : every token uses at least one shared block and leaves at least one block for residual computation.

b) Shared block selection.: The score $\alpha ( \mathbf { x } )$ determines how many blocks to share, but not which blocks. We represent shared block b by the mean of its up-projection keys:

$$
\mu _ { b } = \frac { 1 } { M } \sum _ { h = ( b - 1 ) M + 1 } ^ { b M } \mathbf { K } _ { \mathrm { s h r } } [ : , h ] \in \mathbb { R } ^ { d \times 1 } .\tag{20}
$$

Its priority for token x is

$$
u _ { b } ( \mathbf { x } ) = \mathbf { x } \mu _ { b } .\tag{21}
$$

The $b ( \mathbf { x } )$ blocks with the largest priorities form the shared index set. All other blocks form the residual index set:

$$
\mathcal { T } _ { \mathrm { s h r } } ( \mathbf { x } ) = \mathrm { T o p K } \big ( \{ u _ { b } ( \mathbf { x } ) \} _ { b = 1 } ^ { B } , b ( \mathbf { x } ) \big ) ,\tag{22}
$$

$$
\mathcal { T } _ { \mathrm { r e s } } ( \mathbf { x } ) = \{ 1 , \dots , B \} \setminus \mathcal { T } _ { \mathrm { s h r } } ( \mathbf { x } ) .\tag{23}
$$

Two tokens can therefore use the same shared width while choosing different content. This second decision prevents shared demand from collapsing into a fixed prefix or a scalar width rule.

## C. Cumulative Residual-Expert Routing

The residual embeddings produce an affinity distribution

$$
\mathbf { s } ( \mathbf { x } ) = \mathrm { s o f t m a x } ( \mathbf { x W } _ { g } ) \in \mathbb { R } ^ { 1 \times K } .\tag{24}
$$

Let $q _ { 1 } ( \mathbf { x } ) , \ldots , q _ { K } ( \mathbf { x } )$ be the expert indices sorted by decreasing affinity. We write the sorted values as $p _ { i } ( \mathbf { x } ) = s _ { q _ { i } ( \mathbf { x } ) } ( \mathbf { x } ) \colon$

$$
p _ { 1 } ( \mathbf { x } ) \geq p _ { 2 } ( \mathbf { x } ) \geq \cdots \geq p _ { K } ( \mathbf { x } ) , \qquad p _ { i } ( \mathbf { x } ) = s _ { q _ { i } ( \mathbf { x } ) } ( \mathbf { x } ) .\tag{25}
$$

The residual demand is the complement of the shared demand:

$$
\beta ( \mathbf { x } ) = 1 - \alpha ( \mathbf { x } ) .\tag{26}
$$

Thus, the two pathways divide one budget instead of estimating their importance independently. We activate the smallest prefix whose cumulative affinity covers this demand:

$$
k ( \mathbf { x } ) = \operatorname* { m i n } \left\{ n \in \{ 1 , \dots , K \} : \sum _ { i = 1 } ^ { n } p _ { i } ( \mathbf { x } ) \geq \beta ( \mathbf { x } ) \right\} .\tag{27}
$$

Equation (27) makes expert count conditional on the preceding shared allocation. The value $\beta ( \mathbf { x } )$ says how much computation remains, while the shape of $\mathbf { s } ( \mathbf { x } )$ says how concentrated that demand is across experts. A large residual can still be covered by one expert when the distribution is sharp, whereas a diffuse distribution recruits more experts.

## D. Shared-Residual Output Merging

The selected output of the shared expert and the output of residual expert j are

$$
E _ { \mathrm { s h r } } ^ { S } ( \mathbf { x } ) = \sum _ { b \in \mathbb { Z } _ { \mathrm { s h r } } ( \mathbf { x } ) } E _ { \mathrm { s h r } } ^ { ( b ) } ( \mathbf { x } ) ,\tag{28}
$$

$$
E _ { j } ^ { \mathcal { R } } ( \mathbf { x } ) = \sum _ { b \in \mathbb { Z } _ { \mathrm { r e s } } ( \mathbf { x } ) } E _ { j } ^ { ( b ) } ( \mathbf { x } ) .\tag{29}
$$

The final layer output is

$$
\mathbf { y } = \alpha ( \mathbf { x } ) E _ { \mathrm { { s h r } } } ^ { S } ( \mathbf { x } ) + \sum _ { i = 1 } ^ { k ( \mathbf { x } ) } p _ { i } ( \mathbf { x } ) E _ { q _ { i } ( \mathbf { x } ) } ^ { \mathcal { R } } ( \mathbf { x } ) .\tag{30}
$$

The shared output is weighted by $\alpha ( \mathbf { x } )$ . Selected residual experts keep their affinities. Define $\begin{array} { r } { P _ { n } ( \mathbf { x } ) = \sum _ { i = 1 } ^ { n } p _ { i } ( \mathbf { x } ) } \end{array}$ . By the definition of the smallest sufficient prefix,

$$
\beta ( \mathbf { x } ) \leq P _ { k ( \mathbf { x } ) } ( \mathbf { x } ) < \beta ( \mathbf { x } ) + p _ { k ( \mathbf { x } ) } ( \mathbf { x } ) ,\tag{31}
$$

$$
1 \leq \alpha ( \mathbf { x } ) + P _ { k ( \mathbf { x } ) } ( \mathbf { x } ) < 1 + p _ { k ( \mathbf { x } ) } ( \mathbf { x } ) .\tag{32}
$$

The residual routing mass is at least $\beta ( \mathbf { x } )$ and exceeds it by less than the affinity of the final expert. Because $\alpha ( { \bf x } ) + \beta ( { \bf x } ) =$ 1, the total coefficient mass has a controlled overshoot while preserving the router’s confidence.

## E. Training Objective and Compute

The shared and residual embeddings should remain distinct rather than collapse into interchangeable routes. We therefore initialize the columns of $\mathbf { W } _ { g } ^ { \star }$ orthonormally and regularize their Gram matrix:

$$
{ \mathcal { L } } _ { \mathrm { d i v } } = \left\| ( \mathbf { W } _ { g } ^ { \star } ) ^ { \top } \mathbf { W } _ { g } ^ { \star } - \mathbf { I } _ { K + 1 } \right\| _ { F } ,\tag{33}
$$

$$
\begin{array} { r } { \begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { t a s k } } + \lambda _ { \mathrm { d i v } } \mathcal { L } _ { \mathrm { d i v } } , } \end{array} } \end{array}\tag{34}
$$

where ${ \mathbf { I } } _ { K + 1 }$ is the identity matrix. The diagonal terms keep each embedding at unit scale, yielding a simple normalized routing geometry. The off-diagonal terms separate routing directions, increasing expert diversity. Orthonormal initialization starts from the same geometry when $K + 1 \leq d .$ . Together, these effects promote diverse roles and sparse expert overlap without forcing the expert outputs themselves to be orthogonal.

Measured in blocks, token x activates

$$
C _ { B } ( \mathbf { x } ) = b ( \mathbf { x } ) + k ( \mathbf { x } ) [ B - b ( \mathbf { x } ) ]\tag{35}
$$

blocks. A conventional top-k layer executes kB blocks. When $k ( \mathbf { x } ) = 1$ , the shared and residual blocks together equal one complete FFN. Every additional residual expert adds only the $B - b ( \mathbf { x } )$ blocks that remain. The ordered decomposition therefore performs reusable work once and confines repeated expert cost to the residual pathway.

## IV. EXPERIMENTS

## A. Experimental Setup

a) Benchmarks and implementation details.: We evaluate UniF-MoE on vision and language tasks. For vision, the backbone is DeiT-S/16 [19] pretrained on ImageNet [20]. We use the train-validation selection criterion of DomainBed [21] and report out-of-domain accuracy on PACS [22], VLCS [23], OfficeHome [24], TerraIncognita [17], and DomainNet [25]. K and B are set to 6 and 8, respectively. For language, the backbone is BERT-large [26]. We evaluate CoLA [27], MRPC [28], QNLI [29], MNLI [30], and RTE [31] from GLUE [32], using K = 16 and $B = 1 6 .$ . Learning rate and other settings follow the corresponding baselines. Our results are averaged over three runs.

b) Comparative baselines.: Vision baselines include dense DeiT-S/16; static MoEs GMoE [33], EMoE, and EMoE-L (EMoE-learn) [6]; domain-generalization methods LFME [34], DMDA [35], and PC-MoE [36]; and dynamic MoEs DynMoE [10] and MASS [11]. Language baselines include dense BERT-large, fixed top-k MoEs with $k \in \{ 1 , 2 , 4 , 8 \}$ , DynMoE, and MASS. For fixed routing, we also report the average across k and the per-task best result.

## B. Main Results

a) Vision tasks.: Table I shows that UniF-MoE obtains the best average result on DomainBed, leading on PACS, VLCS, and DomainNet and tying the best result on Office-Home. These four datasets mix category evidence with strong style or source variation, a setting that plays to UniF-MoE’s strength: reusable blocks preserve transferable cues, while residual routing directs the remaining appearance-specific computation to the patches that need it. TerraIncognita is dominated by location-dependent backgrounds and camera viewpoints, where LFME’s explicit domain specialization remains stronger. The overall pattern supports the value of coordinating reusable and residual computation without imposing one budget on every patch.

b) Language tasks.: Table II shows that UniF-MoE performs best on all five GLUE tasks, spanning linguistic acceptability, paraphrase detection, and several forms of textual inference. The strongest fixed k changes across tasks, yet even their per-task oracle envelope remains below UniF-MoE. Choosing k per dataset still assigns one operating point to every token. UniF-MoE instead identifies reusable linguistic computation before adapting the residual expert count within each sequence, replacing a global expert budget with a tokenlevel dependency.

## C. Token-Adaptive Computation and Cost

a) Element-conditioned pathway composition.: Figure 3 groups image patches by semantic element and records their routing choices in Transformer layer 10. Shared blocks recur across several elements, whereas residual experts show more selective preferences. Patches that reuse the same shared block can still activate different residual experts, confirming the dependency at the center of our design: shared selection removes common computation, but does not predetermine who should process the remainder.

b) Patch-wise resource allocation.: Figure 4 shows alarm clocks from different domains. High-compute patches fall on the display in Product and Real World, and on the bells, ticks, and hands in Clipart. The Art example assigns minimal computation to a watermark and several strong but irrelevant strokes. The joint budget therefore responds neither to domain nor contrast alone; it reserves residual capacity for regions that carry useful class evidence after reusable features are handled.

c) Measured cost comparison.: Table III compares UniF-MoE with representative dense, static-MoE, and dynamic-MoE designs. EMoE has the lowest activated parameter count and FLOPs among the MoEs, whereas UniF-MoE achieves the lowest inference time and memory. Relative to top-2 GMoE, it activates 9.1% fewer parameters and uses 16.1% fewer FLOPs, reducing inference time and memory by 45.2% and 52.7%; it also undercuts DynMoE on every compute and runtime statistic. UniF-MoE does not minimize every analytical proxy, but sharing once and repeating only residual blocks yields the strongest measured inference profile among the MoE baselines.

## D. Ablation Study and Hyperparameter Analysis

a) Contribution of the three adaptive decisions.: Table IV breaks the ordered process one decision at a time. Every replacement lowers accuracy and increases computation on both benchmarks. Fixing α is most disruptive because an incorrect shared-residual split propagates to both later decisions. Prefix-based selection loses token-specific shared content, while top-2 residual-expert activation ignores how much demand remains. Their failures show that the gain comes from coordinating the three stages, not merely adding three forms of sparsity.

TABLE I  
OUT-OF-DOMAIN ACCURACY (%) ON DOMAINBED. – AND † DENOTE UNREPORTED AND REPRODUCED RESULTS, RESPECTIVELY.
<table><tr><td>Dataset</td><td>DeiT-S/16</td><td>GMoE</td><td>EMoE</td><td>EMoE-L</td><td>LFME</td><td>DMDA</td><td>PC-MoE</td><td>DynMoE</td><td>MASS</td><td>UniF-MoE</td></tr><tr><td>PACS</td><td>86.2</td><td>88.1</td><td>87.8</td><td>87.2</td><td>88.7</td><td>88.1</td><td>89.4</td><td>88.4</td><td>88.7</td><td>89.6</td></tr><tr><td>VLCS</td><td>79.7</td><td>80.2</td><td>79.5</td><td>79.6</td><td>79.7</td><td>79.4</td><td>80.0</td><td>80.3</td><td>81.1</td><td>81.7</td></tr><tr><td>OfficeHome</td><td>72.2</td><td>74.2</td><td>73.1</td><td>72.5</td><td>73.1</td><td>69.4</td><td>74.1</td><td>73.6</td><td>73.8</td><td>74.2</td></tr><tr><td>TerraInc.</td><td>42.0</td><td>48.5</td><td>45.9</td><td>46.1</td><td>53.4</td><td>49.5</td><td>49.9</td><td>48.9†</td><td>47.5</td><td>52.6</td></tr><tr><td>DomainNet</td><td>47.3</td><td>48.7</td><td>一</td><td>一</td><td>47.5</td><td>45.8</td><td>48.5</td><td>48.2</td><td>一</td><td>49.4</td></tr><tr><td>Avg</td><td>65.5</td><td>67.9</td><td>一</td><td>一</td><td>68.5</td><td>66.4</td><td>68.4</td><td>67.9</td><td></td><td>69.5</td></tr></table>

TABLE II

IN-DOMAIN GLUE TASK SCORES (%). MOE BEST IS THE PER-TASK ORACLE UPPER ENVELOPE OF THE FIXED TOP-k VARIANTS.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">BERT-large</td><td colspan="6">Standard MoE</td><td rowspan="2">DynMoE</td><td rowspan="2">MASS</td><td rowspan="2">UniF-MoE</td></tr><tr><td>Top-1</td><td>Top-2</td><td>Top-4</td><td>Top-8</td><td>Average</td><td>Best</td></tr><tr><td>CoLA</td><td>64.03</td><td>63.63</td><td>64.71</td><td>64.12</td><td>64.37</td><td>64.21</td><td>64.71</td><td>65.17</td><td>65.62</td><td>66.83</td></tr><tr><td>MRPC</td><td>89.36</td><td>89.81</td><td>90.18</td><td>89.74</td><td>90.35</td><td>90.02</td><td>90.35</td><td>90.64</td><td>90.58</td><td>91.57</td></tr><tr><td>QNLI</td><td>92.46</td><td>92.39</td><td>92.53</td><td>92.65</td><td>92.49</td><td>92.52</td><td>92.65</td><td>92.59</td><td>92.62</td><td>93.10</td></tr><tr><td>MNLI</td><td>86.61</td><td>86.63</td><td>86.73</td><td>86.59</td><td>86.51</td><td>86.62</td><td>86.73</td><td>86.37</td><td>86.74</td><td>86.84</td></tr><tr><td>RTE</td><td>74.37</td><td>74.01</td><td>72.32</td><td>75.33</td><td>73.53</td><td>73.80</td><td>75.33</td><td>73.41</td><td>75.33</td><td>75.47</td></tr><tr><td>Avg</td><td>81.37</td><td>81.29</td><td>81.29</td><td>81.69</td><td>81.45</td><td>81.43</td><td>81.95</td><td>81.64</td><td>82.19</td><td>82.76</td></tr></table>

![](images/91e8b9fbb80248d40573d413272ceeb48329b039d6c68453bbafe97f84a99bf4.jpg)  
(a) Original image

![](images/ff4001066b83dfb4f5577b88b60f1b51f9aea70c23c50c2076477c1bcd33e3c7.jpg)  
(b) Patch semantics

![](images/272ab341db73b422b89a81f0cf3f0c4b684dff03b4957033a83af83a6899d233.jpg)  
Fig. 3. Element-conditioned routing for a Person example of PACS. Darker cells indicate higher relative use within each semantic element.  
(c) Shared-block and residual-expert usage

![](images/0174c9b8d7fbde0bd65a90ab5f0d411c58494f1bde0f5c763a8e1cd21827961a.jpg)  
(a) Product

b) Effect of diversity regularization.: Figure 5 examines the qualitative and quantitative effects of the diversity constraint. As shown in Figure 5(a), it reduces mean pair co-activation by 62.9% and moves the router embeddings close to orthonormality. The graph does not become empty:

![](images/29b5df23394ca0fb077fe3c21145b6dc4bbedbc4de2805385dbb5c2eaded2b89.jpg)  
(b) Art

![](images/90e3875ddc4b0f4fa77117456ce4a016c339c91c291b1e72a6d1a5862b182126.jpg)  
(c) Real World

![](images/57d9e726572ce391b33e3d6eef47f42cf704ebe23c646d7e3e9828c904e5e465.jpg)  
(d) Clipart  
Fig. 4. Patch-level block usage $C _ { B } ( \mathbf { x } )$ for the Alarm Clock class across the four OfficeHome domains. High-demand patches with $C _ { B } ( { \bf x } ) \geq 2 0$ are outlined.

useful expert cooperation remains, but routing is no longer concentrated in the same few pairs. Figure 5(b) further favors the intermediate setting $\lambda _ { \mathrm { d i v } } = 0 . 0 1$ on average. A weaker constraint leaves routing directions overly correlated, whereas an excessive one can overwhelm the task loss. Together, the two panels show that moderate regularization preserves distinct residual routes without prohibiting useful cooperation.

TABLE III  
COST COMPARISON ON VLCS. N DENOTES THE TOTAL PARAMETER COUNT, AND $N _ { A }$ DENOTES THE AVERAGE ACTIVATED PARAMETER COUNT ACROSS SAMPLES; BOTH ARE REPORTED IN UNITS OF $2 ^ { 2 0 }$ PARAMETERS. FLOPS ARE REPORTED PER IMAGE IN UNITS OF $2 ^ { 3 0 }$ OPERATIONS. T-TPS AND I-TPS DENOTE TRAINING AND INFERENCE TIME PER STEP, RESPECTIVELY, IN SECONDS; T-RTM AND I-RTM DENOTE THE CORRESPONDING RUN-TIME MEMORY IN GIB. THE BEST MOE VALUES ARE BOLD.
<table><tr><td>Statistic</td><td>DeiT-S/16</td><td>GMoE</td><td>EMoE</td><td>DynMoE</td><td>UniF-MoE</td></tr><tr><td> $N$ </td><td>20.66</td><td>32.12</td><td>20.70</td><td>36.45</td><td>34.19</td></tr><tr><td> $N _ { A }$ </td><td>20.66</td><td>23.11</td><td>18.81</td><td>24.14</td><td>21.01</td></tr><tr><td>FLOPs</td><td>8.57</td><td>10.37</td><td>7.92</td><td>14.63</td><td>8.70</td></tr><tr><td>T-TPS</td><td>0.23</td><td>1.23</td><td>1.19</td><td>3.32</td><td>1.23</td></tr><tr><td>I-TPS</td><td>0.04</td><td>0.31</td><td>0.30</td><td>1.06</td><td>0.17</td></tr><tr><td>T-RTM</td><td>6.81</td><td>8.38</td><td>7.15</td><td>11.80</td><td>8.84</td></tr><tr><td>I-RTM</td><td>0.29</td><td>0.55</td><td>0.39</td><td>0.99</td><td>0.26</td></tr></table>

TABLE IV

ABLATING THE THREE ADAPTIVE DECISIONS. × INDICATES THAT THE CORRESPONDING MECHANISM IS REPLACED BY ITS NON-ADAPTIVE COUNTERPART: FIXED $\alpha = 0 . 4$ , TOP-2 RESIDUAL-EXPERT ACTIVATION, OR PREFIX-BASED BLOCK SELECTION, RESPECTIVELY. THE FULL BLOCK COUNTS ARE 56 ON DOMAINBED AND 272 ON GLUE.
<table><tr><td>Adaptive-α</td><td>Cumulative-β1</td><td>Prototype-µ</td><td colspan="2">DomainBed</td><td colspan="2">GLUE</td></tr><tr><td></td><td></td><td></td><td>Acc</td><td> $\bar { C _ { B } }$ </td><td> $\mathrm { A \bar { c } c }$ </td><td> $\bar { C _ { B } }$ </td></tr><tr><td>X</td><td>√</td><td>√</td><td>68.20</td><td>14.17</td><td>81.73</td><td>78.33</td></tr><tr><td>√</td><td>X</td><td>√</td><td>68.90</td><td>13.41</td><td>82.07</td><td>66.74</td></tr><tr><td>√</td><td>√</td><td>X</td><td>68.80</td><td>13.06</td><td>81.95</td><td>60.14</td></tr><tr><td>√</td><td>√</td><td>√</td><td>69.50</td><td>11.41</td><td>82.76</td><td>47.61</td></tr></table>

c) Shared-block granularity.: Figure 6 shows that $B = 8$ or 16 performs best on four of the five DomainBed datasets. With $B = 4$ , one selection changes a quarter of the FFN, which is too coarse to localize the shared response. With $B =$ 32, each prototype summarizes a narrow group of keys and cannot cover enough semantics. We use $B = 8$ as a stable balance between selection granularity and complexity.

## V. RELATED WORK

## A. Shared Modeling and Expert Specialization

Trained experts may contain overlapping computations. DeepSeekMoE finely segments experts into smaller ones and adds dedicated shared experts [4], while Union-of-Experts builds a virtual shared expert from routing-neuron outputs [5]. A complementary line encourages expert specialization: orthogonality and variance objectives reduce overlap [13], and MP-MoE uses inter-expert covariance to select diverse expert sets [14]. These methods improve reuse or specialization, but do not use token-specific shared allocation to define the computation that specialization should handle. UniF-MoE makes that dependency explicit: the shared pathway handles reusable block responses first, and the Gram constraint preserves distinct routes for what remains.

![](images/b04d367e94f58e0b5218f95f066e4b9ce58b586ba6653e78d7c6bcc2c0902d8b.jpg)  
(a) Residual expert co-activation CoLA

![](images/f5d3f6ca5833707278eb29147c1f7ab97e6af98e88bf22019fb6e1ff384aabf3.jpg)  
(b) Diversity-loss weight  
Fig. 5. Effect of diversity regularization. (a) Residual expert co-activation on PACS without and with $\bar { \mathcal { L } } _ { \mathrm { d i v } } .$ . (b) GLUE results with different values of $\lambda _ { \mathrm { d i v } } .$

## B. Dynamic Routing and Fine-grained Computation

Dynamic MoEs vary expert count by accumulating routing confidence [9], growing or pruning the expert pool [10], combining cumulative mass with expert expansion [11], or distributing a constrained budget across layers and tokens [12]. Fine-grained methods instead focus inside the FFN: Emergent MoE uses key centroids to expose modular structure [6], while nested and slimmable experts vary the executed width [7], [8]. These approaches adapt expert count or width, but generally make the two allocations independently and without conditioning either on a shared response. UniF-MoE orders them through one shared-residual budget: the shared demand determines shared width and content, while the residual demand determines the residual expert count.

## VI. CONCLUSION

Shared modeling, fine-grained computation, and dynamic routing need not be separate mechanisms. By exposing their shared-residual dependency, this work turns them into one ordered rule: identify reusable computation first, then route what remains. UniF-MoE realizes this rule through token-adaptive shared width, shared content, and residual expert count. Across vision and language tasks, the resulting allocation improves predictive performance while reducing activated computation and measured inference overhead. Ablations and routing analyses confirm that the three stages are complementary, establishing shared-residual decomposition as a practical basis for jointly controlling intra-expert and inter-expert computation.

![](images/24fa613f69a6f2b8748d3bee108af36fcbb89fea8664582586a9bbb0c5743aa7.jpg)  
Fig. 6. DomainBed results with different values of B.

## ACKNOWLEDGMENT

This work was funded in part by the National Natural Science Foundation of China grant under number 62536004, 62222603, in part by the Key-Area Research and Development Program of Guangdong Province under number 2023B0303030001, in part by the Program for Guangdong Introducing Innovative and Entrepreneurial Teams (2019ZT08X214), and in part by the Science and Technology Program of Guangzhou under number 2024A04J6310, and in part by the Fundamental Research Funds for the Central Universities 2025ZYGXZR021.

## REFERENCES

[1] R. A. Jacobs, M. I. Jordan, S. J. Nowlan, and G. E. Hinton, “Adaptive mixtures of local experts,” Neural Computation, vol. 3, no. 1, pp. 79–87, 1991.

[2] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, and J. Dean, “Outrageously large neural networks: The sparsely-gated mixture-of-experts layer,” in International Conference on Learning Representations, 2017.

[3] D. Lepikhin, H. Lee, Y. Xu, D. Chen, O. Firat, Y. Huang, M. Krikun, N. Shazeer, and Z. Chen, “GShard: Scaling giant models with conditional computation and automatic sharding,” in International Conference on Learning Representations, 2021.

[4] D. Dai, C. Deng, C. Zhao, R. X. Xu, H. Gao, D. Chen, J. Li, W. Zeng, X. Yu, Y. Wu, Z. Xie, Y. K. Li, P. Huang, F. Luo, C. Ruan, Z. Sui, and W. Liang, “DeepSeekMoE: Towards ultimate expert specialization in mixture-of-experts language models,” in Proceedings ofthe 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2024, pp. 1280–1297.

[5] S. Wu, A. Lv, R. Xie, X. Sun, D. Wang, R. Yan, and Y. Lin, “Union-ofexperts: Neurons in mixture-of-experts are secretly routers,” in Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2026, pp. 36 193–36 206.

[6] Z. Qiu, Z. Huang, and J. Fu, “Unlocking emergent modularity in large language models,” in Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), 2024, pp. 2638–2660.

[7] G. Jain, N. Hegde, A. Kusupati, A. Nagrani, S. Buch, P. Jain, A. Arnab, and S. Paul, “Mixture of nested experts: Adaptive processing of visual tokens,” in Advances in Neural Information Processing Systems, vol. 37, 2024, pp. 58 480–58 497.

[8] N. Tastan, S. Laskaridis, K. Nandakumar, and S. Horvath, “MoSE: Mixture of slimmable experts for efficient and adaptive language models,” in Proceedings of the Forty-third International Conference on Machine Learning (ICML), 2026. [Online]. Available: https: //openreview.net/forum?id=18C6xMcD96

[9] Q. Huang, Z. An, N. Zhuang, M. Tao, C. Zhang, Y. Jin, K. Xu, L. Chen, S. Huang, and Y. Feng, “Harder task needs more experts: Dynamic routing in MoE models,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2024, pp. 12 883–12 895.

[10] Y. Guo, Z. Cheng, X. Tang, Z. Tu, and T. Lin, “Dynamic mixture of experts: An auto-tuning approach for efficient transformer models,” in International Conference on Learning Representations, 2025.

[11] S. Park and N. Park, “How many experts are enough? towards optimal semantic specialization for mixture-of-experts,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, 2026, pp. 24 792– 24 800.

[12] B. Liu, K. Tian, W. Wang, Z. Zhang, L. Qiao, and D. Li, “Alloc-MoE: Budget-aware expert activation allocation for efficient mixture-of-experts inference,” in Proceedings ofthe 64th Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), 2026, pp. 9653– 9667.

[13] H. Guo, H. Lu, G. Nan, B. Chu, J. Zhuang, Y. Yang, W. Che, X. Cao, S. Leng, Q. Cui, and X. Jiang, “Advancing expert specialization for better MoE,” in Advances in Neural Information Processing Systems, vol. 38, 2025.

[14] X. Kang, D. Xue, Z. Wang, C. Du, X. Chen, H. Zhou, H. Chen, and C. Meng, “Breaking the echo chamber: A dynamic ensemble pruning perspective on MoE,” in International Conference on Machine Learning, 2026.

[15] M. Geva, R. Schuster, J. Berant, and O. Levy, “Transformer feed-forward layers are key-value memories,” in Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, 2021, pp. 5484– 5495.

[16] A. Komatsuzaki, J. Puigcerver, J. Lee-Thorp, C. Riquelme Ruiz, B. Mustafa, J. Ainslie, Y. Tay, M. Dehghani, and N. Houlsby, “Sparse upcycling: Training mixture-of-experts from dense checkpoints,” in International Conference on Learning Representations, 2023.

[17] S. Beery, G. Van Horn, and P. Perona, “Recognition in terra incognita,” in Proceedings of the European Conference on Computer Vision, 2018, pp. 456–473.

[18] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” in Advances in Neural Information Processing Systems, vol. 30, 2017, pp. 5998–6008.

[19] H. Touvron, M. Cord, M. Douze, F. Massa, A. Sablayrolles, and H. Jegou, “Training data-efficient image transformers and distillation´ through attention,” in Proceedings of the 38th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 139. PMLR, 2021, pp. 10 347–10 357.

[20] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and F.-F. Li, “ImageNet: A large-scale hierarchical image database,” in Proceedings of the IEEE

Conference on Computer Vision and Pattern Recognition, 2009, pp. 248– 255.

[21] I. Gulrajani and D. Lopez-Paz, “In search of lost domain generalization,” in International Conference on Learning Representations, 2021.

[22] D. Li, Y. Yang, Y.-Z. Song, and T. M. Hospedales, “Deeper, broader and artier domain generalization,” in Proceedings of the IEEE International Conference on Computer Vision, 2017, pp. 5542–5550.

[23] C. Fang, Y. Xu, and D. N. Rockmore, “Unbiased metric learning: On the utilization of multiple datasets and web images for softening bias,” in Proceedings of the IEEE International Conference on Computer Vision, 2013, pp. 1657–1664.

[24] H. Venkateswara, J. Eusebio, S. Chakraborty, and S. Panchanathan, “Deep hashing network for unsupervised domain adaptation,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017, pp. 5018–5027.

[25] X. Peng, Q. Bai, X. Xia, Z. Huang, K. Saenko, and B. Wang, “Moment matching for multi-source domain adaptation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019, pp. 1406–1415.

[26] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pretraining of deep bidirectional transformers for language understanding,” in Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), 2019, pp. 4171–4186.

[27] A. Warstadt, A. Singh, and S. R. Bowman, “Neural network acceptability judgments,” Transactions of the Association for Computational Linguistics, vol. 7, pp. 625–641, 2019.

[28] W. B. Dolan and C. Brockett, “Automatically constructing a corpus of sentential paraphrases,” in Proceedings of the Third International Workshop on Paraphrasing (IWP2005), 2005. [Online]. Available: https://aclanthology.org/I05-5002/

[29] P. Rajpurkar, J. Zhang, K. Lopyrev, and P. Liang, “SQuAD: 100,000+ questions for machine comprehension of text,” in Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, 2016, pp. 2383–2392.

[30] A. Williams, N. Nangia, and S. R. Bowman, “A broad-coverage challenge corpus for sentence understanding through inference,” in Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), 2018, pp. 1112–1122.

[31] L. Bentivogli, I. Dagan, H. T. Dang, D. Giampiccolo, and B. Magnini, “The fifth PASCAL recognizing textual entailment challenge,” in Proceedings of the Second Text Analysis Conference, 2009. [Online]. Available: https://tac.nist.gov/publications/2009/additional. papers/RTE5 overview.proceedings.pdf

[32] A. Wang, A. Singh, J. Michael, F. Hill, O. Levy, and S. R. Bowman, “GLUE: A multi-task benchmark and analysis platform for natural language understanding,” in International Conference on Learning Representations, 2019.

[33] B. Li, Y. Shen, J. Yang, Y. Wang, J. Ren, T. Che, J. Zhang, and Z. Liu, “Sparse mixture-of-experts are domain generalizable learners,” in International Conference on Learning Representations, 2023.

[34] L. Chen, Y. Zhang, Y. Song, Z. Shen, and L. Liu, “LFME: A simple framework for learning from multiple experts in domain generalization,” in Advances in Neural Information Processing Systems, vol. 37, 2024, pp. 102 919–102 947.

[35] S. Long, Q. Zhou, C. Ying, L. Ma, and Y. Luo, “Rethinking domain generalization: Discriminability and generalizability,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 34, no. 11, pp. 11 783–11 797, 2024.

[36] H. Nguyen, P. Akbarian, H. T. Pham, T. T. N. Vu, S. Zhang, and N. Ho, “Statistical advantages of perturbing cosine router in mixture of experts,” in International Conference on Learning Representations, 2025.

## APPENDIX A

## ADDITIONAL EXPERIMENTAL DETAILS

## A. Evaluation Protocol and Statistical Reporting

a) DomainBed.: We follow the train-validation selection criterion. Each domain, referred to as an environment in DomainBed, is split into an 80% in-split and a 20% outsplit. For each held-out test environment, the in-splits of the remaining environments are used for training, and the checkpoint with the highest mean out-split accuracy across these training environments is selected. Its accuracy on the held-out environment’s in-split is reported as the test result. We repeat this procedure with every environment held out, average the resulting environment accuracies to obtain one dataset score, and report the mean and standard deviation over three independently seeded runs. The run seed controls the data split and the Python, NumPy, and PyTorch random generators; deterministic cuDNN execution is enabled.

b) GLUE.: We use the official training and validation splits and select the best epoch according to the validation score. CoLA is evaluated by Matthews correlation, MRPC by the mean of accuracy and F1, and QNLI, MNLI, and RTE by accuracy; MNLI uses the matched validation split. For each task and seed, the learning rate is selected from $\{ 2 \times 1 0 ^ { - 5 } , 3 \times$ $1 0 ^ { - 5 } , 5 \times 1 0 ^ { - 5 } \}$ using validation performance. We report the mean and standard deviation over three independently seeded runs. The same seed is passed to the Python, NumPy, and PyTorch generators through Hugging Face Accelerate.

c) Performance Stability.: Tables V and VI expose the variation terms omitted from the compact main tables. For UniF-MoE, each entry is the mean ± standard deviation over three runs. Baseline variation is retained when it is available from the cited or reproduced result; an entry without a ± term was reported without a standard deviation, and “–” denotes an unavailable result. These variation estimates describe run-torun stability.

## B. Architecture and Optimization Details

a) Vision configuration.: The vision backbone is ImageNet-pretrained DeiT-S/16 with model dimension $d =$ 384 and FFN width $H = 1 5 3 6$ . Transformer layers 8 and 10 under zero-based indexing are converted to UniF-MoE layers. Each converted layer contains one shared expert, $K \ = \ 6$ residual experts, and B = 8 blocks per expert. We use hidden dropout 0.1, stochastic-depth rate 0.1. Images are normalized with ImageNet statistics. Training-domain images use random resized cropping to $2 2 4 \times 2 2 4$ , horizontal flipping, color jitter, and random grayscale; validation and test images are resized directly to $2 2 4 \times 2 2 4$ . Table VII lists the final DomainBed optimization settings.

b) Held-out environments.: PACS contains Art, Cartoon, Photo, and Sketch; VLCS contains Caltech101, LabelMe, SUN09, and VOC2007; OfficeHome contains Art, Clipart, Product, and Real World; TerraIncognita contains locations L100, L38, L43, and L46; and DomainNet contains Clipart, Infograph, Painting, Quickdraw, Real, and Sketch. Each domain is used once as the held-out test environment.

c) Language configuration.: The language backbone is BERT-large-cased. Transformer layers 20 and 22 under zerobased indexing are converted to UniF-MoE layers with $K =$ 16 residual experts and $B ~ = ~ 1 6$ blocks. We train for at most 10 epochs with FP16 mixed precision, dynamic padding, maximum sequence length 128, training and evaluation batch sizes of 32, and no gradient accumulation. Optimization uses AdamW with zero weight decay, a linear learningrate schedule, and no warm-up. The best validation epoch is retained, and training stops after four consecutive epochs without improvement.

## C. Computing Environment

Experiments are run on Ubuntu 20.04.2 LTS workers equipped with an AMD EPYC 75F3 CPU, 503 GiB of host memory, and an NVIDIA GeForce RTX 3090 GPU. The software environment uses Python 3.8.20, PyTorch 2.4.1 with CUDA 12.1 and cuDNN 9.1, torchvision 0.19.1, timm 1.0.20, Tutel 0.2, Transformers 4.46.3, Datasets 3.1.0, Evaluate 0.4.6, and Accelerate 1.0.1.

## APPENDIX B

## GENERATIVE AI USE

Generative AI tools were used solely for language editing and polishing. The authors reviewed and verified all resulting text and take full responsibility for the content of the paper.

TABLE V  
OUT-OF-DOMAIN ACCURACY (%) WITH THE VARIATION AVAILABLE FOR THE DOMAINBED MAIN RESULTS.
<table><tr><td>Method</td><td>PACS</td><td>VLCS</td><td>OfficeHome</td><td>TerraInc.</td><td>DomainNet</td></tr><tr><td>DeiT-S/16</td><td> $8 6 . 2 \pm 0 . 1$ </td><td> $7 9 . 7 \pm 0 . 0$ </td><td> $7 2 . 2 \pm 0 . 4$ </td><td> $4 2 . 0 \pm 0 . 8$ </td><td> $4 7 . 3 \pm 0 . 2$ </td></tr><tr><td>GMoE</td><td> $8 8 . 1 \pm 0 . 1$ </td><td> $8 0 . 2 \pm 0 . 2$ </td><td> ${ \bf 7 4 . 2 \pm 0 . 4 }$ </td><td> $4 8 . 5 \pm 0 . 4$ </td><td> $4 8 . 7 \pm 0 . 2$ </td></tr><tr><td>EMoE</td><td> $8 7 . 8 \pm 0 . 2$ </td><td> $7 9 . 5 \pm 0 . 4$ </td><td> $7 3 . 1 \pm 0 . 2$ </td><td> $4 5 . 9 \pm 0 . 3$ </td><td>一</td></tr><tr><td>EMoE-L</td><td> $8 7 . 2 \pm 0 . 4$ </td><td> $7 9 . 6 \pm 0 . 2$ </td><td> $7 2 . 5 \pm 0 . 2$ </td><td> $4 6 . 1 \pm 0 . 4$ </td><td>一</td></tr><tr><td>LFME</td><td> $8 8 . 7 \pm 0 . 2 $ </td><td> $7 9 . 7 \pm 0 . 1$ </td><td> $7 3 . 1 \pm 0 . 2$ </td><td> ${ \bf 5 3 . 4 \pm 0 . 4 }$ </td><td> $4 7 . 5 \pm 0 . 0$ </td></tr><tr><td>DMDA</td><td>88.1</td><td>79.4</td><td>69.4</td><td>49.5</td><td>45.8</td></tr><tr><td>PC-MoE</td><td>89.4</td><td>80.0</td><td>74.1</td><td>49.9</td><td>48.5</td></tr><tr><td>DynMoE</td><td>88.4</td><td>80.3</td><td>73.6</td><td>48.9 ± 0.5</td><td>48.2</td></tr><tr><td>MASS</td><td>88.7</td><td>81.1</td><td>73.8</td><td>47.5</td><td>一</td></tr><tr><td>UniF-MoE</td><td> ${ \bf 8 9 . 6 \pm 0 . 1 }$ </td><td> ${ \bf 8 1 . 7 \pm 0 . 3 }$ </td><td> ${ \bf 7 4 . 2 \pm 0 . 2 }$ </td><td> $5 2 . 6 \pm 0 . 3$ </td><td> ${ \bf 4 9 . 4 \pm 0 . 1 }$ </td></tr></table>

TABLE VI

GLUE TASK SCORES (%) WITH THE STANDARD DEVIATIONS OMITTED FROM THE COMPACT MAIN TABLE.
<table><tr><td>Method</td><td>CoLA</td><td>MRPC</td><td>QNLI</td><td>MNLI</td><td>RTE</td></tr><tr><td>BERT-large</td><td> $6 4 . 0 3 \pm 0 . 5 4$ </td><td> $8 9 . 3 6 \pm 0 . 0 9$ </td><td> $9 2 . 4 6 \pm 0 . 0 9$ </td><td> $8 6 . 6 1 \pm 0 . 2 6$ </td><td> $7 4 . 3 7 \pm 0 . 7 8$ </td></tr><tr><td>Standard MoE, top-1</td><td> $6 3 . 6 3 \pm 0 . 2 0$ </td><td> $8 9 . 8 1 \pm 0 . 3 0$ </td><td> $9 2 . 3 9 \pm 0 . 2 1$ </td><td> $8 6 . 6 3 \pm 0 . 1 7$ </td><td> $7 4 . 0 1 \pm 0 . 2 9$ </td></tr><tr><td>Standard MoE, top-2</td><td> $6 4 . 7 1 \pm 1 . 2 1$ </td><td> $9 0 . 1 8 \pm 1 . 3 3$ </td><td> $9 2 . 5 3 \pm 0 . 0 7$ </td><td> $8 6 . 7 3 \pm 0 . 4 3$ </td><td> $7 2 . 3 2 \pm 3 . 5 4$ </td></tr><tr><td>Standard MoE, top-4</td><td> $6 4 . 1 2 \pm 1 . 4 2$ </td><td> $8 9 . 7 4 \pm 0 . 4 0$ </td><td> $9 2 . 6 5 \pm 0 . 0 9$ </td><td> $8 6 . 5 9 \pm 0 . 1 6$ </td><td> $7 5 . 3 3 \pm 0 . 9 5$ </td></tr><tr><td>Standard MoE, top-8</td><td> $6 4 . 3 7 \pm 1 . 1 4$ </td><td> $9 0 . 3 5 \pm 0 . 6 8$ </td><td> $9 2 . 4 9 \pm 0 . 1 1$ </td><td> $8 6 . 5 1 \pm 0 . 2 0$ </td><td> $7 3 . 5 3 \pm 2 . 2 1$ </td></tr><tr><td>Standard MoE, average</td><td> $6 4 . 2 1 \pm 0 . 4 5$ </td><td> $9 0 . 0 2 \pm 0 . 2 9$ </td><td> $9 2 . 5 2 \pm 0 . 1 1$ </td><td> $8 6 . 6 2 \pm 0 . 0 9$ </td><td> $7 3 . 8 0 \pm 1 . 2 4$ </td></tr><tr><td>Standard MoE, best</td><td> $6 4 . 7 1 \pm 1 . 2 1$ </td><td> $9 0 . 3 5 \pm 0 . 6 8$ </td><td> $9 2 . 6 5 \pm 0 . 0 9$ </td><td> $8 6 . 7 3 \pm 0 . 4 3$ </td><td> $7 5 . 3 3 \pm 0 . 9 5$ </td></tr><tr><td>DynMoE</td><td> $6 5 . 1 7 \pm 0 . 2 6$ </td><td> $9 0 . 6 4 \pm 0 . 2 6$ </td><td> $9 2 . 5 9 \pm 0 . 0 8$ </td><td> $8 6 . 3 7 \pm 0 . 1 3$ </td><td> $7 3 . 4 1 \pm 1 . 9 6$ </td></tr><tr><td>MASS</td><td> $6 5 . 6 2 \pm 1 . 2 5$ </td><td> $9 0 . 5 8 \pm 0 . 4 5$ </td><td> $9 2 . 6 2 \pm 0 . 1 3$ </td><td> $8 6 . 7 4 \pm 0 . 0 7$ </td><td> $7 5 . 3 3 \pm 1 . 1 2$ </td></tr><tr><td>UniF-MoE</td><td> ${ \bf 6 6 . 8 3 \pm 0 . 1 9 }$ </td><td> ${ \bf 9 1 . 5 7 \pm 0 . 2 6 }$ </td><td> ${ \bf 9 3 . 1 0 \pm 0 . 1 2 }$ </td><td> $\mathbf { 8 6 . 8 4 \pm 0 . 0 7 }$ </td><td> ${ \bf 7 5 . 4 7 \pm 0 . 9 1 }$ </td></tr></table>

TABLE VII  
FINAL DOMAINBED OPTIMIZATION SETTINGS. ALL RUNS USE ADAM, A BATCH SIZE OF 32 PER TRAINING ENVIRONMENT, AND AN EVALUATION BATCH SIZE OF 64.
<table><tr><td>Dataset</td><td>Learning rate</td><td>Weight decay</td><td>Steps</td></tr><tr><td>PACS</td><td> $3 \times 1 0 ^ { - 5 }$ </td><td> $1 0 ^ { - 6 }$ </td><td>5,001</td></tr><tr><td>VLCS</td><td> $3 \times 1 0 ^ { - 5 }$ </td><td> $1 0 ^ { - 6 }$ </td><td>5,001</td></tr><tr><td>OfficeHome</td><td> $1 \times 1 0 ^ { - 5 }$ </td><td> $1 0 ^ { - 6 }$ </td><td>5,001</td></tr><tr><td>TerraInc.</td><td> $5 \times 1 0 ^ { - 5 }$ </td><td> $1 0 ^ { - 4 }$ </td><td>5,001</td></tr><tr><td>DomainNet</td><td> $5 \times 1 0 ^ { - 5 }$ </td><td>0</td><td>15,001</td></tr></table>