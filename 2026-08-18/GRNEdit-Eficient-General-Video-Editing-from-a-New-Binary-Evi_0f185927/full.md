# GRNEdit: Eficient General Video Editing from a New Binary-Evidence Perspective in Generative Refinement Networks

Fen<sub>g</sub> Xie<sup>1,2,\*</sup><sub>,</sub> Jia<sub>g</sub>ao Hu<sup>2</sup><sub>,</sub> Fuhao Li<sup>2</sup><sub>,</sub> Ze<sub>p</sub>en<sub>g</sub> Wan<sub>g</sub><sup>2</sup><sub>,</sub> Yuxuan Chen<sup>2</sup><sub>,</sub> Dahua Gao<sup>1,†</sup><sub>,</sub> Fei Wan<sub>g</sub><sup>2</sup><sub>,</sub> Dai<sub>g</sub>uo Zhou<sup>2</sup>

<sup>1</sup>Xidian University <sup>2</sup>MiLM Plus, Xiaomi Inc.

## Abstract

I<sub>ns</sub>t<sub>ruc</sub>ti<sub>on-</sub>b<sub>ase</sub>d <sub>genera</sub>l <sub>v</sub>id<sub>eo e</sub>diti<sub>ng see</sub>k<sub>s</sub> t<sub>o un</sub>if<sub>y</sub> di<sub>verse e</sub>diti<sub>ng opera-</sub> ti<sub>ons w</sub>ithi<sub>n a s</sub>i<sub>ng</sub>l<sub>e,</sub> i<sub>n</sub>t<sub>u</sub>iti<sub>ve</sub> i<sub>n</sub>t<sub>er</sub>f<sub>ace.</sub> E<sub>x</sub>i<sub>s</sub>ti<sub>ng approac</sub>h<sub>es o</sub>ft<sub>en re</sub>l<sub>y on</sub> <sub>resource-</sub>i<sub>n</sub>t<sub>ens</sub>i<sub>ve con</sub>diti<sub>on</sub>i<sub>ng, us</sub>i<sub>ng e</sub>ith<sub>er</sub> h<sub>eavywe</sub>i<sub>g</sub>ht b<sub>ranc</sub>h<sub>es or cos</sub>tl<sub>y</sub> <sub>source conca</sub>t<sub>ena</sub>ti<sub>on.</sub> I<sub>s</sub> th<sub>ere any e</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>way</sub> t<sub>o mo</sub>d<sub>e</sub>l <sub>e</sub>diti<sub>ng</sub> i<sub>n</sub>t<sub>en</sub>t? Th<sub>us,</sub> <sub>we</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uce</sub> GRNEdit<sub>, a</sub> li<sub>g</sub>ht<sub>we</sub>i<sub>g</sub>ht t<sub>wo-s</sub>t<sub>age</sub> f<sub>ramewor</sub>k<sub>.</sub> GRN i<sub>nsp</sub>i<sub>res our</sub> <sub>approac</sub>h b<sub>y enco</sub>di<sub>ng v</sub>i<sub>sua</sub>l <sub>seman</sub>ti<sub>cs</sub> th<sub>roug</sub>h <sub>com</sub>bi<sub>na</sub>ti<sub>ons o</sub>f bit<sub>s.</sub> Th<sub>roug</sub>h t<sub>as</sub>k<sub>-spec</sub>ifi<sub>c</sub> fi<sub>ne-</sub>t<sub>un</sub>i<sub>ng, we</sub> t<sub>a</sub>k<sub>e</sub> thi<sub>s represen</sub>t<sub>a</sub>ti<sub>on</sub> f<sub>ur</sub>th<sub>er an</sub>d <sub>recas</sub>t <sub>e</sub>diti<sub>ng</sub> <sub>seman</sub>ti<sub>cs as</sub> l<sub>oca</sub>l <sub>re</sub>t<sub>a</sub>i<sub>n-or-</sub>fli<sub>p</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>ons over</sub> i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l bit<sub>s.</sub> S<sub>ource</sub> i<sub>n</sub>f<sub>or-</sub> <sub>ma</sub>ti<sub>on</sub> i<sub>s</sub> <sub>consequen</sub>tl<sub>y</sub> <sub>mo</sub>d<sub>e</sub>l<sub>e</sub>d <sub>as</sub> <sub>coor</sub>di<sub>na</sub>t<sub>e-w</sub>i<sub>se</sub> <sub>ev</sub>id<sub>ence</sub> <sub>suppor</sub>ti<sub>ng</sub> th<sub>e</sub> <sub>o</sub>b<sub>serve</sub>d bi<sub>nary s</sub>t<sub>a</sub>t<sub>es, w</sub>hil<sub>e</sub> th<sub>e</sub> GRN b<sub>ac</sub>kb<sub>one rema</sub>i<sub>ns respons</sub>ibl<sub>e</sub> f<sub>or re-</sub> <sub>so</sub>l<sub>v</sub>i<sub>ng</sub> th<sub>e</sub>i<sub>r g</sub>l<sub>o</sub>b<sub>a</sub>l <sub>compos</sub>iti<sub>on</sub> i<sub>n</sub>t<sub>o co</sub>h<sub>eren</sub>t <sub>genera</sub>ti<sub>ve seman</sub>ti<sub>cs.</sub> I<sub>n</sub> St<sub>age</sub> I<sub>,</sub> <sub>a compac</sub>t <sub>enco</sub>d<sub>er</sub> t<sub>rans</sub>l<sub>a</sub>t<sub>es</sub> di<sub>scre</sub>t<sub>e source co</sub>d<sub>es</sub> i<sub>n</sub>t<sub>o con</sub>ti<sub>nuous ev</sub>id<sub>ence</sub> <sub>s</sub>i<sub>gna</sub>l<sub>s, w</sub>hi<sub>c</sub>h GRN <sub>ass</sub>i<sub>m</sub>il<sub>a</sub>t<sub>es</sub> th<sub>roug</sub>h<sub>ou</sub>t bi<sub>nary re</sub>fi<sub>nemen</sub>t<sub>.</sub> I<sub>nsp</sub>i<sub>re</sub>d b<sub>y</sub> <sub>nu</sub>ll<sub>-promp</sub>t t<sub>ra</sub>i<sub>n</sub>i<sub>ng</sub> f<sub>or c</sub>l<sub>ass</sub>ifi<sub>er-</sub>f<sub>ree gu</sub>id<sub>ance, we</sub> f<sub>ur</sub>th<sub>er ass</sub>i<sub>gn</sub> th<sub>e nu</sub>ll <sub>con</sub>diti<sub>on</sub> <sub>an</sub> <sub>e</sub>diti<sub>ng-spec</sub>ifi<sub>c</sub> <sub>mean</sub>i<sub>ng:</sub> <sub>an</sub> <sub>emp</sub>t<sub>y</sub> i<sub>ns</sub>t<sub>ruc</sub>ti<sub>on</sub> d<sub>eno</sub>t<sub>es</sub> <sub>no</sub> <sub>e</sub>dit <sub>an</sub>d i<sub>s superv</sub>i<sub>se</sub>d th<sub>roug</sub>h <sub>source recons</sub>t<sub>ruc</sub>ti<sub>on.</sub> Thi<sub>s</sub> id<sub>en</sub>tit<sub>y pa</sub>th<sub>way no</sub>t <sub>on</sub>l<sub>y</sub> i<sub>mp</sub>li<sub>c</sub>itl<sub>y s</sub>t<sub>reng</sub>th<sub>ens ev</sub>id<sub>ence u</sub>tili<sub>za</sub>ti<sub>on an</sub>d <sub>con</sub>t<sub>en</sub>t <sub>preserva</sub>ti<sub>on</sub> i<sub>n</sub> St<sub>age</sub> I<sub>,</sub> b<sub>u</sub>t <sub>a</sub>l<sub>so pro</sub>d<sub>uces a source-preserv</sub>i<sub>ng s</sub>t<sub>a</sub>t<sub>e</sub> i<sub>n</sub> th<sub>e same represen</sub>t<sub>a</sub>ti<sub>on space as</sub> th<sub>e e</sub>dit<sub>e</sub>d <sub>s</sub>t<sub>a</sub>t<sub>e.</sub> St<sub>age</sub> II <sub>can</sub> th<sub>ere</sub>f<sub>ore</sub> di<sub>rec</sub>tl<sub>y compare eac</sub>h <sub>e</sub>dit<sub>e</sub>d <sub>s</sub>t<sub>a</sub>t<sub>e w</sub>ith it<sub>s source-preserv</sub>i<sub>ng coun</sub>t<sub>erpar</sub>t <sub>an</sub>d <sub>use</sub> th<sub>e</sub>i<sub>r</sub> di<sub>screpancy</sub> t<sub>o rev</sub>i<sub>se unreso</sub>l<sub>ve</sub>d t<sub>arge</sub>t<sub>-</sub>bit d<sub>ec</sub>i<sub>s</sub>i<sub>ons.</sub> T<sub>ra</sub>i<sub>ne</sub>d <sub>on</sub> <sub>on</sub>l<sub>y</sub> 0<sub>.</sub>6M <sub>pa</sub>i<sub>rs</sub> <sub>w</sub>ith l<sub>ess</sub> th<sub>an</sub> 3% <sub>con</sub>diti<sub>on</sub>i<sub>ng</sub> <sub>p</sub>arameters<sub>,</sub> GRNEdit-2B and GRNEdit-8B achieve scores of 4.03 and 4.18 on O<sub>p</sub>enVE-Bench. The 2B model out<sub>p</sub>erforms multi<sub>p</sub>le 14B o<sub>p</sub>en-source editors<sub>,</sub> <sub>w</sub>hil<sub>e</sub> th<sub>e</sub> 8B <sub>mo</sub>d<sub>e</sub>l <sub>per</sub>f<sub>orms</sub> <sub>on</sub> <sub>par</sub> <sub>w</sub>ith l<sub>ea</sub>di<sub>ng</sub> <sub>open-source</sub> <sub>e</sub>dit<sub>ors.</sub>

Co<sup>d</sup>e resources are avai<sup>l</sup>a<sup>bl</sup>e at https://github.com/Foxerity/GRNEdit.

# Why Binary Evidence Enables Lightweight Editing?

![](images/3bdc5de93167661f06407ff67a0686c0bcfbb7daecf36db4a52250056d6fdadd.jpg)  
Figure 1 Continuous and codebook latents require predicting both edit locations and target v<sub>a</sub>l<sub>ues</sub>/indi<sub>ces,</sub> wh<sub>e</sub>r<sub>eas</sub> <sub>eac</sub>h bin<sub>a</sub>r<sub>y</sub> bit h<sub>as</sub> <sub>a</sub> <sub>u</sub>ni<sub>que</sub> <sub>a</sub>lt<sub>e</sub>rn<sub>a</sub>tiv<sub>e</sub> <sub>s</sub>t<sub>a</sub>t<sub>e,</sub> <sub>e</sub>n<sub>a</sub>blin<sub>g</sub> li<sub>g</sub>htw<sub>e</sub>i<sub>g</sub>ht l<sub>oca</sub>li<sub>za</sub>ti<sub>on-on</sub>l<sub>y con</sub>t<sub>ro</sub>l<sub>.</sub>

## 1 Introduction

I<sub>ns</sub>t<sub>ruc</sub>ti<sub>on-</sub>b<sub>ase</sub>d <sub>genera</sub>l <sub>v</sub>id<sub>eo e</sub>diti<sub>ng un</sub>ifi<sub>es</sub> di<sub>verse e</sub>diti<sub>ng opera</sub>ti<sub>ons un</sub>d<sub>er open-en</sub>d<sub>e</sub>d instructions [1–7]. Most existin<sub>g</sub> s<sub>y</sub>stems o<sub>p</sub>erate in the continuous s<sub>p</sub>aces of lar<sub>g</sub>e difusion <sub>or</sub> fl<sub>ow</sub> b<sub>ac</sub>kb<sub>ones.</sub> Th<sub>e</sub>i<sub>r source-con</sub>diti<sub>on</sub>i<sub>ng mec</sub>h<sub>an</sub>i<sub>sm mus</sub>t b<sub>o</sub>th i<sub>n</sub>t<sub>erpre</sub>t <sub>e</sub>diti<sub>ng</sub> i<sub>n</sub>t<sub>en</sub>t <sub>an</sub>d <sub>cons</sub>t<sub>ruc</sub>t <sub>represen</sub>t<sub>a</sub>ti<sub>ons compa</sub>tibl<sub>e w</sub>ith th<sub>e</sub> b<sub>ac</sub>kb<sub>one</sub>’<sub>s genera</sub>ti<sub>ve s</sub>t<sub>a</sub>t<sub>es.</sub> Thi<sub>s</sub> t<sub>yp</sub>i<sub>ca</sub>ll<sub>y</sub> <sub>requ</sub>i<sub>res</sub> h<sub>eavywe</sub>i<sub>g</sub>ht b<sub>ranc</sub>h<sub>es</sub> <sub>or</sub> <sub>cos</sub>tl<sub>y</sub> <sub>source</sub> <sub>conca</sub>t<sub>ena</sub>ti<sub>on.</sub> A<sub>s</sub> <sub>genera</sub>ti<sub>ve</sub> b<sub>ac</sub>kb<sub>ones</sub> <sub>sca</sub>l<sub>e,</sub> th<sub>e</sub> <sub>con</sub>diti<sub>on</sub>i<sub>ng cos</sub>t <sub>an</sub>d <sub>op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on</sub> b<sub>ur</sub>d<sub>en grow accor</sub>di<sub>ng</sub>l<sub>y.</sub>

S<sub>uc</sub>h <sub>con</sub>diti<sub>on</sub>i<sub>ng</sub> i<sub>s</sub> <sub>common</sub>l<sub>y</sub> i<sub>mp</sub>l<sub>emen</sub>t<sub>e</sub>d <sub>w</sub>ith <sub>cop</sub>i<sub>e</sub>d bl<sub>oc</sub>k<sub>s,</sub> <sub>re</sub>f<sub>erence</sub> <sub>s</sub>t<sub>reams,</sub> <sub>a</sub>d<sub>ap</sub>t<sub>ers,</sub> or source tokens <sub>p</sub>rocessed jointl<sub>y</sub> b<sub>y</sub> the backbone [8–11]. Des<sub>p</sub>ite their diferent forms, these <sub>mec</sub>h<sub>an</sub>i<sub>sms</sub> <sub>mus</sub>t <sub>s</sub>i<sub>mu</sub>lt<sub>aneous</sub>l<sub>y</sub> l<sub>oca</sub>li<sub>ze</sub> th<sub>e</sub> i<sub>n</sub>t<sub>en</sub>d<sub>e</sub>d <sub>c</sub>h<sub>ange</sub> <sub>an</sub>d <sub>pro</sub>d<sub>uce</sub> d<sub>ense,</sub> t<sub>arge</sub>t<sub>-a</sub>li<sub>gne</sub>d <sub>correc</sub>ti<sub>ons.</sub> I<sub>ncreas</sub>i<sub>ng</sub> th<sub>e</sub>i<sub>r capac</sub>it<sub>y</sub> i<sub>mproves represen</sub>t<sub>a</sub>ti<sub>on a</sub>li<sub>gnmen</sub>t b<sub>u</sub>t <sub>ra</sub>i<sub>ses</sub> t<sub>ra</sub>i<sub>n</sub>i<sub>ng an</sub>d i<sub>n</sub>f<sub>erence cos</sub>t<sub>s, w</sub>hil<sub>e s</sub>h<sub>r</sub>i<sub>n</sub>ki<sub>ng</sub> th<sub>em ma</sub>k<sub>es</sub> hi<sub>g</sub>h<sub>-</sub>di<sub>mens</sub>i<sub>ona</sub>l <sub>correc</sub>ti<sub>ons</sub> h<sub>ar</sub>d<sub>er</sub> t<sub>o</sub> l<sub>earn.</sub> W<sub>e</sub> th<sub>ere</sub>f<sub>ore</sub> <sub>as</sub>k<sub>:</sub> C<sub>an</sub> <sub>con</sub>t<sub>en</sub>t <sub>syn</sub>th<sub>es</sub>i<sub>s</sub> <sub>rema</sub>i<sub>n</sub> <sub>w</sub>ith th<sub>e</sub> b<sub>ac</sub>kb<sub>one,</sub> <sub>w</sub>hil<sub>e</sub> <sub>a</sub> li<sub>g</sub>ht<sub>we</sub>i<sub>g</sub>ht b<sub>ranc</sub>h <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>on</sub>l<sub>y source e</sub>diti<sub>ng ev</sub>id<sub>ence</sub>?

The Generative Refinement Network (GRN) <sub>p</sub>rovides a <sub>g</sub>enerative interface distinct from difusion and autore<sub>g</sub>ressive models [12]. Its Hierarchical Binar<sub>y</sub> Quantization (HBQ) encodes video l<sub>a</sub>t<sub>en</sub>t<sub>s</sub> <sub>as</sub> hi<sub>erarc</sub>hi<sub>ca</sub>l bi<sub>nary</sub> <sub>co</sub>d<sub>es,</sub> <sub>w</sub>hi<sub>c</sub>h GRN <sub>pre</sub>di<sub>c</sub>t<sub>s</sub> b<sub>y</sub> <sub>repea</sub>t<sub>e</sub>dl<sub>y</sub> <sub>re</sub>fi<sub>n</sub>i<sub>ng</sub> th<sub>e</sub> f<sub>u</sub>ll bit <sub>map.</sub> Thi<sub>s rep</sub>l<sub>aces</sub> dif<sub>us</sub>i<sub>on-s</sub>t<sub>y</sub>l<sub>e con</sub>ti<sub>nuous regress</sub>i<sub>on w</sub>ith <sub>exp</sub>li<sub>c</sub>it bi<sub>nary</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>ons.</sub> U<sub>n</sub>lik<sub>e causa</sub>l <sub>au</sub>t<sub>oregress</sub>i<sub>on,</sub> it <sub>a</sub>l<sub>so avo</sub>id<sub>s a</sub> l<sub>ong c</sub>h<sub>a</sub>i<sub>n o</sub>f i<sub>rrevers</sub>ibl<sub>e</sub> t<sub>o</sub>k<sub>en pre</sub>di<sub>c</sub>ti<sub>ons.</sub>

When the source and target are quantized by the HBQ tokenizer, they share aligned binary <sub>coor</sub>di<sub>na</sub>t<sub>es.</sub> At <sub>eac</sub>h <sub>coor</sub>di<sub>na</sub>t<sub>e,</sub> th<sub>e source prov</sub>id<sub>es an o</sub>b<sub>serve</sub>d <sub>s</sub>t<sub>a</sub>t<sub>e, an</sub>d <sub>e</sub>diti<sub>ng requ</sub>i<sub>res</sub> d<sub>ec</sub>idi<sub>ng</sub> <sub>w</sub>h<sub>e</sub>th<sub>er</sub> t<sub>o</sub> <sub>re</sub>t<sub>a</sub>i<sub>n</sub> <sub>or</sub> fli<sub>p</sub> it<sub>.</sub> C<sub>ompare</sub>d <sub>w</sub>ith <sub>pre</sub>di<sub>c</sub>ti<sub>ng</sub> <sub>con</sub>ti<sub>nuous</sub> f<sub>ea</sub>t<sub>ure</sub> <sub>correc</sub>ti<sub>ons</sub> <sub>or se</sub>l<sub>ec</sub>ti<sub>ng among mu</sub>lti<sub>p</sub>l<sub>e co</sub>d<sub>e</sub>b<sub>oo</sub>k <sub>en</sub>t<sub>r</sub>i<sub>es,</sub> thi<sub>s</sub> bi<sub>nary c</sub>h<sub>o</sub>i<sub>ce su</sub>b<sub>s</sub>t<sub>an</sub>ti<sub>a</sub>ll<sub>y re</sub>d<sub>uces</sub> th<sub>e</sub> l<sub>oca</sub>l <sub>searc</sub>h b<sub>ur</sub>d<sub>en</sub> <sub>on</sub> th<sub>e</sub> <sub>con</sub>diti<sub>on</sub>i<sub>ng</sub> <sub>pa</sub>th<sub>,</sub> <sub>as</sub> ill<sub>us</sub>t<sub>ra</sub>t<sub>e</sub>d i<sub>n</sub> Fi<sub>g.</sub> 1<sub>.</sub> Thi<sub>s</sub> <sub>o</sub>b<sub>serva</sub>ti<sub>on</sub> <sub>mo</sub>ti<sub>va</sub>t<sub>es</sub> <sub>our</sub> binary evidence perspective: the source is treated not as a residual target representation, but as <sub>ev</sub>id<sub>ence</sub> f<sub>or</sub> t<sub>arge</sub>t<sub>-</sub>bit d<sub>ec</sub>i<sub>s</sub>i<sub>ons.</sub> C<sub>on</sub>ti<sub>nuous em</sub>b<sub>e</sub>ddi<sub>ngs carry</sub> thi<sub>s ev</sub>id<sub>ence, w</sub>hil<sub>e</sub> th<sub>e</sub> GRN b<sub>ac</sub>kb<sub>one reso</sub>l<sub>ves g</sub>l<sub>o</sub>b<sub>a</sub>l bit <sub>compos</sub>iti<sub>on.</sub>

St<sub>age</sub> I <sub>rea</sub>li<sub>zes</sub> thi<sub>s separa</sub>ti<sub>on</sub> th<sub>roug</sub>h li<sub>g</sub>ht<sub>we</sub>i<sub>g</sub>ht <sub>source enco</sub>d<sub>ers</sub> th<sub>a</sub>t <sub>map source</sub> bit<sub>s</sub> i<sub>n</sub>t<sub>o</sub> hidden evidence messages at selected Transformer chunks. Instruction-aware modulation adjusts th<sub>e</sub>i<sub>r</sub> i<sub>n</sub>fl<sub>uence, w</sub>hil<sub>e</sub> GRN’<sub>s na</sub>ti<sub>ve</sub> t<sub>ar e</sub>t<sub>-</sub>bit lik<sub>e</sub>lih<sub>oo</sub>d d<sub>e</sub>t<sub>erm</sub>i<sub>nes w</sub>h<sub>e</sub>th<sub>er eac</sub>h <sub>source s</sub>t<sub>a</sub>t<sub>e</sub> <sub>s</sub>h<sub>ou</sub>ld b<sub>e re</sub>t<sub>a</sub>i<sub>ne</sub>d <sub>or over</sub>t<sub>urne</sub>d<sub>.</sub> Th<sub>e con</sub>diti<sub>on</sub>i<sub>ng pa</sub>th th<sub>ere</sub>f<sub>ore</sub> l<sub>earns</sub> th<sub>e re</sub>l<sub>evance o</sub>f <sub>source</sub> <sub>o</sub>b<sub>serva</sub>ti<sub>ons,</sub> l<sub>eav</sub>i<sub>ng</sub> t<sub>arge</sub>t <sub>syn</sub>th<sub>es</sub>i<sub>s</sub> t<sub>o</sub> th<sub>e pre</sub>t<sub>ra</sub>i<sub>ne</sub>d b<sub>ac</sub>kb<sub>one.</sub>

To anchor evidence injection in preservation semantics, we draw inspiration from null-prompt trainin<sub>g</sub> in classifier-free <sub>g</sub>uidance [13] and reinter<sub>p</sub>ret conventional condition dro<sub>p</sub>out as an id<sub>en</sub>tit<sub>y-a</sub>li<sub>gne</sub>d <sub>nu</sub>ll <sub>con</sub>diti<sub>on.</sub> I<sub>ns</sub>t<sub>ea</sub>d <sub>o</sub>f <sub>as</sub>ki<sub>ng</sub> th<sub>e mo</sub>d<sub>e</sub>l t<sub>o</sub> fit th<sub>e e</sub>dit<sub>e</sub>d t<sub>arge</sub>t <sub>a</sub>ft<sub>er</sub> th<sub>e</sub> t<sub>ex</sub>t <sub>con</sub>diti<sub>on</sub> h<sub>as</sub> b<sub>een</sub> <sub>remove</sub>d<sub>,</sub> <sub>we</sub> d<sub>e</sub>fi<sub>ne</sub> <sub>an</sub> <sub>emp</sub>t<sub>y</sub> i<sub>ns</sub>t<sub>ruc</sub>ti<sub>on</sub> <sub>as</sub> <sub>no</sub> <sub>e</sub>dit <sub>an</sub>d <sub>superv</sub>i<sub>se</sub> it th<sub>roug</sub>h <sub>source</sub> <sub>recons</sub>t<sub>ruc</sub>ti<sub>on.</sub> Th<sub>e</sub> <sub>con</sub>t<sub>ras</sub>t b<sub>e</sub>t<sub>ween</sub> <sub>e</sub>diti<sub>ng</sub> <sub>an</sub>d <sub>no</sub> <sub>e</sub>diti<sub>ng</sub> <sub>s</sub>t<sub>reng</sub>th<sub>ens</sub> St<sub>age</sub> I’<sub>s</sub> i<sub>n</sub>t<sub>erpre</sub>t<sub>a</sub>ti<sub>on o</sub>f <sub>e</sub>diti<sub>ng seman</sub>ti<sub>cs an</sub>d <sub>use o</sub>f <sub>source ev</sub>id<sub>ence, w</sub>hil<sub>e</sub> i<sub>mprov</sub>i<sub>ng con</sub>t<sub>en</sub>t <sub>preserva</sub>ti<sub>on.</sub> It <sub>a</sub>l<sub>so ena</sub>bl<sub>es</sub> th<sub>e same e</sub>dit<sub>or</sub> t<sub>o pro</sub>d<sub>uce a source-preserv</sub>i<sub>ng s</sub>t<sub>a</sub>t<sub>e</sub> i<sub>n</sub> th<sub>e same</sub> <sub>parame</sub>t<sub>er</sub>i<sub>za</sub>ti<sub>on as</sub> th<sub>e e</sub>dit<sub>e</sub>d <sub>s</sub>t<sub>a</sub>t<sub>e, prov</sub>idi<sub>ng a na</sub>t<sub>ura</sub>l <sub>re</sub>f<sub>erence</sub> f<sub>or su</sub>b<sub>sequen</sub>t <sub>rev</sub>i<sub>s</sub>i<sub>on.</sub>

Usin<sub>g</sub> this reference<sub>,</sub> Sta<sub>g</sub>e II freezes Sta<sub>g</sub>e I and em<sub>p</sub>lo<sub>y</sub>s a li<sub>g</sub>htwei<sub>g</sub>ht Bit-Mar<sub>g</sub>in Router to <sub>compare</sub> <sub>e</sub>dit<sub>e</sub>d <sub>an</sub>d <sub>source-preserv</sub>i<sub>ng</sub> <sub>s</sub>t<sub>a</sub>t<sub>es</sub> <sub>an</sub>d <sub>rev</sub>i<sub>se</sub> <sub>unreso</sub>l<sub>ve</sub>d t<sub>arge</sub>t<sub>-</sub>bit d<sub>ec</sub>i<sub>s</sub>i<sub>ons</sub> i<sub>n</sub> l<sub>og</sub>it space. The revised logits are optimized with GRN’s native target-bit likelihood objective. Source <sub>ev</sub>id<sub>ence</sub> fi<sub>rs</sub>t <sub>par</sub>ti<sub>c</sub>i<sub>pa</sub>t<sub>es</sub> i<sub>n genera</sub>ti<sub>on</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>ons</sub> i<sub>n</sub> St<sub>age</sub> I<sub>;</sub> th<sub>e</sub> id<sub>en</sub>tit<sub>y con</sub>diti<sub>on</sub> th<sub>en es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>es</sub> <sub>a preserva</sub>ti<sub>on re</sub>f<sub>erence; an</sub>d St<sub>age</sub> II t<sub>urns</sub> th<sub>a</sub>t <sub>re</sub>f<sub>erence</sub> i<sub>n</sub>t<sub>o</sub> t<sub>arge</sub>t<sub>e</sub>d <sub>rev</sub>i<sub>s</sub>i<sub>ons o</sub>f <sub>res</sub>id<sub>ua</sub>l <sub>errors.</sub> Th<sub>roug</sub>h<sub>ou</sub>t th<sub>e process,</sub> t<sub>arge</sub>t <sub>con</sub>t<sub>en</sub>t <sub>syn</sub>th<sub>es</sub>i<sub>s rema</sub>i<sub>ns</sub> th<sub>e respons</sub>ibilit<sub>y o</sub>f th<sub>e</sub> GRN b<sub>ac</sub>kb<sub>one.</sub>

O<sub>ur con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons are summar</sub>i<sub>ze</sub>d <sub>as</sub> f<sub>o</sub>ll<sub>ows:</sub>

• W<sub>e</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uce a new</sub> bi<sub>nary ev</sub>id<sub>ence perspec</sub>ti<sub>ve</sub> f<sub>or genera</sub>l <sub>v</sub>id<sub>eo e</sub>diti<sub>ng, ena</sub>bli<sub>ng e</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>separa</sub>ti<sub>on</sub> <sub>o</sub>f <sub>genera</sub>ti<sub>ve</sub> <sub>capa</sub>bilit<sub>y</sub> f<sub>rom</sub> <sub>e</sub>diti<sub>ng</sub> <sub>con</sub>t<sub>ro</sub>l <sub>an</sub>d <sub>un</sub>ifi<sub>e</sub>d <sub>mo</sub>d<sub>e</sub>li<sub>ng</sub> <sub>o</sub>f di<sub>verse</sub> <sub>e</sub>diti<sub>ng</sub> i<sub>n</sub>t<sub>en</sub>t<sub>s.</sub>

• W<sub>e</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uce</sub> GRNEdit<sub>,</sub> <sub>a</sub> li<sub>g</sub>ht<sub>we</sub>i<sub>g</sub>ht t<sub>wo-s</sub>t<sub>age</sub> f<sub>ramewor</sub>k th<sub>a</sub>t <sub>r</sub>i<sub>va</sub>l<sub>s</sub> l<sub>ea</sub>di<sub>ng</sub> <sub>v</sub>id<sub>eo</sub> <sub>e</sub>diti<sub>ng</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>w</sub>ith <sub>m</sub>i<sub>n</sub>i<sub>ma</sub>l <sub>con</sub>diti<sub>on</sub>i<sub>ng</sub> <sub>parame</sub>t<sub>ers</sub> <sub>an</sub>d t<sub>ra</sub>i<sub>n</sub>i<sub>ng</sub> d<sub>a</sub>t<sub>a.</sub>

• W<sub>e re</sub>d<sub>es</sub>i<sub>gn c</sub>l<sub>ass</sub>ifi<sub>er-</sub>f<sub>ree nu</sub>ll <sub>con</sub>diti<sub>on</sub>i<sub>ng</sub> f<sub>or e</sub>diti<sub>ng</sub> b<sub>y anc</sub>h<sub>or</sub>i<sub>ng</sub> th<sub>e emp</sub>t<sub>y</sub> i<sub>ns</sub>t<sub>ruc</sub>ti<sub>on</sub> t<sub>o</sub> <sub>source</sub> <sub>recons</sub>t<sub>ruc</sub>ti<sub>on.</sub> Thi<sub>s</sub> id<sub>en</sub>tit<sub>y</sub> <sub>anc</sub>h<sub>or</sub> <sub>s</sub>h<sub>arpens</sub> <sub>e</sub>dit/<sub>no-e</sub>dit <sub>seman</sub>ti<sub>cs,</sub> i<sub>mproves</sub> <sub>source cons</sub>i<sub>s</sub>t<sub>ency, an</sub>d b<sub>r</sub>id<sub>ges</sub> th<sub>e</sub> t<sub>wo s</sub>t<sub>ages.</sub>

![](images/9068a60135804f293fbf68ef8e6a5d9bb5fe0a1ae026b1de802c36b9fcbfea2f.jpg)  
Figure 2 Overview of GRNEdit. Given a source video and an editing instruction, Stage I injects prompt-modulated source evidence into the GRN chunks through lightweight per-chunk projectors, each containin<sub>g</sub> onl<sub>y</sub> about 3 M <sub>p</sub>arameters for GRNEdit-2B and 7 M for GRNEdit-8B. Sta<sub>g</sub>e II <sub>re</sub>fi<sub>nes</sub> th<sub>e</sub> bi<sub>nary pre</sub>di<sub>c</sub>ti<sub>ons</sub> b<sub>y</sub> t<sub>rea</sub>ti<sub>ng source ev</sub>id<sub>ence as suppor</sub>t i<sub>n</sub> k<sub>eep reg</sub>i<sub>ons an</sub>d <sub>as</sub> <sub>coun</sub>t<sub>er-ev</sub>id<sub>ence</sub> i<sub>n</sub> <sub>e</sub>dit <sub>reg</sub>i<sub>ons.</sub>

## 2 Related Work

## 2.1 Video Generation Paradigms

Video <sub>g</sub>eneration s<sub>p</sub>ans continuous iterative models and discrete autore<sub>g</sub>ressive models [14–17]. Dif<sub>us</sub>i<sub>on sys</sub>t<sub>ems opera</sub>t<sub>e on con</sub>ti<sub>nuous</sub> l<sub>a</sub>t<sub>en</sub>t<sub>s</sub> th<sub>roug</sub>h d<sub>eno</sub>i<sub>s</sub>i<sub>ng or vec</sub>t<sub>or-</sub>fi<sub>e</sub>ld i<sub>n</sub>t<sub>egra</sub>ti<sub>on</sub> [18–21]. Visual autore<sub>g</sub>ressive models o<sub>p</sub>timize discrete likelihoods but t<sub>yp</sub>icall<sub>y</sub> commit to tokenor scale-wise <sub>g</sub>eneration orders [22–24]. GRN combines discrete HBQ <sub>p</sub>rediction with <sub>g</sub>lobal <sub>ran</sub>d<sub>om re</sub>fi<sub>nemen</sub>t<sub>, repea</sub>t<sub>e</sub>dl<sub>y rev</sub>i<sub>s</sub>iti<sub>ng</sub> th<sub>e</sub> f<sub>u</sub>ll bi<sub>nary s</sub>t<sub>a</sub>t<sub>e ra</sub>th<sub>er</sub> th<sub>an</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng a</sub> fi<sub>xe</sub>d <sub>causa</sub>l order [12]. This <sub>g</sub>loball<sub>y</sub> revisable <sub>g</sub>enerator forms the backbone of GRNEdit.

## 2.2 From Continuous to Binary Representations

Difusion and flow models o<sub>p</sub>erate on continuous VAE latents [20, 21, 25], whereas VQ-based models <sub>p</sub>redict discrete codebook indices [22, 23, 26]; bitwise re<sub>p</sub>resentations re<sub>p</sub>lace this multi-<sub>way</sub> <sub>pre</sub>di<sub>c</sub>ti<sub>on</sub> <sub>w</sub>ith bi<sub>nary</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>ons.</sub> I<sub>n</sub>fi<sub>n</sub>it<sub>y</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uces</sub> bit<sub>w</sub>i<sub>se</sub> <sub>au</sub>t<sub>oregress</sub>i<sub>on,</sub> <sub>w</sub>hil<sub>e</sub> GRN extends hierarchical binar<sub>y q</sub>uantization to ima<sub>g</sub>es and videos [12, 24, 27]. Des<sub>p</sub>ite hi<sub>g</sub>her com<sub>p</sub>ression<sub>,</sub> GRN matches or sur<sub>p</sub>asses continuous VAEs in reconstruction<sub>,</sub> re<sub>p</sub>ortin<sub>g</sub> 0.56 vs. 0.87 rFID on Ima<sub>g</sub>eNet and near-<sub>p</sub>arit<sub>y</sub> on video.

## 2.3 Instruction-Based General Video Editing

Instruction-based video editing now covers object insertion, removal, and replacement, background modification and st<sub>y</sub>lization. InsV2V<sub>,</sub> InsViE-1M<sub>,</sub> Ditto<sub>,</sub> and O<sub>p</sub>enVE-3M ex<sub>p</sub>and <sub>p</sub>aired trainin<sub>g</sub> d<sub>a</sub>t<sub>a,</sub> <sub>w</sub>hil<sub>e</sub> O<sub>pen</sub>VE<sub>-</sub>B<sub>enc</sub>h <sub>an</sub>d IVEB<sub>enc</sub>h <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub> i<sub>ns</sub>t<sub>ruc</sub>ti<sub>on</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng,</sub> <sub>source</sub> <sub>preserva</sub>ti<sub>on,</sub> <sub>an</sub>d tem<sub>p</sub>oral <sub>q</sub>ualit<sub>y</sub> [1–5]. VideoDirector and InstructVEdit stud<sub>y</sub> conditional control and architecturedata co-desi<sub>g</sub>n<sub>;</sub> VEGGIE<sub>,</sub> VIVA<sub>,</sub> and CoT-Edit introduce multimodal reasonin<sub>g,</sub> <sub>g</sub>roundin<sub>g,</sub> <sub>p</sub>lannin<sub>g,</sub> or reward o<sub>p</sub>timization [6, 7, 10, 28, 29].

## 2.4 Source Conditioning and Parameter-Eficient Adaptation

S<sub>ource con</sub>diti<sub>on</sub>i<sub>ng</sub> f<sub>o</sub>ll<sub>ows severa</sub>l d<sub>es</sub>i<sub>gns.</sub> I<sub>ns</sub>t<sub>ruc</sub>tPi<sub>x</sub>2Pi<sub>x an</sub>d I<sub>ns</sub>ViE f<sub>use source an</sub>d <sub>no</sub>i<sub>sy</sub> latents before a common projection; EasyV2V concatenates separately embedded source tokens, while Omni-Video ada<sub>p</sub>ts multimodal clues into difusion conditionin<sub>g</sub> [2, 11, 30, 31]. ICVE <sub>uses we</sub>i<sub>g</sub>ht<sub>-s</sub>h<sub>are</sub>d <sub>source</sub>/<sub>e</sub>dit <sub>s</sub>t<sub>reams, w</sub>h<sub>ereas</sub> C<sub>on</sub>t<sub>ro</sub>lN<sub>e</sub>t <sub>an</sub>d VACE <sub>a</sub>dd <sub>exp</sub>li<sub>c</sub>it b<sub>ranc</sub>h<sub>es</sub> or ada<sub>p</sub>ters [8, 9, 32]. Shared wei<sub>g</sub>hts and LoRA reduce trainable <sub>p</sub>arameters [33, 34], <sub>y</sub>et th<sub>ese me</sub>th<sub>o</sub>d<sub>s s</sub>till <sub>express con</sub>t<sub>ro</sub>l th<sub>roug</sub>h <sub>con</sub>ti<sub>nuous</sub> b<sub>ac</sub>kb<sub>one s</sub>t<sub>a</sub>t<sub>es.</sub> GRNEdit i<sub>ns</sub>t<sub>ea</sub>d <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>ev</sub>id<sub>ence over</sub> bi<sub>nary</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>ons.</sub>

## 3 Method

## 3.1 Binary Evidence Formulation

Gi<sub>ven a source v</sub>id<sub>eo</sub> $V ^ { s }$ <sub>an</sub>d <sub>an open-en</sub>d<sub>e</sub>d i<sub>ns</sub>t<sub>ruc</sub>ti<sub>on</sub> $c ,$ <sub>genera</sub>l <sub>v</sub>id<sub>eo e</sub>diti<sub>ng genera</sub>t<sub>es</sub> $\widehat { V } ^ { e } =$ $F _ { \Theta } ( V ^ { s } , c )$ <sub>w</sub>hil<sub>e</sub> <sub>preserv</sub>i<sub>ng</sub> <sub>unre</sub>l<sub>a</sub>t<sub>e</sub>d <sub>appearance</sub> <sub>an</sub>d <sub>mo</sub>ti<sub>on.</sub> U<sub>s</sub>i<sub>ng</sub> GRN’<sub>s</sub> bi<sub>nary</sub> i<sub>n</sub>t<sub>er</sub>f<sub>ace,</sub> <sub>we</sub> <sub>cas</sub>t this task as source-referenced target-bit prediction. HBQ represents videos as hierarchical bit maps, which GRN <sub>p</sub>redicts b<sub>y</sub> iterativel<sub>y</sub> refinin<sub>g</sub> all coordinates in <sub>p</sub>arallel. The frozen video encoder E and HBQ tokenizer $Q _ { M }$ <sub>p</sub>ro<sup>d</sup>uce $Y ^ { u } =  { Q _ { M } } (  { { \mathcal { E } } } ( V ^ { u } ) ) \in \{ 0 , 1 \} ^ { N \times D }$ f<sub>or</sub> $u \in \{ s , e \}$ <sub>, w</sub>h<sub>ere</sub> � <sub>an</sub>d � d<sub>eno</sub>t<sub>e</sub> <sub>v</sub>i<sub>sua</sub>l <sub>pos</sub>iti<sub>ons an</sub>d bi<sub>nary c</sub>h<sub>anne</sub>l<sub>s.</sub> Th<sub>e e</sub>dit <sub>map</sub> $A ^ { e } = Y ^ { s }$ ⊕ $Y ^ { e }$ <sub>mar</sub>k<sub>s w</sub>h<sub>e</sub>th<sub>er eac</sub>h <sub>source</sub> bit i<sub>s</sub> <sub>re</sub>t<sub>a</sub>i<sub>ne</sub>d $( A _ { n , d } ^ { e } = 0 )$ <sub>or</sub> fli<sub>ppe</sub>d $( A _ { n , d } ^ { e } = 1 )$ <sub>.</sub> F<sub>or</sub> fi<sub>xe</sub>d $Y ^ { s }$ <sub>,</sub> thi<sub>s</sub> i<sub>s</sub> <sub>an</sub> <sub>exac</sub>t <sub>reparame</sub>t<sub>er</sub>i<sub>za</sub>ti<sub>on</sub> <sub>o</sub>f $Y ^ { e }$ : it <sub>preserves</sub> th<sub>e g</sub>l<sub>o</sub>b<sub>a</sub>l <sub>co</sub>d<sub>e space w</sub>hil<sub>e anc</sub>h<sub>or</sub>i<sub>ng eac</sub>h t<sub>arge</sub>t<sub>-</sub>bit d<sub>ec</sub>i<sub>s</sub>i<sub>on</sub> t<sub>o</sub> th<sub>e source.</sub> Thi<sub>s</sub> source-relative formulation matches GRN’s native objective. For two-class logits $Z _ { n , d } \in \mathbb { R } ^ { 2 }$ <sub>,</sub> d<sub>e</sub>fi<sub>ne</sub> th<sub>e source-a</sub>li<sub>gne</sub>d <sub>marg</sub>i<sub>n</sub>

$$
\kappa _ { n , d } = ( 2 Y _ { n , d } ^ { s } - 1 ) ( Z _ { n , d , 1 } - Z _ { n , d , 0 } ) .\tag{1}
$$

Th<sub>e</sub>n $\sigma ( \kappa _ { n , d } )$ i<sub>s</sub> th<sub>e</sub> <sub>re</sub>t<sub>a</sub>i<sub>n</sub> <sub>pro</sub>b<sub>a</sub>bilit<sub>y,</sub> <sub>an</sub>d th<sub>e</sub> t<sub>arge</sub>t<sub>-</sub>bit <sub>cross-en</sub>t<sub>ropy</sub> b<sub>ecomes</sub>

$$
\begin{array} { r l } & { \ell _ { n , d } = - ( 1 - A _ { n , d } ^ { e } ) \log \sigma ( \kappa _ { n , d } ) } \\ & { \phantom { \ell _ { n , d } } - A _ { n , d } ^ { e } \log \sigma ( - \kappa _ { n , d } ) , } \end{array}\tag{2}
$$

<sub>express</sub>i<sub>ng</sub> t<sub>arge</sub>t <sub>pre</sub>di<sub>c</sub>ti<sub>on</sub> <sub>as</sub> <sub>re</sub>t<sub>a</sub>i<sub>n-or-</sub>fli<sub>p</sub> <sub>superv</sub>i<sub>s</sub>i<sub>on.</sub>

Let $M ^ { s }$ collect the continuous source messages introduced below. We call them source evidence b<sub>ecause</sub> th<sub>e</sub>i<sub>r e</sub>f<sub>ec</sub>t i<sub>s measure</sub>d b<sub>y</sub> th<sub>e c</sub>h<sub>ange</sub> i<sub>n</sub> $\kappa _ { n , d }$ <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub> t<sub>o</sub> $\mathbf { \nabla } { \mathcal { M } } ^ { s } = \mathbf { 0 }$ <sub>,</sub> <sub>w</sub>ith<sub>ou</sub>t <sub>a</sub> <sub>separa</sub>t<sub>e</sub> f<sub>ea</sub>t<sub>ure</sub> t<sub>arge</sub>t<sub>.</sub> S<sub>uperv</sub>i<sub>s</sub>i<sub>on comes on</sub>l<sub>y</sub> f<sub>rom</sub> th<sub>e</sub> fi<sub>na</sub>l bit lik<sub>e</sub>lih<sub>oo</sub>d<sub>;</sub> th<sub>us,</sub> th<sub>e con</sub>diti<sub>on</sub>i<sub>ng pa</sub>th l<sub>earns</sub> h<sub>ow source o</sub>b<sub>serva</sub>ti<sub>ons</sub> bi<sub>as</sub> t<sub>arge</sub>t d<sub>ec</sub>i<sub>s</sub>i<sub>ons, w</sub>hil<sub>e</sub> GRN <sub>rema</sub>i<sub>ns respons</sub>ibl<sub>e</sub> f<sub>or g</sub>l<sub>o</sub>b<sub>a</sub>l <sub>con</sub>t<sub>en</sub>t <sub>syn</sub>th<sub>es</sub>i<sub>s.</sub>

GRN’<sub>s</sub> <sub>g</sub>l<sub>o</sub>b<sub>a</sub>l <sub>ran</sub>d<sub>om</sub> <sub>re</sub>fi<sub>nemen</sub>t k<sub>eeps</sub> thi<sub>s</sub> <sub>ev</sub>id<sub>ence</sub> <sub>rev</sub>i<sub>sa</sub>bl<sub>e.</sub> At <sub>progress</sub> $p ,$ it f<sub>orms</sub>

$$
X _ { p } ( Y ) = S _ { p } \odot Y + ( 1 - S _ { p } ) \odot Y ^ { \mathrm { r a n d } } ,\tag{3}
$$

<sub>w</sub>h<sub>ere</sub> $S _ { p }$ i<sub>s a sc</sub>h<sub>e</sub>d<sub>u</sub>l<sub>e-con</sub>t<sub>ro</sub>ll<sub>e</sub>d bi<sub>nary se</sub>l<sub>ec</sub>ti<sub>on mas</sub>k <sub>an</sub>d $Y ^ { \mathrm { r a n d } }$ <sub>con</sub>t<sub>a</sub>i<sub>ns ran</sub>d<sub>om</sub> bit<sub>s.</sub> GRN <sub>em</sub>b<sub>e</sub>d<sub>s one-</sub>h<sub>o</sub>t $( X _ { p } )$ <sub>w</sub>ith $W _ { x }$ <sub>, processes</sub> it <sub>w</sub>ith th<sub>e</sub> t<sub>ex</sub>t <sub>con</sub>diti<sub>on</sub> $C _ { \mathrm { m a i n } }$ <sub>an</sub>d <sub>progress em</sub>b<sub>e</sub>ddi<sub>ng</sub> $e _ { p }$ th<sub>roug</sub>h th<sub>e</sub> T<sub>rans</sub>f<sub>ormer</sub> $G _ { \theta . }$ <sub>, an</sub>d <sub>app</sub>li<sub>es</sub> th<sub>e</sub> bit <sub>c</sub>l<sub>ass</sub>ifi<sub>er</sub> $B _ { \phi }$ t<sub>o s</sub>t<sub>a</sub>t<sub>es se</sub>l<sub>ec</sub>t<sub>e</sub>d b<sub>y</sub> $\mathrm { \sf s e l _ { \mathrm { v i s } } }$ . Because GRN <sub>rev</sub>i<sub>s</sub>it<sub>s</sub> <sub>every</sub> bit <sub>a</sub>t <sub>eac</sub>h <sub>s</sub>t<sub>ep,</sub> <sub>source</sub> <sub>ev</sub>id<sub>ence</sub> <sub>can</sub> b<sub>e</sub> <sub>repea</sub>t<sub>e</sub>dl<sub>y</sub> <sub>re</sub>i<sub>n</sub>t<sub>erpre</sub>t<sub>e</sub>d <sub>ra</sub>th<sub>er</sub> th<sub>an</sub> i<sub>rrevers</sub>ibl<sub>y</sub> <sub>comm</sub>itt<sub>e</sub>d<sub>.</sub>

Algorithm 1 Two-stage training pipeline of GRNEdit   
Require: Source and target bits $( Y ^ { s } , Y ^ { e } ) ;$ <sub>; con</sub>diti<sub>ons</sub> $( C _ { \mathrm { m a i n } } , C _ { \mathrm { e d i t } } ) ;$ id<sub>en</sub>tit<sub>y pro</sub>b<sub>a</sub>bilit<sub>y</sub> $\rho$   
Ensure: Stage-I editor $( \bar { \theta } , \bar { \eta } , \bar { \phi } )$ and Bit-Mar<sub>g</sub>in Router $\psi$   
1: Stage I: Evidence assimilation and identity anchoring   
2: for each training sample do   
3<sub>:</sub> Sam<sub>p</sub>le refinement <sub>p</sub>ro<sub>g</sub>ress $p$ <sub>an</sub>d $b \sim$ B<sub>ernou</sub>ll $( \rho )$   
4: if $b = 1$ then   
5: $( Y ^ { \star } , C _ { \mathrm { m a i n } } ^ { \star } , C _ { \mathrm { e d i t } } ^ { \star } ) \gets ( Y ^ { s } , C _ { \emptyset } , C _ { \emptyset } )$ {identit<sub>y</sub>}   
6: else   
7: $( Y ^ { \star } , C _ { \mathrm { m a i n } } ^ { \star } , C _ { \mathrm { e d i t } } ^ { \star } ) \gets ( Y ^ { e } , C _ { \mathrm { m a i n } } , C _ { \mathrm { e d i t } } )$ {edit}   
8: end $\mathbf { i f }$   
9: $X _ { p } ^ { \star } \gets X _ { p } ( Y ^ { \star } )$   
10: $Z ^ { \star } \gets \bar { \mathrm { S T A G E I } } ( X _ { p } ^ { \star } , Y ^ { s } , C _ { \mathrm { m a i n } } ^ { \star } , C _ { \mathrm { e d i t } } ^ { \star } , p )$   
11<sub>:</sub> U<sub>p</sub>date $( \theta , \eta , \phi )$ us<sup>i</sup>n<sub>g</sub> $\mathcal { L } _ { \mathrm { b i t } } ( Z ^ { \star } , \bar { Y } ^ { \star } )$   
12: end for   
13<sub>:</sub> Freeze $( \theta , \eta , \phi ) \xrightarrow { } ( { \bar { \theta } } , { \bar { \eta } } , { \bar { \phi } } )$   
14: Stage II: Identity-referenced bit-margin revision   
15: for each training sample do   
16: $X _ { p } ^ { g } \gets X _ { p } ( Y ^ { e } )$   
17: $\dot { \mathcal { T } ^ { g } } \gets ( X _ { p } ^ { g } , Y ^ { s } , C _ { \mathrm { m a i n } } , C _ { \mathrm { e d i t } } , p )$   
18: $\mathcal { T } ^ { r } \gets ( Y ^ { \hat { s } } , Y ^ { s } , C _ { \varnothing } , C _ { \varnothing } , 1 )$   
19: $( H ^ { g } , H ^ { r } ) \gets \mathrm { F }$ rozenStageIHidden $( \mathcal { T } ^ { g } , \mathcal { T } ^ { r } )$   
20: $( Z ^ { g } , Z ^ { r } ) \gets ( B _ { \bar { \phi } } ( H ^ { g } ) , B _ { \bar { \phi } } ( H ^ { r } ) )$   
21: $R \gets \mathcal { R } _ { \psi } ( s \mathbf { g } ( \dot { H ^ { g } } ) , s \mathbf { g } ( \dot { H ^ { r } } ) , C _ { \mathrm { e d i t } } )$   
22: $Z ^ { \mathrm { r e v } } \gets \mathrm { R E V I S E M A R G I N } \left( Z ^ { g } , Z ^ { r } , R \right)$   
23: U<sub>p</sub>date $\psi$ us<sup>i</sup>n<sub>g</sub> $\mathcal { L } _ { \mathrm { b i t } } ( Z ^ { \mathrm { r e v } } , Y ^ { e } )$   
24: end for

## 3.2 GRNEdit: A Two-Stage Evidence Editing Framework

As illustrated in Fig. 2, GRNEdit organizes source-evidence injection, identity anchoring, and id<sub>en</sub>tit<sub>y-re</sub>f<sub>erence</sub>d bit<sub>-marg</sub>i<sub>n rev</sub>i<sub>s</sub>i<sub>on</sub> i<sub>n</sub>t<sub>o</sub> th<sub>e</sub> t<sub>wo-s</sub>t<sub>age p</sub>i<sub>pe</sub>li<sub>ne summar</sub>i<sub>ze</sub>d i<sub>n</sub> Al<sub>gor</sub>ith<sub>m</sub> 1<sub>.</sub> Where oh(·) denotes bitwise one-hot encoding.

Identity-Aligned Null Condition Classi<sup>fi</sup>er-<sup>f</sup>ree guidance (CFG) contrasts a conditional prediction <sub>w</sub>ith <sub>a nu</sub>ll b<sub>ranc</sub>h l<sub>earne</sub>d th<sub>roug</sub>h <sub>con</sub>diti<sub>on</sub> d<sub>ropou</sub>t<sub>,</sub> b<sub>u</sub>t thi<sub>s</sub> b<sub>ranc</sub>h l<sub>ac</sub>k<sub>s e</sub>diti<sub>ng-spec</sub>ifi<sub>c</sub> <sub>preserva</sub>ti<sub>on seman</sub>ti<sub>cs.</sub> F<sub>or e</sub>diti<sub>ng,</sub> th<sub>e nu</sub>ll <sub>opera</sub>ti<sub>on</sub> i<sub>s</sub> id<sub>en</sub>tit<sub>y: an emp</sub>t<sub>y</sub> i<sub>ns</sub>t<sub>ruc</sub>ti<sub>on</sub> d<sub>eno</sub>t<sub>es</sub> <sub>no e</sub>dit<sub>.</sub> W<sub>e</sub> th<sub>ere</sub>f<sub>ore re</sub>i<sub>n</sub>t<sub>erpre</sub>t <sub>con</sub>diti<sub>on</sub> d<sub>ropou</sub>t <sub>as</sub> id<sub>en</sub>tit<sub>y superv</sub>i<sub>s</sub>i<sub>on.</sub> F<sub>or</sub> $b \sim \mathrm { B e r n o u l l i } ( \rho )$ one draw jointly switches conditions, target, and perturbed state:

$$
\begin{array} { r } { ( Y ^ { \star } , C _ { \mathrm { m a i n } } ^ { \star } , C _ { \mathrm { e d i t } } ^ { \star } ) = \left\{ \begin{array} { l l } { ( Y ^ { e } , C _ { \mathrm { m a i n } } , C _ { \mathrm { e d i t } } ) , } & { b = 0 , } \\ { ( Y ^ { s } , C _ { \emptyset } , C _ { \emptyset } ) , } & { b = 1 , } \end{array} \right. } \\ { X _ { p } ^ { \star } = X _ { p } ( Y ^ { \star } ) . \quad } \end{array}\tag{4}
$$

Th<sub>e</sub> $C _ { \mathrm { e d i t } }$ <sub>pre</sub>di<sub>c</sub>t<sub>s</sub> $Y ^ { e }$ <sub>un</sub>d<sub>er</sub> th<sub>e</sub> i<sub>ns</sub>t<sub>ruc</sub>ti<sub>on;</sub> th<sub>e</sub> id<sub>en</sub>tit<sub>y pa</sub>th <sub>recons</sub>t<sub>ruc</sub>t<sub>s</sub> $Y ^ { s }$ f<sub>rom source s</sub>t<sub>a</sub>t<sub>es</sub> <sub>un</sub>d<sub>er nu</sub>ll <sub>con</sub>diti<sub>ons.</sub>

With $Z _ { i } ^ { \star }$ denotin<sub>g</sub> Sta<sub>g</sub>e I lo<sub>g</sub>its and $\omega _ { j }$ <sub>norma</sub>li<sub>ze</sub>d <sub>span we</sub>i<sub>g</sub>ht<sub>s,</sub> b<sub>o</sub>th <sub>pa</sub>th<sub>s op</sub>ti<sub>m</sub>i<sub>ze</sub> $\mathcal { L } _ { \mathrm { I } } ~ =$ $\begin{array} { r } { \sum _ { j } \omega _ { j } \ell _ { \mathrm { b i t } } ^ { ' } ( Z _ { j } ^ { \star } , Y _ { j } ^ { \star } ) } \end{array}$ us<sup>i</sup>n<sub>g</sub> $\ell _ { \mathrm { b i t } }$ from E<sub>q</sub>. (2). Sharin<sub>g</sub> <sub>p</sub>arameters, HBQ coordinates, and hidden s<sub>p</sub>aces t<sub>urns</sub> thi<sub>s con</sub>t<sub>ras</sub>t i<sub>n</sub>t<sub>o a</sub> l<sub>ow-cos</sub>t <sub>s</sub>i <sub>na</sub>l f<sub>or</sub> l<sub>earn</sub>i<sub>n w</sub>h<sub>en source ev</sub>id<sub>ence s</sub>h<sub>ou</sub>ld b<sub>e reserve</sub>d <sub>or</sub> <sub>overr</sub>idd<sub>en,</sub> <sub>s</sub>t<sub>reng</sub>th<sub>en</sub>i<sub>ng</sub> St<sub>age</sub> I <sub>ev</sub>id<sub>ence</sub> <sub>u</sub>tili<sub>za</sub>ti<sub>on</sub> <sub>an</sub>d <sub>con</sub>t<sub>en</sub>t <sub>preserva</sub>ti<sub>on.</sub> Id<sub>en</sub>tit<sub>y</sub> t<sub>ra</sub>i<sub>n</sub>i<sub>ng</sub> across <sub>p</sub>ro<sub>g</sub>ress <sub>y</sub>ields Sta<sub>g</sub>e II’s reference. At its clean end<sub>p</sub>oint $p = 1 , Y ^ { s }$ <sub>serves as</sub> b<sub>o</sub>th th<sub>e</sub> <sub>curren</sub>t <sub>s</sub>t<sub>a</sub>t<sub>e an</sub>d <sub>source ev</sub>id<sub>ence.</sub> Th<sub>e</sub> f<sub>rozen</sub> St<sub>age</sub> I <sub>e</sub>dit<sub>or</sub> th<sub>us prov</sub>id<sub>es a coor</sub>di<sub>na</sub>t<sub>e-a</sub>li<sub>gne</sub>d<sub>,</sub> <sub>source-preserv</sub>i<sub>ng</sub> hidd<sub>en s</sub>t<sub>a</sub>t<sub>e</sub> f<sub>or res</sub>id<sub>ua</sub>l <sub>rev</sub>i<sub>s</sub>i<sub>on, w</sub>ith<sub>ou</sub>t <sub>a separa</sub>t<sub>e enco</sub>d<sub>er.</sub>

![](images/439eeac11634f1982e8341d4802d0923218f93d9789825c0193e6a7d08c1d485.jpg)  
Figure 3 Qualitative comparison across diverse editing tasks. GRNEdit performs more accurate <sub>e</sub>dit<sub>s</sub> <sub>w</sub>hil<sub>e</sub> b<sub>e</sub>tt<sub>er</sub> <sub>preserv</sub>i<sub>ng</sub> <sub>une</sub>dit<sub>e</sub>d <sub>con</sub>t<sub>en</sub>t<sub>,</sub> <sub>w</sub>h<sub>ereas</sub> <sub>compe</sub>ti<sub>ng</sub> <sub>me</sub>th<sub>o</sub>d<sub>s</sub> <sub>o</sub>ft<sub>en</sub> <sub>m</sub>i<sub>ss</sub> th<sub>e</sub> i<sub>n</sub>t<sub>en</sub>d<sub>e</sub>d <sub>c</sub>h<sub>ange</sub> <sub>or</sub> di<sub>srup</sub>t <sub>scene</sub> <sub>cons</sub>i<sub>s</sub>t<sub>ency.</sub>

Stage I: Layerwise Source Evidence Injection With identity de<sup>fi</sup>ning what remains unchanged, Stage I learns how source evidence enters editing. Source and target share HBQ coordinates, but <sub>success</sub>i<sub>ve</sub> GRN <sub>c</sub>h<sub>un</sub>k<sub>s use</sub> dif<sub>eren</sub>t hidd<sub>en</sub> b<sub>ases; we</sub> th<sub>ere</sub>f<sub>ore a</sub>d<sub>ap</sub>t <sub>source</sub> bit<sub>s a</sub>t th<sub>e</sub> i<sub>npu</sub>t <sub>an</sub>d <sub>se</sub>l<sub>ec</sub>t<sub>e</sub>d b<sub>oun</sub>d<sub>ar</sub>i<sub>es.</sub> L<sub>e</sub>t $q ^ { s } = \mathrm { o h } ( Y ^ { s } ) \in \{ 0 , 1 \} ^ { N \times 2 D }$ <sub>.</sub> F<sub>or c</sub>h<sub>un</sub>k $k _ { z }$

$$
\begin{array} { r l } & { u _ { k } = \mathrm { R M S N o r m } \big ( s g ( [ \bar { C } _ { \mathrm { e d i t } } ; e _ { p } ] ) \big ) , } \\ & { \big [ a _ { k } ; g _ { k } \big ] = W _ { k } ^ { \mathrm { o u t } } \mathrm { S i L U } ( W _ { k } ^ { \mathrm { i n } } u _ { k } ) , } \\ & { \qquad M _ { k } ^ { s } = \mathcal { A } _ { k } ( q ^ { s } ) \odot g _ { k } \odot ( 1 + a _ { k } ) . } \end{array}\tag{5}
$$

Here $\mathcal { A } _ { k }$ <sub>maps source</sub> bit<sub>s</sub> i<sub>n</sub>t<sub>o</sub> th<sub>e</sub> �<sub>-</sub>di<sub>mens</sub>i<sub>ona</sub>l b<sub>as</sub>i<sub>s o</sub>f <sub>c</sub>h<sub>un</sub>k �<sub>, w</sub>hil<sub>e</sub> th<sub>e poo</sub>l<sub>e</sub>d <sub>e</sub>dit <sub>con</sub>diti<sub>on</sub> $\bar { C } _ { \mathrm { e d i t } }$ <sub>an</sub>d <sub>progress</sub> <sub>em</sub>b<sub>e</sub>ddi<sub>ng</sub> $e _ { p }$ <sub>p</sub>ro<sup>d</sup>uce t<sup>h</sup>e <sub>g</sub>ate $g _ { k } \in \mathbb { R } ^ { C }$ <sub>an</sub>d <sub>res</sub>id<sub>ua</sub>l <sub>sca</sub>l<sub>e</sub> $a _ { k } \in \mathbb { R } ^ { C }$ <sub>.</sub> Th<sub>us,</sub> $M _ { k } ^ { s } \in \mathbb { R } ^ { N \times C }$ i<sub>s</sub> i<sub>ns</sub>t<sub>ruc</sub>ti<sub>on-aware</sub> <sub>source</sub> <sub>ev</sub>id<sub>ence.</sub> L<sub>e</sub>t $H _ { k }$ b<sub>e</sub> th<sub>e</sub> <sub>pac</sub>k<sub>e</sub>d <sub>s</sub>t<sub>a</sub>t<sub>e,</sub> $m _ { k } \in \{ 0 , 1 \}$ <sub>se</sub>l<sub>ec</sub>t b<sub>oun</sub>d<sub>ar</sub>i<sub>es, an</sub>d $\Pi _ { \mathrm { v i s } }$ <sub>res</sub>t<sub>r</sub>i<sub>c</sub>t <sub>a</sub>dditi<sub>ons</sub> t<sub>o v</sub>i<sub>sua</sub>l <sub>spans.</sub> F<sub>or</sub> th<sub>e</sub> � T<sub>rans</sub>f<sub>ormer c</sub>h<sub>un</sub>k<sub>s</sub> i<sub>n</sub>d<sub>exe</sub>d b<sub>y</sub> $k = 0 , \ldots , K - 1$

$$
\begin{array} { r } { \begin{array} { r l } & { H _ { 0 } = [ W _ { x } \mathrm { o h } ( X _ { p } ) ; C _ { \mathrm { m a i n } } ; e _ { p } ] , } \\ & { H _ { k + 1 } = \mathrm { C h u n k } _ { k } \big ( H _ { k } + m _ { k } \Pi _ { \mathrm { v i s } } ( M _ { k } ^ { s } ) \big ) . } \end{array} } \end{array}\tag{6}
$$

Thi<sub>s preserves</sub> GRN’<sub>s na</sub>ti<sub>ve pac</sub>k<sub>e</sub>d <sub>sequence,</sub> f<sub>u</sub>ll <sub>a</sub>tt<sub>en</sub>ti<sub>on, an</sub>d bit h<sub>ea</sub>d<sub>.</sub> B<sub>ecause</sub> th<sub>e messages</sub> <sub>are superv</sub>i<sub>se</sub>d <sub>on</sub>l<sub>y</sub> th<sub>roug</sub>h <sub>e</sub>dit<sub>e</sub>d t<sub>arge</sub>t<sub>-</sub>bit lik<sub>e</sub>lih<sub>oo</sub>d<sub>,</sub> th<sub>e con</sub>diti<sub>on</sub>i<sub>ng pa</sub>th l<sub>earns</sub> h<sub>ow source</sub> <sub>o</sub>b<sub>serva</sub>ti<sub>ons s</sub>h<sub>ou</sub>ld bi<sub>as eac</sub>h d<sub>ec</sub>i<sub>s</sub>i<sub>on, w</sub>hil<sub>e</sub> GRN <sub>re</sub>t<sub>a</sub>i<sub>ns respons</sub>ibilit<sub>y</sub> f<sub>or</sub> t<sub>arge</sub>t <sub>syn</sub>th<sub>es</sub>i<sub>s.</sub>

Stage II: Identity-Referenced Bit-Margin Revision The identity path turns the CFG-inspired cont<sub>ras</sub>t i<sub>n</sub>t<sub>o</sub> <sub>e</sub>diti<sub>ng-a</sub>li<sub>gne</sub>d <sub>rev</sub>i<sub>s</sub>i<sub>on.</sub> St<sub>age</sub> II f<sub>reezes</sub> St<sub>age</sub> I <sub>an</sub>d <sub>reuses</sub> it<sub>s</sub> <sub>c</sub>l<sub>ean</sub> <sub>en</sub>d<sub>po</sub>i<sub>n</sub>t <sub>as</sub> th<sub>e</sub> <sub>preserva</sub>ti<sub>on re</sub>f<sub>erence, ra</sub>th<sub>er</sub> th<sub>an an uncon</sub>diti<sub>ona</sub>l <sub>or nega</sub>ti<sub>ve s</sub>t<sub>a</sub>t<sub>e.</sub> L<sub>e</sub>t $\bar { F }$ <sub>an</sub>d $B _ { \bar { \phi } }$ d<sub>eno</sub>t<sub>e</sub> th<sub>e</sub> f<sub>rozen</sub> <sub>v</sub>i<sub>sua</sub>l<sub>-</sub>hidd<sub>en</sub> <sub>mapp</sub>i<sub>ng</sub> <sub>an</sub>d bit h<sub>ea</sub>d<sub>,</sub> <sub>an</sub>d <sub>a</sub>bb<sub>rev</sub>i<sub>a</sub>t<sub>e</sub> $C = ( C _ { \mathrm { m a i n } } , C _ { \mathrm { e d i t } } )$ <sub>an</sub>d $\dot { C _ { \emptyset } } = ( C _ { \emptyset } , C _ { \emptyset } )$

$$
\begin{array} { l } { { H ^ { g } = \bar { F } ( X _ { p } ^ { g } ; Y ^ { s } , C , p ) , } } \\ { { H ^ { r } = \bar { F } ( Y ^ { s } ; Y ^ { s } , C _ { \emptyset } , 1 ) . } } \end{array}\tag{7}
$$

<table><tr><td>Method</td><td>Params / Time</td><td></td><td>Overall</td><td>G.Style</td><td>Bkg.Chg.</td><td>L.Chg.</td><td>L.Rem.</td><td>L.Add.</td><td>Sub.</td><td>Cond. Params</td></tr><tr><td>Runway Aleph (Commercial)</td><td>Commercial</td><td></td><td>4.49</td><td>4.41</td><td>4.40</td><td>4.31</td><td>4.64</td><td>4.36</td><td>4.21</td><td>N/A</td></tr><tr><td>VACE (ICCV 2025)</td><td>14B</td><td>/545s</td><td>3.01</td><td>3.46</td><td>2.81</td><td>2.47</td><td>3.99</td><td>1.76</td><td>4.41</td><td>3.05B</td></tr><tr><td>Omni-Video (Tech. Rep. 2025)</td><td>11B</td><td>/312s</td><td>3.66</td><td>3.41</td><td>4.11</td><td>3.75</td><td>4.52</td><td>2.80</td><td>4.95</td><td>&gt; 4.6B</td></tr><tr><td>InsViE (ICCV 2025)</td><td>2B</td><td>/64s</td><td>3.25</td><td>3.63</td><td>2.68</td><td>2.82</td><td>3.56</td><td>2.25</td><td>4.77</td><td>~59M</td></tr><tr><td>Lucy-Edit (Tech. Rep. 2025)</td><td>5B</td><td>/36s</td><td>3.77</td><td>3.64</td><td>3.25</td><td>3.93</td><td>3.95</td><td>3.92</td><td>4.23</td><td>~0.6M</td></tr><tr><td>Kiwi-Edit (arXiv 2026)</td><td>8B</td><td>/45s</td><td>4.17</td><td>4.24</td><td>4.01</td><td>4.30</td><td>4.60</td><td>4.13</td><td>4.23</td><td>3B</td></tr><tr><td>DITTO (CVPR 2026)</td><td>14B</td><td>/611s</td><td>3.44</td><td>4.48</td><td>3.52</td><td>2.89</td><td>3.53</td><td>2.48</td><td>3.69</td><td>3.17B</td></tr><tr><td>OpenVE-Edit (arXiv 2025)</td><td>5B</td><td>/150s</td><td>3.89</td><td>4.24</td><td>4.10</td><td>3.80</td><td>3.50</td><td>3.41</td><td>3.98</td><td>&gt; 3B</td></tr><tr><td>LoomVideo (arXiv 2026)</td><td>5+8B/166s</td><td></td><td>4.09</td><td>4.44</td><td>3.91</td><td>4.02</td><td>4.22</td><td>4.21</td><td>4.66</td><td>89M</td></tr><tr><td>UniVideo (ICLR 2026)</td><td>14B</td><td>/893s</td><td>4.18</td><td>4.05</td><td>3.94</td><td>4.33</td><td>4.42</td><td>4.41</td><td>4.56</td><td>&gt; 7B</td></tr><tr><td>Lance (Tech. Rep. 2026)</td><td>7.1B</td><td>/99s</td><td>4.01</td><td>3.99</td><td>4.01</td><td>3.72</td><td>4.17</td><td>4.29</td><td>4.17</td><td>N/A</td></tr><tr><td>GRNEdit-2B (Ours)</td><td>2B</td><td>/39s</td><td>4.03</td><td>4.36</td><td>3.80</td><td>3.93</td><td>4.56</td><td>3.84</td><td>4.72</td><td>37M</td></tr><tr><td>GRNEdit-8B (Ours)</td><td>8B</td><td>/84s</td><td>4.18</td><td>4.37</td><td>4.12</td><td>4.09</td><td>4.73</td><td>3.86</td><td>4.86</td><td>45M</td></tr></table>

Table 1 Comparison on OpenVE-Bench. Higher is better. Cond. Params counts structural <sub>parame</sub>t<sub>ers</sub> b<sub>eyon</sub>d th<sub>e</sub> li<sub>s</sub>t<sub>e</sub>d <sub>genera</sub>t<sub>or.</sub>

Th<sub>e</sub>i<sub>r</sub> l<sub>og</sub>it<sub>s are</sub> $Z ^ { g } ~ = ~ B _ { \bar { \phi } } ( H ^ { g } )$ <sub>an</sub>d $Z ^ { r } ~ = ~ B _ { \bar { \phi } } ( H ^ { r } )$ <sub>.</sub> Sh<sub>ar</sub>i<sub>ng</sub> th<sub>e</sub> f<sub>rozen e</sub>dit<sub>or</sub> k<sub>eeps</sub> b<sub>o</sub>th <sub>s</sub>t<sub>a</sub>t<sub>es</sub> <sub>coor</sub>di<sub>na</sub>t<sub>e-a</sub>li<sub>gne</sub>d<sub>.</sub>

At <sub>eac</sub>h <sub>coor</sub>di<sub>na</sub>t<sub>e,</sub> th<sub>e</sub> Bit<sub>-</sub>M<sub>arg</sub>i<sub>n</sub> R<sub>ou</sub>t<sub>er quer</sub>i<sub>es w</sub>ith $H ^ { r }$ <sub>an</sub>d <sub>a</sub>tt<sub>en</sub>d<sub>s</sub> t<sub>o</sub> th<sub>e a</sub>li<sub>gne</sub>d <sub>re</sub>f<sub>erence,</sub> generation, and pooled condition as three key–value tokens. Q/K RMS normalization and a GRN-style gated FFN are followed by a zero-initialized per-bit projection. Let ${ \widetilde { H } } ^ { u } = s \ g ( H ^ { u } )$ f<sub>or</sub> $u \in \{ g , r \}$ <sub>.</sub> Th<sub>e rou</sub>t<sub>er pre</sub>di<sub>c</sub>t<sub>s a</sub> t<sub>an</sub>h<sub>-</sub>b<sub>oun</sub>d<sub>e</sub>d $R \in ( - 1 , 1 ) ^ { N \times D }$ <sub>an</sub>d <sub>app</sub>li<sub>es</sub> it t<sub>o</sub> th<sub>e a</sub>li<sub>gne</sub>d <sub>marg</sub>i<sub>n</sub> <sup>di</sup>scre<sub>p</sub>anc<sub>y</sub>:

$$
\begin{array} { r l } & { \delta m = m ( Z ^ { r } ) - s g ( m ( Z ^ { g } ) ) , } \\ & { \quad R = \mathcal { R } _ { \psi } ( \widetilde { H } ^ { g } , \widetilde { H } ^ { r } , C _ { \mathrm { e d i t } } ) , } \\ & { \quad { Z } ^ { \mathrm { r e v } } = Z ^ { g } + \frac { 1 } { 2 } ( R \odot \delta m ) \otimes ( - 1 , 1 ) . } \end{array}\tag{8}
$$

Here $m ( Z ) = Z _ { \cdot , \cdot , 1 } - Z _ { \cdot , \cdot , 0 }$ . The s<sub>y</sub>mmetric u<sub>p</sub>date <sub>g</sub>ives $m ( Z ^ { \mathrm { r e v } } ) = m ( Z ^ { g } ) + R \odot \delta m$ <sub>w</sub>hil<sub>e preserv</sub>i<sub>ng</sub> th<sub>e</sub> <sub>mean</sub> l<sub>og</sub>it<sub>.</sub> Th<sub>e</sub> <sub>rou</sub>t<sub>er</sub> th<sub>ere</sub>f<sub>ore</sub> l<sub>earns</sub> <sub>on</sub>l<sub>y</sub> <sub>a</sub> <sub>per-</sub>bit <sub>s</sub>i<sub>gne</sub>d <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>on</sub> thi<sub>s</sub> di<sub>screpancy,</sub> <sub>ra</sub>th<sub>er</sub> th<sub>an</sub> <sub>syn</sub>th<sub>es</sub>i<sub>z</sub>i<sub>ng</sub> <sub>a</sub> <sub>new</sub> <sub>correc</sub>ti<sub>on</sub> di<sub>rec</sub>ti<sub>on.</sub> H<sub>ence,</sub> $R \ : = \ : 0$ recovers Sta<sub>g</sub>e I, $R  1$ <sub>approac</sub>h<sub>es</sub> th<sub>e</sub> id<sub>en</sub>tit<sub>y-re</sub>f<sub>erence</sub> <sub>marg</sub>i<sub>n,</sub> <sub>an</sub>d $R < 0$ <sub>moves</sub> <sub>away</sub> <sub>w</sub>h<sub>en</sub> <sub>requ</sub>i<sub>re</sub>d b<sub>y</sub> th<sub>e</sub> <sub>e</sub>dit<sub>.</sub> Zero initialization therefore leaves Stage I unchanged before training. Overall, Stage I injects <sub>source</sub> <sub>ev</sub>id<sub>ence</sub> d<sub>ur</sub>i<sub>ng</sub> <sub>syn</sub>th<sub>es</sub>i<sub>s;</sub> id<sub>en</sub>tit<sub>y</sub> <sub>anc</sub>h<sub>ors</sub> <sub>preserva</sub>ti<sub>on;</sub> <sub>an</sub>d St<sub>age</sub> II <sub>rev</sub>i<sub>ses</sub> <sub>res</sub>id<sub>ua</sub>l bit d<sub>ec</sub>i<sub>s</sub>i<sub>ons aga</sub>i<sub>ns</sub>t th<sub>a</sub>t <sub>anc</sub>h<sub>or un</sub>d<sub>er</sub> th<sub>e e</sub>dit <sub>con</sub>diti<sub>on.</sub>

<table><tr><td rowspan="2">Method</td><td colspan="4">Add</td><td colspan="4">Remove</td><td colspan="4">Replace</td><td colspan="4">Style</td></tr><tr><td> $S _ { \mathrm { E A } }$ </td><td>SVN</td><td> $S _ { \mathrm { V Q } }$ </td><td>S</td><td> $S _ { \mathrm { E A } }$ </td><td> $S _ { \mathrm { V N } }$ </td><td> $S _ { \mathrm { V Q } }$ </td><td>S</td><td> $S _ { \mathrm { E A } }$ </td><td> $S _ { \mathrm { V N } }$ </td><td> $S _ { \mathrm { V Q } }$ </td><td>S</td><td> $S _ { \mathrm { E A } }$ </td><td>SvN</td><td> $S _ { \mathrm { V Q } }$ </td><td>S</td></tr><tr><td>InsViE</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>2.60 3.10 3.46 3.05 2.44 3.76 3.29 3.16 2.10 3.91 3.49 3.17 8.17 8.21 7.35 7.91</td><td></td></tr><tr><td>Lucy-Edit 6.47 5.70</td><td></td><td></td><td>6.77</td><td>6.31 7.02</td><td></td><td>6.88</td><td></td><td></td><td></td><td></td><td></td><td>6.81 6.90 7.08 6.21 6.88 6.72</td><td></td><td></td><td></td><td>2 4.65 4.67 5.17 4.83</td></tr><tr><td>DITTO</td><td></td><td>6.70 7.57</td><td>8.41</td><td>7.56 5.48</td><td></td><td>6.67</td><td>6.93</td><td></td><td>6.36 4.56 7.21 7.96 6.58 9.20</td><td></td><td></td><td></td><td></td><td>9.07</td><td>8.77 9.01</td><td></td></tr><tr><td>ReCo</td><td></td><td>8.54 7.55 8.61 8.23 7.28</td><td></td><td></td><td></td><td>6.90</td><td></td><td></td><td>6.82 7.00 9.43 8.01 8.77 8.74 9.42 9.19</td><td></td><td></td><td></td><td></td><td></td><td>8.90 9.17</td><td></td></tr><tr><td>OURS-2B 7.53</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>6.91 7.44 7.29 8.39 7.76 7.81 7.99 8.60 7.24 7.99 7.94 9.03 9.23 8.97 9.08</td><td></td></tr></table>

Table 2 Comparison on ReCo-Bench across four editing tasks. Results are ranked within each t<sub>as</sub>k<sub>;</sub> hi<sub>g</sub>h<sub>er</sub> i<sub>s</sub> b<sub>e</sub>tt<sub>er.</sub>

![](images/e647c0f4b399f483dc0a495af6d981bcb643d413f217eaa0905de9b63dc095d1.jpg)  
Figure 4 Stage II residual evidence refinement. Comparing the Stage I edited state with its id<sub>en</sub>tit<sub>y-a</sub>li<sub>gne</sub>d <sub>source re</sub>f<sub>erence res</sub>t<sub>ores source-cons</sub>i<sub>s</sub>t<sub>en</sub>t d<sub>e</sub>t<sub>a</sub>il<sub>s</sub> i<sub>n preserve</sub>d <sub>reg</sub>i<sub>ons.</sub>

## 4 Experiments

## 4.1 Experimental Setup

Evaluation. We evaluate GRNEdit on OpenVE-Bench and ReCo-Bench [35]. OpenVE-Bench <sub>covers e</sub>i<sub>g</sub>ht <sub>e</sub>diti<sub>ng</sub> t<sub>as</sub>k<sub>s an</sub>d <sub>repor</sub>t<sub>s ca</sub>t<sub>egory-w</sub>i<sub>se an</sub>d O<sub>vera</sub>ll <sub>scores.</sub> R<sub>e</sub>C<sub>o-</sub>B<sub>enc</sub>h <sub>eva</sub>l<sub>ua</sub>t<sub>es</sub> Add<sub>,</sub> Re<sub>p</sub>lace<sub>,</sub> Remove<sub>,</sub> and St<sub>y</sub>le usin<sub>g</sub> $S _ { \mathrm { E A } } , S _ { \mathrm { V N } } , S _ { \mathrm { V Q } }$ <sub>,</sub> <sub>an</sub>d th<sub>e</sub> <sub>aggrega</sub>t<sub>e</sub> <sub>score</sub> �<sub>.</sub> Hi<sub>g</sub>h<sub>er</sub> i<sub>s</sub> b<sub>e</sub>tt<sub>er</sub> for all metrics. O<sub>p</sub>en-source baselines [31, 36–41] in the tables follow their <sub>p</sub>ublic settin<sub>g</sub>s and <sub>na</sub>ti<sub>ve reso</sub>l<sub>u</sub>ti<sub>ons;</sub> R<sub>unway</sub> Al<sub>ep</sub>h i<sub>s</sub> i<sub>nc</sub>l<sub>u</sub>d<sub>e</sub>d <sub>on</sub>l<sub>y as a commerc</sub>i<sub>a</sub>l <sub>re</sub>f<sub>erence.</sub>

Implementation details. We construct a task-balanced set o<sup>f</sup> 0.6M training pairs <sup>f</sup>rom the public OpenVE-3M dataset. For each sample, we use the Qwen3-VL-Flash API to generate a source-aware <sub>repromp</sub>t f<sub>rom</sub> th<sub>e source v</sub>id<sub>eo an</sub>d <sub>raw</sub> i<sub>ns</sub>t<sub>ruc</sub>ti<sub>on.</sub> Th<sub>e repromp</sub>t i<sub>s conca</sub>t<sub>ena</sub>t<sub>e</sub>d <sub>w</sub>ith th<sub>e</sub> <sub>or</sub>i<sub>g</sub>i<sub>na</sub>l i<sub>ns</sub>t<sub>ruc</sub>ti<sub>on.</sub> Th<sub>e</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>are</sub> t<sub>ra</sub>i<sub>ne</sub>d f<sub>or</sub> 60K <sub>op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on</sub> <sub>s</sub>t<sub>eps</sub> <sub>us</sub>i<sub>ng</sub> Ad<sub>am</sub>W <sub>w</sub>ith $\beta _ { 1 } = 0 . 9$ <sub>an</sub>d $\beta _ { 2 } = 0 . 9 9 9$ <sub>,</sub> <sub>w</sub>hil<sub>e</sub> th<sub>e</sub> l<sub>earn</sub>i<sub>ng</sub> <sub>ra</sub>t<sub>e</sub> d<sub>ecays</sub> f<sub>rom</sub> $4 \times 1 0 ^ { - 5 } ~ \mathrm { t o } ~ 1 \times 1 0 ^ { - 5 }$ . Trainin<sub>g</sub> uses 16 H200 GPUs. Further details on im<sub>p</sub>lementation<sub>,</sub> sam<sub>p</sub>lin<sub>g,</sub> and VLM <sub>p</sub>rom<sub>p</sub>tin<sub>g</sub> are <sub>p</sub>rovided in A<sub>pp</sub>endix A1.

<table><tr><td>Configuration</td><td>Overall</td><td>G.Style</td><td>L.Chg.</td><td>Sub.</td></tr><tr><td colspan="5">(a) Source-evidence injection placement</td></tr><tr><td>Head only (1) First + last chunks (2)</td><td>3.88 4.03</td><td>4.03 4.36</td><td>3.68 3.93</td><td>4.87 4.72</td></tr><tr><td>Uniform 3 chunks All 7 chunks</td><td>3.97 3.95</td><td>4.23 4.29</td><td>3.77 3.78</td><td>4.74 4.62</td></tr><tr><td>(b) Identity supervision ratio Identity 0%</td><td>3.97</td><td>4.29</td><td>3.84</td><td>4.75</td></tr><tr><td>Identity 5% Identity 10%</td><td>4.01 4.03</td><td>4.39 4.36</td><td>3.87 3.93</td><td>4.71</td></tr><tr><td>(c) MLLM reprompt</td><td></td><td></td><td></td><td>4.72</td></tr><tr><td>Without reprompt</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>3.97</td><td>4.25</td><td>3.84</td><td>4.63</td></tr><tr><td>With reprompt</td><td>4.03</td><td>4.36</td><td>3.93</td><td>4.72</td></tr><tr><td>(d) Stage II refinement</td><td></td><td></td><td></td><td></td></tr><tr><td>Without Stage II</td><td>4.01</td><td>4.37</td><td>3.88</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>4.68</td></tr><tr><td>With Stage II</td><td>4.03</td><td>4.36</td><td>3.93</td><td>4.72</td></tr></table>

Table 3 Ablation studies on OpenVE-Bench using the 2B backbone. Results are ranked within <sub>eac</sub>h <sub>con</sub>t<sub>ro</sub>ll<sub>e</sub>d <sub>group.</sub>

## 4.2 Experimental Analysis

Th<sub>e ma</sub>i<sub>n resu</sub>lt<sub>s exam</sub>i<sub>ne w</sub>h<sub>e</sub>th<sub>er separa</sub>ti<sub>ng source-ev</sub>id<sub>ence assessmen</sub>t f<sub>rom con</sub>t<sub>en</sub>t <sub>syn</sub>th<sub>es</sub>i<sub>s</sub> <sub>can ac</sub>hi<sub>eve s</sub>t<sub>rong e</sub>diti<sub>ng w</sub>ith li<sub>g</sub>ht<sub>we</sub>i<sub>g</sub>ht <sub>con</sub>diti<sub>on</sub>i<sub>ng.</sub> T<sub>a</sub>bl<sub>e</sub> 1 <sub>con</sub>fi<sub>rms</sub> thi<sub>s sca</sub>l<sub>e–qua</sub>lit<sub>y–</sub> eficienc<sub>y</sub> balance. With onl<sub>y</sub> 37M conditionin<sub>g</sub> <sub>p</sub>arameters<sub>,</sub> GRNEdit-2B achieves 4.03 Overall and out<sub>p</sub>erforms several lar<sub>g</sub>er 5B–14B editors<sub>;</sub> GRNEdit-8B reaches 4.18 with 45M conditionin<sub>g</sub> <sub>parame</sub>t<sub>ers, ma</sub>t<sub>c</sub>hi<sub>ng</sub> th<sub>e</sub> b<sub>es</sub>t <sub>open-source score.</sub> Th<sub>e</sub>i<sub>r per-v</sub>id<sub>eo</sub> i<sub>n</sub>f<sub>erence</sub> ti<sub>mes are</sub> 39 <sub>s an</sub>d 84 <sub>s,</sub> <sub>respec</sub>ti<sub>ve</sub>l<sub>y,</sub> <sub>rema</sub>i<sub>n</sub>i<sub>ng</sub> <sub>su</sub>b<sub>s</sub>t<sub>an</sub>ti<sub>a</sub>ll<sub>y</sub> b<sub>e</sub>l<sub>ow</sub> <sub>mos</sub>t l<sub>arger</sub> b<sub>ase</sub>li<sub>nes.</sub> GRNEdit i<sub>s</sub> <sub>par</sub>ti<sub>cu</sub>l<sub>ar</sub>l<sub>y</sub> <sub>s</sub>t<sub>rong</sub> <sub>on</sub> l<sub>oca</sub>li<sub>za</sub>ti<sub>on-sens</sub>iti<sub>ve</sub> t<sub>as</sub>k<sub>s,</sub> t<sub>a</sub>ki<sub>ng</sub> th<sub>e</sub> b<sub>es</sub>t <sub>open-source</sub> <sub>scores</sub> <sub>on</sub> B<sub>ac</sub>k<sub>groun</sub>d Ch<sub>ange</sub> and L<sub>o</sub>cal R<sub>e</sub>m<sub>o</sub>v<sub>e.</sub>

Fi<sub>g.</sub> 3 <sub>revea</sub>l<sub>s</sub> th<sub>e correspon</sub>di<sub>ng con</sub>t<sub>ro</sub>l b<sub>e</sub>h<sub>av</sub>i<sub>or.</sub> A<sub>cross</sub> di<sub>verse g</sub>l<sub>o</sub>b<sub>a</sub>l <sub>an</sub>d l<sub>oca</sub>l <sub>e</sub>dit<sub>s,</sub> GRNEdit modifies the intended content while preserving source geometry and untargeted subjects. In multi-<sub>person</sub> <sub>scenes,</sub> it <sub>se</sub>l<sub>ec</sub>t<sub>s</sub> th<sub>e</sub> <sub>correc</sub>t <sub>ac</sub>t<sub>or</sub> <sub>an</sub>d <sub>recons</sub>t<sub>ruc</sub>t<sub>s</sub> <sub>remove</sub>d <sub>reg</sub>i<sub>ons</sub> <sub>w</sub>ith<sub>ou</sub>t di<sub>s</sub>t<sub>ur</sub>bi<sub>ng</sub> the remaining content; competing methods more often under-edit, alter the wrong subject, or disrupt unrelated appearance. This behavior directly supports our binary evidence perspective: d<sub>ecoup</sub>li<sub>ng</sub> <sub>e</sub>dit d<sub>ec</sub>i<sub>s</sub>i<sub>ons</sub> f<sub>rom</sub> <sub>seman</sub>ti<sub>c</sub> <sub>syn</sub>th<sub>es</sub>i<sub>s</sub> <sub>re</sub>li<sub>eves</sub> th<sub>e</sub> <sub>con</sub>diti<sub>on</sub>i<sub>ng</sub> b<sub>ranc</sub>h<sub>,</sub> <sub>re</sub>d<sub>uc</sub>i<sub>ng</sub> <sub>parame</sub>t<sub>er over</sub>h<sub>ea</sub>d <sub>w</sub>hil<sub>e ena</sub>bli<sub>ng more prec</sub>i<sub>se con</sub>t<sub>ro</sub>l<sub>.</sub>

T<sub>a</sub>bl<sub>e</sub> 2 f<sub>ur</sub>th<sub>er va</sub>lid<sub>a</sub>t<sub>es</sub> thi<sub>s conc</sub>l<sub>us</sub>i<sub>on on</sub> R<sub>e</sub>C<sub>o-</sub>B<sub>enc</sub>h<sub>, ex</sub>t<sub>en</sub>di<sub>ng eva</sub>l<sub>ua</sub>ti<sub>on</sub> b<sub>eyon</sub>d th<sub>e</sub> di<sub>s-</sub> t<sub>r</sub>ib<sub>u</sub>ti<sub>on o</sub>f <sub>our</sub> O<sub>pen</sub>VE<sub>-</sub>d<sub>er</sub>i<sub>ve</sub>d t<sub>ra</sub>i<sub>n</sub>i<sub>ng se</sub>t<sub>.</sub> It<sub>s</sub> $S _ { \mathrm { V N } }$ <sub>an</sub>d $S _ { \mathrm { V Q } }$ f<sub>ur</sub>th<sub>er assess mo</sub>ti<sub>on na</sub>t<sub>ura</sub>l<sub>ness,</sub> t<sub>empora</sub>l <sub>s</sub>t<sub>a</sub>bilit<sub>y, an</sub>d <sub>e</sub>dit <sub>s</sub>t<sub>a</sub>bilit<sub>y.</sub> GRNEdit <sub>ran</sub>k<sub>s</sub> fi<sub>rs</sub>t <sub>across a</sub>ll R<sub>emove componen</sub>t<sub>s an</sub>d r<sub>eac</sub>h<sub>es</sub> 9<sub>.</sub>08 <sub>o</sub>n St<sub>y</sub>l<sub>e</sub> with th<sub>e</sub> b<sub>es</sub>t $S _ { \mathrm { V N } }$ <sub>an</sub>d $S _ { \mathrm { V Q } }$ <sub>,</sub> d<sub>emons</sub>t<sub>ra</sub>ti<sub>ng</sub> th<sub>a</sub>t GRNEdit’<sub>s</sub> <sub>s</sub>t<sub>reng</sub>th<sub>s</sub> i<sub>n</sub> <sub>e</sub>dit l<sub>oca</sub>li<sub>za</sub>ti<sub>on,</sub> <sub>preserva</sub>ti<sub>on,</sub> <sub>an</sub>d t<sub>empora</sub>l <sub>qua</sub>lit<sub>y</sub> t<sub>rans</sub>f<sub>er</sub> <sub>ro</sub>b<sub>us</sub>tl<sub>y</sub> t<sub>o</sub> <sub>a</sub> <sub>separa</sub>t<sub>e</sub> b<sub>enc</sub>h<sub>mar</sub>k<sub>.</sub>

Why Call It Evidence? Figure 5 shows why the source branch <sup>f</sup>unctions as evidence rather than a <sub>res</sub>id<sub>ua</sub>l <sub>represen</sub>t<sub>a</sub>ti<sub>on.</sub> W<sub>e a</sub>bl<sub>a</sub>t<sub>e</sub> th<sub>e</sub> b<sub>ranc</sub>h <sub>an</sub>d <sub>com</sub>bi<sub>ne s</sub>i<sub>gne</sub>d <sub>marg</sub>i<sub>n c</sub>h<sub>anges w</sub>ith th<sub>e</sub>i<sub>r</sub> RMS <sub>magn</sub>it<sub>u</sub>d<sub>e</sub> t<sub>o v</sub>i<sub>sua</sub>li<sub>ze</sub> b<sub>o</sub>th th<sub>e</sub> di<sub>rec</sub>ti<sub>on an</sub>d <sub>s</sub>t<sub>reng</sub>th <sub>o</sub>f it<sub>s</sub> i<sub>n</sub>fl<sub>uence on</sub> b<sub>ac</sub>kb<sub>one</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>ons.</sub> Th<sub>e</sub> <sub>resu</sub>lt<sub>s s</sub>h<sub>ow</sub> th<sub>a</sub>t <sub>ev</sub>id<sub>ence</sub> h<sub>as a c</sub>l<sub>ear</sub> di<sub>rec</sub>ti<sub>on: pos</sub>iti<sub>ve va</sub>l<sub>ues re</sub>t<sub>a</sub>i<sub>n</sub> th<sub>e source</sub> bit <sub>an</sub>d <sub>nega</sub>ti<sub>ve</sub> <sub>va</sub>l<sub>ues</sub> fli<sub>p</sub> it<sub>.</sub> R<sub>es</sub>id<sub>ua</sub>l f<sub>ea</sub>t<sub>ures</sub> l<sub>ac</sub>k <sub>suc</sub>h <sub>a s</sub>h<sub>are</sub>d <sub>seman</sub>ti<sub>c or</sub>d<sub>er an</sub>d <sub>en</sub>t<sub>ang</sub>l<sub>e e</sub>dit l<sub>oca</sub>ti<sub>on w</sub>ith <sub>con</sub>t<sub>en</sub>t<sub>.</sub> Alth<sub>oug</sub>h <sub>a</sub>tt<sub>en</sub>ti<sub>on we</sub>i<sub>g</sub>ht<sub>s</sub> i<sub>n con</sub>ti<sub>nuous mo</sub>d<sub>e</sub>l<sub>s may appear pro</sub>b<sub>a</sub>bili<sub>s</sub>ti<sub>c,</sub> th<sub>ey are</sub> t<sub>rans</sub>i<sub>en</sub>t <sub>rou</sub>ti<sub>ng coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s;</sub> th<sub>e represen</sub>t<sub>a</sub>ti<sub>on wr</sub>itt<sub>en</sub> b<sub>ac</sub>k t<sub>o</sub> th<sub>e</sub> b<sub>ac</sub>kb<sub>one comes</sub> f<sub>rom va</sub>l<sub>ue</sub> <sub>aggrega</sub>ti<sub>on.</sub> I<sub>n con</sub>t<sub>ras</sub>t<sub>, our suppor</sub>t <sub>s</sub>i<sub>gna</sub>l<sub>s ac</sub>t di<sub>rec</sub>tl<sub>y on</sub> th<sub>e</sub> b<sub>ac</sub>kb<sub>one</sub>’<sub>s na</sub>ti<sub>ve</sub> bi<sub>nary</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>on</sub> <sub>ax</sub>i<sub>s.</sub> E<sub>mp</sub>i<sub>r</sub>i<sub>ca</sub>ll<sub>y,</sub> <sub>ev</sub>id<sub>ence</sub> <sub>a</sub>d<sub>ap</sub>t<sub>s</sub> it<sub>s</sub> <sub>s</sub>t<sub>reng</sub>th t<sub>o</sub> <sub>eac</sub>h t<sub>as</sub>k<sub>,</sub> <sub>preserv</sub>i<sub>ng</sub> <sub>une</sub>dit<sub>e</sub>d <sub>con</sub>t<sub>en</sub>t <sub>w</sub>hil<sub>e</sub> releasin<sub>g</sub> intended edit re<sub>g</sub>ions to the backbone. Retained bits receive 6.2× stron<sub>g</sub>er source-ali<sub>g</sub>ned <sub>responses</sub> th<sub>an c</sub>h<sub>ange</sub>d bit<sub>s, an</sub>d <sub>s</sub>t<sub>reng</sub>th d<sub>ecreases w</sub>ith <sub>source–</sub>t<sub>arge</sub>t di<sub>sagreemen</sub>t i<sub>n e</sub>i<sub>g</sub>ht t<sub>as</sub>k<sub>s.</sub> Th<sub>ese</sub> di<sub>rec</sub>ti<sub>ona</sub>l<sub>, spa</sub>ti<sub>a</sub>ll<sub>y se</sub>l<sub>ec</sub>ti<sub>ve pa</sub>tt<sub>erns es</sub>t<sub>a</sub>bli<sub>s</sub>h <sub>ev</sub>id<sub>ence as a</sub> f<sub>unc</sub>ti<sub>ona</sub>l b<sub>e</sub>h<sub>av</sub>i<sub>or,</sub> <sub>no</sub>t <sub>a re</sub>l<sub>a</sub>b<sub>e</sub>l<sub>e</sub>d <sub>res</sub>id<sub>ua</sub>l f<sub>ea</sub>t<sub>ure; see</sub> A<sub>ppen</sub>di<sub>x</sub> A2<sub>.</sub>

![](images/6e06f1c224dee35027c50a4baedfe9fe4214ec7dfee7ec06e69981bab4578212.jpg)  
Figure 5 Why call it evidence? The heatmaps visualize source-induced changes in target-bit <sub>marg</sub>i<sub>ns.</sub> U<sub>n</sub>lik<sub>e</sub> <sub>conven</sub>ti<sub>ona</sub>l <sub>res</sub>id<sub>ua</sub>l <sub>represen</sub>t<sub>a</sub>ti<sub>ons,</sub> <sub>source</sub> <sub>ev</sub>id<sub>ence</sub> <sub>prov</sub>id<sub>es</sub> <sub>measura</sub>bl<sub>e,</sub> <sub>an</sub>d <sub>or</sub>d<sub>ere</sub>d <sub>suppor</sub>t f<sub>or</sub> bi<sub>nary e</sub>diti<sub>ng</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>ons.</sub>

Cross-backbone convergence and eficiency. Figure 7 compares Evidence and VACE-style conditi<sub>on</sub>i<sub>ng us</sub>i<sub>ng</sub> th<sub>e c</sub>l<sub>oses</sub>t <sub>arc</sub>hit<sub>ec</sub>t<sub>ure-compa</sub>tibl<sub>e</sub> i<sub>mp</sub>l<sub>emen</sub>t<sub>a</sub>ti<sub>ons;</sub> Th<sub>e</sub> VACE<sub>-s</sub>t<sub>y</sub>l<sub>e var</sub>i<sub>an</sub>t <sub>cop</sub>i<sub>es</sub> <sub>on</sub>l<sub>y</sub> th<sub>e</sub> fi<sub>rs</sub>t <sub>an</sub>d l<sub>as</sub>t bl<sub>oc</sub>k<sub>s</sub> t<sub>o a</sub>li<sub>gn w</sub>ith th<sub>e ev</sub>id<sub>ence-</sub>b<sub>ase</sub>d d<sub>es</sub>i<sub>gn.</sub> O<sub>n</sub> GRN<sub>,</sub> GRNEdit <sub>reac</sub>h<sub>es a</sub> com<sub>p</sub>etitive 3.70 after onl<sub>y</sub> 2K u<sub>p</sub>dates and<sub>,</sub> b<sub>y</sub> 6K<sub>,</sub> sur<sub>p</sub>asses VACE at 16K with a branch over 100× smaller. Infinit<sub>y</sub> re<sub>p</sub>roduces this orderin<sub>g</sub> on a se<sub>p</sub>arate binar<sub>y</sub> backbone, ar<sub>g</sub>uin<sub>g</sub> a<sub>g</sub>ainst GRN<sub>-spec</sub>ifi<sub>c arc</sub>hit<sub>ec</sub>t<sub>ure or pre</sub>t<sub>ra</sub>i<sub>n</sub>i<sub>ng as</sub> th<sub>e so</sub>l<sub>e exp</sub>l<sub>ana</sub>ti<sub>on.</sub> W<sub>an prov</sub>id<sub>es a comp</sub>l<sub>emen</sub>t<sub>ary</sub> control: without binary coordinates, Evidence is approximated by direct residual injection in l<sub>a</sub>t<sub>en</sub>t <sub>space, w</sub>h<sub>ere</sub> VACE <sub>per</sub>f<sub>orms</sub> b<sub>e</sub>tt<sub>er.</sub> Thi<sub>s represen</sub>t<sub>a</sub>ti<sub>on-</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>reversa</sub>l<sub>,</sub> t<sub>oge</sub>th<sub>er w</sub>ith f<sub>as</sub>t<sub>er convergence on</sub> b<sub>o</sub>th bi<sub>nary</sub> b<sub>ac</sub>kb<sub>ones, s</sub>h<sub>ows</sub> th<sub>a</sub>t <sub>ev</sub>id<sub>ence</sub> i<sub>s spec</sub>ifi<sub>ca</sub>ll<sub>y su</sub>it<sub>e</sub>d t<sub>o</sub> bi<sub>nary</sub> <sub>represen</sub>t<sub>a</sub>ti<sub>ons,</sub> <sub>ra</sub>th<sub>er</sub> th<sub>an</sub> <sub>mere</sub>l<sub>y</sub> <sub>a</sub> <sub>rename</sub>d f<sub>orm</sub> <sub>o</sub>f <sub>conven</sub>ti<sub>ona</sub>l <sub>con</sub>diti<sub>on</sub>i<sub>ng.</sub>

## 4.3 Ablation Studies

MLLM reprompt. Removing the reprompt reduces the Overall score <sup>f</sup>rom 4.03 to 3.97. Richer <sub>scene</sub> d<sub>escr</sub>i<sub>p</sub>ti<sub>ons</sub> h<sub>e</sub>l<sub>p</sub> <sub>reso</sub>l<sub>ve</sub> <sub>am</sub>bi<sub>gu</sub>iti<sub>es</sub> i<sub>n</sub> <sub>open-en</sub>d<sub>e</sub>d i<sub>ns</sub>t<sub>ruc</sub>ti<sub>ons,</sub> <sub>w</sub>hil<sub>e</sub> th<sub>e</sub> <sub>mo</sub>d<sub>es</sub>t <sub>ga</sub>i<sub>n</sub> <sub>con</sub>fi<sub>rms</sub> th<sub>a</sub>t <sub>repromp</sub>ti<sub>ng</sub> i<sub>s</sub> <sub>a</sub> <sub>comp</sub>l<sub>emen</sub>t<sub>ary</sub> t<sub>ex</sub>t <sub>en</sub>h<sub>ancemen</sub>t <sub>ra</sub>th<sub>er</sub> th<sub>an</sub> th<sub>e</sub> <sub>source</sub> <sub>o</sub>f <sub>p</sub>er<sup>f</sup>ormance.

![](images/f907f64166a6bfa7a1bcd23072078c89fae09336d3b2a52438eb844e0a4d5d26.jpg)  
Figure 6 Under a fixed two-injection budget, we retain the first and last chunks, which exhibit th<sub>e</sub> <sub>s</sub>t<sub>ronges</sub>t <sub>an</sub>d t<sub>empora</sub>ll<sub>y</sub> <sub>comp</sub>l<sub>emen</sub>t<sub>ary</sub> <sub>res</sub>id<sub>ua</sub>l <sub>u</sub>tili<sub>za</sub>ti<sub>on</sub> i<sub>n</sub> th<sub>e</sub> f<sub>u</sub>ll<sub>y</sub> <sub>mo</sub>d<sub>e</sub>l<sub>.</sub>

Evidence injection placement. Injecting all chunks noticeably slows convergence but brings limited gains in Table 3. We therefore visualize residual utilization relative to the pre-injection hidden <sub>s</sub>t<sub>a</sub>t<sub>es ac</sub>r<sub>oss e</sub>ditin<sub>g</sub> t<sub>as</sub>k<sub>s.</sub> Fi<sub>gu</sub>r<sub>e</sub> 6 r<sub>e</sub>v<sub>ea</sub>l<sub>s s</sub>tr<sub>o</sub>n<sub>ge</sub>r <sub>a</sub>nd t<sub>e</sub>m<sub>po</sub>r<sub>a</sub>ll<sub>y co</sub>m<sub>p</sub>l<sub>e</sub>m<sub>e</sub>nt<sub>a</sub>r<sub>y</sub> r<sub>espo</sub>n<sub>ses</sub> <sub>a</sub>t th<sub>e</sub> fi<sub>rs</sub>t <sub>an</sub>d l<sub>as</sub>t <sub>c</sub>h<sub>un</sub>k<sub>s, w</sub>hil<sub>e</sub> i<sub>n</sub>t<sub>erme</sub>di<sub>a</sub>t<sub>e c</sub>h<sub>un</sub>k<sub>s rema</sub>i<sub>n wea</sub>k<sub>.</sub> C<sub>ons</sub>i<sub>s</sub>t<sub>en</sub>tl<sub>y, en</sub>d<sub>po</sub>i<sub>n</sub>t injection <sub>g</sub>ives the best Overall, G.St<sub>y</sub>le, and L.Ch<sub>g</sub>. scores (4.03/4.36/3.93), whereas in<sub>p</sub>ut-onl<sub>y</sub>, <sub>u</sub>nif<sub>o</sub>rm-thr<sub>ee, a</sub>nd <sub>a</sub>ll-<sub>se</sub>v<sub>e</sub>n v<sub>a</sub>ri<sub>a</sub>nt<sub>s</sub> r<sub>eac</sub>h <sub>o</sub>nl<sub>y</sub> 3<sub>.</sub>88<sub>,</sub> 3<sub>.</sub>97<sub>, a</sub>nd 3<sub>.</sub>95 Ov<sub>e</sub>r<sub>a</sub>ll<sub>.</sub> W<sub>e</sub> th<sub>us</sub> r<sub>e</sub>t<sub>a</sub>in th<sub>e</sub> <sub>en</sub>d<sub>po</sub>i<sub>n</sub>t<sub>s</sub> f<sub>or</sub> th<sub>e</sub> b<sub>es</sub>t <sub>accuracy–e</sub>fi<sub>c</sub>i<sub>ency</sub> t<sub>ra</sub>d<sub>e-o</sub>f<sub>.</sub>

Identity-aligned null condition. Without identity supervision, the Overall score is 3.97. Using 5% and 10% identit<sub>y</sub> sam<sub>p</sub>les raises it to 4.01 and 4.03<sub>,</sub> res<sub>p</sub>ectivel<sub>y</sub>. The consistent <sub>g</sub>ain shows that <sub>a</sub>li<sub>g</sub>nin<sub>g</sub> th<sub>e</sub> n<sub>u</sub>ll in<sub>s</sub>tr<sub>uc</sub>ti<sub>o</sub>n with <sub>sou</sub>r<sub>ce</sub> r<sub>eco</sub>n<sub>s</sub>tr<sub>uc</sub>ti<sub>o</sub>n <sub>p</sub>r<sub>o</sub>vid<sub>es a c</sub>l<sub>ea</sub>r <sub>e</sub>dit/n<sub>o</sub>-<sub>e</sub>dit <sub>a</sub>n<sub>c</sub>h<sub>o</sub>r <sub>a</sub>nd b<sub>ene</sub>fit<sub>s con</sub>t<sub>en</sub>t <sub>preserva</sub>ti<sub>on.</sub> B<sub>eyon</sub>d it<sub>s unexpec</sub>t<sub>e</sub>d i<sub>mprovemen</sub>t i<sub>n con</sub>t<sub>en</sub>t <sub>preserva</sub>ti<sub>on,</sub> thi<sub>s</sub> simple objective anchors the edit/no-edit distinction and supplies the source reference connecting Sta<sub>g</sub>e I and Sta<sub>g</sub>e II.

Stage II refinement. Stage II adds only 30M parameters. It reuses the identity-aligned path f<sub>rom</sub> St<sub>age</sub> I <sub>as</sub> <sub>a</sub> <sub>same-space</sub> <sub>source</sub> <sub>re</sub>f<sub>erence</sub> <sub>an</sub>d l<sub>earns</sub> t<sub>arge</sub>t<sub>e</sub>d <sub>rev</sub>i<sub>s</sub>i<sub>ons</sub> f<sub>rom</sub> th<sub>e</sub> <sub>source–</sub> <sub>genera</sub>ti<sub>on</sub> di<sub>screpancy, res</sub>t<sub>or</sub>i<sub>ng source-cons</sub>i<sub>s</sub>t<sub>en</sub>t d<sub>e</sub>t<sub>a</sub>il<sub>s w</sub>ith<sub>ou</sub>t <sub>suppress</sub>i<sub>ng</sub> th<sub>e</sub> i<sub>n</sub>t<sub>en</sub>d<sub>e</sub>d <sub>e</sub>dit<sub>.</sub> B<sub>ecause</sub> VLM<sub>-</sub>b<sub>ase</sub>d O<sub>vera</sub>ll <sub>scor</sub>i<sub>ng</sub> i<sub>s</sub> i<sub>nsens</sub>iti<sub>ve</sub> t<sub>o p</sub>i<sub>xe</sub>l<sub>-</sub>l<sub>eve</sub>l d<sub>r</sub>ift i<sub>n non-e</sub>dit<sub>e</sub>d <sub>reg</sub>i<sub>ons,</sub> th<sub>e</sub> <sub>cons</sub>i<sub>s</sub>t<sub>ency</sub> b<sub>ene</sub>fit <sub>o</sub>f St<sub>age</sub> II i<sub>s on</sub>l<sub>y wea</sub>kl<sub>y re</sub>fl<sub>ec</sub>t<sub>e</sub>d i<sub>n</sub> th<sub>e aggrega</sub>t<sub>e score.</sub> Thi<sub>s</sub> b<sub>ene</sub>fit i<sub>s</sub> more directl<sub>y</sub> evidenced b<sub>y</sub> a 1.8 dB PSNR <sub>g</sub>ain (22.0 to 23.8 dB) over se<sub>g</sub>mentation-identified <sub>non-e</sub>dit<sub>e</sub>d <sub>reg</sub>i<sub>ons</sub> i<sub>n</sub> O<sub>pen</sub>VE<sub>-</sub>B<sub>enc</sub>h’<sub>s non-g</sub>l<sub>o</sub>b<sub>a</sub>l <sub>e</sub>diti<sub>ng</sub> t<sub>as</sub>k<sub>s an</sub>d th<sub>e recovere</sub>d <sub>v</sub>i<sub>sua</sub>l d<sub>e</sub>t<sub>a</sub>il<sub>s</sub> <sub>across</sub> <sub>rep</sub>l<sub>acemen</sub>t <sub>an</sub>d <sub>remova</sub>l t<sub>as</sub>k<sub>s</sub> i<sub>n</sub> Fi<sub>g.</sub> 4<sub>.</sub>

![](images/d2ff66201dd3289a53e3915455ab5277b80300973388d204db9939d6e8a61871.jpg)  
Figure 7 Cross-backbone convergence on OpenVE-Bench. Under matched settings within each b<sub>ac</sub>kb<sub>one, ev</sub>id<sub>ence con</sub>diti<sub>on</sub>i<sub>ng converges</sub> f<sub>as</sub>t<sub>er</sub> th<sub>an</sub> VACE<sub>-s</sub>t<sub>y</sub>l<sub>e con</sub>diti<sub>on</sub>i<sub>ng on</sub> b<sub>o</sub>th GRN <sub>an</sub>d I<sub>n</sub>fi<sub>n</sub>it<sub>y, w</sub>hil<sub>e us</sub>i<sub>ng</sub> f<sub>ewer a</sub>dditi<sub>ona</sub>l <sub>parame</sub>t<sub>ers.</sub> M<sub>ar</sub>k<sub>er s</sub>i<sub>ze</sub> d<sub>eno</sub>t<sub>es a</sub>dditi<sub>ona</sub>l <sub>parame</sub>t<sub>ers.</sub>

## 5 Conclusion

W<sub>e presen</sub>t<sub>e</sub>d GRNEdit<sub>, a</sub> li<sub>g</sub>ht<sub>we</sub>i<sub>g</sub>ht t<sub>wo-s</sub>t<sub>age</sub> f<sub>ramewor</sub>k th<sub>a</sub>t <sub>cas</sub>t<sub>s genera</sub>l <sub>v</sub>id<sub>eo e</sub>diti<sub>ng as</sub> <sub>source-re</sub>f<sub>erence</sub>d bi<sub>nary</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>ons,</sub> d<sub>ecoup</sub>li<sub>ng e</sub>diti<sub>ng con</sub>t<sub>ro</sub>l f<sub>rom con</sub>t<sub>en</sub>t <sub>syn</sub>th<sub>es</sub>i<sub>s.</sub> C<sub>om-</sub> <sub>pac</sub>t <sub>source-ev</sub>id<sub>ence mo</sub>d<sub>u</sub>l<sub>es gu</sub>id<sub>e</sub> t<sub>arge</sub>t<sub>-</sub>bit d<sub>ec</sub>i<sub>s</sub>i<sub>ons, w</sub>hil<sub>e</sub> id<sub>en</sub>tit<sub>y-a</sub>li<sub>gne</sub>d <sub>nu</sub>ll <sub>superv</sub>i<sub>s</sub>i<sub>on</sub> <sub>s</sub>t<sub>reng</sub>th<sub>ens con</sub>t<sub>en</sub>t <sub>preserva</sub>ti<sub>on an</sub>d <sub>prov</sub>id<sub>es an</sub> i<sub>n</sub>t<sub>r</sub>i<sub>ns</sub>i<sub>c re</sub>f<sub>erence</sub> f<sub>or res</sub>id<sub>ua</sub>l bit<sub>-marg</sub>i<sub>n re-</sub> <sub>v</sub>i<sub>s</sub>i<sub>on.</sub> With <sub>on</sub>l<sub>y</sub> 0<sub>.</sub>6M t<sub>ra</sub>i<sub>n</sub>i<sub>ng pa</sub>i<sub>rs an</sub>d <sub>con</sub>diti<sub>on</sub>i<sub>ng over</sub>h<sub>ea</sub>d b<sub>e</sub>l<sub>ow</sub> 3% <sub>o</sub>f b<sub>ac</sub>kb<sub>one s</sub>i<sub>ze,</sub> GRNEdit achieves com<sub>p</sub>etitive or leadin<sub>g</sub> results on O<sub>p</sub>enVE-Bench and ReCo-Bench<sub>,</sub> su<sub>pp</sub>ortin<sub>g</sub> bi<sub>nary ev</sub>id<sub>ence as an e</sub>fi<sub>c</sub>i<sub>en</sub>t i<sub>n</sub>d<sub>uc</sub>ti<sub>ve</sub> bi<sub>as</sub> f<sub>or</sub> di<sub>verse e</sub>dit<sub>s.</sub> GRNEdit <sub>rema</sub>i<sub>ns</sub> li<sub>m</sub>it<sub>e</sub>d i<sub>n</sub> i<sub>n</sub>t<sub>ro-</sub> ducing new semantics with little source evidence, such as object addition, and depends on binary b<sub>ac</sub>kb<sub>ones.</sub> N<sub>ever</sub>th<sub>e</sub>l<sub>ess,</sub> <sub>recen</sub>t <sub>a</sub>d<sub>vances,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> I<sub>n</sub>fi<sub>n</sub>it<sub>y</sub> <sub>an</sub>d GRN<sub>,</sub> hi<sub>g</sub>hli<sub>g</sub>ht th<sub>e</sub> <sub>grow</sub>i<sub>ng</sub> <sub>po</sub>t<sub>en</sub>ti<sub>a</sub>l <sub>o</sub>f bi<sub>nary represen</sub>t<sub>a</sub>ti<sub>ons as a compe</sub>titi<sub>ve genera</sub>ti<sub>ve para</sub>di<sub>gm.</sub>

## Acknowledgements

This work uses the O<sub>p</sub>enVE-3M<sub>,</sub> O<sub>p</sub>enVE-Bench datasets licensed under CC BY-NC 4.0. The ReCo-B<sub>enc</sub>h d<sub>a</sub>t<sub>ase</sub>t li<sub>cense</sub>d <sub>un</sub>d<sub>er</sub> CC BY<sub>-</sub>NC<sub>-</sub>SA 4<sub>.</sub>0<sub>.</sub> Th<sub>e au</sub>th<sub>ors con</sub>fi<sub>rm</sub> th<sub>a</sub>t <sub>a</sub>ll <sub>uses o</sub>f th<sub>e a</sub>b<sub>ove</sub> <sub>resources</sub> <sub>are</sub> <sub>s</sub>t<sub>r</sub>i<sub>c</sub>tl<sub>y</sub> f<sub>or</sub> <sub>aca</sub>d<sub>em</sub>i<sub>c</sub> <sub>researc</sub>h <sub>purposes</sub> <sub>an</sub>d <sub>no</sub>t f<sub>or</sub> <sub>any</sub> <sub>commerc</sub>i<sub>a</sub>l <sub>app</sub>li<sub>ca</sub>ti<sub>on.</sub>

## References

[1] J. Chen<sub>g</sub>, T. Xiao, and T. He, “Consistent video-to-video transfer usin<sub>g</sub> s<sub>y</sub>nthetic dataset,” in International Conference on Learning Representations, vol. 2024, 2024, pp. 16 867–16 879.

[2] Y. Wu, L. Chen, R. Li, S. Wan<sub>g</sub>, C. Xie, and L. Zhan<sub>g</sub>, “Insvie-1m: Efective instructionbased video editing with elaborate dataset construction,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 16 692–16 701.

[3] Q. Bai, Q. Wan<sub>g</sub>, H. Ou<sub>y</sub>an<sub>g</sub>, Y. Yu, H. Wan<sub>g</sub>, W. Wan<sub>g</sub>, K. L. Chen<sub>g</sub>, S. Ma, Y. Zen<sub>g</sub>, Z. Liu et al., “Scaling instruction-based video editing with a high-quality synthetic dataset,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, <sub>pp</sub>. 37 971–37 981.

[4] H. He, J. Wan<sub>g</sub>, J. Zhan<sub>g</sub>, Z. Xue, X. Bu, Q. Yan<sub>g</sub>, S. Wen, and L. Xie, “O<sub>p</sub>enve-3m: A lar<sub>g</sub>e-scale high-quality dataset for instruction-guided video editing,” arXiv preprint arXiv:2512.07826, 2025.

[5] Y. Chen, J. Zhan<sub>g</sub>, T. Hu, Y. Zen<sub>g</sub>, Z. Xue, Q. He, C. Wan<sub>g</sub>, Y. Liu, X. Hu, and S. Yan, “Ivebench: Modern benchmark suite for instruction-guided video editing assessment,” arXiv preprint arXiv:2510.11647, 2025.

[6] X. Con<sub>g</sub>, H. Yan<sub>g</sub>, A. Wan<sub>g</sub>, Y. Wan<sub>g</sub>, Y. Yan<sub>g</sub>, C. Zhan<sub>g</sub>, and C. Ma, “Viva: Vlm-<sub>g</sub>uided instruction-based video editing with reward optimization,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 34 364–34 374.

[7] S. Lian<sub>g</sub>, F. Guan, Y. Zhan<sub>g</sub>, X. Li, and Z. Chen, “Cot-edit: Let cot <sub>g</sub>uide instruction video editing,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026<sub>, pp</sub>. 37 960–37 970.

[8] L. Zhan<sub>g</sub>, A. Rao, and M. A<sub>g</sub>rawala, “Addin<sub>g</sub> conditional control to text-to-ima<sub>g</sub>e difusion models,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, <sub>pp</sub>. 3836–3847.

[9] Z. Jian<sub>g</sub>, Z. Han, C. Mao, J. Zhan<sub>g</sub>, Y. Pan, and Y. Liu, “Vace: All-in-one video creation and editing,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, <sub>pp</sub>. 17 191–17 202.

[10] Y. Wan<sub>g</sub>, L. Wan<sub>g</sub>, Z. Ma, Q. Hu, K. Xu, and Y. Guo, “Videodirector: Precise video editin<sub>g</sub> via text-to-video models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 2589–2598.

[11] J. Mai, C. Wan<sub>g</sub>, G. G. Qian, W. Mena<sub>p</sub>ace, S. Tul<sub>y</sub>akov, B. Ghanem, P. Wonka, and A. Mirzaei, “Easyv2v: A high-quality instruction-based video editing framework,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 30 435–30 445.

[12] J. Han, J. Liu, J. Wan<sub>g</sub>, B. Pen<sub>g</sub>, and Z. Yuan, “Generative refinement networks for visual synthesis,” arXiv preprint arXiv:2604.13030, 2026.

[13] J. Ho and T. Salimans, “Classifier-free difusion guidance,” arXiv preprint arXiv:2207.12598, 2022.

[14] X. Ge, Y. Zhan<sub>g</sub>, Y. Huan<sub>g</sub>, D. He, X. Wan<sub>g</sub>, B. Ma, G. Son<sub>g</sub>, Y. Liu, and J. Zhan<sub>g</sub>, “Salt: S<sub>e</sub>lf<sub>-cons</sub>i<sub>s</sub>t<sub>en</sub>t di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on ma</sub>t<sub>c</sub>hi<sub>ng w</sub>ith <sub>cac</sub>h<sub>e-aware</sub> t<sub>ra</sub>i<sub>n</sub>i<sub>ng</sub> f<sub>or</sub> f<sub>as</sub>t <sub>v</sub>id<sub>eo genera</sub>ti<sub>on,</sub>” 2026. [Online]. Available: htt<sub>p</sub>s://arxiv.or<sub>g</sub>/abs/2604.03118

[15] Y. Guo, C. Yan<sub>g</sub>, H. He, Y. Zhao, M. Wei, Z. Yan<sub>g</sub>, W. Huan<sub>g</sub>, and D. Lin, “End-to-end trainin<sub>g</sub> for autoregressive video difusion via self-resampling,” arXiv preprint arXiv:2512.15702, 2025.

[16] L. Zhan<sub>g</sub>, S. Mo, Z. Cai, J. Lin, Z. Lin, J. Gu, K. K. Sin<sub>g</sub>h, Y. Li, and Y. Li, “Unitem<sub>p</sub>: Unlocking video generation in any temporal order via bidirectional distillation,” arXiv preprint arXiv:2606.18702, 2026.

[17] J. Gu, Y. Shen, T. Chen, L. Dinh, Y. Wan<sub>g</sub>, M. A. Bautista, D. Berthelot, J. Susskind, and S<sub>.</sub> Zh<sub>a</sub>i<sub>,</sub> “St<sub>ar</sub>fl<sub>ow-v:</sub> E<sub>n</sub>d<sub>-</sub>t<sub>o-en</sub>d <sub>v</sub>id<sub>eo</sub> <sub>genera</sub>ti<sub>ve</sub> <sub>mo</sub>d<sub>e</sub>li<sub>ng</sub> <sub>w</sub>ith <sub>au</sub>t<sub>oregress</sub>i<sub>ve</sub> <sub>norma</sub>li<sub>z</sub>i<sub>ng</sub> flows,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2026, pp. 9084–9094.

[18] J. Ho, A. Jain, and P. Abbeel, “Denoising difusion probabilistic models,” Advances in neural information processing systems, vol. 33, pp. 6840–6851, 2020.

[19] Y. Li<sub>p</sub>man, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow matchin<sub>g</sub> for <sub>g</sub>enerative modeling,” arXiv preprint arXiv:2210.02747, 2022.

[20] W. Kong, Q. Tian, Z. Zhang, R. Min, Z. Dai, J. Zhou, J. Xiong, X. Li, B. Wu, J. Zhang et al., “Hunyuanvideo: A systematic framework for large video generative models,” arXiv preprint arXiv:2412.03603, 2024.

[21] A. Wang, B. Ai, B. Wen, C. Mao, C.-W. Xie, D. Chen, F. Yu, H. Zhao, J. Yang, J. Zeng et al., “Wan: Open and advanced large-scale video generative models,” arXiv preprint arXiv:2503.20314, vol. 3<sub>,</sub> no. 4<sub>, p</sub>. 6<sub>,</sub> 2025.

[22] D. Kondrat<sub>y</sub>uk, L. Yu, X. Gu, J. Lezama, J. Huan<sub>g</sub>, G. Schindler, R. Hornun<sub>g</sub>, V. Birodkar, J. Yan, M.-C. Chiu et al., “Videopoet: A large language model for zero-shot video generation,” arXiv preprint arXiv:2312.14125, 2023.

[23] K. Tian, Y. Jian<sub>g</sub>, Z. Yuan, B. Pen<sub>g</sub>, and L. Wan<sub>g</sub>, “Visual autore<sub>g</sub>ressive modelin<sub>g</sub>: Scalable image generation via next-scale prediction,” Advances in neural information processing systems, vol. 37<sub>,</sub> <sub>pp</sub>. 84 839–84 865<sub>,</sub> 2024.

[24] J. Han, J. Liu, Y. Jian<sub>g</sub>, B. Yan, Y. Zhan<sub>g</sub>, Z. Yuan, B. Pen<sub>g</sub>, and X. Liu, “Infinit<sub>y</sub>: Scalin<sub>g</sub> bitwise autoregressive modeling for high-resolution image synthesis,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 15 733–15 744.

[25] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “Hi<sub>g</sub>h-resolution ima<sub>g</sub>e synthesis with latent difusion models,” in Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 10 684–10 695.

[26] A. Van Den Oord, O. Vinyals et al., “Neural discrete representation learning,” Advances in neural information processing systems, vol. 30, 2017.

[27] J. Liu, J. Han, B. Yan, H. Wu, F. Zhu, X. Wan<sub>g</sub>, Y. Jian<sub>g</sub>, B. Pen<sub>g</sub>, and Z. Yuan, “Infinit<sub>y</sub>star: Unified spacetime autoregressive modeling for visual generation,” Advances in Neural Information Processing Systems, vol. 38, pp. 170 054–170 072, 2026.

[28] C. Zhan<sub>g</sub>, C. Fen<sub>g</sub>, F. Yan, Q. Zhan<sub>g</sub>, M. Zhan<sub>g</sub>, Y. Zhon<sub>g</sub>, J. Zhan<sub>g</sub>, and L. Ma, “Instructvedit: A holistic approach for instructional video editing,” arXiv preprint arXiv:2503.17641, 2025.

[29] S. Yu, D. Liu, Z. Ma, Y. Hon<sub>g</sub>, Y. Zhou, H. Tan, J. Chai, and M. Bansal, “Ve<sub>gg</sub>ie: Instructional editing and reasoning video concepts with grounded generation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 15 147–15 158.

[30] T. Brooks, A. Hol<sub>y</sub>nski, and A. A. Efros, “Instruct<sub>p</sub>ix2<sub>p</sub>ix: Learnin<sub>g</sub> to follow ima<sub>g</sub>e editin<sub>g</sub> instructions,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 18 392–18 402.

[31] Z. Tan, H. Yan<sub>g</sub>, L. Qin, J. Gon<sub>g</sub>, M. Yan<sub>g</sub>, and H. Li, “Omni-video: Democratizin<sub>g</sub> unified video understanding and generation,” arXiv preprint arXiv:2507.06119, 2025.

[32] X. Liao, X. Zen<sub>g</sub>, Z. Son<sub>g</sub>, Z. Fu, G. Yu, and G. Lin, “In-context learnin<sub>g</sub> with un<sub>p</sub>aired cli<sub>p</sub>s for instruction-based video editing,” arXiv preprint arXiv:2510.14648, 2025.

[33] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen et al., “Lora: Lowrank adaptation of large language models. arxiv 2021,” arXiv preprint arXiv:2106.09685, vol. 10<sub>,</sub> 2024.

[34] Q. Zhan<sub>g</sub>, M. Chen, A. Bukharin, N. Karam<sub>p</sub>atziakis, P. He, Y. Chen<sub>g</sub>, W. Chen, and T. Zhao, “Adalora: Adaptive budget allocation for parameter-eficient fine-tuning,” arXiv preprint arXiv:2303.10512, 2023.

[35] Z. Zhan<sub>g</sub>, F. Lon<sub>g</sub>, W. Li, Z. Qiu, W. Liu, T. Yao, and T. Mei, “Re<sub>g</sub>ion-constraint in-context <sub>g</sub>eneration for instructional video editin<sub>g</sub>,” 2025. [Online]. Available: htt<sub>p</sub>s://arxiv.or<sub>g</sub>/abs/2512.17650

[36] Runwa<sub>y</sub>, “Runwa<sub>y</sub> ale<sub>p</sub>h: A new wa<sub>y</sub> to edit, transform and <sub>g</sub>enerate video,” 2025. [Online]. Availabl<sub>e:</sub> htt<sub>ps:</sub>//r<sub>u</sub>nwa<sub>y</sub>ml<sub>.</sub>c<sub>o</sub>m/r<sub>ese</sub>arch/intr<sub>o</sub>d<sub>u</sub>cin<sub>g</sub>-r<sub>u</sub>nwa<sub>y</sub>-al<sub>ep</sub>h

[37] Decart AI Team, “Luc<sub>y</sub> edit: O<sub>p</sub>en-wei<sub>g</sub>ht text-<sub>g</sub>uided video editin<sub>g</sub>,” Decart AI, Technical Re<sub>p</sub>ort, 2025. [Online]. Available: htt<sub>p</sub>s://d2drj<sub>p</sub>uinn46lb.cloudfront.net/Luc<sub>y\_</sub>Edit<sub>\_\_</sub>Hi<sub>g</sub> h<sub>\_</sub>Fid<sub>e</sub>lit<sub>y\_</sub>T<sub>e</sub>xt<sub>\_</sub>G<sub>u</sub>id<sub>e</sub>d<sub>\_</sub>Vid<sub>eo\_</sub>Editin<sub>g.p</sub>df

[38] Y. Lin, G. Lian<sub>g</sub>, Z. Zen<sub>g</sub>, Z. Bai, Y. Chen, and M. Z. Shou, “Kiwi-edit: Versatile video editin<sub>g</sub> via instruction and reference <sub>g</sub>uidance,” 2026. [Online]. Available: htt<sub>p</sub>s://arxiv.or<sub>g</sub>/abs/2603.02175

[39] C. Wei, Q. Liu, Z. Ye, Q. Wan<sub>g</sub>, X. Wan<sub>g</sub>, P. Wan, K. Gai, and W. Chen, “Univideo: Unified understanding, generation, and editing for videos,” in The Fourteenth International Conference on Learning Representations, 2026. [Online]. Available: htt<sub>p</sub>s://o<sub>p</sub>enreview.net/forum?id=EDCJTaR9bk

[40] F. Fu, M. Huan<sub>g</sub>, S. Wu, Y. Jian<sub>g</sub>, Y. Huo, H. Li, Y. Son<sub>g</sub>, F. Din<sub>g</sub>, J. Guo, Q. He, Z. Fu, Z<sub>.</sub> M<sub>ao, an</sub>d Y<sub>.</sub> Zh<sub>ang,</sub> “L<sub>ance:</sub> U<sub>n</sub>ifi<sub>e</sub>d <sub>mu</sub>lti<sub>mo</sub>d<sub>a</sub>l <sub>mo</sub>d<sub>e</sub>li<sub>ng</sub> b<sub>y mu</sub>lti<sub>-</sub>t<sub>as</sub>k <sub>synergy,</sub>” 2026<sub>.</sub> [Online]. Available: htt<sub>p</sub>s://arxiv.or<sub>g</sub>/abs/2605.18678

[41] J. Wu, H. Lian, J. Yan<sub>g</sub>, D. Hao, Y. Tian, Y. Ton<sub>g</sub>, J. Zhu, B. Chen, Q. Qi, A. Zhan<sub>g</sub>, W. He, M<sub>.</sub> Li<sub>u,</sub> J<sub>.</sub> Li<sub>u,</sub> P<sub>.</sub> H<sub>uang, an</sub>d H<sub>.</sub> Ji<sub>ang,</sub> “L<sub>oomv</sub>id<sub>eo:</sub> U<sub>n</sub>if<sub>y</sub>i<sub>ng mu</sub>lti<sub>mo</sub>d<sub>a</sub>l i<sub>npu</sub>t<sub>s</sub> i<sub>n</sub>t<sub>o v</sub>id<sub>eo</sub> <sub>g</sub>eneration and editin<sub>g,</sub>” 2026. [Online]. Available: htt<sub>p</sub>s://arxiv.or<sub>g</sub>/abs/2606.06042

## APPENDIX

GRNEdit: Efficient General Video Editing from a New Binary Evidence Perspective

A1 REPRODUCIBILITY & TRAINING   
A1.1 Reproducibility Details: Model Architectures and Training Configurations   
A1.2 Data Preparation, Text Conditioning, and MLLM Reprompting Rules   
A1.3 GRN Training Curves   
A2 BINARY EVIDENCE ANALYSIS   
A2.1 Why Call It Evidence? Ordered Support for Binary Decisions   
A2.2 Attention Comparison: Evidence-Guided GRNEdit vs. Wan 2.1   
A2.3 How Binary Evidence Guides Progressive Editing   
A3 ADDITIONAL RESULTS & LIMITATIONS   
A3.1 More Visualization Results across Different Tasks   
A3.2 Limitations and Failure Cases   
A4 USER STUDY   
A4.1 User Study: Human Preference across Six Evaluation Dimensions

## A1.1 Reproducibility Details: Model Architectures and Training Configurations

<table><tr><td rowspan=1 colspan=3>Model</td><td rowspan=1 colspan=3>Architecture</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>GRNEdit-2B</td><td rowspan=1 colspan=2>GRNEdit-8B</td><td rowspan=1 colspan=1>GRNEdit-8B</td></tr><tr><td rowspan=1 colspan=1>Stage I registry</td><td rowspan=1 colspan=2>GRN2bOfficialEditStage15</td><td rowspan=1 colspan=2>GRN8bOfficialEditStage15</td><td rowspan=1 colspan=1>GRN8bOfficialEditStage15</td></tr><tr><td rowspan=1 colspan=1>Stage Il registry</td><td rowspan=1 colspan=2>GRN2bOfficialEditStage2</td><td rowspan=1 colspan=2>GRN8bOfficialEditStage2</td><td rowspan=1 colspan=1>GRN8bOfficialEditStage2</td></tr><tr><td rowspan=1 colspan=1>Transformer depth</td><td rowspan=1 colspan=2>28</td><td rowspan=1 colspan=2>36</td><td rowspan=1 colspan=1>36</td></tr><tr><td rowspan=1 colspan=1>Transformer chunks</td><td rowspan=1 colspan=2>7× 4 blocks</td><td rowspan=1 colspan=2>6×6 blocks</td><td rowspan=1 colspan=1>6×6 blocks</td></tr><tr><td rowspan=1 colspan=1>Hidden dimension</td><td rowspan=1 colspan=2>2304</td><td rowspan=1 colspan=2>4096</td><td rowspan=1 colspan=1>4096</td></tr><tr><td rowspan=1 colspan=1>Attention heads</td><td rowspan=1 colspan=2>18</td><td rowspan=1 colspan=2>32</td><td rowspan=1 colspan=1>32</td></tr><tr><td rowspan=1 colspan=1>Drop path</td><td rowspan=1 colspan=2>0</td><td rowspan=1 colspan=2>0</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>Norm epsilon</td><td rowspan=1 colspan=2>1(10^{-6}1)</td><td rowspan=1 colspan=2>1(10^{-6}1)</td><td rowspan=1 colspan=1>\(10^{-6}1)</td></tr><tr><td rowspan=1 colspan=1>Source injection mask</td><td rowspan=1 colspan=2>[1,0,0,0,0,0,1]</td><td rowspan=1 colspan=2>[1,0,0,0,0,1]</td><td rowspan=1 colspan=1>[1,0,0,0,0,0]</td></tr><tr><td rowspan=1 colspan=6>Exact additional parameter counts</td></tr><tr><td rowspan=1 colspan=2>Component</td><td rowspan=1 colspan=2>2B</td><td rowspan=1 colspan=2>8B</td></tr><tr><td rowspan=1 colspan=2>Stage I source modules</td><td rowspan=1 colspan=2>24.855M</td><td rowspan=1 colspan=2>37.873M</td></tr><tr><td rowspan=1 colspan=2>Stage II Bit-Margin Router</td><td rowspan=1 colspan=2>37.758M</td><td rowspan=1 colspan=2>118.506M</td></tr><tr><td rowspan=1 colspan=2>Total additional parameters</td><td rowspan=1 colspan=2>62.614M</td><td rowspan=1 colspan=2>156.379M</td></tr><tr><td rowspan=1 colspan=2>Percentage relative to the main trunk</td><td rowspan=1 colspan=2>~3.1%</td><td rowspan=1 colspan=2>~2.0%</td></tr></table>

<table><tr><td rowspan=1 colspan=6>Optimization</td></tr><tr><td rowspan=1 colspan=1>Setting</td><td rowspan=1 colspan=1>2B</td><td rowspan=1 colspan=1>8B</td><td rowspan=1 colspan=1>Setting</td><td rowspan=1 colspan=1>2B</td><td rowspan=1 colspan=1>8B</td></tr><tr><td rowspan=1 colspan=1>Optimizer</td><td rowspan=1 colspan=1>AdamW</td><td rowspan=1 colspan=1>AdamW</td><td rowspan=1 colspan=1>Betas</td><td rowspan=1 colspan=1>(0.9, 0.999)</td><td rowspan=1 colspan=1>(0.9, 0.999)</td></tr><tr><td rowspan=1 colspan=1>Peak LR</td><td rowspan=1 colspan=1>\(4\times10^{-5}\)</td><td rowspan=1 colspan=1>\(4\times10^{-5}\)</td><td rowspan=1 colspan=1>Minimum LR</td><td rowspan=1 colspan=1>\(10^{-5}\)</td><td rowspan=1 colspan=1>\(10^{-5}\)</td></tr><tr><td rowspan=1 colspan=1>Warmup</td><td rowspan=1 colspan=1>100 steps from \(10^{-6}\)</td><td rowspan=1 colspan=1>100 steps from \(10^{-6}\)</td><td rowspan=1 colspan=1>LR decay</td><td rowspan=1 colspan=1>×0.5/5K</td><td rowspan=1 colspan=1>×0.5/4K</td></tr><tr><td rowspan=1 colspan=1>Main zero-LR steps</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>Gradient accumulation</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>Token budget / GPU</td><td rowspan=1 colspan=1>110K</td><td rowspan=1 colspan=1>50K</td><td rowspan=1 colspan=1>Identity ratio</td><td rowspan=1 colspan=1>5%</td><td rowspan=1 colspan=1>5%</td></tr><tr><td rowspan=1 colspan=1>Weight decay</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>Gradient clipping</td><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>0.1</td></tr><tr><td rowspan=1 colspan=1>Precision</td><td rowspan=1 colspan=1>BF16</td><td rowspan=1 colspan=1>BF16</td><td rowspan=1 colspan=1>EMA decay</td><td rowspan=1 colspan=1>0.999</td><td rowspan=1 colspan=1>0.999</td></tr><tr><td rowspan=1 colspan=1>Training updates</td><td rowspan=1 colspan=1>60K</td><td rowspan=1 colspan=1>60K</td><td rowspan=1 colspan=1>Save interval</td><td rowspan=1 colspan=1>1K</td><td rowspan=1 colspan=1>1K</td></tr></table>

## A1.2 Data Preparation, Text Conditioning, and MLLM Reprompting Rules

<table><tr><td rowspan=1 colspan=4>Data Processing</td></tr><tr><td rowspan=1 colspan=1>Setting</td><td rowspan=1 colspan=1>Value</td><td rowspan=1 colspan=1>Setting</td><td rowspan=1 colspan=1>Value</td></tr><tr><td rowspan=1 colspan=1>Training data</td><td rowspan=1 colspan=1>0.6M balanced pairs</td><td rowspan=1 colspan=1>Source dataset</td><td rowspan=1 colspan=1>OpenVE-3M</td></tr><tr><td rowspan=1 colspan=1>Pixel budget</td><td rowspan=1 colspan=1>0.41M</td><td rowspan=1 colspan=1>Aspect ratio</td><td rowspan=1 colspan=1>Dynamic</td></tr><tr><td rowspan=1 colspan=1>Frames / FPS</td><td rowspan=1 colspan=1>61 / 20</td><td rowspan=1 colspan=1>Duration / bin</td><td rowspan=1 colspan=1>3.05 s / 0.25 s</td></tr></table>

<table><tr><td rowspan=1 colspan=4>Text Conditioning</td></tr><tr><td rowspan=1 colspan=1>Setting</td><td rowspan=1 colspan=1>Value</td><td rowspan=1 colspan=1>Setting</td><td rowspan=1 colspan=1>Value</td></tr><tr><td rowspan=1 colspan=1>Encoder</td><td rowspan=1 colspan=1>Frozen UMT5-XXL</td><td rowspan=1 colspan=1>Feature dimension</td><td rowspan=1 colspan=1>4096</td></tr><tr><td rowspan=1 colspan=1>Token limit</td><td rowspan=1 colspan=1>512</td><td rowspan=1 colspan=1>Prompt prefix</td><td rowspan=1 colspan=1>&lt;T2V&gt;</td></tr><tr><td rowspan=1 colspan=1>Backbone condition</td><td rowspan=1 colspan=1>Edit + zero separator + reprompt</td><td rowspan=1 colspan=1>Evidence condition</td><td rowspan=1 colspan=1>Edit only</td></tr><tr><td rowspan=1 colspan=1>Separator</td><td rowspan=1 colspan=1>One zero vector</td><td rowspan=1 colspan=1>Reprompt encoding</td><td rowspan=1 colspan=1>One batched T5 pass</td></tr><tr><td rowspan=1 colspan=1>Null instruction</td><td rowspan=1 colspan=1>Empty string</td><td rowspan=1 colspan=1>Reprompt limit</td><td rowspan=1 colspan=1>50 words</td></tr><tr><td rowspan=1 colspan=1>VLM input</td><td rowspan=1 colspan=1>Source frames + instruction</td><td rowspan=1 colspan=1>Sampled frames</td><td rowspan=1 colspan=1>2</td></tr></table>

<table><tr><td colspan="2">MLLM Reprompting Rules (Randomly Selecting Qwen3.5-Flash or Qwen3-VL-Flash)</td></tr><tr><td>System Prompt</td><td>You are a video descriptor for target-video training prompts. The user provides sampled frames and a visual instruction; infer the intended final video and describe only that plausible result. Never mention editing-related actions or any before/after change process; write it as a normal video description.</td></tr><tr><td>User Prompt</td><td>Rules: - Apply only the rules relevant to the visual instruction. - Use final-state nouns and adjectives, not process words like transformed, replaced, instead of, original, unchanged, absence, or once stood. - Background: describe the main subject and final background. - Style: describe all visible content in the requested style, with natural actions and camera motion. - Add: describe the added object, position, scale, interaction, motion, shadow, or reflection. - Local change: describe the final object or region directly - Remove: describe the natural scene with the target absent; never name the absent object. - Creative: describe a coherent final scene with materials, motion, and environment. - Camera: describe final framing, viewpoint, and visible content only.</td></tr></table>

![](images/933c456e59b6abb80cc605f8f28eae5e87f9f5e60cd90541b484117a3bf5a7f2.jpg)

GRNedited-8B  
![](images/86364f5ae026b401641fcdbfe66538e717f9c5734fe75761405404b131826d13.jpg)

GRN-2B-vace  
![](images/e7432a739d2a52b92eafb3c415a20770f97e70fbbaf64ba7514d03afc6b5de23.jpg)

## A1.3 GRN Training Curves

\- GRNEdit-2B: 40K training steps for the main-paper results. - GRNEdit-8B: 80K training steps for the main-paper results. - GRN-2B-VACE: 20K training steps, used only for the earlystage convergence-efficiency comparison.

![](images/a96cad05169fd58a0657703ec4643ff994dabb25f18c69d238065f3fef0a8318.jpg)

## A2.1 Why Call It Evidence? Ordered Support for Binary Decisions

## Ordered semantic axis

Source-aligned margins share a comparable retain-or-flip meaning across bits.

## 2 Measurable response

Signed margin change gives direction; RMS magnitude reveals spatial strength.

## 3 Direct binary control

Support acts on the backbone's native decisions rather than transient routing weights.

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>ADD</td><td rowspan=1 colspan=1>REMOVE</td><td rowspan=1 colspan=1>BACKGROUND</td><td rowspan=1 colspan=1>STYLE</td><td rowspan=1 colspan=1>LOCAL</td></tr><tr><td rowspan=1 colspan=1>Prompt</td><td rowspan=1 colspan=1>Add a woman with a tanpuffer jacket, wearing bluejeans.</td><td rowspan=1 colspan=1>Completely remove the mansitting on the right side ofthe table.</td><td rowspan=1 colspan=1>Replace the background withdynamic abandoned warehousescene at dusk.</td><td rowspan=1 colspan=1>Make it the style of NeonLight Art.</td><td rowspan=1 colspan=1>Change the pink headband tosleek, metallic silver band</td></tr><tr><td rowspan=1 colspan=1>GRNSource</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>EvidenceStrength</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>西</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>EvidenceDirection</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>西</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>GRN Edit</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr></table>

![](images/166fb699fec35cda746ddcd05220575bef52329030728039b0bba24d9b173286.jpg)

## A2.2 Attention Comparison: Evidence-Guided GRNEdit vs. Wan 2.1

## Different semantics

Attention routes values; evidence acts directly on ordered binary margins.

## 2 Sharper task separation

GRNEdit attention preserves source structure while isolating regions that require synthesis.

## 3 Complementary interaction

Evidence defines what to preserve or revise;   
attention resolves how it fits the generation.

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>ADD</td><td rowspan=1 colspan=1>REMOVE</td><td rowspan=1 colspan=1>BACKGROUND</td><td rowspan=1 colspan=1>STYLE</td><td rowspan=1 colspan=1>LOCAL</td></tr><tr><td rowspan=1 colspan=1>Prompt</td><td rowspan=1 colspan=1>Add a woman with a tanpuffer jacket, wearing bluejeans.</td><td rowspan=1 colspan=1>Completely remove the mansitting on the right side ofthe table.</td><td rowspan=1 colspan=1>Replace the background withdynamic abandoned warehousescene at dusk.</td><td rowspan=1 colspan=1>Make it the style of NeonLight Art.</td><td rowspan=1 colspan=1>Change the pink headband tosleek, metallic silver band</td></tr><tr><td rowspan=1 colspan=1>GRNSource</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>GRNAttentionscore</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>西</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>K</td></tr><tr><td rowspan=1 colspan=1>Wan 2.1Attentionscore</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>GRN Edit</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr></table>

## A2.3 How Binary Evidence Guides Progressive Editing

Early source evidence establishes the editable scope; global refinement then resolves target semantics and visual detail.

![](images/e521c5841ce8908dc21c45d69075a2540fdd7155b6c7f886e1078e6b58cf2832.jpg)  
Editing intent appears early; intermediate iterations resolve semantic structure, while later iterations stabilize identity, geometry, and texture.  
RMS Δ-margin: bright = stronger influence · dark = released edit region

## A3.1 More Visualization Results across Different Tasks

![](images/6e48ece41db64473f02b4278d804da477933393733838d260710962f000c7402.jpg)

Due to data quality concerns, subtitle-related editing samples are excluded from training. Surprisingly, the model naturally generalizes from object removal tasks to text removal, despite never being trained on explicit subtitle removal examples.  
![](images/f9fd79cfd3340165c535f642be110b273043ee52696085678a3f5bbbe2f8e981.jpg)

## A3.2 Limitations and Failure Cases

## ADD failures reveal two bottlenecks: source-absent semantics and imperfect insertion supervision.

![](images/0c2805cea93dbbd7647363d8a656ce54e26c0a3d8edcdbe902a11ca5b45cede4.jpg)  
Weak object synthesis  
Add an animated flying seagull near the upperright sky.

![](images/cee696cfc0107257a43fc3fb40c882069d42312d4c08ab690e6bad678707c781.jpg)

## Pasted-on integration

Add a white ceramic mug on the desk near the keyboard.

![](images/e90d8cf05e22b74047e2367297d32e7fa9f0203437b39108d6140ed1699dc9ad.jpg)

## Physical inconsistency

Add a small framed picture to the blue wall behind the woman.

SOURCE  
![](images/f35138b5d65d1ff205c3f677e698e4710f8070fb27bb4ee538db0841fc0e597a.jpg)

![](images/35775b7e712883824b5bf2399b7d27aa8b31d3b143278c2be2fd845c960aaa5d.jpg)

SOURCE  
![](images/e9113674bd3d2d0040fd114f0cffed53b9f855610e6566a46cd5d3d5c9cd4331.jpg)

![](images/0cddf8d6ef41ba1f7f2ef86c2ba2e4b2300bae57dfd2f550920f0f4eba9f90ab.jpg)

SOURCE  
![](images/4a8944f8519a28c53a7d3ec942a40594245740aa0611e3179fff15e0a36f6262.jpg)

![](images/a3062182af6ad17503faf33ca6b51a02e897bf3a28ee35f8defc48df689c895d.jpg)

OURS  
![](images/ae3ae4e6da3d548a1a61c3de7d46477b90345addeac37d96640451f87a469f07.jpg)  
METHOD-SIDE

![](images/8240657aa1a5079fd979dbfad6c45b7a7132d7bc799f0d72666b3a3ffaeb07a6.jpg)  
OURS

![](images/205827b2fa65cfe96478dda3867a6398442ead1b2166877f242374698c14c485.jpg)  
DATA-SIDE

![](images/8c4197cfed4a472d04f0067aaa340d01c1f7569866e42bd3053cb420ec2226cf.jpg)  
OURS

![](images/ee1c5481d7ffbeebb4d75ab7172fac877d5107bde4c974c6d0d6b855129ed6a2.jpg)  
DATA-SIDE

![](images/c5b1f4b0f3894f492adb6ea1f03702ff026dd2e0d966df042c673bf558c4b69a.jpg)  
The location is correct, but the new entity remains visibly under-rendered.  
The edit follows the prompt, yet the mug remains visually detached from the desk.  
Placement and appearance do not fully respect wall geometry or scene interaction.

## Failure attribution

## METHOD-SIDE LIMITATION

Source-absent semantics create a preservation-generation conflict that a lightweight evidence path cannot fully resolve.

## DATA-SIDE LIMITATION

Overlay-like targets provide weak supervision for geometry, contact, occlusion, lighting, shadows, and reflections.

## A3.3 Analysis of Limitations and Future Work

## Evidence-related Limitations

ADD is structurally harder because the target semantics are absent from the source.

![](images/8328f6e2e98a42de38343b107618dad5da67ac9a775bb7ed212d7a57dddf3504.jpg)

## Preservation–generation conflict

Evidence decoupling is efficient for source-supported edits, but the lightweight branch has limited capacity to mediate retaining the full source while introducing a new, scene-consistent entity.

![](images/4bfc966515f8658e1d644332b63fcc37e8b3c1592c0110755be468bd92314210.jpg)

## Backbone prior drift

Full-parameter fine-tuning on 0.6M pairs can perturb pretrained synthesis priors, further weakening novel-object generation.

Why background change is easier • Preserve the foreground boundary, then generate a largely independent background with limited object–scene interaction.

## Dataset-related Limitations

![](images/28ff6fceceea853bced247375e11e8845382ee59265ee0064f421f9c570ec95f.jpg)

![](images/27514cd705c25bb2fb2793c45d33cb0613c6cd6664a2fcc7fff4696944f2f507.jpg)

Some OpenVE Local Add targets supervise compositing rather than physical insertion.

## Pasted-on supervision

Overlay-like targets provide weak supervision for integrating new objects naturally into the source scene.

## Missing physical cues

Scale, depth, contact, occlusion, folds, illumination, shadows, and reflections can be inconsistent with the scene.

Consequence • The model can reproduce target artifacts even when it follows the instruction.

## Mitigation Strategies and Future Work

![](images/2a8aeb80b996a0b906e896635b42c2ce3bf8b799de3b5750dd376b12e171827e.jpg)

## Task-adaptive evidence

Allocate richer semantic capacity to ADD and learn region-aware gating: preserve source evidence outside the edit area while relaxing it where new content must appear.

![](images/bf9c5bfe67543f47e6cb8a2e9cd192003f05c3cd417dfc3cec7a403091110e39.jpg)

## Prior-preserving tuning

Freeze most of the backbone or use LoRA / partial tuning; adopt staged training to reduce drift of pretrained generative priors.

![](images/6ef26e6f9293c7148aae2b449af0c1d2db29bc7fc4ba04d9ba671e78d2163e94.jpg)

## Higher-quality ADD data

Mix OpenVE with ReCo-Data and Ditto-1M; reweight ADD and filter for temporal consistency, placement, occlusion, contact, lighting, and shadows.

## A4.1 User Study: Human Preference across Six Evaluation Dimensions

Ours is preferred most for faithful editing, artifact suppression, and source preservation.

Normalized preference share

% within each evaluation dimension

![](images/de5ed3c5d61f23fc4d5fd6eb9df89fd0397ecb04573e0f03e16253ff60720506.jpg)  
OVERALL PREFERENCE  
Mean share across six dimensions

![](images/0f499d5b03976aa3271cbaa2e707bd781610d32d8c72650ad72612d4cc2d77ed.jpg)

## DIMENSION RANKING · 1 → 4

Visual Quality UniVideo > Ours = Lance > Kiw

Edit Success Ours > UniVideo > Kiwi = Lance

● Temporal Coherence UniVideo > Ours = Lance > Kiwi

Motion Dynamics Lance > UniVideo > Ours = Kiw

Distortion-Free Ours > Kiwi = UniVideo > Lance

Source Consistency Ours > Kiwi > UniVideo > Lance