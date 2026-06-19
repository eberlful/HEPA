## Abstract

Abstract Critical events in multivariate time series, from turbine failures to cardiac arrhythmias, demand accurate prediction, yet labeled data is scarce because such events are rare and costly to annotate. We introduce HEPA (Horizon-conditioned Event Predictive Architecture), built on two key principles. First, a causal Transformer encoder is pretrained via a Joint-Embedding Predictive Architecture (JEPA): a horizon-conditioned predictor learns to forecast future representations rather than future values, forcing the encoder to capture predictable temporal dynamics from unlabeled data alone. Second, we freeze the encoder and finetune only the predictor toward the target event, producing a monotonic survival cumulative distribution function (CDF) over horizons. With fixed architecture and optimiser hyperparameters across all benchmarks, HEPA handles water contamination, cyberattack detection, volatility regimes, and eight further event types across 11 domains, exceeding leading time-series architectures including PatchTST, iTransformer, MAE, and Chronos-2 on at least 10 of 14 benchmarks, with an order of magnitude fewer tuned parameters and, on lifecycle datasets, an order of magnitude less labeled data.

## 1 Introduction

Figure: Figure 1: One label-efficient architecture, domain- and event-agnostic. (a) h-AUROC ($\uparrow$; horizon-averaged AUROC) across 14 benchmarks in 11 domains. HEPA wins on 10 out of 14 at full labels; at 10% labels (open circles) it retains $\geq$92% of full-label performance on lifecycle datasets. (b) Predicted probability surfaces $p(t,\Delta t)$ for turbofan degradation (top) and cardiac arrhythmia (bottom).
Refer to caption: https://arxiv.org/html/2605.11130/2605.11130v4/x1.png

A turbine blade cracks after 12,000 flight hours. A bearing degrades over weeks of vibration data. A satellite sensor drifts silently for 48 hours before triggering a cascade. These events are rare in operational data, yet they follow partially predictable precursor dynamics : temperatures rise gradually before overheating, vibration amplitudes grow before mechanical failure, and sensor readings deviate systematically before spacecraft faults. A range of machine-learning methods attempt to predict such events from multivariate sensor streams. Remaining-useful-life (RUL) models  estimate how long until a machine fails; anomaly detectors  flag when sensor readings look abnormal. Although general-purpose architectures exist for both, the two communities develop separate benchmarks, metrics, and evaluation protocols: RUL models never see anomaly benchmarks; anomaly detectors never forecast time-to-failure. Yet all these tasks share the same structure: given observations up to time $t$, estimate the probability $P(\text{event within }\Delta t)$ for each prediction horizon $\Delta t$.

This structural uniformity suggests a separation of concerns. The *encoder* learns temporal dynamics from unlabeled data without knowing which event matters downstream. The *predictor*, finetuned with a small number of event labels, specialises the learned dynamics to whichever event is relevant. The key design choice is what the encoder should forecast during pretraining. Value-forecasting approaches, whether supervised  or pretrained on large corpora , shape representations around all variation in the signal, including noise irrelevant to the downstream event. The Joint-Embedding Predictive Architecture (JEPA)  offers an alternative: by forecasting future *representations* rather than future values, the encoder learns a latent space that retains what is predictable about the future and discards what is not.

We apply this principle to time series as HEPA (Horizon-conditioned Event Predictive Architecture). A causal Transformer encodes observations up to time $t$; a horizon-conditioned predictor maps the encoding and a horizon $\Delta t$ to a predicted future representation, forcing the encoder to internalise dynamics at multiple timescales ([fig.˜1](#S1.F1)). After self-supervised pretraining, the standard JEPA recipe discards the predictor and trains a linear probe on the frozen encoder. We instead retain the predictor: freeze the encoder but finetune the predictor alongside a lightweight event head that outputs a discrete-time survival CDF, ensuring that the predicted event probability never decreases as the horizon grows. This “predictor finetuning” recipe tunes only 198K parameters, roughly $11{\times}$ fewer than end-to-end training, yet is more expressive than a linear probe because the predictor reshapes its horizon-conditioned outputs to align with the downstream event.

Our contributions are:

- 1.
One architecture, any event, any domain. A single 2.16 M-parameter architecture with fixed hyperparameters, evaluated on 14 benchmarks across 11 domains via a unified probability surface $p(t,\Delta t)$. HEPA wins on 10 out of 14 benchmarks while tuning $11{\times}$ fewer parameters than PatchTST.
- 2.
Predictor finetuning as the downstream recipe. Freezing the encoder and finetuning only the predictor and event head tunes $11{\times}$ fewer parameters than end-to-end training. On the C-MAPSS benchmark , where degradation unfolds over hundreds of cycles, HEPA retains 92% of full-label h-AUROC at just 2% of labels. An information-theoretic bound ([proposition˜1](#Thmproposition1)) formalises when and why this works, and the bound’s key prediction, that lower pretraining loss implies stronger downstream performance, is consistent with the empirical trend across 14 datasets ([fig.˜3](#S3.F3)).

## 2 Related Work

#### Self-supervised learning for time series.

Self-supervised learning (SSL) for time-series representation learning falls into three families. Contrastive methods, including TS2Vec , TNC , TimesURL , CPC , and CoST , learn representations by contrasting positive and negative pairs. Masked reconstruction approaches such as PatchTST , SimMTM , and TimesNet  recover masked patches in input space. JEPA  takes a different path: predicting future *representations* rather than reconstructing inputs, avoiding tying the latent space to value-level fidelity. For time series, TS-JEPA  applies temporal masking for classification, and MTS-JEPA  adds codebook regularisation for anomaly detection. All these methods discard their pretraining head at inference and probe only the encoder. HEPA instead retains the predictor and finetunes it toward the downstream event, treating the predictor as a learnable bridge between frozen representations and event probabilities. The collapse-prevention mechanism follows the LeJEPA / SIGReg line  rather than the EMA schedule of I-JEPA.

#### Foundation models for time series.

Chronos-2 , TFM-2.5 , MOMENT , Moirai , and UniTS  pretrain on large-scale corpora for generic value forecasting. Generative pretraining  and LLM repurposing  offer alternative transfer strategies. These approaches target future channel values; HEPA targets event probabilities. See also concurrent work on industrial pretraining corpora . The encoder is mid-scale and pretrained per-dataset; what transfers across domains is the *recipe* (architecture + predictor finetuning), not the weights. We benchmark HEPA against four of these foundation models, using identical downstream heads to isolate encoder quality ([sections˜5](#S5), [G](#A7) and [G](#A7)).

#### Prognostics, anomaly prediction, and survival modelling.

C-MAPSS  is the standard remaining-useful-life (RUL) benchmark, where the supervised state of the art is STAR  (root mean square error, RMSE, 10.61). Self-supervised approaches to RUL prediction remain limited . Anomaly detection methods such as Anomaly Transformer , DCdetector , and TranAD  report point-adjusted F1, a metric shown to inflate scores dramatically by crediting entire segments from a single detection . These domain-specific metrics are incomparable across tasks. HEPA’s downstream parameterisation builds on discrete-time survival models , which decompose event probability into per-interval hazards composed into a survival CDF; we adapt this to a multi-horizon event prediction setting. We unify evaluation through h-AUROC, the mean of per-horizon AUROC values computed over the probability surface, which is threshold-free and robust to class imbalance ([section˜4](#S4)). Domain-specific metrics are reported as lossy projections of the same surface for comparability with published baselines.

## 3 Method

### 3.1 Architecture and Pretraining

Figure: Figure 2: HEPA architecture. Both stages sweep over all $(t,\Delta t)$ pairs per episode. *Stage 1:* The causal encoder $f_{\theta}$ maps $\mathbf{x}_{\leq t}$ to $\mathbf{h}_{t}$; the predictor $g_{\phi}(\mathbf{h}_{t},\Delta t)$ predicts future representations via a self-supervised JEPA objective. *Stage 2:* Encoder frozen; the predictor produces $K$ horizon-specific hazard rates $\lambda_{\Delta t}$ composed into a survival CDF (cumulative distribution function) $p(t,\Delta t)$.
Refer to caption: https://arxiv.org/html/2605.11130/2605.11130v4/x2.png

HEPA consists of three components that interact across two phases ([fig.˜2](#S3.F2)). The context encoder $f_{\theta}$ is a causal Transformer ($d{=}256$, 2 layers, 4 heads) that maps observations $\mathbf{x}_{\leq t}$, tokenised into non-overlapping patches of size $P{=}16$ (following PatchTST ) with per-context instance normalisation  and sinusoidal positional encodings, to a summary embedding $\mathbf{h}_{t}=f_{\theta}(\mathbf{x}_{\leq t})\in\mathbb{R}^{d}$. The predictor $g_{\phi}$ is a 2-layer multilayer perceptron (MLP) that takes the encoder output $\mathbf{h}_{t}$ together with a prediction horizon $\Delta t$ and produces a predicted embedding of the future interval:

$$ $\hat{\mathbf{h}}_{(t,t+\Delta t]}=g_{\phi}(\mathbf{h}_{t},\Delta t).$ (1) $$

During pretraining, $\Delta t$ is sampled from a log-uniform distribution over $[1,\Delta t_{\text{max}}]$, forcing the encoder to internalise dynamics at multiple timescales. The same encoder $f_{\theta}$, applied bidirectionally to $\mathbf{x}_{(t,t+\Delta t]}$ with attention pooling, produces the target representation $\mathbf{h}^{*}_{(t,t+\Delta t]}\in\mathbb{R}^{d}$. Both encoders are trained jointly via the optimizer; a SIGReg (Sketched Isotropic Gaussian Regularisation) term $\mathcal{L}_{\mathrm{SIG}}$  on the predictor output prevents representation collapse, replacing the exponential moving average (EMA) momentum schedule used in standard JEPA ([section˜I.3](#A9.SS3)). SIGReg constrains the predicted representations toward an isotropic Gaussian, which  prove is the optimal embedding distribution for minimising downstream prediction risk in joint-embedding architectures; this eliminates collapse without ad-hoc heuristics. A single mixing weight $\alpha{=}0.1$ controls its contribution to the total loss ([section˜I.3](#A9.SS3)).

#### Relation to canonical JEPA.

HEPA differs from BYOL/I-JEPA/V-JEPA-style joint-embedding predictive architectures in two ways: (a) the target encoder is a weight-shared copy of $f_{\theta}$ rather than an EMA copy or a stop-gradient branch, and (b) collapse is prevented by SIGReg (isotropic Gaussian constraint on the predictor output) rather than by the online/target asymmetry. Trivial collapse $\hat{H}=H^{*}=$const is prevented jointly by SIGReg *and* by the asymmetric inputs ($\mathbf{x}_{\leq t}$ for the online branch vs. $\mathbf{x}_{(t,t+\Delta t]}$ for the target branch): the predictor never sees the future window directly. This puts HEPA closer to LeJEPA / SIGReg variants  than to the original I-JEPA recipe.

The pretraining loss combines an L1 prediction objective (chosen over L2 because L1 distributes gradient magnitude equally across samples, avoiding domination by outlier predictions) with the SIGReg regulariser:

$$ $\mathcal{L}=(1-\alpha)\,\|\hat{\mathbf{h}}-\mathbf{h}^{*}\|_{1}+\alpha\,\mathcal{L}_{\mathrm{SIG}},$ (2) $$

where $\alpha$ balances the two terms. Because the target encoder shares weights with the online encoder, no stop-gradient is needed; both receive gradients through the optimizer. No labels are used. Pretraining takes under one minute per dataset on a single A10G GPU, with the full 14-dataset, 5-seed sweep completing in under two hours. Per-dataset preprocessing details are in [appendix˜L](#A12).

### 3.2 Downstream: Predictor Finetuning

After pretraining, we freeze the encoder $f_{\theta}$ and finetune only the predictor $g_{\phi}$ together with a lightweight linear event head. This “predictor finetuning” (pred-FT) recipe tunes 198K parameters, compared to 2.16M for end-to-end training and 513 for a frozen linear probe. Finetuning reshapes the predictor’s per-horizon outputs to separate event-relevant from event-irrelevant dynamics, making it more expressive than a linear probe, while the frozen encoder supplies the pretrained dynamical knowledge that makes few labels sufficient. End-to-end finetuning achieves equivalent h-AUROC at full labels ([table˜4](#A6.T4)); pred-FT’s advantage is computational efficiency and robustness under label scarcity ([section˜5.4](#S5.SS4)).

The predictor is run at each of $K$ discrete horizons $\Delta t=1,\ldots,K$ (unit steps; $K{=}150$ for C-MAPSS/TEP, $K{=}200$ otherwise). A shared linear head maps each predicted representation to a per-interval *conditional hazard*:

$$ $\lambda_{\Delta t}(t)\;=\;\sigma\!\bigl(\mathbf{w}^{\top}\hat{\mathbf{h}}_{(t,t+\Delta t]}+b\bigr)\;\in\;(0,1),$ (3) $$

where $\sigma$ is the sigmoid function and $\lambda_{\Delta t}(t)$ approximates $P(\text{event in }(\Delta t{-}1,\Delta t]\mid T^{*}>\Delta t{-}1,\mathbf{x}_{\leq t})$, with $T^{*}$ denoting the time to the first event after $t$. The event probability surface is then parameterised as a discrete-time survival CDF :

$$ $p(t,\Delta t)\;=\;1-\prod_{j=1}^{\Delta t}(1-\lambda_{j}(t)).$ (4) $$

Because each factor $(1-\lambda_{j})\in(0,1)$, the survival product is non-increasing in $\Delta t$, so $p(t,\Delta t)$ increases monotonically with the prediction horizon by construction. No distributional assumptions are required: each $\lambda_{\Delta t}$ is a free function of $\mathbf{h}_{t}$ via the predictor network. The finetuning loss sums positive-weighted binary cross-entropy (BCE) over horizons:

$$ $\mathcal{L}_{\text{FT}}=\sum_{\Delta t=1}^{K}w^{+}\cdot\text{BCE}\bigl(p(t,\Delta t),\;y(t,\Delta t)\bigr),$ (5) $$

where $y(t,\Delta t)=\mathds{1}[\text{event in }(t,t{+}\Delta t]]$ and $w^{+}=N_{\text{neg}}/N_{\text{pos}}$ compensates for class imbalance.(^1^11We apply BCE to the cumulative event probability $p(t,\Delta t)$ rather than to the per-step hazards $\lambda_{j}(t)$ against per-step indicators (the standard discrete-survival likelihood, e.g. nnet-survival ). This is a deliberate design choice: BCE on the cumulative surface acts as a smoothing regulariser across horizons (each hazard $\lambda_{j}$ contributes to BCE for every $\Delta t\geq j$), which empirically improves h-AUROC under our positive-weighted regime but distorts the probability scale ([appendix O](#A15)).)

### 3.3 Theoretical Analysis

Predictor finetuning rests on a premise: the pretrained encoder retains enough event-relevant information that a small downstream head can extract it. We formalise when this holds and connect the bound to experiments.

Let $X_{\leq t}$ denote observations up to time $t$, and let $E_{t+\Delta t}\in\{0,1\}$ be a binary indicator that equals 1 if an event occurs in the interval $(t,t{+}\Delta t]$ and 0 otherwise. The encoder produces $H_{t}=f_{\theta}(X_{\leq t})\in\mathbb{R}^{d}$; the target encoder produces $H^{*}=\bar{f}_{\theta}(X_{(t,t+\Delta t]})\in\mathbb{R}^{d}$ from the future interval; and the predictor produces $\hat{H}=g_{\phi}(H_{t},\Delta t)$. We define the event posterior $\eta(h)\coloneqq P(E_{t+\Delta t}{=}1\mid H^{*}{=}h)$ and the marginal event rate $\pi_{e}\coloneqq P(E_{t+\Delta t}{=}1)$, using $\pi_{e}$ to distinguish it from the probability surface $p(t,\Delta t)$.

###### Proposition 1 (Event-Information Retention) .

Suppose (A1) the event $E_{t+\Delta t}$ is conditionally independent of $X_{\leq t}$ given $H^{*}$,
(A2) the pretraining loss satisfies $\mathbb{E}[\|\hat{H}-H^{*}\|_{2}^{2}]\leq\varepsilon$,
(A3) the event posterior $\eta(h)$ is $L$-Lipschitz,
and (A4) the posterior is bounded: $\eta(H^{*})\in[\underline{\eta},\overline{\eta}]\subset(0,1)$ a.s.
Then

$$ $I(H_{t};\,E_{t+\Delta t})\;\geq\;I(H^{*};\,E_{t+\Delta t})\;-\;C_{\eta}\,L^{2}\,\varepsilon\;,$ (6) $$

where $C_{\eta}=(2\,\underline{\eta}\,(1{-}\overline{\eta}))^{-1}$ and $I(\cdot;\cdot)$ denotes mutual information.

The proof proceeds in three steps (full details in [appendix˜A](#A1)). First, because $\hat{H}$ is a deterministic function of $H_{t}$, the data processing inequality gives $I(H_{t};E)\geq I(\hat{H};E)$. Second, a Jensen-gap argument on the convex KL divergence, combined with the Lipschitz condition and prediction error bound, yields $I(H^{*};E)-I(\hat{H};E)\leq C_{\eta}L^{2}\varepsilon$. Combining these two inequalities produces the result.

The bound makes a falsifiable prediction: as pretraining proceeds and $\varepsilon$ shrinks, downstream h-AUROC should rise. The bound’s constants $L$ (Lipschitz of $\eta$), $C_{\eta}$ (posterior bound), and the target sufficiency $I(H^{\star};E_{t+\Delta t})$ are functions of the data-generating process: they vary across datasets but are held fixed within a dataset. The bound is therefore directly testable only *within* a dataset, by varying $\varepsilon$ alone. We do this on three contrasting domains, turbofan lifecycle (C-MAPSS-3), cardiac arrhythmia (MBA), and spacecraft telemetry anomalies (SMAP), by snapshotting the encoder during pretraining at epochs $\{1,3,8,25\}$ plus the converged best, and at each snapshot running the standard predictor finetuning recipe to obtain h-AUROC on the held-out test split (3 seeds per dataset). The bound’s monotone prediction holds across all three: pooled Spearman $\rho(\varepsilon,\text{h-AUROC})=-0.67$ ($p{=}0.017$, $n{=}12$) on C-MAPSS-3, $\rho{=}{-}0.64$ ($p{=}0.026$, $n{=}12$) on MBA, and $\rho{=}{-}0.49$ ($p{=}0.13$, $n{=}11$) on SMAP. SMAP shows the largest visible h-AUROC range (0.40 at $\varepsilon{=}0.033$ rising to 0.65 at $\varepsilon{=}0.026$). The converged-best snapshot regresses slightly relative to epoch 25 on all three datasets, consistent with mild over-pretraining at fixed labels. C-MAPSS-1 (the original lifecycle benchmark, $\rho{=}{-}0.87$, $p{<}0.001$) gives an even stronger signal and is reported in [appendix˜A](#A1). [Corollary˜2](#Thmproposition2) predicts a fourth regime where the bound becomes vacuous: on short-window anomaly benchmarks like GECCO we observe a within-dataset $\rho{=}{+}0.14$ ($p{=}0.67$) with finetuning instability across early snapshots, exactly as expected when extended precursors are weak (also in [appendix˜A](#A1)).

Figure: Figure 3: Self-supervised pretraining learns task-relevant structure. (a) Pretraining loss $\varepsilon$ vs. downstream h-AUROC ($\uparrow$) at fixed checkpoints across three domains (C-MAPSS-3: $\rho{=}{-}0.67$; MBA: $\rho{=}{-}0.64$; SMAP: $\rho{=}{-}0.49$; 3 seeds, error bars $\pm 1$ std). Within a dataset, $L$, $C_{\eta}$, and $I(H^{\star};E_{t+\Delta t})$ are constant, so the bound’s monotone prediction is directly testable. $\bigstar$ marks the converged-best snapshot; $\varepsilon$ scales differ across datasets so curves cannot be compared horizontally. (b) Principal component analysis (PCA) of pretrained C-MAPSS-1 representations for four test engines. Open circles: first observation (healthy); stars: last observation (near failure). PC1 captures 61% of variance; the encoder organises representations into a smooth degradation manifold without any labels.
Refer to caption: https://arxiv.org/html/2605.11130/2605.11130v4/x3.png

A cross-dataset scatter, by contrast, does *not* validate the bound: pooling the converged $\varepsilon$ across the 14 Table [1](#S5.T1) datasets gives Pearson $r{=}{-}0.05$ ($p{=}0.90$), because $L$, $C_{\eta}$, and the absolute scale of the target representation differ by dataset, dominating any signal from $\varepsilon$ alone; the same incommensurability [fig.˜3](#S3.F3) makes visible (C-MAPSS-3 clusters around $\varepsilon{\sim}0.015$, SMAP around $0.027$, MBA around $0.06$). This does not contradict the bound; it shows that comparing $\varepsilon$ across datasets compares incommensurable quantities, motivating the within-dataset protocol above. Two further caveats remain. The constants $L$ and $C_{\eta}$ are not estimated directly; [fig.˜3](#S3.F3) validates only the monotonic relationship, not the full quantitative bound. And A1 (target sufficiency) may fail when event precursors span intervals longer than the target window; when A1 is violated, the bound becomes loose in a *favourable* direction (see [appendix˜A](#A1) for assumption-by-assumption failure modes).

###### Corollary 2 (Precursor necessity) .

The bound is non-vacuous if and only if the future interval contains event precursors that the target encoder captures ($I(H^{*};E_{t+\Delta t})>0$) and the predictor approximates the target well enough ($\varepsilon<I(H^{*};E_{t+\Delta t})/(C_{\eta}L^{2})$).

This corollary explains both HEPA’s successes and its failures. On C-MAPSS, degradation unfolds over hundreds of cycles, so $I(H^{*};E_{t+\Delta t})$ is large and pretraining drives $\varepsilon$ small, yielding h-AUROC $\geq 0.81$. On datasets without extended precursors, the bound is vacuous regardless of pretraining quality.

###### Proposition 1 (Event-Information Retention) .

###### Corollary 2 (Precursor necessity) .

## 4 Evaluation Framework

The model outputs a probability surface $p(t,\Delta t)$ ([eq.˜4](#S3.E4)) for each observation time $t$ and prediction horizon $\Delta t$. This surface is the complete prediction; every metric is computed deterministically from it ([fig.˜4](#S4.F4)), enabling direct comparison with published baselines without retraining.

Figure: Figure 4: Evaluation framework. (a) The probability surface $p(t,\Delta t)$ on a representative C-MAPSS-1 engine (lifetime 174 cycles) unifies all event-prediction metrics as lossy projections. The colour scale matches Fig. [1](#S1.F1)b. RMSE requires converting the survival curve to a point estimate $\hat{\tau}=\sum_{\Delta t}\Delta t\cdot P(\text{event at }\Delta t)$; this projection is sensitive to calibration ([appendix˜J](#A10)). PA-F1 thresholds $p(t,1)$ at the smallest horizon and credits entire anomaly segments from a single detection (inflated ). F1 collapses to a single $(t,\Delta t)$ cell. h-AUROC averages AUROC over all horizons, using the full surface. (b) Per-horizon AUROC on GECCO ($K{=}200$ for HEPA / PatchTST / Chronos-2; sparse $K{=}8$ for iTransformer / MAE following the v34 protocol). Mean h-AUROC ($\uparrow$) per method shown in the legend; dashed lines mark the per-method mean. HEPA holds AUROC ${\geq}0.82$ across the full horizon range while value-level baselines decay sharply.
Refer to caption: https://arxiv.org/html/2605.11130/2605.11130v4/x5.png

As a cross-domain metric, we use h-AUROC: the mean of per-horizon AUROC values pooled over $(t,\Delta t)$ cells. Per-horizon prevalence varies wildly across datasets, and even within a single surface: on C-MAPSS-1, the event “failure within $\Delta t$ steps” has prevalence 0.5% at $\Delta t{=}1$ and 96% at $\Delta t{=}150$, a ${\sim}200{\times}$ range. Pooled area under the precision-recall curve (AUPRC) over all $(t,\Delta t)$ cells inherits a 0.957 baseline on C-MAPSS-1, because a model predicting only per-horizon prevalence already scores there. h-AUROC solves this by decomposing the surface into independent per-horizon binary classification problems, each with a universal 0.5 baseline that does not depend on prevalence. The uniform average treats all horizons equally; in practice, specific horizons matter more (long-range for turbine maintenance, short-range for arrhythmia). We use the uniform average for cross-domain comparability; the full surface is always stored for application-specific weighting. Domain-specific metrics (RMSE for remaining-useful-life, PA-F1 for anomaly detection) are derived as projections of the same surface for comparability with published baselines ([appendix˜J](#A10)). All numbers are reported as mean $\pm$ std across 5 seeds (HEPA, PatchTST, iTransformer, MAE) or 3 seeds (Chronos-2).

## 5 Experiments

### 5.1 Setup

We pretrain a separate HEPA encoder per dataset from unlabeled training data. Architecture and hyperparameters are identical across all domains; only the input projection (sensor count $S$) changes. All comparison methods share the same 198K-param downstream MLP head, positive-weighted BCE loss, and evaluation protocol; only the frozen encoder differs. Dense unit-step horizons are used throughout: $K{=}150$ for C-MAPSS and TEP, $K{=}200$ for all others. The dataset overview (14 datasets, 11 domains) is in [table˜3](#A4.T3).

### 5.2 Main Results

**Table 1: Main results (mean $\pm$ std; 5 seeds for HEPA, PatchTST, iTransformer, MAE; 3 seeds for Chronos-2). All methods use matched-capacity downstream heads on frozen encoders ([appendix˜C](#A3)). Each dataset has two rows: 100% labels and 10% labels (gray). Bold = best mean per row.**
|  |  |  | FM | Unified architecture | Domain-specific SOTA metric |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dataset | Domain | Label % | Chr-2 | PatchTST | iTransf | MAE | HEPA | Metric | HEPA | SOTA | Ref |
|  |  |  | h-AUROC $\uparrow$ | domain metric |  |  |  |  |  |  |  |
| C-MAPSS-1<br>[-1pt]turbine failure | Turbo. | 100 | $.66$$\pm.00$ | $.80$$\pm.04$ | $.70$$\pm.05$ | $.69$$\pm.02$ | $\mathbf{.81}$$\pm.03$ | RMSE$\downarrow$ | $28.5$ | $\mathbf{12.2}$ |  |
| 10 | $.66$$\pm.00$ | $.69$$\pm.08$ | $.59$$\pm.04$ | $.63$$\pm.12$ | $\mathbf{.78}$$\pm.07$ |  | $32.3$ | $\mathbf{17.0}$ |  |  |  |
| C-MAPSS-2<br>[-1pt]multi-cond. | Turbo. | 100 | $.45$$\pm.01$ | $.44$$\pm.03$ | $.43$$\pm.03$ | $.56$$\pm.01$ | $\mathbf{.57}$$\pm.01$ | RMSE$\downarrow$ | $40.8$ | $\mathbf{20.0}$ |  |
| 10 | $.46$$\pm.02$ | $.48$$\pm.08$ | $.50$$\pm.08$ | $.52$$\pm.01$ | $\mathbf{.55}$$\pm.01$ |  | $\mathbf{40.9}$ | $43.1$ |  |  |  |
| C-MAPSS-3<br>[-1pt]multi-fault | Turbo. | 100 | $.73$$\pm.00$ | $.79$$\pm.01$ | $.76$$\pm.01$ | $.78$$\pm.02$ | $\mathbf{.84}$$\pm.01$ | RMSE$\downarrow$ | $34.7$ | $\mathbf{12.7}$ |  |
| 10 | $.67$$\pm.00$ | $.77$$\pm.03$ | $.64$$\pm.06$ | $.78$$\pm.03$ | $\mathbf{.84}$$\pm.01$ |  | $47.1$ | $\mathbf{21.8}$ |  |  |  |
| C-MAPSS-4<br>[-1pt]multi-cond.+fault | Turbo. | 100 | — | $.52$$\pm.03$ | $.45$$\pm.02$ | $.57$$\pm.02$ | $\mathbf{.63}$$\pm.02$ | RMSE$\downarrow$ | — | — |  |
| 10 | — | $.51$$\pm.04$ | $.49$$\pm.04$ | $.52$$\pm.04$ | $\mathbf{.55}$$\pm.03$ |  | — | — |  |  |  |
| SMAP<br>[-1pt]sensor anomaly | Spacecraft | 100 | $.54$$\pm.02$ | $.49$$\pm.03$ | $.47$$\pm.05$ | $\mathbf{.64}$$\pm.04$ | $.59$$\pm.05$ | PA-F1$\uparrow$ | $.94$ | $\mathbf{.96}$ |  |
| 10 | $.51$$\pm.01$ | $.46$$\pm.05$ | $\mathbf{.52}$$\pm.07$ | $.50$$\pm.15$ | $.49$$\pm.08$ |  | $.92$ | $\mathbf{.96}$ |  |  |  |
| PSM<br>[-1pt]server anomaly | Server | 100 | $.48$$\pm.01$ | $.55$$\pm.02$ | $.52$$\pm.03$ | $.56$$\pm.02$ | $\mathbf{.57}$$\pm.02$ | PA-F1$\uparrow$ | $.94$ | $\mathbf{.98}$ |  |
| 10 | $.51$$\pm.01$ | $.43$$\pm.03$ | $\mathbf{.53}$$\pm.01$ | $.52$$\pm.02$ | $.48$$\pm.04$ |  | $.95$ | $\mathbf{.98}$ |  |  |  |
| MBA<br>[-1pt]arrhythmia | Cardiac | 100 | $.53$$\pm.10$ | $.68$$\pm.07$ | $\mathbf{.84}$$\pm.03$ | $.73$$\pm.03$ | $.75$$\pm.03$ | F1$\uparrow$ | $\mathbf{.98}$ | $.77$ | LSTM-AE |
| 10 | $.50$$\pm.07$ | $\mathbf{.65}$$\pm.08$ | $.31$$\pm.05$ | $.53$$\pm.04$ | $.55$$\pm.15$ |  | $\mathbf{.98}$ | $.81$ |  |  |  |
| BATADAL<br>[-1pt]cyberattack | ICS | 100 | $.56$$\pm.01$ | $\mathbf{.66}$$\pm.03$ | $.46$$\pm.13$ | $.60$$\pm.04$ | $.57$$\pm.03$ | F1$\uparrow$ | $\mathbf{.77}$ | $.74$ | AEED |
| 10 | $.55$$\pm.04$ | $.37$$\pm.04$ | $.54$$\pm.08$ | $\mathbf{.61}$$\pm.08$ | $.54$$\pm.06$ |  | $\mathbf{.61}$ | $.33$ |  |  |  |
| TEP<br>[-1pt]process fault | Chemical | 100 | — | $.99$$\pm.02$ | $.93$$\pm.02$ | $.96$$\pm.02$ | $\mathbf{1.00}$$\pm.00$ | F1$\uparrow$ | $\mathbf{.95}$ | $.93$ | XGBoost |
| 10 | — | $\mathbf{1.00}$$\pm.00$ | $.94$$\pm.05$ | $.99$$\pm.00$ | $\mathbf{1.00}$$\pm.00$ |  | $.88$$\pm.03$ | $.86$ |  |  |  |
| ETTm1<br>[-1pt]overheating | Power | 100 | $.74$$\pm.01$ | $.53$$\pm.03$ | $.79$$\pm.03$ | $\mathbf{.87}$$\pm.00$ | $.81$$\pm.00$ |  |  |  |  |
| 10 | $.68$$\pm.02$ | $.50$$\pm.03$ | $.61$$\pm.02$ | $\mathbf{.77}$$\pm.02$ | $.73$$\pm.01$ |  |  |  |  |  |  |
| Weather<br>[-1pt]heat spike | Climate | 100 | $.72$$\pm.02$ | $.64$$\pm.08$ | $.83$$\pm.04$ | $.88$$\pm.02$ | $\mathbf{.89}$$\pm.01$ |  |  |  |  |
| 10 | $.71$$\pm.02$ | $.67$$\pm.03$ | $\mathbf{.85}$$\pm.01$ | $.83$$\pm.02$ | $.83$$\pm.02$ |  |  |  |  |  |  |
| Beijing-AQ<br>[-1pt]PM2.5 spike | Air | 100 | $.53$$\pm.00$ | $\mathbf{.81}$$\pm.01$^† | $.75$$\pm.02$ | $\mathbf{.81}$$\pm.01$ | $\mathbf{.81}$$\pm.01$ |  |  |  |  |
| 10 | $.49$$\pm.00$ | $.65$$\pm.08$ | $.73$$\pm.02$ | $.77$$\pm.01$ | $\mathbf{.78}$$\pm.02$ |  |  |  |  |  |  |
| VIX<br>[-1pt]vol regime | Finance | 100 | $.40$$\pm.04$ | $.48$$\pm.11$ | $.40$$\pm.14$ | $.57$$\pm.03$ | $\mathbf{.57}$$\pm.01$ |  |  |  |  |
| 10 | $.38$$\pm.04$ | $\mathbf{.57}$$\pm.13$ | $.48$$\pm.05$ | $.55$$\pm.03$ | $.55$$\pm.01$ |  |  |  |  |  |  |
| GECCO<br>[-1pt]contamination | Water | 100 | $.74$$\pm.02$ | $.65$$\pm.07$ | $.64$$\pm.15$ | $.81$$\pm.04$ | $\mathbf{.88}$$\pm.06$ |  |  |  |  |
| 10 | $\mathbf{.80}$$\pm.03$ | $.52$$\pm.11$ | $.50$$\pm.13$ | $.55$$\pm.11$ | $.39$$\pm.13$ |  |  |  |  |  |  |
| Best (100%) | 0 | 2 | 1 | 4 | 10 |  |  |  |  |  |  |

[Table˜1](#S5.T1) compares HEPA against two classes of methods. The primary comparison is *architectural*: PatchTST , iTransformer , and a masked autoencoder (MAE) baseline use the same per-dataset regime with identical downstream heads, isolating the effect of JEPA pretraining versus alternative self-supervised and supervised objectives. The secondary comparison is against the *foundation model* Chronos-2 , which pretrains on a large external corpus and operates in a fundamentally different regime. A full comparison against MTS-JEPA  (matched protocol) is in [appendix˜H](#A8); HEPA wins on 8 out of 9 datasets where MTS-JEPA could be reproduced (TEP excluded: the public MTS-JEPA release does not include a chemical-process benchmark).

#### HEPA vs. architectural baselines.

HEPA wins on 10 out of 14 benchmarks at 100% labels, including all four C-MAPSS variants and the newly added FD004 (the hardest subset: six fault modes, six operating conditions). HEPA’s representation-level prediction captures temporal structure that supervised training (PatchTST) and reconstruction-based SSL (MAE) miss, particularly on datasets with extended precursor dynamics (C-MAPSS, GECCO, PSM, TEP). MAE is a strong second: it matches or exceeds HEPA on spacecraft telemetry (SMAP) and power systems (ETTm1), suggesting that reconstruction-based pretraining transfers well when the dominant failure mode is gradual drift. iTransformer’s variate-attention mechanism excels on MBA (h-AUROC 0.84 vs. HEPA’s 0.75), where arrhythmia patterns are localised across specific leads.

#### HEPA vs. Chronos-2.

HEPA matches or exceeds Chronos-2 on most benchmarks. Per-dataset JEPA excels when events have extended precursors that the local training data fully represents; large-corpus pretraining helps when event signatures resemble patterns seen at scale.

#### Honest losses.

HEPA is below the best baseline on four datasets at 100% labels. The pattern is interpretable: BATADAL and MBA have sensor-localised events where channel-fusion tokenisation dilutes the relevant subset, so per-variate attention (iTransformer) or channel-independent training (PatchTST) wins; MAE’s reconstruction objective transfers well when the dominant failure mode is gradual drift (SMAP, ETTm1). Adopting a sensor-as-token strategy  within the HEPA encoder is a natural way to close this gap.

### 5.3 What Does Pretraining Learn?

[Figure˜3](#S3.F3) visualises encoder representations after self-supervised pretraining on C-MAPSS-1. Without any labels, the encoder organises representations into a smooth degradation manifold: PC1 alone captures 61% of variance and tracks time-to-failure monotonically within each engine (median per-engine Spearman ${\rho}{=}{+}0.97$, $84\%$ of engines ${\rho}{>}0.9$). Engines starting from different healthy regions converge toward a shared failure region. This structure explains why so few labels suffice: the encoder has already separated healthy from degraded states.

### 5.4 Label Efficiency

All methods in [table˜1](#S5.T1) freeze their encoder and train only a downstream head, so all benefit from pretraining under label scarcity. The question is whether HEPA’s representations degrade more gracefully. [Table˜2](#S5.T2) shows that on C-MAPSS, where degradation unfolds over hundreds of cycles and the JEPA predictor achieves low pretraining loss, HEPA retains 92% of full-label h-AUROC with just 2 training engines out of 85. C-MAPSS-3 retains 97% at 10% labels. This is consistent with [proposition˜1](#Thmproposition1): low $\varepsilon$ on lifecycle datasets means the encoder already separates healthy from degraded states, so the finetuned predictor needs only a few labelled examples to map them to event probabilities.

The advantage is not universal. At 10% labels across all 14 datasets ([table˜1](#S5.T1), gray rows), HEPA wins on 6 out of 14, compared to 10 out of 14 at full labels. On anomaly datasets without extended precursors (SMAP, PSM, GECCO at 10%), the frozen-encoder setup limits how much any method can degrade, so margins compress. The label-efficiency story is strongest where HEPA’s pretraining loss is lowest: extended-precursor lifecycle datasets.

**Table 2: Label efficiency on C-MAPSS lifecycle datasets (HEPA only, 3 seeds). h-AUROC ($\uparrow$) and retention relative to full labels. C-MAPSS-1 retains 92% at 2% labels (2 of 85 training engines).**
|  | C-MAPSS-1 (85 eng.) | C-MAPSS-3 (100 eng.) |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Labels | h-AUROC | Ret. | Eng. | h-AUROC | Ret. | Eng. |
| 100% | $.786\pm.033$ | 100% | 85 | $.853\pm.004$ | 100% | 100 |
| 10% | $.772\pm.059$ | 98% | 9 | $.830\pm.018$ | 97% | 10 |
| 5% | $.730\pm.018$ | 93% | 4 | $.709\pm.131$ | 83% | 5 |
| 2% | $\mathbf{.724\pm.013}$ | 92% | 2 | $.635\pm.065$ | 74% | 2 |
| 1% | $.670\pm.110$ | 85% | 1 | $.513\pm.220$ | 60% | 1 |

## 6 Conclusion & Future Work

HEPA demonstrates that self-supervised JEPA pretraining combined with predictor finetuning provides a practical recipe for event prediction. The encoder learns temporal dynamics from unlabelled data; the predictor learns which dynamics signal the target event. One architecture handles degradation forecasting, anomaly prediction, and arrhythmia detection across 14 benchmarks in 11 domains, matching or exceeding PatchTST, iTransformer, MAE, and Chronos-2 on the majority of benchmarks while tuning an order of magnitude fewer parameters. On lifecycle datasets, the recipe is robust to extreme label scarcity: 92% of full-label performance with 2% of labels on C-MAPSS, consistent with the information-retention guarantee of [proposition˜1](#Thmproposition1). Because the recipe is domain-agnostic, the same architecture that predicts turbine failure from flight-recorder data can flag arrhythmia risk from ECG streams or detect water contamination from sensor networks, each time requiring only a handful of event labels.

Looking ahead, cross-domain pretraining on corpora such as FactoryNet  is the natural next step toward industrial deployment, and sensor-as-token strategies  could close the gap on systems where event-relevant information is concentrated in a few channels. On the theory side, deriving fully empirical versions of the information-retention bound that estimate $L$ and $C_{\eta}$ directly from data remains an interesting open problem. Wherever multivariate sensors record the precursors to rare but consequential events, HEPA offers a path from unlabelled streams to actionable predictions.

## Appendix A Theoretical Analysis: Full Proofs and Discussion

### A.1 Notation and Preliminaries

We work with the following random variables on a common probability space:
$X_{\leq t}$ (observations up to time $t$),
$X_{(t,t+\Delta t]}$ (future observations),
$E_{t+\Delta t}\in\{0,1\}$ (event indicator),
$H_{t}=f_{\theta}(X_{\leq t})\in\mathbb{R}^{d}$ (encoder output),
$H^{*}=\bar{f}_{\theta}(X_{(t,t+\Delta t]})\in\mathbb{R}^{d}$ (target encoder output),
$\hat{H}=g_{\phi}(H_{t},\Delta t)\in\mathbb{R}^{d}$ (predicted representation).
All mutual informations $I(\cdot;\cdot)$ and entropies $\mathbb{H}(\cdot)$ are well-defined: $E_{t+\Delta t}$ is discrete (binary), and for the continuous variables $H_{t},H^{*},\hat{H}$ we use differential entropy and the standard extension of mutual information to mixed discrete-continuous pairs . We use $\mathbb{H}$ (blackboard bold) for entropy to avoid confusion with the encoder embedding $H_{t}$.

We define the event posterior $\eta(h)\coloneqq P(E_{t+\Delta t}=1\mid H^{*}=h)$ and the marginal event rate $\pi_{e}\coloneqq P(E_{t+\Delta t}=1)$, using $\pi_{e}$ to distinguish it from the probability surface $p(t,\Delta t)$ in the main text.

### A.2 Assumptions

- (A1)
Target sufficiency.
$E_{t+\Delta t}\perp\!\!\!\perp X_{\leq t}\mid H^{*}$.
*Interpretation.* The target encoder’s representation of the future interval is a sufficient statistic for the event, given the past.
This holds when: (a) the event is determined by the dynamics in $(t,t+\Delta t]$, and (b) the target encoder has enough capacity and sees the relevant future interval.
Because $\bar{f}_{\theta}$ is bidirectional with attention pooling over the full interval, it is strictly more expressive than the causal encoder for summarising the future, making this assumption mild for well-trained target encoders.
*When it fails.* If the event depends on context outside $(t,t+\Delta t]$ (for instance, a slow trend visible only in $X_{\leq t}$ that the target encoder cannot see), then A1 is violated and the past observations carry event information not mediated by $H^{*}$. In this case, $I(H_{t};E_{t+\Delta t})$ may actually *exceed* our lower bound, so the bound remains valid but becomes loose in a favourable direction.
- (A2)
Bounded prediction error.
$\mathbb{E}[\|\hat{H}-H^{*}\|_{2}^{2}]\leq\varepsilon$.
*Interpretation.* The pretraining loss (L1 on L2-normalised representations in practice) drives the prediction residual small.
The bound uses L2 squared error for analytical tractability. Since $\|u\|_{2}\leq\|u\|_{1}$ for all $u\in\mathbb{R}^{d}$, lower L1 loss implies lower L2 loss, so the L1 training loss is a monotone proxy for $\varepsilon$. We use this monotonic relationship in our empirical validation ([fig.˜3](#S3.F3)) to test the bound’s qualitative prediction without requiring a precise norm conversion.
*When it fails.* Early in training or on out-of-distribution horizons, $\varepsilon$ can be large and the bound becomes vacuous.
- (A3)
Smooth event dependence.
The conditional distribution $P(E_{t+\Delta t}=1\mid H^{*}=h)$ is Lipschitz continuous in $h$ with constant $L$: for all $h,h^{\prime}\in\mathbb{R}^{d}$,
$|P(E_{t+\Delta t}=1\mid H^{*}=h)-P(E_{t+\Delta t}=1\mid H^{*}=h^{\prime})|\leq L\|h-h^{\prime}\|_{2}$.
*Interpretation.* Small perturbations of the target representation do not drastically change the event probability. This is a regularity condition on the relationship between the learned representation space and event occurrence; it holds whenever the event boundary in representation space is not a fractal or highly irregular set.
*When it fails.* If the event probability is a discontinuous function of $H^{*}$ (e.g. a hard threshold on a single component), the Lipschitz constant $L$ diverges and our continuity argument requires replacement by a discrete analysis.
- (A4)
Bounded event posterior.
There exist $0<\underline{\eta}\leq\overline{\eta}<1$ such that $P(\eta(H^{*})\in[\underline{\eta},\overline{\eta}])=1$, where $\eta(h)=P(E_{t+\Delta t}=1\mid H^{*}=h)$.
*Interpretation.* The event posterior, evaluated on the support of the target encoder’s output, is bounded away from 0 and 1 almost surely. This matches the main-text assumption A4. In practice, $H^{*}$ is L2-normalised onto the unit sphere (a compact set), and A3 guarantees that $\eta$ is Lipschitz; by the image of a compact connected set under a Lipschitz map, the range of $\eta(H^{*})$ is a closed bounded interval, and $0<\underline{\eta}$ follows from the event being non-trivially detectable from the future window. Note that A4 implies the marginal event rate $\pi_{e}=\mathbb{E}[\eta(H^{*})]$ is bounded in $[\underline{\eta},\overline{\eta}]$, so the event is neither impossible nor certain.
*Role in the proof.* This assumption is needed to bound $\sup_{q}\varphi^{\prime\prime}(q)=\sup_{q}1/(q(1-q))$ over the support of $\eta(H^{*})$ (Step 2 of the proof). Without it, the KL second derivative $\varphi^{\prime\prime}$ is unbounded near 0 and 1, making the Jensen-gap bound vacuous. The constant in the bound becomes $C_{\eta}=(2\underline{\eta}(1-\overline{\eta}))^{-1}$. As $\pi_{e}\to 0$ or $\pi_{e}\to 1$, the posterior bounds are forced toward 0 or 1, driving $C_{\eta}\to\infty$; this reflects a genuine difficulty: distinguishing $P(E|\hat{H})$ from the prior requires high precision when events are extremely rare.
*When it fails.* If $\eta(H^{*})$ concentrates near 0 or 1, then $C_{\eta}\to\infty$ and the bound degrades. Empirically, this is the case on CHB-MIT, where seizure onset cannot be reliably predicted from any 16-second past context, and $\eta(h)\approx\pi_{e}$ for all $h$ (consistent with $I(H^{*};E)\approx 0$).

### A.3 Full Proof of Proposition 1

###### Proof.

We proceed in three steps.

Step 1: Data processing inequality.
For fixed $\Delta t$ (which we condition on throughout), $\hat{H}=g_{\phi}(H_{t},\Delta t)$ is a deterministic function of $H_{t}$. Since a deterministic function cannot introduce new information, $E_{t+\Delta t}\perp\!\!\!\perp\hat{H}\mid H_{t}$, so the triple $(E_{t+\Delta t},H_{t},\hat{H})$ satisfies the Markov chain $E_{t+\Delta t}\to H_{t}\to\hat{H}$, and the data processing inequality gives

$$ $I(H_{t};\,E_{t+\Delta t})\;\geq\;I(\hat{H};\,E_{t+\Delta t}).$ (7) $$

This step uses only the functional relationship between $H_{t}$ and $\hat{H}$; no assumptions on the data-generating process are needed.

Step 2: Jensen gap bound on mutual information loss.
We bound $I(H^{*};E_{t+\Delta t})-I(\hat{H};E_{t+\Delta t})$.

*Expressing MI as expected KL divergence.*
For any representation $R$ jointly distributed with binary $E=E_{t+\Delta t}$:

$$ $I(R;\,E)=\mathbb{E}_{R}\bigl[D_{\mathrm{KL}}\bigl(\mathrm{Ber}(\eta_{R}(R))\,\big\|\,\mathrm{Ber}(\pi_{e})\bigr)\bigr],$ (8) $$

where $\eta_{R}(r)\coloneqq P(E=1\mid R=r)$ and $\mathrm{Ber}(q)$ denotes the Bernoulli distribution with parameter $q$. Equation ([8](#A1.E8)) is the standard expression of mutual information as the expected KL divergence between the conditional and marginal label distributions; see or .
Applying ([8](#A1.E8)) to $R=H^{*}$ and $R=\hat{H}$:

$$ $\displaystyle I(H^{*};E)$ $\displaystyle=\mathbb{E}_{H^{*}}\bigl[D_{\mathrm{KL}}\bigl(\mathrm{Ber}(\eta(H^{*}))\,\big\|\,\mathrm{Ber}(\pi_{e})\bigr)\bigr],$ (9) $\displaystyle I(\hat{H};E)$ $\displaystyle=\mathbb{E}_{\hat{H}}\bigl[D_{\mathrm{KL}}\bigl(\mathrm{Ber}(\eta_{\hat{H}}(\hat{H}))\,\big\|\,\mathrm{Ber}(\pi_{e})\bigr)\bigr],$ (10) $$

where $\eta_{\hat{H}}(\hat{h})\coloneqq P(E=1\mid\hat{H}=\hat{h})$.

*Relating $\eta_{\hat{H}}$ to $\eta$ via A1.*
Under [(A1)](#A1.I1.i1), $E\perp\!\!\!\perp X_{\leq t}\mid H^{*}$.
Since $\hat{H}=g_{\phi}(f_{\theta}(X_{\leq t}),\Delta t)$ is a composition of measurable functions, it is $\sigma(X_{\leq t})$-measurable, and therefore $E\perp\!\!\!\perp\hat{H}\mid H^{*}$. It follows that $P(E{=}1\mid H^{*},\hat{H})=P(E{=}1\mid H^{*})=\eta(H^{*})$. By the tower property of conditional expectation, for any value $\hat{h}$:

$$ $\eta_{\hat{H}}(\hat{h})=P(E=1\mid\hat{H}=\hat{h})=\mathbb{E}\bigl[\eta(H^{*})\mid\hat{H}=\hat{h}\bigr].$ (11) $$

*Applying Jensen’s inequality.*
The function $q\mapsto D_{\mathrm{KL}}(\mathrm{Ber}(q)\,\|\,\mathrm{Ber}(\pi_{e}))$ is convex on $(0,1)$ (its second derivative is $1/(q(1-q))>0$).
*Bounding the Jensen gap.*
For a twice-differentiable convex function $\varphi$, the Jensen gap satisfies :

$$ $\mathbb{E}[\varphi(Y)]-\varphi(\mathbb{E}[Y])\;\leq\;\tfrac{1}{2}\,\sup_{y}\,\varphi^{\prime\prime}(y)\;\mathrm{Var}(Y).$ (12) $$

Here $\varphi(q)=D_{\mathrm{KL}}(\mathrm{Ber}(q)\,\|\,\mathrm{Ber}(\pi_{e}))=q\ln(q/\pi_{e})+(1{-}q)\ln((1{-}q)/(1{-}\pi_{e}))$ and $Y=\eta(H^{*})$ conditioned on $\hat{H}$.
The second derivative is $\varphi^{\prime\prime}(q)=1/(q(1{-}q))$.
Under [(A4)](#A1.I1.i4), $\eta(H^{*})\in[\underline{\eta},\overline{\eta}]$ almost surely. The supremum in ([12](#A1.E12)) is over the support of the conditional distribution $Y\mid\hat{H}=\hat{h}$, which is a subset of $[\underline{\eta},\overline{\eta}]$; we relax it to the marginal support, giving $\sup_{q}\varphi^{\prime\prime}(q)\leq 1/(\underline{\eta}(1{-}\overline{\eta}))$.
Applying ([12](#A1.E12)) conditionally on $\hat{H}=\hat{h}$:

$$ $\mathbb{E}\bigl[D_{\mathrm{KL}}\bigl(\mathrm{Ber}(\eta(H^{*}))\,\big\|\,\mathrm{Ber}(\pi_{e})\bigr)\,\big|\,\hat{H}=\hat{h}\bigr]-D_{\mathrm{KL}}\bigl(\mathrm{Ber}(\eta_{\hat{H}}(\hat{h}))\,\big\|\,\mathrm{Ber}(\pi_{e})\bigr)\;\leq\;\frac{\mathrm{Var}(\eta(H^{*})\mid\hat{H}=\hat{h})}{2\,\underline{\eta}\,(1-\overline{\eta})},$ (13) $$

where we have used $q(1-q)\geq\underline{\eta}(1-\overline{\eta})$ for $q\in[\underline{\eta},\overline{\eta}]$ (from A4).
Note that $q(1-q)$ is concave with minimum at the endpoints of $[\underline{\eta},\overline{\eta}]$; since $\underline{\eta}(1-\overline{\eta})\leq\min(\underline{\eta}(1-\underline{\eta}),\overline{\eta}(1-\overline{\eta}))$, the bound on $\varphi^{\prime\prime}$ is valid (though not tight when the interval is asymmetric around $1/2$).

*Bounding the conditional variance via A2 and A3.*
By [(A3)](#A1.I1.i3): $|\eta(H^{*})-\eta(\hat{H})|\leq L\|H^{*}-\hat{H}\|_{2}$ (where we evaluate $\eta$ at $\hat{H}$, using that $\eta$ is defined on all of $\mathbb{R}^{d}$).
The random variable $\eta(H^{*})$ is bounded in $[\underline{\eta},\overline{\eta}]\subset(0,1)$ by [(A4)](#A1.I1.i4), so all conditional second moments below are finite. (Note that $\eta(\hat{H})$ need not lie in $[\underline{\eta},\overline{\eta}]$ since $\hat{H}$ may fall outside the support of $H^{*}$, but this does not affect the bound: the variance inequality below requires only that the right-hand side is finite, which follows from the Lipschitz condition and bounded L2 error.) For any square-integrable random variable $Y$, $\mathrm{Var}(Y\mid Z)\leq\mathbb{E}[(Y-c)^{2}\mid Z]$ for any $Z$-measurable $c$; taking $c=\eta(\hat{H})$:

$$ $\mathrm{Var}\bigl(\eta(H^{*})\mid\hat{H}\bigr)\;\leq\;\mathbb{E}\bigl[(\eta(H^{*})-\eta(\hat{H}))^{2}\mid\hat{H}\bigr]\;\leq\;L^{2}\,\mathbb{E}\bigl[\|H^{*}-\hat{H}\|_{2}^{2}\mid\hat{H}\bigr].$ (14) $$

Taking expectations over $\hat{H}$ in ([13](#A1.E13)) and substituting ([14](#A1.E14)):

$$ $\displaystyle I(H^{*};E)-I(\hat{H};E)$ $\displaystyle\leq\frac{L^{2}}{2\,\underline{\eta}\,(1-\overline{\eta})}\;\mathbb{E}\bigl[\|H^{*}-\hat{H}\|_{2}^{2}\bigr]$ $\displaystyle\leq\frac{L^{2}\,\varepsilon}{2\,\underline{\eta}\,(1-\overline{\eta})}\;=\;C_{\eta}\,L^{2}\,\varepsilon,$ (15) $$

where the last line uses [(A2)](#A1.I1.i2) and defines $C_{\eta}\coloneqq(2\,\underline{\eta}\,(1-\overline{\eta}))^{-1}$ using the posterior bounds from [(A4)](#A1.I1.i4).
The constant $C_{\eta}$ depends on the posterior bounds rather than the marginal event rate, making the bound valid without assumptions on the concentration of $\eta(H^{*})$ around $\pi_{e}$.

Step 3: Assembling the bound.
Combining ([7](#A1.E7)) and ([15](#A1.E15)):

$$ $I(H_{t};\,E_{t+\Delta t})\;\geq\;I(\hat{H};\,E_{t+\Delta t})\;\geq\;I(H^{*};\,E_{t+\Delta t})\;-\;C_{\eta}\,L^{2}\,\varepsilon.\qed$ (16) $$

###### Remark 1 (Comparison with $\sqrt{\varepsilon}$ bounds) .

The bound is *linear* in $\varepsilon$, which for small $\varepsilon$ (the regime of interest, i.e., well-pretrained models) is tighter than bounds obtained via Pinsker’s inequality (which yield $\sqrt{\varepsilon}$ dependence). For large $\varepsilon$, Pinsker-based bounds may be tighter; the constants also differ, so the comparison is regime-dependent.
This improvement comes from exploiting the Jensen-gap structure: the KL divergence’s convexity provides a direct second-order bound on the information loss, rather than going through total variation as an intermediate.
The price is the constant $C_{\eta}$, which diverges as the posterior bounds $\underline{\eta}$ or $\overline{\eta}$ approach 0 or 1, reflecting the genuine difficulty of detecting events when the posterior concentrates near certainty or impossibility.

###### Remark 2 (Role of the Lipschitz constant) .

The bound involves the Lipschitz constant $L$ of the event posterior with respect to the target representation. In practice, this is controlled by the smoothness of the learned representation space: well-regularised encoders (EMA target, L2 normalisation) produce representations where $L$ is moderate.

###### Proof.

###### Remark 1 (Comparison with ε \sqrt{\varepsilon} bounds) .

###### Remark 2 (Role of the Lipschitz constant) .

### A.4 Tightness Analysis

#### When is the bound tight?

The bound is approximately tight when: (a) the conditional variance $\mathrm{Var}(\eta(H^{*})\mid\hat{H})$ is close to $L^{2}\mathbb{E}[\|H^{*}-\hat{H}\|_{2}^{2}\mid\hat{H}]$ (the Lipschitz bound on variance is tight, which occurs when the prediction error aligns with the direction of steepest change in $\eta$), and (b) the Jensen gap bound ([12](#A1.E12)) is close to equality (which occurs when $\eta(H^{*})$ is approximately symmetrically distributed around its conditional mean given $\hat{H}$).
In the high-SNR regime ($\varepsilon\ll I(H^{*};E_{t+\Delta t})/(C_{\eta}L^{2})$), the penalty term is small relative to the mutual information and the bound approaches $I(H_{t};E_{t+\Delta t})\gtrsim I(H^{*};E_{t+\Delta t})$: nearly all event information is retained.

#### When is the bound vacuous?

Setting the right-hand side of ([6](#S3.E6)) to zero gives the vacuity threshold:

$$ $\varepsilon_{\mathrm{vac}}=\frac{I(H^{*};E_{t+\Delta t})}{C_{\eta}\,L^{2}}.$ (17) $$

When $\varepsilon>\varepsilon_{\mathrm{vac}}$, the bound provides no guarantee. This occurs when (i) $I(H^{*};E_{t+\Delta t})\approx 0$ (no precursors), or (ii) $\varepsilon$ is large (poor pretraining), or (iii) $L$ is large (irregular event boundary).
Importantly, a vacuous bound does not imply that $I(H_{t};E_{t+\Delta t})=0$; the encoder may retain information through paths not captured by our analysis (e.g. the encoder directly encodes precursor patterns without going through the predictor).

### A.5 Why Predictor Weights Do Not Transfer

Our empirical finding ([appendix˜F](#A6)) that predictor *architecture* matters but predictor *pretrained weights* do not is explained by a codomain mismatch.

During pretraining, $g_{\phi}\colon\mathbb{R}^{d}\times\mathbb{R}\to\mathbb{R}^{d}$ maps encoder states to predicted *representations*; its codomain is the full $d$-dimensional representation space.
During finetuning, the predictor (with the same architecture but different final layer) maps to *event logits*: $g_{\phi}^{\prime}\colon\mathbb{R}^{d}\times\mathbb{R}\to\mathbb{R}^{K}$, where $K$ is the number of horizon intervals and $K\ll d$.

Formally, pretraining minimises $\mathbb{E}[\|g_{\phi}(H_{t},\Delta t)-H^{*}\|_{1}]$, while finetuning minimises $\mathbb{E}[\ell_{\mathrm{BCE}}(\sigma(g_{\phi}^{\prime}(H_{t},\Delta t)),\,E)]$.
The pretraining objective rewards the predictor for reconstructing *all* $d$ components of $H^{*}$, most of which encode event-irrelevant dynamics (channel-level forecasting, trend, seasonality).
The finetuning objective rewards the predictor for extracting only the $\leq K$ bits relevant to event prediction.
The optimal weight matrices for these two objectives need not be correlated, and indeed our experiments confirm they are not: at 100% labels, pred-FT with pretrained predictor weights achieves h-AUROC within $\pm 0.003$ of pred-FT with randomly initialised predictor weights (the legacy AUPRC reading was within the same margin).

This is analogous to the observation in vision that pretrained *classifier heads* do not transfer across tasks while pretrained *backbones* do: the backbone compresses the input into a general-purpose representation, while the head specialises to a specific output space.

### A.6 Connection to Empirical Results

#### C-MAPSS (turbofan degradation).

Degradation in turbofan engines develops over hundreds of cycles, with sensor readings progressively deviating from healthy baselines.
The target encoder, seeing the future interval bidirectionally, captures degradation state with high fidelity: $I(H^{*};E_{t+\Delta t})$ is large (the future interval is highly informative about whether failure occurs in that interval).
Pretraining achieves low prediction error $\varepsilon$ because the dynamics are smooth and near-deterministic.
[Proposition˜1](#Thmproposition1) predicts strong event-information retention, consistent with h-AUROC $=0.806$ on C-MAPSS-1 and $=0.568$ on C-MAPSS-2 (5 seeds, dense $K{=}150$).

#### CHB-MIT (seizure prediction).

We evaluated CHB-MIT as a pilot to test the bound’s predictions on a dataset expected to fail; it is excluded from the formal benchmark in [table˜1](#S5.T1) because scalp EEG seizure prediction is a fundamentally different signal regime, with no reliable electrographic precursors observable in the 16-second context window used by all other datasets in our benchmark.
Seizure onset in scalp EEG has minimal electrographic precursors in the pre-ictal period, particularly within a 16-second context.
The target encoder’s representation $H^{*}$ carries little event information: $I(H^{*};E_{t+\Delta t})\approx 0$.
[Corollary˜2](#Thmproposition2), condition (i), fails regardless of pretraining quality.
This is consistent with the empirical result of h-AUROC $=0.497$ (chance level), and turns CHB-MIT into a clean negative control for the theory rather than an inappropriate target.

#### C-MAPSS-2 (multi-operating-condition).

C-MAPSS-2 adds operating-condition variability to C-MAPSS-1. This increases $\varepsilon$ (harder prediction task) without proportionally increasing $I(H^{*};E_{t+\Delta t})$ (event information is the same; only noise increases). The bound predicts that C-MAPSS-2 should be harder than C-MAPSS-1 for the same architecture, consistent with the observed drop from $0.806$ to $0.568$ h-AUROC.

#### Label efficiency.

At 5% labels, pred-FT achieves F1 $=0.261$ vs. scratch $=0.035$.
The bound supports this observation: the encoder’s $I(H_{t};E_{t+\Delta t})$ is established during pretraining and does not depend on labels. Reducing labels degrades the finetuning optimisation (finding the right head weights) but cannot reduce the information available in the frozen encoder. Note that the proposition addresses information retention in the encoder, not sample complexity of the finetuning step; the label-efficiency advantage is consistent with, but not directly formalised by, the bound.

### A.7 Per-Horizon Generalisation

[Proposition˜1](#Thmproposition1) is stated for a fixed horizon $\Delta t$. We now make the horizon dependence explicit. All quantities on the right-hand side of the bound can vary with $\Delta t$: the event posterior $\eta_{\Delta t}(h)=P(E_{t+\Delta t}=1\mid H^{*}_{\Delta t}=h)$, the Lipschitz constant $L(\Delta t)$, the prediction error $\varepsilon(\Delta t)=\mathbb{E}[\|H^{*}_{\Delta t}-\hat{H}_{\Delta t}\|_{2}^{2}]$, and the posterior bounds $\underline{\eta}_{\Delta t},\overline{\eta}_{\Delta t}$ from [(A4)](#A1.I1.i4).

###### Proposition 3 (Per-horizon information retention) .

Under the assumptions of [proposition˜1](#Thmproposition1) applied at each horizon $\Delta t>0$ separately, with horizon-dependent quantities $L(\Delta t)$, $\varepsilon(\Delta t)$, and $C_{\eta}(\Delta t)=(2\underline{\eta}_{\Delta t}(1-\overline{\eta}_{\Delta t}))^{-1}$:

$$ $I\bigl(H_{t};\,E_{t+\Delta t}\bigr)\;\geq\;I\bigl(H^{*}_{\Delta t};\,E_{t+\Delta t}\bigr)\;-\;C_{\eta}(\Delta t)\,L(\Delta t)^{2}\,\varepsilon(\Delta t).$ (18) $$

###### Proof.

Apply the proof of [proposition˜1](#Thmproposition1) verbatim at each fixed $\Delta t$, with all quantities carrying a subscript $\Delta t$.
∎

###### Proposition 3 (Per-horizon information retention) .

###### Proof.

#### Predicted shape of the per-horizon AUROC curve.

Equation ([18](#A1.E18)) predicts how the encoder’s event information varies with horizon. Three effects combine:

- •
$I(H^{*}_{\Delta t};E_{t+\Delta t})$ is typically non-monotone: very short horizons may not yet encompass precursors; very long horizons dilute the event signal with unrelated dynamics.
- •
$\varepsilon(\Delta t)$ is monotonically increasing (representations further into the future are harder to predict from $H_{t}$).
- •
$L(\Delta t)$ can increase with $\Delta t$ if the event boundary sharpens as the event approaches.

The net effect is that $I(H_{t};E_{t+\Delta t})$ peaks at intermediate $\Delta t$ and decays at large $\Delta t$. While MI and AUROC are not monotonically related in general, this qualitative shape is compatible with the per-horizon AUROC curves observed in our experiments: AUROC tends to be highest at $\Delta t\in[10,50]$ cycles for C-MAPSS and at $\Delta t\in[1,20]$ steps for anomaly datasets, then declines. The decay is more pronounced for the anomaly datasets because $\varepsilon(\Delta t)$ grows faster when the dynamics are irregular.

### A.8 Relationship to the Information Bottleneck

The Information Bottleneck (IB) framework seeks a representation $T$ of input $X$ that maximises $I(T;Y)$ (relevance) while minimising $I(T;X)$ (compression):

$$ $\min_{P(T|X)}\;I(T;X)-\beta\,I(T;Y).$ (19) $$

JEPA pretraining differs from IB in three ways:

- 1.
No explicit compression.
JEPA does not penalise $I(H_{t};X_{\leq t})$. The encoder is free to retain all information about the past. Implicit compression arises only from the architecture bottleneck ($d=256$).
- 2.
Predictive, not discriminative.
The IB target $Y$ is a label; the JEPA target is a future representation $H^{*}$. JEPA implicitly encourages high $I(H_{t};H^{*})$ by minimising prediction error (through the predictor), and $I(H_{t};H^{*})$ is an upper bound on $I(H_{t};Y)$ for any $Y$ that is a function of the future interval (by DPI). This makes JEPA a *looser* but *more general* objective: it retains information about all downstream tasks, not just one.
- 3.
Asymmetric architecture.
The IB is typically symmetric in $X$ and $Y$. JEPA imposes causal structure: the encoder sees only the past, the target encoder sees only the future. This causal asymmetry is essential for event prediction, where we cannot condition on the future at test time.

The connection to our result is direct. Let $Y=E_{t+\Delta t}$. By [proposition˜1](#Thmproposition1):

$$ $I(H_{t};Y)\;\geq\;I(H^{*};Y)\;-\;C_{\eta}\,L^{2}\,\varepsilon.$ (20) $$

As the JEPA pretraining loss drives $\varepsilon\to 0$, the encoder’s mutual information with $Y$ approaches $I(H^{*};Y)$, the maximum achievable by the target representation.
This shows that minimising the JEPA pretraining loss implicitly maximises the IB relevance term $I(H_{t};Y)$ for *any* event $Y$ satisfying A1–A4, without knowing $Y$ at pretraining time.
The price paid relative to IB is that JEPA also retains event-irrelevant information (it does not compress), which manifests as higher representation dimensionality but not as reduced downstream performance.

## Appendix B Horizon Interval Design

All methods (HEPA, PatchTST, iTransformer, MAE, Chronos-2) use dense unit-step horizons: $K{=}150$ for C-MAPSS and TEP ($\Delta t\in\{1,2,\ldots,150\}$), and $K{=}200$ for all other datasets ($\Delta t\in\{1,2,\ldots,200\}$). All methods share the same horizon set per dataset, ensuring fair comparison. The h-AUROC evaluation skips degenerate horizons (event prevalence $<0.001$ or $>0.999$).

## Appendix C Baseline Comparison Protocol

[Table˜1](#S5.T1) compares HEPA against four baselines under a matched protocol designed to isolate the effect of the pretraining objective.
All five methods share:

- •
Matched downstream capacity: a horizon-conditioned MLP that maps a frozen representation $\mathbf{h}_{t}$ and horizon $\Delta t$ to per-horizon event logits (details below), trained with positive-weighted BCE ([section˜3.2](#S3.SS2)).
- •
Identical evaluation: h-AUROC averaged over non-degenerate horizons, same train/val/test splits, same horizon sets ($K{=}150$ or $K{=}200$).
- •
Identical label budgets: 100% and 10% label fractions with the same subsampling procedure.

#### Downstream head architecture.

For HEPA, the finetuned component is the pretrained predictor MLP (a 3-layer MLP mapping $[\mathbf{h}_{t};\Delta t]\to\hat{\mathbf{h}}\in\mathbb{R}^{256}$; 197.6K params) plus a shared linear event head (LayerNorm + linear $\to$ logit; 769 params), totalling 198K finetuned parameters. The predictor is initialised from pretraining; however, [section˜I.2](#A9.SS2) shows that this initialisation carries at most $+0.003$ h-AUROC over random initialisation, confirming that the benefit comes from the frozen encoder, not the predictor’s starting weights.

For all baselines (PatchTST, iTransformer, MAE, Chronos-2), we replace the predictor and event head with a single dt-conditioned MLP head: a linear projection from $d_{\text{input}}$ to 256, a learned horizon embedding (256 entries $\times$ 256 dims), followed by LayerNorm and a 3-layer MLP ($256\to 256\to 256\to 1$). With $d_{\text{input}}{=}256$ this totals 264K parameters, giving the baselines slightly *more* downstream capacity than HEPA’s 198K.

#### Encoder training.

The methods differ only in how the frozen encoder is obtained:

PatchTST : channel-independent patching with bidirectional self-attention, trained end-to-end (supervised, no self-supervised pretraining) on each dataset. Same patch size ($P{=}16$) and context length (512 steps) as HEPA. 2.26M trained parameters; at evaluation time the encoder is frozen and the baseline MLP head is trained from scratch. 5 seeds.

iTransformer : inverted Transformer that treats each variate (sensor channel) as a token and applies self-attention across variates rather than across time steps. Trained end-to-end (supervised) on each dataset with the same context window and horizon set. At evaluation the encoder is frozen and the baseline MLP head is trained. 5 seeds.

MAE (masked autoencoder): uses the same causal Transformer encoder architecture as HEPA but pretrained with a masked reconstruction objective: random patches are masked and the decoder reconstructs the masked input values. This isolates the effect of predicting future *representations* (JEPA) versus reconstructing masked *values* (MAE) under an otherwise identical architecture. After pretraining, the decoder is discarded, the encoder is frozen, and the baseline MLP head is trained. 5 seeds.

Chronos-2 : foundation model (119M parameters) pretrained on a large external time-series corpus. Representations are extracted per-channel (univariate) and mean-pooled across channels for multivariate datasets. The baseline MLP head is trained on the frozen features. 3 seeds. This is the only method that uses external pretraining data; all others are trained exclusively on the target dataset.

#### Why this protocol is fair.

By freezing all encoders and training only the downstream head, we ensure that differences in h-AUROC reflect the quality of the learned representations, not differences in head capacity, loss function, or optimisation. The only free variable is the encoder and how it was trained. HEPA finetunes 198K params (predictor + head); baselines train 264K params (dt-MLP head), giving them a slight capacity advantage. HEPA’s predictor initialisation carries negligible benefit ([section˜I.2](#A9.SS2)). PatchTST and iTransformer serve as supervised baselines (no pretraining benefit); MAE serves as a reconstruction-based SSL baseline (same architecture as HEPA, different objective); Chronos-2 serves as a large-corpus foundation model baseline.

## Appendix D Dataset Overview

**Table 3: Dataset overview (14 datasets, 11 domains). Default: patch size $P{=}16$, sliding window of 512 steps (32 tokens), except C-MAPSS which uses full engine history (8–23 tokens per cycle).**
| Dataset | Target event | Sensors | Rate |
| --- | --- | --- | --- |
| C-MAPSS FD001–FD004 | Engine failure | 14 | 1/cycle |
| SMAP | Spacecraft fault | 25 | 1 Hz |
| PSM | Server fault | 25 | 1/min |
| MBA (ECG) | Cardiac arrhythmia | 2 | 275 Hz |
| GECCO (water) | Water contamination | 9 | 1/min |
| ETTm1 | Transformer overheating | 7 | 15/min |
| BATADAL (ICS) | Cyber-attack on SCADA | 43 | 1/hour |
| TEP | Process fault | 52 | 1/3 min |
| Weather | Heat spike | 21 | 10/min |
| Beijing-AQ | PM2.5 spike | 11 | 1/hour |
| VIX | Volatility regime | 6 | 1/day |

## Appendix E Architectural Decisions

| Decision | Value | Source | Status |
| --- | --- | --- | --- |
| Attention mask | Causal | Ablation study | Fixed |
| Horizon sampling | $\text{LogU}[1,150]$ | Initial sweep | Default |
| Target interval | Cumulative $(t,t{+}\Delta t]$ | Architecture design | Default |
| Output param. | Discrete hazard $\to$ CDF | [section˜3.2](#S3.SS2) | Default |
| Target encoder | Joint-trained (weight-shared) | Ablation study | Default |
| Encoder depth | $L=2$ | Initial sweep | Default |
| $d_{\text{model}}$ | $256$ | Initial sweep | Default |
| Loss | L1 on L2-normalised | Initial sweep | Default |

## Appendix F Finetuning-Mode Ablation

[Table˜4](#A6.T4) compares five finetuning strategies on C-MAPSS under an earlier architecture variant (EMA target, 790K pred-FT params). Predictor finetuning is the only mode that remains competitive at both full and low label budgets; scratch training collapses at 5% labels.

**Table 4: Finetuning-mode ablation on C-MAPSS (5 seeds, F1w; earlier architecture variant with EMA target). This variant uses 790K pred-FT params and 2.37M E2E. Pred-FT outperforms scratch by $+0.226$ at 5% labels ($p{=}0.039$, $d{=}1.35$). Scratch training collapses at low label budgets (RMSE 33), confirming pretraining value. Under the final architecture (198K pred-FT, SIGReg), pred-FT and E2E achieve equivalent performance at all label budgets ($\Delta\leq 0.003$).**
|  | 100% labels | 5% labels |  |  |
| --- | --- | --- | --- | --- |
| Mode | F1w $\uparrow$ | RMSE $\downarrow$ | F1w $\uparrow$ | RMSE $\downarrow$ |
| probe_h (257 p) | $0.30\pm 0.06$ | $16.0\pm 1.5$ | $0.06\pm 0.10$ | $20.4\pm 1.2$ |
| frozen_multi (4K p) | $0.15\pm 0.03$ | $19.0\pm 0.1$ | $0.18\pm 0.14$ | $24.4\pm 4.9$ |
| pred_ft (790K p) | $0.39\pm 0.09$ | $16.9\pm 1.7$ | $\mathbf{0.26\pm 0.17}$ | $24.3\pm 6.8$ |
| e2e (2.37M p) | $0.41\pm 0.12$ | $15.0\pm 1.2$ | $0.18\pm 0.24$ | $\mathbf{20.1\pm 1.9}$ |
| scratch (2.37M p) | $0.40\pm 0.08$ | $14.5\pm 0.7$ | $0.04\pm 0.05$ | $32.9\pm 2.0$ |

## Appendix G Foundation Model Comparisons

[Table˜5](#A7.T5) consolidates the matched-head comparison between HEPA (5 seeds) and four time series foundation models: Chronos-2, MOMENT-1-large  (341.2M parameters), TFM-2.5  (203.6M parameters), and Moirai-1.1-R-base  (91.4M parameters). All five encoders are frozen and feed an identical 198K-param dt-conditioned MLP head trained with positive-weighted BCE under the same labels, splits, and evaluation protocol; only the frozen encoder differs.

**Table 5: HEPA vs. four foundation models (matched 198K MLP head, 100% labels). HEPA: 5 seeds. Chronos-2, MOMENT, TFM-2.5, Moirai: 3 seeds each. Bold = best per row. “—”: model not run on that dataset.**
| Dataset | HEPA (5s) | Chronos-2 | MOMENT | TFM-2.5 | Moirai |
| --- | --- | --- | --- | --- | --- |
| C-MAPSS-1 | $\mathbf{0.73\pm.02}$ | $0.66\pm.00$ | $0.56\pm.01$ | $0.53\pm.00$ | $0.61\pm.00$ |
| C-MAPSS-2 | $0.58\pm.01$ | $0.50\pm.01$ | $\mathbf{0.70\pm.00}$ | $0.60\pm.01$ | $0.66\pm.00$ |
| C-MAPSS-3 | $\mathbf{0.82\pm.02}$ | $0.72\pm.02$ | $0.47\pm.01$ | $0.62\pm.01$ | $0.70\pm.00$ |
| SMAP | $\mathbf{0.60\pm.03}$ | $0.53\pm.01$ | — | $0.51\pm.03$ | — |
| PSM | $0.55\pm.02$ | $0.49\pm.00$ | — | $\mathbf{0.57\pm.01}$ | $0.53\pm.01$ |
| MBA | $0.75\pm.01$ | $0.55\pm.01$ | $\mathbf{0.79\pm.01}$ | $0.76\pm.01$ | $0.57\pm.02$ |
| GECCO | $0.81\pm.07$ | $0.81\pm.01$ | — | $\mathbf{0.93\pm.01}$ | $0.82\pm.01$ |
| BATADAL | $0.64\pm.02$ | $0.58\pm.01$ | $0.54\pm.07$ | $\mathbf{0.65\pm.01}$ | $0.36\pm.01$ |
| ETTm1 | $\mathbf{0.87\pm.00}$ | $0.78\pm.01$ | — | $0.59\pm.01$ | $0.60\pm.00$ |

## Appendix H Full MTS-JEPA Comparison

[Table˜6](#A8.T6) compares HEPA against MTS-JEPA  on event-prediction benchmarks from [table˜1](#S5.T1). Both methods use the same context length, patch size, sequence batching, and matched-capacity downstream head; the only difference is the self-supervised objective (HEPA: predictive JEPA + SIGReg; MTS-JEPA: published contrastive + codebook loss). HEPA wins on 8 out of the 9 datasets where MTS-JEPA could be run, with the largest margins on lifecycle and ICS benchmarks (C-MAPSS-1, ETTm1, BATADAL); MTS-JEPA wins on cardiac arrhythmia (MBA). TEP is excluded because the public MTS-JEPA release does not include a chemical-process benchmark and the encoder fails to converge on its $52$-dimensional input.

**Table 6: HEPA vs. MTS-JEPA . Mean ($\pm$ std) over available seeds. HEPA: 5 seeds. MTS-JEPA reproduction: 1–3 seeds. Both methods use identical patch size, sequence length, and matched-capacity downstream head; only the SSL objective differs.**
| Dataset | HEPA | MTS-JEPA |
| --- | --- | --- |
| C-MAPSS-1 | $\mathbf{0.81}$$\pm 0.04$ | $0.69$$\pm 0.02$ |
| C-MAPSS-2 | $\mathbf{0.57}$$\pm 0.01$ | $0.53$$\pm 0.02$ |
| C-MAPSS-3 | $\mathbf{0.84}$$\pm 0.02$ | $0.78$$\pm 0.00$ |
| SMAP | $\mathbf{0.59}$$\pm 0.06$ | $0.49$$\pm 0.00$ |
| PSM | $\mathbf{0.57}$$\pm 0.02$ | $0.48$$\pm 0.00$ |
| MBA | $0.75$$\pm 0.04$ | $\mathbf{0.88}$$\pm 0.00$ |
| GECCO | $\mathbf{0.88}$$\pm 0.06$ | $0.84$$\pm 0.00$ |
| ETTm1 | $\mathbf{0.81}$$\pm 0.01$ | $0.75$$\pm 0.00$ |
| BATADAL | $\mathbf{0.57}$$\pm 0.04$ | $0.43$$\pm 0.00$ |
| TEP | $1.00$$\pm 0.00$ | — |
| HEPA wins | 8 out of 9 |  |

## Appendix I Additional Ablations

### I.1 Does the Predictor Help at Inference?

With all weights frozen, a probe on $\mathbf{h}_{\text{past}}$ alone achieves 0.299 F1w at 100% labels, while a frozen multi-horizon probe on $[\hat{\mathbf{h}}_{1};\ldots;\hat{\mathbf{h}}_{16}]$ achieves only 0.148. The pretrained predictor’s outputs have high cross-horizon cosine similarity ($>$0.98), so the concatenation is nearly redundant. Finetuning the predictor pushes the per-horizon outputs apart: pred-FT’s value is in reshaping the predictor, not in extracting fixed rollouts.

### I.2 Pretrained vs. Random-Init Predictor

To isolate whether the predictor’s pretrained weights carry useful information, we compared pred-FT with pretrained vs. random-initialised predictor weights (encoder frozen, 3 seeds each). On C-MAPSS-1, pretrained weights yield h-AUROC $0.9257\pm 0.0008$ vs. random-init $0.9235\pm 0.0027$ ($p=0.38$). On SMAP, the difference is $0.3874\pm 0.0205$ vs. $0.3950\pm 0.0286$ ($p=0.34$). On MBA, $0.9465\pm 0.0004$ vs. $0.9435\pm 0.0016$ ($p=0.089$). Pretrained predictor weights carry at most $+0.003$ h-AUROC, confirming that the value of pred-FT lies in the pretrained encoder and the finetuning recipe, not in the predictor’s initial weights.

### I.3 Target Encoder Update Rule: Joint Training vs. EMA

Our default target-encoder strategy is joint training: both encoders share weights and are updated by the same optimizer, with a SIGReg regulariser  ($\alpha{=}0.1$) preventing collapse. We compare this against EMA ($\tau{=}0.99$) across all 12 datasets (3 seeds, pred-FT downstream). Joint training wins on 6 out of 12 datasets (largest gain: MBA $+0.108$), loses on 4 (largest loss: SMAP $-0.048$), and ties on 2. All deltas are within $\pm 0.11$ h-AUROC. Predictor finetuning is robust to the target-encoder update rule; we use joint training for its simplicity (no momentum schedule, no separate sync interval).

### I.4 Predictor Dynamics Visualisation

[Figure˜5](#A9.F5) shows how the finetuned predictor transforms the encoder’s representations across prediction horizons. The encoder outputs (blue) cluster by degradation state; as the horizon $k$ increases, the predicted representations shift progressively away from the encoder manifold, indicating that the predictor learns horizon-dependent mappings rather than simply copying the encoder output. This separation confirms that predictor finetuning reshapes the latent space to distinguish event-relevant from event-irrelevant dynamics at each timescale.

Figure: Figure 5: Predictor outputs in latent space (C-MAPSS-1). t-SNE of 256-dimensional representations; axes are the two t-SNE components (arbitrary units). Blue: encoder output $\mathbf{h}_{t}$. Light to dark red: predicted representations at horizons $k=10,50,100$. Outputs shift progressively with $k$.
Refer to caption: https://arxiv.org/html/2605.11130/2605.11130v4/x6.png

## Appendix J Legacy Metrics

C-MAPSS RMSE is derived from $\text{E}[\Delta t]$ of the stored probability surface; this CDF-based estimator differs from the point-prediction protocol of STAR  (replicated RMSE 12.2), so the gap reflects both protocol and SSL-vs-supervised differences. PA-F1 matches the cited baselines’ protocol. HEPA domain-metric values: SMAP PA-F1 $0.88\pm 0.04$ (vs. MTS-JEPA  0.34), PSM PA-F1 $0.95\pm 0.01$ (vs. MTS-JEPA 0.62).

## Appendix K PA-F1 Inflation

PA-F1 (Point-Adjusted F1) credits an entire anomaly segment as correctly predicted if any single timestep within it exceeds the threshold . This protocol inflates reported F1 dramatically when anomaly segments are long and prevalence is non-trivial: the TSAD-Eval study  shows that a random detector can exceed $F_{1}=0.9$ PA on some datasets. We demonstrate this concretely by contrasting PA-F1 and non-PA F1 for HEPA: SMAP 0.862 PA vs. 0.474 non-PA, PSM 0.950 PA vs. 0.575 non-PA. The gap in PA-F1 between HEPA and MTS-JEPA remains large even under a random-baseline sanity check (a random-init encoder achieves 0.604 PA-F1 on SMAP; HEPA is 0.79), so the margin is not purely inflation, but any PA-F1 number should be read alongside the corresponding non-PA number.

## Appendix L Preprocessing Details

| Dataset | Channels | Preprocessing | Context |
| --- | --- | --- | --- |
| C-MAPSS FD001–FD004 | 14 sensors (after drop) | per-subset min-max | cycle-as-patch |
| SMAP | 25 (after drop) | z-score on train | 100 steps |
| PSM | 25 (after drop-constant) | pre-normalised | 100 steps |
| MBA ECG | 2 leads | z-score on train | 100 steps |
| GECCO water | 9 quality sensors | z-score on train | 512 steps |
| BATADAL SCADA | 43 tank/pump/pressure | z-score on train | 512 steps |
| ETTm1 power | 7 load/temperature | z-score on train | 512 steps |
| TEP | 52 process vars | z-score on train | 512 steps |
| Weather | 21 climate vars | z-score on train | 512 steps |
| Beijing-AQ | 11 air quality vars | z-score on train | 512 steps |
| VIX | 6 market vars | z-score on train | 512 steps |

The normalisation, channel-drop thresholds, and context lengths follow conventions in the cited SSL and anomaly-detection benchmarks. For C-MAPSS we use the SELECTED_SENSORS subset (14 sensors after removing near-constant channels). RUL cap is 125 cycles, following STAR . Specific numeric thresholds are in the code repository.

## Appendix M Hyperparameters

All hyperparameters are fixed across datasets unless noted. Seeds $\{0,1,2,3,4\}$ for 5-seed runs and $\{0,1,2\}$ for 3-seed runs. Pretraining trains the encoder and predictor jointly; finetuning freezes the encoder and trains only the predictor and event head.

**Table 7: Hyperparameters. Fixed across all datasets. Pretraining trains encoder + predictor; finetuning trains predictor + event head (encoder frozen).**
| Stage | Hyperparameter | Value |
| --- | --- | --- |
| Pretraining<br>(encoder +<br>predictor) | Optimizer | AdamW |
| Learning rate | $3\times 10^{-4}$ |  |
| Weight decay | $1\times 10^{-2}$ |  |
| Batch size | 64 |  |
| Epochs | 100 (patience 10) |  |
| SIGReg weight $\alpha$ | 0.1 |  |
| Horizon sampling | LogUniform$[1,K]$ |  |
| Finetuning<br>(predictor +<br>event head) | Optimizer | AdamW |
| Learning rate | $1\times 10^{-3}$ |  |
| Weight decay | $1\times 10^{-2}$ |  |
| Batch size | 64 |  |
| Epochs | 50 (patience 10) |  |
| Pos-weight $w^{+}$ | $N_{\text{neg}}/N_{\text{pos}}$ |  |
| Architecture | Encoder layers | 2 |
| $d_{\text{model}}$ | 256 |  |
| Attention heads | 4 |  |
| Patch size $P$ | 16 |  |
| Dropout | 0.1 |  |

## Appendix N Pairwise Significance Tests

[Table˜8](#A14.T8) reports Welch’s two-sided t-test (unpaired, unequal variance) between HEPA and each architectural baseline at each (dataset, label-fraction) cell with at least 3 seeds per arm. Out of 42 cells (14 datasets $\times$ 3 baselines, 100% labels), HEPA wins 21 at $p<0.05$ and loses 7; the remaining 14 are not significantly different.

**Table 8: Pairwise Welch’s t-test, HEPA vs. each baseline. Markers: ^∗ $p{<}0.05$, ^∗∗ $p{<}0.01$ in HEPA’s favour; ^↓ denotes HEPA loses at $p{<}0.05$; blank = not significant.**
|  | FD001 | FD002 | FD003 | FD004 | SMAP | PSM | MBA | GECCO | BATADAL | TEP | ETTm1 | Weather | Beijing | VIX |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vs. PatchTST |  | ^∗∗ | ^∗∗ | ^∗∗ | ^∗ |  |  | ^∗∗ | ^↓ |  | ^∗∗ | ^∗∗ | ^↓ |  |
| vs. iTransformer | ^∗∗ | ^∗∗ | ^∗∗ | ^∗∗ | ^∗∗ | ^∗∗ | ^↓ | ^∗ |  | ^∗∗ |  | ^∗ | ^∗∗ | ^∗ |
| vs. MAE | ^∗∗ |  | ^∗∗ | ^∗∗ | ^↓ |  |  |  | ^↓ |  | ^↓ |  | ^↓ |  |

The dominant wins are on the C-MAPSS family (all four datasets significant against at least one baseline) and against iTransformer (HEPA wins on 11 out of 13). HEPA’s seven losses ($p{<}0.05$ in the baseline’s favour) are concentrated on SMAP, MBA, BATADAL, ETTm1, and Beijing-AQ, where event signatures are sensor-localised or where MAE’s reconstruction objective transfers well.

## Appendix O Calibration Analysis

[Table˜9](#A15.T9) reports Expected Calibration Error (ECE) and Brier score on five datasets for which we hold verified HEPA-SIGReg probability surfaces (single-seed, $s{=}42$; 10 equal-width bins; Murphy decomposition: Brier = Reliability $-$ Resolution $+$ Uncertainty).

**Table 9: Calibration of HEPA probability surfaces on five datasets.**
| Dataset | Base rate | ECE $\downarrow$ | Brier $\downarrow$ | Reliability | Resolution |
| --- | --- | --- | --- | --- | --- |
| FD001 | 0.78 | 0.272 | 0.238 | 0.129 | 0.062 |
| FD004 | 0.34 | 0.030 | 0.124 | 0.001 | 0.102 |
| Weather | 0.05 | 0.196 | 0.124 | 0.082 | 0.009 |
| BeijingAQI | 0.24 | 0.183 | 0.177 | 0.050 | 0.054 |
| VIX | 0.21 | 0.310 | 0.241 | 0.102 | 0.026 |

Calibration varies substantially: HEPA is well-calibrated on FD004 (ECE 0.030) but miscalibrated on FD001 (ECE 0.272) and VIX (ECE 0.310). The pattern is consistent with the loss design: positive-weighted BCE applied to the cumulative survival CDF up-weights the minority class to preserve recall, distorting the probability scale. Resolution remains positive on every dataset, confirming the surface still discriminates events even when miscalibrated. For applications requiring calibrated probabilities, post-hoc Platt scaling or isotonic regression on a held-out fold is recommended; we verified offline that this reduces ECE on FD001 to below 0.05.
