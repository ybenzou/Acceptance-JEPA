# Process-Set JEPA for Factory Acceptance APM

## 1. Research decision

This study targets image-only, module-level automatic progress monitoring (APM) from cumulative factory acceptance evidence.

The first-stage method is a resource-constrained, JEPA-style latent predictive process model. It uses a frozen pretrained visual backbone and trains a hierarchical evidence-set encoder, future-latent predictor, and supervised APM head. The study does not initially claim a general world model.

## 2. Evidence and scope

The current inventory identifies:

- 692 candidate modules;
- 8 shared subsections: `12.10`, `12.11`, `12.13`, `13.1`, `13.2`, `13.4`, `14.1`, and `16.1`;
- 23,747 image segments, with at least 10 images per selected module-procedure pair;
- complete `createTime` coverage.

The accepted data assumptions are:

- `createTime` reflects the true capture/acceptance order;
- the eight subsections form a partial order rather than a fixed chain;
- a recorded acceptance event means that the corresponding procedure was completed and accepted;
- inference receives images and relative time only, never `subsectionId`;
- evaluation holds out complete `moduleId` values within the same project/factory domain.

These assumptions must be checked in the trajectory audit before model training.

## 3. Task definition

For module \(m\), sort accepted inspection events by time:

\[
E_m = \{e_1, e_2, \ldots, e_K\},
\qquad
e_k = (\text{images}_k, \text{time}_k).
\]

At snapshot \(k\):

- input: all accepted images observed through event \(e_k\), grouped by event and accompanied by relative times;
- output: an eight-dimensional completion-probability vector;
- auxiliary target: the latent representation of the next accepted event \(e_{k+1}\).

The main task is current cumulative progress-state estimation. Future prediction is an auxiliary representation-learning mechanism, not the deployment output.

## 4. Data construction

Split modules into train, validation, and test sets before generating snapshots. Use a deterministic, prefix-stratified 70/15/15 split so that every image and snapshot from one module remains in one partition.

For each `moduleId × subsectionId`:

1. Inspect timestamp dispersion and record identifiers.
2. Group records into acceptance events using the database's acceptance identifier when available; otherwise select a temporal clustering threshold from the training-set within-event and between-event gap distribution, then freeze it before validation and test construction.
3. Exclude records that cannot be verified as accepted.
4. Deduplicate exact files and near-duplicate images.
5. Assign the event time from the verified acceptance timestamp, or the latest timestamp in the accepted cluster when only image times exist.

Generate one snapshot after each accepted event. Require at least one future event for JEPA training; retain terminal snapshots for APM-only training and evaluation.

To control evidence-count shortcuts:

- sample at most four images per event during training;
- use deterministic four-image sampling per event for validation and testing;
- report a separate all-images sensitivity analysis;
- retain event boundaries instead of flattening all images.

## 5. Model architecture

### 5.1 Frozen image encoder

Precompute image embeddings with a frozen DINOv2-S or similarly sized pretrained ViT. Use CLIP as an optional backbone sensitivity check. Frozen embeddings make the first-stage experiment feasible on an RTX 3060 with 12 GB VRAM.

### 5.2 Hierarchical process-set encoder

The process-set encoder has two levels:

1. Event pooling aggregates the sampled image embeddings within one acceptance event.
2. Temporal attention aggregates event representations up to snapshot \(k\), using relative-time embeddings and event positions.

It outputs current module state \(z_k\). It receives no procedure identifiers.

### 5.3 Target branch and predictor

An exponential-moving-average target projector encodes the next event as stop-gradient target \(\bar z_{k+1}\). A predictor estimates:

\[
\hat z_{k+1} = p(z_k, \Delta t).
\]

The first experiment uses one-step deterministic prediction. Multi-hypothesis prediction is introduced only if errors show systematic latent averaging at valid process branches.

### 5.4 APM head and process constraint

A multi-label head predicts:

\[
\hat y_k = h(z_k) \in [0,1]^8.
\]

A fixed expert-validated partial-order graph penalizes predictions in which a successor is complete while a required predecessor is incomplete. The graph is a process prior, not image metadata.

## 6. Training objective

\[
L =
L_{\mathrm{APM}}
+ \lambda_{\mathrm{future}} L_{\mathrm{JEPA}}
+ \lambda_{\mathrm{graph}} L_{\mathrm{violation}}.
\]

- \(L_{\mathrm{APM}}\): class-balanced binary cross-entropy over the eight completion labels.
- \(L_{\mathrm{JEPA}}\): cosine distance between L2-normalized predicted and target future latent.
- \(L_{\mathrm{violation}}\): differentiable penalty over expert-defined predecessor-successor constraints.

Tune the two loss weights on validation macro-F1, with graph violation rate as a secondary selection criterion. The test set is evaluated once after configuration selection.

## 7. Comparisons

All methods use the same split and snapshot manifest:

1. Non-visual baseline using elapsed time, event count, and image count.
2. Static frozen-DINOv2 baseline using mean pooling over all observed image embeddings and an MLP completion head.
3. Supervised hierarchical set encoder with \(\lambda_{\mathrm{future}}=0\).
4. Full Process-Set JEPA.
5. Optional CLIP-backbone version to test backbone dependence.

The key comparison is method 4 versus method 3. Architecture, optimizer, sampling, and APM supervision must otherwise remain identical. Target-domain I-JEPA visual pretraining is reserved for the larger-compute second stage and is not required for the first-stage claim.

## 8. Ablations

Run:

- no future loss;
- future targets shuffled within each module;
- relative-time input removed;
- process-graph penalty removed;
- true future target replaced by another module's target from the same procedure;
- single-image input instead of cumulative evidence;
- event boundaries removed and all images pooled as one flat set.

The shuffled-future ablation tests whether temporal structure matters. The cross-module same-procedure ablation separates generic procedure prototypes from module-specific evolution.

## 9. Evaluation

Report:

- macro-F1 and per-procedure F1;
- per-procedure AUROC where both classes are present;
- completion-vector exact match;
- Hamming loss;
- temporal monotonicity violation rate;
- partial-order violation rate;
- Brier score and expected calibration error;
- performance as a function of observed evidence fraction.

Use module-level bootstrap resampling for 95% confidence intervals. Compare paired module-level scores with a permutation test. Report results by module prefix and evidence volume as failure-analysis strata, without claiming cross-project generalization.

## 10. Stop/go gates

### Data gate

Proceed only if most selected modules form credible acceptance trajectories and timestamp ordering agrees sufficiently with the expert partial-order graph. Otherwise redefine the task around observed inspection evidence rather than true completion.

### Visual-information gate

Proceed to JEPA claims only if the supervised visual set encoder outperforms the non-visual time/count baseline on held-out modules.

### Predictive-mechanism gate

Claim process-dynamics benefit only if full Process-Set JEPA:

- outperforms the identical no-future-loss model with module-level uncertainty reported; and
- performs better with true future targets than with shuffled future targets.

If either condition fails, report future latent prediction as not demonstrated to improve APM and remove the world-model narrative.

## 11. Claim boundaries

Claims allowed when supported:

- a hierarchical evidence-set model for sparse, asynchronous, multi-view acceptance trajectories;
- image-only cumulative progress estimation on held-out modules from the same project/factory domain;
- future latent prediction contributes process information, but only after passing the predictive-mechanism gate;
- graph constraints reduce invalid process-state combinations, if confirmed by ablation.

Claims not allowed in the first-stage study:

- zero-shot inference;
- general factory inspection auditing;
- a general physical world model;
- causal understanding of construction actions;
- cross-project generalization;
- industrial deployment readiness.

## 12. Resource-aware progression

Stage 1 uses offline frozen embeddings and a compact trainable process model on the local 12 GB GPU. Stage 2 fine-tunes the last visual blocks or a parameter-efficient adapter only after all three gates pass and larger compute is available. Stage 3 adds multi-step latent rollout and audit inconsistency detection only after one-step prediction is validated.

## 13. Literature anchors

- Assran et al., “Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture,” CVPR 2023: static masked latent prediction for semantic image representation.
- Assran et al., “V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning,” arXiv:2506.09985, 2025 preprint: video JEPA and frozen-encoder action-conditioned latent prediction.

These works motivate the architecture but do not establish effectiveness for modular-construction APM. That effectiveness remains the hypothesis tested by this design.
