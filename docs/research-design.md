# Action-Conditioned JEPA for Semantic and Process Auditing

## 1. Research decision

This study audits whether a factory acceptance record is supported by its submitted visual evidence and by the previously verified process history of the same module.

The primary task is not automatic progress reconstruction. The database `subsectionId` is treated as a claim to verify, not as hidden information to predict. Progress is a downstream ledger updated only after a record is accepted by the audit process.

The proposed method is an action-conditioned, JEPA-style latent predictive model. It combines multi-view visual evidence, verified event history, elapsed time, and an authoritative partial-order graph derived from the factory SOP.

## 2. Practical motivation and evidence boundary

The current data inventory identifies:

- 692 candidate modules;
- 8 shared subsections: `12.10`, `12.11`, `12.13`, `13.1`, `13.2`, `13.4`, `14.1`, and `16.1`;
- 23,747 image segments, with at least 10 images per selected module-procedure pair;
- complete `createTime` coverage.

The records are assumed to be mostly clean: most accepted records pair the correct subsection with relevant images, while a small unknown fraction may contain errors.

No expert-verified anomaly corpus is currently available. Therefore, the first study evaluates controlled semantic and process corruptions applied to real acceptance records. It does not establish real-world anomaly prevalence, fraud detection, or industrial audit performance.

## 3. Task definition

For the \(k\)-th submitted acceptance record of module \(m\), define:

\[
\mathcal{R}_k =
(a_k, X_k, H_{k-1}, \Delta t_k, G),
\]

where:

- \(a_k\) is the record's claimed `subsectionId`;
- \(X_k\) is the set of images submitted with the current record;
- \(H_{k-1}\) is the sequence of previously verified events, times, procedures, and visual states;
- \(\Delta t_k\) is the elapsed time since the previous verified event;
- \(G\) is the authoritative partial-order graph.

The model evaluates whether the current visual evidence supports the claimed procedure in the context of the module's verified history.

### 3.1 Required outputs

Each audit result uses a fixed structured schema:

```json
{
  "inferred_procedure_distribution": {},
  "claimed_subsection_compatibility": 0.0,
  "evidence_decision": "supported|unsupported|insufficient",
  "sequence_consistency": {
    "legal": true,
    "violated_preconditions": []
  },
  "expected_observed_latent_distance": 0.0,
  "supporting_images": [],
  "contradicting_images": [],
  "confidence": 0.0
}
```

The system produces an audit priority and evidence trace. It does not automatically accuse a record of fraud or replace the final human decision.

## 4. Data construction

Split complete `moduleId` values into deterministic, prefix-stratified train, validation, and test partitions before constructing events or corruptions. Use a 70/15/15 split unless the trajectory audit shows that a different allocation is required for per-procedure coverage.

For each `moduleId × subsectionId`:

1. Verify that the record represents an accepted event.
2. Inspect timestamp dispersion and available event identifiers.
3. Group images by the database event identifier when available.
4. If event identifiers are unavailable, estimate a clustering threshold from training-set time-gap distributions and freeze it before constructing validation and test data.
5. Deduplicate exact files within an event while preserving duplicate counts for later data-quality analysis.
6. Order events using the verified acceptance time, or the latest image timestamp when no separate acceptance time exists.

The trajectory audit must compare empirical event order with the authoritative SOP graph. Frequent but SOP-legal alternative orders remain valid; observed orders must not be used to redefine normative legality.

## 5. Model architecture

### 5.1 Frozen visual backbone

Use a frozen, compact pretrained ViT such as DINOv2-S to produce per-image embeddings. Precompute embeddings offline so the first-stage experiment is feasible on an RTX 3060 with 12 GB VRAM.

CLIP may be evaluated as a backbone sensitivity check. End-to-end visual fine-tuning is reserved for a later stage after the predictive mechanism passes its claim gate.

### 5.2 Multi-view evidence encoder

An event set encoder aggregates the variable-size image set \(X_k\):

\[
z_k = E_{\mathrm{set}}(X_k).
\]

The encoder must preserve per-image representations for evidence attribution. Training uses random view subsampling and subset consistency so that the event representation does not depend on one fixed image count or ordering.

### 5.3 History encoder

A temporal set or transformer encoder represents the previously verified history:

\[
h_{k-1} = E_{\mathrm{history}}(H_{k-1}).
\]

Each historical event includes its visual event latent, accepted procedure, relative time, and position. The current record is excluded from history.

### 5.4 Action-conditioned latent predictor

The predictor receives the verified history, current claimed procedure, elapsed time, and graph-derived procedure representation:

\[
\hat z_k =
P(h_{k-1}, a_k, \Delta t_k, G).
\]

An exponential-moving-average target branch produces the stop-gradient observed target \(z_k\). The audit nonconformity includes:

\[
d_k =
1-\cos(\hat z_k,z_k).
\]

This is a predicted post-action evidence state: it predicts what visual evidence should be observed after the claimed procedure, conditional on the previous verified state.

### 5.5 Independent semantic verifier

A procedure head predicts a distribution directly from the current evidence:

\[
p_k = C(z_k).
\]

The probability assigned to \(a_k\), the distribution entropy, and the disagreement between \(p_k\) and \(a_k\) provide semantic compatibility evidence independent of process history.

### 5.6 Deterministic process verifier

The SOP graph checks whether \(a_k\) is legal given the previously verified procedures. Sequence legality and violated preconditions are deterministic rule outputs, not claimed as learned neural reasoning.

### 5.7 Selective audit and evidence attribution

The audit decision combines calibrated semantic nonconformity, latent prediction error, cross-view disagreement, and process legality.

Calibration primarily uses the clean validation distribution. The model returns `insufficient` when uncertainty or cross-view disagreement exceeds the selected clean-validation threshold.

Supporting and contradicting images are identified through leave-one-image-out score changes. Raw attention weights are not treated as explanations.

## 6. Training objective

Train primarily on original, mostly-clean records:

\[
L =
L_{\mathrm{procedure}}
+ \lambda_{\mathrm{JEPA}}L_{\mathrm{latent}}
+ \lambda_{\mathrm{subset}}L_{\mathrm{view\text{-}consistency}}.
\]

- \(L_{\mathrm{procedure}}\): class-balanced cross-entropy for the accepted subsection.
- \(L_{\mathrm{latent}}\): cosine distance between predicted and stop-gradient observed event latents.
- \(L_{\mathrm{view\text{-}consistency}}\): consistency between embeddings and predictions from different subsets of the same event's images.

Use robust loss trimming or bounded sample weights for the highest-loss training records to reduce sensitivity to a small unknown fraction of incorrect accepted records. Report the trimming rate and include a no-trimming ablation.

The primary setting does not train a binary classifier on synthetic corruptions. A secondary supervised-corruption setting may be reported separately, but it cannot replace the normal-only evaluation.

## 7. Controlled audit benchmark

Keep each clean test record and generate paired corrupted variants only after module-level splitting.

### 7.1 Graph-illegal claim swap

Replace \(a_k\) with a subsection that violates an SOP predecessor constraint. This tests deterministic graph verification and is not sufficient to demonstrate JEPA value.

### 7.2 Graph-legal semantic claim swap

Replace \(a_k\) with a different subsection that remains legal under the same history but is inconsistent with \(X_k\). This is the primary semantic audit test because an SOP-only baseline cannot detect it.

### 7.3 History corruption

Remove a required historical event or perturb history while retaining the current record. This tests whether the process predictor uses the verified trajectory rather than only the claimed procedure.

### 7.4 Controlled evidence insufficiency

Progressively remove images or retain only highly similar views. This measures whether uncertainty increases as controlled visual evidence is reduced. It does not establish real-world evidence sufficiency without expert annotations.

Cross-module image replacement, duplicate reuse, local manipulation, and forgery detection are excluded from the first study because they require provenance or forensic mechanisms distinct from semantic and process auditing.

## 8. Baselines

All methods use the same module split, event manifest, corruption manifest, and frozen visual backbone:

1. SOP graph only.
2. Multi-view procedure classifier only.
3. Procedure classifier plus SOP graph.
4. History encoder without latent prediction.
5. Full action-conditioned JEPA auditor.
6. Optional structured VLM auditor using the same images, claim, history summary, and SOP description.

The key comparison is the full model against classifier plus SOP on graph-legal semantic corruptions.

## 9. Required ablations

Run:

- remove the JEPA latent objective and predictor;
- remove history from the predictor;
- shuffle history between modules;
- shuffle claimed procedure tokens;
- remove elapsed time;
- remove graph-derived representations from the learned predictor while retaining the deterministic legality checker;
- replace the target with another module's event from the same subsection;
- remove subset-consistency training;
- remove robust loss trimming.

If the full model retains the same performance after history shuffling, it has not demonstrated process-conditioned evidence prediction.

## 10. Evaluation

### 10.1 Procedure semantics

- macro-F1;
- per-procedure precision, recall, and F1;
- expected calibration error;
- confusion matrix.

### 10.2 Audit compatibility

Report separately for every corruption type:

- AUROC;
- AUPRC;
- FPR at 95% TPR;
- clean false-alarm rate;
- paired clean-versus-corrupted effect size.

### 10.3 Selective prediction

- risk-coverage curve;
- area under the risk-coverage curve;
- coverage at fixed accepted risk;
- uncertainty trend under progressive evidence removal.

### 10.4 Explanation faithfulness

Measure the change in compatibility after removing images identified as supporting or contradicting. Do not claim human-interpretable localization without expert explanation labels.

### 10.5 Statistical protocol

Use module-level bootstrap resampling for 95% confidence intervals and paired permutation tests for method comparisons. Begin with clean oracle history. Report closed-loop history as a separate stress test so error propagation is not hidden.

## 11. Claim gates

### Data gate

Proceed only if accepted records form credible module trajectories and the authoritative graph covers the selected procedures.

### Semantic gate

The multi-view classifier must distinguish selected procedures on held-out modules with per-class performance reported. Otherwise claimed-subsection compatibility is not established.

### Predictive-mechanism gate

The full model must:

- outperform classifier plus SOP on graph-legal semantic corruptions;
- lose a measurable advantage when history is shuffled or removed;
- maintain a controlled false-alarm rate on clean test records.

If these conditions fail, latent prediction has not demonstrated added audit value beyond static semantic classification and deterministic rules.

### Selective-audit gate

Uncertainty must increase and coverage must decrease predictably under controlled evidence removal. Otherwise `insufficient` is not a validated output.

## 12. Claim boundaries

Claims allowed when directly supported:

- a controlled benchmark for semantic and process corruption of real factory acceptance records;
- a multi-view, history-conditioned method for testing whether evidence supports a claimed subsection;
- improved detection of graph-legal semantic corruption, if the predictive-mechanism gate passes;
- calibrated abstention under controlled evidence removal, if the selective-audit gate passes;
- deterministic SOP-based detection of illegal procedure claims.

Claims not allowed in the first-stage study:

- detection of real fraud, forgery, or malicious manipulation;
- estimates of real anomaly prevalence;
- industrial audit readiness;
- complete automatic progress monitoring;
- cross-project generalization;
- causal understanding of construction actions;
- a general physical world model.

## 13. Resource-aware progression

Stage 1 uses frozen offline image embeddings and compact event, history, and predictor networks on the local 12 GB GPU.

Stage 2 fine-tunes the last visual blocks or parameter-efficient adapters only after the semantic and predictive gates pass.

Stage 3 requires an expert-verified natural anomaly set before making real-world auditing claims. Duplicate reuse and image manipulation should be developed as separate forensic modules rather than added to the semantic JEPA model.

## 14. Literature anchors

- Assran et al., “Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture,” CVPR 2023: static latent prediction for semantic visual representation.
- Assran et al., “V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning,” arXiv:2506.09985, 2025 preprint: video JEPA and frozen-encoder action-conditioned latent prediction.
- Martínez et al., “A vision-based approach for automatic progress tracking of floor paneling in offsite construction facilities,” *Automation in Construction*, 2021: evidence that visual monitoring is an established offsite-construction problem.
- Panahi et al., “Automated Progress Monitoring in Modular Construction Factories Using Computer Vision and Building Information Modeling,” ISARC 2023: factory validation and practical limitations involving occlusion and manual setup.

These works motivate visual process monitoring and latent prediction. They do not establish semantic/process auditing of factory acceptance records; that remains the hypothesis evaluated by this design.
