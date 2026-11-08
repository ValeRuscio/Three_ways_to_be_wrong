# Review-driven experiment suite

This directory contains the original repository unchanged, plus the `revision/` package.
Run commands from this repository root. No experiments have been pushed to GitHub.
The supplied smoke data are synthetic software checks, not paper results.

## Start here

1. Read **Scientific issues found in the original code**, below.
2. Install dependencies and run the CPU smoke workflow.
3. Export the actual paper cohorts with explicit decision points, source spans,
   relations, and group IDs. The GitHub repository does not contain those data.
4. Run extraction once per model and task. Run robustness analysis on those saved
   measurements without loading the model again.
5. Run matched baselines and internal interventions on a prespecified subset before
   committing to the full compute budget.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r revision/requirements.txt
bash revision/run_smoke.sh results/revision_smoke
```

For a CUDA machine install a compatible PyTorch build. Use fp32 and eager attention;
do not substitute quantization or bfloat16 without revisiting the reconstruction
contract. The smoke workflow downloads no pretrained models. It trains tiny models
for only two updates and fabricates target labels; none of its numbers are evidence
for or against the paper. Full commands below use actual model weights and cohorts.

## What is implemented

| Module | Experiment or function |
|---|---|
| `revision.prepare` | Explicit group-disjoint calibration/evaluation split for legacy cohorts, with exclusion audit |
| `revision.extract` | Certified input-origin transport; legacy edge-sum comparison; source-span variants; intermediate-cut transport; cached lens scores; monotone and fixed-dose readout repairs |
| `revision.robustness` | Threshold, calibration-group count/resampling, decoy count/difficulty, span, readability-window sensitivity; leave-relation-out calibration; per-example stability and transition counts |
| `revision.predict` | Grouped held-out utility beyond logits, rank, confidence and dose; train-fold-only preprocessing; paired cluster-bootstrap loss differences |
| `revision.controlled` | Paired toy easy/decoy/competing/hard settings; complete candidate rankings; gradient × activation, matched gate gradient, path IG, exact removal oracle |
| `revision.natural_ranking` | Same matched-removal protocol for natural Llama/Qwen2 cohorts, with optional score-independent random candidate subsampling |
| `revision.internal` | Verified successful-donor residual patches, random and optional wrong-donor norm-matched controls, multiple layers/doses; AtP and candidate-specific IG estimates |
| `revision.internal_report` | Cross-label repair matrix, paired controls, group sign-flip tests and Holm adjustment |
| `revision.stage_controls` | Known-operation controls: input corruption, transport suppression, competitor-only output bias; records failed and ineffective interventions |
| `revision.report` | Vector SVG/PDF plots, ranking summaries and paired uncertainty |
| `revision.tests` | Numerical reconstruction, affine additivity, derivatives, intervention identity and cleanup, split leakage, preprocessing and statistics tests |

## Scientific issues found in the original code

These are reasons to audit and rerun results, not evidence that all published results
are incorrect. The submitted PDF and this public code may represent different revisions.

1. **Transport definitions differ.** `validation_suite.calibrate` and `label_failure`
   sum `api.tdla` attention-edge scores. Equation 8 instead propagates a labelled
   input component through the linear parts of the frozen maps. The new suite keeps
   them separate as `legacy_edge_sum` and `provenance:0`. Edge-message scores summed
   across layers are not automatically an additive input-origin partition.
2. **The old sensitivity script targets another quantity.** `run_sensitivity.py`
   varies the constrained obstruction solver; it does not resample calibration,
   decoys and spans for the main tau/share classification rule.
3. **Prediction preprocessing leaks information.** `run_predictive.py` standardizes
   the full dataset before cross-validation. The new code fits scaling, imputation
   and categorical encoders only on each training fold and holds whole groups out.
4. **Decision-point and correctness implementations need reconciliation.**
   `build_cohort.process` uses the first answer token and prompt-end logits rather
   than implementing semantic divergence alignment. `word_level_correct` accepts
   alias substrings. The new pipeline does not silently claim to solve semantic
   alignment: provide audited `input_ids` for the intended decision point, or
   explicitly run the first-token protocol. It rejects failed records whose target
   is actually top-1 at the supplied position. Correctness annotations themselves
   still require your audit.
5. **The decoy implementation does not guarantee zero target value.**
   `known_circuit2.decoy_edit` moves attention mass to a filler position. It does not
   explicitly zero that position's value projection. A filler can acquire information
   in hidden states. The new tests call this an attention-decoy treatment, not a
   planted zero-value circuit. Annotated answer positions are task ground truth,
   not a proof of the learned model's unique causal route.
6. **Interventions previously compared different operations.** Original head-context
   gradient × source message predicts unrenormalized message removal; the ablation
   renormalizes attention. The new gate gradient differentiates the exact gate-and-
   renormalize operation used by its oracle. The legacy score remains a separate
   comparator, not overwritten.
7. **Readout edits have changed semantics deliberately.** Original projection-to-
   median code can lower an already-high answer projection. The new `answer_up` and
   `alt_down` are monotone (clamped displacement). `combined` adds both displacements
   from the original state, rather than sequentially resetting projections. These
   are new experiments, not bitwise reproductions of Table 2. Random controls match
   the answer-up displacement norm. Fixed-dose controls match their declared norm.

The original scripts are retained so you can reproduce legacy results separately.
Do not combine legacy and new results without naming the protocol differences.

## Cohort and decoy schema

One model and one task per extraction run. Each line of `cohort.jsonl` is:

```json
{"id":"example-001","group_id":"subject-Maryland","split":"evaluation","correct":false,"relation":"capital","task":"recall","decision_scope":"first_token","prompt":"The capital of Maryland is","target_first_token":123,"competitor_token":456,"source_span":[4,5]}
```

The token IDs and span above are schematic: replace them with actual IDs under the
selected model's tokenizer. `source_span` uses inclusive token indices, including
any automatically inserted BOS token. The extractor validates the actual competitor.
Use `input_ids` instead of relying on prompt retokenization when alignment matters.
For `decision_scope="aligned_divergence"`, `input_ids` must contain the entire shared
prefix through the decision point, with the target token updated to that point.

`group_id` must join related observations: same underlying subject/fact, alternate
prompt versions and (when pooling models) copies of the same fact across models.
Group definitions should be chosen before looking at outcomes. Calibration groups
and evaluation groups must be disjoint. Calibration records must be successful.
Keep relations and tasks explicit; missing metadata is an error, not a guessed label.
For extraction, token identity counts only if the target token is inside the tested
span. A random control span does not automatically pass the source criterion.

A separate `decoys.json` maps relation names to token IDs:

```json
{"capital":[100,101,102,103,104,105],"author":[200,201,202,203,204,205]}
```

Build these from an independent type-matched answer inventory, excluding target
aliases and inappropriate first-token collisions. The extractor removes the exact
target token but cannot identify semantic aliases from IDs. Document the inventory,
provenance and tokenization. Never construct the pool from held-out evaluation
outcomes. Empty pools are rejected; requested sizes larger than the available pool
are explicitly excluded and recorded in coverage outputs.

If the old cohort has appropriate metadata:

```bash
python -m revision.prepare --cohort cohorts/llama_recall.jsonl \
  --task recall --decision-scope first_token \
  --calibration-fraction 0.30 --out results/llama_recall_split
```

This conservatively reserves entire groups for calibration, then drops failures in
those reserved groups. It does not claim to preserve the paper's original cohort.
For maximum comparability, export a new audited cohort directly with explicit splits.

## 1. Measure definitions, spans and layer cuts

```bash
python -m revision.extract \
  --model meta-llama/Llama-3.2-1B \
  --cohort results/llama_recall_split/cohort.jsonl \
  --decoy-pool cohorts/llama_recall_decoys.json \
  --out results/revision_llama_recall/measured --device cuda
```

The extractor measures one example at a time, writes it atomically, and releases the
activation cache. It resumes completed records only when configuration, input files
and source-code hashes match. Set the Hugging Face cache directory with `HF_HOME`
if needed. Access to gated model weights must already be authorized in your account.
Use immutable model snapshots for publication; record their resolved revision.

Definitions:

- `provenance:0`: the input-origin component in Equation 8. Each step uses
  `frozen_layer(x) - frozen_layer(0)` to avoid assigning affine offsets to every source.
- `legacy_edge_sum`: original repository attention-edge score sum.
- `provenance:k`: full state at the source span at cut k propagated downstream. This
  is a different source convention; it is not automatically a path for the particular
  decoded fact. Default cuts are 0, floor(L/4), floor(L/2), floor(3L/4).
- Source + complement + propagated biases must reconstruct every residual cut.

The source score is max over positions/windows of the minimum decoy margin within
that same consecutive-layer window. A positive score is exactly the stated
same-position readability criterion. Readability is not proof of stored knowledge.

The original, one-token-wider, first-token, last-token and disjoint random spans are
included. Wider spans can absorb other content: interpret this as sensitivity, not
necessarily a better explanation.

## 2. Main robustness and calibration transfer

```bash
python -m revision.robustness \
  --measurements results/revision_llama_recall/measured \
  --out results/revision_llama_recall/robustness \
  --repeats 200 --sizes 20 40 60 --quantiles 0.05 0.10 0.20 \
  --decoy-sizes 5 10 20 --consecutive 1 2 3 --transfer
```

`--sizes` counts independent calibration groups. Each bootstrap draws that many
groups with replacement and retains their records. A full-calibration-size bootstrap
is always included; requests exceeding available groups are skipped. Raw outputs
include actual sampled example counts. Intervals describe calibration/decoy uncertainty
conditional on the fixed evaluation cohort, not population uncertainty in task shares.

`random` samples decoys without replacement. `hard` deliberately chooses those with
highest observed source-lens scores; this is a declared adversarial sensitivity test,
not an unbiased estimate of deployment performance. The same selected pool is used
through the layer-window test for an example.

`--transfer` calibrates using successful examples from other relations in the same
task. It never calibrates on failures from the held-out relation. If other relations
have no usable successes, that configuration is omitted; inspect coverage. To examine
prompt-condition or difficulty transfer, create a separate cohort where `relation`
encodes the prespecified domain of interest and supply the matching decoy pools.
Do not collapse incompatible semantic decoy types merely to enable the test.

Outputs include replicate proportions, uncertainty summaries, transitions, per-example
stability, exclusions, continuous reference features, and readout repair outcomes.
The reference uses original span, input provenance at cut zero, two consecutive
layers, 10th-percentile thresholds, and the smallest requested random decoy size with
the fixed seed. Its exact definition is recorded in `interpretation.json`.
To compare intermediate cuts, pass e.g. `--definitions provenance:0 provenance:8
legacy_edge_sum`, replacing 8 with an actually recorded cut. Sweep `--epsilon`
in separate output directories to test near-zero denominator exclusions.

## 3. Incremental utility and matched doses

```bash
python -m revision.predict \
  --features results/revision_llama_recall/robustness/reference_features.csv \
  --outcomes results/revision_llama_recall/robustness/repairs.csv \
  --arm answer_up --out results/revision_llama_recall/utility
```

Repeat for `combined`, `alt_down`, or `fixed_answer_1`. The extraction default fixed
norms are 0.25, 0.5, 1 and 2 in the normalized residual space. These are explicitly
chosen perturbation scales, not cross-model comparable effective doses. Actual norm,
gold-logit change and margin change are recorded. Multiple `--features` and matched
`--outcomes` paths allow pooled models while keeping shared group IDs in one fold.

Models are fixed-regularization logistic regression, with numerical imputation,
standardization and categorical encoding fit on training folds only. Four tiers:
logits; logits+dose; logits+dose+source/attention; all plus transport. Report the
increment over the source/attention tier as well as over output-only features.
Single-class training folds use a documented smoothed-prevalence fallback. Inspect
class counts and notes; do not interpret a degenerate task as useful prediction.

The paired bootstrap uses fixed out-of-fold predictions, so it does not capture
training/refitting uncertainty. Use repeated grouped splits as a sensitivity analysis
with different `--seed` and fresh output directories. No hyperparameter tuning is
performed on the test folds. Labels themselves are never prediction targets here.
External binary outcomes can be provided as CSV (`id`, target column); use
`--target column_name`. Continuous outcomes use `--regression`. Missing outcomes
are dropped and never imputed.

Final-readout repair is partly determined by output geometry. A positive utility
result is not independent identification of a causal internal stage; test internal
outcomes too if making stronger claims. Dose-conditioned prediction is a diagnostic
control, not necessarily a feature set available before an intervention is selected.

## 4. Stronger controlled and natural baselines

```bash
python -m revision.controlled --out results/revision_controlled \
  --seeds 12 --steps 2500 --examples 60 --ig-steps 16 --device cuda
python -m revision.report --candidates results/revision_controlled/candidates.csv \
  --out results/revision_controlled/report

python -m revision.natural_ranking \
  --model meta-llama/Llama-3.2-1B \
  --cohort results/llama_recall_split/cohort.jsonl \
  --out results/revision_natural_ranking --candidates 32 --ig-steps 16 --device cuda
python -m revision.report --candidates results/revision_natural_ranking/candidates.csv \
  --out results/revision_natural_ranking/report
```

Toy conditions share prompts within easy/decoy and competing/hard pairs, and training
seeds initialize both model and sampling. Every evaluation example is retained,
including incorrect predictions. Check trained accuracy in the candidate file;
two-update smoke runs are intentionally not trained enough to support claims.
The report uses all examples; a correct-only analysis must be separately specified.
Input-provenance and edge-sum scores are separate methods. Equal-seed summaries and
paired seed-bootstrap differences are exported. With only one seed, intervals are
left undefined instead of reporting a misleading zero-width interval.

Natural candidates are (layer, head) source-column groups. `--candidates 0` measures
all; a positive count selects a uniform subset without consulting any score. Metrics
then apply only to that measured subset. Never call its top-k the full-network oracle.
When fewer than eight candidates are measured, capture8/recall8 use min(8,n); inspect
`n_candidates`. Signed capture can be negative and is undefined when the oracle sum
is nonpositive. The script does not clip away unfavorable effects.

**Baseline names and limits matter:**

- `legacy_gradxact` is native head-context gradient × cached source message.
- `gate_gradient` is the derivative of the exact renormalized removal gate at alpha=1.
- `gate_ig` integrates that derivative from alpha=0 to 1. It is candidate-specific,
  expensive, and should converge to the exact margin drop by the fundamental theorem
  of calculus. It is an accuracy/reference control, not an efficient circuit-discovery
  method or evidence of novel explanatory power.
- The candidate-specific path uses 2K+1 backward evaluations plus four forward
  evaluations at K requested intervals; raw run times, finite-difference gradient
  errors, IG residuals and coarse/fine differences are saved. Inspect numerical
  errors before interpreting results, and increase intervals if needed.

**This package does not implement AtP* or the published EAP-IG circuit-discovery
algorithm.** Do not rename these baselines as either one. To answer that part of the
review fully, add a comparison using the authors' reference implementation and a
matched node/edge intervention and faithfulness metric. That is separate integration
work; these scripts provide the transparent matched baselines and raw data needed
for the comparison. Related references:

- AtP*: https://arxiv.org/abs/2403.00745
- EAP-IG: https://arxiv.org/abs/2403.17806
- Information Flow Routes: https://aclanthology.org/2024.emnlp-main.965/
- MIB: https://mib-bench.github.io/

## 5. Internal repair and attribution patching

Add donor fields to each failed evaluation record:

```json
{"donor_prompt":"Annapolis is the capital of Maryland. The capital of Maryland is","donor_source_span":[10,11]}
```

These are schematic spans. Provide verified token alignment, or `donor_input_ids`.
The donor must predict the exact target token at the specified decision point.
The script rejects unsuccessful donors and unequal span lengths. A wrong donor is
optional via `wrong_donor_prompt`/`wrong_donor_input_ids` and
`wrong_donor_source_span`; it must not predict the target.

```bash
python -m revision.internal --model meta-llama/Llama-3.2-1B \
  --cohort cohorts/llama_recall_with_verified_donors.jsonl \
  --out results/revision_internal --alphas 0.25 0.5 1 --ig-steps 16 --device cuda
python -m revision.internal_report \
  --features results/revision_llama_recall/robustness/reference_features.csv \
  --repairs results/revision_internal/internal_repairs.csv \
  --out results/revision_internal/report
```

Run one layer at a time initially with `--layers 4`. Default is all layers. Every
repair is applied to every available diagnostic category; there is no assumption
that the matching category must respond best. Random and wrong-donor displacements
are norm-matched to the successful-donor displacement at that site. The complete
residual state contains more than the target fact, so successful patching does not
prove a uniquely localized representation.

`atp_full_patch_gain` is the first-order clean-receiver gradient dotted with the
fixed donor displacement, obtained by differentiating alpha at zero.
`ig_full_patch_gain` integrates that same candidate-specific patch path. Full-patch
exact effects and convergence residuals are also saved. AtP is not AtP*.
Prespecify layers/doses or choose them on a separate development cohort. Do not select
the best layer using these outcomes and report its unadjusted significance.
`internal_report` applies Holm across all exported paired-control tests. Its sign-flip
inference assumes independent groups and symmetric/exchangeable paired null effects;
intervals for individual recovery cells remain descriptive.

## 6. Independent known-operation controls

```bash
python -m revision.stage_controls --out results/revision_stage_controls \
  --seeds 12 --steps 2500 --calibration 100 --examples 100 --device cuda
```

Successful clean toy examples are separated into disjoint calibration/evaluation
prompts. Evaluate source-token replacement, removal of its final-query attention
across heads/layers, and an explicit competitor-only logit bias. Export operation
versus diagnostic-label counts, including interventions that leave the answer correct.
Report all attempts and retained successes. If a model does not yield enough
successful examples, coverage is written and no estimate is invented.
These controls test known operations, not universal causal stage identification:
source changes can have downstream consequences, transport can be redundant, and
source readability can fail even on a correctly answered prompt. A diagonal matrix
is a hypothesis, not an enforced test expectation.

## Figures and publication reporting

```bash
python -m revision.report \
  --robustness results/revision_llama_recall/robustness/summary.csv \
  --out results/revision_llama_recall/figures
```

SVG and PDF plots use real numerical outputs and editable vector text. The robustness
plot shows a declared slice: original span, random decoys at the smallest requested
size, two consecutive layers, within-task calibration, largest calibration-group size.
All other settings remain in the CSV files; inspect exclusions before comparing curves.

Recommended paper package, in priority order:

1. Reconcile transport definition and decision-point implementation with the manuscript.
2. Main robustness figure with per-example stability, coverage and calibration transfer.
3. Strong controlled baselines, including honest negative results.
4. Held-out added-value table, with and without intervention-dose controls.
5. Internal repair matrix plus independent known-operation tests if claiming mechanisms.
6. Official contemporary circuit-method comparison if keeping a competitive attribution
   claim. The present code does not eliminate this remaining integration requirement.

Further studies worth doing after these: independently annotated natural failure
cases; common-logit-shift sensitivity of raw target transport; semantically matched
prompt changes without introducing answer tokens; multi-token sequence-level outcomes;
held-out architecture transfer. They require explicit new data/protocols and are not
presented as implemented in this package. More models alone will not resolve validity.

## Boundaries and reproducibility

The new measurement/internal-patch adapters support Llama, Qwen2-style models, and
GPT-NeoX under the original cached-map contract. Live attention-gate experiments
support Llama/Qwen2 only. Qwen3, Gemma and other unsupported architectures are rejected;
passing a numerical check is necessary but not a substitute for an architecture audit.
Tiny models from all three supported families are in the tests. Full pretrained
multi-billion-parameter runs have not been executed here.

Every run records configuration, input-file hashes, source-code hashes, package
versions and repository commit. Use a new output directory after changing code or
configuration. Atomic per-example files permit resumption. Completed computation is
not inferred from a directory existing: inspect the expected CSV and record counts.
No scientific result is hard-coded to be positive. Retain all null results and report
exclusions, numerical failures, sample sizes and subgroup coverage.
