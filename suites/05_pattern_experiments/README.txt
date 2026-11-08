THREE WAYS: PATTERN, SCALING AND TRAINING STUDY

Start with Three_Ways_Pattern_Experiments.ipynb. All source is embedded in it.
No installation, venv, kernel or Hugging Face login steps are included.
Use the environment you already used, with transformers 4.48.3.

INPUT
Use the ORIGINAL causal_v1_results.zip or its extracted directory, or a JSONL
cohort with id, group_id, subject, answer, aliases, relation, split.
The compact gates/followup archives do not contain the required full cohort.
Original model-dependent eligibility and old regime labels are NOT reused to
exclude facts from larger models. Invalid text is logged. Calibration/evaluation
subjects must be disjoint. The code only reads JSON from an input ZIP.
A fresh capture is required for this protocol (in particular MLP input vectors
and all five prompt conditions). Old results are not overwritten.

DEFAULT FULL CAMPAIGN
Llama 3.2 1B, Llama 3.2 3B, Llama 3.1 8B, Qwen 2.5 7B, sequentially.
All valid evaluation facts, all verified failure/success template pairs.
Five conditions: original t0, paraphrases t1/t2, Question: prefix variant t3,
and t4 with three independent calibration demonstrations. t3 is a prefix
variant, not a third independently composed paraphrase. Demonstrations use
known calibration answers without inspecting model success.
Three random-control seeds [0,1,2], rank caps [1,4,16,64], relative norm budgets
[.001,.003,.01,.03], below/above/all layer scopes, eight specificity subjects.
For Llama 1B the scope boundary is layer 11 (zero-based): 0-10 vs 11-15.
Other models use round(depth*11/16), fixed in advance. Empirical crossovers
are reported separately and do not retrospectively alter intervention scopes.

RESOURCE EXPECTATIONS
This is much larger than gates_v1 and is NOT an overnight-time guarantee.
The combinatorial pair/rank/dose/layer curves plus two training trajectories
can take many GPU-days. All five templates are generated unconditionally.
Exact frozen accounting uses fp32/eager attention and one unpadded sequence.
8B weights alone take about 32 GB decimal; gradients/activations and temporary
buffers need additional space. An 80-GB-class GPU is a sensible planning target
for the full campaign, not a measured guarantee. 16/24-GB GPUs cannot hold the
8B fp32 model. No quantization, CPU offload or distributed execution is silently
substituted. CPU RAM is also used for cached gradient factors and fit banks.
Caches can total hundreds of GB to over a TB across all models/checkpoints;
reserve disk for features AND HF model downloads. There is no automatic deletion
of HF cache or useful results. The notebook reports free disk and GPU memory.

RUNNING AND RESUMING
Notebook: choose RUN_STAGE='all' for everything. It launches a detached worker
and returns so the browser may close. Keep the Lambda instance running.
RUN_STAGE can also be capture, analysis, weights, curves, sink, training, report.
Use capture before analyses/interventions; these stages can share the SAME
run name and config. Selecting a stage changes execution only, not the protocol.
Configuration/input/code changes require a NEW run name. Completed per-arm or
per-curve files are skipped; an interrupted current curve is recomputed. A lock
prevents concurrent writers. Model revisions and dataset revisions are pinned.
A failed stage stops with a traceback; it is not silently called successful.

Terminal alternative after extracting Code.zip:
  python -m pattern_study.run --input /home/ubuntu/causal_v1_results.zip \
    --out /home/ubuntu/threeways_pattern_results/patterns_v1 --config config.json
Use --only capture (or another stage) to stage the SAME configuration.
Use --report-only to rebuild statistical summaries without model inference.

EXPERIMENT 1: PATTERN WEIGHT EDITS
All failure/success pairs are discovered from outcomes; no assumed count of 76.
Correct donor means canonical gold first token AND alias-prefix answer match.
Single-pair fit is mathematically rank one. It is evaluated only at rank one.
Higher-rank pattern fits use one independently selected calibration pair per
subject, within the relation, up to 128, PLUS the target pair. A separate
calibration-only arm tests transfer. Calibration selection never uses eval facts.
At each layer X contains native recipient MLP inputs and D=donor preactivation
minus recipient preactivation. Compute ridge least squares D X^T (X X^T+lambda I)^-1
using the equivalent small Gram system, then truncate by thin QR/SVD.
This is ridge fitting followed by weight-space rank truncation, not a claim of
optimal reduced-rank regression. Record attained ranks and live fit residual.
Same norm means the combined Frobenius norm across the selected layers equals
rho * original W_gate block norm, for every arm. Same requested rank CAP is used;
actual rank may be smaller for limited calibration pairs or token-rank-limited
gradients. Do not claim exact-rank matching where recorded ranks differ.
Gradient_gate, gradient_up, gradient_down use the SAME live fixed-competitor
recipient margin derivative, with all input positions included. These are
one-step gradient directions, not equally converged optimization solutions.
Random gate direction controls match the attained pooled-pattern rank per layer
and the same norm. Gradient arms may have lower attained ranks, recorded explicitly.
Weight changes are applied as exact additive low-rank linear-module terms,
W x + U(V^T x), to ALL tokens, including generation and specificity probes.
Tests compare these functional edits with physical dense weight mutations.
No model files are overwritten; hooks always roll back on exceptions.
Full-answer generation is UNCONDITIONAL for all weight arms. Gold top1 uses
strongest CURRENT competitor; fixed-competitor margin is secondary.
Eight other evaluation subjects of the relation are sampled deterministically,
balancing initially correct/incorrect where possible; short coverage is explicit.

EXPERIMENT 2: GATE/EDGE SIZE, SIGN AND MAGNITUDE
Run both failure->success donor repairs and reverse success->failure damage.
Gate order is absolute recipient subject-origin local gold-minus-competitor
contribution, or random. Report per layer and across layers at a logarithmic k
grid INCLUDING k=0 and all coordinates; k_half is first OBSERVED crossing of
half the full signed endpoint effect. It is not an exact minimal integer unless
that integer was sampled. Nonmonotone and near-zero-total curves remain visible.
Live recomputation and recipient-frozen-program curves are separate.
Binarization uses positive SiLU gates at their recipient per-layer mean positive
value, others zero. Signed-pattern swap, positive-mask swap and magnitude swap
are separate. Negative SiLU values exist; no inherent binary gate is assumed.
Some signed swaps can leave SiLU's realizable range. These are gate-variable
interventions, not necessarily realizable preactivation or weight changes.
Attention donor rows are transported monotonically within four semantic roles
(first, subject, template-before, template-after), conserving probability mass.
This is explicit cross-template alignment, not exact raw-token correspondence.
Pairs missing a required recipient role are excluded from EDGE analyses only.
Edges are ranked by frozen partial derivative times donor difference, then
finite patches are explicitly renormalized and recomputed. They are not exact
multi-edge finite gains from a gradient sum. Four full gate/attention hybrids
also receive exact frozen contrastive origin ledgers. Attention effects are
probability interventions, not isolated Q/K geometry edits.

EXPERIMENTS 3, 5 AND PROMPT DISTANCE
Cosines among successes, failures and across groups are computed separately
within relation/template/layer/site, excluding same-subject pairs. Pair counts
are dependent; do not treat them as independent observations.
Centered-PCA participation ratio is computed from the Gram spectrum; it depends
on sample size, so compare with n reported. Calibration success centroids average
within subject before across subjects. Their distances are saved alongside live
margins and all origin terms. Same-fact donor distances include few-shot prompts.
Per-neuron factors use the GOLD logit to keep the target contrast identical
across templates; differing target token IDs are excluded. Signed log-magnitude
ratios satisfy the product identity where all factors are nonzero. Near-zero
factors are excluded at 1e-8, with excluded counts and retained donor absolute
contribution mass. Sign flips are separate. Absolute log-change shares are
DESCRIPTIVE and are not percentages of a causal deficit. Raw arrays remain for
sensitivity analysis. Local neuron contributions overlap across layers.

EXPERIMENT 4: SCALING AND DIAGNOSTIC FIXES
Use native tokenizers and common facts. Qwen is not a same-tokenizer comparison.
Both vocabulary and full backend tokenizer fingerprints are exported.
Recompute all labels/calibration per model. The source diagnostic uses the 5th
percentile of contrastive subject-origin support in calibration successes,
averaged within subject, minimum three subjects per relation. Share is
subject origin contrast / sum(abs(all origin contrasts)), invariant to adding
the same scalar to every logit. These are explicit NEW diagnostic definitions,
not an undocumented reassignment of the original taxonomy or proof of stages.
Uncalibrated relations are labelled uncalibrated. Original source/selection/
transport and proposed absent/misretrieved/overridden names are not conflated.

EXPERIMENT 6: OLMO'S OWN LEARNED FACTS
Use all valid facts in the six PopQA relations, subject-disjoint deterministic
splits, independently of Llama outcomes. Pin PopQA and official OLMo stage1 refs.
Screen every candidate at each OLMo model's LAST stage1 checkpoint, retain its
own t0 strict successes separately in calibration/evaluation. This is a
retrospective final-success cohort, not unconditional population performance.
Do not promise hundreds of acquisition events; actual coverage is measured.
Run 1B and 7B at closest available stage1 checkpoints to 1/5/10/20/40/60/80/100%
of final step. Do not mix stage2 soups/ingredients with stage1.
Track strict correctness, margins, subject-origin margin, both sites' gate-vector
cosines to final form, and three-template variance/crossovers at each checkpoint.
First sustained success requires two observed checkpoints; initial successes
are left-censored and last-checkpoint-only successes do not meet that criterion.
Full per-step features and screen denominators are retained.
Gradient analysis is skipped unless actual pretraining gradient snapshots and
an independent basis are provided. The optional interface is gate_study.gradients:
  python -m gate_study.gradients --manifest snapshots.json --out basis --rank 64
Manifest requires repo, gradient_origin='pretraining', provenance, layers,
basis_snapshots and probe_snapshots with disjoint step sets and NPZ file paths.
Each NPZ stores gate_layer_L [neurons,d_model]. Set gradient_manifest to the built
manifest path, or a map from model repository to its manifest. Compare moving
neurons' probe-gradient projection energy with random neurons. Do not reconstruct
pretraining gradients from released weights or substitute evaluation gradients.

SINK DOSE SWEEP
Positive direct subject-to-answer head support in independent calibration
successes selects up to four heads. Multiply first-token probability by
1/.75/.5/.25/0 and renormalize, with random-head controls. Dose 1 is certified
against the native logits. Report lack of positive head references explicitly.

STATISTICS AND OUTPUTS
Per-arm JSON is authoritative. reports/weights.csv, weight_intervals.csv,
weight_paired_tests.csv, curves.csv, half_effect_sizes.csv, sign_magnitude_variants.csv,
sink_sweep.csv, pairwise_cosines.csv, effective_dimensions.csv,
success_centroid_distances.csv, factor_log_ratios.csv, same_fact_template_distances.csv,
origins_and_diagnostics.csv, and variance tables contain all measurements.
Templates/donors/random seeds are averaged within fact before subject-cluster
bootstrap and sign-flip tests. Exact sign enumeration for <=16 clusters;
100,000 Monte Carlo permutations otherwise. Holm uses the full weight-comparison
family; inference_resolution.json checks achievable resolution. No p-value is
zero. Degenerate bootstrap intervals are omitted rather than implying certainty.
Seeds are control randomness, not independent facts or model-training seeds.
Report-only computation can take substantial CPU time; it needs no GPU inference.
The compact ZIP omits features, large candidate input data, and downloaded weights.
Keep the full output directory on Lambda for future analysis/resume.

VALIDATION
Tiny randomly initialized CPU Llama, Qwen2 and OLMo2 tests cover physical versus
functional edits, live parameter gradients, gate/attention identity, rollback,
rank identifiability, factor identities, attention transport, pipeline/resume,
actual tokenization/capture and mocked-checkpoint training selection.
The full pretrained CUDA campaign has NOT been run by the assistant.

OFFICIAL MODEL/CHECKPOINT REFERENCES
https://huggingface.co/meta-llama/Llama-3.2-1B
https://huggingface.co/meta-llama/Llama-3.2-3B
https://huggingface.co/meta-llama/Llama-3.1-8B
https://huggingface.co/Qwen/Qwen2.5-7B
https://huggingface.co/allenai/OLMo-2-0425-1B
https://huggingface.co/allenai/OLMo-2-1124-7B
https://github.com/allenai/OLMo/blob/main/docs/Checkpoints.md
