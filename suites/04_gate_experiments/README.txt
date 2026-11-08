THREE WAYS — GATES, FACTORS, LIVE WEIGHT EDITS, AND TRAINING
========================================================
Use Three_Ways_Gate_Experiments.ipynb in your existing Lambda environment.
There are NO package-installation, environment-creation, kernel-registration,
or Hugging Face token/login cells. The notebook finds the existing follow-up
Python environment or uses the current kernel executable, checks it, extracts
its embedded source, and runs the experiments.

INPUT
Use the ORIGINAL causal_v1 results directory or causal_v1_results.zip. The
compact followup_v1 archive lacks full measurement records/prompts and cannot
alone reconstruct the cohort. Default discovery finds the standard /home/ubuntu
locations. No /path/to/cohort.jsonl placeholders are required. The exact original
model SHA is reused. The old results and source are not modified.

DEFAULT CAMPAIGN
- All originally eligible evaluation facts (556 in the supplied Llama archive),
  plus eligible calibration facts for independent reference selection.
- Original prompt t0, and THREE alternative templates t1/t2/t3 for every fact.
  t1/t2 are the alternate relation templates; t3 prefixes t1 with 'Question: '.
  They are fixed lexical variants, not independent random paraphrase draws.
- Native gates/factors for every valid evaluation prompt, irrespective of success.
- Calibration prompts are checked for success; only successes receive detailed
  factor caches. Evaluation labels are NEVER used to select independent sets.
- Primary top-k=32; deficit profiles also use k=8 and 128.
- Subject-last-token and final-prompt-token sites, analyzed separately.
- Three control seeds (0,1,2); weight relative-norm doses .003,.01,.03.
- Eight deterministic same-relation other-subject specificity probes per case,
  including correct and incorrect baseline cases when available.
- OLMo-2-0425-1B stage1: eight requested steps, resolved to nearest published
  revisions, printed and saved as immutable model SHAs. No soup-stage mixing.
- Actual training-gradient analysis only if genuine snapshots/bases are supplied.

This is a full-cohort study, not a smoke run. Neuron-set / donor availability
still limits which interventions are meaningful. Missing references, no positive
reference neurons, and no positive preactivation deficit are recorded, not replaced
with arbitrary neurons. Minimum independent reference subjects is 3. Several
relations may lack this in Llama-1B. Report coverage alongside effects.

COMPUTE AND STORAGE
This campaign is substantially larger than the earlier follow-up. Repeated live
weight edits, specificity evaluations, and eight training checkpoints may require
many GPU-hours or days; no hardware timing estimate has been measured here.
FP32 is required. A 16-GB GPU is a practical starting point for 1B models; available
working memory depends on prompt length and library versions. Larger models need
more. Run models sequentially; the code frees each checkpoint before loading next.

save_frozen_cache=True by default. Native A/g/norm/residual caches and full per-neuron
factors for thousands of prompts can require TENS OF GB or more; reserve ample
space (potentially over 100 GB including checkpoint downloads). The notebook shows
free disk. Compact exports exclude heavy caches. Keep the run directory to reuse
those tensors. Setting save_frozen_cache=False still saves all requested site gate
vectors, full answer-position attention rows, and factors in compressed NPZ files;
it only omits complete frozen-program .pt files. Change settings only with a new
RUN_NAME to preserve comparability.

HOW EACH PROPOSAL IS OPERATIONALIZED

1. SUBJECT x TEMPLATE VARIANCE
For each relation, keep one deterministic fact per subject and require the complete
3-paraphrase grid. Compute the exact crossed sum-of-squares partition:
  total = subject main effect + template main effect + subject-template interaction.
The interaction is retained; it is not called random noise. Report per-coordinate
components and layer-level energy-weighted fractions. No activation threshold is
used. Zero total variance is undefined. The plot pools WITHIN-relation sums of
squares, not between-relation variation.

MLP coordinates are neurons, at the subject's last token or final prompt token.
The latter is the site predicting the answer, not a generated answer token.
Subject tokenization can differ across paraphrases; such context effects belong
to the observed template/interaction components and are not pure semantic effects.

Raw attention rows have different lengths/coordinates across templates. The
crossed analysis therefore uses EXACTLY SUMMED mass on four disjoint roles:
first position, subject, template-before-subject, template-after-subject. Subject
membership takes precedence over first position. This is an explicit coarsening;
it does NOT measure all within-role routing variation. Full rows are saved, and a
second analysis measures uncoarsened between-subject row variation within exact
same-template token-layout strata. No fabricated padding alignment is used.
Descriptive subject/template fraction crossings are reported without forcing a
single crossover or interpreting it as proven routing-to-content hand-off.

Files: reports/variance_components.csv, variance_coordinates/*.npz,
attention_exact_layout_variation.csv, variance_coverage.csv, gate_variance.pdf.

2. EXACT LOCAL THREE-FACTOR ACCOUNTING
For a fixed native pass, let z be the native MLP input and z_S its subject-INPUT-
origin component after the frozen within-block attention/normalization map. At
site p and neuron n:
  contribution(n,p) = SiLU(W_gate[n] z[p] + b_gate[n])
                    * (W_up[n] z_S[p])
                    * (W_down[:,n]^T r[p]).
r includes the complete frozen downstream map and OLMo's post-MLP normalization.
Compute gold and competitor contributions. W_up bias is excluded from z_S because
it is an affine bias origin, not subject content. Also save the native full-input
W_up value, native gate preactivation, sigmoid, row norms, and cosine alignments.

This identity is exact for a LOCAL MLP EDGE under the frozen program. Summing such
edges across layers is not a disjoint input-origin decomposition: paths can traverse
multiple MLP edges. A finite neuron-deletion check validates this accounting in the
implementation tests. Live nonlinear causal effects are tested separately below.

Independent 'relation-support neurons' are the top positive mean gold subject-
origin contributors in calibration successes, averaging templates within subject
before averaging subjects. They are not automatically fact-specific memory cells.
For failures with a successful paraphrase, a separate same-fact donor set is built;
its results are never pooled with the independent sets. k sensitivity is reported.
Reference-standardized factor differences use between-calibration-subject standard
deviations when nondegenerate; donor sets have no invented variance estimate.
Competitor-support overlap is exploratory and evaluated separately.

Files: reports/deficit_profiles.csv, neuron_set_coverage.csv, profiles/*.json,
three_factor_profiles_*.pdf, selectors/index.json and per-relation NPZs.

3. CONTINUOUS GATE DISTANCE + LIVE DONOR PATCH RESPONSE
SiLU is smooth and can be negative. It has no exact 'closed/open' hyperplane.
The signed ambient distance to a declared preactivation level a is
  (W_gate[n] z + b_gate[n] - a) / ||W_gate[n]||.
We save distances for a=-2,0,2 as declared sensitivity probes, plus raw preactivation
and sigmoid. These levels are not asserted physiological opening thresholds, and
ambient Euclidean distance is not necessarily a reachable normalized activation edit.

For selected neurons BELOW their reference preactivation, patch .25,.5,1 times
the positive preactivation difference at one site, with live downstream recomputation.
Compare norm-matched random neuron and random direction patches, with three seeds.
Reference may be independent relation success or same-fact paraphrase, explicitly
labelled. This directly tests the donor-patch-distance hypothesis, rather than
inferring it from weight edits alone. Cases without a positive deficit are omitted
from that intervention, not classified as a different cause.

Report strongest-current-competitor margin gain and gold top1. Distance-response
plots are accompanied by descriptive rank correlation and, where sample size and
rank permit, OLS associations adjusting for initial margin, log gold rank, perturbation
norm and relation, with subject-cluster bootstrap. These are observational predictive
associations, not causal effects of distance. Small/rank-deficient subsets are explicit.

Files: activation_patches/*.json, reports/activation_patches.csv,
activation_paired_tests_holm.csv, distance_response_models.csv, distance_*.pdf.

4. LIVE WEIGHT EDITS
Select the layer independently by the largest positive calibration-support sum
among the top-k neurons, separately for each site. Same-fact donor layer selection
is a separate, conditional experiment. Within the selected set, target neurons
whose native preactivation is below the reference. No deficit means no opening edit.

Let a be the unit vector of positive deficits and xhat the unit native MLP input:
  Delta W_gate[selected,:] = budget * a xhat^T.
This is rank one and increases the selected preactivations at the recipient site.
It is a GLOBAL PARAMETER EDIT: it acts at other positions/subjects too. All norms,
gates, attention patterns and downstream activations are recomputed live.

Compare with:
- gate reverse edit;
- random neurons, same input direction;
- selected neurons, random input direction;
- rank-one W_down edit, oriented by the TRUE LIVE margin derivative at W_down output
  and the selected native neuron activation vector;
- W_down random-neuron / random-output-direction controls;
- rank-one W_up edit oriented to improve the same initial live margin.
Random controls use seeds 0,1,2. All arms in a comparison have the SAME Frobenius
norm: rho * original selected W_gate row-block norm. This is equal parameter cost,
not equal functional cost. Final normalized output displacement and KL on other
subjects are reported as additional costs. W_down here is a ROME-inspired rank-one
baseline, NOT the covariance-preserving ROME algorithm or its optimized objective.

Every edit is temporary; try/finally restores exact original selected weights.
Fresh native outputs verify rollback. No edited model is saved over original weights.
Primary endpoint: gold logit minus the strongest OTHER vocabulary logit, plus gold
vocabulary top1. Fixed original-competitor margin is secondary. Conditional greedy
answer generation is performed only when canonical gold is top1; a null full-answer
result is UNMEASURED, not false. Generation scoring uses an alias-prefix match, not
an independently adjudicated semantic metric.

Specificity: deterministic other evaluation subjects of the same relation, excluding
the target subject, with correct/incorrect probes when available. Save every probe,
KL, prediction changes, margin change and top1 retention. Probe selection does not
use edited outcomes. These fixed probes can be shared across target cases; target-
cluster intervals are conditional on this control panel. Inspect efficacy AND
specificity at all declared doses; do not select the best dose on test outcomes.

Files: weight_edits/<case>/*.json, reports/weight_edits.csv, specificity.csv,
weight_intervals.csv, weight_paired_tests_holm.csv, weight_*.pdf.

5. TEMPLATE-KEY ATTENTION PATTERN INTERVENTIONS
Select positive hand-off heads from independent calibration successes, using exact
subject-origin S-position -> answer head contributions to the gold-competitor margin.
Use exact-layout same-template donors from OTHER subjects (success and failure
controls). At the answer query, replace only the template-key entries, and rescale
the complement to stay on the probability simplex. Also compare subject-key edits,
sink-share edits, shuffled-template assignments, and random-head controls.

For same-subject different-template prompts, use explicitly labelled role-MASS
transplants, keeping recipient relative probabilities within each role. Missing
role correspondences are reported; no arbitrary alignment is manufactured.
Values remain the recipient's LIVE values. Subsequent computations remain live.
No direct query or key tensor editing is claimed: patching a probability row does
not uniquely identify the underlying Q/K geometry. Sink-share associations do not
by themselves establish a 'null-gate' failure mechanism.

Files: routing/*.json, reports/routing.csv, routing_intervals.csv,
sink_routing_associations.csv, routing_pattern_effects.pdf.

6. HOW GATES CHANGE DURING TRAINING
Use OLMo-2-0425-1B stage1 published checkpoints. Resolve/pin nearest available
revisions to the requested steps before measurements. The actual revisions/SHAs
are saved. Final-stage1 calibration successes define FIXED neuron IDs/layers for
all checkpoints. When unavailable, a separate retrospective set from the evaluated
fact's final-checkpoint success may be used and is explicitly labelled. Do not pool
that outcome-selected set with independent calibration selection.

Retokenize the SAME fixed facts with OLMo's tokenizer; verify vocabulary consistency
across checkpoints. Llama's regime labels are not used as OLMo error classifications.
Track native gate preactivation, sigmoid, W_up subject read, upstream/read and
W_down downstream cosine alignments, and exact local gold contributions. Learning
uses first sustained canonical-first-token + alias-prefix correctness (two sampled
checkpoints by default). Initial successes are left-censored. First observed success
is interval-censored between sampled training steps, not a precise training moment.
All transitions and final correctness remain available in raw files.

Plot factors aligned to the first sustained success. Also report .25/.5/.75 progress
along each observed initial-to-final change, flagging decreases and absent crossings.
This is a sensitivity analysis, not an opening threshold. Key preactivation depends
on both weights and inputs; downstream alignment also changes with the rest of the
network. Temporal ordering alone does not prove which weight family learned first.

Files: olmo_training/checkpoints.json, fixed_neurons.json, step_*/*.json,
reports/training_status.csv, training_neurons.csv, training_events.csv,
training_event_aligned.csv, training_progress.csv, training_*.pdf.

OPTIONAL ACTUAL GRADIENT SUBSPACE
No gradient snapshots were supplied. Checkpoint trajectories still run; ONLY this
sub-analysis is marked NOT_RUN. Optimizer second moments are not automatically the
full accumulated gradient eigenspace, and fresh PopQA gradients are not pretraining
history. No such substitutions are made.

The provided builder accepts genuine gate-projection gradient arrays [neurons,d_model]
and estimates an INPUT-COORDINATE subspace of SUM G^T G. A neuron gradient row lives
in that space. This is not a flattened parameter-space basis or the earlier output-
projection basis. It must match the definition intended in any companion paper.

Example input manifest (paths refer to actual existing snapshots):
{
  "gradient_origin": "pretraining",
  "provenance": "Actual run/corpus/loss scaling and gradient snapshot sampling",
  "repo": "allenai/OLMo-2-0425-1B",
  "layers": [0,1],
  "basis_snapshots": [{"step": 10000, "file": "step10000.npz"}],
  "probe_snapshots": [{"step": 20000, "file": "step20000.npz"}]
}
Each NPZ has gate_layer_0, gate_layer_1, ... matrices, output-neuron by input-coordinate.
Basis and probe steps MUST be disjoint. The code rejects self-evaluation on the
same gradients that defined the subspace. The accumulation is unweighted over
provided snapshots; describe that sampling rather than calling it the full history.

Build with the existing environment:
python -m gate_study.gradients --manifest actual_gradients.json --out gradient_basis --rank 64
Set gradient_manifest to gradient_basis/manifest.json in a NEW run configuration.
The analysis evaluates unique selected neuron rows and random-neuron controls,
without multiplying sample size by the number of facts sharing a neuron.

RUNNING / RESUMING
The notebook is self-contained and uses your existing environment/credentials.
It defaults to STAGE='all' (core + OLMo trajectory). STAGE='core' and STAGE='training'
are available for scheduling phases separately, using the same settings/run folder.
A filesystem lock prevents concurrent workers in one result directory. A recorded
PID is shown for monitoring. An interrupted monitoring cell leaves its detached
worker running; do not launch a duplicate.

CLI, from extracted source in your existing environment:
python -m gate_study.run --input causal_v1_results.zip --out gate_study_results/gates_v1 --config config.json --stage all
Use --stage report to regenerate summaries without model weights. Raw intervention
JSON files are needed; summary CSVs alone do not reconstruct all reports.

Atomic saves: feature prompt, weight-edit arm/dose, activation/routing case, training
checkpoint/fact. Incomplete individual activation/routing cases restart that case;
completed weight arms/doses are retained. Identical code/config/input resumes;
changes require a new output directory. No uploaded source code is executed.

The compact results ZIP includes source, configurations, selection references,
per-case interventions/profiles, reports and training trajectories. It excludes
frozen_cache/, features/, native_baselines/, duplicated input archives, and model
weights. Keep those large local caches if further tensor analyses are planned.

INTERPRETATION AND PRIOR WORK
Hypotheses are not encoded as expected outcomes. No code automatically relabels
source/selection/transport as absent/misretrieved/overridden mechanisms. An intervention
that helps at one norm budget does not establish unique fact storage or rule out a
value mechanism. Prompt-conditioned exactness does not make live nonlinear causal
attribution unique.

Dai et al. (2022), Knowledge Neurons in Pretrained Transformers:
https://aclanthology.org/2022.acl-long.581/
Their integrated-gradient attribution already depends on the model output. Do not
claim earlier work lacked downstream dependence. The distinction here is explicit
factor accounting conditional on a frozen computation and separately tested live edits.

Meng et al., Locating and Editing Factual Associations in GPT:
https://rome.baulab.info/
The rank-one W_down control here is not presented as a reproduction of ROME.

Official OLMo checkpoint/revision documentation:
https://huggingface.co/allenai/OLMo-2-0425-1B
https://github.com/allenai/OLMo/blob/main/docs/Checkpoints.md

VALIDATION
Tests verify local factorization against finite neuron deletion, nontrivial native
weight edits, exact rollback even on exceptions, live activation and attention
patches, probability conservation, crossed variance identities, fresh tokenization
and frozen-cache saving, the core runner and resume, training runner and temporal
reporting, and gradient-basis/probe separation. Llama, Qwen2 and OLMo2 adapters are
covered. The tests use tiny random models, not scientific data. They do not replace
the full new GPU campaign, which has NOT been executed here.
The existing environment should provide transformers 4.48.3 and the dependencies
used in the previous notebook (torch, numpy, pandas, scipy, matplotlib, huggingface_hub,
tokenizers, pytest). The notebook verifies these; it does not install or log in.
