THREE WAYS: CONTENT / SELECTION CHANNEL STUDY

This package implements all six experiments. No environment installation,
Hugging Face login, or token collection is included. Existing authenticated
PyTorch + Transformers environment is used. Tested locally with transformers
4.48.3 and CPU tiny random Llama, Qwen2, and OLMo2 models. Pretrained GPU runs and
runtime estimates have NOT been validated. FP32 eager attention is required;
turn off TF32. The notebook does that in the worker.

INPUT
Fresh mode (no prior results required):
python -m channel_study.fresh --out /home/ubuntu/threeways_channel_results/channels_fresh_v1 --seed 0 --evaluation-per-relation 100
Uses the existing datasets package to load a pinned PopQA revision. Up to 100
facts per relation across six relations; five templates including three-example
prompts from independent calibration subjects. Up to 3,000 evaluation prompts,
no outcome filtering. Exact counts/provenance/exclusions saved in data_source/.
This is a fresh cohort, not a guarantee of the exact previous cohort. The updated
notebook defaults to INPUT=None and this fresh path.

Alternatively use your existing pattern results directory, a normal results ZIP, or the compact
analysis ZIP containing records.jsonl. Only small prompt JSON metadata is needed;
activation caches are not required. Import deduplicates identical fact/template/
prompt records across models, keeps evaluation prompts regardless of correctness,
and retokenizes for each requested model. Import is restricted to the MAIN model
features folders, not retrospective training checkpoint cohorts.
A standalone prompts.jsonl is also accepted; each line must contain:
  id, group_id, relation, split='evaluation', template (integer), prompt,
  subject, answer, aliases (nonempty list).
The prompt must be raw text, not a pre-rendered chat prompt. Subject span is the
last occurrence, matching the earlier study. Audit ambiguous/repeated names.
Existing partial captures can leave incomplete template coverage: importer does
not invent missing prompts. Check printed fact/template counts before running.

RUN
python -m channel_study.plan --config config.json --prompts prompts.jsonl \
  --stages split geometry
python -u -m channel_study.run --config config.json --prompts prompts.jsonl \
  --out /home/ubuntu/threeways_channel_results/channels_v1 --stages split geometry

Later, with IDENTICAL config, code, prompts and output directory:
  --stages fisher
  --stages perturb
  --stages frontier
  --stages training
  --stages report
All six may be supplied together. Stage selection is not part of the protocol
hash, so adding a stage is supported. Changing cohort/doses/models/config/code
requires a new output directory. Models are pinned to commit SHAs on first run.
The notebook launches a detached supervised worker, provides a log monitor,
cooperative STOP, resume, and compact export. A STOP file is checked between
prompts and intervention arms, so an in-flight generation/gradient may finish.
Completed intervention JSON files survive an interruption. Per-prompt completion
markers prevent incomplete stages entering reports. OS advisory lock prevents
two workers writing one output. Source snapshot and environment accompany output.

DEFAULT CAMPAIGN
Four models: Llama 3.2 1B/3B, Llama 3.1 8B, Qwen2.5 7B. Eight specific model IDs
were not supplied; add supported Llama/Qwen2/OLMo2 models to config if required.
Notebook initially schedules SPLIT + GEOMETRY on ALL imported evaluation prompts
and all five available templates. This is not a tiny-model smoke run. The other
four experiments are fully implemented but deliberately scheduled separately.
Fisher uses 8 sampled next-token score gradients PER PROMPT (16 extra backward
passes, NOT the same 2 margin backward passes). Increase draws in a NEW protocol
if Monte Carlo uncertainty is too large. Each sampled trace is retained.
Perturbation/frontier: 32 facts, one hash-selected template per fact, selected
before observing outcomes; layers at fractions 0, .5, 1; all seven families;
three random seeds; doses .001,.003,.01,.03,.1 times that matrix's Frobenius norm.
Frontier: each nonzero live/content/selection leading-gradient rank-one direction
plus random controls at exactly the same norm. Other-fact controls: up to 8
independent subjects of the same relation and template, with greedy generation.
Training: 128 prespecified facts, template 0, OLMo2 1B and 7B; available stage-1
checkpoints nearest .2/.5/.8 of final step compared to final stage-1 checkpoint.
No final-success preselection. No assumption that 128 selected facts get learned.
Set max_prompts/probe_facts/training_facts to null for all available cases.
The probe selection remains one prompt per fact. For EVERY prompt perturbations,
edit subset call in run.py before starting a new protocol; large cost increase.
The plan prints pass counts. Frontier can require hundreds of thousands of
GENERATIONS across four models. Do not infer overnight completion from the
proposal's runtime estimates. Run one stage, inspect actual timings and results,
then schedule further stages. CPU RAM is needed for token gradient factors;
training additionally stores final and displacement weights in system RAM.

WHAT IS EXACT AND WHAT IS NOT
1. For a fixed canonical-gold vs fixed competing token margin at the last prompt
position, G_live = G_content + G_selection. The frozen program holds the pass's
attention probabilities, SiLU gates, and inverse normalization scales constant.
Selection includes normalization feedback, not solely routing/MLP gating.
Each global matrix derivative sums uses at ALL token positions. It is not an
answer-site-only derivative. W_gate/W_Q/W_K do not occur in the frozen program,
so their content derivative is identically zero by construction, not a discovery.
Token-factor gradients D.T@X agree with dense autograd in the tests. Frobenius
norms/cross-products are computed exactly from factors, with float64 accumulation.
The native frozen forward reconstruction certificate is stored for each prompt;
this alone is NOT a proof/certificate of gradient correctness. Tests independently
check dense gradients, centered finite differences and finite frozen responses.
Norm fraction is ||G_selection||/(||G_selection||+||G_content||). It is bounded,
but is NOT an additive share of ||G_live||^2; cancellation is measured by cosine
and cross term. Report content^2 + selection^2 + 2cross = live^2.

2. Model Fisher is E_y~p [(grad log p_y)(grad log p_y).T] for the next-token
conditional distribution. We sample y from the BASELINE distribution with fixed
seeds and store every draw. Traces and cross term use those score gradients,
not margin gradients. Their Monte Carlo estimates are unbiased conditional on
the baseline model; eight draws may have high variance. Negative cross is valid.
This is not a full sequence Fisher and not an empirical gold-label Fisher.

3. At the input to each MLP gate projection, decompose the frozen residual by
subject/template/first-token/affine-bias origin. Verify the gate preactivation
sum at the answer position. Geometry compares UNCENTERED origin input spans
with an approximate randomized top-r gate-weight row subspace. Principal angles
and random orthonormal subspace controls are saved; effective ranks are saved.
A per-pass origin norm crossing need not coincide with a between-prompt variance
crossing. These norms do not determine whether a gate opens: signs matter.
Origin vectors are small (layers x 2 x hidden), saved on CPU for reanalysis.

4. Change ONE untied matrix at a time. With the whole program frozen, its response
is affine in that matrix; verify the finite frozen response against <G_content,
delta W>. Measure the live response, local-gradient prediction, and their finite
residuals separately. Live-minus-frozen is a defined finite selection RESPONSE;
it is not generally the path integral of the baseline selection gradient.
SiLU/softmax/RMSNorm are smooth; a breakdown scale is a nonlinearity scale, not
a literal distance to a discrete gating boundary. Random signs can cancel:
absolute gaps, signed gaps, and normalized gaps are all saved.

5. Directions are exact leading singular rank-one approximations of the chosen
live/content/selection gradient, computed using thin QR and a small SVD. Norm is
matched to random controls and other families via RELATIVE Frobenius budgets;
this is not functional-equivalence matching. Parameter edits apply at every token
including generated continuations. Reports include fixed-token margin gain,
alias-aware greedy answer correctness, wrong->correct repair and correct->wrong
regression, next-token KL/prediction changes and answer regressions on controls.
Canonical-token top1 remains separate. Accepted-answer matching is normalized
prefix matching, not semantic verification. Raw generations are preserved for
audit (including repetition). A valid alternative answer is not a factual error.
Never choose a best dose on this evaluation cohort and call it confirmatory.

6. DeltaW is FINAL minus EARLIER checkpoint. This is net displacement, not the
accumulated gradient covariance or the full optimization trajectory. Parameter
coordinates must correspond; model architecture and tokenizer are checked.
The rank-r left/right singular subspaces define the tensor-product matrix span
{U B V.T}. Projection energy uses ||U.T G V||_F^2/||G||_F^2; random U,V of the
same ranks are controls. Signed <G,DeltaW> and normalized cosine use the FULL
weight displacement and are saved separately (projection fractions are unsigned).
Classify learned/stayed_wrong/stayed_correct/forgotten from the two endpoint
GENERATIONS, without selecting on final success. Endpoint learned does not
establish when learning occurred or that a particular parameter change caused it.

REGIMES
Without a trusted regime mapping, reports stratify answer_success and
unlabelled_answer_failure. Set regimes_csv to a CSV with model,id,template,regime
to add YOUR original absent/misretrieved/overridden labels. Labels are descriptive,
not treated as proved causal stages. A baseline accepted answer ALWAYS overrides
an old failure label to answer_success. Do not invent thresholds on evaluation
data merely to force all failures into three regimes. Regime file content is
part of the protocol fingerprint; editing it requires a new run directory.

OUTPUTS
Per-prompt baseline, split, Fisher draws, geometry, perturbations, frontier
measurements; small origin vectors; training endpoint generations and alignments;
reports/*_raw.csv, *_summary.csv, and diagnostic PDF figures. Summaries average
seeds within prompt and templates within fact, then use equal-fact means with
subject-cluster bootstrap. CIs withheld with fewer than five subjects. Bootstrap
conditions on saved Fisher draws; it does not separately propagate their MC
uncertainty. These are descriptive/exploratory summaries, not multiplicity-
corrected hypothesis tests. Raw rows support relation-matched/clustered tests.

TESTS (no model downloads)
python -m unittest discover -s tests -v
Tests cover all three supported architectures, every projection family, gradient
factor vs dense-autograd agreement, finite derivatives, frozen finite edits,
origin certificates, Fisher score identity, data import, all six pipeline stages
on tiny random models with a mocked tokenizer, and resumption of completed edits.
