THREE WAYS: REVIEWER FOLLOW-UP SUITE (EIGHT EXPERIMENTS)

Use the self-contained Three_Ways_Reviewer_Followups.ipynb on your existing GPU
Python environment. No installation, virtual-environment, kernel, or token/login
steps. Tested with Transformers 4.48.3 and tiny CPU models; pretrained GPU runtime
is NOT benchmarked. Float32 eager attention, TF32 disabled. Compatible with the
previous Llama/Qwen/OLMo2 native affine engine.

START AND RESUME
Default PREVIOUS points to ~/threeways_channel_results/channels_fresh_v1. If it
exists, reuse its prompts, calibration source data, model commit pins, baseline
generations and compatible gradient split JSONs. If absent, fresh PopQA data are
built with the already-used datasets package. NO PREVIOUS RESULTS ARE REQUIRED.
A summary-only archive cannot supply raw gradients or missing gate caches.
The previous directory is read-only. New output is ~/threeways_channel_followups/
reviewer_v1. Same protocol, code and cohort can resume; changing configuration
requires a new output folder. Adding stages does not change the protocol.

Notebook defaults to stages: profiles, regimes, variance, sink.
Then select separately: nulls; freeze; certificate; matrix.
All can be scheduled together, but see the workload plan first. The eight requested
experiments map to these stages as follows:
1 nulls
2 freeze
3 certificate
4 regimes + profiles (post-hoc reports by the new labels)
5 variance
6 matrix
7 sink
8 profiles on the additional Qwen2.5-3B and OLMo2-7B entries

Models: Llama 3.2 1B/3B, Llama 3.1 8B, Qwen2.5 7B/3B, OLMo-2-1124-7B.
Existing model commits are reused where available; new model commits are pinned.
architecture.json records model type, depth, hidden size, normalization modules,
and actual Q/K/V bias flags. Model-family comparisons cannot isolate the causal
effect of QKV biases, which co-vary with normalization, training and architecture.

EXPERIMENT 1: MATCHED NULLS
Default 100 evaluation prompts selected by a stable hash before observing any
outcomes. Same prompts across random initialization and trained-model random
readout conditions. Native trained margin profiles are retained for paired tests.
Random model uses Llama-1B architecture, same tokenizer, standard HF random
initialization, one fixed seed (1729); tied weights/config conventions preserved.
Margin gold token and competitor are held to the TRAINED counterpart's choices.
This is one initialization realization; intervals across prompts do not measure
uncertainty across random initializations.
Random readout: isotropic Gaussian direction in FINAL NORMALIZED hidden space,
rescaled to ||W_head[g]-W_head[c]|| for that trained prompt, seeds 0/1/2. Direction
and scale are held fixed when differentiating. No artificial output token labels.
Two backward passes per condition/readout draw/PROMPT, not one pass for a dataset.
Summary trained-vs-null differences are paired by prompt/matrix before fact and
subject aggregation. Null conditions inherit trained factual-outcome strata.

EXPERIMENT 2: THREE MECHANISMS
Default 100 outcome-independent prompts each on Llama1B and Llama8B. Set
freeze_prompts=null BEFORE starting a new protocol for every evaluation prompt.
Mechanisms are (A) realized attention probabilities, (G) realized SiLU outputs,
(N) inverse normalization scales. Norm group includes every RMSNorm/LayerNorm,
including Q/K norms where present and final normalization. Norm centering,
affine weights and biases remain explicit. Attention freezes recompute output
using LIVE V/O operations and fixed probabilities. Gate freezes replace SiLU
outputs, not weights. All frozen quantities are held at the original pass.
Run all 8 coalitions, so 8 backward passes/prompt. Each coalition must reproduce
the same unperturbed scalar objective. Define G(S) as gradient when S is frozen.
For mechanism a:
  Phi_a = sum_{S not containing a} |S|!(2-|S|)!/3! [G(S)-G(S union {a})].
Phi_A+Phi_G+Phi_N = G_live-G_all-frozen. Frobenius products use an 8x8 gradient
Gram matrix, avoiding dense parameter gradients and large concatenated factors.
Store Shapley norms, pairwise cross terms, signed <Phi_a,G_selection>/||G_selection||²,
and each one-at-a-time-freeze marginal norm. Signed projection shares sum to one,
but can be negative or exceed one; mechanism norms do not add. Three singleton
freezes alone would NOT yield an additive attribution. Attribution is specific
to this grouping and freeze game; it is not a unique biological/causal mechanism.
The closure check is an algebraic consistency check, not independent gradient
validation. Finite differences and dense-autograd tests provide separate checks.

EXPERIMENT 3: GRADIENT CERTIFICATES
Default 16 preselected prompts per model, layers nearest fractions 0/.5/1, all
7 projection families. Directions: three random rank-one vectors plus the
leading live-gradient rank-one direction. Relative matrix-norm epsilon sweep:
1e-5,1e-4,1e-3. Central live difference is compared with 2*alpha*<G_live,u v.T>.
Tolerance: 2e-4 + .03*max(abs(measured change),abs(predicted change)). Frozen
forward difference is compared with alpha*<G_content,u v.T>, using rtol .002.
Same BASELINE program frozen throughout every finite check. One untied matrix
at a time; this frozen response is affine in that matrix. Parametric gradients
are global over all token uses, not answer-site-only parameter gradients.
A predicted live difference below 10*atol is flagged unresolved; tiny changes
passing an absolute threshold are not strong gradient evidence. Tables include
failures, max errors, direction, epsilon, resolution rate, norm and raw changes.
No claim that finite differences prove exact real-arithmetic identities. The
previous tolerance value only covered forward reconstruction, not this table.

EXPERIMENT 4: LARGE-MODEL ORIGIN REGIMES
New independently calibrated OBSERVATIONAL definitions, versioned here. They are
not identical to earlier non-invariant labels and are not proved causal stages.
At answer site, compute exact subject/template/first-token/affine-bias origin
logit vectors z_o. Center EACH across the vocabulary:
  ztilde_o[t] = z_o[t] - mean_v z_o[v].
Gold/competitor origin shares are ztilde_subject[t]/sum_o |ztilde_o[t]|.
The denominator can be zero; preserve undefined cases. Centering removes uniform
logit shifts without changing target-vs-competitor margins. Full-vocabulary origin
and gate-preactivation reconstruction are checked. This is not the original raw
logit share or the earlier contrastive subject share; comparisons must state it.

Calibration uses subject-disjoint calibration examples from PopQA, template 0,
with BOTH accepted-answer generation and canonical-token top1 success. Average
within subject, then relation-specific lower quantiles of gold source value/share.
Need >=5 independent calibration subjects per relation/model; otherwise retain
uncalibrated_failure. No fallback to evaluation data or pooled relations.
Primary quantile .05, sensitivity .01/.10. Reference files record actual counts.
Gold or competitor is "source present" when its centered subject value AND share
exceed both zero and the corresponding calibrated threshold.
Classification order:
- accepted generated answer -> answer_success (regardless of canonical token);
- incorrect generated answer but canonical gold top1 -> continuation_failure;
- insufficient calibration / undefined share -> explicit unresolved labels;
- competitor source-present and subject margin g-c <0 -> misretrieved_origin;
- gold source-present, subject margin >0 and other-origin sum <0 -> overridden_origin;
- neither gold nor competitor source-present -> absent_origin;
- otherwise -> mixed_origin_failure.
No claim that every failure fits three categories. Negative contribution does
not imply a fact is absent from the weights. Labels change with comparator and
readout conventions; all scores and threshold-sensitivity labels are retained.
regimes.csv is compatible with the previous model,id,template,regime schema.
This suite joins those labels to compatible previous split rows itself, without
editing the old protocol or requiring its GPU run to be repeated. Original old
regimes cannot be compared as if they were unchanged. Null/gradient objectives
still use canonical g against a fixed strongest competing token; factual outcome
is separate normalized accepted-answer-prefix generation, with raw text retained.

EXPERIMENT 5: CROSSED VARIANCE
The previous channels run cached subject/template INPUT-origin vectors, not MLP
gates or attention rows. A new capture is required for actual gate variance.
Templates 0,1,2: three distinct paraphrases, NOT t1/t2/Question-prefix(t1).
One fact per independent subject within relation, chosen by ID without outcomes.
At subject-last and answer sites, decompose realized SiLU gate vector variation
into subject, template, interaction sums of squares. Attention uses per-head
answer-row masses over subject/first/other-template/answer positions, a disjoint
semantic partition. Raw token rows of differing lengths are not zero-padded or
misrepresented as comparable vectors. Headwise and pooled within-relation sums
are available. Exact crossed trace SS are computed from sufficient statistics:
no per-prompt full activation archive. Empty/constant cells stay undefined.
No thresholds choose a crossover: report subject-SS minus template-SS sign changes
by depth, retaining interaction. A descriptive crossover is not a proved handoff.
Resumption atomically saves per-relation sufficient statistics after each fact;
partially processed fact triples are recaptured. Final state can be larger than
summaries but is much smaller than storing every full gate vector.

EXPERIMENT 6: SOURCE x RESELECTING LAYER
Default 100 matched prompts on 1B/8B. Freeze ONLY one downstream layer's SiLU
outputs, all positions, leaving attention and norm selection live. Collect all
source projection derivatives in one backward pass per frozen layer: L+2 passes
per prompt including native/all-frozen reference. Save difference norms and signed
alignment with total selection, pooled across the seven disjoint matrix families.
One-freeze effects overlap; columns do not add to the total selection channel.
Plots mask j<=i and show STRICTLY downstream gate layers. Raw rows include the
diagonal and causal-order zero region as checks; freezing a gate's own layer is
not evidence of a downstream crossover. Similarity to a variance crossover is a
hypothesis to test, not imposed in the intervention or visual.

EXPERIMENT 7: SINK CONSTANCY
Capture first-token, subject, and template-origin contributions to the entire
answer-site gate preactivation vector at every layer. The previous origin cache
did not include the first-token vector, so no CPU-only reuse is claimed.
Within relation and template, use exact running vector sums, squared norms and
unit-vector sums to report trace variance, ||mean vector||, variability fraction
E||v-mean(v)||²/E||v||² and average pairwise cosine. Compare first-token origin
with subject/template origins at matched layer/relation/template. Zero vectors
are unavailable for cosine, not perfectly constant meaningful signals.
First token does not always mean BOS. Coverage records token ID, tokenizer BOS ID,
and subject overlap. Subject takes precedence: Qwen subject-initial prompts can
have an empty first-token origin; zero vectors are explicitly counted and excluded
from meaningful constancy/cosine claims.
This tests a putative gate-offset function. It does not test unigram-frequency
alignment or constancy of full output-logit vectors. A near-constant contribution
alone does not establish a causal prior or prove why the sink exists.

EXPERIMENT 8: FAMILY SIGN COMPARISON
profiles includes Qwen/Qwen2.5-3B and allenai/OLMo-2-1124-7B. Same prompt cohort,
canonical margin definition, gradient collection, accepted-answer scoring and
origin calibration. Compare normalized content-selection cosine, not raw cross
magnitudes alone. Existing four-model splits are reused only with matching pins
and matching baseline g/c/margin. Sign patterns remain descriptive; relation,
template and outcome composition differ. Correlation with QKV-bias presence does
not establish that biases cause the sign difference. OLMo post-attention/post-MLP
normalization is implemented and tested, including its Q/K RMSNorms.

OUTPUT / STATISTICS
Per-prompt atomic JSONs; per-relation compressed sufficient-statistic states;
reports/*_summary.csv, regimes.csv, regime_sensitivity.csv, calibration.csv,
coverage.csv, variance_pooled.csv, variance_crossings.json and diagnostic PDFs.
Norm/cosine summaries average seeds within prompt and templates within fact;
subject-cluster bootstrap, no interval under 5 subjects. Conditional descriptive
estimates, not multiplicity-corrected confirmatory tests. Matched null differences
use same prompt before aggregation. Full per-prompt records are kept for paired
fact/template analyses. Regime differences do not by themselves control relation
and template composition. No unsupported causal-stage assertions are generated.
The summary export intentionally excludes raw per-prompt gradients/statistic
states. Keep a full backup if you want to resume/reanalyse from those later.

OPERATIONS
Run the workload plan first. The default core pass covers all evaluation prompts
and adds calibration captures; expensive follow-up stages use the declared fixed
subsets. Set *_prompts=null before the FIRST run for full-cohort follow-ups.
Detached supervisor writes execution.log and exit.json. STOP file is cooperative;
current capture/Shapley/certificate operation may finish first. Source-layer matrix
checks between downstream layers. Wait for exit.json before terminating instance.
An OS advisory lock prevents overlapping writers in one output. A failed stage
retains completed records. Frozen/source files and results are not deleted.

Offline tests (no model downloads):
python -m unittest discover -s tests -v
Eight tests cover native/frozen gradient agreement for all supported architectures,
coalition baseline equality, Shapley closure, final-down normalization-only effect,
finite differences, causal-order zeros, centered-origin shift invariance, role
partition, streamed ANOVA/moments vs direct arrays, diagnostic branches and all
stages plus resume in a mocked tiny-model workflow. These are not GPU benchmarks.
