THREE WAYS: ORIGINS, ALIGNMENT, TRAINING, AND LIVE REPAIRS
======================================================

Quick start
-----------
Use Three_Ways_Causal_Experiments.ipynb. It embeds this complete source tree.
Upload that ONE notebook to Lambda and run the cells in order. The notebook
creates an isolated venv and runs experiments in a subprocess using that venv;
you do not need to switch kernels. Requires Python >=3.10, a compatible NVIDIA
driver, internet access, and Hugging Face access for gated Meta models.

The default PRESET="one_model" is a REAL experiment: Llama-3.2-1B; 100 fixed
candidate prompts per relation, six relations; up to 250 independent calibration
candidates per relation; all eligible evaluation failures AND successes receive
repairs. No 60-example ranking subset. Planted controls use up to 100 initially
correct evaluation examples; available successes may be fewer. Source repairs
are evaluated only where a verified paraphrase donor exists, with coverage
reported. Other repairs do not require donors.

PRESET="alignment": Llama-3.2-3B and Llama-3.1-8B base/Instruct; Qwen2.5 3B and
7B base/Instruct; OLMo-2-1124-7B base/SFT/DPO/Instruct. Paired fixed prompts,
shared decoy IDs for identical vocabularies. Raw completion primary; optional
chat_check uses the SAME instruct-rendered prompt for both members. Repairs are
off for model entries in this preset; all measurement/alignment analyses run.

PRESET="training": seven Pythia-2.8B revisions, plus seven available stage-1
branches each for OLMo-2-0425-1B and OLMo-2-1124-7B. OLMo branches are discovered
from the official HF repositories and pinned to immutable commits. Missing
branches cause an actionable error; no checkpoint names are invented. Stage-2
step counters are NOT mixed into the stage-1 timeline. Add explicit model specs
if your research question requires a stage-2 timeline with cumulative steps.

PRESET="full": all alignment + training models, with repairs on six explicitly
named core models: Llama-3.2-3B base/Instruct, Llama-3.1-8B base, Qwen2.5-3B base,
Qwen2.5-7B base, OLMo-2-1124-7B base. This is a large campaign, not a short run.
Override the model list to match the exact paper models if they differ.

Memory: fp32 is deliberate for the 1e-4 certificates; no quantization or device
sharding. 7B/8B weights alone need roughly 28-32 GB; native attention, activations,
and pullbacks need extra space. An 80-GB GPU is the safest choice for the largest
preset models. One model is loaded at a time. No runtime guarantee is claimed.

Terminal alternative
--------------------
  bash setup_lambda.sh
  .venv/bin/python -m pytest tests -q
  .venv/bin/python -m causal_suite.run --config config_one_model.json --out results/causal_v1

To rerun reports without GPUs:
  .venv/bin/python -m causal_suite.run --config config_one_model.json --out results/causal_v1 --report-only

Change output directory when changing code/configuration. Resume checks compare
both code and configuration hashes. Never overwrite another experiment's output.
Every measurement, repair example and planted example is saved atomically. A
crash repeats at most the interrupted example, not all previous examples.

Experiment 1
------------
Native model attention probabilities, MLP gates and norm scales define an affine
map exact at the observed pass. This includes OLMo-2 post-norm blocks and GPT-NeoX
parallel blocks. Biases include every affine offset, final norm bias and output
bias. Four DISJOINT origins: subject; remaining first token; all remaining
positions ("template", also includes chat markers); biases. Subject membership
wins if the first position is part of the subject. First token is a position,
not necessarily BOS.

Separate backward pullbacks for gold, actual competitor, and random relation
competitor. Success competitor is the highest non-gold token. The exact per-cut
sum and reconstruction checks must pass 1e-4. Handoff is the FIRST crossing of
half final subject support, only for positive support above the success floor;
curves may be nonmonotone. Non-crossing/ineligible cases remain visible.

Both signed subject/logit ratio and bounded absolute-contribution share are
reported. Neither is a probability of retrieval. Signed origin contributions
can cancel; signed ratios can be negative or exceed one. Random-span/competitor
controls are empirical tests, not assumed to be small. Model-wise AUC bootstrap
and permutation tests use subject groups; Holm corrections are in output names.
This is observational accounting, not proof of causal origin.

Old result archives with only CSVs/JSON summaries lack coefficient fields. They
cannot yield the exact new decomposition. The notebook rebuilds native passes.
Optional save_frozen_cache saves tensors, requiring substantial disk space;
pinned model weights are still needed to apply the maps. Compact reports can
always be regenerated without GPUs.

Experiment 2
------------
Facts are selected before seeing model outcomes, with subject-disjoint inventory,
calibration and evaluation splits. New PopQA study, not an exact reproduction of
the original eight-model paper cohorts. The full relation inventory comprises
answer aliases in the independent inventory split, not a claim to reconstruct
the original paper's unavailable relation families. A custom facts_jsonl can
supply the exact original facts AND their decoy-inventory subjects.

Schema for each custom JSONL row:
  id, group_id, subject, relation, answer, aliases (list), split
split is inventory/calibration/evaluation. IDs unique; subjects cannot cross
splits. Supported relation keys are in causal_suite/data.py. All three splits
are required. No placeholder paths are needed for the default PopQA run.

Full-answer correctness uses normalized answer-prefix + word boundaries;
manual review of generated text is required for publication. Every fixed prompt
remains in accuracy. Canonical-first-token diagnostics exclude valid alternative
initial tokens, later-token errors, ambiguous alias initials, copied answers,
near-zero denominators and inadequate decoy pools. These exclusions are logged,
not silently converted to knowledge failures. First-token metric denominators
are reported when tokenization makes that endpoint unavailable.

Hard-20 decoys are chosen once per prompt by the FIRST model/checkpoint of its
family, then reused across the family. They are not independently reselected
for each model. Thus they are hard for the reference model, not necessarily for
all family members. The full relation inventory is also fixed. The order of
models in config determines this reference. Tokenizer vocabulary mismatch stops
the run. Raw prompt strings are shared; inspect tokenizer-level differences
before strong alignment claims. Chat checks use one shared rendered prompt.

Calibration uses up to 60 independent successful examples, balanced cyclically
across relations; the actual n is reported. If too few successes exist, labels
are undetermined, not forced. Training uses within-relation successes where
sufficient and otherwise successes from other relations, without using evaluation
outcomes. Calibration shift remains a limitation of longitudinal label shifts.

Alignment outputs: four outcome cells; full/hard transition counts; fixed-vs-
still-failing odds ratios for each base label; paired q_run, q_max, tau and margin;
cluster intervals; Holm corrections; transition agreement under decoy conventions.
No assertion of "unchanged q" from a nonsignificant test: equivalence requires a
predeclared scientifically meaningful equivalence bound.

Training outputs: fixed-set accuracy, first-token failure counts, conditional
failure-label shares, coverage, trajectories, last failure before FIRST success,
regression-after-success flags, plots. Requested Cochran-Armitage p-value is marked
DESCRIPTIVE because repeated prompts violate independence. Subject-cluster
bootstrap intervals for log-step source-share slope are the primary uncertainty.

Exposure requires YOUR corpus or counts. Optional CLI causal_suite.exposure counts
lexical document co-occurrences from local JSONL/.jsonl.gz text documents and
records hashes and document counts. It is intentionally not an automatic download
of the entire Pile. Set exposure_csv to its CSV (id,exposure) before analysis, or
rerun report-only with that setting. Actual checkpoint-specific exposure requires
training-order information; corpus totals are only a separate stratifier.

Experiment 3
------------
Three repair families plus secondary/exploratory variants, every labelled failure
and eligible success. Fixed original competitor for graded margin changes;
recovery checks gold against the ENTIRE vocabulary.

Source: independently verified same-fact successful paraphrase (two alternatives
to primary template); span length and target token alignment required. Wrong-
subject donor drawn from independent successful calibration examples. Random and
wrong-subject displacements norm-matched at each layer. If no donor exists, source
repair is missing, NOT zero; all donor attempts/coverage are logged.

Transport: boost subject->answer attention on top subject-input-provenance heads,
computed by local gates under the frozen ledger (NOT legacy contextual edge sums).
Controls: bottom heads, random same-length disjoint span. Gamma grid explicit.
Secondary: add donor input-origin subject component at the answer position.

Selection: answer-up/competitor-down in normalized final-state space; full output
vocabulary recomputed. Random norm-matched controls. Calibrated-to-success-median
variants also exported. Exploratory upstream removal uses the LOCAL pulled-back
competitor direction (raw u_c is not in the source representation's coordinate
system). It is not automatically treated as a selection-specific causal repair.

Native dose units differ: alpha, attention multiplier gamma, normalized-state
Euclidean norm. We DO NOT report the proposed native cross-family cost ratio.
Full dose-response, minimum TESTED dose, and right-censoring are reported. An
exploratory common cost is relative displacement of the final normalized state;
interpolate within observed common ranges only, no extrapolation. Model-fixed-
effect paired comparisons at that proxy cost include subject-cluster covariance.
This proxy is not a causal equivalence of intervention types. Calibration labels
must not be optimized on these test results.

Primary 3x3 slice: middle declared source layer, source alpha=1, transport gamma=2,
selection alternative-down dose=1. Report median gain and matched-control tests;
Holm across available nine cells within model/mode. These doses are NOT equivalent.
Full grid comparisons are corrected separately across each complete output file.
The program never picks a best test dose to make a diagonal look favorable.

Planted source residual noise, zeroed S->T attention, competitor-only output bias
are run on initially correct examples, keeping ineffective plants in coverage.
All three repair families are applied to induced failures. Clean source donor is
an ORACLE positive control, separate from natural paraphrase donors. Exact reversal
is verified. Multiplying an exactly zero edge cannot restore it, so failed boost
on that plant is NOT a diagnostic failure. Multiple stages may be involved;
downstream edits can bypass upstream faults. Diagonal dominance is a hypothesis,
not a necessary consequence of causal stage identification. Success margins need
not remain unchanged under powerful output interventions.

Outputs
-------
reports/ contains CSVs and vector PDFs. Start with:
  coverage.csv, measurements.csv (includes generation eligibility in raw JSON)
  origin_auc_holm.csv, origin_controls_holm.csv, origins.csv, handoff_layers.csv
  alignment_cells.csv, alignment_transitions.csv, alignment_fix_odds_holm.csv
  alignment_paired_effects_holm.csv, alignment_decoy_agreement.csv
  training_summary.csv, training_trends.csv, last_failure.csv
  donor_coverage.csv, dose_response.csv, minimum_tested_dose.csv
  repair_matrix_declared_slice.csv, repair_matrix_nine_control_tests.csv
  repair_cross_family_common_cost_holm.csv, repair_pooled_fixed_effects_holm.csv
  planted_controls.csv, interpretation.json

A run has only those studies supported by its selected preset. Empty comparison
files are not positive results. Measurements.json preserves prompts, generated
text, target IDs, per-origin accounting, decoys, thresholds and certificates.

Validation and limitations
--------------------------
Local CPU tests use small randomly initialized native HF Llama/Qwen2/GPTNeoX/
OLMo2 models with deliberately nonzero biases. Tests cover reconstruction,
handoff conservation, live identity/reversal, planted operations, a tokenizer +
generation measurement path, leakage guards, cluster statistics and report outputs.
These tests validate implementation mechanics, NOT scientific predictions.
Pretrained Lambda-scale runs were not executed by the authoring assistant.
The earlier faulty uniform-grid IG baseline is not used anywhere in this suite.

References used for model presets
---------------------------------
https://huggingface.co/EleutherAI/pythia-2.8b
https://huggingface.co/allenai/OLMo-2-1124-7B-DPO
https://huggingface.co/allenai/OLMo-2-0425-1B
https://www.jmlr.org/papers/v26/23-0058.html

Missing-subject data validation fix
-----------------------------------
Rows with null, empty, or non-string subjects in calibration/evaluation are
excluded before generation. Per-model invalid_data_rows.json lists exclusions;
these rows are omitted from accuracy denominators without replacement sampling.
Inventory pools remain unchanged. For the original interrupted causal_v1 run,
use fix_missing_subject.py to migrate only the known compatible source version.
