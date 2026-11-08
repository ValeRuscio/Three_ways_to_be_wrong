THREE WAYS — FULL FOLLOW-UP EXPERIMENTS
====================================
Run Three_Ways_Full_Followup.ipynb on your Lambda machine. It embeds all source,
installs an isolated environment/kernel, finds causal_v1_results.zip or the prior
results directory, and runs the full eligible evaluation cohort. It never needs
/path/to/cohort.jsonl and does not overwrite the previous experiment.

DEFAULTS
All eligible evaluation examples; no smoke-test subset. Existing pinned model
revision and input token IDs. FP32, eager attention, no KV cache, no TF32.
Llama, Qwen2, GPT-NeoX/Pythia and OLMo2 architecture adapters.
Up to 3 success donors and 3 failure controls per failure, from calibration only.
2 attention choices x 2 gate choices x 2 normalization policies.
Gate, simplex-preserving attention-edge, and joint searches; budget 256 operations
per mode; 3 random-search controls (seeds 0, 1, 2). The greedy algorithm itself is
deterministic. 2,000 subject-cluster bootstrap draws / sign-flip permutations.

This is a bounded full-cohort experiment, not an exhaustive combinatorial search.
Increasing the edit budget or changing any setting requires a NEW RUN_NAME.
With 1B models use a GPU with at least 16 GB VRAM and adequate free disk. FP32
weights alone are about 4 GB per billion parameters. Larger models need more.
Verified edit searches plus controls may take substantial GPU time; no runtime
estimate has been measured on your hardware. Full vocabulary vectors can occupy
several GB across models. The compact results ZIP excludes these vectors; keep
its vectors/ directory on the machine for frequency reanalysis.

WHAT IS INCLUDED
1. Cached CPU diagnostics
- Exact gold-minus-fixed-competitor origin decomposition and certificate.
- Common-logit-shift invariance audit with recalibration on the SAME independent
  saved calibration references, including singular denominator reporting.
- Threshold diagnostics on held-out correct answers, without a correct override.
- Selection/transport matching within relation and declared margin/length
  calipers, vocabulary-rank matching when available. One deterministic fact per
  subject, no repeated subjects, no out-of-caliper fallback. If few pairs match,
  that is a coverage limitation, not permission to loosen calipers post hoc.
- Matched repair response, same arm/dose/layer, from the old repairs.csv.
- The GPU run repeats diagnostic matching with measured target vocabulary rank.
  Cached analyses remain separately available without that adjustment.

2. Routing/gate hybrids
Recipient token sequence/embeddings are unchanged. Whole-program donors must
have identical token count, identical subject-position indices, identical tokens
outside the subject, and the same relation. Same subject length alone is not
sufficient. Independent calibration success donors and failing donor controls
are selected without optimizing recipient recovery. Each donor receives all 8
A/g/normalization combinations. Numerical accounting is verified separately.

Primary contrasts hold recipient normalization coefficients fixed; donor-norm
contrasts are a sensitivity analysis. A/G main effects and their interaction are
reported relative to the corresponding same-normalization baseline. Raw files
also contain changes from the unmodified native recipient. Donor repetitions
are averaged per recipient before statistical summaries.

IMPORTANT: Gates are activation-derived coefficients, not a fact memory bank.
Success on another subject does not establish that its gates contain the
recipient's fact. A/g/norm fields are coupled in the live model; arbitrary
hybrids can be off the live model's reachable computation manifold. These tests
identify effects of frozen-program edits, not a unique natural causal stage.
A negative effect is not proof that routing is generic or irrelevant.

3. Verified edit search
For each first-token failure, try three fixed paraphrase templates of the SAME
fact. Require unchanged target token boundary, exact subject token alignment,
correct native first token, and a verified full-answer prefix match. Log every
attempt. Different-length paraphrases use a conservative monotone partial token
alignment; unmatched positions retain recipient coefficients. No fabricated
padding alignment. The final answer site is an explicit semantic anchor.

Each greedy step computes exact sensitivities with a layerwise reverse sweep,
chooses the highest positive single-edit finite gain, and verifies it by a fresh
frozen forward computation. A single gate coefficient is one gate operation.
An attention operation sets one edge to the donor value AND rescales all other
entries in that row to preserve its probability simplex. Thus edge-operation
counts are not counts of independently changed matrix entries. Previously
edited edges can change through later row redistribution.

Stop when fixed gold-vs-original-competitor margin becomes positive, budget is
exhausted, or there is no improving single edit. Also record whether gold is the
actual vocabulary top-1 token. These outcomes do NOT establish full autoregressive
answer recovery. Full-answer failures already having positive first-token margin
are excluded from distance search and logged.

Reverse deletion produces a deletion-minimal successful edit SEQUENCE, not a
minimum-cardinality solution. Counts are verified upper bounds for the declared
alignment/donor/operation set, not information-theoretic bits. Synergistic edits
can exist even when no single edit improves the margin. Failures to recover are
explicit; do not summarize only successful searches. Curves retain stopped runs
as plateaus. Random controls sample available edits uniformly, not by sensitivity.
Only donor-available failures support the edit-distance comparison; report their
fraction of all failures. Three random seeds are controls, not 3 training runs.

4. Sink / prior hypothesis
Recompute full-vocabulary origin vectors, including bias, with reconstruction
certificates. Existing results contain selected-token ledgers, not these vectors;
this part is NOT purely cached CPU work on the old ZIP. Subsequent aggregation
is CPU-only. Center each vector across vocabulary to remove logit gauge offsets.
Compare within-template variance / second moment for first position, subject,
and remaining template. Singleton or zero-energy variation is undefined. Log
first-position attention mass separately; input-origin attribution is not an
attention-weight measurement. Subject precedence prevents overlapping origins.

Because no frequency artifact was supplied, the notebook can build one from
ALL training rows of WikiText-103, at a pinned dataset revision, using the exact
model tokenizer. Default BUILD_WIKITEXT_REFERENCE=True. This is an independent
public-corpus frequency PROXY, not the model's true pretraining frequency and
not a reconstruction of the paper's Appendix C artifact. Counts use no special
tokens; log(count+.5) is centered and normalized across the full output vocab.
Frequency cosines can depend on smoothing, zero-count tokens and corpus choice;
consider a second corpus before making a general unigram-prior claim.
Dataset: https://huggingface.co/datasets/Salesforce/wikitext

You can supply actual local corpus counts instead:
python -m followup.frequency --corpus YOUR_TEXT.jsonl --model MODEL_REPO \
  --revision EXACT_MODEL_SHA --vocab-size OUTPUT_VOCAB_SIZE \
  --description 'Name and provenance of this corpus' --out counts.npz
Each JSONL line must have a text field (override --text-key if necessary).
Supports .jsonl.gz. Set frequency_file in config, or frequency_files as a mapping
from prior model key to counts.npz. The sidecar counts.json is mandatory and
contains a tokenizer fingerprint. Do not use entity-popularity counts.

5. Gradient subspace (optional, NOT available from the supplied checkpoint)
The code is complete, but no actual pretraining gradient snapshots/eigenvectors
were supplied. Default status is explicitly NOT_RUN. Released model weights
cannot recover accumulated pretraining gradients. Fresh gradients on PopQA,
random directions or weight SVDs must not be labelled pretraining gradients.

A flattened weight-space eigenvector has a different coordinate space from an
activation vector. The implemented operationalization uses head-output residual
coordinates: eigenvectors of SUM_t G_(t,head) G_(t,head)^T, where G is a genuine
pretraining gradient of the attention output projection, shape [d_model,d_head].
This is a specific output-coordinate gradient geometry, not automatically the
same subspace as in another paper. Agree on that definition before comparing.

To build bases if the actual training logs become available:
python -m followup.geometry --gradients gradients_manifest.json --out bases --rank 64
Gradient manifest example:
{
  "gradient_origin": "pretraining",
  "provenance": "Actual run ID, corpus, checkpoints, loss scaling, and which steps were saved",
  "model_sha": "exact evaluated checkpoint SHA",
  "num_heads": 32,
  "layers": [0, 1],
  "snapshots": ["step0001.npz", "step0002.npz"]
}
Each NPZ has keys layer_0, layer_1, etc., arrays [d_model, num_heads*d_head]
for W_o gradients in the model's output-by-input weight convention. Snapshot
sampling is unweighted and must be described; this is not silently a full-history
accumulation. Dense d_model squared eigendecompositions can be CPU-expensive.

Alternatively supply a manifest.json with model_sha, provenance, heads entries
{layer, head, file, key}, and
"space": "head_output_residual_before_post_attention_norm".
Each file/key stores an orthonormal Q with shape [d_model,rank].
Set subspace_manifest to its path. All declared heads are processed.

The geometry stage computes each head's subject-INPUT-origin write to the answer
position, allowing subject-origin information carried at ANY intermediate token.
It projects the local write, propagates projected/complement pieces through the
frozen downstream program, verifies additivity, and reports energy and signed
margin contributions. Rank-matched random subspaces are controls. Local head
writes overlap across computational paths: they cannot be summed as disjoint
final-origin components. A signed contribution ratio can exceed 1 or be negative.

REGIMES AND INFERENCE
Existing source/transport/selection labels are preserved. Per-example results
also contain descriptive subject-margin sign groups (subject_favors_wrong,
subject_favors_gold_but_loses, small_subject_margin), with default magnitude
threshold .1 logit. These are not validated absent/misretrieved/overridden causes.
Raw tables permit both stratifications. Do not rename observational groups as
causal mechanisms solely because a patch changed the output.

RESUME / REPRODUCIBILITY
Each stage/example/search is saved atomically. Completed examples/searches are
skipped; an interrupted individual search restarts that search (up to 256 edits).
A filesystem lock rejects concurrent workers in the same output directory.
Protocol records input measurement hashes, configuration, source hash and supplied
external-data hashes; changes require a new directory. Model revisions come from
prior model.json, never a silently updated main branch. Environment is recorded.
Frequency corpus counting has its own pinned revision and resumable checkpoints.
The archive's Python code is never executed; only its data records are read.

CLI after installing requirements and PyTorch:
python -m followup.run --input causal_v1_results.zip --out followup_results/v1 --config config.json
Add --cpu-only for cached diagnostics, or --report-only to regenerate GPU reports.
Report-only frequency regeneration may load tokenizer files, but no model weights.
Use the SAME config and output directory to resume. Look at coverage.csv and
external_data_status.json before interpreting plots. Numerical failures stop the
run; no silent fallbacks to approximate accounting or fewer examples.

VALIDATION
Tests use tiny random models to verify native/hybrid accounting, exact finite
edit gains, attention probability constraints, origin-vector reconstruction,
projected delivery additivity, gradient-basis schema, search recovery/pruning,
full runner reporting and resume. These checks are not the scientific experiment.
Local validation used CPU torch 2.14.1 and transformers 4.48.3; the notebook installs
CUDA torch 2.7.0 and reruns the tests there before running the real cohort.
No claim that the full new GPU experiments have already been executed.
