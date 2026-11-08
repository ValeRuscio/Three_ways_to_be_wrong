FAST RESUME FOR THE EXISTING PATTERN CAMPAIGN

Use Three_Ways_Fast_Resume.ipynb on the SAME Lambda instance/results directory.
No installation or login. It contains the full updated source.
The notebook explicitly stops only its identified old detached worker group,
then resumes the SAME run, input, configuration and output directory. Completed
results are retained. The currently unfinished curve file may be recomputed.
The original source/protocol are archived, and the upgrade is recorded.
Do not run the old and new notebooks' launch cells concurrently.

WHAT IS FASTER
Gate/edge curves change only the final prompt position. Earlier tokens cannot
be affected by that intervention in a causal transformer. The new evaluator:
- caches earlier-token K/V once for each recipient;
- recomputes only the final token with live gates, attention and downstream norms;
- skips decoder layers before the first edited layer;
- applies the analogous final-token computation to frozen-program curves;
- skips already completed model stages and previously completed cached analysis.
The existing full-sequence evaluator remains the reference and fallback.
All facts, models, donor pairs, ranks, doses, seeds, k grids and controls remain.
Precision remains fp32, eager attention, TF32 disabled. No quantization or
bfloat16/FlashAttention substitution is made. Exact-origin ledgers still use
the original full frozen evaluator. Weight edits, screening, training trajectories
and statistical inference are not replaced by these curve optimizations.

VALIDATION ON YOUR GPU
For EVERY unfinished recipient pair, first compare original and fast logits for
native execution, a late-layer gate edit, multiple gate layers, attention edits,
and combined edits. Maximum absolute and relative error must meet the existing
1e-4 tolerance AND gold-top1/top-token must agree on the probes. If any check
fails, that recipient uses the original path. Near-zero gold margins in actual
curves are checked using the full original evaluator as well.
Each pair writes curves/<pair>/fast_validation.json. It includes a warmed timing
example for a late-layer gate edit. That example is not a whole-stage speedup.
Native-prefix KV tensors are immutable across evaluations; independent cache
containers are created per evaluation. Hooks/temporary forward replacements
are removed on exceptions. Generation and other prompt positions are NOT sent
to the final-token-only path.

EXPECTED SPEED
The optimization removes redundant work, but no GH200 speed multiplier has been
measured here. GPU memory bandwidth, prompt lengths, edit depth and Python/disk
cost affect results. Compare curve counts over 15-30 minutes after resuming with
your previous rate (~148/hour). All-layer edits have less layer-skipping benefit.
The full multi-model run can still take days. Keep the Lambda instance running.

MIGRATION GUARDS
Only the exact delivered original code hash 2af239ee17abee1f33aecf54 is eligible
for migration, and input/configuration must match exactly. Unknown code changes
or changed settings are rejected. The run lock is acquired before migration.
Old completed result files are not rewritten by the curve stage.
source_versions/<original hash>, protocol_before_fast.json, fast_upgrade.json
record provenance. Current source/ points to the faster implementation.
The old notebook will no longer accept this upgraded protocol; resume with the
fast notebook. Do not delete protocol.json or change it by hand.

ADVANCED TERMINAL (worker must already have stopped)
From the extracted fast source directory:
  python -u -m pattern_study.run --input /home/ubuntu/causal_v1_results.zip \
    --out /home/ubuntu/threeways_pattern_results/patterns_v1 \
    --config /home/ubuntu/threeways_pattern_results/patterns_v1/configuration.json \
    --upgrade-fast

Use the SAME existing Python environment. Read README.txt for the experimental
protocol and interpretation limitations; FAST_README.txt describes changes only.
