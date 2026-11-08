# Three Ways to Be Wrong — supplementary code and results

This directory packages the experiments for **Three Ways to Be Wrong: Selected Computation and the Origins of Factual Errors in Language Models** (manuscript reviewed 8 November 2026). The publication title may change. The paper's proposed alternative title is not used as a final publication record here.

## Start here

- [Experiment-to-paper index](docs/EXPERIMENTS.md): find the suite for each result.
- [Reproduction guide](docs/REPRODUCIBILITY.md): dependencies, configurations, data and limits.
- [Definitions and interpretation](docs/DEFINITIONS.md): distinguishes exact accounting from causal claims.
- [Notebooks](notebooks/): editable launchers using the readable source in `suites/`.
- [Observed compact results](results/observed/): selected existing reports, not newly generated results.
- [Release validation](docs/VALIDATION.md): checks performed on this package.

The recommended entry points for the revised paper are **07 reviewer follow-ups**, **08 overnight**, **09 precision**, **10 signed projection**, and **11 adjoints**. Suites 00–06 preserve earlier experiments, including negative results. They are not interchangeable versions of the final diagnostic.

## Running

Open the chosen notebook from `supplementary/notebooks/` in your existing GPU environment. Keep the whole `supplementary/` folder together. No virtual-environment or Hugging Face login cells are included. Some launchers check common existing Python paths; select your own Python if needed. Model access and a compatible CUDA-enabled PyTorch installation are prerequisites. Other dependencies are listed in [requirements.txt](requirements.txt).

Read and execute configuration before launch. Default settings and optional-stage switches are visible in the notebook; they do not necessarily match the completed paper campaign. Use the archived `configuration.json` files in `results/observed/` when reproducing its exact settings. Always choose a new run name for changed settings. Do not launch all notebooks concurrently on one GPU.

Notebooks construct a deterministic ZIP from visible suite source, then use the existing versioned working-directory mechanism. There is no giant embedded base64 payload in the notebook. Editing source changes the working-copy hash; it does not overwrite an earlier experiment.

From the repository root, lightweight checks and result inspection need no GPU:

```bash
python supplementary/tools/check_release.py
python supplementary/tools/index_results.py
```

With the scientific dependencies installed, run the tiny-model implementation tests:

```bash
python supplementary/tools/run_tests.py
# Or one suite:
python supplementary/tools/run_tests.py --suite 11_adjoints
```

Each suite runs in a separate process because several studies intentionally contain different versions of modules with the same names. Do not combine their Python paths or install all suites into one merged package.

## What is included

Readable experiment source, cleaned notebooks, existing tests, configuration defaults, the cohort/model revision evidence available to the later studies, and a compact subset of observed result tables and figures. Source archive hashes are in `SOURCE_PROVENANCE.json`; result archive hashes and subset counts are in `results/PROVENANCE.json`. `MANIFEST.json` records release file hashes.

No model weights, credentials, raw activation caches or large gradient caches are distributed. Compact results cannot resume the original GPU runs or support every new per-prompt analysis. Some earlier suites require fresh PopQA construction or previous run input; their notebook configuration and original README describe those requirements.

## Scientific scope

Origin regimes are observational, success-calibrated categories, not verified causal stages or proof that a fact is absent from weights. Frozen reconstruction is exact conditional accounting; it is not the live Jacobian. Norm fractions are not additive causal shares. Null and singleton-freeze comparisons have the scope described in [definitions](docs/DEFINITIONS.md).

This package preserves the experiment implementations rather than changing them to match stronger prose claims. See the reproduction guide for known manuscript/code differences that must be resolved in the final paper.

## Export a compact subset from a future run

```bash
python supplementary/tools/export_results.py /path/to/completed_run --out /path/to/new_results.zip
```

This reads the run without changing it. It excludes large caches, logs and per-prompt/per-check files by default. Use `--include-rows` if you need those scalar report tables, and inspect `EXPORT_MANIFEST.json` for excluded large files. Export after workers finish for a consistent final artifact.
