# Reproduction and provenance

## Environment

The original implementation tests used Transformers 4.48.3. Use that version: hooks and architecture-specific eager-attention behavior are version-sensitive. Existing GPU runs used CUDA-enabled PyTorch and float32. The release does not choose a CUDA wheel for the user's hardware or install a login token. `requirements.txt` lists other runtime/test dependencies. Archived environment JSONs document observed campaigns; a broad dependency range is not an exact lockfile.

The later engine supports the tested Llama, Qwen2 and OLMo2 architectures. It is not a general implementation for all Hugging Face causal language models. Float32 8B weights alone require roughly 32 GB; gradients, activations and caches add memory. Runtime has not been re-benchmarked for this release.

## Data, checkpoints and configuration

Later suites bundle `evidence/data/cohort.json`, model repository/revision records and, where needed, regime/calibration tables. Earlier suites may build a fresh PopQA cohort or ask for previous results. Follow their configuration rather than supplying placeholder paths. Preserve source dataset attribution and model access/license requirements; no model weights are redistributed here.

Inspect both the notebook settings and `results/observed/**/configuration.json`. Defaults are often bounded exploratory runs, whereas the completed campaigns sometimes use all 2,995 prompts. The notebooks retain the original launch/monitor/export workflows. Optional Jacobian, SGD and curvature stages require enabling their switches before launch.

Observed result metadata retains original scientific settings; absolute home paths in metadata copies are replaced by `<original-run-path>/...`. These files are provenance records, not portable ready-to-run queue plans. Use the notebooks to construct new paths and a new run name. Do not resume a run with changed configuration or another source version.

## Compact results

`results/observed` contains selected summary CSVs, plots, audits and configuration/checkpoint metadata. It excludes per-prompt/per-check tables by filename, logs, large raw tensors and model weights. Some summary tables retain subgroup rows. Archive hashes in `results/PROVENANCE.json` identify the source exports. A successful queue exit is not evidence that every finite-difference certificate passed; inspect the certificate summaries and unresolved counts.

The summary subset supports inspection of the published aggregate findings, not arbitrary reaggregation, relabelling of gradient rows or resuming captures. Full raw outputs remain separate artifacts. The included plot scripts generate study reports; the hand-assembled manuscript figures and tables are not represented as a single guaranteed reproduction command.

## Manuscript reconciliation

Before final submission, align prose with `DEFINITIONS.md`, particularly the signed share and zero-floor classifier, singleton denominators, observational scope of variance/crossover, and subsampled certificates. This release does not modify results to satisfy a stronger claim. Report the distinction between first and persistent crossings and between condition-specific and common-denominator quantities.

## Publication

Copy the entire `supplementary/` folder into the existing repository; review and commit it as a normal change. The package contains no Git history and does not overwrite the repository's existing root README. Add a link from that README to `supplementary/README.md`. No upload or push is performed by these scripts.

No new blanket software license has been selected on the author's behalf. Retain the repository's applicable license and third-party notices; if it has none, the author should choose a license before advertising unrestricted reuse. The experiment code may be shared publicly on GitHub, but a public identified repository is distinct from an anonymized TMLR supplementary upload.
