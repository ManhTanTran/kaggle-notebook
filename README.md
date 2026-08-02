# chunking-benchmark

Reproducible document-chunking benchmark for the
`qa_evidence_retrieval` profile. Reusable implementation lives in
`src/chunkbench`; notebooks only select configs, invoke runners, and inspect
artifacts.

## Install

```powershell
.\scripts\setup_windows.ps1
```

```bash
./scripts/setup_linux.sh
```

Or run `pip install -e ".[dev]"`.

## Datasets and methods

`CORE_DATASETS` contains QASPER, HotpotQA FullWiki, UIT-ViQuAD, and ViMQA.
Adapters normalize their native schemas into one `DatasetBundle` contract.
The repository retains exactly 12 `CORE_METHODS` and 23 `PRIMARY_METRICS`.
All four fixed/sentence-fixed baselines and eight advanced methods run. Advanced
methods have explicit implementation-fidelity labels; none is reported as
paper-exact without a reproduction check. The deterministic advanced smoke uses
mock backends and is deliberately marked non-publishable.

Full datasets are not distributed by this repository. Local fixtures reflect
the documented raw schemas and power offline tests. See `docs/datasets/` for
evidence policies, licensing constraints, and verification status.

## Run

Backward-compatible single-dataset run:

```bash
python -m chunkbench.cli --config configs/experiments/smoke_test.yaml
```

Four-dataset, four-baseline fixture matrix:

```bash
python -m chunkbench.cli \
  --config configs/experiments/all_qa_datasets_smoke.yaml
```

The full 4 × 12 matrix is configured in
`configs/experiments/all_qa_datasets_core_methods.yaml`. Add authorized local
data at the paths in `configs/datasets/` before running it.

Run every advanced method offline:

```bash
python -m chunkbench.cli --config configs/experiments/advanced_methods_mock_smoke.yaml
```

Optional real family configs are in `configs/experiments/*_real_smoke.yaml`.
Install only the relevant extra (`.[semantic]`, `.[ppl]`, or `.[late]`) and
ensure the named model is locally available. A missing dependency/model is not a
validation pass.

## Extension contracts

Add a dataset by implementing `DatasetAdapter`, registering it with
`register_dataset`, and adding config/contract tests. Neither evaluator nor
runner changes. Add a method by subclassing `BaseChunker` and registering its
factory; benchmark notebooks remain unchanged.

## Outputs

Matrix artifacts use:

```text
outputs/<run_name>/<dataset>/<method>/
outputs/<run_name>/benchmark_metrics.csv
outputs/<run_name>/chunk_statistics.csv
outputs/<run_name>/dataset_validation.csv
outputs/<run_name>/experiment_manifest.json
outputs/<run_name>/completed_runs.json
outputs/<run_name>/failed_runs.json
```

Every completed method directory contains config, environment, chunks,
retrieval, evidence coverage, 23 metrics, chunk statistics, runtime, and a
`method_manifest.json` with fidelity, source commit, backend type, and
publishability. Matrix aggregates expose the same provenance fields.
Config hashes are deterministic; manifests include the Git commit when the
workspace is a Git repository.

## VS Code and Git

Open the workspace root and select the `chunkbench` kernel. Keep raw datasets,
weights, indices, credentials, and outputs out of Git. The fixture matrix is the
portable pre-commit smoke test.

## Hybrid Hierarchical Retrieval on DAPR

The repository also contains the Phase 1 HHR benchmark as a separate
`hhr_dapr` package. Its reusable adapters, sparse/dense retrieval, hierarchical
runner, metrics, protocol guards, and artifact export live in `src/hhr_dapr`.
The orchestration notebook is
`notebooks/hhr_dapr/01_phase1_hhr_dapr.ipynb`.

Run its deterministic offline smoke test with:

```bash
python -m pip install -e ".[dev,hhr]"
jupyter notebook notebooks/hhr_dapr/01_phase1_hhr_dapr.ipynb
```

Install `.[hhr-dense]` for DRAGON+/FAISS benchmark runs. Real DAPR corpora are
not committed; the normalized manifest contract and Phase 1 protocol are in
`docs/hhr_dapr/`.
