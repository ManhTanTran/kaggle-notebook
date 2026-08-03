# Phase 1 HHR × DAPR benchmark

The Kaggle notebook owns everything specific to DAPR: Hugging Face download,
official-schema normalization, validation, query sampling, zero-shot protocol,
NQ-hard categories, and the synthetic smoke fixture. The reusable `src/hhr_dapr`
package owns retrieval, metrics, experiment execution, caching, and artifacts.

## Kaggle: notebook-only workflow

1. Push this repository to GitHub.
2. In Kaggle, create a notebook and enable **Internet**. A GPU is recommended for
   real dense runs but is unnecessary for smoke mode.
3. Import
   `notebooks/hhr_dapr/01_phase1_hhr_dapr.ipynb` from GitHub, or upload that file.
4. Run all cells unchanged. The default smoke run clones the repository when
   needed, installs dependencies, and validates the full pipeline without
   downloading DAPR.
5. Edit only the central configuration cell. Change `run_mode` to `baseline` or
   `full`; initially override `datasets` with one dataset and optionally set a
   query sample size.
6. Run all again. Real modes download the pinned
   [`UKPLab/dapr`](https://huggingface.co/datasets/UKPLab/dapr) revision through
   `datasets.load_dataset` and cache it under `/kaggle/working/cache`.
7. Download the artifact directory printed by section 16 from Kaggle Output.

MIRACL and Genomics contain millions of passages. Query sampling reduces query
execution but not the corpus index, so the complete matrix can exceed a single
Kaggle session. Do not interpret synthetic smoke metrics as benchmark results.

## Local workflow

```powershell
python -m pip install -e ".[dev,hhr]"
python -m pytest
jupyter notebook notebooks/hhr_dapr/01_phase1_hhr_dapr.ipynb
```

The paired `01_phase1_hhr_dapr.py` is the text source used to review and rebuild
the notebook. Dataset-specific functions remain notebook-local, not reusable
package APIs.

## Normalized boundary

The reusable runner accepts any object with `name`, `documents`, `passages`,
`queries`, `qrels`, and optional `query_metadata` attributes. The notebook emits:

- documents: `document_id`, `title`, `text`
- passages: `passage_id`, `document_id`, `passage_text`, `passage_position`
- queries: `query_id`, `query_text`, `dataset`, `split`
- qrels: `query_id`, `passage_id`, `relevance`

Artifacts are written under
`outputs/<experiment_name>/<run_mode>/<git_commit_or_timestamp>/`. Unscheduled
matrix entries are marked `not_run`, and smoke metadata is explicitly synthetic.
