# Phase 1 HHR × DAPR benchmark

This directory documents the original HHR prototype retained for compatibility
with the chunking benchmark. Its implementation lives in `src/hhr_dapr`; no HHR
notebook is tracked in this repository. Active development moved to
[`hierarchical-retrieval-benchmark`](https://github.com/ManhTanTran/hierarchical-retrieval-benchmark).

## Quick start

```powershell
python -m pip install -e ".[dev,hhr]"
pytest
```

Install the `hhr-dense` extras for DRAGON+/FAISS runs.

## Real-data directory contract

Create one directory for each of `ms_marco`, `natural_questions`, `miracl_en`, `genomics`, `conditional_qa`, and `nq_hard`:

```text
data/dapr/<dataset>/
├─ manifest.json
├─ documents.parquet
├─ passages.parquet
├─ queries.parquet
├─ qrels.parquet
└─ query_metadata.parquet  # optional; expected for NQ-hard categories
```

Example manifest:

```json
{
  "documents": "documents.parquet",
  "passages": "passages.parquet",
  "queries": "queries.parquet",
  "qrels": "qrels.parquet",
  "query_metadata": "query_metadata.parquet",
  "column_maps": {
    "documents": {"document_id": "docid", "title": "title", "text": "body"},
    "passages": {"passage_id": "pid", "document_id": "docid", "passage_text": "text", "passage_position": "position"},
    "queries": {"query_id": "qid", "query_text": "query", "dataset": "dataset", "split": "split"},
    "qrels": {"query_id": "qid", "passage_id": "pid", "relevance": "score"},
    "query_metadata": {"query_id": "qid", "question_type": "category"}
  }
}
```

When files already use normalized names, omit `column_maps`. NQ-hard `question_type` may contain multiple labels separated by commas, pipes, semicolons, or whitespace; diagnostics count a query in every applicable category.

## Outputs

Each execution writes beneath:

```text
outputs/<experiment_name>/<run_mode>/<git_commit_or_timestamp>/
```

The exporter writes all required tables and marks unexecuted matrix entries as `not_run`. Index caches are stored separately under `cache/` and are excluded from version control.
BM25 structures, dense embeddings, and FAISS document indices are fingerprinted and reused unless `rebuild_indices=True`.
