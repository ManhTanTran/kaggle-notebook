# Phase 1 experiment contract: HHR on DAPR

## Objective

Select the strongest document-retriever + passage-retriever configuration before adding the Phase 2 HiREC reranker or evidence curation. Phase 1 compares all nine sparse/dense/combined pairs under equal query sets, candidate budgets, and metrics.

## Data contract

The notebook downloads a pinned revision of `UKPLab/dapr` directly from Hugging
Face. Its notebook-local normalizers map the official docs, corpus, queries,
qrels, and NQ-hard configurations to the structural interface consumed by the
reusable runner. Validation checks IDs, references, relevance labels, passage
positions, and counts before retrieval. Dataset-specific assumptions do not live
in `src/hhr_dapr`.

## Evaluation protocol

- Tune only on MS MARCO train/dev.
- Freeze fusion and candidate parameters before evaluating MS MARCO test and the four zero-shot datasets.
- Use NQ-hard only for diagnostics. Multi-label examples contribute to every listed category (CR, MT, MHR, AC).
- Primary metrics are passage nDCG@10 and passage Recall@100. Genomics relevance remains graded for nDCG.
- Synthetic smoke results validate software behavior only and must never be reported as DAPR benchmark results.

## Dense model decision

Benchmark runs default to the asymmetric DRAGON+ checkpoints `facebook/dragon-plus-query-encoder` and `facebook/dragon-plus-context-encoder`. The synthetic smoke run uses a deterministic hashing encoder so it does not download a model. These backends are explicitly distinguished in run metadata.

## Phase 2 interface

`HHRPipeline` accepts an optional reranker implementing `rerank(query, rankings, passages)`. Normalization, metrics, the experiment registry, and artifact schemas do not change when a cross-encoder is added.
