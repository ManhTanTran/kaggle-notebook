"""Experiment registry and reusable single-dataset runner."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from .config import ALL_METHODS, RunConfig, validate_config
from .metrics import aggregate_query_metrics, document_qrels, evaluate_rankings
from .retrieval import HHRPipeline, build_retrievers
from .schema import NormalizedDAPRDataset, validate_dataset

Method = Literal["sparse", "dense", "combined"]
CombinedStrategy = Literal["hhr_interleave", "rrf"]


@dataclass(frozen=True)
class HHRExperiment:
    name: str
    document_method: Method
    passage_method: Method
    combined_strategy: CombinedStrategy
    document_top_k: int
    passage_top_k: int


@dataclass
class ExperimentResult:
    dataset: str
    experiment: HHRExperiment
    status: str
    aggregate_metrics: dict[str, float]
    per_query_metrics: pd.DataFrame
    document_rankings: pd.DataFrame
    passage_rankings: pd.DataFrame
    latency: pd.DataFrame
    index_metadata: dict[str, Any]
    run_metadata: dict[str, Any]


def build_experiment_registry(config: RunConfig) -> dict[str, HHRExperiment]:
    registry: dict[str, HHRExperiment] = {}
    for method in ALL_METHODS:
        document_method, passage_method = method.split("+")
        registry[method] = HHRExperiment(
            name=method,
            document_method=document_method,  # type: ignore[arg-type]
            passage_method=passage_method,  # type: ignore[arg-type]
            combined_strategy=config.combined_strategy,  # type: ignore[arg-type]
            document_top_k=config.document_top_k,
            passage_top_k=config.passage_top_k,
        )
    return registry


def run_hhr_experiment(
    dataset: NormalizedDAPRDataset,
    experiment: HHRExperiment,
    config: RunConfig,
    retrievers: dict[str, dict[str, Any]] | None = None,
) -> ExperimentResult:
    validate_config(config)
    validate_dataset(dataset)
    if (
        experiment.document_top_k != config.document_top_k
        or experiment.passage_top_k != config.passage_top_k
    ):
        raise ValueError(
            "Experiment candidate budgets must match the shared run configuration"
        )
    if retrievers is None:
        retrievers = build_retrievers(
            dataset.documents, dataset.passages, config, dataset.name
        )
    pipeline = HHRPipeline(
        retrievers["document"][experiment.document_method],
        retrievers["passage"][experiment.passage_method],
        dataset.passages,
        config,
    )
    document_rows: list[dict[str, Any]] = []
    passage_rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    unique_counts: dict[str, int] = {}
    selected_documents: dict[str, set[str]] = {}

    for query in dataset.queries.itertuples(index=False):
        start = time.perf_counter()
        result = pipeline.retrieve(str(query.query_text))
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        query_id = str(query.query_id)
        latency_rows.append({"query_id": query_id, "latency_ms": elapsed_ms})
        unique_counts[query_id] = result.unique_result_count
        selected_documents[query_id] = {
            str(hit["item_id"]) for hit in result.document_rankings
        }
        for hit in result.document_rankings:
            document_rows.append(
                {
                    "query_id": query_id,
                    "document_id": str(hit["item_id"]),
                    "document_rank": int(hit["document_rank"]),
                    "document_score": float(hit["document_score"]),
                    "sparse_rank": hit.get("sparse_rank"),
                    "dense_rank": hit.get("dense_rank"),
                    "source_method": hit.get("source_method"),
                }
            )
        for hit in result.passage_rankings:
            passage_rows.append(
                {
                    "query_id": query_id,
                    "passage_id": str(hit["item_id"]),
                    "document_id": str(hit["document_id"]),
                    "passage_rank": int(hit["passage_rank"]),
                    "passage_score": float(hit["passage_score"]),
                    "document_rank": int(hit["document_rank"]),
                    "document_score": float(hit["document_score"]),
                    "sparse_rank": hit.get("sparse_rank"),
                    "dense_rank": hit.get("dense_rank"),
                    "source_method": hit.get("source_method"),
                }
            )

    document_rankings = pd.DataFrame(document_rows)
    passage_rankings = pd.DataFrame(passage_rows)
    latency = pd.DataFrame(latency_rows)
    passage_metrics = evaluate_rankings(
        passage_rankings,
        dataset.qrels,
        "passage_id",
        "passage_rank",
        ndcg_ks=(10,),
        recall_ks=(5, 20, 100),
        prefix="passage",
    )
    derived_document_qrels = document_qrels(dataset.qrels, dataset.passages)
    document_metrics = evaluate_rankings(
        document_rankings,
        derived_document_qrels,
        "document_id",
        "document_rank",
        ndcg_ks=(10,),
        recall_ks=(5, 20, 100),
        prefix="document",
    )
    per_query = passage_metrics.merge(
        document_metrics, on="query_id", validate="one_to_one"
    )
    per_query = per_query.merge(latency, on="query_id", validate="one_to_one")
    per_query["unique_result_count"] = (
        per_query["query_id"].map(unique_counts).astype(int)
    )

    passage_doc_map = dict(
        zip(
            dataset.passages["passage_id"].astype(str),
            dataset.passages["document_id"].astype(str),
            strict=True,
        )
    )
    survival: dict[str, float] = {}
    for query_id, group in dataset.qrels.groupby("query_id"):
        relevant_documents = {
            passage_doc_map[str(row.passage_id)]
            for row in group.itertuples(index=False)
            if float(row.relevance) > 0
        }
        survived = relevant_documents & selected_documents.get(str(query_id), set())
        survival[str(query_id)] = (
            len(survived) / len(relevant_documents) if relevant_documents else 0.0
        )
    per_query["candidate_survival_rate"] = (
        per_query["query_id"].map(survival).astype(float)
    )

    aggregate = aggregate_query_metrics(per_query)
    aggregate.update(
        {
            "mean_query_latency_ms": float(latency["latency_ms"].mean()),
            "p50_query_latency_ms": float(latency["latency_ms"].quantile(0.50)),
            "p95_query_latency_ms": float(latency["latency_ms"].quantile(0.95)),
            "mean_unique_result_count": float(per_query["unique_result_count"].mean()),
        }
    )
    document_metadata = retrievers["document"][experiment.document_method].metadata
    passage_metadata = retrievers["passage"][experiment.passage_method].metadata
    index_metadata = {
        "document": document_metadata,
        "passage": passage_metadata,
        "total_storage_bytes": int(
            document_metadata["storage_bytes"] + passage_metadata["storage_bytes"]
        ),
    }
    return ExperimentResult(
        dataset=dataset.name,
        experiment=experiment,
        status="completed",
        aggregate_metrics=aggregate,
        per_query_metrics=per_query,
        document_rankings=document_rankings,
        passage_rankings=passage_rankings,
        latency=latency,
        index_metadata=index_metadata,
        run_metadata={
            "query_count": int(len(dataset.queries)),
            "document_count": int(len(dataset.documents)),
            "passage_count": int(len(dataset.passages)),
            "reranker": None,
            "combined_strategy": experiment.combined_strategy,
            "score_fusion": config.score_fusion,
            "dense_backend": config.dense_backend,
        },
    )


def run_experiment_matrix(
    datasets: list[NormalizedDAPRDataset], config: RunConfig
) -> list[ExperimentResult]:
    registry = build_experiment_registry(config)
    results: list[ExperimentResult] = []
    for dataset in datasets:
        shared_retrievers = build_retrievers(
            dataset.documents, dataset.passages, config, dataset.name
        )
        for method in config.methods:
            results.append(
                run_hhr_experiment(dataset, registry[method], config, shared_retrievers)
            )
    return results
