"""Normalized DAPR tables, integrity validation, audit, and query sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "documents": ("document_id", "title", "text"),
    "passages": ("passage_id", "document_id", "passage_text", "passage_position"),
    "queries": ("query_id", "query_text", "dataset", "split"),
    "qrels": ("query_id", "passage_id", "relevance"),
}


@dataclass(frozen=True)
class NormalizedDAPRDataset:
    name: str
    documents: pd.DataFrame
    passages: pd.DataFrame
    queries: pd.DataFrame
    qrels: pd.DataFrame
    query_metadata: pd.DataFrame | None = None


def _missing_columns(frame: pd.DataFrame, required: tuple[str, ...]) -> list[str]:
    return sorted(set(required) - set(frame.columns))


def validate_dataset(dataset: NormalizedDAPRDataset) -> None:
    errors: list[str] = []
    tables = {
        "documents": dataset.documents,
        "passages": dataset.passages,
        "queries": dataset.queries,
        "qrels": dataset.qrels,
    }
    for name, frame in tables.items():
        missing = _missing_columns(frame, REQUIRED_COLUMNS[name])
        if missing:
            errors.append(f"{name} missing columns {missing}")
    if errors:
        raise ValueError("Invalid normalized DAPR dataset:\n- " + "\n- ".join(errors))

    for table_name, frame, id_column in (
        ("documents", dataset.documents, "document_id"),
        ("passages", dataset.passages, "passage_id"),
        ("queries", dataset.queries, "query_id"),
    ):
        if (
            frame[id_column].isna().any()
            or (frame[id_column].astype(str).str.len() == 0).any()
        ):
            errors.append(f"{table_name}.{id_column} contains null/empty IDs")
        duplicates = (
            frame.loc[frame[id_column].duplicated(), id_column]
            .astype(str)
            .head(5)
            .tolist()
        )
        if duplicates:
            errors.append(
                f"{table_name}.{id_column} is not unique; examples={duplicates}"
            )

    document_ids = set(dataset.documents["document_id"].astype(str))
    passage_document_ids = set(dataset.passages["document_id"].astype(str))
    missing_documents = sorted(passage_document_ids - document_ids)[:5]
    if missing_documents:
        errors.append(
            f"passages reference unknown documents; examples={missing_documents}"
        )

    query_ids = set(dataset.queries["query_id"].astype(str))
    passage_ids = set(dataset.passages["passage_id"].astype(str))
    missing_queries = sorted(set(dataset.qrels["query_id"].astype(str)) - query_ids)[:5]
    missing_passages = sorted(
        set(dataset.qrels["passage_id"].astype(str)) - passage_ids
    )[:5]
    if missing_queries:
        errors.append(f"qrels reference unknown queries; examples={missing_queries}")
    if missing_passages:
        errors.append(f"qrels reference unknown passages; examples={missing_passages}")

    relevance = pd.to_numeric(dataset.qrels["relevance"], errors="coerce")
    if relevance.isna().any() or (relevance < 0).any():
        errors.append("qrels.relevance must contain non-negative numeric labels")

    positions = pd.to_numeric(dataset.passages["passage_position"], errors="coerce")
    supplied = dataset.passages["passage_position"].notna()
    invalid_positions = supplied & (
        positions.isna() | (positions < 0) | (positions % 1 != 0)
    )
    if invalid_positions.any():
        errors.append("passage_position must be a non-negative integer when supplied")

    if (
        dataset.queries["query_text"].isna().any()
        or dataset.queries["split"].isna().any()
    ):
        errors.append("queries require non-null query_text and split")
    if (
        dataset.documents["text"].isna().any()
        or dataset.passages["passage_text"].isna().any()
    ):
        errors.append("document and passage text must be non-null")

    if dataset.query_metadata is not None:
        if (
            "query_id" not in dataset.query_metadata
            or "question_type" not in dataset.query_metadata
        ):
            errors.append("query_metadata requires query_id and question_type")
        else:
            unknown = set(dataset.query_metadata["query_id"].astype(str)) - query_ids
            if unknown:
                errors.append(
                    "query_metadata references unknown queries; "
                    f"examples={sorted(unknown)[:5]}"
                )

    if errors:
        raise ValueError("Invalid normalized DAPR dataset:\n- " + "\n- ".join(errors))


def dataset_audit(dataset: NormalizedDAPRDataset) -> dict[str, Any]:
    validate_dataset(dataset)
    title = dataset.documents["title"]
    title_text = title.fillna("").astype(str).str.strip()
    passages_per_document = dataset.passages.groupby("document_id").size()
    return {
        "dataset": dataset.name,
        "documents": int(len(dataset.documents)),
        "passages": int(len(dataset.passages)),
        "queries": int(len(dataset.queries)),
        "qrels": int(len(dataset.qrels)),
        "splits": ",".join(sorted(dataset.queries["split"].astype(str).unique())),
        "graded_qrels": bool(dataset.qrels["relevance"].max() > 1),
        "missing_title_rate": float((title.isna() | title_text.eq("")).mean()),
        "noisy_title_rate": float(title_text.str.len().gt(300).mean()),
        "mean_passages_per_document": float(passages_per_document.mean()),
    }


def sample_queries(
    dataset: NormalizedDAPRDataset,
    sample_size: int | None,
    seed: int,
    split: str | None = None,
) -> NormalizedDAPRDataset:
    """Sample query IDs only; corpus tables remain intact and qrels are filtered."""
    validate_dataset(dataset)
    queries = dataset.queries
    if split is not None:
        queries = queries.loc[queries["split"].astype(str) == split]
        if queries.empty:
            raise ValueError(
                f"Dataset {dataset.name!r} has no queries for split {split!r}"
            )
    if sample_size is not None and sample_size < len(queries):
        rng = np.random.default_rng(seed)
        chosen = rng.choice(
            queries["query_id"].to_numpy(), size=sample_size, replace=False
        )
        order = {str(query_id): rank for rank, query_id in enumerate(chosen)}
        queries = queries.loc[queries["query_id"].astype(str).isin(order)].copy()
        queries["_sample_order"] = queries["query_id"].astype(str).map(order)
        queries = queries.sort_values("_sample_order").drop(columns="_sample_order")
    else:
        queries = queries.copy()
    selected = set(queries["query_id"].astype(str))
    qrels = dataset.qrels.loc[
        dataset.qrels["query_id"].astype(str).isin(selected)
    ].copy()
    metadata = None
    if dataset.query_metadata is not None:
        metadata = dataset.query_metadata.loc[
            dataset.query_metadata["query_id"].astype(str).isin(selected)
        ].copy()
    result = NormalizedDAPRDataset(
        name=dataset.name,
        documents=dataset.documents.copy(),
        passages=dataset.passages.copy(),
        queries=queries,
        qrels=qrels,
        query_metadata=metadata,
    )
    validate_dataset(result)
    if split is not None and set(result.queries["split"].astype(str)) != {split}:
        raise AssertionError("sampling introduced cross-split leakage")
    return result
