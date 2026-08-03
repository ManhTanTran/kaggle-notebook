"""Graded retrieval metrics and NQ-hard multi-label diagnostics."""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd


def dcg(relevances: Iterable[float], k: int) -> float:
    values = np.asarray(list(relevances), dtype=float)[:k]
    if not len(values):
        return 0.0
    discounts = np.log2(np.arange(2, len(values) + 2))
    gains = np.power(2.0, values) - 1.0
    return float(np.sum(gains / discounts))


def ndcg_at_k(ranked_ids: Iterable[str], relevance: dict[str, float], k: int) -> float:
    ranked = [str(item_id) for item_id in ranked_ids][:k]
    actual = [float(relevance.get(item_id, 0.0)) for item_id in ranked]
    ideal = sorted((float(value) for value in relevance.values()), reverse=True)[:k]
    ideal_dcg = dcg(ideal, k)
    return 0.0 if ideal_dcg == 0.0 else dcg(actual, k) / ideal_dcg


def recall_at_k(
    ranked_ids: Iterable[str], relevance: dict[str, float], k: int
) -> float:
    relevant = {
        str(item_id) for item_id, value in relevance.items() if float(value) > 0.0
    }
    if not relevant:
        return 0.0
    retrieved = set(str(item_id) for item_id in list(ranked_ids)[:k])
    return len(relevant & retrieved) / len(relevant)


def evaluate_rankings(
    rankings: pd.DataFrame,
    qrels: pd.DataFrame,
    item_column: str,
    rank_column: str,
    ndcg_ks: tuple[int, ...] = (10,),
    recall_ks: tuple[int, ...] = (5, 20, 100),
    prefix: str = "passage",
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    qrel_groups = {
        str(query_id): dict(
            zip(
                group[item_column].astype(str),
                group["relevance"].astype(float),
                strict=True,
            )
        )
        for query_id, group in qrels.groupby("query_id", sort=False)
    }
    ranking_groups = {
        str(query_id): group.sort_values(rank_column)[item_column].astype(str).tolist()
        for query_id, group in rankings.groupby("query_id", sort=False)
    }
    for query_id, relevance in qrel_groups.items():
        ranked = ranking_groups.get(query_id, [])
        row: dict[str, float | str] = {"query_id": query_id}
        for k in ndcg_ks:
            row[f"{prefix}_ndcg@{k}"] = ndcg_at_k(ranked, relevance, k)
        for k in recall_ks:
            row[f"{prefix}_recall@{k}"] = recall_at_k(ranked, relevance, k)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_query_metrics(per_query: pd.DataFrame) -> dict[str, float]:
    metric_columns = [
        column for column in per_query if "@" in column or column.endswith("_rate")
    ]
    return {
        column: float(pd.to_numeric(per_query[column], errors="coerce").mean())
        for column in metric_columns
    }


def document_qrels(qrels: pd.DataFrame, passages: pd.DataFrame) -> pd.DataFrame:
    mapped = qrels.merge(
        passages.loc[:, ["passage_id", "document_id"]],
        on="passage_id",
        how="left",
        validate="many_to_one",
    )
    if mapped["document_id"].isna().any():
        raise ValueError(
            "Cannot derive document qrels: a passage has no document mapping"
        )
    return (
        mapped.groupby(["query_id", "document_id"], as_index=False)["relevance"]
        .max()
        .sort_values(["query_id", "document_id"])
    )


def category_labels(value: object) -> set[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    allowed = {"CR", "MT", "MHR", "AC"}
    return {
        part.upper()
        for part in re.split(r"[,;|\s]+", str(value).strip())
        if part.upper() in allowed
    }


def nq_hard_by_category(
    per_query: pd.DataFrame, metadata: pd.DataFrame
) -> pd.DataFrame:
    merged = per_query.merge(
        metadata.loc[:, ["query_id", "question_type"]], on="query_id", how="left"
    )
    rows: list[dict[str, float | int | str]] = []
    categories = ("overall", "CR", "MT", "MHR", "AC")
    for category in categories:
        selected = (
            merged
            if category == "overall"
            else merged.loc[
                merged["question_type"].map(
                    lambda value, selected=category: selected in category_labels(value)
                )
            ]
        )
        row: dict[str, float | int | str] = {
            "category": category,
            "query_count": int(len(selected)),
        }
        for metric in ("passage_ndcg@10", "passage_recall@100"):
            row[metric] = float(selected[metric].mean()) if len(selected) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
