"""Metric suite orchestration independent of retrieval backend."""

from chunkbench.common.types import (
    Chunk,
    DatasetBundle,
    EvidenceCoverage,
    RetrievalHit,
)
from chunkbench.eval.constants import K_VALUES, METRIC_NAMES, TOKEN_BUDGETS
from chunkbench.eval.metrics import (
    document_recall_at_k,
    evidence_coverage_at_k,
    evidence_recall_macro_at_k,
    evidence_recall_micro_at_k,
    hit_at_k,
    mrr_at_k,
)
from chunkbench.eval.redundancy import redundancy_at_k
from chunkbench.eval.token_budget import select_within_token_budget


def evaluate(
    bundle: DatasetBundle,
    chunks: list[Chunk],
    hits: list[RetrievalHit],
    coverage: list[EvidenceCoverage],
    strict_token_budget: bool = True,
) -> dict[str, float]:
    """Compute the canonical 23 benchmark metrics."""
    result: dict[str, float] = {}
    for k in K_VALUES:
        result[f"Hit@{k}"] = hit_at_k(bundle, hits, coverage, k)
        result[f"MRR@{k}"] = mrr_at_k(bundle, hits, coverage, k)
        result[f"EvidenceRecallMacro@{k}"] = evidence_recall_macro_at_k(
            bundle, hits, coverage, k
        )
        result[f"EvidenceRecallMicro@{k}"] = evidence_recall_micro_at_k(
            bundle, hits, coverage, k
        )
        result[f"DocumentRecall@{k}"] = document_recall_at_k(bundle, hits, k)
        result[f"EvidenceCoverage@{k}"] = evidence_coverage_at_k(
            bundle, hits, coverage, k
        )
        result[f"Redundancy@{k}"] = redundancy_at_k(chunks, hits, k)
    for budget in TOKEN_BUDGETS:
        selected = select_within_token_budget(hits, chunks, budget, strict_token_budget)
        depth = max((hit.rank for hit in selected), default=0)
        result[f"EvidenceRecallMacro@{budget}Tokens"] = evidence_recall_macro_at_k(
            bundle, selected, coverage, depth
        )
        result[f"EvidenceCoverage@{budget}Tokens"] = evidence_coverage_at_k(
            bundle, selected, coverage, depth
        )
    if set(result) != set(METRIC_NAMES):
        raise RuntimeError("Evaluator output does not match canonical metric names")
    return result
