"""Pure information-retrieval metric functions."""

from collections import defaultdict

from chunkbench.common.types import (
    DatasetBundle,
    EvidenceCoverage,
    RetrievalHit,
)


def _top(hits: list[RetrievalHit], k: int) -> dict[str, list[RetrievalHit]]:
    grouped: dict[str, list[RetrievalHit]] = defaultdict(list)
    for hit in hits:
        if hit.rank <= k:
            grouped[hit.query_id].append(hit)
    return grouped


def _coverage_lookup(
    rows: list[EvidenceCoverage],
) -> dict[tuple[str, str], frozenset[int]]:
    lookup: dict[tuple[str, str], frozenset[int]] = {}
    for row in rows:
        key = (row.chunk_id, row.evidence_id)
        lookup[key] = lookup.get(key, frozenset()) | row.covered_token_ids
    return lookup


def _covered_evidence(
    query_id: str,
    query_hits: list[RetrievalHit],
    bundle: DatasetBundle,
    rows: list[EvidenceCoverage],
) -> tuple[int, int]:
    lookup = _coverage_lookup(rows)
    items = [item for item in bundle.evidence if item.query_id == query_id]
    covered = 0
    for item in items:
        union = frozenset().union(
            *(
                lookup.get((hit.chunk_id, item.evidence_id), frozenset())
                for hit in query_hits
            )
        )
        covered += int(len(union) >= item.token_count)
    return covered, len(items)


def hit_at_k(
    bundle: DatasetBundle,
    hits: list[RetrievalHit],
    coverage: list[EvidenceCoverage],
    k: int,
) -> float:
    """Mean query indicator that at least one evidence unit is fully covered."""
    grouped = _top(hits, k)
    values = [
        float(
            _covered_evidence(
                query.query_id, grouped[query.query_id], bundle, coverage
            )[0]
            > 0
        )
        for query in bundle.queries
    ]
    return sum(values) / len(values) if values else 0.0


def mrr_at_k(
    bundle: DatasetBundle,
    hits: list[RetrievalHit],
    coverage: list[EvidenceCoverage],
    k: int,
) -> float:
    """Mean reciprocal rank of the first chunk fully covering any evidence."""
    lookup = _coverage_lookup(coverage)
    scores = []
    for query in bundle.queries:
        evidence = [item for item in bundle.evidence if item.query_id == query.query_id]
        reciprocal = 0.0
        for hit in sorted(_top(hits, k)[query.query_id], key=lambda item: item.rank):
            if any(
                len(lookup.get((hit.chunk_id, item.evidence_id), frozenset()))
                >= item.token_count
                for item in evidence
            ):
                reciprocal = 1.0 / hit.rank
                break
        scores.append(reciprocal)
    return sum(scores) / len(scores) if scores else 0.0


def evidence_recall_macro_at_k(
    bundle: DatasetBundle,
    hits: list[RetrievalHit],
    coverage: list[EvidenceCoverage],
    k: int,
) -> float:
    """Average per-query fraction of fully retrieved evidence units."""
    grouped = _top(hits, k)
    values = []
    for query in bundle.queries:
        covered, total = _covered_evidence(
            query.query_id, grouped[query.query_id], bundle, coverage
        )
        values.append(covered / total if total else 0.0)
    return sum(values) / len(values) if values else 0.0


def evidence_recall_micro_at_k(
    bundle: DatasetBundle,
    hits: list[RetrievalHit],
    coverage: list[EvidenceCoverage],
    k: int,
) -> float:
    """Corpus fraction of fully retrieved evidence units."""
    grouped = _top(hits, k)
    covered = total = 0
    for query in bundle.queries:
        query_covered, query_total = _covered_evidence(
            query.query_id, grouped[query.query_id], bundle, coverage
        )
        covered += query_covered
        total += query_total
    return covered / total if total else 0.0


def document_recall_at_k(
    bundle: DatasetBundle, hits: list[RetrievalHit], k: int
) -> float:
    """Macro recall of relevant document identifiers per query."""
    grouped = _top(hits, k)
    values = []
    for query in bundle.queries:
        retrieved = {hit.document_id for hit in grouped[query.query_id]}
        relevant = set(query.relevant_document_ids)
        values.append(len(retrieved & relevant) / len(relevant) if relevant else 0.0)
    return sum(values) / len(values) if values else 0.0


def evidence_coverage_at_k(
    bundle: DatasetBundle,
    hits: list[RetrievalHit],
    coverage: list[EvidenceCoverage],
    k: int,
) -> float:
    """Macro query coverage of unique evidence token identifiers."""
    grouped = _top(hits, k)
    lookup = _coverage_lookup(coverage)
    values = []
    for query in bundle.queries:
        numerator = denominator = 0
        for item in (
            item for item in bundle.evidence if item.query_id == query.query_id
        ):
            covered = frozenset().union(
                *(
                    lookup.get((hit.chunk_id, item.evidence_id), frozenset())
                    for hit in grouped[query.query_id]
                )
            )
            numerator += len(covered)
            denominator += item.token_count
        values.append(numerator / denominator if denominator else 0.0)
    return sum(values) / len(values) if values else 0.0
