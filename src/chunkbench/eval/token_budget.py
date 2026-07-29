"""Rank-preserving token budget selection."""

from collections import defaultdict

from chunkbench.common.types import Chunk, RetrievalHit


def select_within_token_budget(
    hits: list[RetrievalHit],
    chunks: list[Chunk],
    budget: int,
    strict: bool = True,
) -> list[RetrievalHit]:
    """Select ranked hits independently per query within a token budget."""
    sizes = {chunk.chunk_id: chunk.token_count for chunk in chunks}
    grouped: dict[str, list[RetrievalHit]] = defaultdict(list)
    for hit in hits:
        grouped[hit.query_id].append(hit)
    selected = []
    for query_hits in grouped.values():
        used = 0
        for hit in sorted(query_hits, key=lambda item: item.rank):
            size = sizes[hit.chunk_id]
            if strict and used + size > budget:
                break
            selected.append(hit)
            used += size
            if not strict and used >= budget:
                break
    return selected
