"""Default lexical redundancy metric."""

import re
from collections import defaultdict
from itertools import combinations

from chunkbench.common.types import Chunk, RetrievalHit


def _lexical_tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def redundancy_at_k(chunks: list[Chunk], hits: list[RetrievalHit], k: int) -> float:
    """Mean query pairwise Jaccard token overlap among top-k chunks."""
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    grouped: dict[str, list[RetrievalHit]] = defaultdict(list)
    for hit in hits:
        if hit.rank <= k:
            grouped[hit.query_id].append(hit)
    query_scores = []
    for query_hits in grouped.values():
        token_sets = [_lexical_tokens(by_id[hit.chunk_id].text) for hit in query_hits]
        pairs = list(combinations(token_sets, 2))
        if not pairs:
            query_scores.append(0.0)
            continue
        scores = [
            len(left & right) / len(left | right) if left | right else 0.0
            for left, right in pairs
        ]
        query_scores.append(sum(scores) / len(scores))
    return sum(query_scores) / len(query_scores) if query_scores else 0.0
