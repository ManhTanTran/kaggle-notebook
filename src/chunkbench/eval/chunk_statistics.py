"""Chunk distribution and data-quality statistics."""

import re
from collections import Counter
from typing import Any

import numpy as np

from chunkbench.common.types import Chunk


def _percentile(values: list[int], percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values else 0.0


def chunk_statistics(
    chunks: list[Chunk], duplicate_scope: str = "within_document"
) -> dict[str, Any]:
    """Compute all required chunk size and integrity statistics."""
    token_counts = [chunk.token_count for chunk in chunks]
    per_document = list(Counter(chunk.document_id for chunk in chunks).values())
    normalized = [re.sub(r"\s+", " ", chunk.text).strip().lower() for chunk in chunks]
    keys = [
        (chunk.document_id, text) if duplicate_scope == "within_document" else text
        for chunk, text in zip(chunks, normalized, strict=True)
    ]
    duplicates = sum(count - 1 for count in Counter(keys).values() if count > 1)

    def percentage(predicate: list[bool]) -> float:
        return 100.0 * sum(predicate) / len(chunks) if chunks else 0.0

    return {
        "Total Chunks": len(chunks),
        "Chunks per Document Mean": (
            float(np.mean(per_document)) if per_document else 0.0
        ),
        "Chunks per Document Median": (
            float(np.median(per_document)) if per_document else 0.0
        ),
        "Chunks per Document P95": _percentile(per_document, 95),
        "Tokens per Chunk Mean": float(np.mean(token_counts)) if token_counts else 0.0,
        "Tokens per Chunk Median": (
            float(np.median(token_counts)) if token_counts else 0.0
        ),
        "Tokens per Chunk Min": min(token_counts, default=0),
        "Tokens per Chunk Max": max(token_counts, default=0),
        "Tokens per Chunk P90": _percentile(token_counts, 90),
        "Tokens per Chunk P95": _percentile(token_counts, 95),
        "Percentage of Chunks > 256": percentage(
            [count > 256 for count in token_counts]
        ),
        "Percentage of Chunks > 512": percentage(
            [count > 512 for count in token_counts]
        ),
        "Percentage of Chunks > 1024": percentage(
            [count > 1024 for count in token_counts]
        ),
        "Percentage of Non-contiguous Chunks": percentage(
            [not chunk.is_contiguous for chunk in chunks]
        ),
        "Empty Chunk Count": sum(not chunk.text.strip() for chunk in chunks),
        "Duplicate Chunk Count": duplicates,
    }
