"""Embedding-based breakpoint chunking with injectable components."""

from collections.abc import Callable
from typing import Any

from chunkbench.chunking.base import BaseChunker
from chunkbench.chunking.boundaries import cosine_distance, select_boundaries
from chunkbench.chunking.postprocessing import (
    cap_groups,
    contiguous_groups,
    materialize_chunks,
    merge_small_contiguous_groups,
)
from chunkbench.chunking.segments import Segment, sentence_segments
from chunkbench.chunking.validation import validate_advanced_chunks
from chunkbench.common.types import Chunk, Document
from chunkbench.embedding.base import BaseEmbedder, HashingEmbedder


class SemanticBreakpointChunker(BaseChunker):
    """Split where adjacent sentence embedding distances cross a policy threshold."""

    def __init__(
        self,
        segmenter: Callable[[Document], list[Segment]] = sentence_segments,
        embedder: BaseEmbedder | None = None,
        threshold: dict[str, Any] | Callable[[list[float]], float] | None = None,
        min_chunk_tokens: int = 32,
        max_chunk_tokens: int | None = None,
        merge_small_chunks: bool = True,
        **_: object,
    ) -> None:
        self.segmenter = segmenter
        self.embedder = embedder or HashingEmbedder()
        self.threshold = threshold or {"type": "percentile", "value": 90}
        self.min_chunk_tokens = int(min_chunk_tokens)
        self.max_chunk_tokens = (
            int(max_chunk_tokens) if max_chunk_tokens is not None else None
        )
        self.merge_small_chunks = bool(merge_small_chunks)

    def chunk(self, document: Document) -> list[Chunk]:
        """Segment, score adjacency, then materialise lossless contiguous groups."""
        segments = self.segmenter(document)
        if not segments:
            return []
        vectors = self.embedder.encode_documents([segment.text for segment in segments])
        distances = [
            cosine_distance(vectors[index], vectors[index + 1])
            for index in range(len(segments) - 1)
        ]
        threshold, boundaries = select_boundaries(distances, self.threshold)
        groups = contiguous_groups(segments, boundaries)
        if self.merge_small_chunks:
            groups = merge_small_contiguous_groups(groups, self.min_chunk_tokens)
        groups = cap_groups(groups, self.max_chunk_tokens)
        chunks = materialize_chunks(
            document,
            groups,
            "semantic-breakpoint",
            {
                "adjacent_distances": distances,
                "threshold": threshold,
                "selected_boundary_positions": sorted(boundaries),
                "min_chunk_tokens": self.min_chunk_tokens,
                "max_chunk_tokens": self.max_chunk_tokens,
            },
        )
        validate_advanced_chunks(document, segments, chunks, self.max_chunk_tokens)
        return chunks
