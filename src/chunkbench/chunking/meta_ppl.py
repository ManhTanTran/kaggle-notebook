"""PPL-minimum Meta-Chunking reimplementation with explicit dynamic policy."""

from typing import Any

from chunkbench.chunking.base import BaseChunker
from chunkbench.chunking.postprocessing import (
    cap_groups,
    contiguous_groups,
    materialize_chunks,
)
from chunkbench.chunking.segments import sentence_segments
from chunkbench.chunking.validation import validate_advanced_chunks
from chunkbench.common.types import Chunk, Document
from chunkbench.scoring.perplexity import (
    BasePerplexityScorer,
    DeterministicPerplexityScorer,
)


def find_local_minima(values: list[float], prominence: float) -> set[int]:
    """Return local minima with the Meta-Chunking prominence rule."""
    result: set[int] = set()
    for index in range(1, len(values) - 1):
        current, before, after = values[index], values[index - 1], values[index + 1]
        if (
            current < before
            and current <= after
            and max(before - current, after - current) >= prominence
        ):
            result.add(index)
    return result


class MetaPPLChunker(BaseChunker):
    """PPL boundary detection; dynamic_512 adds a documented capped merge policy."""

    def __init__(
        self,
        variant: str,
        scorer: BasePerplexityScorer | None = None,
        prominence: float = 0.05,
        context_policy: str = "all_previous_segments",
        max_chunk_tokens: int | None = None,
        min_chunk_tokens: int = 0,
        **_: Any,
    ) -> None:
        if variant not in {"meta_ppl_raw", "meta_ppl_dynamic_512"}:
            raise ValueError(f"Unknown Meta-PPL variant: {variant}")
        self.variant = variant
        self.scorer = scorer or DeterministicPerplexityScorer()
        self.prominence = float(prominence)
        self.context_policy = context_policy
        self.max_chunk_tokens = (
            int(max_chunk_tokens)
            if max_chunk_tokens is not None
            else (512 if variant == "meta_ppl_dynamic_512" else None)
        )
        self.min_chunk_tokens = int(min_chunk_tokens)

    def chunk(self, document: Document) -> list[Chunk]:
        """Score transitions, split at local minima, then apply a declared cap."""
        segments = sentence_segments(document)
        if not segments:
            return []
        scores: list[float] = []
        for index, segment in enumerate(segments):
            if self.context_policy == "all_previous_segments":
                context_segments = segments[:index]
            elif self.context_policy == "previous_segment":
                context_segments = segments[max(0, index - 1) : index]
            elif self.context_policy == "none":
                context_segments = []
            else:
                raise ValueError(
                    "Unknown Meta-PPL context_policy: " f"{self.context_policy}"
                )
            context = " ".join(item.text for item in context_segments)
            scores.append(self.scorer.score_transition(context, segment.text))
        boundary_after = find_local_minima(scores, self.prominence)
        groups = contiguous_groups(segments, boundary_after)
        groups = cap_groups(groups, self.max_chunk_tokens)
        chunks = materialize_chunks(
            document,
            groups,
            "meta-ppl",
            {
                "variant": self.variant,
                "segment_scores": scores,
                "boundary_after": sorted(boundary_after),
                "prominence": self.prominence,
                "context_policy": self.context_policy,
                "dynamic_merge": self.variant == "meta_ppl_dynamic_512",
                "max_chunk_tokens": self.max_chunk_tokens,
            },
        )
        validate_advanced_chunks(document, segments, chunks, self.max_chunk_tokens)
        return chunks
