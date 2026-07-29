"""Lossless chunk materialisation and explicit token-cap policies."""

from collections.abc import Iterable

from chunkbench.chunking.base import token_spans
from chunkbench.chunking.segments import Segment
from chunkbench.common.types import Chunk, Document


def contiguous_groups(
    segments: list[Segment], boundaries: set[int]
) -> list[list[Segment]]:
    """Group ordered segments; a boundary at i is after segment i."""
    if not segments:
        return []
    groups: list[list[Segment]] = [[]]
    for index, segment in enumerate(segments):
        groups[-1].append(segment)
        if index in boundaries and index != len(segments) - 1:
            groups.append([])
    return groups


def merge_small_contiguous_groups(
    groups: list[list[Segment]], min_chunk_tokens: int
) -> list[list[Segment]]:
    """Merge undersized groups into their right neighbour, or left at the end."""
    if min_chunk_tokens <= 0 or len(groups) < 2:
        return groups
    result = [list(group) for group in groups]
    index = 0
    while index < len(result):
        if sum(item.token_count for item in result[index]) >= min_chunk_tokens:
            index += 1
            continue
        if index + 1 < len(result):
            result[index + 1] = result[index] + result[index + 1]
            result.pop(index)
        else:
            result[index - 1].extend(result[index])
            result.pop(index)
    return result


def _split_group_by_tokens(
    group: list[Segment], max_chunk_tokens: int
) -> list[list[Segment]]:
    """Split only at source-segment boundaries; oversized segments stay intact."""
    if max_chunk_tokens <= 0:
        raise ValueError("max_chunk_tokens must be positive")
    result: list[list[Segment]] = []
    current: list[Segment] = []
    current_count = 0
    for segment in group:
        if current and current_count + segment.token_count > max_chunk_tokens:
            result.append(current)
            current = []
            current_count = 0
        current.append(segment)
        current_count += segment.token_count
    if current:
        result.append(current)
    return result


def cap_groups(
    groups: list[list[Segment]], max_chunk_tokens: int | None
) -> list[list[Segment]]:
    """Apply a declared cap without silently discarding or duplicating segments."""
    if max_chunk_tokens is None:
        return groups
    return [
        piece
        for group in groups
        for piece in _split_group_by_tokens(group, max_chunk_tokens)
    ]


def materialize_chunks(
    document: Document,
    groups: Iterable[list[Segment]],
    family: str,
    metadata: dict[str, object] | None = None,
) -> list[Chunk]:
    """Create chunks with original spans; non-contiguous groups retain all spans."""
    chunks: list[Chunk] = []
    for order, group in enumerate(groups):
        if not group:
            continue
        ordered = sorted(group, key=lambda item: item.order)
        spans = tuple((item.char_start, item.char_end) for item in ordered)
        contiguous = all(
            ordered[index].order + 1 == ordered[index + 1].order
            for index in range(len(ordered) - 1)
        )
        if contiguous:
            start, end = spans[0][0], spans[-1][1]
            text = document.text[start:end]
        else:
            text = "\n".join(item.text for item in ordered)
        chunk_metadata = dict(metadata or {})
        chunk_metadata["source_segment_orders"] = [item.order for item in ordered]
        chunk_metadata["source_segment_count"] = len(ordered)
        chunks.append(
            Chunk(
                chunk_id=f"{document.document_id}:{family}:{order}",
                document_id=document.document_id,
                text=text,
                token_count=len(token_spans(text)),
                chunk_order=order,
                is_contiguous=contiguous,
                source_segment_ids=tuple(item.segment_id for item in ordered),
                source_spans=spans,
                metadata=chunk_metadata,
            )
        )
    return chunks
