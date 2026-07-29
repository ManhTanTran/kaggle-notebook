"""Algorithm-level invariants for advanced chunkers."""

from chunkbench.chunking.segments import Segment
from chunkbench.common.exceptions import ContractError
from chunkbench.common.types import Chunk, Document


def validate_advanced_chunks(
    document: Document,
    segments: list[Segment],
    chunks: list[Chunk],
    max_tokens: int | None,
) -> None:
    """Validate provenance, order, coverage and configured caps where representable."""
    if any(chunk.document_id != document.document_id for chunk in chunks):
        raise ContractError("Chunk document_id does not match source document")
    if [chunk.chunk_order for chunk in chunks] != list(range(len(chunks))):
        raise ContractError("Chunk order must be contiguous")
    expected = {segment.segment_id for segment in segments}
    seen = [segment_id for chunk in chunks for segment_id in chunk.source_segment_ids]
    if set(seen) != expected:
        raise ContractError("Advanced chunker lost a source segment")
    if len(seen) != len(set(seen)):
        raise ContractError("Advanced chunker duplicated a source segment")
    if max_tokens is not None:
        oversized = [
            segment for segment in segments if segment.token_count > max_tokens
        ]
        if not oversized and any(chunk.token_count > max_tokens for chunk in chunks):
            raise ContractError("Chunk violates configured maximum token cap")
