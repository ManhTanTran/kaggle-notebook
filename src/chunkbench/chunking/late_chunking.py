"""Fixed boundaries for late representations; no early chunk embedding occurs here."""

from chunkbench.chunking.base import BaseChunker
from chunkbench.chunking.fixed import FixedTokenChunker
from chunkbench.common.types import Chunk, Document


class LateChunker(BaseChunker):
    """Generate fixed source spans consumed by the late representation strategy."""

    def __init__(self, chunk_size: int, overlap: int = 0, **_: object) -> None:
        self.chunk_size = int(chunk_size)
        self.overlap = int(overlap)
        self._boundary_chunker = FixedTokenChunker(self.chunk_size, self.overlap)

    def chunk(self, document: Document) -> list[Chunk]:
        """Return boundaries only; the runner contextualises before pooling."""
        chunks = self._boundary_chunker.chunk(document)
        return [
            Chunk(
                chunk_id=f"{document.document_id}:late:{chunk.chunk_order}",
                document_id=chunk.document_id,
                text=chunk.text,
                token_count=chunk.token_count,
                chunk_order=chunk.chunk_order,
                is_contiguous=True,
                source_segment_ids=chunk.source_segment_ids,
                source_spans=chunk.source_spans,
                metadata={
                    **chunk.metadata,
                    "late_boundary": True,
                    "chunk_size": self.chunk_size,
                },
            )
            for chunk in chunks
        ]
