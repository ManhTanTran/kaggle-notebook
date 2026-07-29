"""Fixed-size contiguous token chunking."""

from chunkbench.chunking.base import BaseChunker, token_spans
from chunkbench.common.types import Chunk, Document


class FixedTokenChunker(BaseChunker):
    """Split whitespace tokens into fixed-size, optionally overlapping windows."""

    def __init__(self, chunk_size: int, overlap: int = 0) -> None:
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document while preserving exact character source spans."""
        tokens = token_spans(document.text)
        chunks = []
        step = self.chunk_size - self.overlap
        for order, start in enumerate(range(0, len(tokens), step)):
            window = tokens[start : start + self.chunk_size]
            if not window:
                break
            char_start, char_end = window[0][1], window[-1][2]
            chunks.append(
                Chunk(
                    chunk_id=f"{document.document_id}:fixed:{order}",
                    document_id=document.document_id,
                    text=document.text[char_start:char_end],
                    token_count=len(window),
                    chunk_order=order,
                    is_contiguous=True,
                    source_segment_ids=(f"{document.document_id}:tokens:{start}",),
                    source_spans=((char_start, char_end),),
                    metadata={
                        "chunk_size": self.chunk_size,
                        "overlap": self.overlap,
                    },
                )
            )
            if start + self.chunk_size >= len(tokens):
                break
        return chunks
