"""Sentence-aware fixed token-budget chunking."""

import re

from chunkbench.chunking.base import BaseChunker, token_spans
from chunkbench.common.types import Chunk, Document

SENTENCE_RE = re.compile(r"\S.*?(?:[.!?](?=\s|$)|$)", re.DOTALL)


class SentenceFixedChunker(BaseChunker):
    """Pack sentences up to a token target, splitting only oversized sentences."""

    def __init__(self, chunk_size: int) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = chunk_size

    def chunk(self, document: Document) -> list[Chunk]:
        """Create contiguous chunks aligned to sentence boundaries when possible."""
        segments: list[tuple[int, int]] = []
        for sentence in SENTENCE_RE.finditer(document.text):
            spans = token_spans(sentence.group())
            if not spans:
                continue
            absolute = [
                (token, start + sentence.start(), end + sentence.start())
                for token, start, end in spans
            ]
            for index in range(0, len(absolute), self.chunk_size):
                window = absolute[index : index + self.chunk_size]
                segments.append((window[0][1], window[-1][2]))

        packed: list[tuple[int, int, int]] = []
        for start, end in segments:
            count = len(token_spans(document.text[start:end]))
            if packed and packed[-1][2] + count <= self.chunk_size:
                previous_start, _, previous_count = packed[-1]
                packed[-1] = (previous_start, end, previous_count + count)
            else:
                packed.append((start, end, count))

        return [
            Chunk(
                chunk_id=f"{document.document_id}:sentence:{order}",
                document_id=document.document_id,
                text=document.text[start:end],
                token_count=count,
                chunk_order=order,
                is_contiguous=True,
                source_segment_ids=(f"{document.document_id}:sentence:{order}",),
                source_spans=((start, end),),
                metadata={"chunk_size": self.chunk_size, "sentence_aware": True},
            )
            for order, (start, end, count) in enumerate(packed)
        ]
