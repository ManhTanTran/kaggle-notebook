"""Deterministic sentence segments shared by advanced chunkers."""

import re
from dataclasses import dataclass, field
from typing import Any

from chunkbench.chunking.base import token_spans
from chunkbench.common.types import Document

SENTENCE_RE = re.compile(r"\S.*?(?:[.!?](?=\s|$)|$)", re.DOTALL)


@dataclass(frozen=True)
class Segment:
    """One ordered source unit with exact character provenance."""

    segment_id: str
    document_id: str
    text: str
    token_count: int
    order: int
    char_start: int
    char_end: int
    metadata: dict[str, Any] = field(default_factory=dict)


def sentence_segments(document: Document) -> list[Segment]:
    """Split a document into non-empty sentence-like spans without NLP downloads."""
    result: list[Segment] = []
    for match in SENTENCE_RE.finditer(document.text):
        text = match.group()
        count = len(token_spans(text))
        if count == 0:
            continue
        order = len(result)
        result.append(
            Segment(
                segment_id=f"{document.document_id}:sentence:{order}",
                document_id=document.document_id,
                text=text,
                token_count=count,
                order=order,
                char_start=match.start(),
                char_end=match.end(),
            )
        )
    return result
