"""Chunker interface and shared token span logic."""

import re
from abc import ABC, abstractmethod

from chunkbench.common.types import Chunk, Document

TOKEN_RE = re.compile(r"\S+")


def token_spans(text: str) -> list[tuple[str, int, int]]:
    """Return whitespace-token text with character offsets."""
    return [
        (match.group(), match.start(), match.end()) for match in TOKEN_RE.finditer(text)
    ]


class BaseChunker(ABC):
    """Create validated chunks for one document."""

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into chunks."""
