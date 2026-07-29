"""Small pipeline stage helpers."""

from chunkbench.chunking.base import BaseChunker
from chunkbench.common.types import Chunk, Document


def chunk_documents(documents: list[Document], chunker: BaseChunker) -> list[Chunk]:
    """Chunk every document with deterministic document order."""
    return [chunk for document in documents for chunk in chunker.chunk(document)]
