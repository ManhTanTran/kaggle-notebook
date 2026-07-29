import numpy as np

from chunkbench.chunking.late_chunking import LateChunker
from chunkbench.common.types import Document
from chunkbench.embedding.base import HashingEmbedder
from chunkbench.embedding.contextual import HashingContextualEmbedder
from chunkbench.pipeline.representations import (
    IndependentChunkRepresentationStrategy,
    LateChunkingRepresentationStrategy,
)


def test_late_pools_after_document_context_and_normalizes():
    document = Document("doc", "Berlin is large. The city grows.")
    chunks = LateChunker(2).chunk(document)
    embedder = HashingEmbedder(32)
    late = LateChunkingRepresentationStrategy(HashingContextualEmbedder(32))
    late_vectors = late.represent([document], chunks, embedder)
    early_vectors = IndependentChunkRepresentationStrategy().represent(
        [document], chunks, embedder
    )
    assert late_vectors.shape == early_vectors.shape == (3, 32)
    assert np.allclose(np.linalg.norm(late_vectors, axis=1), 1.0)
    assert not np.allclose(late_vectors, early_vectors)
