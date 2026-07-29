"""Independent and late chunk representation strategies."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from chunkbench.common.types import Chunk, Document
from chunkbench.embedding.base import BaseEmbedder
from chunkbench.embedding.contextual import (
    BaseContextualEmbedder,
    HashingContextualEmbedder,
    TransformersContextualEmbedder,
)


class ChunkRepresentationStrategy(ABC):
    """Produce index vectors after boundaries have been generated."""

    name = "independent_chunk_embedding"

    @abstractmethod
    def represent(
        self, documents: list[Document], chunks: list[Chunk], embedder: BaseEmbedder
    ) -> NDArray[np.float32]:
        """Return vectors in exactly the supplied chunk order."""


class IndependentChunkRepresentationStrategy(ChunkRepresentationStrategy):
    """Traditional encode-each-chunk-independently representation."""

    name = "independent_chunk_embedding"

    def represent(
        self, documents: list[Document], chunks: list[Chunk], embedder: BaseEmbedder
    ) -> NDArray[np.float32]:
        return embedder.encode_documents([chunk.text for chunk in chunks])


class LateChunkingRepresentationStrategy(ChunkRepresentationStrategy):
    """Encode each full document before mean-pooling states into chunk spans."""

    name = "late_document_embedding"

    def __init__(
        self, contextual_embedder: BaseContextualEmbedder, normalize: bool = True
    ) -> None:
        self.contextual_embedder = contextual_embedder
        self.normalize = normalize

    def represent(
        self, documents: list[Document], chunks: list[Chunk], embedder: BaseEmbedder
    ) -> NDArray[np.float32]:
        vectors: dict[str, np.ndarray] = {}
        chunk_by_document: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            chunk_by_document.setdefault(chunk.document_id, []).append(chunk)
        for document in documents:
            states = self.contextual_embedder.encode_document(document)
            for chunk in chunk_by_document.get(document.document_id, []):
                indices = [
                    index
                    for index, ((start, end), attended, special) in enumerate(
                        zip(
                            states.offsets,
                            states.attention_mask,
                            states.special_tokens,
                            strict=True,
                        )
                    )
                    if attended
                    and not special
                    and any(
                        start >= span_start and end <= span_end
                        for span_start, span_end in chunk.source_spans
                    )
                ]
                if not indices:
                    raise ValueError(
                        f"No contextual tokens map to chunk {chunk.chunk_id}"
                    )
                vector = states.vectors[indices].mean(axis=0).astype(np.float32)
                if self.normalize:
                    norm = float(np.linalg.norm(vector))
                    if norm:
                        vector /= norm
                vectors[chunk.chunk_id] = vector
        result = np.vstack([vectors[chunk.chunk_id] for chunk in chunks]).astype(
            np.float32
        )
        expected_dimension = getattr(embedder, "dimension", result.shape[1])
        if result.shape[1] != expected_dimension:
            raise ValueError(
                "Late contextual vector dimension must match retrieval query embedder"
            )
        return result


def build_representation_strategy(
    config: dict[str, Any], embedding_dimension: int = 256
) -> ChunkRepresentationStrategy:
    """Build strategy from method configuration; mock is explicitly opt-in metadata."""
    kind = str(config.get("representation_strategy", "independent_chunk_embedding"))
    if kind == "independent_chunk_embedding":
        return IndependentChunkRepresentationStrategy()
    if kind != "late_document_embedding":
        raise ValueError(f"Unknown representation strategy: {kind}")
    representation = dict(config.get("representation", {}))
    backend_type = str(
        representation.get("backend_type", config.get("backend_type", "mock"))
    )
    if backend_type == "mock":
        backend: BaseContextualEmbedder = HashingContextualEmbedder(
            dimension=int(representation.get("dimension", embedding_dimension)),
            max_model_tokens=int(representation.get("max_model_tokens", 8192)),
            long_document_policy=str(
                representation.get("long_document_policy", "error")
            ),
        )
    elif backend_type == "transformers":
        backend = TransformersContextualEmbedder(
            model_name=str(representation["model_name"]),
            model_revision=representation.get("model_revision"),
            device=representation.get("device"),
            max_model_tokens=int(representation.get("max_model_tokens", 8192)),
            long_document_policy=str(
                representation.get("long_document_policy", "error")
            ),
            window_stride=int(representation.get("window_stride", 64)),
        )
    else:
        raise ValueError(f"Unknown late backend_type: {backend_type}")
    return LateChunkingRepresentationStrategy(
        backend, bool(representation.get("normalize", True))
    )
