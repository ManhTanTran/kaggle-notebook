"""NumPy cosine retriever."""

import numpy as np
from numpy.typing import NDArray

from chunkbench.common.types import Chunk, Query, RetrievalHit
from chunkbench.retrieval.base import BaseRetriever


def _normalize(values: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


class CosineRetriever(BaseRetriever):
    """Exact cosine similarity retrieval using NumPy."""

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.embeddings = np.empty((0, 0), dtype=np.float32)

    def index(self, chunks: list[Chunk], embeddings: NDArray[np.float32]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have equal length")
        self.chunks = chunks
        self.embeddings = _normalize(embeddings)

    def search(
        self,
        queries: list[Query],
        query_embeddings: NDArray[np.float32],
        top_k: int,
    ) -> list[RetrievalHit]:
        scores = _normalize(query_embeddings) @ self.embeddings.T
        hits = []
        for query_index, query in enumerate(queries):
            order = np.argsort(-scores[query_index], kind="stable")[:top_k]
            hits.extend(
                RetrievalHit(
                    query_id=query.query_id,
                    chunk_id=self.chunks[index].chunk_id,
                    document_id=self.chunks[index].document_id,
                    rank=rank,
                    score=float(scores[query_index, index]),
                )
                for rank, index in enumerate(order, start=1)
            )
        return hits
