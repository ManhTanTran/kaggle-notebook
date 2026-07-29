"""Optional FAISS cosine retriever."""

import numpy as np
from numpy.typing import NDArray

from chunkbench.common.exceptions import OptionalDependencyError
from chunkbench.common.types import Chunk, Query, RetrievalHit
from chunkbench.retrieval.base import BaseRetriever


class FaissRetriever(BaseRetriever):
    """Exact FAISS inner-product search over L2-normalized vectors."""

    def __init__(self) -> None:
        try:
            import faiss
        except ImportError as error:
            raise OptionalDependencyError(
                "FAISS is required; install chunkbench[retrieval]"
            ) from error
        self.faiss = faiss
        self.chunks: list[Chunk] = []
        self._index = None

    def index(self, chunks: list[Chunk], embeddings: NDArray[np.float32]) -> None:
        values = np.asarray(embeddings, dtype=np.float32).copy()
        self.faiss.normalize_L2(values)
        self._index = self.faiss.IndexFlatIP(values.shape[1])
        self._index.add(values)
        self.chunks = chunks

    def search(
        self,
        queries: list[Query],
        query_embeddings: NDArray[np.float32],
        top_k: int,
    ) -> list[RetrievalHit]:
        if self._index is None:
            raise RuntimeError("Call index before search")
        values = np.asarray(query_embeddings, dtype=np.float32).copy()
        self.faiss.normalize_L2(values)
        scores, indices = self._index.search(values, min(top_k, len(self.chunks)))
        return [
            RetrievalHit(
                query_id=query.query_id,
                chunk_id=self.chunks[int(index)].chunk_id,
                document_id=self.chunks[int(index)].document_id,
                rank=rank,
                score=float(scores[query_index, rank - 1]),
            )
            for query_index, query in enumerate(queries)
            for rank, index in enumerate(indices[query_index], start=1)
            if index >= 0
        ]
