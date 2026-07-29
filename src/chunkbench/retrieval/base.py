"""Retriever interface."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from chunkbench.common.types import Chunk, Query, RetrievalHit


class BaseRetriever(ABC):
    """Index chunk vectors and retrieve ranked results."""

    @abstractmethod
    def index(self, chunks: list[Chunk], embeddings: NDArray[np.float32]) -> None:
        """Build the retrieval index."""

    @abstractmethod
    def search(
        self,
        queries: list[Query],
        query_embeddings: NDArray[np.float32],
        top_k: int,
    ) -> list[RetrievalHit]:
        """Return ranked hits for every query."""
