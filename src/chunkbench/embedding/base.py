"""Embedding interfaces and dependency-free smoke embedder."""

import hashlib
import re
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class BaseEmbedder(ABC):
    """Encode documents and queries into a shared vector space."""

    @abstractmethod
    def encode_documents(self, texts: list[str]) -> NDArray[np.float32]:
        """Encode document or chunk text."""

    @abstractmethod
    def encode_queries(self, texts: list[str]) -> NDArray[np.float32]:
        """Encode query text."""


class HashingEmbedder(BaseEmbedder):
    """Deterministic lexical hashing embedder for offline smoke tests."""

    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension

    def _encode(self, texts: list[str]) -> NDArray[np.float32]:
        result = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"\w+", text.lower()):
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                column = int.from_bytes(digest, "little") % self.dimension
                result[row, column] += 1.0
        return result

    def encode_documents(self, texts: list[str]) -> NDArray[np.float32]:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> NDArray[np.float32]:
        return self._encode(texts)
