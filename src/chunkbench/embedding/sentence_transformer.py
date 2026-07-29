"""Optional sentence-transformers backend."""

import numpy as np
from numpy.typing import NDArray

from chunkbench.common.exceptions import OptionalDependencyError
from chunkbench.embedding.base import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    """Encode text using an explicitly configured sentence-transformers model."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise OptionalDependencyError(
                "sentence-transformers is required; install chunkbench[embedding]"
            ) from error
        self.model = SentenceTransformer(model_name)

    def encode_documents(self, texts: list[str]) -> NDArray[np.float32]:
        return np.asarray(self.model.encode(texts), dtype=np.float32)

    def encode_queries(self, texts: list[str]) -> NDArray[np.float32]:
        return np.asarray(self.model.encode(texts), dtype=np.float32)
