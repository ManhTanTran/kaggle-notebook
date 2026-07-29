"""Retriever registry."""

from collections.abc import Callable
from typing import Any

from chunkbench.retrieval.base import BaseRetriever
from chunkbench.retrieval.cosine import CosineRetriever
from chunkbench.retrieval.faiss_retriever import FaissRetriever

RetrieverFactory = Callable[[dict[str, Any]], BaseRetriever]

RETRIEVER_REGISTRY: dict[str, RetrieverFactory] = {
    "cosine": lambda config: CosineRetriever(),
    "faiss_cosine": lambda config: FaissRetriever(),
}


def build_retriever(name: str, config: dict[str, Any] | None = None) -> BaseRetriever:
    """Build a registered retriever."""
    try:
        return RETRIEVER_REGISTRY[name](config or {})
    except KeyError as error:
        raise KeyError(f"Unknown retriever {name!r}") from error
