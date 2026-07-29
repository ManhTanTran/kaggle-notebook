"""Embedder registry."""

from collections.abc import Callable
from typing import Any

from chunkbench.embedding.base import BaseEmbedder, HashingEmbedder
from chunkbench.embedding.sentence_transformer import SentenceTransformerEmbedder

EmbedderFactory = Callable[[dict[str, Any]], BaseEmbedder]

EMBEDDER_REGISTRY: dict[str, EmbedderFactory] = {
    "hashing": lambda config: HashingEmbedder(int(config.get("dimension", 256))),
    "sentence_transformer": lambda config: SentenceTransformerEmbedder(
        str(config["model_name"])
    ),
}


def build_embedder(name: str, config: dict[str, Any] | None = None) -> BaseEmbedder:
    """Build a registered embedder."""
    try:
        return EMBEDDER_REGISTRY[name](config or {})
    except KeyError as error:
        raise KeyError(f"Unknown embedder {name!r}") from error
