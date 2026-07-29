"""Extension registries."""

from chunkbench.registry.datasets import (
    CORE_DATASETS,
    EVALUATION_PROFILE,
    build_dataset_adapter,
    list_registered_datasets,
    register_dataset,
)
from chunkbench.registry.methods import CORE_METHODS, build_chunker

__all__ = [
    "CORE_DATASETS",
    "CORE_METHODS",
    "EVALUATION_PROFILE",
    "build_chunker",
    "build_dataset_adapter",
    "list_registered_datasets",
    "register_dataset",
]
