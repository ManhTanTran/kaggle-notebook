"""Reusable document chunking benchmark package."""

from chunkbench.common.types import Chunk, DatasetBundle, Document, Evidence, Query
from chunkbench.eval.constants import PRIMARY_METRICS
from chunkbench.pipeline.matrix import ExperimentMatrixRunner
from chunkbench.pipeline.runner import BenchmarkRunner
from chunkbench.registry.datasets import CORE_DATASETS, EVALUATION_PROFILE
from chunkbench.registry.methods import CORE_METHODS

__all__ = [
    "BenchmarkRunner",
    "Chunk",
    "DatasetBundle",
    "Document",
    "Evidence",
    "Query",
    "CORE_DATASETS",
    "CORE_METHODS",
    "EVALUATION_PROFILE",
    "ExperimentMatrixRunner",
    "PRIMARY_METRICS",
]

__version__ = "0.1.0"
