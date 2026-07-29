"""Configuration loading and validation."""

from chunkbench.config.loader import load_config
from chunkbench.config.schema import ExperimentConfig

__all__ = ["ExperimentConfig", "load_config"]
