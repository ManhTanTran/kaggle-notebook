"""Dataset adapter interface."""

from abc import ABC, abstractmethod
from typing import Any

from chunkbench.common.types import DatasetBundle


class DatasetAdapter(ABC):
    """Convert an external dataset into the canonical bundle."""

    def __init__(self, **config: Any) -> None:
        self.config = config

    @abstractmethod
    def load(self) -> DatasetBundle:
        """Load and normalize the configured dataset."""
