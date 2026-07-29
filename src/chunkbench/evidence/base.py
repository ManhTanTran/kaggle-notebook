"""Evidence mapper interface."""

from abc import ABC, abstractmethod

from chunkbench.common.types import Chunk, Evidence, EvidenceCoverage


class EvidenceMapper(ABC):
    """Map canonical evidence tokens to retrieved chunks."""

    @abstractmethod
    def map(
        self, chunks: list[Chunk], evidence: list[Evidence]
    ) -> list[EvidenceCoverage]:
        """Return coverage rows."""
