"""Dataset adapter registry for the QA evidence-retrieval profile."""

from collections.abc import Callable
from typing import Any

from chunkbench.data.base import DatasetAdapter
from chunkbench.data.hotpotqa_fullwiki import HotpotQAFullWikiAdapter
from chunkbench.data.qasper import QasperAdapter
from chunkbench.data.synthetic import SyntheticDatasetAdapter
from chunkbench.data.uit_viquad import UITViQuADAdapter
from chunkbench.data.vimqa import ViMQAAdapter

EVALUATION_PROFILE = "qa_evidence_retrieval"
CORE_DATASETS = [
    "qasper",
    "hotpotqa_fullwiki",
    "uit_viquad",
    "vimqa",
]

DatasetFactory = Callable[[dict[str, Any]], DatasetAdapter]

DATASET_REGISTRY: dict[str, DatasetFactory] = {
    "synthetic": lambda config: SyntheticDatasetAdapter(**config),
    "qasper": lambda config: QasperAdapter(**config),
    "hotpotqa_fullwiki": lambda config: HotpotQAFullWikiAdapter(**config),
    "uit_viquad": lambda config: UITViQuADAdapter(**config),
    "vimqa": lambda config: ViMQAAdapter(**config),
}


def register_dataset(name: str, factory: DatasetFactory) -> None:
    """Register a dataset adapter factory."""
    if name in DATASET_REGISTRY:
        raise ValueError(f"Dataset already registered: {name}")
    DATASET_REGISTRY[name] = factory


def build_dataset_adapter(
    name: str, config: dict[str, Any] | None = None
) -> DatasetAdapter:
    """Build a registered adapter or raise a descriptive error."""
    try:
        factory = DATASET_REGISTRY[name]
    except KeyError as error:
        available = ", ".join(list_registered_datasets())
        raise KeyError(
            f"Unknown dataset {name!r}; registered datasets: {available}"
        ) from error
    return factory(config or {})


def list_registered_datasets() -> list[str]:
    """Return registered names in deterministic insertion order."""
    return list(DATASET_REGISTRY)


def build_dataset(name: str, config: dict[str, Any] | None = None) -> DatasetAdapter:
    """Backward-compatible alias for build_dataset_adapter."""
    return build_dataset_adapter(name, config)
