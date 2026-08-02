"""Manifest-driven adapters for the six DAPR datasets."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

from .schema import REQUIRED_COLUMNS, NormalizedDAPRDataset, validate_dataset


class DAPRDatasetAdapter(ABC):
    dataset_name: str

    def __init__(self, data_root: str | Path):
        self.data_root = Path(data_root).expanduser().resolve()

    @abstractmethod
    def load(self) -> NormalizedDAPRDataset:
        """Load and normalize one dataset."""


def _read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required DAPR artifact is missing: {path}. "
            "Populate the dataset directory or correct its manifest.json."
        )
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".json"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    raise ValueError(f"Unsupported DAPR table format {suffix!r} for {path}")


class ManifestDAPRAdapter(DAPRDatasetAdapter):
    """Normalize local files described by ``<root>/<dataset>/manifest.json``.

    The manifest has ``documents``, ``passages``, ``queries`` and ``qrels`` paths,
    an optional ``query_metadata`` path, and optional ``column_maps``. Each column
    map maps canonical output names to source names.
    """

    directory_name: str

    @property
    def dataset_dir(self) -> Path:
        return self.data_root / self.directory_name

    def load(self) -> NormalizedDAPRDataset:
        manifest_path = self.dataset_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing {manifest_path}. Create it using the contract documented "
                "in README.md."
            )
        manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        required_tables = ("documents", "passages", "queries", "qrels")
        missing = [name for name in required_tables if name not in manifest]
        if missing:
            raise ValueError(f"{manifest_path} is missing table entries: {missing}")
        maps = manifest.get("column_maps", {})
        frames: dict[str, pd.DataFrame] = {}
        for table in required_tables:
            frame = _read_table(self.dataset_dir / manifest[table])
            canonical_to_source = maps.get(table, {})
            rename = {
                source: canonical for canonical, source in canonical_to_source.items()
            }
            frame = frame.rename(columns=rename)
            missing_columns = sorted(set(REQUIRED_COLUMNS[table]) - set(frame.columns))
            if missing_columns:
                raise ValueError(
                    f"{self.dataset_name}.{table} missing normalized columns "
                    f"{missing_columns}"
                )
            frames[table] = frame.loc[:, list(REQUIRED_COLUMNS[table])].copy()

        metadata = None
        if manifest.get("query_metadata"):
            metadata = _read_table(self.dataset_dir / manifest["query_metadata"])
            metadata_map = maps.get("query_metadata", {})
            metadata = metadata.rename(
                columns={
                    source: canonical for canonical, source in metadata_map.items()
                }
            )
            metadata = metadata.loc[:, ["query_id", "question_type"]].copy()

        for frame, columns in (
            (frames["documents"], ("document_id",)),
            (frames["passages"], ("passage_id", "document_id")),
            (frames["queries"], ("query_id", "dataset", "split")),
            (frames["qrels"], ("query_id", "passage_id")),
        ):
            for column in columns:
                frame[column] = frame[column].astype(str)
        frames["qrels"]["relevance"] = pd.to_numeric(frames["qrels"]["relevance"])
        frames["passages"]["passage_position"] = pd.to_numeric(
            frames["passages"]["passage_position"], errors="coerce"
        ).astype("Int64")
        if metadata is not None:
            metadata["query_id"] = metadata["query_id"].astype(str)

        result = NormalizedDAPRDataset(
            self.dataset_name, query_metadata=metadata, **frames
        )
        validate_dataset(result)
        return result


class MSMarcoAdapter(ManifestDAPRAdapter):
    dataset_name = "ms_marco"
    directory_name = "ms_marco"


class NaturalQuestionsAdapter(ManifestDAPRAdapter):
    dataset_name = "natural_questions"
    directory_name = "natural_questions"


class MiraclEnglishAdapter(ManifestDAPRAdapter):
    dataset_name = "miracl_en"
    directory_name = "miracl_en"


class GenomicsAdapter(ManifestDAPRAdapter):
    dataset_name = "genomics"
    directory_name = "genomics"


class ConditionalQAAdapter(ManifestDAPRAdapter):
    dataset_name = "conditional_qa"
    directory_name = "conditional_qa"


class NQHardAdapter(ManifestDAPRAdapter):
    dataset_name = "nq_hard"
    directory_name = "nq_hard"


ADAPTER_REGISTRY = {
    cls.dataset_name: cls
    for cls in (
        MSMarcoAdapter,
        NaturalQuestionsAdapter,
        MiraclEnglishAdapter,
        GenomicsAdapter,
        ConditionalQAAdapter,
        NQHardAdapter,
    )
}


def load_dataset(name: str, data_root: str | Path) -> NormalizedDAPRDataset:
    if name not in ADAPTER_REGISTRY:
        raise KeyError(
            f"Unknown DAPR dataset {name!r}; expected one of {sorted(ADAPTER_REGISTRY)}"
        )
    return ADAPTER_REGISTRY[name](data_root).load()
