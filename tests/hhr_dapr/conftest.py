from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from hhr_dapr.config import config_for_mode
from hhr_dapr.synthetic import make_synthetic_dataset


@pytest.fixture
def synthetic_dataset():
    return make_synthetic_dataset()


@pytest.fixture
def smoke_config(tmp_path: Path):
    return config_for_mode(
        "smoke",
        data_root=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "outputs",
    )


@pytest.fixture
def dapr_root(tmp_path: Path, synthetic_dataset) -> Path:
    root = tmp_path / "dapr"
    for dataset_name in (
        "ms_marco",
        "natural_questions",
        "miracl_en",
        "genomics",
        "conditional_qa",
        "nq_hard",
    ):
        directory = root / dataset_name
        directory.mkdir(parents=True)
        synthetic_dataset.documents.to_csv(directory / "documents.csv", index=False)
        synthetic_dataset.passages.to_csv(directory / "passages.csv", index=False)
        queries = synthetic_dataset.queries.copy()
        queries["dataset"] = dataset_name
        queries.to_csv(directory / "queries.csv", index=False)
        synthetic_dataset.qrels.to_csv(directory / "qrels.csv", index=False)
        manifest = {
            "documents": "documents.csv",
            "passages": "passages.csv",
            "queries": "queries.csv",
            "qrels": "qrels.csv",
        }
        if dataset_name == "nq_hard":
            pd.DataFrame(
                {
                    "query_id": ["q1", "q2", "q3", "q4"],
                    "question_type": ["CR,MT", "MT", "MHR", "AC"],
                }
            ).to_csv(directory / "query_metadata.csv", index=False)
            manifest["query_metadata"] = "query_metadata.csv"
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root
