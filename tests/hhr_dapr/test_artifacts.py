from __future__ import annotations

import json

import pandas as pd

from hhr_dapr.artifacts import export_run_artifacts
from hhr_dapr.config import ALL_METHODS
from hhr_dapr.experiments import build_experiment_registry, run_hhr_experiment


def test_artifact_schema_and_not_run_status(tmp_path, synthetic_dataset, smoke_config):
    experiment = build_experiment_registry(smoke_config)["sparse+dense"]
    result = run_hhr_experiment(synthetic_dataset, experiment, smoke_config)
    run_dir = tmp_path / "run"
    expected = [("synthetic", method) for method in ALL_METHODS]
    paths = export_run_artifacts(
        run_dir,
        smoke_config,
        [result],
        pd.DataFrame(
            [
                {
                    "dataset": synthetic_dataset.name,
                    "documents": len(synthetic_dataset.documents),
                    "passages": len(synthetic_dataset.passages),
                    "queries": len(synthetic_dataset.queries),
                    "qrels": len(synthetic_dataset.qrels),
                }
            ]
        ),
        expected,
        synthetic_smoke=True,
    )
    assert set(paths) == {
        "config",
        "environment",
        "dataset_audit",
        "aggregate_metrics",
        "document_metrics",
        "passage_metrics",
        "per_query_metrics",
        "latency",
        "index_sizes",
        "nq_hard_by_category",
        "experiment_manifest",
        "run_metadata",
    }
    aggregate = pd.read_csv(paths["aggregate_metrics"])
    assert list(aggregate.columns[:3]) == ["dataset", "method", "status"]
    assert (aggregate["status"] == "not_run").sum() == 8
    metadata = json.loads(paths["run_metadata"].read_text(encoding="utf-8"))
    assert metadata["synthetic_smoke"] is True
