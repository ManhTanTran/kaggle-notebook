"""Stable artifact schemas and run-directory export."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd

from .config import RunConfig
from .experiments import ExperimentResult

AGGREGATE_COLUMNS = [
    "dataset",
    "method",
    "status",
    "passage_ndcg@10",
    "passage_recall@100",
]
DOCUMENT_COLUMNS = [
    "dataset",
    "method",
    "status",
    "document_ndcg@10",
    "document_recall@5",
    "document_recall@20",
    "document_recall@100",
]
PASSAGE_COLUMNS = [
    "dataset",
    "method",
    "status",
    "passage_ndcg@10",
    "passage_recall@5",
    "passage_recall@20",
    "passage_recall@100",
    "candidate_survival_rate",
]
LATENCY_COLUMNS = [
    "dataset",
    "method",
    "status",
    "mean_query_latency_ms",
    "p50_query_latency_ms",
    "p95_query_latency_ms",
    "mean_unique_result_count",
]
NQ_CATEGORY_COLUMNS = [
    "dataset",
    "method",
    "category",
    "query_count",
    "passage_ndcg@10",
    "passage_recall@100",
    "status",
]


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def environment_payload() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": {
            name: _package_version(name)
            for name in (
                "numpy",
                "pandas",
                "pyarrow",
                "torch",
                "transformers",
                "faiss-cpu",
                "datasets",
                "huggingface-hub",
            )
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def git_commit_or_timestamp(workdir: Path) -> tuple[str, str | None]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(workdir), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return commit[:12], commit
    except (OSError, subprocess.CalledProcessError):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return timestamp, None


def create_run_directory(
    config: RunConfig, project_root: Path
) -> tuple[Path, str | None]:
    run_id, commit = git_commit_or_timestamp(project_root)
    output = Path(config.output_dir).expanduser()
    if not output.is_absolute():
        output = project_root / output
    run_dir = (output / config.experiment_name / config.run_mode / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, commit


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )


def _metric_rows(results: list[ExperimentResult], columns: list[str]) -> pd.DataFrame:
    rows = []
    for result in results:
        row = {
            "dataset": result.dataset,
            "method": result.experiment.name,
            "status": result.status,
        }
        row.update(result.aggregate_metrics)
        rows.append(row)
    return pd.DataFrame(rows).reindex(columns=columns)


def _append_not_run(
    frame: pd.DataFrame,
    columns: list[str],
    expected_pairs: Iterable[tuple[str, str]],
) -> pd.DataFrame:
    existing = set(zip(frame.get("dataset", []), frame.get("method", []), strict=False))
    rows = [
        {"dataset": dataset, "method": method, "status": "not_run"}
        for dataset, method in expected_pairs
        if (dataset, method) not in existing
    ]
    if rows:
        frame = pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)
    return (
        frame.reindex(columns=columns)
        .sort_values(["dataset", "method"])
        .reset_index(drop=True)
    )


def assert_artifact_schemas(artifacts: dict[str, pd.DataFrame]) -> None:
    expected = {
        "aggregate_metrics": AGGREGATE_COLUMNS,
        "document_metrics": DOCUMENT_COLUMNS,
        "passage_metrics": PASSAGE_COLUMNS,
        "latency": LATENCY_COLUMNS,
        "nq_hard_by_category": NQ_CATEGORY_COLUMNS,
    }
    for name, columns in expected.items():
        if list(artifacts[name].columns) != columns:
            raise AssertionError(
                f"{name} schema mismatch: {list(artifacts[name].columns)} != {columns}"
            )


def export_run_artifacts(
    run_dir: Path,
    config: RunConfig,
    results: list[ExperimentResult],
    dataset_audit: pd.DataFrame,
    expected_pairs: Iterable[tuple[str, str]],
    nq_category_results: pd.DataFrame | None = None,
    git_commit: str | None = None,
    synthetic_smoke: bool = False,
) -> dict[str, Path]:
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    aggregate = _append_not_run(
        _metric_rows(results, AGGREGATE_COLUMNS), AGGREGATE_COLUMNS, expected_pairs
    )
    document = _append_not_run(
        _metric_rows(results, DOCUMENT_COLUMNS), DOCUMENT_COLUMNS, expected_pairs
    )
    passage = _append_not_run(
        _metric_rows(results, PASSAGE_COLUMNS), PASSAGE_COLUMNS, expected_pairs
    )
    latency = _append_not_run(
        _metric_rows(results, LATENCY_COLUMNS), LATENCY_COLUMNS, expected_pairs
    )
    if nq_category_results is None:
        nq_category_results = pd.DataFrame(columns=NQ_CATEGORY_COLUMNS)
    nq_category_results = nq_category_results.reindex(columns=NQ_CATEGORY_COLUMNS)
    tables = {
        "aggregate_metrics": aggregate,
        "document_metrics": document,
        "passage_metrics": passage,
        "latency": latency,
        "nq_hard_by_category": nq_category_results,
    }
    assert_artifact_schemas(tables)

    per_query_parts = []
    for result in results:
        frame = result.per_query_metrics.copy()
        frame.insert(0, "method", result.experiment.name)
        frame.insert(0, "dataset", result.dataset)
        frame["status"] = result.status
        per_query_parts.append(frame)
    per_query = (
        pd.concat(per_query_parts, ignore_index=True)
        if per_query_parts
        else pd.DataFrame(columns=["dataset", "method", "query_id", "status"])
    )
    index_sizes = {
        f"{result.dataset}/{result.experiment.name}": result.index_metadata
        for result in results
    }
    manifest = [
        {
            "dataset": dataset,
            "method": method,
            "status": (
                "completed"
                if any(
                    result.dataset == dataset and result.experiment.name == method
                    for result in results
                )
                else "not_run"
            ),
        }
        for dataset, method in expected_pairs
    ]
    now = datetime.now(timezone.utc).isoformat()
    metadata = {
        "status": "completed",
        "created_at_utc": now,
        "git_commit": git_commit,
        "synthetic_smoke": synthetic_smoke,
        "result_count": len(results),
        "note": "Synthetic smoke metrics validate plumbing only."
        if synthetic_smoke
        else None,
    }
    paths = {
        "config": run_dir / "config.json",
        "environment": run_dir / "environment.json",
        "dataset_audit": run_dir / "dataset_audit.csv",
        "aggregate_metrics": run_dir / "aggregate_metrics.csv",
        "document_metrics": run_dir / "document_metrics.csv",
        "passage_metrics": run_dir / "passage_metrics.csv",
        "per_query_metrics": run_dir / "per_query_metrics.parquet",
        "latency": run_dir / "latency.csv",
        "index_sizes": run_dir / "index_sizes.json",
        "nq_hard_by_category": run_dir / "nq_hard_by_category.csv",
        "experiment_manifest": run_dir / "experiment_manifest.json",
        "run_metadata": run_dir / "run_metadata.json",
    }
    _write_json(paths["config"], config.to_dict())
    _write_json(paths["environment"], environment_payload())
    dataset_audit.to_csv(paths["dataset_audit"], index=False)
    aggregate.to_csv(paths["aggregate_metrics"], index=False)
    document.to_csv(paths["document_metrics"], index=False)
    passage.to_csv(paths["passage_metrics"], index=False)
    per_query.to_parquet(paths["per_query_metrics"], index=False)
    latency.to_csv(paths["latency"], index=False)
    _write_json(paths["index_sizes"], index_sizes)
    nq_category_results.to_csv(paths["nq_hard_by_category"], index=False)
    _write_json(paths["experiment_manifest"], manifest)
    _write_json(paths["run_metadata"], metadata)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise AssertionError(f"Artifact export incomplete: {missing}")
    return paths
