"""Artifact persistence for resumable benchmark runs."""

from dataclasses import asdict
from pathlib import Path
from typing import Any

from chunkbench.common.io import write_csv, write_json, write_jsonl
from chunkbench.common.types import Chunk, EvidenceCoverage, RetrievalHit


def save_method_artifacts(
    method_dir: Path,
    config: dict[str, Any],
    environment: dict[str, Any],
    chunks: list[Chunk],
    hits: list[RetrievalHit],
    coverage: list[EvidenceCoverage],
    metrics: dict[str, float],
    statistics: dict[str, Any],
    runtime_seconds: float,
    method_manifest: dict[str, Any] | None = None,
) -> None:
    """Write the stable per-method artifact contract."""
    method_dir.mkdir(parents=True, exist_ok=True)
    write_json(method_dir / "config.json", config)
    write_json(method_dir / "environment.json", environment)
    write_jsonl(method_dir / "chunks.jsonl", chunks)
    write_csv(method_dir / "retrieval.csv", [asdict(hit) for hit in hits])
    write_jsonl(method_dir / "evidence_coverage.jsonl", coverage)
    write_json(method_dir / "metrics.json", metrics)
    write_json(method_dir / "chunk_statistics.json", statistics)
    write_json(method_dir / "runtime.json", {"runtime_seconds": runtime_seconds})
    if method_manifest is not None:
        write_json(method_dir / "method_manifest.json", method_manifest)


def save_run_summary(
    run_dir: Path,
    metric_rows: list[dict[str, Any]],
    statistic_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    failed: list[dict[str, str]],
) -> None:
    """Write cross-method summary artifacts."""
    write_csv(run_dir / "benchmark_metrics.csv", metric_rows)
    write_csv(run_dir / "chunk_statistics.csv", statistic_rows)
    write_json(run_dir / "experiment_manifest.json", manifest)
    write_json(run_dir / "failed_methods.json", failed)
