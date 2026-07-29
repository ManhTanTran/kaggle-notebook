"""Dataset-by-method experiment matrix execution."""

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from chunkbench.common.environment import environment_fingerprint
from chunkbench.common.io import write_csv, write_json
from chunkbench.config.schema import ExperimentConfig
from chunkbench.data.validation import validate_dataset
from chunkbench.eval.constants import PRIMARY_METRICS
from chunkbench.pipeline.runner import BenchmarkRunner
from chunkbench.registry.datasets import build_dataset_adapter

REQUIRED_METHOD_ARTIFACTS = {
    "config.json",
    "environment.json",
    "chunks.jsonl",
    "retrieval.csv",
    "evidence_coverage.jsonl",
    "metrics.json",
    "chunk_statistics.json",
    "runtime.json",
    "method_manifest.json",
}


@dataclass(frozen=True)
class MatrixRunSpec:
    """One deterministic dataset-method combination."""

    dataset: dict[str, Any]
    method: dict[str, Any]
    config_hash: str
    run_id: str


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _canonical(item)
            for key, item in sorted(value.items())
            if not key.startswith("_")
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _hash_config(
    config: ExperimentConfig,
    dataset: dict[str, Any],
    method: dict[str, Any],
) -> str:
    payload = {
        "dataset": dataset,
        "method": method,
        "embedding": config.embedding,
        "retrieval": config.retrieval,
        "evaluation": asdict(config.evaluation),
        "seed": config.seed,
    }
    encoded = json.dumps(
        _canonical(payload), sort_keys=True, ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _valid_completed_run(method_dir: Path) -> bool:
    if not all((method_dir / name).is_file() for name in REQUIRED_METHOD_ARTIFACTS):
        return False
    try:
        metrics = _read_json(method_dir / "metrics.json")
    except (OSError, json.JSONDecodeError):
        return False
    return set(metrics) == set(PRIMARY_METRICS)


class ExperimentMatrixRunner:
    """Run and resume a deterministic dataset × method matrix."""

    def __init__(
        self, config: ExperimentConfig, project_root: Path | None = None
    ) -> None:
        self.config = config
        self.project_root = project_root or Path.cwd()

    def expand(self) -> list[MatrixRunSpec]:
        """Expand datasets and methods in configured order."""
        specs = []
        for dataset in self.config.datasets:
            dataset_name = str(dataset["name"])
            split = str(dataset.get("split", "unknown"))
            for method in self.config.methods:
                config_hash = _hash_config(self.config, dataset, method)
                method_name = str(method["name"])
                specs.append(
                    MatrixRunSpec(
                        dataset,
                        method,
                        config_hash,
                        f"{dataset_name}:{split}:{method_name}:{config_hash}",
                    )
                )
        return specs

    def run(self) -> dict[str, Any]:
        """Execute the matrix with failure isolation and aggregate artifacts."""
        run_dir = self.config.output_dir / self.config.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        environment = environment_fingerprint(self.project_root)
        specs = self.expand()
        completed: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        validation_rows: list[dict[str, Any]] = []
        result_rows: list[dict[str, Any]] = []
        statistic_names: set[str] = set()
        specs_by_dataset: dict[str, list[MatrixRunSpec]] = {}
        for spec in specs:
            specs_by_dataset.setdefault(str(spec.dataset["name"]), []).append(spec)

        for dataset_name, dataset_specs in specs_by_dataset.items():
            dataset = dataset_specs[0].dataset
            adapter_name = str(dataset.get("adapter", dataset_name))
            bundle = build_dataset_adapter(adapter_name, dataset).load()
            validation = validate_dataset(bundle, dataset.get("validation"))
            validation_rows.append(
                {
                    "Dataset": dataset_name,
                    "Split": bundle.metadata.get("split"),
                    "Language": bundle.metadata.get("language"),
                    "EvaluationProfile": bundle.metadata.get("evaluation_profile"),
                    **validation,
                }
            )
            pending = [
                spec
                for spec in dataset_specs
                if not (
                    (self.config.resume or self.config.skip_completed)
                    and _valid_completed_run(
                        run_dir / dataset_name / str(spec.method["name"])
                    )
                )
            ]
            failure_lookup: dict[str, dict[str, str]] = {}
            if pending:
                subconfig = replace(
                    self.config,
                    run_name=f"{self.config.run_name}/{dataset_name}",
                    dataset=dataset,
                    datasets=(dataset,),
                    methods=tuple(spec.method for spec in pending),
                    matrix=False,
                    resume=False,
                    skip_completed=False,
                )
                subresult = BenchmarkRunner(subconfig, self.project_root).run()
                failure_lookup = {
                    str(item["method"]): item for item in subresult["failed_methods"]
                }

            for spec in dataset_specs:
                method_name = str(spec.method["name"])
                method_dir = run_dir / dataset_name / method_name
                base = {
                    "Dataset": dataset_name,
                    "Method": method_name,
                    "Split": bundle.metadata.get("split"),
                    "Language": bundle.metadata.get("language"),
                    "EvaluationProfile": bundle.metadata.get("evaluation_profile"),
                    "EmbeddingModel": self.config.embedding.get(
                        "model_name", self.config.embedding.get("name")
                    ),
                    "Tokenizer": self.config.embedding.get("tokenizer", "whitespace"),
                    "GitCommit": environment.get("git_commit"),
                    "ConfigHash": spec.config_hash,
                }
                if _valid_completed_run(method_dir):
                    metrics = _read_json(method_dir / "metrics.json")
                    statistics = _read_json(method_dir / "chunk_statistics.json")
                    runtime = _read_json(method_dir / "runtime.json")
                    method_manifest = _read_json(method_dir / "method_manifest.json")
                    statistic_names.update(statistics)
                    row = {
                        **base,
                        "Status": "completed",
                        "RuntimeSeconds": runtime["runtime_seconds"],
                        "MethodFamily": method_manifest.get("family"),
                        "ImplementationFidelity": method_manifest.get(
                            "implementation_fidelity"
                        ),
                        "SourceCommit": method_manifest.get("source_commit"),
                        "ModelName": method_manifest.get("model_name"),
                        "ModelRevision": method_manifest.get("model_revision"),
                        "RepresentationStrategy": method_manifest.get(
                            "representation_strategy"
                        ),
                        "IsMockBackend": method_manifest.get("is_mock_backend"),
                        "IsPublishableBenchmark": method_manifest.get(
                            "is_publishable_benchmark"
                        ),
                        **metrics,
                        **statistics,
                    }
                    result_rows.append(row)
                    completed.append(
                        {
                            "status": "completed",
                            "run_id": spec.run_id,
                            "dataset": dataset_name,
                            "method": method_name,
                            "config_hash": spec.config_hash,
                            "git_commit": environment.get("git_commit"),
                        }
                    )
                    write_json(
                        method_dir / "config.json",
                        {
                            "dataset": dataset,
                            "method": spec.method,
                            "embedding": self.config.embedding,
                            "retrieval": self.config.retrieval,
                            "evaluation": asdict(self.config.evaluation),
                            "config_hash": spec.config_hash,
                        },
                    )
                else:
                    failure = failure_lookup.get(method_name, {})
                    record = {
                        "status": "failed",
                        "error_type": failure.get("error_type", "IncompleteRunError"),
                        "dataset": dataset_name,
                        "method": method_name,
                        "message": failure.get(
                            "error", "Required completed artifacts are missing"
                        ),
                        "run_id": spec.run_id,
                        "config_hash": spec.config_hash,
                        "git_commit": environment.get("git_commit"),
                    }
                    failed.append(record)
                    result_rows.append(
                        {**base, "Status": "failed", "RuntimeSeconds": None}
                    )
                    if self.config.fail_fast:
                        raise RuntimeError(record["message"])

        base_columns = [
            "Dataset",
            "Method",
            "Split",
            "Language",
            "EvaluationProfile",
            "EmbeddingModel",
            "Tokenizer",
            "GitCommit",
            "ConfigHash",
            "Status",
            "RuntimeSeconds",
            "MethodFamily",
            "ImplementationFidelity",
            "SourceCommit",
            "ModelName",
            "ModelRevision",
            "RepresentationStrategy",
            "IsMockBackend",
            "IsPublishableBenchmark",
        ]
        columns = base_columns + PRIMARY_METRICS + sorted(statistic_names)
        normalized_rows = [
            {column: row.get(column) for column in columns} for row in result_rows
        ]
        statistic_columns = base_columns + sorted(statistic_names)
        statistic_rows = [
            {column: row.get(column) for column in statistic_columns}
            for row in result_rows
            if row.get("Status") == "completed"
        ]
        write_csv(run_dir / "benchmark_metrics.csv", normalized_rows)
        write_csv(run_dir / "chunk_statistics.csv", statistic_rows)
        write_csv(run_dir / "dataset_validation.csv", validation_rows)
        write_json(run_dir / "completed_runs.json", completed)
        write_json(run_dir / "failed_runs.json", failed)
        write_json(
            run_dir / "experiment_manifest.json",
            {
                "run_name": self.config.run_name,
                "matrix_size": len(specs),
                "completed_count": len(completed),
                "failed_count": len(failed),
                "git_commit": environment.get("git_commit"),
                "datasets": [item["name"] for item in self.config.datasets],
                "methods": [item["name"] for item in self.config.methods],
                "evaluation_profile": "qa_evidence_retrieval",
            },
        )
        return {
            "run_dir": str(run_dir),
            "matrix_size": len(specs),
            "completed_runs": completed,
            "failed_runs": failed,
            "dataset_validation": validation_rows,
        }
