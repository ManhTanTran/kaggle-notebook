"""YAML configuration loader with single- and multi-dataset compatibility."""

from pathlib import Path
from typing import Any

import yaml

from chunkbench.config.schema import EvaluationConfig, ExperimentConfig
from chunkbench.config.validation import validate_config


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load one YAML mapping."""
    config_path = Path(path)
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping in {config_path}")
    return value


def _load_dataset_spec(spec: Any, project_root: Path) -> dict[str, Any]:
    if isinstance(spec, dict) and "config" not in spec:
        return dict(spec)
    raw_path = spec["config"] if isinstance(spec, dict) else spec
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = project_root / path
    value = load_yaml(path)
    value["_config_path"] = str(path)
    return value


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate a backward-compatible experiment configuration."""
    config_path = Path(path)
    raw = load_yaml(config_path)
    project_root = Path.cwd()
    matrix_mode = "datasets" in raw
    if matrix_mode:
        dataset_configs = tuple(
            _load_dataset_spec(item, project_root) for item in raw["datasets"]
        )
    else:
        dataset_configs = (_load_dataset_spec(raw["dataset"], project_root),)
    evaluation_raw = raw.get("evaluation", {})
    evaluation = EvaluationConfig(
        **{
            **evaluation_raw,
            "k_values": tuple(evaluation_raw.get("k_values", (3, 5, 10))),
            "token_budgets": tuple(evaluation_raw.get("token_budgets", (2048,))),
        }
    )
    execution = raw.get("execution", {})
    config = ExperimentConfig(
        run_name=raw["run_name"],
        dataset=dataset_configs[0],
        datasets=dataset_configs,
        methods=tuple(raw["methods"]),
        embedding=raw.get("embedding", {"name": "hashing", "dimension": 256}),
        retrieval=raw.get("retrieval", {"name": "cosine"}),
        evaluation=evaluation,
        output_dir=Path(raw.get("output_dir", "outputs")),
        seed=int(raw.get("seed", 42)),
        smoke=bool(raw.get("smoke", False)),
        resume=bool(execution.get("resume", raw.get("resume", False))),
        skip_completed=bool(execution.get("skip_completed", False)),
        fail_fast=bool(execution.get("fail_fast", raw.get("fail_fast", False))),
        save_intermediate_artifacts=bool(
            execution.get("save_intermediate_artifacts", True)
        ),
        matrix=matrix_mode,
    )
    validate_config(config)
    return config
