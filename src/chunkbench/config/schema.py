"""Typed configuration schema."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvaluationConfig:
    k_values: tuple[int, ...] = (3, 5, 10)
    token_budgets: tuple[int, ...] = (2048,)
    retrieval_depth: int = 100
    strict_token_budget: bool = True
    duplicate_scope: str = "within_document"


@dataclass(frozen=True)
class ExperimentConfig:
    run_name: str
    dataset: dict[str, Any]
    methods: tuple[dict[str, Any], ...]
    embedding: dict[str, Any]
    retrieval: dict[str, Any]
    datasets: tuple[dict[str, Any], ...] = ()
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output_dir: Path = Path("outputs")
    seed: int = 42
    smoke: bool = False
    resume: bool = False
    skip_completed: bool = False
    fail_fast: bool = False
    save_intermediate_artifacts: bool = True
    matrix: bool = False
