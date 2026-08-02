"""Central run-mode defaults and configuration validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ALL_DATASETS = (
    "ms_marco",
    "natural_questions",
    "miracl_en",
    "genomics",
    "conditional_qa",
    "nq_hard",
)

ALL_METHODS = (
    "sparse+sparse",
    "sparse+dense",
    "sparse+combined",
    "dense+sparse",
    "dense+dense",
    "dense+combined",
    "combined+sparse",
    "combined+dense",
    "combined+combined",
)

RECOMMENDED_METHODS = (
    "sparse+dense",
    "dense+dense",
    "combined+dense",
    "combined+combined",
)

RUN_MODES: dict[str, dict[str, Any]] = {
    "smoke": {
        "datasets": ("synthetic",),
        "methods": RECOMMENDED_METHODS,
        "query_sample_size": 4,
        "document_top_k": 3,
        "passage_top_k": 5,
        "final_top_k": 5,
        "dense_backend": "hashing",
        "dense_index": "exact",
        "run_nq_hard_analysis": False,
    },
    "baseline": {
        "datasets": ALL_DATASETS,
        "methods": RECOMMENDED_METHODS,
        "query_sample_size": None,
        "document_top_k": 20,
        "passage_top_k": 100,
        "final_top_k": 100,
        "dense_backend": "transformers",
        "dense_index": "faiss",
        "run_nq_hard_analysis": True,
    },
    "full": {
        "datasets": ALL_DATASETS,
        "methods": ALL_METHODS,
        "query_sample_size": None,
        "document_top_k": 100,
        "passage_top_k": 1000,
        "final_top_k": 100,
        "dense_backend": "transformers",
        "dense_index": "faiss",
        "run_nq_hard_analysis": True,
    },
}


@dataclass(frozen=True)
class RunConfig:
    experiment_name: str = "phase1_hhr_dapr"
    run_mode: str = "smoke"
    data_root: Path = Path("data/dapr")
    cache_dir: Path = Path("cache/hhr_dapr")
    output_dir: Path = Path("outputs")
    random_seed: int = 42
    datasets: tuple[str, ...] = ("synthetic",)
    methods: tuple[str, ...] = RECOMMENDED_METHODS
    query_sample_size: int | None = 4
    document_top_k: int = 3
    passage_top_k: int = 5
    final_top_k: int = 5
    dense_query_model: str = "facebook/dragon-plus-query-encoder"
    dense_context_model: str = "facebook/dragon-plus-context-encoder"
    dense_document_strategy: str = "title_plus_lead"
    dense_document_lead_chars: int = 4000
    dense_batch_size: int = 32
    dense_device: str = "cpu"
    dense_backend: str = "hashing"
    dense_index: str = "exact"
    hashing_features: int = 512
    sparse_backend: str = "native_bm25"
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    combined_strategy: str = "hhr_interleave"
    rrf_k: int = 60
    score_fusion: str = "passage_only"
    document_score_weight: float = 0.0
    rebuild_indices: bool = False
    run_nq_hard_analysis: bool = False
    save_per_query_results: bool = True
    num_workers: int = 1
    use_synthetic_data: bool = True
    tuning_split: str = "dev"
    evaluation_splits: tuple[str, ...] = ("test",)
    tuned_on_dataset: str = "ms_marco"
    frozen_parameters: bool = True

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in ("data_root", "cache_dir", "output_dir"):
            result[key] = str(result[key])
        return result


def config_for_mode(run_mode: str, **overrides: Any) -> RunConfig:
    if run_mode not in RUN_MODES:
        raise ValueError(
            f"Unknown run_mode {run_mode!r}; expected one of {sorted(RUN_MODES)}"
        )
    values = {**RUN_MODES[run_mode], "run_mode": run_mode}
    values["use_synthetic_data"] = run_mode == "smoke"
    values.update(overrides)
    config = RunConfig(**values)
    validate_config(config)
    return config


def validate_config(config: RunConfig) -> None:
    errors: list[str] = []
    if config.run_mode not in RUN_MODES:
        errors.append(f"run_mode must be one of {sorted(RUN_MODES)}")
    if not config.experiment_name.strip():
        errors.append("experiment_name must not be empty")
    unknown_methods = sorted(set(config.methods) - set(ALL_METHODS))
    if unknown_methods:
        errors.append(f"unknown methods: {unknown_methods}")
    valid_datasets = set(ALL_DATASETS) | {"synthetic"}
    unknown_datasets = sorted(set(config.datasets) - valid_datasets)
    if unknown_datasets:
        errors.append(f"unknown datasets: {unknown_datasets}")
    for name in (
        "document_top_k",
        "passage_top_k",
        "final_top_k",
        "dense_batch_size",
        "num_workers",
    ):
        if getattr(config, name) <= 0:
            errors.append(f"{name} must be positive")
    if config.final_top_k > config.passage_top_k:
        errors.append("final_top_k cannot exceed passage_top_k")
    if config.query_sample_size is not None and config.query_sample_size <= 0:
        errors.append("query_sample_size must be positive or None")
    if config.dense_document_strategy not in {
        "title_plus_lead",
        "title_only",
        "lead_only",
    }:
        errors.append(
            "dense_document_strategy must be title_plus_lead, title_only, or lead_only"
        )
    if config.dense_backend not in {"hashing", "transformers"}:
        errors.append("dense_backend must be hashing or transformers")
    if config.dense_index not in {"exact", "faiss"}:
        errors.append("dense_index must be exact or faiss")
    if config.sparse_backend != "native_bm25":
        errors.append(
            "this Phase 1 package currently supports sparse_backend='native_bm25'"
        )
    if config.combined_strategy not in {"hhr_interleave", "rrf"}:
        errors.append("combined_strategy must be hhr_interleave or rrf")
    if config.rrf_k <= 0:
        errors.append("rrf_k must be positive")
    if config.score_fusion not in {"passage_only", "weighted_sum"}:
        errors.append("score_fusion must be passage_only or weighted_sum")
    if not 0.0 <= config.document_score_weight <= 1.0:
        errors.append("document_score_weight must be in [0, 1]")
    if config.use_synthetic_data and tuple(config.datasets) != ("synthetic",):
        errors.append("use_synthetic_data requires datasets=('synthetic',)")
    if config.run_mode != "smoke" and config.dense_backend == "hashing":
        errors.append(
            "hashing dense backend is synthetic smoke-only; use transformers for "
            "benchmark runs"
        )
    if config.tuned_on_dataset != "ms_marco" or config.tuning_split not in {
        "train",
        "dev",
    }:
        errors.append("zero-shot protocol requires tuning on MS MARCO train/dev only")
    if "test" in {config.tuning_split}:
        errors.append("test labels must never be used for tuning")
    if config.run_mode in {"baseline", "full"} and not config.frozen_parameters:
        errors.append("baseline/full evaluation requires frozen_parameters=True")
    if errors:
        raise ValueError("Invalid HHR configuration:\n- " + "\n- ".join(errors))
