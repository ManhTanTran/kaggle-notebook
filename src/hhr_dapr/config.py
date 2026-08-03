"""Dataset-agnostic retrieval engine configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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


@dataclass(frozen=True)
class RunConfig:
    """Parameters consumed by the reusable HHR retrieval engine."""

    experiment_name: str = "phase1_hhr"
    run_mode: str = "smoke"
    cache_dir: Path = Path("cache/hhr")
    output_dir: Path = Path("outputs")
    methods: tuple[str, ...] = RECOMMENDED_METHODS
    random_seed: int = 42
    document_top_k: int = 3
    passage_top_k: int = 5
    final_top_k: int = 5
    dense_query_model: str = "facebook/dragon-plus-query-encoder"
    dense_context_model: str = "facebook/dragon-plus-context-encoder"
    dense_query_revision: str = "2d3808c087119b953f8494b7638c216c71712cee"
    dense_context_revision: str = "68074e7406bb0061b0d049b58592acafae00e9d4"
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
    save_per_query_results: bool = True
    num_workers: int = 1

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in ("cache_dir", "output_dir"):
            result[key] = str(result[key])
        return result


def validate_config(config: RunConfig) -> None:
    errors: list[str] = []
    if not config.experiment_name.strip():
        errors.append("experiment_name must not be empty")
    unknown_methods = sorted(set(config.methods) - set(ALL_METHODS))
    if unknown_methods:
        errors.append(f"unknown methods: {unknown_methods}")
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
        errors.append("sparse_backend must be 'native_bm25'")
    if config.combined_strategy not in {"hhr_interleave", "rrf"}:
        errors.append("combined_strategy must be hhr_interleave or rrf")
    if config.rrf_k <= 0:
        errors.append("rrf_k must be positive")
    if config.score_fusion not in {"passage_only", "weighted_sum"}:
        errors.append("score_fusion must be passage_only or weighted_sum")
    if not 0.0 <= config.document_score_weight <= 1.0:
        errors.append("document_score_weight must be in [0, 1]")
    if errors:
        raise ValueError("Invalid HHR configuration:\n- " + "\n- ".join(errors))
