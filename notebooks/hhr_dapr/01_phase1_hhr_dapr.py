# %% [markdown]
# # Phase 1 — Hybrid Hierarchical Retrieval on DAPR
#
# Kaggle-ready, reproducible benchmark notebook. DAPR download, normalization,
# validation, sampling, and dataset-specific diagnostics intentionally live here.
# The reusable `src/hhr_dapr` package begins at the normalized dataset contract.
#
# Default `smoke` mode uses a deterministic local corpus and does **not** report
# DAPR benchmark quality. Change only the central configuration cell for real runs.

# %% [markdown]
# ## 1. Kaggle bootstrap
#
# You may upload only this notebook to Kaggle. With Internet enabled, it clones the
# repository if needed. If you cloned the repository yourself, the clone is reused.

# %%
# ruff: noqa: E402
from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

REPOSITORY_URL = "https://github.com/ManhTanTran/kaggle-notebook.git"
START_DIR = Path.cwd().resolve()


def find_project_root(start: Path) -> Path | None:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src" / "hhr_dapr"
        ).is_dir():
            return candidate
    return None


PROJECT_ROOT = find_project_root(START_DIR)
ON_KAGGLE = os.name != "nt" and Path("/kaggle/working").is_dir()
if PROJECT_ROOT is None:
    clone_root = (
        Path("/kaggle/working/kaggle-notebook")
        if ON_KAGGLE
        else START_DIR / "kaggle-notebook"
    )
    if not (clone_root / ".git").is_dir():
        subprocess.check_call(
            ["git", "clone", "--depth", "1", REPOSITORY_URL, str(clone_root)]
        )
    PROJECT_ROOT = clone_root.resolve()

if ON_KAGGLE:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-e", f"{PROJECT_ROOT}[hhr]"]
    )

source_root = str(PROJECT_ROOT / "src")
if source_root not in sys.path:
    sys.path.insert(0, source_root)
os.chdir(PROJECT_ROOT)
print({"project_root": str(PROJECT_ROOT), "on_kaggle": ON_KAGGLE})

# %% [markdown]
# ## 2. Central run configuration
#
# This is the only cell normally edited. `baseline` and `full` use the complete
# selected corpora and can exceed one Kaggle session, especially MIRACL and
# Genomics. Start with one dataset and a query sample; query sampling does not make
# corpus indexing smaller.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from hhr_dapr.artifacts import create_run_directory, export_run_artifacts
from hhr_dapr.config import (
    ALL_METHODS,
    RECOMMENDED_METHODS,
    RunConfig,
    validate_config,
)
from hhr_dapr.experiments import build_experiment_registry, run_experiment_matrix
from hhr_dapr.metrics import ndcg_at_k, recall_at_k

ALL_DAPR_DATASETS = (
    "ms_marco",
    "natural_questions",
    "miracl_en",
    "genomics",
    "conditional_qa",
    "nq_hard",
)
RUN_MODES = {
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
        "datasets": ALL_DAPR_DATASETS,
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
        "datasets": ALL_DAPR_DATASETS,
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

WORK_ROOT = Path("/kaggle/working") if ON_KAGGLE else PROJECT_ROOT
CONFIG = {
    "run_mode": "smoke",  # smoke | baseline | full
    # Optional overrides, e.g. ("natural_questions",) or query_sample_size=100.
    "datasets": None,
    "query_sample_size": None,
    "hf_repo_id": "UKPLab/dapr",
    "hf_revision": "67ae3daa13596700976d20605630f5f9db3bd732",
    "hf_cache_dir": WORK_ROOT / "cache" / "huggingface",
    "cache_dir": WORK_ROOT / "cache" / "hhr_dapr",
    "output_dir": WORK_ROOT / "outputs",
    "random_seed": 42,
    "dense_query_model": "facebook/dragon-plus-query-encoder",
    "dense_context_model": "facebook/dragon-plus-context-encoder",
    "dense_query_revision": "2d3808c087119b953f8494b7638c216c71712cee",
    "dense_context_revision": "68074e7406bb0061b0d049b58592acafae00e9d4",
    "dense_device": "cuda" if ON_KAGGLE else "cpu",
    "rebuild_indices": False,
    # Zero-shot protocol controls; do not tune with test/NQ-hard labels.
    "tuning_dataset": "ms_marco",
    "tuning_split": "dev",
    "evaluation_splits": ("test",),
    "frozen_parameters": True,
}

mode = dict(RUN_MODES[CONFIG["run_mode"]])
if CONFIG["datasets"] is not None:
    mode["datasets"] = tuple(CONFIG["datasets"])
if CONFIG["query_sample_size"] is not None:
    mode["query_sample_size"] = CONFIG["query_sample_size"]

if ON_KAGGLE and CONFIG["run_mode"] != "smoke":
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "-e",
            f"{PROJECT_ROOT}[hhr-dense]",
        ]
    )

RUN_CONFIG = RunConfig(
    experiment_name="phase1_hhr_dapr",
    run_mode=CONFIG["run_mode"],
    cache_dir=Path(CONFIG["cache_dir"]),
    output_dir=Path(CONFIG["output_dir"]),
    methods=tuple(mode["methods"]),
    random_seed=int(CONFIG["random_seed"]),
    document_top_k=int(mode["document_top_k"]),
    passage_top_k=int(mode["passage_top_k"]),
    final_top_k=int(mode["final_top_k"]),
    dense_query_model=str(CONFIG["dense_query_model"]),
    dense_context_model=str(CONFIG["dense_context_model"]),
    dense_query_revision=str(CONFIG["dense_query_revision"]),
    dense_context_revision=str(CONFIG["dense_context_revision"]),
    dense_device=str(CONFIG["dense_device"]),
    dense_backend=str(mode["dense_backend"]),
    dense_index=str(mode["dense_index"]),
    rebuild_indices=bool(CONFIG["rebuild_indices"]),
)
validate_config(RUN_CONFIG)
random.seed(RUN_CONFIG.random_seed)
np.random.seed(RUN_CONFIG.random_seed)
print(json.dumps({**RUN_CONFIG.to_dict(), "datasets": mode["datasets"]}, indent=2))

# %% [markdown]
# ## 3. DAPR schema registry and normalized contract
#
# Official Hugging Face configuration names are mapped explicitly. The revision is
# pinned above so reruns do not silently change data.

# %%
HF_DAPR_SPECS = {
    "ms_marco": {"prefix": "MSMARCO", "query_split": "test"},
    "natural_questions": {"prefix": "NaturalQuestions", "query_split": "test"},
    "miracl_en": {"prefix": "MIRACL", "query_split": "test"},
    "genomics": {"prefix": "Genomics", "query_split": "test"},
    "conditional_qa": {"prefix": "ConditionalQA", "query_split": "test"},
}
NQ_CATEGORY_MAP = {
    "coreference": "CR",
    "main_topic": "MT",
    "multi-hop": "MHR",
    "acronym": "AC",
    "cr": "CR",
    "mt": "MT",
    "mhr": "MHR",
    "ac": "AC",
}


@dataclass(frozen=True)
class NormalizedDAPRDataset:
    name: str
    documents: pd.DataFrame
    passages: pd.DataFrame
    queries: pd.DataFrame
    qrels: pd.DataFrame
    query_metadata: pd.DataFrame | None = None


def as_frame(table: Any) -> pd.DataFrame:
    return table.copy() if isinstance(table, pd.DataFrame) else table.to_pandas()


def normalize_hf_documents(table: Any) -> pd.DataFrame:
    raw = as_frame(table)
    required = {"doc_id", "title", "passage_ids", "passages"}
    if missing := required - set(raw.columns):
        raise ValueError(f"DAPR docs missing columns: {sorted(missing)}")
    bad = raw.apply(lambda row: len(row["passage_ids"]) != len(row["passages"]), axis=1)
    if bad.any():
        raise ValueError("DAPR docs contain mismatched passage_ids/passages lengths")
    return pd.DataFrame(
        {
            "document_id": raw["doc_id"].astype(str),
            "title": raw["title"].fillna("").astype(str),
            "text": raw["passages"].map(lambda values: "\n\n".join(map(str, values))),
        }
    )


def normalize_hf_passages(table: Any) -> pd.DataFrame:
    raw = as_frame(table)
    required = {"_id", "doc_id", "text", "paragraph_no"}
    if missing := required - set(raw.columns):
        raise ValueError(f"DAPR corpus missing columns: {sorted(missing)}")
    normalized = pd.DataFrame(
        {
            "passage_id": raw["_id"].astype(str),
            "document_id": raw["doc_id"].astype(str),
            "passage_text": raw["text"].fillna("").astype(str),
            "passage_position": pd.to_numeric(
                raw["paragraph_no"], errors="raise"
            ).astype(int),
        }
    )
    for column in ("is_candidate", "total_paragraphs"):
        if column in raw:
            normalized[column] = raw[column].values
    return normalized


def normalize_hf_queries(table: Any, dataset_name: str, split: str) -> pd.DataFrame:
    raw = as_frame(table)
    required = {"_id", "text"}
    if missing := required - set(raw.columns):
        raise ValueError(f"DAPR queries missing columns: {sorted(missing)}")
    return pd.DataFrame(
        {
            "query_id": raw["_id"].astype(str),
            "query_text": raw["text"].fillna("").astype(str),
            "dataset": dataset_name,
            "split": split,
        }
    )


def normalize_hf_qrels(table: Any) -> pd.DataFrame:
    raw = as_frame(table)
    required = {"query_id", "corpus_id", "score"}
    if missing := required - set(raw.columns):
        raise ValueError(f"DAPR qrels missing columns: {sorted(missing)}")
    normalized = pd.DataFrame(
        {
            "query_id": raw["query_id"].astype(str),
            "passage_id": raw["corpus_id"].astype(str),
            "relevance": pd.to_numeric(raw["score"], errors="raise"),
        }
    )
    return normalized.groupby(["query_id", "passage_id"], as_index=False)[
        "relevance"
    ].max()


def normalize_nq_hard(table: Any) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = as_frame(table)
    required = {"query_id", "corpus_id", "score", "query", "categories"}
    if missing := required - set(raw.columns):
        raise ValueError(f"nq-hard missing columns: {sorted(missing)}")
    raw = raw.copy()
    raw["query_id"] = raw["query_id"].astype(str)
    queries = (
        raw[["query_id", "query"]]
        .drop_duplicates("query_id")
        .rename(columns={"query": "query_text"})
    )
    queries["dataset"] = "nq_hard"
    queries["split"] = "test"
    qrels = normalize_hf_qrels(raw)

    def collect_categories(values: pd.Series) -> str:
        labels = {
            NQ_CATEGORY_MAP.get(str(category).strip().lower(), str(category).upper())
            for value in values
            if value is not None
            for category in value
        }
        return "|".join(sorted(labels))

    metadata = (
        raw.groupby("query_id")["categories"]
        .agg(collect_categories)
        .rename("question_type")
        .reset_index()
    )
    return queries, qrels, metadata


# %% [markdown]
# ## 4. Notebook-local validation, sampling, audit, and protocol

# %%
REQUIRED_COLUMNS = {
    "documents": {"document_id", "title", "text"},
    "passages": {"passage_id", "document_id", "passage_text", "passage_position"},
    "queries": {"query_id", "query_text", "dataset", "split"},
    "qrels": {"query_id", "passage_id", "relevance"},
}


def validate_dapr_dataset(dataset: NormalizedDAPRDataset) -> None:
    errors: list[str] = []
    for name, required in REQUIRED_COLUMNS.items():
        frame = getattr(dataset, name)
        if missing := required - set(frame.columns):
            errors.append(f"{name} missing columns {sorted(missing)}")
    if errors:
        raise ValueError("Invalid normalized DAPR dataset:\n- " + "\n- ".join(errors))
    if dataset.documents["document_id"].duplicated().any():
        errors.append("duplicate document_id")
    if dataset.passages["passage_id"].duplicated().any():
        errors.append("duplicate passage_id")
    if dataset.queries["query_id"].duplicated().any():
        errors.append("duplicate query_id")
    document_ids = set(dataset.documents["document_id"].astype(str))
    passage_ids = set(dataset.passages["passage_id"].astype(str))
    query_ids = set(dataset.queries["query_id"].astype(str))
    if unknown := set(dataset.passages["document_id"].astype(str)) - document_ids:
        errors.append(f"passages reference {len(unknown)} unknown documents")
    if unknown := set(dataset.qrels["passage_id"].astype(str)) - passage_ids:
        errors.append(f"qrels reference {len(unknown)} unknown passages")
    if unknown := set(dataset.qrels["query_id"].astype(str)) - query_ids:
        errors.append(f"qrels reference {len(unknown)} unknown queries")
    if (pd.to_numeric(dataset.passages["passage_position"]) < 0).any():
        errors.append("passage_position must be non-negative")
    if (pd.to_numeric(dataset.qrels["relevance"]) < 0).any():
        errors.append("relevance must be non-negative")
    for frame_name, column in (
        ("documents", "document_id"),
        ("passages", "passage_id"),
        ("queries", "query_id"),
    ):
        if getattr(dataset, frame_name)[column].astype(str).str.strip().eq("").any():
            errors.append(f"{frame_name}.{column} contains empty values")
    if errors:
        raise ValueError("Invalid normalized DAPR dataset:\n- " + "\n- ".join(errors))


def sample_dapr_queries(
    dataset: NormalizedDAPRDataset, sample_size: int | None, seed: int
) -> NormalizedDAPRDataset:
    if sample_size is None or sample_size >= len(dataset.queries):
        return dataset
    selected = dataset.queries.sample(n=sample_size, random_state=seed)
    selected_ids = set(selected["query_id"].astype(str))
    qrels = dataset.qrels.loc[
        dataset.qrels["query_id"].astype(str).isin(selected_ids)
    ].copy()
    metadata = dataset.query_metadata
    if metadata is not None:
        metadata = metadata.loc[
            metadata["query_id"].astype(str).isin(selected_ids)
        ].copy()
    sampled = replace(
        dataset,
        queries=selected.reset_index(drop=True),
        qrels=qrels.reset_index(drop=True),
        query_metadata=metadata,
    )
    validate_dapr_dataset(sampled)
    return sampled


def audit_dapr_dataset(dataset: NormalizedDAPRDataset) -> dict[str, Any]:
    return {
        "dataset": dataset.name,
        "documents": len(dataset.documents),
        "passages": len(dataset.passages),
        "queries": len(dataset.queries),
        "qrels": len(dataset.qrels),
        "graded_qrels": int((pd.to_numeric(dataset.qrels["relevance"]) > 1).sum()),
        "splits": "|".join(sorted(dataset.queries["split"].astype(str).unique())),
    }


def validate_zero_shot_protocol(
    config: dict[str, Any], selected_datasets: tuple[str, ...]
) -> None:
    if config["tuning_dataset"] != "ms_marco" or config["tuning_split"] not in {
        "train",
        "dev",
    }:
        raise ValueError("Parameter selection is limited to MS MARCO train/dev")
    if "test" == config["tuning_split"]:
        raise ValueError("Test labels must never be used for tuning")
    if "nq_hard" in selected_datasets and config["tuning_dataset"] == "nq_hard":
        raise ValueError("NQ-hard is diagnostic-only")
    if config["run_mode"] != "smoke" and not config["frozen_parameters"]:
        raise ValueError("Real zero-shot evaluation requires frozen parameters")


validate_zero_shot_protocol(CONFIG, tuple(mode["datasets"]))

# %% [markdown]
# ## 5. Hugging Face download and DAPR-specific loading


# %%
def load_hf_config(config_name: str, split: str):
    from datasets import load_dataset as hf_load_dataset

    print(f"Downloading/loading {config_name}:{split}")
    return hf_load_dataset(
        CONFIG["hf_repo_id"],
        config_name,
        split=split,
        revision=CONFIG["hf_revision"],
        cache_dir=str(CONFIG["hf_cache_dir"]),
    )


def load_and_normalize_hf_dapr(dataset_name: str) -> NormalizedDAPRDataset:
    if dataset_name == "nq_hard":
        prefix = "NaturalQuestions"
        documents = normalize_hf_documents(load_hf_config(f"{prefix}-docs", "test"))
        passages = normalize_hf_passages(load_hf_config(f"{prefix}-corpus", "test"))
        queries, qrels, metadata = normalize_nq_hard(load_hf_config("nq-hard", "test"))
        result = NormalizedDAPRDataset(
            dataset_name, documents, passages, queries, qrels, metadata
        )
    else:
        spec = HF_DAPR_SPECS[dataset_name]
        prefix = spec["prefix"]
        split = spec["query_split"]
        result = NormalizedDAPRDataset(
            name=dataset_name,
            documents=normalize_hf_documents(load_hf_config(f"{prefix}-docs", "test")),
            passages=normalize_hf_passages(load_hf_config(f"{prefix}-corpus", "test")),
            queries=normalize_hf_queries(
                load_hf_config(f"{prefix}-queries", split), dataset_name, split
            ),
            qrels=normalize_hf_qrels(load_hf_config(f"{prefix}-qrels", split)),
        )
    validate_dapr_dataset(result)
    return result


# %% [markdown]
# ## 6. Deterministic synthetic smoke dataset


# %%
def make_synthetic_dapr_dataset() -> NormalizedDAPRDataset:
    documents = pd.DataFrame(
        [
            ("d1", "Paris", "Paris France Seine"),
            ("d2", "Marie Curie", "Marie Curie Warsaw radioactivity"),
            ("d3", "Mars", "Mars Phobos Deimos"),
            ("d4", "Pacific Ocean", "Pacific Ocean largest ocean"),
        ],
        columns=["document_id", "title", "text"],
    )
    passages = pd.DataFrame(
        [
            ("p1", "d1", "Paris is the capital of France.", 0),
            ("p2", "d1", "The Seine crosses Paris.", 1),
            ("p3", "d2", "Marie Curie was born in Warsaw.", 0),
            ("p4", "d2", "She researched radioactivity.", 1),
            ("p5", "d3", "Mars is the fourth planet.", 0),
            ("p6", "d3", "Phobos and Deimos are moons of Mars.", 1),
            ("p7", "d4", "The Pacific Ocean is Earth's largest ocean.", 0),
        ],
        columns=["passage_id", "document_id", "passage_text", "passage_position"],
    )
    queries = pd.DataFrame(
        [
            ("q1", "capital of France", "synthetic", "test"),
            ("q2", "where was Marie Curie born", "synthetic", "test"),
            ("q3", "moons of Mars", "synthetic", "test"),
            ("q4", "largest ocean", "synthetic", "test"),
        ],
        columns=["query_id", "query_text", "dataset", "split"],
    )
    qrels = pd.DataFrame(
        [("q1", "p1", 2), ("q2", "p3", 1), ("q3", "p6", 1), ("q4", "p7", 1)],
        columns=["query_id", "passage_id", "relevance"],
    )
    result = NormalizedDAPRDataset("synthetic", documents, passages, queries, qrels)
    validate_dapr_dataset(result)
    return result


# Exercise the official-shape normalizers even when no network dataset is downloaded.
_mini_docs = pd.DataFrame(
    {"doc_id": ["d"], "title": ["T"], "passage_ids": [["p"]], "passages": [["body"]]}
)
_mini_corpus = pd.DataFrame(
    {"_id": ["p"], "doc_id": ["d"], "text": ["body"], "paragraph_no": [0]}
)
_mini_nq_hard = pd.DataFrame(
    {
        "query_id": ["q", "q"],
        "corpus_id": ["p", "p2"],
        "score": [1, 1],
        "query": ["question", "question"],
        "categories": [["coreference"], ["acronym", "coreference"]],
    }
)
assert normalize_hf_documents(_mini_docs).iloc[0].text == "body"
assert normalize_hf_passages(_mini_corpus).iloc[0].passage_id == "p"
_, _mini_nq_qrels, _mini_nq_metadata = normalize_nq_hard(_mini_nq_hard)
assert len(_mini_nq_qrels) == 2
assert set(_mini_nq_metadata.iloc[0].question_type.split("|")) == {"AC", "CR"}

# %% [markdown]
# ## 7. Load, normalize, validate, sample, and audit

# %%
datasets: list[NormalizedDAPRDataset] = []
for dataset_name in mode["datasets"]:
    loaded = (
        make_synthetic_dapr_dataset()
        if dataset_name == "synthetic"
        else load_and_normalize_hf_dapr(dataset_name)
    )
    loaded = sample_dapr_queries(
        loaded, mode["query_sample_size"], RUN_CONFIG.random_seed
    )
    validate_dapr_dataset(loaded)
    datasets.append(loaded)

dataset_audit_table = pd.DataFrame(
    [audit_dapr_dataset(dataset) for dataset in datasets]
)
display(dataset_audit_table)

# %% [markdown]
# ## 8. HHR experiment registry and schedule

# %%
RUN_CONFIG.cache_dir.mkdir(parents=True, exist_ok=True)
EXPERIMENT_REGISTRY = build_experiment_registry(RUN_CONFIG)
assert set(EXPERIMENT_REGISTRY) == set(ALL_METHODS)
experiment_registry_table = pd.DataFrame(
    [
        {
            "method": item.name,
            "document_method": item.document_method,
            "passage_method": item.passage_method,
            "combined_strategy": item.combined_strategy,
        }
        for item in EXPERIMENT_REGISTRY.values()
    ]
)
display(experiment_registry_table)
print(
    "Scheduled:", [(d.name, method) for d in datasets for method in RUN_CONFIG.methods]
)

# %% [markdown]
# ## 9. Metric contract validation

# %%
manual_relevance = {"high": 2, "medium": 1, "none": 0}
manual_actual = 1.0 + 3.0 / np.log2(3)
manual_ideal = 3.0 + 1.0 / np.log2(3)
assert np.isclose(
    ndcg_at_k(["medium", "high", "none"], manual_relevance, 10),
    manual_actual / manual_ideal,
)
assert np.isclose(recall_at_k(["none", "high"], manual_relevance, 2), 0.5)
print("Metric checks passed (graded nDCG and positive-label recall).")

# %% [markdown]
# ## 10. Execute the selected experiment matrix

# %%
results = run_experiment_matrix(datasets, RUN_CONFIG)
assert len(results) == len(datasets) * len(RUN_CONFIG.methods)
assert all(result.status == "completed" for result in results)
assert all(result.run_metadata["reranker"] is None for result in results)
print(f"Completed {len(results)} measured dataset/method runs.")

# %% [markdown]
# ## 11. Effectiveness comparison

# %%
aggregate_rows = [
    {
        "dataset": result.dataset,
        "method": result.experiment.name,
        "status": result.status,
        **result.aggregate_metrics,
    }
    for result in results
]
aggregate_method_dataset = pd.DataFrame(aggregate_rows).sort_values(
    ["dataset", "passage_ndcg@10"], ascending=[True, False]
)
display(aggregate_method_dataset)

# %% [markdown]
# ## 12. Dataset/domain comparison

# %%
domain_comparison = aggregate_method_dataset.pivot_table(
    index="method", columns="dataset", values=["passage_ndcg@10", "passage_recall@100"]
)
display(domain_comparison)

# %% [markdown]
# ## 13. NQ-hard multi-label diagnostics
#
# A query contributes to every listed category (CR, MT, MHR, AC).


# %%
def category_labels(value: object) -> set[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    labels = set()
    for part in re.split(r"[,;|\s]+", str(value).strip()):
        normalized = NQ_CATEGORY_MAP.get(part.lower())
        if normalized:
            labels.add(normalized)
    return labels


def nq_hard_by_category(
    per_query: pd.DataFrame, metadata: pd.DataFrame
) -> pd.DataFrame:
    merged = per_query.merge(
        metadata[["query_id", "question_type"]], on="query_id", how="left"
    )
    rows = []
    for category in ("overall", "CR", "MT", "MHR", "AC"):
        selected = (
            merged
            if category == "overall"
            else merged.loc[
                merged["question_type"].map(
                    lambda value, selected=category: selected in category_labels(value)
                )
            ]
        )
        rows.append(
            {
                "category": category,
                "query_count": len(selected),
                "passage_ndcg@10": selected["passage_ndcg@10"].mean()
                if len(selected)
                else np.nan,
                "passage_recall@100": selected["passage_recall@100"].mean()
                if len(selected)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


nq_rows = []
if mode["run_nq_hard_analysis"]:
    nq_dataset = next(
        (dataset for dataset in datasets if dataset.name == "nq_hard"), None
    )
    if nq_dataset is None or nq_dataset.query_metadata is None:
        raise RuntimeError(
            "NQ-hard diagnostics require the nq_hard dataset and metadata"
        )
    for result in (item for item in results if item.dataset == "nq_hard"):
        frame = nq_hard_by_category(result.per_query_metrics, nq_dataset.query_metadata)
        frame.insert(0, "method", result.experiment.name)
        frame.insert(0, "dataset", "nq_hard")
        frame["status"] = result.status
        nq_rows.append(frame)
nq_category_results = (
    pd.concat(nq_rows, ignore_index=True) if nq_rows else pd.DataFrame()
)
display(
    nq_category_results
    if len(nq_category_results)
    else pd.DataFrame({"status": ["not_run"]})
)

# %% [markdown]
# ## 14. Latency and storage comparison

# %%
efficiency_table = aggregate_method_dataset[
    [
        "dataset",
        "method",
        "passage_ndcg@10",
        "passage_recall@100",
        "mean_query_latency_ms",
        "p50_query_latency_ms",
        "p95_query_latency_ms",
        "mean_unique_result_count",
    ]
].copy()
storage_lookup = {
    (result.dataset, result.experiment.name): result.index_metadata[
        "total_storage_bytes"
    ]
    for result in results
}
efficiency_table["index_storage_bytes"] = [
    storage_lookup[(row.dataset, row.method)] for row in efficiency_table.itertuples()
]
display(efficiency_table)
if len(efficiency_table):
    efficiency_table.plot.scatter(
        x="mean_query_latency_ms", y="passage_ndcg@10", title="Effectiveness vs latency"
    )
    plt.show()

# %% [markdown]
# ## 15. Failure analysis (IDs only)

# %%
failure_rows = []
for result in results:
    dataset = next(item for item in datasets if item.name == result.dataset)
    relevant = dataset.qrels.loc[dataset.qrels["relevance"] > 0]
    observed = result.passage_rankings[
        ["query_id", "passage_id", "passage_rank", "document_rank"]
    ]
    joined = relevant.merge(observed, on=["query_id", "passage_id"], how="left")
    missed = joined.loc[joined["passage_rank"].isna(), ["query_id", "passage_id"]]
    if len(missed):
        missed = missed.copy()
        missed.insert(0, "method", result.experiment.name)
        missed.insert(0, "dataset", result.dataset)
        failure_rows.append(missed)
failure_analysis = (
    pd.concat(failure_rows, ignore_index=True)
    if failure_rows
    else pd.DataFrame(columns=["dataset", "method", "query_id", "passage_id"])
)
display(failure_analysis.head(50))

# %% [markdown]
# ## 16. Artifact export

# %%
run_dir, git_commit = create_run_directory(RUN_CONFIG, PROJECT_ROOT)
expected_pairs = [
    (dataset.name, method) for dataset in datasets for method in ALL_METHODS
]
artifact_paths = export_run_artifacts(
    run_dir=run_dir,
    config=RUN_CONFIG,
    results=results,
    dataset_audit=dataset_audit_table,
    expected_pairs=expected_pairs,
    nq_category_results=nq_category_results if len(nq_category_results) else None,
    git_commit=git_commit,
    synthetic_smoke=CONFIG["run_mode"] == "smoke",
)
print("Artifacts:", run_dir)
display(
    pd.DataFrame(
        {"artifact": artifact_paths.keys(), "path": map(str, artifact_paths.values())}
    )
)

# %% [markdown]
# ## 17. Conclusion and Phase 2 handoff

# %%
if CONFIG["run_mode"] == "smoke":
    recommendation = (
        "No DAPR method recommendation: this is synthetic plumbing validation only. "
        "Run a frozen baseline/full configuration on DAPR before drawing conclusions."
    )
else:
    measured = (
        aggregate_method_dataset.groupby("method")["passage_ndcg@10"]
        .mean()
        .sort_values(ascending=False)
    )
    recommendation = (
        "Measured Phase 1 leader by mean passage nDCG@10: "
        f"{measured.index[0]} ({measured.iloc[0]:.4f})."
    )
print(recommendation)
print(
    "Phase 2 input contract: normalized passage rankings and scores; "
    "reranker remains None in Phase 1."
)
