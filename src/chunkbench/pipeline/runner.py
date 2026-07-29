"""End-to-end benchmark runner."""

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from chunkbench.common.environment import environment_fingerprint
from chunkbench.common.seed import set_seed
from chunkbench.config.schema import ExperimentConfig
from chunkbench.data.validation import validate_dataset
from chunkbench.embedding.cache import build_cache_key, fingerprint
from chunkbench.eval.chunk_statistics import chunk_statistics
from chunkbench.eval.evaluator import evaluate
from chunkbench.evidence.overlap import map_evidence
from chunkbench.pipeline.artifacts import save_method_artifacts, save_run_summary
from chunkbench.pipeline.representations import build_representation_strategy
from chunkbench.pipeline.stages import chunk_documents
from chunkbench.pipeline.validation import validate_chunks
from chunkbench.registry.datasets import build_dataset_adapter
from chunkbench.registry.embedders import build_embedder
from chunkbench.registry.methods import build_method_pipeline_spec
from chunkbench.registry.retrievers import build_retriever


class BenchmarkRunner:
    """Orchestrate all benchmark stages with per-method fault isolation."""

    def __init__(
        self, config: ExperimentConfig, project_root: Path | None = None
    ) -> None:
        self.config = config
        self.project_root = project_root or Path.cwd()

    def run(self) -> dict[str, Any]:
        """Run selected methods and persist per-method and summary artifacts."""
        set_seed(self.config.seed)
        adapter_name = str(
            self.config.dataset.get("adapter", self.config.dataset["name"])
        )
        bundle = build_dataset_adapter(adapter_name, self.config.dataset).load()
        validation_report = validate_dataset(
            bundle, self.config.dataset.get("validation")
        )
        run_dir = self.config.output_dir / self.config.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        environment = environment_fingerprint(self.project_root)
        dataset_fingerprint = fingerprint(
            {
                "metadata": bundle.metadata,
                "documents": [
                    {"id": item.document_id, "text": item.text}
                    for item in bundle.documents
                ],
                "queries": [
                    {"id": item.query_id, "text": item.text} for item in bundle.queries
                ],
            }
        )
        metric_rows: list[dict[str, Any]] = []
        statistic_rows: list[dict[str, Any]] = []
        failed: list[dict[str, str]] = []
        completed: list[str] = []
        for method_spec in self.config.methods:
            name = str(method_spec["name"])
            method_dir = run_dir / name
            if self.config.resume and (method_dir / "metrics.json").exists():
                completed.append(name)
                continue
            started = time.perf_counter()
            try:
                embedder = build_embedder(
                    self.config.embedding["name"], self.config.embedding
                )
                pipeline_spec = build_method_pipeline_spec(name, method_spec, embedder)
                chunks = chunk_documents(bundle.documents, pipeline_spec.chunker)
                validate_chunks(chunks)
                representation_config = {
                    **method_spec,
                    "representation_strategy": pipeline_spec.representation_strategy,
                }
                strategy = build_representation_strategy(
                    representation_config, int(getattr(embedder, "dimension", 256))
                )
                chunk_vectors = strategy.represent(bundle.documents, chunks, embedder)
                query_vectors = embedder.encode_queries(
                    [q.text for q in bundle.queries]
                )
                retriever = build_retriever(
                    self.config.retrieval["name"], self.config.retrieval
                )
                retriever.index(chunks, chunk_vectors)
                hits = retriever.search(
                    bundle.queries,
                    query_vectors,
                    self.config.evaluation.retrieval_depth,
                )
                coverage = map_evidence(chunks, bundle.evidence)
                metrics = evaluate(
                    bundle,
                    chunks,
                    hits,
                    coverage,
                    self.config.evaluation.strict_token_budget,
                )
                statistics = chunk_statistics(
                    chunks, self.config.evaluation.duplicate_scope
                )
                elapsed = time.perf_counter() - started
                backend_type = method_spec.get(
                    "backend_type",
                    method_spec.get("representation", {}).get("backend_type", "real"),
                )
                representation = method_spec.get("representation", {})
                config_hash = build_cache_key(
                    method_name=name,
                    method_config=method_spec,
                    model_name=method_spec.get(
                        "model_name", representation.get("model_name")
                    ),
                    tokenizer_name=method_spec.get(
                        "tokenizer", representation.get("tokenizer")
                    ),
                    model_revision=method_spec.get(
                        "model_revision", representation.get("model_revision")
                    ),
                    dataset_fingerprint=dataset_fingerprint,
                    document_fingerprint=None,
                    code_version=environment.get("git_commit"),
                    implementation_source_commit=pipeline_spec.source_commit,
                    precision=method_spec.get(
                        "precision", representation.get("precision")
                    ),
                    pooling=representation.get("pooling"),
                    long_document_policy=representation.get("long_document_policy"),
                )[:16]
                save_method_artifacts(
                    method_dir,
                    method_spec,
                    environment,
                    chunks,
                    hits,
                    coverage,
                    metrics,
                    statistics,
                    elapsed,
                    {
                        "method": name,
                        "family": pipeline_spec.family,
                        "implementation_fidelity": (
                            pipeline_spec.implementation_fidelity
                        ),
                        "paper_url": pipeline_spec.source_reference,
                        "repository_url": pipeline_spec.repository_url,
                        "source_commit": pipeline_spec.source_commit,
                        "license": "Apache-2.0"
                        if pipeline_spec.source_commit
                        else None,
                        "model_name": method_spec.get(
                            "model_name",
                            method_spec.get("representation", {}).get("model_name"),
                        ),
                        "model_revision": method_spec.get(
                            "model_revision",
                            method_spec.get("representation", {}).get("model_revision"),
                        ),
                        "tokenizer_name": method_spec.get(
                            "tokenizer",
                            method_spec.get("representation", {}).get("tokenizer"),
                        ),
                        "representation_strategy": (
                            pipeline_spec.representation_strategy
                        ),
                        "backend_type": backend_type,
                        "is_mock_backend": backend_type == "mock",
                        "is_publishable_benchmark": backend_type != "mock",
                        "config_hash": config_hash,
                        "dataset_fingerprint": dataset_fingerprint,
                        "git_commit": environment.get("git_commit"),
                        "assumptions": [
                            "No paper-exact claim is made unless fidelity is verified."
                        ],
                        "warnings": [],
                    },
                )
                metric_rows.append({"method": name, **metrics})
                statistic_rows.append({"method": name, **statistics})
                completed.append(name)
            except Exception as error:
                failed.append(
                    {
                        "method": name,
                        "error": str(error),
                        "error_type": type(error).__name__,
                    }
                )
                if self.config.fail_fast:
                    raise
        manifest = {
            "run_name": self.config.run_name,
            "seed": self.config.seed,
            "dataset": self.config.dataset,
            "methods": [item["name"] for item in self.config.methods],
            "completed_methods": completed,
            "environment": environment,
            "config": asdict(self.config),
        }
        save_run_summary(run_dir, metric_rows, statistic_rows, manifest, failed)
        return {
            "run_dir": str(run_dir),
            "completed_methods": completed,
            "failed_methods": failed,
            "metrics": metric_rows,
            "dataset_validation": validation_report,
            "bundle_metadata": bundle.metadata,
        }
