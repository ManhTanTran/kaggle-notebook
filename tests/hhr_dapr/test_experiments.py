from __future__ import annotations

from hhr_dapr.config import ALL_METHODS
from hhr_dapr.experiments import build_experiment_registry, run_hhr_experiment


def test_experiment_registry_is_complete(smoke_config):
    registry = build_experiment_registry(smoke_config)
    assert set(registry) == set(ALL_METHODS)
    assert len(registry) == 9


def test_runner_returns_required_result_contract(synthetic_dataset, smoke_config):
    experiment = build_experiment_registry(smoke_config)["combined+combined"]
    result = run_hhr_experiment(synthetic_dataset, experiment, smoke_config)
    assert result.status == "completed"
    assert "passage_ndcg@10" in result.aggregate_metrics
    assert "document_recall@100" in result.aggregate_metrics
    assert "candidate_survival_rate" in result.per_query_metrics
    assert set(result.passage_rankings["document_id"]).issubset(
        set(result.document_rankings["document_id"])
    )
