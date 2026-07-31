import ast
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from chunkbench.config.loader import load_config
from chunkbench.eval.constants import PRIMARY_METRICS
from chunkbench.pipeline.matrix import (
    REQUIRED_METHOD_ARTIFACTS,
    ExperimentMatrixRunner,
)


def test_matrix_expansion_and_single_dataset_backward_compatibility():
    core = load_config("configs/experiments/all_qa_datasets_core_methods.yaml")
    assert len(ExperimentMatrixRunner(core).expand()) == 48
    smoke = load_config("configs/experiments/all_qa_datasets_smoke.yaml")
    assert len(ExperimentMatrixRunner(smoke).expand()) == 16
    single = load_config("configs/experiments/smoke_test.yaml")
    assert single.dataset["name"] == "synthetic"
    assert len(single.datasets) == 1
    assert not single.matrix
    referenced = load_config("configs/experiments/qasper_fixture_single.yaml")
    assert referenced.dataset["name"] == "qasper"
    assert not referenced.matrix


def test_fixture_matrix_smoke_resume_and_artifacts(tmp_path: Path):
    config = load_config("configs/experiments/all_qa_datasets_smoke.yaml")
    config = replace(config, output_dir=tmp_path)
    first = ExperimentMatrixRunner(config).run()
    assert len(first["completed_runs"]) == 16
    assert first["failed_runs"] == []
    for dataset in ("qasper", "hotpotqa_fullwiki", "uit_viquad", "vimqa"):
        for method in (
            "fixed_256",
            "fixed_512",
            "sentence_fixed_256",
            "sentence_fixed_512",
        ):
            method_dir = tmp_path / config.run_name / dataset / method
            assert all(
                (method_dir / artifact).exists()
                for artifact in REQUIRED_METHOD_ARTIFACTS
            )
            metrics = json.loads(
                (method_dir / "metrics.json").read_text(encoding="utf-8")
            )
            assert set(metrics) == set(PRIMARY_METRICS)
    aggregate = pd.read_csv(tmp_path / config.run_name / "benchmark_metrics.csv")
    assert {"Dataset", "Method"} <= set(aggregate)
    assert len(aggregate) == 16
    assert (tmp_path / config.run_name / "dataset_validation.csv").exists()

    runtime_path = tmp_path / config.run_name / "qasper" / "fixed_256" / "runtime.json"
    runtime_mtime = runtime_path.stat().st_mtime_ns
    resumed = replace(config, resume=True, skip_completed=True)
    second = ExperimentMatrixRunner(resumed).run()
    assert len(second["completed_runs"]) == 16
    assert second["failed_runs"] == []
    assert runtime_path.stat().st_mtime_ns == runtime_mtime


def test_advanced_failure_preserves_baseline_artifacts(tmp_path: Path):
    source = load_config("configs/experiments/all_qa_datasets_smoke.yaml")
    config = replace(
        source,
        run_name="advanced_failure_test",
        output_dir=tmp_path,
        datasets=(source.datasets[0],),
        dataset=source.datasets[0],
        methods=(
            {"name": "fixed_256", "chunk_size": 256},
            {"name": "meta_ppl_raw", "backend_type": "unsupported"},
        ),
    )
    result = ExperimentMatrixRunner(config).run()
    assert len(result["completed_runs"]) == 1
    assert len(result["failed_runs"]) == 1
    assert result["failed_runs"][0]["error_type"] == "ValueError"
    root = tmp_path / config.run_name / "qasper"
    assert (root / "fixed_256" / "metrics.json").exists()
    assert not (root / "meta_ppl_raw" / "metrics.json").exists()


def test_notebooks_contain_no_function_or_class_implementation():
    for path in Path("notebooks").glob("*.ipynb"):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        tree = ast.parse(source or "pass")
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            for node in ast.walk(tree)
        )


def test_kaggle_hotpot_conversion_uses_contexts_containing_gold_documents():
    notebook = json.loads(
        Path("notebooks/05_kaggle_benchmark_vi.ipynb").read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert 'hotpot_config = "distractor"' in source
    assert 'hotpot_config = "fullwiki"' not in source
