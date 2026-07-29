import json
from pathlib import Path

from chunkbench.config.loader import load_config
from chunkbench.eval.constants import PRIMARY_METRICS
from chunkbench.pipeline.runner import BenchmarkRunner


def test_advanced_mock_pipeline_executes_all_eight_methods(tmp_path: Path):
    config = load_config("configs/experiments/advanced_methods_mock_smoke.yaml")
    config = type(config)(**{**config.__dict__, "output_dir": tmp_path})
    result = BenchmarkRunner(config).run()
    assert len(result["completed_methods"]) == 8
    assert result["failed_methods"] == []
    for name in result["completed_methods"]:
        method_dir = tmp_path / config.run_name / name
        metrics = json.loads((method_dir / "metrics.json").read_text())
        manifest = json.loads((method_dir / "method_manifest.json").read_text())
        assert set(metrics) == set(PRIMARY_METRICS)
        assert manifest["is_publishable_benchmark"] is False
