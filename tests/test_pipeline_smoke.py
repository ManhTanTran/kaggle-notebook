from pathlib import Path

from chunkbench.config.loader import load_config
from chunkbench.pipeline.runner import BenchmarkRunner


def test_full_smoke_pipeline(tmp_path: Path):
    config = load_config("configs/experiments/smoke_test.yaml")
    config = type(config)(**{**config.__dict__, "output_dir": tmp_path})
    result = BenchmarkRunner(config, Path.cwd()).run()
    assert len(result["completed_methods"]) == 4
    assert (tmp_path / "smoke_test" / "benchmark_metrics.csv").exists()
