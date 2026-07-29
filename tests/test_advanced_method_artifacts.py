import json
from pathlib import Path

from chunkbench.config.loader import load_config
from chunkbench.pipeline.runner import BenchmarkRunner


def test_advanced_artifact_manifest_marks_mock_non_publishable(tmp_path: Path):
    config = load_config("configs/experiments/advanced_methods_mock_smoke.yaml")
    config = type(config)(
        **{**config.__dict__, "output_dir": tmp_path, "methods": config.methods[:1]}
    )
    result = BenchmarkRunner(config).run()
    assert result["failed_methods"] == []
    manifest = json.loads(
        (
            tmp_path / config.run_name / "semantic_breakpoint" / "method_manifest.json"
        ).read_text()
    )
    assert manifest["is_mock_backend"] is True
    assert manifest["is_publishable_benchmark"] is False
    assert manifest["config_hash"]
