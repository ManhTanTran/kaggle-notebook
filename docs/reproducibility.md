# Reproducibility

Method configuration, experiment embedding/retrieval settings, and implementation
source pin form the method artifact config hash. `method_manifest.json` also
records fidelity, model/tokenizer fields, backend type, and the local Git commit
when available. This workspace is not currently a Git checkout, so that field is
null rather than invented.

Run offline validation with:

```bash
pip install -e ".[dev]"
pytest
ruff check .
python scripts/run_smoke_test.py
python -m chunkbench.cli --config configs/experiments/advanced_methods_mock_smoke.yaml
```

Use the individual real-family smoke profiles only after installing the matching
optional dependency and placing/downloading the configured model through an
authorized workflow. A missing dependency or model is recorded as a failure, not
a completed benchmark.
