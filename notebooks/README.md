# Notebook workflow

Run notebooks from the repository root after `pip install -e ".[dev]"`.
They select YAML configs and call `chunkbench`; implementation stays in `src/`.

The Phase 1 HHR orchestration notebook is under `notebooks/hhr_dapr/` and calls
the separate `hhr_dapr` package from `src/hhr_dapr`. Install `.[hhr]` for its
synthetic smoke run or `.[hhr-dense]` for DRAGON+/FAISS runs.
