# Architecture

Dependencies point inward: native parsers and adapters produce canonical types;
registries construct adapters, chunkers, embedders, and retrievers; runners
orchestrate them; evaluators consume only `DatasetBundle`, retrieval hits, chunks,
and evidence coverage. No dataset-specific branch exists in either evaluator or
runner.

`BenchmarkRunner` preserves single-dataset compatibility.
`ExperimentMatrixRunner` expands dataset × method, computes a deterministic
configuration hash, resumes valid completed runs, isolates failures, and writes
cross-dataset artifacts. `MethodPipelineSpec` keeps chunking and representation
separate: baseline strategies encode chunks independently, while late chunking
encodes each document before pooling token states into fixed spans. Every method
directory additionally has a provenance-bearing `method_manifest.json`.

Dataset parsing, evidence policy, normalization, validation diagnostics, and
matrix orchestration all live under `src/chunkbench`; notebooks stay thin.
