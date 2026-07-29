# Experiment protocol

YAML is the source of truth. Single-dataset configs remain supported, while
`datasets:` expands a matrix in deterministic dataset-then-method order. A run ID
contains dataset, split, method, and a SHA-256-derived config hash. Environment
fingerprints and Git commit are persisted when available.

Resume skips only directories containing every required artifact and exactly 23
metric names. Invalid or incomplete directories rerun. With `fail_fast: false`,
one failure cannot remove completed artifacts. `completed_runs.json` and
`failed_runs.json` record the final state; failed advanced methods retain
the original exception and error type. Advanced mock results have
`is_publishable_benchmark: false` and must be filtered from publishable tables.

Use fixture smoke before full data. Full-data runs require authorized local
dataset copies and must not be described as verified until executed.
