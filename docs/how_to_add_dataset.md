# Add a dataset

1. Implement `DatasetAdapter.load` using the native, documented raw schema.
2. Preserve source evidence; never infer evidence from answer text.
3. Populate standard bundle metadata and evidence locators.
4. Register with `register_dataset(name, factory)`.
5. Add portable fixture, YAML, contract/mapping tests, and dataset documentation.
6. Verify 100% fixture mapping and record full-data verification separately.

Evaluator, `BenchmarkRunner`, and `ExperimentMatrixRunner` require no changes.
