# Meta-PPL

`BasePerplexityScorer` supports an offline deterministic scorer and an optional
causal Transformers scorer. The latter loads once, uses eval/inference mode, and
masks the left context in loss computation. `meta_ppl_dynamic_512` is a
transparent project cap, not a paper-exact dynamic protocol.
