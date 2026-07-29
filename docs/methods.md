# Methods

Four baselines are implemented: fixed and sentence-aware windows at 256/512
tokens. Eight advanced methods are implemented through reusable segment,
boundary, post-processing, scoring, and representation abstractions.

`semantic_breakpoint` is an idea-based adjacent-sentence distance implementation.
`semantic_single_linkage_paper_exact` retains its historical name but its manifest
is `paper_reimplementation_unverified`: it follows Qu et al.'s weighted global
single-linkage formula and intentionally permits non-contiguous clusters.

`meta_ppl_raw` follows PPL local-minimum boundary detection; `dynamic_512` adds a
project-level capped policy. PIC is Pseudo-Instruction for document Chunking
(Wang et al., ACL Findings 2025): it groups adjacent sentences by whether their
summary similarity is above/below the document mean. The mock summary is not
GPT-4o-mini, so both PIC variants remain unverified.

Late methods use fixed boundaries but vectors are generated only after whole
document contextual token embeddings; pooling excludes special and padding
tokens. See `docs/methods/provenance.md` for sources and licenses.
