# Advanced-method provenance

The machine-readable source of truth is `configs/methods/provenance.yaml`.

| Family | Source | Repository / pin | Fidelity |
| --- | --- | --- | --- |
| Semantic breakpoint / single linkage | Qu, Tu, Bao, *Is Semantic Chunking Worth the Computational Cost?*, NAACL Findings 2025 | No official repository located | idea-based / paper reimplementation unverified |
| Meta-PPL | Zhao et al., *Meta-Chunking*, arXiv:2410.12788 | `IAAR-Shanghai/Meta-Chunking@2dbf487a75e44a378eef1a35909bdbe871c396a7`, Apache-2.0 | raw unverified; 512 idea-based |
| PIC | Wang et al., *Document Segmentation Matters for RAG*, ACL Findings 2025 | No official repository located | paper reimplementation unverified / capped idea-based |
| Late chunking | Günther et al., *Late Chunking*, arXiv:2409.04701 | `jina-ai/late-chunking@1d3bb02bf091becd0771455e4e7959463935e26c`, Apache-2.0 | paper reimplementation unverified |

No source code is copied from the cited repositories. Each run writes its
fidelity and source pin to `method_manifest.json`; mock runs are not publishable.
