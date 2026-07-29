# Metrics

The suite has exactly 23 primary outputs (`PRIMARY_METRICS`): Hit, MRR, EvidenceRecallMacro,
EvidenceRecallMicro, DocumentRecall, EvidenceCoverage, and Redundancy at k=3,5,10,
plus EvidenceRecallMacro and EvidenceCoverage under a strict 2048-token budget.
Evidence coverage unions `covered_token_ids`, so repeated retrieval never counts a
token twice. Redundancy is the baseline mean pairwise Jaccard overlap of normalized
lexical token sets.

The evaluator is dataset-agnostic. All adapters use the same evidence-unit and
covered-token contracts under `qa_evidence_retrieval`; matrix aggregation never
changes metric definitions.
