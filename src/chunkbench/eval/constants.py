"""Canonical benchmark dimensions and output names."""

K_VALUES = [3, 5, 10]
TOKEN_BUDGETS = [2048]

RANKED_METRICS = [
    "Hit",
    "MRR",
    "EvidenceRecallMacro",
    "EvidenceRecallMicro",
    "DocumentRecall",
    "EvidenceCoverage",
    "Redundancy",
]
TOKEN_BUDGET_METRICS = ["EvidenceRecallMacro", "EvidenceCoverage"]

PRIMARY_METRICS = [
    *(f"{metric}@{k}" for metric in RANKED_METRICS for k in K_VALUES),
    *(
        f"{metric}@{budget}Tokens"
        for metric in TOKEN_BUDGET_METRICS
        for budget in TOKEN_BUDGETS
    ),
]

METRIC_NAMES = PRIMARY_METRICS

assert len(PRIMARY_METRICS) == 23
