"""Evidence coverage validation."""

from chunkbench.common.exceptions import ContractError
from chunkbench.common.types import Evidence, EvidenceCoverage


def validate_coverage(rows: list[EvidenceCoverage], evidence: list[Evidence]) -> None:
    """Ensure covered token identifiers stay inside evidence boundaries."""
    counts = {item.evidence_id: item.token_count for item in evidence}
    for row in rows:
        if row.evidence_id not in counts:
            raise ContractError(f"Unknown evidence id {row.evidence_id}")
        if any(
            index < 0 or index >= counts[row.evidence_id]
            for index in row.covered_token_ids
        ):
            raise ContractError(f"Coverage out of range for {row.evidence_id}")
