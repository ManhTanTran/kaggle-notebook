"""Guards for the DAPR zero-shot selection/evaluation contract."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

ZERO_SHOT_DATASETS = {"natural_questions", "miracl_en", "genomics", "conditional_qa"}


@dataclass(frozen=True)
class ProtocolEvent:
    purpose: str
    dataset: str
    split: str
    uses_labels: bool


def validate_protocol(events: Iterable[ProtocolEvent]) -> None:
    errors: list[str] = []
    for event in events:
        if event.purpose in {"tune", "select"}:
            if event.dataset != "ms_marco" or event.split not in {"train", "dev"}:
                errors.append(
                    f"{event.purpose} used {event.dataset}/{event.split}; selection "
                    "is limited to MS MARCO train/dev"
                )
        if (
            event.purpose in {"tune", "select"}
            and event.split == "test"
            and event.uses_labels
        ):
            errors.append("test labels were used during tuning/selection")
        if event.dataset == "nq_hard" and event.purpose in {"tune", "select"}:
            errors.append("NQ-hard is diagnostic-only")
    if errors:
        raise ValueError("DAPR zero-shot protocol violation:\n- " + "\n- ".join(errors))
