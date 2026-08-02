from __future__ import annotations

import pytest

from hhr_dapr.protocol import ProtocolEvent, validate_protocol


def test_ms_marco_dev_tuning_is_allowed():
    validate_protocol([ProtocolEvent("tune", "ms_marco", "dev", True)])


def test_test_labels_cannot_be_used_during_tuning():
    with pytest.raises(ValueError, match="selection is limited|test labels"):
        validate_protocol([ProtocolEvent("tune", "ms_marco", "test", True)])


def test_zero_shot_and_nq_hard_labels_cannot_select_parameters():
    with pytest.raises(ValueError, match="selection is limited"):
        validate_protocol([ProtocolEvent("select", "genomics", "test", True)])
    with pytest.raises(ValueError, match="NQ-hard|selection is limited"):
        validate_protocol([ProtocolEvent("tune", "nq_hard", "test", True)])
