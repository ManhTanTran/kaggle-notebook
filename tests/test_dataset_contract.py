from chunkbench.data.synthetic import SyntheticDatasetAdapter
from chunkbench.data.validation import validate_dataset
from chunkbench.registry.datasets import DATASET_REGISTRY, build_dataset


def test_synthetic_contract():
    bundle = SyntheticDatasetAdapter().load()
    validate_dataset(bundle)
    assert len(bundle.documents) == 3
    assert len(bundle.queries) == 2


def test_dataset_adapter_extension_does_not_touch_evaluator():
    class AuditAdapter(SyntheticDatasetAdapter):
        pass

    DATASET_REGISTRY["audit_dataset"] = lambda config: AuditAdapter(**config)
    try:
        bundle = build_dataset("audit_dataset").load()
        validate_dataset(bundle)
        assert bundle.metadata["name"] == "synthetic"
    finally:
        DATASET_REGISTRY.pop("audit_dataset")
