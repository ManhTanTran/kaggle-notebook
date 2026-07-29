from chunkbench.registry.methods import build_method_pipeline_spec


def test_late_spec_selects_late_representation_and_honest_fidelity():
    spec = build_method_pipeline_spec(
        "late_fixed_256", {"representation": {"backend_type": "mock"}}
    )
    assert spec.representation_strategy == "late_document_embedding"
    assert spec.implementation_fidelity == "paper_reimplementation_unverified"
    assert spec.chunker.chunk_size == 256
