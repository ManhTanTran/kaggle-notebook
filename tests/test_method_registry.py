from chunkbench.registry.methods import CORE_METHODS, METHOD_REGISTRY, build_chunker


def test_registry_has_exact_core_names():
    assert CORE_METHODS == list(METHOD_REGISTRY)
    assert len(CORE_METHODS) == 12
    assert build_chunker("fixed_256", {}).chunk
