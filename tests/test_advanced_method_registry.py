from chunkbench.registry.methods import CORE_METHODS, build_chunker


def test_all_advanced_registry_entries_build_without_placeholders():
    configs = {
        "semantic_breakpoint": {"backend_type": "mock"},
        "semantic_single_linkage_paper_exact": {"backend_type": "mock"},
        "meta_ppl_raw": {"backend_type": "mock"},
        "meta_ppl_dynamic_512": {"backend_type": "mock"},
        "pic_paper_reimplementation": {"backend_type": "mock"},
        "pic_reimplementation_capped_512": {"backend_type": "mock"},
        "late_fixed_256": {"representation": {"backend_type": "mock"}},
        "late_fixed_512": {"representation": {"backend_type": "mock"}},
    }
    assert len(CORE_METHODS) == 12
    assert all(build_chunker(name, config).chunk for name, config in configs.items())
