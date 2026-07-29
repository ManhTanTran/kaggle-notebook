from chunkbench.config.loader import load_config


def test_config_loads():
    config = load_config("configs/experiments/smoke_test.yaml")
    assert config.evaluation.k_values == (3, 5, 10)
    assert len(config.methods) == 4
