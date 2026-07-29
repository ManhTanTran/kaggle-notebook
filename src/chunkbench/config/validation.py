"""Configuration contract validation."""

from chunkbench.common.exceptions import ConfigurationError
from chunkbench.config.schema import ExperimentConfig
from chunkbench.eval.constants import K_VALUES, TOKEN_BUDGETS


def validate_config(config: ExperimentConfig) -> None:
    """Raise a descriptive error when an experiment config is invalid."""
    if not config.run_name.strip():
        raise ConfigurationError("run_name must not be empty")
    if not config.methods:
        raise ConfigurationError("at least one method is required")
    if not config.datasets:
        raise ConfigurationError("at least one dataset is required")
    if tuple(config.evaluation.k_values) != tuple(K_VALUES):
        raise ConfigurationError(f"k_values must be {K_VALUES}")
    if tuple(config.evaluation.token_budgets) != tuple(TOKEN_BUDGETS):
        raise ConfigurationError(f"token_budgets must be {TOKEN_BUDGETS}")
    if config.evaluation.duplicate_scope not in {"within_document", "corpus"}:
        raise ConfigurationError("duplicate_scope must be within_document or corpus")
