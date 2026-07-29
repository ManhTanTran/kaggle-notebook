"""Domain-specific exceptions."""


class ChunkBenchError(Exception):
    """Base exception for benchmark failures."""


class ConfigurationError(ChunkBenchError):
    """Raised for invalid or incomplete configuration."""


class ContractError(ChunkBenchError):
    """Raised when data violates a public contract."""


class OptionalDependencyError(ChunkBenchError):
    """Raised when an explicitly selected backend is unavailable."""


class MissingOptionalDependencyError(OptionalDependencyError):
    """Explain the extra required by a selected model-backed implementation."""

    def __init__(self, method: str, extra: str) -> None:
        super().__init__(
            f"{method} requires optional dependencies; install with "
            f'pip install -e ".[{extra}]"'
        )
