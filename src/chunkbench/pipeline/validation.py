"""Pipeline output validation."""

from chunkbench.common.exceptions import ContractError
from chunkbench.common.types import Chunk


def validate_chunks(chunks: list[Chunk]) -> None:
    """Validate required chunk fields and uniqueness."""
    ids = [chunk.chunk_id for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise ContractError("Chunk identifiers must be unique")
    if any(chunk.token_count != len(chunk.text.split()) for chunk in chunks):
        raise ContractError("Chunk token_count must match whitespace token count")
    if any(not chunk.text.strip() for chunk in chunks):
        raise ContractError("Empty chunks are not allowed")
