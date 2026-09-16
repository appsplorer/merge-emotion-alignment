from __future__ import annotations
from typing import List


def chunk_token_ids(token_ids: List[int], chunk_size: int, max_chunks: int) -> List[List[int]]:
    if chunk_size <= 0 or max_chunks <= 0:
        raise ValueError("chunk_size and max_chunks must be positive")
    return [token_ids[i:i + chunk_size] for i in range(0, min(len(token_ids), chunk_size * max_chunks), chunk_size)] or [[]]
