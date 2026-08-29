"""Public API for paper-faithful prefill-only CompilerKV compression."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import torch

from .core import (
    CompiledTables,
    CompilerKVConfig,
    CompilerKVOperator,
    CompressionOutput,
)


class CompilerKVCompressor:
    """Load compiled tables once and apply Algorithm 1 to prefill tensors.

    Unlike the legacy implementation, this class does not silently substitute
    fabricated tables or a constant PPL value. The semantic-risk coordinate is
    part of the paper's policy and must be supplied as token log-probabilities or
    as an already computed local perplexity.
    """

    def __init__(
        self,
        tables: Union[CompiledTables, str, Path],
        observation_window: int = 64,
        missing_ppl: str = "error",
    ) -> None:
        if not isinstance(tables, CompiledTables):
            tables = CompiledTables.from_directory(tables)
        config = CompilerKVConfig(
            observation_window=observation_window,
            entropy_bins=tables.num_entropy_bins,
            ppl_bins=tables.num_ppl_bins,
            missing_ppl=missing_ppl,
        )
        self.tables = tables
        self.operator = CompilerKVOperator(tables, config)

    def compress(
        self,
        attention: torch.Tensor,
        value_cache: torch.Tensor,
        budgets: Union[int, Sequence[int]],
        *,
        key_cache: Optional[torch.Tensor] = None,
        token_logprobs: Optional[torch.Tensor] = None,
        local_ppl: Optional[Union[float, Sequence[float], torch.Tensor]] = None,
        token_mask: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
    ) -> CompressionOutput:
        return self.operator.compress(
            attention=attention,
            value_cache=value_cache,
            budgets=budgets,
            key_cache=key_cache,
            token_logprobs=token_logprobs,
            local_ppl=local_ppl,
            token_mask=token_mask,
            query_mask=query_mask,
        )


def compress_kv_prefill_only(
    attention: torch.Tensor,
    value_cache: torch.Tensor,
    budgets: Union[int, Sequence[int]],
    *,
    key_cache: Optional[torch.Tensor] = None,
    token_logprobs: Optional[torch.Tensor] = None,
    local_ppl: Optional[Union[float, Sequence[float], torch.Tensor]] = None,
    tables_dir: Union[str, Path],
    observation_window: int = 64,
    token_mask: Optional[torch.Tensor] = None,
    query_mask: Optional[torch.Tensor] = None,
) -> CompressionOutput:
    """Run CompilerKV once at the end of prefill.

    Attention may contain the full prefill matrix or only its final observation
    rows. The returned cache is intentionally ragged because Eq. (8) permits
    under-retention; use ``dense_cache_for_batch_one`` for batch-1 generation.
    """

    compressor = CompilerKVCompressor(
        tables=tables_dir,
        observation_window=observation_window,
    )
    return compressor.compress(
        attention=attention,
        value_cache=value_cache,
        budgets=budgets,
        key_cache=key_cache,
        token_logprobs=token_logprobs,
        local_ppl=local_ppl,
        token_mask=token_mask,
        query_mask=query_mask,
    )


def dense_batch_one(
    output: CompressionOutput,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Materialize an exact dense cache when batch size is one."""

    return output.dense_cache_for_batch_one()


KVCompressor = CompilerKVCompressor

__all__ = [
    "CompiledTables",
    "CompilerKVCompressor",
    "CompilerKVConfig",
    "CompilerKVOperator",
    "CompressionOutput",
    "KVCompressor",
    "compress_kv_prefill_only",
    "dense_batch_one",
]
