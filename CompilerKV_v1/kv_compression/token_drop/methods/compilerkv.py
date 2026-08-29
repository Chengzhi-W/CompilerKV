"""Layer-local adapter for the legacy Transformers monkeypatch.

The canonical cross-layer operator lives in :mod:`kv_compression.core`.  This
adapter keeps the old attention monkeypatch usable while fixing its selection
math: scaled observation-window mass, per-head value normalization, reliability
weighted max-pooling, absolute risk thresholds, and elastic under-retention.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from ...core import CompiledTables


class CompilerKVCluster:
    """Compatibility wrapper used by the model-specific attention forwards."""

    def __init__(
        self,
        num_hidden_layers: int = 32,
        num_heads: int = 32,
        window_size: int = 64,
        max_capacity_prompt: int = 512,
        kernel_size: int = 7,
        pooling: str = "none",
        layer_idx: Optional[int] = None,
        tables_dir: str = "tables/compiled",
        radio_max: float = 1.0,
        radio_min: float = 0.0,
        prompt_ppl: Optional[float] = None,
        missing_ppl: str = "median",
    ) -> None:
        del kernel_size, pooling, radio_max, radio_min
        if layer_idx is None:
            raise ValueError("CompilerKV attention integration requires layer_idx")
        if max_capacity_prompt < 0:
            raise ValueError("max_capacity_prompt cannot be negative")
        if missing_ppl not in {"error", "median"}:
            raise ValueError("missing_ppl must be 'error' or 'median'")
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        self.layer_idx = layer_idx
        self.tables = CompiledTables.from_directory(Path(tables_dir))
        self.prompt_ppl = prompt_ppl
        self.missing_ppl = missing_ppl
        self._warned_missing_ppl = False

    def _ppl(self) -> float:
        if self.prompt_ppl is not None:
            return float(self.prompt_ppl)
        if self.missing_ppl == "error":
            raise ValueError(
                "set attention.config.prompt_ppl before the CompilerKV prefill, "
                "or use the canonical API with token_logprobs"
            )
        if not self._warned_missing_ppl:
            warnings.warn(
                "prompt_ppl was not supplied to the legacy monkeypatch; using the "
                "compiled median PPL bin. This is not the paper's two-signal gate.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned_missing_ppl = True
        if len(self.tables.ppl_edges):
            return float(np.median(self.tables.ppl_edges))
        return 1.0

    def _attention(
        self,
        key_states: torch.Tensor,
        query_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        _, _, tokens, head_dim = key_states.shape
        window = min(self.window_size, query_states.shape[-2])
        query = query_states[..., -window:, :]
        logits = torch.matmul(query, key_states.transpose(2, 3)) / math.sqrt(head_dim)

        query_positions = torch.arange(tokens - window, tokens, device=logits.device)
        key_positions = torch.arange(tokens, device=logits.device)
        future = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
        logits = logits.masked_fill(future[None, None], torch.finfo(logits.dtype).min)
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                valid = attention_mask[:, None, None, :].to(dtype=torch.bool)
                logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
            elif attention_mask.ndim == 4:
                logits = logits + attention_mask[..., -window:, :tokens]
        return F.softmax(logits, dim=-1, dtype=torch.float32)

    def _scores(
        self,
        attention: torch.Tensor,
        value_states: torch.Tensor,
        budget: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, heads, window, tokens = attention.shape
        alpha = attention.mean(dim=1).sum(dim=1) * (tokens / float(window))
        norms = torch.linalg.vector_norm(value_states.float(), ord=2, dim=-1)
        rho = norms / norms.mean(dim=-1, keepdim=True).clamp_min(1e-8)
        if rho.shape[1] != heads:
            if heads % rho.shape[1] != 0:
                raise ValueError("attention heads must be divisible by KV heads")
            rho = rho.repeat_interleave(heads // rho.shape[1], dim=1)
        utility = alpha[:, None, :] * rho
        weights = self.tables.head_for(self.num_hidden_layers, heads, budget)[self.layer_idx]
        weights_t = torch.as_tensor(weights, device=utility.device, dtype=utility.dtype)
        score = (utility * weights_t[None, :, None]).amax(dim=1)
        return score, alpha

    def _threshold(self, alpha: torch.Tensor, budget: int) -> torch.Tensor:
        distribution = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        entropy = -(distribution * distribution.clamp_min(1e-8).log()).sum(dim=-1)
        entropy_bin = np.searchsorted(
            self.tables.entropy_edges,
            entropy.detach().cpu().numpy(),
            side="right",
        )
        ppl_bin = int(np.searchsorted(self.tables.ppl_edges, self._ppl(), side="right"))
        gate = self.tables.gate_for(self.num_hidden_layers, budget)[self.layer_idx]
        values = gate[entropy_bin, ppl_bin]
        return torch.as_tensor(values, device=alpha.device, dtype=torch.float32)

    @staticmethod
    def _select(score: torch.Tensor, threshold: torch.Tensor, budget: int) -> List[torch.Tensor]:
        result: List[torch.Tensor] = []
        for sample in range(score.shape[0]):
            indices = torch.where(score[sample] >= threshold[sample])[0]
            if budget == 0:
                indices = indices[:0]
            elif indices.numel() > budget:
                top = torch.topk(score[sample, indices], k=budget, sorted=False).indices
                indices = indices[top]
            result.append(indices.sort().values)
        return result

    def budget_compute_per_layer(
        self,
        key_states: torch.Tensor,
        query_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """Compress one layer without the legacy cross-layer budget heuristic."""

        batch, heads, tokens, head_dim = key_states.shape
        budget = min(self.max_capacity_prompt, tokens)
        if batch != 1:
            raise ValueError(
                "the legacy monkeypatch supports batch size 1 because elastic retention "
                "produces ragged caches; use kv_compression.api for batched inputs"
            )
        attention = self._attention(key_states, query_states, attention_mask)
        score, alpha = self._scores(attention, value_states, budget)
        threshold = self._threshold(alpha, budget)
        selected = self._select(score, threshold, budget)[0]

        gather = selected[None, None, :, None].expand(batch, heads, -1, head_dim)
        key_out = key_states.gather(dim=2, index=gather)
        value_out = value_states.gather(dim=2, index=gather)
        return score[:, None, :], gather, key_out, value_out

    def update_and_reset_budget(
        self,
        budget_k_cache: List[torch.Tensor],
        budget_v_cache: List[torch.Tensor],
        total_gather_indices: List[torch.Tensor],
        advance_attn_cache: List[torch.Tensor],
    ):
        """Compatibility no-op: CompilerKV uses fixed per-layer budgets."""

        del advance_attn_cache
        return budget_k_cache, budget_v_cache, total_gather_indices


def init_compilerkv(self, num_hidden_layers: int) -> None:
    """Attach a configured CompilerKVCluster to a Transformers attention layer."""

    if hasattr(self, "kv_cluster"):
        return
    defaults = {
        "window_size": 64,
        "max_capacity_prompt": 512,
        "tables_dir": "tables/compiled",
        "prompt_ppl": None,
        "missing_ppl": "median",
    }
    for name, value in defaults.items():
        if not hasattr(self.config, name):
            setattr(self.config, name, value)
    self.kv_cluster = CompilerKVCluster(
        num_hidden_layers=num_hidden_layers,
        num_heads=getattr(self.config, "num_attention_heads", 32),
        window_size=self.config.window_size,
        max_capacity_prompt=self.config.max_capacity_prompt,
        layer_idx=self.layer_idx,
        tables_dir=self.config.tables_dir,
        prompt_ppl=self.config.prompt_ppl,
        missing_ppl=self.config.missing_ppl,
    )


__all__ = ["CompilerKVCluster", "init_compilerkv"]
