"""Paper-faithful CompilerKV operators.

This module implements Algorithm 1 from the 2026 CompilerKV paper.  It is
deliberately independent from a particular Transformers attention class so the
math can be tested on CPU and reused by model-specific integrations.

Tensor conventions
------------------
``attention`` has shape ``[layers, batch, heads, queries, tokens]`` and may
contain either the complete causal attention matrix or only the final
observation-window rows.  ``key_cache`` and ``value_cache`` have shape
``[layers, batch, kv_heads, tokens, head_dim]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch


BudgetSpec = Union[int, Sequence[int]]


@dataclass(frozen=True)
class CompilerKVConfig:
    """Runtime configuration matching the main paper setting."""

    observation_window: int = 64
    entropy_bins: int = 20
    ppl_bins: int = 4
    epsilon: float = 1e-8
    missing_ppl: str = "error"

    def __post_init__(self) -> None:
        if self.observation_window <= 0:
            raise ValueError("observation_window must be positive")
        if self.entropy_bins <= 0 or self.ppl_bins <= 0:
            raise ValueError("risk-bin counts must be positive")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.missing_ppl not in {"error", "median"}:
            raise ValueError("missing_ppl must be 'error' or 'median'")


def _interpolate_axis(array: np.ndarray, size: int, axis: int) -> np.ndarray:
    """Linearly map a compiled coordinate axis to a runtime coordinate axis."""

    current = array.shape[axis]
    if current == size:
        return array
    if size <= 0:
        raise ValueError("interpolated axis size must be positive")
    if current == 1:
        return np.repeat(array, size, axis=axis)

    moved = np.moveaxis(array, axis, 0)
    source_x = np.linspace(0.0, 1.0, current)
    target_x = np.linspace(0.0, 1.0, size)
    flat = moved.reshape(current, -1)
    mapped = np.empty((size, flat.shape[1]), dtype=np.float32)
    for column in range(flat.shape[1]):
        mapped[:, column] = np.interp(target_x, source_x, flat[:, column])
    mapped = mapped.reshape((size,) + moved.shape[1:])
    return np.moveaxis(mapped, 0, axis)


@dataclass
class CompiledTables:
    """Validated offline artifacts used by the O(1) runtime lookups.

    Supported shapes are ``[L,H]`` or ``[N_B,L,H]`` for head reliability and
    ``[L,N_ent,N_ppl]`` or ``[N_B,L,N_ent,N_ppl]`` for the risk gate.  The
    optional leading budget axis preserves the paper's ``B_l`` state while
    retaining compatibility with a single-budget compilation.
    """

    head_weights: np.ndarray
    gate_thresholds: np.ndarray
    entropy_edges: np.ndarray
    ppl_edges: np.ndarray
    budget_values: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.head_weights = np.asarray(self.head_weights, dtype=np.float32)
        self.gate_thresholds = np.asarray(self.gate_thresholds, dtype=np.float32)
        self.entropy_edges = np.asarray(self.entropy_edges, dtype=np.float32)
        self.ppl_edges = np.asarray(self.ppl_edges, dtype=np.float32)
        if self.budget_values is not None:
            self.budget_values = np.asarray(self.budget_values, dtype=np.int64)
        self.validate()

    @property
    def num_entropy_bins(self) -> int:
        return int(self.gate_thresholds.shape[-2])

    @property
    def num_ppl_bins(self) -> int:
        return int(self.gate_thresholds.shape[-1])

    @property
    def canonical_layers(self) -> int:
        return int(self.head_weights.shape[-2])

    def validate(self) -> None:
        if self.head_weights.ndim not in {2, 3}:
            raise ValueError("W_head must have shape [L,H] or [N_B,L,H]")
        if self.gate_thresholds.ndim not in {3, 4}:
            raise ValueError(
                "T_gate must have shape [L,N_ent,N_ppl] or [N_B,L,N_ent,N_ppl]"
            )
        if self.head_weights.shape[-2] != self.gate_thresholds.shape[-3]:
            raise ValueError("W_head and T_gate must use the same canonical layer count")
        if len(self.entropy_edges) != self.num_entropy_bins - 1:
            raise ValueError("entropy_edges must contain N_entropy_bins - 1 values")
        if len(self.ppl_edges) != self.num_ppl_bins - 1:
            raise ValueError("ppl_edges must contain N_ppl_bins - 1 values")
        if np.any(np.diff(self.entropy_edges) < 0) or np.any(np.diff(self.ppl_edges) < 0):
            raise ValueError("risk-bin edges must be monotone")
        if not np.isfinite(self.head_weights).all() or not np.isfinite(self.gate_thresholds).all():
            raise ValueError("compiled tables contain NaN or infinity")
        if np.any((self.head_weights < 0.8) | (self.head_weights > 1.5)):
            raise ValueError("W_head values must lie in the paper's [0.8, 1.5] action grid")
        if np.any((self.gate_thresholds < 0.8) | (self.gate_thresholds > 1.0)):
            raise ValueError("T_gate values must lie in the paper's [0.8, 1.0] action grid")

        has_budget_axis = self.head_weights.ndim == 3
        if has_budget_axis != (self.gate_thresholds.ndim == 4):
            raise ValueError("both compiled tables must either include or omit the budget axis")
        if has_budget_axis:
            if self.head_weights.shape[0] != self.gate_thresholds.shape[0]:
                raise ValueError("compiled tables use different budget-axis sizes")
            if self.budget_values is None:
                raise ValueError("budget_values are required for budget-conditioned tables")
            if len(self.budget_values) != self.head_weights.shape[0]:
                raise ValueError("budget_values length does not match the tables")
        elif self.budget_values is not None and len(self.budget_values) != 1:
            raise ValueError("single-budget tables accept at most one budget value")

    @classmethod
    def from_directory(cls, directory: Union[str, Path]) -> "CompiledTables":
        directory = Path(directory)
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing {manifest_path}. Compile real calibration records first; "
                "legacy simulated tables are intentionally not accepted."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        head_path = directory / manifest.get("head_table", "W_head.npy")
        gate_path = directory / manifest.get("gate_table", "T_gate.npy")
        if not head_path.exists() or not gate_path.exists():
            raise FileNotFoundError(
                f"Compiled tables not found: expected {head_path.name} and {gate_path.name}"
            )
        source = manifest.get("source", "")
        if source in {"simulated", "synthetic-paper-figure"}:
            raise ValueError("simulated tables cannot be used as compiled inference artifacts")
        return cls(
            head_weights=np.load(head_path),
            gate_thresholds=np.load(gate_path),
            entropy_edges=np.asarray(manifest["entropy_edges"], dtype=np.float32),
            ppl_edges=np.asarray(manifest["ppl_edges"], dtype=np.float32),
            budget_values=(
                np.asarray(manifest["budget_values"], dtype=np.int64)
                if manifest.get("budget_values") is not None
                else None
            ),
            metadata=manifest,
        )

    def save(self, directory: Union[str, Path]) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "W_head.npy", self.head_weights)
        np.save(directory / "T_gate.npy", self.gate_thresholds)
        manifest = dict(self.metadata)
        manifest.update(
            {
                "format_version": 2,
                "source": manifest.get("source", "offline-calibration"),
                "head_table": "W_head.npy",
                "gate_table": "T_gate.npy",
                "canonical_layers": self.canonical_layers,
                "num_heads": int(self.head_weights.shape[-1]),
                "entropy_edges": self.entropy_edges.tolist(),
                "ppl_edges": self.ppl_edges.tolist(),
                "budget_values": (
                    self.budget_values.tolist() if self.budget_values is not None else None
                ),
            }
        )
        (directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _budget_index(self, budget: int) -> Optional[int]:
        if self.head_weights.ndim == 2:
            return None
        assert self.budget_values is not None
        matches = np.where(self.budget_values == int(budget))[0]
        if len(matches):
            return int(matches[0])
        if self.metadata.get("allow_nearest_budget", False):
            return int(np.abs(self.budget_values - int(budget)).argmin())
        raise ValueError(
            f"budget {budget} was not compiled; available budgets are "
            f"{self.budget_values.tolist()}"
        )

    def head_for(self, layers: int, heads: int, budget: int) -> np.ndarray:
        table = self.head_weights
        budget_index = self._budget_index(budget)
        if budget_index is not None:
            table = table[budget_index]
        table = _interpolate_axis(table, layers, axis=0)
        if table.shape[1] != heads:
            if not self.metadata.get("allow_head_interpolation", False):
                raise ValueError(
                    f"W_head was compiled for {table.shape[1]} heads, but runtime has {heads}"
                )
            table = _interpolate_axis(table, heads, axis=1)
        return np.asarray(table, dtype=np.float32)

    def gate_for(self, layers: int, budget: int) -> np.ndarray:
        table = self.gate_thresholds
        budget_index = self._budget_index(budget)
        if budget_index is not None:
            table = table[budget_index]
        return np.asarray(_interpolate_axis(table, layers, axis=0), dtype=np.float32)


@dataclass
class CompressionOutput:
    """Ragged, exact result of the paper's elastic selection operator."""

    indices: List[List[torch.Tensor]]
    scores: torch.Tensor
    attention_mass: torch.Tensor
    attention_entropy: torch.Tensor
    local_perplexity: torch.Tensor
    thresholds: torch.Tensor
    key_cache: Optional[List[List[torch.Tensor]]] = None
    value_cache: Optional[List[List[torch.Tensor]]] = None

    def dense_cache_for_batch_one(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Return standard dense layer caches when the input batch size is one."""

        if self.key_cache is None or self.value_cache is None:
            raise ValueError("key/value caches were not supplied to the compressor")
        if any(len(layer) != 1 for layer in self.key_cache):
            raise ValueError("dense_cache_for_batch_one requires batch size 1")
        keys = [layer[0].unsqueeze(0) for layer in self.key_cache]
        values = [layer[0].unsqueeze(0) for layer in self.value_cache]
        return keys, values


class CompilerKVOperator:
    """Vectorized Stage 1/2/3 operators plus exact elastic cache gathering."""

    def __init__(self, tables: CompiledTables, config: Optional[CompilerKVConfig] = None):
        self.tables = tables
        self.config = config or CompilerKVConfig(
            entropy_bins=tables.num_entropy_bins,
            ppl_bins=tables.num_ppl_bins,
        )
        if self.config.entropy_bins != tables.num_entropy_bins:
            raise ValueError("config entropy_bins does not match T_gate")
        if self.config.ppl_bins != tables.num_ppl_bins:
            raise ValueError("config ppl_bins does not match T_gate")

    @staticmethod
    def _layered(tensor: torch.Tensor, expected_dims: int, name: str) -> torch.Tensor:
        if tensor.dim() == expected_dims - 1:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != expected_dims:
            raise ValueError(f"{name} must have {expected_dims - 1} or {expected_dims} dimensions")
        return tensor

    @staticmethod
    def _budgets(budgets: BudgetSpec, layers: int, tokens: int) -> List[int]:
        if isinstance(budgets, int):
            result = [budgets] * layers
        else:
            result = [int(value) for value in budgets]
        if len(result) != layers:
            raise ValueError(f"expected {layers} per-layer budgets, got {len(result)}")
        if any(value < 0 for value in result):
            raise ValueError("budgets cannot be negative")
        return [min(value, tokens) for value in result]

    def attention_mass(
        self,
        attention: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Eq. (4): scale-normalized observation-window attention mass ``alpha``."""

        attention = self._layered(attention, 5, "attention")
        layers, batch, _, queries, tokens = attention.shape
        window = min(self.config.observation_window, queries)
        observed = attention[..., -window:, :].float()

        if query_mask is not None:
            if query_mask.ndim != 2 or query_mask.shape[0] != batch or query_mask.shape[1] < queries:
                raise ValueError("query_mask must have shape [batch, >= attention queries]")
            observed_query_mask = query_mask[:, -window:].to(observed.device, observed.dtype)
            observed = observed * observed_query_mask[None, :, None, :, None]
            window_count = observed_query_mask.sum(dim=-1).clamp_min(1.0)
        else:
            window_count = torch.full(
                (batch,), float(window), device=observed.device, dtype=observed.dtype
            )

        mean_attention = observed.mean(dim=(0, 2))
        if token_mask is not None:
            if token_mask.shape != (batch, tokens):
                raise ValueError("token_mask must have shape [batch, tokens]")
            valid_tokens = token_mask.to(observed.device, observed.dtype)
            token_count = valid_tokens.sum(dim=-1).clamp_min(1.0)
        else:
            valid_tokens = None
            token_count = torch.full(
                (batch,), float(tokens), device=observed.device, dtype=observed.dtype
            )

        alpha = mean_attention.sum(dim=-2) * (token_count / window_count).unsqueeze(-1)
        if valid_tokens is not None:
            alpha = alpha * valid_tokens
        return alpha

    def stabilized_utility(
        self,
        attention: torch.Tensor,
        value_cache: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Eq. (4): ``u_t^(l,h) = alpha_t * rho_t^(l,h)``."""

        attention = self._layered(attention, 5, "attention")
        value_cache = self._layered(value_cache, 5, "value_cache")
        layers, batch, attention_heads, _, tokens = attention.shape
        if value_cache.shape[:2] != (layers, batch) or value_cache.shape[-2] != tokens:
            raise ValueError("attention and value_cache layer/batch/token dimensions differ")

        alpha = self.attention_mass(attention, token_mask=token_mask, query_mask=query_mask)
        norms = torch.linalg.vector_norm(value_cache.float(), ord=2, dim=-1)
        if token_mask is None:
            mean_norm = norms.mean(dim=-1, keepdim=True)
        else:
            valid = token_mask.to(norms.device, norms.dtype)[None, :, None, :]
            count = valid.sum(dim=-1, keepdim=True).clamp_min(1.0)
            mean_norm = (norms * valid).sum(dim=-1, keepdim=True) / count
        rho = norms / mean_norm.clamp_min(self.config.epsilon)

        value_heads = rho.shape[2]
        if value_heads != attention_heads:
            if attention_heads % value_heads != 0:
                raise ValueError("attention heads must be divisible by KV heads for GQA")
            rho = rho.repeat_interleave(attention_heads // value_heads, dim=2)
        utility = rho * alpha[None, :, None, :]
        return utility, alpha

    def risk_signals(
        self,
        alpha: torch.Tensor,
        token_logprobs: Optional[torch.Tensor] = None,
        local_ppl: Optional[Union[float, Sequence[float], torch.Tensor]] = None,
        token_mask: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Eq. (7): attention entropy and observation-window perplexity."""

        batch, tokens = alpha.shape
        distribution = alpha.float()
        if token_mask is not None:
            distribution = distribution * token_mask.to(distribution.device, distribution.dtype)
        distribution = distribution / distribution.sum(dim=-1, keepdim=True).clamp_min(
            self.config.epsilon
        )
        entropy = -(distribution * distribution.clamp_min(self.config.epsilon).log()).sum(dim=-1)

        if local_ppl is not None:
            ppl = torch.as_tensor(local_ppl, dtype=torch.float32, device=alpha.device)
            if ppl.ndim == 0:
                ppl = ppl.repeat(batch)
            if ppl.shape != (batch,):
                raise ValueError("local_ppl must be scalar or have shape [batch]")
        elif token_logprobs is not None:
            if token_logprobs.ndim != 2 or token_logprobs.shape[0] != batch:
                raise ValueError("token_logprobs must have shape [batch, queries]")
            queries = token_logprobs.shape[-1]
            window = min(self.config.observation_window, queries)
            observed = token_logprobs[:, -window:].float()
            if query_mask is not None:
                if query_mask.shape != token_logprobs.shape:
                    raise ValueError("query_mask must match token_logprobs when computing PPL")
                mask = query_mask[:, -window:].to(observed.device, observed.dtype)
                mean_logprob = (observed * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
            else:
                mean_logprob = observed.mean(dim=-1)
            ppl = torch.exp(-mean_logprob)
        elif self.config.missing_ppl == "median":
            if len(self.tables.ppl_edges):
                median = float(np.median(self.tables.ppl_edges))
            else:
                median = 1.0
            ppl = torch.full((batch,), median, device=alpha.device, dtype=torch.float32)
        else:
            raise ValueError(
                "CompilerKV's semantic-risk gate requires token_logprobs or local_ppl; "
                "no silent default is used."
            )
        if torch.any(~torch.isfinite(ppl)) or torch.any(ppl <= 0):
            raise ValueError("local perplexity must be finite and positive")
        return entropy, ppl

    def head_aware_scores(self, utility: torch.Tensor, budgets: List[int]) -> torch.Tensor:
        """Eqs. (5)-(6): reliability-weighted, tokenwise max-pooling over heads."""

        layers, _, heads, _ = utility.shape
        if len(set(budgets)) == 1:
            head = self.tables.head_for(layers, heads, budgets[0])
            weights = torch.as_tensor(head, device=utility.device, dtype=utility.dtype)
            return (utility * weights[:, None, :, None]).amax(dim=2)

        scores = []
        for layer, budget in enumerate(budgets):
            all_layers = self.tables.head_for(layers, heads, budget)
            weights = torch.as_tensor(
                all_layers[layer], device=utility.device, dtype=utility.dtype
            )
            scores.append((utility[layer] * weights[None, :, None]).amax(dim=1))
        return torch.stack(scores, dim=0)

    def thresholds(
        self,
        layers: int,
        budgets: List[int],
        entropy: torch.Tensor,
        ppl: torch.Tensor,
    ) -> torch.Tensor:
        """Look up ``T_gate[l,b_ent,b_ppl,B_l]`` for each prompt and layer."""

        entropy_bins = np.searchsorted(
            self.tables.entropy_edges, entropy.detach().cpu().numpy(), side="right"
        )
        ppl_bins = np.searchsorted(
            self.tables.ppl_edges, ppl.detach().cpu().numpy(), side="right"
        )
        output = torch.empty(
            (layers, entropy.shape[0]), device=entropy.device, dtype=torch.float32
        )
        for layer, budget in enumerate(budgets):
            gate = self.tables.gate_for(layers, budget)
            values = gate[layer, entropy_bins, ppl_bins]
            output[layer] = torch.as_tensor(values, device=entropy.device, dtype=torch.float32)
        return output

    @staticmethod
    def elastic_select(
        scores: torch.Tensor,
        thresholds: torch.Tensor,
        budgets: List[int],
        token_mask: Optional[torch.Tensor] = None,
    ) -> List[List[torch.Tensor]]:
        """Eq. (8): retain all under-budget candidates; clamp only overflow."""

        layers, batch, tokens = scores.shape
        selected: List[List[torch.Tensor]] = []
        for layer in range(layers):
            layer_indices: List[torch.Tensor] = []
            for sample in range(batch):
                valid = torch.ones(tokens, dtype=torch.bool, device=scores.device)
                if token_mask is not None:
                    valid &= token_mask[sample].to(device=scores.device, dtype=torch.bool)
                candidate = valid & (scores[layer, sample] >= thresholds[layer, sample])
                indices = torch.where(candidate)[0]
                budget = budgets[layer]
                if budget == 0:
                    indices = indices[:0]
                elif indices.numel() > budget:
                    candidate_scores = scores[layer, sample, indices]
                    local = torch.topk(candidate_scores, k=budget, largest=True, sorted=False).indices
                    indices = indices[local]
                layer_indices.append(indices.sort().values)
            selected.append(layer_indices)
        return selected

    @staticmethod
    def _gather_cache(
        cache: torch.Tensor, indices: List[List[torch.Tensor]]
    ) -> List[List[torch.Tensor]]:
        gathered: List[List[torch.Tensor]] = []
        for layer, layer_indices in enumerate(indices):
            gathered.append(
                [cache[layer, sample].index_select(1, index) for sample, index in enumerate(layer_indices)]
            )
        return gathered

    def compress(
        self,
        attention: torch.Tensor,
        value_cache: torch.Tensor,
        budgets: BudgetSpec,
        key_cache: Optional[torch.Tensor] = None,
        token_logprobs: Optional[torch.Tensor] = None,
        local_ppl: Optional[Union[float, Sequence[float], torch.Tensor]] = None,
        token_mask: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
    ) -> CompressionOutput:
        """Run Algorithm 1 and optionally gather the supplied KV cache."""

        attention = self._layered(attention, 5, "attention")
        value_cache = self._layered(value_cache, 5, "value_cache")
        layers, _, _, _, tokens = attention.shape
        per_layer_budgets = self._budgets(budgets, layers, tokens)

        utility, alpha = self.stabilized_utility(
            attention,
            value_cache,
            token_mask=token_mask,
            query_mask=query_mask,
        )
        scores = self.head_aware_scores(utility, per_layer_budgets)
        entropy, ppl = self.risk_signals(
            alpha,
            token_logprobs=token_logprobs,
            local_ppl=local_ppl,
            token_mask=token_mask,
            query_mask=query_mask,
        )
        thresholds = self.thresholds(layers, per_layer_budgets, entropy, ppl)
        indices = self.elastic_select(
            scores, thresholds, per_layer_budgets, token_mask=token_mask
        )

        layered_keys = None
        if key_cache is not None:
            key_cache = self._layered(key_cache, 5, "key_cache")
            if key_cache.shape[:2] != value_cache.shape[:2] or key_cache.shape[-2:] != value_cache.shape[-2:]:
                raise ValueError("key_cache and value_cache dimensions differ")
            layered_keys = self._gather_cache(key_cache, indices)
        layered_values = self._gather_cache(value_cache, indices)
        return CompressionOutput(
            indices=indices,
            scores=scores,
            attention_mass=alpha,
            attention_entropy=entropy,
            local_perplexity=ppl,
            thresholds=thresholds,
            key_cache=layered_keys,
            value_cache=layered_values,
        )


__all__ = [
    "CompiledTables",
    "CompilerKVConfig",
    "CompilerKVOperator",
    "CompressionOutput",
]
