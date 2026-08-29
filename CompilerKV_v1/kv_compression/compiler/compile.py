"""Compile W_head and T_gate from held-out calibration experience."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from ..core import CompiledTables
from .cql import CQLConfig, fit_horizon_one_cql
from .records import CalibrationRecord, load_records, require_coupled_rewards


HEAD_ACTIONS = np.linspace(0.8, 1.5, 29, dtype=np.float32)
GATE_ACTIONS = np.linspace(0.8, 1.0, 21, dtype=np.float32)


def _quantile_edges(values: Sequence[float], bins: int) -> np.ndarray:
    if bins <= 0:
        raise ValueError("bins must be positive")
    if bins == 1:
        return np.empty((0,), dtype=np.float32)
    return np.quantile(
        np.asarray(values, dtype=np.float64),
        np.arange(1, bins, dtype=np.float64) / bins,
    ).astype(np.float32)


def _action_id(value: float, grid: np.ndarray, kind: str) -> int:
    index = int(np.abs(grid - value).argmin())
    tolerance = float(grid[1] - grid[0]) / 4 if len(grid) > 1 else 1e-6
    if abs(float(grid[index]) - value) > tolerance:
        raise ValueError(f"{kind} action {value} is not on the configured action grid")
    return index


def _fit_table(
    states: List[int],
    actions: List[int],
    rewards: List[float],
    *,
    num_states: int,
    num_actions: int,
    config: CQLConfig,
    device: Optional[torch.device],
) -> Tuple[np.ndarray, np.ndarray]:
    state_tensor = torch.tensor(states, dtype=torch.long)
    action_tensor = torch.tensor(actions, dtype=torch.long)
    reward_tensor = torch.tensor(rewards, dtype=torch.float32)
    q_values = fit_horizon_one_cql(
        state_tensor,
        action_tensor,
        reward_tensor,
        num_states=num_states,
        num_actions=num_actions,
        config=config,
        device=device,
    ).numpy()
    support = np.bincount(np.asarray(states), minlength=num_states)
    return q_values, support


def compile_calibration_records(
    records: Union[Iterable[CalibrationRecord], str, Path],
    output_dir: Union[str, Path],
    *,
    num_layers: int,
    num_heads: int,
    entropy_bins: int = 20,
    ppl_bins: int = 4,
    cql_config: Optional[CQLConfig] = None,
    require_paired_policy: bool = True,
    device: Optional[Union[str, torch.device]] = None,
    input_sha256: Optional[str] = None,
) -> CompiledTables:
    """Compile both factorized tables with coupled reward records.

    Each head-action reward must have been evaluated with the current risk gate
    active, and vice versa.  ``paired_policy`` records that provenance and is
    required by default; pass ``require_paired_policy=False`` only for the
    independent-compilation ablation.
    """

    if isinstance(records, (str, Path)):
        source_path = Path(records)
        loaded = load_records(source_path)
        if input_sha256 is None:
            input_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
        records = loaded
    else:
        records = list(records)
    records = list(records)
    if require_paired_policy:
        require_coupled_rewards(records)
    if num_layers <= 0 or num_heads <= 0:
        raise ValueError("num_layers and num_heads must be positive")

    head_records = [record for record in records if record.kind == "head"]
    gate_records = [record for record in records if record.kind == "gate"]
    if not head_records or not gate_records:
        raise ValueError("both head and gate calibration records are required")
    head_budgets = {record.budget for record in head_records}
    gate_budgets = {record.budget for record in gate_records}
    if head_budgets != gate_budgets:
        raise ValueError(
            "head and gate records must cover the same budgets; "
            f"head={sorted(head_budgets)}, gate={sorted(gate_budgets)}"
        )
    for record in records:
        if record.layer >= num_layers:
            raise ValueError(f"record layer {record.layer} exceeds num_layers={num_layers}")
        if record.head is not None and record.head >= num_heads:
            raise ValueError(f"record head {record.head} exceeds num_heads={num_heads}")

    budgets = np.asarray(sorted({record.budget for record in records}), dtype=np.int64)
    budget_to_index = {int(value): index for index, value in enumerate(budgets)}
    entropy_edges = _quantile_edges(
        [float(record.entropy) for record in gate_records], entropy_bins
    )
    ppl_edges = _quantile_edges([float(record.ppl) for record in gate_records], ppl_bins)
    config = cql_config or CQLConfig()
    torch_device = torch.device(device) if device is not None else None

    head_shape = (len(budgets), num_layers, num_heads)
    head_states: List[int] = []
    head_actions: List[int] = []
    head_rewards: List[float] = []
    for record in head_records:
        assert record.head is not None
        state = np.ravel_multi_index(
            (budget_to_index[record.budget], record.layer, record.head), head_shape
        )
        head_states.append(int(state))
        head_actions.append(_action_id(record.action, HEAD_ACTIONS, "head"))
        head_rewards.append(record.reward)
    head_q, head_support = _fit_table(
        head_states,
        head_actions,
        head_rewards,
        num_states=int(np.prod(head_shape)),
        num_actions=len(HEAD_ACTIONS),
        config=config,
        device=torch_device,
    )
    head_action_ids = head_q.argmax(axis=1)
    head_table = HEAD_ACTIONS[head_action_ids].reshape(head_shape)
    head_table.reshape(-1)[head_support == 0] = 1.0

    gate_shape = (len(budgets), num_layers, entropy_bins, ppl_bins)
    gate_states: List[int] = []
    gate_actions: List[int] = []
    gate_rewards: List[float] = []
    for record in gate_records:
        assert record.entropy is not None and record.ppl is not None
        entropy_bin = int(np.searchsorted(entropy_edges, record.entropy, side="right"))
        ppl_bin = int(np.searchsorted(ppl_edges, record.ppl, side="right"))
        state = np.ravel_multi_index(
            (budget_to_index[record.budget], record.layer, entropy_bin, ppl_bin), gate_shape
        )
        gate_states.append(int(state))
        gate_actions.append(_action_id(record.action, GATE_ACTIONS, "gate"))
        gate_rewards.append(record.reward)
    gate_q, gate_support = _fit_table(
        gate_states,
        gate_actions,
        gate_rewards,
        num_states=int(np.prod(gate_shape)),
        num_actions=len(GATE_ACTIONS),
        config=config,
        device=torch_device,
    )
    gate_action_ids = gate_q.argmax(axis=1)
    gate_table = GATE_ACTIONS[gate_action_ids].reshape(gate_shape)
    # Unsupported cells use the lowest (most conservative) threshold.  They are
    # never represented as learned decisions in the manifest support counts.
    gate_table.reshape(-1)[gate_support == 0] = GATE_ACTIONS[0]

    paired_counts = Counter(record.paired_policy for record in records)
    metadata = {
        "source": (
            "offline-calibration" if require_paired_policy else "offline-independent-ablation"
        ),
        "algorithm": "horizon-1-cql",
        "coupled_reward_evaluation": require_paired_policy,
        "paired_policy_counts": {str(key): value for key, value in paired_counts.items()},
        "input_sha256": input_sha256,
        "num_records": len(records),
        "num_head_records": len(head_records),
        "num_gate_records": len(gate_records),
        "head_supported_states": int(np.count_nonzero(head_support)),
        "gate_supported_states": int(np.count_nonzero(gate_support)),
        "head_actions": HEAD_ACTIONS.tolist(),
        "gate_actions": GATE_ACTIONS.tolist(),
        "cql": {
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "conservative_weight": config.conservative_weight,
            "weight_decay": config.weight_decay,
            "seed": config.seed,
            "balance_states": config.balance_states,
        },
        "reward": {
            "fidelity": "-(L_comp-L_full)",
            "gate_budget_penalty": "max(0,candidates-budget)/sequence_length",
            "head_budget_penalty": 0.0,
        },
    }
    tables = CompiledTables(
        head_weights=head_table,
        gate_thresholds=gate_table,
        entropy_edges=entropy_edges,
        ppl_edges=ppl_edges,
        budget_values=budgets,
        metadata=metadata,
    )
    tables.save(output_dir)
    return tables


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile CompilerKV W_head and T_gate tables from JSONL experience"
    )
    parser.add_argument("records", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--entropy-bins", type=int, default=20)
    parser.add_argument("--ppl-bins", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--conservative-weight", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--allow-independent-records",
        action="store_true",
        help="build the independent-compilation ablation without paired_policy provenance",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    config = CQLConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        conservative_weight=args.conservative_weight,
        seed=args.seed,
    )
    tables = compile_calibration_records(
        args.records,
        args.output_dir,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        entropy_bins=args.entropy_bins,
        ppl_bins=args.ppl_bins,
        cql_config=config,
        require_paired_policy=not args.allow_independent_records,
        device=args.device,
    )
    print(
        f"compiled W_head{tuple(tables.head_weights.shape)} and "
        f"T_gate{tuple(tables.gate_thresholds.shape)} into {args.output_dir}"
    )


if __name__ == "__main__":
    main()


__all__ = [
    "GATE_ACTIONS",
    "HEAD_ACTIONS",
    "compile_calibration_records",
    "main",
]
