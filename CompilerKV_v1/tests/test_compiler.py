import json

import torch

from kv_compression.compiler.compile import compile_calibration_records
from kv_compression.compiler.cql import CQLConfig, fit_horizon_one_cql
from kv_compression.compiler.records import CalibrationRecord
from kv_compression.core import CompiledTables


def test_horizon_one_cql_prefers_supported_high_reward_action():
    states = torch.tensor([0, 0, 0, 0])
    actions = torch.tensor([0, 0, 1, 1])
    rewards = torch.tensor([-1.0, -1.0, 1.0, 1.0])
    q_values = fit_horizon_one_cql(
        states,
        actions,
        rewards,
        num_states=1,
        num_actions=3,
        config=CQLConfig(
            learning_rate=0.1,
            batch_size=4,
            epochs=100,
            conservative_weight=0.75,
            weight_decay=0.0,
            seed=7,
            balance_states=False,
        ),
        device=torch.device("cpu"),
    )
    assert int(q_values[0].argmax()) == 1


def test_compiler_writes_valid_budget_conditioned_tables(tmp_path):
    records = []
    for action, compressed_loss in [(0.8, 1.2), (1.5, 0.8)]:
        records.append(
            CalibrationRecord(
                kind="head",
                prompt_id=f"head-{action}",
                layer=0,
                head=0,
                budget=2,
                action=action,
                full_loss=1.0,
                compressed_loss=compressed_loss,
                sequence_length=8,
                paired_policy="round-1-gate",
            )
        )
    for index, (action, compressed_loss, entropy, ppl) in enumerate(
        [(0.8, 1.2, 0.5, 2.0), (1.0, 0.8, 1.5, 8.0)]
    ):
        records.append(
            CalibrationRecord(
                kind="gate",
                prompt_id=f"gate-{index}",
                layer=0,
                budget=2,
                action=action,
                full_loss=1.0,
                compressed_loss=compressed_loss,
                sequence_length=8,
                entropy=entropy,
                ppl=ppl,
                candidate_count=2,
                paired_policy="round-1-head",
            )
        )

    compile_calibration_records(
        records,
        tmp_path,
        num_layers=1,
        num_heads=1,
        entropy_bins=2,
        ppl_bins=2,
        cql_config=CQLConfig(
            learning_rate=0.1,
            batch_size=4,
            epochs=50,
            conservative_weight=0.75,
            weight_decay=0.0,
        ),
        device="cpu",
    )

    loaded = CompiledTables.from_directory(tmp_path)
    assert loaded.head_weights.shape == (1, 1, 1)
    assert loaded.gate_thresholds.shape == (1, 1, 2, 2)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["coupled_reward_evaluation"] is True
    assert manifest["num_records"] == 4
