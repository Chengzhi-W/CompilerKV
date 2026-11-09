"""Regression coverage for corrupt calibration input and invalid CQL options."""

import json
from dataclasses import replace

import pytest
import torch

from kv_compression.compiler.cql import CQLConfig, fit_horizon_one_cql
from kv_compression.compiler.records import CalibrationRecord, load_records


def record():
    return CalibrationRecord(
        kind="gate", prompt_id="example", layer=0, budget=2, action=0.9,
        full_loss=1.0, compressed_loss=1.2, sequence_length=8,
        entropy=0.5, ppl=2.0, candidate_count=3, paired_policy="head-round-0",
    )


@pytest.mark.parametrize("field", ["action", "full_loss", "compressed_loss", "entropy", "ppl", "reward_override"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_records_reject_nonfinite_values(field, value):
    with pytest.raises(ValueError, match=f"{field} must be finite"):
        replace(record(), **{field: value})


def test_jsonl_errors_include_file_and_line(tmp_path):
    path = tmp_path / "calibration.jsonl"
    data = record().__dict__.copy()
    data["action"] = float("nan")
    path.write_text("\n" + json.dumps(data) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"calibration.jsonl:2: action must be finite"):
        load_records(path)


def test_valid_reward_is_unchanged():
    assert record().reward == pytest.approx(-0.2 - 1 / 8)
    assert replace(record(), reward_override=0.0).reward == 0.0


@pytest.mark.parametrize("field", ["learning_rate", "conservative_weight", "weight_decay"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_cql_config_rejects_nonfinite_values(field, value):
    with pytest.raises(ValueError, match=f"{field} must be finite"):
        CQLConfig(**{field: value})


def test_cql_config_rejects_negative_weight_decay():
    with pytest.raises(ValueError, match="weight_decay"):
        CQLConfig(weight_decay=-0.1)


@pytest.mark.parametrize("field", ["state_ids", "action_ids"])
def test_cql_rejects_fractional_ids(field):
    inputs = dict(state_ids=torch.tensor([0]), action_ids=torch.tensor([0]), rewards=torch.tensor([1.0]))
    inputs[field] = torch.tensor([0.5])
    with pytest.raises(ValueError, match="integer dtypes"):
        fit_horizon_one_cql(**inputs, num_states=1, num_actions=1)


@pytest.mark.parametrize("reward", [float("nan"), float("inf"), -float("inf"), 1e100])
def test_cql_rejects_nonfinite_or_overflowing_rewards(reward):
    with pytest.raises(ValueError, match="rewards must be"):
        fit_horizon_one_cql(
            torch.tensor([0]), torch.tensor([0]), torch.tensor([reward], dtype=torch.float64),
            num_states=1, num_actions=1,
        )
