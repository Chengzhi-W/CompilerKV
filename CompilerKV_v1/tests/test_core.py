import numpy as np
import pytest
import torch

from kv_compression.core import CompiledTables, CompilerKVConfig, CompilerKVOperator


def make_tables(threshold=0.9):
    return CompiledTables(
        head_weights=np.array([[1.0, 1.5]], dtype=np.float32),
        gate_thresholds=np.full((1, 2, 2), threshold, dtype=np.float32),
        entropy_edges=np.array([1.0], dtype=np.float32),
        ppl_edges=np.array([5.0], dtype=np.float32),
        budget_values=np.array([3]),
        metadata={"source": "unit-test"},
    )


def test_stabilized_utility_and_weighted_max_pool_match_equations():
    tables = make_tables()
    operator = CompilerKVOperator(
        tables,
        CompilerKVConfig(observation_window=2, entropy_bins=2, ppl_bins=2),
    )
    attention = torch.tensor(
        [[[[[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]],
           [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]]]]]
    )
    values = torch.tensor(
        [[[[[1.0], [1.0], [1.0]], [[1.0], [2.0], [3.0]]]]]
    )

    utility, alpha = operator.stabilized_utility(attention, values)
    scores = operator.head_aware_scores(utility, [3])

    torch.testing.assert_close(alpha, torch.tensor([[0.75, 1.5, 0.75]]))
    torch.testing.assert_close(scores, torch.tensor([[[0.75, 2.25, 1.6875]]]))


def test_elastic_selection_does_not_pad_under_budget_candidates():
    operator = CompilerKVOperator(
        make_tables(),
        CompilerKVConfig(observation_window=2, entropy_bins=2, ppl_bins=2),
    )
    attention = torch.tensor(
        [[[[[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]],
           [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]]]]]
    )
    values = torch.tensor(
        [[[[[1.0], [1.0], [1.0]], [[1.0], [2.0], [3.0]]]]]
    )
    keys = values.clone()

    output = operator.compress(
        attention,
        values,
        budgets=3,
        key_cache=keys,
        local_ppl=2.0,
    )

    assert output.indices[0][0].tolist() == [1, 2]
    assert output.key_cache[0][0].shape[-2] == 2


def test_budget_clamp_uses_top_scores_within_candidates():
    operator = CompilerKVOperator(
        make_tables(),
        CompilerKVConfig(observation_window=2, entropy_bins=2, ppl_bins=2),
    )
    scores = torch.tensor([[[0.95, 1.4, 1.1]]])
    thresholds = torch.tensor([[0.9]])
    selected = operator.elastic_select(scores, thresholds, [1])
    assert selected[0][0].tolist() == [1]


def test_relative_depth_mapping_interpolates_compiled_layers():
    tables = CompiledTables(
        head_weights=np.array([[0.8], [1.5]], dtype=np.float32),
        gate_thresholds=np.array([[[0.8]], [[1.0]]], dtype=np.float32),
        entropy_edges=np.array([], dtype=np.float32),
        ppl_edges=np.array([], dtype=np.float32),
        metadata={"source": "unit-test"},
    )
    mapped = tables.head_for(layers=3, heads=1, budget=512)
    np.testing.assert_allclose(mapped[:, 0], [0.8, 1.15, 1.5], atol=1e-6)


def test_budget_and_head_mismatches_are_not_silently_interpolated():
    tables = CompiledTables(
        head_weights=np.ones((1, 1, 2), dtype=np.float32),
        gate_thresholds=np.full((1, 1, 1, 1), 0.9, dtype=np.float32),
        entropy_edges=np.array([], dtype=np.float32),
        ppl_edges=np.array([], dtype=np.float32),
        budget_values=np.array([512]),
        metadata={"source": "unit-test"},
    )
    with pytest.raises(ValueError, match="budget 256 was not compiled"):
        tables.gate_for(layers=1, budget=256)
    with pytest.raises(ValueError, match="compiled for 2 heads"):
        tables.head_for(layers=1, heads=4, budget=512)
