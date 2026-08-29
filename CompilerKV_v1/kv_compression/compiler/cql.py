"""Horizon-1 Conservative Q-Learning used by both compiled tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class CQLConfig:
    """Paper defaults for the tabular contextual-bandit objective."""

    learning_rate: float = 3e-4
    batch_size: int = 4096
    epochs: int = 10
    conservative_weight: float = 0.75
    weight_decay: float = 0.01
    seed: int = 42
    balance_states: bool = True

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("learning_rate, batch_size and epochs must be positive")
        if self.conservative_weight < 0:
            raise ValueError("conservative_weight cannot be negative")


def fit_horizon_one_cql(
    state_ids: torch.Tensor,
    action_ids: torch.Tensor,
    rewards: torch.Tensor,
    *,
    num_states: int,
    num_actions: int,
    config: Optional[CQLConfig] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Fit Eq. (1) as a finite Q table and return ``Q[state, action]``.

    Prefill-only retention is a one-step contextual bandit, so there is no
    bootstrap target or target network.  The first loss term pessimistically
    suppresses actions outside behavior support and the second regresses the
    observed fidelity reward.
    """

    config = config or CQLConfig()
    if state_ids.ndim != 1 or action_ids.ndim != 1 or rewards.ndim != 1:
        raise ValueError("state_ids, action_ids and rewards must be 1-D")
    if not (len(state_ids) == len(action_ids) == len(rewards)):
        raise ValueError("CQL record tensors must have equal lengths")
    if len(state_ids) == 0:
        raise ValueError("cannot fit CQL without records")
    if int(state_ids.min()) < 0 or int(state_ids.max()) >= num_states:
        raise ValueError("state id is outside the declared state space")
    if int(action_ids.min()) < 0 or int(action_ids.max()) >= num_actions:
        raise ValueError("action id is outside the declared action space")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)

    states_cpu = state_ids.to(dtype=torch.long, device="cpu")
    actions_cpu = action_ids.to(dtype=torch.long, device="cpu")
    rewards_cpu = rewards.to(dtype=torch.float32, device="cpu")
    q_values = torch.nn.Parameter(torch.zeros(num_states, num_actions, device=device))
    optimizer = torch.optim.AdamW(
        [q_values], lr=config.learning_rate, weight_decay=config.weight_decay
    )

    if config.balance_states:
        counts = torch.bincount(states_cpu, minlength=num_states).clamp_min(1)
        sample_weights = counts[states_cpu].float().reciprocal()
    else:
        sample_weights = None

    for _ in range(config.epochs):
        if sample_weights is None:
            order = torch.randperm(len(states_cpu), generator=generator)
        else:
            order = torch.multinomial(
                sample_weights,
                num_samples=len(states_cpu),
                replacement=True,
                generator=generator,
            )
        for start in range(0, len(order), config.batch_size):
            batch_indices = order[start : start + config.batch_size]
            states = states_cpu[batch_indices].to(device)
            actions = actions_cpu[batch_indices].to(device)
            reward = rewards_cpu[batch_indices].to(device)

            q_state = q_values[states]
            behavior_q = q_state.gather(1, actions.unsqueeze(1)).squeeze(1)
            conservative = (torch.logsumexp(q_state, dim=1) - behavior_q).mean()
            regression = 0.5 * torch.mean((behavior_q - reward) ** 2)
            loss = config.conservative_weight * conservative + regression

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return q_values.detach().cpu()


__all__ = ["CQLConfig", "fit_horizon_one_cql"]
