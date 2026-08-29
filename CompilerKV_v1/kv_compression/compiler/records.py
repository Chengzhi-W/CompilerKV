"""Schema and reward construction for offline calibration records."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, List, Optional, Union


@dataclass(frozen=True)
class CalibrationRecord:
    """One observed state/action/fidelity tuple from a calibration prompt."""

    kind: str
    prompt_id: str
    layer: int
    budget: int
    action: float
    full_loss: float
    compressed_loss: float
    sequence_length: int
    head: Optional[int] = None
    entropy: Optional[float] = None
    ppl: Optional[float] = None
    candidate_count: Optional[int] = None
    paired_policy: Optional[str] = None
    reward_override: Optional[float] = None

    def __post_init__(self) -> None:
        if self.kind not in {"head", "gate"}:
            raise ValueError("record kind must be 'head' or 'gate'")
        if not self.prompt_id:
            raise ValueError("prompt_id cannot be empty")
        if self.layer < 0 or self.budget < 0 or self.sequence_length <= 0:
            raise ValueError("invalid layer, budget or sequence_length")
        if self.kind == "head" and (self.head is None or self.head < 0):
            raise ValueError("head records require a non-negative head index")
        if self.kind == "gate":
            if self.entropy is None or self.ppl is None or self.candidate_count is None:
                raise ValueError("gate records require entropy, ppl and candidate_count")
            if self.ppl <= 0 or self.candidate_count < 0:
                raise ValueError("ppl must be positive and candidate_count non-negative")

    @property
    def reward(self) -> float:
        if self.reward_override is not None:
            return float(self.reward_override)
        fidelity = -(self.compressed_loss - self.full_loss)
        if self.kind == "head":
            return fidelity
        assert self.candidate_count is not None
        over_retention = max(0, self.candidate_count - self.budget)
        return fidelity - over_retention / float(self.sequence_length)

    @classmethod
    def from_dict(cls, value: dict) -> "CalibrationRecord":
        losses = value.get("loss", {})
        reward_override = value.get("reward")
        full_loss = value.get("full_loss", losses.get("full"))
        compressed_loss = value.get("compressed_loss", losses.get("compressed"))
        if reward_override is not None and full_loss is None and compressed_loss is None:
            full_loss = compressed_loss = 0.0
        return cls(
            kind=str(value["kind"]),
            prompt_id=str(value["prompt_id"]),
            layer=int(value["layer"]),
            budget=int(value["budget"]),
            action=float(value["action"]),
            full_loss=float(full_loss),
            compressed_loss=float(compressed_loss),
            sequence_length=int(value["sequence_length"]),
            head=(int(value["head"]) if value.get("head") is not None else None),
            entropy=(float(value["entropy"]) if value.get("entropy") is not None else None),
            ppl=(float(value["ppl"]) if value.get("ppl") is not None else None),
            candidate_count=(
                int(value["candidate_count"])
                if value.get("candidate_count") is not None
                else None
            ),
            paired_policy=value.get("paired_policy"),
            reward_override=(float(reward_override) if reward_override is not None else None),
        )


def load_records(path: Union[str, Path]) -> List[CalibrationRecord]:
    """Load newline-delimited calibration records."""

    path = Path(path)
    records: List[CalibrationRecord] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(CalibrationRecord.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"invalid calibration record at {path}:{line_number}: {exc}") from exc
    if not records:
        raise ValueError(f"no calibration records found in {path}")
    return records


def require_coupled_rewards(records: Iterable[CalibrationRecord]) -> None:
    """Reject independent-table rewards for the paper's joint compilation path."""

    missing = [record.prompt_id for record in records if not record.paired_policy]
    if missing:
        preview = ", ".join(missing[:3])
        raise ValueError(
            "coupled compilation requires paired_policy on every record; "
            f"missing for {preview}{'...' if len(missing) > 3 else ''}"
        )


__all__ = ["CalibrationRecord", "load_records", "require_coupled_rewards"]
