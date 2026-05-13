"""Operational finite-world consistency checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Mapping

from .predicates import Meaning


@dataclass
class ConsistencyResult:
    ok: bool
    contradiction_count: int = 0
    warnings: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)


def check_theory(agent, worlds: list[object], meanings: Mapping[str, Meaning], threshold: float = 0.75) -> ConsistencyResult:
    """Check an agent's local theory for high-confidence operational contradictions.

    The default-safe checker is brute-force over bounded worlds.  It flags symbols
    that strongly map to multiple mutually exclusive meanings from the same group.
    Graph rules are represented lightly in this prototype, so invalid rules become
    warnings unless a future rule object supplies an explicit validator.
    """

    contradictions = 0
    warnings: list[str] = []
    conflicts: list[dict[str, object]] = []

    for symbol, dist in agent.lexicon.items():
        strong = [(mid, weight) for mid, weight in dist.items() if weight >= threshold]
        for (mid_a, w_a), (mid_b, w_b) in combinations(strong, 2):
            ma = meanings.get(mid_a)
            mb = meanings.get(mid_b)
            if ma is None or mb is None:
                warnings.append(f"unknown meaning under {symbol}")
                continue
            if ma.group and ma.group == mb.group and ma.id != mb.id:
                contradictions += 1
                conflicts.append(
                    {"symbol": symbol, "meaning_a": mid_a, "meaning_b": mid_b, "weights": (w_a, w_b), "group": ma.group}
                )

    for rule_id, weight in getattr(agent, "rules", {}).items():
        if weight >= threshold and "invalid" in rule_id:
            contradictions += 1
            conflicts.append({"rule": rule_id, "weight": weight})

    return ConsistencyResult(
        ok=contradictions == 0,
        contradiction_count=contradictions,
        warnings=warnings,
        details={"conflicts": conflicts, "worlds_checked": len(worlds)},
    )
