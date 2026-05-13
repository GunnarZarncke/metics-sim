"""Agent implementation with weighted lexicon learning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .consistency import ConsistencyResult
from .predicates import Meaning


@dataclass
class Agent:
    id: int
    lexicon: dict[str, dict[str, float]]
    rules: dict[str, float] = field(default_factory=dict)
    trust: dict[int, float] = field(default_factory=dict)
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    lr_success: float = 0.15
    lr_failure: float = 0.05
    lr_contradiction: float = 0.30
    decay: float = 0.001

    @classmethod
    def random(cls, agent_id: int, symbols: list[str], meanings: list[Meaning], seed: int = 0, **kwargs) -> "Agent":
        rng = np.random.default_rng(seed)
        lexicon: dict[str, dict[str, float]] = {}
        mids = [m.id for m in meanings]
        for symbol in symbols:
            values = rng.random(len(mids)) + 0.01
            values = values / values.sum()
            lexicon[symbol] = dict(zip(mids, values.astype(float)))
        return cls(id=agent_id, lexicon=lexicon, rng=rng, **kwargs)

    def normalize_symbol(self, symbol: str) -> None:
        dist = self.lexicon.get(symbol, {})
        total = sum(max(0.0, v) for v in dist.values())
        if total <= 0 and dist:
            uniform = 1.0 / len(dist)
            for mid in dist:
                dist[mid] = uniform
            return
        for mid in list(dist):
            dist[mid] = max(0.0, dist[mid]) / total

    def choose_message(self, task, world: object, meanings: Mapping[str, Meaning]) -> tuple[str, str]:
        """Choose a symbol and intended meaning that hold for the task arguments."""

        candidates = [m.id for m in meanings.values() if m.arity == len(task.args) and m.holds(world, task.args)]
        if not candidates:
            candidates = [m.id for m in meanings.values() if m.arity == len(task.args)]
        intended = str(self.rng.choice(candidates))
        best_symbol = max(self.lexicon, key=lambda s: self.lexicon[s].get(intended, 0.0))
        return best_symbol, intended

    def interpret_message(self, symbol: str, world: object | None = None) -> str:
        """Return the highest-weight meaning for a symbol."""

        dist = self.lexicon[symbol]
        return max(dist, key=dist.get)

    def propose_update(self, symbol: str, intended_meaning: str, success: bool, consistency_result: ConsistencyResult) -> dict[str, object]:
        return {
            "symbol": symbol,
            "meaning": intended_meaning,
            "success": success,
            "consistency_ok": consistency_result.ok,
            "contradictions": consistency_result.contradiction_count,
        }

    def apply_update(self, symbol: str, intended_meaning: str, success: bool, consistency_result: ConsistencyResult, consistency_enabled: bool = True) -> None:
        self.lexicon.setdefault(symbol, {})
        self.lexicon[symbol].setdefault(intended_meaning, 0.01)
        w = self.lexicon[symbol][intended_meaning]
        if consistency_enabled and not consistency_result.ok:
            self.lexicon[symbol][intended_meaning] = w - self.lr_contradiction * w
        elif success:
            self.lexicon[symbol][intended_meaning] = w + self.lr_success * (1.0 - w)
        else:
            self.lexicon[symbol][intended_meaning] = w - self.lr_failure * w
        self.normalize_symbol(symbol)

    def decay_weights(self) -> None:
        for dist in self.lexicon.values():
            n = len(dist)
            if n == 0:
                continue
            uniform = 1.0 / n
            for mid in dist:
                dist[mid] = (1.0 - self.decay) * dist[mid] + self.decay * uniform

    def prune(self, threshold: float) -> None:
        for symbol in list(self.lexicon):
            dist = self.lexicon[symbol]
            if len(dist) <= 1:
                continue
            for mid in list(dist):
                if dist[mid] < threshold:
                    del dist[mid]
            if not dist:
                del self.lexicon[symbol]
            else:
                self.normalize_symbol(symbol)

    def innovate(self, probability: float, symbols: list[str], meanings: list[Meaning]) -> None:
        if self.rng.random() >= probability:
            return
        symbol = str(self.rng.choice(symbols + [f"s{len(symbols) + self.id}"]))
        meaning = str(self.rng.choice([m.id for m in meanings]))
        self.lexicon.setdefault(symbol, {})[meaning] = self.lexicon.setdefault(symbol, {}).get(meaning, 0.0) + 0.2
        self.normalize_symbol(symbol)
