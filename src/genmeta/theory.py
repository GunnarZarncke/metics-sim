"""Local theory helpers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Rule:
    """Minimal Horn-like rule placeholder for inspectable experiments."""

    id: str
    antecedents: tuple[str, ...]
    consequent: str
    weight: float = 0.0


@dataclass
class LocalTheory:
    """Accepted mappings and rules above a confidence threshold."""

    mappings: dict[str, str] = field(default_factory=dict)
    rules: dict[str, Rule] = field(default_factory=dict)
