"""Metric computations for simulation logs."""

from __future__ import annotations

from itertools import combinations

import networkx as nx
import numpy as np


def _entropy(dist: dict[str, float]) -> float:
    vals = np.array([v for v in dist.values() if v > 0], dtype=float)
    if vals.size == 0:
        return 0.0
    return float(-(vals * np.log(vals)).sum())


def description_length(agents: list, threshold: float = 0.5) -> int:
    return int(
        sum(1 for a in agents for d in a.lexicon.values() for w in d.values() if w >= threshold)
        + sum(1 for a in agents for w in getattr(a, "rules", {}).values() if w >= threshold)
    )


def active_symbols(agents: list) -> int:
    return len({s for a in agents for s in a.lexicon})


def active_mappings(agents: list, threshold: float = 0.01) -> int:
    return int(sum(1 for a in agents for d in a.lexicon.values() for w in d.values() if w >= threshold))


def high_confidence_mappings(agents: list, threshold: float = 0.5) -> int:
    return active_mappings(agents, threshold=threshold)


def mean_mapping_entropy(agents: list) -> float:
    entropies = [_entropy(d) for a in agents for d in a.lexicon.values()]
    return float(np.mean(entropies)) if entropies else 0.0


def lexical_alignment(agents: list) -> float:
    sims: list[float] = []
    for a, b in combinations(agents, 2):
        for symbol in set(a.lexicon).intersection(b.lexicon):
            mids = sorted(set(a.lexicon[symbol]).union(b.lexicon[symbol]))
            va = np.array([a.lexicon[symbol].get(mid, 0.0) for mid in mids])
            vb = np.array([b.lexicon[symbol].get(mid, 0.0) for mid in mids])
            denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
            if denom > 0:
                sims.append(float(np.dot(va, vb) / denom))
    return float(np.mean(sims)) if sims else 0.0


def cluster_count(agents: list, threshold: float = 0.8) -> int:
    graph = nx.Graph()
    graph.add_nodes_from(a.id for a in agents)
    for a, b in combinations(agents, 2):
        if lexical_alignment([a, b]) >= threshold:
            graph.add_edge(a.id, b.id)
    return nx.number_connected_components(graph)
