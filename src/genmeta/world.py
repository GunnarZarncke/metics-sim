"""Finite synthetic worlds used by the simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import networkx as nx
import numpy as np

COLORS = ("red", "blue", "green")
SHAPES = ("circle", "square", "triangle")
SIZES = ("small", "medium", "large")


@dataclass(frozen=True)
class ObjectEntity:
    """Entity in an object-feature world."""

    id: int
    color: str
    shape: str
    size: str


@dataclass
class ObjectWorld:
    """A bounded world of colored, shaped, sized entities."""

    entities: list[ObjectEntity]
    seed: int = 0
    kind: Literal["object"] = "object"

    @property
    def ids(self) -> list[int]:
        return [e.id for e in self.entities]

    def entity(self, entity_id: int) -> ObjectEntity:
        return self.entities[entity_id]


@dataclass
class GraphWorld:
    """A bounded directed graph world."""

    graph: nx.DiGraph
    seed: int = 0
    kind: Literal["graph"] = "graph"

    @property
    def ids(self) -> list[int]:
        return list(self.graph.nodes)


def make_object_world(size: int = 20, seed: int = 0) -> ObjectWorld:
    """Create a deterministic object-feature world."""

    rng = np.random.default_rng(seed)
    entities = [
        ObjectEntity(
            id=i,
            color=str(rng.choice(COLORS)),
            shape=str(rng.choice(SHAPES)),
            size=str(rng.choice(SIZES)),
        )
        for i in range(size)
    ]
    return ObjectWorld(entities=entities, seed=seed)


def make_graph_world(size: int = 20, seed: int = 0, edge_prob: float = 0.15) -> GraphWorld:
    """Create a deterministic directed graph world with occasional self loops."""

    rng = np.random.default_rng(seed)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(size))
    for i in range(size):
        for j in range(size):
            p = edge_prob * (0.35 if i == j else 1.0)
            if rng.random() < p:
                graph.add_edge(i, j)
    return GraphWorld(graph=graph, seed=seed)


def make_world(world_type: str, size: int, seed: int) -> ObjectWorld | GraphWorld:
    """Factory for supported world types."""

    if world_type == "object":
        return make_object_world(size=size, seed=seed)
    if world_type == "graph":
        return make_graph_world(size=size, seed=seed)
    raise ValueError(f"unknown world_type: {world_type}")
