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


@dataclass(frozen=True)
class ResourceAgent:
    """An agent in the resource-bundle world.

    Bundles are vectors over ordinary goods plus one service/resource component
    for each agent.  The agent's own offer/request bundles are generated from
    its individual vector, so there are no globally valid single-item trades.
    """

    id: int
    inventory: tuple[int, ...]
    values: tuple[float, ...]
    offer_bundle: tuple[int, ...]
    request_bundle: tuple[int, ...]


@dataclass
class ResourceWorld:
    """A no-place/no-production barter world with agent-specific bundles."""

    agents: list[ResourceAgent]
    resource_names: tuple[str, ...]
    seed: int = 0
    kind: Literal["resource"] = "resource"

    @property
    def ids(self) -> list[int]:
        return [a.id for a in self.agents]

    def agent(self, agent_id: int) -> ResourceAgent:
        return self.agents[agent_id]

    def bundle_value(self, agent_id: int, bundle: tuple[int, ...]) -> float:
        agent = self.agent(agent_id)
        return float(sum(v * q for v, q in zip(agent.values, bundle)))

    def has_inventory(self, agent_id: int, bundle: tuple[int, ...]) -> bool:
        agent = self.agent(agent_id)
        return all(have >= need for have, need in zip(agent.inventory, bundle))

    def trade_feasible(self, giver_id: int, receiver_id: int) -> bool:
        if giver_id == receiver_id:
            return False
        giver = self.agent(giver_id)
        receiver = self.agent(receiver_id)
        return self.has_inventory(giver_id, giver.offer_bundle) and self.has_inventory(receiver_id, receiver.offer_bundle)

    def utility_gain(self, agent_id: int, partner_id: int) -> float:
        agent = self.agent(agent_id)
        partner = self.agent(partner_id)
        return self.bundle_value(agent_id, partner.offer_bundle) - self.bundle_value(agent_id, agent.offer_bundle)

    def uses_agent_resource(self, giver_id: int) -> bool:
        agent_resource_offset = len(self.resource_names) - len(self.agents)
        own_resource_index = agent_resource_offset + giver_id
        return self.agent(giver_id).offer_bundle[own_resource_index] > 0


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


def make_resource_world(size: int = 20, seed: int = 0, n_goods: int = 3) -> ResourceWorld:
    """Create a deterministic bundle-trade world.

    There are no places, travel, or production.  The resource basis contains a
    few ordinary goods plus one service/resource component for each agent.  Each
    agent's offer/request bundles are generated from that agent's private value
    vector, which makes communication about whole bundles useful without giving
    agents number or set predicates.
    """

    rng = np.random.default_rng(seed)
    resource_names = tuple([f"g{i}" for i in range(n_goods)] + [f"agent:{i}" for i in range(size)])
    dim = len(resource_names)
    agents: list[ResourceAgent] = []
    for agent_id in range(size):
        values = rng.uniform(0.2, 2.0, size=dim)
        values[n_goods + agent_id] = rng.uniform(1.0, 3.0)
        inventory = rng.integers(1, 5, size=dim)
        inventory[n_goods:] = 0
        inventory[n_goods + agent_id] = 3

        wanted_good = int(np.argmax(values[:n_goods]))
        offered_good = int(np.argmin(values[:n_goods]))
        request_bundle = np.zeros(dim, dtype=int)
        offer_bundle = np.zeros(dim, dtype=int)
        request_bundle[wanted_good] = int(rng.integers(1, 4))
        offer_bundle[offered_good] = int(rng.integers(1, 3))
        if rng.random() < 0.55:
            offer_bundle[n_goods + agent_id] = 1

        agents.append(
            ResourceAgent(
                id=agent_id,
                inventory=tuple(int(x) for x in inventory),
                values=tuple(float(x) for x in values),
                offer_bundle=tuple(int(x) for x in offer_bundle),
                request_bundle=tuple(int(x) for x in request_bundle),
            )
        )
    return ResourceWorld(agents=agents, resource_names=resource_names, seed=seed)


def make_world(world_type: str, size: int, seed: int) -> ObjectWorld | GraphWorld | ResourceWorld:
    """Factory for supported world types."""

    if world_type == "object":
        return make_object_world(size=size, seed=seed)
    if world_type == "graph":
        return make_graph_world(size=size, seed=seed)
    if world_type == "resource":
        return make_resource_world(size=size, seed=seed)
    raise ValueError(f"unknown world_type: {world_type}")
