"""Main agent-based simulation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

from .agent import Agent
from .consistency import check_theory
from .metrics import active_mappings, active_symbols, cluster_count, description_length, lexical_alignment, mean_mapping_entropy
from .predicates import Meaning, meanings_for_world
from .world import make_world


@dataclass
class Config:
    n_agents: int = 50
    n_episodes: int = 5000
    world_type: str = "object"
    world_size: int = 20
    initial_symbols: int = 16
    interaction_graph: str = "random"
    lr_success: float = 0.15
    lr_failure: float = 0.05
    lr_contradiction: float = 0.30
    decay: float = 0.001
    prune_threshold: float = 0.01
    innovation_prob: float = 0.001
    consistency_weight: float = 3.0
    seed: int = 0
    log_every: int = 10
    consistency_enabled: bool = True
    pruning_enabled: bool = True
    innovation_enabled: bool = True


@dataclass(frozen=True)
class Task:
    kind: str
    args: tuple[int, ...]


class Simulation:
    def __init__(self, config: Config):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.meaning_list = meanings_for_world(config.world_type)
        self.meanings: dict[str, Meaning] = {m.id: m for m in self.meaning_list}
        self.symbols = [f"s{i}" for i in range(config.initial_symbols)]
        self.agents = [
            Agent.random(
                i,
                self.symbols,
                self.meaning_list,
                seed=config.seed * 100_000 + i,
                lr_success=config.lr_success,
                lr_failure=config.lr_failure,
                lr_contradiction=config.lr_contradiction,
                decay=config.decay,
            )
            for i in range(config.n_agents)
        ]
        self.graph = self._make_interaction_graph()
        self.worlds = [make_world(config.world_type, config.world_size, config.seed + i) for i in range(3)]
        self.success_window: deque[int] = deque(maxlen=max(1, config.log_every))
        self.contra_window: deque[int] = deque(maxlen=max(1, config.log_every))
        self.mean_contra_window: deque[float] = deque(maxlen=max(1, config.log_every))
        self.rows: list[dict[str, float | int | str | bool]] = []

    def _make_interaction_graph(self) -> nx.Graph:
        n = self.config.n_agents
        if self.config.interaction_graph == "complete":
            return nx.complete_graph(n)
        if self.config.interaction_graph == "small_world":
            k = min(max(2, n // 10), n - 1)
            if k % 2 == 1:
                k -= 1
            return nx.watts_strogatz_graph(n, max(2, k), 0.2, seed=self.config.seed)
        graph = nx.gnp_random_graph(n, min(0.2, 4 / max(n - 1, 1)), seed=self.config.seed)
        for i in range(n):
            if graph.degree(i) == 0 and n > 1:
                graph.add_edge(i, (i + 1) % n)
        return graph

    def _sample_agents(self) -> tuple[Agent, Agent]:
        speaker_id = int(self.rng.integers(0, self.config.n_agents))
        neighbors = list(self.graph.neighbors(speaker_id))
        hearer_id = int(self.rng.choice(neighbors if neighbors else [i for i in range(self.config.n_agents) if i != speaker_id]))
        return self.agents[speaker_id], self.agents[hearer_id]

    def _sample_task(self, world: object) -> Task:
        ids = list(world.ids)  # type: ignore[attr-defined]
        if self.rng.random() < 0.55:
            return Task("unary_reference", (int(self.rng.choice(ids)),))
        return Task("binary_relation", tuple(int(x) for x in self.rng.choice(ids, size=2, replace=True)))


    def _decay_conflicts(self, agent: Agent, symbol: str, intended: str) -> None:
        """Compact a consistent symbol by weakening mutually exclusive alternatives."""

        intended_meaning = self.meanings.get(intended)
        if intended_meaning is None or intended_meaning.group is None or symbol not in agent.lexicon:
            return
        for mid in list(agent.lexicon[symbol]):
            other = self.meanings.get(mid)
            if mid != intended and other is not None and other.group == intended_meaning.group:
                agent.lexicon[symbol][mid] *= 1.0 - (agent.lr_contradiction * 0.5)
        agent.normalize_symbol(symbol)

    def _score_success(self, intended: str, interpreted: str, world: object, task: Task) -> bool:
        if intended == interpreted:
            return True
        mi = self.meanings[intended]
        mh = self.meanings.get(interpreted)
        return bool(mh and mi.arity == mh.arity and mi.holds(world, task.args) == mh.holds(world, task.args) and self.rng.random() < 0.25)

    def run_episode(self, episode: int) -> None:
        world = self.worlds[episode % len(self.worlds)]
        speaker, hearer = self._sample_agents()
        task = self._sample_task(world)
        symbol, intended = speaker.choose_message(task, world, self.meanings)
        interpreted = hearer.interpret_message(symbol)
        success = self._score_success(intended, interpreted, world, task)

        # Always measure contradictions so the no-consistency ablation can be
        # compared directly; the flag only disables contradiction penalties.
        sp_result = check_theory(speaker, self.worlds, self.meanings, threshold=0.35)
        hr_result = check_theory(hearer, self.worlds, self.meanings, threshold=0.35)

        speaker.apply_update(symbol, intended, success, sp_result, self.config.consistency_enabled)
        hearer.apply_update(symbol, intended, success, hr_result, self.config.consistency_enabled)
        if self.config.consistency_enabled and success:
            self._decay_conflicts(speaker, symbol, intended)
            self._decay_conflicts(hearer, symbol, intended)
        speaker.decay_weights()
        hearer.decay_weights()
        if self.config.pruning_enabled:
            speaker.prune(self.config.prune_threshold)
            hearer.prune(self.config.prune_threshold)
        if self.config.innovation_enabled:
            speaker.innovate(self.config.innovation_prob, self.symbols, self.meaning_list)
            hearer.innovate(self.config.innovation_prob, self.symbols, self.meaning_list)

        contradictions = sp_result.contradiction_count + hr_result.contradiction_count
        self.success_window.append(int(success))
        self.contra_window.append(int(contradictions > 0))
        self.mean_contra_window.append(float(contradictions) / 2.0)

    def _log_row(self, episode: int) -> None:
        desc = description_length(self.agents)
        success_rate = float(np.mean(self.success_window)) if self.success_window else 0.0
        self.rows.append(
            {
                "episode": episode,
                "seed": self.config.seed,
                "world_type": self.config.world_type,
                "n_agents": self.config.n_agents,
                "consistency_enabled": self.config.consistency_enabled,
                "communicative_success_rate": success_rate,
                "contradiction_rate": float(np.mean(self.contra_window)) if self.contra_window else 0.0,
                "mean_contradictions": float(np.mean(self.mean_contra_window)) if self.mean_contra_window else 0.0,
                "active_symbols": active_symbols(self.agents),
                "active_mappings": active_mappings(self.agents),
                "mean_mapping_entropy": mean_mapping_entropy(self.agents),
                "lexical_alignment": lexical_alignment(self.agents),
                "closure_size_proxy": desc,
                "description_length": desc,
                "compression_proxy": success_rate / max(desc, 1),
                "cluster_count": cluster_count(self.agents),
            }
        )

    def run(self) -> pd.DataFrame:
        for episode in range(1, self.config.n_episodes + 1):
            self.run_episode(episode)
            if episode % self.config.log_every == 0 or episode == self.config.n_episodes:
                self._log_row(episode)
        return pd.DataFrame(self.rows)
