from genmeta.consistency import check_theory
from genmeta.simulation import Config, Simulation

EXPECTED_COLUMNS = {
    "episode",
    "seed",
    "world_type",
    "n_agents",
    "consistency_enabled",
    "communicative_success_rate",
    "contradiction_rate",
    "mean_contradictions",
    "active_symbols",
    "active_mappings",
    "high_confidence_mappings",
    "current_contradictions_t035",
    "current_contradictions_t075",
    "mean_mapping_entropy",
    "lexical_alignment",
    "closure_size_proxy",
    "description_length",
    "compression_proxy",
    "cluster_count",
}


def test_simulation_runs_100_episodes_and_emits_metrics():
    df = Simulation(Config(n_agents=8, n_episodes=100, log_every=20, seed=1)).run()
    assert len(df) == 5
    assert EXPECTED_COLUMNS.issubset(df.columns)
    assert df["episode"].iloc[-1] == 100


def test_initial_conflicts_make_contradiction_ablation_measurable():
    sim = Simulation(
        Config(
            n_agents=4,
            n_episodes=1,
            initial_symbols=4,
            seed=2,
            consistency_enabled=False,
            initial_conflict_prob=1.0,
            initial_conflict_strength=0.45,
        )
    )
    contradictions = sum(
        check_theory(agent, sim.worlds, sim.meanings, threshold=0.35).contradiction_count
        for agent in sim.agents
    )
    assert contradictions > 0
