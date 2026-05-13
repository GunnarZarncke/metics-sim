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
