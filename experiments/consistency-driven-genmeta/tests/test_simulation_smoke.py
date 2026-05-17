from genmeta.consistency import check_theory
from genmeta.simulation import Config, Simulation

EXPECTED_COLUMNS = {
    "episode",
    "seed",
    "world_type",
    "n_agents",
    "consistency_enabled",
    "initial_conflict_prob",
    "initial_conflict_strength",
    "distractor_count",
    "task_mode",
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


def test_default_initialization_does_not_seed_contradictions():
    sim = Simulation(Config(n_agents=4, n_episodes=1, initial_symbols=4, seed=2))
    contradictions = sum(
        check_theory(agent, sim.worlds, sim.meanings, threshold=0.35).contradiction_count
        for agent in sim.agents
    )
    assert sim.config.initial_conflict_prob == 0.0
    assert contradictions == 0


def test_sampled_tasks_are_discriminative_with_distractors():
    sim = Simulation(Config(n_agents=4, n_episodes=1, world_size=8, distractor_count=3, seed=4))
    task = sim._sample_task(sim.worlds[0])
    assert task.args in task.candidate_args
    assert len(task.candidate_args) > 1
    assert len({args for args in task.candidate_args}) == len(task.candidate_args)


def test_effect_summary_labels_negative_tradeoffs():
    from genmeta.cli import _effect_summary
    import pandas as pd

    final = pd.DataFrame([
        {
            "seed": 0,
            "consistency_enabled": False,
            "communicative_success_rate": 0.8,
            "contradiction_rate": 0.4,
            "mean_contradictions": 1.0,
            "current_contradictions_t035": 5.0,
            "current_contradictions_t075": 0.0,
            "high_confidence_mappings": 4.0,
            "description_length": 4.0,
            "compression_proxy": 0.1,
            "lexical_alignment": 0.4,
        },
        {
            "seed": 0,
            "consistency_enabled": True,
            "communicative_success_rate": 0.6,
            "contradiction_rate": 0.1,
            "mean_contradictions": 0.1,
            "current_contradictions_t035": 1.0,
            "current_contradictions_t075": 0.0,
            "high_confidence_mappings": 2.0,
            "description_length": 2.0,
            "compression_proxy": 0.2,
            "lexical_alignment": 0.5,
        },
    ])
    metrics = [
        "communicative_success_rate",
        "contradiction_rate",
        "mean_contradictions",
        "current_contradictions_t035",
        "current_contradictions_t075",
        "high_confidence_mappings",
        "description_length",
        "compression_proxy",
        "lexical_alignment",
    ]
    effects = _effect_summary(final, metrics)
    assert effects.loc[0, "delta_mean_contradictions"] < 0
    assert effects.loc[0, "delta_lexical_alignment"] > 0
    assert effects.loc[0, "verdict"] == "negative_tradeoff"
    assert effects.loc[1, "seed"] == "mean"


def test_innovation_symbol_can_be_interpreted_by_hearer():
    df = Simulation(
        Config(
            n_agents=4,
            n_episodes=20,
            log_every=10,
            seed=3,
            innovation_prob=1.0,
        )
    ).run()
    assert len(df) == 2


def test_effect_summary_uses_csv_safe_no_effect_label():
    from genmeta.cli import _effect_summary
    import pandas as pd

    final = pd.DataFrame([
        {
            "seed": 0,
            "consistency_enabled": False,
            "communicative_success_rate": 0.5,
            "contradiction_rate": 0.0,
            "mean_contradictions": 0.0,
            "current_contradictions_t035": 0.0,
            "current_contradictions_t075": 0.0,
            "high_confidence_mappings": 1.0,
            "description_length": 1.0,
            "compression_proxy": 0.1,
            "lexical_alignment": 0.5,
        },
        {
            "seed": 0,
            "consistency_enabled": True,
            "communicative_success_rate": 0.5,
            "contradiction_rate": 0.0,
            "mean_contradictions": 0.0,
            "current_contradictions_t035": 0.0,
            "current_contradictions_t075": 0.0,
            "high_confidence_mappings": 1.0,
            "description_length": 1.0,
            "compression_proxy": 0.1,
            "lexical_alignment": 0.5,
        },
    ])
    metrics = [
        "communicative_success_rate",
        "contradiction_rate",
        "mean_contradictions",
        "current_contradictions_t035",
        "current_contradictions_t075",
        "high_confidence_mappings",
        "description_length",
        "compression_proxy",
        "lexical_alignment",
    ]
    effects = _effect_summary(final, metrics)
    assert set(effects["verdict"]) == {"no_effect"}
