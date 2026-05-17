from genmeta.predicates import agent_resource_offer, feasible_trade, mutual_gain, resource_meanings
from genmeta.simulation import Config, Simulation
from genmeta.world import make_resource_world


def test_resource_world_is_deterministic_and_agents_are_resources():
    a = make_resource_world(size=6, seed=5)
    b = make_resource_world(size=6, seed=5)
    assert a.resource_names == b.resource_names
    assert a.agents == b.agents
    assert "agent:0" in a.resource_names
    assert "agent:5" in a.resource_names
    assert any(a.uses_agent_resource(agent.id) for agent in a.agents)


def test_resource_predicates_evaluate_bundle_trade_relations():
    world = make_resource_world(size=6, seed=1)
    args = (0, 1)
    assert feasible_trade().holds(world, args) == world.trade_feasible(*args)
    assert mutual_gain().holds(world, args) == (
        world.utility_gain(0, 1) > 0 and world.utility_gain(1, 0) > 0
    )
    assert agent_resource_offer().holds(world, args) == world.uses_agent_resource(0)
    assert all(m.arity == 2 for m in resource_meanings())


def test_resource_simulation_samples_bundle_trade_tasks():
    sim = Simulation(Config(n_agents=6, world_type="resource", world_size=6, n_episodes=1, seed=4))
    speaker, hearer = sim._sample_agents()
    task = sim._sample_task(sim.worlds[0], speaker.id, hearer.id)
    assert task.kind == "bundle_trade"
    assert task.args == (speaker.id, hearer.id)
    assert all(len(args) == 2 and args[0] != args[1] for args in task.candidate_args)


def test_resource_simulation_smoke_run():
    df = Simulation(Config(n_agents=6, world_type="resource", world_size=6, n_episodes=40, log_every=20, seed=2)).run()
    assert len(df) == 2
    assert set(df["world_type"]) == {"resource"}
