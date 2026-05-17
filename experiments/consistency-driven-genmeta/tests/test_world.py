from genmeta.world import make_graph_world, make_object_world, make_resource_world


def test_object_world_deterministic_under_seed():
    a = make_object_world(10, seed=7)
    b = make_object_world(10, seed=7)
    assert a.entities == b.entities


def test_graph_world_deterministic_under_seed():
    a = make_graph_world(8, seed=3)
    b = make_graph_world(8, seed=3)
    assert sorted(a.graph.edges()) == sorted(b.graph.edges())


def test_resource_world_deterministic_under_seed():
    a = make_resource_world(6, seed=9)
    b = make_resource_world(6, seed=9)
    assert a.resource_names == b.resource_names
    assert a.agents == b.agents
