import networkx as nx

from genmeta.predicates import color_is, edge, not_edge, reachable_bounded, same_color, shape_is
from genmeta.world import GraphWorld, ObjectEntity, ObjectWorld


def test_object_predicates_evaluate_correctly():
    world = ObjectWorld([
        ObjectEntity(0, "red", "square", "small"),
        ObjectEntity(1, "red", "circle", "large"),
    ])
    assert color_is("red").holds(world, (0,))
    assert not shape_is("triangle").holds(world, (0,))
    assert same_color().holds(world, (0, 1))


def test_graph_predicates_evaluate_correctly():
    graph = nx.DiGraph([(0, 1), (1, 2)])
    world = GraphWorld(graph)
    assert edge().holds(world, (0, 1))
    assert not edge().holds(world, (2, 0))
    assert not_edge().holds(world, (2, 0))
    assert not not_edge().holds(world, (0, 1))
    assert reachable_bounded(3).holds(world, (0, 2))
