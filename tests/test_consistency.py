from genmeta.agent import Agent
from genmeta.consistency import check_theory
from genmeta.predicates import object_meanings
from genmeta.world import make_object_world


def test_detects_mutually_exclusive_symbol_mappings():
    meanings = {m.id: m for m in object_meanings()}
    agent = Agent(id=0, lexicon={"s0": {"color:red": 0.8, "color:blue": 0.8}})
    result = check_theory(agent, [make_object_world(seed=0)], meanings, threshold=0.75)
    assert not result.ok
    assert result.contradiction_count == 1
