from genmeta.agent import Agent
from genmeta.consistency import ConsistencyResult


def test_reinforcement_increases_and_failure_decreases_weight():
    agent = Agent(id=0, lexicon={"s0": {"m": 0.5, "n": 0.5}}, lr_success=0.2, lr_failure=0.1)
    agent.apply_update("s0", "m", True, ConsistencyResult(True))
    after_success = agent.lexicon["s0"]["m"]
    assert after_success > 0.5
    agent.apply_update("s0", "m", False, ConsistencyResult(True))
    assert agent.lexicon["s0"]["m"] < after_success


def test_contradiction_penalizes_weight():
    agent = Agent(id=0, lexicon={"s0": {"m": 0.5, "n": 0.5}}, lr_contradiction=0.3)
    agent.apply_update("s0", "m", True, ConsistencyResult(False, contradiction_count=1))
    assert agent.lexicon["s0"]["m"] < 0.5
