import pytest
from constraint_bootstrap.bootstrap_agent_v1 import BootstrapAgentV1
from q_ternary.lane_v1 import Lane

def test_proto_seed_predecision_yields_question():
    """First-ever unseen input must yield Lane.QUESTION and create one proto-handle."""
    agent = BootstrapAgentV1(seed=123, seed_proto_handles=True, seed_eligibility=0.25, eligibility_min_to_consider=0.1)
    
    sent = (1, 2, 3)
    # First predict: should seed and return QUESTION (nudged because eligibility 0.25 >= 0.1)
    # Wait, if seed_eligibility=0.25 and eligibility_min_to_consider=0.1, it is eligible.
    # But truth is 0.0, so it fails speak gate (0.35).
    # It should go to QUESTION (weak knowledge).
    
    decision = agent.predict(sent)
    
    assert decision.lane == Lane.QUESTION
    assert decision.meta.get("was_proto_seeded_predecision") is True
    assert len(agent._handles) == 1
    
    h = agent._handles[0]
    assert h.sent_sig == "1,2,3"
    assert h.resp_sig == "0"
    assert h.truth == 0.0
    assert h.eligibility == 0.25

def test_proto_seed_predecision_stays_question_until_truth():
    """Second time same input remains QUESTION if truth gate not met."""
    agent = BootstrapAgentV1(seed=123, seed_proto_handles=True, seed_eligibility=0.05, eligibility_min_to_consider=0.1)
    # seed_eligibility=0.05 < eligibility_min_to_consider=0.1
    # So first predict should nudge it to QUESTION
    
    sent = (1, 1)
    d1 = agent.predict(sent)
    assert d1.lane == Lane.QUESTION
    assert d1.meta.get("was_nudged_to_question") is True
    
    # Second predict (no changes to handle)
    d2 = agent.predict(sent)
    assert d2.lane == Lane.QUESTION
    assert d2.meta.get("was_proto_seeded_predecision") is False # already exists
    assert d2.meta.get("was_nudged_to_question") is True

def test_proto_seed_disabled_yields_silent():
    """If seeding is disabled, first unseen input is SILENT."""
    agent = BootstrapAgentV1(seed=123, seed_proto_handles=False)
    sent = (9, 9)
    decision = agent.predict(sent)
    assert decision.lane == Lane.SILENT
    assert len(agent._handles) == 0
