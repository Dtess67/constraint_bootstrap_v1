import pytest
from constraint_bootstrap.bootstrap_agent_v1 import BootstrapAgentV1, Handle
from q_ternary.lane_v1 import Lane

def test_question_cooldown_blocks_repeat():
    """Verify that asking a question triggers a cooldown for that sent_sig."""
    agent = BootstrapAgentV1(
        seed_proto_handles=True, 
        question_cooldown_n=2, # Wait 2 steps after a question
        eligibility_min_to_consider=0.1
    )
    
    sent = (1, 2)
    
    # 1. First time: Seed proto and ask QUESTION
    d1 = agent.predict(sent)
    assert d1.lane == Lane.QUESTION
    assert agent._last_question_step["1,2"] == 1
    assert agent._current_step == 1
    
    # 2. Second time (step 2): Should be on cooldown
    d2 = agent.predict(sent)
    assert d2.lane == Lane.NA # Known but on cooldown
    assert d2.meta["on_cooldown"] is True
    assert agent._question_repeats_blocked_round == 1
    assert agent._current_step == 2
    
    # 3. Third time (step 3): Still on cooldown (step 3 - step 1 = 2 <= 2)
    d3 = agent.predict(sent)
    assert d3.lane == Lane.NA
    assert d3.meta["on_cooldown"] is True
    assert agent._question_repeats_blocked_round == 2
    assert agent._current_step == 3
    
    # 4. Fourth time (step 4): Cooldown expired (step 4 - step 1 = 3 > 2)
    d4 = agent.predict(sent)
    assert d4.lane == Lane.QUESTION
    assert d4.meta["on_cooldown"] is False
    assert agent._last_question_step["1,2"] == 4
    assert agent._current_step == 4

def test_question_cooldown_weak_knowledge():
    """Verify cooldown applies to weak knowledge questions too."""
    agent = BootstrapAgentV1(
        min_strength_to_predict=0.5,
        question_cooldown_n=1
    )
    h = Handle(hid="H001", sent_sig="1", resp_sig="10", eligibility=0.3, truth=0.3)
    agent._handles.append(h)
    
    # Step 1: Weak knowledge question
    d1 = agent.predict((1,))
    assert d1.lane == Lane.QUESTION
    assert d1.meta["reason"] == "weak_knowledge"
    
    # Step 2: Cooldown
    d2 = agent.predict((1,))
    assert d2.lane == Lane.NA # weak knowledge routes to NA on cooldown
    assert d2.meta["on_cooldown"] is True

def test_question_cooldown_conflict():
    """Verify cooldown applies to conflict questions too."""
    agent = BootstrapAgentV1(
        min_strength_to_predict=0.1,
        conflict_margin=0.5,
        question_cooldown_n=1
    )
    h1 = Handle(hid="H001", sent_sig="1", resp_sig="10", eligibility=0.6, truth=0.6)
    h2 = Handle(hid="H002", sent_sig="1", resp_sig="20", eligibility=0.5, truth=0.5)
    agent._handles.extend([h1, h2])
    
    # Step 1: Conflict question
    d1 = agent.predict((1,))
    assert d1.lane == Lane.QUESTION
    assert d1.meta["reason"] == "conflict"
    
    # Step 2: Cooldown
    d2 = agent.predict((1,))
    assert d2.lane == Lane.NA # conflict routes to NA on cooldown
    assert d2.meta["on_cooldown"] is True
