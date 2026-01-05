import pytest
from constraint_bootstrap.bootstrap_agent_v1 import BootstrapAgentV1, Handle
from q_ternary.lane_v1 import Lane

def test_silence_penalty_boosts_correct_handle():
    agent = BootstrapAgentV1(seed=123, silence_penalty=0.1)
    # [4] -> []: len=1 (not prime), sum=4 (not prime).
    # We'll use a mapping that is correct: [1,1] -> [5]
    h = Handle(hid="H001", sent_sig="1,1", resp_sig="5", strength=0.2, hits=0)
    
    # Predict for [1,1].
    # To ensure it is SILENT (not QUESTION), we can set min_strength_to_predict very high
    # or ensure no handles exist.
    # In BootstrapAgentV1.predict, if active_candidates is empty but all_candidates is not, it QUESTIONS.
    # So we remove the handle before predict if we want SILENT, but then we can't test boost.
    # Alternative: use a signature that has NO handles for predict, then add it before observe.
    agent.min_strength_to_predict = 0.5
    
    # Check for [1,1] - should be SILENT because no handles yet
    decision = agent.predict((1,1))
    assert decision.lane == Lane.SILENT or decision.lane == Lane.NA
    
    # Now add the handle
    agent._handles.append(h)
    
    # Observe correct response. Silence penalty should trigger because we were SILENT.
    agent.observe((1,1), (5,), learn=True)
    
    # h.strength should have been boosted. 
    # Standard update for hit is +0.08. 
    # silence_penalty is +0.1.
    # So 0.2 + 0.08 + 0.1 = 0.38
    assert h.strength == pytest.approx(0.38)
    assert h.hits == 1

def test_no_silence_penalty_when_zero():
    agent = BootstrapAgentV1(seed=123, silence_penalty=0.0)
    h = Handle(hid="H001", sent_sig="1,1", resp_sig="5", strength=0.2, hits=0)
    agent._handles.append(h)
    
    agent.observe((1,1), (5,), learn=True)
    
    # Just standard update: 0.2 + 0.08 = 0.28
    assert h.strength == pytest.approx(0.28)
