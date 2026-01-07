import pytest
from constraint_bootstrap.bootstrap_agent_v1 import BootstrapAgentV1, Handle
from q_ternary.lane_v1 import Lane

def test_nudge_rule_cases():
    agent = BootstrapAgentV1(seed=123, eligibility_min_to_consider=0.5, min_strength_to_predict=0.5)
    
    # Case 1: no candidates ⇒ still SILENT
    decision1 = agent.predict((1, 2, 3))
    assert decision1.lane == Lane.SILENT
    assert agent._silent_to_question_nudges_round == 0

    # Case 2: proto exists but below gates (eligibility=0.3 < 0.5) ⇒ QUESTION (not SILENT/NA)
    h_proto = Handle(hid="H_PROTO", sent_sig="1,2,3", resp_sig="5", eligibility=0.3, truth=0.0)
    agent._handles.append(h_proto)
    
    decision2 = agent.predict((1, 2, 3))
    assert decision2.lane == Lane.QUESTION
    assert decision2.meta.get("nudge") is True
    assert decision2.meta.get("reason") == "pre_eligible_nudge"
    assert agent._silent_to_question_nudges_round == 1

    # Case 3: Candidate exists but weak knowledge (strength < 0.5) ⇒ QUESTION (standard weak knowledge)
    h_weak = Handle(hid="H_WEAK", sent_sig="4,5,6", resp_sig="7", eligibility=0.6, truth=0.2)
    agent._handles.append(h_weak)
    
    # Reset counter for fresh check
    agent._silent_to_question_nudges_round = 0
    decision3 = agent.predict((4, 5, 6))
    assert decision3.lane == Lane.QUESTION
    assert decision3.meta.get("reason") == "weak_knowledge"
    # This shouldn't count as a "nudge" from SILENT/NA because it was already destined for QUESTION 
    # (actually in previous version it might have been NA if eligibility was low, 
    # but here eligibility is 0.6 > 0.5, so it hits the standard weak_knowledge branch)
    assert decision3.meta.get("nudge") is None 
    assert agent._silent_to_question_nudges_round == 0

def test_nudge_from_na_to_question():
    # If all matches are below eligibility_min_to_consider, it would be NA, now nudged to QUESTION
    agent = BootstrapAgentV1(seed=123, eligibility_min_to_consider=0.5)
    h = Handle(hid="H1", sent_sig="1", resp_sig="10", eligibility=0.1, truth=0.1)
    agent._handles.append(h)
    
    decision = agent.predict((1,))
    assert decision.lane == Lane.QUESTION
    assert decision.meta.get("nudge") is True
    assert decision.meta.get("reason") == "pre_eligible_nudge"
    assert agent._silent_to_question_nudges_round == 1
