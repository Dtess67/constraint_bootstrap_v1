import pytest
from constraint_bootstrap.bootstrap_agent_v1 import BootstrapAgentV1, Handle
from q_ternary.lane_v1 import Lane

def test_proto_handle_seeding_on_observation():
    """Verify that a new handle is seeded when an unknown input is observed."""
    agent = BootstrapAgentV1(seed_proto_handles=True, seed_eligibility=0.25)
    
    sent = (1, 2)
    received = (5,)
    
    # Pre-condition: no handles
    assert len(agent._handles) == 0
    
    # Observe an exchange
    agent.observe(sent, received, learn=True, update_truth=True)
    
    # Post-condition: one handle seeded and immediately updated
    assert len(agent._handles) == 1
    h = agent._handles[0]
    assert h.sent_sig == "1,2"
    assert h.resp_sig == "5"
    # Seeded with 0.25, then updated with +0.08
    assert h.eligibility == pytest.approx(0.33)
    # Truth seeded with 0.0, then updated with +0.08 if update_truth=True
    assert h.truth == pytest.approx(0.08)
    assert agent._proto_seeded_round == 1

def test_proto_handle_no_duplicate_seeding():
    """Verify that we don't seed if a handle for that sent_sig already exists."""
    agent = BootstrapAgentV1(seed_proto_handles=True, seed_eligibility=0.25)
    
    # Seed one manually
    agent._handles.append(Handle(hid="H001", sent_sig="1,2", resp_sig="7", eligibility=0.5, truth=0.1))
    assert len(agent._handles) == 1
    
    # Observe same sent_sig but different response
    agent.observe((1, 2), (5,), learn=True, update_truth=True)
    
    # Should NOT seed a new one because sent_sig "1,2" already has a handle
    assert len(agent._handles) == 1
    assert agent._proto_seeded_round == 0

def test_proto_handle_truth_invariant():
    """Verify that seeding does not increase truth when update_truth=False."""
    agent = BootstrapAgentV1(seed_proto_handles=True, seed_eligibility=0.25)
    
    # Seed via observation with update_truth=False
    agent.observe((1, 2), (5,), learn=True, update_truth=False)
    
    h = agent._handles[0]
    assert h.truth == 0.0
    
    # Subsequent observation of same mapping should NOT increase truth if it's SILENT/QUESTION
    # and update_truth is False
    agent.observe((1, 2), (5,), learn=True, update_truth=False)
    assert h.truth == 0.0

def test_proto_handle_seeding_off():
    """Verify no seeding happens if flag is False."""
    agent = BootstrapAgentV1(seed_proto_handles=False)
    agent.observe((1, 2), (5,), learn=True)
    assert len(agent._handles) == 0

def test_proto_handle_adoption_truth_invariant():
    """Verify that adoption (filling resp_sig) does not gift truth."""
    agent = BootstrapAgentV1(seed_proto_handles=True, seed_eligibility=0.25)
    
    # 1. Trigger pre-decision seeding (predict unknown signal)
    decision = agent.predict((1,2))
    assert len(agent._handles) == 1
    h = agent._handles[0]
    assert h.resp_sig == "0"
    assert h.truth == 0.0
    
    # 2. Trigger adoption (observe with the actual response)
    # Even if update_truth=True, adoption itself should be truth-neutral if hits/misses are 0.
    # Actually, if hits/misses are 0, adoption happens. 
    # But after adoption, standard update is called.
    # To test ADOPTION truth invariant, we should ensure the adoption path itself doesn't GIFT truth.
    # Standard update might increase truth if update_truth=True.
    
    # Let's call observe with update_truth=False to see if adoption gifts truth.
    agent.observe((1,2), (5,), learn=True, update_truth=False)
    
    assert h.resp_sig == "5"
    assert h.truth == 0.0, "Adoption path should not gift truth"
    # Eligibility should increase because standard update is called after adoption
    assert h.eligibility == pytest.approx(0.33)
