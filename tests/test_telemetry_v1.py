from __future__ import annotations

import pytest
from constraint_bootstrap.bootstrap_agent_v1 import BootstrapAgentV1
from constraint_bootstrap.alien_partners_v1 import make_partner
from constraint_bootstrap.channel_v1 import ChannelV1

def test_multi_candidate_occurs():
    """
    With high noise and promote_threshold=1, we ensure telemetry works.
    """
    seed = 42
    agent = BootstrapAgentV1(seed=seed, promote_threshold=1)
    
    # Force two handles for same input
    sent = (3, 3, 3)
    agent.observe(sent, (5,), learn=True) # Promotes H001: (3,3,3) -> (5)
    agent.observe(sent, (6,), learn=True) # Promotes H002: (3,3,3) -> (6)
    
    assert agent.total_multi_candidate_steps >= 1
    assert agent.sum_candidate_count >= 2

def test_multi_candidate_occurs_natural():
    """
    With noise_prob=0.2, noise_jitter=1 and promote_threshold=2, we get at least one step with 
    candidate_count >= 2 within 200 steps (seeded).
    """
    seed = 123
    agent = BootstrapAgentV1(seed=seed, promote_threshold=2)
    partner = make_partner("mixed")
    chan = ChannelV1(noise_prob=0.2, noise_jitter=1, seed=seed)
    
    found_multi = False
    for t in range(1, 201):
        sent = agent.choose_action(t)
        raw_resp = partner.respond(sent)
        ex = chan.transmit(sent, raw_resp)
        agent.observe(ex.sent, ex.received, learn=True)
        
        if agent.total_multi_candidate_steps > 0:
            found_multi = True
            break
            
    assert found_multi, f"Should have seen at least one multi-candidate step in 200 steps, but saw {agent.total_multi_candidate_steps}"
    assert agent.sum_candidate_count > 0
