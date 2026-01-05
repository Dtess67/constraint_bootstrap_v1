import pytest
import random
from constraint_bootstrap.partners.adversarial_partner_v1 import AdversarialPartnerV1

def test_season_flip():
    """
    Season flip test:
    Confirm the dominant response in season 0 is mostly [5] and in season 1 is mostly [7]
    for sent_sig=[4].
    """
    season_len = 50
    p_major = 0.9
    partner = AdversarialPartnerV1(seed=123, season_len=season_len, p_major=p_major, target_sig="4")
    
    # Season 0
    s0_responses = []
    for _ in range(season_len):
        s0_responses.append(partner.respond((4,)))
    
    count_5_s0 = s0_responses.count((5,))
    count_7_s0 = s0_responses.count((7,))
    
    assert count_5_s0 > count_7_s0
    assert count_5_s0 >= season_len * 0.8 # should be around 0.9
    
    # Season 1
    s1_responses = []
    for _ in range(season_len):
        s1_responses.append(partner.respond((4,)))
        
    count_5_s1 = s1_responses.count((5,))
    count_7_s1 = s1_responses.count((7,))
    
    assert count_7_s1 > count_5_s1
    assert count_7_s1 >= season_len * 0.8

def test_drift():
    """
    Drift test:
    Feed sent_sig=[3] for steps around drift_step.
    Confirm output distribution changes after drift_step.
    """
    drift_step = 20
    partner = AdversarialPartnerV1(seed=123, drift_step=drift_step, drift_sig="3")
    
    # Before drift
    # MixedPartnerV1 for [3]: sum=3 is prime -> (7,)
    before_responses = []
    for _ in range(drift_step):
        before_responses.append(partner.respond((3,)))
    
    assert all(r == (7,) for r in before_responses)
    
    # After drift
    # Should swap (7,) to (5,)
    after_responses = []
    for _ in range(10):
        after_responses.append(partner.respond((3,)))
        
    assert all(r == (5,) for r in after_responses)

def test_determinism():
    """
    Determinism test:
    Same seed -> exact same output sequence for same inputs/steps.
    """
    seed = 999
    p1 = AdversarialPartnerV1(seed=seed)
    p2 = AdversarialPartnerV1(seed=seed)
    
    inputs = [(4,), (3,), (1, 2), (4,), (5,), (3,)]
    
    res1 = [p1.respond(inp) for inp in inputs]
    res2 = [p2.respond(inp) for inp in inputs]
    
    assert res1 == res2
