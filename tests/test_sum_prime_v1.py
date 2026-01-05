from constraint_bootstrap.partners.sum_prime_v1 import SumPrimePartnerV1
from constraint_bootstrap.partners import make_partner
from constraint_bootstrap.bootstrap_agent_v1 import BootstrapAgentV1

def test_sum_prime_logic():
    p = SumPrimePartnerV1()
    # sent=[2,2,3] sum=7 ⇒ returns (7,)
    assert p.respond((2, 2, 3)) == (7,)
    # sent=[2,2,4] sum=8 ⇒ returns ()
    assert p.respond((2, 2, 4)) == ()

def test_sum_prime_smoke():
    partner = make_partner("sumprime")
    agent = BootstrapAgentV1(seed=7, promote_threshold=2)
    # Run 80 steps with fixed seed and assert at least one handle forms
    for t in range(1, 81):
        sent = agent.choose_action(t)
        resp = partner.respond(sent)
        _ = agent.observe(sent, resp)
    
    assert len(agent.handles) >= 1
