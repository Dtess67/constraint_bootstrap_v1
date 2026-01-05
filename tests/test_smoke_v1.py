from constraint_bootstrap.alien_partners_v1 import make_partner
from constraint_bootstrap.bootstrap_agent_v1 import BootstrapAgentV1

def test_smoke_runs():
    partner = make_partner("prime")
    agent = BootstrapAgentV1(seed=1, promote_threshold=2)
    for t in range(1, 40):
        sent = agent.choose_action(t)
        resp = partner.respond(sent)
        _ = agent.observe(sent, resp)

    assert len(agent.handles) >= 1
