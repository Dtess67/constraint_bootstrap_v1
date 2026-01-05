from constraint_bootstrap.bootstrap_agent_v1 import BootstrapAgentV1, Handle

def test_topk_gating_in_predict():
    # Only top-K strongest handles should be considered for prediction
    agent = BootstrapAgentV1(compete_topk=1, min_strength_to_predict=0.1)
    
    # H001 is weaker but appears first in list if not sorted
    h1 = Handle(hid="H001", sent_sig="1", resp_sig="10", strength=0.2)
    # H002 is stronger
    h2 = Handle(hid="H002", sent_sig="1", resp_sig="20", strength=0.5)
    
    agent._handles = [h1, h2]
    
    # Prediction should use H002 because it's stronger (handles property sorts them)
    # If compete_topk=1, it should only take H002.
    decision = agent.predict((1,))
    assert decision.act == (20,)
    
def test_topk_gating_in_predict_fixed():
    agent = BootstrapAgentV1(compete_topk=1, min_strength_to_predict=0.1)
    h1 = Handle(hid="H001", sent_sig="1", resp_sig="10", strength=0.2)
    h2 = Handle(hid="H002", sent_sig="1", resp_sig="20", strength=0.5)
    agent._handles = [h1, h2]
    
    # handles property returns [h2, h1]
    # candidates in predict: [h2, h1]
    # compete_topk=1 -> [h2]
    # pred = 20
    decision = agent.predict((1,))
    assert decision.act == (20,)

def test_topk_gating_in_update():
    # Only top-K handles should be updated
    agent = BootstrapAgentV1(compete_topk=1)
    h1 = Handle(hid="H001", sent_sig="1", resp_sig="10", strength=0.5)
    h2 = Handle(hid="H002", sent_sig="1", resp_sig="10", strength=0.2)
    agent._handles = [h1, h2]
    
    # h1 is stronger, so it's the only one in top-1
    agent.observe(sent=(1,), received=(10,), learn=True)
    
    # h1 should have hits=1
    assert h1.hits == 1
    # h2 should have hits=0 because it wasn't in top-1
    assert h2.hits == 0

def test_inhibition_works():
    # Winning handle reduces strength of losers
    agent = BootstrapAgentV1(inhibit_mult=0.5)
    h1 = Handle(hid="H001", sent_sig="1", resp_sig="10", strength=0.6) # Winner
    h2 = Handle(hid="H002", sent_sig="1", resp_sig="20", strength=0.4) # Loser
    agent._handles = [h1, h2]
    
    # observe (1,) -> (10,)
    # h1 matches and is winner.
    # h1.strength after update: 0.6 + 0.08 = 0.68
    # h2 is also checked during update.
    # It has h2.sent_sig == sent_s ("1") but h2.resp_sig != recv_s ("10" != "20")
    # So h2.update(matched=False) is called FIRST.
    # h2.strength becomes 0.4 - 0.12 = 0.28
    # THEN inhibition is applied:
    # h2.strength = max(0.0, 0.28 - (0.68 * 0.5)) = max(0.0, 0.28 - 0.34) = 0.0
    
    agent.observe(sent=(1,), received=(10,), learn=True)
    
    assert abs(h1.strength - 0.68) < 1e-6
    assert h2.strength == 0.0
