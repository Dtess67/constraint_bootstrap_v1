from constraint_bootstrap.bootstrap_agent_v1 import BootstrapAgentV1, Handle

def test_decay_reduces_strength_when_learn_true():
    # Test A: decay reduces strength when learn=True
    agent = BootstrapAgentV1(decay_rate=0.1)
    # Manually add a handle
    h = Handle(hid="H001", sent_sig="1", resp_sig="2", strength=1.0)
    agent._handles.append(h)
    
    # call observe(... learn=True ...) once
    # Use different input to avoid matching and updating the handle
    agent.observe(sent=(9,9), received=(9,9), learn=True)
    
    # assert strength is now 0.9 (or close)
    # 1.0 * (1.0 - 0.1) = 0.9
    assert abs(h.strength - 0.9) < 1e-6

def test_freeze_does_not_decay():
    # Test B: freeze does NOT decay
    agent = BootstrapAgentV1(decay_rate=0.1)
    h = Handle(hid="H001", sent_sig="1", resp_sig="2", strength=1.0)
    agent._handles.append(h)
    
    # Same setup, but learn=False
    agent.observe(sent=(1,), received=(2,), learn=False)
    
    # Assert strength remains unchanged.
    assert h.strength == 1.0

def test_prune_works():
    # Test C: prune works
    # decay_rate high enough to drop under prune_below, confirm handle removed.
    agent = BootstrapAgentV1(decay_rate=0.5, prune_below=0.6)
    h = Handle(hid="H001", sent_sig="1", resp_sig="2", strength=1.0)
    agent._handles.append(h)
    
    # 1.0 * (1.0 - 0.5) = 0.5. 0.5 < 0.6 so it should be pruned.
    agent.observe(sent=(1,), received=(2,), learn=True)
    
    assert len(agent._handles) == 0
