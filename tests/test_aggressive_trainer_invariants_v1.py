import pytest
from constraint_bootstrap.bootstrap_agent_v1 import BootstrapAgentV1, Handle
from q_ternary.lane_v1 import Lane
from q_ternary.training.aggressive_trainer_v1 import AggressiveTrainerV1, TrainingSample

def test_trainer_never_writes_learning_updates_on_approvals():
    """Ensure trainer never writes learning updates on approvals (err=0 and high margin)."""
    agent = BootstrapAgentV1(seed=123)
    # [4] -> []: len=1 (not prime), sum=4 (not prime).
    h = Handle(hid="H001", sent_sig="4", resp_sig="0", strength=1.0, hits=10)
    agent._handles.append(h)
    
    trainer = AggressiveTrainerV1(agent, partner_name="mixed", seed=123)
    sample = TrainingSample(sent=(4,), sent_sig="4")
    
    # Record initial state
    initial_hits = h.hits
    initial_strength = h.strength
    
    # Run one round with batch size 1
    # Ensure uncertainty threshold is low so it doesn't trigger
    trainer.train_round(batch_size=1, drill_n=0, uncertainty_threshold=0.0, fixed_batch=[sample])
    
    # If correct, hits and strength should NOT have changed because AggressiveTrainer
    # should only call observe when it needs to correct or handle uncertainty.
    # Actually, AggressiveTrainerV1.train_round calls self.agent.observe(..., learn=True)
    # ONLY if err > 0 or is_uncertain.
    
    assert h.hits == initial_hits, "Hits should not increase on perfect approval"
    assert h.strength == initial_strength, "Strength should not change on perfect approval"

def test_no_updates_on_question_lane():
    """Absolutely NO core weight updates on lanes: QUESTION."""
    agent = BootstrapAgentV1(seed=123)
    # [4] -> []: len=1 (not prime), sum=4 (not prime).
    h = Handle(hid="H001", sent_sig="4", resp_sig="0", strength=0.5, hits=10)
    agent._handles.append(h)
    
    trainer = AggressiveTrainerV1(agent, partner_name="mixed", seed=123)
    sample = TrainingSample(sent=(4,), sent_sig="4")
    
    # Force uncertainty to trigger QUESTION lane (by setting threshold high)
    # We need at least 2 candidates for conflict or low margin to trigger QUESTION
    h2 = Handle(hid="H002", sent_sig="4", resp_sig="5", strength=0.49, hits=5)
    agent._handles.append(h2)
    
    # Margin is 0.01. conflict_margin is 0.1 by default.
    # predict() should return QUESTION.
    decision = agent.predict((4,))
    assert decision.lane == Lane.QUESTION
    
    initial_h1_hits = h.hits
    initial_h1_strength = h.strength
    
    # Run one round. Even though it's "uncertain", it should NOT update core weights because it's QUESTION lane.
    trainer.train_round(batch_size=1, drill_n=0, uncertainty_threshold=0.2, fixed_batch=[sample])
    
    assert h.hits == initial_h1_hits, "Hits should not change on QUESTION lane"
    assert h.strength == initial_h1_strength, "Strength should not change on QUESTION lane"

def test_no_updates_on_correct_confident_speak():
    """No updates on correct+confident SPEAK."""
    agent = BootstrapAgentV1(seed=123)
    h = Handle(hid="H001", sent_sig="4", resp_sig="0", strength=0.9, hits=10)
    agent._handles.append(h)
    
    trainer = AggressiveTrainerV1(agent, partner_name="mixed", seed=123)
    sample = TrainingSample(sent=(4,), sent_sig="4")
    
    initial_hits = h.hits
    initial_strength = h.strength
    
    # High strength, correct prediction. uncertainty_threshold=0.0.
    trainer.train_round(batch_size=1, drill_n=0, uncertainty_threshold=0.0, fixed_batch=[sample])
    
    assert h.hits == initial_hits
    assert h.strength == initial_strength

def test_updates_only_on_speak_wrong_with_single_label_oracle():
    """Updates only on SPEAK wrong with single-label oracle."""
    agent = BootstrapAgentV1(seed=123)
    # Mapping [1,1,1] -> [5,7] (len=3 is prime, sum=3 is prime) -> ambiguous
    # Mapping [1,1] -> [5] (len=2 is prime, sum=2 is prime) -> Wait, [1,1] sum is 2, len is 2. Both are prime.
    # [1,1] -> (5, 7)
    # [4] -> () (len=1 not prime, sum=4 not prime)
    # [1,4] -> (5) (len=2 is prime, sum=5 is prime) -> Wait, [1,4] sum is 5, len is 2. Both prime.
    # [4,6] -> (5) (len=2 is prime, sum=10 not prime) -> YES, this is single label.
    
    h = Handle(hid="H001", sent_sig="4,6", resp_sig="3", strength=0.5, hits=10)
    agent._handles.append(h)
    
    trainer = AggressiveTrainerV1(agent, partner_name="mixed", seed=123)
    # [4,6] in mixed partner should be (5,) because len=2 (prime) and sum=10 (not prime)
    sample = TrainingSample(sent=(4,6), sent_sig="4,6")
    
    # Ensure it SPEAKs
    decision = agent.predict((4,6))
    assert decision.lane == Lane.SPEAK
    
    metrics = trainer.train_round(batch_size=1, drill_n=0, uncertainty_threshold=0.0, fixed_batch=[sample])
    
    assert metrics["corrections"] == 1
    assert h.strength < 0.5 # Should be penalized because it was wrong (resp_sig="3" vs actual "5")

def test_no_updates_on_ambiguous_oracle():
    """If oracle has comma (e.g. "5,7") => treat as ambiguous => no core update."""
    agent = BootstrapAgentV1(seed=123)
    # We need a partner that returns multi-label.
    # Mixed partner with some specific input? 
    # Actually let's mock the partner or just use one that we know can be ambiguous.
    # For simplicity, I'll just check if AggressiveTrainerV1 handles oracle_ambiguous correctly.
    
    class MultiLabelPartner:
        def respond(self, sent):
            return (5, 7)
            
    trainer = AggressiveTrainerV1(agent, partner_name="mixed", seed=123)
    trainer.partner = MultiLabelPartner()
    
    sample = TrainingSample(sent=(1,), sent_sig="1")
    
    # Create a handle that matches one of them to see if it updates
    h = Handle(hid="H001", sent_sig="1", resp_sig="5", strength=0.5)
    agent._handles.append(h)
    
    metrics = trainer.train_round(batch_size=1, drill_n=0, uncertainty_threshold=0.0, fixed_batch=[sample])
    
    assert metrics["corrections"] == 0
    assert h.strength == 0.5, "Should not update core weights on ambiguous oracle"
