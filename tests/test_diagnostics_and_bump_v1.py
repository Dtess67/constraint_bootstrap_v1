import pytest
from constraint_bootstrap.bootstrap_agent_v1 import BootstrapAgentV1, Handle
from q_ternary.lane_v1 import Lane
from q_ternary.training.aggressive_trainer_v1 import AggressiveTrainerV1, TrainingSample

def test_question_eligibility_bump():
    """Verify that QUESTION-supervised events can bump eligibility significantly."""
    bump = 0.5
    agent = BootstrapAgentV1(seed=123, question_eligibility_bump=bump, min_strength_to_predict=0.8)
    # [4,6] -> [5] (single label)
    h = Handle(hid="H001", sent_sig="4,6", resp_sig="5", eligibility=0.2, truth=0.7)
    agent._handles.append(h)
    
    trainer = AggressiveTrainerV1(agent, partner_name="mixed", seed=123)
    sample = TrainingSample(sent=(4,6), sent_sig="4,6")
    
    # Check it is QUESTION (truth=0.7 < gate=0.8)
    decision = agent.predict((4,6))
    assert decision.lane == Lane.QUESTION
    
    initial_eligibility = h.eligibility
    initial_truth = h.truth
    
    # Run round. Should trigger question_supervised.
    trainer.train_round(batch_size=1, drill_n=0, uncertainty_threshold=0.0, fixed_batch=[sample])
    
    # Normal update + bump
    # Normal update for matched is +0.08
    expected_min_elig = initial_eligibility + 0.08 + bump
    assert h.eligibility >= expected_min_elig
    
    # Truth should only have the normal +0.08 update, NOT the bump
    assert h.truth == pytest.approx(initial_truth + 0.08)

def test_diagnostics_speakable_count():
    """Verify diagnostics for speakable and gated handles."""
    agent = BootstrapAgentV1(seed=123, min_strength_to_predict=0.5, truth_min_to_speak=0.4)
    
    # H1: Speakable (truth >= 0.4 AND strength=min(elig, truth) >= 0.5)
    h1 = Handle(hid="H001", sent_sig="1", resp_sig="1", eligibility=0.6, truth=0.6)
    
    # H2: Gated (truth >= 0.4 but elig=0.3 -> strength=0.3 < 0.5)
    h2 = Handle(hid="H002", sent_sig="2", resp_sig="2", eligibility=0.3, truth=0.6)
    
    # H3: Not even candidate for speaking (truth < 0.4)
    h3 = Handle(hid="H003", sent_sig="3", resp_sig="3", eligibility=0.9, truth=0.2)
    
    agent._handles.extend([h1, h2, h3])
    
    trainer = AggressiveTrainerV1(agent, partner_name="mixed", seed=123)
    
    metrics = trainer.train_round(batch_size=1, drill_n=0, uncertainty_threshold=0.0)
    
    assert metrics["speakable_handle_count"] == 1
    assert metrics["gated_by_eligibility_count"] == 1
