import pytest
from constraint_bootstrap.bootstrap_agent_v1 import BootstrapAgentV1, Handle
from q_ternary.lane_v1 import Lane
from q_ternary.training.aggressive_trainer_v1 import AggressiveTrainerV1, TrainingSample

def test_truth_never_increases_on_question_na_silent():
    """truth never increases on QUESTION/NA/SILENT (even with silence_penalty active)"""
    agent = BootstrapAgentV1(seed=123, silence_penalty=0.1, min_strength_to_predict=0.5)
    # [4,6] -> [5] in mixed partner (len=2, sum=10) - single label
    h = Handle(hid="H001", sent_sig="4,6", resp_sig="5", eligibility=0.3, truth=0.3)
    agent._handles.append(h)
    
    trainer = AggressiveTrainerV1(agent, partner_name="mixed", seed=123)
    sample = TrainingSample(sent=(4,6), sent_sig="4,6")
    
    # Check it is QUESTION (eligible but not strong enough to SPEAK)
    decision = agent.predict((4,6))
    assert decision.lane == Lane.QUESTION
    
    initial_truth = h.truth
    initial_eligibility = h.eligibility
    
    # Run round. Should trigger eligibility_nudge.
    metrics = trainer.train_round(batch_size=1, drill_n=0, uncertainty_threshold=0.0, fixed_batch=[sample])
    
    assert h.truth == initial_truth, "Truth should NOT increase on QUESTION lane"
    assert h.eligibility > initial_eligibility, "Eligibility SHOULD increase on QUESTION lane with silence_penalty"

def test_truth_updates_only_on_speak_wrong():
    """truth updates only occur on SPEAK wrong + trainable oracle"""
    agent = BootstrapAgentV1(seed=123, min_strength_to_predict=0.3)
    # [4,6] -> [5] in mixed partner
    h = Handle(hid="H001", sent_sig="4,6", resp_sig="3", eligibility=0.5, truth=0.5)
    agent._handles.append(h)
    
    trainer = AggressiveTrainerV1(agent, partner_name="mixed", seed=123)
    sample = TrainingSample(sent=(4,6), sent_sig="4,6")
    
    # Verify it SPEAKs and is WRONG
    decision = agent.predict((4,6))
    assert decision.lane == Lane.SPEAK
    assert decision.act == (3,)
    
    initial_truth = h.truth
    
    # Run round. Should trigger correction_truth.
    metrics = trainer.train_round(batch_size=1, drill_n=0, uncertainty_threshold=0.0, fixed_batch=[sample])
    
    assert metrics["corrections"] == 1
    assert h.truth < initial_truth, "Truth should decrease on WRONG SPEAK"
    
    # Now test CORRECT SPEAK
    # [4,6] -> [5]
    h.resp_sig = "5"
    h.truth = 0.5
    h.eligibility = 0.5
    
    initial_truth = h.truth
    trainer.train_round(batch_size=1, drill_n=0, uncertainty_threshold=0.0, fixed_batch=[sample])
    
    # Invariant: no updates on correct+confident SPEAK
    assert h.truth == initial_truth, "Truth should NOT change on correct+confident SPEAK"

def test_truth_decreases_on_mismatch_during_correction():
    """If we are SPEAKing and wrong, truth of the chosen handle should decrease."""
    agent = BootstrapAgentV1(seed=123, min_strength_to_predict=0.3)
    h = Handle(hid="H001", sent_sig="4,6", resp_sig="3", eligibility=0.5, truth=0.5)
    agent._handles.append(h)
    
    trainer = AggressiveTrainerV1(agent, partner_name="mixed", seed=123)
    sample = TrainingSample(sent=(4,6), sent_sig="4,6")
    
    initial_truth = h.truth
    trainer.train_round(batch_size=1, drill_n=0, uncertainty_threshold=0.0, fixed_batch=[sample])
    
    assert h.truth < initial_truth

def test_eligibility_nudge_on_silent_lane():
    """Eligibility can increase on SILENT lane with silence_penalty."""
    agent = BootstrapAgentV1(seed=123, silence_penalty=0.1, eligibility_min_to_consider=0.5)
    # Handle exists but eligibility is too low -> SILENT
    # [4,6] -> [5] single label
    h = Handle(hid="H001", sent_sig="4,6", resp_sig="5", eligibility=0.3, truth=0.5)
    agent._handles.append(h)
    
    trainer = AggressiveTrainerV1(agent, partner_name="mixed", seed=123)
    sample = TrainingSample(sent=(4,6), sent_sig="4,6")
    
    assert agent.predict((4,6)).lane == Lane.SILENT
    
    initial_eligibility = h.eligibility
    initial_truth = h.truth
    
    trainer.train_round(batch_size=1, drill_n=0, uncertainty_threshold=0.0, fixed_batch=[sample])
    
    assert h.eligibility > initial_eligibility
    assert h.truth == initial_truth
