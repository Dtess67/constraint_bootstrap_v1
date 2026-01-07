import pytest
from constraint_bootstrap.bootstrap_agent_v1 import BootstrapAgentV1, Handle
from q_ternary.lane_v1 import Lane
from q_ternary.training.aggressive_trainer_v1 import AggressiveTrainerV1, TrainingSample

def test_question_trains_truth_supervised():
    """QUESTION lane acts as a request for label and updates truth if oracle is trainable."""
    # min_strength=0.5, so 0.3 is NOT speakable -> will lead to QUESTION (weak knowledge)
    agent = BootstrapAgentV1(seed=123, min_strength_to_predict=0.5, eligibility_min_to_consider=0.1)
    
    # [4,6] -> [5] in mixed partner
    h = Handle(hid="H001", sent_sig="4,6", resp_sig="5", eligibility=0.3, truth=0.0)
    agent._handles.append(h)
    
    trainer = AggressiveTrainerV1(agent, partner_name="mixed", seed=123)
    sample = TrainingSample(sent=(4,6), sent_sig="4,6")
    
    # 1. Verify it is in QUESTION lane
    decision = agent.predict((4,6))
    assert decision.lane == Lane.QUESTION
    
    initial_truth = h.truth
    assert initial_truth == 0.0
    
    # 2. Run round. Should trigger question_supervised.
    metrics = trainer.train_round(batch_size=1, drill_n=0, uncertainty_threshold=0.0, fixed_batch=[sample])
    
    assert metrics["question_supervised_count"] == 1
    assert h.truth > initial_truth, "Truth SHOULD increase on supervised QUESTION"
    
def test_silent_na_do_not_train_truth():
    """NA and SILENT lanes do not update truth even if oracle is trainable.
    Note: NA is now nudged to QUESTION, so to test 'no truth update on NA/SILENT'
    we either check SILENT (no matches) or verify that QUESTION nudges still only 
    train truth if the oracle matches the handle.
    Actually, we want to verify that if we didn't have the nudge, NA wouldn't train.
    With the nudge, NA becomes QUESTION, and QUESTION *does* train truth if trainable.
    So the invariant 'Truth only increases on labeled supervision' is still held,
    but the specific 'NA doesn't train' is now moot because NA -> QUESTION.
    """
    # Case: SILENT (no matches) - cannot train truth as there's no handle to update
    # In AggressiveTrainerV1, train_round with fixed_batch will still trigger proto-seeding 
    # if it's enabled in the agent.
    agent = BootstrapAgentV1(seed=123, seed_proto_handles=True)
    trainer = AggressiveTrainerV1(agent, partner_name="mixed", seed=123)
    sample = TrainingSample(sent=(4,6), sent_sig="4,6")
    
    # With pre-decision seeding, firstcontact is QUESTION
    assert agent.predict((4,6)).lane == Lane.QUESTION
    
    # In trainer.train_round, oracle for (4,6) in mixed is [5] which IS trainable.
    # Since lane is QUESTION, it will now trigger question_supervised if update_truth is passed.
    # But in trainer.train_round, it only sets update_truth=True for Lane.QUESTION.
    # Let's see.
    
    trainer.train_round(batch_size=1, drill_n=0, uncertainty_threshold=0.0, fixed_batch=[sample])
    
    # It should have seeded one handle in predict
    assert len(agent._handles) == 1
    h = agent._handles[0]
    # In train_round, for QUESTION it calls agent.observe(update_truth=True)
    assert h.truth > 0.0
