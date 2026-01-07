import os
import time
import pytest
from constraint_bootstrap.bootstrap_agent_v1 import BootstrapAgentV1
from q_ternary.training.aggressive_trainer_v1 import AggressiveTrainerV1, TrainingSample
from q_ternary.lane_v1 import Lane

def test_question_budget_enforcement_no_probe():
    """Verify that with budget=10 and probe_after_budget OFF, questions are blocked after 10."""
    # Ensure no handles exist to avoid SPEAK decisions
    agent = BootstrapAgentV1(seed=123, min_strength_to_predict=0.9, seed_proto_handles=False)
    trainer = AggressiveTrainerV1(agent, seed=123)
    
    # We need to make sure these 20 samples would normally be QUESTIONs.
    # In AggressiveTrainerV1.train_round, a sample becomes QUESTION if:
    # (res.oracle_ambiguous or is_uncertain) and res.decision.lane != Lane.SPEAK
    # and question_preferred is True.
    
    # With no handles and seed_proto_handles=False, predict returns SILENT.
    # uncertainty will be 1.0 (max) because no matches.
    # If uncertainty_threshold is 1.0, is_uncertain might be False (1.0 < 1.0 is False).
    # Let's use uncertainty_threshold=2.0 to be sure.
    
    batch = [TrainingSample(sent=(1, i), sent_sig=f"1,{i}") for i in range(20)]
    
    metrics = trainer.train_round(
        batch_size=20,
        drill_n=0,
        uncertainty_threshold=2.0,
        fixed_batch=batch,
        question_budget_per_round=10,
        probe_after_budget=False,
        question_preferred=True
    )
    
    # With question_preferred=True and is_uncertain=True, 
    # the SILENT decisions should be forced to QUESTION.
    # Then the budget should block them after 10.
    
    assert metrics["question_rate"] == 0.5 # 10/20
    assert metrics["question_budget_hit_count"] == 10
    assert metrics["questions_blocked_count"] == 10
    
    # Verify that blocked questions went to NA
    na_results = [r for r in trainer.history if r.decision.lane == Lane.NA]
    assert len(na_results) == 10

def test_probe_metrics_separation():
    """Verify that probe SPEAK events do not affect speak_precision."""
    agent = BootstrapAgentV1(seed=123, min_strength_to_predict=0.5, promote_threshold=1)
    trainer = AggressiveTrainerV1(agent, partner_name="sumprime", seed=123)
    
    # 1. Add a correct handle so we have a non-probe SPEAK
    # sumprime responds (7,) for (1,1) because 1+1=2 is prime.
    agent.observe((1,1), (7,), learn=True, update_truth=True)
    # Ensure it's strong enough to SPEAK
    h = [h for h in agent._handles if h.sent_sig == "1,1"][0]
    h.truth = 1.0
    h.eligibility = 1.0
    
    # 2. Setup (3,3) for PROBE
    # Give it a handle so it can probe, but make it wrong. 
    # sumprime for (3,3) is silence () because 3+3=6 is not prime.
    agent.observe((3,3), (10,), learn=True, update_truth=False)
    h3 = [h for h in agent._handles if h.sent_sig == "3,3"][0]
    h3.truth = 0.1 # Keep it low so it's not a regular SPEAK
    h3.eligibility = 0.5
    
    # (2,2) for sumprime is silence () because 2+2=4 is not prime.
    
    # We need to make sure (2,2) becomes a QUESTION so it uses the budget.
    # Currently it's SILENT because no matches.
    # We can use seed_proto_handles=True to nudge it to QUESTION.
    agent.seed_proto_handles = True
    
    # Setup batch:
    # 1. (2,2) -> QUESTION (nudge from SILENT) -> Budget Used=1
    # 2. (3,3) -> QUESTION (weak knowledge) -> Budget Hit -> PROBE SPEAK
    # 3. (1,1) -> Regular SPEAK
    batch = [
        TrainingSample(sent=(2,2), sent_sig="2,2"),
        TrainingSample(sent=(3,3), sent_sig="3,3"),
        TrainingSample(sent=(1,1), sent_sig="1,1"),
    ]
    
    metrics = trainer.train_round(
        batch_size=3,
        drill_n=0,
        uncertainty_threshold=1.0,
        fixed_batch=batch,
        question_budget_per_round=1,
        probe_after_budget=True
    )
    
    # Debug: Check lanes
    for r in trainer.history:
        print(f"Sent: {r.sample.sent_sig} | Lane: {r.decision.lane} | Meta: {r.decision.meta}")
    
    # Expected results:
    # (2,2) -> QUESTION (budget used = 1)
    # (3,3) -> PROBE SPEAK (budget hit, probe=True) - Error > 0
    # (1,1) -> SPEAK (Regular) - Error = 0
    
    # The issue might be that (3,3) is NOT recognized as a would-be QUESTION.
    # It needs to be uncertain OR oracle_ambiguous.
    # uncertainty_threshold is 1.0. 
    # compute_uncertainty for (3,3) with one handle (10,) returns 0.1? No, 
    # it returns margin = s1 - s2. s1=min(0.5, 0.1)=0.1. s2=0. Margin=0.1.
    # 0.1 < 1.0 is True. So it IS uncertain.
    
    # Wait, in train_round:
    # if (res.oracle_ambiguous or is_uncertain) and res.decision.lane != Lane.SPEAK:
    # (3,3) with h3.truth=0.1 and truth_min_to_speak=0.35 (default) will be Lane.QUESTION.
    
    assert metrics["question_rate"] == 1/3 # (2,2)
    assert metrics["probe_count"] == 1 # (3,3)
    assert metrics["speak_non_probe_count"] == 1 # (1,1)
    assert metrics["precision"] == 1.0 # Only (1,1) counts
    assert metrics["probe_precision"] == 0.0 # (3,3) is wrong (10 vs ())
    assert metrics["speak_rate"] == 2/3 # (3,3) and (1,1) are both SPEAK lane
    
def test_save_run_uniqueness():
    """Verify that save_run does not overwrite files and uses seconds."""
    from constraint_bootstrap.bootstrap_agent_v1 import BootstrapAgentV1
    from q_ternary.training.aggressive_trainer_v1 import AggressiveTrainerV1
    agent = BootstrapAgentV1(seed=123)
    trainer = AggressiveTrainerV1(agent, seed=123)
    
    os.makedirs("data/training_runs", exist_ok=True)
    
    f1 = trainer.save_run({"test": 1})
    # Small sleep to ensure different second if possible, but the uniqueness check should handle it anyway
    f2 = trainer.save_run({"test": 2})
    
    assert f1 != f2
    assert os.path.exists(f1)
    assert os.path.exists(f2)
    
    # Cleanup
    if os.path.exists(f1): os.remove(f1)
    if os.path.exists(f2): os.remove(f2)
