import pytest
import os
from constraint_bootstrap.bootstrap_agent_v1 import BootstrapAgentV1, Handle
from q_ternary.training.aggressive_trainer_v1 import AggressiveTrainerV1, TrainingSample
from q_ternary.lane_v1 import Lane

def test_question_budget_zero_behavior_unchanged():
    # With budget=0, behavior should be same as before
    agent = BootstrapAgentV1(seed=42)
    # Seed a handle so it has something to question/speak about
    h = Handle(hid="H001", sent_sig="1,2", resp_sig="7", truth=0.1, eligibility=1.0)
    agent._handles.append(h)
    
    trainer = AggressiveTrainerV1(agent, partner_name="sumprime", seed=42)
    
    # Force some questions by setting uncertainty threshold high
    # Use a sample that matches our handle
    sample = TrainingSample(sent=(1, 2), sent_sig="1,2")
    metrics = trainer.train_round(batch_size=1, fixed_batch=[sample], drill_n=0, uncertainty_threshold=1.0, question_budget_per_round=0)
    
    assert metrics["question_rate"] > 0
    assert metrics["probe_count"] == 0
    assert metrics["question_budget_hit_count"] == 0

def test_question_budget_triggers_probes():
    agent = BootstrapAgentV1(seed=42)
    # Seed some handles so it has something to question/speak about
    for i in range(10):
        h = Handle(hid=f"H{i:03d}", sent_sig=str(i), resp_sig="7", truth=0.1, eligibility=1.0)
        agent._handles.append(h)
        
    trainer = AggressiveTrainerV1(agent, partner_name="sumprime", seed=42)
    
    # Small budget=2, probe_after_budget=True
    # batch_size=10, should hit budget quickly
    # Use fixed batch to ensure they want to be questions
    batch = [TrainingSample(sent=(i,), sent_sig=str(i)) for i in range(10)]
    metrics = trainer.train_round(
        batch_size=10, 
        fixed_batch=batch,
        drill_n=0, 
        uncertainty_threshold=1.0, 
        question_budget_per_round=2,
        probe_after_budget=True
    )
    
    assert metrics["question_budget_hit_count"] > 0
    assert metrics["probe_count"] > 0
    # Exactly 2 questions should have been allowed
    # Wait, some samples might naturally be SPEAK if they are strong.
    # But with uncertainty_threshold=1.0, most should want to be QUESTION.
    # Actually, AggressiveTrainerV1.train_round loops through the batch.
    # The first 2 that WANT to be QUESTION will be allowed.
    # The subsequent ones that WANT to be QUESTION will hit budget.
    
    # Verify that we have some Lane.SPEAK with probe=True in history
    probes = [r for r in trainer.history if r.decision.meta.get("probe")]
    assert len(probes) == metrics["probe_count"]
    for p in probes:
        assert p.decision.lane == Lane.SPEAK

def test_probe_never_increases_truth_directly():
    # Requirement: "Truth must never increase from SILENT/NA/QUESTION nudges or probe guesses."
    # We need to ensure that if a probe is CORRECT, truth doesn't increase.
    # If a probe is WRONG/UNCERTAIN, it MAY trigger a correction (supervised truth update).
    
    agent = BootstrapAgentV1(seed=42, truth_min_to_speak=0.5)
    trainer = AggressiveTrainerV1(agent, partner_name="sumprime", seed=42)
    
    # 1. Create a situation where we have a handle with some truth but not enough to speak
    # We can use a fixed batch to ensure consistency
    sample = TrainingSample(sent=(1, 2), sent_sig="1,2")
    
    # Pre-seed a handle for (1, 2) -> (7,)
    # We'll manually add it to the agent
    from constraint_bootstrap.bootstrap_agent_v1 import Handle
    h = Handle(hid="H001", sent_sig="1,2", resp_sig="7", truth=0.1, eligibility=1.0)
    agent._handles.append(h)
    
    initial_truth = h.truth
    
    # 2. Run a round with budget hit and probe
    # If probe is correct (7 is the response for sum 1+2=3 which is prime), 
    # and update_truth=False, truth should stay same.
    
    metrics = trainer.train_round(
        batch_size=1,
        fixed_batch=[sample],
        drill_n=0,
        uncertainty_threshold=1.0, # Force QUESTION
        question_budget_per_round=0, 
    )
    # First sample should be QUESTION, questions_used becomes 1.
    
    sample2 = TrainingSample(sent=(2, 3), sent_sig="2,3") # sum=5 is prime -> (7,)
    h2 = Handle(hid="H002", sent_sig="2,3", resp_sig="7", truth=0.1, eligibility=1.0)
    agent._handles.append(h2)
    
    metrics = trainer.train_round(
        batch_size=2,
        fixed_batch=[sample, sample2],
        drill_n=0,
        uncertainty_threshold=1.0,
        question_budget_per_round=1,
        probe_after_budget=True
    )
    
    # One should be QUESTION, one should be PROBE.
    # The PROBE one should NOT have increased truth if it was correct.
    # Sum(1,2)=3 (prime), Sum(2,3)=5 (prime). Both are single pulse.
    
    # Find which one was probe
    probe_res = [r for r in trainer.history if r.decision.meta.get("probe")][0]
    probe_h = [h for h in agent.handles if h.sent_sig == probe_res.sample.sent_sig][0]
    
    # Since it was correct and probe, it should NOT have triggered a correction.
    # And AggressiveTrainerV1.train_round calls agent.observe(..., update_truth=False) for probes.
    # Handle.update(matched=True, update_truth=False) should increase hits but NOT truth.
    
    assert probe_res.error == 0.0
    assert probe_res.update_type == "probe_speak"
    assert probe_h.truth == 0.1 # Should remain unchanged
    assert probe_h.hits > 0

def test_probe_triggers_correction_if_wrong():
    # Requirement 3: "If oracle/correction is available, a wrong/uncertain probe may trigger a proper supervised truth update"
    agent = BootstrapAgentV1(seed=42, truth_min_to_speak=0.5)
    
    # Handle with WRONG prediction
    sample = TrainingSample(sent=(1, 2), sent_sig="1,2") # Sum is 3, should be (7,)
    h = Handle(hid="H001", sent_sig="1,2", resp_sig="4", truth=0.1, eligibility=1.0) # 4 is wrong
    agent._handles.append(h)
    
    # Another handle to hit budget
    h_q = Handle(hid="H002", sent_sig="2,3", resp_sig="7", truth=0.1, eligibility=1.0)
    agent._handles.append(h_q)
    sample_q = TrainingSample(sent=(2, 3), sent_sig="2,3")
    
    trainer = AggressiveTrainerV1(agent, partner_name="sumprime", seed=42)
    
    metrics = trainer.train_round(
        batch_size=2,
        fixed_batch=[sample_q, sample],
        drill_n=0,
        uncertainty_threshold=1.0,
        question_budget_per_round=1,
        probe_after_budget=True
    )
    
    # sample should be probe and should be wrong
    probes = [r for r in trainer.history if r.decision.meta.get("probe")]
    assert len(probes) > 0
    probe_res = probes[0]
    assert probe_res.sample.sent_sig == "1,2"
    assert probe_res.error > 0.0
    
    # In BootstrapAgentV1.observe:
    # 1. Identify matching candidates for competition
    # 2. allowed = candidates
    # 3. for h_cand in allowed:
    #    if h_cand.resp_sig == recv_s: h_cand.update(matched=True, ...)
    #    else: h_cand.update(matched=False, ...)
    # h.resp_sig is "4", recv_s is "7". So it should call h.update(matched=False, update_truth=True).
    # Handle.update(matched=False, update_truth=True) calls self.truth = max(0.0, self.truth - 0.12).
    # So if initial truth was 0.1, it should become 0.0.
    
    # BUT we want truth to INCREASE on correction.
    # Truth only increases if h_cand.resp_sig == recv_s.
    # On a WRONG probe, the current handle is WRONG.
    # AggressiveTrainerV1.train_round calls self.agent.observe(sample.sent, res.actual, learn=True, update_truth=True)
    # This should find the CORRECT handle (if it exists) and update it, 
    # AND find the WRONG handle and penalize it.
    
    # In this test, we have h (wrong) and h_q (unrelated).
    # observe() will not find a handle for (1,2)->(7,).
    # BUT it will see (1,2)->(7,) enough times (promote_threshold) to create it.
    # Or if we want to see truth increase on an existing handle, we need a handle for the CORRECT mapping.
    
    # Let's add a handle for the CORRECT mapping with low truth.
    h_correct = Handle(hid="H003", sent_sig="1,2", resp_sig="7", truth=0.1, eligibility=1.0)
    agent._handles.append(h_correct)
    
    # Reset seen counts so we don't trigger promotion again if not needed
    agent._seen_counts = {}
    
    metrics = trainer.train_round(
        batch_size=2,
        fixed_batch=[sample_q, sample],
        drill_n=0,
        uncertainty_threshold=1.0,
        question_budget_per_round=1,
        probe_after_budget=True
    )
    
    # Check if ANY correction happened for h_correct
    # In the second sample of train_round:
    # res.decision.lane is SPEAK (probe)
    # res.error > 0.0 (because it predicted "4" and actual is "7")
    # should_correct_truth is True
    # agent.observe(..., update_truth=True) is called.
    
    # print(f"DEBUG: h_correct.truth after train_round: {h_correct.truth}")
    # print(f"DEBUG: h_correct.hits after train_round: {h_correct.hits}")
    assert h_correct.hits > 0
    assert probe_res.corrected == True
    assert probe_res.update_type == "correction_truth_probe"
