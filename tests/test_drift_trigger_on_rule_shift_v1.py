import pytest
import sys
from pathlib import Path

# Add src to sys.path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from constraint_bootstrap.bootstrap_agent_v1 import BootstrapAgentV1
from q_ternary.training.aggressive_trainer_v1 import AggressiveTrainerV1

def test_drift_trigger_on_rule_shift_v1():
    seed = 123
    batch_size = 1000
    
    agent = BootstrapAgentV1(
        seed=seed,
        truth_min_to_speak=0.10,
        min_strength_to_predict=0.10,
        eligibility_min_to_consider=0.10,
        seed_proto_handles=True
    )
    
    trainer = AggressiveTrainerV1(agent, partner_name="mixed_shift", seed=seed)
    
    metrics = trainer.train_round(
        batch_size=batch_size,
        drill_n=0,
        uncertainty_threshold=0.40,
        question_budget_per_round=40,
        probe_after_budget=True
    )
    
    # Assert that running the demo logic with seed=123 triggers >= 1 drift event 
    # after the phase switch (index >= 500).
    assert metrics['drift_triggers'] >= 1
    
    # Check if at least one trigger happened after index 500
    triggers_after_500 = [idx for idx in metrics['drift_trigger_indices'] if idx >= 500]
    assert len(triggers_after_500) >= 1
