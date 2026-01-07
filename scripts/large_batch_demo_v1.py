import sys
import os
from pathlib import Path

# Add src to sys.path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from constraint_bootstrap.bootstrap_agent_v1 import BootstrapAgentV1
from q_ternary.training.aggressive_trainer_v1 import AggressiveTrainerV1

def run_large_demo():
    seed = 123
    batch_size = 10000
    
    agent = BootstrapAgentV1(
        seed=seed,
        silence_penalty=0.02,
        truth_min_to_speak=0.10,
        min_strength_to_predict=0.10,
        eligibility_min_to_consider=0.10,
        seed_proto_handles=True
    )
    
    trainer = AggressiveTrainerV1(agent, partner_name="mixed_shift_large", seed=seed)
    
    print(f"Starting Large Drift/Shift Demo (batch_size={batch_size}, seed={seed})")
    print(f"Phase switch index: 5000")
    print("-" * 60)
    
    metrics = trainer.train_round(
        batch_size=batch_size,
        drill_n=0,
        uncertainty_threshold=0.40,
        question_budget_per_round=40,
        probe_after_budget=True
    )
    
    print("-" * 60)
    print(f"Round 01: Acc={metrics['accuracy']:.2%} | SP={metrics['speak_rate']:.2%} | QU={metrics['question_rate']:.2%}")
    print(f"          Drift Triggers: {metrics['drift_triggers']}")
    print(f"          Drift Probe Steps: {metrics['drift_probe_steps']}")
    
    metadata = {
        "batch": batch_size,
        "rounds": 1,
        "uncertainty_threshold": 0.40,
        "partner": "mixed_shift_large",
        "seed": seed,
        "question_budget_per_round": 40,
        "probe_after_budget": True,
        "drift_detector": "on"
    }
    
    json_path = trainer.save_run(metadata)
    print(f"Run saved to: {json_path}")

if __name__ == "__main__":
    run_large_demo()
