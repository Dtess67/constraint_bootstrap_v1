import sys
import os
from pathlib import Path

# Add src to sys.path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from constraint_bootstrap.bootstrap_agent_v1 import BootstrapAgentV1
from q_ternary.training.aggressive_trainer_v1 import AggressiveTrainerV1

def run_demo():
    seed = 123
    batch_size = 1000
    
    agent = BootstrapAgentV1(
        seed=seed,
        silence_penalty=0.02, # from common settings in previous issue, but description didn't specify. 
        # Actually description said:
        # thresholds: truth_min_to_speak=0.10, min_strength=0.10, eligibility_min_to_consider=0.10, uncertainty_threshold=0.40
        truth_min_to_speak=0.10,
        min_strength_to_predict=0.10,
        eligibility_min_to_consider=0.10,
        seed_proto_handles=True
    )
    
    trainer = AggressiveTrainerV1(agent, partner_name="mixed_shift", seed=seed)
    
    print(f"Starting Drift/Shift Demo (batch_size={batch_size}, seed={seed})")
    print(f"Phase switch index: 500")
    print("-" * 60)
    
    metrics = trainer.train_round(
        batch_size=batch_size,
        drill_n=0,
        uncertainty_threshold=0.40,
        question_budget_per_round=40,
        probe_after_budget=True
    )
    
    print("-" * 60)
    print(f"Round 01: Acc={metrics['accuracy']:.2%} | SP={metrics['speak_rate']:.2%} | QU={metrics['question_rate']:.2%} | Util={metrics['utility']:.2%}")
    print(f"          Drift Triggers: {metrics['drift_triggers']}")
    print(f"          Drift Probe Steps: {metrics['drift_probe_steps']}")
    if metrics['drift_trigger_indices']:
        print(f"          First Drift Trigger at index: {metrics['drift_trigger_indices'][0]}")
    
    print(f"          Na-Rate={metrics['na_rate']:.2%} | Silent-Rate={1.0 - metrics['speak_rate'] - metrics['question_rate'] - metrics['na_rate']:.2%}")
    
    metadata = {
        "batch": batch_size,
        "rounds": 1,
        "uncertainty_threshold": 0.40,
        "partner": "mixed_shift",
        "seed": seed,
        "question_budget_per_round": 40,
        "probe_after_budget": True,
        "drift_detector": "on"
    }
    
    json_path = trainer.save_run(metadata)
    print(f"Run saved to: {json_path}")

if __name__ == "__main__":
    run_demo()
