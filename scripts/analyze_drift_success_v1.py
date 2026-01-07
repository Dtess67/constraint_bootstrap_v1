import sys
import os
from pathlib import Path

# Add src to sys.path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from constraint_bootstrap.bootstrap_agent_v1 import BootstrapAgentV1
from q_ternary.training.aggressive_trainer_v1 import AggressiveTrainerV1

def run_analysis():
    seed = 123
    batch_size = 1000
    
    agent = BootstrapAgentV1(
        seed=seed,
        silence_penalty=0.02,
        truth_min_to_speak=0.10,
        min_strength_to_predict=0.10,
        eligibility_min_to_consider=0.10,
        seed_proto_handles=True
    )
    
    trainer = AggressiveTrainerV1(agent, partner_name="mixed_shift", seed=seed)
    
    trainer.train_round(
        batch_size=batch_size,
        drill_n=0,
        uncertainty_threshold=0.40,
        question_budget_per_round=40,
        probe_after_budget=True
    )
    
    history = trainer.history
    
    # Analyzing success rate (error == 0.0) for the first 500 and second 500
    first_500 = history[:500]
    second_500 = history[500:1000]
    
    def get_success_rate(results):
        # success = error == 0.0
        successes = sum(1 for r in results if r.error == 0.0)
        return successes / len(results) if results else 0.0

    sr_first = get_success_rate(first_500)
    sr_second = get_success_rate(second_500)
    
    print(f"First 500 success rate: {sr_first:.2%}")
    print(f"Second 500 success rate: {sr_second:.2%}")

if __name__ == "__main__":
    run_analysis()
