import os
import sys

# Ensure project root is in path
sys.path.append(os.path.abspath("src"))

from constraint_bootstrap.bootstrap_agent_v1 import BootstrapAgentV1
from q_ternary.training.aggressive_trainer_v1 import AggressiveTrainerV1

def run_verification():
    print("=" * 80)
    print("LONG-RUN VERIFICATION: QUESTION TREND OVER 20 ROUNDS")
    print("Goal: QUESTION starts high then falls as truth accumulates.")
    print("=" * 80)

    # Configure agent with cooldown and seeding
    agent = BootstrapAgentV1(
        seed=123,
        seed_proto_handles=True,
        seed_eligibility=0.25,
        question_cooldown_n=10, # Shorter cooldown to allow more questions
        min_strength_to_predict=0.35,
        truth_min_to_speak=0.35,
        eligibility_min_to_consider=0.25,
        silence_penalty=0.02,
        question_eligibility_bump=0.2 # Bigger bump
    )

    trainer = AggressiveTrainerV1(agent, partner_name="mixed", seed=123)

    print(f"{'Round':<6} | {'Acc':<6} | {'SP':<6} | {'QU':<6} | {'AvgTruth':<8} | {'AvgElig':<8} | {'Speakable':<10} | {'Gated':<6}")
    print("-" * 90)

    for r in range(1, 21):
        metrics = trainer.train_round(
            batch_size=200,
            drill_n=0,
            uncertainty_threshold=0.4,
            question_credit=0.25,
            question_preferred=True
        )

        print(f"{r:02d}     | {metrics['accuracy']:.2%} | {metrics['speak_rate']:.2%} | {metrics['question_rate']:.2%} | {metrics['avg_truth']:.3f}    | {metrics['avg_eligibility']:.3f} | {metrics['speakable_handle_count']:<10} | {metrics['gated_by_eligibility_count']}")

    print("=" * 80)
    print("Verification complete.")

if __name__ == "__main__":
    run_verification()
