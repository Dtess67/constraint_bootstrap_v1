import sys
import os
from pathlib import Path

# Add src to sys.path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from constraint_bootstrap.bootstrap_agent_v1 import BootstrapAgentV1
from q_ternary.training.aggressive_trainer_v1 import AggressiveTrainerV1

def run_analysis_and_plot():
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
    
    # Success over time (1 if error == 0, else 0)
    successes = [1 if r.error == 0.0 else 0 for r in history]
    
    # Rolling average window 50
    window = 50
    rolling_success = []
    for i in range(len(successes)):
        start = max(0, i - window + 1)
        sub = successes[start:i+1]
        rolling_success.append(sum(sub) / len(sub))
    
    # ASCII Graph
    print("Success Rate over Time (Rolling Avg Window 50)")
    print("100% |")
    
    rows = 10
    cols = 50
    step_col = len(rolling_success) // cols
    
    for r in range(rows, 0, -1):
        threshold = r / rows
        line = f"{int(threshold*100):3d}% |"
        for c in range(cols):
            val = rolling_success[min(c * step_col, len(rolling_success)-1)]
            if val >= threshold:
                line += "#"
            elif val >= threshold - (1/(rows*2)):
                line += "+"
            else:
                line += " "
        print(line)
        
    print("  0% |" + "-" * cols)
    print("     " + "0" + " " * (cols // 2 - 2) + "500" + " " * (cols // 2 - 3) + "1000")
    print("     " + " (Step index) ")
    print("\n'#' = High Success Rate, '+' = Moderate, ' ' = Low")
    print("Note the shift at step 500 where success rate drops significantly.")

if __name__ == "__main__":
    run_analysis_and_plot()
