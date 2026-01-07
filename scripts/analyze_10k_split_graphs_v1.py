import json
from pathlib import Path
import sys

def generate_graph(data, title, window=100):
    n = len(data)
    # Successes (1 if error == 0.0)
    successes = [1 if float(r.get("err", 1.0)) == 0.0 else 0 for r in data]
    
    # Rolling average
    rolling = []
    for i in range(len(successes)):
        start = max(0, i - window + 1)
        rolling.append(sum(successes[start:i+1]) / (i+1 - start))
    
    print(f"\n{title} (Rolling Avg Window {window})")
    cols = 50
    rows_g = 10
    step = n // cols
    
    for r in range(rows_g, 0, -1):
        threshold = r / rows_g
        line = f"{int(threshold*100):3d}% |"
        for c in range(cols):
            idx = min(c * step, n - 1)
            val = rolling[idx]
            line += "#" if val >= threshold else "+" if val >= threshold - 0.05 else " "
        print(line)
    print("  0% |" + "-" * cols)
    print("     " + "0" + " " * (cols // 2 - 2) + str(n // 2) + " " * (cols // 2 - 3) + str(n))

def analyze_split_graphs(file_path):
    p = Path(file_path)
    if not p.exists():
        print(f"File not found: {p}")
        return

    obj = json.loads(p.read_text(encoding="utf-8"))
    rows = obj.get("history", [])
    if not rows:
        print("No history found in JSON.")
        return

    # Split into two phases
    phase0 = rows[0:5000]
    phase1 = rows[5000:10000]

    generate_graph(phase0, "Phase 0: Mixed Rule (Steps 0-5000)")
    generate_graph(phase1, "Phase 1: Parity Rule (Steps 5000-10000)")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/training_runs/run_20260106_025204.json"
    analyze_split_graphs(path)
