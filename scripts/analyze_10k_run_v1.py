import json
from pathlib import Path
import sys

def analyze_10k(file_path):
    p = Path(file_path)
    if not p.exists():
        print(f"File not found: {p}")
        return

    obj = json.loads(p.read_text(encoding="utf-8"))
    rows = obj.get("history", [])
    if not rows:
        print("No history found in JSON.")
        return

    n = len(rows)
    print(f"Analyzing {n} samples...")

    # Phase split (0-5000, 5000-10000)
    def acc_slice(a, b):
        sub = rows[a:b]
        xs = [r for r in sub if r.get("err") is not None]
        if not xs: return None
        ok = sum(1 for r in xs if float(r.get("err")) == 0.0)
        return ok, len(xs), ok/len(xs)

    for a, b, name in [(0, 5000, "Phase 0 (Mixed)"), (5000, 10000, "Phase 1 (Parity)")]:
        out = acc_slice(a, b)
        if out:
            ok, total, acc = out
            print(f"{name}: {ok}/{total} = {acc:.2%}")

    # Moving average for graph
    successes = [1 if float(r.get("err", 1.0)) == 0.0 else 0 for r in rows]
    window = 200
    rolling = []
    for i in range(len(successes)):
        start = max(0, i - window + 1)
        rolling.append(sum(successes[start:i+1]) / (i+1 - start))

    # ASCII Graph (compressed for 10k)
    print("\nSuccess Rate over Time (Rolling Avg Window 200)")
    cols = 60
    rows_g = 10
    step = n // cols

    for r in range(rows_g, 0, -1):
        threshold = r / rows_g
        line = f"{int(threshold*100):3d}% |"
        for c in range(cols):
            val = rolling[min(c * step, n - 1)]
            line += "#" if val >= threshold else "+" if val >= threshold - 0.05 else " "
        print(line)
    print("  0% |" + "-" * cols)
    print("     0" + " " * (cols // 2 - 2) + "5000" + " " * (cols // 2 - 3) + "10000")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_10k(sys.argv[1])
    else:
        # Latest 10k run
        analyze_10k("data/training_runs/run_20260106_025204.json")
