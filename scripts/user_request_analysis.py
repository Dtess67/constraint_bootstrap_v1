import json
from pathlib import Path
import sys

# Standardizing to the path used in the user's request
p = Path(r"data/training_runs/run_20260106_024549.json")
if not p.exists():
    print(f"File not found: {p}")
    sys.exit(1)

obj = json.loads(p.read_text(encoding="utf-8"))

# Find the list of per-sample result dicts (robust to key name changes)
candidates = []
for k in ("results", "history", "samples", "events", "last_results"):
    v = obj.get(k)
    if isinstance(v, list) and v and isinstance(v[0], dict):
        candidates.append((k, v))

if not candidates:
    print("No per-sample list found in JSON. Keys:", list(obj.keys()))
    sys.exit(2)

key, rows = max(candidates, key=lambda kv: len(kv[1]))
print(f"Using list key: {key}  (n={len(rows)})")

def get_err(r):
    # err might be stored as 'err' or 'error'
    if "err" in r: return r["err"]
    if "error" in r: return r["error"]
    return None

def get_phase(r, i):
    # Prefer explicit phase metadata if present, else infer by index split at 500.
    meta = r.get("meta") or {}
    if "phase" in meta: return meta["phase"]
    return 0 if i < 500 else 1

phase_counts = {}  # phase -> [correct, total]
for i, r in enumerate(rows):
    err = get_err(r)
    if err is None:
        continue
    ph = get_phase(r, i)
    if ph not in phase_counts:
        phase_counts[ph] = [0, 0]
    phase_counts[ph][1] += 1
    if float(err) == 0.0:
        phase_counts[ph][0] += 1

for ph in sorted(phase_counts.keys()):
    correct, total = phase_counts[ph]
    if total == 0:
        continue
    print(f"Phase {ph}: {correct}/{total} = {correct/total:.2%}")

# Also print a simple first-half / second-half split by index (if we have >=1000 rows)
if len(rows) >= 1000:
    def acc_slice(a, b):
        sub = rows[a:b]
        xs = [r for r in sub if get_err(r) is not None]
        if not xs: return None
        ok = sum(1 for r in xs if float(get_err(r)) == 0.0)
        return ok, len(xs), ok/len(xs)
    for a,b,name in [(0,500,"First 500"), (500,1000,"Second 500")]:
        out = acc_slice(a,b)
        if out:
            ok,n,acc = out
            print(f"{name}: {ok}/{n} = {acc:.2%}")
else:
    print(f"Note: Only {len(rows)} samples available in JSON (history truncated to last_results).")
