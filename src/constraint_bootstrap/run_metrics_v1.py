from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from .run_summary_v1 import parse_steps, StepRow
from q_ternary.lane_v1 import Lane

def parse_csv(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def sig_to_set(sig: str) -> Set[int]:
    """Parse '[5 7]' or '[5]' or '[]' into {5, 7}. Special case [999] for questions."""
    s = sig.strip()
    if s == "[]" or s == "∅":
        return set()
    # Remove brackets
    s = s.strip("[]")
    if not s:
        return set()
    parts = s.split()
    if "999" in parts:
        return {999}
    return {int(p) for p in parts if p.isdigit()}

def compute_metrics(steps: List[Dict[str, Any]], question_credit: float = 0.25) -> Dict[str, Any]:
    if not steps:
        return {}

    n = len(steps)
    exact_matches = 0
    nonempty_steps = 0
    nonempty_exact_matches = 0
    act_empty_count = 0
    
    # For Bitwise
    tp5, fp5, fn5 = 0, 0, 0
    tp7, fp7, fn7 = 0, 0, 0
    
    # Confusion matrix labels
    classes = ["[]", "[5]", "[7]", "[5,7]"]
    
    speak_count = 0
    speak_correct = 0
    question_count = 0

    def canonical(sig: str) -> str:
        vals = sorted(list(sig_to_set(sig)))
        if not vals: return "[]"
        return "[" + " ".join(str(v) for v in vals) + "]"

    # Re-normalize classes for confusion matrix
    norm_classes = [canonical(c) for c in classes]
    # Lane labels for confusion matrix
    lane_labels = ["SPEAK", "QUESTION", "NA", "SILENT"]
    final_labels = norm_classes + lane_labels
    seen = set()
    dedup_labels = []
    for l in final_labels:
        if l not in seen:
            dedup_labels.append(l)
            seen.add(l)
    
    norm_class_map = {c: i for i, c in enumerate(dedup_labels)}
    num_labels = len(dedup_labels)
    conf_matrix = [[0]*num_labels for _ in range(num_labels)]

    for s in steps:
        pred_sig = s.get("pred_sig", "[]")
        act_sig = s.get("act_sig", "[]")
        lane = s.get("lane", "").upper()
        
        if not lane:
            # Infer lane if missing (for legacy logs)
            p_set = sig_to_set(pred_sig)
            if p_set == {999}:
                lane = "QUESTION"
            elif p_set:
                lane = "SPEAK"
            else:
                lane = "SILENT"

        p_set = sig_to_set(pred_sig)
        a_set = sig_to_set(act_sig)
        
        is_exact = (p_set == a_set)
        
        if lane == "SPEAK":
            speak_count += 1
            if is_exact:
                speak_correct += 1
        elif lane == "QUESTION":
            question_count += 1

        if is_exact:
            exact_matches += 1
            
        is_act_empty = (len(a_set) == 0)
        if is_act_empty:
            act_empty_count += 1
        else:
            nonempty_steps += 1
            if is_exact:
                nonempty_exact_matches += 1
        
        # Bitwise (only for SPEAK lane or legacy)
        if lane == "SPEAK" or not s.get("lane"):
            # Bitwise 5
            p5 = (5 in p_set)
            a5 = (5 in a_set)
            if p5 and a5: tp5 += 1
            elif p5 and not a5: fp5 += 1
            elif not p5 and a5: fn5 += 1
            
            # Bitwise 7
            p7 = (7 in p_set)
            a7 = (7 in a_set)
            if p7 and a7: tp7 += 1
            elif p7 and not a7: fp7 += 1
            elif not p7 and a7: fn7 += 1
        
        # Confusion matrix
        # Map actual signature to norm_classes
        a_canon = canonical(act_sig)
        # Map prediction to either SPEAK:sig or QUESTION etc.
        if lane == "SPEAK":
            p_label = canonical(pred_sig)
        else:
            p_label = lane

        if a_canon in norm_class_map and p_label in norm_class_map:
            conf_matrix[norm_class_map[a_canon]][norm_class_map[p_label]] += 1

    acc_exact = exact_matches / n
    acc_nonempty_exact = nonempty_exact_matches / nonempty_steps if nonempty_steps > 0 else 0.0
    p_act_empty = act_empty_count / n
    
    speak_rate = speak_count / n
    question_rate = question_count / n
    precision = speak_correct / speak_count if speak_count > 0 else 0.0
    utility = speak_rate * precision + question_rate * question_credit

    def f1(tp, fp, fn):
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_val = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return prec, rec, f1_val

    p5, r5, f5 = f1(tp5, fp5, fn5)
    p7, r7, f7 = f1(tp7, fp7, fn7)

    return {
        "steps": n,
        "acc_exact": acc_exact,
        "acc_nonempty_exact": acc_nonempty_exact,
        "p_act_empty": p_act_empty,
        "acc_always_silent": p_act_empty,
        "skill_over_silence": acc_exact - p_act_empty,
        "speak_rate": speak_rate,
        "question_rate": question_rate,
        "precision": precision,
        "utility": utility,
        "bitwise": {
            "5": {"precision": p5, "recall": r5, "f1": f5},
            "7": {"precision": p7, "recall": r7, "f1": f7},
        },
        "confusion_matrix": {
            "labels": dedup_labels,
            "matrix": conf_matrix
        }
    }

def print_report(name: str, m: Dict[str, Any]):
    print(f"--- Metrics Report: {name} ---")
    print(f"Steps: {m['steps']}")
    print(f"Overall Exact Match:      {m['acc_exact']:.4f}")
    print(f"Response Efficiency (NE): {m['acc_nonempty_exact']:.4f}")
    print(f"Silence Baseline:         {m['acc_always_silent']:.4f}")
    print(f"Skill over Silence:       {m['skill_over_silence']:.4f}")
    print(f"Speak-rate:               {m.get('speak_rate', 0):.4f}")
    print(f"Question-rate:            {m.get('question_rate', 0):.4f}")
    print(f"Precision (SPEAK):        {m.get('precision', 0):.4f}")
    print(f"Utility (v1.1):           {m.get('utility', 0):.4f}")
    print(f"Skill over Silence (NE):  {m.get('acc_nonempty_exact', 0):.4f}")
    print("Bitwise Metrics:")
    for label in ["5", "7"]:
        b = m['bitwise'][label]
        print(f"  Label {label}: P={b['precision']:.3f} R={b['recall']:.3f} F1={b['f1']:.3f}")
    print("Confusion Matrix (rows=actual, cols=pred):")
    labels = m['confusion_matrix']['labels']
    print("        " + " ".join(f"{l:>6}" for l in labels))
    for i, row in enumerate(m['confusion_matrix']['matrix']):
        print(f"{labels[i]:>6} " + " ".join(f"{val:>6}" for val in row))
    print("-" * 40)

def plot_metrics(steps: List[Dict[str, Any]], roll_window: int):
    if plt is None:
        print("Warning: matplotlib not found, skipping plot.")
        return

    # Error plot
    errs = [float(s['err']) for s in steps]
    xs = [int(s['step']) for s in steps]
    
    # Efficiency plot (rolling)
    # Binary: 1 if non-empty match, 0 if non-empty miss, None if empty
    eff_series = []
    for s in steps:
        p_set = sig_to_set(s.get("pred_sig", "[]"))
        a_set = sig_to_set(s.get("act_sig", "[]"))
        if len(a_set) > 0:
            eff_series.append(1.0 if p_set == a_set else 0.0)
        else:
            # We need to maintain step alignment if we want to plot vs step
            # but efficiency is only defined on non-empty steps.
            # Let's just plot the sequence of non-empty steps.
            pass

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    
    ax1.plot(xs, errs, alpha=0.3, label="Error")
    if len(errs) >= roll_window:
        roll_err = [sum(errs[i:i+roll_window])/roll_window for i in range(len(errs)-roll_window+1)]
        ax1.plot(xs[roll_window-1:], roll_err, color="red", label=f"Rolling Err ({roll_window})")
    ax1.set_ylabel("Error")
    ax1.legend()
    ax1.set_title("Learning Progress")

    if eff_series:
        eff_xs = range(len(eff_series))
        ax2.plot(eff_xs, eff_series, '.', alpha=0.1, color='gray')
        if len(eff_series) >= roll_window:
            roll_eff = [sum(eff_series[i:i+roll_window])/roll_window for i in range(len(eff_series)-roll_window+1)]
            ax2.plot(range(roll_window-1, len(eff_series)), roll_eff, color="blue", label=f"Efficiency ({roll_window})")
        ax2.set_ylabel("Response Efficiency")
        ax2.set_xlabel("Non-empty Step Index")
        ax2.set_ylim(-0.05, 1.05)
        ax2.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Compute advanced metrics for bootstrap runs.")
    parser.add_argument("--in", dest="input_log", help="Path to run log (.txt)")
    parser.add_argument("--csv", help="Path to comparison CSV")
    parser.add_argument("--out-json", help="Path to output JSON results")
    parser.add_argument("--plot", action="store_true", help="Show plots")
    parser.add_argument("--roll", type=int, default=25, help="Rolling window size")
    parser.add_argument("--question-credit", type=float, default=0.25, help="Utility credit for QUESTION lane")
    args = parser.parse_args()

    results = {}

    if args.input_log:
        p = Path(args.input_log)
        if not p.exists():
            print(f"Error: {p} not found")
            return
        
        # Support both .txt (logs) and .json (trainer reports)
        if p.suffix == ".json":
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                # Trainer reports have "last_results"
                if "last_results" in data:
                    raw_steps = data["last_results"]
                    steps = []
                    for i, r in enumerate(raw_steps):
                        steps.append({
                            "step": i + 1,
                            "err": r["err"],
                            "pred_sig": r["pred"],
                            "act_sig": r["act"],
                            "drill": r.get("drill", False)
                        })
                else:
                    # Generic metrics JSON
                    print(f"File {p} looks like a metrics JSON, not a step log.")
                    return
            except Exception as e:
                print(f"Error parsing JSON: {e}")
                return
        else:
            text = p.read_text(encoding="utf-8")
            step_rows = parse_steps(text)
            # Convert to dict list for compute_metrics
            steps = []
            for r in step_rows:
                steps.append({
                    "step": r.step,
                    "err": r.err,
                    "lane": r.lane,
                    "handle_count": r.handle_count,
                    "pred_sig": r.pred_sig,
                    "act_sig": r.act_sig
                })
        
        m = compute_metrics(steps, question_credit=args.question_credit)
        print_report(p.name, m)
        results["single"] = m
        if args.plot:
            plot_metrics(steps, args.roll)

    elif args.csv:
        p = Path(args.csv)
        if not p.exists():
            print(f"Error: {p} not found")
            return
        all_rows = parse_csv(p)
        
        # Split by run_type
        by_type = {}
        for r in all_rows:
            rt = r.get("run_type", "default")
            if rt not in by_type: by_type[rt] = []
            by_type[rt].append(r)
            
        for rt, steps in by_type.items():
            m = compute_metrics(steps, question_credit=args.question_credit)
            print_report(f"{p.name} [{rt}]", m)
            results[rt] = m
            if args.plot and rt == "learn_on":
                plot_metrics(steps, args.roll)
                
        if "learn_on" in results and "frozen" in results:
            lon = results["learn_on"]
            frz = results["frozen"]
            print("--- Comparison Deltas (Learn-On - Frozen) ---")
            print(f"Acc Exact:      {lon['acc_exact'] - frz['acc_exact']:+.4f}")
            print(f"Resp Efficiency: {lon['acc_nonempty_exact'] - frz['acc_nonempty_exact']:+.4f}")
            print(f"Skill over Sil:  {lon['skill_over_silence'] - frz['skill_over_silence']:+.4f}")

    if args.out_json:
        with Path(args.out_json).open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {args.out_json}")

if __name__ == "__main__":
    main()
