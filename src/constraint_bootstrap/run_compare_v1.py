"""
run_compare_v1.py

CLI to compare two constraint_bootstrap demo output logs (typically learn-on vs frozen).

Usage:
  python -m constraint_bootstrap.run_compare_v1 --learn-on out_on.txt --frozen out_off.txt --out-csv compare.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Any

from .run_summary_v1 import parse_steps, parse_final_handles, build_summary, StepRow, Summary


def print_comparison(learn_summary: Summary, frozen_summary: Summary):
    print("========================================================================")
    print("COMPARISON: LEARN-ON vs FROZEN")
    print("========================================================================")
    
    def fmt_row(label: str, val_on: Any, val_off: Any, delta: Any = None):
        if delta is None:
            try:
                delta = val_on - val_off
            except TypeError:
                delta = "-"
        
        if isinstance(val_on, float):
            d_str = f"{delta:+.4f}" if isinstance(delta, float) else str(delta)
            print(f"{label:<18} {val_on:>10.4f} {val_off:>10.4f}   Δ {d_str}")
        else:
            d_str = f"{delta:+d}" if isinstance(delta, int) else str(delta)
            print(f"{label:<18} {val_on:>10} {val_off:>10}   Δ {d_str}")

    print(f"{'Metric':<18} {'Learn-On':>10} {'Frozen':>10}   {'Delta':>10}")
    print("-" * 55)
    fmt_row("steps", learn_summary.steps, frozen_summary.steps)
    fmt_row("mean_err", learn_summary.mean_err, frozen_summary.mean_err)
    fmt_row("po_rate", learn_summary.po_rate, frozen_summary.po_rate)
    fmt_row("de_rate", learn_summary.de_rate, frozen_summary.de_rate)
    fmt_row("final_handles", learn_summary.final_handle_count, frozen_summary.final_handle_count)
    fmt_row("max_strength", learn_summary.max_handle_strength, frozen_summary.max_handle_strength)
    fmt_row("mean_strength", learn_summary.mean_handle_strength, frozen_summary.mean_handle_strength)
    
    print("-" * 72)
    print("Top 10 handles (LEARN-ON only):")
    for h in learn_summary.top_handles:
        print(f"  {h.hid}  str={h.strength:.2f}  hits={h.hits}  {h.pattern} -> {h.outputs}")
    print("========================================================================")


def write_compare_csv(path: Path, learn_steps: List[StepRow], frozen_steps: List[StepRow]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "run_type", "err", "lane", "handle_count", "pred_sig", "act_sig"])
        
        for r in learn_steps:
            w.writerow([r.step, "learn_on", f"{r.err:.6f}", r.lane, r.handle_count, r.pred_sig, r.act_sig])
            
        for r in frozen_steps:
            w.writerow([r.step, "frozen", f"{r.err:.6f}", r.lane, r.handle_count, r.pred_sig, r.act_sig])


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two constraint_bootstrap demo logs.")
    parser.add_argument("--learn-on", required=True, help="Path to learn-on log file.")
    parser.add_argument("--frozen", required=True, help="Path to frozen log file.")
    parser.add_argument("--out-csv", help="Optional CSV output path.")
    args = parser.parse_args()

    path_on = Path(args.learn_on)
    path_off = Path(args.frozen)

    if not path_on.exists():
        print(f"Error: file not found: {path_on}")
        return 1
    if not path_off.exists():
        print(f"Error: file not found: {path_off}")
        return 1

    text_on = path_on.read_text(encoding="utf-8")
    text_off = path_off.read_text(encoding="utf-8")

    steps_on = parse_steps(text_on)
    handles_on = parse_final_handles(text_on)
    summary_on = build_summary(steps_on, handles_on)

    steps_off = parse_steps(text_off)
    handles_off = parse_final_handles(text_off)
    summary_off = build_summary(steps_off, handles_off)

    print_comparison(summary_on, summary_off)

    if args.out_csv:
        write_compare_csv(Path(args.out_csv), steps_on, steps_off)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
