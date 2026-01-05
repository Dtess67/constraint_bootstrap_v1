"""
run_summary_v1.py

CLI to summarize constraint_bootstrap demo output logs.

Usage:
  python -m constraint_bootstrap.run_summary_v1 --in out.txt
Optional:
  --out-json summary.json
  --out-csv  curve.csv

What it extracts:
- Per-step: step index, err, lane, handle_count
- Final handles block: handle id, strength, hits/misses, pattern, outputs
- Summary stats: steps, mean_err, Po/De rates, final handle count, top handles by strength
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---- Data models ----

@dataclass(frozen=True)
class StepRow:
    step: int
    err: float
    lane: str
    handle_count: int
    pred_sig: str = ""
    act_sig: str = ""


@dataclass(frozen=True)
class FinalHandle:
    hid: str
    strength: float
    hits: int
    misses: int
    pattern: str
    outputs: str


@dataclass(frozen=True)
class Summary:
    steps: int
    mean_err: float
    po_rate: float
    de_rate: float
    other_rate: float
    final_handle_count: int
    max_handle_strength: float
    mean_handle_strength: float
    top_handles: List[FinalHandle]


# ---- Parsing helpers ----

# 001  sent=[2 4 6]                pred=[]         act=[5]        err= 2.0  lane=De  handles=(none)
_STEP_RE = re.compile(
    r"^\s*(\d+)\s+.*?pred=(\[[^\]]*\]|[^\s]+)\s+act=(\[[^\]]*\]|[^\s]+)\s+err=\s*([0-9]+(?:\.[0-9]+)?)\s+lane=([A-Za-z]+)\s+handles=(.*)\s*$"
)

# Example:
# H003  strength=0.73  hits=6 misses=0  1,4,3 -> 5,7
# Note: arrow might be "->" only (we enforce ASCII).
_FINAL_HANDLE_RE = re.compile(
    r"^\s*(H\d+)\s+strength=([0-9]+(?:\.[0-9]+)?)\s+hits=(\d+)\s+misses=(\d+)\s+(.*?)\s*->\s*(.+?)\s*$"
)


def _count_handles(handle_blob: str) -> int:
    """handles=(none) OR handles=H002:0.49 ... | H003:0.73 ..."""
    blob = handle_blob.strip()
    if not blob or blob == "(none)":
        return 0
    # split on pipes; each chunk should begin with something like Hxxx:
    parts = [p.strip() for p in blob.split("|")]
    # filter empty chunks just in case
    parts = [p for p in parts if p]
    return len(parts)


def parse_steps(text: str) -> List[StepRow]:
    rows: List[StepRow] = []
    for line in text.splitlines():
        m = _STEP_RE.match(line)
        if not m:
            continue
        step = int(m.group(1))
        pred_sig = m.group(2)
        act_sig = m.group(3)
        err = float(m.group(4))
        lane = m.group(5)
        handle_blob = m.group(6)
        rows.append(
            StepRow(
                step=step,
                err=err,
                lane=lane,
                handle_count=_count_handles(handle_blob),
                pred_sig=pred_sig,
                act_sig=act_sig,
            )
        )
    return rows


def parse_final_handles(text: str) -> List[FinalHandle]:
    """
    Parse the last "Final handles" block in the file.
    We do this by scanning for the marker and then parsing subsequent lines
    until a blank line or prompt-like line.
    """
    lines = text.splitlines()

    # find the LAST occurrence of "Final handles"
    start_idx = -1
    for i, line in enumerate(lines):
        if "Final handles" in line:
            start_idx = i

    if start_idx == -1:
        return []

    out: List[FinalHandle] = []
    for line in lines[start_idx + 1 :]:
        line = line.rstrip("\n")
        if not line.strip():
            # blank line ends the block
            break
        # stop if we hit a shell prompt or separator
        if line.strip().startswith("(.venv)") or line.strip().startswith("PS ") or line.strip().startswith("----"):
            break

        m = _FINAL_HANDLE_RE.match(line)
        if not m:
            continue

        hid = m.group(1)
        strength = float(m.group(2))
        hits = int(m.group(3))
        misses = int(m.group(4))
        pattern = m.group(5).strip()
        outputs = m.group(6).strip()
        out.append(
            FinalHandle(
                hid=hid,
                strength=strength,
                hits=hits,
                misses=misses,
                pattern=pattern,
                outputs=outputs,
            )
        )

    # sort strongest first (in case input isn't strictly ordered)
    out.sort(key=lambda h: h.strength, reverse=True)
    return out


def build_summary(step_rows: List[StepRow], final_handles: List[FinalHandle], top_n: int = 10) -> Summary:
    if not step_rows:
        # keep it safe / explicit rather than crashing
        return Summary(
            steps=0,
            mean_err=0.0,
            po_rate=0.0,
            de_rate=0.0,
            other_rate=0.0,
            final_handle_count=len(final_handles),
            max_handle_strength=0.0,
            mean_handle_strength=0.0,
            top_handles=final_handles[:top_n],
        )

    steps = len(step_rows)
    mean_err = sum(r.err for r in step_rows) / float(steps)

    po = sum(1 for r in step_rows if r.lane.lower() == "po")
    de = sum(1 for r in step_rows if r.lane.lower() == "de")
    other = steps - po - de

    max_str = max((h.strength for h in final_handles), default=0.0)
    mean_str = sum(h.strength for h in final_handles) / float(len(final_handles)) if final_handles else 0.0

    return Summary(
        steps=steps,
        mean_err=mean_err,
        po_rate=po / float(steps),
        de_rate=de / float(steps),
        other_rate=other / float(steps),
        final_handle_count=len(final_handles),
        max_handle_strength=max_str,
        mean_handle_strength=mean_str,
        top_handles=final_handles[:top_n],
    )


def summary_to_dict(summary: Summary) -> Dict[str, Any]:
    d = asdict(summary)
    # dataclass list already serialized by asdict
    return d


# ---- Outputs ----

def write_csv_curve(path: Path, step_rows: List[StepRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "err", "lane", "handle_count"])
        for r in step_rows:
            w.writerow([r.step, f"{r.err:.6f}", r.lane, r.handle_count])


def write_json(path: Path, summary: Summary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary_to_dict(summary), f, indent=2)


def print_human(summary: Summary) -> None:
    print("========================================================================")
    print("SUMMARY v0.1")
    print("========================================================================")
    print(f"steps:            {summary.steps}")
    print(f"mean_err:         {summary.mean_err:.4f}")
    print(f"lane rates:       Po={summary.po_rate:.3f}  De={summary.de_rate:.3f}  Other={summary.other_rate:.3f}")
    print(f"final_handles:    {summary.final_handle_count}")
    print(f"handle strength:  max={summary.max_handle_strength:.2f}  mean={summary.mean_handle_strength:.2f}")
    print("-" * 72)
    print("Top handles (by strength):")
    for h in summary.top_handles:
        print(f"  {h.hid}  str={h.strength:.2f}  hits={h.hits}  {h.pattern} -> {h.outputs}")
    print("========================================================================")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize constraint_bootstrap demo output logs.")
    parser.add_argument("--in", dest="input_file", required=True, help="Input log file path.")
    parser.add_argument("--out-json", help="Optional JSON output path.")
    parser.add_argument("--out-csv", help="Optional CSV output path (step-by-step curve).")
    args = parser.parse_args()

    ipath = Path(args.input_file)
    if not ipath.exists():
        print(f"Error: file not found: {ipath}")
        return 1

    text = ipath.read_text(encoding="utf-8")
    steps = parse_steps(text)
    handles = parse_final_handles(text)
    summary = build_summary(steps, handles)

    print_human(summary)

    if args.out_json:
        write_json(Path(args.out_json), summary)

    if args.out_csv:
        write_csv_curve(Path(args.out_csv), steps)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
