"""
na_gate_cli.py
CLI for NAGateV0
"""

import argparse
import sys
from pathlib import Path

# Add parent dir to sys.path to allow imports if running directly
sys.path.append(str(Path(__file__).resolve().parents[1]))
from na_gate.na_gate_v0 import NAGateV0, log_event

def main():
    parser = argparse.ArgumentParser(prog="na_gate_cli")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("values", type=float, nargs="+", help="Input values to process")
    
    args = parser.parse_args()
    gate = NAGateV0(threshold=args.threshold)
    log_dir = Path(__file__).resolve().parents[1] / "logs"
    
    # Ensure log dir exists (redundant but safe)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    for val in args.values:
        is_open = gate.process(val)
        state = "OPEN" if is_open else "CLOSED"
        print(f"Value: {val} -> Gate: {state}")
        log_event(log_dir, "process_value", {"value": val, "is_open": is_open})

if __name__ == "__main__":
    main()
