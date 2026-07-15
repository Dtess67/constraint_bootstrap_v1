import sys
import yaml
from pathlib import Path
from contract_enforcer_v1 import ContractEnforcerV1

def run_demo():
    # 1. Locate qd_state.yaml
    # We'll try to walk up from the current file
    current = Path(__file__).resolve().parent
    qd_state_path = None
    for _ in range(5):
        candidate = current / "qd_state.yaml"
        if candidate.exists():
            qd_state_path = candidate
            break
        if current.parent == current:
            break
        current = current.parent
    
    if not qd_state_path:
        # Fallback to a relative path if the walk-up fails for some reason
        qd_state_path = Path("../../qd_state.yaml").resolve()

    if not qd_state_path.exists():
        print(f"Error: qd_state.yaml not found at {qd_state_path}")
        return

    # 2. Load Enforcer
    try:
        enforcer = ContractEnforcerV1.from_qd_state(qd_state_path)
    except Exception as e:
        print(f"Error loading contract: {e}")
        return

    print(f"--- Contract Enforcer Demo (Mode: {enforcer.mode}) ---")
    print(f"Source: {qd_state_path}")
    print("-" * 50)

    # 3. Sample Data
    tags = [" na ", "low_evidence", "Care", "???"]
    axes = {"Harm": 0.9, "evidence_strength": 0.2, "WEIRD AXIS": 0.1}

    print("Original Tags:", tags)
    print("Original Axes:", axes)
    print("-" * 50)

    # 4. Normalize
    tags_out, axes_out, notes = enforcer.normalize_payload(tags, axes)

    # 5. Print Results
    print("Normalized Tags:", tags_out)
    print("Normalized Axes:", axes_out)
    print("\nNotes recorded during normalization:")
    if not notes:
        print("  (None)")
    for n in notes:
        print(f"  - {n}")

    print("-" * 50)
    print("Demo complete.")

if __name__ == "__main__":
    run_demo()
