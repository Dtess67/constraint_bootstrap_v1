import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
import yaml

try:
    from .na_gate_v0 import NAGateV0, log_event
except ImportError:
    from na_gate_v0 import NAGateV0, log_event

def _evaluate(task, stakes, assumptions=None):
    if stakes == "low":
        return "RELEASE", [], []
    else:
        return "HOLD", [], []

def get_qd_state_path(provided_path=None):
    if provided_path:
        return Path(provided_path)
    
    env_path = os.environ.get("QD_STATE_PATH")
    if env_path:
        return Path(env_path)
    
    # Fallback: walk up from this file's location
    try:
        current = Path(__file__).resolve().parent
        for _ in range(10): # Limit depth
            candidate = current / "qd_state.yaml"
            if candidate.exists():
                return candidate
            if current.parent == current:
                break
            current = current.parent
    except Exception:
        pass
        
    return None

def get_log_root(qd_state_path): # returns (log_root, err, contract_version)
    if not qd_state_path:
        return None, "missing_qd_state_path", None
    
    try:
        if not qd_state_path.exists():
            return None, f"qd_state_path_not_found:{qd_state_path}", None
            
        with open(qd_state_path, "r", encoding="utf-8") as f:
            content = f.read()
            found_log_root = None
            for line in content.splitlines():
                if "log_root:" in line:
                    path_str = line.split("log_root:")[1].strip().strip('"')
                    found_log_root = Path(path_str)
                    break
            
            state = yaml.safe_load(content)
            contract = state.get("qd_state", {}).get("canonical_naming_contract_v1", {})
            contract_version = contract.get("version")
            yaml_log_root = Path(state["qd_state"]["traceability"]["log_root"])
            return (found_log_root or yaml_log_root), None, contract_version
    except Exception as e:
        return None, f"qd_state_parse_failed:{e}", None

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--stakes", choices=["low", "high"], default="low")
    parser.add_argument("--assumption", action="append")
    parser.add_argument("--evidence", action="append")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--qd-state-path")
    
    parsed_args = parser.parse_args(args)
    
    decision, flags, req = _evaluate(parsed_args.task, parsed_args.stakes, parsed_args.assumption)
    decision_id = str(uuid.uuid4())
    reason = None

    qd_state_path = get_qd_state_path(parsed_args.qd_state_path)
    log_root, path_err, contract_version = get_log_root(qd_state_path)
    if parsed_args.debug:
        print(f"[DEBUG] contract_version={contract_version}")
    
    if path_err:
        decision = "HOLD"
        reason = path_err

    decisions_path = log_root / "decisions.jsonl" if log_root else None
    assumptions_path = log_root / "assumptions.jsonl" if log_root else None
    na_gate_log_path = log_root / "na_gate_log.jsonl" if log_root else None

    # Final fail-closed check for individual paths
    if not reason and (not decisions_path or not assumptions_path or not na_gate_log_path):
        decision = "HOLD"
        reason = "missing_log_paths"

    if parsed_args.debug:
        print(f"DEBUG: qd_state_path={qd_state_path}")
        print(f"DEBUG: log_root={log_root}")
        print(f"DEBUG: decisions_path={decisions_path}")
        print(f"DEBUG: assumptions_path={assumptions_path}")
        print(f"DEBUG: na_gate_log_path={na_gate_log_path}")

    if not reason and log_root:
        try:
            log_root.mkdir(parents=True, exist_ok=True)
            
            assumption_id = None
            if parsed_args.assumption:
                assumption_id = str(uuid.uuid4())
                for a in parsed_args.assumption:
                    with open(assumptions_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "timestamp": time.time(),
                            "assumption_id": assumption_id,
                            "decision_id": decision_id,
                            "content": a
                        }) + "\n")

            evidence_id = None
            if parsed_args.evidence:
                evidence_id = str(uuid.uuid4())

            # na_gate_log.jsonl
            with open(na_gate_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "timestamp": time.time(),
                    "decision_id": decision_id,
                    "task": parsed_args.task,
                    "stakes": parsed_args.stakes,
                    "decision": decision,
                    "assumption_id": assumption_id,
                    "evidence_id": evidence_id
                }) + "\n")
                
            # decisions.jsonl
            with open(decisions_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "timestamp": time.time(),
                    "decision_id": decision_id,
                    "decision": decision,
                    "task": parsed_args.task
                }) + "\n")
        except Exception as e:
            decision = "HOLD"
            reason = f"log_write_failed:{e}"

    if reason:
        print(f"Decision: {decision} ({reason})")
    else:
        print(f"Decision: {decision}")
    print(f"decision_id: {decision_id}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
