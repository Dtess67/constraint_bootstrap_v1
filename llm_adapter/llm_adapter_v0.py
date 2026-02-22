"""
LLM Adapter v0 (skeleton)

Rule: This adapter MUST call Na Gate before returning any response.
For now, it does not call a real LLM. It only demonstrates gating + trace.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import subprocess
import sys
import json
import os
from pathlib import Path


@dataclass
class GateResult:
    decision: str          # "HOLD" or "RELEASE"
    decision_id: Optional[str]
    reason: str


QD_STATE_DEFAULT = r"X:\Dev\QD_Main\qd_state.yaml"


def _get_log_root_from_qd_state(qd_state_path: str) -> str:
    """
    Minimal parser: finds the first line containing 'log_root:' and extracts the quoted value.
    Keeps us dependency-free (no PyYAML requirement).
    """
    p = Path(qd_state_path)
    text = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in text:
        s = line.strip()
        if s.startswith("log_root:"):
            # supports: log_root: "X:\\Dev\\..."
            val = s.split("log_root:", 1)[1].strip().strip('"').strip("'")
            return val
    raise RuntimeError(f"log_root not found in {qd_state_path}")


def _read_last_jsonl(path: Path) -> dict:
    """
    Reads the last non-empty line of a JSONL file and returns it as dict.
    """
    if not path.exists():
        raise FileNotFoundError(str(path))
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in reversed(lines):
        if line.strip():
            return json.loads(line)
    raise RuntimeError(f"No JSON lines found in {path}")


def _find_decision_by_task_id(path: Path, task_id: str) -> dict:
    """
    Scans decisions.jsonl from bottom up to find first record matching task_id.
    """
    if not path.exists():
        raise FileNotFoundError(str(path))
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in reversed(lines):
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("task_id") == task_id:
            return rec
    raise RuntimeError(f"No decision found for task_id={task_id} in {path}")


def run_na_gate(task: str, stakes: str, evidence: list[str]) -> GateResult:
    """
    Calls: python -m constraint_bootstrap_v1.na_gate.na_gate_cli ...
    Expects stdout to contain Decision: HOLD/RELEASE and decision_id: <uuid>.
    """
    cmd = [sys.executable, "-m", "constraint_bootstrap_v1.na_gate.na_gate_cli", "--task", task, "--stakes", stakes]
    for e in evidence:
        cmd += ["--evidence", e]

    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "").strip()
    
    decision = "HOLD"
    decision_id = None
    reason = "na_gate_no_stdout"
    
    if out:
        reason = f"na_gate_stdout_lines={len(out.splitlines())}"
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("Decision:"):
                decision = line.split("Decision:")[1].strip()
            elif line.startswith("decision_id:"):
                decision_id = line.split("decision_id:")[1].strip()
    
    return GateResult(decision=decision, decision_id=decision_id, reason=reason)


def answer(task: str, stakes: str = "low", tone: str = "dry", evidence: list[str] | None = None) -> Tuple[str, GateResult]:
    """
    Returns a response ONLY if Na Gate releases.
    """
    gate = run_na_gate(task=task, stakes=stakes, evidence=(evidence or []))

    if gate.decision != "RELEASE":
        hold_msg = (
            "[NA HOLD]\n"
            f"- decision_id: {gate.decision_id}\n"
            f"- reason: {gate.reason}\n"
            "- next:\n"
            "  • Provide evidence artifacts (files/links/logs). For high stakes, provide 2.\n"
            "  • Or reduce stakes to 'med' or 'low' if appropriate.\n"
        )
        return (hold_msg, gate)

    # Build evidence_block:
    evidence_list = evidence or []
    if not evidence_list:
        evidence_block = "EVIDENCE: (none)"
    else:
        blocks = []
        for i, path_str in enumerate(evidence_list):
            p = Path(path_str)
            try:
                # Read up to first 800 chars as text (errors='replace')
                with p.open("r", encoding="utf-8", errors="replace") as f:
                    content = f.read(800)
                blocks.append(f"EVIDENCE[{i}]: {path_str}\n{content}")
            except Exception as e:
                blocks.append(f"EVIDENCE[{i}]: {path_str}\n[ERROR READING FILE: {e}]")
        evidence_block = "\n\n".join(blocks)

    # Placeholder for real LLM call later:
    # Real LLM call (Ollama) — only runs after RELEASE
    sources_line = ""
    if evidence_list:
        basenames = [os.path.basename(e) for e in evidence_list]
        sources_line = f"\nAfter your 1–2 sentence answer, add a final line exactly like:\nSOURCES: {', '.join(basenames)}\nUse only the basename of each evidence path (e.g., {basenames[0] if basenames else 'evidence_1.txt'})."

    prompt = (
        "You are QD: witty, warm, and a little mischievous, but concise.\n"
        "Do NOT mention being trained by Google/OpenAI or any company.\n"
        "Always start with: hello, field.\n"
        f"Tone: {tone}. If tone is 'emo', be poetic, restrained, and sincere. If tone is 'dry', be intelligent, witty, and dry with occasional dark humor (never cruel, never mean), and avoid goofy cheer.\n"
        "\n--- EVIDENCE snippets ---\n"
        f"{evidence_block}\n"
        "--- END EVIDENCE ---\n"
        "\nHard rules:\n"
        "- You MUST base your answer ONLY on the EVIDENCE snippets above.\n"
        "- If the evidence does not explicitly contain the answer, respond exactly:\n"
        "  'hello, field. [EVIDENCE INSUFFICIENT] Provide two authoritative sources.'\n"
        "- Do NOT use phrases like 'as of today' or name a release unless it appears in evidence.\n"
        f"{sources_line}\n"
        "\nThen answer in 1-2 sentences matching the tone.\n"
        f"User task: {task}"
    )
    p = subprocess.run(
        ["ollama", "run", "gemma3:4b", prompt],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    text = (p.stdout or "").strip() or "[LLM EMPTY OUTPUT]"
    return (text, gate)
