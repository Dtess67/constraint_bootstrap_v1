"""
na_gate_v0.py
QD_v3 — Sodium Gate (Hysteretic)
"""

from dataclasses import dataclass
import json
import time
from pathlib import Path

@dataclass
class NAGateV0:
    threshold: float = 0.5
    is_open: bool = False
    
    def process(self, value: float) -> bool:
        """Update gate state based on input value."""
        if not self.is_open:
            if value >= self.threshold:
                self.is_open = True
        else:
            if value < (self.threshold * 0.5): # Simple hysteresis
                self.is_open = False
        return self.is_open

def log_event(log_dir: Path, event_type: str, data: dict):
    """Append event to JSONL log."""
    log_file = log_dir / "na_gate_events.jsonl"
    event = {
        "timestamp": time.time(),
        "event_type": event_type,
        **data
    }
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
