import re
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

class ContractEnforcerV1:
    def __init__(self, contract: Optional[Dict[str, Any]] = None):
        self.contract = contract or {}
        self.enforcement = self.contract.get("enforcement", {})
        self.mode = self.enforcement.get("mode", "DEBUG_ONLY")
        self.normalization = self.contract.get("normalization", {})
        self.aliases = self.contract.get("aliases", {})
        self.canonical_tags = set(self.contract.get("canonical_tags", []))
        self.canonical_axes = set(self.contract.get("canonical_axes", []))
        
        # Pre-compile patterns
        self.tag_pattern = re.compile(self.normalization.get("allowed_tag_pattern", r"^[A-Z0-9_]+$"))
        self.axis_pattern = re.compile(self.normalization.get("allowed_axis_pattern", r"^[a-z0-9_]+$"))

    @classmethod
    def from_qd_state(cls, qd_state_path: Path) -> 'ContractEnforcerV1':
        with open(qd_state_path, "r", encoding="utf-8") as f:
            state = yaml.safe_load(f)
        contract = state.get("qd_state", {}).get("canonical_naming_contract_v1", {})
        return cls(contract)

    def _clean_string(self, s: str, case_mode: str) -> str:
        if not s:
            return ""
        
        # Trim whitespace
        s = s.strip()
        
        # Collapse internal spaces
        if self.normalization.get("collapse_internal_spaces", True):
            s = " ".join(s.split())
            
        # Case normalization
        if case_mode == "UPPER":
            s = s.upper()
        elif case_mode == "LOWER":
            s = s.lower()
            
        return s

    def normalize_tag(self, raw: str) -> Tuple[str, List[str]]:
        notes = []
        if not raw:
            return "", []

        # 1. Basic cleaning
        tag_case = self.normalization.get("tags_case", "UPPER")
        clean = self._clean_string(raw, tag_case)
        
        # 2. Alias mapping
        tag_aliases = self.aliases.get("tags", {})
        canon = tag_aliases.get(clean, clean)
        if canon != clean:
            notes.append(f"alias_resolved:{clean}->{canon}")
            
        # 3. Validation
        is_canonical = canon in self.canonical_tags
        pattern_match = bool(self.tag_pattern.match(canon))
        
        if not is_canonical or not pattern_match:
            err_type = "unknown_tag" if not is_canonical else "invalid_pattern"
            msg = f"{err_type}:{canon}"
            
            if self.mode == "FAIL":
                raise ValueError(f"Contract violation: {msg}")
            
            notes.append(f"WARN:{msg}")
            
            if self.mode == "WARN":
                # Apply unknown_tag_behavior
                behavior = self.enforcement.get("unknown_tag_behavior", "HOLD_NA")
                notes.append(f"enforced_behavior:{canon}->{behavior}")
                return behavior, notes
                
        return canon, notes

    def normalize_axis(self, raw: str) -> Tuple[str, List[str]]:
        notes = []
        if not raw:
            return "", []

        # 1. Basic cleaning
        axis_case = self.normalization.get("axes_case", "LOWER")
        clean = self._clean_string(raw, axis_case)
        
        # 2. Alias mapping
        axis_aliases = self.aliases.get("axes", {})
        canon = axis_aliases.get(clean, clean)
        if canon != clean:
            notes.append(f"alias_resolved:{clean}->{canon}")
            
        # 3. Validation
        is_canonical = canon in self.canonical_axes
        pattern_match = bool(self.axis_pattern.match(canon))
        
        if not is_canonical or not pattern_match:
            err_type = "unknown_axis" if not is_canonical else "invalid_pattern"
            msg = f"{err_type}:{canon}"
            
            if self.mode == "FAIL":
                raise ValueError(f"Contract violation: {msg}")
            
            notes.append(f"WARN:{msg}")
            
            # unknown_axis_behavior (usually PASSTHROUGH)
            # No change to canon if passthrough, but note is recorded.
                
        return canon, notes

    def normalize_payload(self, 
                          tags: Optional[List[str]] = None, 
                          axes: Optional[Dict[str, float]] = None) -> Tuple[List[str], Dict[str, float], List[str]]:
        all_notes = []
        out_tags = []
        out_axes = {}
        
        if tags:
            for t in tags:
                canon, notes = self.normalize_tag(t)
                out_tags.append(canon)
                all_notes.extend(notes)
                
        if axes:
            for k, v in axes.items():
                canon, notes = self.normalize_axis(k)
                out_axes[canon] = v
                all_notes.extend(notes)
                
        return out_tags, out_axes, all_notes
