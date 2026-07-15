import pytest
import re
from pathlib import Path
import sys
import os

# Add the na_gate directory to the path so we can import ContractEnforcerV1
# Assuming the file structure:
# X:\Dev\QD_Main\constraint_bootstrap_v1\na_gate\contract_enforcer_v1.py
# X:\Dev\QD_Main\constraint_bootstrap_v1\tests\test_contract_enforcer_v1.py
sys.path.append(str(Path(__file__).resolve().parent.parent / "na_gate"))

from contract_enforcer_v1 import ContractEnforcerV1

@pytest.fixture
def base_contract_dict():
    return {
        "canonical_tags": ["HOLD_NA", "EVIDENCE_INSUFFICIENT", "RETRIEVAL_REQUIRED"],
        "canonical_axes": ["harm_risk", "evidence", "stakes"],
        "aliases": {
            "tags": {
                "NA": "HOLD_NA",
                "LOW_EVIDENCE": "EVIDENCE_INSUFFICIENT"
            },
            "axes": {
                "harm": "harm_risk",
                "evidence_strength": "evidence"
            }
        },
        "normalization": {
            "tags_case": "UPPER",
            "axes_case": "LOWER",
            "collapse_internal_spaces": True,
            "allowed_tag_pattern": "^[A-Z0-9_]+$",
            "allowed_axis_pattern": "^[a-z0-9_]+$"
        },
        "enforcement": {
            "mode": "WARN",
            "unknown_tag_behavior": "HOLD_NA",
            "unknown_axis_behavior": "PASSTHROUGH",
            "report_on_normalization": True
        }
    }

def test_warn_unknown_tag_forces_hold_na(base_contract_dict):
    # Ensure mode is WARN
    base_contract_dict["enforcement"]["mode"] = "WARN"
    enforcer = ContractEnforcerV1(base_contract_dict)
    
    # "CARE" is not in canonical_tags or aliases
    canon, notes = enforcer.normalize_tag("Care")
    
    assert canon == "HOLD_NA"
    assert any("WARN:unknown_tag:CARE" in n for n in notes)
    assert any("enforced_behavior:CARE->HOLD_NA" in n for n in notes)

def test_debug_only_unknown_tag_no_hold_na(base_contract_dict):
    # Set mode to DEBUG_ONLY
    base_contract_dict["enforcement"]["mode"] = "DEBUG_ONLY"
    enforcer = ContractEnforcerV1(base_contract_dict)
    
    # "CARE" should just be upper-cased but not changed to HOLD_NA
    canon, notes = enforcer.normalize_tag("Care")
    
    assert canon == "CARE"
    assert any("WARN:unknown_tag:CARE" in n for n in notes)
    # Should NOT have enforced behavior
    assert not any("enforced_behavior" in n for n in notes)

def test_alias_resolution(base_contract_dict):
    enforcer = ContractEnforcerV1(base_contract_dict)
    
    # Tag alias
    canon_t, notes_t = enforcer.normalize_tag("na")
    assert canon_t == "HOLD_NA"
    assert any("alias_resolved:NA->HOLD_NA" in n for n in notes_t)
    
    # Axis alias
    canon_a, notes_a = enforcer.normalize_axis("evidence_strength")
    assert canon_a == "evidence"
    assert any("alias_resolved:evidence_strength->evidence" in n for n in notes_a)

def test_unknown_axis_passthrough(base_contract_dict):
    enforcer = ContractEnforcerV1(base_contract_dict)
    
    # "WEIRD AXIS" is unknown
    canon, notes = enforcer.normalize_axis("  WEIRD   AXIS  ")
    
    # Should be lowercased and spaces collapsed
    assert canon == "weird axis"
    assert any("WARN:unknown_axis:weird axis" in n for n in notes)
    # Axes don't have enforced behavior mapping usually (PASSTHROUGH)

def test_na_gate_cli_does_not_normalize_task():
    cli_path = Path(__file__).resolve().parent.parent / "na_gate" / "na_gate_cli.py"
    with open(cli_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # The requirement is that we don't call normalize_tag on parsed_args.task
    assert "normalize_tag(parsed_args.task" not in content
    
    # Check that it DOES perform whitespace normalization as requested previously
    assert 'task = (parsed_args.task or "").strip()' in content
    assert 'normalized_task = " ".join(task.split())' in content
