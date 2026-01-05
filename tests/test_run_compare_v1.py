import pytest
import csv
from pathlib import Path
from constraint_bootstrap.run_compare_v1 import write_compare_csv
from constraint_bootstrap.run_summary_v1 import parse_steps, parse_final_handles, build_summary

SAMPLE_LOG_ON = """
========================================================================
BOOTSTRAP v0.1  partner=mixed_v1  steps=4
========================================================================
001  sent=[2 4 6]                pred=[]         act=[5]        err= 2.0  lane=De  handles=(none)
002  sent=[1]                    pred=[]         act=[]         err= 0.0  lane=Po  handles=(none)
003  sent=[5 1 3]                pred=[]         act=[5 7]      err= 4.0  lane=De  handles=H001:0.25 hits=1 (5,1,3->5,7)
004  sent=[1]                    pred=[]         act=[]         err= 0.0  lane=Po  handles=H001:0.33 hits=1 (5,1,3->5,7) | H002:0.25 hits=1 (1->∅)

Final handles (strongest first):
  H002  strength=0.50  hits=3 misses=0  1 -> ∅
  H001  strength=0.25  hits=1 misses=0  5,1,3 -> 5,7
"""

SAMPLE_LOG_OFF = """
========================================================================
BOOTSTRAP v0.1  partner=mixed_v1  steps=4
========================================================================
001  sent=[2 4 6]                pred=[]         act=[5]        err= 2.0  lane=De  handles=(none)
002  sent=[1]                    pred=[]         act=[]         err= 0.0  lane=Po  handles=(none)
003  sent=[5 1 3]                pred=[]         act=[5 7]      err= 4.0  lane=De  handles=(none)
004  sent=[1]                    pred=[]         act=[]         err= 0.0  lane=Po  handles=(none)

Final handles (strongest first):
"""

def test_compare_logic():
    steps_on = parse_steps(SAMPLE_LOG_ON)
    handles_on = parse_final_handles(SAMPLE_LOG_ON)
    summary_on = build_summary(steps_on, handles_on)
    
    steps_off = parse_steps(SAMPLE_LOG_OFF)
    handles_off = parse_final_handles(SAMPLE_LOG_OFF)
    summary_off = build_summary(steps_off, handles_off)
    
    assert summary_on.final_handle_count == 2
    assert summary_off.final_handle_count == 0
    
    assert summary_on.mean_err == 1.5
    assert summary_off.mean_err == 1.5

def test_compare_csv(tmp_path):
    steps_on = parse_steps(SAMPLE_LOG_ON)
    steps_off = parse_steps(SAMPLE_LOG_OFF)
    
    csv_path = tmp_path / "compare.csv"
    write_compare_csv(csv_path, steps_on, steps_off)
    
    assert csv_path.exists()
    
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        
    assert len(reader) == 8 # 4 steps * 2 runs
    
    learn_rows = [r for r in reader if r["run_type"] == "learn_on"]
    frozen_rows = [r for r in reader if r["run_type"] == "frozen"]
    
    assert len(learn_rows) == 4
    assert len(frozen_rows) == 4
    
    # Check some values
    assert learn_rows[0]["step"] == "1"
    assert learn_rows[0]["err"] == "2.000000"
    assert learn_rows[2]["handle_count"] == "1"
    assert frozen_rows[2]["handle_count"] == "0"
