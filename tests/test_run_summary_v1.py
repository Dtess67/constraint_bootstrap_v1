import pytest
from constraint_bootstrap.run_summary_v1 import parse_steps, parse_final_handles, build_summary

SAMPLE_LOG = """
========================================================================
BOOTSTRAP v0.1  partner=mixed_v1  steps=60
========================================================================
001  sent=[2 4 6]                pred=[]         act=[5]        err= 2.0  lane=De  handles=(none)
002  sent=[1]                    pred=[]         act=[]         err= 0.0  lane=Po  handles=(none)
003  sent=[5 1 3]                pred=[]         act=[5 7]      err= 4.0  lane=De  handles=H001:0.25 hits=1 (5,1,3->5,7)
004  sent=[1]                    pred=[]         act=[]         err= 0.0  lane=Po  handles=H001:0.33 hits=1 (5,1,3->5,7) | H002:0.25 hits=1 (1->∅)

Final handles (strongest first):
  H002  strength=0.50  hits=3 misses=0  1 -> ∅
  H001  strength=0.25  hits=1 misses=0  5,1,3 -> 5,7
"""

def test_summary_logic():
    steps = parse_steps(SAMPLE_LOG)
    assert len(steps) == 4
    assert steps[0].step == 1
    assert steps[0].err == 2.0
    assert steps[0].lane == "De"
    assert steps[0].handle_count == 0
    
    assert steps[3].handle_count == 2
    
    handles = parse_final_handles(SAMPLE_LOG)
    assert len(handles) == 2
    assert handles[0].hid == "H002"
    assert handles[0].strength == 0.50
    
    summary = build_summary(steps, handles)
    assert summary.steps == 4
    # mean_err = (2.0 + 0.0 + 4.0 + 0.0) / 4 = 1.5
    assert summary.mean_err == 1.5
    assert summary.final_handle_count == 2
    assert summary.top_handles[0].strength == 0.50
