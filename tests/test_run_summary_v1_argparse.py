import pytest
import sys
from unittest.mock import patch
from q_ternary.run_summary_v1 import main

def test_argparse_positional(tmp_path):
    f = tmp_path / "test.json"
    f.write_text('{"metadata": {}, "summary": {}}', encoding="utf-8")
    
    test_args = ["prog", str(f)]
    with patch.object(sys, 'argv', test_args):
        # We just want to see if it doesn't crash on parsing
        # and ideally returns 0 if the file is valid.
        with patch('q_ternary.run_summary_v1.print_summary'): # mock print to keep test quiet
            assert main() == 0

def test_argparse_flag(tmp_path):
    f = tmp_path / "test.json"
    f.write_text('{"metadata": {}, "summary": {}}', encoding="utf-8")
    
    test_args = ["prog", "--in", str(f)]
    with patch.object(sys, 'argv', test_args):
        with patch('q_ternary.run_summary_v1.print_summary'):
            assert main() == 0

def test_argparse_no_input():
    test_args = ["prog"]
    with patch.object(sys, 'argv', test_args):
        # Should return 1 or exit if no input is provided
        # Based on the code: if not input_path_str: return 1
        assert main() == 1
