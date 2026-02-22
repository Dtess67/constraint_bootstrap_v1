import unittest
from pathlib import Path
import sys

# Add base dir to sys.path
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from na_gate.na_gate_v0 import NAGateV0

class TestNAGateV0(unittest.TestCase):
    def test_gate_logic(self):
        # Test with threshold 0.6
        gate = NAGateV0(threshold=0.6)
        self.assertFalse(gate.is_open, "Gate should start closed")
        
        # Below threshold
        self.assertFalse(gate.process(0.5))
        
        # Trigger open (at or above threshold)
        self.assertTrue(gate.process(0.6))
        self.assertTrue(gate.is_open)
        
        # Stay open (hysteresis: above 0.5 * threshold = 0.3)
        self.assertTrue(gate.process(0.4))
        self.assertTrue(gate.is_open)
        
        # Close (below 0.5 * threshold)
        self.assertFalse(gate.process(0.2))
        self.assertFalse(gate.is_open)
        
        # Re-open
        self.assertTrue(gate.process(0.7))
        self.assertTrue(gate.is_open)

if __name__ == "__main__":
    unittest.main()
