"""
Root launcher for PBL Test & Simulation Suite: Agentic AI Security Layer.
Allows direct execution:
    python test_security.py
"""
import sys
import os

# Ensure project root is in sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from tests.test_security import (
    TestAgenticSecurityLayer,
    run_pbl_simulation,
)
import unittest

if __name__ == "__main__":
    run_pbl_simulation()
    print("Running automated unit test assertions...\n")
    unittest.main(argv=[sys.argv[0], "-v"])
