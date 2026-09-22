"""
Wrapper for beyond_laya_jev_suite.py in root evaluation directory.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "apex_decision"))

from apex_decision.evaluation.beyond_laya_jev_suite import *

if __name__ == "__main__":
    main()
