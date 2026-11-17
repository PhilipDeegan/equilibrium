"""Bracket the cells_across=220 vs 250 non-monotonicity: run 180 and 235
(one on each side of 220) with the same run_one from
probe_pol_tor_lockstep_convergence.py, sequentially, to see whether 220 was
itself a resonance-like local minimum (dip) or 250 was the outlier.
"""

import os
import sys

sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from probe_pol_tor_lockstep_convergence import run_one

if __name__ == "__main__":
    run_one(180)
    run_one(235)
