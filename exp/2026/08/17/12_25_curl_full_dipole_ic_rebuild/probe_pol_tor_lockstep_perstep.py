"""Re-check the 218/220/222 dip with PER-STEP reporting (REPORT_EVERY=1)
instead of every-10-steps -- ruling out that the every-10 sampling was
hiding or misrepresenting the true u_rms swing (Philip's question,
2026-08-17: "the points is not enough?").
"""

import os
import sys

sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import probe_pol_tor_lockstep_convergence as C

C.REPORT_EVERY = 1

if __name__ == "__main__":
    C.run_one(218)
    C.run_one(220)
    C.run_one(222)
