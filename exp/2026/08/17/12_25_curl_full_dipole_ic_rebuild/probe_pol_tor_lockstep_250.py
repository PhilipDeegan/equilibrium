"""Single extra convergence point: cells_across=250, between the already-run
110/220 and the RAM-infeasible 300 (estimated ~80M leaves/~80GB+ on this
60GB no-swap machine -- see conversation). Reuses run_one from
probe_pol_tor_lockstep_convergence.py unchanged.
"""

import os
import sys

sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from probe_pol_tor_lockstep_convergence import run_one

if __name__ == "__main__":
    run_one(250)
