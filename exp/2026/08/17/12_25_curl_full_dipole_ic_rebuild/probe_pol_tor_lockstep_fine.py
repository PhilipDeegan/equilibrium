"""Fine bracket right around the cells_across=220 dip: 218 and 222, to see
how narrow/sharp the resonance actually is.
"""

import os
import sys

sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from probe_pol_tor_lockstep_convergence import run_one

if __name__ == "__main__":
    run_one(218)
    run_one(222)
