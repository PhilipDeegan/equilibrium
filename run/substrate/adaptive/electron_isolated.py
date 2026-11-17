"""
Isolated electron dipole -- no-interaction control for
electron_positron_collision.py. Same lattice/geometry/BC/gate, dipole
built at z=+z0 exactly as in the pair run, but the positron at z=-z0 is
never built. Compare this run's surv/H_own history against the pair
run's surv_e/H_up to see how much of the electron's decay is driven by
the positron's presence vs. step_cross v2 acting on it alone.

Run:
  PYTHONPATH=$PWD python3 -u run/substrate/adaptive/electron_isolated.py --n-steps 1000 --n-threads 4
"""

import sys

sys.path.insert(0, ".")
from run.substrate.adaptive.isolated_dipole import build_parser, run

if __name__ == "__main__":
    run(build_parser("electron").parse_args(), species="electron")
