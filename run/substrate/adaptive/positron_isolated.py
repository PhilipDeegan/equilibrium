"""
Isolated positron dipole -- no-interaction control for
electron_positron_collision.py. Same lattice/geometry/BC/gate, mirrored
dipole (amplitude<0) built at z=-z0 exactly as in the pair run, but the
electron at z=+z0 is never built. Compare this run's surv/H_own history
against the pair run's surv_p/H_dn to see how much of the positron's
decay is driven by the electron's presence vs. step_cross v2 acting on
it alone.

Run:
  PYTHONPATH=$PWD python3 -u run/substrate/adaptive/positron_isolated.py --n-steps 1000 --n-threads 4
"""

import sys

sys.path.insert(0, ".")
from run.substrate.adaptive.isolated_dipole import build_parser, run

if __name__ == "__main__":
    run(build_parser("positron").parse_args(), species="positron")
