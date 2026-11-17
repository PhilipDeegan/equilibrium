"""Same question as probe_pol_tor_lockstep.py (does phi_pol/u stay locked
to phi_tor's amplitude under pure wave dynamics, no step_cross), but at the
VALIDATED production resolution (cells_across=220,
project_hop_telescope_resolved_spheromak) instead of the tiny 22-cell test
fixture -- the small-fixture result (u_rms swinging ~2x non-monotonically)
is suspected to be a pure under-resolution artifact (same family of bug as
the old L8 blowup: insufficient information density to sustain the
structure), not a real physical signal. Re-run at the resolution already
known to actually hold together.

Uses the existing t=0 checkpoint directly (no rebuild) -- pure
evolver.step() only, no step_cross, so this is NOT the same run as the
200-step production stepping already done (which included cross_scale).
"""

import sys

sys.path.insert(0, ".")
import time

import numpy as np

from equilibrium import lattice as _lat
from equilibrium.substrate.curl_full_spheromak_ic import find_square_ring_on_leaves
from tst.substrate.adaptive.adaptive_flux_tube import phi_ring_circulation
from run.substrate.adaptive.sweep_spheromak_decay_cross_scale import _compute_dt_l0

IC_PATH = "/bk/curlfull_dipole_uniform_r180_ic220.h5"
HOPS = 41  # same calibration as curl_full_spheromak_ic's production builds -- radius ~1.333*41=54.6, ~half of R_SPH*SCALE=110
N_STEPS = 60
REPORT_EVERY = 10


def main():
    t0 = time.time()
    alat = _lat.AdaptiveLattice.load(IC_PATH)
    alat.set_n_threads(8)
    cl = alat[0]
    print(
        f"loaded {IC_PATH} n_leaves={cl.n_leaves} t={time.time()-t0:.1f}s", flush=True
    )

    ring_js, ring_faces = find_square_ring_on_leaves(cl, axis="z", hops=HOPS)
    assert ring_js is not None, "ring-finding failed on production mesh"
    pos = np.array(cl.leaf_pos)
    centre = pos[ring_js].mean(axis=0)
    print(
        f"ring found: {len(ring_js)} tiles, centre={centre}, t={time.time()-t0:.1f}s",
        flush=True,
    )

    u0 = float(np.sqrt(np.mean(np.array(cl.leaf_u) ** 2)))
    circ0 = phi_ring_circulation(cl, ring_js, centre)
    print(f"t=0  u_rms={u0:.6e}  circ={circ0:.6e}", flush=True)

    dt = _compute_dt_l0(cl, D1=1.0)
    ts = _lat.TimeStepper.by_level(dt)
    evolver = _lat.AdaptiveLatticeEvolver(alat)
    print(f"dt={dt:.6e}", flush=True)

    for step_i in range(1, N_STEPS + 1):
        t1 = time.time()
        evolver.step(ts)  # pure wave dynamics, NO step_cross
        step_dt = time.time() - t1
        if step_i % REPORT_EVERY == 0:
            u_rms = float(np.sqrt(np.mean(np.array(cl.leaf_u) ** 2)))
            circ = phi_ring_circulation(cl, ring_js, centre)
            print(
                f"step {step_i:3d}  step_time={step_dt:.1f}s  u_rms={u_rms:.6e} (ratio {u_rms/u0:.6f})  "
                f"circ={circ:.6e} (ratio {circ/circ0:.6f})",
                flush=True,
            )

    print(f"FINAL total_time={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
