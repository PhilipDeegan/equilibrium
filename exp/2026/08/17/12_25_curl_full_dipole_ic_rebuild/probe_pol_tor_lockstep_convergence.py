"""Resolution-convergence sweep for the pol/tor lockstep question -- does
the earlier finding (u_rms swings ~2x non-monotonically under pure wave
dynamics, no step_cross, while macro ring circulation stays exactly frozen)
hold up as the object is made bigger relative to the fixed discrete floor
(cell size = 1 everywhere in this uniform L0 mesh), or is it itself just an
under-resolution artifact that changes as cells_across grows?

Philip's point (2026-08-17): cells_across=220 was calibrated for a DIFFERENT
purpose (avoiding the old dipole IC's L8 AMR-junction blowup,
project_hop_telescope_resolved_spheromak) -- it was never validated as
"enough" for THIS question. We don't actually know yet. This sweeps
cells_across = [22, 55, 110] fresh here (cheap); 220 is covered separately
by probe_pol_tor_lockstep_production.py (already running against the real
checkpoint) -- combine both tables to see whether the oscillation amplitude
is converging to a fixed answer as cells_across grows, or still drifting.

Same relative geometry at every cells_across (padding/SCALE ratio and
hops/cells_across ratio both held fixed at the production values:
padding=70 at cells_across=220 -> ratio 70/110; hops=41 at 220 -> ratio
41/220) so cells_across is the ONLY varying parameter.
"""

import sys

sys.path.insert(0, ".")
import time

import numpy as np

from equilibrium import lattice as _lat
from equilibrium.substrate.curl_full_dipole_ic import seed_curl_full_dipole_ic
from equilibrium.substrate.curl_full_spheromak_ic import find_square_ring_on_leaves
from tst.substrate.adaptive.adaptive_flux_tube import (
    _FACE_HOP,
    _FACE_NORM,
    phi_ring_circulation,
)
from run.substrate.adaptive.sweep_spheromak_decay_cross_scale import _compute_dt_l0

R_SPH = 1.0
U_MAX = 1.0
AMPLITUDE_POL, AMPLITUDE_TOR = 1.0, 1.0
N_STEPS = 60
REPORT_EVERY = 10
PADDING_RATIO = 70.0 / 110.0  # matches production build_curlfull_dipole_uniform_r180.py
HOPS_RATIO = (
    41.0 / 220.0
)  # matches curl_full_spheromak_ic's production hops=41 @ cells_across=220

CELLS_ACROSS_LIST = [22, 55, 110]


def run_one(cells_across):
    scale = cells_across / 2.0
    padding = PADDING_RATIO * scale
    r_domain = scale + padding
    hops = max(1, round(HOPS_RATIO * cells_across))

    t0 = time.time()
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    alat.set_n_threads(8)
    cl = alat[0]
    _lat.build_bcc_sphere(cl, r_domain, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)
    N = cl.n_leaves
    print(
        f"[cells_across={cells_across}] R_domain={r_domain:.2f} n_leaves={N} hops={hops} "
        f"build_t={time.time()-t0:.1f}s",
        flush=True,
    )

    pos = np.array(cl.leaf_pos)
    cell_size = np.full(N, 1.0)
    seed_curl_full_dipole_ic(
        cl,
        pos,
        cell_size,
        scale,
        _FACE_HOP,
        _FACE_NORM,
        r_sph=R_SPH,
        u_max=U_MAX,
        amplitude_pol=AMPLITUDE_POL,
        amplitude_tor=AMPLITUDE_TOR,
    )

    ring_js, ring_faces = find_square_ring_on_leaves(cl, axis="z", hops=hops)
    if ring_js is None:
        print(
            f"[cells_across={cells_across}] ring-finding FAILED at hops={hops} -- skipping",
            flush=True,
        )
        return
    centre = pos[ring_js].mean(axis=0)

    u0 = float(np.sqrt(np.mean(np.array(cl.leaf_u) ** 2)))
    circ0 = phi_ring_circulation(cl, ring_js, centre)
    print(
        f"[cells_across={cells_across}] t=0  u_rms={u0:.6e}  circ={circ0:.6e}",
        flush=True,
    )

    dt = _compute_dt_l0(cl, D1=1.0)
    ts = _lat.TimeStepper.by_level(dt)
    evolver = _lat.AdaptiveLatticeEvolver(alat)

    max_ratio, min_ratio = 1.0, 1.0
    for step_i in range(1, N_STEPS + 1):
        evolver.step(ts)  # pure wave dynamics, NO step_cross
        if step_i % REPORT_EVERY == 0:
            u_rms = float(np.sqrt(np.mean(np.array(cl.leaf_u) ** 2)))
            circ = phi_ring_circulation(cl, ring_js, centre)
            ratio = u_rms / u0
            max_ratio, min_ratio = max(max_ratio, ratio), min(min_ratio, ratio)
            print(
                f"[cells_across={cells_across}] step {step_i:3d}  u_rms_ratio={ratio:.6f}  "
                f"circ_ratio={circ/circ0:.6f}",
                flush=True,
            )
    print(
        f"[cells_across={cells_across}] SUMMARY: u_rms_ratio range [{min_ratio:.4f}, {max_ratio:.4f}]  "
        f"swing={max_ratio-min_ratio:.4f}  total_t={time.time()-t0:.1f}s",
        flush=True,
    )


def main():
    for ca in CELLS_ACROSS_LIST:
        run_one(ca)


if __name__ == "__main__":
    main()
