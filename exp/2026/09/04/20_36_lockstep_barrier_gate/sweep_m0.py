"""
Follow-up to probe_lockstep_barrier_gate.py: the derived m0=1.494 (rescaled
from doc/dst_next.md sec 8's neutron calibration) turned out to sit far
below this IC's actual drive distribution (median nonzero drive 4.46,
max 9.38) -- 87% of firing cells exceed it, so barrier collapsed to v1.

Philip's prompt: m0 was higher in prior sessions (2.49 un-rescaled, or 12
"empirically found, fixes R1" -- 2026-08-26 entry in now). Sweep m0 across
and beyond the real drive range to see whether the barrier gate can EVER
behave genuinely differently from v1 on this object (i.e. whether there's
an m0 regime where it doesn't just always-fire), rather than testing only
the one derived value.
"""

import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium.substrate.dipole_mode_ic import K_FACTOR, U_and_gradU  # noqa: F401
from equilibrium.substrate.lattice import DampedLattice

sys.path.insert(0, "exp/2026/09/04/20_36_lockstep_barrier_gate")
from probe_lockstep_barrier_gate import (  # noqa: E402
    R,
    R_SPH,
    add_toroidal,
    build_poloidal,
)

CROSS_SCALE = 1e-3
N_STEPS = 2000
REPORT = 500
N_THREADS = 8
M0_VALUES = [1.494, 2.49, 4.46, 7.0, 9.38, 12.0, 20.0]


def drive_stats(lat):
    W = np.where(np.arange(14) < 8, 3.0, 1.0)
    drive = np.sum(W[None, :] * np.maximum(-lat.phi, 0), axis=1)
    nz = drive[drive > 0]
    return drive, nz


def run(m0):
    lat = DampedLattice(radius_cells=R, sponge_fraction=0.15, damp_max=0.003)
    build_poloidal(lat, R_SPH)
    add_toroidal(lat, R_SPH)
    dt = lat.stable_dt()

    circ0_drive, nz0 = drive_stats(lat)
    frac_fire_t0 = np.mean(nz0 > m0) if len(nz0) else float("nan")

    from probe_lockstep_barrier_gate import _phi_circulation

    circ0 = _phi_circulation(lat, np.zeros(3), 4.0)
    for step in range(N_STEPS + 1):
        lat.step(dt, n_threads=N_THREADS)
        lat.step_cross(dt, CROSS_SCALE, n_threads=N_THREADS, gate="barrier", m0=m0)
    circ_end = _phi_circulation(lat, np.zeros(3), 4.0)
    print(
        f"m0={m0:6.3f}  frac(nonzero-drive cells firing @t0)={frac_fire_t0:.3f}"
        f"  circ_ratio@{N_STEPS}={circ_end/circ0:.6f}",
        flush=True,
    )


def main():
    lat = DampedLattice(radius_cells=R, sponge_fraction=0.15, damp_max=0.003)
    build_poloidal(lat, R_SPH)
    add_toroidal(lat, R_SPH)
    drive, nz = drive_stats(lat)
    print(f"drive: max={drive.max():.4f} median(nz)={np.median(nz):.4f} nnz={len(nz)}/{len(drive)}")
    for m0 in M0_VALUES:
        run(m0)


if __name__ == "__main__":
    main()
