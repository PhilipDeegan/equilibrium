"""
v2: use the doc-consistent, currently-validated discrete_curl.py (2-hex+1-sq
triangle plaquette) instead of the older equilibrium.substrate.mhd module
(4-square-face-only SQ_PLAQUETTES, predates discrete_curl.py, not the
canonical BCC triangular loop doc/dst.md itself describes) for "magnetic
energy" -- Philip flagged mhd.py may not be valid anymore.

magnetic_energy = 0.5 * sum(circulation(phi, nb)**2) over all 12*N valid
elementary triangles (equilibrium.substrate.discrete_curl.circulation).
"""

import sys

sys.path.insert(0, ".")

import numpy as np

from equilibrium import lattice as _lat
from equilibrium.substrate.curl_full_dipole_ic import seed_curl_full_dipole_ic
from equilibrium.substrate.discrete_curl import circulation, leaf_nb_rows
from tst.substrate.adaptive.adaptive_flux_tube import (
    _FACE_HOP,
    _FACE_NORM,
    find_toroidal_ring,
    phi_ring_circulation,
)

CELLS_ACROSS = 22
R_SPH = 1.0
R_DOMAIN = 18.0
SCALE = CELLS_ACROSS / 2.0
N_STEPS = 60
REPORT_EVERY = 5


def build():
    alat = _lat.AdaptiveLattice()
    cl = alat[0]
    _lat.build_bcc_sphere(cl, R_DOMAIN, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)
    return alat, cl


def run(amplitude_tor, label):
    alat, cl = build()
    pos = np.array(cl.leaf_pos)
    cell_size = np.full(cl.n_leaves, 1.0)
    nb = leaf_nb_rows(cl)

    seed_curl_full_dipole_ic(
        cl,
        pos,
        cell_size,
        SCALE,
        _FACE_HOP,
        _FACE_NORM,
        r_sph=R_SPH,
        amplitude_pol=1.0,
        amplitude_tor=amplitude_tor,
    )

    ring_js, ring_faces = find_toroidal_ring(cl, axis="z", mode="square", hops=3)
    assert ring_js is not None
    centre = pos[ring_js].mean(axis=0)

    dt = 0.4 / 6.0
    ts = _lat.TimeStepper.by_level(dt)
    evolver = _lat.AdaptiveLatticeEvolver(alat)

    def energies():
        u = np.array(cl.leaf_u)
        phi = np.array(
            cl.leaf_phi
        )  # (14, N) -- matches discrete_curl's expected layout
        circ = circulation(phi, nb)
        e = 0.5 * float(np.sum(u**2))
        m = 0.5 * float(np.sum(circ**2))
        return e, m

    e0, m0 = energies()
    circ0 = phi_ring_circulation(cl, ring_js, centre)
    print(
        f"[{label}] t=0  electric={e0:.6e}  magnetic={m0:.6e}  total={e0+m0:.6e}  macro_circ={circ0:.6e}",
        flush=True,
    )

    for step_i in range(1, N_STEPS + 1):
        evolver.step(ts)
        if step_i % REPORT_EVERY == 0:
            e, m = energies()
            circ = phi_ring_circulation(cl, ring_js, centre)
            tot = e + m
            print(
                f"[{label}] step {step_i:4d}  electric={e:.6e} ({e/e0:.4f}x)  "
                f"magnetic={m:.6e} ({m/max(m0,1e-300):.6f}x)  "
                f"total={tot:.6e} ({tot/(e0+m0):.6f}x)  macro_circ_ratio={circ/circ0:.6f}",
                flush=True,
            )


if __name__ == "__main__":
    print("=== Run A: amplitude_tor=0.0 (poloidal-only floor) ===", flush=True)
    run(0.0, "tor=0")
    print()
    print("=== Run B: amplitude_tor=1.0 (full curl-full dipole) ===", flush=True)
    run(1.0, "tor=1")
