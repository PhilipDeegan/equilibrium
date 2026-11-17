"""
Does DST's plain wave dynamics (evolver.step(), NO step_cross) actually
convert energy between an "electric-like" (u) form and a "magnetic-like"
(curl(phi)=B) form, the way a real EM wave exchanges E<->B energy? Or do
they just evolve independently/incoherently?

Builds on the already-existing curl-full dipole IC (seed_curl_full_dipole_ic
-- separable poloidal u/phi_pol (electric-like, curl-free) + toroidal phi_tor
(magnetic-like, curl-full) halves) and the small test fixture already used
by probe_pol_tor_lockstep.py (CELLS_ACROSS=22, seconds to run).

Definitions (energy, not just amplitude/circulation):
  electric_energy = 0.5 * sum(u_i^2)                    -- the "potential"
  magnetic_energy = 0.5 * sum(|B_i|^2), B = curl(phi)    -- discrete curl,
                     vectorized equivalent of mhd._b_vec via square-face
                     plaquettes (mhd.SQ_PLAQUETTES)
  total_energy     = electric_energy + magnetic_energy

Known caveat (project_dst_discrete_curl / test_curl_full_dipole_ic.py):
the LOCAL per-plaquette curl of a purely poloidal (phi_pol-only, curl-free
in the continuum) field is NOT exactly zero -- it has a real ~0.2-0.3
relative curvature/sampling artifact. So magnetic_energy computed this way
has a nonzero floor even with phi_tor=0. This script measures that floor
directly (phi_tor amplitude=0 run) so the phi_tor>0 run's magnetic_energy
can be judged against it, not against a naive zero baseline.

Two runs:
  A) amplitude_tor=0.0  -- poloidal-only, measures the curl-artifact FLOOR
  B) amplitude_tor=1.0  -- full curl-full dipole, the real test
"""

import sys

sys.path.insert(0, ".")

import numpy as np

from equilibrium import lattice as _lat
from equilibrium.substrate.curl_full_dipole_ic import seed_curl_full_dipole_ic
from equilibrium.substrate.mhd import SQ_PLAQUETTES, _SQ_AXIS_PAIRS
from tst.substrate.adaptive.adaptive_flux_tube import (
    _FACE_HOP,
    _FACE_NORM,
    find_toroidal_ring,
    leaf_nb_j,
    phi_ring_circulation,
)

CELLS_ACROSS = 22
R_SPH = 1.0
R_DOMAIN = 18.0
SCALE = CELLS_ACROSS / 2.0
N_STEPS = 120
REPORT_EVERY = 5


def _plaquette_vectorized(phi_T, nb_j, faces):
    N = phi_T.shape[0]
    c0 = np.arange(N)
    ka, kb, kc, kd = faces
    c1 = nb_j[c0, ka]
    valid = c1 >= 0
    c1s = np.where(valid, c1, 0)
    c2 = nb_j[c1s, kb]
    valid &= c2 >= 0
    c2s = np.where(c2 >= 0, c2, 0)
    c3 = nb_j[c2s, kc]
    valid &= c3 >= 0
    c3s = np.where(c3 >= 0, c3, 0)
    close = nb_j[c3s, kd]
    valid &= close == c0
    term = phi_T[c0, ka] + phi_T[c1s, kb] + phi_T[c2s, kc] + phi_T[c3s, kd]
    return np.where(valid, term, 0.0)


def b_vec_vectorized(phi_T, nb_j):
    Bx = _plaquette_vectorized(phi_T, nb_j, SQ_PLAQUETTES["x"])
    By = _plaquette_vectorized(phi_T, nb_j, SQ_PLAQUETTES["y"])
    Bz = _plaquette_vectorized(phi_T, nb_j, SQ_PLAQUETTES["z"])
    return np.stack([Bx, By, Bz], axis=1)


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
    nb_j = leaf_nb_j(cl)

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
        phi_T = np.array(cl.leaf_phi).T
        B = b_vec_vectorized(phi_T, nb_j)
        e = 0.5 * float(np.sum(u**2))
        m = 0.5 * float(np.sum(B**2))
        return e, m

    e0, m0 = energies()
    circ0 = phi_ring_circulation(cl, ring_js, centre)
    print(
        f"[{label}] t=0  electric={e0:.6e}  magnetic={m0:.6e}  total={e0+m0:.6e}  circ={circ0:.6e}",
        flush=True,
    )

    rows = [(0, e0, m0, e0 + m0, circ0)]
    for step_i in range(1, N_STEPS + 1):
        evolver.step(ts)
        if step_i % REPORT_EVERY == 0:
            e, m = energies()
            circ = phi_ring_circulation(cl, ring_js, centre)
            tot = e + m
            print(
                f"[{label}] step {step_i:4d}  electric={e:.6e} ({e/e0:.4f}x)  "
                f"magnetic={m:.6e} ({m/max(m0,1e-300):.4f}x)  "
                f"total={tot:.6e} ({tot/(e0+m0):.6f}x)  circ_ratio={circ/circ0:.6f}",
                flush=True,
            )
            rows.append((step_i, e, m, tot, circ))
    return rows


if __name__ == "__main__":
    print(
        "=== Run A: amplitude_tor=0.0 (poloidal-only -- measures the curl-artifact floor) ===",
        flush=True,
    )
    rows_a = run(0.0, "tor=0")
    print()
    print(
        "=== Run B: amplitude_tor=1.0 (full curl-full dipole -- the real test) ===",
        flush=True,
    )
    rows_b = run(1.0, "tor=1")
