"""
Does electric (u) <-> magnetic (circulation) energy exchange appear once
sigma varies spatially, matching the doc/dst.md vacuum-hierarchy picture
(lam=0 only at true void; a region near mass carries a nonzero background
lam falling off with distance, per the same 1/r shape as u=GM/(c^2 r))?

Uses the WEIGHT-CONSISTENT (all-square-face, uniform W_sq throughout the
loop) plaquette for "magnetic energy" this time -- NOT the 2-hex+1-sq
triangle (discrete_curl.py), which mixes W_hex=3/W_sq=1 and has a real,
derived non-zero circulation source term (-2*dt*sigma*(u_C-u_A) per step)
independent of any background lam. That triangle is the real neutron
winding topology (doc/dst.md), just the wrong tool for a conserved
circulation/magnetic-energy proxy. mhd.py's SQ_PLAQUETTES (equal W_sq on
all 4 legs) telescopes to an EXACT zero source term whenever sigma is
uniform across the loop -- confirmed analytically and empirically earlier
this session. If sigma varies (nonzero background lam), that cancellation
breaks -- this script tests whether that predicted break actually produces
real, nonzero electric<->magnetic conversion, not just noise.

Two runs:
  A) lam=0 everywhere (pure void baseline -- expect exactly-frozen magnetic
     energy, matching the earlier mhd.py result)
  B) background lam(r) = LAM_SURFACE * R_SPH/max(r_cell, R_SPH), r_cell in
     the same dimensionless units seed_curl_full_dipole_ic uses (r_sph=1 at
     the object's own surface) -- a mild, physically-shaped falloff, not an
     arbitrarily large contrast
"""

import sys

sys.path.insert(0, ".")

import numpy as np

from equilibrium import lattice as _lat
from equilibrium.substrate.curl_full_dipole_ic import seed_curl_full_dipole_ic
from tst.substrate.adaptive.adaptive_flux_tube import _FACE_HOP, _FACE_NORM, leaf_nb_j

CELLS_ACROSS = 22
R_SPH = 1.0
R_DOMAIN = 18.0
SCALE = CELLS_ACROSS / 2.0
N_STEPS = 30
REPORT_EVERY = 5
LAM_SURFACE = 1.0  # order-unity, qualitative existence check -- NOT a calibrated value

SQ_PLAQUETTES = {
    "z": (8, 10, 9, 11),
    "y": (8, 12, 9, 13),
    "x": (10, 12, 11, 13),
}


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


def run(label, lam_background):
    alat = _lat.AdaptiveLattice()
    cl = alat[0]
    _lat.build_bcc_sphere(cl, R_DOMAIN, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)
    alat.set_n_threads(8)
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
        amplitude_tor=1.0,
    )

    if lam_background:
        r_cell = np.linalg.norm(
            pos / SCALE, axis=1
        )  # dimensionless, r_sph=1 at object surface
        lam = LAM_SURFACE * R_SPH / np.maximum(r_cell, R_SPH)
        cl.set_leaf_lam(lam)
        print(
            f"[{label}] background lam: min={lam.min():.4e} max={lam.max():.4e} "
            f"at_domain_edge={lam.min():.4e}",
            flush=True,
        )

    dt = 0.4 / 6.0
    ts = _lat.TimeStepper.by_level(dt)
    evolver = _lat.AdaptiveLatticeEvolver(alat)

    def energies():
        u = np.array(cl.leaf_u)
        phi_T = np.array(cl.leaf_phi).T
        B = b_vec_vectorized(phi_T, nb_j)
        return 0.5 * float(np.sum(u**2)), 0.5 * float(np.sum(B**2))

    e0, m0 = energies()
    print(
        f"[{label}] t=0  electric={e0:.6e}  magnetic={m0:.6e}  total={e0+m0:.6e}",
        flush=True,
    )

    for step_i in range(1, N_STEPS + 1):
        evolver.step(ts)
        if step_i % REPORT_EVERY == 0:
            e, m = energies()
            tot = e + m
            print(
                f"[{label}] step {step_i:3d}/{N_STEPS}  electric={e:.6e} ({e/e0:.4f}x)  "
                f"magnetic={m:.6e} ({m/max(m0,1e-300):.6f}x)  total_ratio={tot/(e0+m0):.6f}",
                flush=True,
            )


if __name__ == "__main__":
    run("A lam=0 (void)      ", lam_background=False)
    print(flush=True)
    run("B lam(r)~1/r (mass) ", lam_background=True)
