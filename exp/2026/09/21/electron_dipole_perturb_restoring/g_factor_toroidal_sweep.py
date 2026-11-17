"""
Electron-scoped follow-up to g_factor_ratio_test.py: does layering a SMALL
toroidal admixture on top of the existing poloidal-only double-ring dipole
(via apply_toroidal_rotation, already built for "a spherical knot") change
the (Gamma/H) ratio toward the target g~2, or change how close the
combined structure is to being invariant (rate-of-change over a short
window, same discipline as electron_charge_test_v4_resolved.py)?

Explicitly scoped to the ELECTRON (poloidal-dominant, toroidal as a small
perturbation) -- NOT the neutron's ~1:1 metastable configuration, per
Philip's correction that the neutron is a bad stand-in (it's a specific,
energetically metastable state that NEEDS external pressure to stay
closed, not evidence that poloidal+toroidal combinations are generally
unstable). Sweeps the toroidal:poloidal amplitude ratio at small-to-
moderate values, not assuming any particular ratio is "the" answer.
"""

import sys
import time

sys.path.insert(0, "/home/p/git/eq")

import numpy as np

from equilibrium import lattice as _lat
from equilibrium.substrate.lattice import W_HEX, W_SQ
from tst.substrate.adaptive import compute_dt
from tst.substrate.adaptive.adaptive_flux_tube import (
    find_toroidal_ring,
    phi_ring_circulation,
    init_electron_dipole_ic_vecpot,
    apply_toroidal_rotation,
    _MIRROR,
)

R_DOMAIN = 40.0
HOPS = 6
SPONGE_FRACTION = 0.15
DAMP_MAX = 0.003
N_THREADS = 8
AMPLITUDE = 1.0
WINDOW_FRAC = 1.5
TOROIDAL_EPS = [0.0, 0.1, 0.3, 0.5, 1.0]
N_STEPS_RATE = 300  # short-window invariance check, same discipline as v4

_W = np.array([W_HEX] * 8 + [W_SQ] * 6, dtype=np.float64)


def seed_constant_current(cl, ring_js, ring_faces, amplitude=1.0):
    N_ring = len(ring_js)
    phi = cl.leaf_phi.copy()
    for idx in range(N_ring):
        j_i = ring_js[idx]
        j_nxt = ring_js[(idx + 1) % N_ring]
        k = ring_faces[idx]
        mk = int(_MIRROR[k])
        a = amplitude / _W[k]
        phi[k, j_i] = a
        phi[mk, j_nxt] = -a
    cl.set_leaf_phi(phi)


def build_cluster():
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    alat.set_n_threads(N_THREADS)
    cl = alat[0]
    _lat.build_bcc_sphere(cl, R_DOMAIN, d1=1.0, u0=0.0, ell0=1.0, lam0=0.0)
    alat.rebuild(0)
    r_all = np.linalg.norm(np.array(cl.leaf_pos), axis=1)
    r_inner = float(r_all.max()) * (1.0 - SPONGE_FRACTION)
    cl.set_bc(
        _lat.BoundaryCondition.quadratic_sponge(
            target=0.0, damp_max=DAMP_MAX, r_all=r_all, r_inner=r_inner
        )
    )
    return alat, cl


def energy_near(cl, centre, r_ring):
    pos = np.array(cl.leaf_pos)
    u = np.array(cl.leaf_u)
    phi = np.array(cl.leaf_phi)
    mask = np.linalg.norm(pos - centre, axis=1) <= WINDOW_FRAC * r_ring
    return float(np.sum(u[mask] ** 2) + 0.5 * np.sum(phi[:, mask] ** 2))


def main():
    alat, cl = build_cluster()
    ring_js, ring_faces = find_toroidal_ring(cl, axis="z", mode="octagon", hops=HOPS)
    pos = np.array(cl.leaf_pos)
    centre = pos[ring_js].mean(axis=0)
    r_ring = float(np.mean(np.linalg.norm(pos[ring_js, :2] - centre[:2], axis=1)))
    dt = compute_dt(cl)
    print(f"N={cl.n_leaves}  ring tiles={len(ring_js)}  r_ring={r_ring:.3f}  dt={dt:.4e}", flush=True)

    seed_constant_current(cl, ring_js, ring_faces, amplitude=AMPLITUDE)
    gamma_single = phi_ring_circulation(cl, ring_js, centre)
    H_single = energy_near(cl, centre, r_ring)
    ratio_single = gamma_single / H_single
    print(f"single-ring baseline: Gamma={gamma_single:.4e}  H={H_single:.4e}  Gamma/H={ratio_single:.4e}\n", flush=True)

    scale_guess = r_ring / 0.6106
    print(f"{'eps':>6} {'Gamma':>10} {'H_near':>10} {'Gamma/H':>10} {'g_eff':>8} | {'rate(300 steps)':>16}", flush=True)
    for eps in TOROIDAL_EPS:
        _, r_axis = init_electron_dipole_ic_vecpot(
            cl, R_sph=1.0, scale=scale_guess, B0=1.0, lam_target=0.0, u_bg=0.0,
            amplitude=AMPLITUDE, div_clean=True, clean_rtol=1e-8,
        )
        if eps > 0.0:
            apply_toroidal_rotation(cl, centre=np.zeros(3), R_core=scale_guess, amplitude=eps, spin_axis="z")

        gamma = phi_ring_circulation(cl, ring_js, centre)
        H = energy_near(cl, centre, r_ring)
        g_eff = (gamma / H) / ratio_single

        # short-window invariance check
        H0 = H
        t0 = time.time()
        for _ in range(N_STEPS_RATE):
            alat.step(0, dt)
            alat.step_sponge(0)
        H1 = energy_near(cl, centre, r_ring)
        rate = (H1 - H0) / H0 / (N_STEPS_RATE * dt) if H0 else float("nan")

        print(f"{eps:6.2f} {gamma:10.4e} {H:10.4e} {gamma/H:10.4e} {g_eff:8.4f} | {rate:+16.4e}  ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
