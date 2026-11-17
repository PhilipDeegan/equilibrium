"""
Does the double-ring (poloidal) dipole electron IC show g~2 relative to a
single ring (g~1), matching the electron's real spin g-factor anomaly?

Motivation (2026-09-21 conversation): setting a self-circulating mass's
angular momentum L=m_e*c*r_Z equal to the electron's actual spin ħ/2 gives
the standard Zitterbewegung radius r_Z=ħ/(2*m_e*c). At that SAME radius, a
single classical current loop's magnetic moment is exactly HALF the
electron's real magnetic moment (mu_classical=(e/2m_e)*L, the textbook
g=1 orbital prediction) -- the real electron needs g~2, i.e. effectively
TWO windings' worth of circulation at the same radius, not one. This
converges with two independent things already on record: the double-ring
dipole IC itself (project_dst_electron_dipole) and the Dirac
chirality-mixing "two components needed for mass" argument
(project_chirality_mixing_two_component_mass). This project has no
validated cell-to-physical-length calibration yet (project_derive_dont_
pick_parameters), so we can't check the ABSOLUTE g-factor -- but we CAN
check the dimensionless RATIO (moment/spin)_double-ring / (moment/spin)
_single-ring against 2.0, entirely inside this project's own units.

Method: build a single-ring "one winding" structure (seed_constant_current
-- genuine, validated nonzero-circulation current, NOT the degenerate
standing-wave winding) and the double-ring dipole IC on the SAME ring
geometry (same radius), and measure BOTH with the SAME functions:
  - circulation Gamma = phi_ring_circulation(ring_js, centre) -- the
    magnetic-moment-like proxy (moment ~ Gamma * Area, but Area is the
    SAME for both cases here so it cancels in the ratio).
  - H_near = field energy in a window around the ring -- the spin-proxy
    denominator (kinematic L ~ H*r/c, r and c also cancel in the ratio
    since the SAME ring geometry is used for both).
So the ratio reduces to (Gamma_double/H_double) / (Gamma_single/H_single)
-- no absolute calibration needed, purely internal to this model.

Both measurements avoid b_vec_vectorized entirely (project_discrete_curl_
directional_bias) -- phi_ring_circulation uses all 14 face directions
directly, no curl reconstruction.
"""

import sys

sys.path.insert(0, "/home/p/git/eq")

import numpy as np

from equilibrium import lattice as _lat
from equilibrium.substrate.lattice import W_HEX, W_SQ
from tst.substrate.adaptive.adaptive_flux_tube import (
    find_toroidal_ring,
    phi_ring_circulation,
    init_electron_dipole_ic_vecpot,
    _MIRROR,
)

R_DOMAIN = 40.0
HOPS = 6
SPONGE_FRACTION = 0.15
DAMP_MAX = 0.003
N_THREADS = 8
AMPLITUDE = 1.0
WINDOW_FRAC = 1.5  # near-ring energy window, in units of r_ring

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
    if ring_js is None:
        raise RuntimeError("ring-finding failed")
    pos = np.array(cl.leaf_pos)
    centre = pos[ring_js].mean(axis=0)
    r_ring = float(np.mean(np.linalg.norm(pos[ring_js, :2] - centre[:2], axis=1)))
    print(f"N={cl.n_leaves}  ring tiles={len(ring_js)}  r_ring={r_ring:.3f}", flush=True)

    # --- single-ring (one winding) ---
    seed_constant_current(cl, ring_js, ring_faces, amplitude=AMPLITUDE)
    gamma_single = phi_ring_circulation(cl, ring_js, centre)
    H_single = energy_near(cl, centre, r_ring)
    print(f"single-ring : Gamma={gamma_single:.4e}  H_near={H_single:.4e}  Gamma/H={gamma_single/H_single:.4e}", flush=True)

    # --- double-ring dipole (poloidal, two lobes) ---
    # scale calibrated so the dipole's own r_axis lands close to r_ring
    # (empirically r_axis ~ 0.6106*scale for this construction)
    scale_guess = r_ring / 0.6106
    _, r_axis = init_electron_dipole_ic_vecpot(
        cl, R_sph=1.0, scale=scale_guess, B0=1.0, lam_target=0.0, u_bg=0.0,
        amplitude=AMPLITUDE, div_clean=True, clean_rtol=1e-8,
    )
    print(f"double-ring : scale={scale_guess:.3f}  r_axis={r_axis:.3f}  (target r_ring={r_ring:.3f})", flush=True)
    gamma_double = phi_ring_circulation(cl, ring_js, centre)
    H_double = energy_near(cl, centre, r_ring)
    print(f"double-ring : Gamma={gamma_double:.4e}  H_near={H_double:.4e}  Gamma/H={gamma_double/H_double:.4e}", flush=True)

    ratio_single = gamma_single / H_single
    ratio_double = gamma_double / H_double
    g_eff = ratio_double / ratio_single
    print(f"\n=== result ===", flush=True)
    print(f"(Gamma/H)_double / (Gamma/H)_single = {g_eff:.4f}   (target ~2.0 if this matches the electron's g-factor anomaly)", flush=True)


if __name__ == "__main__":
    main()
