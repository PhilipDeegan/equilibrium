"""
Hertzian (point oscillating electric) dipole -- first pass, cheap, sanity
check the near field before pushing to a genuine far-field comparison.

Pegged values (not picked ad hoc):
  - c = sqrt(10)  (cell units, from the already-validated dispersion
    relation omega^2 = W_HEX^2*8*(1-cos(...)) + W_SQ^2*(...), continuum
    limit K->0)
  - epsilon_0 = 1  (definitional, doc/dst.md: Q := sum(A_ij du/dr), no
    separate scaling constant)
  - mu_0 = 1/(epsilon_0*c^2) = 0.1  (derived, NOT the naive mu_0=1 guess)
  - wavelength = 8 cells (doc/dst_next.md's recommended safety margin
    against BCC dispersion anisotropy, not today's more lenient
    shift-test result)
  - source radius ~1.5 cells (small compared to wavelength -> small ka,
    point-like without degenerating to a single discrete cell)
  - amplitude = 1 (normalization, not a physical peg)

Method: continuously DRIVE u on the small source region each step to
A*cos(theta)*cos(omega*t) (dipolar angular pattern x oscillation),
everything else evolves via ordinary step() (no step_cross). Same
established pattern as equilibrium/substrate/double_slit.py's
continuously-driven source (avoids needing to analytically derive an
"already outgoing" IC for a localized 3D source, much harder than the
infinite-plane-wave case).
"""

import sys

sys.path.insert(0, ".")
import numpy as np
from equilibrium.substrate.lattice import DampedLattice, W_HEX, W_SQ
from equilibrium.substrate.mhd import _SQ_AXIS_PAIRS, SQ_PLAQUETTES

C_DST = np.sqrt(10.0)
EPS0 = 1.0
MU0 = 1.0 / (EPS0 * C_DST**2)

R = 40.0
WAVELENGTH_CELLS = 8.0
SOURCE_RADIUS = 1.5
AMPLITUDE = 1.0
SPONGE_FRACTION = 0.15


def _plaquette_vectorized(phi, nb, faces):
    N = phi.shape[0]
    c0 = np.arange(N)
    ka, kb, kc, kd = faces
    c1 = nb[c0, ka]
    valid = c1 >= 0
    c1s = np.where(valid, c1, 0)
    c2 = nb[c1s, kb]
    valid &= c2 >= 0
    c2s = np.where(c2 >= 0, c2, 0)
    c3 = nb[c2s, kc]
    valid &= c3 >= 0
    c3s = np.where(c3 >= 0, c3, 0)
    close = nb[c3s, kd]
    valid &= close == c0
    term = phi[c0, ka] + phi[c1s, kb] + phi[c2s, kc] + phi[c3s, kd]
    return np.where(valid, term, 0.0)


def b_vec_vectorized(phi, nb):
    return np.stack(
        [_plaquette_vectorized(phi, nb, SQ_PLAQUETTES[k]) for k in "xyz"], axis=1
    )


def face_vec_vectorized(dphi):
    return np.stack([dphi[:, fp] - dphi[:, fm] for fp, fm in _SQ_AXIS_PAIRS], axis=1)


def main():
    lat = DampedLattice(radius_cells=R, sponge_fraction=SPONGE_FRACTION, damp_max=0.003)
    nb = lat.nb
    print(
        f"N={lat.N}  R={R}  C_DST={C_DST:.4f}  EPS0={EPS0}  MU0={MU0:.4f}", flush=True
    )

    r_all = np.linalg.norm(lat.pos, axis=1)
    theta_all = np.arccos(np.clip(lat.pos[:, 2] / np.maximum(r_all, 1e-9), -1, 1))
    src_mask = r_all < SOURCE_RADIUS
    print(f"source cells: {src_mask.sum()}", flush=True)

    # derive a driving omega that's safely in the propagating band: use the
    # continuum-limit dispersion relation with |K|=2pi/wavelength along z
    # (the established test_maxwell.py formula, not a fresh guess)
    K = 2 * np.pi / WAVELENGTH_CELLS
    omega = np.sqrt(
        8 * W_HEX**2 * (1 - np.cos(K * 0.5)) + 2 * W_SQ**2 * (1 - np.cos(K))
    )
    period = 2 * np.pi / omega
    print(f"K={K:.4f}  omega={omega:.4f}  period={period:.3f}", flush=True)

    dt = lat.stable_dt()
    r_fit_min, r_fit_max = 10.0, 0.75 * R
    fit_mask = (r_all >= r_fit_min) & (r_all <= r_fit_max)
    print(
        f"fit range r=[{r_fit_min},{r_fit_max}]  n_fit_cells={fit_mask.sum()}",
        flush=True,
    )

    # size n_steps for arrival at r_fit_max PLUS one full settled period
    # beyond it -- same fix that resolved the Poynting-ring far-zone saga
    t_arrive = r_fit_max / C_DST
    t_target = t_arrive + period
    n_steps = int(round(t_target / dt))
    print(
        f"dt={dt:.5f}  t_arrive={t_arrive:.3f}  t_target={t_target:.3f}  n_steps={n_steps}  "
        f"({t_target/period:.2f} periods)",
        flush=True,
    )

    cos_theta_src = np.cos(theta_all[src_mask])

    for step in range(n_steps):
        t = step * dt
        lat.u[src_mask] = AMPLITUDE * cos_theta_src * np.cos(omega * t)
        lat.step(dt, n_threads=8)
        if step % max(1, n_steps // 10) == 0:
            print(f"  step={step}/{n_steps}", flush=True)

    print("done stepping, computing E/B/S...", flush=True)
    phi_before = lat.phi.copy()
    t = n_steps * dt
    lat.u[src_mask] = AMPLITUDE * cos_theta_src * np.cos(omega * t)
    lat.step(dt, n_threads=8)
    phi_after = lat.phi.copy()
    dphi = phi_after - phi_before

    E3 = -face_vec_vectorized(dphi) / dt
    B3 = b_vec_vectorized(phi_after, nb) / MU0  # peg mu_0 explicitly, not =1
    S3 = np.cross(E3, B3)
    Smag = np.linalg.norm(S3, axis=1)

    equatorial = (theta_all > np.radians(70)) & fit_mask
    polar = (theta_all < np.radians(20)) & fit_mask
    print(
        f"equatorial cells: {equatorial.sum()}  polar cells: {polar.sum()}", flush=True
    )
    if equatorial.sum() > 5 and polar.sum() > 5:
        s_eq = (
            np.percentile(Smag[equatorial][Smag[equatorial] > 0], 90)
            if (Smag[equatorial] > 0).any()
            else 0.0
        )
        s_pol = (
            np.percentile(Smag[polar][Smag[polar] > 0], 90)
            if (Smag[polar] > 0).any()
            else 0.0
        )
        print(
            f"S (p90) equatorial={s_eq:.4e}  polar={s_pol:.4e}  ratio={s_eq/max(s_pol,1e-300):.3f}"
            f"  (classical Hertzian dipole predicts >>1, sin^2(theta) peaks at equator)",
            flush=True,
        )
    else:
        print("insufficient nonzero cells for a sector comparison", flush=True)


if __name__ == "__main__":
    main()
