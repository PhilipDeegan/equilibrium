"""
Discrete-native Hertzian dipole: instead of continuously sampling cos(theta)
onto a sphere of cells (confirmed structural, not resolution, mismatch --
ratio pinned at ~0.28-0.30 regardless of source radius 1.5-6.0), drive
EXACTLY two cells -- the +z and -z square-face neighbours of the origin,
the lattice's own exact discrete realization of the l=1 (dipole) harmonic
(the 6 square-face directions ARE +-x,+-y,+-z exactly, no sampling needed).
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
    print(f"N={lat.N}  R={R}  C_DST={C_DST:.4f}  MU0={MU0:.4f}", flush=True)

    c0 = int(np.argmin(np.linalg.norm(lat.pos, axis=1)))
    # sq faces: 8,9=+-x  10,11=+-y  12,13=+-z (per _SQ_AXIS_PAIRS convention)
    plus_z = int(nb[c0, 12])
    minus_z = int(nb[c0, 13])
    assert plus_z >= 0 and minus_z >= 0
    print(
        f"c0={c0}  +z neighbour={plus_z} at {lat.pos[plus_z]}  "
        f"-z neighbour={minus_z} at {lat.pos[minus_z]}",
        flush=True,
    )

    r_all = np.linalg.norm(lat.pos, axis=1)
    theta_all = np.arccos(np.clip(lat.pos[:, 2] / np.maximum(r_all, 1e-9), -1, 1))

    K = 2 * np.pi / WAVELENGTH_CELLS
    omega = np.sqrt(
        8 * W_HEX**2 * (1 - np.cos(K * 0.5)) + 2 * W_SQ**2 * (1 - np.cos(K))
    )
    period = 2 * np.pi / omega
    print(f"K={K:.4f}  omega={omega:.4f}  period={period:.3f}", flush=True)

    dt = lat.stable_dt()
    r_fit_min, r_fit_max = 10.0, 0.75 * R
    fit_mask = (r_all >= r_fit_min) & (r_all <= r_fit_max)
    t_arrive = r_fit_max / C_DST
    t_target = t_arrive + period
    n_steps = int(round(t_target / dt))
    print(
        f"dt={dt:.5f}  n_steps={n_steps}  ({t_target/period:.2f} periods)", flush=True
    )

    for step in range(n_steps):
        t = step * dt
        lat.u[plus_z] = AMPLITUDE * np.cos(omega * t)
        lat.u[minus_z] = -AMPLITUDE * np.cos(omega * t)
        lat.step(dt, n_threads=8)
        if step % max(1, n_steps // 10) == 0:
            print(f"  step={step}/{n_steps}", flush=True)

    print("done stepping, computing E/B/S...", flush=True)
    phi_before = lat.phi.copy()
    t = n_steps * dt
    lat.u[plus_z] = AMPLITUDE * np.cos(omega * t)
    lat.u[minus_z] = -AMPLITUDE * np.cos(omega * t)
    lat.step(dt, n_threads=8)
    phi_after = lat.phi.copy()
    dphi = phi_after - phi_before

    E3 = -face_vec_vectorized(dphi) / dt
    B3 = b_vec_vectorized(phi_after, nb) / MU0
    S3 = np.cross(E3, B3)
    Smag = np.linalg.norm(S3, axis=1)

    equatorial = (theta_all > np.radians(70)) & fit_mask
    polar = (theta_all < np.radians(20)) & fit_mask
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
        f"S (p90) equatorial={s_eq:.4e}  polar={s_pol:.4e}  ratio={s_eq/max(s_pol,1e-300):.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
