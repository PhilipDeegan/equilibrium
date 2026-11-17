"""
Philip (2026-09-04): "R30 is tiny, we were doing R180 before with 150/160
spheromaks" -- the validated production scale for this curl-full
poloidal+toroidal IC is domain R=180, SCALE=110 (cells_across=220,
exp/2026/08/17/12_25_curl_full_dipole_ic_rebuild/build_curlfull_dipole_uniform_r180.py),
vs today's inherited R=30/R_SPH=8 (cells_across=16) -- BELOW the smallest
point (22) in project_curl_full_dipole_spheromak's own resolution sweep,
which already showed strong resolution-dependence (swing 0.718 at 22 down
to ~0.09-0.10 only at cells_across>=218).

This is a cheap intermediate check (cells_across=110, ~3.97M leaves --
13x today's leaf count, matches a table point already in that memory) to
(1) time a real run at closer-to-production scale before committing to the
full 220 (~31.7M leaves, ~27-29GB RAM) run, and (2) get a fresh apples-to-
apples point on both the swing AND the circulation-freeze-under-detuning
question this session has been chasing, at a size not obviously too small.
"""

import sys
import time

import numpy as np
from scipy.special import spherical_jn

sys.path.insert(0, ".")
from equilibrium.substrate.lattice import DampedLattice

R = 180.0
R_SPH = 110.0  # cells_across=220, the validated production scale (Philip, 2026-09-04)
N_STEPS = 6000
REPORT = 200
N_THREADS = 8
CIRC_R_HOP = R_SPH * 0.5
K0 = 4.493
DELTAS = [0.0, 0.10]


def U_and_gradU_K(pos, scale, K, centre=np.zeros(3)):
    p = (pos - centre) / scale
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    r_safe = np.where(r > 0, r, 1.0)
    j1 = spherical_jn(1, K * r_safe)
    j1p = spherical_jn(1, K * r_safe, derivative=True)
    g = j1 / r_safe
    gp = K * j1p / r_safe - j1 / r_safe**2
    U = z * g
    dUdx = z * gp * x / r_safe
    dUdy = z * gp * y / r_safe
    dUdz = g + z * gp * z / r_safe
    zero = r <= 0
    U = np.where(zero, 0.0, U)
    dUdx = np.where(zero, 0.0, dUdx)
    dUdy = np.where(zero, 0.0, dUdy)
    dUdz = np.where(zero, 0.0, dUdz)
    gradU = np.stack([dUdx, dUdy, dUdz], axis=1) / scale
    return U, gradU


def _phihat(pos, centre):
    p = pos - centre
    rho = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2)
    on_axis = rho <= 0
    rho_safe = np.where(on_axis, 1.0, rho)
    tang = np.zeros_like(p)
    tang[:, 0] = -p[:, 1] / rho_safe
    tang[:, 1] = p[:, 0] / rho_safe
    tang[on_axis] = 0.0
    return tang


def _toroidal_envelope(pos, scale, K, centre):
    p = (pos - centre) / scale
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    rho = np.sqrt(x**2 + y**2)
    r = np.sqrt(x**2 + y**2 + z**2)
    r_safe = np.where(r > 0, r, 1.0)
    sin_theta = np.where(r > 0, rho / r_safe, 0.0)
    return spherical_jn(1, K * r_safe) * sin_theta


def build_poloidal(lat, r_sph, K, centre=None, u_max=1.0, amplitude=1.0):
    if centre is None:
        centre = np.zeros(3)
    pos, nb = lat.pos, lat.nb
    r_cell = np.linalg.norm(pos - centre, axis=1) / r_sph
    inside = r_cell <= 1.0
    U_vals, _ = U_and_gradU_K(pos, r_sph, K, centre)
    u_max_val = float(np.abs(U_vals[inside]).max()) if inside.any() else 1.0
    lat.u = np.where(inside, u_max * U_vals / (u_max_val or 1.0), 0.0)
    valid = nb >= 0
    safe_nb = np.where(valid, nb, 0)
    mid = 0.5 * (pos[:, None, :] + pos[safe_nb])
    _, gU_mid = U_and_gradU_K(mid.reshape(-1, 3), r_sph, K, centre)
    gU_mid = gU_mid.reshape(pos.shape[0], 14, 3)
    diff = pos[safe_nb] - pos[:, None, :]
    dlen = np.linalg.norm(diff, axis=-1, keepdims=True)
    n_hat = diff / np.where(dlen > 0, dlen, 1.0)
    phi = np.einsum("nkd,nkd->nk", gU_mid, n_hat)
    phi_max = float(np.abs(phi).max()) or 1.0
    phi = phi * (amplitude / phi_max)
    nbr_r = np.linalg.norm(pos[safe_nb] - centre, axis=-1) / r_sph
    mask = valid & (nbr_r <= 1.0) & (r_cell[:, None] <= 1.0)
    lat.phi = np.where(mask, phi, 0.0)


def add_toroidal(lat, r_sph, K, centre=None, amplitude=1.0):
    if centre is None:
        centre = np.zeros(3)
    pos, nb = lat.pos, lat.nb
    valid = nb >= 0
    safe_nb = np.where(valid, nb, 0)
    mid = 0.5 * (pos[:, None, :] + pos[safe_nb])
    env_mid = _toroidal_envelope(mid.reshape(-1, 3), r_sph, K, centre).reshape(pos.shape[0], 14)
    tang_mid = _phihat(mid.reshape(-1, 3), centre).reshape(pos.shape[0], 14, 3)
    diff = pos[safe_nb] - pos[:, None, :]
    dlen = np.linalg.norm(diff, axis=-1, keepdims=True)
    n_hat = diff / np.where(dlen > 0, dlen, 1.0)
    proj = np.einsum("nkd,nkd->nk", tang_mid, n_hat)
    phi_tor = env_mid * proj
    tor_max = float(np.abs(phi_tor).max()) or 1.0
    phi_tor = phi_tor * (amplitude / tor_max)
    r_cell = np.linalg.norm(pos - centre, axis=1) / r_sph
    nbr_r = np.linalg.norm(pos[safe_nb] - centre, axis=-1) / r_sph
    mask = valid & (nbr_r <= 1.0) & (r_cell[:, None] <= 1.0)
    lat.phi = lat.phi + np.where(mask, phi_tor, 0.0)


def _phi_circulation(lat, centre, r_hop, dr):
    c = np.asarray(centre, dtype=float)
    nb_safe = np.where(lat.nb >= 0, lat.nb, 0)
    pos_j = lat.pos[nb_safe]
    pos_i = lat.pos[:, np.newaxis, :]
    mid = 0.5 * (pos_i + pos_j) - c
    r_mid = np.linalg.norm(mid, axis=-1)
    on_shell = (lat.nb >= 0) & (r_mid >= r_hop - dr) & (r_mid <= r_hop + dr)
    upper = on_shell & (np.arange(len(lat.pos))[:, None] < nb_safe)
    mx, my = mid[..., 0], mid[..., 1]
    rho = np.sqrt(mx**2 + my**2)
    rho_s = np.where(rho > 1e-10, rho, 1.0)
    diff = pos_j - pos_i
    d_len = np.linalg.norm(diff, axis=-1, keepdims=True)
    n_hat = diff / np.where(d_len > 0, d_len, 1.0)
    proj = n_hat[..., 0] * (-my / rho_s) + n_hat[..., 1] * (mx / rho_s)
    return float(np.sum(lat.phi[upper] * proj[upper]))


def run(delta):
    K = K0 * (1.0 + delta)
    t_build0 = time.time()
    lat = DampedLattice(radius_cells=R, sponge_fraction=0.15, damp_max=0.003)
    build_poloidal(lat, R_SPH, K)
    add_toroidal(lat, R_SPH, K)
    dt = lat.stable_dt()
    t_build = time.time() - t_build0

    u_rms0 = float(np.sqrt(np.mean(lat.u**2)))
    circ0 = _phi_circulation(lat, np.zeros(3), CIRC_R_HOP, R_SPH * 0.1)
    H0 = float(np.sum(lat.u**2) + 0.5 * np.sum(lat.phi**2))
    print(
        f"\n=== delta={delta:+.2f} (K={K:.4f}) ===\n"
        f"N={lat.N}  build_time={t_build:.1f}s  dt={dt:.4e}"
        f"  u_rms0={u_rms0:.4e}  circ0={circ0:+.4e}  H0={H0:.4e}",
        flush=True,
    )

    t0 = time.time()
    for step in range(N_STEPS + 1):
        if step % REPORT == 0:
            elapsed = time.time() - t0
            u_rms = float(np.sqrt(np.mean(lat.u**2)))
            circ = _phi_circulation(lat, np.zeros(3), CIRC_R_HOP, R_SPH * 0.1)
            H = float(np.sum(lat.u**2) + 0.5 * np.sum(lat.phi**2))
            rc = circ / circ0 if abs(circ0) > 1e-30 else float("nan")
            swing = u_rms / u_rms0 - rc
            rate = step / elapsed if elapsed > 0 else float("nan")
            print(
                f"  {step:6d}  u_ratio={u_rms/u_rms0:.6f}  circ_ratio={rc:.6f}"
                f"  swing={swing:+.6f}  H/H0={H/H0:.6f}  ({rate:.2f} steps/s)",
                flush=True,
            )
        lat.step(dt, n_threads=N_THREADS)


def main():
    for delta in DELTAS:
        run(delta)


if __name__ == "__main__":
    main()
