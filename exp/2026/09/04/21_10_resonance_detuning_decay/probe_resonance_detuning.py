"""
Direct test of doc/dst.md's actual decay mechanism ("Decay rate = distance
from nearest resonance": an off-resonance winding "cannot close perfectly;
a small persistent phi imbalance bleeds out as radiation... tick by tick")
-- NOT step_cross (now.md 2026-09-04: step_cross is the translation/inertia
operator, doc/dst.md never ties it to decay; the resonance mechanism is
specified as a plain face-hopping phenomenon).

Concrete target, per Philip (2026-09-04): "a proton dipole should not
decay, the neutron/spheromak may decay to a point where it becomes a
proton dipole". This probe tests both halves: (1) does detuning the
neutron candidate off its exact resonance produce genuine monotonic
radiation (not just conservative oscillation), scaling with detuning
amount, and (2) does the field configuration that remains actually
converge toward the standalone proton-dipole IC's own shape, not just
lose generic energy.

Resonance = a literal, already-existing eigenmode condition in this
codebase: K_FACTOR=4.493 is j1's first zero, chosen in dipole_mode_ic.py
specifically so U vanishes at r=R_SPH (a genuine closure condition, not a
free choice). Detuning K away from 4.493 breaks that closure -- U no
longer vanishes at the boundary, so the masked-at-r<=1 construction cannot
close perfectly, the literal discrete analogue of "off-resonance" as
described. Local re-implementation of U_and_gradU here (parametrized K)
rather than editing the validated production dipole_mode_ic.py module.
"""

import sys

import numpy as np
from scipy.special import spherical_jn

sys.path.insert(0, ".")
from equilibrium.substrate.lattice import DampedLattice

R = 30.0
R_SPH = 8.0
N_STEPS = 6000
REPORT = 200
N_THREADS = 8
CIRC_R_HOP = 4.0
K0 = 4.493  # j1's first zero -- the on-resonance value
DELTAS = [0.0, 0.02, 0.05, 0.10]


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


def _phi_circulation(lat, centre, r_hop, dr=0.8):
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


def _overlap(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-30 or nb < 1e-30:
        return float("nan")
    return float(np.dot(a.ravel(), b.ravel()) / (na * nb))


def run(delta):
    K = K0 * (1.0 + delta)
    label = f"delta={delta:+.2f} (K={K:.4f})"
    print(f"\n=== {label} ===")

    lat = DampedLattice(radius_cells=R, sponge_fraction=0.15, damp_max=0.003)
    build_poloidal(lat, R_SPH, K)
    add_toroidal(lat, R_SPH, K)
    dt = lat.stable_dt()

    # target proton-dipole profile: on-resonance (K0), poloidal-only -- what
    # the field is claimed to converge toward as the toroidal part decays.
    target = DampedLattice(radius_cells=R, sponge_fraction=0.15, damp_max=0.003)
    build_poloidal(target, R_SPH, K0)
    target_u = target.u.copy()
    target_phi = target.phi.copy()

    u_rms0 = float(np.sqrt(np.mean(lat.u**2)))
    circ0 = _phi_circulation(lat, np.zeros(3), CIRC_R_HOP)
    H0 = float(np.sum(lat.u**2) + 0.5 * np.sum(lat.phi**2))
    print(f"N={lat.N}  dt={dt:.4e}  u_rms0={u_rms0:.4e}  circ0={circ0:+.4e}  H0={H0:.4e}")

    history = []
    for step in range(N_STEPS + 1):
        if step % REPORT == 0:
            u_rms = float(np.sqrt(np.mean(lat.u**2)))
            circ = _phi_circulation(lat, np.zeros(3), CIRC_R_HOP)
            H = float(np.sum(lat.u**2) + 0.5 * np.sum(lat.phi**2))
            rc = circ / circ0 if abs(circ0) > 1e-30 else float("nan")
            ov_u = _overlap(lat.u, target_u)
            ov_phi = _overlap(lat.phi, target_phi)
            history.append((step, u_rms / u_rms0, rc, H / H0, ov_u, ov_phi))
            print(
                f"  {step:6d}  u_ratio={u_rms/u_rms0:.6f}  circ_ratio={rc:.6f}"
                f"  H/H0={H/H0:.6f}  overlap_u={ov_u:.4f}  overlap_phi={ov_phi:.4f}",
                flush=True,
            )
        lat.step(dt, n_threads=N_THREADS)

    return history


def main():
    results = {}
    for delta in DELTAS:
        results[delta] = run(delta)

    print("\n=== summary: circulation retained + shape overlap at final step ===")
    for delta, hist in results.items():
        step, ru, rc, hh, ov_u, ov_phi = hist[-1]
        print(
            f"delta={delta:+.2f}  circ_ratio@{step}={rc:.6f}  H/H0={hh:.6f}"
            f"  overlap_u(->proton)={ov_u:.4f}  overlap_phi(->proton)={ov_phi:.4f}"
        )


if __name__ == "__main__":
    main()
