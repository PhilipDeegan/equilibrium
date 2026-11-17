"""Self-consistent scalar dipole (l=1) standing-wave IC for the AdaptiveLattice
wave update (`_step_pass1`/`_step_pass2`: du/dt=-sigma*div(phi),
dphi/dt=-sigma*grad(u), giving d^2u/dt^2 = sigma^2 * Laplacian(u) -- a real
scalar wave equation, sigma=1 in vacuum/lam=0).

Replaces the old Chandrasekhar-Kendall/Taylor-state B-field IC (u=|B|,
phi=B.n_hat), which seeded u and phi from two DIFFERENT equations (u from a
force-free curl-B system, discarding B's sign via the magnitude; phi from
B's actual vector components) -- an O(1), resolution-independent
inconsistency, most visible on the z-axis where u started at an artificial
exact zero (from |B|) while phi's real, unrelated z-asymmetry was free to
leak in unopposed.

Here both fields come from ONE real solution of the wave equation's own
governing PDE: U(r,z) = j1(K*r)/r * z, a genuine dipole (l=1) Helmholtz
eigenmode (same K=4.493/R_SPH convention as before -- 4.493 is j1's first
zero, so U already vanishes at r=R_SPH). For the general standing-wave pair
u(x,t)=a*U(x)*cos(wt)+..., phi(x,t) is forced (not a free choice) to be
proportional to grad(U) at t=0 for ANY amplitude ratio a:b -- so u=a*U(0),
phi=b*gradU(0) is an exact solution of the continuum PDE for any a,b.
Amplitude normalization stays independent (each field to its own peak-1),
since that freedom was never what caused the inconsistency.

Promoted 2026-08-13 from per-experiment exp/ copies (e.g.
exp/2026/08/11/21_27_everything_active/dipole_mode_ic.py) to a real module --
it's the current, validated production IC (see project_dipole_mode_ic
memory), not a one-off experiment script, and needed a stable import path
for tst/ to depend on directly.
"""

import numpy as np
from scipy.special import spherical_jn

K_FACTOR = 4.493  # first zero of j1 -- U vanishes at r=R_SPH by construction


def U_and_gradU(pos, scale, centre=np.zeros(3)):
    """pos: (N,3) array, absolute units. scale: R_SPH in absolute units.
    Returns (U, gradU) with gradU shape (N,3), both zero outside r<=1
    (normalized radius) is NOT applied here -- caller masks separately.
    """
    p = (pos - centre) / scale
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    r_safe = np.where(r > 0, r, 1.0)
    K = K_FACTOR  # R_SPH=1 in normalized coords p

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

    gradU = (
        np.stack([dUdx, dUdy, dUdz], axis=1) / scale
    )  # chain rule: d/dpos = (1/scale) d/dp
    return U, gradU


def seed_dipole_ic(
    cl,
    pos,
    cell_size,
    scale,
    face_hop,
    face_norm,
    centre=np.zeros(3),
    r_sph=1.0,
    u_max=1.0,
    amplitude=1.0,
    batch=500_000,
):
    """Seeds leaf u from U (peak-normalized) and leaf phi from grad(U) at each
    face midpoint (peak-normalized independently), both masked to zero
    outside r<=r_sph (normalized radius) -- same convention as the old B-field
    IC. Returns nothing; mutates cl in place via set_leaf_u/set_leaf_phi.
    """
    N = pos.shape[0]
    r_cell = np.linalg.norm((pos - centre) / scale, axis=1)
    inside = r_cell <= r_sph

    U_vals, _ = U_and_gradU(pos, scale, centre)
    u_max_val = float(np.abs(U_vals[inside]).max()) if inside.any() else 1.0
    u_max_val = u_max_val or 1.0
    u_all = np.where(inside, u_max * U_vals / u_max_val, 0.0)
    cl.set_leaf_u(u_all)

    phi_max = 0.0
    for start in range(0, N, batch):
        end = min(start + batch, N)
        hop_off = face_hop[None, :, :] * cell_size[start:end, None, None]
        midpoints = pos[start:end, None, :] + 0.5 * hop_off
        _, gU_mid = U_and_gradU(midpoints.reshape(-1, 3), scale, centre)
        gU_mid = gU_mid.reshape(end - start, 14, 3)
        phi_batch = np.einsum("jkd,kd->kj", gU_mid, face_norm)
        phi_max = max(phi_max, float(np.abs(phi_batch).max()))
    phi_max = phi_max or 1.0

    phi_full = np.zeros((14, N), dtype=np.float64)
    for start in range(0, N, batch):
        end = min(start + batch, N)
        hop_off = face_hop[None, :, :] * cell_size[start:end, None, None]
        midpoints = pos[start:end, None, :] + 0.5 * hop_off
        _, gU_mid = U_and_gradU(midpoints.reshape(-1, 3), scale, centre)
        gU_mid = gU_mid.reshape(end - start, 14, 3)
        phi_batch = np.einsum("jkd,kd->kj", gU_mid, face_norm)
        phi_batch *= amplitude / phi_max
        nbr_r = np.linalg.norm(
            (pos[start:end, None, :] + hop_off - centre) / scale, axis=2
        )
        phi_batch[(nbr_r > r_sph).T] = 0.0
        phi_batch[:, r_cell[start:end] > r_sph] = 0.0
        phi_full[:, start:end] = phi_batch

    cl.set_leaf_phi(np.ascontiguousarray(phi_full))
    return u_all, phi_full
