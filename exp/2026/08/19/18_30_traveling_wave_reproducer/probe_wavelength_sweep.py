"""
Wavelength sweep: does the traveling-wave shift-match hold across a range
of wavelengths, or only near N=8 cells/wavelength (doc/dst_next.md's
recommended safety margin against BCC dispersion anisotropy, minimum 3.6)?
Domain scaled proportionally to wavelength (HALF_LEN=2.5*wavelength,
R=1.5*wavelength, floor R at 6) so each case gets a comparable relative
test, not conflated with a separate "domain too small" confound.
Wavelength is an EXACT integer multiple of the true axial lattice period
(2/sqrt(3), confirmed from lat.pos directly, NOT assumed =1).
"""

import sys

sys.path.insert(0, ".")
import numpy as np
from equilibrium.substrate.lattice import DampedLattice, W_HEX, W_SQ

ENVELOPE_WAVELENGTHS = 4.0
AXIS = 2


def build_faces(lat):
    c0 = int(np.argmin(np.linalg.norm(lat.pos, axis=1)))
    assert lat._valid[c0].all()
    offsets = lat.pos[lat.nb[c0]] - lat.pos[c0]
    w = np.array([W_HEX] * 8 + [W_SQ] * 6, dtype=np.float64)
    return offsets, w


def companion_phi(lat, offsets, w, axis, K, amplitude, envelope_sigma, z0, omega):
    theta = offsets[:, axis] * K
    sigma0 = 1.0
    phi = np.zeros_like(lat.phi)
    for k in range(14):
        Phi_k_amp = (2 * sigma0 * w[k] / omega) * np.sin(0.5 * theta[k])
        mid_axis_pos = lat.pos[:, axis] + 0.5 * offsets[k, axis]
        mid_env = np.exp(-0.5 * ((mid_axis_pos - z0) / envelope_sigma) ** 2)
        phi[:, k] = amplitude * Phi_k_amp * mid_env * np.cos(mid_axis_pos * K)
    return phi


def u_field(lat, axis, K, amplitude, envelope_sigma, z0):
    z = lat.pos[:, axis] - z0
    env = np.exp(-0.5 * (z / envelope_sigma) ** 2)
    return amplitude * env * np.cos(lat.pos[:, axis] * K)


def run(n_cells_per_wavelength):
    a_axial_guess = 2.0 / np.sqrt(3.0)
    HALF_LEN = 2.5 * n_cells_per_wavelength * a_axial_guess
    R = max(6.0, 1.5 * n_cells_per_wavelength * a_axial_guess)

    lat = DampedLattice.cylinder(
        R, HALF_LEN, axis=AXIS, sponge_fraction=0.15, damp_max=0.003
    )
    offsets, w = build_faces(lat)
    a_axial = float(abs(offsets[13, AXIS]))
    wavelength = n_cells_per_wavelength * a_axial
    K = 2 * np.pi / wavelength
    envelope_sigma = ENVELOPE_WAVELENGTHS * wavelength
    z0 = 0.0

    theta = offsets[:, AXIS] * K
    omega = np.sqrt(float(np.sum(w**2 * (1 - np.cos(theta)))))
    T = 2 * np.pi / omega
    v_phase = wavelength / T

    dt = lat.stable_dt()
    n_steps = int(round(T / dt))

    lat.u = u_field(lat, AXIS, K, 1.0, envelope_sigma, z0)
    lat.phi = companion_phi(lat, offsets, w, AXIS, K, 1.0, envelope_sigma, z0, omega)
    u0 = lat.u.copy()
    phi0 = lat.phi.copy()

    for step in range(n_steps):
        lat.step(dt)

    u_T = lat.u.copy()
    phi_T = lat.phi.copy()

    r_perp = np.linalg.norm(np.delete(lat.pos, AXIS, axis=1), axis=1)
    z_axis = lat.pos[:, AXIS]
    axial_ok = (np.abs(z_axis) < 0.5 * HALF_LEN) & (
        np.abs(z_axis + wavelength) < 0.5 * HALF_LEN
    )
    core = (r_perp < 0.5 * R) & axial_ok

    from scipy.spatial import cKDTree

    tree = cKDTree(lat.pos)
    shift = np.zeros(3)
    shift[AXIS] = wavelength
    dist, idx = tree.query(lat.pos + shift[None, :])
    valid_shift = (dist < 1e-3) & core

    a = u_T[valid_shift]
    b = u0[idx][valid_shift]
    ap = phi_T[valid_shift]
    bp = phi0[idx][valid_shift]

    n_cmp = int(valid_shift.sum())
    if n_cmp < 10:
        print(
            f"N_cells/wavelength={n_cells_per_wavelength:5.1f}  wavelength={wavelength:.3f}  "
            f"N={lat.N}  n_steps={n_steps}  N_compared={n_cmp}  INSUFFICIENT DATA",
            flush=True,
        )
        return

    corr_u = np.corrcoef(a, b)[0, 1]
    corr_phi = np.corrcoef(ap.ravel(), bp.ravel())[0, 1]
    rel_u = np.sqrt(np.mean((a - b) ** 2)) / (
        np.sqrt(np.mean(a**2)) + np.sqrt(np.mean(b**2))
    )
    rel_phi = np.sqrt(np.mean((ap - bp) ** 2)) / (
        np.sqrt(np.mean(ap**2)) + np.sqrt(np.mean(bp**2))
    )

    print(
        f"N_cells/wavelength={n_cells_per_wavelength:5.1f}  wavelength={wavelength:.3f}  "
        f"N={lat.N:7d}  n_steps={n_steps:5d}  N_compared={n_cmp:5d}  "
        f"corr_u={corr_u:.4f}  corr_phi={corr_phi:.4f}  rel_u={rel_u:.4f}  rel_phi={rel_phi:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    for n in [2, 3, 3.6, 4, 5, 6, 8, 10, 14, 20]:
        run(n)
