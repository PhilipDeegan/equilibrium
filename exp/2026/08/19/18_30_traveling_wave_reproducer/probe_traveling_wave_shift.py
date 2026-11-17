"""
Sharper test than centroid drift: does the field at t=T match the field at
t=0 shifted by exactly one wavelength along the propagation axis? T is
derived directly from the same dispersion relation used to build the
companion-phi traveling wave (omega^2 = sigma^2 * sum w_k^2*(1-cos(K.offset_k))),
T = 2*pi/omega. Uses DampedLattice.cylinder (sponge active) to remove the
boundary-reflection confound from the earlier centroid-drift check.
"""

import sys

sys.path.insert(0, ".")
import numpy as np
from equilibrium.substrate.lattice import DampedLattice, W_HEX, W_SQ

R = 12.0
HALF_LEN = 60.0
N_CELLS_PER_WAVELENGTH = 8
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


def main():
    lat = DampedLattice.cylinder(
        R, HALF_LEN, axis=AXIS, sponge_fraction=0.15, damp_max=0.003
    )
    print(f"N={lat.N}  R={R}  HALF_LEN={HALF_LEN}", flush=True)
    offsets, w = build_faces(lat)
    a_axial = float(
        abs(offsets[13, AXIS])
    )  # true +z sq-face hop distance = 2/sqrt(3), NOT 1.0
    wavelength = (
        N_CELLS_PER_WAVELENGTH * a_axial
    )  # exact integer multiple of the real lattice period
    K = 2 * np.pi / wavelength
    envelope_sigma = ENVELOPE_WAVELENGTHS * wavelength
    z0 = 0.0

    theta = offsets[:, AXIS] * K
    omega = np.sqrt(float(np.sum(w**2 * (1 - np.cos(theta)))))
    T = 2 * np.pi / omega
    v_phase = wavelength / T

    dt = lat.stable_dt()
    n_steps = int(round(T / dt))
    T_actual = n_steps * dt

    print(
        f"wavelength={wavelength}  K={K:.4f}  omega={omega:.4f}  T={T:.4f}  "
        f"v_phase={v_phase:.4f}  dt={dt:.5f}  n_steps={n_steps}  T_actual={T_actual:.4f}",
        flush=True,
    )

    lat.u = u_field(lat, AXIS, K, 1.0, envelope_sigma, z0)
    lat.phi = companion_phi(lat, offsets, w, AXIS, K, 1.0, envelope_sigma, z0, omega)
    u0 = lat.u.copy()
    phi0 = lat.phi.copy()

    for step in range(n_steps):
        lat.step(dt)

    u_T = lat.u.copy()
    phi_T = lat.phi.copy()

    # compare u(z, T) to u(z - wavelength, 0) -- nearest-cell lookup along axis,
    # restricted to a region well inside the sponge (envelope center + a few
    # sigma, away from both the origin transient and the wall)
    z_all = lat.pos[:, AXIS]
    r_perp = np.linalg.norm(np.delete(lat.pos, AXIS, axis=1), axis=1)
    axial_ok = (np.abs(z_all) < 0.5 * HALF_LEN) & (
        np.abs(z_all + wavelength) < 0.5 * HALF_LEN
    )
    core = (r_perp < 0.5 * R) & axial_ok  # away from BOTH lateral wall and end caps

    # build a lookup from (rounded z, rounded transverse pos) -> u0, shifted by wavelength
    # simplest robust approach: for each cell, find u0 at the SAME (y or x, whichever
    # is not the axis) but z shifted by +wavelength, via nearest neighbour in position
    from scipy.spatial import cKDTree

    tree = cKDTree(lat.pos)
    shift = np.zeros(3)
    shift[AXIS] = wavelength
    query_pts = lat.pos + shift[None, :]
    dist, idx = tree.query(query_pts)
    valid_shift = (
        dist < 1e-3
    ) & core  # now an exact lattice translation -- should be a near-exact match

    u0_shifted = u0[idx]

    a = u_T[valid_shift]
    b = u0_shifted[valid_shift]
    print(f"N compared cells: {valid_shift.sum()}", flush=True)
    print("--- u ---")
    print(
        f"u_T rms: {np.sqrt(np.mean(a**2)):.5f}  u0_shifted rms: {np.sqrt(np.mean(b**2)):.5f}",
        flush=True,
    )
    diff = a - b
    print(f"diff rms: {np.sqrt(np.mean(diff**2)):.5f}", flush=True)
    denom = np.sqrt(np.mean(a**2)) + np.sqrt(np.mean(b**2))
    print(f"relative rms diff: {np.sqrt(np.mean(diff**2))/denom:.5f}", flush=True)
    print(f"correlation: {np.corrcoef(a,b)[0,1]:.5f}", flush=True)

    print(
        "--- phi (all 14 faces, same k -- z-shift by an integer number of\n"
        "    wavelengths preserves face-direction correspondence exactly) ---"
    )
    phi0_shifted = phi0[idx]
    ap = phi_T[valid_shift]
    bp = phi0_shifted[valid_shift]
    print(
        f"phi_T rms: {np.sqrt(np.mean(ap**2)):.5f}  phi0_shifted rms: {np.sqrt(np.mean(bp**2)):.5f}",
        flush=True,
    )
    diffp = ap - bp
    print(f"diff rms: {np.sqrt(np.mean(diffp**2)):.5f}", flush=True)
    denomp = np.sqrt(np.mean(ap**2)) + np.sqrt(np.mean(bp**2))
    print(f"relative rms diff: {np.sqrt(np.mean(diffp**2))/denomp:.5f}", flush=True)
    print(f"correlation: {np.corrcoef(ap.ravel(),bp.ravel())[0,1]:.5f}", flush=True)

    print("--- combined field energy (u^2 + 0.5*sum(phi^2)) ---")
    e_T = a**2 + 0.5 * np.sum(ap**2, axis=1)
    e_0shift = b**2 + 0.5 * np.sum(bp**2, axis=1)
    diffe = e_T - e_0shift
    denome = np.sqrt(np.mean(e_T**2)) + np.sqrt(np.mean(e_0shift**2))
    print(
        f"e_T rms: {np.sqrt(np.mean(e_T**2)):.5f}  e_0shift rms: {np.sqrt(np.mean(e_0shift**2)):.5f}",
        flush=True,
    )
    print(f"relative rms diff: {np.sqrt(np.mean(diffe**2))/denome:.5f}", flush=True)
    print(f"correlation: {np.corrcoef(e_T,e_0shift)[0,1]:.5f}", flush=True)


if __name__ == "__main__":
    main()
