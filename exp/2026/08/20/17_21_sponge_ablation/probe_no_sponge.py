"""
Ablation of exp/2026/08/19/18_30_traveling_wave_reproducer/probe_traveling_wave_shift.py:
same domain, IC, and one-period phase-shift measurement, but with the sponge
BC removed entirely (plain SubstrateLattice.cylinder, no damping). The run
only lasts one phase period T, during which the wave pattern can't reach the
wall anyway, so if the sponge was truly inert for this measurement the two
runs should agree to numerical precision in the core AND show the same
(small, IC-truncation) mismatch out at the wall -- not a larger one caused by
sponge damping of the envelope tails.

Also dumps u(z) at t=0 and t=T for both runs (sponge vs no-sponge) along the
axis for plotting, so the sponge's effect on the tail region is visible
directly rather than only through the masked correlation metric.
"""

import sys

sys.path.insert(0, ".")
import numpy as np
from equilibrium.substrate.lattice import DampedLattice, SubstrateLattice, W_HEX, W_SQ

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


def run(lat, label):
    print(f"[{label}] N={lat.N}  R={R}  HALF_LEN={HALF_LEN}", flush=True)
    offsets, w = build_faces(lat)
    a_axial = float(abs(offsets[13, AXIS]))
    wavelength = N_CELLS_PER_WAVELENGTH * a_axial
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
        f"[{label}] wavelength={wavelength}  K={K:.4f}  omega={omega:.4f}  T={T:.4f}  "
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

    z_all = lat.pos[:, AXIS]
    r_perp = np.linalg.norm(np.delete(lat.pos, AXIS, axis=1), axis=1)
    axial_ok = (np.abs(z_all) < 0.5 * HALF_LEN) & (
        np.abs(z_all + wavelength) < 0.5 * HALF_LEN
    )
    core = (r_perp < 0.5 * R) & axial_ok

    from scipy.spatial import cKDTree

    tree = cKDTree(lat.pos)
    shift = np.zeros(3)
    shift[AXIS] = wavelength
    query_pts = lat.pos + shift[None, :]
    dist, idx = tree.query(query_pts)
    valid_shift = (dist < 1e-3) & core

    u0_shifted = u0[idx]
    a = u_T[valid_shift]
    b = u0_shifted[valid_shift]
    diff = a - b
    denom = np.sqrt(np.mean(a**2)) + np.sqrt(np.mean(b**2))
    corr_u = np.corrcoef(a, b)[0, 1]
    rel_u = np.sqrt(np.mean(diff**2)) / denom
    print(
        f"[{label}] N compared cells: {valid_shift.sum()}  u corr: {corr_u:.6f}  "
        f"u rel rms diff: {rel_u:.6f}",
        flush=True,
    )

    phi0_shifted = phi0[idx]
    ap = phi_T[valid_shift]
    bp = phi0_shifted[valid_shift]
    diffp = ap - bp
    denomp = np.sqrt(np.mean(ap**2)) + np.sqrt(np.mean(bp**2))
    corr_phi = np.corrcoef(ap.ravel(), bp.ravel())[0, 1]
    rel_phi = np.sqrt(np.mean(diffp**2)) / denomp
    print(
        f"[{label}] phi corr: {corr_phi:.6f}  phi rel rms diff: {rel_phi:.6f}",
        flush=True,
    )

    # axial profile for plotting: near-axis cells (small r_perp), all z
    near_axis = r_perp < 1.0
    order = np.argsort(z_all[near_axis])
    z_line = z_all[near_axis][order]
    u0_line = u0[near_axis][order]
    uT_line = u_T[near_axis][order]

    return dict(
        label=label,
        wavelength=wavelength,
        envelope_sigma=envelope_sigma,
        corr_u=corr_u,
        rel_u=rel_u,
        corr_phi=corr_phi,
        rel_phi=rel_phi,
        z_line=z_line,
        u0_line=u0_line,
        uT_line=uT_line,
    )


def main():
    lat_sponge = DampedLattice.cylinder(
        R, HALF_LEN, axis=AXIS, sponge_fraction=0.15, damp_max=0.003
    )
    res_sponge = run(lat_sponge, "sponge")

    lat_plain = SubstrateLattice.cylinder(R, HALF_LEN, axis=AXIS)
    res_plain = run(lat_plain, "no_sponge")

    print("\n--- comparison ---")
    print(
        f"corr_u   sponge={res_sponge['corr_u']:.6f}  no_sponge={res_plain['corr_u']:.6f}  "
        f"delta={res_plain['corr_u']-res_sponge['corr_u']:+.2e}"
    )
    print(
        f"corr_phi sponge={res_sponge['corr_phi']:.6f}  no_sponge={res_plain['corr_phi']:.6f}  "
        f"delta={res_plain['corr_phi']-res_sponge['corr_phi']:+.2e}"
    )

    np.savez(
        "exp/2026/08/20/17_21_sponge_ablation/result.npz",
        z_sponge=res_sponge["z_line"],
        u0_sponge=res_sponge["u0_line"],
        uT_sponge=res_sponge["uT_line"],
        z_plain=res_plain["z_line"],
        u0_plain=res_plain["u0_line"],
        uT_plain=res_plain["uT_line"],
        corr_u_sponge=res_sponge["corr_u"],
        corr_u_plain=res_plain["corr_u"],
        corr_phi_sponge=res_sponge["corr_phi"],
        corr_phi_plain=res_plain["corr_phi"],
        half_len=HALF_LEN,
    )
    print("saved exp/2026/08/20/17_21_sponge_ablation/result.npz")


if __name__ == "__main__":
    main()
