"""
Confirmatory test for the already-proven ∂B/∂t≡0 identity (now, 2026-09-05):
hand-build a photon (the validated companion-phi traveling wave,
exp/2026/08/19/18_30_traveling_wave_reproducer/) with a REAL static B baked
in on top of it (a circular-current-loop phi, same analytic construction as
tst/substrate/test_curl_loop.py), then step under plain step() only and
watch the wave packet sweep past the loop's location while curl(phi)=B at
the loop stays frozen to floating-point precision throughout -- not just
algebraically true, but directly demonstrated with a real propagating
disturbance moving through/past a real curl-full structure.

Superposition is exact here (step() is linear in (u,phi) at constant sigma,
no lam): B(phi_wave + phi_loop) = B(phi_wave) + B(phi_loop) identically,
since b_vec is linear in phi, and phi_wave is already known curl-free
(companion-phi is a per-face copy of a scalar gradient) -- so this result is
GUARANTEED by facts already established, not a genuine unknown. The point is
to make the frozen-B mechanism concrete and visible with a real propagating
wave, not to test whether it holds (it must).
"""

import sys

sys.path.insert(0, ".")
import numpy as np
from scipy.spatial import cKDTree

from equilibrium.substrate.curl import b_vec_vectorized
from equilibrium.substrate.lattice import DampedLattice, W_HEX, W_SQ

R = 12.0
HALF_LEN = 90.0
AXIS = 2
N_CELLS_PER_WAVELENGTH = 8
ENVELOPE_WAVELENGTHS = 3.0
Z0_WAVE = -45.0
N_PERIODS = 12
LOOP_A = 4.0
Z_LOOP = 0.0


# --- wave construction, verbatim from probe_traveling_wave_shift.py ---
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


# --- loop construction, adapted from tst/substrate/test_curl_loop.py,
# shifted along AXIS to Z_LOOP instead of fixed at the origin ---
def _k2(rho, z, a):
    return 4.0 * a * rho / ((a + rho) ** 2 + z**2)


def _a_phi(rho, z, a):
    from scipy.special import ellipe, ellipk

    rho = np.asarray(rho, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    safe_rho = np.where(rho > 1e-9, rho, 1.0)
    k2 = np.clip(_k2(safe_rho, z, a), 0.0, 1.0 - 1e-12)
    K, E = ellipk(k2), ellipe(k2)
    val = (
        (1.0 / np.pi)
        * np.sqrt(a / safe_rho)
        / np.sqrt(k2)
        * ((1.0 - k2 / 2.0) * K - E)
    )
    return np.where(rho > 1e-6, val, 0.0)


def _b_z_axis(z, a):
    return a**2 / (2.0 * (a**2 + z**2) ** 1.5)


def build_loop_phi(lat, a, z_loop):
    pos, nb = lat.pos, lat.nb
    valid = nb >= 0
    safe_nb = np.where(valid, nb, 0)
    mid = 0.5 * (pos[:, None, :] + pos[safe_nb])
    rho_mid = np.linalg.norm(np.delete(mid, AXIS, axis=2), axis=2)
    z_mid = mid[..., AXIS] - z_loop
    a_phi_mid = _a_phi(rho_mid, z_mid, a)
    safe_rho_mid = np.where(rho_mid > 1e-9, rho_mid, 1.0)
    perp = [i for i in range(3) if i != AXIS]
    phihat = np.zeros_like(mid)
    phihat[..., perp[0]] = -mid[..., perp[1]] / safe_rho_mid
    phihat[..., perp[1]] = mid[..., perp[0]] / safe_rho_mid
    diff = pos[safe_nb] - pos[:, None, :]
    a_dot_dl = a_phi_mid * np.sum(phihat * diff, axis=2)
    return np.where(valid, a_dot_dl, 0.0)


def main():
    lat = DampedLattice.cylinder(
        R, HALF_LEN, axis=AXIS, sponge_fraction=0.15, damp_max=0.003
    )
    print(f"N={lat.N}  R={R}  HALF_LEN={HALF_LEN}", flush=True)

    offsets, w = build_faces(lat)
    a_axial = float(abs(offsets[13, AXIS]))
    wavelength = N_CELLS_PER_WAVELENGTH * a_axial
    K = 2 * np.pi / wavelength
    envelope_sigma = ENVELOPE_WAVELENGTHS * wavelength

    theta = offsets[:, AXIS] * K
    omega = np.sqrt(float(np.sum(w**2 * (1 - np.cos(theta)))))
    T = 2 * np.pi / omega
    dt = lat.stable_dt()
    n_steps_per_period = int(round(T / dt))
    n_steps_total = N_PERIODS * n_steps_per_period
    z_final_envelope = Z0_WAVE + N_PERIODS * wavelength

    print(
        f"wavelength={wavelength:.4f}  envelope_sigma={envelope_sigma:.4f}  "
        f"T={T:.4f}  dt={dt:.5f}  n_steps/period={n_steps_per_period}  "
        f"n_periods={N_PERIODS}  n_steps_total={n_steps_total}",
        flush=True,
    )
    print(
        f"wave envelope center: {Z0_WAVE} -> {z_final_envelope:.2f}  "
        f"(loop at z={Z_LOOP}, sponge core half-extent ~{0.85*HALF_LEN:.1f})",
        flush=True,
    )

    # nearest cell to the loop's own center (on-axis, rho=0)
    tree = cKDTree(lat.pos)
    loop_pt = np.zeros(3)
    loop_pt[AXIS] = Z_LOOP
    _, i_loop = tree.query(loop_pt)
    i_loop = int(i_loop)
    print(
        f"loop-center cell: idx={i_loop}  pos={lat.pos[i_loop]}",
        flush=True,
    )

    phi_wave0 = companion_phi(
        lat, offsets, w, AXIS, K, 1.0, envelope_sigma, Z0_WAVE, omega
    )
    phi_loop = build_loop_phi(lat, LOOP_A, Z_LOOP)

    # sanity check 1: the loop alone reproduces the classical on-axis Bz(z)
    # profile in THIS cylinder geometry too (not just the sphere domain
    # test_curl_loop.py used) -- confirms the superposed structure really is
    # a genuine curl-full field, not a construction bug.
    lat.phi = phi_loop.copy()
    B_loop_only = b_vec_vectorized(lat.pos, lat.nb, lat.phi)
    axis_mask = (
        np.isfinite(B_loop_only[:, 0])
        & (np.linalg.norm(np.delete(lat.pos, AXIS, axis=1), axis=1) < 1.0)
        & (np.abs(lat.pos[:, AXIS]) < 40.0)
    )
    zs = np.round(lat.pos[axis_mask, AXIS])
    ratios = []
    _HOP_SQ_FACTOR = (2.0 / np.sqrt(3.0)) ** 2
    for zv in np.unique(zs):
        m = axis_mask & (np.abs(lat.pos[:, AXIS] - zv) < 0.5)
        target = _b_z_axis(zv - Z_LOOP, LOOP_A)
        if abs(target) < 1e-9 or m.sum() < 1:
            continue
        ratios.append(float(np.mean(B_loop_only[m, AXIS])) / target)
    ratios = np.array(ratios)
    print(
        f"[sanity] loop-alone on-axis Bz ratio to classical target: "
        f"mean={ratios.mean():.4f} std={ratios.std():.4f} "
        f"(expect ~{_HOP_SQ_FACTOR:.4f}, ~0 std)",
        flush=True,
    )

    # sanity check 2: the wave alone is curl-free everywhere (matches the
    # already-established E_full≡0 / companion-phi-is-a-scalar-gradient
    # result, now checked directly via b_vec on this exact traveling-wave IC)
    lat.phi = phi_wave0.copy()
    B_wave_only = b_vec_vectorized(lat.pos, lat.nb, lat.phi)
    finite = np.isfinite(B_wave_only).all(axis=1)
    b_wave_rms = float(np.sqrt(np.nanmean(B_wave_only[finite] ** 2)))
    print(
        f"[sanity] companion-phi wave ALONE: B rms over domain = {b_wave_rms:.3e} "
        f"(expect ~0, floating-point noise)",
        flush=True,
    )

    # --- the actual run: superpose wave + loop, step under plain step() only ---
    lat.u = u_field(lat, AXIS, K, 1.0, envelope_sigma, Z0_WAVE)
    lat.phi = phi_wave0 + phi_loop

    B0 = b_vec_vectorized(lat.pos, lat.nb, lat.phi)
    B_loop_initial = B0[i_loop].copy()
    print(f"B at loop center, t=0: {B_loop_initial}", flush=True)

    max_dev = 0.0
    u_trace = []
    energy_trace = []
    b_trace = []
    t_trace = []
    sample_every = max(1, n_steps_per_period // 20)

    for step in range(n_steps_total + 1):
        if step % sample_every == 0 or step == n_steps_total:
            B_now = b_vec_vectorized(lat.pos, lat.nb, lat.phi)[i_loop]
            dev = float(np.max(np.abs(B_now - B_loop_initial)))
            max_dev = max(max_dev, dev)
            u_trace.append(float(lat.u[i_loop]))
            energy_trace.append(
                float(lat.u[i_loop] ** 2 + 0.5 * np.sum(lat.phi[i_loop] ** 2))
            )
            b_trace.append(B_now.copy())
            t_trace.append(step * dt)
        if step < n_steps_total:
            lat.step(dt)

    u_trace = np.array(u_trace)
    energy_trace = np.array(energy_trace)
    b_trace = np.array(b_trace)
    t_trace = np.array(t_trace)

    print(f"\nmax |B_loop(t) - B_loop(0)| over the whole run: {max_dev:.3e}", flush=True)
    print(f"B_loop(0)  = {B_loop_initial}", flush=True)
    print(f"B_loop(end)= {b_trace[-1]}", flush=True)
    print(
        f"u at loop location: min={u_trace.min():.5f} max={u_trace.max():.5f} "
        f"(t=0 value {u_trace[0]:.5f}) -- rises as the wave sweeps past, "
        f"confirming the disturbance really did reach the loop",
        flush=True,
    )
    print(
        f"local energy density at loop: min={energy_trace.min():.5e} "
        f"max={energy_trace.max():.5e}",
        flush=True,
    )
    peak_idx = int(np.argmax(np.abs(u_trace)))
    print(
        f"|u| peaks at t={t_trace[peak_idx]:.2f} (sample {peak_idx}/{len(t_trace)-1}), "
        f"B deviation AT THAT SAME INSTANT: "
        f"{float(np.max(np.abs(b_trace[peak_idx]-B_loop_initial))):.3e}",
        flush=True,
    )

    out_dir = "exp/2026/09/05/21_10_frozen_b_photon_confirmation"
    np.savez(
        f"{out_dir}/frozen_b_photon_trace.npz",
        t=t_trace,
        u=u_trace,
        energy=energy_trace,
        b=b_trace,
        b_loop_initial=B_loop_initial,
        wavelength=wavelength,
        z_loop=Z_LOOP,
        z0_wave=Z0_WAVE,
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(8, 4.5))
        ax1.plot(t_trace, u_trace, "b-", label="u at loop location")
        ax1.set_xlabel("t")
        ax1.set_ylabel("u", color="b")
        ax1.tick_params(axis="y", labelcolor="b")
        ax2 = ax1.twinx()
        b_dev = np.max(np.abs(b_trace - B_loop_initial[None, :]), axis=1)
        ax2.plot(t_trace, b_dev, "r-", label="|B(t)-B(0)| at loop")
        ax2.set_ylabel("|B deviation|", color="r")
        ax2.tick_params(axis="y", labelcolor="r")
        ax2.set_ylim(-1e-12, max(1e-11, b_dev.max() * 5))
        plt.title("Wave passes the loop; B at the loop stays frozen")
        fig.tight_layout()
        fig.savefig(f"{out_dir}/frozen_b_photon.png", dpi=130)
        print(f"plot saved: {out_dir}/frozen_b_photon.png", flush=True)
    except Exception as e:
        print(f"(plot skipped: {e})", flush=True)


if __name__ == "__main__":
    main()
