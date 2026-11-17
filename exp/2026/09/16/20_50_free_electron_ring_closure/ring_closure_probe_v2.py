"""
v2: same free-electron ring-closure question as ring_closure_probe.py, but
with a genuine companion-phi (WKB generalization of equilibrium.substrate.
traveling_wave.companion_phi to a curved/circular path) instead of phi=0.

v1 finding: phi=0 is a "shocked state, not an exact fixed point" IC (same
lesson as project_dst_proton_translation) -- every config lost 45-99% of
its ring-region field energy within a few periods. v1 DID find a clean,
r-independent omega(n) signature though (ratio omega_meas/omega_pred was
~1.50 for n=2 and ~1.33 for n=6 across every tested r), suggesting the
topological/frequency part is real even though the shocked IC disperses.

This version builds a real companion phi: at each face midpoint, the LOCAL
tangential wavevector K_vec(pos) = (n/rho(pos)) * theta_hat(pos) (not a
fixed global axis) gives a per-face phase increment theta_k(pos) =
K_vec(pos) . offsets[k], and Phi_k_amp = (2*w_k/omega)*sin(0.5*theta_k(pos))
-- the same formula traveling_wave.companion_phi uses for a straight axis,
generalized to a curved path (a WKB/slowly-varying-wavevector
approximation, valid when n/r isn't too large relative to 1/hop-distance).

Measures, in addition to v1's frequency/energy-decay check: the field-
energy-weighted RADIAL SPREAD of the ring over time -- does the envelope
hold its width (a genuine non-spreading mode), or grow like a free
wavepacket (expected default per project_dst_electron_topology's
"capacitance not heat" discussion -- delocalization without a confining
mechanism is not itself evidence against anything, it's the QM-expected
default for a free particle)?
"""

import sys

sys.path.insert(0, ".")

import numpy as np

from equilibrium.substrate.lattice import SubstrateLattice
from equilibrium.substrate.traveling_wave import dispersion_omega, face_offsets


def build_ring_ic_companion(lat, offsets, w, n, r, width, omega, amplitude=1.0):
    pos = lat.pos
    rho = np.hypot(pos[:, 0], pos[:, 1])
    z = pos[:, 2]
    theta = np.arctan2(pos[:, 1], pos[:, 0])
    env = np.exp(-0.5 * ((rho - r) / width) ** 2) * np.exp(-0.5 * (z / width) ** 2)
    lat.u = amplitude * env * np.cos(n * theta)

    phi = np.zeros_like(lat.phi)
    for k in range(14):
        mid = pos + 0.5 * offsets[k]
        rho_mid = np.hypot(mid[:, 0], mid[:, 1])
        theta_mid = np.arctan2(mid[:, 1], mid[:, 0])
        safe_rho = np.where(rho_mid > 1e-9, rho_mid, 1.0)
        that_x = -mid[:, 1] / safe_rho
        that_y = mid[:, 0] / safe_rho
        K_local = n / safe_rho
        theta_k = K_local * (that_x * offsets[k, 0] + that_y * offsets[k, 1])
        Phi_k_amp = (2 * w[k] / omega) * np.sin(0.5 * theta_k)
        z_mid = mid[:, 2]
        env_mid = np.exp(-0.5 * ((rho_mid - r) / width) ** 2) * np.exp(-0.5 * (z_mid / width) ** 2)
        phi[:, k] = amplitude * Phi_k_amp * env_mid * np.cos(n * theta_mid)
    lat.phi = phi
    return theta, env, rho


def ring_amplitude(lat, n, theta, w):
    return float(np.sum(lat.u * np.cos(n * theta) * w) / np.sum(w**2 + 1e-300))


def field_energy(lat, mask):
    u2 = lat.u**2
    p2 = 0.5 * np.sum(lat.phi**2, axis=1)
    return float(np.sum((u2 + p2)[mask]))


def radial_spread(lat, mask, rho):
    e = lat.u[mask] ** 2 + 0.5 * np.sum(lat.phi[mask] ** 2, axis=1)
    tot = e.sum()
    if tot < 1e-300:
        return float("nan"), float("nan")
    rho_m = rho[mask]
    mean_rho = float(np.sum(rho_m * e) / tot)
    std_rho = float(np.sqrt(np.sum((rho_m - mean_rho) ** 2 * e) / tot))
    return mean_rho, std_rho


def run_one(n, r, width, R_domain, n_periods_target, report_every):
    lat = SubstrateLattice(radius_cells=R_domain)
    offsets, w = face_offsets(lat)
    K = n / r
    omega_pred = dispersion_omega(offsets, w, axis=1, K=K)
    theta, env, rho = build_ring_ic_companion(lat, offsets, w, n, r, width, omega_pred)
    mask = env > 0.05 * env.max()

    dt = lat.stable_dt()
    T_pred = 2 * np.pi / omega_pred if omega_pred > 0 else 1.0
    n_steps = max(int(round(n_periods_target * T_pred / dt)), report_every * 8)

    ts, amps, energies, mean_rhos, std_rhos = [], [], [], [], []
    for s in range(n_steps + 1):
        if s % report_every == 0:
            ts.append(s * dt)
            amps.append(ring_amplitude(lat, n, theta, mask))
            energies.append(field_energy(lat, mask))
            mr, sr = radial_spread(lat, mask, rho)
            mean_rhos.append(mr)
            std_rhos.append(sr)
        if s < n_steps:
            lat.step(dt)

    ts = np.array(ts)
    amps = np.array(amps)
    energies = np.array(energies)
    mean_rhos = np.array(mean_rhos)
    std_rhos = np.array(std_rhos)

    omega_meas = 0.0
    if len(ts) > 8:
        freqs = np.fft.rfftfreq(len(ts), d=(ts[1] - ts[0]))
        A = np.fft.rfft(amps - amps.mean())
        if len(A) > 1:
            dom = int(np.argmax(np.abs(A[1:]))) + 1
            omega_meas = 2 * np.pi * freqs[dom]

    H0, Hend = (energies[0], energies[-1]) if len(energies) else (0.0, 0.0)
    spread0 = std_rhos[len(std_rhos) // 10] if len(std_rhos) > 10 else std_rhos[0]  # skip initial transient
    spread_end = std_rhos[-1]
    return {
        "n": n,
        "r": r,
        "N": lat.pos.shape[0],
        "dt": dt,
        "n_steps": n_steps,
        "omega_pred": omega_pred,
        "omega_meas": omega_meas,
        "H0": H0,
        "H_end": Hend,
        "H_decay_frac": (H0 - Hend) / H0 if H0 else float("nan"),
        "spread0": spread0,
        "spread_end": spread_end,
        "spread_growth": (spread_end - spread0) / spread0 if spread0 else float("nan"),
        "ts": ts,
        "amps": amps,
        "energies": energies,
        "mean_rhos": mean_rhos,
        "std_rhos": std_rhos,
    }


def main():
    R_domain = 34.0
    width = 2.0
    n_periods_target = 6.0
    report_every = 10

    configs = [
        (2, 6.0),
        (2, 10.0),
        (2, 14.0),
        (4, 6.0),
        (4, 10.0),
        (4, 14.0),
        (6, 10.0),
        (6, 14.0),
        (6, 18.0),
    ]

    print(
        f"{'n':>3} {'r':>6} {'steps':>7} {'w_pred':>9} {'w_meas':>9} {'ratio':>7}"
        f" {'decay%':>7} {'spread0':>8} {'spread_end':>10} {'spread_grow%':>12}"
    )
    results = []
    for n, r in configs:
        if r + 4 * width > R_domain * 0.8:
            print(f"{n:3d} {r:6.1f}  -- skipped (too close to domain edge)")
            continue
        res = run_one(n, r, width, R_domain, n_periods_target, report_every)
        ratio = res["omega_meas"] / res["omega_pred"] if res["omega_pred"] else float("nan")
        print(
            f"{n:3d} {r:6.1f} {res['n_steps']:7d}"
            f" {res['omega_pred']:9.4f} {res['omega_meas']:9.4f} {ratio:7.3f}"
            f" {100*res['H_decay_frac']:7.2f} {res['spread0']:8.3f} {res['spread_end']:10.3f}"
            f" {100*res['spread_growth']:12.2f}",
            flush=True,
        )
        results.append(res)

    np.savez(
        "ring_closure_v2_results.npz",
        n=[r["n"] for r in results],
        r=[r["r"] for r in results],
        omega_pred=[r["omega_pred"] for r in results],
        omega_meas=[r["omega_meas"] for r in results],
        H_decay_frac=[r["H_decay_frac"] for r in results],
        spread0=[r["spread0"] for r in results],
        spread_end=[r["spread_end"] for r in results],
    )
    print("\nsaved -> ring_closure_v2_results.npz")


if __name__ == "__main__":
    main()
