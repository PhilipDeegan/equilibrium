"""
v7: genuinely non-degenerate two-component ring IC, queued next-step (2a)
from exp/2026/09/17/free_electron_soliton_alpha_scan/status.md.

v5 (exp/2026/09/17/.../alpha_scan_v5_standing.py) summed a +n and -n
companion-phi wave at equal amplitude and the SAME time reference (phase=0)
-- worked out algebraically (and confirmed by the worse v5 results) that
this cancels phi EXACTLY, reducing to v1's already-known-bad phi=0
"shocked" standing-wave IC. Not a real test of the "two coupled components"
idea from project_chirality_mixing_two_component_mass.

Here the minus-branch companion wave is built at a PHASE OFFSET (not
phase=0) relative to the plus branch, so the cancellation is not exact.
At phase=0 this construction reduces exactly to v5 (verified below in
`_verify_degenerate_limit`); at phase!=0 it does not.

Construction, per branch (+n / -n = which sign of K_local goes into the
companion-phi's Phi_k_amp -- the direction convention, NOT the spatial
winding number, which stays n for both so both branches decorate the SAME
ring): u_branch(theta) = amplitude*env*cos(n*theta - phase_branch),
phi_branch built as the WKB companion for that phase-shifted profile
(replace cos(n*theta_mid) with cos(n*theta_mid - phase_branch) in the
existing companion-phi formula; Phi_k_amp's sign, set by K_local's sign,
is unaffected by phase). plus branch: phase=0. minus branch: phase=phase.
Fields are summed. u always collapses to a single cos(n*theta - phase/2)
mode scaled by 2*cos(phase/2) (two same-n cosines always do, regardless of
phase) -- the physically new content at phase!=0 is that phi no longer
matches the clean single-branch companion relationship to u; the combined
(u,phi) state is no longer a solution of a single dispersion branch, unlike
v4's plain single-component IC.
"""

import sys
from multiprocessing import Pool

sys.path.insert(0, "/home/p/git/eq")

import numpy as np

from equilibrium.substrate.lattice import SubstrateLattice
from equilibrium.substrate.traveling_wave import dispersion_omega, face_offsets

N_WORKERS = 6


def build_two_component_ic(lat, offsets, w, n, r, width, omega, phase, amplitude=1.0):
    pos = lat.pos
    rho = np.hypot(pos[:, 0], pos[:, 1])
    z = pos[:, 2]
    theta = np.arctan2(pos[:, 1], pos[:, 0])
    env = np.exp(-0.5 * ((rho - r) / width) ** 2) * np.exp(-0.5 * (z / width) ** 2)
    lat.u = amplitude * env * (np.cos(n * theta) + np.cos(n * theta - phase))

    phi = np.zeros_like(lat.phi)
    for k in range(14):
        mid = pos + 0.5 * offsets[k]
        rho_mid = np.hypot(mid[:, 0], mid[:, 1])
        theta_mid = np.arctan2(mid[:, 1], mid[:, 0])
        safe_rho = np.where(rho_mid > 1e-9, rho_mid, 1.0)
        that_x = -mid[:, 1] / safe_rho
        that_y = mid[:, 0] / safe_rho
        z_mid = mid[:, 2]
        env_mid = np.exp(-0.5 * ((rho_mid - r) / width) ** 2) * np.exp(-0.5 * (z_mid / width) ** 2)

        K_plus = n / safe_rho
        theta_k_plus = K_plus * (that_x * offsets[k, 0] + that_y * offsets[k, 1])
        Phi_plus_amp = (2 * w[k] / omega) * np.sin(0.5 * theta_k_plus)
        phi_plus_k = amplitude * Phi_plus_amp * env_mid * np.cos(n * theta_mid)

        K_minus = -n / safe_rho
        theta_k_minus = K_minus * (that_x * offsets[k, 0] + that_y * offsets[k, 1])
        Phi_minus_amp = (2 * w[k] / omega) * np.sin(0.5 * theta_k_minus)
        phi_minus_k = amplitude * Phi_minus_amp * env_mid * np.cos(n * theta_mid - phase)

        phi[:, k] = phi_plus_k + phi_minus_k
    lat.phi = phi
    return theta, env, rho


def _verify_degenerate_limit():
    """phase=0 must reduce to v5's exact phi=0 standing wave (sanity check
    on the construction before trusting any nonzero-phase result)."""
    lat = SubstrateLattice(radius_cells=20.0)
    offsets, w = face_offsets(lat)
    omega = dispersion_omega(offsets, w, axis=1, K=2 / 10.0)
    build_two_component_ic(lat, offsets, w, n=2, r=10.0, width=2.0, omega=omega, phase=0.0)
    phi_max = float(np.abs(lat.phi).max())
    assert phi_max < 1e-10, f"phase=0 should give exactly phi=0, got max|phi|={phi_max}"
    print(f"[verify] phase=0 degenerate limit OK, max|phi|={phi_max:.2e}")


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


def local_energy_density(lat):
    return lat.u**2 + 0.5 * np.sum(lat.phi**2, axis=1)


def run_one(n, r, width, R_domain, phase, n_periods_target, report_every, alpha):
    lat = SubstrateLattice(radius_cells=R_domain)
    offsets, w = face_offsets(lat)
    K = n / r
    omega_pred = dispersion_omega(offsets, w, axis=1, K=K)
    theta, env, rho = build_two_component_ic(lat, offsets, w, n, r, width, omega_pred, phase)
    mask = env > 0.05 * env.max()

    dt = lat.stable_dt()
    T_pred = 2 * np.pi / omega_pred if omega_pred > 0 else 1.0
    n_steps = max(int(round(n_periods_target * T_pred / dt)), report_every * 8)

    ts, energies, std_rhos, lam_maxs = [], [], [], []
    for s in range(n_steps + 1):
        if alpha > 0.0:
            lat.lam = alpha * local_energy_density(lat)
        if s % report_every == 0:
            ts.append(s * dt)
            energies.append(field_energy(lat, mask))
            _, sr = radial_spread(lat, mask, rho)
            std_rhos.append(sr)
            lam_maxs.append(float(lat.lam.max()) if alpha > 0.0 else 0.0)
        if s < n_steps:
            lat.step(dt)

    ts = np.array(ts)
    energies = np.array(energies)
    std_rhos = np.array(std_rhos)
    spread0 = std_rhos[len(std_rhos) // 10] if len(std_rhos) > 10 else std_rhos[0]
    spread_end = std_rhos[-1]

    half = slice(len(ts) // 2, len(ts))
    t_h, y_h = ts[half], std_rhos[half]
    slope_frac_per_period = float("nan")
    if len(t_h) >= 4 and y_h.mean() > 0:
        A = np.vstack([t_h, np.ones_like(t_h)]).T
        slope, _ = np.linalg.lstsq(A, y_h, rcond=None)[0]
        slope_frac_per_period = slope * T_pred / y_h.mean()

    return {
        "n": n,
        "r": r,
        "phase": phase,
        "alpha": alpha,
        "n_steps": n_steps,
        "T_pred": T_pred,
        "H0": energies[0],
        "H_end": energies[-1],
        "H_decay_frac": (energies[0] - energies[-1]) / energies[0],
        "spread0": spread0,
        "spread_end": spread_end,
        "spread_growth": (spread_end - spread0) / spread0,
        "slope_frac_per_period": slope_frac_per_period,
        "lam_max": max(lam_maxs) if lam_maxs else 0.0,
    }


def find_crossover(alphas, growths):
    for i in range(len(alphas) - 1):
        g0, g1 = growths[i], growths[i + 1]
        if np.isnan(g0) or np.isnan(g1):
            continue
        if g0 >= 0.0 >= g1 or g0 <= 0.0 <= g1:
            if g1 == g0:
                continue
            a0, a1 = alphas[i], alphas[i + 1]
            return a0 + (a1 - a0) * (0.0 - g0) / (g1 - g0)
    return float("nan")


def _worker(task):
    n, r, width, R_domain, phase, n_periods_target, report_every, alpha = task
    return run_one(n, r, width, R_domain, phase, n_periods_target, report_every, alpha)


def main():
    _verify_degenerate_limit()

    width = 2.0
    R_domain = 34.0
    n_periods_target = 6.0
    report_every = 5

    n, r = 2, 10.0  # v3/v4 baseline config
    # phase offsets: pi/2 (quarter of a full 2*pi winding cycle -- a clean,
    # well-motivated first non-degenerate point) and pi (maximal departure,
    # amplitude factor 2*cos(phase/2)=0 for u -- also worth seeing) plus a
    # small-offset point (pi/8) to check the near-degenerate limit behaves
    # continuously, not just the two extremes.
    phases = [np.pi / 8, np.pi / 2, np.pi]
    alphas = [0.0, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 2.0]

    tasks = [
        (n, r, width, R_domain, phase, n_periods_target, report_every, alpha)
        for phase in phases
        for alpha in alphas
    ]

    print(f"running {len(tasks)} (phase,alpha) tasks across {N_WORKERS} workers...", flush=True)
    with Pool(processes=N_WORKERS) as pool:
        raw = pool.map(_worker, tasks)

    by_phase = {}
    for res in raw:
        by_phase.setdefault(res["phase"], []).append(res)

    all_results = {}
    for phase, res_list in sorted(by_phase.items()):
        res_list.sort(key=lambda r: r["alpha"])
        print(f"\n=== phase={phase:.4f} rad ({phase/np.pi:.3f}*pi) ===")
        print(
            f"{'alpha':>6} {'lam_max':>9} {'decay%':>7} {'spread0':>8}"
            f" {'spread_end':>10} {'grow%':>8} {'slope/period%':>13}"
        )
        for res in res_list:
            print(
                f"{res['alpha']:6.2f} {res['lam_max']:9.4f} {100*res['H_decay_frac']:7.2f}"
                f" {res['spread0']:8.3f} {res['spread_end']:10.3f}"
                f" {100*res['spread_growth']:8.2f} {100*res['slope_frac_per_period']:13.2f}",
                flush=True,
            )
        alphas_sorted = [r["alpha"] for r in res_list]
        growths = [r["spread_growth"] for r in res_list]
        alpha_star = find_crossover(alphas_sorted, growths)
        print(f"--> alpha* (spread_growth==0, linear interp): {alpha_star:.3f}")
        all_results[phase] = {"alphas": alphas_sorted, "res": res_list, "alpha_star": alpha_star}

    print("\n=== summary: crossover alpha* per phase ===")
    print(f"{'phase/pi':>10} {'alpha*':>8}")
    for phase, d in all_results.items():
        print(f"{phase/np.pi:10.3f} {d['alpha_star']:8.3f}")

    np.savez(
        "v7_two_component_results.npz",
        phases=np.array(list(all_results.keys())),
        alphas=np.array(alphas),
        alpha_stars=np.array([d["alpha_star"] for d in all_results.values()]),
        **{
            f"phase{i}_growth": np.array([res["spread_growth"] for res in d["res"]])
            for i, (phase, d) in enumerate(all_results.items())
        },
        **{
            f"phase{i}_slope": np.array([res["slope_frac_per_period"] for res in d["res"]])
            for i, (phase, d) in enumerate(all_results.items())
        },
    )
    print("\nsaved -> v7_two_component_results.npz")


if __name__ == "__main__":
    main()
