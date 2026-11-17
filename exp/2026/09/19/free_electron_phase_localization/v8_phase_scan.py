"""
v8: localize the v7 finding. exp/2026/09/18/.../status.md found that the
two-component phase-offset IC at (n=2,r=10), phase=0.125*pi is the first
config across v1-v7 where spread_growth and slope_frac_per_period agree in
sign (both slightly negative near alpha~0.9-1.1); phase=0.5*pi and pi did
NOT show this. Queued next step #1: finer phase scan 0-0.25*pi around
(2,10) to localize/refine the effect -- not done until now.

Same construction as v7_two_component.py (copied verbatim, not imported,
per that script's own "self-contained" convention), just a finer
(phase, alpha) grid: 8 phases in (0, 0.25*pi] and a denser alpha grid
around the v7 crossover region (0.7-1.3) instead of v7's coarser 9 points.
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
    width = 2.0
    R_domain = 34.0
    n_periods_target = 6.0
    report_every = 5

    n, r = 2, 10.0

    # finer localization around v7's phase=0.125*pi hit, staying in (0, 0.25*pi]
    phases = (np.array([0.03, 0.0625, 0.09375, 0.125, 0.15625, 0.1875, 0.21875, 0.25]) * np.pi).tolist()
    # denser alpha grid around v7's crossover region (alpha* was 0.725-0.858 across phases)
    alphas = [0.0, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 2.0]

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
        print(f"\n=== phase={phase:.4f} rad ({phase/np.pi:.4f}*pi) ===")
        print(
            f"{'alpha':>6} {'lam_max':>9} {'decay%':>7} {'spread0':>8}"
            f" {'spread_end':>10} {'grow%':>8} {'slope/period%':>13}"
        )
        best_agree = None
        for res in res_list:
            grow = 100 * res["spread_growth"]
            slope = 100 * res["slope_frac_per_period"]
            agree = "  <-- AGREE" if (grow < 0 and slope < 0) else ""
            if agree:
                best_agree = res["alpha"] if best_agree is None else best_agree
            print(
                f"{res['alpha']:6.2f} {res['lam_max']:9.4f} {100*res['H_decay_frac']:7.2f}"
                f" {res['spread0']:8.3f} {res['spread_end']:10.3f}"
                f" {grow:8.2f} {slope:13.2f}{agree}",
                flush=True,
            )
        alphas_sorted = [r["alpha"] for r in res_list]
        growths = [r["spread_growth"] for r in res_list]
        alpha_star = find_crossover(alphas_sorted, growths)
        print(f"--> alpha* (spread_growth==0, linear interp): {alpha_star:.3f}")
        print(f"--> first alpha with grow<0 AND slope<0: {best_agree}")
        all_results[phase] = {
            "alphas": alphas_sorted,
            "res": res_list,
            "alpha_star": alpha_star,
            "first_agree_alpha": best_agree,
        }

    print("\n=== summary: crossover alpha* and first-agreement alpha per phase ===")
    print(f"{'phase/pi':>10} {'alpha*':>8} {'first_agree':>12}")
    for phase, d in all_results.items():
        fa = d["first_agree_alpha"]
        fa_str = f"{fa:.2f}" if fa is not None else "none"
        print(f"{phase/np.pi:10.4f} {d['alpha_star']:8.3f} {fa_str:>12}")

    phases_arr = np.array(list(all_results.keys()))
    np.savez(
        "v8_phase_scan_results.npz",
        phases=phases_arr,
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
    print("\nsaved -> v8_phase_scan_results.npz")


if __name__ == "__main__":
    main()
