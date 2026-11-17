"""
v6: resolution check on the v4 alpha-scan family
(exp/2026/09/17/free_electron_soliton_alpha_scan/alpha_scan_v4.py), queued
next-step (1) from that session's status.md.

v4's ring envelope used `width=2.0` cells -- only ~2 lattice cells per
Gaussian sigma against the lattice's own unit-1.0 hop spacing, an
unresolved risk flagged but never tested. Since the lattice has fixed
hop-spacing 1.0 (no mesh refinement available), "higher resolution" here
means enlarging the whole construction (width, and the domain that must
contain it) so more lattice cells represent the same shape -- the same
convergence-test idea as `project_hop_telescope_resolved_spheromak`'s
`cells_across` sweep, applied here to the ring's cross-section instead of
a spheromak's radius. `r` and `n` are held FIXED per config (changing them
would test a different physical object, not resolution).

R_domain is matched across widths to keep the SAME relative clearance
fraction ((r+4*width)/R_domain ~ 0.5) for every (config, width) pair --
this also serves queued next-step (3), since it gives (6,14) proper
clearance instead of the tight ~19% it had in v4 at fixed R_domain=34.

Self-contained, same physics/IC as v4 (single traveling companion-phi
component, self-sourced lam = alpha*(u^2+0.5*sum(phi^2)) nonlinearity).
"""

import sys
from multiprocessing import Pool

sys.path.insert(0, "/home/p/git/eq")

import numpy as np

from equilibrium.substrate.lattice import SubstrateLattice
from equilibrium.substrate.traveling_wave import dispersion_omega, face_offsets

N_WORKERS = 6
CLEARANCE_FRAC = 0.5  # (r + 4*width) / R_domain, matched across all runs


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


def run_one(n, r, width, R_domain, n_periods_target, report_every, alpha):
    lat = SubstrateLattice(radius_cells=R_domain)
    offsets, w = face_offsets(lat)
    K = n / r
    omega_pred = dispersion_omega(offsets, w, axis=1, K=K)
    theta, env, rho = build_ring_ic_companion(lat, offsets, w, n, r, width, omega_pred)
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
        "width": width,
        "R_domain": R_domain,
        "alpha": alpha,
        "n_cells": int(lat.n_cells) if hasattr(lat, "n_cells") else -1,
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


def _worker(task):
    n, r, width, R_domain, n_periods_target, report_every, alpha = task
    return run_one(n, r, width, R_domain, n_periods_target, report_every, alpha)


def matched_R_domain(r, width):
    return round((r + 4 * width) / CLEARANCE_FRAC, 1)


def main():
    n_periods_target = 6.0
    report_every = 5

    # (n, r, widths-to-test) -- widths=[2.0] is the v4 baseline resolution;
    # 4.0/6.0 are 2x/3x. Reduced alpha grid (not v4's full 9) to keep the
    # larger-R_domain runs (cost ~ R_domain^3) tractable: alpha=0 (baseline
    # growth direction), 0.9 (v4's (2,10) alpha* crossover), 1.3 (v4's
    # upper tested value) -- enough to see whether the QUALITATIVE v4
    # conclusion ("slope stays positive at every alpha, never a genuine
    # plateau") survives at higher resolution, without a full re-scan.
    # width=6.0 (3x) dropped from the first pass -- timing check showed
    # R=36 (width=2) costs ~400s/run single-threaded, and cost scales ~R^3
    # (cell count), so width=6 (R~68) would cost ~45min PER (config,alpha)
    # run, dominating the whole batch. width=4.0 (2x) already answers the
    # main convergence question; add width=6.0 as a follow-up only if 4.0
    # shows meaningful drift from the 2.0 baseline.
    configs = [
        (2, 10.0, [2.0, 4.0]),
        (6, 14.0, [2.0, 4.0]),
    ]
    alphas = [0.0, 0.9, 1.3]

    tasks = []
    for n, r, widths in configs:
        for width in widths:
            R_domain = matched_R_domain(r, width)
            for alpha in alphas:
                tasks.append((n, r, width, R_domain, n_periods_target, report_every, alpha))

    print(f"running {len(tasks)} tasks across {N_WORKERS} workers...", flush=True)
    for n, r, width, R_domain, *_ in tasks[::len(alphas)]:
        print(f"  n={n} r={r} width={width} -> R_domain={R_domain}", flush=True)

    with Pool(processes=N_WORKERS) as pool:
        raw = pool.map(_worker, tasks)

    by_config = {}
    for res in raw:
        by_config.setdefault((res["n"], res["r"], res["width"]), []).append(res)

    print("\n=== results ===")
    print(
        f"{'n':>3} {'r':>5} {'width':>6} {'R_dom':>6} {'alpha':>6} {'lam_max':>9}"
        f" {'decay%':>7} {'grow%':>8} {'slope/period%':>13}"
    )
    for (n, r, width), res_list in sorted(by_config.items()):
        res_list.sort(key=lambda x: x["alpha"])
        for res in res_list:
            print(
                f"{n:3d} {r:5.1f} {width:6.1f} {res['R_domain']:6.1f}"
                f" {res['alpha']:6.2f} {res['lam_max']:9.4f} {100*res['H_decay_frac']:7.2f}"
                f" {100*res['spread_growth']:8.2f} {100*res['slope_frac_per_period']:13.2f}",
                flush=True,
            )

    np.savez(
        "v6_resolution_check_results.npz",
        results=np.array(raw, dtype=object),
    )
    print("\nsaved -> v6_resolution_check_results.npz")


if __name__ == "__main__":
    main()
