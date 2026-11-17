"""
v5: same self-sourced-lam alpha scan as v4, but on the STANDING (phi=0,
v1-style) IC instead of the traveling companion-phi (v2/v3/v4-style) IC --
follow-up to today's QM discussion (project_chirality_mixing_two_component_
mass): the Dirac mass term is literally the coupling between two
counter-propagating (at c) chirality components, and zitterbewegung is the
observable beat between them. A standing wave u=cos(n*theta)*env(rho,z),
phi=0 at t=0 IS the equal-amplitude superposition of a +n and -n traveling
companion wave sampled at their common zero-velocity turning point (worked
out by hand before writing this: adding the +n and -n companion-phi
formulas at matched amplitude/time cancels phi EXACTLY, since Phi_k_amp
flips sign under K_local -> -K_local -- so "two counter-propagating
components at t=0" reduces to exactly the ORIGINAL v1 IC, not a new
construction). v1 already found this IC "decays" (45-99% energy loss from
the measurement window) under LINEAR dynamics, same as the traveling case,
later shown to be genuine reversible envelope diffraction (Derrick's
theorem), not real dissipation -- so the open question this script answers
is specifically: does the self-sourced-lam nonlinearity behave differently
on the STANDING (two-component) IC than it did on the TRAVELING
(one-component) IC in v4, using the identical alpha grid/configs/diagnostics
for a direct comparison?

Self-contained; only the IC construction differs from v4 (build_ring_ic
below vs. build_ring_ic_companion) -- everything else (field_energy,
radial_spread, local_energy_density, run_one, slope diagnostic, crossover
finder, config/alpha grid, multiprocessing) is unchanged from v4 so the two
runs are directly comparable number-for-number.
"""

import sys
from multiprocessing import Pool

sys.path.insert(0, "/home/p/git/eq")

import numpy as np

from equilibrium.substrate.lattice import SubstrateLattice
from equilibrium.substrate.traveling_wave import dispersion_omega, face_offsets

N_WORKERS = 6


def build_ring_ic(lat, n, r, width, amplitude=1.0):
    pos = lat.pos
    rho = np.hypot(pos[:, 0], pos[:, 1])
    z = pos[:, 2]
    theta = np.arctan2(pos[:, 1], pos[:, 0])
    env = np.exp(-0.5 * ((rho - r) / width) ** 2) * np.exp(-0.5 * (z / width) ** 2)
    lat.u = amplitude * env * np.cos(n * theta)
    lat.phi = np.zeros_like(lat.phi)
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
    theta, env, rho = build_ring_ic(lat, n, r, width)
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
    n, r, width, R_domain, n_periods_target, report_every, alpha = task
    return (n, r, alpha, run_one(n, r, width, R_domain, n_periods_target, report_every, alpha))


def main():
    width = 2.0
    R_domain = 34.0
    n_periods_target = 6.0
    report_every = 5

    configs = [
        (2, 6.0),
        (2, 10.0),
        (6, 14.0),
    ]

    alphas = [0.0, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 2.0]

    tasks = []
    for n, r in configs:
        if r + 4 * width > R_domain * 0.8:
            print(f"n={n} r={r} -- skipped (too close to domain edge)")
            continue
        for alpha in alphas:
            tasks.append((n, r, width, R_domain, n_periods_target, report_every, alpha))

    print(f"running {len(tasks)} (config,alpha) tasks across {N_WORKERS} workers...", flush=True)
    with Pool(processes=N_WORKERS) as pool:
        raw = pool.map(_worker, tasks)

    by_config = {}
    for n, r, alpha, res in raw:
        by_config.setdefault((n, r), []).append((alpha, res))

    all_results = {}
    for (n, r), pairs in by_config.items():
        pairs.sort(key=lambda p: p[0])
        print(f"\n=== n={n} r={r} (STANDING, phi=0 IC) ===")
        print(
            f"{'alpha':>6} {'lam_max':>9} {'decay%':>7} {'spread0':>8}"
            f" {'spread_end':>10} {'grow%':>8} {'slope/period%':>13}"
        )
        res_list = []
        for alpha, res in pairs:
            res_list.append(res)
            print(
                f"{alpha:6.2f} {res['lam_max']:9.4f} {100*res['H_decay_frac']:7.2f}"
                f" {res['spread0']:8.3f} {res['spread_end']:10.3f}"
                f" {100*res['spread_growth']:8.2f} {100*res['slope_frac_per_period']:13.2f}",
                flush=True,
            )
        alphas_sorted = [p[0] for p in pairs]
        growths = [r["spread_growth"] for r in res_list]
        alpha_star = find_crossover(alphas_sorted, growths)
        print(f"--> alpha* (spread_growth==0, linear interp): {alpha_star:.3f}")
        all_results[(n, r)] = {"alphas": alphas_sorted, "res": res_list, "alpha_star": alpha_star}

    print("\n=== summary: crossover alpha* per config (STANDING IC) ===")
    print(f"{'n':>3} {'r':>6} {'alpha*':>8}")
    for (n, r), d in all_results.items():
        print(f"{n:3d} {r:6.1f} {d['alpha_star']:8.3f}")

    np.savez(
        "alpha_scan_v5_standing_results.npz",
        configs=np.array(list(all_results.keys())),
        alpha_stars=np.array([d["alpha_star"] for d in all_results.values()]),
        alphas=np.array(alphas),
        **{
            f"n{n}_r{r}_growth": np.array([res["spread_growth"] for res in d["res"]])
            for (n, r), d in all_results.items()
        },
        **{
            f"n{n}_r{r}_slope": np.array([res["slope_frac_per_period"] for res in d["res"]])
            for (n, r), d in all_results.items()
        },
    )
    print("\nsaved -> alpha_scan_v5_standing_results.npz")


if __name__ == "__main__":
    main()
