"""
Resume script for v8b_phase_scan_fast.py, interrupted (clean shutdown, not
a crash) after 6/32 tasks. Reads v8b_results.jsonl for what's already done,
runs only the remaining (phase,alpha) pairs, APPENDS (not overwrites) to
the same file. Same grid/physics as v8b_phase_scan_fast.py -- copy that
file's docstring for why this grid (leaner than the original v8, which was
going to take ~9h) was chosen.
"""

import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import time
from multiprocessing import Pool

sys.path.insert(0, "/home/p/git/eq")

import numpy as np

from equilibrium.substrate.lattice import SubstrateLattice
from equilibrium.substrate.traveling_wave import dispersion_omega, face_offsets

N_WORKERS = 6
RESULTS_FILE = "v8b_results.jsonl"


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


def radial_spread(lat, mask, rho):
    e = lat.u[mask] ** 2 + 0.5 * np.sum(lat.phi[mask] ** 2, axis=1)
    tot = e.sum()
    if tot < 1e-300:
        return float("nan"), float("nan")
    rho_m = rho[mask]
    mean_rho = float(np.sum(rho_m * e) / tot)
    std_rho = float(np.sqrt(np.sum((rho_m - mean_rho) ** 2 * e) / tot))
    return mean_rho, std_rho


def field_energy(lat, mask):
    u2 = lat.u**2
    p2 = 0.5 * np.sum(lat.phi**2, axis=1)
    return float(np.sum((u2 + p2)[mask]))


def local_energy_density(lat):
    return lat.u**2 + 0.5 * np.sum(lat.phi**2, axis=1)


def run_one(n, r, width, R_domain, phase, n_periods_target, report_every, alpha):
    t0 = time.time()
    lat = SubstrateLattice(radius_cells=R_domain)
    offsets, w = face_offsets(lat)
    K = n / r
    omega_pred = dispersion_omega(offsets, w, axis=1, K=K)
    theta, env, rho = build_two_component_ic(lat, offsets, w, n, r, width, omega_pred, phase)
    mask = env > 0.05 * env.max()

    dt = lat.stable_dt()
    T_pred = 2 * np.pi / omega_pred if omega_pred > 0 else 1.0
    n_steps = max(int(round(n_periods_target * T_pred / dt)), report_every * 8)

    ts, energies, std_rhos = [], [], []
    for s in range(n_steps + 1):
        if alpha > 0.0:
            lat.lam = alpha * local_energy_density(lat)
        if s % report_every == 0:
            ts.append(s * dt)
            energies.append(field_energy(lat, mask))
            _, sr = radial_spread(lat, mask, rho)
            std_rhos.append(sr)
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
        "H_decay_frac": float((energies[0] - energies[-1]) / energies[0]),
        "spread_growth": float((spread_end - spread0) / spread0),
        "slope_frac_per_period": slope_frac_per_period,
        "wall_s": time.time() - t0,
    }


def _worker(task):
    n, r, width, R_domain, phase, n_periods_target, report_every, alpha = task
    return run_one(n, r, width, R_domain, phase, n_periods_target, report_every, alpha)


def main():
    width = 2.0
    R_domain = 34.0
    n_periods_target = 3.0
    report_every = 5

    n, r = 2, 10.0
    phases = (np.array([0.0625, 0.125, 0.1875, 0.25]) * np.pi).tolist()
    alphas = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]

    done = set()
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            for line in f:
                r_ = json.loads(line)
                done.add((round(r_["phase"], 6), round(r_["alpha"], 6)))
        print(f"found {len(done)} already-done results in {RESULTS_FILE}, skipping those", flush=True)

    tasks = [
        (n, r, width, R_domain, phase, n_periods_target, report_every, alpha)
        for phase in phases
        for alpha in alphas
        if (round(phase, 6), round(alpha, 6)) not in done
    ]

    print(f"running {len(tasks)} remaining (phase,alpha) tasks across {N_WORKERS} workers", flush=True)

    t_start = time.time()
    n_done = 0
    with open(RESULTS_FILE, "a") as f:
        with Pool(processes=N_WORKERS) as pool:
            for res in pool.imap_unordered(_worker, tasks):
                n_done += 1
                f.write(json.dumps(res) + "\n")
                f.flush()
                grow = 100 * res["spread_growth"]
                slope = 100 * res["slope_frac_per_period"]
                agree = "  <-- AGREE (both neg)" if (grow < 0 and slope < 0) else ""
                elapsed = time.time() - t_start
                eta = elapsed / n_done * (len(tasks) - n_done)
                print(
                    f"[{n_done:3d}/{len(tasks)}] phase={res['phase']/np.pi:.4f}pi alpha={res['alpha']:.2f} "
                    f"grow%={grow:+7.2f} slope%={slope:+7.2f}{agree}  "
                    f"(task {res['wall_s']:.0f}s, elapsed {elapsed/60:.1f}min, ETA {eta/60:.1f}min)",
                    flush=True,
                )

    print(f"\ndone with remaining tasks in {(time.time()-t_start)/60:.1f} min", flush=True)

    print("\n=== full summary (all 32, old + new) from v8b_results.jsonl ===", flush=True)
    all_res = []
    with open(RESULTS_FILE) as f:
        for line in f:
            all_res.append(json.loads(line))
    by_phase = {}
    for r_ in all_res:
        by_phase.setdefault(r_["phase"], []).append(r_)
    for phase, res_list in sorted(by_phase.items()):
        res_list.sort(key=lambda r_: r_["alpha"])
        print(f"\n=== phase={phase/np.pi:.4f}*pi ===")
        for r_ in res_list:
            grow = 100 * r_["spread_growth"]
            slope = 100 * r_["slope_frac_per_period"]
            agree = "  <-- AGREE" if (grow < 0 and slope < 0) else ""
            print(f"  alpha={r_['alpha']:.2f}  grow%={grow:+7.2f}  slope%={slope:+7.2f}{agree}")


if __name__ == "__main__":
    main()
