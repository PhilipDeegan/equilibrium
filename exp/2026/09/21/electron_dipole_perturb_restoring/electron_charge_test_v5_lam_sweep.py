"""
v5: extends v4's short-window rate-of-change read to a wider lam sweep
(0, 1, 2, 4, 8), same resolved R=80/electron_scale=50 setup, looking for
where frac_rate trends -- toward a zero-crossing (a genuine lam* where the
combined dipole+deficit state is invariant) or an asymptotic nonzero floor
(never truly invariant, just increasingly slow). Philip's framing: this is
an equilibrium SEARCH, not a one-shot construction -- the first guess
(lam=0) isn't expected to already be the fixed point.
"""

import sys
import time

sys.path.insert(0, "/home/p/git/eq")

import numpy as np

from equilibrium import lattice as _lat
from tst.substrate.adaptive import compute_dt
from tst.substrate.adaptive.adaptive_flux_tube import init_electron_dipole_ic_vecpot

R_DOMAIN = 80.0
ELECTRON_SCALE = 50.0
SPONGE_CELLS = 10.0
SPONGE_FRACTION = SPONGE_CELLS / R_DOMAIN
DAMP_MAX = 0.003
N_THREADS = 8

DEFICIT_AMP = 0.2
DEFICIT_WIDTH_FRAC = 0.3
N_STEPS = 400
REPORT_EVERY = 10

LAM_TARGETS = [0.0, 1.0, 2.0, 4.0, 8.0]


def build_cluster():
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    alat.set_n_threads(N_THREADS)
    cl = alat[0]
    _lat.build_bcc_sphere(cl, R_DOMAIN, d1=1.0, u0=0.0, ell0=1.0, lam0=0.0)
    alat.rebuild(0)
    r_all = np.linalg.norm(np.array(cl.leaf_pos), axis=1)
    r_inner = float(r_all.max()) * (1.0 - SPONGE_FRACTION)
    cl.set_bc(
        _lat.BoundaryCondition.quadratic_sponge(
            target=0.0, damp_max=DAMP_MAX, r_all=r_all, r_inner=r_inner
        )
    )
    return alat, cl


def run_lam(alat, cl, lam_target, r_from_center, mask_window, deficit, dt):
    _, r_axis = init_electron_dipole_ic_vecpot(
        cl, R_sph=1.0, scale=ELECTRON_SCALE, B0=1.0, lam_target=lam_target, u_bg=0.0,
        amplitude=1.0, div_clean=True, clean_rtol=1e-8,
    )
    phi0 = np.array(cl.leaf_phi)
    lam0 = np.array(cl.leaf_lam)
    u0 = np.zeros(cl.n_leaves) - deficit
    phi, u, lam = phi0.copy(), u0.copy(), lam0.copy()

    times, deficit_hist = [], []
    t0 = time.time()
    for step in range(N_STEPS + 1):
        if step % REPORT_EVERY == 0:
            times.append(step * dt)
            deficit_hist.append(float((-u[mask_window]).sum()))
        if step < N_STEPS:
            cl.set_leaf_phi(phi.copy())
            cl.set_leaf_u(u.copy())
            cl.set_leaf_lam(lam.copy())
            alat.step(0, dt)
            alat.step_sponge(0)
            phi, u = np.array(cl.leaf_phi), np.array(cl.leaf_u)

    times = np.array(times)
    deficit_hist = np.array(deficit_hist)
    A = np.vstack([times, np.ones_like(times)]).T
    slope, _ = np.linalg.lstsq(A, deficit_hist, rcond=None)[0]
    frac_rate = slope / deficit_hist[0] if deficit_hist[0] else float("nan")
    lam_a = np.array(cl.leaf_lam)
    print(
        f"[lam_target={lam_target:5.1f}] frac_rate={frac_rate:+.4e}/t  "
        f"lam_static mean(nonzero)={lam_a[lam_a>0].mean() if (lam_a>0).any() else 0:.4f}  "
        f"max={lam_a.max():.4f}  ({time.time()-t0:.1f}s)",
        flush=True,
    )
    return {"lam_target": lam_target, "times": times, "deficit": deficit_hist, "frac_rate": float(frac_rate)}


def main():
    alat, cl = build_cluster()
    dt = compute_dt(cl)
    pos = np.array(cl.leaf_pos)
    r_from_center = np.linalg.norm(pos, axis=1)
    width = DEFICIT_WIDTH_FRAC * ELECTRON_SCALE
    deficit = DEFICIT_AMP * np.exp(-0.5 * (r_from_center / width) ** 2)
    mask_window = r_from_center <= 3.0 * width
    print(f"N={cl.n_leaves}  electron_scale={ELECTRON_SCALE}  dt={dt:.4e}", flush=True)

    print(f"\n=== lam-target sweep for zero-rate crossing, {N_STEPS} steps each ===", flush=True)
    results = {}
    for lam_target in LAM_TARGETS:
        results[lam_target] = run_lam(alat, cl, lam_target, r_from_center, mask_window, deficit, dt)

    print("\n=== summary: frac_rate vs lam_target ===", flush=True)
    for lam_target in LAM_TARGETS:
        print(f"  lam_target={lam_target:5.1f}  frac_rate={results[lam_target]['frac_rate']:+.4e}/t", flush=True)

    lams = np.array(LAM_TARGETS)
    rates = np.array([results[l]["frac_rate"] for l in LAM_TARGETS])
    if np.all(rates < 0):
        # crude log-linear extrapolation to see if/where it would cross zero
        logr = np.log(-rates)
        slope, intercept = np.polyfit(lams, logr, 1)
        print(
            f"\nlog(-frac_rate) vs lam_target: slope={slope:+.4f}  "
            f"({'decaying toward an asymptote, never reaching exactly 0' if slope < 0 else 'growing -- not converging'})",
            flush=True,
        )

    out = {"lam_targets": lams, "frac_rates": rates}
    for lam_target in LAM_TARGETS:
        out[f"t_{lam_target:g}"] = results[lam_target]["times"]
        out[f"deficit_{lam_target:g}"] = results[lam_target]["deficit"]
    np.savez("exp/2026/09/21/electron_dipole_perturb_restoring/results_lam_sweep_v5.npz", **out)
    print("\nsaved results_lam_sweep_v5.npz", flush=True)


if __name__ == "__main__":
    main()
