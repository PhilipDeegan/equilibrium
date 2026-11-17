"""
v4: same free (unpinned) u-deficit + double-ring dipole test as
electron_charge_test.py, but (1) at a properly resolved scale --
R_DOMAIN=80, electron scale=50 (cells_across=100, vs v3's unresolved
cells_across=9 -- Philip caught this: this project has a validated
precedent at cells_across=220, R=180/scale=110, for a related
poloidal+toroidal construction, project_curl_full_dipole_spheromak) --
and (2) reading only a SHORT early-time window, not chasing a long
horizon. Philip's correction: "an electron at rest, the rate of change
should be basically zero... we don't need to run it forever to see that
it doesn't change" -- the bare dipole's own stationarity check (v1, 200
steps) never needed a long run either (rel H change 1.3e-14). The question
here is the same kind of cheap, short-window rate-of-change read, just
applied to the u-deficit instead of the bare IC: does adding it move the
system at a rate comparable to free dispersal (vacuum), or does the
resolved structure genuinely slow it down -- readable in a few hundred
steps, no need to resolve the long-horizon boundary-reflection question
at all this time.
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
N_STEPS = 400  # short-window rate-of-change read, not a long-horizon chase
REPORT_EVERY = 10

CONFIGS = ["vacuum", "dipole_lam0", "dipole_lam1", "dipole_lam2"]


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
    print(f"domain R={R_DOMAIN}  sponge from r={r_inner:.1f} to {r_all.max():.1f} ({SPONGE_CELLS} cells)", flush=True)
    return alat, cl


def run_config(alat, cl, name, r_from_center, mask_window, deficit, dt):
    if name == "vacuum":
        phi0 = np.zeros((14, cl.n_leaves))
        lam0 = np.zeros(cl.n_leaves)
    else:
        lam_target = {"dipole_lam0": 0.0, "dipole_lam1": 1.0, "dipole_lam2": 2.0}[name]
        init_electron_dipole_ic_vecpot(
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
    # rate of change over the short window: fit a line, report as a FRACTION
    # of the initial deficit per unit time -- directly comparable to how
    # the bare dipole's own stationarity was judged (relative change).
    A = np.vstack([times, np.ones_like(times)]).T
    slope, _ = np.linalg.lstsq(A, deficit_hist, rcond=None)[0]
    frac_rate = slope / deficit_hist[0] if deficit_hist[0] else float("nan")
    rel_change_end = abs(deficit_hist[-1] - deficit_hist[0]) / abs(deficit_hist[0])

    print(
        f"[{name:>12s}] deficit0={deficit_hist[0]:.4e}  deficit@t={times[-1]:.2f}={deficit_hist[-1]:.4e}  "
        f"linear_rate={slope:+.4e}/t  frac_rate={frac_rate:+.4e}/t  "
        f"rel_change_over_window={rel_change_end:.4f}  ({time.time()-t0:.1f}s)",
        flush=True,
    )
    return {"name": name, "times": times, "deficit": deficit_hist, "frac_rate": float(frac_rate)}


def main():
    alat, cl = build_cluster()
    dt = compute_dt(cl)
    pos = np.array(cl.leaf_pos)
    r_from_center = np.linalg.norm(pos, axis=1)
    width = DEFICIT_WIDTH_FRAC * ELECTRON_SCALE
    deficit = DEFICIT_AMP * np.exp(-0.5 * (r_from_center / width) ** 2)
    mask_window = r_from_center <= 3.0 * width
    print(
        f"N={cl.n_leaves}  electron_scale={ELECTRON_SCALE}  cells_across={2*ELECTRON_SCALE:.0f}  "
        f"dt={dt:.4e}  N_STEPS={N_STEPS} (t_final={N_STEPS*dt:.2f})  "
        f"deficit_width={width:.2f}  window cells={int(mask_window.sum())}",
        flush=True,
    )

    print(f"\n=== short-window rate-of-change sweep, {N_STEPS} steps each ===", flush=True)
    results = {}
    for name in CONFIGS:
        results[name] = run_config(alat, cl, name, r_from_center, mask_window, deficit, dt)

    print("\n=== summary: fractional rate of change (per unit time) ===", flush=True)
    base = results["vacuum"]["frac_rate"]
    print(f"vacuum baseline: frac_rate = {base:+.4e}/t", flush=True)
    for name in CONFIGS[1:]:
        r = results[name]
        ratio = abs(r["frac_rate"] / base) if base else float("nan")
        print(f"  {name:>12s}: frac_rate = {r['frac_rate']:+.4e}/t  ({ratio*100:.1f}% of vacuum's rate)", flush=True)

    out = {}
    for name, r in results.items():
        out[f"t_{name}"] = r["times"]
        out[f"deficit_{name}"] = r["deficit"]
    np.savez("exp/2026/09/21/electron_dipole_perturb_restoring/results_charge_test_v4.npz", **out)
    print("\nsaved results_charge_test_v4.npz", flush=True)


if __name__ == "__main__":
    main()
