"""
v2: static (invariant) lam instead of step_lam_relax.

Philip's correction (2026-09-21): if we want an electron "at rest", lam
shouldn't be a dynamically-relaxing field chasing alpha*|phi| every step --
that's what made v1's result uninterpretable (see project_dipole_perturb_
restoring_force memory: step_lam_relax at alpha=0.5-2.0 was found to drive
sigma down by 15-70% across the WHOLE dipole envelope within ~10 time
units, so the two paired worlds were diverging because they were each
independently rewriting their own propagation speed, not because of any
localized effect near the perturbation).

Instead: lam is set ONCE at IC construction (init_electron_dipole_ic_vecpot's
existing lam_target parameter, proportional to |B|=|curl(phi)|, already a
static, non-degenerate profile local to the structure) and never touched
again -- no step_lam_relax, no step_cross, just plain step()+sponge. This
also fixes the earlier test's real methodological weakness: both paired
worlds (with-bump, without-bump) now share the IDENTICAL static sigma(pos)
field, so any difference between them can ONLY come from the bump itself,
not from two nonlinear lam fields independently drifting apart.

Question: does a static, structure-shaped sigma(pos) profile (a
"refractive index" set by the electron's own field, like a graded-index
fiber) actually confine/reflect a nearby perturbation, or does the
perturbation just disperse through it exactly as in vacuum? Swept across
lam_target = 0 (linear baseline, sigma=1 everywhere) vs 0.5/1.0/2.0.

Includes a PROPER spatial-localization check (radial energy binning at
several times, not one late snapshot) this time -- v1's apparent signal
did not survive that check, so it must be checked here too before trusting
any window-level number.
"""

import sys
import time

sys.path.insert(0, "/home/p/git/eq")

import numpy as np

from equilibrium import lattice as _lat
from tst.substrate.adaptive import compute_dt
from tst.substrate.adaptive.adaptive_flux_tube import init_electron_dipole_ic_vecpot

R_DOMAIN = 30.0
SPH_FRAC = 0.15
SPONGE_FRACTION = 0.15
DAMP_MAX = 0.003
N_THREADS = 8

EPS_U = 0.05
BUMP_WIDTH_FRAC = 0.3
N_STEPS = 3000
REPORT_EVERY = 50

LAM_TARGETS = [0.0, 0.5, 1.0, 2.0]
RADIAL_BINS = np.array([0, 2, 4, 6, 10, 15, 20, 25, 30, 40])
SPATIAL_CHECK_STEPS = [500, 1500, 2450]  # early / mid / near-minimum


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


def radial_profile(de, r_from_bump):
    out = []
    for lo, hi in zip(RADIAL_BINS[:-1], RADIAL_BINS[1:]):
        m = (r_from_bump >= lo) & (r_from_bump < hi)
        out.append(float(de[m].sum()) if m.any() else 0.0)
    return out


def run_lam_target(alat, cl, lam_val, scale, r_axis, bump_center, bump_w, r_bump, mask_bump, dt):
    b_max, r_axis_chk = init_electron_dipole_ic_vecpot(
        cl, R_sph=1.0, scale=scale, B0=1.0, lam_target=lam_val, u_bg=0.0,
        amplitude=1.0, div_clean=True, clean_rtol=1e-8,
    )
    phi_clean = np.array(cl.leaf_phi)
    u_clean = np.array(cl.leaf_u)
    lam_static = np.array(cl.leaf_lam)  # set once, never touched again
    sigma_static = 1.0 / (1.0 + lam_static)
    print(
        f"  lam_target={lam_val}: b_max={b_max:.4f}  lam_static mean(dipole-nonzero)="
        f"{lam_static[lam_static > 0].mean() if (lam_static > 0).any() else 0.0:.4f}  "
        f"max={lam_static.max():.4f}  sigma_min={sigma_static.min():.4f}",
        flush=True,
    )

    bump = EPS_U * np.exp(-0.5 * (r_bump / bump_w) ** 2)
    phi_A, u_A = phi_clean.copy(), u_clean + bump
    phi_B, u_B = phi_clean.copy(), u_clean.copy()

    times, ebump_hist = [], []
    spatial_snaps = {}
    t0 = time.time()
    for step in range(N_STEPS + 1):
        if step % REPORT_EVERY == 0:
            du = u_A - u_B
            dphi = phi_A - phi_B
            de = du**2 + 0.5 * np.sum(dphi**2, axis=0)
            times.append(step * dt)
            ebump_hist.append(float(de[mask_bump].sum()))
        if step in SPATIAL_CHECK_STEPS:
            du = u_A - u_B
            dphi = phi_A - phi_B
            de = du**2 + 0.5 * np.sum(dphi**2, axis=0)
            spatial_snaps[step] = (float(de.sum()), radial_profile(de, r_bump))
        if step < N_STEPS:
            cl.set_leaf_phi(phi_A.copy())
            cl.set_leaf_u(u_A.copy())
            cl.set_leaf_lam(lam_static.copy())
            alat.step(0, dt)
            alat.step_sponge(0)
            phi_A, u_A = np.array(cl.leaf_phi), np.array(cl.leaf_u)

            cl.set_leaf_phi(phi_B.copy())
            cl.set_leaf_u(u_B.copy())
            cl.set_leaf_lam(lam_static.copy())
            alat.step(0, dt)
            alat.step_sponge(0)
            phi_B, u_B = np.array(cl.leaf_phi), np.array(cl.leaf_u)

    times = np.array(times)
    ebump_hist = np.array(ebump_hist)
    i_min = int(np.argmin(ebump_hist))
    print(
        f"  lam_target={lam_val}: E_bump min={ebump_hist[i_min]:.4e} at t={times[i_min]:.2f} "
        f"(step {i_min*REPORT_EVERY}); final={ebump_hist[-1]:.4e} at t={times[-1]:.2f}  ({time.time()-t0:.1f}s)",
        flush=True,
    )
    for step, (total, prof) in spatial_snaps.items():
        near = prof[0] + prof[1]  # r in [0,4)
        far = sum(prof[2:])  # r in [4,40)
        print(
            f"    spatial@step={step:5d}: total_excess={total:.4e}  near(r<4)={near:.4e} "
            f"({100*near/total if total>0 else 0:.2f}%)  far(r>=4)={far:.4e}",
            flush=True,
        )

    return {
        "lam_target": lam_val,
        "times": times,
        "E_bump": ebump_hist,
        "min_val": float(ebump_hist[i_min]),
        "min_t": float(times[i_min]),
        "spatial_snaps": spatial_snaps,
    }


def main():
    alat, cl = build_cluster()
    scale = R_DOMAIN * SPH_FRAC
    dt = compute_dt(cl)

    # geometry (r_axis fixed across lam_target values -- same B0/k_)
    _, r_axis = init_electron_dipole_ic_vecpot(
        cl, R_sph=1.0, scale=scale, B0=1.0, lam_target=0.0, u_bg=0.0, amplitude=1.0, div_clean=True
    )
    pos = np.array(cl.leaf_pos)
    bump_center = np.array([r_axis, 0.0, 0.0])
    bump_w = BUMP_WIDTH_FRAC * scale
    r_bump = np.linalg.norm(pos - bump_center, axis=1)
    mask_bump = r_bump <= 3.0 * bump_w
    print(f"N={cl.n_leaves}  scale={scale:.2f}  r_axis={r_axis:.3f}  dt={dt:.4e}  bump_w={bump_w:.2f}", flush=True)

    print(f"\n=== static-lam sweep, {N_STEPS} steps each, paired bump/no-bump ===", flush=True)
    results = {}
    for lam_val in LAM_TARGETS:
        results[lam_val] = run_lam_target(
            alat, cl, lam_val, scale, r_axis, bump_center, bump_w, r_bump, mask_bump, dt
        )

    print("\n=== summary ===", flush=True)
    base = results[0.0]["min_val"]
    print(f"lam_target=0.0 (vacuum baseline): E_bump min = {base:.4e}", flush=True)
    for lam_val in LAM_TARGETS[1:]:
        r = results[lam_val]
        ratio = r["min_val"] / base if base > 0 else float("nan")
        print(f"  lam_target={lam_val}: E_bump min = {r['min_val']:.4e}  (x{ratio:.2f} vs baseline)", flush=True)

    out = {}
    for lam_val, r in results.items():
        key = f"lam{lam_val:g}"
        out[f"t_{key}"] = r["times"]
        out[f"Ebump_{key}"] = r["E_bump"]
    np.savez("exp/2026/09/21/electron_dipole_perturb_restoring/results_static_lam.npz", **out)
    print("\nsaved results_static_lam.npz", flush=True)


if __name__ == "__main__":
    main()
