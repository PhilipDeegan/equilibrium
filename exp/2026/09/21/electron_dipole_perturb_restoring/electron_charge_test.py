"""
Does the double-ring dipole electron IC support a genuine charge (u-deficit)
signature, and does its own field (phi structure and/or static lam) help
that deficit persist rather than just dispersing?

Motivation (2026-09-21 conversation): the dipole IC's u is exactly flat
(confirmed: min=max=std=0.0) by construction -- it carries zero charge
signature, only the phi/magnetic-moment content. doc/dst.md's own text says
proton=high-u open knot (charge), and this project's convention documents
electron as the opposite, a u-DEFICIT -- never implemented in any tested
IC (see project_electron_u_phi_convention_conflict). Rather than PINNING a
u source cell (the yesterday's Coulomb test's method, `fixed_u=True` --
flagged by feedback_pinning_signals_aphysicality as a red flag, not a real
demonstration), this seeds a FREE, unpinned local u-deficit and just lets
it evolve. Prior result this session: an unpinned local u perturbation in
vacuum (the "bump" in perturb_restoring_v1/v2) disperses essentially
completely within a few time units under plain linear dynamics -- so the
default expectation here is that a free deficit ALSO just disperses,
unless something about the dipole's own structure (phi circulation and/or
a nonzero static lam, which today's v2 test showed gives real partial
wave confinement) changes that.

Four configs, same free u-deficit (centered at the dipole's own origin,
not pinned) on top of each:
  vacuum       : no dipole phi at all, lam=0 -- pure control
  dipole_lam0  : double-ring dipole phi present, lam=0 (pure-phi null hyp.)
  dipole_lam1  : dipole phi + static lam_target=1.0 (today's v2 partial-
                 confinement regime)
  dipole_lam2  : dipole phi + static lam_target=2.0 (stronger confinement)

Diagnostic: total |deficit| = sum(u_bg - u) within a fixed window around
the origin, tracked over time (same "read the minimum before boundary-
reflection contamination returns" discipline as v2), plus a snapshot
radial profile of u(r) at a few checkpoints to see whether any config
develops something resembling a persistent, spatially-structured profile
(cf. the classical q(1/r - 1/R) sponge-bounded solution from
project_dst_proton_translation-era Coulomb work) versus just flattening
back toward u_bg.
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

DEFICIT_AMP = 0.2  # magnitude of the u-deficit, u = u_bg - amp*exp(...)
DEFICIT_WIDTH_FRAC = 0.3  # fraction of the dipole's own physical scale
N_STEPS = 3000
REPORT_EVERY = 50
PROFILE_STEPS = [0, 500, 1500, 2450]

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
    return alat, cl


def radial_u_profile(u, r_from_center, bins):
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (r_from_center >= lo) & (r_from_center < hi)
        out.append(float(u[m].mean()) if m.any() else float("nan"))
    return out


def run_config(alat, cl, name, scale, r_from_center, mask_window, deficit, dt):
    if name == "vacuum":
        phi0 = np.zeros((14, cl.n_leaves))
        lam0 = np.zeros(cl.n_leaves)
    else:
        lam_target = {"dipole_lam0": 0.0, "dipole_lam1": 1.0, "dipole_lam2": 2.0}[name]
        init_electron_dipole_ic_vecpot(
            cl, R_sph=1.0, scale=scale, B0=1.0, lam_target=lam_target, u_bg=0.0,
            amplitude=1.0, div_clean=True, clean_rtol=1e-8,
        )
        phi0 = np.array(cl.leaf_phi)
        lam0 = np.array(cl.leaf_lam)

    u0 = np.zeros(cl.n_leaves) - deficit  # u_bg=0, deficit is positive-valued

    phi, u, lam = phi0.copy(), u0.copy(), lam0.copy()

    times, deficit_hist = [], []
    profiles = {}
    bins = np.array([0, 1, 2, 3, 4, 6, 8, 10, 15, 20])
    t0 = time.time()
    for step in range(N_STEPS + 1):
        if step % REPORT_EVERY == 0:
            times.append(step * dt)
            deficit_hist.append(float((-u[mask_window]).sum()))
        if step in PROFILE_STEPS:
            profiles[step] = radial_u_profile(u, r_from_center, bins)
        if step < N_STEPS:
            cl.set_leaf_phi(phi.copy())
            cl.set_leaf_u(u.copy())
            cl.set_leaf_lam(lam.copy())  # static -- never modified after IC construction
            alat.step(0, dt)
            alat.step_sponge(0)
            phi, u = np.array(cl.leaf_phi), np.array(cl.leaf_u)

    times = np.array(times)
    deficit_hist = np.array(deficit_hist)
    i_min = int(np.argmin(np.abs(deficit_hist - deficit_hist[-1])))  # for reference only
    print(
        f"[{name:>12s}] deficit0={deficit_hist[0]:.4e} deficit_final={deficit_hist[-1]:.4e} "
        f"retained={deficit_hist[-1]/deficit_hist[0] if deficit_hist[0] else float('nan'):.4f}  "
        f"({time.time()-t0:.1f}s)",
        flush=True,
    )
    for step in PROFILE_STEPS:
        prof = profiles[step]
        prof_str = "  ".join(f"{v:+.4f}" for v in prof)
        print(f"    u(r) profile @step={step:5d} (bins 0-1,1-2,...15-20): {prof_str}", flush=True)

    return {"name": name, "times": times, "deficit": deficit_hist, "profiles": profiles, "bins": bins}


def main():
    alat, cl = build_cluster()
    scale = R_DOMAIN * SPH_FRAC
    dt = compute_dt(cl)
    pos = np.array(cl.leaf_pos)
    r_from_center = np.linalg.norm(pos, axis=1)
    width = DEFICIT_WIDTH_FRAC * scale
    deficit = DEFICIT_AMP * np.exp(-0.5 * (r_from_center / width) ** 2)
    mask_window = r_from_center <= 3.0 * width
    print(
        f"N={cl.n_leaves}  scale={scale:.2f}  dt={dt:.4e}  deficit_amp={DEFICIT_AMP}  "
        f"width={width:.2f}  window cells={int(mask_window.sum())}",
        flush=True,
    )

    print(f"\n=== charge-persistence sweep, {N_STEPS} steps each ===", flush=True)
    results = {}
    for name in CONFIGS:
        results[name] = run_config(alat, cl, name, scale, r_from_center, mask_window, deficit, dt)

    out = {}
    for name, r in results.items():
        out[f"t_{name}"] = r["times"]
        out[f"deficit_{name}"] = r["deficit"]
    np.savez("exp/2026/09/21/electron_dipole_perturb_restoring/results_charge_test.npz", **out)
    print("\nsaved results_charge_test.npz", flush=True)


if __name__ == "__main__":
    main()
