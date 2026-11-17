"""
Bohr orbit stability probe.

Imposes a u-deficit shell at a target orbit radius and tracks the radial
centre-of-mass of the deficit over time.

Physical lam calibration
------------------------
The fine structure constant α ≈ 1/137 is the electron's orbital speed in units of c.
In DST, orbital speed = 1 - σ_orbit, so:

    σ_orbit = 1 - α   →   lam_orbit = α / (1 - α) ≈ α ≈ 0.0073

The BCC ring sits at r_ring ≈ D1 × 4/√3 ≈ 1.394 lattice units from the origin.
For a 1/r lam profile, if the orbit is at r_orbit:

    lam_orbit = lam_0 × (r_ring / r_orbit)

Setting lam_orbit = α and r_orbit ≈ 10 × r_ring gives lam_0 ≈ 10 × α ≈ 0.073.

This is much less than the old chi=0.9 (lam=9) simulation value, which was chosen
to make the gradient visible in small domains, not for physical accuracy.

At lam_0 = 0.073:
    σ at ring  ≈ 0.932  (7% time dilation at the proton surface)
    σ at orbit ≈ 0.993  ≈ 1 - α  (Bohr speed at the orbit)

The orbit radius at this lam is unknown — the --r-orbit parameter must be swept.

If the orbit is stable: moat COM stays near r_orbit.
If not: COM drifts inward/outward (also important physics).
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")

from equilibrium import lattice as _lat
from tst.substrate.adaptive import (
    add_amr_args,
    add_criterion_args,
    add_lattice_args,
    add_warmup_args,
    apply_split_criterion,
    compute_dt,
    find_origin,
    get_r_physical,
    setup_knot_and_origin,
)
from tst.substrate.adaptive.adaptive_flux_tube import (
    find_toroidal_ring,
    init_toroidal_winding,
)

D1 = 1.0
R = 20.0  # larger domain — orbit radius at low lam is uncertain

ALPHA_FINE = 1.0 / 137.036
# 10 × α puts σ_orbit ≈ 1 - α at r ~ 10 × ring_r (see module docstring)
LAM_BOHR = 10.0 * ALPHA_FINE  # ≈ 0.073

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_lattice_args(parser, lam=LAM_BOHR)
    add_amr_args(parser)
    add_warmup_args(parser, n_warm=20_000)
    add_criterion_args(parser)
    parser.add_argument(
        "--r-orbit",
        type=float,
        default=7.0,
        help="moat shell centre radius — MUST BE SWEPT; orbit radius at low lam is unknown",
    )
    parser.add_argument("--dr-moat", type=float, default=2.0, help="moat half-width")
    parser.add_argument(
        "--u-def",
        type=float,
        default=-0.1,
        help="moat u amplitude (smaller at low lam — moat is shallow)",
    )
    parser.add_argument("--n-steps", type=int, default=100_000)
    parser.add_argument("--n-track", type=int, default=2_000)
    args = parser.parse_args()

    R_ORBIT = args.r_orbit
    DR_MOAT = args.dr_moat
    U_DEF = args.u_def
    N_WARM = args.n_warm
    N_STEPS = args.n_steps
    N_TRACK = args.n_track

    alat = _lat.AdaptiveLattice()
    alat.set_n_threads(6)
    alat.add_cluster()
    cl = alat[0]

    _lat.build_bcc_sphere(cl, R, D1, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)

    info = _lat.DSTInfo()
    info.split_hi = 0.0
    info.split_lo = 0.0
    info.max_level = args.max_level
    cl.info = info

    origin_j = find_origin(cl)
    setup_knot_and_origin(cl, origin_j, args.lam)

    lam_arr = np.array(cl.leaf_lam, dtype=np.float64)
    sigma_origin = 1.0 / (1.0 + float(lam_arr[origin_j]))
    lam_ring_mean = float(np.mean(lam_arr[lam_arr > 0]))
    print(
        f"lam_0={args.lam:.4f}  σ_origin={sigma_origin:.4f}  lam_ring_mean={lam_ring_mean:.4f}"
    )
    print(f"σ_orbit target: {1.0 - ALPHA_FINE:.4f}  (α_fine={ALPHA_FINE:.5f})")

    ring_js, ring_faces = find_toroidal_ring(cl, axis="z")
    if ring_js is None:
        print("ERROR: ring not found")
        sys.exit(1)
    init_toroidal_winding(cl, ring_js, ring_faces, amplitude=1.0, frozen=False)
    cl.set_bc(_lat.BoundaryCondition.absorbing())

    dt = compute_dt(cl, D1=D1)

    if N_WARM > 0:
        print(f"warmup {N_WARM} steps at L0  dt={dt:.3e} …", flush=True)
        for s in range(1, N_WARM + 1):
            alat.step(0, dt)
            if s % 5000 == 0:
                u_arr = np.array(cl.leaf_u)
                print(
                    f"  {s}/{N_WARM}  n_leaves={cl.n_leaves}  u_mean={float(np.mean(u_arr)):.4f}  u_max={float(np.max(u_arr)):.4f}",
                    flush=True,
                )
        print("warmup done", flush=True)

    if args.max_level > 0:
        apply_split_criterion(cl, args, level=0)
        for lv in range(1, args.max_level + 1):
            print(f"adapt → level {lv} …", flush=True)
            alat.adapt(0)
            dt = compute_dt(cl, D1=D1)
            print(
                f"  n_leaves={cl.n_leaves}  dt={dt:.3e}  relaxing {args.n_relax} steps …",
                flush=True,
            )
            for s in range(1, args.n_relax + 1):
                alat.step(0, dt)
                if s % 5000 == 0:
                    u_arr = np.array(cl.leaf_u)
                    print(
                        f"  {s}/{args.n_relax}  n_leaves={cl.n_leaves}  u_mean={float(np.mean(u_arr)):.4f}",
                        flush=True,
                    )
            print(f"  level {lv} done", flush=True)
            if lv < args.max_level:
                apply_split_criterion(cl, args, level=lv)

    r_phys = get_r_physical(cl)
    origin_j = find_origin(cl)
    if args.max_level > 0:
        ring_js, _ = find_toroidal_ring(cl, axis="z")
        if ring_js is None:
            ring_js = []

    # Report σ at candidate orbit radius from the actual post-warmup lam field
    lam_arr = np.array(cl.leaf_lam, dtype=np.float64)
    orbit_mask = np.abs(r_phys - R_ORBIT) < 0.5
    if orbit_mask.any():
        sigma_orbit = float(np.mean(1.0 / (1.0 + lam_arr[orbit_mask])))
        print(f"σ at r≈{R_ORBIT}: {sigma_orbit:.5f}  (target 1-α = {1-ALPHA_FINE:.5f})")

    exclude = set(ring_js) | {origin_j}
    shell_js = [
        j
        for j in range(cl.n_leaves)
        if abs(float(r_phys[j]) - R_ORBIT) < DR_MOAT and j not in exclude
    ]

    if not shell_js:
        print(f"ERROR: no cells found in shell r={R_ORBIT}±{DR_MOAT}")
        sys.exit(1)

    u_bg = np.array(cl.leaf_u).copy()
    u_arr = u_bg.copy()
    for j in shell_js:
        u_arr[j] += U_DEF
    cl.set_leaf_u(u_arr)
    print(f"R={R}  r_orbit={R_ORBIT}  DR={DR_MOAT}  shell_cells={len(shell_js)}")
    print(
        f"dt={dt:.3e}  N_steps={N_STEPS}  U_DEF={U_DEF}  u_bg_mean={float(np.mean(u_bg)):.4f}"
    )
    print(
        f"\n{'step':>8}  {'r_com':>8}  {'drift':>8}  {'moat_depth':>12}  {'n_cells':>8}"
    )
    print("  " + "-" * 55)

    def _moat_com(threshold=U_DEF * 0.3):
        u_now = np.array(cl.leaf_u)
        du = u_now - u_bg
        mask = (du < threshold) & (~np.isin(np.arange(cl.n_leaves), list(exclude)))
        if not np.any(mask):
            return float("nan"), 0, float("nan")
        weights = np.abs(du[mask])
        r_com = float(np.sum(r_phys[mask] * weights) / np.sum(weights))
        depth = float(np.min(du[mask]))
        return r_com, int(np.sum(mask)), depth

    r_com0, n0, depth0 = _moat_com()
    print(f"{'0':>8}  {r_com0:8.3f}  {'---':>8}  {depth0:12.5f}  {n0:>8}")

    for step in range(1, N_STEPS + 1):
        alat.step(0, dt)
        if step % N_TRACK == 0:
            r_com, n, depth = _moat_com()
            drift = r_com - r_com0 if not np.isnan(r_com) else float("nan")
            print(
                f"{step:>8}  {r_com:8.3f}  {drift:+8.3f}  {depth:12.5f}  {n:>8}",
                flush=True,
            )
            if np.isnan(r_com):
                print("  moat fully dissipated")
                break
