"""
Criterion calibration probe for the D1=1 Bohr radius setup.

Imposes all ICs analytically (u ~ 1/r, lam ~ 1/r, phi on ring), then runs a
short L0 warmup to let phi propagate throughout the sphere.  After that,
adapts one level at a time with a relax phase between each, reporting
criterion_max percentiles AND the normalised values for the chosen criterion
type so that hi/lo thresholds can be calibrated for scale-invariant splitting.

Run:
    PYTHONPATH=$PWD python3 run/substrate/adaptive/probe_bohr_criterion.py [options]

Criterion types:
    absolute    raw criterion_max              (diagnostic / unit-test use)
    relative_u  criterion_max / |u|            (scale-invariant, recommended)
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")

from equilibrium import lattice as _lat
from equilibrium.substrate.amr import auto_split_criterion
from tst.substrate.adaptive import (
    add_lattice_args,
    add_warmup_args,
    compute_dt,
    find_origin,
)
from tst.substrate.adaptive.adaptive_flux_tube import (
    find_toroidal_ring,
    init_toroidal_winding,
    set_knot,
    set_u_background,
)

D1 = 1.0
R = 20.0

ALPHA_FINE = 1.0 / 137.036
LAM_BOHR = 10.0 * ALPHA_FINE  # ≈ 0.073 — places σ_orbit ≈ 1-α at r~10×ring_r


def _normalised(cmax, u, level, lam, criterion_type, ref=0.0):
    """Compute the value that the chosen criterion actually thresholds on."""
    u_eff = np.maximum(np.abs(u), ref)
    if criterion_type == "relative_u":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(u_eff > 0, cmax / u_eff, np.nan)
    return cmax  # absolute


def criterion_stats(cl, label="", criterion_type="gradient", ref=0.0):
    cmax = np.array(cl.criterion_max)
    lev = np.array(cl.leaf_level, dtype=np.int32)
    u = np.array(cl.leaf_u)
    lam = np.array(cl.leaf_lam, dtype=np.float64)
    mask = np.isfinite(cmax)

    print(f"\n--- criterion stats ({label}) ---")
    print(f"  n_leaves={cl.n_leaves}  finite={mask.sum()}")
    if mask.sum() == 0:
        return

    normed = _normalised(cmax, u, lev, lam, criterion_type, ref)
    nmask = np.isfinite(normed) & mask

    print("  criterion_max  (raw):")
    for p in (50, 75, 90, 95, 99, 100):
        print(f"    p{p:5.1f}: {np.percentile(cmax[mask], p):.4e}")

    if criterion_type != "absolute" and nmask.sum() > 0:
        print(
            f"  criterion_max  (normalised — {criterion_type}, thresholds this value):"
        )
        for p in (50, 75, 90, 95, 99, 100):
            print(f"    p{p:5.1f}: {np.percentile(normed[nmask], p):.4e}")

    levels = np.unique(lev)
    if len(levels) > 1:
        print(f"  per-level breakdown (normalised={criterion_type}):")
        for lv in levels:
            m = mask & (lev == lv)
            if m.sum() == 0:
                continue
            nm = nmask & (lev == lv)
            nval = normed[nm] if nm.sum() > 0 else np.array([np.nan])
            print(
                f"    level {lv}: n={m.sum():5d}  "
                f"raw_med={np.median(cmax[m]):.3e}  "
                f"norm_med={np.median(nval):.3e}  "
                f"norm_p95={np.percentile(nval, 95) if nm.sum() > 0 else float('nan'):.3e}  "
                f"norm_max={np.max(nval) if nm.sum() > 0 else float('nan'):.3e}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_lattice_args(parser, lam=LAM_BOHR)
    add_warmup_args(parser, n_warm=5_000, n_relax=10_000, n_report=5_000)
    parser.add_argument("--max-level", type=int, default=3)
    parser.add_argument(
        "--criterion",
        choices=["absolute", "relative_u"],
        default="relative_u",
        help="Split criterion type (default: relative_u — scale-invariant)",
    )
    # Manual overrides — if omitted, auto_split_criterion derives them after warmup.
    parser.add_argument(
        "--hi",
        type=float,
        default=None,
        help="Split threshold (auto-derived from cluster if omitted)",
    )
    parser.add_argument(
        "--lo", type=float, default=None, help="Coarsen threshold (auto: hi*0.1)"
    )
    parser.add_argument(
        "--ref",
        type=float,
        default=None,
        help="u floor for relative_u (auto: p5 of |u|)",
    )
    parser.add_argument(
        "--split-frac",
        type=float,
        default=0.05,
        help="Target fraction of cells to split per level (default 0.05)",
    )
    parser.add_argument(
        "--memory-budget",
        type=int,
        default=0,
        help="Max leaf count after adapt; 0 = unlimited",
    )
    args = parser.parse_args()

    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    cl = alat[0]

    _lat.build_bcc_sphere(cl, R, D1, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)
    print(
        f"lam_0={args.lam:.5f}  (α_fine={ALPHA_FINE:.5f}, lam_0/α = {args.lam/ALPHA_FINE:.1f}×)"
    )

    info = _lat.DSTInfo()
    info.max_level = args.max_level
    # Disable legacy thresholds — SplitCriterion takes over.
    info.split_hi = 0.0
    info.split_lo = 0.0
    cl.info = info

    # Split criterion is wired after warmup when auto-deriving; pre-wire with
    # manual values if all three are supplied explicitly.
    _manual = args.hi is not None and args.lo is not None and args.ref is not None
    if _manual:
        sc_factory = getattr(_lat.SplitCriterion, args.criterion)
        cl.set_split_criterion(sc_factory(hi=args.hi, lo=args.lo, ref=args.ref))

    origin_j = find_origin(cl)
    set_knot(cl, origin_j, args.lam, profile="1/r")
    set_u_background(cl, origin_j, u_origin=1.0, profile="1/r")

    fu = np.zeros(cl.n_leaves, dtype=np.uint8)
    fu[origin_j] = 1
    cl.set_leaf_fixed_u(fu)

    ring_js, ring_faces = find_toroidal_ring(cl, axis="z")
    if ring_js is None:
        print("ERROR: ring not found")
        sys.exit(1)
    init_toroidal_winding(cl, ring_js, ring_faces, amplitude=1.0, frozen=False)
    cl.set_bc(_lat.BoundaryCondition.absorbing())

    dt = compute_dt(cl, D1=D1)

    u_arr = np.array(cl.leaf_u)
    print(f"lam={args.lam}  max_level={args.max_level}")
    print(
        f"criterion={args.criterion}  split_frac={args.split_frac}"
        f"  memory_budget={args.memory_budget or 'unlimited'}"
    )
    print(f"dt={dt:.3e}  n_warm={args.n_warm}  n_relax={args.n_relax}")
    print(f"u_mean={float(np.mean(u_arr)):.4f}  u_max={float(np.max(u_arr)):.4f}")

    if args.n_warm > 0:
        print(f"\nwarmup {args.n_warm} steps at L0 (phi propagation) …", flush=True)
        for s in range(1, args.n_warm + 1):
            alat.step(0, dt)
            if s % args.n_report == 0:
                u_arr = np.array(cl.leaf_u)
                print(
                    f"  {s}/{args.n_warm}  u_mean={float(np.mean(u_arr)):.4f}",
                    flush=True,
                )

    # Derive or confirm split criterion from post-warmup cluster state.
    if _manual:
        sc = _lat.SplitCriterion.relative_u(hi=args.hi, lo=args.lo, ref=args.ref)
        print(
            f"\ncriterion (manual): hi={args.hi:.4f}  lo={args.lo:.4f}  ref={args.ref:.4f}"
        )
    else:
        sc = auto_split_criterion(
            cl,
            split_frac=args.split_frac,
            memory_budget=args.memory_budget or None,
        )
        print(f"\ncriterion (auto):   hi={sc.hi:.4f}  lo={sc.lo:.4f}  ref={sc.ref:.4f}")
    cl.set_split_criterion(sc)

    ref_display = sc.ref
    criterion_stats(
        cl, f"after L0 warmup  n_leaves={cl.n_leaves}", args.criterion, ref_display
    )
    sys.stdout.flush()

    for lv in range(1, args.max_level + 1):
        print(f"\nadapt → level {lv} …", flush=True)
        alat.adapt(0)
        alat.rebuild(0)
        dt = compute_dt(cl, D1=D1)
        print(
            f"  n_leaves={cl.n_leaves}  dt={dt:.3e}  relaxing {args.n_relax} steps …",
            flush=True,
        )
        for s in range(1, args.n_relax + 1):
            alat.step(0, dt)
            if s % args.n_report == 0:
                u_arr = np.array(cl.leaf_u)
                print(
                    f"  {s}/{args.n_relax}  n_leaves={cl.n_leaves}  "
                    f"u_mean={float(np.mean(u_arr)):.4f}",
                    flush=True,
                )
        criterion_stats(
            cl,
            f"after level {lv} relax  n_leaves={cl.n_leaves}",
            args.criterion,
            ref_display,
        )
        sys.stdout.flush()

    criterion_stats(
        cl,
        f"final  max_level={args.max_level}  n_leaves={cl.n_leaves}",
        args.criterion,
        ref_display,
    )
