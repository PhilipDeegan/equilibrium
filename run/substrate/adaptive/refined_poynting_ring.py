"""
Refined Poynting-ring probe — cross-check for the odd27 deep-telescope AMR
investigation (doc/mems/project_odd27_amr_junction_leak.md, CLAWD status
2026-07-20). That investigation characterised a bounded, persistent H
oscillation (not monotonic runaway) using the real spheromak/neutron-decay
IC. This script asks the same question with a genuinely radiating,
independently-calibrated source instead: does the barrier-gated oscillating
ring (doc/mems/project_poynting_dipole_calibration.md — uniform-mesh polar
falloff slope -2.44, close to the classical dipole -2) show the same bounded
oscillation in H when a shell around it is refined to --max-level, and does
the AMR telescope degrade the measured far-field falloff exponent?

Builds on run/substrate/adaptive/probe_poynting_dipole_compare.py (uniform
L0 baseline + falloff-slope fit machinery, copied here) and
refined_neutron_decay.py (telescope construction via
refine_region_to_convergence, TimeStepper.by_level + AdaptiveLatticeEvolver
stepping). Ring-protection follows test_adaptive_lattice.py's
TestByLevelStability/TestByLevelMultiLevelStability pattern: the ring's own
tiles must NEVER split (init_toroidal_winding's omega=2*sin(pi/N) formula is
tuned for single-leaf L0 ring cells; splitting a ring tile breaks
find_toroidal_ring's assumed 8-tile connectivity and corrupts the winding).

Refined region: a shell around the ring's own radius (|r - ring_r| < width,
measured from ring_centre), NOT a solid ball from the ring centre — a solid
ball would mostly refine the empty space inside the torus hole, missing the
ring's own near field entirely. Width shrinks by SPLIT_RATIO per level (same
cell-count-conserving telescope shape as refined_neutron_decay), so deep
levels concentrate resolution right at the ring while shallower levels
extend further into the near/far field.

Usage:
  PYTHONPATH=$PWD python3 -u run/substrate/adaptive/refined_poynting_ring.py \\
      --R 50 --hops 4 --max-level 4 --n-steps 8000 --m0 1e-4 --h-report 200
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium import lattice as _lat
from equilibrium.substrate.mhd import _SQ_AXIS_PAIRS, SQ_PLAQUETTES
from tst.substrate.adaptive import refine_region_to_convergence
from tst.substrate.adaptive.adaptive_flux_tube import (
    find_toroidal_ring,
    init_toroidal_winding,
    leaf_nb_j,
    winding_number,
)

SPLIT_RATIO = 3.0  # Odd27 refinement ratio


def _compute_dt_l0(cl, w2_sum=8 * 9.0 + 6 * 1.0, D1=1.0):
    """L0 CFL dt -- TimeStepper.by_level subcycles finer cells internally,
    so the caller only ever needs the coarse-cell dt (see
    sweep_spheromak_decay_cross_scale._compute_dt_l0, same formula)."""
    fu_arr = np.array(cl.leaf_fixed_u).astype(bool)
    lam_arr = np.array(cl.leaf_lam, dtype=np.float64)
    free = ~fu_arr
    sigma_max = float(np.max(1.0 / (1.0 + lam_arr[free])))
    return 0.4 * D1 / (sigma_max * w2_sum)


def _plaquette_vectorized(phi_T, nb_j, faces):
    """Vectorized eq. of mhd._plaquette_curl over ALL cells at once."""
    N = phi_T.shape[0]
    c0 = np.arange(N)
    ka, kb, kc, kd = faces
    c1 = nb_j[c0, ka]
    valid = c1 >= 0
    c1s = np.where(valid, c1, 0)
    c2 = nb_j[c1s, kb]
    valid &= c2 >= 0
    c2s = np.where(c2 >= 0, c2, 0)
    c3 = nb_j[c2s, kc]
    valid &= c3 >= 0
    c3s = np.where(c3 >= 0, c3, 0)
    close = nb_j[c3s, kd]
    valid &= close == c0
    term = phi_T[c0, ka] + phi_T[c1s, kb] + phi_T[c2s, kc] + phi_T[c3s, kd]
    return np.where(valid, term, 0.0)


def b_vec_vectorized(phi_T, nb_j):
    Bx = _plaquette_vectorized(phi_T, nb_j, SQ_PLAQUETTES["x"])
    By = _plaquette_vectorized(phi_T, nb_j, SQ_PLAQUETTES["y"])
    Bz = _plaquette_vectorized(phi_T, nb_j, SQ_PLAQUETTES["z"])
    return np.stack([Bx, By, Bz], axis=1)


def face_vec_vectorized(dphi):
    return np.stack([dphi[:, fp] - dphi[:, fm] for fp, fm in _SQ_AXIS_PAIRS], axis=1)


def fit_radial_slope(
    r_from_centre,
    Smag,
    mask,
    r_min_fit,
    r_max_fit,
    percentile=99.0,
    n_bins=16,
    min_per_bin=5,
):
    """log-log slope of the percentile envelope of |S| vs r, in-mask,
    in-range -- copy of probe_poynting_dipole_compare.py's nested _fit_slope,
    lifted to module level so it can be reused here without re-running the
    barrier-gate cross-scale on the uniform baseline."""
    m = mask & (r_from_centre >= r_min_fit) & (r_from_centre <= r_max_fit) & (Smag > 0)
    n_total = int(m.sum())
    if n_total < min_per_bin * 3:
        return None, None, n_total, None
    r_m, s_m = r_from_centre[m], Smag[m]
    edges = np.logspace(np.log10(r_min_fit), np.log10(r_max_fit), n_bins + 1)
    centres, envelope = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (r_m >= lo) & (r_m < hi)
        if in_bin.sum() < min_per_bin:
            continue
        centres.append(np.sqrt(lo * hi))
        envelope.append(np.percentile(s_m[in_bin], percentile))
    if len(centres) < 3:
        return None, None, n_total, None
    centres, envelope = np.array(centres), np.array(envelope)
    lr, ls = np.log(centres), np.log(envelope)
    A = np.vstack([lr, np.ones_like(lr)]).T
    slope, intercept = np.linalg.lstsq(A, ls, rcond=None)[0]
    return float(slope), float(intercept), n_total, (centres, envelope)


parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("--R", type=float, default=50.0)
parser.add_argument("--hops", type=int, default=4)
parser.add_argument("--amplitude", type=float, default=1.0)
parser.add_argument("--lam-ring", type=float, default=1.0)
parser.add_argument("--cross-scale", type=float, default=0.01)
parser.add_argument("--m0", type=float, default=1e-4)
parser.add_argument("--axis", default="z")
parser.add_argument("--n-steps", type=int, default=8000)
parser.add_argument("--sponge", type=float, default=0.15)
parser.add_argument("--damp-max", type=float, default=0.003)
parser.add_argument("--threads", type=int, default=10)
parser.add_argument("--sector-half-angle", type=float, default=20.0)
parser.add_argument("--r-min-fit", type=float, default=None, help="default: 1.5*ring_r")
parser.add_argument("--r-max-fit", type=float, default=None, help="default: 0.8*R")
parser.add_argument(
    "--max-level",
    type=int,
    default=0,
    help="deepest refinement level for the shell hugging the ring (0 = uniform L0, "
    "reproduces probe_poynting_dipole_compare.py exactly)",
)
parser.add_argument(
    "--shell-width0",
    type=float,
    default=None,
    help="L1 shell half-width around ring_r, in sim-units (default: 2*ring_r); "
    "shrinks by SPLIT_RATIO per level after that",
)
parser.add_argument(
    "--h-report",
    type=int,
    default=200,
    help="print an H (=sum(u^2)+0.5*sum(phi^2)) trajectory line every this many steps",
)
parser.add_argument("--plot-out", type=str, default="tmp/refined_poynting_ring.png")
parser.add_argument(
    "--max-cells",
    type=int,
    default=2_000_000,
    help="abort after telescope construction if N_total exceeds this (safety cap)",
)


def _build_shell_telescope(
    alat, cl, ring_centre, ring_r, ring_js_fn, shell_width0, max_level
):
    """Refine a shell |r - ring_r| < width around the ring (never the ring
    tiles themselves, re-resolved + protected every round) down to
    max_level, width shrinking by SPLIT_RATIO**3 per level.

    refined_neutron_decay._build_telescope shrinks a SOLID BALL's radius by
    SPLIT_RATIO per level -- volume ~ radius**3 shrinks 27x/level, exactly
    cancelling Odd27's 27x/level density growth, so added cell count per
    level stays ~constant. A thin SHELL's volume scales linearly with width
    (~surface_area * width, surface_area roughly fixed), so shrinking width
    by only SPLIT_RATIO (not SPLIT_RATIO**3) leaves a net ~9x/level
    UNCANCELLED growth -- confirmed empirically (max_level=4 at R=20/hops=2
    blew up to 24.2M cells / hung past 300s before this fix). Shrinking by
    SPLIT_RATIO**3 restores the same cancellation as the ball case."""
    cl.info.split_hi = 0.5
    cl.info.split_lo = 0.0
    for lvl in range(1, max_level + 1):
        width = shell_width0 / (SPLIT_RATIO ** (3 * (lvl - 1)))

        def _region(cl, width=width):
            pos = np.array(cl.leaf_pos)
            r = np.linalg.norm(pos - ring_centre, axis=1)
            return np.abs(r - ring_r) < width

        refine_region_to_convergence(
            alat, cl, _region, stage_max_level=lvl, protected_js_fn=ring_js_fn
        )
    cl.set_criterion_max(np.full(cl.n_leaves, np.nan))
    cl.clear_split_criterion()
    cl.info = _lat.DSTInfo()


def main():
    args = parser.parse_args()
    ax_idx = {"x": 0, "y": 1, "z": 2}[args.axis]
    perp = [i for i in range(3) if i != ax_idx]

    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    cl = alat[0]
    _lat.build_bcc_sphere(cl, args.R, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)
    alat.set_n_threads(args.threads)

    r_all = np.linalg.norm(np.array(cl.leaf_pos), axis=1)
    r_max = float(r_all.max())
    r_inner = r_max * (1.0 - args.sponge)
    cl.set_bc(
        _lat.BoundaryCondition.quadratic_sponge(
            target=0.0, damp_max=args.damp_max, r_all=r_all, r_inner=r_inner
        )
    )

    def ring_js_fn(cl):
        js, _ = find_toroidal_ring(cl, axis=args.axis, mode="square", hops=args.hops)
        return js

    ring_js, ring_faces = find_toroidal_ring(
        cl, axis=args.axis, mode="square", hops=args.hops
    )
    if ring_js is None:
        print("ERROR: no toroidal ring found")
        sys.exit(1)
    w = winding_number(cl, ring_js, axis=args.axis)
    print(f"winding={w:+.4f}")
    omega = init_toroidal_winding(
        cl, ring_js, ring_faces, amplitude=args.amplitude, phi_only=True
    )
    pos = np.array(cl.leaf_pos)
    ring_pos = pos[ring_js]
    ring_centre = ring_pos.mean(axis=0)
    ring_r = float(np.linalg.norm(ring_pos - ring_centre, axis=1).mean())

    lam = np.array(cl.leaf_lam)
    for j in ring_js:
        lam[j] = args.lam_ring
    cl.set_leaf_lam(lam)

    dt = _compute_dt_l0(cl)
    period = 2.0 * np.pi / omega if omega > 0 else None
    print(
        f"N0={cl.n_leaves}  R={args.R}  hops={args.hops}  ring_r={ring_r:.3f}  "
        f"omega={omega:.4f}  period={period}  dt={dt:.4e}  m0={args.m0:.2e}  "
        f"max_level={args.max_level}",
        flush=True,
    )

    if args.max_level > 0:
        shell_width0 = (
            args.shell_width0 if args.shell_width0 is not None else 2.0 * ring_r
        )
        _build_shell_telescope(
            alat, cl, ring_centre, ring_r, ring_js_fn, shell_width0, args.max_level
        )
        r_all = np.linalg.norm(np.array(cl.leaf_pos), axis=1)
        r_max = float(r_all.max())
        r_inner = r_max * (1.0 - args.sponge)
        cl.set_bc(
            _lat.BoundaryCondition.quadratic_sponge(
                target=0.0, damp_max=args.damp_max, r_all=r_all, r_inner=r_inner
            )
        )
        ring_js, _ = find_toroidal_ring(
            cl, axis=args.axis, mode="square", hops=args.hops
        )
        lvl_arr = np.array(cl.leaf_level)
        print(
            f"telescope built: shell_width0={shell_width0:.3f}  N_total={cl.n_leaves}  "
            f"level_max={int(lvl_arr.max())}  n_at_max_level={int((lvl_arr == args.max_level).sum())}",
            flush=True,
        )
        if cl.n_leaves > args.max_cells:
            print(
                f"ABORT: N_total={cl.n_leaves} exceeds --max-cells={args.max_cells} -- "
                f"telescope geometry too large for this max_level/shell_width0, refusing "
                f"to run the step loop. Reduce --shell-width0 or --max-level.",
                flush=True,
            )
            sys.exit(1)

    pos = np.array(cl.leaf_pos)
    r_from_centre = np.linalg.norm(pos - ring_centre, axis=1)
    nb_j = leaf_nb_j(cl)

    r_min_fit = args.r_min_fit if args.r_min_fit is not None else 1.5 * ring_r
    r_max_fit = args.r_max_fit if args.r_max_fit is not None else 0.8 * args.R

    rel = pos - ring_centre
    ax_comp = rel[:, ax_idx]
    perp_comp = np.linalg.norm(rel[:, perp], axis=1)
    theta = np.degrees(np.arctan2(perp_comp, np.abs(ax_comp)))
    half = args.sector_half_angle
    equatorial = theta > (90.0 - half)
    polar = theta < half

    has_fine = int(np.max(np.array(cl.leaf_level))) > 0
    ts = _lat.TimeStepper.by_level(dt) if has_fine else _lat.TimeStepper.fixed(dt)
    evolver = _lat.AdaptiveLatticeEvolver(alat)

    def _H():
        u = np.array(cl.leaf_u)
        phi = np.array(cl.leaf_phi)
        return float(np.sum(u**2) + 0.5 * np.sum(phi**2))

    H0 = _H()
    print(f"H0={H0:.6e}", flush=True)
    h_hist = [(0, H0, 1.0)]

    for step in range(1, args.n_steps + 1):
        evolver.step(ts)
        alat.step_sponge(0)
        if args.cross_scale > 0.0:
            alat.step_cross(0, dt, args.cross_scale, gate="barrier", m0=args.m0)
        if step % args.h_report == 0 or step == args.n_steps:
            h_now = _H()
            ratio = h_now / H0 if H0 > 0 else float("nan")
            h_hist.append((step, h_now, ratio))
            print(f"  step={step:6d}  H={h_now:.6e}  H/H0={ratio:.4f}", flush=True)

    print(
        f"\nreached step={args.n_steps}, computing E/B/S over full domain...",
        flush=True,
    )

    phi_before = np.array(cl.leaf_phi).T.copy()
    alat.step_cross(0, dt, args.cross_scale, gate="barrier", m0=args.m0)
    phi_after = np.array(cl.leaf_phi).T
    dphi_cross = phi_after - phi_before

    E3 = -face_vec_vectorized(dphi_cross) / dt
    B3 = b_vec_vectorized(phi_after, nb_j)
    S3 = np.cross(E3, B3)
    Smag = np.linalg.norm(S3, axis=1)

    slope_eq, intercept_eq, n_eq, env_eq = fit_radial_slope(
        r_from_centre, Smag, equatorial, r_min_fit, r_max_fit
    )
    slope_pol, intercept_pol, n_pol, env_pol = fit_radial_slope(
        r_from_centre, Smag, polar, r_min_fit, r_max_fit
    )

    print()
    print("=== Radial |S| falloff (log-log slope; classical far-zone dipole = -2) ===")
    print(
        "    (uniform-L0 calibrated reference: polar=-2.44, equatorial=-6.19, "
        "see doc/mems/project_poynting_dipole_calibration.md)"
    )
    if slope_eq is not None:
        print(f"equatorial (theta>{90-half:.0f} deg, n={n_eq}): slope={slope_eq:+.3f}")
    else:
        print(f"equatorial: insufficient nonzero cells in fit range (n={n_eq})")
    if slope_pol is not None:
        print(f"polar      (theta<{half:.0f} deg, n={n_pol}): slope={slope_pol:+.3f}")
    else:
        print(f"polar: insufficient nonzero cells in fit range (n={n_pol})")

    print()
    print("=== H trajectory summary ===")
    ratios = [r for _, _, r in h_hist]
    print(f"H/H0: min={min(ratios):.4f}  max={max(ratios):.4f}  final={ratios[-1]:.4f}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"[plot] SKIPPED -- matplotlib import failed ({e})", flush=True)
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    steps_h = [s for s, _, _ in h_hist]
    ratios_h = [r for _, _, r in h_hist]
    ax.plot(steps_h, ratios_h, "o-")
    ax.set_xlabel("step")
    ax.set_ylabel("H/H0")
    ax.set_title(f"H trajectory  max_level={args.max_level}")

    ax = axes[1]
    for mask, label, color in [
        (equatorial, "equatorial", "C0"),
        (polar, "polar", "C1"),
    ]:
        m = mask & (Smag > 0) & (r_from_centre > ring_r)
        ax.scatter(
            r_from_centre[m],
            Smag[m],
            s=3,
            alpha=0.08,
            color=color,
            label=f"_{label} cells",
        )
    for env, label, color in [
        (env_eq, "equatorial p99 envelope", "C0"),
        (env_pol, "polar p99 envelope", "C1"),
    ]:
        if env is not None:
            ax.plot(
                env[0], env[1], "o", color=color, ms=6, mec="k", mew=0.5, label=label
            )
    r_ref = np.array([r_min_fit, r_max_fit])
    if slope_eq is not None:
        ax.plot(
            r_ref,
            np.exp(intercept_eq) * r_ref**slope_eq,
            "C0-",
            lw=2,
            label=f"equatorial fit slope={slope_eq:+.2f}",
        )
    if slope_pol is not None:
        ax.plot(
            r_ref,
            np.exp(intercept_pol) * r_ref**slope_pol,
            "C1-",
            lw=2,
            label=f"polar fit slope={slope_pol:+.2f}",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("r (cells, from ring centre)")
    ax.set_ylabel("|S|")
    ax.set_title(f"|S| falloff  R={args.R} hops={args.hops} max_level={args.max_level}")
    ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(args.plot_out, dpi=130, bbox_inches="tight")
    print(f"\n[plot] {args.plot_out}", flush=True)


if __name__ == "__main__":
    main()
