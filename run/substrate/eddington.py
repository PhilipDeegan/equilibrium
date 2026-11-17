"""
current status (2026-05-26)

Both point and ring source tests confirm DST produces FOCUS (correct Eddington
direction) at moderate-to-late times.  Early steps show defocus because fast
high-rho paths (avoiding the star) arrive first; once all paths arrive the
rho-centroid with star is consistently lower than baseline, and total u at the
detector is higher (gravitational lensing amplification).

The sign is correct.  Magnitude comparison (factor 2 vs 4 vs Newton) requires
a cleaner geometry and a reference calculation — not yet done.

Key lessons from failed approaches:
  - u-field moat drift is the wrong probe (moat = electron, not photon)
  - wave equations are approximations; DST has no wave equation by design
  - light in DST = pressure finding path of least resistance through sigma field
  - double-slit "interference" = preference for least action, no cancellation

options:
  - AMR around the star — refine the sigma gradient region for a sharper profile
  - Vary lam0 and R_star — measure how focus magnitude scales
  - Vary impact parameter b — deflection angle vs distance curve
  - Larger R with sponge BC — push boundary far from measurement zone
  - Magnitude calibration — compare Δrho/rho_baseline against GR prediction (4GM/c²b)


---

Eddington pressure-path test — DST least-resistance geometry.

The substrate is under pressure everywhere.  A localised source (elevated u)
releases pressure that propagates along paths of least resistance, determined
by the sigma = 1/(1+lam) throughput field.  Near a star (lam > 0) throughput
is lower; in vacuum (lam = 0) throughput is 1.

Two test cases
--------------
  point  : Gaussian u source at (−d, 0, 0) — isotropic pressure release
  ring   : annular u source at x=−d, rho=b  — pressure released on a ring
           at impact parameter b, mimicking a ray bundle grazing the star

Measurement
-----------
After N steps, the radial profile of u at x > x_detect is histogrammed as a
function of rho = sqrt(y²+z²).  The rho-centroid of that distribution is
compared between the star and no-star (baseline) runs:

  rho_centroid decreases  →  pressure focused toward axis  →  star attracts
  rho_centroid increases  →  pressure defocused             →  star repels
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")

from equilibrium import lattice as _lat
from equilibrium.substrate.discrete import LAM_MAX
from tst.substrate.adaptive import compute_dt

D1 = 1.0


# ---------------------------------------------------------------------------
# Build cluster once, reuse for both runs
# ---------------------------------------------------------------------------
def build_cluster(lam0, R_star, threads, args):
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    cl = alat[0]
    _lat.build_bcc_sphere(cl, args.R, d1=D1, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)
    alat.set_n_threads(threads)
    cl.info.split_hi = 0.0
    cl.info.split_lo = 0.0
    cl.info.max_level = 0

    pos = np.array(cl.leaf_pos)
    r = np.sqrt(np.sum(pos**2, axis=1))

    if lam0 > 0:
        # 1/r winding profile: lam0 at centre, lam0*R_star/r outside
        safe_r = np.where(
            r > 0, r, 1.0
        )  # avoid 0-division; r=0 cells take float(lam0) anyway
        lam_r = np.where(r > R_star, lam0 * R_star / safe_r, float(lam0))
        lam_r = np.clip(lam_r, 0.0, LAM_MAX)
        cl.set_leaf_lam(lam_r.astype(np.float64))

    cl.set_bc(_lat.BoundaryCondition.sponge(target=0.0, weight=0.5))
    return alat, cl, pos, r


def init_source(cl, pos, source_type, d, b, sigma, amp):
    x0 = -d
    if source_type == "point":
        r2 = (pos[:, 0] - x0) ** 2 + pos[:, 1] ** 2 + pos[:, 2] ** 2
    else:  # ring: torus at x=x0, rho=b
        rho2 = pos[:, 1] ** 2 + pos[:, 2] ** 2
        r2 = (pos[:, 0] - x0) ** 2 + (np.sqrt(rho2) - b) ** 2
    u = amp * np.exp(-r2 / (2.0 * sigma**2))
    cl.set_leaf_u(u)


def measure(cl, pos, x_detect):
    """rho-centroid and total u at x > x_detect."""
    u = np.array(cl.leaf_u)
    mask = pos[:, 0] > x_detect
    w = np.maximum(u[mask], 0.0)
    wsum = float(np.sum(w))
    if wsum < 1e-30:
        return float("nan"), float("nan")
    rho = np.sqrt(pos[mask, 1] ** 2 + pos[mask, 2] ** 2)
    rho_c = float(np.dot(w, rho) / wsum)
    return rho_c, wsum


def steady_delta(rho_star_hist, rho_base_hist, n_avg=3):
    """Mean Δrho over the last n_avg report steps."""
    pairs = [
        (rs, rb)
        for rs, rb in zip(rho_star_hist[-n_avg:], rho_base_hist[-n_avg:])
        if np.isfinite(rs) and np.isfinite(rb)
    ]
    if not pairs:
        return float("nan")
    return float(np.mean([rs - rb for rs, rb in pairs]))


def print_step_summary(rho_star, rho_base, args):
    print(f"{'step':>6}  {'rho_star':>10}  {'rho_base':>10}  {'Δrho':>10}  {'verdict'}")
    steps = [args.report * (i + 1) for i in range(len(rho_star))]
    for s, rs, rb in zip(steps, rho_star, rho_base):
        if np.isfinite(rs) and np.isfinite(rb):
            dr = rs - rb
            verdict = "FOCUS (→star)" if dr < 0 else "DEFOCUS (←star)"
        else:
            dr, verdict = float("nan"), "no signal"
        print(f"{s:6d}  {rs:10.4f}  {rb:10.4f}  {dr:+10.4f}  {verdict}")


def main():
    # ---------------------------------------------------------------------------
    # CLI
    # ---------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Eddington pressure-path test")
    parser.add_argument("--R", type=float, default=80.0)
    parser.add_argument(
        "--lam0",
        type=float,
        default=1.0,
        help="winding at star centre (lam0=1 → sigma=0.5)",
    )
    parser.add_argument("--R-star", type=float, default=3.0)
    parser.add_argument(
        "--d", type=float, default=40.0, help="source distance from origin along -x"
    )
    parser.add_argument(
        "--b",
        type=float,
        default=10.0,
        help="ring impact parameter rho (ring test only)",
    )
    parser.add_argument(
        "--sigma", type=float, default=2.5, help="Gaussian source width"
    )
    parser.add_argument("--amp", type=float, default=1.0, help="source u amplitude")
    parser.add_argument(
        "--x-detect", type=float, default=30.0, help="x threshold for detector plane"
    )
    parser.add_argument(
        "--source", type=str, default="point", choices=["point", "ring"]
    )
    parser.add_argument("--n-steps", type=int, default=6000)
    parser.add_argument("--report", type=int, default=500)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument(
        "--b-sweep",
        type=str,
        default=None,
        help="comma-separated b values for impact parameter sweep, e.g. 3,5,7,10,15",
    )
    parser.add_argument(
        "--lam0-sweep",
        type=str,
        default=None,
        help="comma-separated lam0 values for field strength sweep, e.g. 1,2,5,10",
    )
    args = parser.parse_args()

    def run_case(label, lam0):
        alat, cl, pos, r = build_cluster(lam0, args.R_star, args.threads, args)
        init_source(cl, pos, args.source, args.d, args.b, args.sigma, args.amp)
        dt = compute_dt(cl, D1=D1)

        lam_arr = np.array(cl.leaf_lam)
        lam_max_actual = float(lam_arr.max())
        print(
            f"\n--- {label}  lam0={lam0}  lam_max={lam_max_actual:.4g}  N={cl.n_leaves}  dt={dt:.4e} ---"
        )
        print(f"{'step':>6}  {'rho_cent':>10}  {'u_total':>10}")

        rho_history = []
        for s in range(1, args.n_steps + 1):
            alat.step(0, dt)
            if s % args.report == 0:
                rho_c, wsum = measure(cl, pos, args.x_detect)
                rho_history.append(rho_c)
                tag = f"{rho_c:10.4f}" if np.isfinite(rho_c) else "       nan"
                print(f"{s:6d}  {tag}  {wsum:10.4f}", flush=True)

        return rho_history

    def run_pair(b_val, lam0_val):
        """Run STAR + BASELINE for given b and lam0; return (delta_rho, rho_star_hist, rho_base_hist)."""
        args.b = b_val
        rho_star = run_case(f"STAR   b={b_val:.1f} lam0={lam0_val:.2f}", lam0_val)
        rho_base = run_case(f"BASE   b={b_val:.1f}", 0.0)
        return steady_delta(rho_star, rho_base), rho_star, rho_base

    # ---------------------------------------------------------------------------
    # Sweep or single run
    # ---------------------------------------------------------------------------
    print(f"R={args.R}  source={args.source}  d={args.d}")
    print(f"lam0={args.lam0}  R_star={args.R_star}  x_detect={args.x_detect}")
    print(f"n_steps={args.n_steps}  sigma={args.sigma}")
    print("gravitational well via lam (float64): sigma = 1/(1+lam)")

    if args.b_sweep:
        b_values = [float(v) for v in args.b_sweep.split(",")]
        print(f"\n=== b-sweep  lam0={args.lam0}  R_star={args.R_star} ===")
        print(f"{'b':>8}  {'Δrho':>10}  {'Δrho/b':>10}  {'verdict'}")
        for bv in b_values:
            delta, rh_s, rh_b = run_pair(bv, args.lam0)
            print_step_summary(rh_s, rh_b, args)
            drob = delta / bv if np.isfinite(delta) else float("nan")
            verdict = "FOCUS" if np.isfinite(delta) and delta < 0 else "DEFOCUS/nan"
            print(
                f"\n>>> b={bv:6.1f}  Δrho={delta:+10.4f}  Δrho/b={drob:+10.4f}  {verdict}\n"
            )

    elif args.lam0_sweep:
        lam0_values = [float(v) for v in args.lam0_sweep.split(",")]
        print(f"\n=== lam0-sweep  b={args.b}  R_star={args.R_star} ===")
        print(f"{'lam0':>8}  {'Δrho':>10}  {'Δrho/lam0':>12}  {'verdict'}")
        for lv in lam0_values:
            delta, rh_s, rh_b = run_pair(args.b, lv)
            print_step_summary(rh_s, rh_b, args)
            droc = delta / lv if (np.isfinite(delta) and lv > 0) else float("nan")
            verdict = "FOCUS" if np.isfinite(delta) and delta < 0 else "DEFOCUS/nan"
            print(
                f"\n>>> lam0={lv:6.2f}  Δrho={delta:+10.4f}  Δrho/lam0={droc:+12.4f}  {verdict}\n"
            )

    else:
        rho_star = run_case("STAR", args.lam0)
        rho_baseline = run_case("BASELINE", 0.0)
        print("\n=== summary ===")
        print_step_summary(rho_star, rho_baseline, args)

    print("\ndone")


if __name__ == "__main__":
    main()
