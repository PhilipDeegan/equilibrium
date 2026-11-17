"""
DST proton + electron (moat) probe.

Sets up a proton as a pinned high-u source at the origin (open knot —
u radiates outward through all 14 faces, creating a 1/r field).
Initialises the background u to the 1/r steady state.

Runs lat.step only.  Tracks the radial u profile over time, looking for
spontaneous moat (u-deficit shell) formation.

In DST every stable knot/moat pair is a product of the update rule.
If the moat does NOT form here, it indicates that cell-size contrast
(lambda / chi, absent in DampedLattice) is required — i.e. the moat is
a multi-resolution effect, not a uniform-sigma effect.
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")

from equilibrium.substrate.lattice import DampedLattice


def radial_profile(u, r, edges):
    means = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r >= lo) & (r < hi)
        means.append(float(np.mean(u[m])) if m.any() else 0.0)
    return np.array(means)


def u_moat_depth(u, r, r_moat_lo=4.0, r_moat_hi=16.0, r_ref_lo=1.5, r_ref_hi=3.5):
    """
    Minimum u in the moat band minus expected 1/r value there.
    Negative = genuine deficit below the 1/r envelope.
    """
    ref_mask = (r >= r_ref_lo) & (r < r_ref_hi)
    moat_mask = (r >= r_moat_lo) & (r < r_moat_hi)
    if not ref_mask.any() or not moat_mask.any():
        return 0.0
    # fit 1/r: u_ref = A/r_ref → A = mean(u*r) in ref band
    A = float(np.mean(u[ref_mask] * r[ref_mask]))
    u_expected = A / r[moat_mask]
    deficit = u[moat_mask] - u_expected
    return float(np.min(deficit))


def u_at_r(u, r, target, dr=0.6):
    m = np.abs(r - target) < dr
    return float(np.mean(u[m])) if m.any() else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--R", type=float, default=40.0)
    parser.add_argument(
        "--u-central",
        type=float,
        default=10.0,
        help="fixed u at origin cell (proton source)",
    )
    parser.add_argument("--n-steps", type=int, default=20000)
    parser.add_argument("--report", type=int, default=2000)
    parser.add_argument("--n-threads", type=int, default=1)
    parser.add_argument(
        "--r-max-plot",
        type=float,
        default=20.0,
        help="max radius for radial profile output",
    )

    args = parser.parse_args()

    lat = DampedLattice(radius_cells=args.R, sponge_fraction=0.15, damp_max=0.003)
    pos = lat.pos
    r = np.linalg.norm(pos, axis=1)
    dt = lat.stable_dt()

    # Pin origin cell to u_central — the proton is a sustained u source
    origin = int(np.argmin(r))
    lat.fixed_u[origin] = True
    lat.u[origin] = args.u_central

    # Initialise background to 1/r steady state so phi starts near zero
    safe_r = np.where(r > 0.5, r, 1.0)
    lat.u = np.where(lat.fixed_u, lat.u, args.u_central / safe_r)

    print(f"N={lat.N}  R={args.R}  dt={dt:.4e}")
    print(f"origin cell: j={origin}  r={r[origin]:.3f}  u={lat.u[origin]:.2f}")
    print(f"u_central={args.u_central}  n_steps={args.n_steps}")
    print("\nNo lambda, no winding — pure lat.step.  Watching for moat.\n")

    # Radial bins for profile
    r_bins = np.arange(0.5, args.r_max_plot + 0.5, 0.5)
    r_edges = np.arange(0.25, args.r_max_plot + 0.75, 0.5)

    print(
        f"{'step':>8}  {'phi_rms':>10}  {'moat_depth':>12}  {'u@r=1':>8}  {'u@r=5':>8}  {'u@r=10':>8}  {'u@r=15':>8}"
    )
    print("  " + "-" * 76)

    for s in range(args.n_steps + 1):
        if s % args.report == 0:
            phi_rms = float(np.sqrt(np.mean(lat.phi**2)))
            depth = u_moat_depth(lat.u, r)
            u1 = u_at_r(lat.u, r, 1.0)
            u5 = u_at_r(lat.u, r, 5.0)
            u10 = u_at_r(lat.u, r, 10.0)
            u15 = u_at_r(lat.u, r, 15.0)
            print(
                f"{s:8d}  {phi_rms:10.4e}  {depth:+12.5f}  {u1:8.4f}  {u5:8.4f}  {u10:8.4f}  {u15:8.4f}",
                flush=True,
            )

        if s < args.n_steps:
            lat.step(dt, n_threads=args.n_threads)

    # Final radial profile
    prof = radial_profile(lat.u, r, r_edges)
    print("\nFinal radial u profile (r, u_mean):")
    for rc, um in zip(r_bins, prof):
        bar = "#" * max(0, int(um * 4))
        print(f"  r={rc:5.1f}  u={um:+8.4f}  {bar}")

    print("\ndone")


if __name__ == "__main__":
    main()
