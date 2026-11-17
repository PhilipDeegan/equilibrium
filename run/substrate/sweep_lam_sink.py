"""
Parameter sweep for probe_lam_sink: lam_scale × sink_scale grid.

Reports a summary table of final lam_sum, u_asym, and sep change for each
combination so we can find a stable regime where:
  - lam_sum is stable (not collapsing or exploding)
  - u_asym < 0 consistently (sinkhole depletion working)
  - sep change is visible (attraction emerging)
"""

import argparse
import itertools
import sys

import numpy as np

sys.path.insert(0, ".")

from equilibrium.substrate.lattice import DampedLattice

parser = argparse.ArgumentParser()
parser.add_argument("--R", type=float, default=12.0)
parser.add_argument("--sep", type=float, default=6.0)
parser.add_argument("--lam", type=float, default=1.0)
parser.add_argument("--n-warm", type=int, default=1000)
parser.add_argument("--n-steps", type=int, default=3000)
parser.add_argument("--lam-scales", type=float, nargs="+", default=[0.0])
parser.add_argument("--force-scales", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2])
parser.add_argument("--sink-scales", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2])
parser.add_argument("--chi-scales", type=float, nargs="+", default=[0.0])


def run_one(
    R, sep, lam_0, n_warm, n_steps, lam_scale, force_scale, sink_scale, chi_scale=0.0
):
    lat = DampedLattice(radius_cells=R, sponge_fraction=0.15, damp_max=0.003)
    pos = lat.pos
    dt = lat.stable_dt()

    half = sep / 2.0
    chi_0 = lam_0 / (1.0 + lam_0)

    def nearest(target):
        return lat.tiles[int(np.argmin(np.linalg.norm(pos - target, axis=1)))]

    kA = nearest(np.array([0.0, 0.0, half]))
    kB = nearest(np.array([0.0, 0.0, -half]))

    lat.set_winding(kA, chi_0=chi_0)
    lam_A = lat.lam.copy()
    lat.lam[:] = 0.0
    lat.set_winding(kB, chi_0=chi_0)
    lat.lam[:] = np.maximum(lam_A, lat.lam)

    for t in lat.tiles:
        rA = float(np.linalg.norm(t.pos - kA.pos))
        rB = float(np.linalg.norm(t.pos - kB.pos))
        lat.u[t.idx] = 1.0 / max(rA, 0.5) + 1.0 / max(rB, 0.5)

    for _ in range(n_warm):
        lat.step(dt, chi_scale=chi_scale)

    def lam_com(lam_w):
        total = lam_w.sum()
        return (
            (lam_w[:, None] * pos).sum(axis=0) / total if total > 1e-30 else np.zeros(3)
        )

    def knot_mask(centre):
        return np.linalg.norm(pos - centre, axis=1) < 2.5

    def u_slab(z_lo, z_hi):
        m = (pos[:, 2] >= z_lo) & (pos[:, 2] < z_hi)
        return float(lat.u[m].mean()) if m.any() else 0.0

    lam_sum_0 = float(lat.lam.sum())
    mA = knot_mask(kA.pos)
    mB = knot_mask(kB.pos)
    sep_0 = float(np.linalg.norm(lam_com(lat.lam * mA) - lam_com(lat.lam * mB)))

    u_asym_sum = 0.0
    stable = True

    for s in range(n_steps):
        lat.step(dt, chi_scale=chi_scale)
        lat.step_lam_sink(dt, scale=sink_scale)
        if lam_scale:
            lat.step_lam(dt, scale=lam_scale)
        if force_scale:
            lat.step_lam_force(dt, scale=force_scale)

        if not np.isfinite(lat.u).all() or not np.isfinite(lat.lam).all():
            stable = False
            break
        if lat.lam.sum() < 0.01 * lam_sum_0:
            stable = False  # knots dissolved
            break

        if s % 200 == 0:
            u_btw = u_slab(-half, +half)
            u_outA = u_slab(+half, +half + sep)
            u_outB = u_slab(-half - sep, -half)
            u_asym_sum += u_btw - (u_outA + u_outB) / 2.0

    if not stable:
        return dict(
            lam_scale=lam_scale,
            force_scale=force_scale,
            sink_scale=sink_scale,
            chi_scale=chi_scale,
            stable=False,
            dlam=float("nan"),
            u_asym_mean=float("nan"),
            dsep=float("nan"),
        )

    lam_sum_f = float(lat.lam.sum())
    sep_f = float(np.linalg.norm(lam_com(lat.lam * mA) - lam_com(lat.lam * mB)))
    n_samples = n_steps // 200
    u_asym_mean = u_asym_sum / max(n_samples, 1)

    return dict(
        lam_scale=lam_scale,
        force_scale=force_scale,
        sink_scale=sink_scale,
        chi_scale=chi_scale,
        stable=True,
        dlam=lam_sum_f - lam_sum_0,
        u_asym_mean=u_asym_mean,
        dsep=sep_f - sep_0,
    )


def main():
    args = parser.parse_args()

    lam_scales = args.lam_scales
    force_scales = args.force_scales
    sink_scales = args.sink_scales
    chi_scales = args.chi_scales

    print(
        f"R={args.R}  sep={args.sep}  lam_0={args.lam}  "
        f"n_warm={args.n_warm}  n_steps={args.n_steps}"
    )
    print(f"grid: force_scale {force_scales}")
    print(f"      sink_scale  {sink_scales}")
    print(f"      chi_scale   {chi_scales}\n")

    hdr = (
        f"{'force_scale':>11}  {'sink_scale':>10}  {'chi_scale':>10}  "
        f"{'stable':>6}  {'Δlam':>10}  {'u_asym_mean':>12}  {'Δsep':>10}"
    )
    print(hdr)
    print("-" * len(hdr))

    for sink_scale, force_scale, chi_scale in itertools.product(
        sink_scales, force_scales, chi_scales
    ):
        r = run_one(
            args.R,
            args.sep,
            args.lam,
            args.n_warm,
            args.n_steps,
            lam_scales[0],
            force_scale,
            sink_scale,
            chi_scale,
        )
        flag = ""
        if r["stable"] and r["u_asym_mean"] < -1e-4 and r["dsep"] < -1e-4:
            flag = "  <-- ATTRACT"
        print(
            f"{force_scale:>11.2e}  {sink_scale:>10.2e}  {chi_scale:>10.2e}  "
            f"{r['stable']!s:>6}  "
            f"{r['dlam']:>+10.3f}  {r['u_asym_mean']:>+12.6f}  {r['dsep']:>+10.6f}{flag}",
            flush=True,
        )

    print("\ndone")


if __name__ == "__main__":
    main()
