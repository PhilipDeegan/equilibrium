"""
AMR interface flux conservation diagnostic.

Checks whether the coarse-fine interface introduces spurious u loss.
For a closed system (no BC), total u must be exactly conserved.
Any decay = interface artifact.

Compares:
  A) uniform L0 lattice, no BC
  B) same lattice with a split shell, no BC

Usage:
    PYTHONPATH=$PWD python3 run/substrate/adaptive/flux_conservation.py
    PYTHONPATH=$PWD python3 run/substrate/adaptive/flux_conservation.py --r-shell 9 --shell-width 2 --n-steps 500
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")

from tst.substrate.adaptive import compute_dt


def _build(split=False):
    from equilibrium import lattice as _lat

    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    cl = alat[0]
    _lat.build_bcc_sphere(cl, args.R, args.D1, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)
    alat.set_n_threads(args.threads)
    cl.set_bc(_lat.BoundaryCondition.absorbing())

    if split:
        info = _lat.DSTInfo()
        info.split_hi = 0.5
        info.split_lo = 0.0
        for level in range(args.max_level):
            info.max_level = level + 1
            cl.info = info
            pos = np.array(cl.leaf_pos)
            r = np.sqrt(np.sum(pos**2, axis=1))
            cmax = np.full(cl.n_leaves, np.nan)
            cmax[np.abs(r - args.r_shell) < args.shell_width] = 1.0
            cl.set_criterion_max(cmax)
            alat.adapt(0)
            alat.rebuild(0)

    # Set IC after splitting so all leaf cells get the same Gaussian profile
    pos = np.array(cl.leaf_pos)
    r = np.sqrt(np.sum(pos**2, axis=1))
    cl.set_leaf_u(np.exp(-0.5 * (r / 3.0) ** 2))

    return alat, cl


def _run(label, split):
    alat, cl = _build(split=split)
    dt = compute_dt(cl, D1=args.D1)
    N = cl.n_leaves

    u0 = float(np.sum(np.array(cl.leaf_u)))
    for _ in range(args.n_steps):
        alat.step(0, dt)
    u_end = float(np.sum(np.array(cl.leaf_u)))
    drift = u_end - u0
    pct = 100.0 * drift / u0 if u0 != 0 else float("nan")

    lvl_arr = np.array(cl.leaf_level)
    levels = {int(lv): int(np.sum(lvl_arr == lv)) for lv in np.unique(lvl_arr)}
    dist = "  ".join(f"L{lv}:{n}" for lv, n in sorted(levels.items()))

    print(f"{label}")
    print(f"  N={N}  dt={dt:.3e}  levels=[{dist}]")
    print(f"  total_u: {u0:.6f} → {u_end:.6f}  drift={drift:+.3e}  ({pct:+.4f}%)")
    print()
    return drift / args.n_steps  # drift per step


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--R", type=float, default=15.0)
    parser.add_argument("--D1", type=float, default=1.0)
    parser.add_argument("--r-shell", type=float, default=9.0, help="AMR shell centre")
    parser.add_argument(
        "--shell-width", type=float, default=2.0, help="AMR shell half-width"
    )
    parser.add_argument("--max-level", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    print(
        f"R={args.R}  r_shell={args.r_shell}  shell_width={args.shell_width}"
        f"  max_level={args.max_level}  n_steps={args.n_steps}\n"
    )

    rate_uniform = _run("A) uniform (no AMR, no BC)", split=False)
    rate_amr = _run(
        f"B) AMR shell ±{args.shell_width} at r={args.r_shell}  (no BC)", split=True
    )

    excess = rate_amr - rate_uniform
    print(f"excess drift/step from AMR interface: {excess:+.3e}")
    if abs(rate_uniform) > 0:
        print(f"relative to uniform: {100*excess/abs(rate_uniform):+.2f}%")
