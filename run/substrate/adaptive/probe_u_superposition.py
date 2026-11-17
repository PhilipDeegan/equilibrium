"""
Linear superposition baseline -- do a bare u-excess and u-deficit (no phi
winding, no lam, no cross) cancel when combined, purely from the wave-step
equations' own linearity?

Motivation: candidate charge-sign carriers under discussion are (a) winding-
number chirality of a phi loop, or (b) excess/deficit of the u density. This
tests the null case of (b) with no protecting structure at all. Since
du/dt = -sum(w_k phi_k) and dphi/dt = -dt*sigma*W*(u_k-u_i) are linear when
lam=0 (sigma constant), superposition should hold EXACTLY on the real
discrete/adaptive lattice (not just analytically): running the sum-IC should
reproduce the pointwise sum of the two separate runs at every step, and full
co-location should cancel to identically zero for all time. This is the
"no protecting structure -> trivial, permanent cancellation" control that
the electron/positron winding test (counter_rotating dipole collision) is
contrasted against.

Run:
  PYTHONPATH=$PWD python3 -u run/substrate/adaptive/probe_u_superposition.py
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium import lattice as _lat
from tst.substrate.adaptive import compute_dt

parser = argparse.ArgumentParser()
parser.add_argument("--R", type=float, default=30.0)
parser.add_argument("--width", type=float, default=4.0, help="Gaussian bump width")
parser.add_argument("--amp", type=float, default=1.0)
parser.add_argument("--z-sep", type=float, default=10.0)
parser.add_argument("--n-steps", type=int, default=3000)
parser.add_argument("--report", type=int, default=500)
parser.add_argument("--sponge", type=float, default=0.15)
parser.add_argument("--damp-max", type=float, default=0.003)
parser.add_argument("--n-threads", type=int, default=2)


def build(args):
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    alat.set_n_threads(args.n_threads)
    cl = alat[0]
    _lat.build_bcc_sphere(cl, args.R, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)
    pos = np.array(cl.leaf_pos)
    r_all = np.linalg.norm(pos, axis=1)
    r_max = float(r_all.max())
    cl.set_bc(
        _lat.BoundaryCondition.quadratic_sponge(
            target=0.0,
            damp_max=args.damp_max,
            r_all=r_all,
            r_inner=r_max * (1.0 - args.sponge),
        )
    )
    return alat, cl, pos


def bump(pos, centre, width, amp):
    d2 = np.sum((pos - np.array(centre)) ** 2, axis=1)
    return amp * np.exp(-0.5 * d2 / width**2)


def run(args, u0, report_steps):
    alat, cl, pos = build(args)
    cl.set_leaf_u(u0.copy())
    dt = compute_dt(cl)
    snaps = {}
    for step in range(args.n_steps + 1):
        if step in report_steps:
            snaps[step] = np.array(cl.leaf_u).copy()
        if step < args.n_steps:
            alat.step(0, dt)
            alat.step_sponge(0)
    return snaps, dt


def main():
    args = parser.parse_args()
    report_steps = set(range(0, args.n_steps + 1, args.report))
    _, _, pos0 = build(args)

    z0 = args.z_sep / 2.0
    u_pos = bump(pos0, [0, 0, +z0], args.width, +args.amp)
    u_neg = bump(pos0, [0, 0, -z0], args.width, -args.amp)
    u_sum = u_pos + u_neg

    print(f"=== separated (z_sep={args.z_sep}) -- superposition check ===")
    snaps_p, dt = run(args, u_pos, report_steps)
    snaps_n, _ = run(args, u_neg, report_steps)
    snaps_c, _ = run(args, u_sum, report_steps)

    print(f"{'step':>6} {'t':>8} {'rms(combined)':>14} {'max|combined-(p+n)|':>22}")
    for step in sorted(report_steps):
        rc = float(np.sqrt(np.mean(snaps_c[step] ** 2)))
        diff = float(np.abs(snaps_c[step] - (snaps_p[step] + snaps_n[step])).max())
        print(f"{step:6d} {step*dt:8.2f} {rc:14.4e} {diff:22.4e}")

    print(f"\n=== fully co-located (z_sep=0) -- exact permanent cancellation ===")
    args0 = argparse.Namespace(**vars(args))
    args0.z_sep = 0.0
    u_c0 = bump(pos0, [0, 0, 0], args.width, args.amp) + bump(
        pos0, [0, 0, 0], args.width, -args.amp
    )
    print(f"max|u| at t=0: {np.abs(u_c0).max():.3e}  (exact algebraic cancellation)")
    snaps0, dt0 = run(args0, u_c0, report_steps)
    print(f"{'step':>6} {'t':>8} {'rms(u)':>10} {'max|u|':>10}")
    for step in sorted(report_steps):
        u = snaps0[step]
        print(
            f"{step:6d} {step*dt0:8.2f} {np.sqrt(np.mean(u**2)):10.4e} {np.abs(u).max():10.4e}"
        )


if __name__ == "__main__":
    main()
