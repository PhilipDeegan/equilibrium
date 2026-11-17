"""
No imposed shape: seed a single localized bump of u at ONE ring tile
(everything else zero background), then just print the raw phi values
around the ring at several times.  No cos/sin ansatz anywhere -- whatever
periodicity or lack of it shows up is discovered from the update rule's
own relaxation dynamics, not assumed going in.

Run:
  PYTHONPATH=$PWD python3 -u run/substrate/adaptive/probe_ring_bump_relaxation.py
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium import lattice as _lat
from tst.substrate.adaptive import compute_dt
from tst.substrate.adaptive.adaptive_flux_tube import find_toroidal_ring

parser = argparse.ArgumentParser()
parser.add_argument("--R", type=float, default=15.0)
parser.add_argument("--hops", type=int, default=3)
parser.add_argument("--bump", type=float, default=1.0)
parser.add_argument("--n-steps", type=int, default=2000)
parser.add_argument("--report", type=int, default=100)
parser.add_argument("--n-threads", type=int, default=4)
parser.add_argument(
    "--print-vals", action="store_true", help="print full per-tile array each report"
)
parser.add_argument(
    "--save-npz",
    type=str,
    default=None,
    help="save every checkpoint's raw ring_u array (plus IC, N_ring, dt) so "
    "any diagnostic can be re-run offline without re-simulating",
)
args = parser.parse_args()


def main():
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    alat.set_n_threads(args.n_threads)
    cl = alat[0]
    _lat.build_bcc_sphere(cl, args.R, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)

    ring_js, ring_faces = find_toroidal_ring(
        cl, axis="z", mode="square", hops=args.hops
    )
    if ring_js is None:
        raise RuntimeError("ring-finding failed")
    N_ring = len(ring_js)

    # single localized bump -- one ring tile gets extra u, nothing else touched
    u = np.array(cl.leaf_u)
    u[ring_js[0]] = args.bump
    cl.set_leaf_u(u)

    dt = compute_dt(cl)
    print(f"N_ring={N_ring}  bump at ring-index 0 only  dt={dt:.4e}\n")

    ic_ring_u = None
    saved_steps, saved_u = [], []

    def snapshot(step, print_vals=True):
        nonlocal ic_ring_u
        u_arr = np.array(cl.leaf_u)
        ring_u = np.array([u_arr[j] for j in ring_js])
        if ic_ring_u is None:
            ic_ring_u = ring_u.copy()
        if args.save_npz:
            saved_steps.append(step)
            saved_u.append(ring_u.copy())
        peak = int(np.argmax(np.abs(ring_u)))
        spread = int(np.sum(np.abs(ring_u) > 0.05 * args.bump))
        dist = float(np.linalg.norm(ring_u - ic_ring_u))
        dist0 = float(np.linalg.norm(ic_ring_u))
        line = (
            f"step {step:6d}  peak_idx={peak:3d}  n_cells>5%={spread:3d}  "
            f"|u-IC|={dist:.4f}  |u-IC|/|IC|={dist / dist0:.4f}"
        )
        if print_vals:
            vals = "  ".join(f"{v:+.3f}" for v in ring_u)
            line += f"  vals: {vals}"
        print(line)

    snapshot(0)
    for step in range(1, args.n_steps + 1):
        alat.step(0, dt)
        if step % args.report == 0:
            snapshot(step, print_vals=args.print_vals)

    if args.save_npz:
        np.savez(
            args.save_npz,
            steps=np.array(saved_steps),
            ring_u=np.array(saved_u),  # (n_snapshots, N_ring)
            ic_ring_u=ic_ring_u,
            N_ring=N_ring,
            dt=dt,
            R=args.R,
            hops=args.hops,
            bump=args.bump,
        )
        print(f"\nsaved {len(saved_steps)} snapshots -> {args.save_npz}")


if __name__ == "__main__":
    main()
