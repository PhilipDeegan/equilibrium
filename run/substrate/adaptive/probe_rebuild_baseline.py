"""
Post-rebuild speed baseline -- one config, R=100, 8 threads, 10 steps.

Same electron/barrier config as the R100 gate sweeps
(probe_resolution_scan.py / probe_cross_v2_gates.py), just short and timed,
to check per-step wall time after a rebuild. Saves the final lattice state
natively via AdaptiveLattice.save (HDF5).

Run:
  PYTHONPATH=$PWD python3 -u run/substrate/adaptive/probe_rebuild_baseline.py
"""

import argparse
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from run.substrate.adaptive.cross_v2 import CrossV2
from run.substrate.adaptive.probe_cross_v2_gates import build_lattice, seed_ic
from tst.substrate.adaptive import compute_dt

parser = argparse.ArgumentParser()
parser.add_argument("--R", type=float, default=100.0)
parser.add_argument("--sph-frac", type=float, default=0.2)
parser.add_argument("--ic", type=str, default="electron")
parser.add_argument("--m0", type=float, default=2.49)
parser.add_argument("--cross-scale", type=float, default=0.1)
parser.add_argument("--n-steps", type=int, default=10)
parser.add_argument("--n-threads", type=int, default=8)
parser.add_argument("--sponge", type=float, default=0.15)
parser.add_argument("--damp-max", type=float, default=0.003)
parser.add_argument("--save-out", type=str, default="/bk/dst/xyz.hdf5")


def main():
    args = parser.parse_args()
    args.lam_target = 1.0
    args.u_max = 1.0
    args.B0 = 1.0
    scale = args.R * args.sph_frac

    print(f"R={args.R}  scale={scale}  n_threads={args.n_threads}  ic={args.ic}")
    alat, cl = build_lattice(args)
    print(f"n_leaves={cl.n_leaves}")

    seed_ic(cl, args.ic, args, scale)
    dt = compute_dt(cl)
    op = CrossV2(cl, gate="barrier", m0=args.m0)

    t_step = []
    for step in range(1, args.n_steps + 1):
        t0 = time.time()
        alat.step(0, dt)
        alat.step_sponge(0)
        if args.cross_scale > 0.0:
            op.step(dt, args.cross_scale)
        t_step.append(time.time() - t0)
        print(f"step {step:3d}  t={t_step[-1]:6.3f}s", flush=True)

    print(f"\nmean step_time={np.mean(t_step):.3f}s  total={sum(t_step):.3f}s")

    alat.save(args.save_out)
    print(f"final state saved -> {args.save_out}")


if __name__ == "__main__":
    main()
