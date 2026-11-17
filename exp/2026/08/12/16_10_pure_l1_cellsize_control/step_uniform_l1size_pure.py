"""Step the pure-L1-cell-size control mesh: evolver.step(ts) plus
step_cross_all(dt, cross_scale) each step, identical convention to
step_uniform_r180_active.py / step_l1_wider_margin_active.py
(exp/2026/08/11/21_27_everything_active/). dt uses D1=CELL_SIZE (1/3) so this
mesh runs at the SAME effective per-step time resolution L1 cells get via
subcycling in the AMR case -- the only intended difference from that
subcycling is floating-point roundoff, not a physically different dt.

Resumable via --snapshot/--start-step, same convention as the 08/11 scripts.
"""

import argparse
import re
import sys

sys.path.insert(0, ".")
import os
import resource
import time

import numpy as np

from equilibrium import lattice as _lat
from run.substrate.adaptive.sweep_spheromak_decay_cross_scale import _compute_dt_l0
from run.substrate.adaptive.plot_telescope_xz import plot_cluster

CELL_SIZE = 1.0 / 3.0


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def _H_of(path):
    alat = _lat.AdaptiveLattice.load(path)
    cl = alat[0]
    u = np.array(cl.leaf_u)
    phi = np.array(cl.leaf_phi)
    return float(np.sum(u**2) + 0.5 * np.sum(phi**2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ic",
        default="/bk/uniform_l1size_pure_ic220.h5",
        help="true t=0 IC, used only for H0",
    )
    ap.add_argument(
        "--snapshot", default=None, help="state to resume from (default: --ic)"
    )
    ap.add_argument("--start-step", type=int, default=None)
    ap.add_argument(
        "--cross-scale",
        type=float,
        default=1e-3,
        help="step_cross resistive-coupling strength",
    )
    ap.add_argument("--out-dir", default="/bk/uniform_l1size_pure_stepped")
    ap.add_argument(
        "--plot-dir", default="exp/2026/08/12/16_10_pure_l1_cellsize_control/plots"
    )
    ap.add_argument("--n-steps", type=int, default=130)
    ap.add_argument("--plot-every", type=int, default=25)
    ap.add_argument("--clip-r", type=float, default=60.0)
    ap.add_argument("--n-threads", type=int, default=8)
    args = ap.parse_args()

    snapshot = args.snapshot or args.ic
    start_step = args.start_step
    if start_step is None:
        m = re.search(r"step_(\d+)_final\.h5$", snapshot)
        start_step = int(m.group(1)) if m else 0

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    t0 = time.time()
    H0 = _H_of(args.ic)
    print(f"H0 (from {args.ic})={H0:.6e}", flush=True)
    print(f"cross_scale={args.cross_scale:g}", flush=True)

    alat = _lat.AdaptiveLattice.load(snapshot)
    alat.set_n_threads(args.n_threads)
    cl = alat[0]
    lam0 = np.array(cl.leaf_lam)
    print(
        f"loaded {snapshot} (resuming from absolute step {start_step}) n_leaves={cl.n_leaves} "
        f"lam_max={float(lam0.max()):.6e} lam_mean={float(lam0.mean()):.6e} "
        f"t={time.time()-t0:.1f}s peak_rss_GB={rss_gb():.2f}",
        flush=True,
    )

    dt = _compute_dt_l0(cl, D1=CELL_SIZE)
    ts = _lat.TimeStepper.by_level(dt)
    evolver = _lat.AdaptiveLatticeEvolver(alat)
    print(f"dt={dt:.6e}", flush=True)

    for step_i in range(start_step + 1, start_step + args.n_steps + 1):
        t1 = time.time()
        evolver.step(ts)
        if args.cross_scale > 0.0:
            alat.step_cross_all(dt, args.cross_scale)
        step_dt = time.time() - t1

        u = np.array(cl.leaf_u)
        phi = np.array(cl.leaf_phi)
        h = u**2 + 0.5 * np.sum(phi**2, axis=0)
        H = float(h.sum())
        print(
            f"step {step_i} step_time={step_dt:.1f}s H={H:.6e} H/H0={H/H0:.6f} peak_rss_GB={rss_gb():.2f}",
            flush=True,
        )

        if step_i % args.plot_every == 0:
            t2 = time.time()
            label = f"step_{step_i:03d}"
            out_png = f"{args.plot_dir}/xz_{label}.png"
            plot_cluster(cl, label, out_png, args.clip_r, 260, 1e-16)
            print(
                f"step {step_i} plotted {out_png} plot_time={time.time()-t2:.1f}s",
                flush=True,
            )

    t3 = time.time()
    end_step = start_step + args.n_steps
    final_path = f"{args.out_dir}/step_{end_step:03d}_final.h5"
    alat.save(final_path)
    print(f"saved final {final_path} save_time={time.time()-t3:.1f}s", flush=True)
    print(
        f"FINAL peak_rss_GB={rss_gb():.2f} total_time={time.time()-t0:.1f}s", flush=True
    )


if __name__ == "__main__":
    main()
