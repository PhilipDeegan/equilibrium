"""Step the uniform-L0 baseline (no coarse-fine junction anywhere), plotting
intermediate states on the fly (no per-step checkpoint) -- only the final
state is saved to disk, for comparison against the R60/L1@R40 boundary
case's H/H0 trajectory.

Resumable: --snapshot points at whatever state to step FROM (the original
IC for a fresh run, or a prior run's "*_final.h5" to continue evolving
further); --ic always points at the true t=0 IC so H0 (and hence H/H0)
stays defined relative to the real origin, not the resume point, so the
ratio is continuous and comparable across chained runs. --start-step is
auto-detected from a "..._NNN_final.h5"-shaped --snapshot name (absolute
step count already reached) unless given explicitly.
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
        "--ic", default="/bk/uniform_l0_ic220.h5", help="true t=0 IC, used only for H0"
    )
    ap.add_argument(
        "--snapshot",
        default=None,
        help="state to resume from (default: --ic, i.e. fresh start)",
    )
    ap.add_argument(
        "--start-step",
        type=int,
        default=None,
        help="absolute step already reached by --snapshot (auto-detected if omitted)",
    )
    ap.add_argument("--out-dir", default="/bk/uniform_l0_stepped")
    ap.add_argument(
        "--plot-dir", default="exp/2026/08/10/19_24_l0_l1_boundary_error/plots/uniform"
    )
    ap.add_argument(
        "--n-steps",
        type=int,
        default=30,
        help="additional steps to run from start-step",
    )
    ap.add_argument("--plot-every", type=int, default=9)
    ap.add_argument("--clip-r", type=float, default=140.0)
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

    alat = _lat.AdaptiveLattice.load(snapshot)
    alat.set_n_threads(args.n_threads)
    cl = alat[0]
    print(
        f"loaded {snapshot} (resuming from absolute step {start_step}) n_leaves={cl.n_leaves} "
        f"t={time.time()-t0:.1f}s peak_rss_GB={rss_gb():.2f}",
        flush=True,
    )

    dt = _compute_dt_l0(cl, D1=1.0)
    ts = _lat.TimeStepper.by_level(dt)
    evolver = _lat.AdaptiveLatticeEvolver(alat)
    print(f"dt={dt:.6e}", flush=True)

    for step_i in range(start_step + 1, start_step + args.n_steps + 1):
        t1 = time.time()
        evolver.step(ts)
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
