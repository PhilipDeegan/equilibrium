import argparse
import sys

sys.path.insert(0, ".")
import resource
import time

import numpy as np

from equilibrium import lattice as _lat
from run.substrate.adaptive.sweep_spheromak_decay_cross_scale import _compute_dt_l0


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-steps", type=int, default=1)
    ap.add_argument("--n-threads", type=int, default=8)
    args = ap.parse_args()

    t0 = time.time()
    alat = _lat.AdaptiveLattice.load(args.snapshot)
    alat.set_n_threads(args.n_threads)
    cl = alat[0]
    print(
        f"loaded {args.snapshot} n_leaves={cl.n_leaves} t={time.time()-t0:.1f}s "
        f"peak_rss_GB={rss_gb():.2f}",
        flush=True,
    )

    u0 = np.array(cl.leaf_u)
    phi0 = np.array(cl.leaf_phi)
    H0 = float(np.sum(u0**2) + 0.5 * np.sum(phi0**2))
    print(f"H0={H0:.6e}", flush=True)

    dt = _compute_dt_l0(cl, D1=1.0)
    ts = _lat.TimeStepper.by_level(dt)
    evolver = _lat.AdaptiveLatticeEvolver(alat)
    print(f"dt={dt:.6e}", flush=True)

    for step_i in range(1, args.n_steps + 1):
        t1 = time.time()
        evolver.step(ts)
        step_dt = time.time() - t1

        u = np.array(cl.leaf_u)
        phi = np.array(cl.leaf_phi)
        H = float(np.sum(u**2) + 0.5 * np.sum(phi**2))
        print(
            f"step {step_i} step_time={step_dt:.2f}s H={H:.6e} H/H0={H/H0:.6f} "
            f"peak_rss_GB={rss_gb():.2f}",
            flush=True,
        )

    t2 = time.time()
    alat.save(args.out)
    print(f"saved {args.out} save_time={time.time()-t2:.1f}s", flush=True)
    print(
        f"FINAL peak_rss_GB={rss_gb():.2f} total_time={time.time()-t0:.1f}s", flush=True
    )


if __name__ == "__main__":
    main()
