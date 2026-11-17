import argparse
import sys

sys.path.insert(0, ".")
import resource
import time

import numpy as np

from equilibrium import lattice as _lat
from run.substrate.adaptive.sweep_spheromak_decay_cross_scale import _compute_dt_l0

CORE_LEVEL = 5
SNAPSHOT = "/bk/l5_core_diameter9_cells_across_220.h5"
# re-earned ladder L0-L4 (project_odd27_amr_junction_leak.md Part 15 / `now`)
ETA_H_L0_L4 = [0.0, 0.0, 1e-5, 1e-7, 3e-9]


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--eta-h-l5",
        type=float,
        default=0.0,
        help="eta_h at level 5 only, L0-L4 held at 0",
    )
    ap.add_argument("--n-steps", type=int, default=3)
    ap.add_argument("--n-threads", type=int, default=4)
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    eta_h_by_level = ETA_H_L0_L4 + [args.eta_h_l5]

    t0 = time.time()
    alat = _lat.AdaptiveLattice.load(SNAPSHOT)
    alat.set_n_threads(args.n_threads)
    cl = alat[0]
    print(
        f"[{args.tag}] loaded {SNAPSHOT} n_leaves={cl.n_leaves} "
        f"t={time.time()-t0:.1f}s peak_rss_GB={rss_gb():.2f}",
        flush=True,
    )

    u0 = np.array(cl.leaf_u)
    phi0 = np.array(cl.leaf_phi)
    levels0 = np.array(cl.leaf_level)
    h0 = u0**2 + 0.5 * np.sum(phi0**2, axis=0)
    H0 = float(h0.sum())
    H_by_lvl_0 = np.bincount(levels0, weights=h0)
    print(f"[{args.tag}] H0={H0:.6e} H_by_level_0={H_by_lvl_0.tolist()}", flush=True)

    dt = _compute_dt_l0(cl, D1=1.0)
    ts = _lat.TimeStepper.by_level(dt)
    evolver = _lat.AdaptiveLatticeEvolver(alat)
    print(f"[{args.tag}] dt={dt:.6e} eta_h_by_level={eta_h_by_level}", flush=True)

    for step_i in range(1, args.n_steps + 1):
        t1 = time.time()
        evolver.step(ts)
        alat.step_boundary_hyper_resistivity_per_level_all(
            dt, eta_h_by_level, imex=True
        )
        step_dt = time.time() - t1
        u = np.array(cl.leaf_u)
        phi = np.array(cl.leaf_phi)
        levels = np.array(cl.leaf_level)
        h = u**2 + 0.5 * np.sum(phi**2, axis=0)
        H = float(h.sum())
        H_by_lvl = np.bincount(levels, weights=h)
        print(
            f"[{args.tag}] step {step_i} step_time={step_dt:.1f}s H={H:.6e} H/H0={H/H0:.6f} "
            f"H_by_level={H_by_lvl.tolist()} peak_rss_GB={rss_gb():.2f}",
            flush=True,
        )

    print(
        f"[{args.tag}] FINAL peak_rss_GB={rss_gb():.2f} total_time={time.time()-t0:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
