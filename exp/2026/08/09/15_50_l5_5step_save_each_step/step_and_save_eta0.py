import sys

sys.path.insert(0, ".")
import resource
import time

import numpy as np

from equilibrium import lattice as _lat
from run.substrate.adaptive.sweep_spheromak_decay_cross_scale import _compute_dt_l0

SNAPSHOT = "/bk/l5_core_diameter9_cells_across_220.h5"
OUT_DIR = "/bk/l5_5step_save_each_step_eta0"
N_STEPS = 5
N_THREADS = 8
ETA_H_BY_LEVEL = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def main():
    import os

    os.makedirs(OUT_DIR, exist_ok=True)

    t0 = time.time()
    alat = _lat.AdaptiveLattice.load(SNAPSHOT)
    alat.set_n_threads(N_THREADS)
    cl = alat[0]
    print(
        f"loaded {SNAPSHOT} n_leaves={cl.n_leaves} t={time.time()-t0:.1f}s peak_rss_GB={rss_gb():.2f}",
        flush=True,
    )

    u0 = np.array(cl.leaf_u)
    phi0 = np.array(cl.leaf_phi)
    levels0 = np.array(cl.leaf_level)
    h0 = u0**2 + 0.5 * np.sum(phi0**2, axis=0)
    H0 = float(h0.sum())
    print(
        f"H0={H0:.6e} H_by_level_0={np.bincount(levels0, weights=h0).tolist()}",
        flush=True,
    )

    dt = _compute_dt_l0(cl, D1=1.0)
    ts = _lat.TimeStepper.by_level(dt)
    evolver = _lat.AdaptiveLatticeEvolver(alat)
    print(f"dt={dt:.6e} eta_h_by_level={ETA_H_BY_LEVEL}", flush=True)

    for step_i in range(1, N_STEPS + 1):
        t1 = time.time()
        evolver.step(ts)
        # eta_h all zero -> step_boundary_hyper_resistivity_per_level_all is a no-op;
        # skip calling it entirely to make the "no eta_h at all" comparison unambiguous.
        step_dt = time.time() - t1

        u = np.array(cl.leaf_u)
        phi = np.array(cl.leaf_phi)
        levels = np.array(cl.leaf_level)
        h = u**2 + 0.5 * np.sum(phi**2, axis=0)
        H = float(h.sum())
        print(
            f"step {step_i} step_time={step_dt:.1f}s H={H:.6e} H/H0={H/H0:.6f} "
            f"H_by_level={np.bincount(levels, weights=h).tolist()} peak_rss_GB={rss_gb():.2f}",
            flush=True,
        )

        t2 = time.time()
        out_path = f"{OUT_DIR}/step_{step_i:02d}.h5"
        alat.save(out_path)
        print(
            f"step {step_i} saved {out_path} save_time={time.time()-t2:.1f}s",
            flush=True,
        )

    print(
        f"FINAL peak_rss_GB={rss_gb():.2f} total_time={time.time()-t0:.1f}s", flush=True
    )


if __name__ == "__main__":
    main()
