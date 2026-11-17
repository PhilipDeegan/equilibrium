import sys

sys.path.insert(0, ".")
import gc
import time

import numpy as np

from equilibrium import lattice as _lat

SNAPSHOT_BASE = "/bk/l5_core_diameter9_cells_across_220.h5"
STEP_DIR = "/bk/l5_5step_save_each_step"
N_STEPS = 5
THRESHOLDS = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
N_RBINS = 240
R_MAX = 3.0


def front_radii(path, label):
    t0 = time.time()
    alat = _lat.AdaptiveLattice.load(path)
    cl = alat[0]
    pos = np.array(cl.leaf_pos)
    u = np.array(cl.leaf_u)
    r = np.linalg.norm(pos, axis=1)

    bins = np.linspace(0.0, R_MAX, N_RBINS + 1)
    idx = np.clip(np.digitize(r, bins) - 1, 0, N_RBINS - 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    pmax = np.zeros(N_RBINS)
    for b in range(N_RBINS):
        mask = idx == b
        if mask.any():
            pmax[b] = np.abs(u[mask]).max()

    fronts = {}
    for th in THRESHOLDS:
        above = np.where(pmax >= th)[0]
        fronts[th] = float(centers[above].max()) if above.size else float("nan")

    print(
        f"[{label}] load_t={time.time()-t0:.1f}s  fronts(max|u|>=thresh): "
        + ", ".join(f"{th:.0e}->r={fronts[th]:.4f}" for th in THRESHOLDS),
        flush=True,
    )

    del cl, alat
    gc.collect()
    return fronts


def main():
    rows = []
    rows.append((0, front_radii(SNAPSHOT_BASE, "IC")))
    for i in range(1, N_STEPS + 1):
        rows.append((i, front_radii(f"{STEP_DIR}/step_{i:02d}.h5", f"step{i}")))

    print("\nstep  " + "  ".join(f"r(>{th:.0e})" for th in THRESHOLDS))
    for step_i, fronts in rows:
        print(f"{step_i:4d}  " + "  ".join(f"{fronts[th]:8.4f}" for th in THRESHOLDS))


if __name__ == "__main__":
    main()
