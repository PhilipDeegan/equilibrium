import sys

sys.path.insert(0, ".")
import time

import numpy as np

from equilibrium import lattice as _lat
from tst.substrate.adaptive.adaptive_flux_tube import leaf_nb_j

STEP_DIR = "/bk/l5_5step_save_each_step"
SNAPSHOT_BASE = "/bk/l5_core_diameter9_cells_across_220.h5"


def fine_radial_profile(r, field, r_max=3.0, n_bins=60):
    bins = np.linspace(0.0, r_max, n_bins + 1)
    idx = np.clip(np.digitize(r, bins) - 1, 0, n_bins - 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    mean = np.full(n_bins, np.nan)
    pmax = np.full(n_bins, np.nan)
    n_in_bin = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        mask = idx == b
        n_in_bin[b] = mask.sum()
        if mask.any():
            vals = field[mask]
            mean[b] = vals.mean()
            pmax[b] = np.abs(vals).max()
    return centers, mean, pmax, n_in_bin


def level_pair_breakdown(cl):
    u = np.array(cl.leaf_u)
    levels = np.array(cl.leaf_level)
    pos = np.array(cl.leaf_pos)
    r = np.linalg.norm(pos, axis=1)
    nb_j = leaf_nb_j(cl)
    N = u.shape[0]
    valid = nb_j >= 0
    self_idx = np.broadcast_to(np.arange(N)[:, None], nb_j.shape)
    si = self_idx[valid]
    nj = nb_j[valid]
    du = np.abs(u[si] - u[nj])
    lev_i = levels[si]
    lev_j = levels[nj]
    r_pair = np.minimum(r[si], r[nj])

    pairs = sorted({(int(min(a, b)), int(max(a, b))) for a, b in zip(lev_i, lev_j)})
    print(f"  level-level pair types present: {pairs}", flush=True)
    for lo, hi in pairs:
        mask = (np.minimum(lev_i, lev_j) == lo) & (np.maximum(lev_i, lev_j) == hi)
        n = int(mask.sum())
        if n == 0:
            continue
        dv = du[mask]
        rv = r_pair[mask]
        tag = "SAME" if lo == hi else "CROSS"
        print(
            f"  [{tag} L{lo}-L{hi}] n={n:9d}  mean|du|={dv.mean():.4e}  "
            f"p95={np.percentile(dv, 95):.4e}  max={dv.max():.4e}  "
            f"r_range=[{rv.min():.3f}, {rv.max():.3f}]",
            flush=True,
        )


def analyze(label, path):
    t0 = time.time()
    alat = _lat.AdaptiveLattice.load(path)
    cl = alat[0]
    print(
        f"\n=== [{label}] {path} n_leaves={cl.n_leaves} load_t={time.time()-t0:.1f}s ===",
        flush=True,
    )

    levels = np.array(cl.leaf_level)
    lvl_hist = np.bincount(levels)
    print(f"[{label}] leaf count by level: {lvl_hist.tolist()}", flush=True)

    u = np.array(cl.leaf_u)
    pos = np.array(cl.leaf_pos)
    r = np.linalg.norm(pos, axis=1)

    centers, mean_u, pmax_u, n_in_bin = fine_radial_profile(r, u, r_max=3.0, n_bins=60)
    print(f"[{label}] fine radial |u| profile (r in [0,3], 60 bins):", flush=True)
    for c, m, pm, n in zip(centers, mean_u, pmax_u, n_in_bin):
        if n > 0:
            print(
                f"    r={c:5.3f}  n={n:8d}  mean_u={m: .4e}  max|u|={pm:.4e}",
                flush=True,
            )

    print(f"[{label}] level-pair junction breakdown:", flush=True)
    level_pair_breakdown(cl)


def main():
    analyze("IC", SNAPSHOT_BASE)
    analyze("step5", f"{STEP_DIR}/step_05.h5")


if __name__ == "__main__":
    main()
