import sys

sys.path.insert(0, ".")
import gc
import time

import numpy as np

from equilibrium import lattice as _lat
from tst.substrate.adaptive.adaptive_flux_tube import leaf_nb_j

SNAPSHOT_BASE = "/bk/l5_core_diameter9_cells_across_220.h5"
STEP_DIR = "/bk/l5_5step_save_each_step"
N_STEPS = 5
N_RADIAL_BINS = 40
R_MAX = 30.0


def radial_profile(r, field, n_bins=N_RADIAL_BINS, r_max=R_MAX):
    bins = np.linspace(0.0, r_max, n_bins + 1)
    idx = np.clip(np.digitize(r, bins) - 1, 0, n_bins - 1)
    mean = np.full(n_bins, np.nan)
    std = np.full(n_bins, np.nan)
    pmax = np.full(n_bins, np.nan)
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            vals = field[mask]
            mean[b] = vals.mean()
            std[b] = vals.std()
            pmax[b] = np.abs(vals).max()
    return bins, mean, std, pmax


def junction_diff_stats(cl):
    """Compare |u[i]-u[nb]| across same-level vs cross-level face pairs."""
    u = np.array(cl.leaf_u)
    levels = np.array(cl.leaf_level)
    nb_j = leaf_nb_j(cl)  # (N, 14)
    N = u.shape[0]
    valid = nb_j >= 0
    self_idx = np.broadcast_to(np.arange(N)[:, None], nb_j.shape)
    si = self_idx[valid]
    nj = nb_j[valid]
    du = np.abs(u[si] - u[nj])
    cross = levels[si] != levels[nj]

    def stats(x):
        if x.size == 0:
            return dict(n=0, mean=np.nan, median=np.nan, p95=np.nan, max=np.nan)
        return dict(
            n=int(x.size),
            mean=float(x.mean()),
            median=float(np.median(x)),
            p95=float(np.percentile(x, 95)),
            max=float(x.max()),
        )

    return stats(du[cross]), stats(du[~cross])


def analyze(label, path):
    t0 = time.time()
    alat = _lat.AdaptiveLattice.load(path)
    cl = alat[0]
    print(
        f"[{label}] loaded {path} n_leaves={cl.n_leaves} t={time.time()-t0:.1f}s",
        flush=True,
    )

    u = np.array(cl.leaf_u)
    phi = np.array(cl.leaf_phi)
    pos = np.array(cl.leaf_pos)
    r = np.linalg.norm(pos, axis=1)

    n_nan_u = int(np.isnan(u).sum())
    n_inf_u = int(np.isinf(u).sum())
    n_nan_phi = int(np.isnan(phi).sum())
    n_inf_phi = int(np.isinf(phi).sum())
    print(
        f"[{label}] NaN/Inf: u_nan={n_nan_u} u_inf={n_inf_u} phi_nan={n_nan_phi} phi_inf={n_inf_phi}",
        flush=True,
    )

    bins, mean_u, std_u, pmax_u = radial_profile(r, u)
    print(f"[{label}] radial |u| profile (bin_center, mean, std, max_abs):", flush=True)
    centers = 0.5 * (bins[:-1] + bins[1:])
    for c, m, s, pm in zip(centers, mean_u, std_u, pmax_u):
        if not np.isnan(m):
            print(
                f"    r={c:5.2f}  mean_u={m: .4e}  std_u={s:.4e}  max|u|={pm:.4e}",
                flush=True,
            )

    cross_stats, same_stats = junction_diff_stats(cl)
    print(
        f"[{label}] level-junction |du|: cross-level n={cross_stats['n']} "
        f"mean={cross_stats['mean']:.4e} median={cross_stats['median']:.4e} "
        f"p95={cross_stats['p95']:.4e} max={cross_stats['max']:.4e}",
        flush=True,
    )
    print(
        f"[{label}] level-junction |du|: same-level  n={same_stats['n']} "
        f"mean={same_stats['mean']:.4e} median={same_stats['median']:.4e} "
        f"p95={same_stats['p95']:.4e} max={same_stats['max']:.4e}",
        flush=True,
    )
    ratio = (
        cross_stats["p95"] / same_stats["p95"] if same_stats["p95"] else float("nan")
    )
    print(f"[{label}] cross/same p95 ratio = {ratio:.3f}", flush=True)
    print(flush=True)

    del cl, alat
    gc.collect()


def main():
    analyze("IC", SNAPSHOT_BASE)
    for i in range(1, N_STEPS + 1):
        analyze(f"step{i}", f"{STEP_DIR}/step_{i:02d}.h5")


if __name__ == "__main__":
    main()
