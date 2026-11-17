import sys

sys.path.insert(0, ".")
import resource

import numpy as np

from equilibrium import lattice as _lat
from tst.substrate.adaptive import refine_region_to_convergence

R = 30.0
MAX_LEVEL = 8


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def hop_distance_from_center(cl, max_hops):
    pos = np.array(cl.leaf_pos)
    N = pos.shape[0]
    r = np.linalg.norm(pos, axis=1)
    start_j = int(np.argmin(r))
    leaf_nb_g = cl.leaf_nb
    leaf_indices = np.array(cl.leaf_indices)
    max_g = int(leaf_indices.max())
    g_to_j = -np.ones(max_g + 1, dtype=np.int64)
    g_to_j[leaf_indices] = np.arange(N)
    nb_g_clipped = np.where(leaf_nb_g >= 0, leaf_nb_g, 0)
    nb_j = np.where(leaf_nb_g >= 0, g_to_j[nb_g_clipped], -1)

    hop = np.full(N, -1, dtype=np.int64)
    hop[start_j] = 0
    frontier = np.array([start_j])
    for d in range(1, max_hops + 1):
        nbrs = nb_j[frontier].ravel()
        nbrs = nbrs[nbrs >= 0]
        nbrs = nbrs[hop[nbrs] == -1]
        nbrs = np.unique(nbrs)
        if len(nbrs) == 0:
            break
        hop[nbrs] = d
        frontier = nbrs
    return hop


alat = _lat.AdaptiveLattice()
alat.add_cluster()
alat.set_n_threads(8)
cl = alat[0]
_lat.build_bcc_sphere(cl, R, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
alat.rebuild(0)
print(f"[base] n_leaves={cl.n_leaves}  peak_rss_GB={rss_gb():.2f}", flush=True)

for lvl in range(1, MAX_LEVEL + 1):
    target_hops = MAX_LEVEL - lvl

    def region_fn(cl, target_hops=target_hops):
        hop = hop_distance_from_center(cl, target_hops)
        return (hop >= 0) & (hop <= target_hops)

    refine_region_to_convergence(alat, cl, region_fn, stage_max_level=lvl)
    print(
        f"[level {lvl}] target_hops={target_hops}  n_leaves={cl.n_leaves}  "
        f"peak_rss_GB={rss_gb():.2f}",
        flush=True,
    )

levels = np.array(cl.leaf_level)
print(f"\nFINAL n_leaves={cl.n_leaves}  peak_rss_GB={rss_gb():.2f}")
print(f"level histogram: {np.bincount(levels).tolist()}")
