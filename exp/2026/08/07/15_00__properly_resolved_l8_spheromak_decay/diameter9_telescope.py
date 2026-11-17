import sys

sys.path.insert(0, ".")
import resource

import numpy as np

from equilibrium import lattice as _lat
from tst.substrate.adaptive import refine_region_to_convergence

R = 30.0
CORE_LEVEL = 5


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


alat = _lat.AdaptiveLattice()
alat.add_cluster()
alat.set_n_threads(8)
cl = alat[0]
_lat.build_bcc_sphere(cl, R, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
alat.rebuild(0)
n0 = cl.n_leaves
print(f"[base] n_leaves={n0}  peak_rss_GB={rss_gb():.2f}", flush=True)

# BFS on the BASE (unrefined) L0 mesh -- computed once, fixed hop-distance
# per ORIGINAL leaf index, independent of later splitting.
pos0 = np.array(cl.leaf_pos)
r0 = np.linalg.norm(pos0, axis=1)
start_j = int(np.argmin(r0))
leaf_nb_g = cl.leaf_nb
leaf_indices = np.array(cl.leaf_global_indices)
max_g = int(leaf_indices.max())
g_to_j = -np.ones(max_g + 1, dtype=np.int64)
g_to_j[leaf_indices] = np.arange(n0)
nb_g_clipped = np.where(leaf_nb_g >= 0, leaf_nb_g, 0)
nb_j0 = np.where(leaf_nb_g >= 0, g_to_j[nb_g_clipped], -1)

hop = np.full(n0, -1, dtype=np.int64)
hop[start_j] = 0
frontier = np.array([start_j])
for d in range(1, CORE_LEVEL + 1):
    nbrs = nb_j0[frontier].ravel()
    nbrs = nbrs[nbrs >= 0]
    nbrs = nbrs[hop[nbrs] == -1]
    nbrs = np.unique(nbrs)
    if len(nbrs) == 0:
        break
    hop[nbrs] = d
    frontier = nbrs

for d in range(CORE_LEVEL + 1):
    print(f"  hop={d}: n_original_cells={int(np.sum(hop == d))}")

orig_g_at_hop = {d: leaf_indices[hop == d] for d in range(CORE_LEVEL + 1)}

cl.info.split_hi = 0.5
cl.info.split_lo = 0.0

# Refine from shallow (hop=CORE_LEVEL-1, target L1) to deep (hop=0, target CORE_LEVEL),
# same outside-in order as _build_telescope, using membership in the FIXED
# original-g set for each hop (works even as leaf indices reshuffle, since
# we test descent from the original g via cl.leaf_global_indices each round).
for target_level in range(1, CORE_LEVEL + 1):
    d = CORE_LEVEL - target_level
    target_roots = orig_g_at_hop[d]

    def region_fn(cl, target_roots=target_roots):
        # leaf_ancestry packs (root_g << 32 | path); root_g is the ORIGINAL
        # L0 ancestor and stays stable across splits, unlike leaf_global_indices.
        roots = np.array(cl.leaf_ancestry) >> 32
        return np.isin(roots, target_roots)

    refine_region_to_convergence(alat, cl, region_fn, stage_max_level=target_level)
    levels = np.array(cl.leaf_level)
    print(
        f"[hop={d} -> L{target_level}] n_leaves={cl.n_leaves}  "
        f"level_hist={np.bincount(levels).tolist()}  peak_rss_GB={rss_gb():.2f}",
        flush=True,
    )

print(f"\nFINAL n_leaves={cl.n_leaves}  peak_rss_GB={rss_gb():.2f}")
