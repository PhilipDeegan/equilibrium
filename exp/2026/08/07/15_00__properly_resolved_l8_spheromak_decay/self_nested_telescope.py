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


def nearest_to_origin(cl):
    pos = np.array(cl.leaf_pos)
    r = np.linalg.norm(pos, axis=1)
    mask = np.zeros(pos.shape[0], dtype=bool)
    mask[int(np.argmin(r))] = True
    return mask


alat = _lat.AdaptiveLattice()
alat.add_cluster()
alat.set_n_threads(8)
cl = alat[0]
_lat.build_bcc_sphere(cl, R, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
alat.rebuild(0)
print(f"[base] n_leaves={cl.n_leaves}  peak_rss_GB={rss_gb():.2f}", flush=True)

cl.info.split_hi = 0.5
cl.info.split_lo = 0.0

for lvl in range(1, MAX_LEVEL + 1):
    refine_region_to_convergence(alat, cl, nearest_to_origin, stage_max_level=lvl)
    levels = np.array(cl.leaf_level)
    print(
        f"[level {lvl}] n_leaves={cl.n_leaves}  max_level_present={levels.max()}  "
        f"level_hist={np.bincount(levels).tolist()}  peak_rss_GB={rss_gb():.2f}",
        flush=True,
    )

print(f"\nFINAL n_leaves={cl.n_leaves}  peak_rss_GB={rss_gb():.2f}")
