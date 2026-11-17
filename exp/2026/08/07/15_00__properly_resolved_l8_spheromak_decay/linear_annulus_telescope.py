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
    r_hi = MAX_LEVEL - lvl + 1  # lvl=8 -> [0,1), lvl=1 -> [7,8)
    r_lo = MAX_LEVEL - lvl

    def region_fn(cl, r_lo=r_lo, r_hi=r_hi):
        pos = np.array(cl.leaf_pos)
        r = np.linalg.norm(pos, axis=1)
        return (r >= r_lo) & (r < r_hi)

    refine_region_to_convergence(alat, cl, region_fn, stage_max_level=lvl)
    levels = np.array(cl.leaf_level)
    print(
        f"[level {lvl}] shell=[{r_lo},{r_hi})  n_leaves={cl.n_leaves}  "
        f"max_level_present={levels.max()}  level_hist={np.bincount(levels).tolist()}  "
        f"peak_rss_GB={rss_gb():.2f}",
        flush=True,
    )

print(f"\nFINAL n_leaves={cl.n_leaves}  peak_rss_GB={rss_gb():.2f}")
