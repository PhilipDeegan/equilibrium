import sys

sys.path.insert(0, ".")
import resource

import numpy as np
from scipy.special import spherical_jn

from equilibrium import lattice as _lat
from tst.substrate.adaptive import refine_region_to_convergence

R = 30.0
MAX_LEVEL = 8
SCALE = 0.022862  # chosen so cells_across = 300 at L8 (2*SCALE*3**8)
R_SPH = 1.0
B0 = 1.0
K_ = 4.493 / R_SPH


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def b_mag_at(pos):
    p = pos / SCALE
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    out = np.zeros(pos.shape[0])
    mask = (r <= R_SPH) & (r > 0)
    rm = r[mask]
    cf = B0 * spherical_jn(1, K_ * rm) / rm
    bx = cf * (z[mask] * x[mask] / rm - y[mask])
    by = cf * (z[mask] * y[mask] / rm + x[mask])
    bz = cf * (-(x[mask] ** 2 + y[mask] ** 2) / rm)
    out[mask] = np.sqrt(bx**2 + by**2 + bz**2)
    return out


_call_count = [0]


def strongest_field_cell(cl):
    _call_count[0] += 1
    pos = np.array(cl.leaf_pos)
    bmag = b_mag_at(pos)
    j = int(np.argmax(bmag))
    mask = np.zeros(pos.shape[0], dtype=bool)
    mask[j] = True
    print(
        f"    call={_call_count[0]}  n_leaves={pos.shape[0]}  picked_j={j}  "
        f"pos={pos[j]}  bmag={bmag[j]:.6e}",
        flush=True,
    )
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
    refine_region_to_convergence(alat, cl, strongest_field_cell, stage_max_level=lvl)
    levels = np.array(cl.leaf_level)
    pos = np.array(cl.leaf_pos)
    peak_j = int(np.argmax(b_mag_at(pos)))
    print(
        f"[level {lvl}] n_leaves={cl.n_leaves}  max_level_present={levels.max()}  "
        f"level_hist={np.bincount(levels).tolist()}  peak_cell_pos={pos[peak_j]}  "
        f"peak_cell_level={levels[peak_j]}  peak_rss_GB={rss_gb():.2f}",
        flush=True,
    )

print(f"\nFINAL n_leaves={cl.n_leaves}  peak_rss_GB={rss_gb():.2f}")
