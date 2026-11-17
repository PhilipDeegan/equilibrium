import sys

sys.path.insert(0, ".")
import resource
import time

import numpy as np

from equilibrium import lattice as _lat
from run.substrate.adaptive.refined_neutron_decay import (
    _build_telescope,
    init_spheromak_ic_refinable,
)
from run.substrate.adaptive.sweep_spheromak_decay_cross_scale import _compute_dt_l0

R = 90.0
MAX_LEVEL = 8
SCALE = 0.022862
REGION_R0 = 65.000


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


t0 = time.time()
alat = _lat.AdaptiveLattice()
alat.add_cluster()
alat.set_n_threads(8)
cl = alat[0]
_lat.build_bcc_sphere(cl, R, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
alat.rebuild(0)
print(
    f"[bare_sphere] n_leaves={cl.n_leaves}  t={time.time()-t0:.1f}s  peak_rss_GB={rss_gb():.2f}",
    flush=True,
)

_build_telescope(alat, cl, REGION_R0, MAX_LEVEL)
print(
    f"[telescope_built] n_leaves={cl.n_leaves}  t={time.time()-t0:.1f}s  peak_rss_GB={rss_gb():.2f}",
    flush=True,
)

init_spheromak_ic_refinable(
    cl,
    R_sph=1.0,
    scale=SCALE,
    B0=1.0,
    lam_target=1.0,
    u_max=1.0,
    amplitude=1.0,
    sigma=None,
)
cl.set_bc(_lat.BoundaryCondition.dirichlet(0.0))
print(f"[ic_seeded] t={time.time()-t0:.1f}s  peak_rss_GB={rss_gb():.2f}", flush=True)

u0 = np.array(cl.leaf_u)
phi0 = np.array(cl.leaf_phi)
levels0 = np.array(cl.leaf_level)
H0 = float(np.sum(u0**2) + 0.5 * np.sum(phi0**2))
print(
    f"H0={H0:.6e}  max_level_present={levels0.max()}  levels_hist={np.bincount(levels0).tolist()}"
)

dt = _compute_dt_l0(cl, D1=1.0)
print(f"dt={dt:.6e}")
