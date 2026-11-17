import sys

sys.path.insert(0, ".")
import numpy as np

from equilibrium import lattice as _lat
from run.substrate.adaptive.refined_neutron_decay import (
    SPLIT_RATIO,
    _build_telescope,
    init_spheromak_ic_refinable,
)
from run.substrate.adaptive.sweep_spheromak_decay_cross_scale import _compute_dt_l0

R = 30.0
BASE_SCALE = 1.3606e-3
MAX_LEVEL = 8
MARGIN = 1.3  # modest margin -- generous 1.68x (used for mult=1/2) gets expensive fast at larger mult

for mult in (1, 2, 4):
    scale = BASE_SCALE * mult
    region_r0 = MARGIN * scale * SPLIT_RATIO ** (MAX_LEVEL - 1)
    finest_region_r = region_r0 / (SPLIT_RATIO ** (MAX_LEVEL - 1))
    cells_across = 2 * scale / (1.0 / SPLIT_RATIO**MAX_LEVEL)

    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    alat.set_n_threads(8)
    cl = alat[0]
    _lat.build_bcc_sphere(cl, R, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)
    _build_telescope(alat, cl, region_r0, MAX_LEVEL)
    init_spheromak_ic_refinable(
        cl,
        R_sph=1.0,
        scale=scale,
        B0=1.0,
        lam_target=1.0,
        u_max=1.0,
        amplitude=1.0,
        sigma=None,
    )
    cl.set_bc(_lat.BoundaryCondition.dirichlet(0.0))

    u0 = np.array(cl.leaf_u).copy()
    phi0 = np.array(cl.leaf_phi).copy()
    H0 = float(np.sum(u0**2) + 0.5 * np.sum(phi0**2))
    dt = _compute_dt_l0(cl, D1=1.0)
    ts = _lat.TimeStepper.by_level(dt)
    evolver = _lat.AdaptiveLatticeEvolver(alat)
    evolver.step(ts)
    u1 = np.array(cl.leaf_u)
    phi1 = np.array(cl.leaf_phi)
    H1 = float(np.sum(u1**2) + 0.5 * np.sum(phi1**2))
    print(
        f"mult={mult}  scale={scale:.4e}  cells_across_diameter={cells_across:.1f}  "
        f"N_leaves={cl.n_leaves}  H0={H0:.4e}  H1={H1:.4e}  H/H0={H1/H0:.4f}",
        flush=True,
    )
