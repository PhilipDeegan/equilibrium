import sys

sys.path.insert(0, ".")
import numpy as np

from equilibrium import lattice as _lat
from run.substrate.adaptive.refined_neutron_decay import init_spheromak_ic_refinable
from run.substrate.adaptive.sweep_spheromak_decay_cross_scale import _compute_dt_l0

scale = 1.3606e-3  # same real physical core size as the L8 telescope run
d1_fine = 1.0 / 3.0**8  # same finest cell size as an L8 telescope leaf

for R_cells in (20, 40, 60):
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    alat.set_n_threads(8)
    cl = alat[0]
    _lat.build_bcc_sphere(cl, R_cells, d1=d1_fine, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)
    phys_R = R_cells * d1_fine
    print(
        f"R_cells={R_cells}  phys_R={phys_R:.4e} (scale={scale:.4e})  d1={d1_fine:.4e}  "
        f"N_leaves={cl.n_leaves}  levels_present={sorted(set(np.array(cl.leaf_level).tolist()))}"
    )
    if cl.n_leaves > 1_500_000 or cl.n_leaves < 2:
        print("  SKIP")
        continue
    init_spheromak_ic_refinable(
        cl,
        R_sph=1.0,
        scale=scale,
        B0=1.0,
        lam_target=1.0,
        u_max=1.0,
        amplitude=0.0,
        sigma=None,
    )
    u0 = np.array(cl.leaf_u).copy()
    phi0 = np.array(cl.leaf_phi).copy()
    H0 = float(np.sum(u0**2) + 0.5 * np.sum(phi0**2))
    dt = _compute_dt_l0(cl, D1=d1_fine)
    ts = _lat.TimeStepper.fixed(dt)
    evolver = _lat.AdaptiveLatticeEvolver(alat)
    evolver.step(ts)
    u1 = np.array(cl.leaf_u)
    phi1 = np.array(cl.leaf_phi)
    H1 = float(np.sum(u1**2) + 0.5 * np.sum(phi1**2))
    ratio = H1 / H0 if H0 > 0 else float("nan")
    print(f"  H0={H0:.6e}  H1={H1:.6e}  H/H0={ratio:.6f}  dt={dt:.4e}")
