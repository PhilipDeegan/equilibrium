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


def build(
    max_level, R=20.0, region_r0=5.0, scale_override=1.3606e-3, scale_ref_level=8
):
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    alat.set_n_threads(8)
    cl = alat[0]
    _lat.build_bcc_sphere(cl, R, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)
    if max_level > 0:
        _build_telescope(alat, cl, region_r0, max_level)
    scale = scale_override * SPLIT_RATIO ** (scale_ref_level - max_level)
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
    return alat, cl


def run_one(max_level, zero_u=False, zero_phi=False):
    alat, cl = build(max_level)
    if zero_u:
        cl.set_leaf_u(np.zeros(cl.n_leaves))
    if zero_phi:
        cl.set_leaf_phi(np.zeros((14, cl.n_leaves)))
    u0 = np.array(cl.leaf_u).copy()
    phi0 = np.array(cl.leaf_phi).copy()
    H0 = float(np.sum(u0**2) + 0.5 * np.sum(phi0**2))
    dt = _compute_dt_l0(cl)
    ts = _lat.TimeStepper.by_level(dt)
    evolver = _lat.AdaptiveLatticeEvolver(alat)
    evolver.step(ts)
    u1 = np.array(cl.leaf_u)
    phi1 = np.array(cl.leaf_phi)
    H1 = float(np.sum(u1**2) + 0.5 * np.sum(phi1**2))
    return H0, H1, (H1 / H0 if H0 > 0 else float("nan"))


for L in (4, 6, 8):
    H0_full, H1_full, r_full = run_one(L, zero_u=False, zero_phi=False)
    H0_nou, H1_nou, r_nou = run_one(L, zero_u=True, zero_phi=False)
    H0_nophi, H1_nophi, r_nophi = run_one(L, zero_u=False, zero_phi=True)
    print(
        f"L={L}  full: H0={H0_full:.4e} H1={H1_full:.4e} H/H0={r_full:.4f}   "
        f"u=0 (phi+lam only): H0={H0_nou:.4e} H1={H1_nou:.4e} H/H0={r_nou:.4f}   "
        f"phi=0 (u+lam only): H0={H0_nophi:.4e} H1={H1_nophi:.4e} H/H0={r_nophi:.4f}"
    )
