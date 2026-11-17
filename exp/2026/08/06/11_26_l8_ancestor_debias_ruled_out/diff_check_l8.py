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
        amplitude=0.0,
        sigma=None,
    )
    return alat, cl


def run_one(enable_debias, max_level=8):
    _lat.AdaptiveLattice.set_enable_ancestor_debias_correction(enable_debias)
    alat, cl = build(max_level)
    dt = _compute_dt_l0(cl)
    ts = _lat.TimeStepper.by_level(dt)
    evolver = _lat.AdaptiveLatticeEvolver(alat)
    u0 = np.array(cl.leaf_u).copy()
    evolver.step(ts)
    u1 = np.array(cl.leaf_u).copy()
    phi1 = np.array(cl.leaf_phi).copy()
    H = float(np.sum(u1**2) + 0.5 * np.sum(phi1**2))
    H0 = float(np.sum(u0**2))
    return u1, phi1, H, H0


u1a, phi1a, Ha, H0 = run_one(False)
u1b, phi1b, Hb, _ = run_one(True)

print(
    "H0=", H0, "H(off)=", Ha, "H(on)=", Hb, "H/H0(off)=", Ha / H0, "H/H0(on)=", Hb / H0
)
print("u1 max abs diff:", np.max(np.abs(u1a - u1b)), "max abs u1:", np.max(np.abs(u1a)))
print("n cells u1 differ:", np.sum(u1a != u1b), "/", u1a.size)
print("relative u diff:", np.max(np.abs(u1a - u1b)) / np.max(np.abs(u1a)))
