"""
Isolate WHY organic's init_spheromak_ic and refined's init_spheromak_ic_refinable
give different phi_rms at max_level=1, on the SAME actual organic mesh.

Builds organic's mesh exactly as sweep_neutron_decay_cross_scale.py's
_build_and_run does (build_bcc_sphere -> pre-adapt warm-up rounds -> re-seed
pristine IC on the converged mesh), matching probe_refined_vs_organic_amr.py's
max_level=1 parameters. Then, on that SAME mesh (same leaf_pos/leaf_level),
recomputes phi TWO ways:
  (a) organic's actual init_spheromak_ic (fixed level-0-sized hop for every leaf)
  (b) a per-leaf cell_size-scaled hop (the refined script's fix)
to see whether (b) on organic's own mesh lands close to refined's independently
measured phi_rms (2.2314e-2 from the probe log), confirming the entire gap is
explained by the known hop-scaling difference and not a separate bug.
"""

import sys

sys.path.insert(0, ".")

import numpy as np
from scipy.special import spherical_jn

from equilibrium import lattice as _lat
from tst.substrate.adaptive.adaptive_flux_tube import (
    _FACE_HOP,
    _FACE_NORM,
    init_spheromak_ic,
)

SPLIT_RATIO = 3.0
R = 80.0
scale = 4.8  # matches probe's scale @ L1
max_level = 1
split_hi = 0.02
split_lo = 0.0
split_ref = 1e-6
B0 = 1.0
lam_target = 1.0
u_max = 1.0
amplitude = 1.0
sigma = None

alat = _lat.AdaptiveLattice()
alat.add_cluster()
alat.set_n_threads(8)
cl = alat[0]
_lat.build_bcc_sphere(cl, R, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
alat.rebuild(0)

r_all = np.linalg.norm(np.array(cl.leaf_pos), axis=1)
r_max = float(r_all.max())
r_inner = r_max * 0.85
cl.set_bc(
    _lat.BoundaryCondition.quadratic_sponge(
        target=0.0, damp_max=0.003, r_all=r_all, r_inner=r_inner
    )
)


def _seed_ic():
    return init_spheromak_ic(
        cl,
        R_sph=1.0,
        scale=scale,
        B0=B0,
        lam_target=lam_target,
        u_max=u_max,
        amplitude=amplitude,
        sigma=sigma,
    )


b_max = _seed_ic()
dt = 5.1282e-3

sc = _lat.SplitCriterion.relative_u(hi=split_hi, lo=split_lo, ref=split_ref)
has_fine = int(np.max(np.array(cl.leaf_level))) > 0
ts = _lat.TimeStepper.by_level(dt) if has_fine else _lat.TimeStepper.fixed(dt)
evolver = _lat.AdaptiveLatticeEvolver(alat)

pre_info = _lat.DSTInfo()
pre_info.max_level = max_level
pre_info.split_interval = 1
cl.info = pre_info
cl.set_split_criterion(sc)
for i in range(50):
    evolver.step(ts)
    n_before = cl.n_leaves
    alat.adapt(0)
    if cl.n_leaves == n_before:
        print(f"pre-adapt converged after {i+1} rounds, n_leaves={cl.n_leaves}")
        break

# Re-seed pristine IC on the converged mesh -- this is organic's ACTUAL final IC.
b_max = _seed_ic()
phi_organic_actual = np.array(cl.leaf_phi).copy()
levels = np.array(cl.leaf_level)
pos = np.array(cl.leaf_pos)
N = pos.shape[0]
print(f"n_leaves={N}  level counts: {np.bincount(levels)}")
print(
    f"organic ACTUAL phi_rms = {float(np.sqrt(np.mean(phi_organic_actual**2))):.6e}  "
    f"(probe log value: 2.2254e-02)"
)

# --- Recompute phi on the SAME mesh with a per-leaf cell_size-scaled hop -----
centre = np.zeros(3)
k_ = 4.493 / 1.0


def _B(p3):
    p = (p3 - centre) / scale
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    B = np.zeros_like(p)
    mask = (r <= 1.0) & (r > 0)
    rm = r[mask]
    cf = B0 * spherical_jn(1, k_ * rm) / rm
    B[mask, 0] = cf * (z[mask] * x[mask] / rm - y[mask])
    B[mask, 1] = cf * (z[mask] * y[mask] / rm + x[mask])
    B[mask, 2] = cf * (-(x[mask] ** 2 + y[mask] ** 2) / rm)
    return B


cell_size = 1.0 / (SPLIT_RATIO**levels)
r_cell = np.linalg.norm((pos - centre) / scale, axis=1)

hop_scaled = _FACE_HOP[None, :, :] * cell_size[:, None, None]
midpoints_scaled = pos[:, None, :] + 0.5 * hop_scaled
B_mid_scaled = _B(midpoints_scaled.reshape(-1, 3)).reshape(N, 14, 3)
phi_scaled = np.einsum("jkd,kd->kj", B_mid_scaled, _FACE_NORM)
phi_max_scaled = float(np.abs(phi_scaled).max()) or 1.0
phi_scaled *= amplitude / phi_max_scaled
nbr_r_scaled = np.linalg.norm((pos[:, None, :] + hop_scaled - centre) / scale, axis=2)
phi_scaled[(nbr_r_scaled > 1.0).T] = 0.0
phi_scaled[:, r_cell > 1.0] = 0.0

print(
    f"same-mesh, cell_size-SCALED hop, phi_rms = "
    f"{float(np.sqrt(np.mean(phi_scaled**2))):.6e}  "
    f"(refined's independently-measured value: 2.2314e-02)"
)

# --- Also recompute organic's own fixed-hop formula directly here as a sanity
# cross-check that this script's re-implementation matches the C++/Python
# path's actual stored value exactly.
hop_fixed = _FACE_HOP[None, :, :]
midpoints_fixed = pos[:, None, :] + 0.5 * hop_fixed
B_mid_fixed = _B(midpoints_fixed.reshape(-1, 3)).reshape(N, 14, 3)
phi_fixed = np.einsum("jkd,kd->kj", B_mid_fixed, _FACE_NORM)
phi_max_fixed = float(np.abs(phi_fixed).max()) or 1.0
phi_fixed *= amplitude / phi_max_fixed
nbr_r_fixed = np.linalg.norm((pos[:, None, :] + hop_fixed - centre) / scale, axis=2)
phi_fixed[(nbr_r_fixed > 1.0).T] = 0.0
phi_fixed[:, r_cell > 1.0] = 0.0
print(
    f"same-mesh, FIXED (organic-style) hop reimplementation, phi_rms = "
    f"{float(np.sqrt(np.mean(phi_fixed**2))):.6e}  (sanity cross-check vs organic ACTUAL above)"
)

# Breakdown by level: how much does each level's own phi differ between the
# two hop conventions?
for lev in sorted(set(levels.tolist())):
    m = levels == lev
    rms_fixed = float(np.sqrt(np.mean(phi_fixed[:, m] ** 2)))
    rms_scaled = float(np.sqrt(np.mean(phi_scaled[:, m] ** 2)))
    print(
        f"  level {lev}: n_leaves={int(m.sum())}  "
        f"phi_rms(fixed-hop)={rms_fixed:.6e}  phi_rms(scaled-hop)={rms_scaled:.6e}  "
        f"ratio={rms_scaled / rms_fixed if rms_fixed > 0 else float('nan'):.4f}"
    )
