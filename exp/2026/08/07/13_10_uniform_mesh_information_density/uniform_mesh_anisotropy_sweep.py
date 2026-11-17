import sys

sys.path.insert(0, ".")
import time

import numpy as np
from scipy.spatial import cKDTree

from equilibrium import lattice as _lat
from run.substrate.adaptive.refined_neutron_decay import init_spheromak_ic_refinable
from run.substrate.adaptive.sweep_spheromak_decay_cross_scale import _compute_dt_l0

FILL_FRAC = 0.9  # spheromak radius = 90% of domain radius
N_STEPS = 5
R_VALUES = [20, 40, 80, 160]
SAMPLE_FRACS = [0.5, 0.9]  # fraction of spheromak scale to sample the field at

AXIS_DIRS = np.array(
    [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ],
    dtype=float,
)

DIAG_DIRS = np.array(
    [
        [1, 1, 1],
        [1, 1, -1],
        [1, -1, 1],
        [1, -1, -1],
        [-1, 1, 1],
        [-1, 1, -1],
        [-1, -1, 1],
        [-1, -1, -1],
    ],
    dtype=float,
)
DIAG_DIRS /= np.linalg.norm(DIAG_DIRS, axis=1, keepdims=True)


def anisotropy_ratio(tree, field, scale, frac):
    r_s = frac * scale
    axis_pts = r_s * AXIS_DIRS
    diag_pts = r_s * DIAG_DIRS
    _, axis_idx = tree.query(axis_pts)
    _, diag_idx = tree.query(diag_pts)
    axis_vals = field[axis_idx]
    diag_vals = field[diag_idx]
    axis_mean = float(np.mean(axis_vals))
    diag_mean = float(np.mean(diag_vals))
    denom = 0.5 * (abs(axis_mean) + abs(diag_mean))
    rel = abs(axis_mean - diag_mean) / denom if denom > 0 else float("nan")
    return axis_mean, diag_mean, rel


print(
    f"{'R':>5}  {'cells_across':>12}  {'N_leaves':>10}  {'build_s':>8}  {'step_s':>8}  "
    + "  ".join(f"phi_asym@{f}" for f in SAMPLE_FRACS)
    + "  "
    + "  ".join(f"u_asym@{f}" for f in SAMPLE_FRACS),
    flush=True,
)

for R in R_VALUES:
    scale = FILL_FRAC * R
    cells_across = 2 * scale / 1.0

    t0 = time.time()
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    alat.set_n_threads(8)
    cl = alat[0]
    _lat.build_bcc_sphere(cl, R, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)
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
    build_s = time.time() - t0

    dt = _compute_dt_l0(cl, D1=1.0)
    ts = _lat.TimeStepper.fixed(dt)
    evolver = _lat.AdaptiveLatticeEvolver(alat)

    t0 = time.time()
    for _ in range(N_STEPS):
        evolver.step(ts)
    step_s = time.time() - t0

    pos = np.array(cl.leaf_pos)
    tree = cKDTree(pos)
    phi = np.array(cl.leaf_phi)
    phi_mag = np.sqrt(np.sum(phi**2, axis=0))
    u_mag = np.abs(np.array(cl.leaf_u))

    phi_results = [anisotropy_ratio(tree, phi_mag, scale, f) for f in SAMPLE_FRACS]
    u_results = [anisotropy_ratio(tree, u_mag, scale, f) for f in SAMPLE_FRACS]

    row = (
        f"{R:>5}  {cells_across:>12.1f}  {cl.n_leaves:>10}  {build_s:>8.2f}  {step_s:>8.2f}  "
        + "  ".join(f"{r[2]:.4f}" for r in phi_results)
        + "  "
        + "  ".join(f"{r[2]:.4f}" for r in u_results)
    )
    print(row, flush=True)
    del alat, cl, tree, pos, phi, phi_mag, u_mag
