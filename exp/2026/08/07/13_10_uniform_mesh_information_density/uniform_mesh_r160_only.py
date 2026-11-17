import sys

sys.path.insert(0, ".")
import resource
import time

import numpy as np
from scipy.spatial import cKDTree

from equilibrium import lattice as _lat
from run.substrate.adaptive.refined_neutron_decay import init_spheromak_ic_refinable
from run.substrate.adaptive.sweep_spheromak_decay_cross_scale import _compute_dt_l0

FILL_FRAC = 0.9
N_STEPS = 5
R = 120
SAMPLE_FRACS = [0.5, 0.9]

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


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def mark(tag, t0):
    print(f"[{tag}] t={time.time()-t0:.2f}s  peak_rss_GB={rss_gb():.2f}", flush=True)


def anisotropy_ratio(tree, field, scale, frac):
    r_s = frac * scale
    _, axis_idx = tree.query(r_s * AXIS_DIRS)
    _, diag_idx = tree.query(r_s * DIAG_DIRS)
    axis_mean = float(np.mean(field[axis_idx]))
    diag_mean = float(np.mean(field[diag_idx]))
    denom = 0.5 * (abs(axis_mean) + abs(diag_mean))
    rel = abs(axis_mean - diag_mean) / denom if denom > 0 else float("nan")
    return axis_mean, diag_mean, rel


t0 = time.time()
scale = FILL_FRAC * R
cells_across = 2 * scale / 1.0
print(f"R={R}  scale={scale}  cells_across={cells_across}", flush=True)

alat = _lat.AdaptiveLattice()
alat.add_cluster()
alat.set_n_threads(8)
cl = alat[0]
_lat.build_bcc_sphere(cl, R, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
alat.rebuild(0)
mark("built_sphere", t0)
print(f"n_leaves={cl.n_leaves}", flush=True)

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
mark("seeded_ic", t0)

cl.set_bc(_lat.BoundaryCondition.dirichlet(0.0))
mark("set_bc", t0)

dt = _compute_dt_l0(cl, D1=1.0)
ts = _lat.TimeStepper.fixed(dt)
evolver = _lat.AdaptiveLatticeEvolver(alat)
mark("stepper_ready", t0)

for i in range(N_STEPS):
    evolver.step(ts)
    mark(f"step_{i}", t0)

pos = np.array(cl.leaf_pos)
mark("pos_copied", t0)
tree = cKDTree(pos)
mark("kdtree_built", t0)

phi = np.array(cl.leaf_phi)
phi_mag = np.sqrt(np.sum(phi**2, axis=0))
u_mag = np.abs(np.array(cl.leaf_u))
mark("fields_copied", t0)

phi_results = [anisotropy_ratio(tree, phi_mag, scale, f) for f in SAMPLE_FRACS]
u_results = [anisotropy_ratio(tree, u_mag, scale, f) for f in SAMPLE_FRACS]
mark("anisotropy_done", t0)

row = (
    f"{R:>5}  {cells_across:>12.1f}  {cl.n_leaves:>10}  "
    + "  ".join(f"{r[2]:.4f}" for r in phi_results)
    + "  "
    + "  ".join(f"{r[2]:.4f}" for r in u_results)
)
print("RESULT: " + row, flush=True)
