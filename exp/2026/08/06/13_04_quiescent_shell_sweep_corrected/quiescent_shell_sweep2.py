import sys

sys.path.insert(0, ".")
import numpy as np

from equilibrium import lattice as _lat
from run.substrate.adaptive.refined_neutron_decay import init_spheromak_ic_refinable
from run.substrate.adaptive.sweep_spheromak_decay_cross_scale import _compute_dt_l0
from tst.substrate.adaptive import refine_region_to_convergence

R = 100.0
SCALE = 45.0  # spheromak radius = 45% of R

SHELLS = [
    (47.4, 47.6),  # thin, well clear of the spheromak (baseline quiescent case)
    (46.0, 47.6),  # wider, still just outside
    (45.0, 47.6),  # inner edge exactly at the spheromak's own boundary
    (43.0, 47.6),  # overlaps the spheromak's outer edge
    (40.0, 47.6),  # well inside the spheromak
]


def build_base():
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    alat.set_n_threads(8)
    cl = alat[0]
    _lat.build_bcc_sphere(cl, R, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)
    return alat, cl


def shell_mask_fn(lo, hi):
    def _fn(cl):
        pos = np.array(cl.leaf_pos)
        r = np.linalg.norm(pos, axis=1)
        return (r >= lo) & (r <= hi)

    return _fn


def report_region(cl, mask, tag):
    u = np.array(cl.leaf_u)[mask]
    phi = np.array(cl.leaf_phi)[:, mask]
    n = mask.sum()
    print(
        f"    [{tag}] N={n}  max|u|={np.max(np.abs(u)) if n else float('nan'):.3e}  "
        f"max|phi|={np.max(np.abs(phi)) if n else float('nan'):.3e}"
    )


print("=== BASELINE: pure uniform (no refinement anywhere), IC seeded directly ===")
alat_a, cl_a = build_base()
init_spheromak_ic_refinable(
    cl_a,
    R_sph=1.0,
    scale=SCALE,
    B0=1.0,
    lam_target=1.0,
    u_max=1.0,
    amplitude=1.0,
    sigma=None,
)
cl_a.set_bc(_lat.BoundaryCondition.dirichlet(0.0))
H0_a = float(
    np.sum(np.array(cl_a.leaf_u) ** 2) + 0.5 * np.sum(np.array(cl_a.leaf_phi) ** 2)
)
dt_a = _compute_dt_l0(cl_a, D1=1.0)
ev_a = _lat.AdaptiveLatticeEvolver(alat_a)
ev_a.step(_lat.TimeStepper.fixed(dt_a))
u1_a = np.array(cl_a.leaf_u)
phi1_a = np.array(cl_a.leaf_phi)
H1_a = float(np.sum(u1_a**2) + 0.5 * np.sum(phi1_a**2))
print(f"N_total={cl_a.n_leaves}  dt={dt_a:.4e}  H/H0={H1_a/H0_a:.6f}")

for lo, hi in SHELLS:
    print(
        f"=== SHELL [{lo}, {hi}]: REFINE FIRST, then seed IC analytically (no prolongation) ==="
    )
    alat_b, cl_b = build_base()
    n_before = cl_b.n_leaves
    cl_b.info.split_hi = 0.5
    cl_b.info.split_lo = 0.0
    refine_region_to_convergence(alat_b, cl_b, shell_mask_fn(lo, hi), stage_max_level=1)
    cl_b.set_criterion_max(np.full(cl_b.n_leaves, np.nan))
    cl_b.clear_split_criterion()
    print(
        f"    N_before={n_before}  N_after={cl_b.n_leaves}  (+{cl_b.n_leaves - n_before})"
    )

    # IC seeded AFTER refinement, directly on the final mesh's own leaf
    # positions/levels -- no prolongation/interpolation involved at all.
    init_spheromak_ic_refinable(
        cl_b,
        R_sph=1.0,
        scale=SCALE,
        B0=1.0,
        lam_target=1.0,
        u_max=1.0,
        amplitude=1.0,
        sigma=None,
    )

    cl_b.set_bc(_lat.BoundaryCondition.dirichlet(0.0))
    m_b = shell_mask_fn(lo, hi)(cl_b)
    report_region(cl_b, m_b, "pre-step")

    H0_b = float(
        np.sum(np.array(cl_b.leaf_u) ** 2) + 0.5 * np.sum(np.array(cl_b.leaf_phi) ** 2)
    )
    dt_b = _compute_dt_l0(cl_b, D1=1.0)
    has_fine = int(np.max(np.array(cl_b.leaf_level))) > 0
    ts_b = _lat.TimeStepper.by_level(dt_b) if has_fine else _lat.TimeStepper.fixed(dt_b)
    ev_b = _lat.AdaptiveLatticeEvolver(alat_b)
    ev_b.step(ts_b)
    u1_b = np.array(cl_b.leaf_u)
    phi1_b = np.array(cl_b.leaf_phi)
    H1_b = float(np.sum(u1_b**2) + 0.5 * np.sum(phi1_b**2))
    m_b_post = shell_mask_fn(lo, hi)(cl_b)
    report_region(cl_b, m_b_post, "post-step")
    print(
        f"    N_total={cl_b.n_leaves}  dt={dt_b:.4e}  H0={H0_b:.6e}  H1={H1_b:.6e}  "
        f"H/H0={H1_b/H0_b:.6f}"
    )
