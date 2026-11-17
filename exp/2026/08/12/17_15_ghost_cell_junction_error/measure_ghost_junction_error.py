"""Ghost-cell measurement of the L0/L1 junction's absolute error, using REAL
same-level cells (constant-prolongation split, the same convention the whole
codebase already uses -- NOT the retired enable_cross_level_gradient_correction
extrapolation, which is a confirmed dead end, see adaptive_lattice.cpp
2026-08-08 comments: it makes the curvature error 4-15x WORSE because
collapsing both sample points to the same location leaves the flux weight's
spacing-normalization denominator unchanged).

Method: load the real checkpoint twice.
  - "real": stepped unchanged (production path, ancestor-substitution at the
    L0/L1 junction).
  - "ghost": BEFORE stepping, the L0 shell (all level-0 leaves) is refined to
    level 1 via refine_region_to_convergence (constant prolongation, the
    existing split() convention) -- this eliminates the junction entirely for
    the ghost copy. The original L1 leaves keep their global ids unchanged
    (alloc_children only ever allocates NEW slots), so real/ghost can be
    diffed leaf-for-leaf on that unchanged set after stepping.

Run twice per real/ghost pair: wave-step only (evolver.step, isolates the
ancestor-substitution/AMR junction formula alone) and wave-step+step_cross
(the full "everything active" physics) -- the DELTA between those two error
levels isolates whether step_cross's own per-level treatment (delta ~ dt *
cell.L * cross_scale, applied with the un-subcycled base dt at every level
every outer step, not per-substep) adds its own artifact on top of the pure
AMR junction error.
"""

import sys

sys.path.insert(0, ".")

import numpy as np

from equilibrium import lattice as _lat
from run.substrate.adaptive.sweep_spheromak_decay_cross_scale import _compute_dt_l0
from tst.substrate.adaptive import refine_region_to_convergence

CKPT = "/bk/l1_wider_margin_active_stepped/step_130_final.h5"
CROSS_SCALE = 1e-3


def load(path):
    return _lat.AdaptiveLattice.load(path)


def build_ghost(path):
    alat = load(path)
    cl = alat[0]
    levels = np.array(cl.leaf_level)
    n_l0_before = int((levels == 0).sum())

    def region_fn(cl_):
        return np.array(cl_.leaf_level) == 0

    refine_region_to_convergence(alat, cl, region_fn, stage_max_level=1)
    levels_after = np.array(cl.leaf_level)
    print(
        f"[ghost build] L0 leaves before={n_l0_before} -> after refine, "
        f"level counts={np.bincount(levels_after).tolist()} n_leaves={cl.n_leaves}",
        flush=True,
    )
    return alat


def run_case(real_path, use_cross, label):
    alat_real = load(real_path)
    cl_real = alat_real[0]
    g_real = np.array(cl_real.leaf_global_indices)
    levels_real = np.array(cl_real.leaf_level)

    alat_ghost = build_ghost(real_path)
    cl_ghost = alat_ghost[0]
    g_ghost = np.array(cl_ghost.leaf_global_indices)
    g_to_j_ghost = {int(g): j for j, g in enumerate(g_ghost)}

    dt_real = _compute_dt_l0(cl_real, D1=1.0)
    dt_ghost = _compute_dt_l0(cl_ghost, D1=1.0)
    print(f"[{label}] dt_real={dt_real:.6e} dt_ghost={dt_ghost:.6e}", flush=True)

    ts_real = _lat.TimeStepper.by_level(dt_real)
    ts_ghost = _lat.TimeStepper.by_level(dt_ghost)

    ev_real = _lat.AdaptiveLatticeEvolver(alat_real)
    ev_ghost = _lat.AdaptiveLatticeEvolver(alat_ghost)

    ev_real.step(ts_real)
    if use_cross:
        alat_real.step_cross_all(dt_real, CROSS_SCALE)

    ev_ghost.step(ts_ghost)
    if use_cross:
        alat_ghost.step_cross_all(dt_ghost, CROSS_SCALE)

    u_real = np.array(cl_real.leaf_u)
    u_ghost_full = np.array(cl_ghost.leaf_u)

    # Only L1 leaves that existed in BOTH real and ghost from the start (i.e.
    # were level 1 already in "real" -- their global ids are untouched by
    # ghost's L0-shell refinement).
    is_l1_real = levels_real == 1
    l1_g = g_real[is_l1_real]
    l1_u_real = u_real[is_l1_real]

    # Map those same global ids into the ghost copy's own leaf ordering.
    idx_in_ghost = np.array([g_to_j_ghost[int(g)] for g in l1_g])
    l1_u_ghost = u_ghost_full[idx_in_ghost]

    du = np.abs(l1_u_real - l1_u_ghost)

    # leaf_is_boundary (already exposed): true iff any of the 14 neighbours
    # sits at a different cell.L -- exactly the coarse-fine junction
    # indicator, computed on "real" BEFORE any ghost refinement so it's the
    # same physical leaf set in both copies.
    at_junction_full = np.array(cl_real.leaf_is_boundary).astype(bool)
    at_junction_l1 = at_junction_full[is_l1_real]

    print(
        f"[{label}] L1 leaves={du.size}  at_junction={int(at_junction_l1.sum())}  "
        f"interior={int((~at_junction_l1).sum())}",
        flush=True,
    )
    print(
        f"[{label}] JUNCTION   mean|du|={du[at_junction_l1].mean():.6e}  max|du|={du[at_junction_l1].max():.6e}",
        flush=True,
    )
    print(
        f"[{label}] INTERIOR   mean|du|={du[~at_junction_l1].mean():.6e}  max|du|={du[~at_junction_l1].max():.6e}",
        flush=True,
    )
    return du, at_junction_l1


if __name__ == "__main__":
    print(
        "=== case A: wave-step only (isolates AMR/ancestor-substitution junction error) ===",
        flush=True,
    )
    run_case(CKPT, use_cross=False, label="wave_only")

    print(
        "\n=== case B: wave-step + step_cross (full physics, matches the real 08/11 runs) ===",
        flush=True,
    )
    run_case(CKPT, use_cross=True, label="wave_plus_cross")
