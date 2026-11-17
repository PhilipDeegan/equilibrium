"""
Reusable helpers for adaptive lattice (AdaptiveCellCluster).
"""

import numpy as np
from scipy.spatial import KDTree

from equilibrium.substrate.discrete import LAM_MAX

# ---------------------------------------------------------------------------
# BCC face geometry — mirrors bcc_geometry.hpp
# ---------------------------------------------------------------------------
_HEX_DIR = np.array(
    [
        [-1, -1, -1],
        [-1, -1, +1],
        [-1, +1, -1],
        [-1, +1, +1],
        [+1, -1, -1],
        [+1, -1, +1],
        [+1, +1, -1],
        [+1, +1, +1],
    ],
    dtype=np.float64,
)
_SQ_DIR = np.array(
    [[-1, 0, 0], [+1, 0, 0], [0, -1, 0], [0, +1, 0], [0, 0, -1], [0, 0, +1]],
    dtype=np.float64,
)

_ALL_DIR = np.concatenate([_HEX_DIR, _SQ_DIR])  # shape (14, 3)


def wire_periodic_bcc(cl, H: float, axes: str = "xyz", verbose: bool = False) -> int:
    """Wire periodic boundary conditions for a BCC box cluster.

    Uses a full 3D KDTree over all leaf cells.  For each unwired boundary face
    the natural neighbour position is wrapped into the box; the nearest existing
    leaf is selected as the periodic partner.  Both directions (cell→partner and
    partner.mirror→cell) are set in one step, so antisymmetry is always satisfied
    even when two valid BCC candidates are equidistant from the wrapped target
    (which happens for hex faces when the box period is an odd multiple of H).

    Must be called after build_bcc_box AND after every adapt+rebuild.

    Parameters
    ----------
    cl      : AdaptiveCellCluster
    H       : float — BCC lattice step, d1 / sqrt(3)
    axes    : str   — which pairs to wire, any subset of "xyz" (default "xyz")
    verbose : bool  — print total wired face count

    Returns
    -------
    Total number of faces wired (counts both src→dst and dst→src).

    Raises
    ------
    ValueError
        If any boundary face in a wired axis remains unwired after all wiring
        attempts.  This indicates a wiring bug.
    """
    p = cl.leaf_pos
    g = cl.leaf_global_indices
    lev = cl.leaf_level
    n = cl.leaf_nb.copy()  # local copy — updated as faces are wired
    N = len(g)

    lo = p.min(axis=0)  # (3,)
    hi = p.max(axis=0)  # (3,)
    period = hi - lo + H  # (3,) — one BCC step wider than extent
    ax_active = np.array([a in axes for a in "xyz"], dtype=bool)

    tree = KDTree(p)
    total = 0

    def _wire_pass(src_iter, k_near, max_dist):
        nonlocal total
        for j, k, tgt in src_iter:
            if n[j, k] >= 0:
                continue
            mk = k ^ 7 if k < 8 else k ^ 1
            g_j = int(g[j])
            dists, idxs = tree.query(tgt, k=k_near)
            for dist_c, idx_c in zip(
                (dists,) if np.ndim(dists) == 0 else dists,
                (idxs,) if np.ndim(idxs) == 0 else idxs,
            ):
                if dist_c > max_dist:
                    break
                if n[idx_c, mk] >= 0:
                    continue
                pg = int(g[idx_c])
                if pg == g_j:
                    continue
                cl.set_nb(g_j, k, pg)
                cl.set_nb(pg, mk, g_j)
                n[j, k] = pg
                n[idx_c, mk] = g_j
                total += 2
                break

    # Collect all boundary faces (target wraps outside the box in a wired axis).
    boundary_faces = []
    for j in range(N):
        cell_H = H * (0.5 ** int(lev[j]))
        for k in range(14):
            if n[j, k] >= 0:
                continue
            step = (_HEX_DIR[k] if k < 8 else _SQ_DIR[k - 8] * 2.0) * cell_H
            tgt = p[j] + step
            wrap = ax_active & ((tgt < lo) | (tgt > hi))
            if not wrap.any():
                continue
            for ax in np.where(wrap)[0]:
                tgt[ax] = lo[ax] + (tgt[ax] - lo[ax]) % period[ax]
            boundary_faces.append((j, k, tgt.copy()))

    # Pass 1: fast greedy — nearest 50 candidates within 3.5H.
    # Covers shells at H, sqrt(2)H, 2H, 2sqrt(2)H for typical BCC incommensurability.
    _wire_pass(boundary_faces, k_near=50, max_dist=H * 3.5)

    # Pass 2: exhaustive fallback for stragglers — greedy ordering can leave some
    # cells stuck when all nearby candidates are claimed.  Re-query ALL leaves with no
    # distance cutoff; this always succeeds if a valid unclaimed partner exists anywhere.
    stragglers = [(j, k, tgt) for (j, k, tgt) in boundary_faces if n[j, k] < 0]
    if stragglers:
        _wire_pass(stragglers, k_near=N, max_dist=np.inf)

    if verbose:
        print(f"wire_periodic_bcc: {total} faces wired")

    # Post-wiring sanity check: any remaining nb=-1 face whose target wraps into a
    # wired axis is a genuine failure (bug or completely incommensurable box).
    # We re-use the same wrap logic to identify which faces SHOULD have been wired.
    n_after = cl.leaf_nb  # live read — reflects all set_nb calls above
    bad = 0
    for j in range(N):
        cell_H = H * (0.5 ** int(lev[j]))
        for k in range(14):
            if n_after[j, k] >= 0:
                continue
            step = (_HEX_DIR[k] if k < 8 else _SQ_DIR[k - 8] * 2.0) * cell_H
            tgt = p[j] + step
            wrap = ax_active & ((tgt < lo) | (tgt > hi))
            if wrap.any():
                bad += 1
    if bad:
        raise ValueError(
            f"wire_periodic_bcc: {bad} boundary faces could not be wired in axes='{axes}'."
        )

    return total


def dst_state(cl) -> dict:
    """Structural introspection summary of an AdaptiveCellCluster's CURRENT
    leaf set: cells-per-level histogram plus hex-face neighbour-level-jump
    stats, for spotting AMR topology problems (e.g. a >1-level neighbour
    gap) without hand-walking the mesh.

    Returns a dict:
      n_leaves               : int
      max_level              : int
      cells_per_level        : {level: count}
      max_hex_leaf_jump      : int or None — largest |level_i - level_nb|
                                seen over HEX faces (k<8) where BOTH
                                endpoints are leaves (a direct, unambiguous
                                leaf_nb pointer). None if no such face exists.
      n_hex_leaf_jumps_gt1   : int — count of such pairs with jump > 1.
                                Nonzero means a genuine 2:1-balance violation
                                exists in the mesh (`can_split` enforces no
                                such balance — see `now`, 2026-08-13 stale-
                                proxy investigation, and the
                                project_2to1_balance_question memory — this is
                                exactly the class of topology that broke
                                TestMultipleSplits).
      n_hex_faces_unresolved : int — hex faces whose neighbour is a NON-LEAF
                                cell (ancestor-substitution proxy) or has no
                                leaf_nb pointer; these carry real cross-level
                                flux but can't be level-checked here without a
                                global level-by-global-id lookup (leaf_level
                                is leaf-indexed only, not exposed by global id)
                                — NOT included in max_hex_leaf_jump /
                                n_hex_leaf_jumps_gt1, so those two are a LOWER
                                BOUND, not exact. Square faces (k>=8) are
                                entirely excluded too, for the same reason
                                plus genuine multi-child fan-in (see `now`).

    Every real face is seen from both endpoints' own hex-face lists, so a
    genuine jump is counted twice (once from each side) — this inflates
    n_hex_leaf_jumps_gt1 by ~2x but does not affect max_hex_leaf_jump or
    whether a violation exists at all.
    """
    level = np.asarray(cl.leaf_level)
    g = np.asarray(cl.leaf_global_indices)
    nb_hex = np.asarray(cl.leaf_nb)[:, :8]
    N = len(g)

    cells_per_level = (
        {int(lv): int(c) for lv, c in zip(*np.unique(level, return_counts=True))}
        if N
        else {}
    )

    order = np.argsort(g)
    g_sorted = g[order]
    level_sorted = level[order]

    valid = nb_hex >= 0
    flat_nb = nb_hex[valid]
    li = np.repeat(level[:, None], 8, axis=1)[valid]

    idx = np.searchsorted(g_sorted, flat_nb)
    in_range = idx < len(g_sorted)
    idx_c = np.clip(idx, 0, max(len(g_sorted) - 1, 0))
    found = (
        in_range & (g_sorted[idx_c] == flat_nb)
        if len(g_sorted)
        else np.zeros_like(idx, dtype=bool)
    )

    jumps = np.abs(li[found] - level_sorted[idx_c[found]])

    return {
        "n_leaves": N,
        "max_level": int(level.max()) if N else 0,
        "cells_per_level": cells_per_level,
        "max_hex_leaf_jump": int(jumps.max()) if jumps.size else None,
        "n_hex_leaf_jumps_gt1": int((jumps > 1).sum()),
        "n_hex_faces_unresolved": int((~found).sum()),
    }


def step_lam_relax(cl, dt: float, rate: float, alpha: float = 1.0):
    """
    Local lam relaxation toward alpha * |phi| for an AdaptiveCellCluster.

    lam_target_i = clip(alpha * |phi_i|, 0, LAM_MAX)
    dlam_i/dt    = rate * (lam_target_i - lam_i)

    No transport between cells — pure local coupling to field amplitude.
    """
    phi_mag = np.sqrt((cl.leaf_phi**2).sum(axis=0))
    lam = cl.leaf_lam
    lam_target = np.clip(alpha * phi_mag, 0.0, LAM_MAX)
    cl.set_leaf_lam(np.clip(lam + dt * rate * (lam_target - lam), 0.0, LAM_MAX))


def make_ell_boundary_gate(cl, ell_vacuum, H, axes="xy", band_cells=2):
    """Return a callable that resets ell → ell_vacuum in a boundary layer.

    The gate is a selective BC for the chi/ell field: phi and u circulate freely
    through periodic faces, but substrate compression (ell) perturbations are
    absorbed at the walls — equivalent to embedding the domain in an infinite
    vacuum substrate.

    The mask is recomputed each call from cl.leaf_pos so it stays valid after
    AMR adapts that add or remove cells.

    Parameters
    ----------
    cl          : AdaptiveCellCluster
    ell_vacuum  : float   — vacuum ell value to restore at boundary (typically args.ell0)
    H           : float   — BCC lattice step (d1 / sqrt(3)); sets spatial scale
    axes        : str     — which boundary pairs to gate: any combination of 'x','y','z'
    band_cells  : float   — gate thickness in units of H (default 2 ≈ one Kelvin cell)
    """
    bw = band_cells * H

    def gate():
        pos = cl.leaf_pos
        mask = np.zeros(len(pos), dtype=bool)
        if "x" in axes:
            x = pos[:, 0]
            mask |= (x < x.min() + bw) | (x > x.max() - bw)
        if "y" in axes:
            y = pos[:, 1]
            mask |= (y < y.min() + bw) | (y > y.max() - bw)
        if "z" in axes:
            z = pos[:, 2]
            mask |= (z < z.min() + bw) | (z > z.max() - bw)
        if not mask.any():
            return
        ell = cl.leaf_ell
        ell[mask] = ell_vacuum
        cl.set_leaf_ell(ell)

    return gate
