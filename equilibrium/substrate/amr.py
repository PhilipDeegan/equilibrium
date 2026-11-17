"""
AMR utility helpers for AdaptiveLattice / SplitCriterion.

auto_split_criterion(cl, ...)
    Derive hi, lo, ref from the current cluster state so the caller does not
    have to pick thresholds by hand.  Works at any refinement level and any
    physical scale because it normalises by |u|.

    Parameters
    ----------
    cl              : AdaptiveCellCluster
    split_frac      : float  (default 0.05)
        Target fraction of current leaves to split.  hi is set to the
        (1 - split_frac) percentile of the normalised criterion distribution.
    coarsen_factor  : float  (default 0.1)
        lo = hi * coarsen_factor.  Set to 0 to disable coarsening.
    ref_percentile  : float  (default 5)
        |u| percentile used as the void/boundary floor for the relative_u
        criterion.  Prevents blowup in cells where |u|→0.
    memory_budget   : int or None  (default None)
        Hard cap on total leaf count after the next adapt pass.  Each split
        adds 7 net leaves (8 children − 1 parent).  If the naive split count
        would exceed the budget, hi is raised until the projection fits.
        Passing 0 or None disables the budget check.

    Returns
    -------
    SplitCriterion configured with the derived hi, lo, ref.

    Notes
    -----
    Call this after a warmup / relax phase so criterion_max is populated.
    NaN entries (unstepped cells) are excluded from the percentile computation.
"""

import numpy as np


def auto_split_criterion(
    cl, *, split_frac=0.05, coarsen_factor=0.0, ref_percentile=5, memory_budget=None
):
    from equilibrium import lattice as _lat

    u = np.array(cl.leaf_u)
    cmax = np.array(cl.criterion_max)
    mask = np.isfinite(cmax)

    # ref: low percentile of |u| — natural floor for void/boundary cells.
    ref = float(np.percentile(np.abs(u), ref_percentile))

    if mask.sum() == 0:
        return _lat.SplitCriterion.relative_u(hi=0.3, lo=0.0, ref=ref)

    u_eff = np.maximum(np.abs(u), ref)
    normed = np.where(u_eff > 0, cmax / u_eff, np.nan)
    nmask = np.isfinite(normed) & mask

    if nmask.sum() == 0:
        return _lat.SplitCriterion.relative_u(hi=0.3, lo=0.0, ref=ref)

    n_leaves = int(cl.n_leaves)
    n_valid = int(nmask.sum())

    # hi: threshold that splits the top split_frac of valid cells.
    target_pct = (1.0 - split_frac) * 100.0
    hi = float(np.percentile(normed[nmask], target_pct))

    # Apply memory budget — raise hi until projected leaf count fits.
    if memory_budget and memory_budget > 0:
        n_split = int(np.sum(normed[nmask] > hi))
        projected = n_leaves + n_split * 7  # 8 children − 1 parent
        if projected > memory_budget:
            headroom = max(0, memory_budget - n_leaves)
            allowed = headroom // 7
            if allowed < n_valid:
                # hi = value such that only `allowed` cells exceed it
                adjusted_pct = (1.0 - allowed / n_valid) * 100.0
                hi = float(np.percentile(normed[nmask], adjusted_pct))
            else:
                pass  # budget is generous enough

    lo = hi * coarsen_factor if coarsen_factor > 0.0 else 0.0

    return _lat.SplitCriterion.relative_u(hi=hi, lo=lo, ref=ref)
