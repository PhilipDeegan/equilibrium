"""
Shared utilities for adaptive Bohr orbit experiments.

  W2_SUM               — BCC weight constant (Σ W_k² = 78)
  compute_dt            — CFL timestep from cluster state
  GrowingAMRSchedule    — geometrically growing adapt-call cadence
  find_origin           — leaf index nearest the spatial origin
  get_r_physical        — radial distances for all leaves
  setup_knot_and_origin — set_knot + u_origin + optional pin
  setup_ring            — find ring + init winding + absorbing BC
  run_warmup            — timed step loop with progress reporting
  probe_orbit           — fork warmup state and probe along +z
  compute_orbit_ratio   — orbit_r / ring_r
  refine_region_to_convergence — repeatedly adapt()+rebuild() a region until
                          nothing more splits (max_level reached or the
                          region criterion stops selecting new leaves)

Composable ArgumentParser helpers:
  add_lattice_args  — --lam
  add_amr_args      — --max-level, --split-hi, --split-lo
  add_warmup_args   — --n-warm, --n-relax, --n-report
  add_probe_args    — --n-probe, --n-pos, --u-def, --rmin-factor
"""

import sys

import numpy as np

from equilibrium import lattice as _lat
from tst.substrate.adaptive.adaptive_flux_tube import (
    find_toroidal_ring,
    init_toroidal_winding,
    set_knot,
)

W2_SUM = 8 * 9.0 + 6 * 1.0  # Σ W_k² for BCC: 8 hex × 9 + 6 sq × 1 = 78


def compute_dt(cl, w2_sum=W2_SUM, D1=1.0):
    """CFL timestep: 0.4 * L_min / (sigma_max * w2_sum), excluding fixed-u cells."""
    fu_arr = np.array(cl.leaf_fixed_u).astype(bool)
    lam_arr = np.array(cl.leaf_lam, dtype=np.float64)
    lev_arr = np.array(cl.leaf_level)
    free = ~fu_arr
    sigma_max = float(np.max(1.0 / (1.0 + lam_arr[free])))
    L_min = D1 / (2.0 ** int(lev_arr.max()))
    return 0.4 * L_min / (sigma_max * w2_sum)


class GrowingAMRSchedule:
    """Adapt-call cadence with a geometrically growing interval, capped at a
    maximum.

    Meshes typically need frequent re-evaluation early (the field is still
    settling near the IC) and far less often once refinement has converged
    to the evolving structure — a fixed --amr-interval either over-adapts
    late in a run or under-adapts early.  This schedule adapts at
    `start_interval`, then waits `start_interval*growth` before the next,
    doubling (by default) each time up to `max_interval`, after which the
    interval holds steady.

    Usage — replaces a fixed modulo cadence check in a step loop:
        schedule = GrowingAMRSchedule(start_interval=10, max_interval=1000)
        for step in range(1, n_steps + 1):
            ...
            if schedule.due(step):
                alat.adapt(0)

    Not coordinated with DSTInfo.split_interval — the caller should set
    cl.info.split_interval = 1 so every Python-triggered adapt() call
    evaluates immediately, and drive cadence entirely from `due()`.
    """

    def __init__(self, start_interval=10, max_interval=1000, growth=2.0):
        if start_interval <= 0:
            raise ValueError("start_interval must be positive")
        if max_interval < start_interval:
            raise ValueError("max_interval must be >= start_interval")
        self.interval = float(start_interval)
        self.max_interval = float(max_interval)
        self.growth = growth
        self.next_step = start_interval

    def due(self, step):
        """True if `step` is an adapt step; advances the internal schedule."""
        if step < self.next_step:
            return False
        self.interval = min(self.interval * self.growth, self.max_interval)
        self.next_step = step + self.interval
        return True


def find_origin(cl):
    """Return leaf index closest to the spatial origin."""
    pos = np.array(cl.leaf_pos)
    return int(np.argmin(np.sum(pos**2, axis=1)))


def get_r_physical(cl):
    """Return ndarray of radial distances for all leaves."""
    pos = np.array(cl.leaf_pos)
    return np.sqrt(np.sum(pos**2, axis=1))


def setup_knot_and_origin(cl, origin_j, lam, u_origin=1.0, profile="1/r", fix=True):
    """Apply lam knot profile, set u_origin at origin, and optionally pin it."""
    set_knot(cl, origin_j, lam, profile=profile)
    u_arr = np.array(cl.leaf_u)
    u_arr[origin_j] = u_origin
    cl.set_leaf_u(u_arr)
    if fix:
        fu = np.zeros(cl.n_leaves, dtype=np.uint8)
        fu[origin_j] = 1
        cl.set_leaf_fixed_u(fu)


def setup_ring(cl, axis="z", amplitude=1.0, frozen=False):
    """Find toroidal ring, init winding, set absorbing BC.

    Returns (ring_js, ring_faces). Exits with error if ring not found.
    """
    ring_js, ring_faces = find_toroidal_ring(cl, axis=axis)
    if ring_js is None:
        print("ERROR: toroidal ring not found")
        sys.exit(1)
    init_toroidal_winding(cl, ring_js, ring_faces, amplitude=amplitude, frozen=frozen)
    cl.set_bc(_lat.BoundaryCondition.absorbing())
    return ring_js, ring_faces


def run_warmup(
    alat, n_steps, dt_fn, *, cluster_idx=0, report_interval=5000, label="warmup"
):
    """Step n_steps, reporting u_mean every report_interval steps.

    dt_fn() is called at each report point so dt tracks level changes.
    Returns final dt.
    """
    cl = alat[cluster_idx]
    dt = dt_fn()
    print(f"{label} {n_steps} steps  dt={dt:.3e} …", flush=True)
    for s in range(1, n_steps + 1):
        alat.step(cluster_idx, dt)
        if s % report_interval == 0:
            dt = dt_fn()
            u_arr = np.array(cl.leaf_u)
            print(
                f"  {s}/{n_steps}  n_leaves={cl.n_leaves}"
                f"  u_mean={float(np.mean(u_arr)):.4f}",
                flush=True,
            )
    return dt_fn()


def probe_orbit(
    alat,
    cl,
    dt,
    origin_j,
    ring_js,
    r_min,
    r_max,
    *,
    n_pos=16,
    n_probe=2000,
    u_def=-0.1,
    cluster_idx=0,
):
    """Fork warmup state and probe along +z for n_pos radial positions.

    Returns (orbit_r, rows) where rows = list of (r_target, r_actual, u_init, u_final, du).
    Restores cluster state on exit.
    """
    u_warm = np.array(cl.leaf_u).copy()
    phi_warm = np.array(cl.leaf_phi).copy()
    pos = np.array(cl.leaf_pos)
    r_phys = np.sqrt(np.sum(pos**2, axis=1))

    exclude = set(ring_js) | {origin_j}
    z_axis = np.array([0.0, 0.0, 1.0])
    r_probe_vals = np.linspace(r_min, r_max, n_pos)

    best_du, best_r = 0.0, float("nan")
    rows = []

    for r_target in r_probe_vals:
        dists = np.linalg.norm(pos - r_target * z_axis, axis=1)
        j_probe = next((int(i) for i in np.argsort(dists) if int(i) not in exclude), -1)
        if j_probe < 0:
            continue
        cl.set_leaf_u(u_warm.copy())
        cl.set_leaf_phi(phi_warm.copy())
        u_pert = u_warm.copy()
        u_pert[j_probe] += u_def
        cl.set_leaf_u(u_pert)
        u_init = float(u_warm[j_probe])
        for _ in range(n_probe):
            alat.step(cluster_idx, dt)
        u_final = float(np.array(cl.leaf_u)[j_probe])
        du = u_final - u_init
        rows.append((r_target, float(r_phys[j_probe]), u_init, u_final, du))
        if du < best_du:
            best_du, best_r = du, float(r_phys[j_probe])

    cl.set_leaf_u(u_warm)
    cl.set_leaf_phi(phi_warm)
    return best_r, rows


def compute_orbit_ratio(orbit_r, ring_r):
    """orbit_r / ring_r, or nan if inputs are invalid."""
    if ring_r > 0 and not np.isnan(orbit_r):
        return orbit_r / ring_r
    return float("nan")


# ---------------------------------------------------------------------------
# Composable ArgumentParser helpers
# ---------------------------------------------------------------------------


def add_lattice_args(parser, *, lam=9.0):
    """Add --lam."""
    parser.add_argument("--lam", type=float, default=lam)


def add_amr_args(parser, *, max_level=0, split_hi=0.0, split_lo=0.0):
    """Add --max-level, --split-hi, --split-lo."""
    parser.add_argument("--max-level", type=int, default=max_level)
    parser.add_argument("--split-hi", type=float, default=split_hi)
    parser.add_argument("--split-lo", type=float, default=split_lo)


def add_warmup_args(parser, *, n_warm=50_000, n_relax=10_000, n_report=5_000):
    """Add --n-warm, --n-relax, --n-report."""
    parser.add_argument("--n-warm", type=int, default=n_warm)
    parser.add_argument("--n-relax", type=int, default=n_relax)
    parser.add_argument("--n-report", type=int, default=n_report)


def add_probe_args(parser, *, n_probe=2_000, n_pos=16, u_def=-0.1, rmin_factor=2.5):
    """Add --n-probe, --n-pos, --u-def, --rmin-factor."""
    parser.add_argument("--n-probe", type=int, default=n_probe)
    parser.add_argument("--n-pos", type=int, default=n_pos)
    parser.add_argument("--u-def", type=float, default=u_def)
    parser.add_argument("--rmin-factor", type=float, default=rmin_factor)


def add_criterion_args(
    parser, *, criterion="relative_u", split_frac=0.05, memory_budget=0
):
    """Add SplitCriterion arguments.

    --criterion       : absolute | relative_u (default: relative_u)
    --split-frac      : target fraction of cells to split per level (auto mode)
    --memory-budget   : max leaf count after adapt; 0 = unlimited (auto mode)
    --hi / --lo / --ref : manual thresholds — all three must be given to skip auto-derive
    """
    parser.add_argument(
        "--criterion",
        choices=["absolute", "relative_u"],
        default=criterion,
    )
    parser.add_argument(
        "--split-frac",
        type=float,
        default=split_frac,
        help="Target fraction of cells to split (auto mode, default 0.05)",
    )
    parser.add_argument(
        "--memory-budget",
        type=int,
        default=memory_budget,
        help="Max leaf count after adapt; 0 = unlimited",
    )
    parser.add_argument(
        "--hi",
        type=float,
        default=None,
        help="Split threshold (auto-derived if omitted)",
    )
    parser.add_argument(
        "--lo", type=float, default=None, help="Coarsen threshold (auto: hi * 0.1)"
    )
    parser.add_argument(
        "--ref",
        type=float,
        default=None,
        help="u floor for relative_u (auto: p5 of |u|)",
    )


def apply_split_criterion(cl, args, *, level=0):
    """Wire SplitCriterion onto cl from parsed add_criterion_args args.

    split_frac is halved per level (split_frac / 2**level) to bound cell-count
    growth — re-derive after each relax by passing the current level.
    If --hi/--lo/--ref are all supplied, uses them directly (no per-level scaling).
    Returns the SplitCriterion that was applied.
    """
    from equilibrium import lattice as _lat
    from equilibrium.substrate.amr import auto_split_criterion

    manual = args.hi is not None and args.lo is not None and args.ref is not None
    if manual:
        sc = getattr(_lat.SplitCriterion, args.criterion)(
            hi=args.hi, lo=args.lo, ref=args.ref
        )
        print(f"criterion (manual):  hi={sc.hi:.4f}  lo={sc.lo:.4f}  ref={sc.ref:.4f}")
    else:
        sf = args.split_frac / (2**level)
        sc = auto_split_criterion(
            cl,
            split_frac=sf,
            memory_budget=getattr(args, "memory_budget", 0) or None,
        )
        print(
            f"criterion (auto, level={level}, split_frac={sf:.4f}):"
            f"  hi={sc.hi:.4f}  lo={sc.lo:.4f}  ref={sc.ref:.4f}"
        )
    cl.set_split_criterion(sc)
    return sc


# ---------------------------------------------------------------------------
# Kept for backwards compatibility (used by test_adaptive_lattice.py)
# ---------------------------------------------------------------------------


def build_adaptive_sphere(
    R: float,
    max_level: int = 0,
    u_origin: float = 1.0,
    split_hi: float = 0.0,
    split_lo: float = 0.0,
    d1: float = 1.0,
    lam0: int = 0,
    fix_origin: bool = False,
):
    """Build a flat AdaptiveLattice BCC sphere.

    Returns (alat, origin_leaf).
    """
    alat = _lat.AdaptiveLattice()
    cl = alat[0]
    cl.info.max_level = max_level
    cl.info.split_hi = split_hi
    cl.info.split_lo = split_lo
    _lat.build_bcc_sphere(cl, R, d1=d1, u0=0.0, ell0=1.0, lam0=lam0)
    alat.rebuild(0)
    pos = cl.leaf_pos
    r2 = (pos**2).sum(axis=1)
    origin_leaf = int(np.argmin(r2))
    if u_origin != 0.0:
        u_arr = np.zeros(cl.n_leaves, dtype=np.float64)
        u_arr[origin_leaf] = u_origin
        cl.set_leaf_u(u_arr)
    if fix_origin:
        fixed_arr = np.zeros(cl.n_leaves, dtype=np.uint8)
        fixed_arr[origin_leaf] = 1
        cl.set_leaf_fixed_u(fixed_arr)
    return alat, origin_leaf


def refine_region_to_convergence(
    alat, cl, region_fn, stage_max_level, protected_js_fn=None, max_rounds=20
):
    """Drive every leaf region_fn selects up to stage_max_level.

    A single adapt() call only deepens a leaf by one level — a freshly split
    child must go through rebuild() and become eligible again before it can
    split further — and a 2:1-balance refusal can also delay a split by one
    round (see AdaptiveLattice::split()'s can_split guard). So "fully
    refined to a target level" is not one adapt() call, it's calling
    adapt()+rebuild() repeatedly, re-evaluating region_fn against the
    CURRENT leaf set each round (leaf indices reshuffle after every
    rebuild), until a round produces no new leaves — either every selected
    cell has reached stage_max_level, or region_fn no longer selects
    anything new to split.

    region_fn(cl) -> boolean mask over the CURRENT leaves: which cells
    should be driven toward stage_max_level.
    protected_js_fn(cl), optional -> leaf-j indices that must never split
    regardless of region_fn (e.g. active ring/winding tiles whose topology
    AdaptiveLattice::split()'s phi-prolongation would corrupt) — re-resolved
    every round since protected cells' own leaf-j indices shift too.
    Temporarily overrides cl.info.max_level to stage_max_level for the
    duration (restored after), so callers can build a nested L0->L1->L2
    cascade by calling this once per level with a narrower region_fn each
    time.
    """
    prev_max_level = cl.info.max_level
    cl.info.max_level = stage_max_level
    try:
        for _ in range(max_rounds):
            cmax = np.full(cl.n_leaves, np.nan)
            cmax[region_fn(cl)] = 1.0
            if protected_js_fn is not None:
                protected = protected_js_fn(cl)
                if protected is not None and len(protected):
                    cmax[protected] = np.nan
            cl.set_criterion_max(cmax)
            n_before = cl.n_leaves
            alat.adapt(0)
            alat.rebuild(0)
            if cl.n_leaves == n_before:
                break
        else:
            raise RuntimeError(
                f"refine_region_to_convergence: did not converge within {max_rounds} rounds"
            )
    finally:
        cl.info.max_level = prev_max_level
