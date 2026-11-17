"""
AMR adapt() coarsen-pass g_to_j benchmark.

equilibrium::dst::adapt()'s coarsen pass (inc/eq/core/data/lattice/adaptive_amr.hpp)
rebuilds a full leaves-sized std::unordered_map<int32_t, size_t> every call
whenever split_lo > 0 -- UNCONDITIONALLY, before it even checks which parents
have "quiet" children, so the map-build cost is paid every call regardless of
whether anything actually coarsens. This isolates that cost via the existing
equilibrium::dst::adapt scope timer (same EQ_SCOPE_TIMER used by run/substrate/
perf.py), by holding the lattice in a stable AMR configuration (criterion held
strictly between split_lo and split_hi everywhere, so nothing actually splits
or coarsens) and calling adapt() N_BENCH times.

Use this to compare g_to_j implementations (std::unordered_map, a dense
std::vector<int32_t> lookup like _step_cross_local/_step_cross_mirror already
use in adaptive_lattice.hpp, or ankerl::unordered_dense) -- rebuild the
extension between runs and compare wall time / the printed adapt() scope
timing.

Usage:
    PYTHONPATH=$PWD python3 run/substrate/perf_amr_adapt_map.py
    PYTHONPATH=$PWD python3 run/substrate/perf_amr_adapt_map.py --R 20 --max-level 4 --n-bench 500
"""

import argparse
import os
import sys
import time

import numpy as np

os.environ["EQUILIBRIUM_SCOPE_TIMING"] = "1"
sys.path.insert(0, ".")

from equilibrium import lattice as _lat

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    "--R", type=float, default=15.0, help="BCC sphere radius in cell units"
)
parser.add_argument("--D1", type=float, default=1.0, help="Level-0 cell spacing")
parser.add_argument(
    "--r-shell", type=float, default=9.0, help="Radius of the refinement shell"
)
parser.add_argument(
    "--dr", type=float, default=1.5, help="Half-width of the refinement shell"
)
parser.add_argument(
    "--max-level",
    type=int,
    default=3,
    help="Number of AMR refinement levels applied to the shell",
)
parser.add_argument(
    "--n-bench", type=int, default=200, help="Number of adapt() calls to time"
)
parser.add_argument(
    "--threads", type=int, default=1, help="Thread count passed to AdaptiveLattice"
)
args, _ = parser.parse_known_args()

R = args.R
D1 = args.D1
R_SHELL = args.r_shell
DR = args.dr
MAX_LEVEL = args.max_level
N_BENCH = args.n_bench
THREADS = args.threads


def _split_shell(cl, r_shell, dr):
    pos = np.array(cl.leaf_pos)
    r = np.sqrt(np.sum(pos**2, axis=1))
    cmax = np.full(cl.n_leaves, np.nan)
    cmax[np.abs(r - r_shell) < dr] = 1.0
    cl.set_criterion_max(cmax)


def _print_timings(path=".phlop/timings.txt"):
    if not os.path.exists(path):
        print("(no .phlop/timings.txt -- check EQUILIBRIUM_SCOPE_TIMING build support)")
        return
    import phlop.timing.scope_timer as phst

    phst.print_scope_timings(path, sort_worst_first=True)


if __name__ == "__main__":
    _lat.start_phlop()
    print(
        f"R={R}  D1={D1}  r_shell={R_SHELL}  dr={DR}  max_level={MAX_LEVEL}"
        f"  n_bench={N_BENCH}  threads={THREADS}"
    )

    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    cl = alat[0]

    _lat.build_bcc_sphere(cl, R, D1, u0=1.0, ell0=1.0, lam0=0)
    alat.rebuild(0)
    alat.set_n_threads(THREADS)

    info = _lat.DSTInfo()
    info.split_hi = 1.0
    info.split_lo = 0.0
    cl.set_bc(_lat.BoundaryCondition.absorbing())
    evolver = _lat.AdaptiveLatticeEvolver(alat)

    # Warm-up: refine a shell up to max_level (same recipe as perf.py's "shell" mode).
    for level in range(MAX_LEVEL):
        info.max_level = level + 1
        cl.info = info
        _split_shell(cl, R_SHELL, DR)
        evolver.adapt()
        evolver.rebuild()

    N = cl.n_leaves
    print(f"N={N} leaves after warm-up refinement\n")

    # Steady state: criterion held strictly between split_lo and split_hi
    # everywhere, so the split/coarsen passes are both ENABLED (split_lo>0
    # triggers the g_to_j build every call) but nothing actually changes leaf
    # count -- isolates the map-build cost from split/coarsen churn.
    info.split_hi = 0.75
    info.split_lo = 0.25
    cl.info = info
    cmax = np.full(N, 0.5)
    cl.set_criterion_max(cmax)

    t0 = time.perf_counter()
    for _ in range(N_BENCH):
        evolver.adapt()
    elapsed = time.perf_counter() - t0

    N_after = cl.n_leaves
    print(f"N after bench: {N_after} (should equal {N} -- steady state, no churn)")
    print(f"{N_BENCH} adapt() calls  {elapsed:.4f}s  {N_BENCH / elapsed:.1f} calls/s")
    print()

    _lat.stop_phlop()
    print("=== scope timings ===")
    _print_timings()
