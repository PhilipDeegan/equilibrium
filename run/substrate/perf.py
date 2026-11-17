"""
AdaptiveLattice step performance benchmark.

Builds a BCC sphere and refines a thin shell at r_shell up to max_level,
then benchmarks evolver.step() on the resulting AMR lattice.  Mirrors the
real Bohr-orbit workload (coarse vacuum + refined shell) without blowing up RAM.

Usage:
    PYTHONPATH=$PWD python3 run/substrate/perf.py
    PYTHONPATH=$PWD python3 run/substrate/perf.py --max-level 4 --n-bench 500 --threads 6
    PYTHONPATH=$PWD python3 run/substrate/perf.py --r-shell 9.0 --dr 1.5 --max-level 4
"""

import argparse
import os
import sys
import time

import numpy as np

os.environ["EQUILIBRIUM_SCOPE_TIMING"] = "1"
sys.path.insert(0, ".")

from equilibrium import lattice as _lat

W2_SUM = 8 * 9.0 + 6 * 1.0

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    "--R", type=float, default=15.0, help="BCC sphere radius in cell units"
)
parser.add_argument("--D1", type=float, default=1.0, help="Level-0 cell spacing")
parser.add_argument(
    "--mode",
    choices=["shell", "radial"],
    default="radial",
    help="shell: refine thin orbit shell; radial: finest at centre, coarsest at edge",
)
parser.add_argument(
    "--r-shell",
    type=float,
    default=9.0,
    help="Radius of the refinement shell (shell mode)",
)
parser.add_argument(
    "--dr",
    type=float,
    default=1.0,
    help="Half-width of the refinement shell (shell mode)",
)
parser.add_argument(
    "--max-level",
    type=int,
    default=0,
    help="Number of AMR refinement levels applied to the shell",
)
parser.add_argument("--n-bench", type=int, default=10, help="Number of steps to time")
parser.add_argument(
    "--threads", type=int, default=6, help="Thread count passed to AdaptiveLattice"
)
args, _ = parser.parse_known_args()

R = args.R
D1 = args.D1
MODE = args.mode
R_SHELL = args.r_shell
DR = args.dr
MAX_LEVEL = args.max_level
N_BENCH = args.n_bench
THREADS = args.threads


def _dt(cl):
    ell_arr = np.array(cl.leaf_ell)
    fu_bool = np.array(cl.leaf_fixed_u).astype(bool)
    lam_arr = np.array(cl.leaf_lam, dtype=np.float64)
    free = ~fu_bool
    sigma_max = float(np.max((ell_arr / (1.0 + lam_arr))[free])) if free.any() else 1.0
    L_min = D1 / (2.0 ** int(np.array(cl.leaf_level).max()))
    return 0.4 * L_min / (sigma_max * W2_SUM)


def _split_shell(cl, r_shell, dr=1.0):
    pos = np.array(cl.leaf_pos)
    r = np.sqrt(np.sum(pos**2, axis=1))
    cmax = np.full(cl.n_leaves, np.nan)
    cmax[np.abs(r - r_shell) < dr] = 1.0
    cl.set_criterion_max(cmax)


def _split_radial(cl, r_max, level, max_level):
    """Split cells within a shrinking radius — finest at centre, coarsest at edge."""
    pos = np.array(cl.leaf_pos)
    r = np.sqrt(np.sum(pos**2, axis=1))
    r_thresh = r_max * (max_level - level) / max_level
    cmax = np.full(cl.n_leaves, np.nan)
    cmax[r < r_thresh] = 1.0
    cl.set_criterion_max(cmax)


def _print_timings(path=".phlop/timings.txt"):
    if not os.path.exists(path):
        print("(no .phlop/timings.txt — check EQUILIBRIUM_SCOPE_TIMING build support)")
        return
    import phlop.timing.scope_timer as phst

    phst.print_scope_timings(path, sort_worst_first=True)


if __name__ == "__main__":
    _lat.start_phlop()
    print(
        f"R={R}  D1={D1}  r_shell={R_SHELL}  dr={DR}  max_level={MAX_LEVEL}"
        f"  n_bench={N_BENCH}  threads={THREADS}"
    )

    def _run():
        alat = _lat.AdaptiveLattice()
        alat.add_cluster()
        cl = alat[0]

        _lat.build_bcc_sphere(cl, R, D1, u0=1.0, ell0=1.0, lam0=0)
        alat.rebuild(0)
        alat.set_n_threads(THREADS)

        info = _lat.DSTInfo()
        info.split_hi = 0.5
        info.split_lo = 0.0
        cl.set_bc(_lat.BoundaryCondition.absorbing())

        evolver = _lat.AdaptiveLatticeEvolver(alat)

        for level in range(MAX_LEVEL):
            info.max_level = level + 1
            cl.info = info
            if MODE == "radial":
                _split_radial(cl, R, level, MAX_LEVEL)
            else:
                _split_shell(cl, R_SHELL, DR)
            evolver.adapt()
            evolver.rebuild()

        lvl_arr = np.array(cl.leaf_level)
        counts = {int(lv): int(np.sum(lvl_arr == lv)) for lv in np.unique(lvl_arr)}
        dist = "  ".join(f"L{lv}:{n}" for lv, n in sorted(counts.items()))
        dt = _dt(cl)
        N = cl.n_leaves
        print(f"N={N}  dt={dt:.3e}  levels=[{dist}]\n")

        t0 = time.perf_counter()
        evolver.step(dt, N_BENCH)
        elapsed = time.perf_counter() - t0
        return elapsed

    elapsed = _run()
    print(f"{N_BENCH} steps  {elapsed:.3f}s  {N_BENCH/elapsed:.1f} steps/s")
    print()

    _lat.stop_phlop()
    print("=== scope timings ===")
    _print_timings()
