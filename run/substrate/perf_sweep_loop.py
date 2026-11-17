"""
Amdahl breakdown of the neutron-decay sweep's per-step loop.

Mirrors run/substrate/adaptive/sweep_neutron_decay_cross_scale.py's
_build_and_run main loop — Taylor spheromak IC, AMR pre-adapt, then the
interleaved evolver.step / step_sponge / step_cross / step_lam tick — and
times each phase separately across a list of thread counts, so the serial
fraction of every phase can be read straight off the scaling table
(efficiency = T1 / (n * Tn); an Amdahl fit s = (n*Tn/T1 - 1)/(n - 1) is
printed per phase).

Also times, once, the Python-side costs the sweep pays outside the C++ tick:
  - record()'s legacy per-shell phi_toroidal_circulation loop over R_SCAN
    vs the vectorized face_flux_geometry + phi_toroidal_scan equivalent
  - steady-state adapt() calls (converged mesh, split_interval=1)
  - the _rebuild_sponge Python round-trip

Usage:
    PYTHONPATH=$PWD python3 run/substrate/perf_sweep_loop.py
    PYTHONPATH=$PWD python3 run/substrate/perf_sweep_loop.py \\
        --R 30 --max-level 1 --split-hi 0.02 --n-bench 200 --threads 1 4 10
"""

import argparse
import os
import sys
import time

import numpy as np

os.environ["EQUILIBRIUM_SCOPE_TIMING"] = "1"
sys.path.insert(0, ".")

from equilibrium import lattice as _lat
from run.substrate.adaptive.sweep_spheromak_decay_cross_scale import _compute_dt_l0
from tst.substrate.adaptive.adaptive_flux_tube import (
    face_flux_geometry,
    init_spheromak_ic,
    phi_toroidal_circulation,
    phi_toroidal_scan,
)

R_SCAN = (2, 4, 6, 8, 10, 12, 15, 20, 25, 30)
BUILD_THREADS = 10  # pre-adapt is not what we measure — always build fast

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--R", type=float, default=24.0)
parser.add_argument("--sph-frac", type=float, default=0.2)
parser.add_argument("--max-level", type=int, default=1)
parser.add_argument("--split-hi", type=float, default=0.02)
parser.add_argument("--split-lo", type=float, default=0.0)
parser.add_argument("--split-ref", type=float, default=1e-6)
parser.add_argument("--cross-scale", type=float, default=1e-3)
parser.add_argument("--m0", type=float, default=1e-2)
parser.add_argument("--lam-scale", type=float, default=0.05)
parser.add_argument("--sponge", type=float, default=0.15)
parser.add_argument("--damp-max", type=float, default=0.003)
parser.add_argument("--n-bench", type=int, default=100)
parser.add_argument("--threads", type=int, nargs="+", default=[1, 4, 10])
parser.add_argument("--phlop", action="store_true", help="print scope-timer totals")
args, _ = parser.parse_known_args()


def _build():
    """Sweep-identical build: sphere + sponge + spheromak IC + AMR pre-adapt."""
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    alat.set_n_threads(BUILD_THREADS)
    cl = alat[0]

    _lat.build_bcc_sphere(cl, args.R, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)

    r_all = np.linalg.norm(np.array(cl.leaf_pos), axis=1)
    r_inner = float(r_all.max()) * (1.0 - args.sponge)

    def _rebuild_sponge():
        r_now = np.linalg.norm(np.array(cl.leaf_pos), axis=1)
        cl.set_bc(
            _lat.BoundaryCondition.quadratic_sponge(
                target=0.0, damp_max=args.damp_max, r_all=r_now, r_inner=r_inner
            )
        )

    _rebuild_sponge()
    scale = args.R * args.sph_frac

    def _seed_ic():
        return init_spheromak_ic(
            cl,
            R_sph=1.0,
            scale=scale,
            B0=1.0,
            lam_target=1.0,
            u_max=1.0,
            amplitude=1.0,
            sigma=None,
        )

    _seed_ic()
    dt = _compute_dt_l0(cl)
    evolver = _lat.AdaptiveLatticeEvolver(alat)

    amr_enabled = args.max_level > 0 and args.split_hi > 0.0
    if amr_enabled:
        sc = _lat.SplitCriterion.relative_u(
            hi=args.split_hi, lo=args.split_lo, ref=args.split_ref
        )
        pre_info = _lat.DSTInfo()
        pre_info.max_level = args.max_level
        pre_info.split_interval = 1
        cl.info = pre_info
        cl.set_split_criterion(sc)
        ts_pre = _lat.TimeStepper.by_level(dt)
        for _ in range(50):
            evolver.step(ts_pre)
            n_before = cl.n_leaves
            alat.adapt(0)
            if cl.n_leaves == n_before:
                break
            _rebuild_sponge()
        _seed_ic()
        dt = _compute_dt_l0(cl)

    has_fine = int(np.max(np.array(cl.leaf_level))) > 0
    ts = (
        _lat.TimeStepper.by_level(dt)
        if (amr_enabled or has_fine)
        else _lat.TimeStepper.fixed(dt)
    )
    return alat, cl, evolver, ts, dt, _rebuild_sponge


def _tick_loop(alat, evolver, ts, dt, n):
    """The sweep's per-step body, with per-phase wall-time accumulators."""
    pc = time.perf_counter
    acc = dict(step=0.0, sponge=0.0, cross=0.0, lam=0.0)
    for _ in range(n):
        t0 = pc()
        evolver.step(ts)
        acc["step"] += pc() - t0
        t0 = pc()
        alat.step_sponge(0)
        acc["sponge"] += pc() - t0
        t0 = pc()
        alat.step_cross(0, dt, args.cross_scale, gate="barrier", m0=args.m0)
        acc["cross"] += pc() - t0
        t0 = pc()
        alat.step_lam(0, dt, args.lam_scale)
        acc["lam"] += pc() - t0
    return acc


def _print_phlop_totals(path=".phlop/timings.txt"):
    if not os.path.exists(path):
        print("(no .phlop/timings.txt — extension built without phlop?)")
        return
    names, totals, counts = {}, {}, {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if len(parts) == 3 and ":" in parts[2]:  # "tid sid start:dur"
                sid = parts[1]
                dur = int(parts[2].split(":")[1])
                totals[sid] = totals.get(sid, 0) + dur
                counts[sid] = counts.get(sid, 0) + 1
            elif len(parts) == 2:  # "sid name"
                names[parts[0]] = parts[1]
    rows = sorted(totals, key=lambda s: -totals[s])
    print(f"{'scope':<70} {'total_s':>9} {'calls':>8} {'ms/call':>9}")
    for sid in rows:
        nm = names.get(sid, sid)
        tot = totals[sid] / 1e9
        print(f"{nm:<70} {tot:9.3f} {counts[sid]:8d} {1e3 * tot / counts[sid]:9.3f}")


if __name__ == "__main__":
    # phlop's scope timer must be initialised before the first EQ_SCOPE_TIMER
    # entry on any thread (see vis/substrate/perf.py) — totals therefore also
    # include the build/pre-adapt phase; use the calls column to normalise.
    if args.phlop:
        _lat.start_phlop()
    results = {}
    n_leaves = lvl_max = None
    keep = None  # last build, reused for the one-off serial measurements

    for nt in args.threads:
        alat, cl, evolver, ts, dt, _rebuild_sponge = _build()
        alat.set_n_threads(nt)
        if n_leaves is None:
            lvl = np.array(cl.leaf_level)
            n_leaves, lvl_max = cl.n_leaves, int(lvl.max())
            dist = "  ".join(f"L{v}:{int((lvl == v).sum())}" for v in np.unique(lvl))
            print(
                f"R={args.R}  N={n_leaves}  [{dist}]  dt={dt:.3e}  "
                f"max_level={args.max_level}  split_hi={args.split_hi}  "
                f"cross_scale={args.cross_scale}  m0={args.m0}  "
                f"n_bench={args.n_bench}\n"
            )
        _tick_loop(alat, evolver, ts, dt, 3)  # warm-up
        results[nt] = _tick_loop(alat, evolver, ts, dt, args.n_bench)
        keep = (alat, cl, evolver, ts, dt, _rebuild_sponge)

    phases = ["step", "sponge", "cross", "lam"]
    n1 = args.threads[0]
    print(
        f"{'phase':<8}"
        + "".join(f"{f'T({nt})ms/it':>14}" for nt in args.threads)
        + f"{'speedup':>9}{'eff':>7}{'serial%':>9}"
    )
    for ph in phases:
        t1 = results[n1][ph] / args.n_bench
        row = f"{ph:<8}"
        for nt in args.threads:
            row += f"{1e3 * results[nt][ph] / args.n_bench:14.2f}"
        nt = args.threads[-1]
        tn = results[nt][ph] / args.n_bench
        sp = t1 / tn if tn > 0 else float("nan")
        eff = sp / (nt / n1)
        s = (nt * tn / t1 - 1) / (nt - 1) if nt > 1 else float("nan")
        row += f"{sp:9.2f}{eff:7.2f}{100 * max(s, 0.0):9.1f}"
        print(row)
    tot1 = sum(results[n1][p] for p in phases) / args.n_bench
    totn = sum(results[args.threads[-1]][p] for p in phases) / args.n_bench
    print(
        f"{'TOTAL':<8}{1e3 * tot1:14.2f}"
        + " " * 14 * (len(args.threads) - 2)
        + f"{1e3 * totn:14.2f}{tot1 / totn:9.2f}\n"
    )

    # ------------------------------------------------------------------
    # One-off serial costs (thread-count independent), at the final state.
    # ------------------------------------------------------------------
    alat, cl, evolver, ts, dt, _rebuild_sponge = keep
    pc = time.perf_counter

    centre = np.zeros(3)
    t0 = pc()
    circ_legacy = [phi_toroidal_circulation(cl, centre, r) for r in R_SCAN]
    t_legacy = pc() - t0
    t0 = pc()
    mid, n_hat, phi_f = face_flux_geometry(cl, centre)
    circ_vec = phi_toroidal_scan(mid, n_hat, phi_f, R_SCAN)
    t_vec = pc() - t0
    err = max(abs(a - b) / (abs(a) + 1e-30) for a, b in zip(circ_legacy, circ_vec))
    print(f"record() probe, {len(R_SCAN)} shells at N={cl.n_leaves}:")
    print(f"  legacy phi_toroidal_circulation loop : {t_legacy:8.3f} s")
    print(
        f"  vectorized face_flux_geometry + scan : {t_vec:8.3f} s   "
        f"(x{t_legacy / t_vec:.0f}, max rel diff {err:.2e})"
    )

    if args.max_level > 0 and args.split_hi > 0.0:
        n_ad = 20
        t0 = pc()
        for _ in range(n_ad):
            alat.adapt(0)
        t_ad = (pc() - t0) / n_ad
        print(f"steady-state adapt()                   : {1e3 * t_ad:8.3f} ms/call")

    t0 = pc()
    _rebuild_sponge()
    print(f"_rebuild_sponge (Python round-trip)    : {1e3 * (pc() - t0):8.3f} ms/call")

    if args.phlop:
        _lat.stop_phlop()
        print("\n=== scope-timer totals (all phases, all thread counts) ===")
        _print_phlop_totals()
