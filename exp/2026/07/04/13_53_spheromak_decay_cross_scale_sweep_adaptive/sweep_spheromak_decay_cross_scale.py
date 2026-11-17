"""
Adaptive spheromak decay — cross_scale calibration sweep.

Re-run of exp/2026/06/18/14_00_spheromak_decay_sweep (cs=0.005 viable,
cs=0.010 "too aggressive") — now stale for two reasons:
  1. That sweep predates the sigma(lam)^2 resistive fix to step_cross — lam
     stayed pinned at 1.0 the whole run with zero resistance, so decay was
     completely lam-blind.
  2. It also predates the adaptive step_cross sign-convention fix (2026-07-04):
     the old adaptive step_cross double-processed every antisymmetric face
     pair (see doc/mems/project_step_cross.md), so even a lam-blind comparison
     would have decayed ~2x faster than the now-fixed single-processing path.

Multiprocessing pattern mirrors tst/substrate/validate_step_cross.py: each
cross_scale value runs in its own subprocess via a top-level picklable worker
that builds its own AdaptiveLattice from scratch (deterministic IC — no
randomness in find_toroidal_spheromak/init_toroidal_winding — so every worker
reproduces the identical IC without needing to pickle large arrays across the
process boundary).

Default sweep: 8 log-spaced cross_scale values in [1e-4, 1e-2].

Run:
  PYTHONPATH=$PWD python3 tst/substrate/adaptive/sweep_spheromak_decay_cross_scale.py
  PYTHONPATH=$PWD python3 tst/substrate/adaptive/sweep_spheromak_decay_cross_scale.py \\
      --cross-scales 1e-4 3e-4 1e-3 3e-3 1e-2 --n-steps 50000
"""

import argparse
import multiprocessing as mp
import sys

import numpy as np

sys.path.insert(0, ".")

from equilibrium import lattice as _lat
from tst.substrate.adaptive import compute_dt
from tst.substrate.adaptive.adaptive_flux_tube import (
    find_toroidal_spheromak,
    init_toroidal_winding,
    phi_ring_circulation,
)

parser = argparse.ArgumentParser()
parser.add_argument("--R", type=float, default=40.0)
parser.add_argument("--amplitude", type=float, default=1.0)
parser.add_argument(
    "--lam-rings", type=float, default=1.0, help="lam on ring tiles (0=vacuum)"
)
parser.add_argument("--lam-scale", type=float, default=0.05)
parser.add_argument("--lam-mode", choices=["dynamic", "frozen"], default="dynamic")
parser.add_argument("--axis", default="z")
parser.add_argument("--sponge", type=float, default=0.15)
parser.add_argument("--damp-max", type=float, default=0.003)
parser.add_argument("--n-steps", type=int, default=50000)
parser.add_argument("--report", type=int, default=1000)
parser.add_argument("--n-jobs", type=int, default=8)
parser.add_argument("--lo", type=float, default=1e-4)
parser.add_argument("--hi", type=float, default=1e-2)
parser.add_argument(
    "--cross-scales",
    type=float,
    nargs="+",
    default=None,
    help="explicit cross_scale values (overrides --n-jobs log-spacing between --lo/--hi)",
)
parser.add_argument(
    "--out", type=str, default="tmp/spheromak_decay_cross_scale_sweep.png"
)


def _build_and_run(
    cross_scale,
    R,
    amplitude,
    lam_rings,
    lam_scale,
    lam_mode,
    axis,
    sponge,
    damp_max,
    n_steps,
    report,
):
    """Build a fresh Hopf-ring spheromak IC and run one cross_scale to n_steps.

    Deterministic — every call with the same kwargs (aside from cross_scale)
    produces a bit-identical IC, so this is safe to call independently in
    each worker process without sharing state.
    """
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    alat.set_n_threads(
        1
    )  # avoid oversubscription — parallelism is at the process level
    cl = alat[0]

    _lat.build_bcc_sphere(cl, R, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)

    r_all = np.linalg.norm(np.array(cl.leaf_pos), axis=1)
    r_max = float(r_all.max())
    r_inner = r_max * (1.0 - sponge)
    cl.set_bc(
        _lat.BoundaryCondition.quadratic_sponge(
            target=0.0, damp_max=damp_max, r_all=r_all, r_inner=r_inner
        )
    )

    A_js, A_faces, B_js, B_faces = find_toroidal_spheromak(cl, axis=axis)
    if A_js is None:
        raise RuntimeError("no toroidal spheromak found — lattice too small?")
    init_toroidal_winding(cl, A_js, A_faces, amplitude=amplitude, phi_only=True)
    init_toroidal_winding(cl, B_js, B_faces, amplitude=amplitude, phi_only=True)

    ring_idx = set(A_js) | set(B_js)
    if lam_rings > 0.0:
        lam = np.array(cl.leaf_lam)
        for i in ring_idx:
            lam[i] = lam_rings
        cl.set_leaf_lam(lam)

    centre_A = np.array(cl.leaf_pos)[A_js].mean(axis=0)
    centre_B = np.array(cl.leaf_pos)[B_js].mean(axis=0)
    dt = compute_dt(cl)

    n_rep = n_steps // report + 1
    t_hist = np.zeros(n_rep)
    H_hist = np.zeros(n_rep)
    phi_rms_h = np.zeros(n_rep)
    lam_A_h = np.zeros(n_rep)
    lam_B_h = np.zeros(n_rep)
    circ_A_h = np.zeros(n_rep)
    circ_B_h = np.zeros(n_rep)

    def record(si, step):
        phi_a = np.array(cl.leaf_phi)
        u_a = np.array(cl.leaf_u)
        lam_a = np.array(cl.leaf_lam)
        t_hist[si] = step * dt
        H_hist[si] = float(np.sum(u_a**2) + 0.5 * np.sum(phi_a**2))
        phi_rms_h[si] = float(np.sqrt(np.mean(phi_a**2)))
        lam_A_h[si] = float(np.mean(lam_a[A_js]))
        lam_B_h[si] = float(np.mean(lam_a[B_js]))
        circ_A_h[si] = phi_ring_circulation(cl, A_js, centre_A)
        circ_B_h[si] = phi_ring_circulation(cl, B_js, centre_B)

    si = 0
    record(0, 0)
    for step in range(1, n_steps + 1):
        alat.step(0, dt)
        alat.step_sponge(0)
        if cross_scale > 0.0:
            alat.step_cross(0, dt, cross_scale)
        if lam_mode == "dynamic":
            alat.step_lam(0, dt, lam_scale)
        if step % report == 0:
            si += 1
            record(si, step)

    return dict(
        t=t_hist[: si + 1],
        H=H_hist[: si + 1],
        phi_rms=phi_rms_h[: si + 1],
        lam_A=lam_A_h[: si + 1],
        lam_B=lam_B_h[: si + 1],
        circ_A=circ_A_h[: si + 1],
        circ_B=circ_B_h[: si + 1],
        n_A=len(A_js),
        n_B=len(B_js),
        dt=dt,
    )


def _worker(task):
    """Top-level, picklable — runs one cross_scale value in its own subprocess."""
    cross_scale, kw = task
    return cross_scale, _build_and_run(cross_scale, **kw)


def main():
    args = parser.parse_args()

    if args.cross_scales:
        cross_scales = sorted(args.cross_scales)
    else:
        cross_scales = list(
            np.logspace(np.log10(args.lo), np.log10(args.hi), args.n_jobs)
        )

    kw = dict(
        R=args.R,
        amplitude=args.amplitude,
        lam_rings=args.lam_rings,
        lam_scale=args.lam_scale,
        lam_mode=args.lam_mode,
        axis=args.axis,
        sponge=args.sponge,
        damp_max=args.damp_max,
        n_steps=args.n_steps,
        report=args.report,
    )

    print(
        f"R={args.R}  n_steps={args.n_steps}  lam_rings={args.lam_rings}  "
        f"lam_scale={args.lam_scale}  lam_mode={args.lam_mode}"
    )
    print(
        f"cross_scales ({len(cross_scales)}): "
        + ", ".join(f"{c:.4e}" for c in cross_scales)
    )

    tasks = [(cs, kw) for cs in cross_scales]
    with mp.Pool(len(tasks)) as pool:
        results = dict(pool.map(_worker, tasks))

    # ---------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------
    any_cs = cross_scales[0]
    print(
        f"\nN_A={results[any_cs]['n_A']}  N_B={results[any_cs]['n_B']}  "
        f"dt={results[any_cs]['dt']:.4e}"
    )
    hdr = (
        f"{'cross_scale':>12}  {'phi_rms_0':>10}  {'phi_rms_f':>10}  "
        f"{'H_0':>10}  {'H_f':>10}  {'lam_A_f':>8}  {'lam_B_f':>8}  "
        f"{'circ_B_0':>10}  {'circ_B_f':>10}"
    )
    print(f"\n{hdr}")
    print("-" * len(hdr))
    for cs in cross_scales:
        h = results[cs]
        print(
            f"{cs:12.4e}  {h['phi_rms'][0]:10.4e}  {h['phi_rms'][-1]:10.4e}  "
            f"{h['H'][0]:10.4e}  {h['H'][-1]:10.4e}  {h['lam_A'][-1]:8.4f}  "
            f"{h['lam_B'][-1]:8.4f}  {h['circ_B'][0]:+10.4e}  {h['circ_B'][-1]:+10.4e}"
        )

    # ---------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    cmap = plt.get_cmap("viridis")
    for i, cs in enumerate(cross_scales):
        h = results[cs]
        color = cmap(i / max(len(cross_scales) - 1, 1))
        axes[0].semilogy(h["t"], h["H"] + 1e-30, color=color, label=f"cs={cs:.2e}")
        axes[1].semilogy(
            h["t"], np.abs(h["circ_B"]) + 1e-30, color=color, label=f"cs={cs:.2e}"
        )
        axes[2].plot(h["t"], h["lam_A"], color=color, label=f"cs={cs:.2e}")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("H (log)")
    axes[0].set_title("Total energy")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("|circ_B| (log)")
    axes[1].set_title("Ring B circulation")
    axes[2].set_xlabel("time")
    axes[2].set_ylabel("lam_A")
    axes[2].set_title("Ring A lam (resistance check)")
    for ax in axes:
        ax.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(args.out, dpi=130)
    print(f"\nplot saved -> {args.out}")


if __name__ == "__main__":
    main()
