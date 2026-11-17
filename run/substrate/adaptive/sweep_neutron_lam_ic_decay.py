"""
Neutron decay sweep — converged lam IC vs baseline.

Two-phase experiment for each alpha:
  Phase 1 (frozen phi): run step_lam_relax until convergence to find the
           natural lam equilibrium for the Taylor spheromak phi structure.
  Phase 2 (full decay): use the converged lam as IC, release phi, and run
           the standard neutron decay to compare knot lifetime.

Baseline uses lam from init_spheromak_ic directly (no convergence phase).

Run:
  PYTHONPATH=$PWD python3 run/substrate/adaptive/sweep_neutron_lam_ic_decay.py
  PYTHONPATH=$PWD python3 run/substrate/adaptive/sweep_neutron_lam_ic_decay.py \\
      --alphas 0.5 1.0 2.0 --n-steps 10000 --n-threads 10

  # Skip phase 1 by loading pre-computed lam ICs from sweep_lam_ic_convergence:
  PYTHONPATH=$PWD python3 run/substrate/adaptive/sweep_neutron_lam_ic_decay.py \\
      --lam-ic-dir tmp/lam_ic_sweep --alphas 0.5 1.0 2.0 --n-steps 10000
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from equilibrium import lattice as _lat
from equilibrium.substrate.adaptive import step_lam_relax
from tst.substrate.adaptive import compute_dt
from tst.substrate.adaptive.adaptive_flux_tube import (
    init_spheromak_ic,
    phi_toroidal_circulation,
)

parser = argparse.ArgumentParser()
parser.add_argument("--R", type=float, default=40.0)
parser.add_argument("--sph-frac", type=float, default=0.2)
parser.add_argument("--B0", type=float, default=1.0)
parser.add_argument("--lam-target", type=float, default=1.0)
parser.add_argument("--u-max", type=float, default=1.0)
parser.add_argument("--amplitude", type=float, default=1.0)
parser.add_argument("--sigma", type=float, default=None)
parser.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 2.0])
parser.add_argument(
    "--converge-rate",
    type=float,
    default=2.0,
    help="lam_relax rate for phase 1 convergence",
)
parser.add_argument(
    "--converge-tol",
    type=float,
    default=1e-5,
    help="convergence threshold for phase 1",
)
parser.add_argument("--converge-max-steps", type=int, default=20000)
parser.add_argument("--lam-scale", type=float, default=0.05)
parser.add_argument("--cross-scale", type=float, default=0.01)
parser.add_argument("--n-steps", type=int, default=8000)
parser.add_argument("--report", type=int, default=500)
parser.add_argument("--sponge", type=float, default=0.15)
parser.add_argument("--damp-max", type=float, default=0.003)
parser.add_argument("--n-threads", type=int, default=10)
parser.add_argument("--out-dir", type=str, default="tmp/neutron_lam_ic_decay")
parser.add_argument(
    "--lam-ic-dir",
    type=str,
    default=None,
    help="directory with pre-computed lam ICs from sweep_lam_ic_convergence.py; "
    "skips phase 1 for any alpha whose alpha_<a>.npz exists there",
)
parser.add_argument(
    "--no-baseline", action="store_true", help="skip the baseline decay run"
)
parser.add_argument("--no-plot", action="store_true")

R_SCAN = (2, 4, 6, 8, 10, 12, 15, 20)


def build_grid(args):
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    alat.set_n_threads(args.n_threads)
    cl = alat[0]
    _lat.build_bcc_sphere(cl, args.R, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)
    r_all = np.linalg.norm(np.array(cl.leaf_pos), axis=1)
    r_inner = float(r_all.max()) * (1.0 - args.sponge)
    cl.set_bc(
        _lat.BoundaryCondition.quadratic_sponge(
            target=0.0, damp_max=args.damp_max, r_all=r_all, r_inner=r_inner
        )
    )
    return alat, cl


def set_phi_ic(cl, args):
    scale = args.R * args.sph_frac
    init_spheromak_ic(
        cl,
        R_sph=1.0,
        scale=scale,
        B0=args.B0,
        lam_target=args.lam_target,
        u_max=args.u_max,
        amplitude=args.amplitude,
        sigma=args.sigma,
    )
    return scale


def converge_lam(cl, dt, args, alpha, out_dir):
    """Phase 1: freeze phi, run lam_relax to convergence. Returns converged lam."""
    cl.set_leaf_lam(np.zeros(cl.n_leaves))
    phi_mag = np.sqrt((np.array(cl.leaf_phi) ** 2).sum(axis=0))
    threshold = args.converge_tol * (float(phi_mag.max()) or 1.0)
    for step in range(args.converge_max_steps):
        lam_old = np.array(cl.leaf_lam)
        step_lam_relax(cl, dt, args.converge_rate, alpha)
        delta = float(np.max(np.abs(np.array(cl.leaf_lam) - lam_old)))
        if delta < threshold:
            print(f"    lam converged at step {step+1}  max|Δlam|={delta:.2e}")
            break
    else:
        print(f"    lam did not converge  max|Δlam|={delta:.2e}")
    return np.array(cl.leaf_lam).copy()


def run_decay(alat, cl, dt, lam_ic, args, label, out_dir):
    """Phase 2: set lam_ic, release phi, run full decay. Returns timeseries dict."""
    centre = np.zeros(3)
    cl.set_leaf_lam(lam_ic)

    n_snaps = args.n_steps // args.report + 1
    times = np.zeros(n_snaps)
    H_hist = np.zeros(n_snaps)
    rms_hist = np.zeros(n_snaps)
    circ_hist = np.zeros((n_snaps, len(R_SCAN)))
    lam_max_h = np.zeros(n_snaps)
    lam_mean_h = np.zeros(n_snaps)

    si = 0
    for step in range(args.n_steps + 1):
        if step % args.report == 0:
            phi_a = np.array(cl.leaf_phi)
            u_a = np.array(cl.leaf_u)
            lam_a = np.array(cl.leaf_lam)
            times[si] = step * dt
            H_hist[si] = float(np.sum(u_a**2) + 0.5 * np.sum(phi_a**2))
            rms_hist[si] = float(np.sqrt(np.mean(phi_a**2)))
            lam_max_h[si] = float(lam_a.max())
            lam_mean_h[si] = float(lam_a.mean())
            for ki, r in enumerate(R_SCAN):
                circ_hist[si, ki] = phi_toroidal_circulation(cl, centre, r)
            cs = "  ".join(f"{circ_hist[si,ki]:+.2e}" for ki in range(len(R_SCAN)))
            print(
                f"    {step:6d}  H={H_hist[si]:.4e}  rms={rms_hist[si]:.4e}"
                f"  lam={lam_max_h[si]:.3f}/{lam_mean_h[si]:.2e}  {cs}",
                flush=True,
            )
            si += 1

        if step < args.n_steps:
            alat.step(0, dt)
            alat.step_sponge(0)
            if args.cross_scale > 0.0:
                alat.step_cross(0, dt, args.cross_scale)
            alat.step_lam(0, dt, args.lam_scale)

    pos = np.array(cl.leaf_pos)
    result = dict(
        times=times[:si],
        H=H_hist[:si],
        phi_rms=rms_hist[:si],
        circ=circ_hist[:si],
        lam_max=lam_max_h[:si],
        lam_mean=lam_mean_h[:si],
        R_SCAN=np.array(R_SCAN),
        label=label,
        pos=pos,
        lam_end=np.array(cl.leaf_lam),
    )
    np.savez(
        out_dir / f"{label}.npz", **{k: v for k, v in result.items() if k != "label"}
    )
    return result


def main():
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    alat, cl = build_grid(args)
    scale = set_phi_ic(cl, args)
    dt = compute_dt(cl)

    print(f"N={cl.n_leaves}  R={args.R}  scale={scale:.1f}  dt={dt:.4e}")
    print(f"alphas: {args.alphas}  n_steps: {args.n_steps}\n")

    results = {}
    lam_ic_dir = Path(args.lam_ic_dir) if args.lam_ic_dir else None

    # Load any previously saved results so they appear in plots without rerunning
    for npz in sorted(out_dir.glob("*.npz")):
        label = npz.stem
        d = np.load(npz)
        results[label] = {k: d[k] for k in d.files}
        results[label]["label"] = label
        print(f"loaded existing result: {label}")

    if not args.no_baseline and "baseline" not in results:
        # --- Baseline: lam from init_spheromak_ic (mirrors phi) ---
        print("=== baseline (lam from spheromak IC) ===")
        lam_baseline = np.array(cl.leaf_lam).copy()
        results["baseline"] = run_decay(
            alat, cl, dt, lam_baseline, args, "baseline", out_dir
        )

    # --- Converged lam IC for each alpha ---
    for alpha in args.alphas:
        label = f"alpha_{alpha:.2f}"
        if label in results:
            print(f"\n=== {label} — already done, skipping ===")
            continue
        print(f"\n=== {label} ===")

        # Fresh lattice for each alpha — build_bcc_sphere appends, not replaces
        alat, cl = build_grid(args)
        set_phi_ic(cl, args)
        dt = compute_dt(cl)

        precomputed = lam_ic_dir / f"alpha_{alpha:.4f}.npz" if lam_ic_dir else None
        if precomputed and precomputed.exists():
            d = np.load(precomputed)
            lam_pre = d["lam"].astype(np.float64)
            cur_pos = np.array(cl.leaf_pos)
            if len(lam_pre) == cl.n_leaves:
                lam_ic = lam_pre
                print(
                    f"  Phase 1: loaded lam IC from {precomputed}  (max={lam_ic.max():.4f})"
                )
            else:
                from scipy.spatial import KDTree

                tree = KDTree(d["pos"])
                _, idx = tree.query(cur_pos, workers=-1)
                lam_ic = lam_pre[idx]
                print(
                    f"  Phase 1: interpolated lam IC ({len(lam_pre)} → {cl.n_leaves})  "
                    f"(max={lam_ic.max():.4f})"
                )
        else:
            print(f"  Phase 1: converging lam (alpha={alpha})...")
            lam_ic = converge_lam(cl, dt, args, alpha, out_dir)
            set_phi_ic(cl, args)

        print("  Phase 2: decay run...")
        results[label] = run_decay(alat, cl, dt, lam_ic, args, label, out_dir)

    if args.no_plot:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    preferred = ([] if args.no_baseline else ["baseline"]) + [
        f"alpha_{a:.2f}" for a in args.alphas
    ]
    labels = [lb for lb in preferred if lb in results]
    cmap = plt.cm.tab10(np.linspace(0, 0.9, len(labels)))
    r_mid = len(R_SCAN) // 2

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    fig.suptitle(
        f"Neutron decay — converged lam IC vs baseline  R={args.R}  scale={scale:.0f}",
        fontsize=10,
    )
    ax_H, ax_rms, ax_lmax = axes[0]
    ax_lmean, ax_circ, ax_circ2 = axes[1]

    for c, label in zip(cmap, labels):
        d = results[label]
        t = d["times"]
        ax_H.semilogy(t, d["H"], color=c, lw=1.5, label=label)
        ax_rms.plot(t, d["phi_rms"], color=c, lw=1.5, label=label)
        ax_lmax.plot(t, d["lam_max"], color=c, lw=1.5, label=label)
        ax_lmean.plot(t, d["lam_mean"], color=c, lw=1.5, label=label)
        ax_circ.plot(t, d["circ"][:, r_mid], color=c, lw=1.5, label=label)
        ax_circ2.plot(t, d["circ"][:, -1], color=c, lw=1.5, label=label)

    for ax, title, ylabel in [
        (ax_H, "Total energy (log)", "H"),
        (ax_rms, "phi RMS", "phi_rms"),
        (ax_lmax, "lam max", "lam_max"),
        (ax_lmean, "lam mean", "lam_mean"),
        (ax_circ, f"circ r={R_SCAN[r_mid]}", "circ"),
        (ax_circ2, f"circ r={R_SCAN[-1]}", "circ"),
    ]:
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xlabel("time", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    plt.savefig(out_dir / "decay_comparison.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("\nsaved decay_comparison.png")


if __name__ == "__main__":
    main()
