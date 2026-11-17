"""
Sweep lam-relax-rate × lam-alpha for the neutron decay (Taylor spheromak IC).

Runs test_neutron_decay.py for each (rate, alpha) pair with --no-plot --data-out.
rate=0.0 is run once as the shared baseline (alpha is irrelevant when rate=0).

Produces:
  sweep_alpha_<a>.png  — one figure per alpha: rate comparison timeseries
  sweep_summary.png    — final-step lam_max and phi_rms as a heat map over (rate, alpha)

Usage:
  PYTHONPATH=$PWD python3 run/substrate/adaptive/sweep_neutron_lam_relax.py
  PYTHONPATH=$PWD python3 run/substrate/adaptive/sweep_neutron_lam_relax.py \\
      --rates 0.0 0.01 0.05 0.1 0.3 1.0 --alphas 0.1 0.5 1.0 2.0 \\
      --R 40 --n-steps 10000 --n-threads 10
"""

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--rates", type=float, nargs="+", default=[0.0, 0.01, 0.05, 0.1, 0.3, 1.0]
)
parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.5, 1.0, 2.0])
parser.add_argument("--R", type=float, default=40.0)
parser.add_argument("--sph-frac", type=float, default=0.2)
parser.add_argument("--B0", type=float, default=1.0)
parser.add_argument("--lam-target", type=float, default=1.0)
parser.add_argument("--cross-scale", type=float, default=0.01)
parser.add_argument("--lam-scale", type=float, default=0.05)
parser.add_argument("--n-steps", type=int, default=10000)
parser.add_argument("--report", type=int, default=500)
parser.add_argument("--n-threads", type=int, default=10)
parser.add_argument("--out-dir", type=str, default="tmp/neutron_lam_relax_sweep")


def cache_path(out_dir, rate, alpha):
    if rate == 0.0:
        return out_dir / "baseline.npz"
    return out_dir / f"rate_{rate:.4f}_alpha_{alpha:.4f}.npz"


def run_one(rate, alpha, args, out_dir):
    path = cache_path(out_dir, rate, alpha)
    if path.exists():
        print(f"  skip rate={rate} alpha={alpha} (cached)")
        return path
    cmd = [
        sys.executable,
        "run/substrate/adaptive/neutron_decay.py",
        "--R",
        str(args.R),
        "--sph-frac",
        str(args.sph_frac),
        "--B0",
        str(args.B0),
        "--lam-target",
        str(args.lam_target),
        "--cross-scale",
        str(args.cross_scale),
        "--lam-scale",
        str(args.lam_scale),
        "--n-steps",
        str(args.n_steps),
        "--report",
        str(args.report),
        "--n-threads",
        str(args.n_threads),
        "--lam-relax-rate",
        str(rate),
        "--lam-alpha",
        str(alpha),
        "--no-plot",
        "--data-out",
        str(path),
    ]
    label = "baseline" if rate == 0.0 else f"rate={rate} alpha={alpha}"
    print(f"  running {label} ...", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAILED {label}:\n{result.stderr[-2000:]}")
        return None
    print(f"  done {label}")
    return path


def load(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def plot_alpha(baseline, results_by_rate, alpha, args, out_dir):
    """One figure: rate comparison for a fixed alpha."""
    active_rates = sorted(r for r in results_by_rate if r > 0.0)
    n_lines = 1 + len(active_rates)
    cmap = plt.cm.plasma(np.linspace(0.1, 0.85, n_lines))

    R_SCAN = baseline["R_SCAN"]
    r_mid = len(R_SCAN) // 2

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    fig.suptitle(
        f"Neutron decay — rate sweep  (alpha={alpha})\n"
        f"R={args.R}  sph_frac={args.sph_frac}  cross={args.cross_scale}"
        f"  n_steps={args.n_steps}",
        fontsize=10,
    )
    ax_lmax, ax_lstd = axes[0]
    ax_lmean, ax_rms = axes[1]
    ax_H, ax_circ = axes[2]

    def _plot(d, label, c):
        t = d["times"]
        ax_lmax.plot(t, d["lam_max"], color=c, lw=1.5, label=label)
        ax_lstd.plot(t, d["lam_std"], color=c, lw=1.5, label=label)
        ax_lmean.plot(t, d["lam_mean"], color=c, lw=1.5, label=label)
        ax_rms.plot(t, d["phi_rms"], color=c, lw=1.5, label=label)
        ax_H.semilogy(t, d["H"], color=c, lw=1.5, label=label)
        ax_circ.plot(t, d["circ"][:, r_mid], color=c, lw=1.5, label=label)

    _plot(baseline, "baseline (rate=0)", cmap[0])
    for i, rate in enumerate(active_rates):
        _plot(results_by_rate[rate], f"rate={rate}", cmap[1 + i])

    for ax, title, ylabel in [
        (ax_lmax, "lam max", "lam_max"),
        (ax_lstd, "lam std", "lam_std"),
        (ax_lmean, "lam mean", "lam_mean"),
        (ax_rms, "phi RMS", "phi_rms"),
        (ax_H, "total energy (log)", "H"),
        (ax_circ, f"toroidal circ r={R_SCAN[r_mid]}", "circ"),
    ]:
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xlabel("time", fontsize=8)
        ax.grid(True, alpha=0.3)
    ax_lmax.legend(fontsize=7, ncol=2)

    out = out_dir / f"sweep_alpha_{alpha:.4f}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"plot saved → {out}")


def plot_summary(baseline, grid, args, out_dir):
    """Heat map of final lam_max and phi_rms over (rate, alpha)."""
    active_rates = sorted({r for r, _ in grid})
    alphas = sorted({a for _, a in grid})
    if not active_rates or not alphas:
        return

    lam_max_grid = np.zeros((len(active_rates), len(alphas)))
    rms_grid = np.zeros((len(active_rates), len(alphas)))
    for ri, rate in enumerate(active_rates):
        for ai, alpha in enumerate(alphas):
            d = grid.get((rate, alpha))
            if d is not None:
                lam_max_grid[ri, ai] = d["lam_max"][-1]
                rms_grid[ri, ai] = d["phi_rms"][-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    fig.suptitle(
        f"Neutron decay — final-step summary  R={args.R}  n_steps={args.n_steps}",
        fontsize=10,
    )

    def _heatmap(ax, data, title):
        im = ax.imshow(
            data,
            aspect="auto",
            origin="lower",
            extent=[-0.5, len(alphas) - 0.5, -0.5, len(active_rates) - 0.5],
        )
        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels([str(a) for a in alphas], fontsize=8)
        ax.set_yticks(range(len(active_rates)))
        ax.set_yticklabels([str(r) for r in active_rates], fontsize=8)
        ax.set_xlabel("alpha", fontsize=9)
        ax.set_ylabel("rate", fontsize=9)
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.85)
        baseline_val = (
            baseline["lam_max"][-1]
            if title.startswith("lam")
            else baseline["phi_rms"][-1]
        )
        ax.set_title(f"{title}  (baseline={baseline_val:.4f})", fontsize=9)

    _heatmap(ax1, lam_max_grid, "lam_max (final)")
    _heatmap(ax2, rms_grid, "phi_rms (final)")

    out = out_dir / "sweep_summary.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"summary saved → {out}")


def main():
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # baseline — rate=0, alpha irrelevant
    baseline_path = run_one(0.0, 1.0, args, out_dir)
    if baseline_path is None:
        print("baseline run failed — aborting")
        return
    baseline = load(baseline_path)

    # (rate, alpha) grid — skip rate=0
    active_rates = [r for r in args.rates if r > 0.0]
    grid = {}
    for rate in active_rates:
        for alpha in args.alphas:
            path = run_one(rate, alpha, args, out_dir)
            if path is not None:
                grid[(rate, alpha)] = load(path)

    # per-alpha timeseries plots
    for alpha in args.alphas:
        by_rate = {r: grid[(r, alpha)] for r in active_rates if (r, alpha) in grid}
        if by_rate:
            plot_alpha(baseline, by_rate, alpha, args, out_dir)

    # summary heatmap
    plot_summary(baseline, grid, args, out_dir)


if __name__ == "__main__":
    main()
