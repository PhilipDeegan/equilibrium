"""
Harris sheet IC sweep — parameter study with lam_bg.

Runs harris_sheet_phare.py at each (B0, lam_bg) configuration, 10k steps,
saving a 4/5-panel PNG every 500 steps and a diagnostics CSV.
After all configs run, generates a comparison overlay plot.

Edit CONFIGS to adjust the parameter grid.  Results land in:
  <out_dir>/<label>/plots/step_NNNNNN.png
  <out_dir>/<label>/diag.csv
  <out_dir>/<label>/harris_phare_{init,final}.h5
  <out_dir>/comparison.png

Usage:
  PYTHONPATH=$PWD python3 run/substrate/adaptive/harris_ic_sweep.py [--out-dir tmp/harris_sweep]
  PYTHONPATH=$PWD python3 run/substrate/adaptive/harris_ic_sweep.py --dry-run
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _label(B0, lam_bg, lam_peak, lam_relax_rate=0.0, lam_alpha=1.0):
    ms = (
        f"_rr{int(lam_relax_rate*1000):04d}_a{int(lam_alpha):02d}"
        if lam_relax_rate > 0.0
        else ""
    )
    return f"B{int(B0*1000):03d}_lb{int(lam_bg)}_lp{int(lam_peak*10):02d}{ms}"


# u_bg confirmed inert in first sweep — fixed at 0.4 throughout.
# lam_alpha calibrated via reconnection alpha sweep (exp/2026/06/20/22_36_reconnection_alpha_sweep):
#   alpha=5  → causal threshold entry (~20% suppression expected in harris geometry)
#   alpha=10 → confirmed causal in spheromak (35% suppression, reconnection completes)
#   alpha=15 → stronger suppression, may approach near-stall regime
# Rate=0.1 fixed (confirmed stable). Baseline (rate=0.0) included for comparison.
CONFIGS = [
    dict(
        B0=B0,
        lam_bg=lam_bg,
        lam_peak=lam_peak,
        u_bg=0.4,
        lam_relax_rate=lam_relax_rate,
        lam_alpha=lam_alpha,
        label=_label(B0, lam_bg, lam_peak, lam_relax_rate, lam_alpha),
    )
    for B0 in [0.05, 0.10]
    for lam_bg in [0.4, 2.0]
    for lam_peak in [0.0, 1.0]
    for lam_relax_rate, lam_alpha in (
        [(0.0, 1.0)]  # baseline: no relax (alpha irrelevant)
        + [(0.1, a) for a in [5, 10, 15]]  # alpha sweep at calibrated rate
    )
]  # 2×2×2×4 = 32 configs

# Targeted rerun for lb=0 configs that either grew (lp=1) or crashed (lp=2.5).
# Higher blowup_factor + longer run to see if they saturate.
RERUN_LB0 = [
    dict(
        B0=B0,
        lam_bg=0.0,
        lam_peak=lam_peak,
        u_bg=0.4,
        lam_relax_rate=0.0,
        n_steps=30000,
        blowup_factor=500.0,
        label=_label(B0, 0.0, lam_peak) + "_rerun",
    )
    for B0 in [0.05, 0.10]
    for lam_peak in [1.0, 2.5]
]

SHARED = dict(
    Lx=100,
    Ly=100,
    Lz=3.5,
    n_steps=10000,
    report=500,
    plot_every=500,
    dt=0.05,
    perturb=0.1,
    blowup_factor=50.0,
    no_amr=True,
    threads=10,
)

DIAG_COLS = [
    "step",
    "phi_E",
    "phi_E_ratio",
    "phi_null_rms",
    "phi_xpt_by",
    "u_rms",
    "lam_std",
]
DIAG_LABELS = {
    "phi_E_ratio": "phi_E / phi_E0  (growth = instability)",
    "phi_null_rms": "phi_rms at null line  (field at current sheet)",
    "phi_xpt_by": "|By| at X-point  (reconnecting flux)",
    "u_rms": "u rms  (pressure evolution)",
    "lam_std": "lam std  (winding redistribution)",
}

parser = argparse.ArgumentParser()
parser.add_argument("--out-dir", default="tmp/harris_sweep")
parser.add_argument("--dry-run", action="store_true", help="print commands, don't run")


def build_cmd(cfg, out_dir):
    label = cfg["label"]
    run_dir = os.path.join(out_dir, label)
    plot_dir = os.path.join(run_dir, "plots")
    prefix = os.path.join(run_dir, "harris_phare")
    diag_out = os.path.join(run_dir, "diag.csv")

    cmd = [
        sys.executable,
        "run/substrate/adaptive/harris_sheet_phare.py",
        "--Lx",
        str(SHARED["Lx"]),
        "--Ly",
        str(SHARED["Ly"]),
        "--Lz",
        str(SHARED["Lz"]),
        "--B0",
        str(cfg["B0"]),
        "--lam-bg",
        str(cfg["lam_bg"]),
        "--lam-peak",
        str(cfg.get("lam_peak", 0.0)),
        "--u-bg",
        str(cfg.get("u_bg", 0.4)),
        "--n-steps",
        str(cfg.get("n_steps", SHARED["n_steps"])),
        "--report",
        str(SHARED["report"]),
        "--plot-every",
        str(SHARED["plot_every"]),
        "--plot-dir",
        plot_dir,
        "--save-prefix",
        prefix,
        "--diag-out",
        diag_out,
        "--dt",
        str(SHARED["dt"]),
        "--perturb",
        str(SHARED["perturb"]),
        "--threads",
        str(SHARED["threads"]),
        "--blowup-factor",
        str(cfg.get("blowup_factor", SHARED["blowup_factor"])),
        "--lam-relax-rate",
        str(cfg.get("lam_relax_rate", 0.0)),
        "--lam-alpha",
        str(cfg.get("lam_alpha", 1.0)),
    ]
    if SHARED.get("no_amr"):
        cmd.append("--no-amr")
    return cmd, run_dir


def load_diag(run_dir):
    path = os.path.join(run_dir, "diag.csv")
    if not os.path.exists(path):
        return None
    data = {}
    with open(path) as f:
        header = f.readline().strip().split(",")
        rows = [list(map(float, ln.strip().split(","))) for ln in f if ln.strip()]
    if not rows:
        return None
    arr = np.array(rows)
    for i, col in enumerate(header):
        data[col] = arr[:, i]
    return data


def _smooth(y, w=5):
    """Rolling mean — reveals envelope through phase oscillation."""
    if len(y) < w:
        return y
    out = np.convolve(y, np.ones(w) / w, mode="valid")
    # pad to same length by repeating edges
    pad = len(y) - len(out)
    return np.concatenate(
        [out[:1].repeat(pad // 2), out, out[-1:].repeat(pad - pad // 2)]
    )


def comparison_plot(configs, out_dir):
    """
    Two figures:
      comparison.png      — raw phi_E_ratio for all configs (one panel per metric)
      comparison_env.png  — rolling-average envelope, collapsed over u_bg (which has no effect),
                            grouped by (B0, lam_bg), lam_peak as colour
    """
    diags = {}
    for cfg in configs:
        run_dir = os.path.join(out_dir, cfg["label"])
        d = load_diag(run_dir)
        if d is not None:
            diags[cfg["label"]] = (cfg, d)

    if not diags:
        print("No diag CSVs found — skipping comparison plot.")
        return

    plot_cols = list(DIAG_LABELS.keys())

    # --- raw plot (kept for reference, but usually unreadable) ---
    fig0, axes0 = plt.subplots(
        len(plot_cols), 1, figsize=(10, 3 * len(plot_cols)), constrained_layout=True
    )
    fig0.suptitle("Harris sweep — raw diagnostics (all configs)", fontsize=11)
    lam_vals = sorted({cfg["lam_bg"] for cfg, _ in diags.values()})
    b0_vals = sorted({cfg["B0"] for cfg, _ in diags.values()})
    lp_vals = sorted({cfg.get("lam_peak", 0.0) for cfg, _ in diags.values()})
    cmap0 = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(lam_vals), 1)))
    lam_color = {v: cmap0[i] for i, v in enumerate(lam_vals)}
    b0_style = {v: ls for v, ls in zip(b0_vals, ["-", "--", ":", "-."])}
    lp_width = {
        v: 0.8 + 1.5 * i / max(len(lp_vals) - 1, 1) for i, v in enumerate(lp_vals)
    }

    for ax, col in zip(axes0, plot_cols):
        for label, (cfg, d) in diags.items():
            if col not in d:
                continue
            ax.plot(
                d["step"],
                d[col],
                color=lam_color[cfg["lam_bg"]],
                linestyle=b0_style[cfg["B0"]],
                lw=lp_width[cfg.get("lam_peak", 0.0)],
                alpha=0.5,
                label=label,
            )
        ax.set_ylabel(DIAG_LABELS[col], fontsize=8)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
    axes0[0].legend(fontsize=6, ncol=3, loc="upper right")
    fig0.savefig(os.path.join(out_dir, "comparison.png"), dpi=120)
    plt.close(fig0)

    # --- envelope plot: collapse u_bg, smooth, group by (B0, lam_bg) ---
    # Rows = metrics, cols = B0.
    # Within each panel: one line per (lam_peak, lam_relax_rate, lam_alpha) combo.
    # Colour = lam_alpha (blue=baseline/no-relax, then plasma scale for alpha values).
    # Linestyle = lam_peak.
    u_vals = sorted({cfg.get("u_bg", 0.4) for cfg, _ in diags.values()})
    u_pick = u_vals[len(u_vals) // 2]

    groups = {}  # (B0, lam_bg, lam_peak, lam_relax_rate, lam_alpha) → (cfg, d)
    for label, (cfg, d) in diags.items():
        if abs(cfg.get("u_bg", 0.4) - u_pick) > 1e-6:
            continue
        key = (
            cfg["B0"],
            cfg["lam_bg"],
            cfg.get("lam_peak", 0.0),
            cfg.get("lam_relax_rate", 0.0),
            cfg.get("lam_alpha", 1.0),
        )
        groups[key] = (cfg, d)

    if not groups:
        return

    # Colours: baseline (rate=0) in grey; alpha values in plasma scale
    relax_alphas = sorted({a for (_, _, _, rr, a) in groups if rr > 0})
    alpha_colors = {
        a: c
        for a, c in zip(
            relax_alphas,
            plt.cm.plasma(np.linspace(0.15, 0.85, max(len(relax_alphas), 1))),
        )
    }
    alpha_colors[None] = (0.5, 0.5, 0.5, 1.0)  # baseline (no relax)

    lp_style = {v: ls for v, ls in zip(sorted(lp_vals), ["-", "--", ":", "-."])}

    for lb in lam_vals:
        fig, axes = plt.subplots(
            len(plot_cols),
            len(b0_vals),
            figsize=(6 * len(b0_vals), 3 * len(plot_cols)),
            constrained_layout=True,
            squeeze=False,
        )
        fig.suptitle(
            f"Harris sweep — lam_bg={lb}  (smoothed, u_bg={u_pick})\n"
            f"colour=lam_alpha (grey=baseline/no-relax), linestyle=lam_peak",
            fontsize=10,
        )

        for ci, b0 in enumerate(b0_vals):
            for ri, col in enumerate(plot_cols):
                ax = axes[ri, ci]
                ax.set_title(f"B0={b0}", fontsize=8)
                for (B0, lam_bg, lp, rr, la), (cfg, d) in sorted(groups.items()):
                    if abs(B0 - b0) > 1e-6 or abs(lam_bg - lb) > 1e-6:
                        continue
                    if col not in d:
                        continue
                    color = (
                        alpha_colors[None] if rr == 0.0 else alpha_colors.get(la, "k")
                    )
                    lbl = f"α={la:.0f},lp={lp}" if rr > 0 else f"baseline,lp={lp}"
                    y = _smooth(d[col])
                    ax.plot(
                        d["step"],
                        y,
                        color=color,
                        lw=1.8,
                        linestyle=lp_style.get(lp, "-"),
                        label=lbl,
                    )
                ax.set_ylabel(DIAG_LABELS[col], fontsize=7)
                ax.set_xlabel("step")
                ax.grid(True, alpha=0.3)
                if ri == 0:
                    ax.legend(fontsize=7, loc="upper right", ncol=2)

        fname = os.path.join(out_dir, f"comparison_lb{int(lb)}.png")
        fig.savefig(fname, dpi=120)
        plt.close(fig)
        print(f"  envelope plot → {fname}")

    print(f"\nComparison plots → {out_dir}/comparison*.png")


def main():
    args = parser.parse_args()

    configs = CONFIGS
    print(f"Sweep: {len(configs)} configs → {args.out_dir}")
    for cfg in configs:
        cmd, run_dir = build_cmd(cfg, args.out_dir)
        Path(run_dir).mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Config: {cfg['label']}")
        print("  " + " ".join(cmd))

        if args.dry_run:
            continue

        log_path = os.path.join(run_dir, "run.log")
        with open(log_path, "w") as log:
            log.write(" ".join(cmd) + "\n\n")
            log.flush()
            result = subprocess.run(
                cmd,
                env={**os.environ, "PYTHONPATH": os.getcwd()},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            log.write(result.stdout)

        lines = result.stdout.splitlines()
        for ln in lines[-20:]:
            print(f"  {ln}")

        status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
        print(f"  → {status}  log: {log_path}")

    if not args.dry_run:
        comparison_plot(configs, args.out_dir)
        print("Sweep complete.")
    else:
        print("\nDry run — no jobs submitted.")


if __name__ == "__main__":
    main()
