"""
DST vs PHARE mirror instability — side-by-side comparison at IC / early / late.

Loads the two npz snapshot files produced by:
    run/substrate/mirror_instability_phare.py --extract
    run/substrate/mirror_instability_dst.py

Top row: PHARE's R=T_perp/T_para as a genuine 2D spatial field (it has a
real periodic box). Bottom row: beta_para vs R scatter (the Brazil-plot
style from vis/substrate/plot_brazil_instabilities.py), PHARE and DST
overlaid, with the firehose/mirror threshold curves.

The scatter is the fair comparison, not a second spatial map: DST's B-cells
here aren't a 2D-extended patch (they cluster in a thin filament along the
solenoid's branch-cut line, see mirror_instability_dst.py docstring), so a
DST spatial plot next to PHARE's field would visually imply a structural
correspondence that doesn't exist. What *is* comparable between the two is
how the (beta_para, R) population evolves relative to the stability
boundaries — that's what the bottom row shows.

Usage:
    PYTHONPATH=$PWD python3 vis/substrate/plot_mirror_instability_compare.py \\
        --phare mirror_instability_phare.npz --dst mirror_instability_dst.npz \\
        --save mirror_compare.png
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")

from equilibrium.substrate.mhd import firehose_curve, mirror_curve


def _phare_beta(d, label):
    bx, by, bz = d[f"{label}_bx"], d[f"{label}_by"], d[f"{label}_bz"]
    B_sq = bx**2 + by**2 + bz**2
    bhx, bhy, bhz = bx / np.sqrt(B_sq), by / np.sqrt(B_sq), bz / np.sqrt(B_sq)
    Pxx, Pxy, Pxz = d[f"{label}_Pxx"], d[f"{label}_Pxy"], d[f"{label}_Pxz"]
    Pyy, Pyz, Pzz = d[f"{label}_Pyy"], d[f"{label}_Pyz"], d[f"{label}_Pzz"]
    p_para = (
        bhx * bhx * Pxx
        + bhy * bhy * Pyy
        + bhz * bhz * Pzz
        + 2 * bhx * bhy * Pxy
        + 2 * bhx * bhz * Pxz
        + 2 * bhy * bhz * Pyz
    )
    trace = Pxx + Pyy + Pzz
    p_perp = (trace - p_para) / 2.0
    beta_para = 2.0 * p_para / B_sq
    beta_perp = 2.0 * p_perp / B_sq
    return beta_para.ravel(), beta_perp.ravel()


def _dst_beta(d, label):
    return d[f"{label}_beta_para"], d[f"{label}_beta_perp"]


def main(args):
    dp = np.load(args.phare)
    dd = np.load(args.dst)

    beta_para_ic = float(dd["beta_para_ic"])
    aniso_ic = float(dd["aniso_ic"])
    R_mirror = 1.0 + 1.0 / beta_para_ic
    R_fh = (1.0 - 1.0 / beta_para_ic) if beta_para_ic > 1.0 else float("nan")

    labels = ["ic", "early", "late"]
    vmax = args.rmax
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        f"Mirror instability: DST vs PHARE (hybrid)   "
        f"IC: beta_para={beta_para_ic:g}  R={aniso_ic:g}   "
        f"(R_mirror={R_mirror:.2f}, R_firehose={R_fh:.2f})",
        fontsize=12,
    )

    bp_fill = np.logspace(-3, 2, 400)

    for col, label in enumerate(labels):
        # --- row 0: PHARE spatial field (DST has no comparable 2D extent) ---
        ax = axes[0, col]
        bp_ph, bq_ph = _phare_beta(dp, label)
        R_ph_grid = (bq_ph / bp_ph).reshape(dp[f"{label}_density"].shape)
        t = float(dp[f"{label}_time"])
        im = ax.imshow(
            np.clip(R_ph_grid.T, 0, vmax),
            origin="lower",
            cmap="inferno",
            vmin=0,
            vmax=vmax,
            aspect="equal",
            extent=[dp["x"].min(), dp["x"].max(), dp["y"].min(), dp["y"].max()],
        )
        ax.set_title(f"PHARE field  t={t:.2f}", fontsize=10)
        ax.set_xlabel("x (d_i)")
        if col == 0:
            ax.set_ylabel("y (d_i)")
        fig.colorbar(im, ax=ax, shrink=0.85, label="R=T_perp/T_para")

        # --- row 1: beta_para vs R scatter, PHARE + DST overlaid ---
        ax = axes[1, col]
        ax.plot(
            bp_fill,
            firehose_curve(bp_fill.copy()),
            "k--",
            lw=1,
            label="firehose" if col == 0 else None,
        )
        ax.plot(
            bp_fill,
            mirror_curve(bp_fill.copy()),
            "k:",
            lw=1,
            label="mirror" if col == 0 else None,
        )

        n_sub = min(len(bp_ph), args.phare_sample)
        sub = rng.choice(len(bp_ph), n_sub, replace=False)
        ax.scatter(
            bp_ph[sub],
            bq_ph[sub] / bp_ph[sub],
            s=3,
            alpha=0.25,
            color="tab:blue",
            label="PHARE" if col == 0 else None,
        )

        bp_dst, bq_dst = _dst_beta(dd, label)
        finite = np.isfinite(bp_dst) & np.isfinite(bq_dst)
        ax.scatter(
            bp_dst[finite],
            bq_dst[finite] / bp_dst[finite],
            s=25,
            color="tab:orange",
            edgecolors="k",
            linewidths=0.3,
            label="DST" if col == 0 else None,
        )

        step = int(dd[f"{label}_step"])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-3, 1e2)
        ax.set_ylim(0.05, 20.0)
        ax.set_title(f"t={t:.2f} / step={step}", fontsize=10)
        ax.set_xlabel("beta_para")
        if col == 0:
            ax.set_ylabel("R = T_perp/T_para")
            ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"saved -> {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--phare", default="mirror_instability_phare.npz")
    parser.add_argument("--dst", default="mirror_instability_dst.npz")
    parser.add_argument(
        "--rmax",
        type=float,
        default=8.0,
        help="colour-scale clip for the PHARE field panels",
    )
    parser.add_argument(
        "--phare-sample",
        type=int,
        default=3000,
        help="random subsample of PHARE grid points for the scatter panels",
    )
    parser.add_argument("--save", default=None)
    args = parser.parse_args()
    main(args)
