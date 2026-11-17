"""
Per-level XZ midplane panels -- one subplot per level, EACH with its own
log10|u| color scale (not a shared scale across levels), and raw leaf
scatter (no griddata interpolation) so texture is never an interpolation
artifact.

Why: the combined multi-level plot (plot_telescope_xz.py) shares one color
scale across ~16 orders of magnitude of |u|, so coarser levels (L0-L3,
which have real but much smaller-amplitude structure) look empty even
when they aren't. This tool zooms each level to its own occupied extent
and its own amplitude range so the shape/texture at every level is
visible on equal footing.

Run:
    PYTHONPATH=$PWD python3 exp/2026/08/10/12_11_l5_1step_eta0/plot_per_level.py \
        /bk/l5_5step_save_each_step/step_01.h5 --out exp/2026/08/10/12_11_l5_1step_eta0/plots/per_level_step01.png
"""

import argparse
import sys

sys.path.insert(0, ".")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from equilibrium import lattice as _lat

SPLIT_RATIO = 3.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--log-floor", type=float, default=1e-16)
    args = ap.parse_args()

    alat = _lat.AdaptiveLattice.load(args.path)
    cl = alat[0]
    levels = np.array(cl.leaf_level)
    pos = np.array(cl.leaf_pos)
    u = np.array(cl.leaf_u)
    label = args.path

    present = sorted(int(l) for l in np.unique(levels))
    n = len(present)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.3 * ncols, 4.3 * nrows), squeeze=False
    )

    for i, lvl in enumerate(present):
        ax = axes[i // ncols][i % ncols]
        cell_size = 1.0 / (SPLIT_RATIO**lvl)
        thresh = cell_size * 0.6
        m = (levels == lvl) & (np.abs(pos[:, 1]) < thresh)
        n_slice = int(m.sum())
        if n_slice == 0:
            ax.set_title(f"L{lvl}: empty slice")
            ax.axis("off")
            continue

        x, z, uv = pos[m, 0], pos[m, 2], u[m]
        au = np.abs(uv)
        nz = au[au > 0]
        if nz.size == 0:
            ax.set_title(f"L{lvl}: all exactly zero (n={n_slice})")
            ax.axis("off")
            continue

        vmax = float(np.log10(au.max()))
        vmin = float(np.log10(max(nz.min(), args.log_floor)))
        logv = np.log10(np.clip(au, args.log_floor, None))
        r_max = float(np.sqrt(x**2 + z**2).max())

        sc = ax.scatter(
            x, z, c=logv, cmap="viridis", vmin=vmin, vmax=vmax, s=3, marker="s"
        )
        ax.set_aspect("equal")
        ax.set_title(
            f"L{lvl}  n_slice={n_slice}  r<={r_max:.3f}\n"
            f"log10|u| in [{vmin:.2f},{vmax:.2f}]  cell={cell_size:.4f}",
            fontsize=8,
        )
        fig.colorbar(sc, ax=ax, shrink=0.8)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(f"{label}  per-level XZ midplane (own scale per level)", fontsize=10)
    plt.tight_layout()
    plt.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
