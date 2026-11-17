"""Per-level XZ-midplane slice plot of an AdaptiveLattice checkpoint's `u`
field -- one panel per level, unlike plot_telescope_xz.py's single panel
combining every level's cells into one griddata interpolation.  Useful for a
coarse-fine (e.g. L0/L1) config where mixing levels in one interpolation
blurs the junction itself rather than showing each level's own resolution.

Reuses plot_telescope_xz.py's per-level "|y| < threshold" convention (sized
to each level's own cell size, 1 / SPLIT_RATIO**level) so every level gets a
proper single-cell-thick midplane layer, then plots each level in its own
column instead of merging them.

Run:
    PYTHONPATH=$PWD python3 run/substrate/adaptive/plot_per_level_xz.py \
        /bk/l1_wider_margin_active_stepped/step_300_final.h5 --out-dir /tmp/plots
"""

import argparse
import gc
import os
import sys

sys.path.insert(0, ".")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

from equilibrium import lattice as _lat
from run.substrate.adaptive.plot_telescope_xz import SPLIT_RATIO


def plot_cluster_per_level(cl, label, out_png, clip_r, nr, log_floor):
    levels = np.array(cl.leaf_level)
    pos = np.array(cl.leaf_pos)
    u = np.array(cl.leaf_u)

    present = sorted(int(lvl) for lvl in np.unique(levels))
    fig, axes = plt.subplots(
        2, len(present), figsize=(5.5 * len(present), 10.4), squeeze=False
    )

    xi = np.linspace(-clip_r, clip_r, nr)
    zi = np.linspace(-clip_r, clip_r, nr)
    Xi, Zi = np.meshgrid(xi, zi)
    r_xz = np.sqrt(pos[:, 0] ** 2 + pos[:, 2] ** 2)

    for col, lvl in enumerate(present):
        cell_size = 1.0 / (SPLIT_RATIO**lvl)
        thresh = cell_size * 0.6
        mask = (levels == lvl) & (np.abs(pos[:, 1]) < thresh) & (r_xz < clip_r)
        x, z, uv = pos[mask, 0], pos[mask, 2], u[mask]
        n_this = int(mask.sum())

        if n_this >= 4:
            Zp = griddata((x, z), uv, (Xi, Zi), method="linear")
            # griddata fills the CONVEX HULL of the input points -- a level whose
            # leaves only exist in an annulus (e.g. L0 outside an L1 core) has a
            # convex hull that is the full disk, so linear interpolation bridges
            # straight across the physically-empty hole where that level has NO
            # leaves at all (not zero field -- no data). Mask any grid point
            # whose nearest real data point is farther than a few cell-widths
            # away back to NaN so the hole renders as blank, not fabricated.
            tree = cKDTree(np.column_stack([x, z]))
            grid_pts = np.column_stack([Xi.ravel(), Zi.ravel()])
            dist, _ = tree.query(grid_pts, k=1)
            far = (dist > cell_size * 3.0).reshape(Xi.shape)
            Zp[far] = np.nan
        else:
            Zp = np.full_like(Xi, np.nan)

        ax_lin, ax_log = axes[0, col], axes[1, col]
        im1 = ax_lin.pcolormesh(
            Xi, Zi, Zp, cmap="RdBu_r", vmin=-1.0, vmax=1.0, shading="auto"
        )
        ax_lin.set_aspect("equal")
        ax_lin.set_title(f"L{lvl}  u (linear)  n={n_this}")
        ax_lin.set_xlabel("x")
        ax_lin.set_ylabel("z")
        fig.colorbar(im1, ax=ax_lin, shrink=0.8, label="u")

        Zlog = np.log10(np.clip(np.abs(Zp), log_floor, None))
        im2 = ax_log.pcolormesh(
            Xi,
            Zi,
            Zlog,
            cmap="viridis",
            vmin=np.log10(log_floor),
            vmax=0,
            shading="auto",
        )
        ax_log.set_aspect("equal")
        ax_log.set_title(f"L{lvl}  log10|u|")
        ax_log.set_xlabel("x")
        fig.colorbar(im2, ax=ax_log, shrink=0.8, label="log10|u|")

    fig.suptitle(
        f"{label}  per-level XZ midplane, r<{clip_r:g}  n_leaves={cl.n_leaves}",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[{label}] n_leaves={cl.n_leaves} levels={present} -> {out_png}", flush=True)


def plot_one(path, out_png, clip_r, nr, log_floor):
    alat = _lat.AdaptiveLattice.load(path)
    cl = alat[0]
    label = os.path.splitext(os.path.basename(path))[0]
    plot_cluster_per_level(cl, label, out_png, clip_r, nr, log_floor)
    del cl, alat
    gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="AdaptiveLattice .h5 checkpoint paths")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument(
        "--clip-r", type=float, default=3.0, help="plot radius (raw L0 units)"
    )
    ap.add_argument("--nr", type=int, default=260, help="grid resolution per axis")
    ap.add_argument("--log-floor", type=float, default=1e-16)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for p in args.paths:
        label = os.path.splitext(os.path.basename(p))[0]
        out_png = os.path.join(args.out_dir, f"xz_per_level_{label}.png")
        plot_one(p, out_png, args.clip_r, args.nr, args.log_floor)


if __name__ == "__main__":
    main()
