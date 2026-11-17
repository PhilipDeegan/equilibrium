"""
XZ-midplane slice plot of an AdaptiveLattice checkpoint's `u` field, across
whatever levels are present in the telescope (not restricted to one level).

One PNG per input checkpoint (not a combined multi-column figure) -- meant
to be re-run against however many checkpoints a long stepping run produces,
without needing to know that count in advance.

Level-aware slicing: a single fixed "|y| < threshold" cut (as in
run/substrate/adaptive/neutron_decay.py's xz_slice, sized for one uniform
resolution) badly under-samples coarser levels when multiple levels share
the same slice -- a coarse cell is much larger than a fine one, so few of
its centres fall inside a threshold sized for the finest level. Here each
level gets its own threshold, scaled to that level's own cell size
(1 / SPLIT_RATIO**level), so every level contributes one proper single-cell
midplane layer.

Run:
    PYTHONPATH=$PWD python3 run/substrate/adaptive/plot_telescope_xz.py \
        /bk/l5_5step_save_each_step/step_*.h5 --out-dir /bk/l5_5step_save_each_step/plots

    PYTHONPATH=$PWD python3 run/substrate/adaptive/plot_telescope_xz.py \
        /bk/some_long_run/step_*.h5 --out-dir /bk/some_long_run/plots --clip-r 10
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

from equilibrium import lattice as _lat

SPLIT_RATIO = 3.0  # Odd27 refinement ratio: cell size shrinks by this factor per level


def level_aware_xz_mask(pos, levels, clip_r):
    y = pos[:, 1]
    mask = np.zeros(len(pos), dtype=bool)
    for lvl in np.unique(levels):
        cell_size = 1.0 / (SPLIT_RATIO ** int(lvl))
        thresh = cell_size * 0.6
        mask |= (levels == lvl) & (np.abs(y) < thresh)
    r_xz = np.sqrt(pos[:, 0] ** 2 + pos[:, 2] ** 2)
    mask &= r_xz < clip_r
    return mask


def plot_cluster(cl, label, out_png, clip_r, nr, log_floor):
    """Plots directly from an in-memory cluster (no checkpoint load/save
    round-trip needed) -- lets a stepping loop plot intermediate states on
    the fly and only persist the final checkpoint to disk.
    """
    levels = np.array(cl.leaf_level)
    pos = np.array(cl.leaf_pos)
    u = np.array(cl.leaf_u)

    mask = level_aware_xz_mask(pos, levels, clip_r)
    x, z, uv = pos[mask, 0], pos[mask, 2], u[mask]
    lvl_slice = levels[mask]

    xi = np.linspace(-clip_r, clip_r, nr)
    zi = np.linspace(-clip_r, clip_r, nr)
    Xi, Zi = np.meshgrid(xi, zi)
    Zp = griddata((x, z), uv, (Xi, Zi), method="linear")

    lvl_hist = np.bincount(lvl_slice) if lvl_slice.size else np.array([])

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(11, 5.2))

    im1 = ax_lin.pcolormesh(
        Xi, Zi, Zp, cmap="RdBu_r", vmin=-1.0, vmax=1.0, shading="auto"
    )
    ax_lin.set_aspect("equal")
    ax_lin.set_title("u (linear)")
    ax_lin.set_xlabel("x")
    ax_lin.set_ylabel("z")
    fig.colorbar(im1, ax=ax_lin, shrink=0.8, label="u")

    Zlog = np.log10(np.clip(np.abs(Zp), log_floor, None))
    im2 = ax_log.pcolormesh(
        Xi, Zi, Zlog, cmap="viridis", vmin=np.log10(log_floor), vmax=0, shading="auto"
    )
    ax_log.set_aspect("equal")
    ax_log.set_title("log10|u|")
    ax_log.set_xlabel("x")
    fig.colorbar(im2, ax=ax_log, shrink=0.8, label="log10|u|")

    fig.suptitle(
        f"{label}  XZ midplane, r<{clip_r:g}  n_leaves={cl.n_leaves}  "
        f"n_slice={x.size}  levels_in_slice={lvl_hist.tolist()}",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[{label}] n_leaves={cl.n_leaves} n_slice={x.size} -> {out_png}", flush=True)


def plot_one(path, out_png, clip_r, nr, log_floor):
    alat = _lat.AdaptiveLattice.load(path)
    cl = alat[0]
    label = os.path.splitext(os.path.basename(path))[0]
    plot_cluster(cl, label, out_png, clip_r, nr, log_floor)
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
        out_png = os.path.join(args.out_dir, f"xz_{label}.png")
        plot_one(p, out_png, args.clip_r, args.nr, args.log_floor)


if __name__ == "__main__":
    main()
