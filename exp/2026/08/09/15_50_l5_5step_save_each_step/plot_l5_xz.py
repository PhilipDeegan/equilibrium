"""
XZ-midplane slice plots of the L5-only leaves across the 5-step checkpoint
sequence, in the style of run/substrate/adaptive/neutron_decay.py's
xz_slice/griddata cross-section panels -- just u (not phi_toroidal), just
level-5 leaves, for now.

Run:
    PYTHONPATH=$PWD python3 exp/2026/08/09/15_50_l5_5step_save_each_step/plot_l5_xz.py
"""

import sys

sys.path.insert(0, ".")
import gc
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

from equilibrium import lattice as _lat

SNAPSHOT_BASE = "/bk/l5_core_diameter9_cells_across_220.h5"
STEP_DIR = "/bk/l5_5step_save_each_step"
N_STEPS = 5
OUT_PNG = "/home/p/git/eq/exp/2026/08/09/15_50_l5_5step_save_each_step/l5_xz_slices.png"
CLIP_R = 0.85
NR = 220
LOG_FLOOR = 1e-16


def load_l5_xz(path, label):
    t0 = time.time()
    alat = _lat.AdaptiveLattice.load(path)
    cl = alat[0]
    levels = np.array(cl.leaf_level)
    pos = np.array(cl.leaf_pos)
    u = np.array(cl.leaf_u)

    l5 = levels == 5
    pos5 = pos[l5]
    u5 = u[l5]

    y = pos5[:, 1]
    y_u = np.sort(np.abs(np.unique(y)))
    ztol = float(y_u[y_u > 1e-9][0]) * 1.2 if (y_u > 1e-9).any() else 0.01
    mask = np.abs(y) < ztol
    r_xz = np.sqrt(pos5[:, 0] ** 2 + pos5[:, 2] ** 2)
    mask &= r_xz < CLIP_R

    x, z, uv = pos5[mask, 0], pos5[mask, 2], u5[mask]
    print(
        f"[{label}] load_t={time.time()-t0:.1f}s  n_L5={l5.sum()}  n_slice={x.size}",
        flush=True,
    )

    del cl, alat
    gc.collect()
    return x, z, uv


def main():
    labels_paths = [("IC", SNAPSHOT_BASE)] + [
        (f"step{i}", f"{STEP_DIR}/step_{i:02d}.h5") for i in range(1, N_STEPS + 1)
    ]

    slices = [load_l5_xz(p, lbl) for lbl, p in labels_paths]

    xi = np.linspace(-CLIP_R, CLIP_R, NR)
    zi = np.linspace(-CLIP_R, CLIP_R, NR)
    Xi, Zi = np.meshgrid(xi, zi)

    n = len(labels_paths)
    fig, axes = plt.subplots(2, n, figsize=(3.0 * n, 6.2))

    for col, ((lbl, _), (x, z, uv)) in enumerate(zip(labels_paths, slices)):
        Zp = griddata((x, z), uv, (Xi, Zi), method="linear")

        ax_lin = axes[0, col]
        im1 = ax_lin.pcolormesh(
            Xi, Zi, Zp, cmap="RdBu_r", vmin=-1.0, vmax=1.0, shading="auto"
        )
        ax_lin.set_aspect("equal")
        ax_lin.set_title(f"{lbl}  u (linear)", fontsize=9)
        if col == 0:
            ax_lin.set_ylabel("z")

        ax_log = axes[1, col]
        Zlog = np.log10(np.clip(np.abs(Zp), LOG_FLOOR, None))
        im2 = ax_log.pcolormesh(
            Xi, Zi, Zlog, cmap="viridis", vmin=-16, vmax=0, shading="auto"
        )
        ax_log.set_aspect("equal")
        ax_log.set_title(f"{lbl}  log10|u|", fontsize=9)
        ax_log.set_xlabel("x")
        if col == 0:
            ax_log.set_ylabel("z")

    fig.colorbar(im1, ax=axes[0, :].tolist(), shrink=0.8, label="u")
    fig.colorbar(im2, ax=axes[1, :].tolist(), shrink=0.8, label="log10|u|")

    fig.suptitle("L5-only XZ midplane slice: u field, IC through step5", fontsize=11)
    plt.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    print(f"plot saved -> {OUT_PNG}", flush=True)


if __name__ == "__main__":
    main()
