"""
XZ meridional cross-section plots of |S| (local energy-flux magnitude) at
several snapshot times, from probe_poynting_fine_and_snapshots.py's saved
output -- visualizes the wavefront expanding outward and (at late times)
the still-oscillating, non-settled field found in the time-evolution
check.

Reads the probe's per-snapshot-time `<out_prefix>_smag/t_<time>.npy` files
(one full-domain |S| array each) rather than a single combined archive, so
plotting works against a still-running or partially-completed sweep too --
see probe_poynting_fine_and_snapshots.py's `save_smag_snapshot` (perf fix,
2026-09-09: rendering used to happen inline in the physics loop, blocking
it; now the probe only writes raw arrays and this script does the plotting
offline).

Works against a still-running job -- just plots whatever snapshots exist yet.

Usage:
    python3 exp/2026/09/08/12_15_poynting_time_evolution/plot_poynting_cross_sections.py [--prefix PREFIX]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_PREFIX = "exp/2026/09/08/12_15_poynting_time_evolution/fine_R100_octagon"
SLAB_HALF_THICKNESS = 1.5  # |y| < this, cell units


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prefix", type=str, default=DEFAULT_PREFIX, help="probe script's --out-prefix value")
    args = parser.parse_args()
    PREFIX = args.prefix

    pos = np.load(f"{PREFIX}_pos.npy")
    smag_dir = Path(f"{PREFIX}_smag")
    snap_files = sorted(smag_dir.glob("t_*.npy"))
    times = [float(p.stem[2:]) for p in snap_files]

    slab = np.abs(pos[:, 1]) < SLAB_HALF_THICKNESS
    x, z = pos[slab, 0], pos[slab, 2]

    n = len(times)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows), squeeze=False)

    # PER-PANEL color scale (each snapshot's own 1st-99th percentile) -- a
    # shared/global scale is swamped by the early snapshot's huge dynamic
    # range (still-expanding wavefront: center vs untouched vacuum can span
    # ~60+ orders of magnitude), which washes out the much subtler
    # (~2-3 orders of magnitude) late-time structure into flat color.
    for i, (t, snap_file) in enumerate(zip(times, snap_files)):
        ax = axes[i // ncols][i % ncols]
        Smag = np.load(snap_file)[slab]
        finite_pos = Smag[np.isfinite(Smag) & (Smag > 0)]
        vmin = np.log10(np.percentile(finite_pos, 1)) if finite_pos.size else -10
        vmax = np.log10(np.percentile(finite_pos, 99)) if finite_pos.size else 0
        c = np.log10(np.clip(Smag, 10.0 ** vmin, None))
        sc = ax.scatter(x, z, c=c, s=3, cmap="inferno", vmin=vmin, vmax=vmax, linewidths=0)
        ax.set_title(f"t={t:.2f}  (log10|S| in [{vmin:.1f},{vmax:.1f}])", fontsize=9)
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].axis("off")

    fig.suptitle("Poynting-ring |S| (XZ slice, |y|<{:.1f}), per-panel log10 scale".format(SLAB_HALF_THICKNESS))
    fig.tight_layout()

    out_path = f"{PREFIX}_cross_sections.png"
    fig.savefig(out_path, dpi=130)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
