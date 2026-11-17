"""
Cross-section comparison across multiple decay snapshots (uniform lattice).

Geometry is reconstructed from DampedLattice(R) — ordering is deterministic
for a given R so no pos needed in the file.

Usage:
  PYTHONPATH=$PWD python3 run/substrate/probe_decay_compare.py \\
      proton_candidate_R40.npz \\
      proton_candidate_R40_decay2.npz \\
      proton_candidate_R40_decay3.npz

Optional zoom:
  ... --zoom 6
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium.substrate.lattice import DampedLattice

parser = argparse.ArgumentParser()
parser.add_argument("files", nargs="+", help=".npz snapshot files to compare")
parser.add_argument("--R", type=float, default=40.0)
parser.add_argument("--sponge", type=float, default=0.15)
parser.add_argument("--damp-max", type=float, default=0.003)
parser.add_argument("--zoom", type=float, default=None)
parser.add_argument(
    "--thickness", type=float, default=2.0, help="half-thickness of the midplane slice"
)
parser.add_argument("--gridsize", type=int, default=60)
parser.add_argument("--plot-dir", type=str, default="tmp")
parser.add_argument("--no-plot", action="store_true")


def load_snapshot(lat, path, r_all):
    data = np.load(path)
    lat.u[:] = data["u"]
    lat.phi[:] = data["phi"]
    lat.lam[:] = data["lam"]
    A_tiles = list(data["A_tiles"].astype(int))
    B_tiles = list(data["B_tiles"].astype(int)) if "B_tiles" in data else []
    ring_r = (
        float(np.mean([r_all[i] for i in A_tiles + B_tiles]))
        if (A_tiles or B_tiles)
        else 0.0
    )
    return ring_r


def main():
    args = parser.parse_args()

    lat = DampedLattice(
        radius_cells=args.R, sponge_fraction=args.sponge, damp_max=args.damp_max
    )
    r_all = np.linalg.norm(lat.pos, axis=1)
    pos = lat.pos

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    n = len(args.files)

    snapshots = []
    for path in args.files:
        ring_r = load_snapshot(lat, path, r_all)
        snapshots.append(
            {
                "label": os.path.basename(path).replace(".npz", ""),
                "ring_r": ring_r,
                "phi": lat.phi.copy(),
                "u": lat.u.copy(),
                "lam": lat.lam.copy(),
            }
        )
        print(f"Loaded {path}  ring_r≈{ring_r:.2f}", flush=True)

    if args.no_plot:
        return

    planes = [
        ("XY", 2, 0, 1, "x", "y"),
        ("XZ", 1, 0, 2, "x", "z"),
        ("YZ", 0, 1, 2, "y", "z"),
    ]

    for plane_tag, perp_ax, ix, iy, xl, yl in planes:
        fig, axes = plt.subplots(3, n, figsize=(5 * n, 14), squeeze=False)
        fig.suptitle(f"DST decay comparison — {plane_tag}", fontsize=11)

        for col, snap in enumerate(snapshots):
            phi_mag = np.sqrt((snap["phi"] ** 2).sum(axis=1))
            u = snap["u"]
            lam = snap["lam"]

            mask = np.abs(pos[:, perp_ax]) < args.thickness
            if args.zoom is not None:
                mask &= r_all < args.zoom

            px, py = pos[mask, ix], pos[mask, iy]
            domain_r = args.zoom if args.zoom is not None else args.R
            gs = max(20, int(args.gridsize * domain_r / args.R))

            axes[0, col].set_title(snap["label"], fontsize=8)

            # row 0: |phi| — max per hex so sparse ring tiles survive IC panel
            ax = axes[0, col]
            phi_vals = phi_mag[mask]
            phi_pos = phi_vals[phi_vals > 0]
            if phi_pos.size > 0:
                norm_phi = LogNorm(
                    vmin=max(float(np.percentile(phi_pos, 1)), 1e-14),
                    vmax=float(phi_pos.max()),
                )
            else:
                norm_phi = None
            hb = ax.hexbin(
                px,
                py,
                C=phi_vals,
                reduce_C_function=np.max,
                gridsize=gs,
                cmap="inferno",
                norm=norm_phi,
            )
            plt.colorbar(hb, ax=ax, shrink=0.8)
            ax.set_xlabel(xl)
            ax.set_aspect("equal")
            ax.set_ylabel(f"|phi|\n{yl}" if col == 0 else yl)

            # row 1: u — symlog so both sign and dynamic range are visible
            ax = axes[1, col]
            um = u[mask]
            vm = float(np.abs(um).max()) or 1.0
            linthresh = max(vm * 1e-3, 1e-9)
            from matplotlib.colors import SymLogNorm

            norm_u = SymLogNorm(linthresh=linthresh, vmin=-vm, vmax=vm)
            hb = ax.hexbin(
                px,
                py,
                C=um,
                reduce_C_function=np.mean,
                gridsize=gs,
                cmap="seismic",
                norm=norm_u,
            )
            plt.colorbar(hb, ax=ax, shrink=0.8)
            ax.set_xlabel(xl)
            ax.set_aspect("equal")
            ax.set_ylabel(f"u (symlog)\n{yl}" if col == 0 else yl)

            # row 2: lam — max per hex so ring tiles are visible
            ax = axes[2, col]
            hb = ax.hexbin(
                px,
                py,
                C=lam[mask],
                reduce_C_function=np.max,
                gridsize=gs,
                cmap="viridis",
            )
            plt.colorbar(hb, ax=ax, shrink=0.8)
            ax.set_xlabel(xl)
            ax.set_aspect("equal")
            ax.set_ylabel(f"lam\n{yl}" if col == 0 else yl)

            if snap["ring_r"] > 0:
                theta = np.linspace(0, 2 * np.pi, 128)
                rc = snap["ring_r"]
                for row in range(3):
                    axes[row, col].plot(
                        rc * np.cos(theta), rc * np.sin(theta), "w--", lw=0.7, alpha=0.5
                    )

        plt.tight_layout()
        zoom_tag = f"_zoom{int(args.zoom)}" if args.zoom is not None else ""
        out = f"{args.plot_dir}/decay_compare_{plane_tag}{zoom_tag}.png"
        plt.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"saved → {out}", flush=True)


if __name__ == "__main__":
    main()
