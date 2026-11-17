"""Quick cross-section visualisation of a saved AdaptiveLattice HDF5 snapshot."""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")
from equilibrium import lattice as _lat

gridsize = 250


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", help="HDF5 snapshot path")
    parser.add_argument(
        "--thickness", type=float, default=2.0, help="slice half-thickness"
    )
    parser.add_argument("--out", default="snapshot_slices.png")
    args = parser.parse_args()

    print(f"Loading {args.snapshot} ...", flush=True)
    alat = _lat.AdaptiveLattice.load(args.snapshot)
    cl = alat[0]
    print(f"  N={cl.n_leaves}  max_level={cl.info.max_level}", flush=True)

    pos = np.array(cl.leaf_pos)  # (N, 3)
    phi = np.array(cl.leaf_phi).T  # (N, 14)
    u = np.array(cl.leaf_u)  # (N,)
    lam = np.array(cl.leaf_lam)  # (N,)

    phi_mag = np.sqrt((phi**2).sum(axis=1))  # (N,)

    t = args.thickness

    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    fig.suptitle(f"Snapshot: {args.snapshot}", fontsize=12)

    slices = [
        ("XY (z≈0)", (np.abs(pos[:, 2]) < t), 0, 1, "x", "y"),
        ("XZ (y≈0)", (np.abs(pos[:, 1]) < t), 0, 2, "x", "z"),
        ("YZ (x≈0)", (np.abs(pos[:, 0]) < t), 1, 2, "y", "z"),
    ]

    for col, (title, mask, ix, iy, xl, yl) in enumerate(slices):
        px, py = pos[mask, ix], pos[mask, iy]
        pm = phi_mag[mask]
        um = u[mask]
        lm = lam[mask]

        # phi magnitude
        ax = axes[0, col]
        hb = ax.hexbin(
            px, py, C=pm, reduce_C_function=np.mean, gridsize=gridsize, cmap="inferno"
        )
        plt.colorbar(hb, ax=ax)
        ax.set_title(f"{title} — |phi|")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_aspect("equal")

        # u field
        ax = axes[1, col]
        vm = max(abs(um.min()), abs(um.max())) or 1.0
        hb = ax.hexbin(
            px,
            py,
            C=um,
            reduce_C_function=np.mean,
            gridsize=gridsize,
            cmap="seismic",
            vmin=-vm,
            vmax=vm,
        )
        plt.colorbar(hb, ax=ax)
        ax.set_title(f"{title} — u")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_aspect("equal")

        # lam field
        ax = axes[2, col]
        hb = ax.hexbin(
            px, py, C=lm, reduce_C_function=np.mean, gridsize=gridsize, cmap="viridis"
        )
        plt.colorbar(hb, ax=ax)
        ax.set_title(f"{title} — lam")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"Saved → {args.out}", flush=True)


if __name__ == "__main__":
    main()
