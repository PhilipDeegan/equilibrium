"""
2-D slice visualisation of the H0 proton IC.

Shows chi and u fields on a thin planar cut through the lattice
after warmup, using the same proton setup as run_h0_simulation.

Usage:
    PYTHONPATH=$PWD python3 vis/substrate/dst_h0_slice.py
"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from tst.substrate.explore_flux_ic import (
    DampedLattice,
    find_ring_interior,
    find_toroidal_ring,
    init_toroidal_winding,
)

from equilibrium.substrate.lattice import L_MIN


def build_proton_ic(radius_cells=40.0, alpha=0.05, chi_target=0.9):
    lat = DampedLattice(radius_cells=radius_cells, alpha=alpha)
    ring_tiles, ring_faces = find_toroidal_ring(lat, axis="z")
    interior_idx = find_ring_interior(lat, ring_tiles)
    lat.set_knot(lat.tiles[interior_idx], chi_target, 1.0, profile="1/r")
    lat.ell[interior_idx] = L_MIN
    lat.fixed_u[interior_idx] = True
    lat.u[interior_idx] = 1.0
    init_toroidal_winding(lat, ring_tiles, ring_faces, amplitude=1.0, frozen=False)
    centre_pos = lat.pos[interior_idx].copy()
    return lat, ring_tiles, interior_idx, centre_pos


def run_warmup(lat, warmup_steps=10000, n_threads=6):
    dt = lat.stable_dt()
    for _ in range(warmup_steps):
        lat.step(dt, n_threads)


def slice_lattice(lat, normal="z", thickness=0.6):
    """Return mask and in-plane coordinates for a planar slice."""
    pos = lat.pos
    if normal == "z":
        mask = np.abs(pos[:, 2]) < thickness
        h, v, hl, vl = pos[mask, 0], pos[mask, 1], "x", "y"
    elif normal == "y":
        mask = np.abs(pos[:, 1]) < thickness
        h, v, hl, vl = pos[mask, 0], pos[mask, 2], "x", "z"
    else:  # x-normal
        mask = np.abs(pos[:, 0]) < thickness
        h, v, hl, vl = pos[mask, 1], pos[mask, 2], "y", "z"
    return mask, h, v, hl, vl


def visualize_slice(
    radius_cells=40.0,
    alpha=0.05,
    warmup_steps=10000,
    n_threads=6,
    normals=("z", "y"),
    save=True,
):
    print(f"Building lattice r={radius_cells}, alpha={alpha}...")
    lat, ring_tiles, interior_idx, centre_pos = build_proton_ic(radius_cells, alpha)
    print(f"  N={lat.N}  Running {warmup_steps} warmup steps...")
    run_warmup(lat, warmup_steps, n_threads)
    print("  Done.")

    chi = lat.chi  # 1 - ell; high near knot
    u = lat.u

    ring_pos = lat.pos[np.array(ring_tiles)]

    for normal in normals:
        mask, h, v, hl, vl = slice_lattice(lat, normal=normal)
        chi_s = chi[mask]
        u_s = u[mask]

        # ring tile positions in this slice
        if normal == "z":
            ring_h = ring_pos[:, 0]
            ring_v = ring_pos[:, 1]
            ring_mask = np.abs(ring_pos[:, 2]) < 0.6
        elif normal == "y":
            ring_h = ring_pos[:, 0]
            ring_v = ring_pos[:, 2]
            ring_mask = np.abs(ring_pos[:, 1]) < 0.6
        else:
            ring_h = ring_pos[:, 1]
            ring_v = ring_pos[:, 2]
            ring_mask = np.abs(ring_pos[:, 0]) < 0.6

        fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
        fig.suptitle(
            f"H0 proton IC — {normal}-normal slice  "
            f"(r={radius_cells}, α={alpha}, {warmup_steps} steps, N={lat.N})",
            fontsize=12,
        )

        # --- chi field ---
        ax = axes[0]
        sc = ax.scatter(
            h, v, c=chi_s, cmap="hot", s=3, vmin=0.0, vmax=float(chi_s.max())
        )
        plt.colorbar(sc, ax=ax, label="χ  (1 − ℓ)", fraction=0.046)
        if ring_mask.any():
            ax.scatter(
                ring_h[ring_mask],
                ring_v[ring_mask],
                c="cyan",
                s=30,
                marker="D",
                zorder=5,
                label="ring tiles",
            )
            ax.legend(fontsize=8)
        ax.set_xlabel(hl)
        ax.set_ylabel(vl)
        ax.set_title("chi field — cell compression")
        ax.set_aspect("equal")

        # --- u field (diverging: moat is negative) ---
        ax = axes[1]
        u_abs = float(np.abs(u_s).max()) or 1e-9
        norm = mcolors.TwoSlopeNorm(vmin=-u_abs, vcenter=0.0, vmax=u_abs)
        sc = ax.scatter(h, v, c=u_s, cmap="RdBu_r", s=3, norm=norm)
        plt.colorbar(sc, ax=ax, label="u  (essence)", fraction=0.046)
        if ring_mask.any():
            ax.scatter(
                ring_h[ring_mask],
                ring_v[ring_mask],
                c="lime",
                s=30,
                marker="D",
                zorder=5,
                label="ring tiles",
            )
            ax.legend(fontsize=8)
        ax.set_xlabel(hl)
        ax.set_ylabel(vl)
        ax.set_title("u-field — Coulomb + moat structure")
        ax.set_aspect("equal")

        plt.tight_layout()

        if save:
            fname = f"vis/substrate/dst_h0_slice_{normal}.png"
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            print(f"  Saved {fname}")
        plt.show()


if __name__ == "__main__":
    visualize_slice(
        radius_cells=40.0,
        alpha=0.05,
        warmup_steps=10000,
        n_threads=6,
        normals=("z", "y"),
        save=True,
    )
