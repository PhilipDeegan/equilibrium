"""
Cross-section comparison across multiple adaptive-lattice decay snapshots.

Positions are read directly from the snapshot (saved as 'pos' by the adaptive
tests) — no DampedLattice reconstruction needed.

Usage:
  PYTHONPATH=$PWD python3 run/substrate/adaptive/probe_decay_compare.py \\
      proton_candidate_A_R40_IC.npz \\
      proton_candidate_A_R40_decay1.npz \\
      proton_candidate_A_R40_decay2.npz

Optional zoom:
  ... --zoom 6
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from tst.substrate.adaptive.adaptive_flux_tube import _FACE_NORM


def _toroidal_phi(pos, phi_N14):
    """Signed azimuthal (z-axis toroidal) phi component per cell.

    pos     : (N, 3)
    phi_N14 : (N, 14)   — snap["phi"] format
    returns   (N,)
    """
    phi_14N = phi_N14.T  # (14, N)
    rho = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
    rho_s = np.where(rho > 1e-10, rho, 1.0)
    hx = -pos[:, 1] / rho_s  # φ̂_x = -y/ρ
    hy = pos[:, 0] / rho_s  # φ̂_y =  x/ρ
    proj = (
        _FACE_NORM[:, 0:1] * hx[None, :] + _FACE_NORM[:, 1:2] * hy[None, :]
    )  # (14, N)
    return np.sum(phi_14N * proj, axis=0)  # (N,)


def _poloidal_phi(pos, phi_N14, ring_r):
    """Signed poloidal phi component per cell — tangent to the meridional
    circle of radius sqrt((rho-ring_r)^2 + z^2) centred on the ring's own
    tube centreline at (rho, z) = (ring_r, 0).

    Unlike the toroidal direction (globally defined from the z-axis alone),
    this needs the tube's own minor-radius reference (ring_r) to know which
    circle it's tangent to — well-defined near the ring, but singular right
    through the tube's own centreline (rho=ring_r, z=0), same edge-case
    pattern as the rho=0 singularity in _toroidal_phi.

    pos     : (N, 3)
    phi_N14 : (N, 14)   — snap["phi"] format
    ring_r  : float — tube (minor-radius) reference, from snap["ring_r"]
    returns   (N,)
    """
    phi_14N = phi_N14.T  # (14, N)
    rho = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
    rho_s = np.where(rho > 1e-10, rho, 1.0)
    rho_hat_x = pos[:, 0] / rho_s
    rho_hat_y = pos[:, 1] / rho_s
    theta_pol = np.arctan2(pos[:, 2], rho - ring_r)
    s, c = np.sin(theta_pol), np.cos(theta_pol)
    hx = -s * rho_hat_x
    hy = -s * rho_hat_y
    hz = c
    proj = (
        _FACE_NORM[:, 0:1] * hx[None, :]
        + _FACE_NORM[:, 1:2] * hy[None, :]
        + _FACE_NORM[:, 2:3] * hz[None, :]
    )  # (14, N)
    return np.sum(phi_14N * proj, axis=0)  # (N,)


parser = argparse.ArgumentParser()
parser.add_argument("files", nargs="+", help=".npz snapshot files to compare")
parser.add_argument("--zoom", type=float, default=None)
parser.add_argument(
    "--thickness", type=float, default=2.0, help="half-thickness of the midplane slice"
)
parser.add_argument("--gridsize", type=int, default=60)
parser.add_argument("--plot-dir", type=str, default="tmp")
parser.add_argument("--no-plot", action="store_true")


def load_snapshot(path):
    data = np.load(path)
    pos = data["pos"]
    r_all = np.linalg.norm(pos, axis=1)
    A_tiles = list(data["A_tiles"].astype(int))
    B_tiles = list(data["B_tiles"].astype(int)) if "B_tiles" in data else []
    ring_r = (
        float(np.mean([r_all[i] for i in A_tiles + B_tiles]))
        if (A_tiles or B_tiles)
        else 0.0
    )
    return {
        "label": os.path.basename(path).replace(".npz", ""),
        "phi": data[
            "phi"
        ].T,  # saved as (14, N) by --save-snapshot-dir; transpose to (N, 14)
        "lam": data["lam"],
        "u": data["u"],
        "pos": pos,
        "ring_r": ring_r,
    }


def plot_snapshots(
    snapshots, *, zoom=None, thickness=2.0, gridsize=60, plot_dir="tmp", tag=""
):
    """
    Plot |phi|, phi_tor, phi_pol, u (symlog and linear) cross-sections for a
    list of in-memory snapshots.

    Each snapshot is a dict with keys:
        phi    : (N, 14) float
        u      : (N,)   float
        lam    : (N,)   float  (loaded but no longer plotted)
        pos    : (N, 3) float
        label  : str
        ring_r : float  (0 = no ring circle drawn; also used as the tube
                          reference for phi_pol — see _poloidal_phi)

    tag    : appended to output filenames to distinguish cadence calls.
    """
    import matplotlib

    matplotlib.use("Agg")
    import os

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, SymLogNorm

    os.makedirs(plot_dir, exist_ok=True)

    n = len(snapshots)
    R_ref = float(np.linalg.norm(snapshots[0]["pos"], axis=1).max())
    tag_s = f"_{tag}" if tag else ""

    planes = [
        ("XY", 2, 0, 1, "x", "y"),
        ("XZ", 1, 0, 2, "x", "z"),
        ("YZ", 0, 1, 2, "y", "z"),
    ]

    # 5 field rows (|phi|, phi_tor, phi_pol, u symlog, u linear) stacked in a
    # single column reads as a very tall, narrow strip when there's only one
    # snapshot (the periodic per-step field-plot case) — fold that case into
    # a 2x3 grid instead (row 0: the three phi views; row 1: the two u views,
    # with one unused cell hidden below).
    single = n == 1
    n_rows, n_cols = (2, 3) if single else (5, n)

    def ax_for(axes, quantity_row, col):
        if single:
            return axes[quantity_row // n_cols, quantity_row % n_cols]
        return axes[quantity_row, col]

    for plane_tag, perp_ax, ix, iy, xl, yl in planes:
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows), squeeze=False
        )
        fig.suptitle(f"DST decay comparison (adaptive) — {plane_tag}", fontsize=11)

        for col, snap in enumerate(snapshots):
            pos = snap["pos"]
            r_all = np.linalg.norm(pos, axis=1)
            phi_mag = np.sqrt((snap["phi"] ** 2).sum(axis=1))
            phi_tor = _toroidal_phi(pos, snap["phi"])  # signed, (N,)
            phi_pol = _poloidal_phi(pos, snap["phi"], snap["ring_r"])  # signed, (N,)
            u = snap["u"]

            mask = np.abs(pos[:, perp_ax]) < thickness
            if zoom is not None:
                mask &= r_all < zoom

            px, py = pos[mask, ix], pos[mask, iy]
            domain_r = zoom if zoom is not None else R_ref
            gs = max(20, int(gridsize * domain_r / R_ref))

            ax_for(axes, 0, col).set_title(snap["label"], fontsize=8)

            # row 0: |phi|
            ax = ax_for(axes, 0, col)
            phi_vals = phi_mag[mask]
            phi_pos = phi_vals[phi_vals > 0]
            norm_phi = (
                LogNorm(
                    vmin=max(float(np.percentile(phi_pos, 1)), 1e-14),
                    vmax=float(phi_pos.max()),
                )
                if phi_pos.size > 0
                else None
            )
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

            # row 1: signed toroidal phi — reveals two-lobe structure in XZ/YZ
            ax = ax_for(axes, 1, col)
            pt = phi_tor[mask]
            vt = float(np.abs(pt).max()) or 1.0
            lt = max(vt * 1e-3, 1e-9)
            norm_t = SymLogNorm(linthresh=lt, vmin=-vt, vmax=vt)
            hb = ax.hexbin(
                px,
                py,
                C=pt,
                reduce_C_function=np.mean,
                gridsize=gs,
                cmap="RdBu_r",
                norm=norm_t,
            )
            plt.colorbar(hb, ax=ax, shrink=0.8)
            ax.set_xlabel(xl)
            ax.set_aspect("equal")
            ax.set_ylabel(f"phi_tor\n{yl}" if col == 0 else yl)

            # row 2: signed poloidal phi — the linked ring's other half;
            # tangent to the tube's own meridional circle (needs ring_r, so
            # ring_r=0 draws it centred on the origin instead — still valid,
            # just not meaningful without a real ring).
            ax = ax_for(axes, 2, col)
            pp = phi_pol[mask]
            vp = float(np.abs(pp).max()) or 1.0
            lp = max(vp * 1e-3, 1e-9)
            norm_p = SymLogNorm(linthresh=lp, vmin=-vp, vmax=vp)
            hb = ax.hexbin(
                px,
                py,
                C=pp,
                reduce_C_function=np.mean,
                gridsize=gs,
                cmap="RdBu_r",
                norm=norm_p,
            )
            plt.colorbar(hb, ax=ax, shrink=0.8)
            ax.set_xlabel(xl)
            ax.set_aspect("equal")
            ax.set_ylabel(f"phi_pol\n{yl}" if col == 0 else yl)

            # row 3: u (symlog) — vmax/linthresh from a percentile of |u|, not
            # the true max: with AMR, the finest leaves at the core resolve a
            # peak that a raw max lets dominate, pushing linthresh up with it
            # and collapsing every coarser cell into the near-zero linear
            # band (only the centre shows contrast). A percentile bound keeps
            # a few outlier fine cells from swamping the whole scale.
            ax = ax_for(axes, 3, col)
            um = u[mask]
            um_abs = np.abs(um)
            nz = um_abs[um_abs > 0]
            vm = max(float(np.percentile(nz, 99)), 1e-9) if nz.size else 1.0
            linthresh = max(
                float(np.percentile(nz, 50)) if nz.size else 0.0, vm * 1e-3, 1e-9
            )
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

            # row 4: u (linear) — same robust bound as row 3 but without log
            # compression, so the bulk magnitude structure (which the symlog
            # row compresses near zero) is visible directly. Replaces the lam
            # plot, which stopped being informative once AMR made lam depend
            # on refinement level rather than dynamics.
            ax = ax_for(axes, 4, col)
            hb = ax.hexbin(
                px,
                py,
                C=um,
                reduce_C_function=np.mean,
                gridsize=gs,
                cmap="seismic",
                vmin=-vm,
                vmax=vm,
            )
            plt.colorbar(hb, ax=ax, shrink=0.8)
            ax.set_xlabel(xl)
            ax.set_aspect("equal")
            ax.set_ylabel(f"u (linear)\n{yl}" if col == 0 else yl)

            if snap["ring_r"] > 0:
                theta = np.linspace(0, 2 * np.pi, 128)
                rc = snap["ring_r"]
                for row in range(5):
                    ax_for(axes, row, col).plot(
                        rc * np.cos(theta), rc * np.sin(theta), "w--", lw=0.7, alpha=0.5
                    )

        if single:
            axes[1, 2].axis(
                "off"
            )  # 5 quantities into a 2x3 grid leaves one cell unused

        plt.tight_layout()
        zoom_tag = f"_zoom{int(zoom)}" if zoom is not None else ""
        out = f"{plot_dir}/decay_compare_{plane_tag}{zoom_tag}{tag_s}.png"
        plt.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"saved → {out}", flush=True)


def main():
    args = parser.parse_args()

    snapshots = []
    for path in args.files:
        snap = load_snapshot(path)
        snapshots.append(snap)
        print(
            f"Loaded {path}  N={snap['pos'].shape[0]}  ring_r≈{snap['ring_r']:.2f}",
            flush=True,
        )

    if args.no_plot:
        return

    plot_snapshots(
        snapshots,
        zoom=args.zoom,
        thickness=args.thickness,
        gridsize=args.gridsize,
        plot_dir=args.plot_dir,
    )


if __name__ == "__main__":
    main()
