"""Per-level XZ-midplane vector-field reconstruction from leaf_phi (the
14-face flux array, this IC's B-field analogue -- see equilibrium.substrate.
dipole_mode_ic) and streamplot it, to check for flux-tube-like field-line
topology (poloidal loops linking the two u lobes) that a scalar u plot alone
cannot show.

Reconstruction: for each leaf, phi[k] ~= face_norm[k].v is an overdetermined
system (14 equations, 3 unknowns) -- solved as a face_weight-WEIGHTED least
squares fit (face_weight: hex=3, sq=1, matching bcc::face_weight, the same
weights the actual flux formula uses), NOT a naive unweighted fit:

    v = (N^T W N)^-1 N^T W phi = PROJ @ phi,  PROJ precomputed once (3,14)

N^T W N = 10*I exactly for this face geometry (isotropic BCC), so PROJ has a
clean closed form. This is a diagnostic reconstruction (best-fit vector
consistent with the 14 face fluxes), not a claim that DST's phi is fundamentally
a continuum vector field.

Run:
    PYTHONPATH=$PWD python3 exp/.../plot_phi_vector_per_level.py \
        /bk/l1_wider_margin_active_ic220_pp1fix.h5 \
        /bk/l1_wider_margin_active_pp1fix_stepped/step_100_final.h5 \
        /bk/l1_wider_margin_active_pp1fix_stepped/step_300_final.h5 \
        --out-dir plots/L1_R60_phi_vector --clip-r 60
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
from run.substrate.adaptive.plot_telescope_xz import SPLIT_RATIO
from tst.substrate.adaptive.adaptive_flux_tube import _FACE_NORM

_W = np.array([3.0] * 8 + [1.0] * 6)
_M = _FACE_NORM.T @ (_W[:, None] * _FACE_NORM)
_PROJ = np.linalg.inv(_M) @ (_FACE_NORM.T * _W[None, :])  # (3, 14)


def reconstruct_vec(phi):
    """phi: (14, N) leaf_phi array -> (N, 3) best-fit vector per leaf."""
    return (_PROJ @ phi).T


def plot_phi_vector_per_level(cl, label, out_png, clip_r, nr, u_floor=1e-16):
    levels = np.array(cl.leaf_level)
    pos = np.array(cl.leaf_pos)
    u = np.array(cl.leaf_u)
    phi = np.array(cl.leaf_phi)
    vec = reconstruct_vec(phi)  # (N, 3): vx, vy, vz

    present = sorted(int(lvl) for lvl in np.unique(levels))
    fig, axes = plt.subplots(
        1, len(present), figsize=(6.5 * len(present), 6.2), squeeze=False
    )
    axes = axes[0]

    xi = np.linspace(-clip_r, clip_r, nr)
    zi = np.linspace(-clip_r, clip_r, nr)
    Xi, Zi = np.meshgrid(xi, zi)
    r_xz = np.sqrt(pos[:, 0] ** 2 + pos[:, 2] ** 2)

    for col, lvl in enumerate(present):
        cell_size = 1.0 / (SPLIT_RATIO**lvl)
        thresh = cell_size * 0.6
        mask = (levels == lvl) & (np.abs(pos[:, 1]) < thresh) & (r_xz < clip_r)
        x, z = pos[mask, 0], pos[mask, 2]
        vx, vy, vz = vec[mask, 0], vec[mask, 1], vec[mask, 2]
        uv = u[mask]
        n_this = int(mask.sum())

        ax = axes[col]
        if n_this < 4:
            ax.set_title(f"L{lvl}  n={n_this} (too few for interpolation)")
            continue

        Uv = griddata((x, z), uv, (Xi, Zi), method="linear")
        Vx = griddata((x, z), vx, (Xi, Zi), method="linear")
        Vz = griddata((x, z), vz, (Xi, Zi), method="linear")
        Vy = griddata((x, z), vy, (Xi, Zi), method="linear")

        im = ax.pcolormesh(
            Xi, Zi, Uv, cmap="RdBu_r", vmin=-1.0, vmax=1.0, shading="auto"
        )
        fig.colorbar(im, ax=ax, shrink=0.8, label="u")

        speed = np.sqrt(Vx**2 + Vz**2)
        speed_safe = np.where(np.isfinite(speed) & (speed > 0), speed, np.nan)
        lw = 0.5 + 2.5 * (speed_safe / np.nanmax(speed_safe))
        try:
            ax.streamplot(
                Xi,
                Zi,
                np.nan_to_num(Vx),
                np.nan_to_num(Vz),
                color="k",
                density=1.4,
                linewidth=np.nan_to_num(lw, nan=0.5),
                arrowsize=1.0,
            )
        except (
            Exception
        ) as e:  # noqa: BLE001 -- diagnostic plot, don't fail the whole run on a bad slice
            ax.text(
                0.5, 0.5, f"streamplot failed: {e}", transform=ax.transAxes, ha="center"
            )

        ax.set_aspect("equal")
        ax.set_title(
            f"L{lvl}  u (bg) + in-plane (vx,vz) field lines from phi  n={n_this}"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("z")

    fig.suptitle(
        f"{label}  phi-reconstructed vector field, XZ midplane, r<{clip_r:g}  n_leaves={cl.n_leaves}",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[{label}] n_leaves={cl.n_leaves} levels={present} -> {out_png}", flush=True)


def plot_one(path, out_png, clip_r, nr):
    alat = _lat.AdaptiveLattice.load(path)
    cl = alat[0]
    label = os.path.splitext(os.path.basename(path))[0]
    plot_phi_vector_per_level(cl, label, out_png, clip_r, nr)
    del cl, alat
    gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="AdaptiveLattice .h5 checkpoint paths")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--clip-r", type=float, default=60.0)
    ap.add_argument("--nr", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for p in args.paths:
        label = os.path.splitext(os.path.basename(p))[0]
        out_png = os.path.join(args.out_dir, f"xz_phi_vec_{label}.png")
        plot_one(p, out_png, args.clip_r, args.nr)


if __name__ == "__main__":
    main()
