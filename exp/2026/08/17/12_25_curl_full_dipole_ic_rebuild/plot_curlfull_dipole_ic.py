"""Per-level XZ-midplane plots of the new volume-filling curl-full dipole
spheromak IC (equilibrium.substrate.curl_full_dipole_ic), t=0.

Three panels per level:
  1. u (the poloidal scalar, two-lobed dipole -- unchanged from
     dipole_mode_ic) as a linear RdBu colormap.
  2. log10|u| -- same convention as plot_per_level_xz.py (that script's
     second row), useful for seeing the decay tail/boundary zeroing that
     the linear panel saturates out.
  3. The reconstructed phi vector's OUT-OF-PLANE component (vy) as a
     background colormap -- this is where the genuinely-toroidal phi_tor
     content actually lives in an XZ (y=0) meridional slice, since
     phihat=(-y,x,0)/rho reduces to the pure +/-y direction at y=0 -- PLUS
     the in-plane (vx,vz) reconstructed field streamlined on top (the
     poloidal field lines phi_pol contributes). This is the standard
     poloidal-field-lines + toroidal-color-background convention for
     visualizing a Taylor-state/spheromak meridional slice.

Reuses plot_phi_vector_per_level.py's weighted-least-squares phi->vector
reconstruction (PROJ, 3x14) unchanged.

Run:
    PYTHONPATH=$PWD python3 exp/2026/08/17/12_25_curl_full_dipole_ic_rebuild/plot_curlfull_dipole_ic.py
"""

import gc
import os
import sys
import time

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


def plot_one(cl, label, out_png, clip_r, nr, log_floor=1e-16):
    levels = np.array(cl.leaf_level)
    pos = np.array(cl.leaf_pos)
    u = np.array(cl.leaf_u)
    phi = np.array(cl.leaf_phi)
    vec = reconstruct_vec(phi)  # (N, 3): vx, vy, vz

    present = sorted(int(lvl) for lvl in np.unique(levels))
    fig, axes = plt.subplots(
        3, len(present), figsize=(6.5 * len(present), 18.0), squeeze=False
    )

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

        ax_u, ax_log, ax_tor = axes[0, col], axes[1, col], axes[2, col]
        if n_this < 4:
            ax_u.set_title(f"L{lvl}  n={n_this} (too few for interpolation)")
            ax_log.set_title(f"L{lvl}  n={n_this} (too few for interpolation)")
            ax_tor.set_title(f"L{lvl}  n={n_this} (too few for interpolation)")
            continue

        Uv = griddata((x, z), uv, (Xi, Zi), method="linear")
        Vx = griddata((x, z), vx, (Xi, Zi), method="linear")
        Vz = griddata((x, z), vz, (Xi, Zi), method="linear")
        Vy = griddata((x, z), vy, (Xi, Zi), method="linear")

        im_u = ax_u.pcolormesh(
            Xi, Zi, Uv, cmap="RdBu_r", vmin=-1.0, vmax=1.0, shading="auto"
        )
        fig.colorbar(im_u, ax=ax_u, shrink=0.8, label="u (poloidal scalar)")
        ax_u.set_aspect("equal")
        ax_u.set_title(f"L{lvl}  u -- two-lobed dipole  n={n_this}")
        ax_u.set_xlabel("x")
        ax_u.set_ylabel("z")

        Ulog = np.log10(np.clip(np.abs(Uv), log_floor, None))
        im_log = ax_log.pcolormesh(
            Xi,
            Zi,
            Ulog,
            cmap="viridis",
            vmin=np.log10(log_floor),
            vmax=0,
            shading="auto",
        )
        fig.colorbar(im_log, ax=ax_log, shrink=0.8, label="log10|u|")
        ax_log.set_aspect("equal")
        ax_log.set_title(f"L{lvl}  log10|u|")
        ax_log.set_xlabel("x")
        ax_log.set_ylabel("z")

        vy_max = np.nanmax(np.abs(Vy)) or 1.0
        im_tor = ax_tor.pcolormesh(
            Xi, Zi, Vy, cmap="PuOr", vmin=-vy_max, vmax=vy_max, shading="auto"
        )
        fig.colorbar(im_tor, ax=ax_tor, shrink=0.8, label="vy (toroidal, out-of-plane)")

        speed = np.sqrt(Vx**2 + Vz**2)
        speed_safe = np.where(np.isfinite(speed) & (speed > 0), speed, np.nan)
        lw = 0.5 + 2.5 * (speed_safe / np.nanmax(speed_safe))
        try:
            ax_tor.streamplot(
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
            ax_tor.text(
                0.5,
                0.5,
                f"streamplot failed: {e}",
                transform=ax_tor.transAxes,
                ha="center",
            )

        ax_tor.set_aspect("equal")
        ax_tor.set_title(
            f"L{lvl}  poloidal field lines (phi_pol) + toroidal vy (phi_tor)  n={n_this}"
        )
        ax_tor.set_xlabel("x")
        ax_tor.set_ylabel("z")

    fig.suptitle(
        f"{label}  XZ midplane (y=0), r<{clip_r:g}  n_leaves={cl.n_leaves}", fontsize=11
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[{label}] n_leaves={cl.n_leaves} levels={present} -> {out_png}", flush=True)


def process(label, ic_path, out_dir, clip_r, nr=220):
    t0 = time.time()
    alat = _lat.AdaptiveLattice.load(ic_path)
    cl = alat[0]
    print(
        f"[{label}] loaded {ic_path}  n_leaves={cl.n_leaves}  t={time.time()-t0:.1f}s",
        flush=True,
    )

    os.makedirs(out_dir, exist_ok=True)
    plot_one(cl, label, f"{out_dir}/xz_curlfull_dipole_{label}.png", clip_r, nr)
    print(f"[{label}] done  total_t={time.time()-t0:.1f}s", flush=True)
    del cl, alat
    gc.collect()


def main():
    exp_dir = "exp/2026/08/17/12_25_curl_full_dipole_ic_rebuild"
    process(
        "uniform_r180",
        "/bk/curlfull_dipole_uniform_r180_ic220.h5",
        f"{exp_dir}/plots",
        clip_r=150.0,
    )
    process(
        "l1_wider_margin",
        "/bk/curlfull_dipole_l1_wider_margin_ic220.h5",
        f"{exp_dir}/plots",
        clip_r=60.0,
    )


if __name__ == "__main__":
    main()
