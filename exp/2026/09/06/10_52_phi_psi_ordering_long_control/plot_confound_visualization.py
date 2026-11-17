"""
Field-snapshot plot for the c2=0 confound-check control (gen_confound_snapshots.py):
same layout as plot_leak_visualization.py so the two are directly comparable
side by side -- phi_mag and circ_density at t=0 vs t=25.9, under plain
step() with the (phi,psi) coupling OFF.
"""

import sys

sys.path.insert(0, ".")
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from equilibrium.plot import savefig

OUT_DIR = "exp/2026/09/06/10_52_phi_psi_ordering_long_control"


def slice_mask(pos, r_sph):
    y = pos[:, 1]
    y_u = np.sort(np.abs(np.unique(y)))
    ytol = float(y_u[y_u > 1e-6][0]) * 1.2 if (y_u > 1e-6).any() else 1.5
    clip = r_sph * 1.15
    return (np.abs(y) < ytol) & (np.sqrt(pos[:, 0] ** 2 + pos[:, 2] ** 2) < clip), clip


def to_grid(pos, mask, vals, clip, nr=220):
    xi_g = np.linspace(-clip, clip, nr)
    zi_g = np.linspace(-clip, clip, nr)
    Xi, Zi = np.meshgrid(xi_g, zi_g)
    xv, zv, pv = pos[mask, 0], pos[mask, 2], vals[mask]
    Zp = griddata((xv, zv), pv, (Xi, Zi), method="linear")
    return Xi, Zi, Zp


def main():
    d = np.load(f"{OUT_DIR}/confound_snapshots.npz")
    pos, r_sph, t_end = d["pos"], float(d["r_sph"]), float(d["t_phys_end"])
    mask, clip = slice_mask(pos, r_sph)

    rows = [
        ("phi_mag", "|phi| (ring flux magnitude)", d["phi_mag0"], d["phi_mag1"]),
        ("circd", "local circulation density", d["circd0"], d["circd1"]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9, 8.5))
    col_titles = ["t=0 (IC)", f"t={t_end:.1f} (c2=0 control -- coupling OFF)"]

    for r, (key, label, v0, v1) in enumerate(rows):
        vmax = max(np.nanpercentile(v0[mask], 99.5), np.nanpercentile(v1[mask], 99.5), 1e-9)
        for c, vals in enumerate((v0, v1)):
            ax = axes[r, c]
            Xi, Zi, Zp = to_grid(pos, mask, vals, clip)
            im = ax.pcolormesh(Xi, Zi, Zp, cmap="inferno", vmin=0.0, vmax=vmax, shading="auto")
            ax.set_aspect("equal")
            ax.set_xlabel("x")
            ax.set_ylabel("z")
            if r == 0:
                ax.set_title(col_titles[c])
            fig.colorbar(im, ax=ax, shrink=0.85, label=label if c == 1 else None)
        axes[r, 0].set_ylabel(f"{label}\nz")

    fig.suptitle(
        "Confound check: SAME IC/domain, coupling OFF (c2=0) -- circulation stays essentially frozen\n"
        f"R=50, sph_frac=0.9, XZ meridional slice (|y|<~1 cell)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    savefig(f"{OUT_DIR}/confound_visualization.png", fig=fig, dpi=140)


if __name__ == "__main__":
    main()
