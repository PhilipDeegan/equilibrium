"""
Compare two orbit_sandbox checkpoints.

Usage:
  PYTHONPATH=$PWD python3 vis/substrate/orbit_compare.py before.h5 after.h5

Shows:
  - lam_before, lam_after, delta_lam  (equatorial xy slice)
  - printed summary: where lam grew/shrank, centroid shift
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

sys.path.insert(0, ".")
import argparse

from equilibrium import lattice as _lat


def main():
    _ap = argparse.ArgumentParser()
    _ap.add_argument("before")
    _ap.add_argument("after")
    _ap.add_argument(
        "--R-central",
        type=float,
        default=3.0,
        help="Sun knot radius in cells — scales region boundaries",
    )
    _ap.add_argument(
        "--short",
        action="store_true",
        help="one-line summary: outer_mean inner_mean core_mean",
    )
    _args = _ap.parse_args()

    f_before, f_after = _args.before, _args.after
    _Rc = _args.R_central

    # ---------------------------------------------------------------------------
    # Load
    # ---------------------------------------------------------------------------
    print(f"loading {f_before} …")
    lat_b = _lat.AdaptiveLattice.load(f_before)
    cl_b = lat_b[0]
    print(f"loading {f_after} …")
    lat_a = _lat.AdaptiveLattice.load(f_after)
    cl_a = lat_a[0]

    pos_b = np.array(cl_b.leaf_pos)
    lam_b = np.array(cl_b.leaf_lam)
    lam_a = np.array(cl_a.leaf_lam)
    u_b = np.array(cl_b.leaf_u)
    u_a = np.array(cl_a.leaf_u)
    phi_b = np.array(cl_b.leaf_phi)
    phi_a = np.array(cl_a.leaf_phi)

    assert (
        cl_b.n_leaves == cl_a.n_leaves
    ), "leaf counts differ — lattice changed between checkpoints"

    chi_b = lam_b / (1.0 + lam_b)  # χ = λ/(1+λ), used internally for centroid estimates
    chi_a = lam_a / (1.0 + lam_a)
    d_lam = lam_a - lam_b
    d_u = u_a - u_b
    d_phi = phi_a - phi_b  # (14, N)

    # ---------------------------------------------------------------------------
    # Printed summary
    # ---------------------------------------------------------------------------
    r = np.linalg.norm(pos_b, axis=1)

    print("\n=== lam delta ===")
    print(f"  max increase : {d_lam.max():.6f}  at r={r[np.argmax(d_lam)]:.2f} cells")
    print(f"  max decrease : {d_lam.min():.6f}  at r={r[np.argmin(d_lam)]:.2f} cells")
    print(f"  rms change   : {np.sqrt(np.mean(d_lam**2)):.6f}")

    # Region breakdown — thresholds scale with R_central
    _r1, _r2, _r3 = _Rc * 1.5, _Rc * 3.5, _Rc * 7.0
    regions = [
        (f"Sun core (<{_r1:.1f})", 0, _r1),
        (f"inner ({_r1:.1f}-{_r2:.1f})", _r1, _r2),
        (f"mid ({_r2:.1f}-{_r3:.1f})", _r2, _r3),
        (f"outer (>{_r3:.1f})", _r3, 9999),
    ]
    region_means = {}
    for label, rlo, rhi in regions:
        m = (r >= rlo) & (r < rhi)
        if m.any():
            mean_v = d_lam[m].mean()
            region_means[label] = mean_v
            if not _args.short:
                print(
                    f"  {label:26s}: mean Δlam={mean_v:.6f}  max|Δlam|={np.abs(d_lam[m]).max():.6f}"
                )

    if _args.short:
        vals = list(region_means.values())
        labels = [k.split("(")[0].strip() for k in region_means]
        parts = "  ".join(f"{lbl}={v:+.4f}" for lbl, v in zip(labels, vals))
        print(parts)
        sys.exit(0)

    print("\n=== u delta ===")
    print(f"  max increase : {d_u.max():.6f}  at r={r[np.argmax(d_u)]:.2f} cells")
    print(f"  max decrease : {d_u.min():.6f}  at r={r[np.argmin(d_u)]:.2f} cells")
    print(f"  rms change   : {np.sqrt(np.mean(d_u**2)):.6f}")

    print("\n=== phi delta (all faces) ===")
    d_phi_mag = np.sqrt((d_phi**2).sum(axis=0))
    print(
        f"  max |dphi|   : {d_phi_mag.max():.6f}  at r={r[np.argmax(d_phi_mag)]:.2f} cells"
    )
    print(f"  rms |dphi|   : {np.sqrt(np.mean(d_phi_mag**2)):.6f}")

    # Chi-excess centroid before/after (rough, no Sun radius arg)
    # Estimate sun background from the minimum-chi region
    # Simple 1/r fit: use cells between r=5..15 to estimate chi_central * R_central
    mid = (r > 5) & (r < 15)
    if mid.any():
        coeff = float(
            np.median(chi_b[mid] * r[mid])
        )  # chi * r ≈ chi_central * R_central
        chi_sun = coeff / np.maximum(r, 1e-9)
    else:
        chi_sun = np.zeros_like(chi_b)

    exc_b = np.maximum(chi_b - chi_sun, 0.0)
    exc_a = np.maximum(chi_a - chi_sun, 0.0)

    # Centroid in annular region r=5..25
    ann = (r > 5) & (r < 25)

    def centroid(exc, mask):
        w = exc[mask]
        wsum = w.sum()
        if wsum < 1e-20:
            return np.full(3, float("nan"))
        return (w[:, None] * pos_b[mask]).sum(axis=0) / wsum

    c_b = centroid(exc_b, ann)
    c_a = centroid(exc_a, ann)
    print("\n=== lam-excess centroid (r=5..25) ===")
    print(f"  before: ({c_b[0]:.3f}, {c_b[1]:.3f}, {c_b[2]:.3f})")
    print(f"  after : ({c_a[0]:.3f}, {c_a[1]:.3f}, {c_a[2]:.3f})")
    if np.all(np.isfinite(c_b)) and np.all(np.isfinite(c_a)):
        shift = c_a - c_b
        print(
            f"  shift : ({shift[0]:.3f}, {shift[1]:.3f}, {shift[2]:.3f})  |shift|={np.linalg.norm(shift):.3f}"
        )

    # ---------------------------------------------------------------------------
    # Plot: equatorial slice
    # ---------------------------------------------------------------------------
    Z_THICK = 3.0
    GRID_N = 300

    mask_xy = np.abs(pos_b[:, 2]) < Z_THICK
    px, py = pos_b[mask_xy, 0], pos_b[mask_xy, 1]
    tri = Delaunay(np.column_stack([px, py]))
    xi = np.linspace(px.min(), px.max(), GRID_N)
    yi = np.linspace(py.min(), py.max(), GRID_N)
    Xi, Yi = np.meshgrid(xi, yi)

    def interp(vals):
        Z = LinearNDInterpolator(tri, vals[mask_xy])(Xi, Yi)
        return np.where(np.isfinite(Z), Z, np.nan)

    Z_cb = interp(chi_b)
    Z_ca = interp(chi_a)
    Z_dc = interp(d_lam)
    Z_du = interp(d_u)

    ext = [xi.min(), xi.max(), yi.min(), yi.max()]
    fig, axes = plt.subplots(1, 4, figsize=(26, 6))
    fig.patch.set_facecolor("#0a0a1a")

    titles = [
        "lam before (χ)",
        "lam after (χ)",
        "Δlam (after−before)",
        "Δu (after−before)",
    ]
    data = [Z_cb, Z_ca, Z_dc, Z_du]
    cmaps = ["inferno", "inferno", "RdBu_r", "RdBu_r"]
    sym = [False, False, True, True]

    for ax, title, Z, cmap, symmetric in zip(axes, titles, data, cmaps, sym):
        ax.set_facecolor("#0a0a1a")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")

        if symmetric:
            vabs = (
                float(np.nanpercentile(np.abs(Z[np.isfinite(Z)]), 99))
                if np.any(np.isfinite(Z))
                else 1e-6
            )
            vmin, vmax = -vabs, vabs
        else:
            vmin = float(np.nanmin(Z)) if np.any(np.isfinite(Z)) else 0
            vmax = float(np.nanmax(Z)) if np.any(np.isfinite(Z)) else 1

        im = ax.imshow(
            Z,
            extent=ext,
            origin="lower",
            cmap=cmap,
            aspect="equal",
            interpolation="bilinear",
            vmin=vmin,
            vmax=vmax,
        )
        cb = plt.colorbar(im, ax=ax, pad=0.02)
        cb.ax.yaxis.set_tick_params(color="white")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
        ax.set_title(title, color="white")
        ax.set_xlabel("x [cells]", color="white")
        ax.set_ylabel("y [cells]", color="white")
        ax.plot(0, 0, "o", color="yellow", ms=4)

        # Centroid markers
        if np.all(np.isfinite(c_b)):
            ax.plot(c_b[0], c_b[1], "o", color="cyan", ms=6, alpha=0.7, label="before")
        if np.all(np.isfinite(c_a)):
            ax.plot(c_a[0], c_a[1], "+", color="lime", ms=10, mew=2, label="after")

    axes[0].legend(fontsize=7, facecolor="#111", labelcolor="white")

    plt.tight_layout()
    out = f_after.replace(".h5", "_compare.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nsaved → {out}")
    plt.show()


if __name__ == "__main__":
    main()
