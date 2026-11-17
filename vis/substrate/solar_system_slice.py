"""
Solar system chi terrain visualiser.

Loads an AdaptiveLattice checkpoint and renders:
  1. Chi terrain — equatorial (z≈0) slice, chi = 1-ell, terrain-style heatmap
  2. Chi gradient vectors — direction of steepest descent (gravity direction)
  3. u field overlay — planet blob positions
  4. Centre-of-mass map — lam-weighted centroid per Voronoi region around each body

Usage:
  PYTHONPATH=$PWD python3 vis/substrate/solar_system_slice.py checkpoint.h5 \\
      --bodies Sun,Mercury,Venus,Earth,Mars,Jupiter \\
      --out solar_system.png

  python3 vis/substrate/solar_system_slice.py /tmp/sol_amr.h5 \
      --body-orbits 9.7,18.1,25.0,38.0,130.0 \
      --out sol_terrain.png

"""

import argparse
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

sys.path.insert(0, ".")

from equilibrium import lattice as _lat


def _auto_detect_orbits(r_centres, u_rmax, n_bodies):
    """Return up to n_bodies peak radii from u_rmax, sorted by r, excluding r≈0."""
    _kb = max(3, len(r_centres) // 7)
    _bg = uniform_filter1d(np.nan_to_num(u_rmax), size=_kb)
    _exc = np.nan_to_num(u_rmax - _bg)
    _thresh = max(float(_exc.max()) * 0.05, 1e-12)
    _peaks, _ = find_peaks(_exc, height=_thresh, distance=max(2, len(r_centres) // 60))
    _peaks = _peaks[r_centres[_peaks] > r_centres[2]]
    _peaks = _peaks[np.argsort(r_centres[_peaks])]
    return [float(r_centres[i]) for i in _peaks[:n_bodies]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument(
        "--bodies", type=str, default="Sun,Mercury,Venus,Earth,Mars,Jupiter"
    )
    parser.add_argument(
        "--body-orbits",
        type=str,
        default="",
        help="comma-separated orbit radii in cells (order matches --bodies minus Sun)",
    )
    parser.add_argument(
        "--z-thick", type=float, default=3.0, help="half-thickness of equatorial slice"
    )
    parser.add_argument(
        "--grid-n", type=int, default=400, help="interpolation grid resolution"
    )
    parser.add_argument(
        "--grad-stride", type=int, default=20, help="gradient arrow stride on grid"
    )
    parser.add_argument(
        "--out", type=str, default="", help="output file (empty = display)"
    )
    parser.add_argument(
        "--log-chi", action="store_true", default=True, help="log-scale chi terrain"
    )

    parser.add_argument(
        "--min-level",
        type=int,
        default=0,
        help="only include cells at this AMR level or finer (default 0 = all cells; "
        "use 1 to drop coarse background and reveal planet blobs)",
    )
    args = parser.parse_args()

    bodies = [b.strip() for b in args.bodies.split(",")]

    # ---------------------------------------------------------------------------
    # Load checkpoint
    # ---------------------------------------------------------------------------
    print(f"loading {args.checkpoint} …", flush=True)
    alat = _lat.AdaptiveLattice.load(args.checkpoint)
    cl = alat[0]

    pos = np.array(cl.leaf_pos)  # (N, 3)
    ell = np.array(cl.leaf_ell)  # (N,)
    u = np.array(cl.leaf_u)  # (N,)
    lev = np.array(cl.leaf_level)  # (N,)
    chi = 1.0 - ell

    print(
        f"  N={cl.n_leaves}  levels={lev.min()}-{lev.max()}"
        f"  chi=[{chi.min():.4f}, {chi.max():.4f}]  u=[{u.min():.4f}, {u.max():.4f}]"
    )

    # ---------------------------------------------------------------------------
    # Equatorial slice (|z| < z_thick), optionally filtered to finer AMR levels
    # ---------------------------------------------------------------------------
    mask = (np.abs(pos[:, 2]) < args.z_thick) & (lev >= args.min_level)
    if args.min_level > 0:
        print(
            f"  keeping levels >= {args.min_level}: {mask.sum()} / {len(mask)} cells",
            flush=True,
        )
    px, py = pos[mask, 0], pos[mask, 1]
    cx = chi[mask]
    ux = u[mask]

    # interpolate to regular grid
    xi = np.linspace(px.min(), px.max(), args.grid_n)
    yi = np.linspace(py.min(), py.max(), args.grid_n)
    Xi, Yi = np.meshgrid(xi, yi)
    points = np.column_stack([px, py])

    print("interpolating chi field …", flush=True)
    Zchi = griddata(points, cx, (Xi, Yi), method="linear")
    Zu = griddata(points, ux, (Xi, Yi), method="linear")

    # ---------------------------------------------------------------------------
    # Chi gradient (finite difference on the regular grid)
    # ---------------------------------------------------------------------------
    dchi_dy, dchi_dx = np.gradient(Zchi, yi, xi)
    # gradient points toward increasing chi (toward mass) — this IS the gravity direction
    # stride for quiver plot
    s = args.grad_stride
    Xq, Yq = Xi[::s, ::s], Yi[::s, ::s]
    Gx, Gy = dchi_dx[::s, ::s], dchi_dy[::s, ::s]
    Gm = np.sqrt(Gx**2 + Gy**2)
    Gm_safe = np.where(Gm > 0, Gm, 1.0)
    Gx_n, Gy_n = Gx / Gm_safe, Gy / Gm_safe  # unit vectors

    # ---------------------------------------------------------------------------
    # Radial profile — chi_max and u_max binned by r over the equatorial slice.
    # Background (smooth solar falloff) is subtracted so planet peaks are visible.
    # ---------------------------------------------------------------------------
    r_eq = np.sqrt(px**2 + py**2)
    r_max_domain = r_eq.max()
    n_bins = 300
    r_edges = np.linspace(0.0, r_max_domain, n_bins + 1)
    r_centres = 0.5 * (r_edges[:-1] + r_edges[1:])
    chi_rmax = np.full(n_bins, np.nan)
    u_rmax = np.full(n_bins, np.nan)
    for k in range(n_bins):
        mask_k = (r_eq >= r_edges[k]) & (r_eq < r_edges[k + 1])
        if mask_k.any():
            chi_rmax[k] = cx[mask_k].max()
            u_rmax[k] = ux[mask_k].max()

    # Smooth background captures the broad solar falloff; residual isolates planets.
    # Kernel width = ~15 % of domain so individual planet widths (~3-5 bins) survive.
    _kernel = max(3, n_bins // 7)
    chi_bg = uniform_filter1d(np.nan_to_num(chi_rmax), size=_kernel)
    chi_excess = np.where(np.isfinite(chi_rmax), chi_rmax - chi_bg, np.nan)
    u_bg = uniform_filter1d(np.nan_to_num(u_rmax), size=_kernel)
    u_excess = np.where(np.isfinite(u_rmax), u_rmax - u_bg, np.nan)

    # ---------------------------------------------------------------------------
    # Resolve orbit radii: explicit override OR auto-detect from u_max peaks.
    # ---------------------------------------------------------------------------
    if args.body_orbits:
        _planet_orbits = [float(v) for v in args.body_orbits.split(",")]
    else:
        _n_planets = max(0, len(bodies) - 1)
        _planet_orbits = _auto_detect_orbits(r_centres, u_rmax, _n_planets)
        if _planet_orbits:
            print(
                f"  auto-detected orbits: {[f'{r:.1f}' for r in _planet_orbits]}",
                flush=True,
            )
        else:
            print(
                "  WARNING: no planet peaks detected — pass --body-orbits manually",
                flush=True,
            )

    orbit_vals_all = [0.0] + _planet_orbits
    _bodies_trimmed = bodies[: len(orbit_vals_all)]

    # ---------------------------------------------------------------------------
    # Centre-of-mass per body (lam-weighted centroid in local window)
    # ---------------------------------------------------------------------------
    com_positions = {"Sun": (0.0, 0.0)}
    for name, orb in zip(_bodies_trimmed[1:], _planet_orbits):
        centre = np.array([0.0, orb])
        window = max(orb * 0.4, 5.0)
        dist = np.sqrt((px - centre[0]) ** 2 + (py - centre[1]) ** 2)
        w_mask = dist < window
        w = np.maximum(cx[w_mask], 0.0)
        wsum = float(np.sum(w))
        if wsum > 1e-20:
            com_positions[name] = (
                float(np.dot(w, px[w_mask]) / wsum),
                float(np.dot(w, py[w_mask]) / wsum),
            )

    # ---------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(26, 8))
    fig.patch.set_facecolor("#0a0a1a")
    for ax in axes:
        ax.set_facecolor("#0a0a1a")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    # --- panel 1: chi terrain ---
    ax = axes[0]
    chi_plot = np.log1p(Zchi * 1000) if args.log_chi else Zchi
    im = ax.imshow(
        chi_plot,
        extent=[xi.min(), xi.max(), yi.min(), yi.max()],
        origin="lower",
        cmap="inferno",
        aspect="equal",
        interpolation="bilinear",
    )
    cb = plt.colorbar(im, ax=ax, pad=0.02)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
    cb.set_label("log(1 + chi×1000)" if args.log_chi else "chi", color="white")

    # gradient arrows
    valid = np.isfinite(Gx_n) & np.isfinite(Gy_n)
    ax.quiver(
        Xq[valid],
        Yq[valid],
        Gx_n[valid],
        Gy_n[valid],
        color="cyan",
        alpha=0.4,
        scale=60,
        width=0.002,
    )

    ax.set_title("Chi terrain + gradient (gravity direction)")
    ax.set_xlabel("x [cells]")
    ax.set_ylabel("y [cells]")

    # body labels
    for name, orb in zip(_bodies_trimmed, orbit_vals_all):
        ax.plot(0, orb, "o", color="white", ms=4)
        ax.annotate(
            name,
            (0, orb),
            color="white",
            fontsize=7,
            xytext=(5, 5),
            textcoords="offset points",
        )
    for name, (cx_, cy_) in com_positions.items():
        ax.plot(cx_, cy_, "+", color="lime", ms=8, mew=1.5)

    # --- panel 2: u field (planet blobs) ---
    ax = axes[1]
    u_plot = np.where(np.isfinite(Zu), Zu, 0.0)
    im2 = ax.imshow(
        u_plot,
        extent=[xi.min(), xi.max(), yi.min(), yi.max()],
        origin="lower",
        cmap="plasma",
        aspect="equal",
        interpolation="bilinear",
        norm=mcolors.PowerNorm(
            gamma=0.4, vmin=0, vmax=float(np.nanpercentile(u_plot, 99.5))
        ),
    )
    cb2 = plt.colorbar(im2, ax=ax, pad=0.02)
    cb2.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb2.ax.yaxis.get_ticklabels(), color="white")
    cb2.set_label("u (pressure)", color="white")
    ax.set_title("u field — planet blob positions")
    ax.set_xlabel("x [cells]")
    ax.set_ylabel("y [cells]")

    for name, orb in zip(_bodies_trimmed, orbit_vals_all):
        ax.plot(0, orb, "o", color="white", ms=4)
        ax.annotate(
            name,
            (0, orb),
            color="white",
            fontsize=7,
            xytext=(5, 5),
            textcoords="offset points",
        )

    # --- panel 3: chi/u excess over solar background — all planet peaks visible ---
    ax3 = axes[2]
    ax3.set_facecolor("#0a0a1a")

    # Both raw (log, faint) and excess (bright) so scale context is preserved.
    with np.errstate(divide="ignore", invalid="ignore"):
        chi_log = np.log10(np.where(chi_rmax > 0, chi_rmax, np.nan))
    ax3_log = ax3.twinx()
    ax3_log.plot(r_centres, chi_log, color="gold", lw=0.6, alpha=0.3)
    ax3_log.set_ylabel("log₁₀(chi_max)", color="gold", alpha=0.5)
    ax3_log.tick_params(axis="y", colors="gold", labelsize=7)
    for spine in ax3_log.spines.values():
        spine.set_edgecolor("#444")

    # excess chi (planet peaks above smooth solar background)
    ax3.plot(r_centres, chi_excess, color="gold", lw=1.4, label="Δchi")
    ax3.plot(r_centres, u_excess, color="#88aaff", lw=1.0, alpha=0.85, label="Δu")
    ax3.axhline(0, color="#444", lw=0.6, linestyle=":")

    colors_body = plt.cm.tab10(np.linspace(0, 1, len(_bodies_trimmed)))
    for name, orb, col in zip(_bodies_trimmed, orbit_vals_all, colors_body):
        ax3.axvline(orb, color=col, lw=0.8, alpha=0.7, linestyle="--")
        ax3.text(
            orb,
            1.0,
            f" {name}",
            color=col,
            fontsize=7,
            rotation=90,
            va="top",
            transform=ax3.get_xaxis_transform(),
        )

    ax3.set_xlabel("r [cells]")
    ax3.set_ylabel("chi / u excess over background", color="white")
    ax3.tick_params(axis="y", colors="white")
    ax3.set_title("Radial excess chi/u — planet peaks")
    ax3.legend(fontsize=8, facecolor="#111", labelcolor="white")
    for spine in ax3.spines.values():
        spine.set_edgecolor("#444")

    plt.tight_layout()
    if args.out:
        plt.savefig(
            args.out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor()
        )
        print(f"saved → {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
