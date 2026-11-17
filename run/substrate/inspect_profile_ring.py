"""
inspect_profile_ring.py — load saved profile_ring start/end lattices and
visualise the phi field angular structure to check for dipole character.

Produces:
  - Radial profile (inner region, start vs end)
  - 2D cross-section slices through the ring (axial plane + equatorial plane)
  - Angular structure at fixed radii: phi_rms vs theta to check 2:1 axial/equatorial

Usage:
    PYTHONPATH=$PWD python3 run/substrate/inspect_profile_ring.py \
        --start tmp/profile_ring_start.npz \
        --end   tmp/profile_ring_end.npz \
        --axis  z
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium.substrate.lattice import SubstrateLattice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="tmp/profile_ring_start.npz")
    parser.add_argument("--end", type=str, default="tmp/profile_ring_end.npz")
    parser.add_argument("--axis", type=str, default="z")
    parser.add_argument("--out", type=str, default="inspect_profile_ring.png")
    parser.add_argument(
        "--r-shell-cells",
        type=float,
        nargs="+",
        default=[4.0, 6.0, 8.0],
        help="radii (cells) at which to measure angular phi_rms profile",
    )
    parser.add_argument(
        "--shell-width",
        type=float,
        default=0.8,
        help="half-width of shell in cells for angular measurement",
    )
    parser.add_argument(
        "--ring-n-tiles",
        type=int,
        default=8,
        help="number of ring tiles — used to find geometric ring centre",
    )
    args = parser.parse_args()

    ax_idx = {"x": 0, "y": 1, "z": 2}[args.axis]
    perp = [i for i in range(3) if i != ax_idx]

    print(f"Loading {args.start} ...")
    lat_s = SubstrateLattice.load(args.start)
    print(f"Loading {args.end} ...")
    lat_e = SubstrateLattice.load(args.end)

    pos = lat_e.pos

    # Ring centre: centroid of the top-N phi_rms cells in start lattice (the ring tiles)
    phi_rms_s = np.sqrt((lat_s.phi**2).mean(axis=1))
    top_idx = np.argpartition(phi_rms_s, -args.ring_n_tiles)[-args.ring_n_tiles :]
    ring_centre = pos[top_idx].mean(axis=0)
    print(f"Ring tile positions:\n{pos[top_idx]}")
    print(
        f"Ring centre (centroid of top-{args.ring_n_tiles} phi_rms tiles): {ring_centre}"
    )

    # Radial distance from ring centre
    r_all = np.linalg.norm(pos - ring_centre, axis=1)
    phi_rms_e = np.sqrt((lat_e.phi**2).mean(axis=1))

    # Polar angle from ring axis (theta=0 is along axis)
    axial_dist = pos[:, ax_idx] - ring_centre[ax_idx]
    perp_dist = np.linalg.norm(pos[:, perp] - ring_centre[perp], axis=1)
    theta = np.arctan2(perp_dist, np.abs(axial_dist))  # 0=axial, pi/2=equatorial

    # ---------------------------------------------------------------------------
    # 1. Radial profile — inner region, start vs end
    # ---------------------------------------------------------------------------
    r_max_prof = 12.0
    n_bins = 30
    edges = np.linspace(0.5, r_max_prof, n_bins + 1)
    r_mid = 0.5 * (edges[:-1] + edges[1:])

    def radial_profile(phi_rms):
        totals = np.zeros(n_bins)
        counts = np.zeros(n_bins, dtype=int)
        idx = ((r_all - 0.5) / (r_max_prof - 0.5) * n_bins).astype(int)
        valid = (r_all >= 0.5) & (r_all < r_max_prof) & (idx >= 0) & (idx < n_bins)
        np.add.at(totals, idx[valid], phi_rms[valid])
        np.add.at(counts, idx[valid], 1)
        m = counts > 0
        return np.where(m, totals / np.maximum(counts, 1), np.nan)

    prof_s = radial_profile(phi_rms_s)
    prof_e = radial_profile(phi_rms_e)

    # Power-law fit to end profile inner region (r=2-8, excluding ring itself r<1.5)
    fit_mask = (r_mid > 1.5) & (r_mid < 8.0) & np.isfinite(prof_e) & (prof_e > 0)
    if fit_mask.sum() >= 3:
        coeffs = np.polyfit(np.log(r_mid[fit_mask]), np.log(prof_e[fit_mask]), 1)
        n_fit = coeffs[0]
        A_fit = np.exp(coeffs[1])
        print(
            f"Inner region power law (r=2-7): phi_rms ~ {A_fit:.3e} * r^{n_fit:.3f}  (A-field=-2, B-field=-3)"
        )
    else:
        n_fit, A_fit = None, None

    # ---------------------------------------------------------------------------
    # 2. Angular structure at fixed radii
    # ---------------------------------------------------------------------------
    print("\nAngular structure (phi_rms vs theta):")
    print(f"  Shell width = ±{args.shell_width} cells")
    n_theta = 18
    theta_edges = np.linspace(0, np.pi / 2, n_theta + 1)
    theta_mid = 0.5 * (theta_edges[:-1] + theta_edges[1:])

    shell_results = []
    for r_shell in args.r_shell_cells:
        in_shell = (r_all > r_shell - args.shell_width) & (
            r_all < r_shell + args.shell_width
        )
        ang_tot = np.zeros(n_theta)
        ang_cnt = np.zeros(n_theta, dtype=int)
        t_idx = (
            ((theta[in_shell]) / (np.pi / 2) * n_theta).astype(int).clip(0, n_theta - 1)
        )
        np.add.at(ang_tot, t_idx, phi_rms_e[in_shell])
        np.add.at(ang_cnt, t_idx, 1)
        ang_m = ang_cnt > 0
        ang_prof = np.where(ang_m, ang_tot / np.maximum(ang_cnt, 1), np.nan)

        axial_val = ang_prof[0] if ang_m[0] else np.nan
        eq_val = ang_prof[-1] if ang_m[-1] else np.nan
        ratio = axial_val / eq_val if (np.isfinite(eq_val) and eq_val > 0) else np.nan
        print(
            f"  r={r_shell:.1f}: phi_axial={axial_val:.3e}  phi_equatorial={eq_val:.3e}  "
            f"ratio={ratio:.2f}  (dipole=2.0)"
        )
        shell_results.append((r_shell, theta_mid, ang_prof, ratio))

    # ---------------------------------------------------------------------------
    # 3. 2D cross-section slice — axial plane (perp[0] vs axis)
    # ---------------------------------------------------------------------------
    slice_w = 1.0  # half-width in cells for slice
    in_slice = np.abs(pos[:, perp[1]] - ring_centre[perp[1]]) < slice_w

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        fig = plt.figure(figsize=(20, 10))
        fig.suptitle(f"Profile ring inspection  axis={args.axis}", fontsize=11)

        # Panel 1: radial profile start vs end
        ax1 = fig.add_subplot(2, 3, 1)
        m_s = np.isfinite(prof_s) & (prof_s > 0)
        m_e = np.isfinite(prof_e) & (prof_e > 0)
        ax1.semilogy(r_mid[m_s], prof_s[m_s], "b--o", ms=3, lw=1, label="start")
        ax1.semilogy(r_mid[m_e], prof_e[m_e], "r-o", ms=3, lw=1, label="end")
        if A_fit is not None:
            r_fit = np.linspace(2.0, 7.0, 50)
            ax1.semilogy(
                r_fit, A_fit * r_fit**n_fit, "k--", lw=1.5, label=f"fit r^{n_fit:.2f}"
            )
            ax1.semilogy(
                r_fit,
                A_fit * (r_fit / 2.0) ** (-2) * A_fit * 2.0 ** (-n_fit),
                "g:",
                lw=1.2,
                label="r^-2 (A-field)",
            )
        ax1.set_xlabel("r (cells)")
        ax1.set_ylabel("phi_rms")
        ax1.set_title("Radial profile: start vs end\n(inner region)")
        ax1.legend(fontsize=7)
        ax1.set_xlim(0, 12)

        # Panels 2-4: angular profile at each shell radius
        colors = ["steelblue", "tomato", "seagreen"]
        for k, (r_shell, tm, ap, ratio) in enumerate(shell_results):
            ax = fig.add_subplot(2, 3, k + 2)
            m = np.isfinite(ap)
            ax.plot(np.degrees(tm[m]), ap[m], "o-", color=colors[k], ms=4)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xlabel("θ from axis (deg)")
            ax.set_ylabel("phi_rms")
            ax.set_title(
                f"Angular profile r={r_shell:.1f}\naxial/equatorial={ratio:.2f} (dipole=2.0)"
            )
            ax.set_xlim(0, 90)

        # Panel 5: 2D slice — axial plane (axis vs perp[0])
        ax5 = fig.add_subplot(2, 3, 5)
        xs = pos[in_slice, perp[0]] - ring_centre[perp[0]]
        ys = pos[in_slice, ax_idx] - ring_centre[ax_idx]
        cs = phi_rms_e[in_slice]
        vmin = cs[cs > 0].min() if (cs > 0).any() else 1e-6
        sc5 = ax5.scatter(
            xs, ys, c=cs, s=8, cmap="hot", norm=LogNorm(vmin=vmin, vmax=cs.max())
        )
        plt.colorbar(sc5, ax=ax5, label="phi_rms (end)")
        ax5.set_xlabel(f"{['x','y','z'][perp[0]]} - centre")
        ax5.set_ylabel(f"{args.axis} - centre")
        ax5.set_title("2D axial slice (phi_rms, end)\nlog scale")
        ax5.set_aspect("equal")

        # Panel 6: 2D slice — equatorial plane (perp[0] vs perp[1])
        ax6 = fig.add_subplot(2, 3, 6)
        in_eq = np.abs(pos[:, ax_idx] - ring_centre[ax_idx]) < slice_w
        xe = pos[in_eq, perp[0]] - ring_centre[perp[0]]
        ye = pos[in_eq, perp[1]] - ring_centre[perp[1]]
        ce = phi_rms_e[in_eq]
        vmin6 = ce[ce > 0].min() if (ce > 0).any() else 1e-6
        sc6 = ax6.scatter(
            xe, ye, c=ce, s=8, cmap="hot", norm=LogNorm(vmin=vmin6, vmax=ce.max())
        )
        plt.colorbar(sc6, ax=ax6, label="phi_rms (end)")
        ax6.set_xlabel(f"{['x','y','z'][perp[0]]} - centre")
        ax6.set_ylabel(f"{['x','y','z'][perp[1]]} - centre")
        ax6.set_title("2D equatorial slice (phi_rms, end)\nlog scale")
        ax6.set_aspect("equal")

        plt.tight_layout()
        fig.savefig(args.out, dpi=120)
        print(f"\n[plot] {args.out}")
        plt.close(fig)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
