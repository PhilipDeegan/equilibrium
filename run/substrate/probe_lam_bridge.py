"""
probe_lam_bridge.py — lam gradient topology and phi propagation, two blobs.

Phase 1 (always): static lam gradient diagnostic — quantifies gradient
cancellation in the bridge vs the outward slope.

Phase 2 (--n-steps > 0): phi wave propagation through the fixed lam
landscape.  lam is frozen; a u impulse at the midpoint seeds phi and
step() runs for n_steps.  Tracks phi_rms in the bridge region vs the
outward region to expose whether the current wave equation channels phi
through the bridge or lets it escape isotropically.

Usage:
    PYTHONPATH=$PWD python3 run/substrate/probe_lam_bridge.py [options]
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium.substrate.lattice import DampedLattice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--R", type=float, default=25.0, help="domain radius in cells")
    parser.add_argument(
        "--sep", type=float, default=8.0, help="blob separation (blobs at z=±sep/2)"
    )
    parser.add_argument(
        "--chi0", type=float, default=0.5, help="chi_0 winding amplitude per blob"
    )
    parser.add_argument(
        "--n-steps", type=int, default=0, help="phi wave steps (0=static only)"
    )
    parser.add_argument(
        "--u-kick", type=float, default=1.0, help="u impulse amplitude at midpoint"
    )
    parser.add_argument(
        "--r-cut", type=float, default=1.5, help="z-axis selection radius (cells)"
    )
    parser.add_argument("--report", type=int, default=200, help="print every N steps")
    parser.add_argument(
        "--snap-dir",
        type=str,
        default="probe_lam_bridge_snaps",
        help="directory for cross-section snapshots ('' to disable)",
    )
    parser.add_argument(
        "--slice-w",
        type=float,
        default=1.2,
        help="half-width of y=0 slice for cross-section plots",
    )
    parser.add_argument(
        "--sink-scale",
        type=float,
        default=0.0,
        help="phi->lam conversion scale in step_lam_sink (0=off)",
    )
    parser.add_argument(
        "--sponge",
        type=float,
        default=0.15,
        help="sponge layer fraction of domain radius",
    )
    parser.add_argument(
        "--damp-max",
        type=float,
        default=0.003,
        help="max damping coefficient at outer boundary",
    )
    args = parser.parse_args()

    # ---------------------------------------------------------------------------
    # Setup
    # ---------------------------------------------------------------------------
    lat = DampedLattice(
        radius_cells=args.R, sponge_fraction=args.sponge, damp_max=args.damp_max
    )
    pos = lat.pos
    dt = lat.stable_dt()
    half = args.sep / 2.0

    def nearest(target):
        return lat.tiles[int(np.argmin(np.linalg.norm(pos - target, axis=1)))]

    kA = nearest(np.array([0.0, 0.0, half]))
    kB = nearest(np.array([0.0, 0.0, -half]))

    # Superpose two lam blobs (element-wise max preserves both cores).
    lat.set_winding(kA, chi_0=args.chi0)
    lam_A = lat.lam.copy()
    lat.lam[:] = 0.0
    lat.set_winding(kB, chi_0=args.chi0)
    lat.lam[:] = np.maximum(lam_A, lat.lam)

    lat.phi[:] = 0.0

    print(
        f"N={lat.N}  R={args.R}  sep={args.sep}  chi0={args.chi0:.3f}  "
        f"sponge={args.sponge}  damp_max={args.damp_max}"
    )
    print(f"kA={kA.pos}  kB={kB.pos}")
    print(f"lam_sum={lat.lam.sum():.4f}\n")

    # ---------------------------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------------------------

    def z_axis_profile():
        near = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2) < args.r_cut
        idx = np.where(near)[0]
        order = np.argsort(pos[idx, 2])
        idx_s = idx[order]
        return pos[idx_s, 2], lat.lam[idx_s]

    def bridge_stats(z_s, lam_s):
        """Return (grad_bridge, grad_outward, ratio) from finite differences along z."""
        dz = np.diff(z_s)
        dlam = np.diff(lam_s)
        grad = np.abs(dlam / np.where(dz > 1e-12, dz, 1e-12))
        z_m = 0.5 * (z_s[:-1] + z_s[1:])

        m_bridge = np.abs(z_m) < half / 2.0
        m_out = (np.abs(z_m) > half) & (np.abs(z_m) < half + args.sep)

        g_b = float(grad[m_bridge].mean()) if m_bridge.any() else float("nan")
        g_o = float(grad[m_out].mean()) if m_out.any() else float("nan")
        return g_b, g_o, (g_b / g_o if g_o > 0 else float("nan"))

    def phi_zone_rms():
        """phi_rms in bridge (|z|<half/2) vs outward (half<|z|<half+sep) zone."""
        r_xy = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
        phi_rms = np.sqrt((lat.phi**2).mean(axis=1))

        m_bridge = (np.abs(pos[:, 2]) < half / 2.0) & (r_xy < half)
        m_out = (np.abs(pos[:, 2]) > half) & (np.abs(pos[:, 2]) < half + args.sep)

        p_b = float(phi_rms[m_bridge].mean()) if m_bridge.any() else float("nan")
        p_o = float(phi_rms[m_out].mean()) if m_out.any() else float("nan")
        return p_b, p_o, (p_b / p_o if p_o > 0 else float("nan"))

    def save_cross_section(step, snap_dir, phi_vmax=None):
        """
        Save a 2-panel cross-section figure at the current step.

        Left panel  — x-z slice (|y| < slice_w): phi_rms per cell as colour.
        Right panel — x-y slice at z≈0 (|z| < slice_w): same colouring.

        Colour scale is log10(phi_rms), shared across both panels so frames are
        comparable.  phi_vmax fixes the colour ceiling across all snapshots so
        the animation is consistent; pass the step-0 max or leave None to auto.
        """
        import os

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        phi_rms = np.sqrt((lat.phi**2).mean(axis=1))
        # log scale — floor at 1e-12 to avoid log(0)
        log_phi = np.log10(np.maximum(phi_rms, 1e-12))

        vmin = -12.0
        vmax = float(np.log10(phi_vmax)) if phi_vmax else float(log_phi.max())

        # xz slice: cells with |y| < slice_w
        m_xz = np.abs(pos[:, 1]) < args.slice_w
        # xy slice: cells with |z| < slice_w
        m_xy = np.abs(pos[:, 2]) < args.slice_w

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            f"φ density — step {step}  (sep={args.sep}, chi0={args.chi0})", fontsize=10
        )

        sc1 = ax1.scatter(
            pos[m_xz, 0],
            pos[m_xz, 2],
            c=log_phi[m_xz],
            cmap="hot",
            s=4,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
        ax1.axhline(
            half, color="cyan", lw=0.8, ls="--", alpha=0.7, label=f"blobs z=±{half:.1f}"
        )
        ax1.axhline(-half, color="cyan", lw=0.8, ls="--", alpha=0.7)
        ax1.axhline(0, color="white", lw=0.5, ls=":", alpha=0.5, label="midpoint")
        ax1.set_xlabel("x")
        ax1.set_ylabel("z")
        ax1.set_title(f"x-z slice (|y| < {args.slice_w:.1f})")
        ax1.set_aspect("equal")
        ax1.legend(fontsize=7, loc="upper right")
        plt.colorbar(sc1, ax=ax1, label="log₁₀(φ rms)")

        sc2 = ax2.scatter(
            pos[m_xy, 0],
            pos[m_xy, 1],
            c=log_phi[m_xy],
            cmap="hot",
            s=4,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_title(f"x-y slice at z≈0 (|z| < {args.slice_w:.1f})")
        ax2.set_aspect("equal")
        plt.colorbar(sc2, ax=ax2, label="log₁₀(φ rms)")

        plt.tight_layout()
        os.makedirs(snap_dir, exist_ok=True)
        fname = os.path.join(snap_dir, f"snap_{step:06d}.png")
        fig.savefig(fname, dpi=110)
        plt.close(fig)
        print(f"  [snap] {fname}", flush=True)

    # ---------------------------------------------------------------------------
    # Static snapshot
    # ---------------------------------------------------------------------------
    print("=== static lam gradient (phi=0, no steps) ===")
    z_s, lam_s = z_axis_profile()
    g_b, g_o, ratio = bridge_stats(z_s, lam_s)
    print(f"  grad_bridge={g_b:.4e}   grad_outward={g_o:.4e}   ratio={ratio:.4f}")
    print(f"  lam at midpoint (z≈0):   {lam_s[np.argmin(np.abs(z_s))]:.6f}")
    print(f"  lam at blob A (z≈+half): {lam_s[np.argmin(np.abs(z_s - half))]:.6f}")

    # ---------------------------------------------------------------------------
    # Optional phi wave propagation through frozen lam landscape
    # ---------------------------------------------------------------------------
    history_phi = []
    if args.n_steps > 0:
        # u impulse at the midpoint — seeds phi from between the two blobs.
        # lam is never updated so the landscape stays fixed throughout.
        mid_idx = int(np.argmin(np.linalg.norm(pos, axis=1)))
        lat.u[mid_idx] = args.u_kick

        sink_desc = f"  sink_scale={args.sink_scale:.1e}" if args.sink_scale else ""
        print(
            f"\nrunning {args.n_steps} wave steps  (u_kick={args.u_kick:.2f} at midpoint"
            f"{sink_desc})"
        )
        print(
            f"{'step':>8}  {'phi_rms_global':>16}  {'phi_bridge':>12}  {'phi_outward':>12}  {'ratio':>8}"
        )

        snap_dir = args.snap_dir
        do_snaps = bool(snap_dir)
        phi_vmax = None  # set on first snap so colour scale is fixed across all frames

        for s in range(1, args.n_steps + 1):
            lat.step(dt)
            if args.sink_scale:
                lat.step_lam_sink(dt, scale=args.sink_scale)
            if s % args.report == 0:
                p_b, p_o, ratio = phi_zone_rms()
                phi_global = float(np.sqrt(np.mean(lat.phi**2)))
                print(
                    f"{s:8d}  {phi_global:16.4e}  {p_b:12.4e}  {p_o:12.4e}  {ratio:8.4f}",
                    flush=True,
                )
                history_phi.append((s, phi_global, p_b, p_o, ratio))
                if do_snaps:
                    if phi_vmax is None:
                        phi_vmax = (
                            float(np.sqrt((lat.phi**2).mean(axis=1)).max()) or 1e-6
                        )
                    save_cross_section(s, snap_dir, phi_vmax)

    # ---------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        z_s, lam_s = z_axis_profile()
        dz = np.diff(z_s)
        dlam = np.diff(lam_s)
        grad = np.abs(dlam / np.where(dz > 1e-12, dz, 1e-12))
        z_m = 0.5 * (z_s[:-1] + z_s[1:])

        n_panels = 3 if (args.n_steps > 0 and history_phi) else 2
        fig, axes = plt.subplots(
            n_panels, 1, figsize=(10, 4 * n_panels), sharex=(n_panels == 2)
        )

        ax1, ax2 = axes[0], axes[1]
        ax1.plot(z_s, lam_s, "r-", lw=1.5)
        ax1.axvline(half, color="k", lw=0.8, ls="--", label=f"blobs ±{half:.1f}")
        ax1.axvline(-half, color="k", lw=0.8, ls="--")
        ax1.axvline(0, color="gray", lw=0.5, ls=":", label="midpoint")
        ax1.set_ylabel("λ")
        ax1.set_title(f"z-axis lam  (sep={args.sep}, chi0={args.chi0})")
        ax1.legend(fontsize=8)

        ax2.plot(z_m, grad, "b-", lw=1.5)
        ax2.axvline(half, color="k", lw=0.8, ls="--")
        ax2.axvline(-half, color="k", lw=0.8, ls="--")
        ax2.axvline(0, color="gray", lw=0.5, ls=":")
        ax2.set_ylabel("|dλ/dz|")
        ax2.set_title("lam gradient magnitude along z-axis")
        if n_panels == 2:
            ax2.set_xlabel("z")

        if n_panels == 3:
            ax3 = axes[2]
            steps_h = [h[0] for h in history_phi]
            phi_b_h = [h[2] for h in history_phi]
            phi_o_h = [h[3] for h in history_phi]
            ax3.plot(steps_h, phi_b_h, "g-", lw=1.5, label="φ_rms bridge")
            ax3.plot(steps_h, phi_o_h, "m--", lw=1.2, label="φ_rms outward")
            ax3.set_ylabel("φ rms")
            ax3.set_xlabel("step")
            ax3.set_title(
                "phi accumulation: bridge vs outward  (lam frozen, u_kick at midpoint)"
            )
            ax3.legend(fontsize=8)

        plt.tight_layout()
        fname = "probe_lam_bridge.png"
        fig.savefig(fname, dpi=100)
        print(f"\n[plot] {fname}")
        plt.close(fig)

    except ImportError:
        print("matplotlib not available — skipping plot")


if __name__ == "__main__":
    main()
