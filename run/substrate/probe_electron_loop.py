"""
probe_electron_loop.py — stability and inertia of the 8-tile toroidal winding.

Phase 1: init the minimal toroidal ring (electron topology), run wave steps,
track circulation amplitude and leakage — measures winding decay rate.

Phase 2 (--gradient > 0): apply a small u-gradient across the ring after
the stability run and track the ring COM drift.  Rate of drift / gradient
gives 1/mass in DST units.

Usage:
    PYTHONPATH=$PWD python3 run/substrate/probe_electron_loop.py [options]
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium.substrate.flux_tube import (
    find_toroidal_ring,
    init_toroidal_winding,
    winding_number,
)
from equilibrium.substrate.lattice import DampedLattice

parser = argparse.ArgumentParser()
parser.add_argument("--R", type=float, default=15.0, help="domain radius in cells")
parser.add_argument("--axis", type=str, default="z", help="ring axis: x, y, or z")
parser.add_argument(
    "--amplitude", type=float, default=1.0, help="phi circulation amplitude"
)
parser.add_argument(
    "--lam-ring", type=float, default=1.0, help="lambda on ring tiles (0=vacuum)"
)
parser.add_argument("--n-steps", type=int, default=5000, help="stability run steps")
parser.add_argument(
    "--gradient",
    type=float,
    default=0.0,
    help="u-gradient amplitude for inertia phase (0=skip)",
)
parser.add_argument("--n-grad", type=int, default=2000, help="steps in gradient phase")
parser.add_argument("--report", type=int, default=500, help="print every N steps")
parser.add_argument("--sponge", type=float, default=0.20)
parser.add_argument("--damp-max", type=float, default=0.005)
parser.add_argument(
    "--dt-scale",
    type=float,
    default=1.0,
    help="scale factor on stable dt (< 1 for finer time resolution)",
)
parser.add_argument(
    "--snap-dir",
    type=str,
    default="",
    help="directory for cross-section snaps (''=off)",
)
parser.add_argument("--slice-w", type=float, default=1.2)
parser.add_argument(
    "--snap-r", type=float, default=4.0, help="plot radius around ring centre (cells)"
)


def run(args):
    # -----------------------------------------------------------------------
    # Build lattice and find ring
    # -----------------------------------------------------------------------
    lat = DampedLattice(
        radius_cells=args.R, sponge_fraction=args.sponge, damp_max=args.damp_max
    )
    pos = lat.pos
    dt = lat.stable_dt() * args.dt_scale

    ring_tiles, ring_faces = find_toroidal_ring(lat, axis=args.axis)
    if ring_tiles is None:
        print("ERROR: could not find toroidal ring — try a larger R")
        sys.exit(1)

    ring_pos = pos[ring_tiles]
    ring_centre = ring_pos.mean(axis=0)
    ring_r = float(np.linalg.norm(ring_pos - ring_centre, axis=1).mean())

    print(f"N={lat.N}  R={args.R}  axis={args.axis}  dt={dt:.4e}")
    print(f"Ring tiles: {ring_tiles}")
    print(f"Ring centre: {ring_centre}  ring_r≈{ring_r:.3f}")
    print(f"Winding number: {winding_number(lat, ring_tiles, axis=args.axis):+.3f}\n")

    # Optionally set lam on ring tiles to give them substrate depletion
    if args.lam_ring > 0:
        for t in ring_tiles:
            lat.lam[t] = args.lam_ring

    omega = init_toroidal_winding(
        lat, ring_tiles, ring_faces, amplitude=args.amplitude, frozen=False
    )
    print(f"Resonant ω = {omega:.4f}  (period = {2*np.pi/omega:.1f} steps)\n")

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------
    _ring_set = set(ring_tiles)

    # Cells adjacent to the ring but not in it — measure leakage into these
    _adj_set = set()
    for t in ring_tiles:
        for nb in lat.nb[t]:
            if nb >= 0 and nb not in _ring_set:
                _adj_set.add(int(nb))
    _adj = np.array(sorted(_adj_set), dtype=int)
    _ring = np.array(ring_tiles, dtype=int)

    def ring_circulation():
        """
        Amplitude of the n=1 winding mode (first Fourier coefficient around the ring).

        For N=8 tiles with K=2π/8, the resonant mode has phi_j = A·cos(K·(j+½)).
        The signed sum is zero by symmetry; this returns the complex amplitude
        |Σ phi_j · exp(i·2π·j/N)| / (N/2) — non-zero only for the circulating mode.
        """
        N = len(ring_tiles)
        total = sum(
            float(lat.phi[t, f])
            * complex(np.cos(2 * np.pi * j / N), np.sin(2 * np.pi * j / N))
            for j, (t, f) in enumerate(zip(ring_tiles, ring_faces))
        )
        return abs(total) / (N / 2)

    def ring_metrics():
        circ = ring_circulation()
        ring_prms = float(np.sqrt(np.mean(lat.phi[_ring] ** 2)))
        ring_urms = float(np.sqrt(np.mean(lat.u[_ring] ** 2)))
        adj_prms = float(np.sqrt(np.mean(lat.phi[_adj] ** 2))) if len(_adj) else 0.0
        global_prms = float(np.sqrt(np.mean(lat.phi**2)))
        return circ, ring_prms, ring_urms, adj_prms, global_prms

    def ring_com(axis_idx):
        """phi_rms-weighted centre of mass of ring tiles along one axis."""
        w = np.sqrt(np.mean(lat.phi[_ring] ** 2, axis=1))
        w_sum = w.sum()
        if w_sum < 1e-30:
            return float(ring_centre[axis_idx])
        return float(np.dot(w, pos[_ring, axis_idx]) / w_sum)

    def save_snap(step, snap_dir, phi_vmax):
        import os

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ax_idx = {"x": 0, "y": 1, "z": 2}[args.axis]
        perp = [i for i in range(3) if i != ax_idx]
        N_ring = len(ring_tiles)
        K = 2 * np.pi / N_ring

        # Order ring tiles by angle around the ring centre in the ring plane
        dx = ring_pos[:, perp[0]] - ring_centre[perp[0]]
        dy = ring_pos[:, perp[1]] - ring_centre[perp[1]]
        angles = np.arctan2(dy, dx)
        order = np.argsort(angles)
        ordered_tiles = [ring_tiles[i] for i in order]
        ordered_faces = [ring_faces[i] for i in order]
        ordered_angles = angles[order]

        # phi on each directed ring edge, in angle order
        phi_ring = np.array(
            [float(lat.phi[t, f]) for t, f in zip(ordered_tiles, ordered_faces)]
        )
        # expected resonant mode shape (amplitude fitted to data)
        amp = float(np.sqrt(2) * np.std(phi_ring)) if np.std(phi_ring) > 1e-15 else 1.0
        phi_expected = amp * np.cos(
            K * np.arange(N_ring)
            + np.arctan2(
                np.sum(phi_ring * np.sin(K * np.arange(N_ring))),
                np.sum(phi_ring * np.cos(K * np.arange(N_ring))),
            )
        )

        # per-tile phi_rms for ring + adjacent shell
        all_show = list(_ring_set | _adj_set)
        show_pos = pos[all_show]
        show_prms = np.sqrt((lat.phi[all_show] ** 2).mean(axis=1))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
        fig.suptitle(f"step {step}  |winding|={ring_circulation():.4f}", fontsize=10)

        # --- Left: phi on each ring edge vs position around ring ---
        bar_x = np.arange(N_ring)
        colors = ["tomato" if v >= 0 else "steelblue" for v in phi_ring]
        ax1.bar(bar_x, phi_ring, color=colors, alpha=0.8, label="φ on ring edge")
        ax1.plot(bar_x, phi_expected, "k--", lw=1.5, label="fitted mode")
        ax1.axhline(0, color="k", lw=0.5)
        ax1.set_xticks(bar_x)
        ax1.set_xticklabels(
            [f"{np.degrees(a):.0f}°" for a in ordered_angles], fontsize=7
        )
        ax1.set_xlabel("tile angle around ring")
        ax1.set_ylabel("φ (directed)")
        ax1.set_title("ring edge φ — cosine = intact winding")
        ax1.legend(fontsize=8)

        # --- Right: spatial layout of ring + 1-shell neighbours ---
        sc = ax2.scatter(
            show_pos[:, perp[0]],
            show_pos[:, perp[1]],
            c=show_prms,
            cmap="hot",
            s=120,
            vmin=0,
            vmax=max(float(phi_vmax), 1e-12),
        )
        # label each ring tile with its index in the ring
        for idx, (t, angle) in enumerate(zip(ordered_tiles, ordered_angles)):
            ax2.annotate(
                str(idx),
                (pos[t, perp[0]], pos[t, perp[1]]),
                ha="center",
                va="center",
                fontsize=7,
                color="cyan",
                fontweight="bold",
            )
        plt.colorbar(sc, ax=ax2, label="φ rms")
        ax2.set_xlabel(["x", "y", "z"][perp[0]])
        ax2.set_ylabel(["x", "y", "z"][perp[1]])
        ax2.set_aspect("equal")
        ax2.set_title("ring + neighbours (cyan = ring index)")

        plt.tight_layout()
        os.makedirs(snap_dir, exist_ok=True)
        fname = os.path.join(snap_dir, f"snap_{step:07d}.png")
        fig.savefig(fname, dpi=110)
        plt.close(fig)
        print(f"  [snap] {fname}", flush=True)

    # -----------------------------------------------------------------------
    # Phase 1 — stability run
    # -----------------------------------------------------------------------
    print("=== Phase 1: stability ===")
    print(
        f"{'step':>8}  {'circulation':>14}  {'ring_phi':>10}  {'ring_u':>10}  "
        f"{'adj_phi':>10}  {'global_phi':>12}"
    )

    phi_vmax = None
    history = []

    circ0, rp0, ru0, ap0, gp0 = ring_metrics()
    print(
        f"{'0':>8}  {circ0:+14.6e}  {rp0:10.4e}  {ru0:10.4e}  {ap0:10.4e}  {gp0:12.4e}"
    )
    if args.snap_dir:
        _ring_phi = np.sqrt((lat.phi[_ring] ** 2).mean(axis=1)).max()
        phi_vmax = max(float(_ring_phi), 1e-12)
        save_snap(0, args.snap_dir, phi_vmax)

    for s in range(1, args.n_steps + 1):
        lat.step(dt)
        if s % args.report == 0:
            circ, rp, ru, ap, gp = ring_metrics()
            print(
                f"{s:8d}  {circ:+14.6e}  {rp:10.4e}  {ru:10.4e}  {ap:10.4e}  {gp:12.4e}",
                flush=True,
            )
            history.append((s, circ, rp, ru, ap, gp))
            if args.snap_dir:
                if phi_vmax is None:
                    phi_vmax = max(
                        float(np.sqrt((lat.phi[_ring] ** 2).mean(axis=1)).max()), 1e-12
                    )
                save_snap(s, args.snap_dir, phi_vmax)

    # -----------------------------------------------------------------------
    # Phase 2 — inertia measurement (optional)
    # -----------------------------------------------------------------------
    if args.gradient > 0:
        ax_idx = {"x": 0, "y": 1, "z": 2}[args.axis]
        # Apply a static u-gradient perpendicular to the ring axis
        # Use the first perpendicular axis
        grad_ax = [i for i in range(3) if i != ax_idx][0]
        for i in range(lat.N):
            if i not in _ring_set:
                lat.u[i] += args.gradient * float(pos[i, grad_ax])

        com0 = ring_com(grad_ax)
        print(
            f"\n=== Phase 2: inertia (gradient={args.gradient:.2e} along axis {grad_ax}) ==="
        )
        print(f"  ring COM at start: {com0:.6f}")
        print(f"{'step':>8}  {'COM':>12}  {'ΔCOM':>12}  {'circulation':>14}")

        grad_hist = []
        grad_snap_dir = args.snap_dir + "_grad" if args.snap_dir else ""
        for s in range(1, args.n_grad + 1):
            lat.step(dt)
            if s % args.report == 0:
                com = ring_com(grad_ax)
                circ = ring_circulation()
                print(
                    f"{s:8d}  {com:12.6f}  {com-com0:+12.6f}  {circ:+14.6e}",
                    flush=True,
                )
                grad_hist.append((s, com, circ))
                if grad_snap_dir:
                    save_snap(s, grad_snap_dir, phi_vmax)

        if len(grad_hist) >= 2:
            drift_rate = (grad_hist[-1][1] - grad_hist[0][1]) / (
                (grad_hist[-1][0] - grad_hist[0][0]) * dt
            )
            force = args.gradient * ring_r * len(ring_tiles)
            mass_dst = force / max(abs(drift_rate), 1e-30)
            print(f"\n  drift rate  = {drift_rate:.4e} (DST units/tick)")
            print(f"  applied force ≈ {force:.4e}")
            print(f"  mass (DST)  ≈ {mass_dst:.4e}")

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    if history:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            steps = [h[0] for h in history]
            circs = [h[1] for h in history]
            rphis = [h[2] for h in history]
            aphis = [h[4] for h in history]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
            fig.suptitle(
                f"Electron loop stability  (R={args.R}, axis={args.axis}, "
                f"amp={args.amplitude}, lam_ring={args.lam_ring})",
                fontsize=10,
            )

            ax1.plot(steps, circs, "b-", lw=1.5, label="|n=1 mode amplitude|")
            ax1.set_ylabel("winding amplitude")
            ax1.set_title("n=1 mode amplitude vs step")
            ax1.legend(fontsize=8)

            ax2.semilogy(steps, rphis, "b-", lw=1.5, label="ring φ_rms")
            ax2.semilogy(steps, aphis, "r--", lw=1.2, label="adjacent φ_rms (leakage)")
            ax2.set_ylabel("φ rms")
            ax2.set_xlabel("step")
            ax2.set_title("φ amplitude: ring vs leakage (log scale)")
            ax2.legend(fontsize=8)

            plt.tight_layout()
            fname = "probe_electron_loop.png"
            fig.savefig(fname, dpi=110)
            print(f"\n[plot] {fname}")
            plt.close(fig)
        except ImportError:
            pass


def main():
    run(parser.parse_args())


if __name__ == "__main__":
    main()
