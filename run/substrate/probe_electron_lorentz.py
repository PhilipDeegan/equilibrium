"""
probe_electron_lorentz.py — Lorentz deflection of a moving electron winding.

Initialises the 8-tile toroidal ring with a translational momentum kick
(phase gradient k_x along x) so the winding hops in the +x direction.
Optionally superimposes a background phi circulation representing a uniform
B field in z.  Tracks ring position via phi_rms-weighted COM across the
whole domain, and winding mode amplitude on the current best-fit ring.

Physics: if the ring moves in x with B in z, Lorentz force is in y.
A non-zero y-COM drift confirms the coupling.

Usage:
    PYTHONPATH=$PWD python3 run/substrate/probe_electron_lorentz.py [options]
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
from equilibrium.substrate.lattice import _K_REV, DampedLattice

parser = argparse.ArgumentParser()
parser.add_argument("--R", type=float, default=20.0)
parser.add_argument("--axis", type=str, default="z")
parser.add_argument(
    "--amplitude", type=float, default=1.0, help="winding phi amplitude"
)
parser.add_argument(
    "--kick-px",
    type=float,
    default=0.3,
    help="translational phase gradient k_x (lattice units⁻¹)",
)
parser.add_argument(
    "--lam-ring",
    type=float,
    default=0.0,
    help="lam on ring tiles (0 = pure phi winding, no substrate cost)",
)
parser.add_argument(
    "--B-lam",
    type=float,
    default=0.0,
    help="peak lam of the cylindrical B-field tension shell",
)
parser.add_argument(
    "--B-radius",
    type=float,
    default=8.0,
    help="radius of the B-field lam shell (cells)",
)
parser.add_argument(
    "--B-width",
    type=float,
    default=2.0,
    help="Gaussian half-width of the lam shell (cells)",
)
parser.add_argument("--n-steps", type=int, default=10000)
parser.add_argument("--dt-scale", type=float, default=0.1)
parser.add_argument("--report", type=int, default=500)
parser.add_argument("--sponge", type=float, default=0.20)
parser.add_argument("--damp-max", type=float, default=0.005)
parser.add_argument("--snap-dir", type=str, default="tmp/lorentz_snaps")
parser.add_argument(
    "--snap-r", type=float, default=10.0, help="plot radius for snapshots (cells)"
)


def run(args):
    # ---------------------------------------------------------------------------
    # Build lattice
    # ---------------------------------------------------------------------------
    lat = DampedLattice(
        radius_cells=args.R, sponge_fraction=args.sponge, damp_max=args.damp_max
    )
    pos = lat.pos
    dt = lat.stable_dt() * args.dt_scale
    N = lat.N

    ring_tiles, ring_faces = find_toroidal_ring(lat, axis=args.axis)
    if ring_tiles is None:
        print("ERROR: no toroidal ring found — try larger R")
        sys.exit(1)

    ring_pos = pos[ring_tiles]
    ring_centre = ring_pos.mean(axis=0)
    ring_r = float(np.linalg.norm(ring_pos - ring_centre, axis=1).mean())
    N_ring = len(ring_tiles)
    K_ring = 2 * np.pi / N_ring

    ax_idx = {"x": 0, "y": 1, "z": 2}[args.axis]
    perp = [i for i in range(3) if i != ax_idx]  # [0,1] for axis=z

    print(f"N={N}  R={args.R}  axis={args.axis}  dt={dt:.4e}")
    print(f"Ring tiles : {ring_tiles}")
    print(f"Ring centre: {ring_centre}  ring_r≈{ring_r:.3f}")
    print(f"Winding    : {winding_number(lat, ring_tiles, axis=args.axis):+.3f}")
    print(f"kick_px={args.kick_px:.3f}  B_lam={args.B_lam:.3e}\n")

    # Optional lam on ring tiles
    if args.lam_ring > 0:
        for t in ring_tiles:
            lat.lam[t] = args.lam_ring

    # ---------------------------------------------------------------------------
    # Initialise winding with translational momentum kick
    # Phase: K_ring*(j+0.5) + k_x * x_j  (standard resonant + Doppler tilt)
    # ---------------------------------------------------------------------------
    init_toroidal_winding(
        lat, ring_tiles, ring_faces, amplitude=args.amplitude, frozen=False
    )

    if args.kick_px != 0.0:
        for j, (t, f) in enumerate(zip(ring_tiles, ring_faces)):
            phase = K_ring * (j + 0.5) + args.kick_px * float(pos[t, perp[0]])
            lat.phi[t, f] = args.amplitude * np.cos(phase)
            # maintain antisymmetry on the reverse face
            nb_t = lat.nb[t, f]
            if nb_t >= 0:
                f_rev = int(_K_REV[f])
                lat.phi[nb_t, f_rev] = -lat.phi[t, f]

    # ---------------------------------------------------------------------------
    # Seed B field as a cylindrical lam tension shell around the ring axis.
    # A phi wave passing through this shell sees lower σ (slower propagation)
    # on one side than the other — that asymmetry is the deflecting force.
    # lam(r) = B_lam * exp(-(r - B_radius)^2 / (2*B_width^2))
    # where r = cylindrical radius from the ring axis.
    # ---------------------------------------------------------------------------
    if args.B_lam > 0.0:
        r_cyl = np.sqrt(pos[:, perp[0]] ** 2 + pos[:, perp[1]] ** 2)
        lam_B = args.B_lam * np.exp(
            -((r_cyl - args.B_radius) ** 2) / (2.0 * args.B_width**2)
        )
        lat.lam += lam_B
        print(
            f"B lam shell: peak={lam_B.max():.4f}  "
            f"lam_sum={lat.lam.sum():.1f}  "
            f"σ_min={1/(1+lam_B.max()):.4f}"
        )

    # ---------------------------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------------------------
    _ring_set = set(ring_tiles)
    _ring = np.array(ring_tiles, dtype=int)

    def winding_amplitude():
        """n=1 Fourier mode amplitude on the original ring tiles."""
        total = sum(
            float(lat.phi[t, f])
            * complex(np.cos(2 * np.pi * j / N_ring), np.sin(2 * np.pi * j / N_ring))
            for j, (t, f) in enumerate(zip(ring_tiles, ring_faces))
        )
        return abs(total) / (N_ring / 2)

    def phi_com():
        """phi_rms-weighted COM of the whole domain (x and y)."""
        prms = np.sqrt((lat.phi**2).mean(axis=1))
        w = prms.sum()
        if w < 1e-30:
            return 0.0, 0.0
        cx = float(np.dot(prms, pos[:, perp[0]]) / w)
        cy = float(np.dot(prms, pos[:, perp[1]]) / w)
        return cx, cy

    # ---------------------------------------------------------------------------
    # Snapshot
    # ---------------------------------------------------------------------------
    def save_snap(step, snap_dir, phi_vmax):
        import os

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        prms = np.sqrt((lat.phi**2).mean(axis=1))
        cx, cy = phi_com()

        # Crop to snap_r around the current COM (so plot follows the ring)
        com_pt = np.array([0.0, 0.0, 0.0])
        com_pt[perp[0]] = cx
        com_pt[perp[1]] = cy
        com_pt[ax_idx] = ring_centre[ax_idx]
        dist_com = np.linalg.norm(pos - com_pt, axis=1)
        m = (np.abs(pos[:, ax_idx] - ring_centre[ax_idx]) < 1.5) & (
            dist_com < args.snap_r
        )

        fig, ax = plt.subplots(figsize=(7, 7))
        sc = ax.scatter(
            pos[m, perp[0]],
            pos[m, perp[1]],
            c=prms[m],
            cmap="hot",
            s=15,
            vmin=0,
            vmax=max(float(phi_vmax), 1e-12),
            rasterized=True,
        )
        # original ring tile markers
        ax.scatter(
            ring_pos[:, perp[0]],
            ring_pos[:, perp[1]],
            s=60,
            facecolors="none",
            edgecolors="cyan",
            lw=1.2,
            zorder=5,
            label="orig. ring",
        )
        # COM marker
        ax.plot(cx, cy, "g+", ms=14, mew=2, label=f"COM ({cx:.2f},{cy:.2f})", zorder=6)
        plt.colorbar(sc, ax=ax, label="φ rms")
        ax.set_xlabel(["x", "y", "z"][perp[0]])
        ax.set_ylabel(["x", "y", "z"][perp[1]])
        ax.set_aspect("equal")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_title(
            f"step {step}  |wind|={winding_amplitude():.3f}  "
            f"COM=({cx:.2f},{cy:.2f})"
        )
        plt.tight_layout()
        os.makedirs(snap_dir, exist_ok=True)
        fname = os.path.join(snap_dir, f"snap_{step:07d}.png")
        fig.savefig(fname, dpi=110)
        plt.close(fig)
        print(f"  [snap] {fname}", flush=True)

    # ---------------------------------------------------------------------------
    # Run
    # ---------------------------------------------------------------------------
    print(
        f"{'step':>8}  {'|wind|':>10}  {'COM_x':>10}  {'COM_y':>10}  "
        f"{'phi_ring':>10}  {'global_phi':>12}"
    )

    com_x0, com_y0 = phi_com()
    phi_vmax = max(float(np.sqrt((lat.phi[_ring] ** 2).mean(axis=1)).max()), 1e-12)

    wa0 = winding_amplitude()
    prms0 = float(np.sqrt(np.mean(lat.phi[_ring] ** 2)))
    gp0 = float(np.sqrt(np.mean(lat.phi**2)))
    print(
        f"{'0':>8}  {wa0:+10.6f}  {com_x0:+10.4f}  {com_y0:+10.4f}  "
        f"{prms0:10.4e}  {gp0:12.4e}"
    )
    if args.snap_dir:
        save_snap(0, args.snap_dir, phi_vmax)

    history = []
    for s in range(1, args.n_steps + 1):
        lat.step(dt)
        if s % args.report == 0:
            wa = winding_amplitude()
            cx, cy = phi_com()
            prms = float(np.sqrt(np.mean(lat.phi[_ring] ** 2)))
            gp = float(np.sqrt(np.mean(lat.phi**2)))
            print(
                f"{s:8d}  {wa:+10.6f}  {cx:+10.4f}  {cy:+10.4f}  "
                f"{prms:10.4e}  {gp:12.4e}",
                flush=True,
            )
            history.append((s, wa, cx, cy, prms, gp))
            if args.snap_dir:
                save_snap(s, args.snap_dir, phi_vmax)

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    if len(history) >= 2:
        _, _, cx0, cy0, _, _ = history[0]
        _, _, cxf, cyf, _, _ = history[-1]
        steps_elapsed = history[-1][0] - history[0][0]
        t_elapsed = steps_elapsed * dt
        vx = (cxf - cx0) / t_elapsed if t_elapsed > 0 else 0
        vy = (cyf - cy0) / t_elapsed if t_elapsed > 0 else 0
        print(
            f"\nCOM drift: Δx={cxf-cx0:+.4f}  Δy={cyf-cy0:+.4f}  "
            f"over {steps_elapsed} steps ({t_elapsed:.3f} time units)"
        )
        print(f"Drift velocity: vx={vx:+.4e}  vy={vy:+.4e}")
        if args.kick_px != 0 and args.B_lam != 0 and abs(vx) > 1e-6:
            print(
                f"Lorentz deflection vy/vx = {vy/vx:+.4f}  "
                f"(expected sign: {'+' if args.B_lam > 0 else '-'})"
            )

    # ---------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------
    if history:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            steps = [h[0] for h in history]
            winds = [h[1] for h in history]
            xs = [h[2] for h in history]
            ys = [h[3] for h in history]

            fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
            fig.suptitle(
                f"Lorentz probe  kick_px={args.kick_px}  B_lam={args.B_lam}",
                fontsize=10,
            )

            axes[0].plot(steps, winds, "b-", lw=1.5)
            axes[0].axhline(1.0, color="k", lw=0.5, ls="--")
            axes[0].set_ylabel("|winding amplitude|")
            axes[0].set_title("winding mode amplitude")

            axes[1].plot(steps, xs, "r-", lw=1.5, label="COM x")
            axes[1].axhline(0, color="k", lw=0.5, ls="--")
            axes[1].set_ylabel("COM x")
            axes[1].set_title("x drift (motion direction)")
            axes[1].legend(fontsize=8)

            axes[2].plot(steps, ys, "g-", lw=1.5, label="COM y")
            axes[2].axhline(0, color="k", lw=0.5, ls="--")
            axes[2].set_ylabel("COM y")
            axes[2].set_xlabel("step")
            axes[2].set_title("y deflection (Lorentz direction)")
            axes[2].legend(fontsize=8)

            plt.tight_layout()
            fname = "probe_electron_lorentz.png"
            fig.savefig(fname, dpi=110)
            print(f"\n[plot] {fname}")
            plt.close(fig)
        except ImportError:
            pass


def main():
    run(parser.parse_args())


if __name__ == "__main__":
    main()
