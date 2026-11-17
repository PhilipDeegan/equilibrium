"""
Probe: 6-spheromak cross configuration in Saha vacuum.

Six single-ring spheromaks placed symmetrically along ±x, ±y, ±z at distance
OFFSET from the centre.  Each ring axis is aligned with its displacement axis
(+x ring winds around x, lies in the y-z plane, etc.).

Questions:
  - Do the 6 moats interact or remain independent?
  - Does the centre (origin, equidistant from all 6 rings) show elevated or
    depleted u?
  - Does the Saha fraction profile develop a characteristic multi-ring shape?
  - Do the phi circulations of the 6 rings stay independent?

Run:
  PYTHONPATH=$PWD python3 run/substrate/probe_saha_cross.py
  PYTHONPATH=$PWD python3 run/substrate/probe_saha_cross.py --R 30 --n-steps 20000
"""

import argparse
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from equilibrium.substrate.flux_tube import (
    _TOROIDAL_PATTERNS,
    _trace_loop,
    init_toroidal_winding,
    phi_ring_circulation,
)
from equilibrium.substrate.lattice import DampedLattice

U_BG = 1.0
RNG_SEED = 42

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class ReciprocalBC:
    def __init__(self, lat, u_target, rng):
        nb = lat.nb
        boundary = np.any(nb < 0, axis=1) & ~lat.fixed_u
        self._idx = np.where(boundary)[0]
        self._target = float(u_target)
        self._rng = rng

    def step(self, lat, noise_amp=0.0):
        deficit = self._target - float(lat.u.sum())
        n = len(self._idx)
        if n == 0:
            return
        w = self._rng.exponential(size=n)
        w /= w.sum()
        inj = deficit * w
        if noise_amp > 0:
            noise = noise_amp * self._rng.standard_normal(size=n)
            noise -= noise.mean()
            inj += noise
        lat.u[self._idx] += inj


def find_ring_near(lat, target_pos, axis):
    """Find a toroidal ring near target_pos with the given winding axis."""
    face_seq, _ = _TOROIDAL_PATTERNS[axis]
    dists = np.linalg.norm(lat.pos - np.array(target_pos), axis=1)
    for start in np.argsort(dists)[:120]:
        tiles, faces = _trace_loop(lat, int(start), face_seq)
        if tiles is not None:
            return tiles, faces
    return None, None


def radial_profile(lat_pos, values, r_all, r_edges):
    out = np.zeros(len(r_edges) - 1)
    for i, (r0, r1) in enumerate(zip(r_edges[:-1], r_edges[1:])):
        m = (r_all >= r0) & (r_all < r1)
        out[i] = float(values[m].mean()) if m.any() else float("nan")
    return out


def main():
    # ---------------------------------------------------------------------------
    # CLI
    # ---------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--R", type=float, default=20.0)
    parser.add_argument(
        "--offset", type=float, default=6.0, help="distance from centre to each ring"
    )
    parser.add_argument("--n-steps", type=int, default=8000)
    parser.add_argument("--report", type=int, default=800)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--damp-max", type=float, default=0.003)
    parser.add_argument("--lam", type=float, default=5.0)
    parser.add_argument("--amp", type=float, default=1.0)
    parser.add_argument("--out", type=str, default="saha_cross.png")
    args = parser.parse_args()

    # ---------------------------------------------------------------------------
    # Build lattice
    # ---------------------------------------------------------------------------
    rng = np.random.default_rng(RNG_SEED)
    lat = DampedLattice(radius_cells=args.R, u_bg=U_BG, damp_max=args.damp_max)
    lat.u[:] = U_BG
    dt = lat.stable_dt()
    r_all = np.linalg.norm(lat.pos, axis=1)

    # 6-ring cross: ±x (axis=x), ±y (axis=y), ±z (axis=z)
    d = args.offset
    CONFIGS = [
        (np.array([+d, 0, 0]), "x"),
        (np.array([-d, 0, 0]), "x"),
        (np.array([0, +d, 0]), "y"),
        (np.array([0, -d, 0]), "y"),
        (np.array([0, 0, +d]), "z"),
        (np.array([0, 0, -d]), "z"),
    ]

    print(f"N={lat.N}  R={args.R}  offset={args.offset}  dt={dt:.4e}")
    print(f"lam={args.lam}  noise={args.noise}  damp={args.damp_max}")

    rings = []  # list of (tiles_array, faces_list, target_pos, axis, r_mean)
    used_tiles = set()

    for target, axis in CONFIGS:
        tiles, faces = find_ring_near(lat, target, axis)
        if tiles is None:
            print(f"  WARN: no ring found near {target} axis={axis}")
            continue
        tile_set = set(tiles)
        if tile_set & used_tiles:
            print(f"  WARN: ring near {target} overlaps existing ring — skipped")
            continue
        used_tiles |= tile_set
        r_mean = float(np.linalg.norm(lat.pos[tiles].mean(axis=0)))
        print(
            f"  ring @ {np.array(target).round(1)}  axis={axis}  "
            f"actual_centre={lat.pos[tiles].mean(axis=0).round(2)}  r≈{r_mean:.2f}"
        )
        init_toroidal_winding(lat, tiles, faces, amplitude=args.amp, frozen=False)
        if args.lam > 0:
            lat.lam[np.array(tiles)] = args.lam
        rings.append((np.array(tiles), faces, target, axis, r_mean))

    if not rings:
        print("ERROR: no rings found — try increasing R or reducing offset")
        sys.exit(1)

    print(f"\n{len(rings)}/6 rings placed  total ring tiles={len(used_tiles)}")

    bc = ReciprocalBC(lat, u_target=float(lat.u.sum()), rng=rng)

    # ---------------------------------------------------------------------------
    # History
    # ---------------------------------------------------------------------------
    r_edges = np.arange(0.0, args.R * 0.85, 1.0)
    r_mids = 0.5 * (r_edges[:-1] + r_edges[1:])
    snap_steps = sorted({0, args.n_steps // 4, args.n_steps // 2, args.n_steps})
    snap_profiles = {}
    snap_saha = {}

    n_rep = args.n_steps // args.report + 1
    t_hist = np.zeros(n_rep)
    u_std_h = np.zeros(n_rep)
    u_mean_h = np.zeros(n_rep)
    circ_h = np.zeros((n_rep, len(rings)))
    u_ring_h = np.zeros((n_rep, len(rings)))

    def record(si, step):
        t_hist[si] = step * dt
        u_std_h[si] = float(lat.u.std())
        u_mean_h[si] = float(lat.u.mean())
        for ri, (tiles, _, _, axis, _) in enumerate(rings):
            ring_centre = lat.pos[tiles].mean(axis=0)
            circ_h[si, ri] = phi_ring_circulation(
                lat, list(tiles), ring_centre, axis=axis
            )
            u_ring_h[si, ri] = float(lat.u[tiles].mean())
        if step in snap_steps:
            snap_profiles[step] = radial_profile(lat.pos, lat.u.copy(), r_all, r_edges)
            thresh = U_BG + 0.5 * max(float(lat.u.std()), 1e-6)
            snap_saha[step] = radial_profile(
                lat.pos, (lat.u > thresh).astype(float), r_all, r_edges
            )

    # ---------------------------------------------------------------------------
    # Run
    # ---------------------------------------------------------------------------
    hdr = "  ".join(f"circ{i+1:d}" for i in range(len(rings)))
    print(f"\n{'step':>6}  {'u_mean':>8}  {'u_std':>7}  {hdr}")
    print("-" * (30 + 10 * len(rings)))

    si = 0
    record(si, 0)
    circs = "  ".join(f"{circ_h[0, i]:>8.3f}" for i in range(len(rings)))
    print(f"{0:>6}  {u_mean_h[0]:>8.4f}  {u_std_h[0]:>7.4f}  {circs}")

    for s in range(1, args.n_steps + 1):
        lat.step(dt)
        bc.step(lat, noise_amp=args.noise)
        if s % args.report == 0:
            si += 1
            record(si, s)
            circs = "  ".join(f"{circ_h[si, i]:>8.3f}" for i in range(len(rings)))
            print(f"{s:>6}  {u_mean_h[si]:>8.4f}  {u_std_h[si]:>7.4f}  {circs}")

    u_final = lat.u.copy()
    pos_final = lat.pos.copy()

    # ---------------------------------------------------------------------------
    # Plot — 2×2
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        f"6-spheromak cross  R={args.R}  offset={args.offset}  "
        f"lam={args.lam}  noise={args.noise}  steps={args.n_steps}",
        fontsize=11,
    )
    cmap = plt.get_cmap("viridis")
    snaps = sorted(snap_profiles.keys())
    colors = [cmap(i / max(len(snaps) - 1, 1)) for i in range(len(snaps))]

    # ── 0,0 : radial u profile ───────────────────────────────────────────────
    ax = axes[0, 0]
    for step, col in zip(snaps, colors):
        ax.plot(
            r_mids,
            snap_profiles[step],
            color=col,
            label=f"t={step*dt:.1f}",
            linewidth=1.5,
        )
    ax.axhline(U_BG, color="gray", linestyle="--", linewidth=0.8, label="U_BG")
    ax.axvline(
        args.offset,
        color="red",
        linestyle=":",
        linewidth=0.9,
        label=f"rings r≈{args.offset:.0f}",
    )
    ax.set_xlabel("r")
    ax.set_ylabel("⟨u⟩ (shell mean)")
    ax.set_title("Radial u profile")
    ax.legend(fontsize=7)

    # ── 0,1 : phi circulation per ring ───────────────────────────────────────
    ax = axes[0, 1]
    ax2 = ax.twinx()
    ts = t_hist[: si + 1]
    lcolors = ["steelblue", "steelblue", "seagreen", "seagreen", "coral", "coral"]
    lstyles = ["-", "--", "-", "--", "-", "--"]
    for ri in range(len(rings)):
        ax.plot(
            ts,
            circ_h[: si + 1, ri],
            color=lcolors[ri],
            linestyle=lstyles[ri],
            linewidth=1.2,
            label=f"ring{ri+1}",
        )
    ax.set_ylabel("phi circulation")
    ax.set_xlabel("time")
    ax.set_title("Phi circulation — all 6 rings")
    ax.legend(fontsize=7, ncol=2)

    # mean u_ring across all rings on right axis
    mean_u_ring = u_ring_h[: si + 1].mean(axis=1)
    ax2.plot(
        ts, mean_u_ring, color="gold", linewidth=1.5, linestyle=":", label="⟨u_ring⟩"
    )
    ax2.axhline(U_BG, color="gray", linewidth=0.6, linestyle=":")
    ax2.set_ylabel("mean u at ring cells", color="gold")

    # ── 1,0 : 2-D z≈0 slice ─────────────────────────────────────────────────
    ax = axes[1, 0]
    z_sl = np.abs(pos_final[:, 2]) < 1.5
    x_sl = pos_final[z_sl, 0]
    y_sl = pos_final[z_sl, 1]
    u_sl = u_final[z_sl]
    vd = 2 * float(u_final.std())
    sc = ax.scatter(
        x_sl, y_sl, c=u_sl, cmap="RdBu_r", s=5, vmin=U_BG - vd, vmax=U_BG + vd
    )
    plt.colorbar(sc, ax=ax, label="u")
    # mark ring tiles in slice — different colour per ring
    ring_colors = ["gold", "orange", "lime", "cyan", "magenta", "white"]
    for ri, (tiles, _, _, _, _) in enumerate(rings):
        in_sl = [i for i in tiles if np.abs(pos_final[i, 2]) < 1.5]
        if in_sl:
            rx = pos_final[in_sl, 0]
            ry = pos_final[in_sl, 1]
            ax.scatter(
                rx, ry, c=ring_colors[ri % len(ring_colors)], s=50, marker="*", zorder=5
            )
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("u field  z≈0  (final state)")

    # ── 1,1 : Saha fraction vs radius ────────────────────────────────────────
    ax = axes[1, 1]
    for step, col in zip(snaps, colors):
        ax.plot(
            r_mids, snap_saha[step], color=col, label=f"t={step*dt:.1f}", linewidth=1.5
        )
    ax.axvline(
        args.offset,
        color="red",
        linestyle=":",
        linewidth=0.9,
        label=f"rings r≈{args.offset:.0f}",
    )
    ax.set_xlabel("r")
    ax.set_ylabel("fraction  u > U_BG + ½σ")
    ax.set_title("Saha fraction vs radius")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"\nPlot saved → {args.out}")


if __name__ == "__main__":
    main()
