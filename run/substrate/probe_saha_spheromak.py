"""
Probe: single-ring spheromak (proton model) in Saha vacuum.

The small spheromak — ring A only, 8-tile toroidal loop — embedded in a
Saha-equilibrated vacuum (DampedLattice + reciprocal BC).  lam on ring tiles
gives the ring mass; the Saha vacuum provides the thermal reference.

Key questions:
  - Does the ring's lam create a u-gradient (substrate depletion / time dilation)?
  - Does the phi circulation persist in the thermal bath?
  - How does the Saha fraction shift near the ring vs far from it?

Plot on exit: 4 panels — radial u profile, phi+ring diagnostics over time,
2-D z=0 slice at final state, Saha fraction vs radius.

Short run (default): R=12, 2000 steps — quick diagnostic.
Longer run:  --R 25 --n-steps 20000 --report 1000
"""

import argparse
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from equilibrium.substrate.flux_tube import (
    find_toroidal_ring,
    init_toroidal_winding,
    phi_ring_circulation,
)
from equilibrium.substrate.lattice import DampedLattice

U_BG = 1.0
RNG_SEED = 42


# ---------------------------------------------------------------------------
# Reciprocal BC — inline
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


def main():
    # ---------------------------------------------------------------------------
    # CLI
    # ---------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--R", type=float, default=12.0)
    parser.add_argument("--n-steps", type=int, default=2000)
    parser.add_argument("--report", type=int, default=200)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--damp-max", type=float, default=0.003)
    parser.add_argument(
        "--lam",
        type=float,
        default=5.0,
        help="lam on ring tiles (0 = massless flux tube)",
    )
    parser.add_argument(
        "--amp", type=float, default=1.0, help="initial phi amplitude on ring"
    )
    parser.add_argument("--out", type=str, default="saha_spheromak.png")
    args = parser.parse_args()

    # ---------------------------------------------------------------------------
    # Build lattice + spheromak IC
    # ---------------------------------------------------------------------------
    rng = np.random.default_rng(RNG_SEED)
    lat = DampedLattice(radius_cells=args.R, u_bg=U_BG, damp_max=args.damp_max)
    lat.u[:] = U_BG

    dt = lat.stable_dt()
    r_all = np.linalg.norm(lat.pos, axis=1)

    ring_tiles, ring_faces = find_toroidal_ring(lat, axis="z")
    if ring_tiles is None:
        print("ERROR: toroidal ring not found — increase R")
        sys.exit(1)

    init_toroidal_winding(lat, ring_tiles, ring_faces, amplitude=args.amp, frozen=False)
    ring_idx = np.array(ring_tiles)
    ring_r = float(r_all[ring_idx].mean())

    if args.lam > 0:
        lat.lam[ring_idx] = args.lam

    bc = ReciprocalBC(lat, u_target=float(lat.u.sum()), rng=rng)

    print(f"N={lat.N}  R={args.R}  ring={len(ring_tiles)} tiles  r_ring≈{ring_r:.2f}")
    print(f"dt={dt:.4e}  lam={args.lam}  noise={args.noise}  damp={args.damp_max}")
    print(f"n_steps={args.n_steps}  report={args.report}  out={args.out}")

    # ---------------------------------------------------------------------------
    # History arrays
    # ---------------------------------------------------------------------------
    r_edges = np.arange(0.0, args.R * 0.85, 1.0)
    r_mids = 0.5 * (r_edges[:-1] + r_edges[1:])

    snap_steps = sorted({0, args.n_steps // 4, args.n_steps // 2, args.n_steps})
    snap_profiles = {}  # step → radial u profile
    snap_saha = {}  # step → Saha fraction vs radius

    n_rep = args.n_steps // args.report + 1
    t_hist = np.zeros(n_rep)
    circ_h = np.zeros(n_rep)  # phi circulation on ring
    u_ring_h = np.zeros(n_rep)  # mean u at ring tiles
    u_std_h = np.zeros(n_rep)  # domain u std
    u_mean_h = np.zeros(n_rep)  # domain u mean (conservation check)

    def _radial_profile(field):
        out = np.zeros(len(r_mids))
        for i, (r0, r1) in enumerate(zip(r_edges[:-1], r_edges[1:])):
            m = (r_all >= r0) & (r_all < r1)
            out[i] = float(field[m].mean()) if m.any() else float("nan")
        return out

    def _saha_fraction(u, threshold):
        """Fraction of cells above threshold in each radial bin."""
        above = (u > threshold).astype(float)
        return _radial_profile(above)

    ring_centre = lat.pos[ring_idx].mean(axis=0)

    def record(si, step):
        t_hist[si] = step * dt
        circ_h[si] = phi_ring_circulation(lat, list(ring_idx), ring_centre, axis="z")
        u_ring_h[si] = float(lat.u[ring_idx].mean())
        u_std_h[si] = float(lat.u.std())
        u_mean_h[si] = float(lat.u.mean())
        if step in snap_steps:
            snap_profiles[step] = _radial_profile(lat.u.copy())
            thresh = U_BG + 0.5 * max(float(lat.u.std()), 1e-6)
            snap_saha[step] = _saha_fraction(lat.u.copy(), thresh)

    # ---------------------------------------------------------------------------
    # Run
    # ---------------------------------------------------------------------------
    print(
        f"\n{'step':>6}  {'u_mean':>8}  {'u_std':>7}  {'u_ring':>8}  {'phi_circ':>10}"
    )
    print("-" * 50)

    si = 0
    record(si, 0)
    print(
        f"{0:>6}  {u_mean_h[0]:>8.4f}  {u_std_h[0]:>7.4f}  "
        f"{u_ring_h[0]:>8.4f}  {circ_h[0]:>10.4f}"
    )

    for s in range(1, args.n_steps + 1):
        lat.step(dt)
        bc.step(lat, noise_amp=args.noise)
        if s % args.report == 0:
            si += 1
            record(si, s)
            print(
                f"{s:>6}  {u_mean_h[si]:>8.4f}  {u_std_h[si]:>7.4f}  "
                f"{u_ring_h[si]:>8.4f}  {circ_h[si]:>10.4f}"
            )

    # Capture final state for 2-D slice
    u_final = lat.u.copy()
    pos_final = lat.pos.copy()

    # ---------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        f"Spheromak in Saha vacuum  R={args.R}  lam={args.lam}  "
        f"noise={args.noise}  steps={args.n_steps}",
        fontsize=11,
    )
    cmap = plt.get_cmap("viridis")
    snap_list = sorted(snap_profiles.keys())
    colors = [cmap(i / max(len(snap_list) - 1, 1)) for i in range(len(snap_list))]

    # ── Panel 0,0 : radial u profile over time ──────────────────────────────
    ax = axes[0, 0]
    for step, col in zip(snap_list, colors):
        prof = snap_profiles[step]
        ax.plot(r_mids, prof, color=col, label=f"t={step*dt:.2f}", linewidth=1.5)
    ax.axhline(U_BG, color="gray", linestyle="--", linewidth=0.8, label="U_BG")
    ax.axvline(
        ring_r, color="red", linestyle=":", linewidth=0.8, label=f"ring r≈{ring_r:.1f}"
    )
    ax.set_xlabel("r")
    ax.set_ylabel("⟨u⟩ (shell mean)")
    ax.set_title("Radial u profile over time")
    ax.legend(fontsize=7)

    # ── Panel 0,1 : phi circulation + u_ring over time ──────────────────────
    ax = axes[0, 1]
    ax2 = ax.twinx()
    ts = t_hist[: si + 1]
    ax.plot(ts, circ_h[: si + 1], color="steelblue", label="phi circ")
    ax2.plot(ts, u_ring_h[: si + 1], color="coral", label="u_ring", linestyle="--")
    ax2.axhline(U_BG, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("time")
    ax.set_ylabel("phi circulation", color="steelblue")
    ax2.set_ylabel("u at ring cells", color="coral")
    ax.set_title("Ring diagnostics over time")
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, fontsize=8)

    # ── Panel 1,0 : 2-D z=0 slice of u (final state) ───────────────────────
    ax = axes[1, 0]
    z_slice = np.abs(pos_final[:, 2]) < 1.5
    x_sl = pos_final[z_slice, 0]
    y_sl = pos_final[z_slice, 1]
    u_sl = u_final[z_slice]
    sc = ax.scatter(
        x_sl,
        y_sl,
        c=u_sl,
        cmap="RdBu_r",
        s=8,
        vmin=U_BG - 2 * float(u_final.std()),
        vmax=U_BG + 2 * float(u_final.std()),
    )
    plt.colorbar(sc, ax=ax, label="u")
    # mark ring cells in the slice
    ring_in_slice = [i for i in ring_idx if np.abs(pos_final[i, 2]) < 1.5]
    if ring_in_slice:
        rx = pos_final[ring_in_slice, 0]
        ry = pos_final[ring_in_slice, 1]
        ax.scatter(rx, ry, c="gold", s=60, marker="*", zorder=5, label="ring")
        ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("u field  z≈0  (final state)")

    # ── Panel 1,1 : Saha fraction vs radius ─────────────────────────────────
    ax = axes[1, 1]
    for step, col in zip(snap_list, colors):
        frac = snap_saha[step]
        ax.plot(r_mids, frac, color=col, label=f"t={step*dt:.2f}", linewidth=1.5)
    ax.axvline(
        ring_r, color="red", linestyle=":", linewidth=0.8, label=f"ring r≈{ring_r:.1f}"
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
