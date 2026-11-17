"""
probe_lam_differential.py — Seebeck-analogue flux-packing differential probe.

Tests whether a u-differential spontaneously emerges between high-lam poles
and a low-lam flux tube (deadzone channel).

Physics: phi routes to AVOID high-lam regions (light bends around mass in DST).
The flux tube is therefore a low-lam deadzone carved between two high-lam proton
pole knots.  The Seebeck differential is:

  - Poles (high lam): sigma small -> u builds up, cannot drain
  - Tube (lam=0):     sigma=1    -> u turns over fast, stays depleted

Sustained u_pole > u_tube should emerge purely from the sigma contrast at the
pole-tube junction, driven by the wave step.  step_lam can be added to test
whether the lam gradient self-reinforces.

IC:
  - Toroidal phi ring (ring A, frozen) in the xy-plane -- the deadzone path
  - lam=0 on ring tiles (tube/deadzone -- phi routes here)
  - tight axial clusters at +/-z get lam=lam_pole (the proton knots)
  - u = flat background; no fixed_u pins

Diagnostics:
  - u_pole   : mean u over pole tiles (high-lam -- should rise)
  - u_tube   : mean u over ring tiles (lam=0 -- should stay low)
  - u_diff   : u_pole - u_tube  (positive = poles hotter than tube)
  - lam_tube : mean lam on ring tiles (should stay ~0 unless step_lam active)
  - phi_rms  : rms phi (should stay ~constant -- no step_cross)

Usage:
  PYTHONPATH=$PWD python3 run/substrate/probe_lam_differential.py
  PYTHONPATH=$PWD python3 run/substrate/probe_lam_differential.py \\
      --R 30 --lam-pole 2.0 --lam-scale 1e-3 --n-steps 10000
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium.substrate.flux_tube import find_toroidal_ring, init_toroidal_winding
from equilibrium.substrate.lattice import DampedLattice
from tst.substrate import mpl_agg

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--R", type=float, default=20.0, help="lattice radius in cells")
parser.add_argument("--sponge", type=float, default=0.15)
parser.add_argument("--damp-max", type=float, default=0.003)
parser.add_argument(
    "--lam-pole", type=float, default=1.0, help="lam amplitude at pole knot tiles"
)
parser.add_argument(
    "--pole-r", type=float, default=1.0, help="xy-radius cutoff for axial pole cluster"
)
parser.add_argument("--amplitude", type=float, default=1.0, help="ring phi amplitude")
parser.add_argument(
    "--lam-scale", type=float, default=0.0, help="step_lam scale (0 = off)"
)
parser.add_argument(
    "--u-pole",
    type=float,
    default=0.0,
    help="pinned u value at pole tiles (depleted sink)",
)
parser.add_argument("--u-bg", type=float, default=1.0, help="background substrate u")
parser.add_argument(
    "--pin-poles",
    action="store_true",
    help="hold u fixed at pole tiles (sink) throughout the run",
)
parser.add_argument("--n-steps", type=int, default=50000)
parser.add_argument("--report", type=int, default=2000)
parser.add_argument("--plot-out", type=str, default="tmp/probe_lam_diff.png")
parser.add_argument("--no-plot", action="store_true")


# ---------------------------------------------------------------------------
def pole_tiles(pos, ring_tiles, pole_r):
    """
    Tight axial clusters at +/-z -- the proton knot locations.

    Only tiles within pole_r of the z-axis, above/below the ring plane.
    Ring tiles excluded (they are the deadzone tube, not the knots).
    """
    r_xy = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
    on_axis = r_xy < pole_r
    z = pos[:, 2]
    ring_set = set(ring_tiles)

    pos_z = [int(i) for i in np.where(on_axis & (z > 0))[0] if i not in ring_set]
    neg_z = [int(i) for i in np.where(on_axis & (z < 0))[0] if i not in ring_set]
    return pos_z, neg_z


def main():
    args = parser.parse_args()

    lat = DampedLattice(
        radius_cells=args.R,
        sponge_fraction=args.sponge,
        damp_max=args.damp_max,
        u_bg=args.u_bg,
    )
    pos = lat.pos
    dt = lat.stable_dt()

    print(
        f"N={lat.N}  R={args.R}  dt={dt:.4e}  lam_pole={args.lam_pole}  lam_scale={args.lam_scale}"
    )

    # --- IC: toroidal ring phi (frozen -- no step_cross) ---
    A_tiles, A_faces = find_toroidal_ring(lat, axis="z")
    if A_tiles is None:
        print("ERROR: no toroidal ring found -- try a larger R")
        sys.exit(1)

    init_toroidal_winding(lat, A_tiles, A_faces, amplitude=args.amplitude, frozen=True)
    ring_r = float(np.linalg.norm(pos[A_tiles], axis=1).mean())
    print(f"Ring A: {len(A_tiles)} tiles  r~{ring_r:.2f}  lam=0 (deadzone)")

    # --- IC: lam on pole tiles (proton knots -- high lam, slow sigma) ---
    pos_poles, neg_poles = pole_tiles(pos, A_tiles, args.pole_r)
    if not pos_poles or not neg_poles:
        print(
            f"WARNING: pole cluster empty -- try increasing --pole-r (currently {args.pole_r})"
        )
    print(
        f"Pole clusters: +z={len(pos_poles)} tiles  -z={len(neg_poles)} tiles  lam={args.lam_pole}"
    )

    for idx in pos_poles + neg_poles:
        lat.lam[idx] = args.lam_pole
    # ring tiles stay lam=0 (deadzone)

    # --- IC: background u; poles depleted and pinned as sinks ---
    lat.u[:] = args.u_bg
    if args.pin_poles:
        for idx in pos_poles + neg_poles:
            lat.u[idx] = args.u_pole
            lat.fixed_u[idx] = True
        print(f"Poles pinned at u={args.u_pole} (sink)  background u={args.u_bg}")

    print(f"phi_rms_init = {float(np.sqrt((lat.phi**2).mean())):.4e}")
    print(f"lam_sum_init = {lat.lam.sum():.4f}")

    # --- Diagnostics setup ---
    all_pole_idx = np.array(pos_poles + neg_poles, dtype=int)
    tube_idx = np.array(A_tiles, dtype=int)
    lam_active = args.lam_scale > 0.0

    times = []
    u_pole_h = []
    u_tube_h = []
    u_diff_h = []
    lam_tube_h = []
    phi_rms_h = []

    def record(step):
        t = step * dt
        up = float(lat.u[all_pole_idx].mean()) if len(all_pole_idx) else 0.0
        ut = float(lat.u[tube_idx].mean())
        times.append(t)
        u_pole_h.append(up)
        u_tube_h.append(ut)
        u_diff_h.append(up - ut)  # positive = poles hotter than tube
        lam_tube_h.append(float(lat.lam[tube_idx].mean()))
        phi_rms_h.append(float(np.sqrt((lat.phi**2).mean())))

    record(0)

    # --- Main loop: wave step + optional step_lam; NO step_cross ---
    for step in range(1, args.n_steps + 1):
        lat.step(dt)
        if lam_active:
            lat.step_lam(dt, scale=args.lam_scale)

        if step % args.report == 0:
            record(step)
            ud = u_diff_h[-1]
            print(
                f"  step {step:6d}  u_pole={u_pole_h[-1]:.4f}  u_tube={u_tube_h[-1]:.4f}"
                f"  Du(pole-tube)={ud:+.4f}  lam_tube={lam_tube_h[-1]:.4f}"
                f"  phi_rms={phi_rms_h[-1]:.4e}"
            )

    if args.no_plot:
        return

    # --- Plot ---
    import os

    os.makedirs(os.path.dirname(args.plot_out) or ".", exist_ok=True)

    lam_label = f"lam_scale={args.lam_scale}" if lam_active else "step_lam off"
    pin_label = (
        f"  poles pinned u={args.u_pole} bg={args.u_bg}" if args.pin_poles else ""
    )
    plt = mpl_agg()
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    ax = axes[0]
    ax.plot(times, u_pole_h, label="u_pole (high lam)", color="tomato")
    ax.plot(times, u_tube_h, label="u_tube (deadzone)", color="steelblue")
    ax.set_ylabel("mean u")
    ax.legend(fontsize=9)
    ax.set_title(
        f"Seebeck-analogue probe  R={args.R}  lam_pole={args.lam_pole}"
        f"  amp={args.amplitude}  {lam_label}{pin_label}"
    )

    ax = axes[1]
    ax.plot(times, u_diff_h, color="darkorchid")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("Du = u_pole - u_tube")

    ax = axes[2]
    ax.plot(times, phi_rms_h, color="steelblue", label="phi_rms")
    ax2 = ax.twinx()
    ax2.plot(times, lam_tube_h, color="darkorange", lw=0.8, label="lam_tube (mean)")
    ax.set_ylabel("phi_rms")
    ax2.set_ylabel("lam_tube mean", color="darkorange")
    ax.set_xlabel("t (lattice units)")
    ax.legend(fontsize=9, loc="upper left")
    ax2.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    plt.savefig(args.plot_out, dpi=130, bbox_inches="tight")
    print(f"plot saved -> {args.plot_out}")


if __name__ == "__main__":
    main()
