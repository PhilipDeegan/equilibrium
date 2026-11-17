"""
DST electron (lam-free winding) translation probe -- uniform lattice.

Direct companion to probe_proton_translation.py, same protocol, same lattice
class (equilibrium.substrate.lattice.DampedLattice), same diagnostics, so the
two are a controlled apples-to-apples comparison of a circulation-type
(curl(phi) != 0 globally, an 8-tile toroidal winding) versus a non-circulating
lam concentration -- the only prior electron-vs-proton comparison
(exp/2026/07/10/21_15_electron_translation_gate/ vs
exp/2026/07/11/19_45_proton_knot_translation/) used two DIFFERENT lattice
classes (AMR adaptive cluster vs this uniform lattice) with different IC
constructions, so this closes that gap.

IC: an 8-tile toroidal ring (equilibrium.substrate.flux_tube.find_toroidal_ring)
carrying a CONSTANT circulation flux (not the resonant/travelling-wave mode,
which deliberately oscillates, and not the frozen/fixed_phi mode, which is a
permanently-pinned "inexhaustible battery" -- the same pathology found for the
proton's first two failed ICs), with u=0 everywhere and nothing frozen.
Verified numerically to be an EXACT fixed point of step() alone (max|u|,
max|phi| unchanged over 5000 steps): each ring tile has equal inward/outward
ring flux (divergence-free), and u is uniform (zero) everywhere so phi is
never driven. lam=0 everywhere (electron: no lam, per DST convention).

Unlike the proton IC, step_cross(cross_scale>0) does NOT leave this state
alone even with no kick -- inward ring flux exists (half the ring's edges are
negative), so step_cross fires on the ring's own circulation at rest. This
reproduces the already-known R1 pathology (step_cross erodes a resting
winding) and is the expected baseline, not new; the question here is whether,
on TOP of that baseline erosion, the same external tilt used on the proton
probe produces a sign-consistent (kick-and-coast, same direction) COM drift,
or only erosion/noise as found on the adaptive lattice.

Run:
  PYTHONPATH=$PWD python3 -u run/substrate/probe_electron_translation_uniform.py
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium.substrate.flux_tube import _MIRROR, find_toroidal_ring
from equilibrium.substrate.lattice import DampedLattice

parser = argparse.ArgumentParser()
parser.add_argument("--R", type=float, default=20.0)
parser.add_argument("--amplitude", type=float, default=1.0)
parser.add_argument(
    "--ring-axis", type=str, default="z", help="winding axis of the ring IC"
)
parser.add_argument("--cross-scale", type=float, default=0.01)
parser.add_argument("--gradient", type=float, default=5e-4)
parser.add_argument("--kick-axis", type=str, default="z")
parser.add_argument("--n-base", type=int, default=1000)
parser.add_argument("--n-kick", type=int, default=6000)
parser.add_argument("--n-coast", type=int, default=4000)
parser.add_argument("--report", type=int, default=500)
parser.add_argument("--sponge", type=float, default=0.15)
parser.add_argument("--damp-max", type=float, default=0.003)
parser.add_argument("--n-threads", type=int, default=4)
parser.add_argument(
    "--plot-out", type=str, default="probe_electron_translation_uniform.png"
)
parser.add_argument("--no-plot", action="store_true")
parser.add_argument("--data-out", type=str, default=None)

_AX = {"x": 0, "y": 1, "z": 2}


def weighted_com(pos, w):
    total = w.sum()
    if total < 1e-30:
        return np.zeros(3), 0.0
    return (pos * w[:, None]).sum(axis=0) / total, float(total)


def weighted_spread(pos, w, com):
    total = w.sum()
    if total < 1e-30:
        return 0.0
    d2 = ((pos - com[None, :]) ** 2).sum(axis=1)
    return float(np.sqrt((w * d2).sum() / total))


def build_lattice(args):
    lat = DampedLattice(
        radius_cells=args.R, sponge_fraction=args.sponge, damp_max=args.damp_max
    )
    pos = lat.pos
    r = np.linalg.norm(pos, axis=1)

    ring_tiles, ring_faces = find_toroidal_ring(lat, axis=args.ring_axis)
    if ring_tiles is None:
        raise RuntimeError("no toroidal ring found")

    n = len(ring_tiles)
    for j in range(n):
        i = ring_tiles[j]
        nxt = ring_tiles[(j + 1) % n]
        k = ring_faces[j]
        mk = int(_MIRROR[k])
        lat.phi[i, k] = args.amplitude
        lat.phi[nxt, mk] = -args.amplitude
    # u=0, lam=0 everywhere already (fresh lattice) -- exact fixed point of
    # step() alone; step_cross fires on the ring's own inward flux (expected,
    # matches the known R1 baseline erosion, not a bug).
    return lat, pos, r, ring_tiles


def run(args):
    ax_i = _AX[args.kick_axis]
    lat, pos, r_all, ring_tiles = build_lattice(args)
    dt = lat.stable_dt()
    r_max = float(r_all.max())
    r_inner = r_max * (1.0 - args.sponge)
    battery = (r_all > r_inner - 2.0) & (r_all <= r_inner)
    tilt = args.gradient * pos[:, ax_i]

    def phys_step():
        lat.step(dt, n_threads=args.n_threads)
        if args.cross_scale > 0.0:
            lat.step_cross(dt, args.cross_scale, n_threads=args.n_threads)

    print(
        f"N={lat.N}  dt={dt:.4e}  ring_axis={args.ring_axis}  ring size={len(ring_tiles)}"
    )
    print(f"max|u|={np.abs(lat.u).max():.3e}  max|phi|={np.abs(lat.phi).max():.3e}")

    history = []  # (t, phase, com3, mass, spread, phi_rms, radiated_cum)
    radiated_cum = 0.0

    def weight():
        return np.sqrt((lat.phi**2).mean(axis=1))

    def record(step, phase):
        w = weight()
        com, mass = weighted_com(pos, w)
        spread = weighted_spread(pos, w, com)
        phi_rms = float(np.sqrt(np.mean(lat.phi**2)))
        history.append((step * dt, phase, com, mass, spread, phi_rms, radiated_cum))
        h = history[-1]
        print(
            f"  {step:6d} {phase:>5}  com=({h[2][0]:+8.4f},{h[2][1]:+8.4f},{h[2][2]:+8.4f})"
            f"  mass={h[3]:10.4e}  spread={h[4]:8.4f}  phi_rms={h[5]:9.3e}"
            f"  rad_cum={h[6]:+9.3e}",
            flush=True,
        )

    def advance(n, phase, step0):
        nonlocal radiated_cum
        for s in range(1, n + 1):
            radiated_cum += -lat.bc_flux() * dt
            phys_step()
            if s % args.report == 0:
                record(step0 + s, phase)
        return step0 + n

    record(0, "base")
    step = advance(args.n_base, "base", 0)

    lat.u = lat.u + tilt
    lat.fixed_u[battery] = True
    step = advance(args.n_kick, "kick", step)

    lat.fixed_u[battery] = False
    lat.u = lat.u - tilt
    step = advance(args.n_coast, "coast", step)

    def fit_v(phase):
        rows = [(t, c[ax_i]) for t, p, c, *_ in history if p == phase]
        if len(rows) < 3:
            return 0.0
        rows = rows[len(rows) // 2 :]
        t = np.array([r[0] for r in rows])
        x = np.array([r[1] for r in rows])
        return float(np.polyfit(t, x, 1)[0])

    com_base0 = history[0][2]
    com_end_kick = next(c for t, p, c, *_ in reversed(history) if p == "kick")
    mass0_h, mass_end = history[0][3], history[-1][3]
    spread0, spread_end = history[0][4], history[-1][4]

    res = dict(
        v_base=fit_v("base"),
        v_kick=fit_v("kick"),
        v_coast=fit_v("coast"),
        d_kick=float(com_end_kick[ax_i] - com_base0[ax_i]),
        mass_frac=mass_end / max(mass0_h, 1e-30),
        spread_frac=spread_end / max(spread0, 1e-30),
        radiated_frac=history[-1][6] / max(mass0_h, 1e-30),
        history=history,
        ax_i=ax_i,
        dt=dt,
    )
    print(
        f"\n-> v_base={res['v_base']:+.3e}  v_kick={res['v_kick']:+.3e}"
        f"  v_coast={res['v_coast']:+.3e}  d_kick={res['d_kick']:+.4f}"
        f"  mobility v/g={res['v_kick']/args.gradient:+.3e}"
        f"  mass_frac={res['mass_frac']:.4f}  spread_frac={res['spread_frac']:.4f}"
        f"  radiated/mass0={res['radiated_frac']:+.3e}"
    )
    return res


def main():
    args = parser.parse_args()
    res = run(args)

    if args.data_out:
        np.savez(
            args.data_out,
            history=np.array(
                [
                    (
                        t,
                        {"base": 0, "kick": 1, "coast": 2}[p],
                        *com,
                        mass,
                        spread,
                        phi_rms,
                        rad,
                    )
                    for t, p, com, mass, spread, phi_rms, rad in res["history"]
                ]
            ),
            gradient=args.gradient,
        )
        print(f"data saved -> {args.data_out}")

    if args.no_plot:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    fig.suptitle(
        f"Electron (winding) translation probe -- uniform lattice  R={args.R}"
        f"  g={args.gradient}  cross={args.cross_scale}",
        fontsize=10,
    )
    h = res["history"]
    t = np.array([x[0] for x in h])
    d = np.array([x[2][res["ax_i"]] for x in h])
    mass = np.array([x[3] for x in h])
    spread = np.array([x[4] for x in h])
    axes[0].plot(t, d - d[0], color="C1", lw=1.5)
    axes[1].plot(t, mass / mass[0], color="C1", lw=1.5)
    axes[2].plot(t, spread / spread[0], color="C1", lw=1.5)
    dt0 = res["dt"]
    for ax in axes:
        ax.axvline(args.n_base * dt0, color="k", lw=0.5, ls="--")
        ax.axvline((args.n_base + args.n_kick) * dt0, color="k", lw=0.5, ls="--")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_ylabel("COM displacement (kick axis)")
    axes[0].set_title("Movement")
    axes[1].set_ylabel("mass / mass0")
    axes[1].set_title("Winding-energy conservation")
    axes[2].set_ylabel("spread / spread0")
    axes[2].set_title("Shape rigidity (heat/erosion proxy)")
    axes[2].set_xlabel("time")
    plt.tight_layout()
    plt.savefig(args.plot_out, dpi=130, bbox_inches="tight")
    print(f"plot saved -> {args.plot_out}")


if __name__ == "__main__":
    main()
