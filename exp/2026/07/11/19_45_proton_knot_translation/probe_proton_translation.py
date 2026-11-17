"""
DST proton-knot translation probe (uniform lattice).

Companion to tst/substrate/adaptive/probe_electron_translation.py, testing the
same question -- does external pressure asymmetry convert into COHERENT
displacement, or only into erosion? -- for a lam-carrying "open knot" (a
proton-like mass concentration) instead of the lam-free electron double-ring
winding.

Electron result (exp/2026/07/10/21_15_electron_translation_gate): cross=0 is
an exact null (superposition), cross>0 gives erosion without transport (COM
wander sub-cell, sign-inconsistent, winding decays). The current operator set
had no channel turning pressure into coherent motion for a CIRCULATION-type
(curl != 0 globally, i.e. nonzero winding number) structure.

Hypothesis under test here: a proton is not a winding -- it is a lam
concentration with NO seeded circulation (curl(phi) = 0 everywhere, phi is
zero, not merely radial). The question is whether THIS topological class --
unlike the electron's circulation type -- already shows collective, rigid
translation using only the existing state (u, phi, lam, sigma(lambda)) and
existing operators (step, step_cross, step_lam_force), or whether it too only
erodes -- which would mean the "moves and resists as one" relationship is not
yet encoded anywhere and a new quantity is needed.

IC: uniform u=0 + phi=0 + a lam(r) profile from SubstrateLattice.set_winding.
This is an EXACT fixed point of the full operator set for ANY lam(r) profile
(verified: max|u|, max|phi| stay at machine 0 over 5000 steps with no kick) --
sigma(lambda) only affects the RATE of departure from equilibrium, not
whether zero-flux/uniform-u is an equilibrium, so no warmup or point-spike
seeding is needed (earlier attempts at both -- an unpinned point spike, which
rang in an undamped whole-domain breathing mode, and a permanently pinned
point spike, which pumped energy in forever with nothing to cap the rate --
were the wrong kind of IC entirely, not a tuning problem). This mirrors
exactly how the electron probe used an exact fixed point (uniform u +
divergence-free phi) rather than a shocked IC.

Also tracked: the J = curl(B) analogy -- an external push that cannot be
absorbed as curl has to go SOMEWHERE. Three channels tracked (as coherence
PATTERNS of the same u/phi/lam state, not separate physical quantities):
  movement  -- COM displacement along the kick axis (coherent transport)
  heat      -- local erosion/smearing: growth of the lam profile's weighted
               spread around its own COM (irreversible, undirected)
  light     -- energy actually leaving through the sponge boundary
               (DampedLattice.bc_flux(), integrated over the run)

Run:
  PYTHONPATH=$PWD python3 -u tst/substrate/probe_proton_translation.py
  PYTHONPATH=$PWD python3 -u tst/substrate/probe_proton_translation.py \
      --R 15 --n-base 100 --n-kick 400 --n-coast 300 --report 50
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium.substrate.lattice import DampedLattice

parser = argparse.ArgumentParser()
parser.add_argument("--R", type=float, default=20.0)
parser.add_argument("--chi0", type=float, default=0.9, help="lam depletion depth")
parser.add_argument("--profile", type=str, default="1/r2", help="set_winding profile")
parser.add_argument("--cross-scale", type=float, default=0.01)
parser.add_argument(
    "--force-scale", type=float, default=1e-2, help="step_lam_force scale"
)
parser.add_argument("--gradient", type=float, default=5e-4)
parser.add_argument("--kick-axis", type=str, default="z")
parser.add_argument("--n-base", type=int, default=500)
parser.add_argument("--n-kick", type=int, default=3000)
parser.add_argument("--n-coast", type=int, default=2000)
parser.add_argument("--report", type=int, default=250)
parser.add_argument("--sponge", type=float, default=0.15)
parser.add_argument("--damp-max", type=float, default=0.003)
parser.add_argument("--n-threads", type=int, default=4)
parser.add_argument("--plot-out", type=str, default="probe_proton_translation.png")
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
    origin = int(np.argmin(r))
    lat.set_winding(lat.tiles[origin], chi_0=args.chi0, profile=args.profile)
    # u=0, phi=0 already (fresh lattice) -- exact fixed point, no warmup needed.
    return lat, pos, r, origin


def run(args):
    ax_i = _AX[args.kick_axis]
    lat, pos, r_all, origin = build_lattice(args)
    dt = lat.stable_dt()
    r_max = float(r_all.max())
    r_inner = r_max * (1.0 - args.sponge)
    battery = (r_all > r_inner - 2.0) & (r_all <= r_inner)
    tilt = args.gradient * pos[:, ax_i]

    def phys_step():
        lat.step(dt, n_threads=args.n_threads)
        if args.cross_scale > 0.0:
            lat.step_cross(dt, args.cross_scale, n_threads=args.n_threads)
        if args.force_scale > 0.0:
            lat.step_lam_force(dt, args.force_scale)

    print(f"N={lat.N}  dt={dt:.4e}  chi0={args.chi0}  profile={args.profile}")
    print(
        f"max|u|={np.abs(lat.u).max():.3e}  max|phi|={np.abs(lat.phi).max():.3e}"
        f"  lam_sum={lat.lam.sum():.4f}"
    )

    history = []  # (t, phase, com3, mass, spread, phi_rms, radiated_cum)
    radiated_cum = 0.0

    def record(step, phase):
        w = lat.lam
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
        f"Proton-knot translation probe  R={args.R}  g={args.gradient}"
        f"  cross={args.cross_scale}  force={args.force_scale}",
        fontsize=10,
    )
    h = res["history"]
    t = np.array([x[0] for x in h])
    d = np.array([x[2][res["ax_i"]] for x in h])
    mass = np.array([x[3] for x in h])
    spread = np.array([x[4] for x in h])
    axes[0].plot(t, d - d[0], color="C0", lw=1.5)
    axes[1].plot(t, mass / mass[0], color="C0", lw=1.5)
    axes[2].plot(t, spread / spread[0], color="C0", lw=1.5)
    dt0 = res["dt"]
    for ax in axes:
        ax.axvline(args.n_base * dt0, color="k", lw=0.5, ls="--")
        ax.axvline((args.n_base + args.n_kick) * dt0, color="k", lw=0.5, ls="--")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_ylabel("COM displacement (kick axis)")
    axes[0].set_title("Movement")
    axes[1].set_ylabel("mass / mass0")
    axes[1].set_title("Mass conservation")
    axes[2].set_ylabel("spread / spread0")
    axes[2].set_title("Shape rigidity (heat/erosion proxy)")
    axes[2].set_xlabel("time")
    plt.tight_layout()
    plt.savefig(args.plot_out, dpi=130, bbox_inches="tight")
    print(f"plot saved -> {args.plot_out}")


if __name__ == "__main__":
    main()
