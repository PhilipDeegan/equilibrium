"""
DST electron-as-moat translation probe (uniform lattice).

Follow-up to probe_proton_translation.py and probe_electron_translation_uniform.py.
Both of those tested the electron as a phi-CIRCULATION structure (a toroidal
winding) and found it erodes without translating, in sharp contrast to the
lam-carrying, non-circulating proton knot (which translates rigidly). But
CLAUDE.md's canonical DST spec describes the electron differently: "a moat --
a reciprocal deficit of u surrounding the proton knot... analogous to a doping
hole in a semiconductor" -- not a circulation topology at all, but a
u/lam-family object (the same family as the proton), just a DEFICIT instead of
an excess, displaced to an orbital shell instead of sitting at the centre.
project_dst_electron_topology.md already flags free-electron topology as
unresolved; the dipole/winding tests were one candidate, not a settled model.

If the moat is topologically a u/lam-type object rather than a phi-circulation
object, it should belong to the SAME class we just showed translates
coherently (the proton knot), not the class that erodes (the phi ring) --
resolving the apparent electron/proton asymmetry as a modelling artefact
(wrong topology tested), not a real physical asymmetry.

IC: a proton knot (SubstrateLattice.set_winding, chi_0, profile) with an
additional LOCAL DEFICIT carved out of lam in a shell at r_moat (width w,
fractional depth deficit_frac): lam_final = max(0, lam_full - deficit_frac *
exp(-((r-r_moat)/w)^2) * lam_full). u=0 and phi=0 everywhere, exactly as for
the plain proton knot -- this is STILL an exact fixed point of the full
operator set regardless of the lam(r) shape (verified for the plain knot;
carrying a local dip instead of a monotonic profile doesn't change the
argument: Δu ∝ Σφ = 0 since phi=0, Δφ ∝ (u_j-u_i) = 0 since u is uniform,
step_lam_force's pull ∝ max(u_nb-u_i,0) = 0 since u is uniform -- none of
these terms reference the SHAPE of lam(r), only its role in σ(λ), which sets
rate, not equilibrium). The moat itself is tracked as the DEFICIT field
(lam_full - lam_actual), isolating the shell from the underlying knot for the
COM/mass/spread diagnostics -- same three-channel accounting (movement/
heat/light) as the proton probe.

Run:
  PYTHONPATH=$PWD python3 -u tst/substrate/probe_moat_translation.py
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium.substrate.lattice import DampedLattice

parser = argparse.ArgumentParser()
parser.add_argument("--R", type=float, default=20.0)
parser.add_argument("--chi0", type=float, default=0.9, help="knot depletion depth")
parser.add_argument("--profile", type=str, default="1/r2", help="set_winding profile")
parser.add_argument("--r-moat", type=float, default=5.0, help="moat shell radius")
parser.add_argument("--moat-width", type=float, default=1.0)
parser.add_argument("--moat-depth", type=float, default=0.6, help="fractional deficit")
parser.add_argument("--cross-scale", type=float, default=0.01)
parser.add_argument(
    "--force-scale", type=float, default=1e-2, help="step_lam_force scale"
)
parser.add_argument("--gradient", type=float, default=5e-4)
parser.add_argument("--kick-axis", type=str, default="z")
parser.add_argument("--n-base", type=int, default=1000)
parser.add_argument("--n-kick", type=int, default=6000)
parser.add_argument("--n-coast", type=int, default=4000)
parser.add_argument("--report", type=int, default=500)
parser.add_argument("--sponge", type=float, default=0.15)
parser.add_argument("--damp-max", type=float, default=0.003)
parser.add_argument("--n-threads", type=int, default=4)
parser.add_argument("--plot-out", type=str, default="probe_moat_translation.png")
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

    lam_full = lat.lam.copy()
    shell = np.exp(-(((r - args.r_moat) / args.moat_width) ** 2))
    lam_new = np.maximum(0.0, lam_full - args.moat_depth * shell * lam_full)
    lat.lam = lam_new
    deficit0 = lam_full - lam_new
    # u=0, phi=0 already (fresh lattice) -- exact fixed point, no warmup needed.
    return lat, pos, r, origin, lam_full, deficit0


def run(args):
    ax_i = _AX[args.kick_axis]
    lat, pos, r_all, origin, lam_full, deficit0 = build_lattice(args)
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

    com0_check, mass0_check = weighted_com(pos, deficit0)
    print(f"N={lat.N}  dt={dt:.4e}  chi0={args.chi0}  profile={args.profile}")
    print(
        f"moat: r={args.r_moat} w={args.moat_width} depth={args.moat_depth}"
        f"  deficit_mass0={mass0_check:.4f}  com0_r={np.linalg.norm(com0_check):.4f}"
    )
    print(f"max|u|={np.abs(lat.u).max():.3e}  max|phi|={np.abs(lat.phi).max():.3e}")

    history = []  # (t, phase, com3, mass, spread, phi_rms, radiated_cum)
    radiated_cum = 0.0

    def deficit_field():
        return np.maximum(0.0, lam_full - lat.lam)

    def record(step, phase):
        w = deficit_field()
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
        f"Moat (u/lam-deficit) translation probe  R={args.R}  g={args.gradient}"
        f"  cross={args.cross_scale}  force={args.force_scale}",
        fontsize=10,
    )
    h = res["history"]
    t = np.array([x[0] for x in h])
    d = np.array([x[2][res["ax_i"]] for x in h])
    mass = np.array([x[3] for x in h])
    spread = np.array([x[4] for x in h])
    axes[0].plot(t, d - d[0], color="C2", lw=1.5)
    axes[1].plot(t, mass / mass[0], color="C2", lw=1.5)
    axes[2].plot(t, spread / spread[0], color="C2", lw=1.5)
    dt0 = res["dt"]
    for ax in axes:
        ax.axvline(args.n_base * dt0, color="k", lw=0.5, ls="--")
        ax.axvline((args.n_base + args.n_kick) * dt0, color="k", lw=0.5, ls="--")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_ylabel("COM displacement (kick axis)")
    axes[0].set_title("Movement")
    axes[1].set_ylabel("mass / mass0")
    axes[1].set_title("Deficit conservation")
    axes[2].set_ylabel("spread / spread0")
    axes[2].set_title("Shape rigidity (heat/erosion proxy)")
    axes[2].set_xlabel("time")
    plt.tight_layout()
    plt.savefig(args.plot_out, dpi=130, bbox_inches="tight")
    print(f"plot saved -> {args.plot_out}")


if __name__ == "__main__":
    main()
