"""
DST electron guide-field probe (uniform lattice) -- boundary/waveguide
confinement diagnostic, complementary to the bulk Klein-Gordon mass-term
test in probe_electron_trap.py.

Context (doc/dst_next.md and project_dst_electron_topology.md, 2026-07-12
discussion): two candidate confinement mechanisms were identified for a
free "ball of charge" u-deficit that cannot sit statically on its own
(dphi_ik = -dt*sigma_face*W*(u_k-u_i) has no collective escape hatch the
way the u-update does, so any nonuniform u free-disperses):

  (a) BULK -- Klein-Gordon mass term added uniformly everywhere (a
      plasma-frequency-like cutoff). Built in probe_electron_trap.py;
      inconclusive so far (mass=0.05 predicts a confinement length ~1/mass
      comparable to the whole domain, so it was never distinguishable from
      the sponge boundary swallowing the wave).
  (b) BOUNDARY/GUIDE FIELD (this probe) -- a real, spatially-localised
      sigma(lam) step or shell somewhere INSIDE the domain (not the sponge,
      which is built to absorb -- this is meant to REFLECT), acting like a
      fibre-optic core/cladding boundary (total internal reflection) or a
      waveguide wall. Philip's framing: probably more physically natural
      than a bulk mass term, and the natural stepping stone toward
      self-trapping (lam sourced by the structure's own presence, soliton-
      style) -- but that self-consistent version is a stretch goal. This
      probe starts with the EXTERNALLY IMPOSED version first, same
      ion-trap-analogy discipline as the mass-term probe: a deliberately
      artificial diagnostic apparatus, not a claim about the free
      electron's own intrinsic physics.

Mechanism: sigma = 1/(1+lam) already enters the phi update as a
FACE-AVERAGED value sigma_face = (sigma_i + sigma_nb)/2
(equilibrium/substrate/lattice.py SubstrateLattice.step()) -- so a static
lam shell directly changes the local wave impedance at exactly the face
where two differing-lam cells meet, with no new term needed in the update
rule. lam is never evolved by step()/step_cpp() (only step_lam() touches
it, and this probe never calls it), so imposing a lam profile once at IC
time behaves as a genuine static boundary condition -- the same trick
already used for the "B field" lam shell in probe_electron_lorentz.py.

IC: symmetric negative Gaussian u-deficit at the origin (r_wall > u_width,
so the deficit starts well inside the wall), phi=0, mass=0 (bulk KG term
OFF by default -- isolates the boundary mechanism; can be combined with
--mass later once each is understood separately). lam=0 inside the wall,
a Gaussian lam shell of height lam_wall / width wall_width centred at
r_wall, lam=0 again outside the shell out to the sponge. r_wall must sit
well inside r_inner=(1-sponge_fraction)*R so the wall's effect is not
confounded with the sponge's own absorption -- the defaults below leave a
~15-cell lam=0 buffer between wall and sponge at R=30.

Diagnostics: same spread(t)/mass(t)/energy(t) as probe_electron_trap.py,
plus frac_inside(t) = deficit-weighted mass still at r < r_wall / total
deficit-weighted mass. A genuine TIR-style trap should hold frac_inside
near 1; a leaky/no-op wall should see it decay toward whatever the free
(no-wall) spreading case gives.

Options (mark [x] when implemented AND run with a recorded result):
- [ ] Baseline: --lam-wall 0 (wall geometry present but inert) should
      reproduce probe_electron_trap.py's free-spreading result -- confirms
      this probe's own plumbing before trusting a nonzero-wall result.
- [ ] Single wall height/width at the defaults below: does frac_inside
      plateau (reflection) or decay (leakage/transmission)?
- [ ] lam_wall sweep at fixed geometry: is there a critical wall height
      (TIR-critical-angle analogue) below which the wall is fully
      transparent and above which it fully reflects, or is reflection
      always partial (steady erosion, no true confinement)?
- [ ] wall_width sweep: does a sharper (thinner) wall reflect better or
      worse than a smooth ramp, at fixed lam_wall?
- [ ] Compare a genuinely confining wall's plateaued spread against the
      mass-term probe's plateau at matched confinement length, if/when
      the mass-term probe produces one -- do the two mechanisms agree on
      what a confined electron's internal profile looks like?
- [ ] Self-trapping / soliton stretch goal: replace the externally-imposed
      lam_wall with a lam profile sourced by the deficit's own (small,
      mass-ratio-scaled) lam -- not started, only meaningful once the
      externally-imposed version above shows genuine reflection at all.

Run:
  PYTHONPATH=$PWD python3 -u run/substrate/probe_electron_guide_field.py
  PYTHONPATH=$PWD python3 -u run/substrate/probe_electron_guide_field.py --lam-wall 0
      (baseline -- wall geometry present but inert, should match the free-
      spreading case)
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium.substrate.lattice import DampedLattice

parser = argparse.ArgumentParser()
parser.add_argument("--R", type=float, default=30.0)
parser.add_argument("--u-depth", type=float, default=1.0, help="initial deficit depth")
parser.add_argument(
    "--u-width", type=float, default=2.0, help="initial deficit Gaussian width"
)
parser.add_argument(
    "--r-wall", type=float, default=10.0, help="wall shell centre radius"
)
parser.add_argument(
    "--wall-width", type=float, default=1.5, help="wall shell Gaussian width"
)
parser.add_argument(
    "--lam-wall",
    type=float,
    default=5.0,
    help="peak lam at the wall shell (0 = no wall, baseline free-spreading case)",
)
parser.add_argument(
    "--mass",
    type=float,
    default=0.0,
    help="bulk Klein-Gordon mass term, OFF by default to isolate the boundary "
    "mechanism (see probe_electron_trap.py for the bulk-only test)",
)
parser.add_argument("--n-steps", type=int, default=8000)
parser.add_argument("--report", type=int, default=250)
parser.add_argument("--sponge", type=float, default=0.15)
parser.add_argument("--damp-max", type=float, default=0.003)
parser.add_argument("--n-threads", type=int, default=4)
parser.add_argument("--plot-out", type=str, default="probe_electron_guide_field.png")
parser.add_argument("--no-plot", action="store_true")
parser.add_argument("--data-out", type=str, default=None)


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


def field_energy(lat, mass=0.0):
    """Discrete field energy: 0.5*sum(u^2)*(1+mass^2) + 0.5*sum(W*phi^2).

    Same form as probe_electron_trap.py -- the lam-shell wall only changes
    sigma (propagation rate), not the stored-energy bookkeeping, so this
    conservation check is directly comparable between the two probes.
    """
    valid = lat._valid
    from equilibrium.substrate.lattice import _W

    phi_e = 0.5 * float((np.where(valid, _W[None, :], 0.0) * lat.phi**2).sum())
    u_e = 0.5 * (1.0 + mass**2) * float((lat.u**2).sum())
    return u_e, phi_e


def build_lattice(args):
    lat = DampedLattice(
        radius_cells=args.R, sponge_fraction=args.sponge, damp_max=args.damp_max
    )
    pos = lat.pos
    r = np.linalg.norm(pos, axis=1)
    # symmetric negative Gaussian deficit -- the "ball of charge"
    lat.u = -args.u_depth * np.exp(-((r / args.u_width) ** 2))
    # static lam shell -- imposed once, never touched by step_lam(), so it
    # behaves as a genuine boundary condition for the rest of the run
    if args.lam_wall > 0:
        lat.lam = args.lam_wall * np.exp(
            -((r - args.r_wall) ** 2) / (2.0 * args.wall_width**2)
        )
        sigma_wall_min = 1.0 / (1.0 + args.lam_wall)
        print(
            f"wall: r={args.r_wall}  width={args.wall_width}  lam_peak={args.lam_wall}"
            f"  sigma_min_at_wall={sigma_wall_min:.4f}"
        )
    return lat, pos, r


def run(args):
    lat, pos, r_all = build_lattice(args)
    dt = lat.stable_dt()
    r_inner = args.R * (1.0 - args.sponge)
    deficit0 = -np.minimum(lat.u, 0.0)
    com0, mass0 = weighted_com(pos, deficit0)
    spread0 = weighted_spread(pos, deficit0, com0)
    u_e0, phi_e0 = field_energy(lat, args.mass)

    print(
        f"N={lat.N}  dt={dt:.4e}  R={args.R}  r_inner={r_inner:.2f}  u_depth={args.u_depth}"
        f"  u_width={args.u_width}  mass={args.mass}"
    )
    print(
        f"deficit0: mass={mass0:.4f}  com_r={np.linalg.norm(com0):.4f}  spread={spread0:.4f}"
    )
    print(f"field energy0: u_e={u_e0:.4e}  phi_e={phi_e0:.4e}  total={u_e0+phi_e0:.4e}")

    history = []  # (t, mass, spread, frac_inside, phi_rms, u_e, phi_e, total_e)

    def record(step):
        w = -np.minimum(lat.u, 0.0)
        com, mass = weighted_com(pos, w)
        spread = weighted_spread(pos, w, com)
        total = w.sum()
        inside = float(w[r_all < args.r_wall].sum())
        frac_inside = inside / total if total > 1e-30 else 0.0
        phi_rms = float(np.sqrt(np.mean(lat.phi**2)))
        u_e, phi_e = field_energy(lat, args.mass)
        history.append(
            (step * dt, mass, spread, frac_inside, phi_rms, u_e, phi_e, u_e + phi_e)
        )
        h = history[-1]
        print(
            f"  {step:6d}  mass={h[1]:10.4e}  spread={h[2]:8.4f}  frac_in={h[3]:6.4f}"
            f"  phi_rms={h[4]:9.3e}  u_e={h[5]:9.3e}  phi_e={h[6]:9.3e}  total_e={h[7]:9.3e}",
            flush=True,
        )

    record(0)
    for s in range(1, args.n_steps + 1):
        lat.step(dt, n_threads=args.n_threads)
        if args.mass > 0:
            # optional bulk KG term, off by default -- see probe_electron_trap.py
            sigma = 1.0 / (1.0 + lat.lam)
            mass_term = dt * sigma[:, None] * (args.mass**2) * lat.u[:, None] / 30.0
            lat.phi = np.where(lat.fixed_phi, lat.phi, lat.phi + mass_term)
        if s % args.report == 0:
            record(s)

    mass0_h, mass_end = history[0][1], history[-1][1]
    spread0_h, spread_end = history[0][2], history[-1][2]
    frac_end = history[-1][3]
    e0, e_end = history[0][7], history[-1][7]

    res = dict(
        mass_frac=mass_end / max(mass0_h, 1e-30),
        spread_frac=spread_end / max(spread0_h, 1e-30),
        frac_inside_end=frac_end,
        energy_frac=e_end / max(e0, 1e-30),
        history=history,
        dt=dt,
    )
    print(
        f"\n-> mass_frac={res['mass_frac']:.4f}  spread_frac={res['spread_frac']:.4f}"
        f"  frac_inside_end={res['frac_inside_end']:.4f}  energy_frac={res['energy_frac']:.4f}"
    )
    print(
        "   (frac_inside_end ~= 1 with energy_frac ~= 1 => genuine reflection/confinement;"
        " frac_inside_end decaying toward the no-wall value => leaky/no-op wall)"
    )
    return res


def main():
    args = parser.parse_args()
    res = run(args)

    if args.data_out:
        np.savez(
            args.data_out,
            history=np.array(res["history"]),
            lam_wall=args.lam_wall,
            r_wall=args.r_wall,
            mass=args.mass,
        )
        print(f"data saved -> {args.data_out}")

    if args.no_plot:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = res["history"]
    t = np.array([x[0] for x in h])
    mass = np.array([x[1] for x in h])
    spread = np.array([x[2] for x in h])
    frac_inside = np.array([x[3] for x in h])
    total_e = np.array([x[7] for x in h])

    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    fig.suptitle(
        f"Electron guide-field probe  R={args.R}  r_wall={args.r_wall}  lam_wall={args.lam_wall}",
        fontsize=10,
    )
    axes[0].plot(t, mass / mass[0], color="C0", lw=1.5)
    axes[0].set_ylabel("mass / mass0")
    axes[0].set_title("Deficit conservation")
    axes[1].plot(t, spread / spread[0], color="C1", lw=1.5)
    axes[1].set_ylabel("spread / spread0")
    axes[1].set_title("Localisation (>>1 = delocalising, ~1 = held)")
    axes[2].plot(t, frac_inside, color="C4", lw=1.5)
    axes[2].axhline(1.0, color="k", lw=0.5, ls="--")
    axes[2].set_ylabel("frac inside wall")
    axes[2].set_title("Confinement (1 = fully reflected, ->0 = fully leaked)")
    axes[3].plot(t, total_e / total_e[0], color="C3", lw=1.5)
    axes[3].set_ylabel("field_energy / energy0")
    axes[3].set_title("Conservation check")
    axes[3].set_xlabel("time")
    plt.tight_layout()
    plt.savefig(args.plot_out, dpi=130, bbox_inches="tight")
    print(f"plot saved -> {args.plot_out}")


if __name__ == "__main__":
    main()
