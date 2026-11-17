"""
DST electron trap probe (uniform lattice) -- external confinement diagnostic.

Context: a free (lam=0) u/phi structure cannot sit statically without global
u-uniformity (see project_step_cross_v2 / project_dst_electron_topology
discussion, 2026-07-12) -- any nonuniform u sources phi via
dphi_ik = -dt*sigma_face*W_k*(u_k-u_i), a per-face condition with no
collective/relational escape hatch (unlike the u-update, which sums over 14
faces and so DOES admit divergence-free-phi cancellation -- exactly the
winding/dipole electron's trick). BUT: per-face sourcing that spreads a
symmetric disturbance outward while EXACTLY conserving a discrete field
energy is reversible delocalization (a free-particle wavepacket spreading,
per the strong Huygens principle in 3D), not decay/dissipation (heat,
photon emission) -- those require an actual irreversible sink, not just
"the field ends up somewhere else". A genuinely BOUND (non-spreading)
electron needs a potential well (e.g. a small, mass-ratio-scaled intrinsic
lam profile, ~1/1836 of the proton's, per the mc^2-bound-phi reframing of
lam as substrate depletion) -- that's the physically meaningful target and
requires a companion proton.

This probe sidesteps the bound-state requirement using an ion-trap analogy:
a real Paul/Penning trap is an EXTERNAL, clearly-artificial confining field
used purely as a diagnostic apparatus, not a claim about the trapped
particle's own intrinsic physics. Implemented as a genuine SOURCE TERM added
to u's own update (mass=0 recovers the untrapped dynamics exactly) --
NOT as fixed_u/fixed_phi pinning, which freezes field VALUES outright and
silently asserts an answer instead of letting the dynamics respond to an
added potential. The point is to hold a symmetric u-deficit ("ball of
charge") localized long enough to observe whatever internal steady-state
behaviour it settles into (does phi organise into a stable circulation
pattern? does it look like a standing wave / discrete-level structure,
consistent with the "offloads phi like Bohr energy levels" idea, rather than
continuously radiating?).

Options (mark [x] when implemented AND run with a recorded result):
- [x] Baseline conservation check (mass=0/no trap): verified on the plain
      SubstrateLattice (no sponge, no C++) that the discrete field energy
      (0.5*sum(u^2) + 0.5*sum(W*phi^2)) jumps once during an initial ~100-
      step leapfrog "phase-in" transient (phi starting at exactly 0 is not
      the natural staggered-leapfrog phase relative to u) and then holds
      flat (33.90 -> 33.78 -> 33.78 -> 33.78 over steps 200-500) while the
      deficit's spread keeps growing. Confirms genuine reversible
      delocalisation (free-particle-wavepacket-like spreading), not decay,
      once past the startup artefact -- 2026-07-12.
- [x] Linear radial well (relaxation-to-target): du_i += -dt*sigma_i*
      kappa_trap*(u_i - u_target(r_i)), same mechanism as DampedLattice's
      own sponge, centred on the origin instead of the outer boundary.
      RESULT (negative, but informative): does NOT confine -- spread stayed
      ~11 regardless of kappa_trap (tested 0.02 and 0.5), while "mass" drained
      to 8% and field energy to 34% of initial over 4000 steps. This is
      dissipative (linear damping/friction toward a fixed target), not a
      conservative confining potential -- draining energy, not redirecting
      it. NOTE: an earlier attempt used a bare "-kappa*r_i" additive force,
      which is even more wrong -- u_i lives at a FIXED grid location, not a
      moving particle's position, so a constant additive term never
      restores, it just injects energy forever (energy_frac ran to 270x in
      500 steps before this was caught) -- 2026-07-12.
- [ ] Klein-Gordon mass term. FIRST ATTEMPT (WRONG, caught before trusting
      results): added "du_i += -dt*sigma_i*m**2*u_i" directly to u's own
      equation -- that's first-order exponential decay/friction toward
      zero, NOT the KG equation (confirmed empirically: energy still drained
      to 46% over 4000 steps at m=0.05, spread never plateaued). The real KG
      equation is SECOND order: d^2u/dt^2 = grad^2(u) - m^2*u. Since
      du_i/dt=-div(phi) already gives d^2u/dt^2=grad^2(u) (via
      d(du/dt)/dt=-div(dphi/dt)=-div(-grad(u))=grad^2(u)), the mass term
      must instead modify PHI's equation so its divergence reproduces
      "-m^2*u": dphi_ik/dt += sigma_i*m**2*u_i/30 (uniform across all 14
      faces, sourced by that cell's own u_i; sum_k W_k = 30 exactly, so
      summing this correction over k reproduces +m^2*u_i in d(du/dt)/dt).
      This version is conservative by construction (built from the same
      -div/-grad structure as the existing terms), unlike a direct decay
      term on u. (CORRECTED VERSION implemented below -- current primary
      target; textbook conservative way to give a wave equation a finite
      confinement length -- m=0 only supports freely-dispersing solutions,
      m>0 gives a genuine static, localised Yukawa-profile ground state
      (u(r) ~ exp(-m*r)/r in the continuum limit), sidestepping Earnshaw's
      theorem since it's a field-theory mass term, not a classical static
      force field sourced by charges. Directly connects to the earlier
      "electron carries a small mass-ratio-scaled lam (~1/1836)" reframing
      from the 2026-07-12 conversation -- a mass term IS what
      confinement/inertia means in this formalism.)
- [ ] Harmonic radial well: du_i += -dt*sigma_i*kappa_trap*r_i**2
      (superseded by the mass-term approach above -- keep as fallback only
      if the mass term doesn't produce a genuine plateau)
- [ ] Trap shaped like the hypothesised intrinsic electron lam-well
      (rehearsal for the bound/proton case -- not claimed as real physics)
- [ ] Weak-mass sweep: does the deficit relax toward a stable quasi-bound
      (Yukawa) shape, or just get held wherever it started?
- [ ] Strong-mass sweep: quick read of internal phi circulation pattern
      once confined
- [ ] Oscillating (Paul-trap / ponderomotive) time-dependent confinement --
      demoted to fallback since the mass term is conservative and better-
      justified; still worth trying if the mass term fails to confine
- [ ] Bound case: proton + electron with mass-ratio-scaled intrinsic lam
      (~1/1836), NO external trap -- the physically meaningful target
- [ ] Curl-feedback extension to the phi update (deferred -- may matter for
      step_cross_v2 linkage/reconnection, not basic electron stability)

IC: symmetric negative Gaussian u-deficit at the lattice centre (the "ball
of charge"), phi=0, lam=0 everywhere (free-electron null hypothesis -- no
lam channel used in this probe at all).

Run:
  PYTHONPATH=$PWD python3 -u run/substrate/probe_electron_trap.py
  PYTHONPATH=$PWD python3 -u run/substrate/probe_electron_trap.py --kappa-trap 0
      (baseline conservation/spreading check, no trap)
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium.substrate.lattice import DampedLattice

parser = argparse.ArgumentParser()
parser.add_argument("--R", type=float, default=20.0)
parser.add_argument("--u-depth", type=float, default=1.0, help="initial deficit depth")
parser.add_argument(
    "--u-width", type=float, default=2.0, help="initial deficit Gaussian width"
)
parser.add_argument(
    "--mass",
    type=float,
    default=0.05,
    help="Klein-Gordon mass term m (0 = off, recovers plain massless wave dynamics)",
)
parser.add_argument("--n-steps", type=int, default=6000)
parser.add_argument("--report", type=int, default=250)
parser.add_argument("--sponge", type=float, default=0.15)
parser.add_argument("--damp-max", type=float, default=0.003)
parser.add_argument("--n-threads", type=int, default=4)
parser.add_argument("--plot-out", type=str, default="probe_electron_trap.png")
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

    The (1+mass^2) factor folds in the Klein-Gordon mass term's own potential
    energy contribution (0.5*mass^2*u^2 per cell), so the conservation check
    stays meaningful when mass > 0.
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
    # symmetric negative Gaussian deficit -- the "ball of charge", lam=0 everywhere
    lat.u = -args.u_depth * np.exp(-((r / args.u_width) ** 2))
    return lat, pos, r


def run(args):
    lat, pos, r_all = build_lattice(args)
    dt = lat.stable_dt()
    deficit0 = -np.minimum(lat.u, 0.0)  # positive weight where u is a deficit
    com0, mass0 = weighted_com(pos, deficit0)
    spread0 = weighted_spread(pos, deficit0, com0)
    u_e0, phi_e0 = field_energy(lat, args.mass)

    print(
        f"N={lat.N}  dt={dt:.4e}  R={args.R}  u_depth={args.u_depth}"
        f"  u_width={args.u_width}  mass={args.mass}"
    )
    print(
        f"deficit0: mass={mass0:.4f}  com_r={np.linalg.norm(com0):.4f}  spread={spread0:.4f}"
    )
    print(f"field energy0: u_e={u_e0:.4e}  phi_e={phi_e0:.4e}  total={u_e0+phi_e0:.4e}")

    history = []  # (t, mass, spread, phi_rms, u_e, phi_e, total_e)

    def record(step):
        w = -np.minimum(lat.u, 0.0)
        com, mass = weighted_com(pos, w)
        spread = weighted_spread(pos, w, com)
        phi_rms = float(np.sqrt(np.mean(lat.phi**2)))
        u_e, phi_e = field_energy(lat, args.mass)
        history.append((step * dt, mass, spread, phi_rms, u_e, phi_e, u_e + phi_e))
        h = history[-1]
        print(
            f"  {step:6d}  mass={h[1]:10.4e}  spread={h[2]:8.4f}  phi_rms={h[3]:9.3e}"
            f"  u_e={h[4]:9.3e}  phi_e={h[5]:9.3e}  total_e={h[6]:9.3e}",
            flush=True,
        )

    record(0)
    for s in range(1, args.n_steps + 1):
        lat.step(dt, n_threads=args.n_threads)
        if args.mass > 0:
            # KG mass term lives in phi's equation (see docstring) -- a direct
            # decay term on u is NOT conservative and was ruled out empirically.
            sigma = 1.0 / (1.0 + lat.lam)
            mass_term = dt * sigma[:, None] * (args.mass**2) * lat.u[:, None] / 30.0
            lat.phi = np.where(lat.fixed_phi, lat.phi, lat.phi + mass_term)
        if s % args.report == 0:
            record(s)

    mass0_h, mass_end = history[0][1], history[-1][1]
    spread0_h, spread_end = history[0][2], history[-1][2]
    e0, e_end = history[0][6], history[-1][6]

    res = dict(
        mass_frac=mass_end / max(mass0_h, 1e-30),
        spread_frac=spread_end / max(spread0_h, 1e-30),
        energy_frac=e_end / max(e0, 1e-30),
        history=history,
        dt=dt,
    )
    print(
        f"\n-> mass_frac={res['mass_frac']:.4f}  spread_frac={res['spread_frac']:.4f}"
        f"  energy_frac={res['energy_frac']:.4f}"
    )
    print(
        "   (spread_frac >> 1 with energy_frac ~= 1 => reversible delocalisation, not decay;"
        " spread_frac ~= 1 with the trap on => confinement holding)"
    )
    return res


def main():
    args = parser.parse_args()
    res = run(args)

    if args.data_out:
        np.savez(
            args.data_out,
            history=np.array(res["history"]),
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
    phi_rms = np.array([x[3] for x in h])
    total_e = np.array([x[6] for x in h])

    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    fig.suptitle(
        f"Electron trap probe  R={args.R}  mass={args.mass}",
        fontsize=10,
    )
    axes[0].plot(t, mass / mass[0], color="C0", lw=1.5)
    axes[0].set_ylabel("mass / mass0")
    axes[0].set_title("Deficit conservation")
    axes[1].plot(t, spread / spread[0], color="C1", lw=1.5)
    axes[1].set_ylabel("spread / spread0")
    axes[1].set_title("Localisation (>>1 = delocalising, ~1 = held)")
    axes[2].plot(t, phi_rms, color="C2", lw=1.5)
    axes[2].set_ylabel("phi_rms")
    axes[2].set_title("Internal circulation activity")
    axes[3].plot(t, total_e / total_e[0], color="C3", lw=1.5)
    axes[3].set_ylabel("field_energy / energy0")
    axes[3].set_title("Conservation check")
    axes[3].set_xlabel("time")
    plt.tight_layout()
    plt.savefig(args.plot_out, dpi=130, bbox_inches="tight")
    print(f"plot saved -> {args.plot_out}")


if __name__ == "__main__":
    main()
