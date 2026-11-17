"""
Three-way energy partition (translation KE / heat retained in the core /
radiated into the transit+sponge zones) for A2's kick/coast cycling-ring
knot -- resolves the still-open "not yet determined whether this is
legitimate radiation or a leak" question flagged for A2's H_kick -> H_coast
drop (`now`, 2026-08-20 entry).

Reuses A2's IC construction verbatim (chi0-sourced knot + cycling ring via
find_toroidal_ring/init_toroidal_winding,
exp/2026/08/20/18_00_cycling_ring_charge_test/probe_cycling_ring_curl.py)
but at a larger domain (default R=35 vs A2's original R=20) so a coast has
real room to propagate before hitting the sponge -- same "big domain, small
object" idea as isolated_dipole.py/electron_positron_collision.py (R=60,
scale=8), scaled down from a first attempt at R=60 (1.175M cells on this
DampedLattice/BCC-sphere implementation -- ~6 CPU-minutes per 500 steps,
too slow for iteration).

Drops the W=+-1 chirality sweep (not needed here, single W=+1 run) and adds:
  - energy-weighted centroid (A2's own weighted_com used lat.lam, which is
    NEVER seeded by this IC -- always zero -- so its com/v_coast were
    silently degenerate the whole time, confirmed in the 2026-08-27
    rest-only cross_scale=0 rerun's log: com=(+0.000,...) throughout. Fixed
    here to weight by per-cell field energy instead.)
  - zone energy tracking: core (r<shell_rmin), transit (shell_rmin<=r<
    r_inner), sponge (r>=r_inner) -- reuses A2's own shell_rmin convention
  - final energy budget relative to the rest-phase baseline:
      KE_translation = 0.5 * M_eff * v_coast^2   (M_eff = core-zone H, proxy)
      heat_core      = (H_core_coast - H_core_rest) - KE_translation
      radiated       = (H_transit_coast - H_transit_rest)
                        + (H_sponge_coast - H_sponge_rest)

Run:
  PYTHONPATH=$PWD python3 -u exp/2026/08/27/18_24_kick_coast_energy_partition/probe_energy_partition.py
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium.substrate.lattice import DampedLattice
from equilibrium.substrate.flux_tube import (
    find_toroidal_ring,
    init_toroidal_winding,
    resonant_freq,
)

_AX = {"x": 0, "y": 1, "z": 2}


def field_energy_cell(lat):
    return 0.5 * lat.u**2 + 0.5 * np.sum(lat.phi**2, axis=1)


def weighted_com(pos, w):
    total = w.sum()
    if total < 1e-30:
        return np.zeros(3), 0.0
    return (pos * w[:, None]).sum(axis=0) / total, float(total)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--R", type=float, default=35.0)
    p.add_argument("--chi0", type=float, default=0.9)
    p.add_argument("--profile", type=str, default="1/r2")
    p.add_argument("--ring-amp", type=float, default=0.05)
    p.add_argument(
        "--cross-scale",
        type=float,
        default=0.0,
        help="0 = pure bare-step() run (default, primary question); >0 compares step_cross's effect, applied only during kick/coast, never rest",
    )
    p.add_argument("--gradient", type=float, default=5e-4)
    p.add_argument("--kick-axis", type=str, default="z")
    p.add_argument("--n-periods-rest", type=float, default=5.0)
    p.add_argument("--n-kick", type=int, default=3000)
    p.add_argument("--n-coast", type=int, default=3000)
    p.add_argument("--report", type=int, default=250)
    p.add_argument("--sponge", type=float, default=0.15)
    p.add_argument("--damp-max", type=float, default=0.003)
    p.add_argument("--n-threads", type=int, default=8)
    p.add_argument(
        "--shell-rmin", type=float, default=3.0, help="core/transit boundary"
    )
    outdir = "exp/2026/08/27/18_24_kick_coast_energy_partition"
    p.add_argument("--out-npz", type=str, default=f"{outdir}/result.npz")
    args = p.parse_args()

    ax_i = _AX[args.kick_axis]
    lat = DampedLattice(
        radius_cells=args.R, sponge_fraction=args.sponge, damp_max=args.damp_max
    )
    pos = lat.pos
    r_all = np.linalg.norm(pos, axis=1)
    origin = int(np.argmin(r_all))
    lat.set_winding(lat.tiles[origin], chi_0=args.chi0, profile=args.profile)

    ring_tiles, ring_faces = find_toroidal_ring(
        lat, axis=args.kick_axis, mode="minimal"
    )
    if ring_tiles is None:
        print("ERROR: no ring found")
        sys.exit(1)
    omega = resonant_freq(len(ring_tiles), face_type="sq")
    init_toroidal_winding(
        lat, ring_tiles, ring_faces, amplitude=args.ring_amp, frozen=False
    )

    dt = lat.stable_dt()
    period = 2.0 * np.pi / omega
    n_rest = int(round(args.n_periods_rest * period / dt))
    r_max = float(r_all.max())
    r_inner = r_max * (1.0 - args.sponge)
    battery = (r_all > r_inner - 2.0) & (r_all <= r_inner)
    tilt = args.gradient * pos[:, ax_i]

    core_mask = r_all < args.shell_rmin
    transit_mask = (r_all >= args.shell_rmin) & (r_all < r_inner)
    sponge_mask = r_all >= r_inner

    def phys_step(apply_cross):
        lat.step(dt, n_threads=args.n_threads)
        if apply_cross and args.cross_scale > 0.0:
            lat.step_cross(dt, args.cross_scale, n_threads=args.n_threads)

    def zone_energies():
        Hc = field_energy_cell(lat)
        return (
            float(Hc[core_mask].sum()),
            float(Hc[transit_mask].sum()),
            float(Hc[sponge_mask].sum()),
        )

    print(
        f"N={lat.N}  R={args.R}  dt={dt:.4e}  omega={omega:.4f}  period={period:.2f}  "
        f"n_rest={n_rest}  core_n={int(core_mask.sum())}  transit_n={int(transit_mask.sum())}"
        f"  sponge_n={int(sponge_mask.sum())}  r_inner={r_inner:.1f}",
        flush=True,
    )

    hist = []  # (t, phase, H_core, H_transit, H_sponge, com, mass)

    def record(step, phase):
        Hc = field_energy_cell(lat)
        com, mass = weighted_com(pos, Hc)
        hc, ht, hs = (
            float(Hc[core_mask].sum()),
            float(Hc[transit_mask].sum()),
            float(Hc[sponge_mask].sum()),
        )
        hist.append((step * dt, phase, hc, ht, hs, com.copy(), mass))
        if step % args.report == 0:
            print(
                f"  {step:6d} {phase:>5}  H_core={hc:.4e}  H_transit={ht:.4e}  H_sponge={hs:.4e}  "
                f"com=({com[0]:+.3f},{com[1]:+.3f},{com[2]:+.3f})",
                flush=True,
            )

    step = 0
    record(step, "base")
    for s in range(1, n_rest + 1):
        phys_step(apply_cross=False)
        step += 1
        if step % args.report == 0:
            record(step, "base")
    record(step, "base")
    H_core_rest, H_transit_rest, H_sponge_rest = zone_energies()

    lat.u = lat.u + tilt
    lat.fixed_u[battery] = True
    for s in range(1, args.n_kick + 1):
        phys_step(apply_cross=True)
        step += 1
        if step % args.report == 0:
            record(step, "kick")
    record(step, "kick")
    H_core_kick, H_transit_kick, H_sponge_kick = zone_energies()

    lat.fixed_u[battery] = False
    lat.u = lat.u - tilt
    for s in range(1, args.n_coast + 1):
        phys_step(apply_cross=True)
        step += 1
        if step % args.report == 0:
            record(step, "coast")
    record(step, "coast")
    H_core_coast, H_transit_coast, H_sponge_coast = zone_energies()

    coast_rows = [
        (t, com, mass) for t, phase, hc, ht, hs, com, mass in hist if phase == "coast"
    ]
    if len(coast_rows) >= 3:
        rows = coast_rows[len(coast_rows) // 2 :]
        t = np.array([r[0] for r in rows])
        x = np.array([r[1][ax_i] for r in rows])
        v_coast = float(np.polyfit(t, x, 1)[0])
    else:
        v_coast = 0.0
    M_eff = H_core_coast  # field-energy mass proxy, core zone only
    KE_translation = 0.5 * M_eff * v_coast**2

    heat_core = (H_core_coast - H_core_rest) - KE_translation
    radiated = (H_transit_coast - H_transit_rest) + (H_sponge_coast - H_sponge_rest)
    kick_input = (H_core_kick + H_transit_kick + H_sponge_kick) - (
        H_core_rest + H_transit_rest + H_sponge_rest
    )

    print("\n--- energy partition (relative to rest baseline) ---")
    print(f"kick input (total H added)   = {kick_input:.4e}")
    print(f"v_coast                      = {v_coast:+.4e}")
    print(f"KE_translation (core, proxy) = {KE_translation:.4e}")
    print(f"heat_core (core excess - KE) = {heat_core:.4e}")
    print(f"radiated (transit+sponge dH) = {radiated:.4e}")
    accounted = KE_translation + heat_core + radiated
    denom = max(abs(kick_input), 1e-30)
    print(
        f"accounted total               = {accounted:.4e}  ({100 * accounted / denom:.1f}% of kick input)"
    )

    np.savez(
        args.out_npz,
        H_core_rest=H_core_rest,
        H_transit_rest=H_transit_rest,
        H_sponge_rest=H_sponge_rest,
        H_core_kick=H_core_kick,
        H_transit_kick=H_transit_kick,
        H_sponge_kick=H_sponge_kick,
        H_core_coast=H_core_coast,
        H_transit_coast=H_transit_coast,
        H_sponge_coast=H_sponge_coast,
        v_coast=v_coast,
        KE_translation=KE_translation,
        heat_core=heat_core,
        radiated=radiated,
        kick_input=kick_input,
    )
    print(f"saved {args.out_npz}")


if __name__ == "__main__":
    main()
