"""
Does a bound, topologically-closed knot (real internal circulation, not a
diffuse pressure blob) drift toward a `lam`/`sigma` gradient under BARE
step() alone -- same "star" lam profile as run/substrate/eddington.py
(sigma = 1/(1+lam), lower throughput near the star), same knot construction
as A2 (exp/2026/08/20/18_00_cycling_ring_charge_test/probe_cycling_ring_
curl.py: chi0-sourced core + cycling ring), NO step_cross anywhere.

Motivation (this session): Eddington's own light-bending test already shows
a diffuse pressure/u source refracts toward a lam-gradient star under plain
alat.step() alone -- "light in DST = pressure finding path of least
resistance through sigma field" (eddington.py docstring), no step_cross
involved. But step_cross was always orthogonal for photons (pure face-
hopping content, never in step_cross's domain), so that result says nothing
about whether a genuinely CLOSED structure (which step_cross's whole
"crossing between cells" framing WAS meant to apply to) behaves the same
way. This is the direct, standalone test of that question.

Geometry: knot stays at the lattice's own origin (find_toroidal_ring has no
off-center option, always searches near (0,0,0) -- confirmed by reading
equilibrium/substrate/flux_tube.py), so the star's lam profile is offset
instead, centred at (+d, 0, 0), reusing eddington.py's exact
lam_r = where(r>R_star, lam0*R_star/r, lam0) formula. Two runs: STAR
(lam0>0) and BASELINE (lam0=0, same knot, same domain) -- any drift in the
BASELINE is confound/asymmetry to subtract, not signal.

Run:
  PYTHONPATH=$PWD python3 -u exp/2026/08/27/18_28_knot_in_star_gradient/probe_knot_star_gradient.py
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium.substrate.discrete import LAM_MAX
from equilibrium.substrate.lattice import DampedLattice
from equilibrium.substrate.flux_tube import (
    find_toroidal_ring,
    init_toroidal_winding,
    resonant_freq,
)


def field_energy_cell(lat):
    return 0.5 * lat.u**2 + 0.5 * np.sum(lat.phi**2, axis=1)


def weighted_com(pos, w):
    total = w.sum()
    if total < 1e-30:
        return np.zeros(3), 0.0
    return (pos * w[:, None]).sum(axis=0) / total, float(total)


def run_one(args, lam0):
    lat = DampedLattice(
        radius_cells=args.R, sponge_fraction=args.sponge, damp_max=args.damp_max
    )
    pos = lat.pos
    r0 = np.linalg.norm(pos, axis=1)
    origin = int(np.argmin(r0))
    lat.set_winding(lat.tiles[origin], chi_0=args.chi0, profile=args.profile)

    ring_tiles, ring_faces = find_toroidal_ring(
        lat, axis=args.ring_axis, mode="minimal"
    )
    if ring_tiles is None:
        print("ERROR: no ring found")
        sys.exit(1)
    omega = resonant_freq(len(ring_tiles), face_type="sq")
    init_toroidal_winding(
        lat, ring_tiles, ring_faces, amplitude=args.ring_amp, frozen=False
    )

    if lam0 > 0.0:
        star_c = np.array([args.d, 0.0, 0.0])
        r_star = np.linalg.norm(pos - star_c[None, :], axis=1)
        safe_r = np.where(r_star > 0, r_star, 1.0)
        lam_r = np.where(r_star > args.R_star, lam0 * args.R_star / safe_r, float(lam0))
        lat.lam = np.clip(lam_r, 0.0, LAM_MAX)

    dt = lat.stable_dt()
    period = 2.0 * np.pi / omega
    n_rest = int(round(args.n_periods_rest * period / dt))
    core_mask = r0 < args.shell_rmin

    print(
        f"[lam0={lam0}] N={lat.N}  R={args.R}  d={args.d}  R_star={args.R_star}"
        f"  dt={dt:.4e}  omega={omega:.4f}  n_rest={n_rest}  core_n={int(core_mask.sum())}",
        flush=True,
    )

    hist = []

    def record(step):
        Hc = field_energy_cell(lat)
        com, mass = weighted_com(pos, Hc * core_mask)
        hist.append((step * dt, com[0], mass))
        if step % args.report == 0:
            print(
                f"  [lam0={lam0}] {step:6d}  com_x={com[0]:+.5f}  mass={mass:.4e}",
                flush=True,
            )

    step = 0
    record(step)
    for s in range(1, n_rest + 1):
        lat.step(dt, n_threads=args.n_threads)
        step += 1
        if step % args.report == 0:
            record(step)
    record(step)

    return np.array(hist)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--R", type=float, default=60.0)
    p.add_argument("--chi0", type=float, default=0.9)
    p.add_argument("--profile", type=str, default="1/r2")
    p.add_argument("--ring-axis", type=str, default="z")
    p.add_argument("--ring-amp", type=float, default=0.05)
    p.add_argument("--d", type=float, default=30.0, help="star offset along +x")
    p.add_argument("--R-star", type=float, default=3.0)
    p.add_argument(
        "--lam0",
        type=float,
        default=2.0,
        help="star strength (sigma=1/(1+lam0) at centre)",
    )
    p.add_argument(
        "--n-periods-rest",
        type=float,
        default=40.0,
        help="run duration, in ring periods",
    )
    p.add_argument("--report", type=int, default=500)
    p.add_argument("--sponge", type=float, default=0.15)
    p.add_argument("--damp-max", type=float, default=0.003)
    p.add_argument("--n-threads", type=int, default=6)
    p.add_argument("--shell-rmin", type=float, default=3.0)
    outdir = "exp/2026/08/27/18_28_knot_in_star_gradient"
    p.add_argument("--out-npz", type=str, default=f"{outdir}/result.npz")
    args = p.parse_args()

    hist_star = run_one(args, args.lam0)
    hist_base = run_one(args, 0.0)

    x_star0, x_starN = hist_star[0, 1], hist_star[-1, 1]
    x_base0, x_baseN = hist_base[0, 1], hist_base[-1, 1]
    print("\n--- summary ---")
    print(
        f"STAR      com_x: {x_star0:+.5f} -> {x_starN:+.5f}  (delta={x_starN - x_star0:+.5f})"
    )
    print(
        f"BASELINE  com_x: {x_base0:+.5f} -> {x_baseN:+.5f}  (delta={x_baseN - x_base0:+.5f})"
    )
    net = (x_starN - x_star0) - (x_baseN - x_base0)
    print(
        f"net drift toward star (STAR delta - BASELINE delta) = {net:+.5f}"
        f"  {'(toward star, +x)' if net > 0 else '(away from star or null)'}"
    )

    np.savez(args.out_npz, hist_star=hist_star, hist_base=hist_base, net=net)
    print(f"saved {args.out_npz}")


if __name__ == "__main__":
    main()
