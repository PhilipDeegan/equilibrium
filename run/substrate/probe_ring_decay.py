"""
Ring phi concentration decay probe.

Initialises a free toroidal ring (frozen=False) and tracks how phi energy
distributes over time between:
  - ring tiles      (the 8-tile loop itself)
  - inner bulk      (non-ring interior cells)
  - sponge layer    (absorbed at boundary)

No lam transport, no u-gradient source — pure wave equation only.
This establishes a baseline: how long does the winding stays concentrated
before leaking into the bulk.

Result: ring_frac ≈ 0.98 throughout — winding is self-sustaining.
The --lam / --phi-advect modes (step_lam_inertial) have been removed;
J_lam and phi_advect are obsolete (see doc/dst.md, through-cell transit).
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")

from equilibrium.substrate.flux_tube import find_toroidal_ring, init_toroidal_winding
from equilibrium.substrate.lattice import DampedLattice

parser = argparse.ArgumentParser()
parser.add_argument("--R", type=float, default=20.0)
parser.add_argument("--n-steps", type=int, default=50000)
parser.add_argument("--report", type=int, default=2500)
parser.add_argument("--n-threads", type=int, default=1)


def phi_energy(lat, mask):
    return float(np.sum(lat.phi[mask] ** 2))


def main():
    args = parser.parse_args()

    lat = DampedLattice(radius_cells=args.R, sponge_fraction=0.15, damp_max=0.003)
    pos = lat.pos
    dt = lat.stable_dt()
    R = args.R

    # Identify sponge vs interior cells
    r_all = np.linalg.norm(pos, axis=1)
    sponge_r = R * (1.0 - 0.15)
    sponge_mask = r_all >= sponge_r
    interior_mask = ~sponge_mask

    ring_tiles, ring_faces = find_toroidal_ring(lat, axis="z")
    if ring_tiles is None:
        print("ERROR: no toroidal ring found")
        sys.exit(1)

    init_toroidal_winding(lat, ring_tiles, ring_faces, amplitude=1.0, frozen=False)

    ring_mask = np.zeros(lat.N, dtype=bool)
    ring_mask[ring_tiles] = True
    bulk_mask = interior_mask & ~ring_mask

    ring_r = float(np.mean(np.linalg.norm(pos[ring_tiles], axis=1)))

    print(f"N={lat.N}  R={R}  dt={dt:.4e}")
    print(f"ring: {len(ring_tiles)} tiles  r≈{ring_r:.3f}")
    print(
        f"sponge boundary at r≈{sponge_r:.1f}  "
        f"interior={interior_mask.sum()}  sponge={sponge_mask.sum()}"
    )
    print()

    e0 = phi_energy(lat, interior_mask) + phi_energy(lat, sponge_mask)
    print(
        f"{'step':>8}  {'ring_frac':>10}  {'bulk_frac':>10}  {'sponge_abs':>11}  {'phi_rms':>10}"
    )
    print("  " + "-" * 60)

    absorbed = 0.0

    for s in range(args.n_steps + 1):
        if s % args.report == 0:
            e_ring = phi_energy(lat, ring_mask)
            e_bulk = phi_energy(lat, bulk_mask)
            phi_rms = float(np.sqrt(np.mean(lat.phi**2)))
            ring_frac = e_ring / (e0 + 1e-30)
            bulk_frac = e_bulk / (e0 + 1e-30)
            print(
                f"{s:8d}  {ring_frac:10.4f}  {bulk_frac:10.4f}  "
                f"{absorbed/(e0+1e-30):11.4f}  {phi_rms:10.4e}",
                flush=True,
            )

        if s < args.n_steps:
            e_before = phi_energy(lat, interior_mask) + phi_energy(lat, sponge_mask)
            lat.step(dt, n_threads=args.n_threads)
            e_after = phi_energy(lat, interior_mask) + phi_energy(lat, sponge_mask)
            absorbed += max(0.0, e_before - e_after)

    print("\ndone")


if __name__ == "__main__":
    main()
