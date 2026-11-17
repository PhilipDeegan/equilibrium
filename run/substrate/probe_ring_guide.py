"""
DST electron guiding-centre probe.

Two co-axial toroidal rings (both wound around z):
  - guide ring  at z≈0          frozen phi — sustained flux source
  - electron ring at z=+z_sep   free — no pins

Runs lat.step only.  Tracks whether the free ring translates along z
toward (or away from) the guide ring.

In plasma physics a current loop in a non-uniform B_z field experiences
a force F = ∇(m·B).  The guide ring creates a phi_z field that falls off
with distance along z.  If the free ring couples to this field it should
drift toward z=0 (toward the guide).

No lam transport.  Pure DST update rule.
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")

from equilibrium.substrate.flux_tube import (
    _TOROIDAL_PATTERNS,
    _sq_hop_dist,
    _trace_loop,
    init_toroidal_winding,
)
from equilibrium.substrate.lattice import DampedLattice

parser = argparse.ArgumentParser()
parser.add_argument("--R", type=float, default=20.0)
parser.add_argument(
    "--z-sep",
    type=float,
    default=4.0,
    help="target z offset for the free (electron) ring",
)
parser.add_argument("--amplitude", type=float, default=1.0)
parser.add_argument("--n-steps", type=int, default=20000)
parser.add_argument("--report", type=int, default=1000)
parser.add_argument("--n-threads", type=int, default=1)


def find_ring_near(lat, target, axis="z", n_candidates=100):
    """Find a toroidal ring near the given target position."""
    face_seq, _ = _TOROIDAL_PATTERNS[axis]
    target = np.asarray(target, dtype=float)
    dists = np.linalg.norm(lat.pos - target, axis=1)
    for start in np.argsort(dists)[:n_candidates]:
        tiles, faces = _trace_loop(lat, int(start), face_seq)
        if tiles is not None:
            return tiles, faces
    return None, None


def phi_com_z(lat, mask=None):
    phi = lat.phi if mask is None else lat.phi[mask]
    pos = lat.pos if mask is None else lat.pos[mask]
    e = np.sum(phi**2, axis=1)
    total = e.sum()
    if total < 1e-30:
        return 0.0
    return float((e * pos[:, 2]).sum() / total)


def ring_energy(lat, tiles):
    return float(np.sum(lat.phi[tiles] ** 2))


def main():
    args = parser.parse_args()

    lat = DampedLattice(radius_cells=args.R, sponge_fraction=0.15, damp_max=0.003)
    pos = lat.pos
    dt = lat.stable_dt()
    a = _sq_hop_dist(lat)

    # Guide ring — at natural z=0 position
    guide_tiles, guide_faces = find_ring_near(lat, [a, 0.0, 0.0])
    if guide_tiles is None:
        print("ERROR: guide ring not found")
        sys.exit(1)

    guide_pos = pos[guide_tiles]
    guide_z = float(np.mean(guide_pos[:, 2]))
    guide_r = float(np.mean(np.linalg.norm(guide_pos[:, :2], axis=1)))

    # Electron ring — offset along z
    elec_tiles, elec_faces = find_ring_near(lat, [a, 0.0, args.z_sep])
    if elec_tiles is None:
        print(f"ERROR: no ring found near z={args.z_sep}")
        sys.exit(1)

    elec_pos = pos[elec_tiles]
    elec_z = float(np.mean(elec_pos[:, 2]))
    elec_r = float(np.mean(np.linalg.norm(elec_pos[:, :2], axis=1)))

    if set(guide_tiles) & set(elec_tiles):
        print("ERROR: guide and electron rings overlap — increase --z-sep")
        sys.exit(1)

    print(f"N={lat.N}  R={args.R}  dt={dt:.4e}  a={a:.3f}")
    print(f"guide  ring: z≈{guide_z:+.3f}  ρ≈{guide_r:.3f}  (frozen)")
    print(f"electron ring: z≈{elec_z:+.3f}  ρ≈{elec_r:.3f}  (free)")
    print(f"separation: Δz={elec_z - guide_z:.3f}")
    print("\nNo lam transport — pure lat.step.\n")

    # Init guide ring: frozen phi circulation — sustained flux source
    init_toroidal_winding(
        lat, guide_tiles, guide_faces, amplitude=args.amplitude, frozen=True
    )

    # Init electron ring: free resonant mode
    init_toroidal_winding(
        lat, elec_tiles, elec_faces, amplitude=args.amplitude, frozen=False
    )

    elec_mask = np.zeros(lat.N, dtype=bool)
    elec_mask[elec_tiles] = True

    elec_z0 = phi_com_z(lat, elec_mask)
    print(
        f"{'step':>8}  {'elec_COM_z':>12}  {'Δz_from_0':>12}  {'e_elec':>10}  {'phi_rms':>10}"
    )
    print("  " + "-" * 62)

    for s in range(args.n_steps + 1):
        if s % args.report == 0:
            ecz = phi_com_z(lat, elec_mask)
            e_elec = ring_energy(lat, elec_tiles)
            phi_rms = float(np.sqrt(np.mean(lat.phi**2)))
            print(
                f"{s:8d}  {ecz:+12.5f}  {ecz-elec_z0:+12.5f}  "
                f"{e_elec:10.4e}  {phi_rms:10.4e}",
                flush=True,
            )

        if s < args.n_steps:
            lat.step(dt, n_threads=args.n_threads)

    ecz_f = phi_com_z(lat, elec_mask)
    print(f"\nelectron ring z drift: {ecz_f - elec_z0:+.6f}")
    print("  (positive = drifted away from guide, negative = attracted)")
    print("done")


if __name__ == "__main__":
    main()
