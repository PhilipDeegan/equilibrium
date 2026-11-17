"""
Winding translation probe — pure DST gate test.

A free toroidal ring under a persistent u-gradient.  No auxiliary fields.

The face/cell asymmetry means translating the winding requires ALL ring tiles
to simultaneously relay their phi pattern to the adjacent tile layer.  Below a
threshold forcing this doesn't happen (phi oscillates, ring stays).  Above
threshold all 8 tiles cross together and phi_COM_z steps discretely.

The gate is not a stored field — it is an emergent collective condition on the
8-tile ring topology under the plain DST update rule.

Key columns:
  phi_z@ring  — mean (phi_13 - phi_12) at ring tiles: the net z-directed flux
  phi_COM_z   — phi-energy COM within ±5 cells of ring_z (watch for discrete hops)
  phi_rms     — global phi RMS (stability check)
  u_grad_rms  — RMS of the u_grad field (sanity)
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
from equilibrium.substrate.lattice import _FACE_ADJ, DampedLattice

parser = argparse.ArgumentParser()
parser.add_argument("--R", type=float, default=20.0)
parser.add_argument("--amplitude", type=float, default=1.0)
parser.add_argument(
    "--u-bias",
    type=float,
    default=1e-2,
    help="per-step additive z-gradient: u += bias*(1-z/R) - mean",
)
parser.add_argument(
    "--cross-scale",
    type=float,
    default=1e-3,
    help="face-adjacent crossing coupling (smaller = more resistance = more mass)",
)
parser.add_argument("--n-steps", type=int, default=20000)
parser.add_argument("--report", type=int, default=500)
parser.add_argument("--n-threads", type=int, default=1)


def find_ring_near(lat, target, axis="z", n_candidates=100):
    face_seq, _ = _TOROIDAL_PATTERNS[axis]
    dists = np.linalg.norm(lat.pos - np.asarray(target, float), axis=1)
    for start in np.argsort(dists)[:n_candidates]:
        tiles, faces = _trace_loop(lat, int(start), face_seq)
        if tiles is not None:
            return tiles, faces
    return None, None


def main():
    args = parser.parse_args()

    lat = DampedLattice(radius_cells=args.R, sponge_fraction=0.15, damp_max=0.003)
    pos = lat.pos
    dt = lat.stable_dt()
    R = float(args.R)
    a = _sq_hop_dist(lat)

    # One-shot z-gradient: applied once at t=0, then free evolution.
    # Conservative — no per-step energy injection.  The gradient drives φ
    # which back-reacts on u; ring either hops before gradient disperses or stays.
    u_grad = args.u_bias * (1.0 - pos[:, 2] / R)
    u_grad -= u_grad.mean()

    ring_tiles, ring_faces = find_ring_near(lat, [a, 0.0, 0.0])
    if ring_tiles is None:
        print("ERROR: ring not found")
        sys.exit(1)

    ring_z = float(np.mean(pos[ring_tiles, 2]))
    init_toroidal_winding(
        lat, ring_tiles, ring_faces, amplitude=args.amplitude, frozen=False
    )
    lat.u += u_grad  # one-shot: apply gradient once at t=0

    print(f"N={lat.N}  R={R}  dt={dt:.4e}")
    print(f"ring: {len(ring_tiles)} tiles  z≈{ring_z:+.3f}  (free)")
    print(f"u_bias={args.u_bias}  cross_scale={args.cross_scale}  (one-shot)")
    print(f"u_grad range: [{u_grad.min():+.4e}, {u_grad.max():+.4e}]")

    # Show what face 12 (-z sq) short-circuits to for ring tiles
    adj_12 = [j for j in _FACE_ADJ[12] if j >= 0]
    print(f"face-12 short-circuit exits: {adj_12}  (hex faces adjacent to -z sq)")
    print()

    def phi_z_ring():
        return float(np.mean(lat.phi[ring_tiles, 13] - lat.phi[ring_tiles, 12]))

    def phi_hex_ring():
        return float(np.mean(np.abs(lat.phi[ring_tiles, :8])))

    def phi_com_z():
        local = np.abs(pos[:, 2] - ring_z) < 5.0
        e = np.sum(lat.phi[local] ** 2, axis=1)
        t = e.sum()
        return float((e * pos[local, 2]).sum() / t) if t > 1e-30 else ring_z

    hdr = (
        f"{'step':>8}  {'phi_z@ring':>12}  {'phi_hex@ring':>13}  "
        f"{'phi_COM_z':>10}  {'phi_rms':>10}"
    )
    print(hdr)
    print("  " + "-" * 62)

    for s in range(args.n_steps + 1):
        if s % args.report == 0:
            print(
                f"{s:8d}  {phi_z_ring():+12.5e}  {phi_hex_ring():13.5e}  "
                f"{phi_com_z():+10.5f}  "
                f"{float(np.sqrt(np.mean(lat.phi**2))):10.4e}",
                flush=True,
            )

        if s < args.n_steps:
            lat.step(dt, n_threads=args.n_threads)
            lat.step_cross(dt, scale=args.cross_scale)

    print(f"\nfinal phi_z@ring:  {phi_z_ring():+.6e}")
    print(f"final phi_hex@ring:{phi_hex_ring():.6e}")
    print(f"final phi_COM_z:   {phi_com_z():+.6f}  (ring_z={ring_z:+.5f})")
    print("done")


if __name__ == "__main__":
    main()
