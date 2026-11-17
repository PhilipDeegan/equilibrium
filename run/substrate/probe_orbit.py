"""
Electron orbit probe — frozen proton source + free electron ring.

A single frozen high-u cell at the origin acts as a proton placeholder.
Its u field decays as ~1/r from the source, providing a persistent
attractive gradient without injecting energy into the rest of the domain.

The electron ring starts at z=+z_start above the source and is free to
translate toward it via step_cross.  No per-step energy injection.

Key columns:
  sep_z      — ring phi-COM z minus source z: tracks approach/orbit
  phi_z@ring — net z-directed flux at ring tiles (coupling signal)
  phi_COM_z  — phi-energy weighted z of the full domain
  phi_rms    — global phi RMS (should stay flat once source field equilibrates)
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
parser.add_argument("--R", type=float, default=40.0)
parser.add_argument("--amplitude", type=float, default=1.0)
parser.add_argument(
    "--u-source",
    type=float,
    default=0.2,
    help="frozen u at source cell (proton placeholder)",
)
parser.add_argument(
    "--z-start", type=float, default=5.0, help="initial ring z offset above source"
)
parser.add_argument(
    "--cross-scale", type=float, default=1e-3, help="face-adjacent crossing coupling"
)
parser.add_argument("--n-steps", type=int, default=20000)
parser.add_argument("--report", type=int, default=500)
parser.add_argument("--n-threads", type=int, default=1)


def find_ring_near(lat, target, n_candidates=100):
    face_seq, _ = _TOROIDAL_PATTERNS["z"]
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

    # Proton placeholder: cell nearest to origin, frozen at u_source
    src_idx = int(np.argmin(np.linalg.norm(pos, axis=1)))
    src_z = float(pos[src_idx, 2])
    lat.u[src_idx] = args.u_source
    lat.fixed_u[src_idx] = True

    # Electron ring: toroidal z-loop starting z_start above source
    ring_target = [a, 0.0, src_z + args.z_start]
    ring_tiles, ring_faces = find_ring_near(lat, ring_target)
    if ring_tiles is None:
        print("ERROR: ring not found near", ring_target)
        sys.exit(1)

    ring_z0 = float(np.mean(pos[ring_tiles, 2]))
    init_toroidal_winding(
        lat, ring_tiles, ring_faces, amplitude=args.amplitude, frozen=False
    )

    print(f"N={lat.N}  R={R}  dt={dt:.4e}")
    print(f"source: cell {src_idx}  z={src_z:+.3f}  u={args.u_source}")
    print(
        f"ring: {len(ring_tiles)} tiles  z0={ring_z0:+.3f}  sep0={ring_z0 - src_z:+.3f}"
    )
    print(f"cross_scale={args.cross_scale}")
    print()

    def phi_z_ring():
        return float(np.mean(lat.phi[ring_tiles, 13] - lat.phi[ring_tiles, 12]))

    def phi_com_z():
        # phi-energy weighted z, windowed around current estimated ring position
        e = np.sum(lat.phi**2, axis=1)
        t = e.sum()
        return float((e * pos[:, 2]).sum() / t) if t > 1e-30 else ring_z0

    hdr = (
        f"{'step':>8}  {'sep_z':>8}  {'phi_z@ring':>12}  "
        f"{'phi_COM_z':>10}  {'phi_rms':>10}"
    )
    print(hdr)
    print("  " + "-" * 58)

    for s in range(args.n_steps + 1):
        if s % args.report == 0:
            cz = phi_com_z()
            sep = cz - src_z
            print(
                f"{s:8d}  {sep:+8.3f}  {phi_z_ring():+12.5e}  "
                f"{cz:+10.5f}  "
                f"{float(np.sqrt(np.mean(lat.phi**2))):10.4e}",
                flush=True,
            )

        if s < args.n_steps:
            lat.step(dt, n_threads=args.n_threads)
            lat.step_cross(dt, scale=args.cross_scale)

    cz = phi_com_z()
    print(f"\nfinal sep_z:     {cz - src_z:+.4f}  (started {ring_z0 - src_z:+.4f})")
    print(f"final phi_COM_z: {cz:+.6f}")
    print(f"final phi_rms:   {float(np.sqrt(np.mean(lat.phi**2))):.4e}")
    print("done")


if __name__ == "__main__":
    main()
