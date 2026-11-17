"""
Hydrogen probe — frozen proton (high u + high λ) + free electron ring.

The proton is a constant, persistent winding knot: fixed_u=True freezes
both u and λ from t=0 (lattice.py:384).  No warmup needed — the proton's
state is topologically stable and the Coulomb field propagates outward
naturally during the run.  The ring starts cold and begins feeling the
proton's field once it arrives (~z_start/c steps).

High λ → σ = 1/(1+λ) ≈ 0 at the core → DST time-dilation barrier.

Physics question: does the λ barrier produce a stable non-zero equilibrium
separation (Bohr-like orbit), or does the ring still collapse to zero?

Key columns:
  sep_z      — ring phi-COM z minus proton z (watch for stable non-zero value)
  phi_z@ring — net z-directed flux at ring (negative=attracted, positive=repelled)
  phi_COM_z  — windowed phi-energy COM tracking the ring
  phi_rms    — global phi RMS (stability)
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
from equilibrium.substrate.lattice import DampedLattice, RadiationBCLattice

parser = argparse.ArgumentParser()
parser.add_argument("--R", type=float, default=40.0)
parser.add_argument("--amplitude", type=float, default=1.0)
parser.add_argument(
    "--u-proton", type=float, default=1.3e-3, help="frozen u at proton cell"
)
parser.add_argument(
    "--lam-proton",
    type=float,
    default=100.0,
    help="frozen λ at proton (σ = 1/(1+λ) barrier strength)",
)
parser.add_argument(
    "--z-start", type=float, default=5.0, help="initial ring z above proton"
)
parser.add_argument(
    "--snapshot",
    type=str,
    default=None,
    help="pre-equilibrated proton field from probe_equilibrate.py",
)
parser.add_argument("--cross-scale", type=float, default=1e-3)
parser.add_argument("--n-steps", type=int, default=40000)
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

    if args.snapshot:
        lat = RadiationBCLattice.load_snapshot(args.snapshot)
        dt = lat.saved_step_dt or lat.stable_dt()
        print(
            f"loaded snapshot: {args.snapshot}  phi_rms={float(np.sqrt(np.mean(lat.phi**2))):.4e}"
        )
    else:
        lat = DampedLattice(radius_cells=args.R, sponge_fraction=0.15, damp_max=0.003)
        dt = lat.stable_dt()
        # Proton: cell nearest origin — freeze both u and λ from t=0
        src_idx = int(np.argmin(np.linalg.norm(lat.pos, axis=1)))
        lat.u[src_idx] = args.u_proton
        lat.lam[src_idx] = args.lam_proton
        lat.fixed_u[src_idx] = True  # also freezes lam (lattice.py:384)

    pos = lat.pos
    R = float(args.R)
    a = _sq_hop_dist(lat)
    src_idx = int(np.argmin(np.linalg.norm(pos, axis=1)))
    src_z = float(pos[src_idx, 2])

    # Electron ring: toroidal z-loop at z_start above proton
    ring_target = [a, 0.0, src_z + args.z_start]
    ring_tiles, ring_faces = find_ring_near(lat, ring_target)
    if ring_tiles is None:
        print("ERROR: ring not found near", ring_target)
        sys.exit(1)

    ring_z0 = float(np.mean(pos[ring_tiles, 2]))
    init_toroidal_winding(
        lat, ring_tiles, ring_faces, amplitude=args.amplitude, frozen=False
    )

    sigma_p = 1.0 / (1.0 + args.lam_proton)
    print(f"N={lat.N}  R={R}  dt={dt:.4e}")
    print(
        f"proton: cell {src_idx}  z={src_z:+.3f}  "
        f"u={args.u_proton:.2e}  λ={args.lam_proton:.1f}  σ={sigma_p:.4f}"
    )
    print(
        f"ring: {len(ring_tiles)} tiles  z0={ring_z0:+.3f}  sep0={ring_z0 - src_z:+.3f}"
    )
    print(f"field reaches ring in ~{int(args.z_start / dt)} steps")
    print()

    # Narrow-window phi-energy COM — ±WIN cells around last known z.
    # WIN=2 keeps the window tight enough to exclude the proton at z=0
    # while the ring is at sep>2.  Slides with the ring after each hop.
    WIN = 2.0
    current_z = ring_z0

    def ring_com_z():
        nonlocal current_z
        mask = np.abs(pos[:, 2] - current_z) <= WIN
        e = np.sum(lat.phi[mask] ** 2, axis=1)
        t = e.sum()
        if t > 1e-30:
            cz = float((e * pos[mask, 2]).sum() / t)
            current_z = cz  # slide window for next call
            return cz
        return current_z

    def phi_z_ring():
        mask = np.abs(pos[:, 2] - current_z) <= WIN
        return float(np.mean(lat.phi[mask, 13] - lat.phi[mask, 12]))

    hdr = f"{'step':>8}  {'sep_z':>8}  {'phi_z@ring':>12}  {'phi_rms':>10}"
    print(hdr)
    print("  " + "-" * 40)

    for s in range(args.n_steps + 1):
        if s % args.report == 0:
            cz = ring_com_z()
            sep = cz - src_z
            print(
                f"{s:8d}  {sep:+8.3f}  {phi_z_ring():+12.5e}  "
                f"{float(np.sqrt(np.mean(lat.phi**2))):10.4e}",
                flush=True,
            )

        if s < args.n_steps:
            lat.step(dt, n_threads=args.n_threads)
            lat.step_cross(dt, scale=args.cross_scale)

    cz = ring_com_z()
    print(f"\nfinal sep_z:     {cz - src_z:+.4f}  (started {ring_z0 - src_z:+.4f})")
    print(f"final phi_rms:   {float(np.sqrt(np.mean(lat.phi**2))):.4e}")
    print("done")


if __name__ == "__main__":
    main()
