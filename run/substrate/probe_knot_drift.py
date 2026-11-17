"""
Pure DST knot drift probe.

Sets up a toroidal phi circulation (the winding) plus a background u gradient,
then runs only lat.step — no lambda transport, no J_lam, no auxiliary fields.

Question: does the knot (phi-energy COM) drift under a u gradient using only
the fundamental DST update rule?

Tracks:
  phi_COM_z  — centre of mass of phi energy in z (global)
  u_COM_z    — centre of mass of |u| in z (isolates the knot from background)
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")

from equilibrium.substrate.flux_tube import find_toroidal_ring, init_toroidal_winding
from equilibrium.substrate.lattice import DampedLattice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--R", type=float, default=20.0)
    parser.add_argument(
        "--amplitude", type=float, default=1.0, help="knot phi/u amplitude"
    )
    parser.add_argument(
        "--u-bias",
        type=float,
        default=1e-2,
        help="per-step additive z-gradient: u += bias*(1-z/R) - mean",
    )
    parser.add_argument(
        "--n-warm",
        type=int,
        default=2000,
        help="steps to establish gradient before tracking",
    )
    parser.add_argument("--n-steps", type=int, default=20000)
    parser.add_argument("--report", type=int, default=1000)
    parser.add_argument("--n-threads", type=int, default=1)

    args = parser.parse_args()

    lat = DampedLattice(radius_cells=args.R, sponge_fraction=0.15, damp_max=0.003)
    pos = lat.pos
    dt = lat.stable_dt()
    R = args.R

    u_grad_source = args.u_bias * (1.0 - pos[:, 2] / R)
    u_grad_source -= u_grad_source.mean()

    ring_tiles, ring_faces = find_toroidal_ring(lat, axis="z")
    if ring_tiles is None:
        print("ERROR: no toroidal ring found")
        sys.exit(1)

    init_toroidal_winding(
        lat, ring_tiles, ring_faces, amplitude=args.amplitude, frozen=False
    )

    ring_r = float(np.mean(np.linalg.norm(pos[ring_tiles], axis=1)))

    print(f"N={lat.N}  R={R}  dt={dt:.4e}  amplitude={args.amplitude}")
    print(f"ring: {len(ring_tiles)} tiles  r≈{ring_r:.3f}")
    print(f"u_bias={args.u_bias}  n_warm={args.n_warm}  n_steps={args.n_steps}")
    print("\nNo lam transport — pure lat.step only.\n")

    print(f"warmup {args.n_warm} steps …", flush=True)
    for _ in range(args.n_warm):
        lat.step(dt, n_threads=args.n_threads)
        lat.u += u_grad_source
    print("warmup done\n")

    def phi_com_z(lat):
        e = np.sum(lat.phi**2, axis=1)
        total = e.sum()
        if total < 1e-30:
            return 0.0
        return float((e * pos[:, 2]).sum() / total)

    def u_com_z(lat):
        w = np.abs(lat.u)
        total = w.sum()
        if total < 1e-30:
            return 0.0
        return float((w * pos[:, 2]).sum() / total)

    phi_com_0 = phi_com_z(lat)
    u_com_0 = u_com_z(lat)

    print(f"{'step':>8}  {'phi_COM_z':>10}  {'u_COM_z':>10}  {'phi_rms':>10}")
    print("  " + "-" * 50)

    for s in range(args.n_steps + 1):
        if s % args.report == 0:
            pcz = phi_com_z(lat)
            ucz = u_com_z(lat)
            phi_rms = float(np.sqrt(np.mean(lat.phi**2)))
            print(f"{s:8d}  {pcz:+10.5f}  {ucz:+10.5f}  {phi_rms:10.4e}", flush=True)

        if s < args.n_steps:
            lat.step(dt, n_threads=args.n_threads)
            lat.u += u_grad_source

    print(f"\nphi_COM drift: {phi_com_z(lat) - phi_com_0:+.6f}")
    print(f"u_COM drift:   {u_com_z(lat) - u_com_0:+.6f}")
    print("done")


if __name__ == "__main__":
    main()
