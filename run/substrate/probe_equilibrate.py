"""
Proton field equilibration — produce a stable initial condition for hydrogen runs.

Pins the proton (fixed u + λ) in a RadiationBCLattice which preserves the
Coulomb 1/r tail at the boundary instead of absorbing it.  After an initial
build-up phase, freeze_bc() locks in the fitted A and u_bg so subsequent
steps maintain u = A/r + u_bg at the BC shell.

Once phi_rms plateaus (outgoing flux from source == BC absorption), the
snapshot is saved.  probe_hydrogen.py loads it via --snapshot so every run
starts in the live field without waiting for propagation.

Output:
  <output>.npz  — full lattice state (u, phi, lam, fixed_u, ...) ready for
                   RadiationBCLattice.load_snapshot()
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")

from equilibrium.substrate.lattice import RadiationBCLattice

parser = argparse.ArgumentParser()
parser.add_argument("--R", type=float, default=40.0)
parser.add_argument("--u-proton", type=float, default=1.3e-3)
parser.add_argument("--lam-proton", type=float, default=100.0)
parser.add_argument(
    "--n-build",
    type=int,
    default=5000,
    help="steps before freeze_bc() — field propagates outward",
)
parser.add_argument(
    "--n-run",
    type=int,
    default=20000,
    help="steps after freeze_bc() — run until phi_rms plateaus",
)
parser.add_argument("--report", type=int, default=1000)
parser.add_argument("--n-threads", type=int, default=1)
parser.add_argument("--output", type=str, default="tst/substrate/proton_field.npz")


def phi_rms(lat):
    return float(np.sqrt(np.mean(lat.phi**2)))


def main():
    args = parser.parse_args()

    lat = RadiationBCLattice(radius_cells=args.R)
    dt = lat.stable_dt()

    src_idx = int(np.argmin(np.linalg.norm(lat.pos, axis=1)))
    src_z = float(lat.pos[src_idx, 2])
    lat.u[src_idx] = args.u_proton
    lat.lam[src_idx] = args.lam_proton
    lat.fixed_u[src_idx] = True
    lat.refresh_masks()  # exclude proton from BC shell and fit band

    sigma_p = 1.0 / (1.0 + args.lam_proton)
    print(f"N={lat.N}  R={args.R}  dt={dt:.4e}")
    print(
        f"proton: cell {src_idx}  z={src_z:+.3f}  "
        f"u={args.u_proton:.2e}  λ={args.lam_proton:.1f}  σ={sigma_p:.4f}"
    )
    print(f"field reaches R in ~{int(args.R / dt)} steps")
    print()

    # Phase 1: build-up — field propagates, BC not yet frozen
    print(f"phase 1: build-up ({args.n_build} steps) ...")
    for s in range(args.n_build):
        lat.step(dt, n_threads=args.n_threads)
        if (s + 1) % args.report == 0:
            print(
                f"  step {s+1:6d}  phi_rms={phi_rms(lat):.4e}  "
                f"u_src={lat.u[src_idx]:.4e}",
                flush=True,
            )

    lat.freeze_bc()
    print(f"freeze_bc: A={lat._A:.4e}  u_bg={lat._u_bg:.4e}")
    print()

    # Phase 2: run with frozen Coulomb BC until phi_rms plateaus
    print(f"phase 2: equilibrate ({args.n_run} steps) ...")
    prev_rms = phi_rms(lat)
    for s in range(args.n_run):
        lat.step(dt, n_threads=args.n_threads)
        if (s + 1) % args.report == 0:
            rms = phi_rms(lat)
            drms = rms - prev_rms
            print(f"  step {s+1:6d}  phi_rms={rms:.4e}  Δ={drms:+.4e}", flush=True)
            prev_rms = rms

    lat.save(args.output, step_dt=dt)
    print(f"\nsnapshot saved → {args.output}")
    print(f"final phi_rms: {phi_rms(lat):.4e}  A={lat._A:.4e}")
    print("done")


if __name__ == "__main__":
    main()
