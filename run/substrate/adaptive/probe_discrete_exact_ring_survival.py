"""
Discrete-exact single ring (unlinked, electron-like) vs continuum-sampled
double-ring dipole: does a GENUINELY topologically-protected winding
survive a Saha-style ambient background, where the continuum-sampled
double-ring construction (probe_electron_saha_survival.py) eroded to
~30% of its initial circulation?

Construction: find_toroidal_ring (mode="square", hops>=4, clear of the
BCC anisotropy band) + seed_constant_current -- a weighted (phi_k =
amplitude/W_k) flat current, the SAME discrete-exact construction already
proven in test_curl_divergence_functional.py / test_ring_circulation_
conservation.py to have exactly zero per-cell divergence and exactly
conserved circulation under plain step(). This is a single, UNLINKED ring
(no Hopf partner) -- structurally the electron-relevant case (neutron
decay/protection comes from linkage; this tests whether an unlinked but
genuinely discrete-exact loop is protected on its own).

Two phases:
  1. Clean vacuum, pure step() -- sanity check that circulation is exactly
     conserved (Kelvin's theorem), no noise, no cross_scale.
  2. Saha-style background (nonzero u_bg, sponge + reciprocal reinjection +
     boundary thermalize, same convention as probe_electron_saha_survival.py)
     -- the actual survival test, same protocol as the double-ring run for
     direct comparison.

Run:
  PYTHONPATH=$PWD python3 -u run/substrate/adaptive/probe_discrete_exact_ring_survival.py
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium import lattice as _lat
from tst.substrate.adaptive import compute_dt
from tst.substrate.adaptive.adaptive_flux_tube import (
    find_toroidal_ring,
    phi_ring_circulation,
)
from tst.substrate.adaptive.test_ring_circulation_conservation import (
    seed_constant_current,
)
from equilibrium.substrate.reciprocal_bc import ReciprocalBCCluster as ReciprocalBC

parser = argparse.ArgumentParser()
parser.add_argument("--R", type=float, default=40.0)
parser.add_argument("--hops", type=int, default=6)
parser.add_argument("--amplitude", type=float, default=1.0)
parser.add_argument(
    "--ac-amplitude",
    type=float,
    default=0.0,
    help="traveling-wave component added on top of the DC circulation "
    "(0 = pure DC, matches original test)",
)
parser.add_argument("--clean-steps", type=int, default=2000)
parser.add_argument("--saha-steps", type=int, default=8000)
parser.add_argument("--report", type=int, default=400)
parser.add_argument("--u-bg", type=float, default=0.05)
parser.add_argument("--sponge-frac", type=float, default=0.2)
parser.add_argument("--damp-max", type=float, default=0.01)
parser.add_argument("--thermalize-amp", type=float, default=0.02)
parser.add_argument("--n-threads", type=int, default=8)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()


def build_ring():
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    alat.set_n_threads(args.n_threads)
    cl = alat[0]
    _lat.build_bcc_sphere(cl, args.R, d1=1.0, u0=args.u_bg, ell0=1.0, lam0=0)
    alat.rebuild(0)
    ring_js, ring_faces = find_toroidal_ring(
        cl, axis="z", mode="square", hops=args.hops
    )
    if ring_js is None:
        raise RuntimeError(f"ring-finding failed at R={args.R}, hops={args.hops}")
    seed_constant_current(cl, ring_js, ring_faces, amplitude=args.amplitude)
    centre = np.array(cl.leaf_pos)[ring_js].mean(axis=0)
    return alat, cl, ring_js, centre


def main():
    # --- phase 1: clean vacuum sanity check -------------------------------
    print("=== phase 1: clean vacuum, pure step(), exact-conservation check ===")
    alat, cl, ring_js, centre = build_ring()
    dt = compute_dt(cl)
    circ_0 = phi_ring_circulation(cl, ring_js, centre)
    print(
        f"N={cl.n_leaves}  R={args.R}  hops={args.hops}  ring_size={len(ring_js)}  "
        f"circ_0={circ_0:.6f}  dt={dt:.4e}"
    )
    for step in range(1, args.clean_steps + 1):
        alat.step(0, dt)
        if step % args.report == 0 or step == args.clean_steps:
            c = phi_ring_circulation(cl, ring_js, centre)
            print(f"  step {step:6d}  circ={c:.6f}  circ/circ0={c / circ_0:.8f}")

    # --- phase 2: Saha-style background survival test ----------------------
    print("\n=== phase 2: Saha-style ambient background survival test ===")
    alat, cl, ring_js, centre = build_ring()
    r_all = np.linalg.norm(np.array(cl.leaf_pos), axis=1)
    r_inner = args.R * (1.0 - args.sponge_frac)
    cl.set_bc(
        _lat.BoundaryCondition.quadratic_sponge(
            target=args.u_bg, damp_max=args.damp_max, r_all=r_all, r_inner=r_inner
        )
    )
    bc = ReciprocalBC(cl, rng=np.random.default_rng(args.seed))
    dt = compute_dt(cl)
    circ_0 = phi_ring_circulation(cl, ring_js, centre)
    print(
        f"N={cl.n_leaves}  R={args.R}  r_inner(sponge)={r_inner:.1f}  "
        f"u_bg={args.u_bg}  circ_0={circ_0:.6f}  dt={dt:.4e}"
    )
    print(f"\n{'step':>6}  {'t':>9}  {'circ':>10}  {'circ/circ0':>10}  {'H/H0':>8}")
    u0, phi0 = np.array(cl.leaf_u), np.array(cl.leaf_phi)
    H0 = float(np.sum(u0**2) + 0.5 * np.sum(phi0**2))
    for step in range(1, args.saha_steps + 1):
        alat.step(0, dt)
        alat.step_sponge(0)
        bc.compensate(cl)
        bc.thermalize(cl, args.thermalize_amp)
        if step % args.report == 0 or step == args.saha_steps:
            c = phi_ring_circulation(cl, ring_js, centre)
            u, phi = np.array(cl.leaf_u), np.array(cl.leaf_phi)
            H = float(np.sum(u**2) + 0.5 * np.sum(phi**2))
            print(
                f"{step:6d}  {step * dt:9.2f}  {c:10.4f}  {c / circ_0:10.4f}  {H / H0:8.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
