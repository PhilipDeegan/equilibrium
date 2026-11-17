"""
Phi-driven lam transport probe — uniform Python/C++ lattice.

Tests that Δlam_i = -σ_i · lam_i · Σ W_k · φ_{ik} moves winding in
response to phi imbalance and stays put when phi is balanced.

Phase 1 (warmup):  DampedLattice with set_winding knot; let u/phi settle.
Phase 2 (ring):    Inject free toroidal phi ring at natural BCC position.
Phase 3 (coupled): step() + step_lam() together; track lam COM and radial
                   profile to confirm winding redistribution follows phi flux.

Key check: lam_sum must be conserved (no sources/sinks), COM drift must be
consistent with net phi imbalance direction.
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")

from equilibrium.substrate.flux_tube import find_toroidal_ring, init_toroidal_winding
from equilibrium.substrate.lattice import DampedLattice
from tst.substrate import lam_com

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--R", type=float, default=20.0)
parser.add_argument("--lam", type=float, default=1.0, help="initial knot lam_0")
parser.add_argument("--u-central", type=float, default=1.0)
parser.add_argument("--amplitude", type=float, default=1.0, help="ring phi amplitude")
parser.add_argument("--n-warm", type=int, default=5000, help="warmup steps (no ring)")
parser.add_argument(
    "--n-steps", type=int, default=20000, help="coupled transport steps"
)
parser.add_argument(
    "--lam-scale", type=float, default=1e-3, help="step_lam scale factor"
)
parser.add_argument("--report", type=int, default=2000)


def lam_profile(pos, lam, r_edges):
    """Mean lam in each radial shell."""
    r = np.linalg.norm(pos, axis=1)
    vals = []
    for r0, r1 in zip(r_edges[:-1], r_edges[1:]):
        m = (r >= r0) & (r < r1)
        vals.append(float(lam[m].mean()) if m.any() else 0.0)
    return vals


def main():
    args = parser.parse_args()

    lat = DampedLattice(radius_cells=args.R, sponge_fraction=0.15, damp_max=0.003)
    N = lat.N
    pos = lat.pos
    r_all = np.linalg.norm(pos, axis=1)
    dt = lat.stable_dt()

    # Knot: origin tile
    origin_idx = int(np.argmin(r_all))
    origin = lat.tiles[origin_idx]

    # Set lam profile: 1/r well with lam_0 at the ring radius (~1.4 cells)
    # set_winding uses chi_0 — convert: chi_0 = lam_0/(1+lam_0), but set_winding
    # takes chi_0 and treats it as the amplitude at r=1, so pass args.lam directly
    # as chi_0 (valid for lam_0 << 1; for lam_0=1 this slightly overestimates chi).
    chi_0 = args.lam / (1.0 + args.lam)
    lat.set_winding(origin, chi_0=chi_0)

    # u field: 1/r Coulomb-like, pin origin
    for t in lat.tiles:
        r = float(np.linalg.norm(t.pos))
        lat.u[t.idx] = args.u_central / max(r, 0.5)
    lat.fixed_u[origin_idx] = True

    print(
        f"N={N}  R={args.R}  lam_0={args.lam:.4f}  dt={dt:.4e}  lam_scale={args.lam_scale}"
    )
    print(f"lam_sum_init = {lat.lam.sum():.4f}")

    # -----------------------------------------------------------------------
    # Phase 1 — warmup: u/phi settle, no ring, no lam transport
    # -----------------------------------------------------------------------
    print(f"\nwarmup {args.n_warm} steps …", flush=True)
    for s in range(1, args.n_warm + 1):
        lat.step(dt)
        if s % (args.n_warm // 4) == 0:
            print(
                f"  {s}/{args.n_warm}  u_mean={lat.u.mean():.4f}  phi_rms={np.sqrt(np.mean(lat.phi**2)):.4e}",
                flush=True,
            )
    print("warmup done\n")

    # -----------------------------------------------------------------------
    # Phase 2 — inject free toroidal ring
    # -----------------------------------------------------------------------
    ring_tiles, ring_faces = find_toroidal_ring(lat, axis="z")
    if ring_tiles is None:
        print("ERROR: no toroidal ring found")
        sys.exit(1)

    ring_pos = pos[ring_tiles]
    ring_r = float(np.mean(np.linalg.norm(ring_pos, axis=1)))
    print(f"ring: {len(ring_tiles)} tiles  r≈{ring_r:.3f}")

    init_toroidal_winding(
        lat, ring_tiles, ring_faces, amplitude=args.amplitude, frozen=False
    )

    phi_rms_init = float(np.sqrt(np.mean(lat.phi**2)))
    lam_sum_init = float(lat.lam.sum())
    com_0 = lam_com(pos, lat.lam)
    print(f"phi_rms after ring injection = {phi_rms_init:.4e}")
    print(f"lam_sum = {lam_sum_init:.4f}  COM = {com_0}")

    # Radial shells for profile

    print(
        f"\n{'step':>8}  {'lam_sum':>10}  {'phi_rms':>10}  {'COM_z':>8}  {'COM_r':>8}  {'lam@r1':>8}  {'lam@r5':>8}  {'lam@r10':>8}"
    )
    print("  " + "-" * 80)

    # -----------------------------------------------------------------------
    # Phase 3 — coupled step() + step_lam()
    # -----------------------------------------------------------------------
    for s in range(args.n_steps + 1):
        if s % args.report == 0:
            com = lam_com(pos, lat.lam)

            def lam_at(r_target):
                m = np.abs(r_all - r_target) < 0.6
                return float(lat.lam[m].mean()) if m.any() else 0.0

            phi_rms = float(np.sqrt(np.mean(lat.phi**2)))
            lam_sum = float(lat.lam.sum())
            com_r = float(np.linalg.norm(com[:2]))
            print(
                f"{s:8d}  {lam_sum:10.4f}  {phi_rms:10.4e}  {com[2]:8.4f}  {com_r:8.4f}"
                f"  {lam_at(1):8.5f}  {lam_at(5):8.5f}  {lam_at(10):8.5f}",
                flush=True,
            )

        if s < args.n_steps:
            lat.step(dt)
            lat.step_lam(dt, scale=args.lam_scale)

    com_f = lam_com(pos, lat.lam)
    drift = float(np.linalg.norm(com_f - com_0))
    print(f"\nCOM drift over {args.n_steps} steps: {drift:.6f}")
    print(
        f"lam_sum conservation: {lat.lam.sum():.4f} vs {lam_sum_init:.4f}  "
        f"(Δ={lat.lam.sum()-lam_sum_init:+.4f})"
    )
    print("done")


if __name__ == "__main__":
    main()
