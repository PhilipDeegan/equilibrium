"""
Hydrogen moat probe — proton as winding ring, electron as spontaneous u-deficit.

The proton IS the toroidal ring at the natural BCC z-position (r≈1.4 cells).
No electron is placed — the moat emerges from the proton ring's phi field.

  Phase 1 (warmup): seed 1/r u background; let phi field establish.
  Phase 2 (ring):   init phi circulation on proton ring; run step_lam;
                    scan for spontaneous moat_u < 0 (DST electron).
  Phase 3 (probe):  drop u-deficit at candidate radii along +z;
                    Bohr orbit = radius where perturbation deepens most.

Key metric: moat_r / ring_r → 2π ≈ 6.28  (n=1 Bohr condition).
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
parser.add_argument(
    "--u-central",
    type=float,
    default=0.1,
    help="u at proton ring; background seeded as u_central/r",
)
parser.add_argument(
    "--lam-proton",
    type=float,
    default=9.0,
    help="winding content at proton ring (chi≈0.9 → lam=9)",
)
parser.add_argument(
    "--amplitude",
    type=float,
    default=0.0,
    help="optional extra phi traveling-wave on proton ring (0=natural only)",
)
parser.add_argument(
    "--lam-scale", type=float, default=0.1, help="step_lam coupling strength"
)
parser.add_argument("--n-warmup", type=int, default=5000)
parser.add_argument("--n-ring", type=int, default=30000)
parser.add_argument("--n-probe", type=int, default=2000)
parser.add_argument("--probe-rmax", type=float, default=20.0)
parser.add_argument("--report", type=int, default=1000)
parser.add_argument("--n-threads", type=int, default=1)


def find_proton_ring(lat):
    """Find the natural BCC z-toroidal ring near origin."""
    a = _sq_hop_dist(lat)
    face_seq, _ = _TOROIDAL_PATTERNS["z"]
    ref = np.array([a, 0.0, 0.0])
    dists = np.linalg.norm(lat.pos - ref, axis=1)
    for start in np.argsort(dists)[:50]:
        tiles, faces = _trace_loop(lat, int(start), face_seq)
        if tiles is not None:
            return tiles, faces
    return None, None


def main():
    args = parser.parse_args()

    lat = DampedLattice(radius_cells=args.R, sponge_fraction=0.15, damp_max=0.003)
    dt = lat.stable_dt()
    pos = lat.pos
    r_all = np.linalg.norm(pos, axis=1)

    # Proton ring — natural BCC z-toroidal position near origin
    ring_tiles, ring_faces = find_proton_ring(lat)
    if ring_tiles is None:
        print("ERROR: proton ring not found")
        sys.exit(1)

    ring_pos = pos[ring_tiles]
    ring_r = float(np.mean(np.sqrt(ring_pos[:, 0] ** 2 + ring_pos[:, 1] ** 2)))
    ring_z = float(np.mean(ring_pos[:, 2]))

    sigma_p = 1.0 / (1.0 + args.lam_proton)
    print(f"N={lat.N}  R={args.R}  dt={dt:.4e}")
    print(
        f"proton ring: {len(ring_tiles)} tiles  r≈{ring_r:.3f}  z≈{ring_z:+.3f}"
        f"  λ={args.lam_proton:.1f}  σ={sigma_p:.4f}"
    )
    print(f"target moat_r ≈ {ring_r * 2 * np.pi:.2f}  (2π × ring_r, Bohr n=1)")

    # Seed 1/r Coulomb u background — pre-establishes equilibrium profile
    r_safe = np.where(r_all > ring_r, r_all, ring_r)
    lat.u[:] = args.u_central * ring_r / r_safe
    lat.lam[ring_tiles] = args.lam_proton
    # Freeze ring: proton is topologically stable — u and lam don't drain.
    # phi on ring tiles still evolves freely (fixed_u does not touch phi).
    for t in ring_tiles:
        lat.fixed_u[t] = True

    # ------------------------------------------------------------------
    # Phase 1 — warmup: u background settles, no phi circulation yet
    # ------------------------------------------------------------------
    print(f"\nPhase 1: warmup {args.n_warmup} steps …", flush=True)
    for s in range(args.n_warmup):
        lat.step(dt, n_threads=args.n_threads)
        if (s + 1) % 1000 == 0:
            phi_rms = float(np.sqrt(np.mean(lat.phi**2)))
            print(f"  {s+1}/{args.n_warmup}  phi_rms={phi_rms:.3e}", flush=True)

    # ------------------------------------------------------------------
    # Phase 2 — scan for moat from natural proton phi field
    # ------------------------------------------------------------------
    if args.amplitude > 0:
        init_toroidal_winding(lat, ring_tiles, ring_faces, amplitude=args.amplitude)
        print(f"\nPhase 2: added phi circulation (amplitude={args.amplitude})")
    else:
        print("\nPhase 2: natural phi only (no explicit circulation)")
    print()

    # Outer moat: skip ring near-field (r < ring_r*4); target r ≈ ring_r*2π
    OUTER_RMIN = ring_r * 4.0
    OUTER_RMAX = args.R * 0.6

    # Report radii bracketing the Bohr target (ring_r × 2π ≈ 8.76)
    REPORT_R = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0]

    # Z-axis masks: cells near the z-axis (|xy| < 0.6) at each report radius.
    # Ring has azimuthal symmetry → all ring cells equidistant from z-axis →
    # constructive interference → Bohr moat forms here.
    r_xy = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
    z_masks = []
    for rc in REPORT_R:
        m = np.where((np.abs(r_all - rc) < 0.5) & (r_xy < 0.6))[0]
        z_masks.append(m)

    def find_moat_outer():
        # Z-axis cone minimum in outer region
        mask = (r_all > OUTER_RMIN) & (r_all < OUTER_RMAX) & (r_xy < 1.0)
        if not mask.any():
            mask = (r_all > OUTER_RMIN) & (r_all < OUTER_RMAX)
        sub = np.where(mask)[0]
        idx = int(np.argmin(lat.u[sub]))
        return float(r_all[sub[idx]]), float(lat.u[sub[idx]])

    r_hdr = "".join(f"{'r='+str(int(rc)):>11}" for rc in REPORT_R)
    print(
        f"{'step':>8}  {'moat_r':>7}  {'moat_u':>12}  {'phi_rms':>10}{r_hdr}  (inst u, z-axis)"
    )
    print("  " + "-" * (42 + 11 * len(REPORT_R)))

    for s in range(args.n_ring + 1):
        if s % args.report == 0:
            mr, mu = find_moat_outer()
            u_vals = "".join(
                (
                    f"{float(lat.u[z_masks[k]].mean()):+11.3e}"
                    if len(z_masks[k])
                    else f"{'nan':>11}"
                )
                for k in range(len(REPORT_R))
            )
            print(
                f"{s:8d}  {mr:7.3f}  {mu:+12.5e}"
                f"  {float(np.sqrt(np.mean(lat.phi**2))):10.4e}{u_vals}",
                flush=True,
            )

        if s < args.n_ring:
            lat.step(dt, n_threads=args.n_threads)

    # ------------------------------------------------------------------
    # Phase 3 — moat probe: deepening at candidate radii along +z
    # ------------------------------------------------------------------
    u_saved = lat.u.copy()
    phi_saved = lat.phi.copy()
    lam_saved = lat.lam.copy()

    # Probe along +z axis: ring is in XY plane, all ring cells equidistant from
    # every z-axis point → coherent phi interference → Bohr moat forms here.
    z_axis = np.array([0.0, 0.0, 1.0])
    r_probe = list(np.arange(2.0, args.probe_rmax + 0.1, 1.0))
    U_DEF = -0.1 * args.u_central
    exclude = set(ring_tiles)

    print(f"\nPhase 3: moat probe  u_def={U_DEF:.3e}  n_probe={args.n_probe}")
    print(f"  {'r_target':>9}  {'r_actual':>9}  {'u_init':>10}  {'du_moat':>12}")
    print("  " + "-" * 48)

    probe_results = []
    for r_target in r_probe:
        dists = np.linalg.norm(pos - r_target * z_axis, axis=1)
        j_probe = next((int(i) for i in np.argsort(dists) if int(i) not in exclude), -1)
        if j_probe < 0:
            continue

        lat.u[:] = u_saved
        lat.phi[:] = phi_saved
        lat.lam[:] = lam_saved
        u_init = float(u_saved[j_probe])
        lat.u[j_probe] += U_DEF

        for _ in range(args.n_probe):
            lat.step(dt, n_threads=args.n_threads)
            lat.step_lam(dt, scale=args.lam_scale)

        du = float(lat.u[j_probe]) - u_init
        r_actual = float(r_all[j_probe])
        spont = " *" if u_init < 0 else ""
        bar = "▼" * min(int(abs(du) * 1000), 20) if du < U_DEF * 0.5 else ""
        print(
            f"  {r_target:9.1f}  {r_actual:9.3f}  {u_init:10.6f}  {du:+12.5e}  {bar}{spont}",
            flush=True,
        )
        probe_results.append((r_actual, u_init, du))

    if probe_results:
        # Spontaneous: most negative u_init in outer region (excluding ring near-field)
        outer_spont = [(r, u0, du) for r, u0, du in probe_results if r > OUTER_RMIN]
        if outer_spont:
            spont = min(outer_spont, key=lambda x: x[1])
            spont_r, spont_u0 = spont[0], spont[1]
            # Deepest deepening in outer region (most negative du_moat)
            orbit_r = min(outer_spont, key=lambda x: x[2])[0]
            print(
                f"\n  spontaneous moat  r = {spont_r:.3f}  u = {spont_u0:.3e}"
                f"  ratio = {spont_r/ring_r:.2f}  (target 2π={2*np.pi:.2f})"
            )
            print(
                f"  deepest deepening r = {orbit_r:.3f}"
                f"  ratio = {orbit_r/ring_r:.2f}"
            )

    print("\ndone")


if __name__ == "__main__":
    main()
