"""
Follow-up to today's directional-current null: `curl(phi)=B` is a PROVABLY
frozen invariant under plain step() (E_full=-grad(u)-dphi/dt=0 identically
=> d(curl phi)/dt=0 for any field) -- so no test of "does a current SOURCE
a field" can ever succeed under plain dynamics. The productive question
instead: given TWO already-curl-carrying objects (real, reachable states --
frozen-flux toroidal rings, same construction validated against the
classical current-loop field shape in test_curl_loop.py), does DST's field
ENERGY correctly encode the classical magnetic dipole-dipole interaction?

Two coaxial toroidal rings ("pretend protons" -- NOT a lam-only proxy,
per [[project_dst_lam_definition]]/the deleted 2026-09-02 lam-blob-knot
work: lam without real phi circulation is a state step_lam could never
produce. A frozen phi flux loop IS a real, reachable, externally-maintained
state -- exactly what set_flux_loop/fixed_phi already models, a "battery-
held" current loop, same category as the already-validated static wire/
loop tests) with SAME-sense (parallel moments along the connecting axis --
classically attractive) vs OPPOSITE-sense (antiparallel -- classically
repulsive) circulation, at a range of separations.

Because the loops are FIXED FLUX (fixed_phi=True), not fixed-current-via-
battery, the naive total field energy 0.5*sum(phi^2) is the theoretically
correct quantity to compare (no battery-work/energy-paradox correction
needed, unlike the classical fixed-current case -- see Griffiths' magnetic
energy sign subtlety, which only applies to fixed-CURRENT loops). With
u=0 (no charge anywhere, matching the real magnetostatic vacuum solution
for a pure current loop with no scalar potential source), H=0.5*sum(phi^2)
exactly -- no time-stepping needed for the static part.

Classical target: dipole-dipole interaction energy for two coaxial
dipoles separated along their shared axis:
    U_int = (mu0/4*pi*d^3) * (m1.m2 - 3*(m1.dhat)*(m2.dhat))
          = -2*mu0*m1*m2/(4*pi*d^3)   [same-sense, moments along d-axis]
          = +2*mu0*m1*m2/(4*pi*d^3)   [opposite-sense]
i.e. same-sense should have LOWER total field energy than opposite-sense,
at every finite separation, with the difference falling off ~1/d^3 and
vanishing as d->infinity (self-energies dominate, common to both).
"""

import sys

import numpy as np

sys.path.insert(0, ".")
import equilibrium.substrate.flux_tube as ft
from equilibrium.substrate.curl import b_vec_vectorized
from equilibrium.substrate.lattice import SubstrateLattice

R = 25.0
SEPARATIONS = [6.0, 8.0, 10.0, 12.0, 16.0, 20.0]
N_SETTLE_STEPS = 2000
REPORT = 1000


def find_ring_near(lat, axis, center, mode="minimal", hops=1, n_candidates=200):
    """Same construction as flux_tube.find_toroidal_ring, but searchable
    around an arbitrary center instead of near the lattice origin."""
    if mode == "minimal":
        face_seq, ref_dir = ft._TOROIDAL_PATTERNS[axis]
        scale = 1
    elif mode == "square":
        face_seq = ft._square_face_seq(axis, hops)
        ref_dir = ft._AXIS_REF_DIR[axis]
        scale = hops
    else:
        raise ValueError(mode)
    a = ft._sq_hop_dist(lat)
    target = np.asarray(center, dtype=np.float64) + ref_dir * a * scale
    dists = np.linalg.norm(lat.pos - target, axis=1)
    for start in np.argsort(dists)[:n_candidates]:
        tiles, faces = ft._trace_loop(lat, int(start), face_seq)
        if tiles is not None:
            return tiles, faces
    return None, None


def build_pair(lat, d, sense):
    A_tiles, A_faces = find_ring_near(lat, "z", center=(0.0, 0.0, -d / 2.0))
    B_tiles, B_faces = find_ring_near(lat, "z", center=(0.0, 0.0, +d / 2.0))
    if A_tiles is None or B_tiles is None:
        return None, None
    ft.init_toroidal_winding(lat, A_tiles, A_faces, amplitude=1.0, frozen=True)
    ft.init_toroidal_winding(lat, B_tiles, B_faces, amplitude=sense * 1.0, frozen=True)
    return A_tiles, B_tiles


def field_energy(lat):
    return 0.5 * float(np.sum(lat.phi**2))


def single_ring_self_energy():
    lat = SubstrateLattice(radius_cells=R)
    tiles, faces = find_ring_near(lat, "z", center=(0.0, 0.0, 0.0))
    ft.init_toroidal_winding(lat, tiles, faces, amplitude=1.0, frozen=True)
    return field_energy(lat)


def static_scan():
    H_self = single_ring_self_energy()
    print(f"single-ring self-energy H_self={H_self:.6e} (each ring, both same)\n")
    print(f"{'d':>6} {'H_same':>12} {'H_opp':>12} {'cross_same':>12} "
          f"{'cross_opp':>12} {'sum_check':>12} {'d^3*cross_same':>16}")
    rows = []
    for d in SEPARATIONS:
        lat_s = SubstrateLattice(radius_cells=R)
        A, B = build_pair(lat_s, d, +1.0)
        if A is None:
            print(f"  d={d}: ring construction failed, skipping")
            continue
        H_same = field_energy(lat_s)

        lat_o = SubstrateLattice(radius_cells=R)
        build_pair(lat_o, d, -1.0)
        H_opp = field_energy(lat_o)

        cross_same = H_same - 2 * H_self
        cross_opp = H_opp - 2 * H_self
        sum_check = cross_same + cross_opp  # should be ~0 (bilinear form antisymmetry)
        d3_cross = d**3 * cross_same
        rows.append((d, H_same, H_opp, cross_same, cross_opp))
        print(f"{d:6.1f} {H_same:12.4e} {H_opp:12.4e} {cross_same:12.4e} "
              f"{cross_opp:12.4e} {sum_check:12.4e} {d3_cross:16.4e}")

    print(f"\nclassical expectation: cross_same < 0 < cross_opp (same-sense "
          f"attractive/lower-energy), |cross| falling off ~1/d^3 (d^3*cross "
          f"roughly constant across separations if so).")
    return rows


def midpoint_curl(lat):
    """|curl(phi)| at the cell nearest the origin (midpoint between the two
    rings, on their shared axis) -- tests whether real step() dynamics ever
    builds up nonzero circulation there, or whether it stays pinned at its
    t=0 value (0) as the E_full=-grad(u)-dphi/dt=0 identity predicts for ANY
    plaquette loop not touching one of the rings' own frozen edges."""
    i = int(np.argmin(np.linalg.norm(lat.pos, axis=1)))
    B = b_vec_vectorized(lat.pos, lat.nb, lat.phi)[i]
    return float(np.linalg.norm(B)) if np.isfinite(B).all() else float("nan")


def dynamic_check(d, sense, n_steps):
    """Freeze the loops, let u/phi relax under real step(), track total H
    and the midpoint curl(phi) over time -- checks whether the static
    (u=0, no time-stepping) energy comparison is stable, and directly
    tests the prediction that curl(phi) between the rings can never
    change from 0 under plain step(), per this afternoon's frozen-B
    identity (only loop-edge phi is frozen by fixed_phi -- everything in
    between is free to evolve, so this is a real test, not a tautology)."""
    lat = SubstrateLattice(radius_cells=R)
    build_pair(lat, d, sense)
    dt = lat.stable_dt()
    print(f"\ndynamic check: d={d}, sense={sense:+.0f}, dt={dt:.4e}, N={lat.N}")
    for step in range(n_steps + 1):
        if step % REPORT == 0:
            H = float(np.sum(lat.u**2) + 0.5 * np.sum(lat.phi**2))
            u_rms = float(np.sqrt(np.mean(lat.u**2)))
            b_mid = midpoint_curl(lat)
            print(f"  {step:6d}  H={H:.6e}  u_rms={u_rms:.6e}  |B|_mid={b_mid:.6e}")
        lat.step(dt)
    return lat


if __name__ == "__main__":
    static_scan()
    dynamic_check(10.0, +1.0, N_SETTLE_STEPS)
    dynamic_check(10.0, -1.0, N_SETTLE_STEPS)
