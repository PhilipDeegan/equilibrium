"""
Queued item from `now` (top of file, 2026-09-05): the two-point-source
vacuum-current probe (exp/2026/09/05/14_45_vacuum_current_bfield/) confirmed
DST reproduces the classical B=0 target for a spherically-symmetric current
-- a clean null, but by the pseudovector/reflection symmetry argument it can
NEVER see a nonzero field, no matter how the mechanism works. This probe
builds the genuinely DIRECTIONAL case: a real current confined to a long
thin channel (not radiating symmetrically), driven by real step() dynamics
(not a hand-set analytic phi, unlike tst/substrate/test_curl.py's static
straight-wire check), and tests actual Ampere's law -- does the ALREADY-
VALIDATED discrete curl(phi)=B reconstruction (equilibrium.substrate.curl,
matches classical wire/loop profiles exactly, see test_curl.py/
test_curl_loop.py) produce a circulating B around this real current whose
magnitude satisfies B_phi(rho) = I_enc(rho)/(2*pi*rho) -- the actual
Ampere's-law relation, using the ENCLOSED CURRENT MEASURED DIRECTLY FROM THE
SIMULATION at each radius, not an assumed uniform-current-density profile --
at the same (2/sqrt(3))^2 hop-distance^2 calibration constant found for the
static wire/loop tests.

Geometry: DampedLattice.cylinder (built exactly for "two points separated
along one axis by a large distance" per its own docstring) -- a long thin
domain, radius_cells << half_length_cells, so current between two fixed-u
sources placed near the ends is geometrically forced to flow mostly along
the axis rather than spread out spherically. No hard confinement/topology
hack -- the directionality comes from the domain shape alone, same
"battery terminal" fixed_u mechanism as the point-source probe.
"""

import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium.substrate.curl import b_vec_vectorized
from equilibrium.substrate.lattice import DampedLattice

R_PERP = 8.0
HALF_L = 40.0
Z_SRC = 25.0
U0 = 1.0
N_STEPS = 20000
REPORT = 1000
N_THREADS = 8
MIDPLANE_HALF_WIDTH = 1.0
DATA_OUT = "exp/2026/09/05/17_44_directional_current_ampere/directional_current_state.npz"


def build(lat):
    pos = lat.pos
    src_a = np.array([0.0, 0.0, -Z_SRC])
    src_b = np.array([0.0, 0.0, Z_SRC])
    ia = int(np.argmin(np.linalg.norm(pos - src_a, axis=1)))
    ib = int(np.argmin(np.linalg.norm(pos - src_b, axis=1)))
    lat.u[ia] = U0
    lat.u[ib] = -U0
    lat.fixed_u[ia] = True
    lat.fixed_u[ib] = True
    return ia, ib


def crossing_current(lat, rho_max=None):
    """Signed flux crossing the z=0 plane (positive = +z direction), summed
    over cells within rho_max of the axis if given (else the whole plane)."""
    pos, nb = lat.pos, lat.nb
    near = np.abs(pos[:, 2]) <= MIDPLANE_HALF_WIDTH
    if rho_max is not None:
        rho = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
        near = near & (rho <= rho_max)
    valid = nb >= 0
    safe_nb = np.where(valid, nb, 0)
    nbr_z = pos[safe_nb, 2]
    z_i = pos[:, 2, None].repeat(14, axis=1)
    crossing = valid & near[:, None] & (
        ((z_i < 0) & (nbr_z > 0)) | ((z_i > 0) & (nbr_z < 0))
    )
    sign = np.where(z_i < 0, 1.0, -1.0)
    W = np.where(np.arange(14) < 8, 3.0, 1.0)
    return float(np.sum(np.where(crossing, sign * W[None, :] * lat.phi, 0.0)))


def run():
    lat = DampedLattice.cylinder(
        radius_cells=R_PERP, half_length_cells=HALF_L, axis=2,
        sponge_fraction=0.15, damp_max=0.003,
    )
    ia, ib = build(lat)
    dt = lat.stable_dt()
    print(f"N={lat.N}  dt={dt:.4e}  src_a={lat.pos[ia]}  src_b={lat.pos[ib]}", flush=True)

    for step in range(N_STEPS + 1):
        if step % REPORT == 0:
            I = crossing_current(lat)
            H = float(np.sum(lat.u**2) + 0.5 * np.sum(lat.phi**2))
            print(f"  {step:6d}  I_z0={I:+.6e}  H={H:.6e}", flush=True)
        lat.step(dt, n_threads=N_THREADS)

    np.savez(
        DATA_OUT,
        pos=lat.pos, phi=lat.phi, u=lat.u, lam=lat.lam,
        ia=ia, ib=ib, R_PERP=R_PERP, HALF_L=HALF_L, Z_SRC=Z_SRC, U0=U0,
        n_steps=N_STEPS,
    )
    print(f"state saved -> {DATA_OUT}", flush=True)
    return lat, ia, ib


def analyze(lat, ia, ib):
    pos, nb = lat.pos, lat.nb
    B = b_vec_vectorized(pos, nb, lat.phi)
    have = np.isfinite(B).all(axis=1)

    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    rho = np.sqrt(x**2 + y**2)

    hop_sq_factor = (2.0 / np.sqrt(3.0)) ** 2

    # settled cross-section: near z=0 (midway between the two sources, far
    # from both end sponges), inside the radial sponge's inner edge
    # (0.85*R_PERP with sponge_fraction=0.15)
    near_plane = have & (np.abs(z) <= 1.5) & (rho > 0.5) & (rho < 0.75 * R_PERP)

    b_azimuthal = (-B[:, 0] * y + B[:, 1] * x) / np.where(rho > 0, rho, 1.0)
    b_radial_perp = (B[:, 0] * x + B[:, 1] * y) / np.where(rho > 0, rho, 1.0)
    b_z = B[:, 2]

    print(f"\ncells with finite B: {int(have.sum())}/{len(have)}")
    print(f"near-plane cells: {int(near_plane.sum())}")

    print(f"\nnear z=0 plane, whole measured band (rho in (0.5, {0.75*R_PERP:.1f})):")
    print(f"  B_azimuthal rms={np.sqrt(np.mean(b_azimuthal[near_plane]**2)):.6e}  "
          f"B_radial(perp) rms={np.sqrt(np.mean(b_radial_perp[near_plane]**2)):.6e}  "
          f"B_z rms={np.sqrt(np.mean(b_z[near_plane]**2)):.6e}")

    I_total = crossing_current(lat)
    print(f"\nI crossing full z=0 plane: {I_total:+.6e}")

    edges = np.linspace(1.0, 0.75 * R_PERP, 8)
    print(f"\nAmpere's-law check: B_azimuthal(rho) vs I_enc(rho)/(2*pi*rho), "
          f"expect a constant ratio ~ hop_sq_factor={hop_sq_factor:.4f} "
          f"(same calibration as the static wire/loop tests):")
    print(f"  {'rho_lo':>8} {'rho_hi':>8} {'I_enc':>12} {'B_phi_meas':>12} "
          f"{'B_phi_target':>12} {'ratio':>10}")
    ratios = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = near_plane & (rho >= lo) & (rho < hi)
        if in_bin.sum() < 3:
            continue
        rho_mid = 0.5 * (lo + hi)
        I_enc = crossing_current(lat, rho_max=rho_mid)
        b_phi_target = I_enc / (2.0 * np.pi * rho_mid)
        b_phi_meas = float(np.mean(b_azimuthal[in_bin]))
        ratio = b_phi_meas / b_phi_target if abs(b_phi_target) > 1e-12 else float("nan")
        ratios.append(ratio)
        print(f"  {lo:8.2f} {hi:8.2f} {I_enc:12.4e} {b_phi_meas:12.4e} "
              f"{b_phi_target:12.4e} {ratio:10.4f}")

    ratios = np.array([r for r in ratios if np.isfinite(r)])
    if len(ratios):
        print(f"\nratio mean={ratios.mean():.4f}  std={ratios.std():.4f}  "
              f"(hop_sq_factor={hop_sq_factor:.4f})")
    else:
        print("\nno valid bins to summarize")

    return B, near_plane


if __name__ == "__main__":
    lat, ia, ib = run()
    analyze(lat, ia, ib)
