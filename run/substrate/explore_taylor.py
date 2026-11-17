"""
Taylor spheromak initialisation and decay experiments for DST.

The Taylor spheromak is the force-free MHD equilibrium inside a sphere of
radius R:  curl B = (4.493/R) B.  The analytical field is the spherical
Bessel j1 solution used by PHARE.  In PHARE, this decays via Taylor
relaxation (resistive dissipation of high-k helicity) into a dipole with a
residual open toroidal ring — the proposed neutron→proton beta-decay analogue.

Here we initialise the DST phi field from the same analytical expression and
watch whether a similar decay occurs under the DST update rule alone.

Mapping:
    PHARE B field  <->  DST phi face flux
    phi[i,k] = B(midpoint_{ik}) · n_hat(i->j)

Run with:
    PYTHONPATH=$PWD python3 run/substrate/explore_taylor.py
"""

import numpy as np
from scipy.special import spherical_jn

from equilibrium.substrate.lattice import DampedLattice, SubstrateLattice


def _phi_circulation(lat, centre, r_hop, dr=0.8):
    """Azimuthal phi circulation summed over faces on shell r_hop ± dr."""
    c = np.asarray(centre, dtype=float)
    nb_safe = np.where(lat.nb >= 0, lat.nb, 0)
    pos_j = lat.pos[nb_safe]
    pos_i = lat.pos[:, np.newaxis, :]
    mid = 0.5 * (pos_i + pos_j) - c
    r_mid = np.linalg.norm(mid, axis=-1)
    on_shell = lat._valid & (r_mid >= r_hop - dr) & (r_mid <= r_hop + dr)
    upper = on_shell & (np.arange(len(lat.pos))[:, None] < nb_safe)
    mx, my = mid[..., 0], mid[..., 1]
    rho = np.sqrt(mx**2 + my**2)
    rho_s = np.where(rho > 1e-10, rho, 1.0)
    diff = pos_j - pos_i
    d_len = np.linalg.norm(diff, axis=-1, keepdims=True)
    n_hat = diff / np.where(d_len > 0, d_len, 1.0)
    proj = n_hat[..., 0] * (-my / rho_s) + n_hat[..., 1] * (mx / rho_s)
    return float(np.sum(lat.phi[upper] * proj[upper]))


# ---------------------------------------------------------------------------
# Taylor spheromak phi initialiser
# ---------------------------------------------------------------------------


def init_taylor_spheromak_phi(
    lat: SubstrateLattice,
    centre: np.ndarray | None = None,
    R: float = 5.0,
    B0: float = 1.0,
) -> int:
    """
    Set lat.phi from the analytical Taylor (force-free) spheromak field.

    The spheromak field (Chandrasekhar-Kendall spherical solution):
        B_x = common * (z*x/r - y)
        B_y = common * (z*y/r + x)
        B_z = common * (-(x²+y²)/r)
        common = B0 * j1(k*r) / r,   k = 4.493/R

    phi[i,k] is set to the dot product of B at the face midpoint with the
    unit vector from tile i to tile j.  Faces outside the sphere are zeroed.

    Returns the number of faces initialised (non-zero phi set).
    """
    if centre is None:
        centre = np.zeros(3, dtype=float)
    centre = np.asarray(centre, dtype=float)

    k_bessel = 4.493 / R

    # Build neighbour positions, clamping -1 → 0 (guarded by lat._valid mask)
    nb_safe = np.where(lat.nb >= 0, lat.nb, 0)  # (N, 14)

    # Face midpoints relative to centre
    pos_i = lat.pos[:, np.newaxis, :]  # (N, 1, 3)
    pos_j = lat.pos[nb_safe]  # (N, 14, 3)
    mid = 0.5 * (pos_i + pos_j) - centre  # (N, 14, 3)

    dx, dy, dz = mid[..., 0], mid[..., 1], mid[..., 2]
    r = np.sqrt(dx**2 + dy**2 + dz**2)

    # Only initialise faces inside the sphere with a valid neighbour
    in_sphere = lat._valid & (r > 1e-10) & (r <= R)

    # Evaluate spherical Bessel j1 — use safe r to avoid divide-by-zero
    r_safe = np.where(in_sphere, r, 1.0)
    rho = k_bessel * r_safe
    j1r = spherical_jn(1, rho)
    common = np.where(in_sphere, B0 * j1r / r_safe, 0.0)

    bx = common * (dz * dx / r_safe - dy)
    by = common * (dz * dy / r_safe + dx)
    bz = common * (-(dx**2 + dy**2) / r_safe)

    # Face unit vectors i→j
    diff = pos_j - pos_i  # (N, 14, 3)
    d_len = np.linalg.norm(diff, axis=-1, keepdims=True)
    d_safe = np.where(d_len > 0, d_len, 1.0)
    n_hat = diff / d_safe  # (N, 14, 3)

    # phi = B · n_hat, zero outside sphere
    B_dot_n = bx * n_hat[..., 0] + by * n_hat[..., 1] + bz * n_hat[..., 2]
    lat.phi = np.where(in_sphere, B_dot_n, 0.0)

    return int(in_sphere.sum())


# ---------------------------------------------------------------------------
# Experiment: watch the Taylor spheromak decay in DST
# ---------------------------------------------------------------------------


def run_taylor_spheromak_decay(
    lam_target: float = 9.0,
    radius_cells: float = 25.0,
    R_sph: float = 6.0,
    B0: float = 1.0,
    run_steps: int = 15000,
    measure_every: int = 1000,
    r_scan: tuple = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18),
    sponge_fraction: float = 0.15,
    damp_max: float = 0.003,
    n_threads: int = 6,
):
    """
    Initialise DST with an analytical Taylor spheromak phi field and run.

    Initial conditions:
      - phi: Taylor spheromak within sphere of radius R_sph, zero outside
      - u:   zero everywhere
      - lam: lam_target within sphere (sigma = 1/(1+lam_target)); 0 (vacuum) outside

    lam_target=9 gives sigma=0.1 inside the sphere — 10× slower local time.
    lam_target=0 is pure vacuum everywhere (see run_taylor_spheromak_vacuum).

    Measurements every measure_every steps:
      - radial u profile (looking for moat at r > R_sph)
      - phi circulation at each r in r_scan (looking for toroidal winding)
      - total H = sum(u²) + 0.5*sum(phi²)
    """
    print(f"\n=== Taylor spheromak decay  (R={R_sph}, lam={lam_target}) ===")

    lat = DampedLattice(
        radius_cells=radius_cells,
        sponge_fraction=sponge_fraction,
        damp_max=damp_max,
    )
    centre = np.zeros(3, dtype=float)
    r_all = np.linalg.norm(lat.pos, axis=1)

    # Set winding within the spheromak sphere; vacuum outside
    in_sph = r_all <= R_sph
    lat.lam[in_sph] = float(lam_target)
    lat.lam[~in_sph] = 0.0

    # Initialise phi from Taylor spheromak
    n_faces = init_taylor_spheromak_phi(lat, centre=centre, R=R_sph, B0=B0)
    print(f"  N={lat.N}  R_sph={R_sph}  faces_init={n_faces}")
    print(f"  phi_max={lat.phi.max():.4f}  phi_min={lat.phi.min():.4f}")

    dt = lat.stable_dt()
    print(f"  dt={dt:.6f}")

    # Header
    r_cols = "  ".join(f"r={r:2d}" for r in r_scan)
    print(f"\n  step  |  H_total  |  u_rms_r<R  |  {r_cols}")
    print(f"  {'-'*80}")

    for step in range(0, run_steps + measure_every, measure_every):
        if step > 0:
            for _ in range(measure_every):
                lat.step(dt, n_threads)

        H = float(np.sum(lat.u**2) + 0.5 * np.sum(lat.phi**2))
        u_rms_core = float(np.sqrt(np.mean(lat.u[in_sph] ** 2)))

        circs = [_phi_circulation(lat, centre, r_hop=r) for r in r_scan]
        circ_str = "  ".join(f"{c:+.3e}" for c in circs)

        print(f"  {step:5d}  |  {H:.4e}  |  {u_rms_core:.4e}  |  {circ_str}")

    # Final radial u profile
    print("\n  Radial u profile (final):")
    n_bins = int(radius_cells * 0.8)
    for rb in range(1, n_bins, 2):
        mask = (r_all >= rb - 0.5) & (r_all < rb + 0.5)
        if mask.sum() > 0:
            u_mean = float(lat.u[mask].mean())
            u_rms = float(np.sqrt(np.mean(lat.u[mask] ** 2)))
            marker = " <-- moat?" if u_mean < -0.005 else ""
            print(f"    r={rb:3d}  u_mean={u_mean:+.5f}  u_rms={u_rms:.5f}{marker}")


def run_taylor_spheromak_vacuum(
    radius_cells: float = 25.0,
    R_sph: float = 6.0,
    B0: float = 1.0,
    run_steps: int = 15000,
    measure_every: int = 1000,
    r_scan: tuple = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18),
    sponge_fraction: float = 0.15,
    damp_max: float = 0.003,
    n_threads: int = 6,
):
    """
    Taylor spheromak in vacuum (lam=0 everywhere, sigma=1).

    Lesson from run_taylor_spheromak_decay: setting lam=lam_target inside the
    sphere slows the internal dynamics (lam=9 → sigma=0.1 → 10× slower).
    The spheromak barely evolves over 15000 steps.

    This variant uses lam=0 everywhere so the phi field evolves at full vacuum
    throughput.  We watch whether the DST wave dynamics alone produce any
    relaxation or moat — without the winding slow-down confounding the result.
    """
    print(f"\n=== Taylor spheromak (vacuum lam=0)  R={R_sph} ===")

    lat = DampedLattice(
        radius_cells=radius_cells,
        sponge_fraction=sponge_fraction,
        damp_max=damp_max,
    )
    centre = np.zeros(3, dtype=float)
    r_all = np.linalg.norm(lat.pos, axis=1)

    # lam=0 everywhere (default) — sigma=1
    n_faces = init_taylor_spheromak_phi(lat, centre=centre, R=R_sph, B0=B0)
    print(f"  N={lat.N}  R_sph={R_sph}  faces_init={n_faces}")
    print(f"  phi_max={lat.phi.max():.4f}  phi_min={lat.phi.min():.4f}")

    dt = lat.stable_dt()
    print(f"  dt={dt:.6f}")

    r_cols = "  ".join(f"r={r:2d}" for r in r_scan)
    print(f"\n  step  |  H_total  |  u_rms_r<R  |  {r_cols}")
    print(f"  {'-'*80}")

    in_sph = r_all <= R_sph
    for step in range(0, run_steps + measure_every, measure_every):
        if step > 0:
            for _ in range(measure_every):
                lat.step(dt, n_threads)

        H = float(np.sum(lat.u**2) + 0.5 * np.sum(lat.phi**2))
        u_rms_core = float(np.sqrt(np.mean(lat.u[in_sph] ** 2)))
        circs = [_phi_circulation(lat, centre, r_hop=r) for r in r_scan]
        circ_str = "  ".join(f"{c:+.3e}" for c in circs)
        print(f"  {step:5d}  |  {H:.4e}  |  {u_rms_core:.4e}  |  {circ_str}")

    print("\n  Radial u profile (final):")
    for rb in range(1, int(radius_cells * 0.8), 2):
        mask = (r_all >= rb - 0.5) & (r_all < rb + 0.5)
        if mask.sum() > 0:
            u_mean = float(lat.u[mask].mean())
            u_rms = float(np.sqrt(np.mean(lat.u[mask] ** 2)))
            marker = " <-- moat?" if u_mean < -0.005 else ""
            print(f"    r={rb:3d}  u_mean={u_mean:+.5f}  u_rms={u_rms:.5f}{marker}")


if __name__ == "__main__":
    run_taylor_spheromak_vacuum()
