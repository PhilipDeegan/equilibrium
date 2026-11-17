"""Volume-filling, two-lobed, genuinely curl-full spheromak IC — 2026-08-17,
the actual deliverable flagged (not yet built) at the top of `now`: the
wire-loop Hopf-pair diagnostic in `equilibrium.substrate.curl_full_spheromak_ic`
validated the step_cross-erosion MECHANISM on a bare 1-cell-wide filament
(u=0 everywhere, no volume, no thickness) — it is explicitly not a spheromak.
This module builds the real object: comparable in shape to
`equilibrium.substrate.dipole_mode_ic` (two lobes, u/phi filling the whole
volume) and to PHARE's Taylor spheromak (`run/phare_plasmoses.py`'s
`get_taylor_spheromak`, same K=4.493 eigenvalue), but with real, verifiable
circulation rather than being curl-free by construction.

Design: keep the poloidal and toroidal halves of the classic axisymmetric
l=1 Chandrasekhar-Kendall/Taylor force-free state STRICTLY SEPARATE and
additive, instead of PHARE's single non-separable Cartesian cross-product
formula (`bx = cf*(dz*dx/r - dy)`, etc., which mixes both terms together and
cannot be decomposed or independently curl-checked):

- Poloidal half (u, phi_pol): UNCHANGED call into `dipole_mode_ic.
  seed_dipole_ic` (u=U, phi_pol=grad(U) sampled at face midpoints, same
  K=4.493 mode). Known curl-free-in-the-continuum, real-but-modest
  discrete circulation (~0.21 relative to phi's own RMS, a finite-hop
  curvature artifact — see project_dst_discrete_curl memory). Kept as-is:
  this half already gives the two-lobed, volume-filling shape and is
  self-consistent with the wave equation.
- Toroidal half (phi_tor, NEW): `B_phi(r,theta) = j1(K*r)*sin(theta)` — the
  standard toroidal ingredient of the same l=1 force-free mode (Chandrasekhar
  & Kendall 1957), projected onto face normals via the azimuthal unit vector
  `phihat = (-y,x,0)/rho` — the SAME geometric primitive already used
  unmodified elsewhere in this codebase (`adaptive_flux_tube.
  apply_toroidal_rotation`/`phi_toroidal_circulation`), not a new Cartesian
  formula. At any fixed (rho,z) this term is constant all the way around the
  azimuthal loop (no idx-dependent sign reversal) — the same
  "constant-around-a-closed-loop" condition
  `test_ring_circulation_conservation.py` already proved CANNOT be any
  scalar's discrete derivative (loop-sum = amplitude*N_ring != 0) — so this
  term is genuinely, provably curl-full by the same standard already
  validated on the wire-loop diagnostic, not merely assumed because the
  continuum formula "should" be rotational (that assumption is exactly what
  burned the original |B|/B.n_hat IC and the sampled-gradient dipole phi —
  see dipole_mode_ic.py's and discrete_curl.py's docstrings). Verify with
  `equilibrium.substrate.discrete_curl.circulation`, don't just trust it.

`sin(theta) = rho/r` is symmetric under z -> -z (unlike U's z-antisymmetric
two-lobe pattern), so phi_tor forms a single toroidal band threading through
BOTH lobes of the poloidal field — matching the real Taylor-state/spheromak
picture (poloidal loops through both hemispheres, one toroidal twist
component), not a second independent dipole.
"""

import numpy as np
from scipy.special import spherical_jn

from equilibrium.substrate.dipole_mode_ic import K_FACTOR, seed_dipole_ic


def toroidal_envelope(pos, scale, centre=np.zeros(3)):
    """B_phi(r,theta) = j1(K*r)*sin(theta), z the polar axis (matches
    dipole_mode_ic's U=z*g(r) convention). Zero on-axis (rho=0) and at
    r=1 (K=4.493 is j1's first zero, same eigenvalue as the poloidal mode).
    """
    p = (pos - centre) / scale
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    rho = np.sqrt(x**2 + y**2)
    r = np.sqrt(x**2 + y**2 + z**2)
    r_safe = np.where(r > 0, r, 1.0)
    sin_theta = np.where(r > 0, rho / r_safe, 0.0)
    return spherical_jn(1, K_FACTOR * r_safe) * sin_theta


def _phihat(pos, centre=np.zeros(3)):
    """Unit azimuthal direction (-y,x,0)/rho about the z-axis -- zero
    on-axis. Same primitive as adaptive_flux_tube.apply_toroidal_rotation.
    """
    p = pos - centre
    rho = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2)
    on_axis = rho <= 0
    rho_safe = np.where(on_axis, 1.0, rho)
    tang = np.zeros_like(p)
    tang[:, 0] = -p[:, 1] / rho_safe
    tang[:, 1] = p[:, 0] / rho_safe
    tang[on_axis] = 0.0
    return tang


def _phi_tor_raw(pos, cell_size, scale, face_hop, face_norm, centre, batch):
    """Directed phi_tor at each leaf face, BEFORE amplitude normalization
    and boundary masking -- factored out so seed_curl_full_dipole_ic can do
    its two-pass (max-then-store) normalization like seed_dipole_ic does.
    """
    N = pos.shape[0]
    out = np.zeros((14, N), dtype=np.float64)
    for start in range(0, N, batch):
        end = min(start + batch, N)
        hop_off = face_hop[None, :, :] * cell_size[start:end, None, None]
        mid = (pos[start:end, None, :] + 0.5 * hop_off).reshape(-1, 3)
        A_mid = toroidal_envelope(mid, scale, centre).reshape(end - start, 14)
        tang_mid = _phihat(mid, centre).reshape(end - start, 14, 3)
        proj = np.einsum("ikd,kd->ik", tang_mid, face_norm)
        out[:, start:end] = (A_mid * proj).T
    return out


def seed_curl_full_dipole_ic(
    cl,
    pos,
    cell_size,
    scale,
    face_hop,
    face_norm,
    centre=np.zeros(3),
    r_sph=1.0,
    u_max=1.0,
    amplitude_pol=1.0,
    amplitude_tor=1.0,
    batch=500_000,
):
    """Seeds u (poloidal, same as seed_dipole_ic) and phi = phi_pol + phi_tor
    (poloidal grad(U) + genuinely-toroidal j1(Kr)*sin(theta) term), both
    masked to zero outside r<=r_sph (dual-zero convention: both the inner
    cell's outward face AND the outer cell's own faces are zeroed, so every
    boundary-crossing face pair presents (0,0) -- same requirement as
    init_spheromak_ic/seed_dipole_ic). amplitude_pol/amplitude_tor are
    independent peak-normalizations (max|phi_pol|=amplitude_pol,
    max|phi_tor|=amplitude_tor before summing), consistent with
    seed_dipole_ic's "each field to its own peak-1" convention.

    Returns (u_all, phi_pol, phi_tor, phi_full) -- the split is kept in the
    return value (not just the sum) so callers/tests can independently
    verify each half's circulation content.
    """
    centre = np.asarray(centre, dtype=float)
    N = pos.shape[0]
    r_cell = np.linalg.norm((pos - centre) / scale, axis=1)

    u_all, phi_pol = seed_dipole_ic(
        cl,
        pos,
        cell_size,
        scale,
        face_hop,
        face_norm,
        centre=centre,
        r_sph=r_sph,
        u_max=u_max,
        amplitude=amplitude_pol,
        batch=batch,
    )

    phi_tor_raw = _phi_tor_raw(
        pos, cell_size, scale, face_hop, face_norm, centre, batch
    )
    tor_max = float(np.abs(phi_tor_raw).max()) or 1.0
    phi_tor = phi_tor_raw * (amplitude_tor / tor_max)

    # Batched (not a single (N,14,3) temporary -- at production scale, e.g.
    # N~31.7M leaves, that intermediate alone is ~10GB and OOM-killed the
    # first attempt at this rebuild, 2026-08-17).
    for start in range(0, N, batch):
        end = min(start + batch, N)
        hop_off = face_hop[None, :, :] * cell_size[start:end, None, None]
        nbr_r = np.linalg.norm(
            (pos[start:end, None, :] + hop_off - centre) / scale, axis=2
        )
        phi_tor[:, start:end][(nbr_r > r_sph).T] = 0.0
    phi_tor[:, r_cell > r_sph] = 0.0

    phi_full = np.ascontiguousarray(phi_pol + phi_tor)
    cl.set_leaf_phi(phi_full)
    return u_all, phi_pol, phi_tor, phi_full
