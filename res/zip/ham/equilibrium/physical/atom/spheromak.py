"""
Taylor spheromak u-field profile for the DST neutron model.

Scale and resolution note
─────────────────────────
The neutron — like the proton — is a resonant pattern that spans an
enormous number of Planck cells (~10⁵⁸ at Planck scale).  A single
simulation cell is itself a coarse-grained stand-in for billions of
Planck cells; the spheromak profile here describes the *envelope* of
the winding across the simulation lattice, not individual cell steps.
One or two simulation cells can only ever be an approximation of that
collective pattern.  The profile functions in this module operate at
simulation scale (SI or oscillator units) and should be understood in
that coarse-grained sense.

Topology
────────
The neutron is a toroidal knot: a closed loop of circulating u-flux
whose minimal BCC carrier is the triangular plaquette formed by
2 hexagonal faces + 1 square face.  There are 72 such plaquettes in
the Kelvin lattice.  The winding closes on itself, so no net outward flux
field escapes — charge neutrality by topology, not cancellation.

The stable winding is the Woltjer-Taylor force-free state: the
configuration that minimises substrate impedance for a given winding
number, subject to the boundary constraint that u = 0 at radius R.

Confinement quantization
────────────────────────
The winding fits inside a sphere of radius R when:

    k R = 4.493    (first zero of the spherical Bessel function j₁)

This is not an empirical fit.  It is the condition that makes u vanish
at the boundary without external confinement — the winding is
self-confining.  k = 4.493 / R.

u-field profile (scalar magnitude approximation)
────────────────────────────────────────────────
    u(r) = B₀ × j₁(kr) / r,    r ≤ R
    u(r) = 0,                    r > R

j₁(0) = 0  →  u vanishes at the centre  (no divergence at origin)
j₁(kR) = 0  →  u vanishes at the boundary  (self-confined)

Peak at kr ≈ 2.08, i.e. r ≈ R / 2.16.

Internal structure
──────────────────
The characteristic length scale of the internal mode: k⁻¹ = R / 4.493

For a neutron of radius R ≈ 0.86 fm: k⁻¹ ≈ 0.19 fm.  Deep inelastic
scattering probes this scale and sees sub-structure.  In DST this is
the internal j₁ mode of the toroidal u-flux, not a separate constituent
— there are no quarks.

Decay
─────
The spheromak is metastable.  Left to evolve freely, it unravels into a
toroidal dipole (the true minimum-impedance state).  In PHARE simulations
this occurs at t ≈ 3 (60 000 timesteps).  In DST the corresponding event
is beta decay: the toroidal winding opens, the compensating moat
re-expands to the Bohr orbit (emitted electron), and the topological
deficit closes as an antineutrino.

Full toroidal implementation
────────────────────────────
The scalar magnitude profile captures the impedance distribution but not
the circulation topology.  A full implementation assigns SIGNED u values
around each 2-hex+1-sq triangular plaquette such that the face fluxes
(u_j − u_i) / d_ij reproduce the PHARE vector components:

    face flux ≈ B(r) · n̂_face

This is an inverse problem: find u(cell) such that the discrete gradient
matches the Taylor spheromak vector field.
"""

import numpy as np
from scipy.special import spherical_jn

# First zero of j₁ — the confinement quantization constant
J1_FIRST_ZERO = 4.493


def taylor_u_profile(r, R, B0=1.0):
    """
    Scalar u-field magnitude for the Taylor spheromak: u(r) = B₀ j₁(kr)/r.

    Parameters
    ----------
    r  : array of radial distances
    R  : confinement radius (same units as r)
    B0 : peak amplitude

    Returns
    -------
    u  : array, same shape as r
    """
    k = J1_FIRST_ZERO / R
    u = np.zeros_like(r, dtype=float)
    mask = (r <= R) & (r > 0)
    u[mask] = B0 * spherical_jn(1, k * r[mask]) / r[mask]
    return u


def taylor_vector_field(pos, centre, R, B0=1.0):
    """
    Full Taylor spheromak vector field (Bx, By, Bz) at each position.

    This is the PHARE get_taylor_spheromak formula translated to numpy arrays.
    In DST the three components give the TARGET face fluxes along x, y, z
    directions — used when initialising the full toroidal u-circulation.

    Parameters
    ----------
    pos    : (N, 3) positions
    centre : (3,) centre of spheromak
    R      : confinement radius
    B0     : amplitude

    Returns
    -------
    bx, by, bz : (N,) arrays — target face flux components
    """
    dx = pos[:, 0] - centre[0]
    dy = pos[:, 1] - centre[1]
    dz = pos[:, 2] - centre[2]
    r = np.sqrt(dx**2 + dy**2 + dz**2)

    bx = np.zeros(len(pos))
    by = np.zeros(len(pos))
    bz = np.zeros(len(pos))

    mask = (r <= R) & (r > 0)
    k = J1_FIRST_ZERO / R
    rho = k * r[mask]

    j1_rho = spherical_jn(1, rho)
    cfactor = B0 * j1_rho / r[mask]

    bx[mask] = cfactor * (dz[mask] * dx[mask] / r[mask] - dy[mask])
    by[mask] = cfactor * (dz[mask] * dy[mask] / r[mask] + dx[mask])
    bz[mask] = cfactor * (-(dx[mask] ** 2 + dy[mask] ** 2) / r[mask])

    return bx, by, bz


def confinement_radius_from_k(k):
    """R = 4.493 / k  (inverse of the quantization condition)."""
    return J1_FIRST_ZERO / k


def internal_wavelength(R):
    """
    Characteristic length of the internal j₁ mode: k⁻¹ = R / 4.493.

    For a neutron of radius R ≈ 0.86 fm → k⁻¹ ≈ 0.19 fm.
    This is the scale DIS experiments probe as 'quark' sub-structure in DST.
    """
    return R / J1_FIRST_ZERO


def peak_radius(R):
    """Radius at which u(r) is maximum: kr ≈ 2.082 → r ≈ R / 2.16."""
    return R * 2.082 / J1_FIRST_ZERO
