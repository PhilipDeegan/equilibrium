"""
Casimir-Polder-style vacuum force between two point scatterers ("knot-knot")
on the Kelvin cell substrate.

Method: real-space (imaginary-frequency) Green's function via 3-D Brillouin
zone quadrature, reusing casimir.py's exact lattice dispersion relation
`_dispersion(Kx,Ky,Kz,a)` directly -- no `SubstrateLattice` instance needed,
same as casimir.py itself.

    G(r, xi) = INT_BZ cos(Kx*r) / (omega(K)^2 + xi^2) d^3K / (2*pi)^3

r is separation along the x-axis (the lattice is anisotropic, so direction
matters -- x is the arbitrary reference direction here, not a claim that all
directions are equivalent). BZ approximated as the cube [-pi/a, pi/a]^3: half
the BCC zone (the dispersion's period is 4 pi/a per axis; a full domain is
[-pi/a, pi/a]^2 x [-2 pi/a, 2 pi/a]). The missing half is gapped (omega > 0),
so only the short range is affected.

Numerics: naive equal-weight cube quadrature converges only algebraically
(~1/N) in the outer (Kx) direction because cos(Kx*r) is generically
incommensurate with the cube's periodicity -- a real risk at large r, unlike
the plate-Casimir case's clean analytic Euler-Maclaurin split. Instead the
outer Kx integral is done with scipy's Filon-type oscillatory quadrature
(`weight='cos'`), which handles the oscillation exactly; only the smooth
inner (Ky,Kz) average needs grid convergence (`Nyz`, checked via
`converged_G_point` by doubling).

Cross-check: the long-range (r >> a) limit of G(r,0) recovers the ordinary
Coulomb-like propagator 1/(4*pi*c^2*r), and G(r,xi) generally recovers the
Yukawa form G_continuum, with the SAME c = sqrt(10)*a already used as
casimir.py's continuum reference speed -- confirming the plate-Casimir and
point-scatterer effects come from one dispersion relation, not two unrelated
formulas.

Scope: scalar/monopole coupling (couples linearly to u), giving a ~1/r^3
vacuum energy via
    U(r) = -(1/(2*pi)) INT_0^inf G(r, xi)^2 dxi
NOT a literal reproduction of the EM dipole-dipole 1/r^6 (near-field) /
1/r^7 (retarded far-field) laws, which would need a vector (phi) coupling
instead.
"""

import numpy as np
from scipy import integrate

from equilibrium.substrate.casimir import _dispersion

from equilibrium.substrate.casimir import W_HEX, W_SQ  # noqa: E402

# c = C_COEFF * a: step()'s long-wavelength speed, c^2 = W_HEX^2 + W_SQ^2 in these units
# (sqrt10 at the pre-2026-09-28 W = 3:1, 2 at W = sqrt3:1).
C_COEFF = float(np.sqrt(W_HEX**2 + W_SQ**2))


def _H_transverse(Kx: float, xi: float, a: float, Nyz: int) -> float:
    """Smooth inner average: INT_(-pi/a)^(pi/a) INT dKy dKz / (omega(Kx,Ky,Kz)^2 + xi^2)."""
    dk = 2.0 * np.pi / (Nyz * a)
    k1d = np.arange(Nyz) * dk - np.pi / a + dk / 2.0
    Ky, Kz = np.meshgrid(k1d, k1d, indexing="ij")
    omega2 = _dispersion(Kx, Ky, Kz, a) ** 2
    return float(np.mean(1.0 / (omega2 + xi**2))) * (2.0 * np.pi / a) ** 2


def G_point(
    r: float, xi: float, a: float = 1.0, Nyz: int = 200, limit: int = 400
) -> float:
    """
    Real-space imaginary-frequency Green's function at separation r (along x)
    and imaginary frequency xi, on the Kelvin lattice.

    Outer Kx integral over the exact BZ boundary [0, pi/a] via Filon-type
    oscillatory quadrature (exploiting evenness in Kx to halve the range and
    move the sole singular point of the inner integral, at Kx=xi=0, to an
    endpoint); inner (Ky,Kz) integral via equal-weight grid (`Nyz` points
    per dimension) -- see module docstring for why the split is needed.
    """
    Kx_max = np.pi / a

    def h(kx):
        return _H_transverse(kx, xi, a, Nyz)

    if abs(r) < 1e-12:
        val, _ = integrate.quad(h, 0.0, Kx_max, limit=limit)
    else:
        val, _ = integrate.quad(h, 0.0, Kx_max, weight="cos", wvar=r, limit=limit)
    return 2.0 * val / (2.0 * np.pi) ** 3


def converged_G_point(
    r: float,
    xi: float,
    a: float = 1.0,
    Nyz0: int = 100,
    rtol: float = 1e-2,
    max_doublings: int = 3,
    limit: int = 400,
):
    """
    G_point with an explicit grid-doubling convergence check (Nyz -> 2*Nyz
    until the relative change is below `rtol` or `max_doublings` is hit).

    Returns (value, Nyz_used, converged).
    """
    Nyz = Nyz0
    prev = G_point(r, xi, a=a, Nyz=Nyz, limit=limit)
    for _ in range(max_doublings):
        Nyz *= 2
        cur = G_point(r, xi, a=a, Nyz=Nyz, limit=limit)
        if prev == 0.0 or abs(cur - prev) <= rtol * abs(cur):
            return cur, Nyz, True
        prev = cur
    return prev, Nyz, False


def G_continuum(r: float, xi: float, a: float = 1.0) -> float:
    """Continuum (Yukawa) limit: exp(-xi*r/c) / (4*pi*c^2*r), c = C_COEFF*a."""
    c = C_COEFF * a
    return np.exp(-xi * r / c) / (4.0 * np.pi * c**2 * r)


def U_continuum(r: float, a: float = 1.0) -> float:
    """
    Closed-form continuum scalar vacuum energy between two point scatterers,
    obtained analytically from U(r) = -(1/(2*pi)) INT_0^inf G_continuum(r,xi)^2 dxi:
        U_continuum(r) = -1 / (64 * pi^3 * c^3 * r^3),  c = C_COEFF*a
    """
    c = C_COEFF * a
    return -1.0 / (64.0 * np.pi**3 * c**3 * r**3)


def vacuum_energy_scalar(
    r: float,
    a: float = 1.0,
    Nyz: int = 200,
    n_xi: int = 64,
    xi_max_factor: float = 50.0,
    limit: int = 400,
) -> float:
    """
    U(r) = -(1/(2*pi)) INT_0^inf G(r, xi)^2 dxi.

    xi is integrated via a tan substitution (xi = xi_scale*tan(theta),
    theta in (0, pi/2)) so a finite grid covers the semi-infinite range;
    xi_scale = xi_max_factor * c/r sets the natural decay scale of G(r,xi)
    in xi (G falls off like exp(-xi*r/c) in the continuum limit). theta
    never touches 0, so xi never touches 0 -- xi=0 is where the transverse
    inner integral is only conditionally (log-)singular, and this scheme
    avoids evaluating exactly there.
    """
    c = C_COEFF * a
    xi_scale = xi_max_factor * c / max(r, a)
    dtheta = (np.pi / 2.0) / n_xi
    theta = (np.arange(n_xi) + 0.5) * dtheta
    xi_vals = xi_scale * np.tan(theta)
    dxi_dtheta = xi_scale / np.cos(theta) ** 2

    g_vals = np.array([G_point(r, xi, a=a, Nyz=Nyz, limit=limit) for xi in xi_vals])
    integral = float(np.sum(g_vals**2 * dxi_dtheta) * dtheta)
    return -integral / (2.0 * np.pi)
