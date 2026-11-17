"""
Classical (continuum) constant-current loop antenna -- the reference target
for the DST Poynting-ring radiation calibration (project_poynting_dipole_
calibration memory), which so far only ever compared against the naive
point-magnetic-dipole far-field law S(theta) ~ sin^2(theta)/r^2. That
comparison assumes the source is small compared to its own wavelength
(ka << 1); DST's ring is NOT (ka ~ 1.04, ring_r/lambda ~ 0.166 at every
tested hops value) -- this module is the finite-size (any ka) far-field
generalisation, standard antenna theory (Balanis, "Antenna Theory", loop
antennas), not a new derivation.

Far-field radiation pattern of a circular loop of radius `a` carrying a
spatially uniform (in-phase, "constant current" -- matching
tst.substrate.adaptive.test_ring_circulation_conservation.seed_constant_
current, n=0 azimuthal mode) time-harmonic current, at angle theta from the
loop axis, wavenumber k = 2*pi/lambda:

    S(theta; ka) ~ sin^2(theta) * [ J_1(ka*sin(theta)) / (ka*sin(theta)) ]^2

As ka -> 0, J_1(x)/x -> 1/2 (constant), recovering the point-dipole
sin^2(theta) pattern exactly. At finite ka this reshapes the ANGULAR
weighting -- it does NOT change the radial 1/r^2 far-field falloff, which
is fixed by energy conservation (geometric spreading) for any compact
source regardless of ka. This module only provides the angular shape
function; it is not a full near-to-far-zone solution (that needs the
general vector-multipole/spherical-Hankel-function treatment, not attempted
here -- see the module docstring's caveat in the accompanying discussion).

No lattice stepping -- module + tests only, same convention as casimir.py /
casimir_polder.py.
"""

import numpy as np
from scipy.special import j1


def pattern_shape(theta, ka):
    """
    Normalised angular power pattern of a finite constant-current loop,
    S(theta)/S(pi/2, ka->0) up to an overall (unfitted) constant -- i.e.
    sin^2(theta) * [2*J_1(ka*sin(theta)) / (ka*sin(theta))]^2.

    theta : array_like, radians, angle from the loop axis (0 = pole)
    ka    : loop radius * wavenumber (2*pi*a/lambda) -- dimensionless

    At ka=0 this returns exactly sin(theta)**2 (the point-dipole pattern).
    NOTE: the bracket must be normalised by 2 -- J_1(x)/x on its own tends
    to 1/2 (not 1) as x->0, which would leave a spurious constant 0.25
    factor relative to the point-dipole limit (caught by
    tst/substrate/test_loop_antenna.py's ka=0 exact-match test).
    """
    theta = np.asarray(theta, dtype=np.float64)
    s = np.sin(theta)
    x = ka * s
    near_zero = np.abs(x) < 1e-12
    safe_x = np.where(near_zero, 1.0, x)  # avoid 0/0; result discarded there anyway
    bracket = np.where(near_zero, 1.0, 2.0 * j1(safe_x) / safe_x)
    return (s * bracket) ** 2


def point_dipole_shape(theta):
    """The ka->0 limit, sin^2(theta) -- kept separate so callers can compare
    both explicitly rather than trusting pattern_shape's ka=0 branch alone."""
    return np.sin(np.asarray(theta, dtype=np.float64)) ** 2


def far_zone_slope():
    """
    The radial power-law exponent S ~ r^exponent in the TRUE far zone
    (kr >> 1), for ANY compact source regardless of ka -- fixed by energy
    conservation / geometric spreading through a sphere, not specific to a
    point dipole. This is the calibration target for a large-r rerun of the
    Poynting-ring probe: both polar and equatorial fitted slopes should
    converge to this value once kr is large enough, independent of the
    ring's finite size. The angular WEIGHTING (how much power goes to each
    theta) is what ka reshapes -- not this exponent.
    """
    return -2.0


def kr(r, wavelength):
    """k*r, the standard far-zone figure of merit (>>1 required to trust
    the asymptotic far-field pattern/slope at radius r)."""
    return 2.0 * np.pi * np.asarray(r, dtype=np.float64) / wavelength
