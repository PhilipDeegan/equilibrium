"""
(phi, psi) candidate curl-conjugate pair -- a genuinely stable, structurally
derived (not bolted-on) alternative to a bare curl(curl(phi)) self-loop
(confirmed unconditionally unstable, doc/dst/failures/naive-reverse-curl-
term-unconditionally-unstable.txt).

Structure
---------
psi lives one value per (cell, triangle-family) -- 12 families per cell,
see equilibrium.substrate.discrete_curl's TRIANGLES / triangle_triples.
It forms a second symplectic pair with phi's curl-carrying content,
mirroring how (u, phi) are already a genuine conjugate pair:

    dpsi_f/dt   = -circ_D(phi)_f          (Faraday-like)
    dphi_ik/dt += +c2 * scatter_DT(psi)_ik (Ampere-like)

circ_D/scatter_DT here are WEIGHTED by the SAME W_HEX/W_SQ face weights the
base (u, phi) system already uses on both of its own legs (du/dt=-sum W_k
phi_k, dphi/dt=-W_k grad(u)) -- every TRIANGLES family (h1, h2, sqk) has
h1, h2 always hex-type (weight W_HEX) and mirror(sqk) always sq-type
(weight W_SQ), confirmed exactly from discrete_curl.py's construction, so
the weighting is the same (W_HEX, W_HEX, W_SQ) triple on every family, no
per-family branching needed. scatter_DT is built as the EXACT transpose of
circ_D (same per-leg weight applied when depositing back onto each
destination face) -- this is the correct adjoint pairing, reusing values
the lattice already defines, not a new free parameter.

Status (exp/2026/09/08/{09_34_c2_speed_matching_sweep,10_32_weighted_circ_
scatter,11_11_resolution_convergence}, see `now`): calibrating c2 (the one
free coefficient -- the DST-native analogue of Maxwell's eps0*mu0=1/c^2,
found empirically rather than assumed from a continuum formula) to c2~0.7
produces a curl-carrying disturbance that co-propagates with a real
traveling (u,phi) wave at ~90% of the wave's own phase velocity -- a huge
improvement over an uncalibrated/unweighted coupling (which leaves B
essentially frozen where deposited, an unbounded trailing wake). The
residual ~9-12% velocity deficit does NOT shrink with resolution (checked
at n_cells_per_wavelength = 8/12/16) -- it is a genuine, resolution-
independent property of this specific construction at c2=0.7, not a
discretization artifact. Whether that residual is itself a real predicted
signature of this candidate mechanism, or a sign that some other structural
piece is still missing, is open -- not resolved here. Per project
convention (doc/dst.md), this is emergent SCAFFOLDING for testing a
candidate mechanism against real phenomenology, not a claimed final theory
of DST magnetism.
"""

import numpy as np

from equilibrium.substrate.lattice import W_HEX, W_SQ

N_FAMILIES = 12

# Empirically calibrated coefficient (see module docstring / `now` for the
# calibration sweep) -- NOT derived from a continuum formula, found by
# direct measurement on the discrete lattice. Callers may override.
C2_CALIBRATED = 0.7


def circ_D(phi, triples, N, mirror):
    """Faraday-like leg: weighted circulation of phi around each (cell,
    triangle-family) instance, W_HEX on the two hex legs, W_SQ on the
    mirrored-sq leg (see module docstring for why this weighting is exact
    for every family, not an approximation)."""
    out = np.zeros((N, N_FAMILIES))
    for f, ((h1, h2, sqk), a, b, c) in enumerate(triples):
        m = mirror[sqk]
        out[a, f] = W_HEX * phi[a, h1] + W_HEX * phi[b, h2] + W_SQ * phi[c, m]
    return out


def scatter_DT(psi_dense, phi_shape, triples, mirror):
    """Ampere-like leg: the EXACT transpose of circ_D -- same per-leg
    weight applied when depositing psi back onto the 3 faces that
    contributed to its circulation."""
    inc = np.zeros(phi_shape)
    for f, ((h1, h2, sqk), a, b, c) in enumerate(triples):
        m = mirror[sqk]
        v = psi_dense[a, f]
        np.add.at(inc, (a, np.full_like(a, h1)), W_HEX * v)
        np.add.at(inc, (b, np.full_like(b, h2)), W_HEX * v)
        np.add.at(inc, (c, np.full_like(c, m)), W_SQ * v)
    return inc


def step_psi_pair(lat, psi_dense, dt, triples, mirror, c2=C2_CALIBRATED):
    """One symplectic-Euler sub-step of the (phi, psi) pair, to be composed
    AFTER the caller's own base (u, phi) step (e.g. lat.step(dt) or the
    manual symplectic update used in the calibration probes) each tick.

    Mutates nothing in place; returns (new_psi_dense, dphi_extra) so the
    caller controls exactly how dphi_extra is folded into lat.phi (matching
    how the calibration probes add it alongside the base step's own dphi).
    """
    vals = circ_D(lat.phi, triples, lat.N, mirror)
    psi_new = psi_dense - dt * vals
    dphi_extra = dt * c2 * scatter_DT(psi_new, lat.phi.shape, triples, mirror)
    return psi_new, dphi_extra
