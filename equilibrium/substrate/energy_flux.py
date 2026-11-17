"""
Local discrete energy-flux vector for the (u, phi) substrate.

Derived directly from the real update rule (equilibrium/substrate/lattice.py
SubstrateLattice.step(), step_cpp and the C++ AdaptiveLattice step(): sigma is relational, taken
at the face on both legs since 2026-09-29):
    du_i/dt      = -sum_k sigma_face_k * W_k * phi[i,k]
    dphi[i,k]/dt = -sigma_face_k * W_k * (u_nb - u_i)
The per-cell energy E_i = u_i^2/2 + (1/4)*sum_k phi[i,k]^2
obeys an EXACT local continuity law in continuous time:
    dE_i/dt = -sum_k S_i,k          S_i,k = sigma_face_k * W_k * phi[i,k] * ubar_k
where ubar_k = (u_i + u_nb)/2 is the face-average scalar value. Verified
numerically against the real step() (nonuniform sigma, random field): the
residual between the actual per-step dE_i and -dt*sum_k(S_i,k) scales
cleanly as O(dt^2) (tst/substrate/test_energy_flux.py) -- the leftover is the expected symplectic-
Euler splitting error, not a wrong formula. See doc/mems/
project_dst_em_field_convention_crisis.md for why this replaces the
previously-used (and three-times-falsified) E:=-dphi/dt, B:=curl(phi),
S:=ExB convention: DST's core (u at cell, phi on faces) is structurally a
SCALAR/acoustic-style staggered wave scheme, not a full vector-EM one --
S_i,k is a scalar flux per face (like acoustic intensity p*v), not a cross
product. B=curl(phi) (the 4-face plaquette loop) is a separate, composite
quantity that plays no role in this local energy law.

This module is backend-agnostic: it only needs cell positions, the 14-slot
neighbor index array (-1 = invalid/boundary), phi, and u -- works for both
the pure-Python SubstrateLattice and the C++ AdaptiveLattice leaf arrays
(transpose leaf_phi from (14,N) to (N,14) first).
"""

import numpy as np

from equilibrium.substrate.lattice import W_HEX, W_SQ, _W as W_FACE  # step()'s per-leg W (sqrt3 : 1)


def face_unit_vectors(pos, nb, pos_own=None):
    """
    Unit direction vector from each cell to each of its 14 neighbours,
    read directly off real cell positions (no hardcoded BCC geometry
    constants) -- works for any leaf layout, including AMR-refined cells
    of mixed size. Shape (N_rows,14,3); zero vector where nb[i,k] < 0.

    pos: (N_global,3), must cover every id nb can reference (global gather
    for the neighbour side). nb: (N_rows,14) GLOBAL neighbour indices.

    pos_own: each row's OWN centre, in nb's row order, shape (N_rows,3).
    Defaults to `pos` itself, correct only when nb covers the FULL domain
    (row i's own cell IS global id i). Pass this explicitly whenever nb is a
    ROW SUBSET (e.g. pos_own=pos_full[subset_idx]) -- this keeps every
    intermediate array shaped (N_rows,14,3) instead of (N_global,14,3), which
    matters a lot at large N: computing this for a small row subset must
    never materialize a full-domain temporary (verified 2026-09-09 -- a
    18.4M-cell run OOM'd here even though the result was immediately sliced
    down and discarded, because the same-shaped (N_global,14,3) temporaries
    inside this function were the actual peak, not what was kept after).
    """
    if pos_own is None:
        pos_own = pos
    valid = nb >= 0
    safe_nb = np.where(valid, nb, 0)
    d = pos[safe_nb] - pos_own[:, None, :]  # (N_rows,14,3)
    dist = np.linalg.norm(d, axis=2)
    n_hat = np.zeros_like(d)
    nz = dist > 0
    n_hat[nz] = d[nz] / dist[nz, None]
    n_hat[~valid] = 0.0
    return n_hat


def local_energy_flux_vector(
    pos, nb, phi, u, w_face=W_FACE, n_hat=None, u_own=None, pos_own=None, sigma_face=None
):
    """
    Per-cell local energy-flux (Poynting-analogue) 3-vector:

        S_i = sum_k sigma_face_k * W_k * phi[i,k] * ubar_i,k * n_hat[i,k]

    sigma_face: (N_rows,14) face sigma (mean of the two cells' 1/(1+lam)); None = 1 (lam = 0).

    ubar_i,k = 0.5*(u_i + u_nb) is the face-average scalar value. Pass the
    phi and u snapshots that actually bracket one real step() call the way
    step() itself does internally (du computed from OLD phi, dphi computed
    from NEW u) -- i.e. phi from BEFORE the step, u from AFTER it. That
    pairing gave the smallest residual in the direct numerical check (see
    module docstring) because it matches step()'s own symplectic-Euler
    ordering, not an arbitrary choice.

    pos: (N_rows,3) cell centres. nb: (N_rows,14) GLOBAL neighbour indices
    into `u` (-1 = invalid). phi: (N_rows,14). u: (N_global,) -- must cover
    every id nb can reference, since u[safe_nb] is a global gather. Returns
    (N_rows,3); Smag = norm(S, axis=1) is the scalar magnitude used for
    radial falloff fits.

    n_hat: precomputed face_unit_vectors(pos, nb), (N_rows,14,3) -- pass this
    when calling repeatedly on a static (non-AMR) lattice, since pos/nb
    never change and recomputing it every call is pure waste (it dominates
    wall time at large N). Recomputed from pos/nb if omitted.

    u_own: each row's OWN u value, in pos/nb/phi's row order, shape
    (N_rows,). Defaults to `u` itself, correct only when pos/nb/phi cover
    the FULL domain (row i's own cell IS global id i). Pass this explicitly
    whenever pos/nb/phi are a ROW SUBSET of the full domain (e.g.
    u_own=u_full[subset_idx]) -- nb's neighbour ids stay global regardless,
    so the neighbour gather (u[safe_nb]) and the own-cell value need
    different indexing in that case.

    pos_own: same idea as u_own, forwarded to face_unit_vectors when n_hat
    isn't precomputed -- required (not just an optimization) for a row
    subset, since face_unit_vectors' internal temporaries are shaped
    (N_rows,14,3): a subset call without it would still be correct-shaped
    on `pos` (full, needed for the neighbour gather) but wrong on the "own
    centre" side unless pos_own is given.
    """
    if u_own is None:
        u_own = u
    valid = nb >= 0
    safe_nb = np.where(valid, nb, 0)
    u_nb = np.where(valid, u[safe_nb], u_own[:, None])
    ubar = 0.5 * (u_own[:, None] + u_nb)
    coef = w_face[None, :] if sigma_face is None else sigma_face * w_face[None, :]
    S_face = np.where(valid, coef * phi * ubar, 0.0)  # (N_rows,14)
    if n_hat is None:
        n_hat = face_unit_vectors(pos, nb, pos_own=pos_own)  # (N_rows,14,3)
    return (S_face[:, :, None] * n_hat).sum(axis=1)
