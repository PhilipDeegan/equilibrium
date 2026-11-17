"""
Rotating-frame correction terms for DST's discrete (u,phi) update, derived
directly from SubstrateLattice.step()'s own equations (du/dt=-sigma*div(phi),
dphi/dt=-sigma*grad(u)) -- see `now`, 2026-09-06 night entry, for the
derivation.

Two per-cell primitives both terms need: reconstructing a Cartesian (x,y,z)
vector from a cell's 14 per-face values, and projecting a Cartesian vector
back onto the 14 face directions. Built on
equilibrium.substrate.traveling_wave.face_offsets (empirically derived from
real lat.pos/lat.nb geometry, so convention-safe by construction) --
deliberately NOT tst/substrate/adaptive/adaptive_flux_tube.py's
_HEX_DIR/_SQ_DIR tables, which that file's own docstring flags as the
OPPOSITE sign convention (C++ AdaptiveCellCluster, not Python
SubstrateLattice).

phi's own convention is unscaled by hop length (the curl-free identity is
the exact `phi[i,k] = u[nb[i,k]] - u[i]`, no division by distance) -- so
reconstruction divides by hop length before the least-squares fit (to put
phi and one-sided u-differences on the same directional-derivative footing
as a true gradient), and projection multiplies back through the real
offset vectors (not unit directions) to land back in phi's own raw
convention.
"""

import numpy as np


def face_design(offsets):
    """(hop_len, pinv): pinv is the (3,14) least-squares pseudo-inverse
    mapping 14 per-face directional-derivative samples to one Cartesian
    vector. Same design matrix for every interior cell (homogeneous
    lattice) -- compute once, reuse for every cell/step."""
    hop_len = np.linalg.norm(offsets, axis=1)
    unit_dirs = offsets / hop_len[:, None]
    pinv = np.linalg.pinv(unit_dirs)  # (3,14)
    return hop_len, pinv


def reconstruct_vector(face_values, hop_len, pinv, valid=None):
    """(N,14) per-face samples (phi, or one-sided u_nb-u_i differences)
    -> (N,3) Cartesian vector."""
    rhs = face_values / hop_len[None, :]
    if valid is not None:
        rhs = np.where(valid, rhs, 0.0)
    return rhs @ pinv.T


def project_to_faces(vec, offsets):
    """(N,3) Cartesian vector -> (N,14) per-face values in phi's own raw
    (hop-length-scaled) convention -- the inverse map of reconstruct_vector."""
    return vec @ offsets.T


def apply_rotating_frame_correction(lat, offsets, hop_len, pinv, omega_vec, dt):
    """
    Advances (lat.u, lat.phi) by the two rotating-frame correction terms
    ONLY (not the base step() physics -- call lat.step(dt) separately,
    same operator-split pattern as probe_dlamdt_correction.py):

        du[i]   += -dt * (omega_vec x x_i) . grad(u)_i
        phi[i,k] += -dt * (omega_vec x Phi_vec_i) . offsets[k]

    omega_vec: constant (3,) angular-velocity vector (e.g. (0,0,omega) for
    rotation about z). Respects lat.fixed_u/lat.fixed_phi the same way
    step() does.
    """
    valid = lat._valid
    nb = lat.nb

    u_nb = np.where(valid, lat.u[np.where(valid, nb, 0)], lat.u[:, None])
    du_face = u_nb - lat.u[:, None]
    grad_u = reconstruct_vector(du_face, hop_len, pinv, valid=valid)

    omega_cross_x = np.cross(omega_vec[None, :], lat.pos)
    du_corr = -dt * np.sum(omega_cross_x * grad_u, axis=1)
    lat.u = np.where(lat.fixed_u, lat.u, lat.u + du_corr)

    Phi_vec = reconstruct_vector(lat.phi, hop_len, pinv, valid=valid)
    omega_cross_phi = np.cross(np.broadcast_to(omega_vec, Phi_vec.shape), Phi_vec)
    phi_corr = project_to_faces(-dt * omega_cross_phi, offsets)
    lat.phi = np.where(valid & ~lat.fixed_phi, lat.phi + phi_corr, lat.phi)


def light_cylinder_radius(c, omega):
    """r_lc = c/|omega| -- beyond this radius the rotating frame's own
    advective velocity exceeds the wave speed and the coordinate
    description stops being well-posed (see `now`, 2026-09-06 night)."""
    return c / abs(omega)
