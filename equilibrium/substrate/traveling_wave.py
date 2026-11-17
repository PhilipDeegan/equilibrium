"""
Companion-phi traveling wave: a curl-full (u,phi) wave packet that satisfies
the SubstrateLattice's own discrete dispersion relation exactly, rather than
a hand-picked continuum wave. Extracted from
exp/2026/08/19/18_30_traveling_wave_reproducer/ (previously duplicated
inline across 3 probe scripts) -- validated there via an exact one-
wavelength-shift check (tst/substrate/test_traveling_wave.py).

Dispersion relation: omega^2 = sum_k w_k^2 * (1 - cos(K . offset_k)), the
same relation the discrete step() update itself obeys for a plane wave along
`axis`. Phi's per-face amplitude Phi_k = (2*w_k/omega)*sin(theta_k/2) is the
companion flux that makes (u,phi) a genuine traveling (not standing) wave
solution of the discrete update, not just a u snapshot with phi=0.
"""

import numpy as np

from equilibrium.substrate.lattice import W_HEX, W_SQ

FACE_W = np.array([W_HEX] * 8 + [W_SQ] * 6, dtype=np.float64)


def face_offsets(lat):
    """(offsets, w): offsets[k] = pos[nb[c0,k]] - pos[c0] for an interior
    cell c0 (nearest the origin), w = the fixed per-face weight table."""
    c0 = int(np.argmin(np.linalg.norm(lat.pos, axis=1)))
    assert lat._valid[c0].all()
    offsets = lat.pos[lat.nb[c0]] - lat.pos[c0]
    return offsets, FACE_W


def dispersion_omega(offsets, w, axis, K):
    theta = offsets[:, axis] * K
    return float(np.sqrt(np.sum(w**2 * (1 - np.cos(theta)))))


def wave_params(lat, axis, n_cells_per_wavelength=8, envelope_wavelengths=4.0):
    """Derives wavelength/K/omega/period/envelope_sigma from the lattice's
    own real hop distance along `axis`, so the wavelength is always an exact
    integer multiple of the true lattice period (not an arbitrary length)."""
    offsets, w = face_offsets(lat)
    a_axial = float(abs(offsets[13, axis]))  # true +axis sq-face hop distance
    wavelength = n_cells_per_wavelength * a_axial
    K = 2 * np.pi / wavelength
    omega = dispersion_omega(offsets, w, axis, K)
    T = 2 * np.pi / omega
    envelope_sigma = envelope_wavelengths * wavelength
    return {
        "offsets": offsets,
        "w": w,
        "wavelength": wavelength,
        "K": K,
        "omega": omega,
        "T": T,
        "envelope_sigma": envelope_sigma,
    }


def u_field(lat, axis, K, amplitude, envelope_sigma, z0, centre=None):
    pos = lat.pos if centre is None else lat.pos - centre
    z = pos[:, axis] - z0
    env = np.exp(-0.5 * (z / envelope_sigma) ** 2)
    return amplitude * env * np.cos(pos[:, axis] * K)


def companion_phi(lat, offsets, w, axis, K, amplitude, envelope_sigma, z0, omega, centre=None):
    theta = offsets[:, axis] * K
    pos = lat.pos if centre is None else lat.pos - centre
    phi = np.zeros_like(lat.phi)
    for k in range(14):
        Phi_k_amp = (2 * w[k] / omega) * np.sin(0.5 * theta[k])
        mid_axis_pos = pos[:, axis] + 0.5 * offsets[k, axis]
        mid_env = np.exp(-0.5 * ((mid_axis_pos - z0) / envelope_sigma) ** 2)
        phi[:, k] = amplitude * Phi_k_amp * mid_env * np.cos(mid_axis_pos * K)
    return phi


def seed_traveling_wave(lat, axis, amplitude=1.0, n_cells_per_wavelength=8, envelope_wavelengths=4.0, z0=0.0, centre=None):
    """Seeds lat.u/lat.phi with a validated traveling wave packet centred at
    z0 (along `axis`, optionally offset by `centre` for an off-axis launch).
    Returns the wave_params dict (wavelength/K/omega/T/...) for the caller's
    own step-count/measurement bookkeeping."""
    p = wave_params(lat, axis, n_cells_per_wavelength, envelope_wavelengths)
    lat.u = u_field(lat, axis, p["K"], amplitude, p["envelope_sigma"], z0, centre=centre)
    lat.phi = companion_phi(lat, p["offsets"], p["w"], axis, p["K"], amplitude, p["envelope_sigma"], z0, p["omega"], centre=centre)
    return p
