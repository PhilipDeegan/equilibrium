"""
Cross-check: does the companion-phi photon IC's large 12-triangle
circulation (probe_photon_staircase_circulation.py, circ_rms/phi_rms~0.97,
robust across wavelength and direction) also show up as a nonzero B via
the OTHER, independently-validated curl measure used everywhere else this
session (equilibrium.substrate.curl.b_vec_vectorized, the 4-cell sq-face
loop, matches the classical straight-wire/current-loop profile exactly)?
If both agree the finding is real; if they disagree, something is wrong
in one of the two measurements, not a real property of the IC.
"""

import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium.substrate.curl import b_vec_vectorized
from equilibrium.substrate.discrete_curl import circulation
from equilibrium.substrate.lattice import SubstrateLattice, W_HEX, W_SQ

WAVELENGTH = 16.0
ENVELOPE_WAVELENGTHS = 2.0
R_SPHERE = 40.0


def build_faces(lat):
    c0 = int(np.argmin(np.linalg.norm(lat.pos, axis=1)))
    offsets = lat.pos[lat.nb[c0]] - lat.pos[c0]
    w = np.array([W_HEX] * 8 + [W_SQ] * 6, dtype=np.float64)
    return offsets, w


def companion_wave(lat, offsets, w, k_hat, amplitude, envelope_sigma):
    K = 2 * np.pi / WAVELENGTH
    Kvec = K * k_hat
    theta = offsets @ Kvec
    sigma0 = 1.0
    omega = np.sqrt(float(np.sum(w**2 * (1 - np.cos(theta)))))
    pos = lat.pos
    proj = pos @ k_hat
    env = np.exp(-0.5 * (proj / envelope_sigma) ** 2)
    u = amplitude * env * np.cos(pos @ Kvec)
    phi = np.zeros_like(lat.phi)
    for k in range(14):
        Phi_k_amp = (2 * sigma0 * w[k] / omega) * np.sin(0.5 * theta[k])
        mid = pos + 0.5 * offsets[k][None, :]
        mid_proj = mid @ k_hat
        mid_env = np.exp(-0.5 * (mid_proj / envelope_sigma) ** 2)
        phi[:, k] = amplitude * Phi_k_amp * mid_env * np.cos(mid @ Kvec)
    return u, phi


def check(label, k_hat):
    lat = SubstrateLattice(radius_cells=R_SPHERE)
    offsets, w = build_faces(lat)
    envelope_sigma = ENVELOPE_WAVELENGTHS * WAVELENGTH
    u, phi = companion_wave(lat, offsets, w, k_hat, 1.0, envelope_sigma)
    lat.u, lat.phi = u, phi

    phi_rms = float(np.sqrt(np.mean(lat.phi**2)))
    circ = circulation(lat.phi.T, lat.nb)
    circ_rms = float(np.sqrt(np.mean(circ**2)))

    B = b_vec_vectorized(lat.pos, lat.nb, lat.phi)
    have = np.isfinite(B).all(axis=1)
    Bmag = np.linalg.norm(B[have], axis=1)
    b_rms = float(np.sqrt(np.mean(Bmag**2)))

    print(f"{label:30s}  phi_rms={phi_rms:.4e}  "
          f"12-tri circ_rms/phi_rms={circ_rms/phi_rms:.4f}  "
          f"4-sq |B|_rms/phi_rms={b_rms/phi_rms:.4f}")


if __name__ == "__main__":
    check("sq-axis (z)", np.array([0.0, 0.0, 1.0]))
    check("hex-axis (1,1,1)/sqrt3", np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0))
    generic = np.array([3.0, 1.3, 0.4])
    check("generic", generic / np.linalg.norm(generic))
