"""
Fills in the missing half of the 2x2 grid isolating envelope_wavelengths
from n_cells_per_wavelength: n=64 at both envelope widths is already known
(probe_high_res_convergence.py: 0.9806 @ env=4.0; probe_dt_accuracy_check.py
baseline: 0.9370 @ env=1.5). This runs n=8 at both widths (cheap, small
domains) to see whether the same envelope-width sensitivity shows up
independent of resolution, or whether it's resolution-specific.
"""

import sys

sys.path.insert(0, ".")
import time

import numpy as np
from scipy.spatial import cKDTree

from equilibrium.substrate.lattice import SubstrateLattice
from equilibrium.substrate.traveling_wave import seed_traveling_wave, face_offsets

AXIS = 2
N_CELLS_PER_WAVELENGTH = 8


def run(R, HALF_LEN, envelope_wavelengths, label):
    lat = SubstrateLattice.cylinder(R, HALF_LEN, axis=AXIS)
    p = seed_traveling_wave(lat, AXIS, amplitude=1.0, n_cells_per_wavelength=N_CELLS_PER_WAVELENGTH, envelope_wavelengths=envelope_wavelengths)
    wavelength, T = p["wavelength"], p["T"]
    dt = lat.stable_dt()
    n_steps = int(round(T / dt))
    print(f"[{label}] N={lat.N} wavelength={wavelength:.3f} T={T:.3f} n_steps={n_steps}", flush=True)

    u0, phi0 = lat.u.copy(), lat.phi.copy()
    t_run = time.time()
    for step in range(n_steps):
        lat.step(dt)
    print(f"[{label}] stepping done, wall={time.time()-t_run:.1f}s", flush=True)

    u_T, phi_T = lat.u.copy(), lat.phi.copy()
    z_all = lat.pos[:, AXIS]
    r_perp = np.linalg.norm(np.delete(lat.pos, AXIS, axis=1), axis=1)
    axial_ok = (np.abs(z_all) < 0.5 * HALF_LEN) & (np.abs(z_all + wavelength) < 0.5 * HALF_LEN)
    core = (r_perp < 0.5 * R) & axial_ok

    tree = cKDTree(lat.pos)
    shift = np.zeros(3)
    shift[AXIS] = wavelength
    dist, idx = tree.query(lat.pos + shift[None, :])
    valid_shift = (dist < 1e-3) & core

    a, b = u_T[valid_shift], u0[idx][valid_shift]
    ap, bp = phi_T[valid_shift], phi0[idx][valid_shift]
    corr_u = np.corrcoef(a, b)[0, 1]
    corr_phi = np.corrcoef(ap.ravel(), bp.ravel())[0, 1]
    diff = a - b
    denom = np.sqrt(np.mean(a**2)) + np.sqrt(np.mean(b**2))
    rel_rms = np.sqrt(np.mean(diff**2)) / denom
    print(f"[{label}] N_compared={valid_shift.sum()}  u_corr={corr_u:.6f}  phi_corr={corr_phi:.6f}  rel_rms={rel_rms:.6f}", flush=True)
    return corr_u, corr_phi, rel_rms


def main():
    run(R=12.0, HALF_LEN=60.0, envelope_wavelengths=4.0, label="n=8, env=4.0")
    run(R=15.0, HALF_LEN=25.0, envelope_wavelengths=1.5, label="n=8, env=1.5")


if __name__ == "__main__":
    main()
