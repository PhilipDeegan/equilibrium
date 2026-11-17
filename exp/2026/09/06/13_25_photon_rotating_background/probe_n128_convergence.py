"""
Continues the resolution sweep (probe_envelope_vs_resolution.py: n=8 -> 0.9782,
n=64 -> 0.9806, at fixed envelope_wavelengths=4.0) up to n=128: does the
shift-check correlation keep improving (converging further), or plateau?
"""

import sys

sys.path.insert(0, ".")
import time

import numpy as np
from scipy.spatial import cKDTree

from equilibrium.substrate.lattice import SubstrateLattice
from equilibrium.substrate.traveling_wave import seed_traveling_wave

R = 15.0
HALF_LEN = 960.0
AXIS = 2
N_CELLS_PER_WAVELENGTH = 128
ENVELOPE_WAVELENGTHS = 4.0


def main():
    t_build = time.time()
    lat = SubstrateLattice.cylinder(R, HALF_LEN, axis=AXIS)
    print(f"N={lat.N}  build_time={time.time()-t_build:.1f}s", flush=True)

    p = seed_traveling_wave(lat, AXIS, amplitude=1.0, n_cells_per_wavelength=N_CELLS_PER_WAVELENGTH, envelope_wavelengths=ENVELOPE_WAVELENGTHS)
    wavelength, T = p["wavelength"], p["T"]
    dt = lat.stable_dt()
    n_steps = int(round(T / dt))
    print(f"wavelength={wavelength:.3f} T={T:.3f} dt={dt:.6f} n_steps={n_steps}", flush=True)

    u0, phi0 = lat.u.copy(), lat.phi.copy()
    t_run = time.time()
    for step in range(n_steps):
        lat.step_cpp(dt, n_threads=1)
        if step % 1000 == 0:
            print(f"step={step}/{n_steps} wall={time.time()-t_run:.1f}s", flush=True)
    print(f"stepping done, wall={time.time()-t_run:.1f}s", flush=True)

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
    print(f"N_compared={valid_shift.sum()}  u_corr={corr_u:.6f}  phi_corr={corr_phi:.6f}  rel_rms={rel_rms:.6f}", flush=True)


if __name__ == "__main__":
    main()
