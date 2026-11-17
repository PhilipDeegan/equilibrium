"""
Once-off dynamical (real step()-driven, not just the dispersion FORMULA)
convergence validation: at n_cells_per_wavelength=64 (8x finer than today's
n=8 experiments), does the traveling wave's actual measured v_phase (via a
genuine one-wavelength-shift check, same method as
tst/substrate/test_traveling_wave.py) converge to the corrected continuum
speed c=sqrt(10)*a_axial as tightly as the pure dispersion-formula sweep
predicted (-0.0033% at n=64)?
"""

import sys

sys.path.insert(0, ".")
import time

import numpy as np
from scipy.spatial import cKDTree

from equilibrium.substrate.lattice import SubstrateLattice
from equilibrium.substrate.traveling_wave import seed_traveling_wave, face_offsets

R = 15.0
HALF_LEN = 480.0
AXIS = 2
N_CELLS_PER_WAVELENGTH = 64


def main():
    t_build = time.time()
    lat = SubstrateLattice.cylinder(R, HALF_LEN, axis=AXIS)
    print(f"N={lat.N}  build_time={time.time()-t_build:.1f}s", flush=True)

    p = seed_traveling_wave(lat, AXIS, amplitude=1.0, n_cells_per_wavelength=N_CELLS_PER_WAVELENGTH, envelope_wavelengths=4.0)
    wavelength, T = p["wavelength"], p["T"]
    offsets, _ = face_offsets(lat)
    a_axial = float(abs(offsets[13, AXIS]))
    c_continuum = np.sqrt(10.0) * a_axial
    v_phase = wavelength / T

    dt = lat.stable_dt()
    n_steps = int(round(T / dt))
    print(
        f"wavelength={wavelength:.3f} T={T:.3f} v_phase={v_phase:.5f} "
        f"c_continuum(corrected)={c_continuum:.5f} v_phase/c_continuum={v_phase/c_continuum:.6f} "
        f"dt={dt:.6f} n_steps={n_steps}",
        flush=True,
    )

    u0, phi0 = lat.u.copy(), lat.phi.copy()
    t_run = time.time()
    for step in range(n_steps):
        lat.step(dt)
        if step % 500 == 0:
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
    print(f"N compared cells: {valid_shift.sum()}", flush=True)
    print(f"u correlation: {np.corrcoef(a,b)[0,1]:.6f}", flush=True)
    print(f"phi correlation: {np.corrcoef(ap.ravel(),bp.ravel())[0,1]:.6f}", flush=True)
    diff = a - b
    denom = np.sqrt(np.mean(a**2)) + np.sqrt(np.mean(b**2))
    print(f"u relative rms diff: {np.sqrt(np.mean(diff**2))/denom:.6f}", flush=True)


if __name__ == "__main__":
    main()
