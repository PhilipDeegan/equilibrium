"""
Tests whether the worse-than-expected correlation at n_cells_per_wavelength=64
(probe_high_res_convergence.py: 0.9806/0.9787, 10.1% relative rms diff,
despite the dispersion FORMULA predicting 0.013% agreement) is accumulated
symplectic-Euler integration error from the large step count (3947 steps at
the stability-derived dt), not a stability/CFL problem. dt is a stability
bound here, not an accuracy one, and is independent of spatial resolution
(stable_dt() has no spacing/L dependence at all) -- so refining space
without also refining time can make overall fidelity WORSE despite better
dispersion-relation agreement.

Also applies the "resolution not spatial extent" + "a photon only needs a
few periods resolved" lessons: envelope_wavelengths cut from 4.0 to 1.5,
letting HALF_LEN shrink proportionally (~180 instead of 480) for the same
n_cells_per_wavelength=64 -- same resolution, far cheaper domain, so a
smaller dt is actually affordable.
"""

import sys

sys.path.insert(0, ".")
import time

import numpy as np
from scipy.spatial import cKDTree

from equilibrium.substrate.lattice import SubstrateLattice
from equilibrium.substrate.traveling_wave import seed_traveling_wave, face_offsets

R = 15.0
HALF_LEN = 180.0
AXIS = 2
N_CELLS_PER_WAVELENGTH = 64
ENVELOPE_WAVELENGTHS = 1.5
DT_FACTOR = 4  # dt = stable_dt()/DT_FACTOR


def run(dt_factor, label):
    lat = SubstrateLattice.cylinder(R, HALF_LEN, axis=AXIS)
    t_build = time.time()
    print(f"[{label}] N={lat.N} build_time={time.time()-t_build:.1f}s", flush=True)

    p = seed_traveling_wave(lat, AXIS, amplitude=1.0, n_cells_per_wavelength=N_CELLS_PER_WAVELENGTH, envelope_wavelengths=ENVELOPE_WAVELENGTHS)
    wavelength, T = p["wavelength"], p["T"]
    offsets, _ = face_offsets(lat)
    a_axial = float(abs(offsets[13, AXIS]))
    c_continuum = np.sqrt(10.0) * a_axial
    v_phase = wavelength / T

    dt_stable = lat.stable_dt()
    dt = dt_stable / dt_factor
    n_steps = int(round(T / dt))
    print(
        f"[{label}] wavelength={wavelength:.3f} T={T:.3f} v_phase={v_phase:.5f} "
        f"c_continuum(corrected)={c_continuum:.5f} v_phase/c_continuum={v_phase/c_continuum:.6f} "
        f"dt_stable={dt_stable:.6f} dt_used={dt:.6f} n_steps={n_steps}",
        flush=True,
    )

    u0, phi0 = lat.u.copy(), lat.phi.copy()
    t_run = time.time()
    for step in range(n_steps):
        lat.step(dt)
        if step % 1000 == 0:
            print(f"[{label}] step={step}/{n_steps} wall={time.time()-t_run:.1f}s", flush=True)
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
    print(
        f"[{label}] N compared cells={valid_shift.sum()}  u_corr={corr_u:.6f}  "
        f"phi_corr={corr_phi:.6f}  u_rel_rms_diff={rel_rms:.6f}",
        flush=True,
    )
    return corr_u, corr_phi, rel_rms


def main():
    run(1, "dt=stable_dt (baseline)")
    run(DT_FACTOR, f"dt=stable_dt/{DT_FACTOR}")


if __name__ == "__main__":
    main()
