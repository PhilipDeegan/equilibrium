"""
Verify equilibrium.substrate.traveling_wave against an exact one-wavelength
shift: u/phi at t=T (T derived from the wave's own discrete dispersion
relation) must match u/phi at t=0 shifted by exactly one wavelength along
the propagation axis. Extracted/promoted from
exp/2026/08/19/18_30_traveling_wave_reproducer/probe_traveling_wave_shift.py.
"""

import unittest

import numpy as np
from scipy.spatial import cKDTree

from equilibrium.substrate.lattice import DampedLattice
from equilibrium.substrate.traveling_wave import seed_traveling_wave

R = 12.0
HALF_LEN = 60.0
AXIS = 2


class TestTravelingWaveShift(unittest.TestCase):
    def test_one_wavelength_shift(self):
        lat = DampedLattice.cylinder(R, HALF_LEN, axis=AXIS, sponge_fraction=0.15, damp_max=0.003)
        p = seed_traveling_wave(lat, AXIS, amplitude=1.0, n_cells_per_wavelength=8, envelope_wavelengths=4.0)
        wavelength, T = p["wavelength"], p["T"]

        u0, phi0 = lat.u.copy(), lat.phi.copy()
        dt = lat.stable_dt()
        n_steps = int(round(T / dt))
        for _ in range(n_steps):
            lat.step(dt)
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
        self.assertGreater(valid_shift.sum(), 50)
        self.assertGreater(np.corrcoef(a, b)[0, 1], 0.95)

        ap, bp = phi_T[valid_shift], phi0[idx][valid_shift]
        self.assertGreater(np.corrcoef(ap.ravel(), bp.ravel())[0, 1], 0.95)


if __name__ == "__main__":
    unittest.main()
