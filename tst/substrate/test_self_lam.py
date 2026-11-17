import os
import unittest

import numpy as np

from equilibrium.substrate.lattice import DampedLattice, SubstrateLattice
from equilibrium.substrate.self_lam import SelfLamMidpoint, energy_cell, step_midpoint_numpy

_N_THREADS = int(os.environ.get("EQ_TEST_N_THREADS", 1))


def _dipole(lat, width=1.5):
    r = np.linalg.norm(lat.pos, axis=1)
    lat.u[:] = (lat.pos[:, 2] / width) * np.exp(-(r**2) / (2 * width**2))
    lat.phi[:] = 0.0


class TestSelfLamMidpointCpp(unittest.TestCase):
    def test_matches_numpy(self):
        """C++ kernel == numpy reference (same iterates, same convergence test), with and without threads."""
        for n_threads in sorted({1, max(2, _N_THREADS)}):
            ref, lat = SubstrateLattice(radius_cells=6.0), SubstrateLattice(radius_cells=6.0)
            _dipole(ref), _dipole(lat)
            stepper = SelfLamMidpoint(lat, alpha=30.0, eps=0.01, n_threads=n_threads)
            dt = lat.stable_dt()
            for s in range(150):
                it_ref = step_midpoint_numpy(ref, 30.0, dt, eps=0.01)
                it = stepper.step(dt)
                self.assertEqual(it, it_ref)
            np.testing.assert_allclose(lat.u, ref.u, rtol=0, atol=1e-12)
            np.testing.assert_allclose(lat.phi, ref.phi, rtol=0, atol=1e-12)
            np.testing.assert_allclose(lat.lam, ref.lam, rtol=1e-12, atol=1e-12)

    def test_energy_conserved(self):
        """Implicit midpoint conserves H with state-dependent sigma (Euler would drift O(dt))."""
        lat = SubstrateLattice(radius_cells=6.0)
        _dipole(lat)
        H0 = energy_cell(lat).sum()
        stepper = SelfLamMidpoint(lat, alpha=100.0, n_threads=_N_THREADS)
        dt = lat.stable_dt()
        dev = 0.0
        for _ in range(400):
            stepper.step(dt)
            dev = max(dev, abs(energy_cell(lat).sum() / H0 - 1))
        self.assertLess(dev, 1e-10)

    def test_with_sponge(self):
        """Buffer swap survives an external sponge between steps (DampedLattice.apply_sponge reassigns u / phi)."""
        ref, lat = (DampedLattice(radius_cells=6.0, sponge_fraction=0.25) for _ in range(2))
        _dipole(ref), _dipole(lat)
        stepper = SelfLamMidpoint(lat, alpha=10.0, n_threads=_N_THREADS)
        dt = lat.stable_dt()
        for _ in range(100):
            step_midpoint_numpy(ref, 10.0, dt)
            ref.apply_sponge(dt)
            stepper.step(dt)
            lat.apply_sponge(dt)
        np.testing.assert_allclose(lat.u, ref.u, rtol=0, atol=1e-12)
        np.testing.assert_allclose(lat.phi, ref.phi, rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
