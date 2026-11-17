"""
Invariants of the discrete kernels with non-uniform lam (2026-09-29).

step() / step_cpp: sigma at the face on both legs, so face antisymmetry, sum u and the energy
H = 1/2 sum u^2 + 1/2 sum_faces phi^2 hold for any lam (the u leg used the cell's sigma before:
sum u drifted +1.6 in 20 steps with lam in [0, 2]); numpy and C++ agree to round-off.
mhd.step_phi_diffuse: one update per face for both copies (the per-cell sigma broke antisymmetry
by 0.83); dissipative by design.
"""

import unittest

import numpy as np

from equilibrium.substrate import mhd
from equilibrium.substrate.ambient import energy, thermal_state
from equilibrium.substrate.lattice import _K_REV, SubstrateLattice


def _state(seed=0):
    lat = SubstrateLattice(radius_cells=6.0)
    rng = np.random.default_rng(seed)
    lat.u[:], lat.phi[:] = thermal_state(lat.nb, 1.0, rng)
    lat.lam[:] = rng.uniform(0.0, 2.0, lat.N)
    return lat


def _antisym(lat):
    i, k = np.nonzero(lat.nb >= 0)
    return float(np.max(np.abs(lat.phi[i, k] + lat.phi[lat.nb[i, k], _K_REV[k]])))


class TestStepWithLam(unittest.TestCase):
    def _check(self, advance):
        lat = _state()
        dt = lat.stable_dt()
        u0, h0 = lat.u.sum(), energy(lat.u, lat.phi, lat.nb)
        for _ in range(200):
            advance(lat, dt)
        self.assertEqual(_antisym(lat), 0.0)
        self.assertLess(abs(lat.u.sum() - u0), 1e-10)
        self.assertAlmostEqual(energy(lat.u, lat.phi, lat.nb) / h0, 1.0, delta=5e-3)

    def test_numpy(self):
        self._check(lambda lat, dt: lat.step(dt))

    def test_cpp(self):
        self._check(lambda lat, dt: lat.step_cpp(dt))

    def test_numpy_matches_cpp(self):
        a, b = _state(3), _state(3)
        dt = a.stable_dt()
        for _ in range(50):
            a.step(dt)
            b.step_cpp(dt)
        np.testing.assert_allclose(a.u, b.u, atol=1e-13)
        np.testing.assert_allclose(a.phi, b.phi, atol=1e-13)


class TestPhiDiffuse(unittest.TestCase):
    def test_antisymmetric_and_dissipative(self):
        lat = _state(1)
        dt = lat.stable_dt()
        h = [energy(lat.u, lat.phi, lat.nb)]
        for _ in range(50):
            mhd.step_phi_diffuse(lat, dt, eta=0.1)
            h.append(energy(lat.u, lat.phi, lat.nb))
        self.assertEqual(_antisym(lat), 0.0)
        self.assertTrue(np.all(np.diff(h) <= 1e-12))


if __name__ == "__main__":
    unittest.main()
