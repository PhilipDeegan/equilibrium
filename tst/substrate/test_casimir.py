"""
Kelvin-lattice Casimir energy from the exact mode sum between two plates.

1. Sign: the separation-dependent energy is negative (attractive plates).
2. Continuum limit: the 1/d^3 coefficient equals the scalar Dirichlet
   continuum value -pi^2 c / (1440 d^3) with c the lattice wave speed.
3. Universality: the same holds for other face weights (the Casimir term is
   set by the long-wavelength end alone).
4. The chain reproduces the bulk lattice dispersion.
"""

import unittest

import numpy as np

from equilibrium.substrate.casimir import (
    W_HEX,
    W_SQ,
    _dispersion,
    casimir_coefficient,
    plate_energy,
    wave_speed,
)


class TestCasimirContinuumLimit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = casimir_coefficient()

    def test_casimir_energy_negative(self):
        self.assertLess(self.r["casimir"], 0.0)

    def test_matches_continuum(self):
        self.assertAlmostEqual(self.r["ratio"], 1.0, delta=1e-3)

    def test_wave_speed_is_lattice_speed(self):
        self.assertAlmostEqual(self.r["c"], 2.0, places=12)


class TestCasimirUniversal(unittest.TestCase):
    def test_other_weights_match_continuum(self):
        r = casimir_coefficient(w_hex=3.0, w_sq=1.0)
        self.assertAlmostEqual(r["c"], np.sqrt(10.0), places=12)
        self.assertAlmostEqual(r["ratio"], 1.0, delta=1e-3)


class TestPlateEnergy(unittest.TestCase):
    def test_bulk_term_is_mean_zero_point_energy(self):
        """E(N+1) - E(N) -> per-layer bulk zero-point energy: 1/2 <omega> over the
        zone per site (two sites per cubic cell a^3, layer spacing a/2)."""
        dE = plate_energy(33) - plate_energy(32)
        rng = np.random.default_rng(0)
        K = rng.uniform(-np.pi, np.pi, size=(3, 400_000))
        K[2] *= 2.0  # Kz spans [-2 pi/a, 2 pi/a): layer spacing a/2
        bulk = 0.5 * float(np.mean(_dispersion(K[0], K[1], K[2], 1.0)))
        self.assertAlmostEqual(dE, bulk, delta=2e-3 * bulk)

    def test_chain_matches_bulk_dispersion(self):
        """omega^2 of the layer chain with a Bloch phase in z equals _dispersion."""
        a, h = 1.0, 0.5
        rng = np.random.default_rng(1)
        Kx, Ky, Kz = rng.uniform(-np.pi, np.pi, size=(3, 50))
        D0 = 8 * W_HEX**2 + 2 * W_SQ**2 + W_SQ**2 * (4 - 2 * np.cos(Kx) - 2 * np.cos(Ky))
        t1 = 4 * W_HEX**2 * np.cos(h * Kx) * np.cos(h * Ky)
        w2 = D0 - 2 * t1 * np.cos(h * Kz) - 2 * W_SQ**2 * np.cos(2 * h * Kz)
        np.testing.assert_allclose(np.sqrt(w2), _dispersion(Kx, Ky, Kz, a), rtol=1e-12)
        self.assertAlmostEqual(wave_speed(), 2.0, places=12)


if __name__ == "__main__":
    unittest.main()
