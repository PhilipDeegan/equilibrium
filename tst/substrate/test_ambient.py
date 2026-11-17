"""
Thermal ambient state as an initial condition (equilibrium/substrate/ambient.py), discrete
numpy step() on a closed lattice (no BC, no noise).

Replaces test_saha.py's TestSahaFraction, which pumped zero-mean noise into a lattice with no
sink: u performed a random walk (std ~ sqrt(steps), 2.4 -> 7.4 over 1k -> 10k steps), so there
was no equilibrium to measure (exp/2026/09/29). The equipartition ensemble is stationary under
step(), and <u^2> = T is exact, so the references here are derived, not pinned.
"""

import unittest

import numpy as np
from scipy.special import erfc

from equilibrium.substrate.ambient import energy, thermal_state
from equilibrium.substrate.lattice import _K_REV, SubstrateLattice, wave_speed

R = 8.0
T = 0.5


def _run(seed, crossings=3.0, n_snap=30):
    lat = SubstrateLattice(radius_cells=R)
    lat.u[:], lat.phi[:] = thermal_state(lat.nb, T, np.random.default_rng(seed))
    dt = lat.stable_dt()
    n = int(round(crossings * 2 * R / wave_speed() / dt))
    every = max(1, n // n_snap)
    h0 = energy(lat.u, lat.phi, lat.nb)
    snaps = []
    for s in range(1, n + 1):
        lat.step(dt)
        if s % every == 0:
            snaps.append(lat.u.copy())
    return lat, np.array(snaps), h0


class TestThermalState(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lat, cls.snaps, cls.h0 = _run(1)
        cls.N = cls.lat.N

    def test_initial_moments(self):
        lat = SubstrateLattice(radius_cells=R)
        u, phi = thermal_state(lat.nb, T, np.random.default_rng(2))
        self.assertAlmostEqual(float(np.mean(u**2)) / T, 1.0, delta=4 * np.sqrt(2 / lat.N))
        i, k = np.nonzero(lat.nb >= 0)
        self.assertEqual(float(np.max(np.abs(phi[i, k] + phi[lat.nb[i, k], _K_REV[k]]))), 0.0)
        self.assertEqual(float(np.abs(phi[lat.nb < 0]).max()), 0.0)

    def test_stationary_variance(self):
        """<u^2> stays T over 3 crossings (per-snapshot scatter ~ sqrt(2/N))."""
        v = np.mean(self.snaps**2, axis=1) / T
        self.assertLess(float(np.max(np.abs(v - 1.0))), 5 * np.sqrt(2 / self.N))

    def test_energy_conserved(self):
        h = energy(self.lat.u, self.lat.phi, self.lat.nb)
        self.assertAlmostEqual(h / self.h0, 1.0, delta=0.01)

    def test_two_seeds_agree(self):
        _, snaps2, _ = _run(3)
        a, b = float(np.mean(self.snaps**2)), float(np.mean(snaps2**2))
        self.assertAlmostEqual(a / b, 1.0, delta=4 * np.sqrt(2 / self.N))

    def test_tail_fraction_is_gaussian(self):
        """Fraction of cells above z sqrt(T) = 1/2 erfc(z / sqrt2): the exceedance fraction is a
        property of the ensemble (binomial error over pooled snapshots, which are correlated, so
        3 sigma with N cells as the effective count)."""
        u = self.snaps.ravel() / np.sqrt(T)
        for z in (0.5, 1.0, 2.0):
            p = 0.5 * erfc(z / np.sqrt(2))
            f = float(np.mean(u > z))
            self.assertLess(abs(f - p), 3 * np.sqrt(p * (1 - p) / self.N), f"z={z}: {f:.4f} vs {p:.4f}")


if __name__ == "__main__":
    unittest.main()
