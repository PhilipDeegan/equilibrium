"""
Aharonov-Bohm effect tests on the Kelvin cell substrate.

Tests verify:
  1. A-field (phi) values follow the correct formula.
  2. phi is antisymmetric across each face.
  3. Loop integral along an actual lattice path = flux_quanta.
  4. Wave propagation is stable.
  5. Non-zero flux shifts the detector pattern vs zero flux.
"""

import unittest

import numpy as np

from equilibrium.substrate.ab import (
    _inject_pulse,
    _set_solenoid,
    _solenoid_cells,
    detector_slice,
    run,
)
from equilibrium.substrate.lattice import SubstrateLattice


def _mirror_k(k: int) -> int:
    """Face index of the reverse direction for face k."""
    if k < 8:
        return 7 - k  # hex: offsets are ordered so mirror = 7-k
    return k + 1 if k % 2 == 0 else k - 1  # sq pairs: (8,9),(10,11),(12,13)


def _sq_loop_around_origin(lat: SubstrateLattice) -> list:
    """
    Return a closed 8-cell sq-face loop that provably encircles the z-axis.

    The loop visits corner atoms at (±a, 0, 0) and (0, ±a, 0) in CCW order,
    where a = lattice spacing.  All steps are sq-face connections (distance d2).

    Proof of winding: each of the 8 steps advances by exactly π/4 in azimuth
    (with atan2 wrapping at the ±x axis handled correctly), so Σ Δθ = 2π.

    Returns [] if any target cell is missing from the lattice.
    """
    from equilibrium.math.geo.kelvin import KelvinLattice

    a = KelvinLattice(lattice_spacing=1.0).a  # sq-face distance = a

    targets = (
        np.array(
            [
                [-a, -a, 0],
                [0, -a, 0],
                [a, -a, 0],
                [a, 0, 0],
                [a, a, 0],
                [0, a, 0],
                [-a, a, 0],
                [-a, 0, 0],
            ]
        )
        * 1.0
    )

    idx = []
    for t in targets:
        dists = np.linalg.norm(lat.pos - t, axis=1)
        best = int(np.argmin(dists))
        if dists[best] > 0.1 * a:
            return []  # cell not in lattice
        idx.append(best)
    return idx


def _loop_phi_sum(lat: SubstrateLattice, path: list) -> float:
    """Sum phi[i,k] along a closed lattice path (last cell connects back to first)."""
    total = 0.0
    n = len(path)
    for step in range(n):
        i = path[step]
        j = path[(step + 1) % n]
        hits = np.where(lat.nb[i] == j)[0]
        if len(hits):
            total += float(lat.phi[i, hits[0]])
    return total


class TestPhiValues(unittest.TestCase):
    """phi must follow formula and be antisymmetric."""

    def setUp(self):
        self.lat = SubstrateLattice(radius_cells=8.0)
        sol = _solenoid_cells(self.lat, 1.5)
        _set_solenoid(self.lat, sol, flux_quanta=1)

    def test_formula(self):
        lat = self.lat
        errors = []
        for i in range(min(200, lat.N)):
            xi, yi = lat.pos[i, 0], lat.pos[i, 1]
            theta_i = np.arctan2(yi, xi)
            for k in range(14):
                j = lat.nb[i, k]
                if j < 0:
                    continue
                xj, yj = lat.pos[j, 0], lat.pos[j, 1]
                dtheta = np.arctan2(yj, xj) - theta_i
                dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
                expected = dtheta / (2 * np.pi)
                errors.append(abs(lat.phi[i, k] - expected))

        self.assertLess(max(errors), 1e-12, "phi[i,k] must equal Δθ / 2π")

    def test_antisymmetry(self):
        lat = self.lat
        errors = []
        for i in range(min(200, lat.N)):
            for k in range(14):
                j = lat.nb[i, k]
                if j < 0:
                    continue
                mk = _mirror_k(k)
                errors.append(abs(lat.phi[i, k] + lat.phi[j, mk]))

        self.assertLess(max(errors), 1e-12, "phi must be antisymmetric across faces")


class TestLoopIntegral(unittest.TestCase):
    """
    Loop integral of phi around a provably enclosing sq-face path must
    equal flux_quanta.  The 8-cell CCW square loop (±a, 0) and (0, ±a)
    encloses the origin: each step advances Δθ = π/4, total = 2π.
    """

    def _check(self, flux):
        lat = SubstrateLattice(radius_cells=4.0)
        sol = _solenoid_cells(lat, 1.5)
        _set_solenoid(lat, sol, flux_quanta=flux)

        loop = _sq_loop_around_origin(lat)
        if not loop:
            self.skipTest("loop cells not present at this radius")

        integral = _loop_phi_sum(lat, loop)
        self.assertAlmostEqual(
            integral,
            float(flux),
            delta=1e-10,
            msg=f"loop integral should be {flux}, got {integral:.6f}",
        )

    def test_flux_1(self):
        self._check(1)

    def test_flux_2(self):
        self._check(2)

    def test_flux_minus_1(self):
        self._check(-1)


class TestWavePropagation(unittest.TestCase):
    def test_stable_and_spreads(self):
        lat = SubstrateLattice(radius_cells=8.0)
        _inject_pulse(lat, x0=-4.0, sigma=1.0)
        lat.run(n_steps=40)
        self.assertFalse(np.any(np.isnan(lat.u)))
        self.assertFalse(np.any(np.isinf(lat.u)))
        self.assertLess(lat.u.max(), 10.0)

    def test_stable_dt_positive(self):
        lat = SubstrateLattice(radius_cells=5.0)
        self.assertGreater(lat.stable_dt(), 0.0)


class TestABShift(unittest.TestCase):
    """Non-zero flux must shift the detector pattern."""

    def test_flux_changes_pattern(self):
        lat0 = run(radius_cells=12.0, flux_quanta=0, pulse_sigma=2.0, n_steps=150)
        lat1 = run(radius_cells=12.0, flux_quanta=1, pulse_sigma=2.0, n_steps=150)

        x_det = lat0.pos[:, 0].max() * 0.5
        _, u0 = detector_slice(lat0, x_det)
        _, u1 = detector_slice(lat1, x_det)

        diff = np.abs(u1 - u0).sum()
        self.assertGreater(
            diff, 0.0, "AB pattern must differ between flux=0 and flux=1"
        )

    def test_zero_flux_self_consistent(self):
        # Two independent zero-flux runs must be identical
        lat_a = run(radius_cells=10.0, flux_quanta=0, pulse_sigma=2.0, n_steps=80)
        lat_b = run(radius_cells=10.0, flux_quanta=0, pulse_sigma=2.0, n_steps=80)

        diff = np.abs(lat_a.u - lat_b.u).max()
        self.assertAlmostEqual(
            diff,
            0.0,
            delta=1e-12,
            msg="identical zero-flux runs must give identical results",
        )


if __name__ == "__main__":
    unittest.main()
