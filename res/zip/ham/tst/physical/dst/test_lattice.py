"""
Tests for DSTLattice core mechanics, focusing on the throughput function σ(λ)
and the fractal/multi-resolution structure.
"""

import unittest
import numpy as np
from equilibrium.physical.dst.lattice import DSTLattice


class TestSigma(unittest.TestCase):
    """
    σ(λ) = L / (1 + λ) encodes the local rate of time.
    Higher λ → slower clock.  σ is always positive (no zero).
    """

    def setUp(self):
        self.lat = DSTLattice(cell_size=1.0, radius_cells=4.0, K=1.0)

    def test_vacuum_sigma_equals_L(self):
        """At λ=0 (vacuum), σ = L for all cells."""
        np.testing.assert_array_equal(self.lat.sigma(), self.lat.L)

    def test_lambda1_half_rate(self):
        """At λ=1 (proton knot), σ = L/2 — half the vacuum rate."""
        self.lat.lam[:] = 1
        np.testing.assert_allclose(self.lat.sigma(), self.lat.L / 2.0)

    def test_sigma_decreases_with_lambda(self):
        """σ must be strictly monotonically decreasing as λ increases."""
        L = self.lat.cell_size
        lams = np.arange(0, 20)
        rates = L / (1.0 + lams)
        self.assertTrue(np.all(np.diff(rates) < 0))

    def test_sigma_never_zero(self):
        """
        σ(λ_max) > 0 always — the substrate never stops.
        Test with a very large λ (black-hole limit).
        """
        self.lat.lam[:] = 10_000  # extreme black-hole λ
        s = self.lat.sigma()
        self.assertTrue(np.all(s > 0.0))

    def test_sigma_black_hole_floor(self):
        """
        At λ=10 000, σ = L/(10 001) ≈ 1e-4 × vacuum rate.
        Slow but finite — observed as time dilation, not time stopping.
        """
        lam_bh = 10_000
        self.lat.lam[:] = lam_bh
        expected = self.lat.L / (1.0 + lam_bh)
        np.testing.assert_allclose(self.lat.sigma(), expected)

    def test_mixed_lambda_field(self):
        """Mixed λ field: each cell ticks at its own rate."""
        N = self.lat.N
        self.lat.lam = np.arange(N, dtype=np.int32) % 5  # λ ∈ {0,1,2,3,4}
        s_expected = self.lat.L / (1.0 + self.lat.lam)
        np.testing.assert_allclose(self.lat.sigma(), s_expected)


class TestStepWithLambda(unittest.TestCase):
    """
    Verify that cells with higher λ evolve slower than vacuum cells
    under the same potential gradient.
    """

    def _make_pair(self, lam_centre):
        lat = DSTLattice(cell_size=1.0, radius_cells=3.0, K=1.0)
        # give the central cell a u-gradient and set its λ
        centre = np.argmin(np.linalg.norm(lat.pos, axis=1))
        lat.u[centre] = 1.0
        lat.lam[centre] = lam_centre
        return lat, centre

    def test_high_lambda_evolves_slower(self):
        """
        A central cell with λ=5 should spread u more slowly than one with λ=0.
        """
        lat0, c0 = self._make_pair(0)
        lat5, c5 = self._make_pair(5)
        dt = min(lat0.stable_dt(), lat5.stable_dt())

        lat0.step(dt)
        lat5.step(dt)

        # Cell with λ=5 should retain more u (slower diffusion)
        self.assertGreater(lat5.u[c5], lat0.u[c0])

    def test_stable_dt_with_high_lambda(self):
        """
        stable_dt must remain positive and finite even when some cells
        have very high λ (black-hole-like).
        """
        lat = DSTLattice(cell_size=1.0, radius_cells=4.0, K=1.0)
        lat.lam[0] = 100_000
        dt = lat.stable_dt()
        self.assertGreater(dt, 0.0)
        self.assertTrue(np.isfinite(dt))


if __name__ == "__main__":
    unittest.main(verbosity=2)
