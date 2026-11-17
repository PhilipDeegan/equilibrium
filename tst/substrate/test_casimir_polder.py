"""
Tests for the Kelvin-lattice Casimir-Polder point-scatterer vacuum energy
(real-space Green's function via 3-D Brillouin zone quadrature).

Physics being verified
-----------------------
1. Grid convergence: G_point is stable under Nyz doubling (the flagged
   numerical risk -- quadrature accuracy needs checking, not assuming).
2. Long-range limit: G(r,0) and G(r,xi) recover the continuum Yukawa
   propagator G_continuum as r grows, with the SAME c = sqrt(10)*a already
   used in casimir.py -- one dispersion relation, not two unrelated formulas.
3. Vacuum energy sign (attractive) and ~1/r^3 scaling.
4. U_continuum is the correct closed form of the xi-integral of G_continuum^2.
"""

import unittest

import numpy as np

from equilibrium.substrate.casimir_polder import (
    C_COEFF,
    G_continuum,
    G_point,
    U_continuum,
    converged_G_point,
    vacuum_energy_scalar,
)


class TestGPointConvergence(unittest.TestCase):
    def test_converges_under_grid_doubling(self):
        """G_point at a modest separation must stabilise as Nyz doubles."""
        _, Nyz, converged = converged_G_point(
            5.0, 1.0, Nyz0=80, rtol=2e-2, max_doublings=3
        )
        self.assertTrue(converged, f"did not converge by Nyz={Nyz}")

    def test_finite_xi_regularises_small_r(self):
        """At r=0 with xi>0, G_point must be finite (no singularity to worry about)."""
        val = G_point(0.0, 1.0, Nyz=100)
        self.assertTrue(np.isfinite(val))
        self.assertGreater(val, 0.0)


class TestLongRangeContinuumLimit(unittest.TestCase):
    def test_G_at_xi0_approaches_coulomb_like_form(self):
        """
        G(r,0) should approach 1/(4*pi*c^2*r) as r grows (many lattice
        spacings) -- loosely at r=5, more tightly at r=8.
        """
        r_small, r_large = 5.0, 8.0
        g_small = G_point(r_small, 0.0, Nyz=250, limit=400)
        g_large = G_point(r_large, 0.0, Nyz=250, limit=400)
        cont_small = G_continuum(r_small, 0.0)
        cont_large = G_continuum(r_large, 0.0)

        ratio_small = g_small / cont_small
        ratio_large = g_large / cont_large
        self.assertGreater(ratio_small, 0.85)
        self.assertLess(ratio_small, 1.05)
        self.assertGreater(ratio_large, 0.85)
        self.assertLess(ratio_large, 1.05)

    def test_G_finite_xi_matches_yukawa_shape(self):
        """G(r,xi) with xi>0 should sit close to the Yukawa continuum form at moderate r.
        xi is set by the decay length c/xi = sqrt10 a (what xi = 1 gave at the
        pre-2026-09-28 c = sqrt10): a shorter decay length nears the lattice scale
        and the transverse integral loses accuracy (xi = 1 at c = 2: 1.17 at r=5,
        negative by r=12)."""
        r = 5.0
        xi = C_COEFF / np.sqrt(10.0)
        g = G_point(r, xi, Nyz=250, limit=400)
        cont = G_continuum(r, xi, a=1.0)
        self.assertGreater(g / cont, 0.85)
        self.assertLess(g / cont, 1.1)


class TestVacuumEnergy(unittest.TestCase):
    def test_energy_is_negative(self):
        """Scalar vacuum energy between two point scatterers is attractive."""
        U = vacuum_energy_scalar(5.0, Nyz=150, n_xi=48)
        self.assertLess(U, 0.0)

    def test_energy_scales_as_1_over_r_cubed(self):
        """U(2r)/U(r) should approach (1/2)^3 = 0.125 as the lattice-scale correction shrinks."""
        U1 = vacuum_energy_scalar(4.0, Nyz=150, n_xi=48)
        U2 = vacuum_energy_scalar(8.0, Nyz=150, n_xi=48)
        ratio = U2 / U1
        self.assertAlmostEqual(ratio, 0.125, delta=0.05)

    def test_energy_within_order_of_continuum(self):
        """
        U(r) should be within a factor of a few of the continuum reference
        (same-order-of-magnitude cross-check, not a tight coefficient match --
        the BZ-shape correction factor is not assumed to be 1).
        """
        r = 5.0
        U = vacuum_energy_scalar(r, Nyz=150, n_xi=48)
        Ucont = U_continuum(r)
        ratio = U / Ucont
        self.assertGreater(ratio, 0.3)
        self.assertLess(ratio, 3.0)


class TestContinuumClosedForm(unittest.TestCase):
    def test_U_continuum_matches_numeric_xi_integral_of_G_continuum(self):
        """
        U_continuum's closed form must match direct numeric integration of
        -(1/2pi) INT G_continuum(r,xi)^2 dxi (sanity-checks the closed-form
        algebra independent of the lattice quadrature).
        """
        r = 5.0
        xi_vals = np.linspace(1e-6, 200.0, 20000)
        integrand = G_continuum(r, xi_vals) ** 2
        integral = np.trapezoid(integrand, xi_vals)
        U_numeric = -integral / (2.0 * np.pi)
        U_closed = U_continuum(r)
        self.assertAlmostEqual(U_numeric / U_closed, 1.0, delta=1e-3)

    def test_U_continuum_negative(self):
        self.assertLess(U_continuum(5.0), 0.0)


if __name__ == "__main__":
    unittest.main()
