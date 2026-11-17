"""
Tests for DST gravitational time dilation.

Verifies that the DST cell-size prescription (L(r) = L₀·(1 − GM/rc²))
reproduces measured gravitational time dilation:

  Pound-Rebka (1959)  : first direct measurement of gravitational redshift
  Hafele-Keating 1971 : airborne atomic clocks, Δf/f ≈ g·h/c²
  GPS (ongoing)       : gravitational gain ≈ +45.9 μs/day at 20 200 km
"""

import unittest
import numpy as np

from equilibrium.physical.dst.gravity import (
    altitude_dilation,
    newtonian_approx,
    clock_rate,
    phi,
    R_EARTH,
    M_EARTH,
    cell_size_profile,
    cell_size_gradient_fm_per_km,
    cell_compression_hierarchy,
    phi_sources,
    L_VACUUM_CLAWD,
    L_PLANCK_M,
)
from equilibrium.physical.dst.lattice import DSTLattice


class TestGravitationalPotential(unittest.TestCase):
    def test_surface_potential(self):
        """Earth's surface potential φ = GM_E/(R_E c²) ≈ 6.96e-10."""
        p = phi(R_EARTH, M_EARTH)
        self.assertAlmostEqual(p, 6.96e-10, delta=6.96e-10 * 0.01)

    def test_clock_rate_below_one_at_surface(self):
        """Clock at Earth's surface runs slower than at infinity."""
        self.assertLess(clock_rate(R_EARTH), 1.0)

    def test_clock_rate_increases_with_radius(self):
        """Clock rate must increase monotonically with altitude."""
        radii = [R_EARTH + h for h in (0, 1e3, 1e4, 1e6, 1e7, 1e8)]
        rates = [clock_rate(r) for r in radii]
        for i in range(1, len(rates)):
            self.assertGreater(rates[i], rates[i - 1])


class TestAltitudeDilation(unittest.TestCase):
    def test_1km_matches_newtonian(self):
        """
        At 1 km, DST matches g·h/c² to better than 0.2%.

        The ~0.15% residual is physical: newtonian_approx uses the standard
        surface gravity (9.80665 m/s²), while the DST formula uses GM_E/R_E²
        (9.820 m/s²).  These differ because G_SURFACE includes centrifugal
        correction and polar flattening; our spherical model does not.
        """
        h = 1_000.0
        dst = altitude_dilation(h)
        appr = newtonian_approx(h)
        self.assertAlmostEqual(dst / appr, 1.0, delta=0.002)

    def test_10km_matches_newtonian(self):
        """At 10 km (cruising altitude), agreement is still better than 0.2%."""
        h = 10_000.0
        dst = altitude_dilation(h)
        appr = newtonian_approx(h)
        self.assertAlmostEqual(dst / appr, 1.0, delta=0.002)

    def test_gps_gravitational_gain(self):
        """
        GPS orbit (h = 20 200 km): gravitational gain ≈ 45.9 μs/day.
        Fractional rate: 45.9e-6 / 86400 ≈ 5.31e-10.
        DST should agree to 1%.
        """
        h_gps = 20_200e3
        gps_ref = 45.9e-6 / 86_400  # ≈ 5.31e-10
        dst = altitude_dilation(h_gps)
        self.assertAlmostEqual(dst, gps_ref, delta=gps_ref * 0.01)

    def test_altitude_gain_is_positive(self):
        """Higher altitude always means faster clock."""
        for h in (100.0, 1e3, 1e4, 1e5, 1e6, 1e7):
            with self.subTest(h=h):
                self.assertGreater(altitude_dilation(h), 0.0)

    def test_monotone_with_altitude(self):
        """Dilation increases monotonically: more altitude, more gain."""
        hs = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
        ds = [altitude_dilation(h) for h in hs]
        for i in range(1, len(ds)):
            self.assertGreater(ds[i], ds[i - 1])


class TestDSTLatticeGravity(unittest.TestCase):
    """Verify set_gravity() on a small lattice reproduces the expected profile."""

    @classmethod
    def setUpClass(cls):
        # Small lattice: cell_size=1, 6-cell radius sphere
        cls.lat = DSTLattice(cell_size=1.0, radius_cells=6.0, K=1.0)
        # Use an Earth-like Schwarzschild radius scaled to lattice units
        # rs_planck just needs to be << lattice size so weak-field holds
        cls.rs = 1e-3  # Planck lengths (tiny — weak field throughout)
        cls.lat.set_gravity(rs_planck=cls.rs)

    def test_u_decreases_with_radius(self):
        """u(r) = rs/(2r) must fall off with distance."""
        r = np.linalg.norm(self.lat.pos, axis=1)
        u = self.lat.u
        # Cells further out should have smaller u on average
        inner = u[r < r.mean()]
        outer = u[r > r.mean()]
        self.assertGreater(inner.mean(), outer.mean())

    def test_cell_size_decreases_near_mass(self):
        """Cells closer to the mass must be smaller (slower time)."""
        r = np.linalg.norm(self.lat.pos, axis=1)
        L = self.lat.L
        inner = L[r < r.mean()]
        outer = L[r > r.mean()]
        self.assertLess(inner.mean(), outer.mean())

    def test_clock_rate_field_range(self):
        """All clock rates must be in (0, 1] for weak-field source."""
        f = self.lat.clock_rate_field()
        self.assertTrue(np.all(f > 0.0))
        self.assertTrue(np.all(f <= 1.0))

    def test_clock_rate_matches_gravity_module(self):
        """
        For each cell, f(r) = 1 − rs/(2r) from the lattice must
        match the gravity.clock_rate formula (scaled to Planck units).
        """
        r = np.linalg.norm(self.lat.pos, axis=1)
        r = np.maximum(r, self.lat.cell_size)
        f_expected = 1.0 - self.rs / (2.0 * r)
        f_actual = self.lat.clock_rate_field()
        np.testing.assert_allclose(f_actual, f_expected, rtol=1e-10)


class TestCellSizeProfile(unittest.TestCase):
    """
    Verify DST cell-size variation with altitude using L_vacuum = 10^30 l_P
    (CLAWD deep-space reference).  At this scale, altitude differences map
    to femtometre-range cell size changes.
    """

    def test_sea_level_deficit(self):
        """
        At Earth's surface, cells are compressed by φ_surface × L_vacuum.
        For L_vacuum = 10^30 l_P: deficit ≈ 6.96e20 l_P ≈ 11.25 fm.
        """
        L_sea = cell_size_profile(0.0, L_vacuum_lP=L_VACUUM_CLAWD)[()]
        deficit_lP = L_VACUUM_CLAWD - L_sea
        deficit_fm = deficit_lP * L_PLANCK_M * 1e15
        # Should be ~11 fm (about 13 proton radii)
        self.assertAlmostEqual(deficit_fm, 11.25, delta=0.1)

    def test_gps_delta_is_subnuclear(self):
        """
        ΔL between sea level and GPS orbit (L_vacuum = 10^30 l_P) ≈ 8.55 fm.
        This is subnuclear — about 10 proton radii of cell growth.
        """
        L_sea = cell_size_profile(0.0, L_vacuum_lP=L_VACUUM_CLAWD)[()]
        L_gps = cell_size_profile(20_200e3, L_vacuum_lP=L_VACUUM_CLAWD)[()]
        dL_fm = (L_gps - L_sea) * L_PLANCK_M * 1e15
        self.assertAlmostEqual(dL_fm, 8.55, delta=0.05)

    def test_cell_size_increases_with_altitude(self):
        """Cells must grow monotonically with altitude."""
        alts = np.linspace(0, 5e7, 50)
        L = cell_size_profile(alts, L_vacuum_lP=L_VACUUM_CLAWD)
        self.assertTrue(np.all(np.diff(L) > 0))

    def test_cell_size_ratio_equals_time_dilation(self):
        """
        The cell-size ratio L(r2)/L(r1) must equal the clock-rate ratio f(r2)/f(r1).
        This is the core DST identity: time rate ∝ cell size.
        """
        h1, h2 = 0.0, 20_200e3
        L1 = cell_size_profile(h1, L_vacuum_lP=L_VACUUM_CLAWD)[()]
        L2 = cell_size_profile(h2, L_vacuum_lP=L_VACUUM_CLAWD)[()]
        f1 = clock_rate(R_EARTH + h1)
        f2 = clock_rate(R_EARTH + h2)
        self.assertAlmostEqual(L2 / L1, f2 / f1, delta=1e-15)

    def test_gradient_fm_per_km(self):
        """
        Near Earth's surface, dL/dr ≈ g/c² × L_vacuum ≈ 1.77e-3 fm/km
        for L_vacuum = 10^30 l_P.
        """
        grad = cell_size_gradient_fm_per_km(L_VACUUM_CLAWD)
        self.assertAlmostEqual(grad, 1.766e-3, delta=1e-5)


class TestVacuumHierarchy(unittest.TestCase):
    """
    Verify the multi-scale vacuum hierarchy:
    extragalactic → intragalactic → solar system → Earth surface → altitude.

    The galaxy's potential at Earth's location dominates everything else.
    All Earth-based experiments measure against the intragalactic+solar background.
    """

    def setUp(self):
        self.src = phi_sources()
        self.hier = cell_compression_hierarchy(L_VACUUM_CLAWD)

    def test_galaxy_dominates_over_earth(self):
        """Galaxy potential at solar orbit must be >> Earth surface potential."""
        self.assertGreater(self.src["galaxy"], self.src["earth"] * 100)

    def test_galaxy_dominates_over_sun(self):
        """Galaxy potential must exceed Sun's potential at 1 AU."""
        self.assertGreater(self.src["galaxy"], self.src["sun"] * 10)

    def test_sun_dominates_over_earth(self):
        """Sun's potential at Earth's orbit must exceed Earth surface potential."""
        self.assertGreater(self.src["sun"], self.src["earth"] * 10)

    def test_galaxy_phi_order_of_magnitude(self):
        """Galaxy φ ≈ 5.4e-7 (from circular velocity ≈ 220 km/s at 8.5 kpc)."""
        self.assertAlmostEqual(self.src["galaxy"], 5.4e-7, delta=5.4e-7 * 0.05)

    def test_sun_phi_order_of_magnitude(self):
        """Sun φ at 1 AU ≈ 9.9e-9."""
        self.assertAlmostEqual(self.src["sun"], 9.9e-9, delta=9.9e-9 * 0.02)

    def test_galaxy_compression_is_picometre(self):
        """
        For L_vacuum = 10^30 l_P, the galaxy compresses cells by ~8.7 pm
        — sub-atomic bond-length scale.
        """
        dL_pm = self.hier["galaxy (extragalactic→intragalactic)"] * 1e12
        self.assertAlmostEqual(dL_pm, 8.7, delta=0.5)

    def test_sun_compression_is_femtometre(self):
        """Sun compresses cells by ~160 fm at 1 AU (about 190 proton radii)."""
        dL_fm = self.hier["sun (intragalactic→solar system)"] * 1e15
        self.assertAlmostEqual(dL_fm, 159.5, delta=2.0)

    def test_earth_compression_is_femtometre(self):
        """Earth compresses cells by ~11.25 fm at its surface."""
        dL_fm = self.hier["earth (solar system→surface)"] * 1e15
        self.assertAlmostEqual(dL_fm, 11.25, delta=0.1)

    def test_galaxy_larger_than_altitude_gps(self):
        """
        The galactic contribution (8.7 pm) dwarfs the sea-level→GPS altitude
        effect (8.6 fm) by a factor of ~1000.
        """
        dL_gal = self.hier["galaxy (extragalactic→intragalactic)"]
        dL_alt = self.hier["altitude (sea level→GPS orbit)"]
        self.assertGreater(dL_gal / dL_alt, 500)

    def test_intra_vs_extra_galactic_reference(self):
        """
        A measurement between two Earth altitudes correctly gives Δf/f from
        Earth's potential only — galaxy and Sun cancel as background constants.
        """
        phi_sea = phi(R_EARTH)
        phi_gps = phi(R_EARTH + 20_200e3)
        # altitude_dilation uses Earth's phi only → should match delta_earth
        delta_earth_only = phi_sea - phi_gps
        gps_dilation = altitude_dilation(20_200e3)
        self.assertAlmostEqual(
            gps_dilation, delta_earth_only, delta=delta_earth_only * 0.001
        )


if __name__ == "__main__":
    from equilibrium.physical.dst.gravity import (
        print_table,
        print_cell_sizes,
        print_hierarchy,
    )

    print("DST gravitational time dilation — Earth (SI units)")
    print_table()
    print()
    print_cell_sizes()
    print()
    print_hierarchy()
    print()
    unittest.main(verbosity=2)
