"""
Validation of equilibrium.substrate.loop_antenna -- the classical finite-
loop far-field reference for the DST Poynting-ring comparison. No lattice
stepping, pure analytic checks against known limits/properties.
"""

import unittest

import numpy as np

from equilibrium.substrate.loop_antenna import (
    far_zone_slope,
    kr,
    pattern_shape,
    point_dipole_shape,
)

THETA = np.linspace(1e-3, np.pi - 1e-3, 361)


class TestSmallKaLimitRecoversPointDipole(unittest.TestCase):
    """ka -> 0 must recover the plain sin^2(theta) dipole pattern exactly
    at ka=0, and increasingly closely as ka shrinks toward it."""

    def test_ka_zero_is_exactly_sin_squared(self):
        np.testing.assert_allclose(
            pattern_shape(THETA, ka=0.0), point_dipole_shape(THETA), atol=1e-12
        )

    def test_small_ka_converges_to_point_dipole(self):
        errs = [
            float(np.max(np.abs(pattern_shape(THETA, ka) - point_dipole_shape(THETA))))
            for ka in (1e-1, 1e-2, 1e-3)
        ]
        self.assertTrue(
            errs[0] > errs[1] > errs[2], f"not monotonically converging: {errs}"
        )
        self.assertLess(errs[2], 1e-6)


class TestPatternShapeBasicProperties(unittest.TestCase):
    def test_zero_at_the_pole_any_ka(self):
        """sin(0)=0 kills the pattern at theta=0 regardless of ka -- a
        constant-current loop, like a point dipole, radiates nothing along
        its own axis. (DST's measured nonzero small-theta signal is
        therefore NOT explained by this shape function at all -- it must
        come from elsewhere, e.g. BCC lattice-axis propagation, not from
        finite-loop physics.)"""
        for ka in (0.0, 0.5, 1.043, 3.0):
            self.assertAlmostEqual(float(pattern_shape(0.0, ka)), 0.0, places=12)

    def test_ka_at_dst_ring_value_deviates_from_point_dipole_at_equator(self):
        """At DST's actual ka=1.043 (ring_r=5.32, lambda=32.05, both from
        project_poynting_dipole_calibration), confirm the equatorial
        (theta=pi/2) value is a REAL, nontrivial deviation from the naive
        sin^2 prediction -- not a negligible correction."""
        ka = 1.043
        shape_val = float(pattern_shape(np.pi / 2, ka))
        dipole_val = float(point_dipole_shape(np.pi / 2))
        self.assertAlmostEqual(dipole_val, 1.0, places=12)
        self.assertLess(shape_val, dipole_val)
        # 2*J_1(1.043)/1.043 ~ 0.870 (vs the ka->0 normalised limit of 1.0)
        # -> shape ~ 0.757, a real ~24% deviation, not negligible
        self.assertAlmostEqual(shape_val, 0.757, delta=0.005)

    def test_no_extra_sidelobes_yet_at_dst_ka(self):
        """First null of the loop pattern is at ka*sin(theta) = first zero
        of J_1 (~3.8317). DST's ka=1.043 can reach at most ka*1=1.043,
        nowhere near the first null -- the pattern should still be a single
        broad lobe peaking at the equator, not yet developing structure."""
        ka = 1.043
        vals = pattern_shape(THETA, ka)
        i_max = int(np.argmax(vals))
        theta_at_max = THETA[i_max]
        self.assertAlmostEqual(theta_at_max, np.pi / 2, delta=0.05)
        # monotonically increasing from pole to equator on [0, pi/2]
        half = vals[: len(vals) // 2 + 1]
        self.assertTrue(
            np.all(np.diff(half) >= -1e-9), "unexpected non-monotonic lobe structure"
        )


class TestFarZoneSlopeIsUniversal(unittest.TestCase):
    """The radial exponent target does not depend on ka -- this is the
    calibration value the (not-yet-run) large-r Poynting rerun should
    converge to at BOTH polar and equatorial angles, once kr is large
    enough, regardless of the ring's finite size."""

    def test_far_zone_slope_is_minus_two(self):
        self.assertEqual(far_zone_slope(), -2.0)


class TestKrDiagnostic(unittest.TestCase):
    """Reproduces the concrete numbers that motivated skepticism of the
    original -6.19 equatorial slope fit: r=8-40 at lambda=32.05 spans
    kr=1.57-7.84, nowhere near the kr>>1 regime the -2 comparison assumes."""

    def test_probed_range_is_not_deep_far_zone(self):
        lam = 32.05
        kr_lo = float(kr(8, lam))
        kr_hi = float(kr(40, lam))
        self.assertAlmostEqual(kr_lo, 1.568, places=2)
        self.assertAlmostEqual(kr_hi, 7.842, places=2)
        self.assertLess(kr_lo, 10.0, "r=8 should NOT already be deep far-zone")


if __name__ == "__main__":
    unittest.main()
