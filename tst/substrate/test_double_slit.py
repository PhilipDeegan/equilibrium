"""
Double-slit diffraction tests on the Kelvin cell substrate.

Tests verify:
  1. Wave propagation is stable.
  2. The dense slab transmits (dose behind the solid part is non-zero, just
     well below the dose behind an aperture).
  3. Two slits give a backplate dose pattern that differs from one slit
     (interference, not just a wider aperture).

Geometry/drive defaults (see equilibrium/substrate/double_slit.py) are
tuned so: the source's drive frequency sits well inside the lattice's
propagating band (below the dispersion cutoff: 6.93 at W = sqrt3:1) and the
slit separation is several wavelengths (not sub-wavelength, which gives
under one fringe period across the whole screen).
"""

import unittest

import numpy as np

from equilibrium.substrate.double_slit import (
    _wall_cells,
    run,
    screen_dose_slice,
    screen_leakage,
)


class TestWaveStability(unittest.TestCase):
    def test_stable(self):
        lat, dose = run(n_slits=2)
        self.assertFalse(np.any(np.isnan(lat.u)))
        self.assertFalse(np.any(np.isinf(lat.u)))
        self.assertFalse(np.any(np.isnan(dose)))
        self.assertLess(np.abs(lat.u).max(), 10.0)


class TestWallLeaks(unittest.TestCase):
    def test_body_dose_nonzero_but_small(self):
        """Dose behind the wall BODY is non-zero and below the dose behind the aperture. The
        dense slab is an index contrast (delay), not an absorber, so "how far below" is a
        geometry/run-length detail (0.52 of the aperture dose at the defaults with sigma at the
        face on both legs, 2026-09-29), not a property to pin."""
        lat, dose = run(wall_lam=300.0, n_slits=1)
        behind_slit, behind_body = screen_leakage(lat, dose)

        self.assertGreater(
            behind_body, 0.0, "wall body must leak some dose, not be a hard block"
        )
        self.assertLess(
            behind_body,
            behind_slit,
            "dose behind the slab body must be below the dose behind the aperture",
        )


class TestTwoSlitDiffersFromOneSlit(unittest.TestCase):
    def test_pattern_differs(self):
        lat1, dose1 = run(n_slits=1, wall_lam=300.0)
        lat2, dose2 = run(n_slits=2, wall_lam=300.0)

        _, d1 = screen_dose_slice(lat1, dose1)
        _, d2 = screen_dose_slice(lat2, dose2)

        n = min(len(d1), len(d2))
        diff = np.abs(d1[:n] - d2[:n]).sum()
        self.assertGreater(diff, 0.0, "one-slit and two-slit dose patterns must differ")

    def test_two_slit_not_just_a_scaled_envelope(self):
        """
        The two-slit/one-slit dose ratio should NOT be constant across y --
        a merely wider aperture (no real interference) would scale the
        envelope uniformly; genuine interference makes the ratio vary
        (the two-slit curve is non-monotonic where the one-slit curve is
        smooth, see module calibration notes).
        """
        lat1, dose1 = run(n_slits=1, wall_lam=300.0)
        lat2, dose2 = run(n_slits=2, wall_lam=300.0)

        _, d1 = screen_dose_slice(lat1, dose1)
        _, d2 = screen_dose_slice(lat2, dose2)

        n = min(len(d1), len(d2))
        # restrict to the well-illuminated centre, away from the sponge-
        # affected far edge
        centre = slice(n // 4, 3 * n // 4)
        ratio = d2[centre] / d1[centre]
        self.assertGreater(
            float(np.std(ratio)),
            0.02 * float(np.mean(ratio)),
            "two-slit/one-slit ratio must vary across y (interference), "
            "not sit at one constant scale factor",
        )


class TestWallCells(unittest.TestCase):
    def test_aperture_excluded_from_wall(self):
        from equilibrium.substrate.lattice import SubstrateLattice

        lat = SubstrateLattice(radius_cells=10.0)
        wall = _wall_cells(
            lat, x_wall=0.0, thickness=1.5, slit_centres=[(0.0, 0.0)], slit_radius=1.5
        )
        in_slab = np.abs(lat.pos[:, 0]) <= 0.75
        near_centre = np.sqrt(lat.pos[:, 1] ** 2 + lat.pos[:, 2] ** 2) <= 1.5
        should_be_open = np.where(in_slab & near_centre)[0]
        self.assertTrue(np.all(~np.isin(should_be_open, wall)))


if __name__ == "__main__":
    unittest.main()
