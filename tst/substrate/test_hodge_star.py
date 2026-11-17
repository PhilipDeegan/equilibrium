"""
Hex:square Hodge star ratio r = *1_hex / *1_sq on the BCC Delaunay / Kelvin complex
(equilibrium.substrate.bcc_symbol; scan in exp/2026/09/28/hodge_star/star_ratio.py):
- geometry      the Kelvin cell's |dual face| / |edge| is exactly 3:1, volume 1/2
- step()        step()'s u symbol (W^2 on the two legs) is the DEC 0-form symbol at r = 3,
                the Kelvin geometry (W_HEX = sqrt3 since 2026-09-28; was 3 -> r = 9)
- consistency   any r gives a direction-independent c at k -> 0 in both sectors:
                c_u^2 = (r+1)/4, c_T^2 = 4/(r+1) (with *2 = 1); equal only at r = 3
- isotropy      the 2nd-order speed error is direction independent only at r = 2, in both
                the u and the transverse sector (no birefringence) -- doc/dst.md's
                w = 2/3 : 1/3; r = 3 and r = 9 are anisotropic, 9 the more so
"""

import unittest

import numpy as np

from equilibrium.substrate import bcc_symbol as B
from equilibrium.substrate import casimir


def _spread(r, dirs):
    res = [B.speed_error(n, r) for n in dirs]
    return np.ptp([a for a, _ in res]), np.ptp([b for _, b in res])


class TestHodgeStar(unittest.TestCase):
    dirs = B.directions(60)

    def test_kelvin_geometry_is_3_to_1(self):
        sh, ss, vol = B.kelvin_star1()
        self.assertAlmostEqual(sh / ss, 3.0, places=12)
        self.assertAlmostEqual(vol, B.STAR0, places=12)

    def test_step_is_kelvin_3_to_1(self):
        # casimir._dispersion is step()'s omega (W on both legs), cube constant a
        k = np.random.default_rng(2).uniform(-3, 3, size=(20, 3))
        w2_step = casimir._dispersion(k[:, 0], k[:, 1], k[:, 2], 1.0) ** 2
        w2_dec = np.array([B.w2_u(kk, 3) for kk in k])
        ratio = w2_step / w2_dec
        self.assertLess(np.ptp(ratio) / ratio.mean(), 1e-12)

    def test_any_ratio_is_consistent(self):
        k0 = 1e-3
        for r in (1, 2, 3, 9):
            u = np.array([B.w2_u(k0 * n, r) for n in self.dirs]) / k0**2
            T = np.array([B.w2_T(k0 * n, r) for n in self.dirs]) / k0**2
            np.testing.assert_allclose(u, (r + 1) / 4, rtol=1e-6)
            np.testing.assert_allclose(T, 4 / (r + 1), rtol=1e-5)

    def test_isotropic_only_at_2_to_1(self):
        su2, sT2 = _spread(2, self.dirs)
        self.assertLess(su2, 1e-5)
        self.assertLess(sT2, 1e-5)
        s3 = _spread(3, self.dirs)
        s9 = _spread(9, self.dirs)
        for s in (s3, s9):
            self.assertGreater(min(s), 1e-3)
        self.assertGreater(s9[0], 2 * s3[0])
        self.assertGreater(s9[1], 2 * s3[1])


if __name__ == "__main__":
    unittest.main()
