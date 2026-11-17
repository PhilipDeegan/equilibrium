"""
Tests for the multiclock (per-cell σ) functionality.

σ = 1 / (1 + λ) is always ≤ 1.  Void (λ=0, σ=1) is the master tick rate;
higher winding (λ>0) reduces throughput.  Tests set lam directly to produce
sigma gradients.
"""

import unittest

import numpy as np

from equilibrium.substrate.lattice import SubstrateLattice


class TestMulticlock(unittest.TestCase):
    def test_backward_compatibility(self):
        """
        With lam=0 everywhere (uniform void), sigma=1 everywhere and dynamics
        are identical to the original implementation.
        """
        lat = SubstrateLattice(radius_cells=5.0)
        self.assertTrue(np.all(lat.lam == 0.0))
        self.assertTrue(np.all(lat.sigma == 1.0))

        c0 = int(np.argmin(np.linalg.norm(lat.pos, axis=1)))
        a = float(lat.pos[int(lat.nb[c0, 8]), 0])
        n_cells = 4
        K = 2 * np.pi / (n_cells * a)
        lat.u = np.sin(K * lat.pos[:, 0])

        i_peak = int(np.argmax(lat.u))
        self.assertAlmostEqual(float(lat.u[i_peak]), 1.0, places=10)

        dt = lat.stable_dt()
        u_prev = float(lat.u[i_peak])
        t_cross = None
        for n in range(1, 100):
            lat.step(dt)
            u_curr = float(lat.u[i_peak])
            if u_curr <= 0.0 and u_prev > 0.0:
                t_cross = dt * (n - 1 + u_prev / (u_prev - u_curr))
                break
            u_prev = u_curr

        self.assertIsNotNone(t_cross)
        # Regression pin, NOT an analytic prediction: the plane wave is not an
        # eigenmode of the finite free-boundary sphere, so t_cross depends on
        # the R=5 fixture (R=8 gives ~0.391, R=12 ~0.388).  0.4535 is the
        # stable measured value for this fixture; the previous pin (0.48)
        # predates the repo history (fails at the init commit) and its
        # provenance is unrecoverable.  Re-pinned 2026-07-16.  Re-pinned
        # 2026-09-28 for step()'s W = sqrt3:1 (was 3:1): 0.4535 -> 0.723, x1.595,
        # the c ratio 3.651/2.309 = 1.581 to within the fixture's non-eigenmode slack.
        self.assertAlmostEqual(t_cross, 0.723, delta=0.02)

    def test_phi_antisymmetry(self):
        """
        Verify that dphi remains antisymmetric even with a lam gradient
        (and therefore a sigma gradient), thanks to the symmetric face clock.
        phi[i,k] + phi[j, mirror(k)] must remain zero.
        """
        mirror = np.array([7, 6, 5, 4, 3, 2, 1, 0, 9, 8, 11, 10, 13, 12])
        lat = SubstrateLattice(radius_cells=5.0)

        # sigma gradient via lam: sigma = 0.7 + 0.1*x  →  lam = 1/sigma - 1
        sigma_target = np.clip(0.7 + 0.1 * lat.pos[:, 0], 0.1, 1.0)
        lat.lam = (1.0 / sigma_target) - 1.0
        lat.u = lat.pos[:, 1].copy()
        lat.step(lat.stable_dt())

        max_err = 0.0
        for k in range(14):
            mk = int(mirror[k])
            j = lat.nb[:, k]
            mask = j >= 0
            errs = lat.phi[mask, k] + lat.phi[j[mask], mk]
            if errs.size:
                max_err = max(max_err, float(np.max(np.abs(errs))))

        self.assertEqual(max_err, 0.0, "phi antisymmetry broken by lam gradient")

    def test_gps_time_dilation(self):
        """
        Validate the core multiclock prediction: oscillation frequency is
        proportional to sigma = 1/(1+lam).

        T_void is the master tick (lam=0, sigma=1).  A wound region has
        lam > 0 and oscillates proportionally slower.

        Region A: void — lam=0.0, sigma=1.0
        Region B: wound — lam=0.25, sigma=0.8
        Expected freq_A / freq_B = sigma_A / sigma_B = 1.0 / 0.8 = 1.25.
        """
        lat = SubstrateLattice(radius_cells=8.0)

        lam_A = 0.0  # void — master tick rate
        lam_B = 0.25  # wound region — sigma_B = 1/1.25 = 0.8

        region_A = np.where(lat.pos[:, 0] < -4)[0]
        region_B = np.where(lat.pos[:, 0] > 4)[0]
        self.assertGreater(len(region_A), 0)
        self.assertGreater(len(region_B), 0)

        lat.lam[region_A] = lam_A
        lat.lam[region_B] = lam_B
        # smooth transition
        mid_mask = (lat.pos[:, 0] >= -4) & (lat.pos[:, 0] <= 4)
        x_mid = lat.pos[mid_mask, 0]
        lat.lam[mid_mask] = lam_A + (lam_B - lam_A) * (x_mid + 4) / 8.0

        i_A = region_A[int(np.argmin(lat.pos[region_A, 0]))]
        i_B = region_B[int(np.argmax(lat.pos[region_B, 0]))]

        lat.u[i_A] = 1.0
        lat.u[i_B] = 1.0

        dt = lat.stable_dt()
        u_prev_A, u_prev_B = 1.0, 1.0
        t_cross_A, t_cross_B = None, None

        for n in range(1, 200):
            lat.step(dt)
            u_curr_A = float(lat.u[i_A])
            u_curr_B = float(lat.u[i_B])

            if t_cross_A is None and u_curr_A <= 0.0 and u_prev_A > 0.0:
                t_cross_A = dt * (n - 1 + u_prev_A / (u_prev_A - u_curr_A))
            if t_cross_B is None and u_curr_B <= 0.0 and u_prev_B > 0.0:
                t_cross_B = dt * (n - 1 + u_prev_B / (u_prev_B - u_curr_B))

            if t_cross_A is not None and t_cross_B is not None:
                break
            u_prev_A, u_prev_B = u_curr_A, u_curr_B

        self.assertIsNotNone(t_cross_A, "Clock A (void) did not oscillate")
        self.assertIsNotNone(t_cross_B, "Clock B (wound) did not oscillate")

        # freq_A / freq_B = t_cross_B / t_cross_A = sigma_A / sigma_B
        freq_ratio = t_cross_B / t_cross_A
        sigma_ratio = 1.0 / (1.0 + lam_A) / (1.0 / (1.0 + lam_B))  # 1.25

        self.assertAlmostEqual(freq_ratio, sigma_ratio, delta=0.02)


if __name__ == "__main__":
    unittest.main()
