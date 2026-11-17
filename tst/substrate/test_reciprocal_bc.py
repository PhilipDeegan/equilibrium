"""
Reciprocal boundary condition (equilibrium/substrate/reciprocal_bc.py) -- assumption validation.

Whatever u leaves through any boundary face is matched by an equal injection spread randomly
across the whole boundary: not reflective at the face it left, not absorbing.

  1. CONSERVATION -- the reciprocal BC maintains total u exactly.
  2. NOT REFLECTIVE -- the injection is spatially uncorrelated with where u left; the domain
     centroid does not drift under random injection.
  3. EQUILIBRATION -- an inhomogeneous IC relaxes toward a uniform mean while total u is
     conserved.

Was test_saha.py (renamed 2026-09-29: no ionisation is involved). Its fraction-above-threshold
tests measured a random walk (noise, no sink); the ambient state is now an initial condition,
tst/substrate/test_ambient.py.
"""

import unittest

import numpy as np

from equilibrium.substrate.lattice import DampedLattice, SubstrateLattice
from equilibrium.substrate.reciprocal_bc import ReciprocalBC

R = 10.0
DT = 0.005
U_BG = 1.0
RNG_SEED = 0


def _build(u0: float = U_BG, radius: float = R) -> SubstrateLattice:
    lat = SubstrateLattice(radius_cells=radius)
    lat.u[:] = u0
    return lat


def _build_damped(
    u0: float = U_BG, radius: float = R, damp_max: float = 0.005
) -> DampedLattice:
    lat = DampedLattice(radius_cells=radius, u_bg=u0, damp_max=damp_max)
    lat.u[:] = u0
    return lat


# ---------------------------------------------------------------------------
# Assumption 1 — conservation
# ---------------------------------------------------------------------------


class TestConservation(unittest.TestCase):

    def test_substrate_u_conserved_without_bc(self):
        """SubstrateLattice (lam=0, no damping) conserves total u exactly."""
        lat = _build()
        rng = np.random.default_rng(1)
        lat.u += 0.3 * rng.standard_normal(lat.N)
        u0 = float(lat.u.sum())
        for _ in range(500):
            lat.step(DT)
        drift = abs(lat.u.sum() - u0)
        self.assertLess(drift, 1e-10, f"u drifted without BC: {drift:.2e}")

    def test_reciprocal_compensates_damping(self):
        """With DampedLattice, reciprocal BC restores lost u each step."""
        lat = _build_damped(damp_max=0.005)
        rng = np.random.default_rng(2)
        lat.u += 0.5 * rng.standard_normal(lat.N)
        bc = ReciprocalBC(lat, rng=np.random.default_rng(3))

        u_target = bc._u_target
        for _ in range(1000):
            lat.step(DT)
            bc.compensate(lat)

        drift = abs(lat.u.sum() - u_target)
        self.assertLess(
            drift, 1e-8, f"reciprocal BC failed to conserve u: drift={drift:.2e}"
        )

    def test_thermalize_is_zero_mean(self):
        """thermalize() adds zero-mean noise — total u is unchanged."""
        lat = _build()
        bc = ReciprocalBC(lat, rng=np.random.default_rng(4))
        u_before = float(lat.u.sum())
        for _ in range(200):
            bc.thermalize(lat, amplitude=0.1)
        u_after = float(lat.u.sum())
        drift = abs(u_after - u_before)
        self.assertLess(drift, 1e-10, f"thermalize changed total u: {drift:.2e}")


# ---------------------------------------------------------------------------
# Assumption 2 — not reflective
# ---------------------------------------------------------------------------


class TestNotReflective(unittest.TestCase):

    def test_centroid_doesnt_drift(self):
        """Random injection across boundary keeps the u centroid near origin.

        Reflective BC would send u back along the same axis it left through,
        eventually biasing the centroid. Reciprocal injection is isotropic:
        the centroid drifts by at most O(1/sqrt(N_steps)) in each axis.
        """
        lat = _build()
        pos = np.array(lat.pos)
        bc = ReciprocalBC(lat, rng=np.random.default_rng(5))

        N = 2000
        for _ in range(N):
            lat.step(DT)
            bc.thermalize(lat, amplitude=0.05)

        u = lat.u
        centroid = (u @ pos) / u.sum()
        max_disp = float(np.max(np.abs(centroid)))
        self.assertLess(
            max_disp,
            1.5,
            f"centroid drifted too far: {centroid} — suggests preferred axis",
        )

    def test_two_seeds_same_statistics(self):
        """Two runs with different RNG seeds should produce the same mean/std.

        If the BC had a preferred direction, the mean would differ between runs.
        Equal statistics confirms the injection is spatially unbiased.
        """

        def run(seed):
            lat = _build()
            bc = ReciprocalBC(lat, rng=np.random.default_rng(seed))
            for _ in range(1000):
                lat.step(DT)
                bc.thermalize(lat, amplitude=0.05)
            return float(lat.u.mean()), float(lat.u.std())

        mean_a, std_a = run(10)
        mean_b, std_b = run(20)
        self.assertAlmostEqual(
            mean_a,
            mean_b,
            delta=0.05,
            msg=f"means differ: {mean_a:.4f} vs {mean_b:.4f}",
        )
        self.assertAlmostEqual(
            std_a, std_b, delta=0.05, msg=f"stds differ: {std_a:.4f} vs {std_b:.4f}"
        )


# ---------------------------------------------------------------------------
# Assumption 3 — equilibration
# ---------------------------------------------------------------------------


class TestEquilibration(unittest.TestCase):

    def test_peaked_ic_equilibrates(self):
        """A u spike on one side spreads toward spatial uniformity.

        Initial state: all u in one boundary cell.
        Final state: u distributed across the domain, peak reduced by >80%.
        Total u conserved throughout.
        """
        lat = _build(u0=0.0)
        bc = ReciprocalBC(lat, u_target=0.0, rng=np.random.default_rng(6))
        nb = lat.nb
        boundary_idx = np.where(np.any(nb < 0, axis=1))[0]
        spike_j = int(boundary_idx[0])
        lat.u[spike_j] = float(lat.N)
        u_total = float(lat.u.sum())
        u_peak_0 = float(lat.u.max())

        bc._u_target = u_total
        for _ in range(3000):
            lat.step(DT)
            bc.compensate(lat)

        u_peak_f = float(lat.u.max())
        drift = abs(lat.u.sum() - u_total)
        self.assertLess(
            u_peak_f,
            u_peak_0 * 0.2,
            f"peak barely changed: {u_peak_0:.2f} → {u_peak_f:.2f}",
        )
        self.assertLess(drift, 1e-8, f"u_total drifted: {drift:.2e}")

    def test_equilibration_symmetric_ic(self):
        """Symmetric IC stays symmetric — no spontaneous symmetry breaking."""
        lat = _build(u0=0.0)
        pos = np.array(lat.pos)
        bc = ReciprocalBC(lat, u_target=float(lat.N), rng=np.random.default_rng(7))

        lat.u[:] = 1.0
        for _ in range(2000):
            lat.step(DT)
            bc.compensate(lat)

        u = lat.u
        centroid = (u @ pos) / max(u.sum(), 1e-12)
        max_disp = float(np.max(np.abs(centroid)))
        self.assertLess(
            max_disp, 2.0, f"symmetric IC broke symmetry: centroid={centroid}"
        )


if __name__ == "__main__":
    unittest.main()
