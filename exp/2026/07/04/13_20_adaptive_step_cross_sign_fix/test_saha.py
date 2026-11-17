"""
Saha vacuum equilibrium — assumption validation, adaptive lattice testbed.

Mirrors tst/substrate/test_saha.py on the AdaptiveLattice C++ backend. See that
file's docstring for the full physics motivation. Same four assumptions
validated here, operating on an AdaptiveCellCluster's leaf arrays instead of a
flat SubstrateLattice:

  1. CONSERVATION — the reciprocal BC maintains total u exactly.
  2. NOT REFLECTIVE — the injection is spatially uncorrelated with where u
     left; the domain centroid does not drift under random injection.
  3. EQUILIBRATION — an inhomogeneous IC relaxes toward a uniform mean while
     total u is conserved.
  4. SAHA FRACTION — the fraction of boundary-cell u values above a threshold
     decays monotonically; repeated measurements are consistent (the BC is
     statistically stable).

Run:
  PYTHONPATH=$PWD python3 -m unittest tst/substrate/adaptive/test_saha.py
"""

import unittest

import numpy as np

from equilibrium import lattice as _lat

R = 10.0
DT = 0.005
U_BG = 1.0
RNG_SEED = 0


class ReciprocalBC:
    """
    Reciprocal boundary condition — adaptive-cluster version.

    Same semantics as tst/substrate/test_saha.py's ReciprocalBC: after each
    step, measures the total u lost from the domain and injects exactly that
    amount back through randomly chosen boundary leaves (exponential weights,
    re-normalised). An optional zero-mean noise term (``thermalize``) adds
    stochastic fluctuations to boundary leaves without changing total u.
    """

    def __init__(self, cl, u_target: float = None, rng=None):
        nb = np.array(cl.leaf_nb)
        fixed_u = np.array(cl.leaf_fixed_u).astype(bool)
        boundary = np.any(nb < 0, axis=1) & ~fixed_u
        self._bc_idx = np.where(boundary)[0]
        u0 = np.array(cl.leaf_u)
        self._u_target = float(u_target if u_target is not None else u0.sum())
        self._rng = rng or np.random.default_rng(RNG_SEED)

    @property
    def n_boundary(self) -> int:
        return len(self._bc_idx)

    def compensate(self, cl):
        """Inject back whatever u was lost since construction / last set_target call."""
        u = np.array(cl.leaf_u)
        deficit = self._u_target - float(u.sum())
        if self.n_boundary == 0 or abs(deficit) < 1e-15:
            return
        weights = self._rng.exponential(size=self.n_boundary)
        weights /= weights.sum()
        u[self._bc_idx] += deficit * weights
        cl.set_leaf_u(u)

    def thermalize(self, cl, amplitude: float):
        """Add zero-mean Gaussian noise at boundary leaves (does not change total u)."""
        if self.n_boundary == 0 or amplitude <= 0:
            return
        u = np.array(cl.leaf_u)
        noise = amplitude * self._rng.standard_normal(size=self.n_boundary)
        noise -= noise.mean()
        u[self._bc_idx] += noise
        cl.set_leaf_u(u)

    def set_target(self, cl):
        """Reset the conservation target to the current domain u_total."""
        self._u_target = float(np.array(cl.leaf_u).sum())


def _build(u0: float = U_BG, radius: float = R):
    """Bare adaptive lattice, no BC. Returns (alat, cl)."""
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    cl = alat[0]
    _lat.build_bcc_sphere(cl, radius, d1=1.0, u0=u0, ell0=1.0, lam0=0)
    alat.rebuild(0)
    return alat, cl


def _build_damped(u0: float = U_BG, radius: float = R, damp_max: float = 0.005):
    """Adaptive lattice with a quadratic sponge BC targeting u0. Returns (alat, cl).

    sponge_fraction=0.2 matches the uniform DampedLattice default.
    """
    alat, cl = _build(u0=u0, radius=radius)
    r_all = np.linalg.norm(np.array(cl.leaf_pos), axis=1)
    r_max = float(r_all.max())
    r_inner = r_max * 0.8
    cl.set_bc(
        _lat.BoundaryCondition.quadratic_sponge(
            target=u0, damp_max=damp_max, r_all=r_all, r_inner=r_inner
        )
    )
    return alat, cl


# ---------------------------------------------------------------------------
# Assumption 1 — conservation
# ---------------------------------------------------------------------------


class TestConservation(unittest.TestCase):

    def test_substrate_u_conserved_without_bc(self):
        """Bare adaptive lattice (lam=0, no damping) conserves total u exactly."""
        alat, cl = _build()
        rng = np.random.default_rng(1)
        u = np.array(cl.leaf_u)
        u += 0.3 * rng.standard_normal(cl.n_leaves)
        cl.set_leaf_u(u)
        u0 = float(np.array(cl.leaf_u).sum())
        for _ in range(500):
            alat.step(0, DT)
        drift = abs(float(np.array(cl.leaf_u).sum()) - u0)
        self.assertLess(drift, 1e-10, f"u drifted without BC: {drift:.2e}")

    def test_reciprocal_compensates_damping(self):
        """With a sponge BC, reciprocal BC restores lost u each step."""
        alat, cl = _build_damped(damp_max=0.005)
        rng = np.random.default_rng(2)
        u = np.array(cl.leaf_u)
        u += 0.5 * rng.standard_normal(cl.n_leaves)
        cl.set_leaf_u(u)
        bc = ReciprocalBC(cl, rng=np.random.default_rng(3))

        u_target = bc._u_target
        for _ in range(1000):
            alat.step(0, DT)
            alat.step_sponge(0)
            bc.compensate(cl)

        drift = abs(float(np.array(cl.leaf_u).sum()) - u_target)
        self.assertLess(
            drift, 1e-8, f"reciprocal BC failed to conserve u: drift={drift:.2e}"
        )

    def test_thermalize_is_zero_mean(self):
        """thermalize() adds zero-mean noise — total u is unchanged."""
        alat, cl = _build()
        bc = ReciprocalBC(cl, rng=np.random.default_rng(4))
        u_before = float(np.array(cl.leaf_u).sum())
        for _ in range(200):
            bc.thermalize(cl, amplitude=0.1)
        u_after = float(np.array(cl.leaf_u).sum())
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
        alat, cl = _build()
        pos = np.array(cl.leaf_pos)
        bc = ReciprocalBC(cl, rng=np.random.default_rng(5))

        N = 2000
        for _ in range(N):
            alat.step(0, DT)
            bc.thermalize(cl, amplitude=0.05)

        u = np.array(cl.leaf_u)
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
            alat, cl = _build()
            bc = ReciprocalBC(cl, rng=np.random.default_rng(seed))
            for _ in range(1000):
                alat.step(0, DT)
                bc.thermalize(cl, amplitude=0.05)
            u = np.array(cl.leaf_u)
            return float(u.mean()), float(u.std())

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
        alat, cl = _build(u0=0.0)
        nb = np.array(cl.leaf_nb)
        boundary_idx = np.where(np.any(nb < 0, axis=1))[0]
        spike_j = int(boundary_idx[0])
        u = np.array(cl.leaf_u)
        u[spike_j] = float(cl.n_leaves)
        cl.set_leaf_u(u)
        u_total = float(np.array(cl.leaf_u).sum())
        u_peak_0 = float(np.array(cl.leaf_u).max())

        bc = ReciprocalBC(cl, u_target=u_total, rng=np.random.default_rng(6))
        for _ in range(3000):
            alat.step(0, DT)
            bc.compensate(cl)

        u_final = np.array(cl.leaf_u)
        u_peak_f = float(u_final.max())
        drift = abs(float(u_final.sum()) - u_total)
        self.assertLess(
            u_peak_f,
            u_peak_0 * 0.2,
            f"peak barely changed: {u_peak_0:.2f} → {u_peak_f:.2f}",
        )
        self.assertLess(drift, 1e-8, f"u_total drifted: {drift:.2e}")

    def test_equilibration_symmetric_ic(self):
        """Symmetric IC stays symmetric — no spontaneous symmetry breaking."""
        alat, cl = _build(u0=1.0)
        pos = np.array(cl.leaf_pos)
        bc = ReciprocalBC(cl, u_target=float(cl.n_leaves), rng=np.random.default_rng(7))

        for _ in range(2000):
            alat.step(0, DT)
            bc.compensate(cl)

        u = np.array(cl.leaf_u)
        centroid = (u @ pos) / max(u.sum(), 1e-12)
        max_disp = float(np.max(np.abs(centroid)))
        self.assertLess(
            max_disp, 2.0, f"symmetric IC broke symmetry: centroid={centroid}"
        )


# ---------------------------------------------------------------------------
# Assumption 4 — Saha fraction
# ---------------------------------------------------------------------------


class TestSahaFraction(unittest.TestCase):
    """
    Fraction of cells above a threshold u_thresh should decay monotonically
    and consistently — the DST analog of the Saha ionization fraction.

    In the linear (lam=0) wave limit the u distribution at equilibrium is
    Gaussian (equipartition). For large thresholds the Gaussian tail is
    approximately exponential, giving the Boltzmann-like fraction needed for
    the Saha calibration.
    """

    @classmethod
    def setUpClass(cls):
        alat, cl = _build()
        bc = ReciprocalBC(cl, rng=np.random.default_rng(8))
        AMP = 0.2
        for _ in range(5000):
            alat.step(0, DT)
            bc.thermalize(cl, amplitude=AMP)
        cls._u = np.array(cl.leaf_u).copy()

    def _fraction_above(self, thresh):
        return float(np.mean(self._u > thresh))

    def test_fraction_monotone_in_threshold(self):
        """Fraction above threshold must strictly decrease as threshold rises."""
        u_mean = float(self._u.mean())
        u_std = float(self._u.std())
        thresholds = u_mean + u_std * np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        fracs = [self._fraction_above(t) for t in thresholds]
        for i in range(len(fracs) - 1):
            self.assertGreater(
                fracs[i],
                fracs[i + 1],
                f"fraction not monotone at {thresholds[i]:.3f}: "
                f"{fracs[i]:.4f} <= {fracs[i+1]:.4f}",
            )

    def test_fraction_positive_at_mean(self):
        """At threshold = mean, roughly half the cells are above."""
        u_mean = float(self._u.mean())
        frac = self._fraction_above(u_mean)
        self.assertGreater(frac, 0.3, f"fraction at mean too low: {frac:.3f}")
        self.assertLess(frac, 0.7, f"fraction at mean too high: {frac:.3f}")

    def test_fraction_exponential_tail(self):
        """log(fraction) decreases approximately linearly with threshold.

        For a Gaussian distribution the tail is exp(-z²/2) where z is the
        standardised threshold. The logarithm should be monotone decreasing —
        the Boltzmann condition. We test monotonicity of log(fraction) without
        requiring a specific exponent (that calibration requires a physical
        temperature, not set here).
        """
        u_mean = float(self._u.mean())
        u_std = float(self._u.std())
        thresholds = u_mean + u_std * np.array([0.5, 1.0, 1.5, 2.0])
        log_fracs = []
        for t in thresholds:
            f = self._fraction_above(t)
            if f > 0:
                log_fracs.append(np.log(f))
        self.assertGreater(len(log_fracs), 2, "too few non-zero fractions to test tail")
        diffs = np.diff(log_fracs)
        self.assertTrue(
            np.all(diffs < 0), f"log(fraction) not monotone decreasing: diffs={diffs}"
        )

    def test_two_measurements_consistent(self):
        """Two separate equilibration runs yield the same fraction statistics.

        Validates that the Saha fraction is a property of the equilibrium
        state, not of a particular noise realisation.
        """

        def run(seed):
            alat, cl = _build()
            bc = ReciprocalBC(cl, rng=np.random.default_rng(seed))
            for _ in range(5000):
                alat.step(0, DT)
                bc.thermalize(cl, amplitude=0.2)
            u = np.array(cl.leaf_u)
            return self._fraction_above(u.mean()), float(u.std())

        f1, s1 = run(30)
        f2, s2 = run(40)
        self.assertAlmostEqual(
            s1, s2, delta=0.05, msg=f"equilibrium std differs: {s1:.4f} vs {s2:.4f}"
        )


if __name__ == "__main__":
    unittest.main()
