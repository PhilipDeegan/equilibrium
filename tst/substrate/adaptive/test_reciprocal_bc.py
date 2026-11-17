"""
Reciprocal boundary condition on the AdaptiveLattice C++ backend (ReciprocalBCCluster,
equilibrium/substrate/reciprocal_bc.py). Mirrors tst/substrate/test_reciprocal_bc.py:

  1. CONSERVATION -- the reciprocal BC maintains total u exactly.
  2. NOT REFLECTIVE -- the injection is spatially uncorrelated with where u left; the domain
     centroid does not drift under random injection.
  3. EQUILIBRATION -- an inhomogeneous IC relaxes toward a uniform mean while total u is
     conserved.
plus the reciprocal-vs-sponge comparison in the ring / cross_scale context.

Was adaptive/test_saha.py (renamed 2026-09-29); its fraction-above-threshold tests measured a
noise random walk and were removed (thermal state: tst/substrate/test_ambient.py).

Run:
  PYTHONPATH=$PWD python3 -m unittest tst/substrate/adaptive/test_reciprocal_bc.py
"""

import unittest

import numpy as np

from equilibrium import lattice as _lat
from equilibrium.substrate.reciprocal_bc import ReciprocalBCCluster as ReciprocalBC
from tst.substrate.adaptive import compute_dt
from tst.substrate.adaptive.adaptive_flux_tube import (
    find_toroidal_spheromak,
    init_toroidal_winding,
    phi_ring_circulation,
)

R = 10.0
DT = 0.005
U_BG = 1.0
RNG_SEED = 0


def _build(u0: float = U_BG, radius: float = R):
    """Bare adaptive lattice, no BC. Returns (alat, cl)."""
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    cl = alat[0]
    _lat.build_bcc_sphere(cl, radius, d1=1.0, u0=u0, ell0=1.0, lam0=0)
    alat.rebuild(0)
    return alat, cl


def _build_ring(
    radius: float = 15.0,
    axis: str = "z",
    amplitude: float = 1.0,
    lam_rings: float = 1.0,
):
    """Hopf-ring spheromak IC (same construction as
    sweep_spheromak_decay_cross_scale.py), no BC set yet. Returns
    (alat, cl, A_js, B_js, dt). radius=15 (not the module's R=10) — needs
    enough room around the origin for the linked A+B ring pair; matches the
    H0 Bohr-orbit study's fast-test convention (project_h0_study memory).
    """
    alat, cl = _build(u0=0.0, radius=radius)
    A_js, A_faces, B_js, B_faces = find_toroidal_spheromak(cl, axis=axis)
    if A_js is None:
        raise RuntimeError(f"no toroidal spheromak found at R={radius} — too small?")
    init_toroidal_winding(cl, A_js, A_faces, amplitude=amplitude, phi_only=True)
    init_toroidal_winding(cl, B_js, B_faces, amplitude=amplitude, phi_only=True)
    if lam_rings > 0.0:
        lam = np.array(cl.leaf_lam)
        for i in set(A_js) | set(B_js):
            lam[i] = lam_rings
        cl.set_leaf_lam(lam)
    dt = compute_dt(cl)
    return alat, cl, A_js, B_js, dt


def _boundary_phi_frac(cl, sponge_frac: float = 0.15):
    """Fraction of total phi-energy sitting in the outer sponge shell
    (r > r_max*(1-sponge_frac)) — the diagnostic behind the cross_scale
    boundary-buildup finding (project_cross_scale_calibration memory)."""
    pos = np.array(cl.leaf_pos)
    phi = np.array(cl.leaf_phi)  # (14, N)
    r = np.linalg.norm(pos, axis=1)
    r_inner = float(r.max()) * (1.0 - sponge_frac)
    shell = r > r_inner
    return float(np.sum(phi[:, shell] ** 2) / max(np.sum(phi**2), 1e-300))


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
# ReciprocalBC vs the absorbing sponge, in the actual ring/cross_scale context
# ---------------------------------------------------------------------------


class TestReciprocalWithRingDecay(unittest.TestCase):
    """
    The cross_scale calibration sweep (sweep_spheromak_decay_cross_scale.py)
    found that its absorbing quadratic_sponge develops a sharp boundary-shell
    buildup right at r_inner=R*(1-sponge) — confirmed (2026-07-05, R=40 vs
    R=60 comparison) to be a soft-onset BC-seam artifact: zero damping exactly
    at r_inner lets outward flux from step_cross dam up there before the ramp
    is strong enough (further out) to absorb it. See
    project_cross_scale_calibration memory for the full writeup.

    This does NOT use thermalize()/noise — only compensate() (pure
    conservation, no injected thermal bath) — so it's a different combination
    from the one feedback_saha_noise_scope warns against (a FIXED noise floor
    swamping a decay signal). The question here is whether replacing the
    sponge's one-way drain with reciprocal reinjection (a stand-in for "the
    rest of the universe," not vacuum) removes the artifact, not whether
    ambient thermal noise is compatible with decay probes.

    R=15 (not the sweep's R=40) purely for test speed — the buildup mechanism
    is r_inner-relative, not tied to a specific domain size (that was exactly
    the R-scan finding).
    """

    CROSS_SCALE = 0.05
    LAM_SCALE = 0.05
    N_STEPS = 4000
    SPONGE_FRAC = 0.15

    def _run(self, use_reciprocal):
        alat, cl, A_js, B_js, dt = _build_ring(radius=15.0)
        centre_B = np.array(cl.leaf_pos)[B_js].mean(axis=0)

        if use_reciprocal:
            bc = ReciprocalBC(cl, rng=np.random.default_rng(0))
        else:
            r_all = np.linalg.norm(np.array(cl.leaf_pos), axis=1)
            r_max = float(r_all.max())
            r_inner = r_max * (1.0 - self.SPONGE_FRAC)
            cl.set_bc(
                _lat.BoundaryCondition.quadratic_sponge(
                    target=0.0, damp_max=0.003, r_all=r_all, r_inner=r_inner
                )
            )

        circ_0 = phi_ring_circulation(cl, B_js, centre_B)

        for _ in range(self.N_STEPS):
            alat.step(0, dt)
            if use_reciprocal:
                bc.compensate(cl)
            else:
                alat.step_sponge(0)
            alat.step_cross(0, dt, self.CROSS_SCALE)
            alat.step_lam(0, dt, self.LAM_SCALE)

        circ_f = phi_ring_circulation(cl, B_js, centre_B)
        bnd_frac = _boundary_phi_frac(cl, self.SPONGE_FRAC)
        return circ_0, circ_f, bnd_frac

    def test_reciprocal_alone_is_worse_than_sponge(self):
        """Measured 2026-07-05: swapping the sponge for compensate() alone
        makes the boundary buildup WORSE (~35x), not better, at these
        constants (R=15, cross_scale=0.05, N_STEPS=4000) — the opposite of
        the original hypothesis. compensate() is a pure global u-bookkeeping
        correction; it does no local wave absorption. Removing the sponge
        removes the only absorbing mechanism in the domain, so outgoing
        phi/u waves reach the true edge and reflect (missing-neighbour terms
        just drop out of the flux sum — a natural no-flux/reflecting
        condition, not a radiating one), which piles up MORE energy at the
        boundary than the (imperfect) absorbing sponge did. Conclusion: a
        working version of "pretend there's more universe out there" needs
        an absorbing/non-reflecting layer to prevent this reflection PLUS
        reciprocal reinjection of what that layer removes — reciprocal
        instead of the sponge is not enough on its own. See
        project_cross_scale_calibration memory. This test pins the negative
        result so a future change silently reintroducing reflection (e.g.
        someone "fixing" this by just deleting the sponge again) gets
        flagged instead of quietly regressing."""
        _, _, bnd_sponge = self._run(use_reciprocal=False)
        _, _, bnd_recip = self._run(use_reciprocal=True)
        self.assertGreater(
            bnd_recip,
            bnd_sponge,
            f"expected reciprocal-alone to still be worse than the sponge "
            f"(no absorption at all -> reflection); if this now passes the "
            f"other way, the mechanism above may no longer apply and the "
            f"memory note should be revisited: "
            f"sponge={bnd_sponge:.3e} vs reciprocal={bnd_recip:.3e}",
        )

    def test_u_conserved_with_ring_and_cross_active(self):
        """compensate() should hold u_total fixed even with the ring's
        step_cross/step_lam physics active, not just the bare-lattice case
        TestConservation already covers."""
        alat, cl, A_js, B_js, dt = _build_ring(radius=15.0)
        bc = ReciprocalBC(cl, rng=np.random.default_rng(1))
        u_target = bc._u_target
        for _ in range(self.N_STEPS):
            alat.step(0, dt)
            bc.compensate(cl)
            alat.step_cross(0, dt, self.CROSS_SCALE)
            alat.step_lam(0, dt, self.LAM_SCALE)
        drift = abs(float(np.array(cl.leaf_u).sum()) - u_target)
        self.assertLess(
            drift, 1e-5, f"u drifted with ring+cross physics active: {drift:.2e}"
        )


if __name__ == "__main__":
    unittest.main()
