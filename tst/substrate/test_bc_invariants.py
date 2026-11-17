"""
Boundary-condition invariants, discrete and continuous models
(exp/2026/09/29/bc_review/status.md).

1. Face antisymmetry: each face is stored on both cells, phi_ki = -phi_ik (and
   e for the continuous E sector). step() keeps it; nothing restores it, and a
   broken (symmetric) part is frozen by step(). A BC must keep it.
2. dt consistency: a sponge is a damping rate. Evolving to the same time with
   dt and dt/2 must give the same damping.
3. Sponge target: a sponge relaxes toward its target; a state at the target
   is left alone.
4. Continuous sponge damps the whole state: u, A (phi) and E together.

Open wall, periodic wiring and plain steps are the controls.
"""

import unittest

import numpy as np

from equilibrium import lattice as _lat
from equilibrium.substrate import continuous as C
from equilibrium.substrate.lattice import _K_REV, DampedLattice, SubstrateLattice
from tst.substrate.adaptive import compute_dt

R = 6.0
SPONGE = 0.3
DAMP = 0.05


def _antisym_err(phi, nb):
    """max |phi_ik + phi_{nb, rev k}| over interior faces; phi, nb (N, 14)."""
    i, k = np.nonzero(nb >= 0)
    return float(np.max(np.abs(phi[i, k] + phi[nb[i, k], _K_REV[k]])))


def _random_antisym(nb, rng):
    """Random face field, antisymmetric across interior faces, 0 on boundary faces."""
    phi = np.zeros(nb.shape)
    i, k = np.nonzero(nb >= 0)
    once = i < nb[i, k]
    i, k = i[once], k[once]
    g = rng.standard_normal(len(i))
    phi[i, k] = g
    phi[nb[i, k], _K_REV[k]] = -g
    return phi


class _Discrete:
    """Level-0 C++ discrete AdaptiveLattice sphere with a random antisymmetric state."""

    def __init__(self, seed=0):
        self.alat = _lat.discrete.AdaptiveLattice()
        self.cl = self.alat[0]
        _lat.discrete.build_bcc_sphere(self.cl, R, d1=1.0)
        self.alat.rebuild(0)
        self.pos = np.array(self.cl.leaf_pos)
        self.r = np.linalg.norm(self.pos, axis=1)
        self.r_inner = float(self.r.max()) * (1.0 - SPONGE)
        self.nb = np.array(self.cl.leaf_nb)
        rng = np.random.default_rng(seed)
        self.cl.set_leaf_u(rng.standard_normal(len(self.pos)))
        self.cl.set_leaf_phi(np.ascontiguousarray(_random_antisym(self.nb, rng).T))
        self.dt = compute_dt(self.cl)

    @property
    def phi(self):
        return np.array(self.cl.leaf_phi).T

    @property
    def u(self):
        return np.array(self.cl.leaf_u)


def _continuous(seed=0):
    ac = C.anchor_cluster(R, 1.0, n_threads=1)
    rng = np.random.default_rng(seed)
    ac.u = rng.standard_normal(ac.N)
    ac.phi = _random_antisym(ac.nb, rng)
    ac.e = _random_antisym(ac.nb, rng)
    ac.r = np.linalg.norm(ac.pos, axis=1)
    ac.r_inner = float(ac.r.max()) * (1.0 - SPONGE)
    return ac


class TestFaceAntisymmetryDiscreteNumpy(unittest.TestCase):
    def _state(self, lat, seed=0):
        rng = np.random.default_rng(seed)
        lat.u[:] = rng.standard_normal(lat.N)
        lat.phi[:] = _random_antisym(lat.nb, rng)

    def test_open_wall(self):
        lat = SubstrateLattice(radius_cells=R)
        self._state(lat)
        for _ in range(20):
            lat.step(lat.stable_dt())
        self.assertLess(_antisym_err(lat.phi, lat.nb), 1e-12)

    def test_damped_lattice(self):
        lat = DampedLattice(radius_cells=R, sponge_fraction=SPONGE, damp_max=DAMP)
        self._state(lat)
        for _ in range(20):
            lat.step(lat.stable_dt())
        self.assertLess(_antisym_err(lat.phi, lat.nb), 1e-12)


class TestFaceAntisymmetryDiscreteCpp(unittest.TestCase):
    def test_open_wall(self):
        d = _Discrete()
        for _ in range(20):
            d.alat.step(0, d.dt)
        self.assertLess(_antisym_err(d.phi, d.nb), 1e-12)

    def test_quadratic_sponge(self):
        d = _Discrete()
        BC = _lat.discrete.BoundaryCondition
        d.cl.set_bc(BC.quadratic_sponge(target=0.0, damp_max=DAMP, r_all=d.r, r_inner=d.r_inner))
        for _ in range(20):
            d.alat.step(0, d.dt)
            d.alat.step_sponge(0)
        self.assertLess(_antisym_err(d.phi, d.nb), 1e-12)

    def test_quadratic_sponge_rate(self):
        d = _Discrete()
        BC = _lat.discrete.BoundaryCondition
        d.cl.set_bc(BC.quadratic_sponge(target=0.0, gamma_max=DAMP / d.dt, r_all=d.r, r_inner=d.r_inner))
        for _ in range(20):
            d.alat.step(0, d.dt)
            d.alat.step_sponge(0, d.dt)
        self.assertLess(_antisym_err(d.phi, d.nb), 1e-12)

    def test_quadratic_sponge_integrated(self):
        d = _Discrete()
        BC = _lat.discrete.BoundaryCondition
        d.cl.set_bc(
            BC.quadratic_sponge_integrated(target=0.0, gamma_max=DAMP / d.dt, r_all=d.r, r_inner=d.r_inner)
        )
        for _ in range(20):
            d.alat.step_damped(0, d.dt)
        self.assertLess(_antisym_err(d.phi, d.nb), 1e-12)

    def test_quadratic_sponge_directional(self):
        d = _Discrete()
        BC = _lat.discrete.BoundaryCondition
        d.cl.set_bc(
            BC.quadratic_sponge_directional(target=0.0, gamma_max=DAMP / d.dt, pos_all=d.pos, r_inner=d.r_inner)
        )
        for _ in range(20):
            d.alat.step_dir_damped(0, d.dt)
        self.assertLess(_antisym_err(d.phi, d.nb), 1e-12)

    def test_quadratic_sponge_pml(self):
        d = _Discrete()
        BC = _lat.discrete.BoundaryCondition
        d.cl.set_bc(BC.quadratic_sponge_pml(target=0.0, gamma_max=DAMP / d.dt, pos_all=d.pos, r_inner=d.r_inner))
        for _ in range(20):
            d.alat.step_pml(0, d.dt)
        self.assertLess(_antisym_err(d.phi, d.nb), 1e-12)


class TestFaceAntisymmetryContinuous(unittest.TestCase):
    def _sponge(self, ac):
        BC = _lat.continuous.BoundaryCondition
        ac.cl.set_bc(BC.quadratic_sponge(target=0.0, damp_max=DAMP, r_all=ac.r, r_inner=ac.r_inner))

    def test_step_ae_open(self):
        ac = _continuous()
        dt = C.stable_dt(ac)
        for _ in range(20):
            ac.step_ae(dt)
        self.assertLess(_antisym_err(ac.phi, ac.nb), 1e-12)
        self.assertLess(_antisym_err(ac.e, ac.nb), 1e-12)

    def test_step_sponge(self):
        ac = _continuous()
        self._sponge(ac)
        dt = C.stable_dt(ac)
        for _ in range(20):
            ac.alat.step(0, dt)
            ac.alat.step_sponge(0)
        self.assertLess(_antisym_err(ac.phi, ac.nb), 1e-12)

    def test_step_ae_sponge(self):
        ac = _continuous()
        BC = _lat.continuous.BoundaryCondition
        dt = C.stable_dt(ac)
        ac.cl.set_bc(BC.quadratic_sponge(target=0.0, gamma_max=DAMP / dt, r_all=ac.r, r_inner=ac.r_inner))
        for _ in range(20):
            ac.step_ae(dt)
            ac.alat.step_sponge(0, dt)
        self.assertLess(_antisym_err(ac.phi, ac.nb), 1e-12)
        self.assertLess(_antisym_err(ac.e, ac.nb), 1e-12)


class TestContinuousSpongeDampsE(unittest.TestCase):
    def test_e_damped_with_phi(self):
        """The sponge scales E by the same per-face factor as phi, and actually damps it."""
        ac = _continuous()
        BC = _lat.continuous.BoundaryCondition
        ac.cl.set_bc(BC.quadratic_sponge(target=0.0, gamma_max=1.0, r_all=ac.r, r_inner=ac.r_inner))
        phi0, e0 = ac.phi, ac.e
        ac.alat.step_sponge(0, 0.1)
        m = (ac.nb >= 0) & (np.abs(phi0) > 1e-3) & (np.abs(e0) > 1e-3)
        np.testing.assert_allclose(ac.e[m] / e0[m], ac.phi[m] / phi0[m], rtol=1e-12)
        deep = ac.r > ac.r_inner + 0.8 * (ac.r.max() - ac.r_inner)
        self.assertLess(float(np.max(np.abs(ac.e[deep]) / np.maximum(np.abs(e0[deep]), 1e-300))), 1.0)


def _sponge_decay_numpy(dt_div):
    """Flat u = 1 relaxing to u_bg = 0 in a DampedLattice, to a fixed time."""
    lat = DampedLattice(radius_cells=R, sponge_fraction=SPONGE, damp_max=DAMP)
    lat.u[:] = 1.0
    dt = lat.stable_dt() / dt_div
    for _ in range(40 * dt_div):
        lat.step(dt)
    return lat.u, lat._sponge > 0


def _sponge_decay_cpp(dt_div, integrated):
    d = _Discrete()
    d.cl.set_leaf_u(np.ones(len(d.pos)))
    d.cl.set_leaf_phi(np.zeros((14, len(d.pos))))
    BC = _lat.discrete.BoundaryCondition
    dt = d.dt / dt_div
    if integrated:
        d.cl.set_bc(BC.quadratic_sponge_integrated(target=0.0, gamma_max=DAMP / d.dt, r_all=d.r, r_inner=d.r_inner))
    else:
        d.cl.set_bc(BC.quadratic_sponge(target=0.0, gamma_max=DAMP / d.dt, r_all=d.r, r_inner=d.r_inner))
    for _ in range(40 * dt_div):
        if integrated:
            d.alat.step_damped(0, dt)
        else:
            d.alat.step(0, dt)
            d.alat.step_sponge(0, dt)
    return d.u, d.r > d.r_inner


class TestSpongeDtConsistent(unittest.TestCase):
    """Same physical time, dt vs dt/2: the damping must agree (to O(dt))."""

    def _check(self, run):
        (u1, s), (u2, _) = run(1), run(2)
        loss1, loss2 = 1.0 - u1[s], 1.0 - u2[s]
        rel = float(np.max(np.abs(loss2 - loss1)) / np.max(np.abs(loss1)))
        self.assertLess(rel, 0.1, f"sponge loss changes by {rel:.2f} when dt halves")

    def test_numpy_damped_lattice(self):
        self._check(_sponge_decay_numpy)

    def test_cpp_quadratic_sponge(self):
        self._check(lambda k: _sponge_decay_cpp(k, integrated=False))

    def test_cpp_quadratic_sponge_integrated(self):
        self._check(lambda k: _sponge_decay_cpp(k, integrated=True))


class TestRateSpongeNeedsDt(unittest.TestCase):
    def test_step_sponge_without_dt_raises(self):
        d = _Discrete()
        BC = _lat.discrete.BoundaryCondition
        d.cl.set_bc(BC.quadratic_sponge(target=0.0, gamma_max=1.0, r_all=d.r, r_inner=d.r_inner))
        with self.assertRaises(ValueError):
            d.alat.step_sponge(0)


class TestSpongeTarget(unittest.TestCase):
    """A state already at the sponge target is not changed by the sponge."""

    def test_numpy_damped_lattice(self):
        lat = DampedLattice(radius_cells=R, sponge_fraction=SPONGE, damp_max=DAMP, u_bg=1.0)
        lat.u[:] = 1.0
        for _ in range(20):
            lat.step(lat.stable_dt())
        np.testing.assert_allclose(lat.u, 1.0, atol=1e-12)

    def test_cpp_quadratic_sponge(self):
        d = _Discrete()
        d.cl.set_leaf_u(np.ones(len(d.pos)))
        d.cl.set_leaf_phi(np.zeros((14, len(d.pos))))
        BC = _lat.discrete.BoundaryCondition
        d.cl.set_bc(BC.quadratic_sponge(target=1.0, damp_max=DAMP, r_all=d.r, r_inner=d.r_inner))
        for _ in range(20):
            d.alat.step(0, d.dt)
            d.alat.step_sponge(0)
        np.testing.assert_allclose(d.u, 1.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
