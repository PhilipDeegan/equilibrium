import unittest

import numpy as np

from equilibrium.substrate.ambient_bc import AmbientBC, proton_microfield, uniform_field
from equilibrium.substrate.charge_exterior import divergence
from equilibrium.substrate.lattice import _W, DampedLattice


def _lattice():
    return DampedLattice(radius_cells=8.0, sponge_fraction=0.25)


def _pulse(lat, amp=1.0, width=1.5):
    r = np.linalg.norm(lat.pos, axis=1)
    return amp * np.exp(-(r**2) / (2 * width**2))


class TestProtonMicrofield(unittest.TestCase):
    def test_interior_divergence_free_and_coulomb(self):
        lat = _lattice()
        charges = [((14.0, 0.0, 0.0), 1.0), ((-6.0, 11.0, 3.0), 0.5)]
        phi, psi = proton_microfield(lat.pos, lat.nb, _W, charges)
        inner = ~np.any(lat.nb < 0, axis=1)
        d = divergence(phi, lat.nb, _W)
        self.assertLess(np.abs(d[inner]).max(), 1e-10)
        exact = sum(Q / np.linalg.norm(lat.pos - np.array(rp), axis=1) for rp, Q in charges)
        core = np.linalg.norm(lat.pos, axis=1) < 5.0
        rel = np.abs((psi - exact)[core] - (psi - exact)[core].mean()).max() / np.ptp(exact[core])
        self.assertLess(rel, 1e-2)
        np.testing.assert_allclose(phi[lat.nb < 0], 0.0)

    def test_static_under_step_in_interior(self):
        lat = _lattice()
        phi, _ = proton_microfield(lat.pos, lat.nb, _W, [((14.0, 0.0, 0.0), 1.0)])
        lat.phi = phi.copy()
        dt = lat.stable_dt()
        for _ in range(50):
            lat.step_cpp(dt)
        r = np.linalg.norm(lat.pos, axis=1)
        core = r < 4.0
        self.assertLess(np.abs(lat.u[core]).max(), 1e-3 * np.abs(phi).max())


class TestAmbientBC(unittest.TestCase):
    def test_empty_domain_stays_ambient(self):
        lat = _lattice()
        bc = AmbientBC(lat, T=0.5, rng=np.random.default_rng(1))
        dt = lat.stable_dt()
        for _ in range(200):
            bc.step(dt)
        du, dphi = bc.departure()
        self.assertLess(max(np.abs(du).max(), np.abs(dphi).max()), 1e-12)

    def test_linear_superposition(self):
        """Linear step(): domain - bath evolves exactly as the departure under a zero-target sponge."""
        lat = _lattice()
        ref = _lattice()
        lat.u = _pulse(lat)
        ref.u = _pulse(ref)
        bc = AmbientBC(lat, T=0.5, rng=np.random.default_rng(2))
        dt = lat.stable_dt()
        for _ in range(300):
            bc.step(dt)
            ref.step(dt)
        du, dphi = bc.departure()
        self.assertLess(np.abs(du - ref.u).max(), 1e-10)
        self.assertLess(np.abs(np.where(lat.nb >= 0, dphi - ref.phi, 0.0)).max(), 1e-10)
        self.assertGreater(np.abs(ref.u).max(), 1e-3)

    def test_static_field_held_exactly(self):
        """Held proton microfield: external flux through the missing boundary faces is restored each step."""
        lat = _lattice()
        phi, _ = proton_microfield(lat.pos, lat.nb, _W, [((14.0, 0.0, 0.0), 1.0), ((0.0, -12.0, 5.0), -0.7)])
        bc = AmbientBC(lat, T=0.3, rng=np.random.default_rng(3), static_phi=phi)
        dt = lat.stable_dt()
        for _ in range(500):
            bc.step(dt)
        du, dphi = bc.departure()
        self.assertLess(max(np.abs(du).max(), np.abs(np.where(lat.nb >= 0, dphi, 0.0)).max()), 1e-12)


class TestUniformField(unittest.TestCase):
    def test_linear_potential_exact(self):
        lat = _lattice()
        E = np.array([0.3, -0.2, 0.5])
        phi, psi = uniform_field(lat.pos, lat.nb, _W, E)
        self.assertLess(np.abs(psi + lat.pos @ E).max(), 1e-12)
        interior = ~np.any(lat.nb < 0, axis=1)
        self.assertLess(np.abs(divergence(phi, lat.nb, _W)[interior]).max(), 1e-12)

    def test_held_exactly(self):
        lat = _lattice()
        phi, _ = uniform_field(lat.pos, lat.nb, _W, [0.0, 0.0, 1.0])
        bc = AmbientBC(lat, T=0.3, rng=np.random.default_rng(4), static_phi=phi)
        dt = lat.stable_dt()
        for _ in range(500):
            bc.step(dt)
        du, dphi = bc.departure()
        self.assertLess(max(np.abs(du).max(), np.abs(np.where(lat.nb >= 0, dphi, 0.0)).max()), 1e-12)


if __name__ == "__main__":
    unittest.main()
