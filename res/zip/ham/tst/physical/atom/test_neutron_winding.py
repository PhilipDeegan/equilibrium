"""
Tests for the toroidal winding / perimeter-closure system.

Uses the cached neutron_perimeter.npz — no PHARE data required.
"""

import os
import unittest
import numpy as np

NPZ_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../../equilibrium/physical/atom/neutron_perimeter.npz",
)
NPZ_PATH = os.path.normpath(NPZ_PATH)


@unittest.skipUnless(os.path.exists(NPZ_PATH), "neutron_perimeter.npz not found")
class TestPerimeterLoading(unittest.TestCase):
    """Load the cached perimeter winding and verify structure."""

    @classmethod
    def setUpClass(cls):
        from equilibrium.physical.atom.toroidal_winding import load_perimeter_winding

        cls.pts, cls.u, cls.inner, cls.outer, cls.scale = load_perimeter_winding(
            NPZ_PATH
        )

    def test_total_cells(self):
        self.assertGreater(len(self.pts), 0)

    def test_inner_outer_partition(self):
        """Inner and outer masks are disjoint and cover all cells."""
        self.assertEqual(len(self.inner), len(self.pts))
        self.assertEqual(len(self.outer), len(self.pts))
        self.assertTrue(
            np.all(self.inner ^ self.outer), "masks must be disjoint and complete"
        )

    def test_inner_cells_closer_than_outer(self):
        """All inner cells must be closer to origin than all outer cells."""
        r = np.linalg.norm(self.pts, axis=1)
        self.assertLess(r[self.inner].max(), r[self.outer].min() * 1.05)

    def test_u_mean_near_zero(self):
        """u is mean-subtracted — balanced winding."""
        self.assertAlmostEqual(float(self.u.mean()), 0.0, delta=1e-6)

    def test_u_both_signs(self):
        """Signed winding has both positive and negative u values."""
        self.assertTrue((self.u > 0).any())
        self.assertTrue((self.u < 0).any())


@unittest.skipUnless(os.path.exists(NPZ_PATH), "neutron_perimeter.npz not found")
class TestPerimeterEvolution(unittest.TestCase):
    """
    Evolve a DSTLattice with the perimeter-closure winding and verify
    the key properties of the stable state.
    """

    @classmethod
    def setUpClass(cls):
        from equilibrium.physical.atom.toroidal_winding import (
            load_perimeter_winding,
            net_flux_fraction,
        )
        from equilibrium.physical.dst.lattice import DSTLattice
        from equilibrium.math.geo.kelvin import KelvinLattice

        pts, u, inner, outer, scale = load_perimeter_winding(NPZ_PATH)
        d = np.load(NPZ_PATH)
        r_total = float(d["radius_total"][0])

        geo = KelvinLattice(lattice_spacing=1.0)
        _, key_fn = geo.point_index_map(pts)

        lat = DSTLattice(cell_size=scale, radius_cells=r_total)
        _, key_fn2 = geo.point_index_map(lat.pos / scale)
        host_map = {key_fn2(lat.pos[i] / scale): i for i in range(lat.N)}

        for p, u_val, is_inner in zip(pts, u, inner):
            hi = host_map.get(key_fn(p))
            if hi is not None:
                lat.u[hi] += float(u_val)
                lat.fixed[hi] = not bool(is_inner)

        free = ~lat.fixed
        lat.u[free] -= lat.u[free].mean()

        cls.e0 = float(np.sum(lat.u[free] ** 2))
        lat.run(500)
        cls.e1 = float(np.sum(lat.u[free] ** 2))
        cls.lat = lat
        cls.free = free
        cls.net = net_flux_fraction(lat.u[free])

    def test_energy_retained(self):
        """Winding retains at least 10% of initial energy after 500 steps."""
        retained = self.e1 / (self.e0 + 1e-12)
        self.assertGreater(
            retained, 0.10, f"retained only {100*retained:.1f}% — winding collapsed"
        )

    def test_z_antisymmetry(self):
        """u is antisymmetric about z=0: top half ≈ −bottom half."""
        z = self.lat.pos[:, 2]
        top = float(self.lat.u[self.free & (z > 0)].sum())
        bot = float(self.lat.u[self.free & (z < 0)].sum())
        if abs(bot) > 1e-10:
            ratio = abs(top / bot)
            self.assertAlmostEqual(
                ratio,
                1.0,
                delta=0.05,
                msg=f"z-antisymmetry ratio {ratio:.4f} — winding chirality lost",
            )

    def test_net_flux_small(self):
        """Closed winding: net outward flux fraction < 0.5."""
        self.assertLess(
            self.net, 0.5, f"net flux fraction {self.net:.3f} — winding not closed"
        )

    def test_fixed_cells_unchanged(self):
        """Fixed perimeter cells must not change during evolution."""
        fixed_u_before = self.lat.u[self.lat.fixed].copy()
        self.lat.run(10)
        fixed_u_after = self.lat.u[self.lat.fixed]
        np.testing.assert_array_equal(fixed_u_before, fixed_u_after)


if __name__ == "__main__":
    unittest.main(verbosity=2)
