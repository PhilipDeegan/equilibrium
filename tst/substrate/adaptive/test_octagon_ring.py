"""
Regression tests for find_toroidal_ring(mode="octagon") (adaptive_flux_tube
py), built 2026-09-08 after realising mode="square" is a literal 4-sided
square (permanently sharp-cornered at any hops/resolution -- confirmed in
`now`: the "diamond" near-field pattern in the Poynting-dipole cross-
section plots reproduces identically at multiple domain sizes AND multiple
ring sizes, ruling out both a domain-too-small artifact and a resolution
artifact, leaving the square's own genuine corner structure as the likely
cause). mode="octagon" chamfers each corner with a diagonal staircase leg,
giving a genuine 8-direction (not just 8-tile) loop -- a rounder
approximation to a circle, to test whether that removes the near-field
anisotropy.
"""

import unittest

import numpy as np

from equilibrium import lattice as _lat
from tst.substrate.adaptive.adaptive_flux_tube import (
    _octagon_face_seq,
    _square_face_seq,
    find_toroidal_ring,
    winding_number,
)

R_DOMAIN = 30.0


def _build():
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    cl = alat[0]
    _lat.build_bcc_sphere(cl, R_DOMAIN, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)
    return alat, cl


class TestOctagonFaceSeqClosure(unittest.TestCase):
    def test_algebraic_closure(self):
        """Net displacement of the face sequence is exactly zero for every
        hops value -- a necessary (not sufficient) condition for a valid
        closed loop, checked in idealised 2D direction-vector space."""
        d = {
            11: np.array([0, 1]),
            8: np.array([-1, 0]),
            10: np.array([0, -1]),
            9: np.array([1, 0]),
        }
        for hops in (1, 2, 3, 4, 12):
            seq = _octagon_face_seq("z", hops)
            total = sum(d[int(f)] for f in seq)
            self.assertEqual(tuple(total), (0, 0), f"hops={hops}")
            self.assertEqual(len(seq), 16 * hops)


class TestOctagonRingOnRealLattice(unittest.TestCase):
    def test_closes_and_has_winding_one(self):
        """The octagon actually closes on the real lattice graph (not just
        in idealised vector space) and winds exactly once around its axis,
        for several sizes."""
        alat, cl = _build()
        for hops in (1, 2, 3, 4):
            ring_js, ring_faces = find_toroidal_ring(cl, axis="z", mode="octagon", hops=hops)
            self.assertIsNotNone(ring_js, f"hops={hops}: no ring found")
            self.assertEqual(len(ring_js), 16 * hops)
            w = winding_number(cl, ring_js, axis="z")
            self.assertAlmostEqual(abs(w), 1.0, places=6, msg=f"hops={hops}")

    def test_ring_centers_at_origin(self):
        """find_toroidal_ring's search target is meant to land the loop's
        CENTROID at the origin, not just its start tile -- confirmed 2026-
        09-08 that mode="square" only did this by coincidence (its
        particular face_seq happens to start at its own centroid-symmetric
        point) while mode="octagon" was landing at 2x that offset (a real
        physical bug: an off-center ring in a domain built around the
        origin sits asymmetrically close to one side of the sponge, not
        just a plotting inconvenience). Both modes must center at the
        origin for every hops value, not just by luck at one size."""
        alat, cl = _build()
        pos = np.array(cl.leaf_pos)
        for mode in ("square", "octagon"):
            for hops in (1, 2, 4, 8):
                ring_js, _ = find_toroidal_ring(cl, axis="z", mode=mode, hops=hops)
                self.assertIsNotNone(ring_js, f"mode={mode} hops={hops}: no ring found")
                centre = pos[ring_js].mean(axis=0)
                self.assertLess(
                    float(np.linalg.norm(centre)),
                    1e-9,
                    f"mode={mode} hops={hops}: ring centroid {centre} not at origin",
                )

    def test_rounder_than_square_at_matched_hops(self):
        """The octagon's radial spread (std/mean over its own tiles' distance
        from centroid) should be meaningfully smaller than the square's at
        the same hops -- a direct, real-lattice-measured roundness check,
        not just a visual impression."""
        alat, cl = _build()
        hops = 4
        pos = np.array(cl.leaf_pos)

        sq_js, _ = find_toroidal_ring(cl, axis="z", mode="square", hops=hops)
        oct_js, _ = find_toroidal_ring(cl, axis="z", mode="octagon", hops=hops)
        self.assertIsNotNone(sq_js)
        self.assertIsNotNone(oct_js)

        def rel_std(js):
            p = pos[js]
            c = p.mean(axis=0)
            r = np.linalg.norm(p - c, axis=1)
            return float(r.std() / r.mean())

        sq_rel_std = rel_std(sq_js)
        oct_rel_std = rel_std(oct_js)
        self.assertLess(oct_rel_std, 0.6 * sq_rel_std)


if __name__ == "__main__":
    unittest.main()
