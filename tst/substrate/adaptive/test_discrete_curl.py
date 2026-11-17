"""Validation of the coordinate-free discrete curl/circulation primitive on
the Kelvin/BCC lattice (equilibrium.substrate.discrete_curl) -- 2026-08-16
investigation into why DST's dipole-mode spheromak decay shows only
expansion+diffusion (never PHARE-style lobe-merging into a flux tube, see
`~/tmp/EM_B.vtkhdf` comparison and `run/phare_plasmoses.py`'s genuinely
curl-full Chandrasekhar-Kendall/Taylor-state IC).

Three things validated here, cheaply (CELLS_ACROSS=22, matching
test_step_cross_per_level.py's fast-test convention):

1. The 12-elementary-triangle geometric identity (two hex hops == one direct
   sq hop, to the SAME cell) holds on real leaf/nb data in the domain
   interior -- this is what makes the "loop" coordinate-free and well-defined
   at all (no continuum embedding needed, just the fixed face-index table).
2. A discrete-derivative field (phi[k,i] = u[nb[i,k]]-u[i], for ANY scalar
   u) has EXACTLY zero circulation -- pure algebraic telescoping, not an
   approximation. This is the discrete analogue of "curl-free".
3. By contrast, `seed_dipole_ic`'s phi -- built by SAMPLING the continuum
   gradient grad(U) at face midpoints, not by discrete-differencing u --
   has a real nonzero circulation (a finite-hop curvature artifact of U's
   own second derivative, not evidence of rotational/"magnetic" content).
   This is the finding that motivated the whole investigation: the
   production dipole IC's phi is curl-free in the CONTINUUM sense but that
   is a different condition from discrete-exactness on this lattice, and
   only the latter is the meaningful zero for lattice dynamics.
"""

import unittest

import numpy as np

from equilibrium import lattice as _lat
from equilibrium.substrate.dipole_mode_ic import seed_dipole_ic
from equilibrium.substrate.discrete_curl import (
    TRIANGLES,
    leaf_nb_rows,
    triangle_triples,
    circulation,
    circulation_from_scalar,
)
from tst.substrate.adaptive.adaptive_flux_tube import _FACE_HOP, _FACE_NORM

CELLS_ACROSS = 22
R_SPH = 1.0
U_MAX, AMPLITUDE = 1.0, 1.0
R_DOMAIN = 18.0  # matches test_step_cross_per_level.py's scale


def _build():
    scale = CELLS_ACROSS / 2.0
    alat = _lat.AdaptiveLattice()
    cl = alat[0]
    _lat.build_bcc_sphere(cl, R_DOMAIN, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)

    pos = np.array(cl.leaf_pos)
    cell_size = np.full(cl.n_leaves, 1.0)
    seed_dipole_ic(
        cl,
        pos,
        cell_size,
        scale,
        _FACE_HOP,
        _FACE_NORM,
        r_sph=R_SPH,
        u_max=U_MAX,
        amplitude=AMPLITUDE,
    )
    return alat, cl


class TestDiscreteCurl(unittest.TestCase):
    def test_twelve_triangle_families(self):
        # Sanity on the table itself: 6 sq-directions x 2 hex-pairs each.
        self.assertEqual(len(TRIANGLES), 12)
        by_sqk = {}
        for h1, h2, sqk in TRIANGLES:
            by_sqk.setdefault(sqk, []).append((h1, h2))
        self.assertEqual(sorted(by_sqk.keys()), list(range(8, 14)))
        for sqk, pairs in by_sqk.items():
            self.assertEqual(
                len(pairs), 2, f"sq face {sqk} should have exactly 2 hex-pairs"
            )

    def test_triangle_identity_holds_in_interior(self):
        _, cl = _build()
        nb = leaf_nb_rows(cl)
        pos = np.array(cl.leaf_pos)
        r = np.linalg.norm(pos, axis=1)
        # Exclude the outer few cells so the geometric identity isn't tested
        # against real domain-boundary cells (expected to fail there -- no
        # neighbour to close the loop, not a bug).
        idx = np.nonzero(r < R_DOMAIN - 3.0)[0]
        self.assertGreater(idx.size, 0)

        triples = triangle_triples(nb, idx)
        total_candidates = 0
        total_matched = 0
        for (h1, h2, sqk), a, b, c in triples:
            # Recompute the candidate count (before the identity filter) to
            # get a real match-rate denominator: re-walk h1 then h2 from idx.
            bstep = nb[idx, h1]
            vb = bstep >= 0
            cstep = nb[bstep[vb], h2]
            vc = cstep >= 0
            total_candidates += int(vc.sum())
            total_matched += a.size
        self.assertGreater(total_candidates, 0)
        match_rate = total_matched / total_candidates
        self.assertGreater(
            match_rate,
            0.999,
            f"two-hex-hop == direct-sq-hop identity should hold on ~all interior leaves, got {match_rate:.4%}",
        )

    def test_discrete_derivative_has_exactly_zero_circulation(self):
        _, cl = _build()
        nb = leaf_nb_rows(cl)
        pos = np.array(cl.leaf_pos)
        r = np.linalg.norm(pos, axis=1)
        idx = np.nonzero(r < R_DOMAIN - 3.0)[0]

        u = np.array(cl.leaf_u)  # the production dipole u field -- any scalar works
        circ = circulation_from_scalar(u, nb, idx)
        self.assertGreater(circ.size, 0)
        rms = float(np.sqrt(np.mean(circ**2)))
        self.assertLess(
            rms,
            1e-10,
            "circulation of a genuine discrete derivative must vanish by pure "
            "telescoping, at any resolution -- not an approximation",
        )

    def test_continuum_sampled_phi_has_nonzero_circulation(self):
        # Documents the distinction found 2026-08-16: seed_dipole_ic's phi
        # is grad(U) SAMPLED at face midpoints, not a discrete derivative of
        # u -- it satisfies continuum Stokes (~0 in the infinitesimal-loop
        # limit) but NOT discrete-exactness, so its measured circulation is
        # a real, nonzero, resolution-dependent number (finite-hop curvature
        # of U), not a bug. This test pins that number's order of magnitude
        # so a future change to seed_dipole_ic (or the face geometry) that
        # silently makes phi discrete-exact (circulation -> 0) or wildly
        # different is caught, not silently assumed unchanged.
        _, cl = _build()
        nb = leaf_nb_rows(cl)
        pos = np.array(cl.leaf_pos)
        r = np.linalg.norm(pos, axis=1)
        idx = np.nonzero(r < R_DOMAIN - 3.0)[0]

        phi = np.array(cl.leaf_phi)
        circ = circulation(phi, nb, idx)
        phi_rms = float(np.sqrt(np.mean(phi**2)))
        circ_rms = float(np.sqrt(np.mean(circ**2)))
        ratio = circ_rms / phi_rms
        self.assertGreater(
            ratio,
            1e-3,
            "continuum-sampled phi is expected to have a real (not float-noise) "
            "circulation -- if this drops near float noise, seed_dipole_ic may "
            "have changed to a genuine discrete derivative, which would be worth "
            "noting explicitly, not silently passing",
        )
        self.assertLess(ratio, 1.0, "circulation should not exceed phi's own scale")


if __name__ == "__main__":
    unittest.main()
