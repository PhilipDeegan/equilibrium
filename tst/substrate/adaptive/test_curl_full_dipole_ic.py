"""Validation of equilibrium.substrate.curl_full_dipole_ic -- the
volume-filling, two-lobed, genuinely curl-full spheromak IC that was still
missing as of `now`'s 2026-08-16 entry (the wire-loop Hopf-pair diagnostic
in curl_full_spheromak_ic.py validates the step_cross-erosion mechanism on a
bare 1-cell-wide filament, u=0 everywhere -- explicitly not this object).

Same fixture/style as test_discrete_curl.py and
test_ring_circulation_conservation.py.

**A real finding from building this test, not assumed**: the per-elementary-
-triangle circulation/phi-RMS ratio (equilibrium.substrate.discrete_curl,
the metric that flagged phi_pol's ~0.2-0.3 curvature artifact in
test_discrete_curl.py) does NOT distinguish phi_tor from phi_pol -- measured
directly, phi_tor's ratio (~0.32) is essentially the SAME order as phi_pol's
(~0.33) on this fixture. That metric measures a small, LOCAL 3-cell-loop
sampling artifact common to any face-midpoint-sampled continuum field
(both phi_pol and phi_tor are built this way); it is NOT a measure of real,
persistent, macroscopic circulation. The metric that actually distinguishes
them is the pre-existing MACRO ring circulation
(adaptive_flux_tube.phi_ring_circulation, the same tool
test_ring_circulation_conservation.py used on the wire-loop): phi_pol
(axisymmetric, no azimuthal component anywhere -- U has no phi-dependence)
gives EXACTLY 0.0 net circulation around any full ring encircling the axis;
phi_tor gives a large, real, nonzero net circulation. That is the correct,
non-misleading test for "genuinely curl-full", and is what this file checks.

Five things checked:

1. u matches dipole_mode_ic's own poloidal profile exactly (unchanged half).
2. phi_tor is genuinely volume-filling (nonzero over most of the interior,
   not a thin filament).
3. phi_pol's macro ring circulation is exactly zero while phi_tor's is
   large and real -- the actual "genuinely curl-full, not just a
   continuum-sampling artifact" evidence (see note above; NOT the
   per-triangle ratio).
4. phi_tor's macro ring circulation scales linearly with its own amplitude.
5. phi_tor is symmetric across the equator (single toroidal band threading
   both lobes) while u is antisymmetric (the two-lobed dipole) -- confirms
   the two halves are geometrically distinct, not accidentally the same
   shape.
"""

import unittest

import numpy as np

from equilibrium import lattice as _lat
from equilibrium.substrate.dipole_mode_ic import seed_dipole_ic, U_and_gradU
from equilibrium.substrate.curl_full_dipole_ic import (
    seed_curl_full_dipole_ic,
    toroidal_envelope,
)
from equilibrium.substrate.discrete_curl import leaf_nb_rows, circulation
from tst.substrate.adaptive.adaptive_flux_tube import (
    _FACE_HOP,
    _FACE_NORM,
    find_toroidal_ring,
    phi_ring_circulation,
)

CELLS_ACROSS = 22
R_SPH = 1.0
U_MAX = 1.0
AMPLITUDE_POL, AMPLITUDE_TOR = 1.0, 1.0
R_DOMAIN = 18.0
SCALE = CELLS_ACROSS / 2.0


def _build():
    alat = _lat.AdaptiveLattice()
    cl = alat[0]
    _lat.build_bcc_sphere(cl, R_DOMAIN, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)
    return alat, cl


def _interior_idx(cl):
    pos = np.array(cl.leaf_pos)
    r = np.linalg.norm(pos, axis=1)
    return np.nonzero(r < R_DOMAIN - 3.0)[0]


class TestCurlFullDipoleIC(unittest.TestCase):
    def test_u_matches_poloidal_dipole_shape(self):
        _, cl_ref = _build()
        pos = np.array(cl_ref.leaf_pos)
        cell_size = np.full(cl_ref.n_leaves, 1.0)
        u_ref, _ = seed_dipole_ic(
            cl_ref,
            pos,
            cell_size,
            SCALE,
            _FACE_HOP,
            _FACE_NORM,
            r_sph=R_SPH,
            u_max=U_MAX,
            amplitude=AMPLITUDE_POL,
        )

        _, cl = _build()
        u_all, phi_pol, phi_tor, phi_full = seed_curl_full_dipole_ic(
            cl,
            pos,
            cell_size,
            SCALE,
            _FACE_HOP,
            _FACE_NORM,
            r_sph=R_SPH,
            u_max=U_MAX,
            amplitude_pol=AMPLITUDE_POL,
            amplitude_tor=AMPLITUDE_TOR,
        )
        np.testing.assert_allclose(u_all, u_ref)
        self.assertGreater(np.count_nonzero(u_all), 0)

    def test_phi_tor_is_volume_filling_not_a_filament(self):
        _, cl = _build()
        pos = np.array(cl.leaf_pos)
        cell_size = np.full(cl.n_leaves, 1.0)
        _, _, phi_tor, _ = seed_curl_full_dipole_ic(
            cl,
            pos,
            cell_size,
            SCALE,
            _FACE_HOP,
            _FACE_NORM,
            r_sph=R_SPH,
            amplitude_pol=AMPLITUDE_POL,
            amplitude_tor=AMPLITUDE_TOR,
        )
        r_cell = np.linalg.norm(pos / SCALE, axis=1)
        inside = r_cell <= R_SPH
        self.assertGreater(
            int(inside.sum()), 100, "sanity: sphere should contain many leaves"
        )

        nonzero_faces_per_leaf = np.count_nonzero(np.abs(phi_tor) > 1e-15, axis=0)
        frac_leaves_touched = float(
            np.count_nonzero(nonzero_faces_per_leaf[inside]) / inside.sum()
        )
        self.assertGreater(
            frac_leaves_touched,
            0.5,
            "phi_tor should be nonzero across most of the interior volume -- "
            "a thin filament (like the wire-loop diagnostic) would touch only "
            "a tiny fraction of leaves",
        )

    def test_phi_pol_macro_ring_circulation_is_zero_phi_tor_is_not(self):
        # The actual "genuinely curl-full" evidence -- see module docstring
        # for why the per-triangle circulation/RMS ratio does NOT
        # distinguish phi_tor from phi_pol (both are continuum-sampled and
        # pick up similar local curvature noise); net circulation around a
        # full macroscopic ring is what actually discriminates them.
        _, cl_pol = _build()
        pos = np.array(cl_pol.leaf_pos)
        cell_size = np.full(cl_pol.n_leaves, 1.0)
        seed_curl_full_dipole_ic(
            cl_pol,
            pos,
            cell_size,
            SCALE,
            _FACE_HOP,
            _FACE_NORM,
            r_sph=R_SPH,
            amplitude_pol=AMPLITUDE_POL,
            amplitude_tor=0.0,
        )
        ring_js, ring_faces = find_toroidal_ring(
            cl_pol, axis="z", mode="square", hops=3
        )
        self.assertIsNotNone(
            ring_js, "ring-finding failed -- check R_DOMAIN/hops fit the domain"
        )
        centre = pos[ring_js].mean(axis=0)
        circ_pol = phi_ring_circulation(cl_pol, ring_js, centre)
        self.assertAlmostEqual(
            circ_pol,
            0.0,
            delta=1e-9,
            msg="phi_pol (axisymmetric grad(U), no phi-dependence) must have "
            "exactly zero net circulation around any ring encircling the "
            "axis -- it has no azimuthal component to begin with",
        )

        _, cl_tor = _build()
        seed_curl_full_dipole_ic(
            cl_tor,
            pos,
            cell_size,
            SCALE,
            _FACE_HOP,
            _FACE_NORM,
            r_sph=R_SPH,
            amplitude_pol=0.0,
            amplitude_tor=AMPLITUDE_TOR,
        )
        circ_tor = phi_ring_circulation(cl_tor, ring_js, centre)
        self.assertGreater(
            abs(circ_tor),
            1.0,
            "phi_tor should carry real, O(1)+ net circulation around the same ring",
        )

    def test_phi_tor_has_nonzero_per_triangle_circulation(self):
        # Weaker, secondary check: phi_tor is not a pure scalar discrete
        # derivative either (same qualitative status as phi_pol -- both are
        # continuum-sampled fields, see module docstring). NOT a
        # distinguishing test on its own.
        _, cl = _build()
        pos = np.array(cl.leaf_pos)
        cell_size = np.full(cl.n_leaves, 1.0)
        _, _, phi_tor, _ = seed_curl_full_dipole_ic(
            cl,
            pos,
            cell_size,
            SCALE,
            _FACE_HOP,
            _FACE_NORM,
            r_sph=R_SPH,
            amplitude_pol=AMPLITUDE_POL,
            amplitude_tor=AMPLITUDE_TOR,
        )
        nb = leaf_nb_rows(cl)
        idx = _interior_idx(cl)

        circ = circulation(phi_tor, nb, idx)
        self.assertGreater(circ.size, 0)
        phi_rms = float(np.sqrt(np.mean(phi_tor**2)))
        circ_rms = float(np.sqrt(np.mean(circ**2)))
        ratio = circ_rms / phi_rms
        self.assertGreater(ratio, 1e-3)
        self.assertLess(ratio, 1.0)

    def test_phi_tor_circulation_scales_linearly_with_amplitude(self):
        _, cl1 = _build()
        pos = np.array(cl1.leaf_pos)
        cell_size = np.full(cl1.n_leaves, 1.0)
        seed_curl_full_dipole_ic(
            cl1,
            pos,
            cell_size,
            SCALE,
            _FACE_HOP,
            _FACE_NORM,
            r_sph=R_SPH,
            amplitude_pol=0.0,
            amplitude_tor=1.0,
        )
        ring_js, ring_faces = find_toroidal_ring(cl1, axis="z", mode="square", hops=3)
        self.assertIsNotNone(
            ring_js, "ring-finding failed -- check R_DOMAIN/hops fit the domain"
        )
        centre = pos[ring_js].mean(axis=0)
        circ1 = phi_ring_circulation(cl1, ring_js, centre)

        _, cl2 = _build()
        seed_curl_full_dipole_ic(
            cl2,
            pos,
            cell_size,
            SCALE,
            _FACE_HOP,
            _FACE_NORM,
            r_sph=R_SPH,
            amplitude_pol=0.0,
            amplitude_tor=3.0,
        )
        circ2 = phi_ring_circulation(cl2, ring_js, centre)

        self.assertGreater(
            abs(circ1), 1e-6, "toroidal-only ring should carry real circulation"
        )
        self.assertAlmostEqual(circ2 / circ1, 3.0, delta=0.05 * 3.0)

    def test_phi_tor_symmetric_u_antisymmetric_across_equator(self):
        # Direct analytic check (not lattice ring-finding -- axis="z" rings
        # sit at z=0, the equatorial plane itself, where U=z*g(r) is
        # identically zero by construction; that would test nothing about
        # the north/south antisymmetry). Two synthetic mirrored points at
        # the same (rho, r), opposite z.
        rho0, z0 = 0.4 * SCALE, 0.4 * SCALE
        north = np.array([[rho0, 0.0, z0]])
        south = np.array([[rho0, 0.0, -z0]])

        U_north, _ = U_and_gradU(north, SCALE)
        U_south, _ = U_and_gradU(south, SCALE)
        self.assertGreater(abs(float(U_north[0])), 1e-6)
        self.assertLess(
            float(U_north[0]) * float(U_south[0]),
            0,
            "u (poloidal, U=z*g(r)) must flip sign across the equator",
        )

        A_north = toroidal_envelope(north, SCALE)
        A_south = toroidal_envelope(south, SCALE)
        self.assertGreater(abs(float(A_north[0])), 1e-6)
        self.assertGreater(
            float(A_north[0]) * float(A_south[0]),
            0,
            "phi_tor's envelope j1(Kr)*sin(theta) must NOT flip sign across "
            "the equator -- sin(theta) is the same both hemispheres, so it "
            "is one toroidal band threading both poloidal lobes, not a "
            "second dipole",
        )
        self.assertAlmostEqual(float(A_north[0]), float(A_south[0]), places=10)


if __name__ == "__main__":
    unittest.main()
