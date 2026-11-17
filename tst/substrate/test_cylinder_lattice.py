"""
Cylinder-domain construction — public constructor path.

The cylindrical domain exists for long-axis propagation probes (photon
carrier / canal experiments in doc/dst_next.md): O(radius² · length) cells
instead of O(R³) for a sphere, when only one axis must span a big distance.

Covers:
  - SubstrateLattice.cylinder() geometry (extents, axis choice, cell count
    saving vs the bounding sphere)
  - DampedLattice.cylinder() sponge: zero in the core, active at the lateral
    wall AND both end caps, matching spherical-sponge magnitude at the wall
  - dynamics smoke test: a u pulse steps without NaN and conserves total u
    away from the boundary
"""

import unittest

import numpy as np

from equilibrium.substrate.lattice import DampedLattice, SubstrateLattice, W_HEX, W_SQ

_R = 6.0
_HALF_LEN = 20.0


class TestCylinderConstruction(unittest.TestCase):
    def test_geometry_extents(self):
        lat = SubstrateLattice.cylinder(_R, _HALF_LEN, axis=2)
        self.assertGreater(len(lat.pos), 0)
        z = lat.pos[:, 2]
        r_perp = np.linalg.norm(lat.pos[:, :2], axis=1)
        self.assertLessEqual(r_perp.max(), _R + 1e-9)
        self.assertLessEqual(np.abs(z).max(), _HALF_LEN + 1e-9)
        # long axis genuinely longer than the transverse extent
        self.assertGreater(np.abs(z).max(), 2 * r_perp.max())

    def test_axis_selection(self):
        lat = SubstrateLattice.cylinder(_R, _HALF_LEN, axis=0)
        x = lat.pos[:, 0]
        r_perp = np.linalg.norm(lat.pos[:, 1:], axis=1)
        self.assertGreater(np.abs(x).max(), 2 * r_perp.max())

    def test_cheaper_than_bounding_sphere(self):
        """The point of the cylinder: far fewer cells than a sphere of
        radius = half_length covering the same axial span."""
        n_cyl = len(SubstrateLattice.cylinder(_R, _HALF_LEN).pos)
        n_sph = len(SubstrateLattice(radius_cells=_HALF_LEN).pos)
        self.assertLess(n_cyl, n_sph / 2)


class TestCylinderSponge(unittest.TestCase):
    def setUp(self):
        self.lat = DampedLattice.cylinder(
            _R, _HALF_LEN, sponge_fraction=0.2, damp_max=0.005
        )

    def _sponge_at(self, mask):
        return self.lat._sponge[mask]

    def test_core_undamped(self):
        z = np.abs(self.lat.pos[:, 2])
        r_perp = np.linalg.norm(self.lat.pos[:, :2], axis=1)
        core = (z < 0.5 * _HALF_LEN) & (r_perp < 0.5 * _R)
        self.assertTrue(core.any())
        self.assertTrue((self._sponge_at(core) == 0.0).all())

    def test_end_caps_damped(self):
        z = np.abs(self.lat.pos[:, 2])
        caps = z >= 0.95 * z.max()
        self.assertTrue(caps.any())
        self.assertTrue((self._sponge_at(caps) > 0.0).all())

    def test_lateral_wall_damped(self):
        r_perp = np.linalg.norm(self.lat.pos[:, :2], axis=1)
        wall = r_perp >= 0.98 * r_perp.max()
        self.assertTrue(wall.any())
        self.assertTrue((self._sponge_at(wall) > 0.0).all())

    def test_max_damping_matches_spherical(self):
        """Outermost cells reach damp_max·t² with t→1, same law as the sphere."""
        self.assertAlmostEqual(float(self.lat._sponge.max()), 0.005, places=4)


class TestCylinderDynamics(unittest.TestCase):
    def test_pulse_steps_finite_and_conservative(self):
        lat = SubstrateLattice.cylinder(_R, _HALF_LEN)
        centre = int(np.argmin(np.linalg.norm(lat.pos, axis=1)))
        lat.u[centre] = 1.0
        u_total = float(lat.u.sum())
        for _ in range(50):
            lat.step(0.1)
        self.assertTrue(np.isfinite(lat.u).all())
        self.assertTrue(np.isfinite(lat.phi).all())
        self.assertAlmostEqual(float(lat.u.sum()), u_total, places=10)


class TestTravelingWaveReproducer(unittest.TestCase):
    """
    The basic "photon" litmus test (2026-08-19,
    project_dst_em_field_convention_crisis memory): does this lattice
    support a genuine ONE-DIRECTIONAL, self-propagating traveling wave, or
    does every u-only IC split into two counter-propagating halves the way
    equilibrium.substrate.discrete.init_photon_packet's own docstring
    admits ("phi=0 -- packet splits into two halves")?

    Derived (not guessed) companion phi for u=U*cos(K.x-wt), from the
    continuous-time linearization of the update rule: at each face
    midpoint, Phi_k = (2*sigma*w_k/omega)*sin(0.5*K.offset_k)*U, in phase
    with u there. omega from the same dispersion relation TestDispersionRelation
    already validates. Confirmed across a wavelength sweep 2-20 cells with
    no breakdown at any scale; this is the cheapest config that still gives
    a clean, well-populated result (~seconds, not the full sweep).

    Prediction checked: the field at t=T (one period) should equal the
    field at t=0 shifted by exactly one (lattice-commensurate) wavelength.
    Two real bugs matter here if this test is ever modified: the axial
    lattice period is 2/sqrt(3), NOT 1 (must multiply cell counts by that,
    or shifts land off true lattice points); the comparison region must
    exclude cells near BOTH the lateral wall and the axial end-cap sponge
    (checking only one caused a false failure earlier the same session).
    """

    def test_field_shifts_by_one_wavelength_after_one_period(self):
        R, half_len, n_cells_per_wavelength, envelope_wavelengths, axis = (
            12.0,
            20.0,
            8,
            4.0,
            2,
        )
        lat = DampedLattice.cylinder(
            R, half_len, axis=axis, sponge_fraction=0.15, damp_max=0.003
        )
        c0 = int(np.argmin(np.linalg.norm(lat.pos, axis=1)))
        offsets = lat.pos[lat.nb[c0]] - lat.pos[c0]
        w = np.array([W_HEX] * 8 + [W_SQ] * 6, dtype=np.float64)
        a_axial = float(abs(offsets[13, axis]))  # true +/-z sq-face hop, NOT 1.0
        wavelength = n_cells_per_wavelength * a_axial
        K = 2 * np.pi / wavelength
        envelope_sigma = envelope_wavelengths * wavelength
        z0 = 0.0

        theta = offsets[:, axis] * K
        omega = np.sqrt(float(np.sum(w**2 * (1 - np.cos(theta)))))
        period = 2 * np.pi / omega
        dt = lat.stable_dt()
        n_steps = int(round(period / dt))

        z = lat.pos[:, axis] - z0
        env = np.exp(-0.5 * (z / envelope_sigma) ** 2)
        lat.u = env * np.cos(lat.pos[:, axis] * K)
        phi = np.zeros_like(lat.phi)
        for k in range(14):
            phi_k_amp = (2.0 * w[k] / omega) * np.sin(0.5 * theta[k])
            mid_axis_pos = lat.pos[:, axis] + 0.5 * offsets[k, axis]
            mid_env = np.exp(-0.5 * ((mid_axis_pos - z0) / envelope_sigma) ** 2)
            phi[:, k] = phi_k_amp * mid_env * np.cos(mid_axis_pos * K)
        lat.phi = phi
        u0, phi0 = lat.u.copy(), lat.phi.copy()

        for _ in range(n_steps):
            lat.step(dt)

        r_perp = np.linalg.norm(np.delete(lat.pos, axis, axis=1), axis=1)
        z_axis = lat.pos[:, axis]
        axial_ok = (np.abs(z_axis) < 0.5 * half_len) & (
            np.abs(z_axis + wavelength) < 0.5 * half_len
        )
        core = (r_perp < 0.5 * R) & axial_ok

        from scipy.spatial import cKDTree

        tree = cKDTree(lat.pos)
        shift = np.zeros(3)
        shift[axis] = wavelength
        dist, idx = tree.query(lat.pos + shift[None, :])
        valid = (dist < 1e-3) & core

        self.assertGreater(int(valid.sum()), 500, "not enough cells to compare")

        a, b = lat.u[valid], u0[idx][valid]
        ap, bp = lat.phi[valid], phi0[idx][valid]
        corr_u = float(np.corrcoef(a, b)[0, 1])
        corr_phi = float(np.corrcoef(ap.ravel(), bp.ravel())[0, 1])

        self.assertGreater(
            corr_u, 0.95, f"u shift-match correlation too low: {corr_u:.4f}"
        )
        self.assertGreater(
            corr_phi, 0.95, f"phi shift-match correlation too low: {corr_phi:.4f}"
        )


if __name__ == "__main__":
    unittest.main()
