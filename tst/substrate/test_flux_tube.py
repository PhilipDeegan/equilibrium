"""
Tests for flux-loop utilities.

Option 1 (fixed plaquette) — TestFluxTube:
  - connected 4-cycle through sq faces (8–13)
  - closed, divergence-free, antisymmetric
  - faces frozen; clear_flux_loop restores state
  - phi unchanged under dynamics while frozen

Option 2 (free resonant loop) — TestResonantLoop:
  - u and phi set to K=2π/N traveling-wave mode
  - no faces frozen
  - first-step Δu matches analytic wave-equation prediction exactly
  - u at the initially-maximum tile oscillates through zero
"""

import unittest

import numpy as np

from equilibrium.substrate.flux_tube import (
    _MIRROR,
    TOROIDAL_RING_SIZE,
    clear_flux_loop,
    find_linked_loop,
    find_shortest_loop,
    find_toroidal_ring,
    find_toroidal_spheromak,
    init_resonant_loop,
    init_spheromak,
    init_toroidal_winding,
    linking_number,
    resonant_freq,
    set_flux_loop,
    winding_number,
)
from equilibrium.substrate.lattice import W_SQ, SubstrateLattice

_W = np.array([3] * 8 + [1] * 6, dtype=np.float64)


def _interior_start(lat) -> int:
    """Tile nearest the origin.  find_shortest_loop needs an interior tile
    with all 14 neighbours; generate_sphere's point ordering puts tile 0 on
    the domain boundary (7 neighbours), where no 4-plaquette closes."""
    return int(np.argmin(np.linalg.norm(lat.pos, axis=1)))


class TestFluxTube(unittest.TestCase):
    def setUp(self):
        self.lat = SubstrateLattice(radius_cells=5.0)
        self.tiles, self.faces = find_shortest_loop(
            self.lat, start=_interior_start(self.lat)
        )

    # ------------------------------------------------------------------
    # Structural validity
    # ------------------------------------------------------------------

    def test_loop_found(self):
        self.assertIsNotNone(self.tiles, "No sq-face loop found at tile 0")
        self.assertEqual(len(self.tiles), 4)
        self.assertEqual(len(self.faces), 4)

    def test_loop_connectivity(self):
        self.assertIsNotNone(self.tiles)
        nb = self.lat.nb
        for step in range(4):
            i = self.tiles[step]
            j = self.tiles[(step + 1) % 4]
            k = self.faces[step]
            self.assertEqual(
                int(nb[i, k]),
                j,
                f"step {step}: tile {i} face {k} does not reach tile {j}",
            )

    def test_loop_closed(self):
        self.assertIsNotNone(self.tiles)
        nb = self.lat.nb
        self.assertEqual(int(nb[self.tiles[-1], self.faces[-1]]), self.tiles[0])

    def test_loop_sq_faces_only(self):
        self.assertIsNotNone(self.tiles)
        for k in self.faces:
            self.assertGreaterEqual(k, 8)
            self.assertLessEqual(k, 13)

    # ------------------------------------------------------------------
    # Flux properties after set_flux_loop
    # ------------------------------------------------------------------

    def test_zero_divergence(self):
        self.assertIsNotNone(self.tiles)
        set_flux_loop(self.lat, self.tiles, self.faces, flux=2.5)
        for i in self.tiles:
            div = float(np.dot(_W, self.lat.phi[i]))
            self.assertAlmostEqual(
                div, 0.0, places=12, msg=f"Non-zero divergence at tile {i}: {div}"
            )

    def test_phi_antisymmetry(self):
        self.assertIsNotNone(self.tiles)
        set_flux_loop(self.lat, self.tiles, self.faces, flux=1.0)
        for step in range(4):
            i = self.tiles[step]
            j = self.tiles[(step + 1) % 4]
            k = self.faces[step]
            mk = int(_MIRROR[k])
            self.assertAlmostEqual(
                self.lat.phi[i, k] + self.lat.phi[j, mk],
                0.0,
                places=15,
                msg=f"phi[{i},{k}] + phi[{j},{mk}] != 0",
            )

    def test_faces_frozen(self):
        self.assertIsNotNone(self.tiles)
        set_flux_loop(self.lat, self.tiles, self.faces, flux=1.0)
        for step in range(4):
            i = self.tiles[step]
            j = self.tiles[(step + 1) % 4]
            k = self.faces[step]
            mk = int(_MIRROR[k])
            self.assertTrue(self.lat.fixed_phi[i, k], f"fixed_phi[{i},{k}] not set")
            self.assertTrue(self.lat.fixed_phi[j, mk], f"fixed_phi[{j},{mk}] not set")

    # ------------------------------------------------------------------
    # Clear / restore
    # ------------------------------------------------------------------

    def test_clear_restores_phi(self):
        self.assertIsNotNone(self.tiles)
        set_flux_loop(self.lat, self.tiles, self.faces, flux=1.0)
        clear_flux_loop(self.lat, self.tiles, self.faces)
        for step in range(4):
            i = self.tiles[step]
            j = self.tiles[(step + 1) % 4]
            k = self.faces[step]
            mk = int(_MIRROR[k])
            self.assertEqual(self.lat.phi[i, k], 0.0)
            self.assertEqual(self.lat.phi[j, mk], 0.0)

    def test_clear_unfreezes_faces(self):
        self.assertIsNotNone(self.tiles)
        set_flux_loop(self.lat, self.tiles, self.faces, flux=1.0)
        clear_flux_loop(self.lat, self.tiles, self.faces)
        for step in range(4):
            i = self.tiles[step]
            j = self.tiles[(step + 1) % 4]
            k = self.faces[step]
            mk = int(_MIRROR[k])
            self.assertFalse(self.lat.fixed_phi[i, k])
            self.assertFalse(self.lat.fixed_phi[j, mk])

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def test_flux_persists_under_dynamics(self):
        """Frozen loop phi must be unchanged after several lattice steps."""
        self.assertIsNotNone(self.tiles)
        set_flux_loop(self.lat, self.tiles, self.faces, flux=1.0)
        snapshot = {
            (self.tiles[s], self.faces[s]): self.lat.phi[self.tiles[s], self.faces[s]]
            for s in range(4)
        }
        self.lat.run(20)
        for step in range(4):
            i = self.tiles[step]
            k = self.faces[step]
            self.assertEqual(
                self.lat.phi[i, k],
                snapshot[(i, k)],
                f"Frozen phi changed at tile {i} face {k}",
            )


class TestSpheromak(unittest.TestCase):
    """
    Hopf-linked dual flux loop (spheromak initialisation).

    Ring A: 4-tile corner-tile plaquette (find_shortest_loop).
    Ring B: 4-tile body-centre plaquette threaded through Ring A.
    Linking number = ±1; no shared tiles.
    """

    def setUp(self):
        self.lat = SubstrateLattice(radius_cells=5.0)
        self.A_tiles, self.A_faces = find_shortest_loop(
            self.lat, start=_interior_start(self.lat)
        )
        self.assertIsNotNone(self.A_tiles)
        self.B_tiles, self.B_faces = find_linked_loop(
            self.lat, self.A_tiles, self.A_faces
        )

    # ------------------------------------------------------------------
    # Topology
    # ------------------------------------------------------------------

    def test_linked_loop_found(self):
        self.assertIsNotNone(self.B_tiles, "find_linked_loop returned None")
        self.assertEqual(len(self.B_tiles), 4)
        self.assertEqual(len(self.B_faces), 4)

    def test_no_shared_tiles(self):
        self.assertIsNotNone(self.B_tiles)
        shared = set(self.A_tiles) & set(self.B_tiles)
        self.assertEqual(shared, set(), f"Rings share tiles: {shared}")

    def test_linking_number(self):
        self.assertIsNotNone(self.B_tiles)
        lk = linking_number(self.lat, self.A_tiles, self.B_tiles)
        self.assertAlmostEqual(
            abs(lk), 1.0, places=6, msg=f"Linking number = {lk}, expected ±1"
        )

    def test_linking_number_antisymmetric(self):
        """lk(A,B) = -lk(B,A) for a Hopf link."""
        self.assertIsNotNone(self.B_tiles)
        lk_AB = linking_number(self.lat, self.A_tiles, self.B_tiles)
        lk_BA = linking_number(self.lat, self.B_tiles, self.A_tiles)
        self.assertAlmostEqual(lk_AB, -lk_BA, places=6)

    # ------------------------------------------------------------------
    # Ring B structural validity
    # ------------------------------------------------------------------

    def test_ring_B_connectivity(self):
        self.assertIsNotNone(self.B_tiles)
        nb = self.lat.nb
        for step in range(4):
            i = self.B_tiles[step]
            j = self.B_tiles[(step + 1) % 4]
            k = self.B_faces[step]
            self.assertEqual(int(nb[i, k]), j)

    def test_ring_B_sq_faces(self):
        self.assertIsNotNone(self.B_tiles)
        for k in self.B_faces:
            self.assertGreaterEqual(k, 8)
            self.assertLessEqual(k, 13)

    # ------------------------------------------------------------------
    # Static spheromak (frozen)
    # ------------------------------------------------------------------

    def test_static_both_divergence_free(self):
        self.assertIsNotNone(self.B_tiles)
        init_spheromak(
            self.lat,
            self.A_tiles,
            self.A_faces,
            self.B_tiles,
            self.B_faces,
            amplitude=1.0,
            frozen=True,
        )
        for ring in (self.A_tiles, self.B_tiles):
            for i in ring:
                div = float(np.dot(_W, self.lat.phi[i]))
                self.assertAlmostEqual(
                    div, 0.0, places=12, msg=f"Non-zero divergence at tile {i}: {div}"
                )

    def test_static_all_faces_frozen(self):
        self.assertIsNotNone(self.B_tiles)
        init_spheromak(
            self.lat,
            self.A_tiles,
            self.A_faces,
            self.B_tiles,
            self.B_faces,
            amplitude=1.0,
            frozen=True,
        )
        for tiles, faces in (
            (self.A_tiles, self.A_faces),
            (self.B_tiles, self.B_faces),
        ):
            for step in range(4):
                i, j = tiles[step], tiles[(step + 1) % 4]
                k, mk = faces[step], int(_MIRROR[faces[step]])
                self.assertTrue(self.lat.fixed_phi[i, k])
                self.assertTrue(self.lat.fixed_phi[j, mk])

    def test_static_spheromak_persists(self):
        self.assertIsNotNone(self.B_tiles)
        init_spheromak(
            self.lat,
            self.A_tiles,
            self.A_faces,
            self.B_tiles,
            self.B_faces,
            amplitude=1.0,
            frozen=True,
        )
        snap = {}
        for tiles, faces in (
            (self.A_tiles, self.A_faces),
            (self.B_tiles, self.B_faces),
        ):
            for s in range(4):
                snap[(tiles[s], faces[s])] = self.lat.phi[tiles[s], faces[s]]
        self.lat.run(20)
        for (i, k), v in snap.items():
            self.assertEqual(self.lat.phi[i, k], v)

    # ------------------------------------------------------------------
    # Free resonant spheromak
    # ------------------------------------------------------------------

    def test_free_no_faces_frozen(self):
        self.assertIsNotNone(self.B_tiles)
        init_spheromak(
            self.lat,
            self.A_tiles,
            self.A_faces,
            self.B_tiles,
            self.B_faces,
            amplitude=1.0,
            frozen=False,
        )
        for tiles, faces in (
            (self.A_tiles, self.A_faces),
            (self.B_tiles, self.B_faces),
        ):
            for step in range(4):
                i, j = tiles[step], tiles[(step + 1) % 4]
                k, mk = faces[step], int(_MIRROR[faces[step]])
                self.assertFalse(self.lat.fixed_phi[i, k])
                self.assertFalse(self.lat.fixed_phi[j, mk])

    def test_free_both_rings_have_energy(self):
        """After free init (phi_only=True default) both rings carry nonzero phi."""
        self.assertIsNotNone(self.B_tiles)
        init_spheromak(
            self.lat,
            self.A_tiles,
            self.A_faces,
            self.B_tiles,
            self.B_faces,
            amplitude=1.0,
            frozen=False,
        )
        for ring, faces in ((self.A_tiles, self.A_faces), (self.B_tiles, self.B_faces)):
            phi_energy = sum(
                float(self.lat.phi[t, k]) ** 2 for t, k in zip(ring, faces)
            )
            self.assertGreater(phi_energy, 0.0)

    def test_free_phi_only_u_zero(self):
        """phi_only=True (default): u must be zero on both rings after init."""
        self.assertIsNotNone(self.B_tiles)
        init_spheromak(
            self.lat,
            self.A_tiles,
            self.A_faces,
            self.B_tiles,
            self.B_faces,
            amplitude=1.0,
            frozen=False,
            phi_only=True,
        )
        for ring in (self.A_tiles, self.B_tiles):
            for t in ring:
                self.assertEqual(
                    float(self.lat.u[t]),
                    0.0,
                    msg=f"u[{t}] = {self.lat.u[t]:.6e} != 0 with phi_only=True",
                )

    def test_free_full_u_nonzero(self):
        """phi_only=False: u must be nonzero on both rings after init."""
        self.assertIsNotNone(self.B_tiles)
        init_spheromak(
            self.lat,
            self.A_tiles,
            self.A_faces,
            self.B_tiles,
            self.B_faces,
            amplitude=1.0,
            frozen=False,
            phi_only=False,
        )
        for ring in (self.A_tiles, self.B_tiles):
            u_sq = sum(float(self.lat.u[t]) ** 2 for t in ring)
            self.assertGreater(u_sq, 0.0)


class TestToroidalRing(unittest.TestCase):
    """
    Option 3: 8-tile toroidal ring winding once around a coordinate axis.

    The ring lies in the plane perpendicular to the axis and uses only sq faces.
    It is topologically distinct from the 4-tile plaquette: its winding number
    around the enclosed axis is ±1.
    """

    def setUp(self):
        self.lat = SubstrateLattice(radius_cells=5.0)

    # ------------------------------------------------------------------
    # Structural validity — tested for all three axes
    # ------------------------------------------------------------------

    def _check_ring(self, axis):
        tiles, faces = find_toroidal_ring(self.lat, axis=axis)
        self.assertIsNotNone(tiles, f"No toroidal ring found for axis={axis}")
        self.assertEqual(len(tiles), TOROIDAL_RING_SIZE)
        self.assertEqual(len(faces), TOROIDAL_RING_SIZE)

        nb = self.lat.nb
        for step in range(TOROIDAL_RING_SIZE):
            i = tiles[step]
            j = tiles[(step + 1) % TOROIDAL_RING_SIZE]
            k = faces[step]
            self.assertGreaterEqual(k, 8, f"axis={axis} step {step}: not a sq face")
            self.assertLessEqual(k, 13)
            self.assertEqual(
                int(nb[i, k]), j, f"axis={axis} step {step}: connectivity broken"
            )

        # closed
        self.assertEqual(int(nb[tiles[-1], faces[-1]]), tiles[0])
        return tiles, faces

    def test_ring_z_axis(self):
        self._check_ring("z")

    def test_ring_y_axis(self):
        self._check_ring("y")

    def test_ring_x_axis(self):
        self._check_ring("x")

    # ------------------------------------------------------------------
    # Topological invariant: winding number = ±1
    # ------------------------------------------------------------------

    def test_winding_number_z(self):
        tiles, _ = find_toroidal_ring(self.lat, axis="z")
        self.assertIsNotNone(tiles)
        w = winding_number(self.lat, tiles, axis="z")
        self.assertAlmostEqual(
            abs(w), 1.0, places=6, msg=f"Winding number = {w}, expected ±1"
        )

    def test_winding_number_y(self):
        tiles, _ = find_toroidal_ring(self.lat, axis="y")
        self.assertIsNotNone(tiles)
        w = winding_number(self.lat, tiles, axis="y")
        self.assertAlmostEqual(abs(w), 1.0, places=6)

    def test_winding_number_x(self):
        tiles, _ = find_toroidal_ring(self.lat, axis="x")
        self.assertIsNotNone(tiles)
        w = winding_number(self.lat, tiles, axis="x")
        self.assertAlmostEqual(abs(w), 1.0, places=6)

    # ------------------------------------------------------------------
    # Static frozen winding (option-1 style on the toroidal ring)
    # ------------------------------------------------------------------

    def test_static_winding_zero_divergence(self):
        tiles, faces = find_toroidal_ring(self.lat, axis="z")
        self.assertIsNotNone(tiles)
        init_toroidal_winding(self.lat, tiles, faces, amplitude=1.0, frozen=True)
        for i in tiles:
            div = float(np.dot(_W, self.lat.phi[i]))
            self.assertAlmostEqual(
                div, 0.0, places=12, msg=f"Non-zero divergence at tile {i}: {div}"
            )

    def test_static_winding_faces_frozen(self):
        tiles, faces = find_toroidal_ring(self.lat, axis="z")
        self.assertIsNotNone(tiles)
        init_toroidal_winding(self.lat, tiles, faces, amplitude=1.0, frozen=True)
        for step in range(TOROIDAL_RING_SIZE):
            i = tiles[step]
            j = tiles[(step + 1) % TOROIDAL_RING_SIZE]
            k = faces[step]
            mk = int(_MIRROR[k])
            self.assertTrue(self.lat.fixed_phi[i, k])
            self.assertTrue(self.lat.fixed_phi[j, mk])

    def test_static_winding_persists(self):
        tiles, faces = find_toroidal_ring(self.lat, axis="z")
        self.assertIsNotNone(tiles)
        init_toroidal_winding(self.lat, tiles, faces, amplitude=1.0, frozen=True)
        snap = {
            (tiles[s], faces[s]): self.lat.phi[tiles[s], faces[s]]
            for s in range(TOROIDAL_RING_SIZE)
        }
        self.lat.run(20)
        for step in range(TOROIDAL_RING_SIZE):
            i, k = tiles[step], faces[step]
            self.assertEqual(self.lat.phi[i, k], snap[(i, k)])

    # ------------------------------------------------------------------
    # Free resonant winding (option-2 style on the toroidal ring)
    # ------------------------------------------------------------------

    def test_resonant_winding_u_pattern(self):
        tiles, faces = find_toroidal_ring(self.lat, axis="z")
        self.assertIsNotNone(tiles)
        A = 1.0
        init_toroidal_winding(self.lat, tiles, faces, amplitude=A, frozen=False)
        N = TOROIDAL_RING_SIZE
        K = 2.0 * np.pi / N
        for j, t in enumerate(tiles):
            self.assertAlmostEqual(float(self.lat.u[t]), A * np.cos(K * j), places=15)

    def test_resonant_winding_no_frozen(self):
        tiles, faces = find_toroidal_ring(self.lat, axis="z")
        self.assertIsNotNone(tiles)
        init_toroidal_winding(self.lat, tiles, faces, amplitude=1.0, frozen=False)
        for step in range(TOROIDAL_RING_SIZE):
            i = tiles[step]
            j = tiles[(step + 1) % TOROIDAL_RING_SIZE]
            self.assertFalse(self.lat.fixed_phi[i, faces[step]])
            self.assertFalse(self.lat.fixed_phi[j, int(_MIRROR[faces[step]])])

    def test_resonant_winding_oscillates(self):
        """u at the peak tile (j=0) must cross zero within two full periods."""
        tiles, faces = find_toroidal_ring(self.lat, axis="z")
        self.assertIsNotNone(tiles)
        omega = init_toroidal_winding(
            self.lat, tiles, faces, amplitude=1.0, frozen=False
        )
        dt = self.lat.stable_dt()
        max_steps = int(2.0 * 2.0 * np.pi / omega / dt)
        u_prev = float(self.lat.u[tiles[0]])
        for _ in range(max_steps):
            self.lat.step(dt)
            u_curr = float(self.lat.u[tiles[0]])
            if u_curr <= 0.0 and u_prev > 0.0:
                return
            u_prev = u_curr
        self.fail("Toroidal winding did not oscillate through zero within 2·T")


class TestToroidalSpheromak(unittest.TestCase):
    """
    Geometrically correct spheromak: 8-tile toroidal ring (Ring A, winding_number=±1)
    + 4-tile threading plaquette (Ring B, linking_number=±1 with Ring A).

    Rings are on different BCC sublattices (corner vs body-centre), so they
    share no tiles.  Their resonant frequencies differ (ω_A≈0.77, ω_B≈1.41),
    eliminating the parametric resonance of the equal-N two-plaquette spheromak.
    """

    def setUp(self):
        self.lat = SubstrateLattice(radius_cells=5.0)
        (
            self.A_tiles,
            self.A_faces,
            self.B_tiles,
            self.B_faces,
        ) = find_toroidal_spheromak(self.lat, axis="z")

    # ------------------------------------------------------------------
    # Topology
    # ------------------------------------------------------------------

    def test_rings_found(self):
        self.assertIsNotNone(self.A_tiles, "toroidal ring not found")
        self.assertIsNotNone(self.B_tiles, "threading plaquette not found")
        self.assertEqual(len(self.A_tiles), TOROIDAL_RING_SIZE)
        self.assertEqual(len(self.B_tiles), 4)

    def test_no_shared_tiles(self):
        self.assertIsNotNone(self.B_tiles)
        shared = set(self.A_tiles) & set(self.B_tiles)
        self.assertEqual(shared, set(), f"Rings share tiles: {shared}")

    def test_ring_A_winding_number(self):
        self.assertIsNotNone(self.A_tiles)
        w = winding_number(self.lat, self.A_tiles, axis="z")
        self.assertAlmostEqual(abs(w), 1.0, places=6)

    def test_linking_number(self):
        self.assertIsNotNone(self.B_tiles)
        lk = linking_number(self.lat, self.A_tiles, self.B_tiles)
        self.assertAlmostEqual(
            abs(lk), 1.0, places=6, msg=f"Linking number = {lk}, expected ±1"
        )

    def test_different_sublattices(self):
        """Ring A tiles (z=0) and Ring B tiles (z=±a/2) must be on different z-levels."""
        self.assertIsNotNone(self.B_tiles)
        z_A = set(round(float(self.lat.pos[t, 2]), 6) for t in self.A_tiles)
        z_B = set(round(float(self.lat.pos[t, 2]), 6) for t in self.B_tiles)
        self.assertEqual(
            z_A & z_B, set(), msg=f"Ring A and Ring B share z-levels: {z_A & z_B}"
        )

    # ------------------------------------------------------------------
    # Frequencies
    # ------------------------------------------------------------------

    def test_different_frequencies(self):
        """ω_A(N=8) and ω_B(N=4) must differ — eliminates parametric resonance."""
        omega_A = resonant_freq(len(self.A_tiles))
        omega_B = resonant_freq(len(self.B_tiles))
        self.assertNotAlmostEqual(
            omega_A,
            omega_B,
            places=6,
            msg=f"ω_A={omega_A:.6f} == ω_B={omega_B:.6f}: same frequency → parametric instability",
        )
        self.assertAlmostEqual(omega_A, 2.0 * W_SQ * np.sin(np.pi / 8), places=12)
        self.assertAlmostEqual(omega_B, 2.0 * W_SQ * np.sin(np.pi / 4), places=12)

    # ------------------------------------------------------------------
    # Initialisation (reuses init_spheromak)
    # ------------------------------------------------------------------

    def test_phi_only_u_zero(self):
        self.assertIsNotNone(self.B_tiles)
        init_spheromak(
            self.lat,
            self.A_tiles,
            self.A_faces,
            self.B_tiles,
            self.B_faces,
            amplitude=1.0,
            frozen=False,
            phi_only=True,
        )
        for ring in (self.A_tiles, self.B_tiles):
            for t in ring:
                self.assertEqual(float(self.lat.u[t]), 0.0)

    def test_both_rings_have_phi_energy(self):
        self.assertIsNotNone(self.B_tiles)
        init_spheromak(
            self.lat,
            self.A_tiles,
            self.A_faces,
            self.B_tiles,
            self.B_faces,
            amplitude=1.0,
            frozen=False,
        )
        for ring, faces in ((self.A_tiles, self.A_faces), (self.B_tiles, self.B_faces)):
            phi_energy = sum(
                float(self.lat.phi[t, k]) ** 2 for t, k in zip(ring, faces)
            )
            self.assertGreater(phi_energy, 0.0)

    def test_ring_A_oscillates(self):
        """u at Ring A peak tile must cross zero within two full periods."""
        self.assertIsNotNone(self.A_tiles)
        init_spheromak(
            self.lat,
            self.A_tiles,
            self.A_faces,
            self.B_tiles,
            self.B_faces,
            amplitude=1.0,
            frozen=False,
            phi_only=False,
        )
        omega_A = resonant_freq(len(self.A_tiles))
        dt = self.lat.stable_dt()
        max_steps = int(2.0 * 2.0 * np.pi / omega_A / dt)
        u_prev = float(self.lat.u[self.A_tiles[0]])
        for _ in range(max_steps):
            self.lat.step(dt)
            u_curr = float(self.lat.u[self.A_tiles[0]])
            if u_curr <= 0.0 and u_prev > 0.0:
                return
            u_prev = u_curr
        self.fail("Ring A did not oscillate through zero within 2·T")


class TestResonantLoop(unittest.TestCase):
    """
    Option 2: free traveling-wave initialization on a sq-face plaquette.

    The 4-tile loop (N=4, K=π/2, W_SQ=1) has dispersion ω=√2.
    Initial state:
        u  = [A,  0, −A,  0]          (cosine, one full wavelength around loop)
        φ  = [A/√2, −A/√2, −A/√2, A/√2]  (half-step offset)
    No faces are frozen.
    """

    def setUp(self):
        self.lat = SubstrateLattice(radius_cells=5.0)
        self.tiles, self.faces = find_shortest_loop(
            self.lat, start=_interior_start(self.lat)
        )
        self.assertIsNotNone(self.tiles)
        self.A = 1.0
        self.omega = init_resonant_loop(
            self.lat, self.tiles, self.faces, amplitude=self.A
        )

    # ------------------------------------------------------------------
    # Initial state
    # ------------------------------------------------------------------

    def test_resonant_freq_value(self):
        """ω = 2·W_SQ·|sin(π/4)| = √2 for N=4."""
        self.assertAlmostEqual(self.omega, 2.0 * W_SQ * np.sin(np.pi / 4), places=15)

    def test_u_cosine_pattern(self):
        N = len(self.tiles)
        K = 2.0 * np.pi / N
        for j, tile_idx in enumerate(self.tiles):
            self.assertAlmostEqual(
                float(self.lat.u[tile_idx]),
                self.A * np.cos(K * j),
                places=15,
            )

    def test_phi_half_step_pattern(self):
        N = len(self.tiles)
        K = 2.0 * np.pi / N
        for j in range(N):
            i = self.tiles[j]
            k = self.faces[j]
            self.assertAlmostEqual(
                float(self.lat.phi[i, k]),
                self.A * np.cos(K * (j + 0.5)),
                places=15,
            )

    def test_initial_phi_antisymmetry(self):
        for step in range(4):
            i = self.tiles[step]
            j = self.tiles[(step + 1) % 4]
            k = self.faces[step]
            mk = int(_MIRROR[k])
            self.assertAlmostEqual(
                self.lat.phi[i, k] + self.lat.phi[j, mk],
                0.0,
                places=15,
            )

    def test_no_faces_frozen(self):
        for step in range(4):
            i = self.tiles[step]
            j = self.tiles[(step + 1) % 4]
            k = self.faces[step]
            mk = int(_MIRROR[k])
            self.assertFalse(self.lat.fixed_phi[i, k])
            self.assertFalse(self.lat.fixed_phi[j, mk])

    # ------------------------------------------------------------------
    # First-step dynamics (exact while non-loop phi is still zero)
    # ------------------------------------------------------------------

    def test_first_step_u_update(self):
        """
        With all non-loop phi=0 at t=0, the first-step Δu is determined solely
        by loop phi.  The wave equation gives:
            Δu_j=0 = 0  (at cosine maximum, du/dt=0)
            Δu_j=1 = +dt·A·ω  (zero-crossing, rising)
            Δu_j=2 = 0  (at cosine minimum, du/dt=0)
            Δu_j=3 = −dt·A·ω  (zero-crossing, falling)
        """
        dt = self.lat.stable_dt()
        u_before = [float(self.lat.u[t]) for t in self.tiles]
        self.lat.step(dt)
        u_after = [float(self.lat.u[t]) for t in self.tiles]

        expected_du = [0.0, dt * self.A * self.omega, 0.0, -dt * self.A * self.omega]
        for j in range(4):
            self.assertAlmostEqual(
                u_after[j] - u_before[j],
                expected_du[j],
                places=12,
                msg=f"Δu at j={j}: got {u_after[j]-u_before[j]:.6e}, expected {expected_du[j]:.6e}",
            )

    # ------------------------------------------------------------------
    # Oscillation
    # ------------------------------------------------------------------

    def test_wave_oscillates(self):
        """
        u at the initially-maximum tile (j=0, u=A) must cross zero within
        one full period T = 2π/ω, even with leakage into surrounding tiles.
        """
        dt = self.lat.stable_dt()
        T = 2.0 * np.pi / self.omega
        max_steps = int(2.0 * T / dt)  # two full periods as generous margin

        u_prev = float(self.lat.u[self.tiles[0]])
        for _ in range(max_steps):
            self.lat.step(dt)
            u_curr = float(self.lat.u[self.tiles[0]])
            if u_curr <= 0.0 and u_prev > 0.0:
                return  # zero crossing found
            u_prev = u_curr

        self.fail(
            f"u at peak tile did not oscillate through zero within {max_steps} steps (2·T)"
        )


if __name__ == "__main__":
    unittest.main()
