"""
Minimal, single-cell functional-expectation tests for step_cross's gate
modes -- deliberately NOT using any winding/knot/spheromak aggregate. The
question here is purely: does each gate do what its own formula says it
should do, at the smallest possible scale (one cell, one active face, one
tick), with an exactly hand-derived expected value?

Motivation (2026-08-18 discussion): before trusting any aggregate-scale
result (electron/neutron/photon probes, spheromak merging, PHARE
comparison), gate="v1" vs gate="barrier" must first be checked at the level
where their formulas can be verified by hand -- this is that check.

Kernel reference (equilibrium/substrate/lattice.py step_cross /
inc/eq/substrate.hpp substrate_step_cross[_barrier], both gates share the
same drain+mirror redistribution once "fired"):

    inward_k = max(-phi[i,k], 0)                    (>= 0)
    v1:      fire_k = 1 always (per face, independent of any threshold)
    barrier: drive_i = sum_k W_k * inward_k (POOLED over the whole cell)
             fire_i  = 1 if drive_i > m0 else 0      (same for all 14 faces)
    delta_k = dt * scale * sigma(lam)^2 * fire * inward_k
    phi[i,k]        += n_adj(k) * delta_k             (drain the source face)
    phi[i,j] (j adj) += delta_k   for each of the (6 hex / 4 sq) edge-adjacent
                                    faces j of k on the SAME cell           (*)
    [+ antisymmetric mirror writes on the neighbour cell, irrelevant here --
     this file only inspects cell i's own resulting phi]

(*) is the "exhaust": whatever this file calls the redistribution onto
face-adjacent faces of the one face that fired.
"""

import unittest

import numpy as np

from equilibrium.substrate.lattice import (
    W_HEX,
    W_SQ,
    SubstrateLattice,
    _FACE_ADJ,
    _K_REV,
)

DT = 1e-4
SCALE = 1.0
K_HEX = 0  # a hex face, n_adj = 6
ADJ_HEX = [3, 5, 6, 8, 10, 12]  # _FACE_ADJ[0], edge-adjacent faces of face 0
N_ADJ_HEX = 6

K_SQ = 8  # a sq face (-x), n_adj = 4
ADJ_SQ = [0, 1, 2, 3]  # _FACE_ADJ[8][:4], edge-adjacent faces of face 8
N_ADJ_SQ = 4


def _center_lattice():
    """Radius-4 lattice, its centre cell has all 14 neighbours valid."""
    lat = SubstrateLattice(radius_cells=4.0)
    i = int(np.argmin(np.linalg.norm(lat.pos, axis=1)))
    assert lat._valid[i].all(), "centre cell must have all 14 neighbours"
    return lat, i


class TestV1UnconditionalErosion(unittest.TestCase):
    """
    gate="v1" has no threshold anywhere in its formula: ANY nonzero inward
    phi on ANY face erodes, every tick, regardless of amplitude. This is the
    direct, single-cell demonstration that v1 cannot distinguish "a knot
    attempting a collective crossing" from "an arbitrarily small disturbance
    passing through" -- there is no concept of the latter being exempt.
    """

    def _run(self, A):
        lat, i = _center_lattice()
        lat.phi[i, K_HEX] = -A
        lat.step_cross(DT, scale=SCALE, gate="v1")
        return lat, i

    def test_erodes_at_ordinary_amplitude(self):
        A = 1.0
        lat, i = self._run(A)
        delta = DT * SCALE * A  # sigma=1 (lam=0)
        expected_face0 = -A + N_ADJ_HEX * delta
        self.assertAlmostEqual(float(lat.phi[i, K_HEX]), expected_face0, places=12)

    def test_erodes_at_photon_scale_amplitude(self):
        """
        Same formula, amplitude dropped 8 orders of magnitude (representative
        of a face-hopping wave disturbance, not a knot). v1 erodes it by
        exactly the same PROPORTION -- there is no amplitude below which it
        stops firing. This is v1 failing the "a photon must not erode"
        requirement (doc/mems/project_step_cross_v2.md, requirement R0),
        demonstrated at the smallest possible scale.
        """
        A = 1e-8
        lat, i = self._run(A)
        delta = DT * SCALE * A
        expected_face0 = -A + N_ADJ_HEX * delta
        self.assertNotEqual(
            float(lat.phi[i, K_HEX]), -A, "v1 must have moved phi at all"
        )
        self.assertAlmostEqual(float(lat.phi[i, K_HEX]), expected_face0, places=20)
        # the fractional erosion is IDENTICAL to the A=1.0 case -- v1 is
        # scale-invariant, i.e. genuinely has no threshold at any amplitude.
        frac = (expected_face0 - (-A)) / (-A)
        frac_at_unit_amplitude = (N_ADJ_HEX * DT * SCALE * 1.0) / 1.0
        self.assertAlmostEqual(frac, -frac_at_unit_amplitude, places=12)


class TestV1ExhaustIsotropic(unittest.TestCase):
    """
    Checks Philip's "secondary magnetic exhaust" hypothesis directly against
    the CURRENT kernel: when face k fires, the drained flux is redistributed
    onto its edge-adjacent faces. Is that redistribution isotropic (same
    delta on all 6 adjacent faces, i.e. no preferred direction -- cannot by
    itself produce a rotational/curl signature) or does it already carry
    some directional weighting?

    Result (verified here): CURRENTLY isotropic, by construction -- the
    kernel adds the exact same `delta` to every adjacent face regardless of
    its relative orientation to face k. A rotational "exhaust" signature is
    NOT present in the kernel as it exists today; it would require making
    the redistribution weight depend on face geometry (e.g. relative to a
    winding's own crossing direction), which nothing currently does.
    """

    def test_all_six_adjacent_faces_receive_identical_delta(self):
        A = 1.0
        lat, i = _center_lattice()
        lat.phi[i, K_HEX] = -A
        lat.step_cross(DT, scale=SCALE, gate="v1")

        deltas = np.array([float(lat.phi[i, j]) for j in ADJ_HEX])
        self.assertTrue(
            np.allclose(deltas, deltas[0], atol=0, rtol=0),
            f"exhaust is not isotropic: {deltas}",
        )
        expected = DT * SCALE * A
        self.assertAlmostEqual(float(deltas[0]), expected, places=12)

    def test_no_faces_outside_source_and_adjacent_set_change(self):
        """Only face k and its 6 edge-adjacent faces move; the other 7 faces
        of the centre cell (the ones NOT edge-adjacent to k=0) are untouched."""
        lat, i = _center_lattice()
        lat.phi[i, K_HEX] = -1.0
        lat.step_cross(DT, scale=SCALE, gate="v1")

        untouched = [k for k in range(14) if k not in (K_HEX, *ADJ_HEX)]
        for k in untouched:
            self.assertEqual(float(lat.phi[i, k]), 0.0, f"face {k} should be untouched")


class TestBarrierThresholdBoundary(unittest.TestCase):
    """
    gate="barrier": the pooled per-cell drive (here, a single active hex
    face, drive = W_HEX * A) must be compared against m0 as a hard,
    deterministic switch -- exactly zero change below, exact v1-style
    kernel value above. No partial/fractional firing near the boundary.
    """

    M0 = 1e-3

    def test_just_below_threshold_is_a_bit_exact_no_op(self):
        A = (self.M0 / W_HEX) * 0.99  # drive = 0.99 * m0
        lat, i = _center_lattice()
        phi_before = lat.phi[i].copy()
        lat.phi[i, K_HEX] = -A
        phi_before[K_HEX] = -A

        lat.step_cross(DT, scale=SCALE, gate="barrier", m0=self.M0)

        np.testing.assert_array_equal(
            np.asarray(lat.phi[i]),
            np.asarray(phi_before),
            err_msg="barrier gate fired below threshold",
        )

    def test_just_above_threshold_fires_exact_v1_style_value(self):
        A = (self.M0 / W_HEX) * 1.01  # drive = 1.01 * m0
        lat, i = _center_lattice()
        lat.phi[i, K_HEX] = -A

        lat.step_cross(DT, scale=SCALE, gate="barrier", m0=self.M0)

        delta = DT * SCALE * A
        expected_face0 = -A + N_ADJ_HEX * delta
        self.assertAlmostEqual(float(lat.phi[i, K_HEX]), expected_face0, places=12)
        for j in ADJ_HEX:
            self.assertAlmostEqual(float(lat.phi[i, j]), delta, places=12)

    def test_exactly_at_threshold_does_not_fire(self):
        """drive > m0 is strict -- drive == m0 must NOT fire."""
        A = self.M0 / W_HEX  # drive == m0 exactly
        lat, i = _center_lattice()
        lat.phi[i, K_HEX] = -A

        lat.step_cross(DT, scale=SCALE, gate="barrier", m0=self.M0)

        self.assertEqual(
            float(lat.phi[i, K_HEX]), -A, "drive==m0 must not fire (strict >)"
        )


class TestBarrierVsV1SameSetupDivergentOutcome(unittest.TestCase):
    """
    The single decisive comparison: identical minimal IC (one cell, one
    small hex-face disturbance, amplitude chosen to sit below a barrier
    threshold), run through both gates. v1 must move it; barrier must not.
    This is requirement R0 ("a photon must not erode") reduced to its
    smallest possible falsifiable case -- no wave packet, no domain, no
    winding topology required to make the point.
    """

    def test_same_ic_diverges_under_the_two_gates(self):
        A = 1e-6
        m0 = 1e-3  # drive = W_HEX*A = 3e-6, comfortably < m0

        lat_v1, i = _center_lattice()
        lat_v1.phi[i, K_HEX] = -A
        lat_v1.step_cross(DT, scale=SCALE, gate="v1")

        lat_bar, j = _center_lattice()
        lat_bar.phi[j, K_HEX] = -A
        lat_bar.step_cross(DT, scale=SCALE, gate="barrier", m0=m0)

        self.assertNotEqual(
            float(lat_v1.phi[i, K_HEX]), -A, "v1 should have eroded this face"
        )
        self.assertEqual(
            float(lat_bar.phi[j, K_HEX]),
            -A,
            "barrier should have left this face untouched (sub-threshold)",
        )


class TestSqFaceUnconditionalErosionAndExhaust(unittest.TestCase):
    """
    Same claims as TestV1UnconditionalErosion / TestV1ExhaustIsotropic, but
    on a SQUARE face (k=8, n_adj=4, W_SQ=1) rather than a hex face (k=0,
    n_adj=6, W_HEX=3) -- confirms nothing above was an artefact of picking a
    hex face specifically. v1's per-face delta formula does not reference
    W_HEX/W_SQ at all (only n_adj differs between the two face types); the
    barrier gate's POOLED drive does use W_k, so W_SQ=1 changes the
    drive-vs-m0 arithmetic (drive = W_SQ*A = A, not 3*A as for a hex face).
    """

    def test_v1_erodes_sq_face_at_photon_scale(self):
        A = 1e-8
        lat, i = _center_lattice()
        lat.phi[i, K_SQ] = -A
        lat.step_cross(DT, scale=SCALE, gate="v1")

        delta = DT * SCALE * A
        expected_face8 = -A + N_ADJ_SQ * delta
        self.assertNotEqual(float(lat.phi[i, K_SQ]), -A)
        self.assertAlmostEqual(float(lat.phi[i, K_SQ]), expected_face8, places=20)

    def test_v1_sq_exhaust_isotropic_over_four_adjacent_faces(self):
        A = 1.0
        lat, i = _center_lattice()
        lat.phi[i, K_SQ] = -A
        lat.step_cross(DT, scale=SCALE, gate="v1")

        deltas = np.array([float(lat.phi[i, j]) for j in ADJ_SQ])
        self.assertTrue(np.allclose(deltas, deltas[0], atol=0, rtol=0), deltas)
        self.assertAlmostEqual(float(deltas[0]), DT * SCALE * A, places=12)

        untouched = [k for k in range(14) if k not in (K_SQ, *ADJ_SQ)]
        for k in untouched:
            self.assertEqual(float(lat.phi[i, k]), 0.0, f"face {k} should be untouched")

    def test_barrier_sq_face_threshold_uses_W_SQ_not_W_HEX(self):
        """drive = W_SQ*A = A for a sq face -- the boundary sits at A=m0
        exactly, not A=m0/3 as it would for a hex face."""
        m0 = 1e-3

        lat_below, i = _center_lattice()
        A_below = m0 * 0.99
        lat_below.phi[i, K_SQ] = -A_below
        lat_below.step_cross(DT, scale=SCALE, gate="barrier", m0=m0)
        self.assertEqual(float(lat_below.phi[i, K_SQ]), -A_below)

        lat_above, j = _center_lattice()
        A_above = m0 * 1.01
        lat_above.phi[j, K_SQ] = -A_above
        lat_above.step_cross(DT, scale=SCALE, gate="barrier", m0=m0)
        delta = DT * SCALE * A_above
        expected = -A_above + N_ADJ_SQ * delta
        self.assertAlmostEqual(float(lat_above.phi[j, K_SQ]), expected, places=12)


class TestAntisymmetricMirrorPreserved(unittest.TestCase):
    """
    step_cross only ever directly computes deltas from cell i's own
    perspective (whichever cell sees NEGATIVE/inward phi on a face); the
    neighbouring cell's mirrored face is updated by an equal-and-opposite
    "mirror" write, not by independently re-deriving it. These tests check,
    with hand-derived numbers on the smallest fixture that has a real
    neighbour (two/few cells, not one), that the antisymmetry invariant
    phi[i,k] + phi[nb[i,k], mirror(k)] == 0 survives a step_cross tick
    EXACTLY -- the same invariant test_maxwell.py's
    TestDivergencePreservation checks for the plain wave step, here checked
    for step_cross specifically.
    """

    def test_drain_mirror_pair_stays_antisymmetric(self):
        A = 1.0
        lat, i = _center_lattice()
        j = int(lat.nb[i, K_HEX])
        mk = int(_K_REV[K_HEX])  # mirror of face 0 is face 7

        lat.phi[i, K_HEX] = -A
        lat.phi[j, mk] = A  # antisymmetric IC: phi[i,k] + phi[j,mk] == 0

        lat.step_cross(DT, scale=SCALE, gate="v1")

        delta = DT * SCALE * A
        expected_i = -A + N_ADJ_HEX * delta
        expected_j = A - N_ADJ_HEX * delta
        self.assertAlmostEqual(float(lat.phi[i, K_HEX]), expected_i, places=12)
        self.assertAlmostEqual(float(lat.phi[j, mk]), expected_j, places=12)
        # antisymmetry preserved EXACTLY (bit-exact telescoping, not just close)
        self.assertEqual(float(lat.phi[i, K_HEX]) + float(lat.phi[j, mk]), 0.0)

    def test_adjacent_face_mirror_pair_stays_antisymmetric(self):
        """Face k=0 firing also redistributes onto its adjacent face k=3 on
        cell i; that redistribution mirrors onto nb[i,3]'s face K_REV[3]=4,
        which started at 0 (antisymmetric with cell i's own face 3, also 0)."""
        A = 1.0
        lat, i = _center_lattice()
        adj_face = 3
        m = int(lat.nb[i, adj_face])
        mj = int(_K_REV[adj_face])  # = 4

        lat.phi[i, K_HEX] = -A  # the only active face; face 3 and its mirror start at 0

        lat.step_cross(DT, scale=SCALE, gate="v1")

        delta = DT * SCALE * A
        self.assertAlmostEqual(float(lat.phi[i, adj_face]), delta, places=12)
        self.assertAlmostEqual(float(lat.phi[m, mj]), -delta, places=12)
        self.assertEqual(float(lat.phi[i, adj_face]) + float(lat.phi[m, mj]), 0.0)

    def test_full_14_face_antisymmetry_invariant_after_step(self):
        """General regression check (not just the two hand-picked pairs
        above): starting from a fully antisymmetric random field, every one
        of the centre cell's 14 face pairs stays antisymmetric to machine
        precision after step_cross -- same style as test_maxwell.py's
        TestDivergencePreservation.test_div_b_zero_initial_preserved."""
        rng = np.random.default_rng(0)
        lat, i = _center_lattice()
        valid = lat._valid
        safe_nb = np.where(valid, lat.nb, 0)

        raw = rng.normal(size=lat.phi.shape)
        antisym = np.where(valid, raw - raw[safe_nb, _K_REV[None, :]], 0.0)
        lat.phi = antisym

        lat.step_cross(DT, scale=SCALE, gate="v1")

        max_err = 0.0
        for k in range(14):
            mk = int(_K_REV[k])
            j = lat.nb[:, k]
            mask = j >= 0
            errs = lat.phi[mask, k] + lat.phi[j[mask], mk]
            if errs.size:
                max_err = max(max_err, float(np.max(np.abs(errs))))
        self.assertLess(
            max_err, 1e-12, f"antisymmetry broken after step_cross: {max_err}"
        )


class TestStepCrossIsMotionBlind(unittest.TestCase):
    """
    The "photon must not erode" question (R0) isn't really about amplitude
    alone -- it's about whether step_cross can tell "this flux is part of a
    genuinely propagating face-hopping disturbance" from "this flux is a
    static value someone assigned directly". It structurally cannot: its
    formula (equilibrium/substrate/lattice.py step_cross /
    inc/eq/substrate.hpp) is a pure function of the instantaneous (phi, lam)
    arrays -- it never reads u, never reads a velocity, has no memory of the
    previous tick. This is checked here empirically, not just by code
    inspection: build a REAL propagating wave (plain lat.step(), Faraday +
    Ampere, genuine du/dt != 0 motion) up to some tick, snapshot its phi,
    then show step_cross applied to that snapshot is bit-for-bit identical
    to step_cross applied to a freshly constructed lattice that never moved
    at all and had that exact phi array assigned directly.
    """

    def _propagate_and_snapshot(self, n_steps):
        """A real, multi-face-active traveling disturbance: 1-D plane wave
        u=sin(Kx), phi=0 initially, stepped with the plain (no step_cross)
        wave equation so phi becomes genuinely wave-carried, not hand-set."""
        lat = SubstrateLattice(radius_cells=6.0)
        lat.u = np.sin(0.5 * lat.pos[:, 0])
        dt = lat.stable_dt()
        for _ in range(n_steps):
            lat.step(dt)
        self.assertGreater(
            float(np.max(np.abs(lat.phi))), 0.0, "wave must have produced nonzero phi"
        )
        return lat, dt

    def test_propagated_and_hand_set_phi_erode_identically_under_v1(self):
        lat_moving, dt_wave = self._propagate_and_snapshot(n_steps=20)
        phi_snapshot = lat_moving.phi.copy()

        lat_static = SubstrateLattice(radius_cells=6.0)  # never stepped: u=0, no motion
        lat_static.phi = phi_snapshot.copy()

        lat_moving.step_cross(DT, scale=SCALE, gate="v1")
        lat_static.step_cross(DT, scale=SCALE, gate="v1")

        np.testing.assert_array_equal(
            lat_moving.phi,
            lat_static.phi,
            err_msg="step_cross gave different results for a moving wave vs "
            "a static twin with the identical phi snapshot -- it should be "
            "impossible for these to differ, since step_cross reads only "
            "(phi, lam), never u or history",
        )

    def test_propagated_and_hand_set_phi_erode_identically_under_barrier(self):
        lat_moving, dt_wave = self._propagate_and_snapshot(n_steps=20)
        phi_snapshot = lat_moving.phi.copy()
        m0 = float(np.abs(phi_snapshot).mean())  # pick something mid-distribution

        lat_static = SubstrateLattice(radius_cells=6.0)
        lat_static.phi = phi_snapshot.copy()

        lat_moving.step_cross(DT, scale=SCALE, gate="barrier", m0=m0)
        lat_static.step_cross(DT, scale=SCALE, gate="barrier", m0=m0)

        np.testing.assert_array_equal(lat_moving.phi, lat_static.phi)

    def test_v1_erodes_the_moving_wave_regardless_of_amplitude(self):
        """Same wave, amplitude scaled down 1e6x (still genuinely moving --
        u=1e-6*sin(Kx) still propagates under the plain wave step, just with
        a smaller phi). v1 still erodes it by the identical FRACTION as the
        unscaled wave -- motion doesn't earn any exemption, same conclusion
        as TestV1UnconditionalErosion but now on a real propagating field
        instead of one hand-set face."""
        lat_unit, _ = self._propagate_and_snapshot(n_steps=20)

        lat_small = SubstrateLattice(radius_cells=6.0)
        lat_small.u = 1e-6 * np.sin(0.5 * lat_small.pos[:, 0])
        dt = lat_small.stable_dt()
        for _ in range(20):
            lat_small.step(dt)

        phi_unit_before = lat_unit.phi.copy()
        phi_small_before = lat_small.phi.copy()
        self.assertTrue(
            np.allclose(phi_small_before, 1e-6 * phi_unit_before, rtol=1e-6)
        )

        lat_unit.step_cross(DT, scale=SCALE, gate="v1")
        lat_small.step_cross(DT, scale=SCALE, gate="v1")

        frac_unit = np.divide(
            lat_unit.phi - phi_unit_before,
            phi_unit_before,
            out=np.zeros_like(phi_unit_before),
            where=phi_unit_before != 0,
        )
        frac_small = np.divide(
            lat_small.phi - phi_small_before,
            phi_small_before,
            out=np.zeros_like(phi_small_before),
            where=phi_small_before != 0,
        )
        np.testing.assert_allclose(frac_small, frac_unit, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
