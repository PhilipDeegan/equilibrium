"""
Tests for the C++ AdaptiveLattice.

TestAdaptiveAMRSweep       — adapt() splits/coarsens correctly driven by phi-flux criterion
TestAdaptiveFluxTube       — toroidal ring IC geometry and phi antisymmetry
TestFluxConservation       — total u conserved at AMR coarse/fine interfaces
TestStepCrossConservation  — total phi conserved by step_cross, incl. at AMR interfaces
TestStepCrossMultiLevelConservation — same, across genuine multi-level (>1) jumps
TestByLevelStability       — by_level wave stepping alone does not amplify H at AMR interfaces
TestByLevelMultiLevelStability — by_level (+ step_cross, + step_lam, + regrow) at
                             R=8 across a nested L0/L1/L2 cascade — scaffolding
                             that does NOT yet reproduce the real runaway
TestSpheromakDecayMultiLevelRunaway — full production sweep pipeline at R=30;
                             confirmed and now resolved the max_level=2 runaway
                             (step_cross CFL scaling + split() phi duplication,
                             both fixed)
TestStepCrossUniformVsJunctionMismatch — uniform (no coarse-fine boundary)
                             level-2 mesh vs partial (genuine junction) refinement —
                             confirmed the residual was a coarse-fine
                             neighbour-mismatch issue (split() phi duplication,
                             now fixed), not an inherent per-cell level-2 rate
TestGrowingAMRSchedule     — geometrically growing adapt-call cadence helper
"""

import os
import unittest

import numpy as np

from tst.substrate.adaptive import GrowingAMRSchedule, build_adaptive_sphere

# Override via EQ_TEST_N_THREADS=<n> to avoid oversubscribing when other
# n_threads-heavy runs are active on the same machine.
_N_THREADS = int(os.environ.get("EQ_TEST_N_THREADS", 1))


def _nearest_neighbour(cl, origin_leaf: int) -> int:
    """Return the leaf index of the cell geometrically nearest to origin_leaf."""
    pos = np.array(cl.leaf_pos)
    d2 = ((pos - pos[origin_leaf]) ** 2).sum(axis=1)
    d2[origin_leaf] = np.inf
    return int(np.argmin(d2))


class TestAdaptiveAMRSweep(unittest.TestCase):
    """AMR sweep: adapt() splits/coarsens correctly; more levels → more leaves near origin."""

    R = 12.0
    DT = 0.003
    N_WARM = 200
    N_POST = 100
    SPLIT_HI = 1e-3
    SPLIT_LO = 1e-12

    def _build_and_run(self, max_level: int):
        alat, origin_leaf = build_adaptive_sphere(
            self.R,
            max_level=max_level,
            split_hi=self.SPLIT_HI,
            split_lo=self.SPLIT_LO,
            fix_origin=True,
        )
        for _ in range(self.N_WARM):
            alat.step(0, self.DT)
        alat.adapt(0)
        for _ in range(self.N_POST):
            alat.step(0, self.DT)
        return alat, alat[0]

    def test_max_level_0_no_splits(self):
        """With max_level=0, adapt() must not split any cell (already at limit)."""
        alat, origin_leaf = build_adaptive_sphere(
            self.R,
            max_level=0,
            split_hi=self.SPLIT_HI,
            split_lo=self.SPLIT_LO,
            fix_origin=True,
        )
        cl = alat[0]
        n_before = cl.n_leaves
        for _ in range(self.N_WARM):
            alat.step(0, self.DT)
        alat.adapt(0)
        self.assertEqual(
            cl.n_leaves, n_before, "max_level=0: adapt() must not create new leaves"
        )

    def test_max_level_1_splits_occur(self):
        """With max_level=1, adapt() must split at least one cell near the source."""
        alat, origin_leaf = build_adaptive_sphere(
            self.R,
            max_level=1,
            split_hi=self.SPLIT_HI,
            split_lo=self.SPLIT_LO,
            fix_origin=True,
        )
        cl = alat[0]
        n_before = cl.n_leaves
        for _ in range(self.N_WARM):
            alat.step(0, self.DT)
        alat.adapt(0)
        self.assertGreater(
            cl.n_leaves,
            n_before,
            "max_level=1: adapt() must produce at least one split",
        )

    def test_higher_max_level_more_leaves(self):
        """Leaf count after adapt() must be non-decreasing with max_level."""
        leaf_counts = {}
        for lvl in (1, 2, 3):
            _, cl = self._build_and_run(lvl)
            leaf_counts[lvl] = cl.n_leaves
        self.assertLessEqual(
            leaf_counts[1],
            leaf_counts[2],
            "max_level=2 must produce >= leaves than max_level=1",
        )
        self.assertLessEqual(
            leaf_counts[2],
            leaf_counts[3],
            "max_level=3 must produce >= leaves than max_level=2",
        )

    def test_refined_has_more_cells_near_origin(self):
        """After adapt(), higher max_level must place more cells within r<3 than flat."""

        def count_near(cl, r_thresh=3.0):
            pos = np.array(cl.leaf_pos)
            r = np.sqrt((pos**2).sum(axis=1))
            return int((r < r_thresh).sum())

        _, cl_flat = self._build_and_run(max_level=0)
        _, cl_refined = self._build_and_run(max_level=2)
        n_flat = count_near(cl_flat)
        n_refined = count_near(cl_refined)
        self.assertGreater(
            n_refined,
            n_flat,
            f"refined (max_level=2) must have more cells near origin "
            f"than flat (max_level=0): got {n_refined} vs {n_flat}",
        )


class TestAdaptiveFluxTube(unittest.TestCase):
    """Toroidal ring IC on the C++ adaptive lattice."""

    R = 5.0

    def _build(self):
        return build_adaptive_sphere(self.R, u_origin=0.0)

    def test_ring_found(self):
        from tst.substrate.adaptive.adaptive_flux_tube import find_toroidal_ring

        alat, _ = self._build()
        cl = alat[0]
        ring_js, ring_faces = find_toroidal_ring(cl, axis="z")
        self.assertIsNotNone(ring_js, "No toroidal ring found for axis=z")
        self.assertEqual(len(ring_js), 8)
        self.assertEqual(len(ring_faces), 8)

    def test_ring_connectivity(self):
        from tst.substrate.adaptive.adaptive_flux_tube import find_toroidal_ring

        alat, _ = self._build()
        cl = alat[0]
        ring_js, ring_faces = find_toroidal_ring(cl, axis="z")
        self.assertIsNotNone(ring_js)
        nb_g = np.array(cl.leaf_nb)
        leaf_g = np.array(cl.leaf_global_indices)
        g_to_j = {int(g): j for j, g in enumerate(leaf_g)}
        n = len(ring_js)
        for step in range(n):
            j_i = ring_js[step]
            j_nxt = ring_js[(step + 1) % n]
            k = ring_faces[step]
            g_nb = int(nb_g[j_i, k])
            self.assertGreaterEqual(g_nb, 0, f"step {step}: no neighbour via face {k}")
            j_via_nb = g_to_j.get(g_nb, -1)
            self.assertEqual(j_via_nb, j_nxt, f"step {step}: connectivity broken")

    def test_ring_sq_faces_only(self):
        from tst.substrate.adaptive.adaptive_flux_tube import find_toroidal_ring

        alat, _ = self._build()
        cl = alat[0]
        _, ring_faces = find_toroidal_ring(cl, axis="z")
        for k in ring_faces:
            self.assertGreaterEqual(k, 8)
            self.assertLessEqual(k, 13)

    def test_winding_phi_antisymmetry(self):
        from tst.substrate.adaptive.adaptive_flux_tube import (
            _MIRROR,
            find_toroidal_ring,
            init_toroidal_winding,
        )

        alat, _ = self._build()
        cl = alat[0]
        ring_js, ring_faces = find_toroidal_ring(cl, axis="z")
        self.assertIsNotNone(ring_js)
        init_toroidal_winding(cl, ring_js, ring_faces, amplitude=1.0)
        phi = np.array(cl.leaf_phi)
        n = len(ring_js)
        for step in range(n):
            j_i = ring_js[step]
            j_nxt = ring_js[(step + 1) % n]
            k = ring_faces[step]
            mk = int(_MIRROR[k])
            self.assertAlmostEqual(
                phi[k, j_i] + phi[mk, j_nxt],
                0.0,
                places=15,
                msg=f"step {step}: phi antisymmetry broken",
            )

    def test_frozen_winding_persists(self):
        from tst.substrate.adaptive.adaptive_flux_tube import (
            find_toroidal_ring,
            init_toroidal_winding,
        )

        alat, _ = self._build()
        cl = alat[0]
        ring_js, ring_faces = find_toroidal_ring(cl, axis="z")
        self.assertIsNotNone(ring_js)
        init_toroidal_winding(cl, ring_js, ring_faces, amplitude=1.0, frozen=True)
        phi_before = {
            (ring_js[s], ring_faces[s]): float(
                np.array(cl.leaf_phi)[ring_faces[s], ring_js[s]]
            )
            for s in range(len(ring_js))
        }
        dt = 0.003
        for _ in range(30):
            alat.step(0, dt)
        phi_after = np.array(cl.leaf_phi)
        for (j, k), v in phi_before.items():
            self.assertAlmostEqual(
                phi_after[k, j],
                v,
                places=15,
                msg=f"Frozen phi changed at leaf {j} face {k}",
            )


class TestFluxConservation(unittest.TestCase):
    """Total u must be conserved at AMR interfaces (no absorbing BC)."""

    R = 8.0
    D1 = 1.0
    W2_SUM = 8 * 9.0 + 6 * 1.0

    def _dt(self, cl):
        lam = np.array(cl.leaf_lam, dtype=np.float64)
        lvl = np.array(cl.leaf_level)
        sigma_max = float(np.max(1.0 / (1.0 + lam)))
        L_min = self.D1 / (2.0 ** int(lvl.max()))
        return 0.4 * L_min / (sigma_max * self.W2_SUM)

    def _total_u(self, cl):
        return float(np.sum(np.array(cl.leaf_u)))

    def _gaussian_ic(self, cl):
        pos = np.array(cl.leaf_pos)
        r = np.sqrt((pos**2).sum(axis=1))
        cl.set_leaf_u(np.exp(-0.5 * (r / 3.0) ** 2))

    def _split_shell(self, alat, cl, r_centre, width):
        from equilibrium import lattice as _lat

        info = _lat.DSTInfo()
        info.split_hi = 0.5
        info.split_lo = 0.0
        info.max_level = 1
        cl.info = info
        pos = np.array(cl.leaf_pos)
        r = np.sqrt((pos**2).sum(axis=1))
        cmax = np.full(cl.n_leaves, np.nan)
        cmax[np.abs(r - r_centre) < width] = 1.0
        cl.set_criterion_max(cmax)
        alat.adapt(0)
        alat.rebuild(0)

    def test_uniform_two_step(self):
        """Uniform: Σu conserved after two steps (machine precision)."""
        alat, _ = build_adaptive_sphere(self.R, u_origin=0.0)
        cl = alat[0]
        self._gaussian_ic(cl)
        u0 = self._total_u(cl)
        dt = self._dt(cl)
        alat.step(0, dt)
        alat.step(0, dt)
        drift = self._total_u(cl) - u0
        self.assertAlmostEqual(
            drift, 0.0, delta=1e-10 * abs(u0), msg=f"uniform 2-step drift={drift:.3e}"
        )

    def test_coarse_fine_two_step(self):
        """Narrow split shell: Σu conserved after two steps."""
        alat, _ = build_adaptive_sphere(self.R, u_origin=0.0)
        cl = alat[0]
        self._split_shell(alat, cl, r_centre=4.0, width=1.0)
        self._gaussian_ic(cl)
        u0 = self._total_u(cl)
        dt = self._dt(cl)
        alat.step(0, dt)
        alat.step(0, dt)
        drift = self._total_u(cl) - u0
        self.assertAlmostEqual(
            drift,
            0.0,
            delta=1e-10 * abs(u0),
            msg=f"coarse-fine 2-step drift={drift:.3e}",
        )

    def test_common_denominator_two_step(self):
        """Wide split shell: Σu conserved after two steps."""
        alat, _ = build_adaptive_sphere(self.R, u_origin=0.0)
        cl = alat[0]
        self._split_shell(alat, cl, r_centre=3.5, width=2.5)
        self._gaussian_ic(cl)
        u0 = self._total_u(cl)
        dt = self._dt(cl)
        alat.step(0, dt)
        alat.step(0, dt)
        drift = self._total_u(cl) - u0
        self.assertAlmostEqual(
            drift,
            0.0,
            delta=1e-10 * abs(u0),
            msg=f"common-denom 2-step drift={drift:.3e}",
        )

    def test_uniform_conservation(self):
        """Uniform lattice: total u is conserved to machine precision over 200 steps."""
        alat, _ = build_adaptive_sphere(self.R, u_origin=0.0)
        cl = alat[0]
        self._gaussian_ic(cl)
        u0 = self._total_u(cl)
        dt = self._dt(cl)
        for _ in range(200):
            alat.step(0, dt)
        drift_pct = 100.0 * (self._total_u(cl) - u0) / u0
        self.assertAlmostEqual(
            drift_pct, 0.0, delta=1e-6, msg=f"uniform 200-step drift={drift_pct:.3e}%"
        )

    def test_coarse_fine_conservation(self):
        """Narrow split shell: total u conserved to <0.001% over 200 steps."""
        alat, _ = build_adaptive_sphere(self.R, u_origin=0.0)
        cl = alat[0]
        self._split_shell(alat, cl, r_centre=4.0, width=1.0)
        self._gaussian_ic(cl)
        u0 = self._total_u(cl)
        dt = self._dt(cl)
        for _ in range(200):
            alat.step(0, dt)
        drift_pct = 100.0 * (self._total_u(cl) - u0) / u0
        self.assertAlmostEqual(
            drift_pct,
            0.0,
            delta=0.001,
            msg=f"coarse-fine 200-step drift={drift_pct:.3e}%",
        )

    def test_common_denominator_conservation(self):
        """Wide split shell: total u conserved to <0.001% over 200 steps."""
        alat, _ = build_adaptive_sphere(self.R, u_origin=0.0)
        cl = alat[0]
        self._split_shell(alat, cl, r_centre=3.5, width=2.5)
        self._gaussian_ic(cl)
        u0 = self._total_u(cl)
        dt = self._dt(cl)
        for _ in range(200):
            alat.step(0, dt)
        drift_pct = 100.0 * (self._total_u(cl) - u0) / u0
        self.assertAlmostEqual(
            drift_pct,
            0.0,
            delta=0.001,
            msg=f"common-denom 200-step drift={drift_pct:.3e}%",
        )


class TestStepCrossConservation(unittest.TestCase):
    """step_cross must conserve total phi.

    In the uniform path every inward-face crossing changes the local cell by
    +2·n_adj·delta and its neighbours by -2·n_adj·delta (mirror corrections),
    so Σphi is exact to machine precision.  The same must hold across AMR
    coarse/fine interfaces — a leaf whose nb[k] is a non-leaf proxy still
    represents physical face pairs whose corrections must balance.

    These tests pinned the 2026-07-08 bug behind the max_level>0 cross_scale
    runaway (m=4 lattice-locked "octupole", L1 vs L0 R=100 comparison):
    _step_cross_mirror resolved neighbours via g_to_j (leaves only), so
    mirror corrections into refinement proxies were silently dropped, and
    dead faces (a child's face toward its parent centre — invisible to the
    wave step) both fired on inert prolongation phi and swallowed
    redistribution boosts.  Fixed by _step_cross_mirror_proxy + _dead_face
    handling in adaptive_lattice.hpp.
    """

    R = 8.0
    DT = 0.01
    SCALE = 7e-3
    N_CALLS = 5

    def _build_ring(self):
        from tst.substrate.adaptive.adaptive_flux_tube import (
            find_toroidal_ring,
            init_toroidal_winding,
        )

        alat, _ = build_adaptive_sphere(self.R, u_origin=0.0)
        cl = alat[0]
        ring_js, ring_faces = find_toroidal_ring(cl, axis="z")
        self.assertIsNotNone(ring_js, "no toroidal ring found")
        init_toroidal_winding(cl, ring_js, ring_faces, amplitude=1.0, phi_only=True)
        return alat, cl, ring_js

    def _split_where(self, alat, cl, mask):
        """Split every masked leaf once via the scalar criterion path."""
        from equilibrium import lattice as _lat

        info = _lat.DSTInfo()
        info.split_hi = 0.5
        info.split_lo = 0.0
        info.max_level = 1
        cl.info = info
        cmax = np.full(cl.n_leaves, np.nan)
        cmax[mask] = 1.0
        cl.set_criterion_max(cmax)
        alat.adapt(0)
        alat.rebuild(0)

    def _total_phi(self, cl):
        return float(np.sum(np.array(cl.leaf_phi)))

    def _run_cross(self, alat, cl):
        s0 = self._total_phi(cl)
        for _ in range(self.N_CALLS):
            alat.step_cross(0, self.DT, self.SCALE)
        return self._total_phi(cl) - s0

    def test_uniform_sum_phi_conserved(self):
        """Uniform: Σphi drift is machine-zero after step_cross calls."""
        alat, cl, _ = self._build_ring()
        drift = self._run_cross(alat, cl)
        self.assertAlmostEqual(
            drift, 0.0, delta=1e-12, msg=f"uniform Σphi drift={drift:.3e}"
        )

    def test_interface_at_ring_sum_phi_conserved(self):
        """Coarse ring inside a refined ball: Σphi must not drift.

        Splits every cell within r<3 except the ring tiles, so the
        coarse-fine interface passes directly through the phi-carrying
        faces — the configuration the decay sweeps create when the absolute
        criterion refines around the (split-masked) ring.
        """
        alat, cl, ring_js = self._build_ring()
        r = np.sqrt((np.array(cl.leaf_pos) ** 2).sum(axis=1))
        mask = r < 3.0
        mask[ring_js] = False
        self._split_where(alat, cl, mask)
        drift = self._run_cross(alat, cl)
        self.assertAlmostEqual(
            drift, 0.0, delta=1e-9, msg=f"interface-at-ring Σphi drift={drift:.3e}"
        )

    def test_refined_ball_sum_phi_conserved(self):
        """Ring and surroundings all split: Σphi must not drift.

        The interface sits at r=3, away from the ring, but step_cross
        redistribution reaches it within a few calls.
        """
        alat, cl, _ = self._build_ring()
        r = np.sqrt((np.array(cl.leaf_pos) ** 2).sum(axis=1))
        self._split_where(alat, cl, r < 3.0)
        drift = self._run_cross(alat, cl)
        self.assertAlmostEqual(
            drift, 0.0, delta=1e-9, msg=f"refined-ball Σphi drift={drift:.3e}"
        )

    def _assert_leaf_pair_antisymmetry(self, cl, places=12, delta=None):
        """phi[k, j] == -phi[mirror_k, j_nb] for every same-level leaf-leaf pair."""
        from tst.substrate.adaptive.adaptive_flux_tube import _MIRROR

        phi = np.array(cl.leaf_phi)
        nb_g = np.array(cl.leaf_nb)
        lvl = np.array(cl.leaf_level)
        g_to_j = {int(g): j for j, g in enumerate(np.array(cl.leaf_global_indices))}
        checked = 0
        for j in range(cl.n_leaves):
            for k in range(14):
                g_nb = int(nb_g[j, k])
                if g_nb < 0:
                    continue
                j_nb = g_to_j.get(g_nb, -1)
                if j_nb < 0 or lvl[j_nb] != lvl[j]:
                    continue  # proxy or cross-level face — not a 1:1 pair
                mk = int(_MIRROR[k])
                kwargs = dict(delta=delta) if delta is not None else dict(places=places)
                self.assertAlmostEqual(
                    phi[k, j],
                    -phi[mk, j_nb],
                    msg=f"antisymmetry broken at leaf {j} face {k}",
                    **kwargs,
                )
                checked += 1
        self.assertGreater(checked, 0, "no leaf-leaf pairs checked")

    def test_uniform_antisymmetry_preserved(self):
        """Uniform: step_cross keeps every face pair antisymmetric."""
        alat, cl, _ = self._build_ring()
        self._run_cross(alat, cl)
        self._assert_leaf_pair_antisymmetry(cl)

    def test_coarse_fine_leaf_pair_antisymmetry_preserved(self):
        """Refined: step_cross keeps same-level leaf-leaf pairs antisymmetric,
        up to a small Odd27 coarse-proxy floor.

        Cross-level pairs are excluded (their balance is the Σphi tests'
        job). A coarse leaf has only 14 faces, but a coarse-proxy boundary
        can route more independent fine directions into it than that —
        some of a same-level pair's face can carry a small amount of
        unrelated cross-level junction flux aliased onto the same index.
        That crosstalk cannot be relocated to a different face without
        breaking test_junction_sum_antisymmetry_preserved's exact sum for
        whichever junction it gets moved onto instead — the two invariants
        compete for the same finite face budget. Observed floor (max over
        all same-level pairs, not just the first found): 1.52e-9.
        """
        alat, cl, ring_js = self._build_ring()
        r = np.sqrt((np.array(cl.leaf_pos) ** 2).sum(axis=1))
        mask = r < 3.0
        mask[ring_js] = False
        self._split_where(alat, cl, mask)
        self._run_cross(alat, cl)
        self._assert_leaf_pair_antisymmetry(cl, delta=2e-9)

    def _find_proxy_junction(self, cl, sq_only=False):
        """First (jQ, k, mk, desc_js): coarse leaf jQ whose face k neighbours
        a refinement proxy, plus the fine leaves whose face mk pairs with it."""
        from tst.substrate.adaptive.adaptive_flux_tube import _MIRROR

        nb_g = np.array(cl.leaf_nb)
        lvl = np.array(cl.leaf_level)
        gids = np.array(cl.leaf_global_indices)
        leaf_set = set(int(x) for x in gids)
        for j in np.flatnonzero(lvl == 0):
            for k in range(8, 14) if sq_only else range(14):
                g_nb = int(nb_g[j, k])
                if g_nb < 0 or g_nb in leaf_set:
                    continue
                mk = int(_MIRROR[k])
                desc = np.flatnonzero((lvl > 0) & (nb_g[:, mk] == int(gids[j])))
                if desc.size >= 2:
                    return int(j), k, mk, [int(d) for d in desc]
        self.fail("no coarse-proxy junction found")

    def test_proxy_mirror_locality(self):
        """A coarse-side crossing corrects only the fine faces that carry the
        pair phi — uninvolved sibling faces stay exactly untouched.

        Discriminates proportional (proper) mirroring from junction-sum
        balancing, which would smear the correction across all fine faces."""
        alat, _ = build_adaptive_sphere(self.R, u_origin=0.0)
        cl = alat[0]
        r = np.sqrt((np.array(cl.leaf_pos) ** 2).sum(axis=1))
        self._split_where(alat, cl, r < 3.0)
        jQ, k, mk, desc = self._find_proxy_junction(cl)

        phi = np.zeros_like(np.array(cl.leaf_phi))
        phi[k, jQ] = -1.0  # inward on the coarse face
        phi[mk, desc[0]] = 1.0  # entire fine-side pair phi on one face
        cl.set_leaf_phi(phi)
        s0 = float(np.sum(phi))
        alat.step_cross(0, self.DT, self.SCALE)
        phi_a = np.array(cl.leaf_phi)

        drained = phi_a[k, jQ] - (-1.0)  # n_adj*delta applied to the coarse face
        self.assertGreater(drained, 0.0, "coarse inward face was not drained")
        self.assertAlmostEqual(
            phi_a[mk, desc[0]],
            1.0 - drained,
            places=12,
            msg="loaded fine face must absorb the full mirror correction",
        )
        for jj in desc[1:]:
            self.assertEqual(
                phi_a[mk, jj],
                0.0,
                f"uninvolved fine face at leaf {jj} was modified",
            )
        drift = float(np.sum(phi_a)) - s0
        self.assertAlmostEqual(drift, 0.0, delta=1e-12, msg=f"Σphi drift={drift:.3e}")

    def test_junction_sum_antisymmetry_preserved(self):
        """phi_Q[k] + Σ phi_fine[mk] stays zero at every coarse-proxy junction
        (the sum convention _step_pass2_amr_correct maintains for the wave
        step), after step_cross activity reaches the interface."""
        from tst.substrate.adaptive.adaptive_flux_tube import _MIRROR

        alat, cl, ring_js = self._build_ring()
        r = np.sqrt((np.array(cl.leaf_pos) ** 2).sum(axis=1))
        mask = r < 3.0
        mask[ring_js] = False
        self._split_where(alat, cl, mask)
        self._run_cross(alat, cl)

        phi = np.array(cl.leaf_phi)
        nb_g = np.array(cl.leaf_nb)
        lvl = np.array(cl.leaf_level)
        gids = np.array(cl.leaf_global_indices)
        leaf_set = set(int(x) for x in gids)
        checked = active = 0
        for j in np.flatnonzero(lvl == 0):
            for k in range(14):
                g_nb = int(nb_g[j, k])
                if g_nb < 0 or g_nb in leaf_set:
                    continue
                mk = int(_MIRROR[k])
                desc = np.flatnonzero((lvl > 0) & (nb_g[:, mk] == int(gids[j])))
                if desc.size == 0:
                    continue
                s = float(phi[k, j] + phi[mk, desc].sum())
                self.assertAlmostEqual(
                    s,
                    0.0,
                    places=12,
                    msg=f"junction sum broken at coarse leaf {j} face {k}",
                )
                checked += 1
                if abs(phi[k, j]) > 0:
                    active += 1
        self.assertGreater(checked, 0, "no coarse-proxy junctions found")
        self.assertGreater(
            active, 0, "no junction carried phi — test exercised nothing"
        )


class TestStepCrossMultiLevelConservation(unittest.TestCase):
    """step_cross Σphi conservation across genuine multi-level (>1) jumps.

    adapt() enforces NO 2:1 neighbour balance (can_split only checks
    level < max_level; confirmed by reading adaptive_amr.hpp 2026-07-08) —
    so a level-0 leaf can directly border a level-2+ leaf whenever the
    split criterion refines a small region repeatedly while its immediate
    exterior stays unrefined.  TestStepCrossConservation only exercised
    single-level (L0<->L1) junctions; this class pins the SEPARATE bug that
    only appeared at max_level>=2 with a real level gap, found and fixed
    2026-07-08.

    Root cause: a face with nb[k] < 0 is invisible to the wave step
    (_step_pass1/2/3 all gate on nb[k] >= 0) — either a genuine domain edge
    or an AMR-inherited "dead" direction (wire_split_children sets exactly
    one child's face to -1 at every split, and every further split
    generation re-inherits that -1 unchanged into its non-adjacent
    descendants).  A cell refined N generations deep can carry up to N such
    inert faces, one contributed by each ancestor — but the original fix
    (_dead_face) only ever computed ONE (the immediate parent's), so a
    doubly-refined leaf's INHERITED dead face (from its grandparent) was
    still treated as live: step_cross drained/boosted it locally with no
    mirror partner (nb<0 has none), an unbalanced write with no possible
    offset — leaking Σphi independent of any hex/sq multiplicity question.
    Fixed by replacing _dead_face with a direct nb[k]<0 "live" check at
    every face, in both the local and mirror passes (adaptive_lattice.hpp)
    — matches the wave step's own liveness gate exactly, and generalizes to
    any refinement depth (verified up to a genuine L0<->L3 direct border).
    """

    R = 8.0
    DT = 0.01
    SCALE = 7e-3
    N_CALLS = 5

    def _split_two_rounds(self, alat, cl):
        """Round 1: split r<1.5 to level 1.  Round 2: split only the
        level-1 cells with r<1.0 to level 2 — their level-0 exterior
        neighbours are untouched, producing direct L0<->L2 borders."""
        from equilibrium import lattice as _lat

        info = _lat.DSTInfo()
        info.split_hi = 0.5
        info.split_lo = 0.0
        info.max_level = 2
        cl.info = info

        r = np.sqrt((np.array(cl.leaf_pos) ** 2).sum(axis=1))
        cmax = np.full(cl.n_leaves, np.nan)
        cmax[r < 1.5] = 1.0
        cl.set_criterion_max(cmax)
        alat.adapt(0)
        alat.rebuild(0)

        r = np.sqrt((np.array(cl.leaf_pos) ** 2).sum(axis=1))
        lvl = np.array(cl.leaf_level)
        cmax = np.full(cl.n_leaves, np.nan)
        cmax[(lvl == 1) & (r < 1.0)] = 1.0
        cl.set_criterion_max(cmax)
        alat.adapt(0)
        alat.rebuild(0)

    def test_two_level_jump_exists(self):
        """Sanity check that the split plan actually produces a direct
        level-0 <-> level-2 leaf-leaf border (else the drift test below
        would pass vacuously)."""
        from tst.substrate.adaptive.adaptive_flux_tube import (
            find_toroidal_ring,
            init_toroidal_winding,
        )

        alat, _ = build_adaptive_sphere(self.R, u_origin=0.0)
        cl = alat[0]
        ring_js, ring_faces = find_toroidal_ring(cl, axis="z")
        init_toroidal_winding(cl, ring_js, ring_faces, amplitude=1.0, phi_only=True)
        self._split_two_rounds(alat, cl)

        lvl = np.array(cl.leaf_level)
        gids = np.array(cl.leaf_global_indices)
        nb_g = np.array(cl.leaf_nb)
        lvl_by_g = {int(g): int(lv) for g, lv in zip(gids, lvl)}
        n_l0_l2 = 0
        for j in np.flatnonzero(lvl == 2):
            for k in range(14):
                g_nb = int(nb_g[j, k])
                if lvl_by_g.get(g_nb, -1) == 0:
                    n_l0_l2 += 1
        self.assertGreater(n_l0_l2, 0, "split plan produced no L0<->L2 border")

    def test_two_level_jump_sum_phi_conserved(self):
        """Σphi must not drift across a direct L0<->L2 junction (see class
        docstring for the inherited-inert-face bug this pins, fixed 2026-07-08)."""
        from tst.substrate.adaptive.adaptive_flux_tube import (
            find_toroidal_ring,
            init_toroidal_winding,
        )

        alat, _ = build_adaptive_sphere(self.R, u_origin=0.0)
        cl = alat[0]
        ring_js, ring_faces = find_toroidal_ring(cl, axis="z")
        init_toroidal_winding(cl, ring_js, ring_faces, amplitude=1.0, phi_only=True)
        self._split_two_rounds(alat, cl)

        s0 = float(np.sum(np.array(cl.leaf_phi)))
        for _ in range(self.N_CALLS):
            alat.step_cross(0, self.DT, self.SCALE)
        drift = float(np.sum(np.array(cl.leaf_phi))) - s0
        self.assertAlmostEqual(
            drift, 0.0, delta=1e-9, msg=f"L0<->L2 junction Σphi drift={drift:.3e}"
        )

    def test_three_level_jump_sum_phi_conserved(self):
        """Σphi must not drift across a genuine L0<->L3 direct junction —
        guards that the fix is a general nb[k]<0 liveness check (any number
        of inherited inert faces), not a fix specific to a single inherited
        generation."""
        from equilibrium import lattice as _lat
        from tst.substrate.adaptive.adaptive_flux_tube import (
            find_toroidal_ring,
            init_toroidal_winding,
        )

        alat, _ = build_adaptive_sphere(self.R, u_origin=0.0)
        cl = alat[0]
        ring_js, ring_faces = find_toroidal_ring(cl, axis="z")
        init_toroidal_winding(cl, ring_js, ring_faces, amplitude=1.0, phi_only=True)

        info = _lat.DSTInfo()
        info.split_hi = 0.5
        info.split_lo = 0.0
        info.max_level = 3
        cl.info = info

        # A shrinking-radius-threshold recipe (the old approach) only
        # produces GRADED shells (L0/L1, L1/L2, L2/L3, each a 1-level jump)
        # under Odd27's ratio-3 splitting -- confirmed directly, every
        # radius-threshold variant tried gave max_gap=1, never 3 (see
        # project_odd27_amr_junction_leak memory). It apparently relied on
        # geometric coincidence under the old ratio-2 split model that
        # doesn't hold here. Build the direct L0<->L3 border explicitly
        # instead: split ONE leaf 3 times in place, each round continuing
        # into whichever child still has a live neighbour OUTSIDE its own
        # lineage (an "outer" child) -- under any split model, an
        # exterior-facing child inherits its parent's own external
        # neighbour reference verbatim (wire_split_children), so chasing
        # the same outer branch through 3 generations leaves a genuine,
        # never-split L0 cell directly adjacent to the final L3 descendant.
        r = np.sqrt((np.array(cl.leaf_pos) ** 2).sum(axis=1))
        ring_set = set(int(x) for x in ring_js)
        candidates = [j for j in range(cl.n_leaves) if j not in ring_set]
        seed_j = min(candidates, key=lambda j: abs(r[j] - 2.5))
        seed_g = int(np.array(cl.leaf_global_indices)[seed_j])

        # Track lineage explicitly in Python rather than via leaf_ancestry
        # (its path field is packed 3 bits/level -- too narrow for Odd27's
        # 27 children, unsafe to decode here even though its root field,
        # unused, would be fine). At each round, the candidate pool must be
        # ONLY this round's freshly-created children (an accumulated-
        # descendant pool would keep re-finding an earlier round's already
        # "outer" sibling instead of progressing deeper).
        current_g = seed_g
        lineage = {seed_g}
        for _ in range(3):
            gids = np.array(cl.leaf_global_indices)
            gids_before = set(int(x) for x in gids)
            j = int(np.flatnonzero(gids == current_g)[0])
            cmax = np.full(cl.n_leaves, np.nan)
            cmax[j] = 1.0
            cl.set_criterion_max(cmax)
            alat.adapt(0)
            alat.rebuild(0)

            gids = np.array(cl.leaf_global_indices)
            new_idx = np.flatnonzero(
                np.array([int(g) not in gids_before for g in gids])
            )
            lineage |= set(int(x) for x in gids[new_idx])
            nb = np.array(cl.leaf_nb)
            outer = next(
                (
                    jj
                    for jj in new_idx
                    if any(
                        int(nb[jj, k]) >= 0 and int(nb[jj, k]) not in lineage
                        for k in range(14)
                    )
                ),
                None,
            )
            self.assertIsNotNone(
                outer, "no outer child with a live external neighbour found"
            )
            current_g = int(gids[outer])

        lvl = np.array(cl.leaf_level)
        gids = np.array(cl.leaf_global_indices)
        nb_g = np.array(cl.leaf_nb)
        lvl_by_g = {int(g): int(lv) for g, lv in zip(gids, lvl)}
        max_gap = max(
            (
                lvl[j] - lvl_by_g[int(nb_g[j, k])]
                for j in np.flatnonzero(lvl == 3)
                for k in range(14)
                if int(nb_g[j, k]) in lvl_by_g
            ),
            default=0,
        )
        self.assertEqual(
            max_gap, 3, "split plan did not produce a direct L0<->L3 border"
        )

        s0 = float(np.sum(np.array(cl.leaf_phi)))
        for _ in range(self.N_CALLS):
            alat.step_cross(0, self.DT, self.SCALE)
        drift = float(np.sum(np.array(cl.leaf_phi))) - s0
        self.assertAlmostEqual(
            drift, 0.0, delta=1e-9, msg=f"L0<->L3 junction Σphi drift={drift:.3e}"
        )


class TestByLevelStability(unittest.TestCase):
    """AdaptiveLatticeEvolver + TimeStepper.by_level on a refined mesh must not
    amplify energy — isolates the wave step from step_cross as a runaway
    source (the decay-sweep stepping loop minus sponge/cross/lam).

    RESOLVED (2026-07-09): a real bug was found and fixed along the way
    (step_subcycle's phantom lev=0 iteration double-applying the sibling-face
    correction — see _step_pass2_siblings's level_exact filter in
    adaptive_lattice.hpp), but it wasn't sufficient on its own: H still grew
    ~2.67x over 200 steps after that fix. Root-caused by direct
    experimentation (not just AMR/subcycling code reading) to a TEST
    CONSTRUCTION bug, not an engine defect: this test's refinement criterion
    (`cmax[r < 3.0] = 1.0`) refines everything inside r=3 — including the
    toroidal ring's own tiles, which sit at r~1.15-1.63. Splitting a ring
    tile breaks find_toroidal_ring's assumed single-leaf 8-tile connectivity
    (AdaptiveLattice::split() prolongs the parent's phi by dividing it evenly
    across every child that inherits the parent's outward neighbour pointer,
    not just the one child that would actually continue the true ring path),
    corrupting the ring's stationary winding pattern and causing it to
    radiate/relax — which is what showed up as H growth.
    sweep_spheromak_decay_cross_scale.py already guards against exactly this
    (masks its own ring tiles, A_js/B_js, to NaN before every adapt() call —
    see that script's run loop) — this test just wasn't doing the same.
    Fixed by masking `ring_js` to NaN in the criterion here too. Confirmed by
    direct A/B test: refining a shell that excludes the ring (2<r<4, or
    r<1.0) stays at machine precision; refining r<3 without the ring mask
    grows ~1.03x-3.3x depending on exact IC/steps; with the ring masked, H
    stays flat (ratio ~0.99) over the full 200-step run.
    """

    R = 8.0
    D1 = 1.0
    W2_SUM = 8 * 9.0 + 6 * 1.0
    N_STEPS = 200

    def _H(self, cl):
        u = np.array(cl.leaf_u)
        phi = np.array(cl.leaf_phi)
        return float(np.sum(u**2) + 0.5 * np.sum(phi**2))

    def test_by_level_no_runaway_without_cross(self):
        from equilibrium import lattice as _lat
        from tst.substrate.adaptive import refine_region_to_convergence
        from tst.substrate.adaptive.adaptive_flux_tube import (
            find_toroidal_ring,
            init_toroidal_winding,
        )

        alat, _ = build_adaptive_sphere(self.R, u_origin=0.0)
        cl = alat[0]
        ring_js, ring_faces = find_toroidal_ring(cl, axis="z")
        self.assertIsNotNone(ring_js, "no toroidal ring found")
        init_toroidal_winding(cl, ring_js, ring_faces, amplitude=1.0, phi_only=True)

        cl.info.split_hi = 0.5
        cl.info.split_lo = 0.0

        def _region(cl):
            r = np.sqrt((np.array(cl.leaf_pos) ** 2).sum(axis=1))
            return r < 3.0

        def _ring_js_now(cl):
            js, _ = find_toroidal_ring(cl, axis="z")
            return js

        refine_region_to_convergence(
            alat, cl, _region, stage_max_level=1, protected_js_fn=_ring_js_now
        )

        dt = 0.4 * self.D1 / self.W2_SUM  # L0 CFL; by_level subcycles fine cells
        ts = _lat.TimeStepper.by_level(dt)
        evolver = _lat.AdaptiveLatticeEvolver(alat)
        H0 = self._H(cl)
        for _ in range(self.N_STEPS):
            evolver.step(ts)
        Hf = self._H(cl)
        self.assertLess(
            Hf,
            2.0 * H0,
            f"by_level wave step amplified H at AMR interface: {H0:.3e} -> {Hf:.3e}",
        )


class TestByLevelMultiLevelStability(unittest.TestCase):
    """by_level (+ step_cross, + step_lam, + organic regrow) on a nested
    L0->L1->L2 cascade must not run per-cell fields away from the IC scale —
    extends TestByLevelStability's single-level (max_level=1) check to two
    levels, working toward a fast reproduction of the 2026-07-08
    spheromak-decay sweep runaway
    (run/substrate/adaptive/sweep_spheromak_decay_cross_scale.py, R=60,
    cs=7e-3): max_level=0/1 stayed at u<=0.09 / phi_rms<=1.2e-3 after 1000
    steps, but max_level=2 reached u=33.5 / phi_rms=0.74 — 3-4 orders of
    magnitude higher, concentrated in the level-2 core, with the
    characteristic alternating-sign ring structure replaced by a single
    runaway positive blob.

    RESOLVED (2026-07-09): the first three tests below (static two-level
    mesh, no further adapt() during stepping) briefly FAILED here (H grew
    ~8.3-8.7x) for the same reason root-caused via TestByLevelStability: the
    two-round mesh construction (r<3 -> level 1, then inner r<1.5 -> level
    2) refined the toroidal ring's own tiles instead of protecting them —
    worse than the single-level case since both rounds corrupted the ring's
    winding topology, compounding the leak. Fixed by masking the ring tiles
    (find_toroidal_ring, re-resolved every adapt round since leaf-j indices
    shift) out of both stages' refinement criteria — see
    TestByLevelStability's docstring and _build_two_level's use of
    refine_region_to_convergence (tst/substrate/adaptive/__init__.py) for
    the shared fix. test_two_level_no_runaway_full_pipeline_with_regrow
    (organic regrow, the closest match to the real sweep loop) passed even
    before this fix, since sweep_spheromak_decay_cross_scale.py-style
    organic regrow already NaN-masks its ring tiles (A_js/B_js) every round
    — this class's other three tests just weren't doing the same for their
    static two-level construction.
    """

    R = 8.0
    D1 = 1.0
    W2_SUM = 8 * 9.0 + 6 * 1.0
    N_STEPS = 300
    CROSS_SCALE = 7e-3
    LAM_SCALE = 0.05
    SPLIT_HI = 0.02

    def _H(self, cl):
        u = np.array(cl.leaf_u)
        phi = np.array(cl.leaf_phi)
        return float(np.sum(u**2) + 0.5 * np.sum(phi**2))

    def _build_two_level(self):
        from tst.substrate.adaptive import refine_region_to_convergence
        from tst.substrate.adaptive.adaptive_flux_tube import (
            find_toroidal_ring,
            init_toroidal_winding,
        )

        alat, _ = build_adaptive_sphere(self.R, u_origin=0.0)
        cl = alat[0]
        ring_js, ring_faces = find_toroidal_ring(cl, axis="z")
        self.assertIsNotNone(ring_js, "no toroidal ring found")
        init_toroidal_winding(cl, ring_js, ring_faces, amplitude=1.0, phi_only=True)

        cl.info.split_hi = 0.5
        cl.info.split_lo = 0.0

        # Ring tiles themselves must stay single leaves at every stage —
        # splitting one breaks find_toroidal_ring's assumed 8-tile
        # connectivity (same protection sweep_spheromak_decay_cross_scale.py
        # applies to its A_js/B_js ring tiles before every adapt() call).
        # Re-resolved fresh each round (inside refine_region_to_convergence)
        # since the ring tiles' own leaf-j indices shift as other cells split.
        def _ring_js_now(cl):
            js, _ = find_toroidal_ring(cl, axis="z")
            return js

        # Stage 1: shell r<3 -> level 1 (same split TestByLevelStability uses).
        def _region_lvl1(cl):
            r = np.sqrt((np.array(cl.leaf_pos) ** 2).sum(axis=1))
            return r < 3.0

        refine_region_to_convergence(
            alat, cl, _region_lvl1, stage_max_level=1, protected_js_fn=_ring_js_now
        )

        # Stage 2: only the inner level-1 core -> level 2, leaving a level-1
        # buffer shell between it and the level-0 exterior — the progressive
        # cascade the real sweep's criterion produces (not a direct L0<->L2
        # skip, which TestStepCrossMultiLevelConservation already covers).
        def _region_lvl2(cl):
            r = np.sqrt((np.array(cl.leaf_pos) ** 2).sum(axis=1))
            lvl = np.array(cl.leaf_level)
            return (lvl == 1) & (r < 1.5)

        refine_region_to_convergence(
            alat, cl, _region_lvl2, stage_max_level=2, protected_js_fn=_ring_js_now
        )

        lvl = np.array(cl.leaf_level)
        self.assertGreater(int(lvl.max()), 1, "split plan did not reach level 2")
        return alat, cl

    def test_two_level_no_runaway_without_cross(self):
        """Wave step alone at two nested AMR levels must not blow up H."""
        from equilibrium import lattice as _lat

        alat, cl = self._build_two_level()
        dt = 0.4 * self.D1 / self.W2_SUM
        ts = _lat.TimeStepper.by_level(dt)
        evolver = _lat.AdaptiveLatticeEvolver(alat)
        H0 = self._H(cl)
        for _ in range(self.N_STEPS):
            evolver.step(ts)
        Hf = self._H(cl)
        self.assertLess(
            Hf,
            2.0 * H0,
            f"by_level wave step amplified H at 2-level AMR interface: "
            f"{H0:.3e} -> {Hf:.3e}",
        )

    def test_two_level_no_runaway_with_cross(self):
        """Wave step + step_cross together (the real decay-sweep pairing) at
        two nested AMR levels must not blow up H — pins the runaway described
        in the class docstring."""
        from equilibrium import lattice as _lat

        alat, cl = self._build_two_level()
        dt = 0.4 * self.D1 / self.W2_SUM
        ts = _lat.TimeStepper.by_level(dt)
        evolver = _lat.AdaptiveLatticeEvolver(alat)
        H0 = self._H(cl)
        for _ in range(self.N_STEPS):
            evolver.step(ts)
            alat.step_cross(0, dt, self.CROSS_SCALE)
        Hf = self._H(cl)
        self.assertLess(
            Hf,
            2.0 * H0,
            f"wave+cross amplified H at 2-level AMR interface: {H0:.3e} -> {Hf:.3e}",
        )

    def test_two_level_no_runaway_with_cross_and_lam(self):
        """Wave step + step_cross + dynamic step_lam (lam_mode="dynamic" is
        the sweep script's default) on a static two-level mesh must not blow
        up H or lam. The R=60 blowup snapshot showed lam itself running away
        (max 5.40 vs ~1.0 in the L0/L1 baselines) alongside u/phi — this
        isolates whether lam feedback alone (no organic regrow) reproduces
        it."""
        from equilibrium import lattice as _lat

        alat, cl = self._build_two_level()
        dt = 0.4 * self.D1 / self.W2_SUM
        ts = _lat.TimeStepper.by_level(dt)
        evolver = _lat.AdaptiveLatticeEvolver(alat)
        H0 = self._H(cl)
        lam0_max = float(np.max(np.array(cl.leaf_lam)))
        for _ in range(self.N_STEPS):
            evolver.step(ts)
            alat.step_cross(0, dt, self.CROSS_SCALE)
            alat.step_lam(0, dt, self.LAM_SCALE)
        Hf = self._H(cl)
        lamf_max = float(np.max(np.array(cl.leaf_lam)))
        self.assertLess(
            Hf,
            2.0 * H0,
            f"wave+cross+lam amplified H at 2-level AMR interface: "
            f"{H0:.3e} -> {Hf:.3e}",
        )
        self.assertLess(
            lamf_max,
            2.0 * max(lam0_max, 1.0),
            f"wave+cross+lam ran away lam at 2-level AMR interface: "
            f"{lam0_max:.3e} -> {lamf_max:.3e}",
        )

    def test_two_level_no_runaway_full_pipeline_with_regrow(self):
        """Closest reproduction of the real sweep loop: organic
        SplitCriterion.absolute-driven refinement (no manual pre-split),
        periodic re-adapt() via GrowingAMRSchedule, wave step, step_cross,
        and dynamic step_lam every step — per-cell field magnitudes (u, phi,
        lam) must stay near the IC's O(1) scale, and must actually reach
        level 2 within the step budget (else the test exercises nothing).

        Deliberately checks per-cell max/rms, NOT the raw extensive H = Σu² +
        0.5Σphi² used elsewhere in this file: split_lo=0 here (matching the
        real bug's config) disables coarsening, so n_leaves grows ~10x over
        1000 steps as refinement spreads and never retreats. Split's
        constant-copy prolongation multiplies a region's contribution to an
        unweighted sum by n_children every time it splits, so raw H inflates
        from mesh growth alone even when every individual cell is decaying
        normally — confirmed by hand: at this R/step budget, u_max stays
        <=0.14 and phi_rms *decreases* monotonically even though raw H rises
        ~100x. An extensive per-leaf sum is the wrong invariant once
        coarsening is disabled; use intensive (per-cell) quantities instead.
        """
        from equilibrium import lattice as _lat
        from tst.substrate.adaptive.adaptive_flux_tube import (
            find_toroidal_ring,
            init_toroidal_winding,
        )

        alat, _ = build_adaptive_sphere(self.R, u_origin=0.0)
        cl = alat[0]
        ring_js, ring_faces = find_toroidal_ring(cl, axis="z")
        self.assertIsNotNone(ring_js, "no toroidal ring found")
        init_toroidal_winding(cl, ring_js, ring_faces, amplitude=1.0, phi_only=True)
        lam = np.array(cl.leaf_lam)
        lam[ring_js] = 1.0
        cl.set_leaf_lam(lam)

        info = _lat.DSTInfo()
        info.max_level = 2
        info.split_interval = 1
        cl.info = info
        cl.set_split_criterion(_lat.SplitCriterion.absolute(hi=self.SPLIT_HI, lo=0.0))

        dt = 0.4 * self.D1 / self.W2_SUM
        ts = _lat.TimeStepper.by_level(dt)
        evolver = _lat.AdaptiveLatticeEvolver(alat)
        schedule = GrowingAMRSchedule(start_interval=10, max_interval=100)

        n_steps = 1000
        AMPLITUDE = 1.0
        for step in range(1, n_steps + 1):
            evolver.step(ts)
            alat.step_cross(0, dt, self.CROSS_SCALE)
            alat.step_lam(0, dt, self.LAM_SCALE)
            if schedule.due(step):
                alat.adapt(0)
        lvl_max = int(np.max(np.array(cl.leaf_level)))
        u = np.array(cl.leaf_u)
        phi = np.array(cl.leaf_phi)
        lamf_max = float(np.max(np.array(cl.leaf_lam)))
        u_max = float(np.max(np.abs(u)))
        phi_max = float(np.max(np.abs(phi)))

        self.assertGreaterEqual(
            lvl_max, 2, "organic regrow did not reach level 2 — test exercised nothing"
        )
        self.assertLess(
            u_max,
            2.0 * AMPLITUDE,
            f"full pipeline (organic regrow) ran away u: max={u_max:.3e} "
            f"(lvl_max={lvl_max})",
        )
        self.assertLess(
            phi_max,
            2.0 * AMPLITUDE,
            f"full pipeline (organic regrow) ran away phi: max={phi_max:.3e} "
            f"(lvl_max={lvl_max})",
        )
        self.assertLess(
            lamf_max,
            2.0,
            f"full pipeline (organic regrow) ran away lam: max={lamf_max:.3e} "
            f"(lvl_max={lvl_max})",
        )


class TestSpheromakDecayMultiLevelBisection(unittest.TestCase):
    """Same R=30 production pipeline as TestSpheromakDecayMultiLevelRunaway
    (reuses _build_and_run directly), adding one physics step at a time to
    find which ingredient first introduces the runaway:

        1. wave step + organic AMR regrow only (cross_scale=0, lam frozen)
        2. + step_cross                        (lam still frozen)
        3. + step_lam                          == TestSpheromakDecayMultiLevelRunaway

    Each stage compares max_level=2 against a max_level=0 baseline run with
    the IDENTICAL cross/lam settings (AMR is the only variable), so a
    passing stage means that ingredient combination is NOT sufficient to
    trigger the runaway on its own — the first stage that fails is where it
    is introduced.

    RESOLVED (2026-07-09): the runaway was step_cross's interaction with the
    nested L0/L1/L2 mesh under organic regrow, isolated to two independent
    bugs, both now fixed: (1) step_cross's per-crossing delta wasn't CFL-safe
    at fine levels (adaptive_lattice.hpp _step_cross_local/_step_cross_mirror,
    delta now scales by cell.L); (2) AdaptiveLattice::split() copied a
    parent's whole phi vector verbatim into every child instead of
    dividing it across the live fan-out, inflating Σ|phi| by up to nc-1
    (=7x) every time an already-active region got refined
    (bcc_geometry.hpp wire_split_children, Step 5). All three stages pass now.
    """

    R = 30.0
    N_STEPS = 1000
    BASE_KW = dict(
        amplitude=1.0,
        lam_rings=1.0,
        lam_scale=0.05,
        axis="z",
        sponge=0.15,
        damp_max=0.003,
        report=200,
        criterion="absolute",
        split_hi=0.02,
        split_lo=0.0,
        split_ref=1e-6,
        amr_interval=1000,
        amr_growing=True,
        amr_growing_start=10,
        n_threads=_N_THREADS,
    )

    def _run(self, max_level, cross_scale, lam_mode):
        from run.substrate.adaptive.sweep_spheromak_decay_cross_scale import (
            _build_and_run,
        )

        kw = dict(self.BASE_KW, cross_scale=cross_scale, lam_mode=lam_mode)
        return _build_and_run(R=self.R, n_steps=self.N_STEPS, max_level=max_level, **kw)

    def _assert_not_much_worse(self, r0, r2, tag):
        phi_ratio = r2["phi_rms"][-1] / max(r0["phi_rms"][-1], 1e-300)
        self.assertLess(
            phi_ratio,
            5.0,
            f"{tag}: max_level=2 phi_rms diverged from max_level=0 baseline: "
            f"L0={r0['phi_rms'][-1]:.3e} L2={r2['phi_rms'][-1]:.3e} "
            f"ratio={phi_ratio:.2f} (n_leaves={r2['n_leaves']}, "
            f"level_max={r2['level_max']})",
        )
        for k in ("lam_A", "lam_B"):
            lam0, lam2 = r0[k][-1], r2[k][-1]
            self.assertLess(
                lam2,
                2.0 * max(lam0, 1.0),
                f"{tag}: max_level=2 {k} ran away vs max_level=0 baseline: "
                f"L0={lam0:.4f} L2={lam2:.4f}",
            )

    def test_1_wave_and_regrow_only(self):
        """Wave step + organic 2-level regrow, no step_cross, lam frozen."""
        r0 = self._run(max_level=0, cross_scale=0.0, lam_mode="frozen")
        r2 = self._run(max_level=2, cross_scale=0.0, lam_mode="frozen")
        self._assert_not_much_worse(r0, r2, "wave+regrow only")

    def test_2_with_cross_lam_frozen(self):
        """Wave step + regrow + step_cross, lam still frozen."""
        r0 = self._run(max_level=0, cross_scale=7e-3, lam_mode="frozen")
        r2 = self._run(max_level=2, cross_scale=7e-3, lam_mode="frozen")
        self._assert_not_much_worse(r0, r2, "wave+regrow+cross, lam frozen")


class TestSpheromakDecayMultiLevelRunaway(unittest.TestCase):
    """Full production spheromak-decay pipeline — reuses
    sweep_spheromak_decay_cross_scale._build_and_run directly (the exact
    code path the real sweep script runs: toroidal winding on two rings,
    sponge BC, dynamic step_lam, organic SplitCriterion.absolute-driven
    regrow via GrowingAMRSchedule) at R=30, the smallest/fastest scale found
    to reproduce the 2026-07-08 R=60 sweep runaway (TestByLevelMultiLevelStability's
    R=8 scaffolding does NOT reproduce it — see that class's docstring).

    RESOLVED (2026-07-09): originally phi_rms grew ~850x, H ~880,000x,
    lam_A/B >3x vs the max_level=0 baseline over 1000 steps, while leaf count
    grew only ~12% — a genuine per-cell blowup. Two independent bugs, both
    now fixed (see TestSpheromakDecayMultiLevelBisection's docstring for the
    bisection that separated them): step_cross's per-crossing CFL scaling
    (cell.L in adaptive_lattice.hpp) and AdaptiveLattice::split()'s
    unscaled phi duplication into children (bcc_geometry.hpp
    wire_split_children, Step 5).
    """

    R = 30.0
    N_STEPS = 1000
    KW = dict(
        cross_scale=7e-3,
        amplitude=1.0,
        lam_rings=1.0,
        lam_scale=0.05,
        lam_mode="dynamic",
        axis="z",
        sponge=0.15,
        damp_max=0.003,
        report=200,
        criterion="absolute",
        split_hi=0.02,
        split_lo=0.0,
        split_ref=1e-6,
        amr_interval=1000,
        amr_growing=True,
        amr_growing_start=10,
        n_threads=_N_THREADS,
    )

    def _run(self, max_level):
        from run.substrate.adaptive.sweep_spheromak_decay_cross_scale import (
            _build_and_run,
        )

        return _build_and_run(
            R=self.R, n_steps=self.N_STEPS, max_level=max_level, **self.KW
        )

    def test_max_level_2_not_much_worse_than_flat(self):
        """max_level=2 phi_rms/lam must stay within a small multiple of the
        max_level=0 baseline at run's end."""
        r0 = self._run(max_level=0)
        r2 = self._run(max_level=2)

        phi_ratio = r2["phi_rms"][-1] / max(r0["phi_rms"][-1], 1e-300)
        self.assertLess(
            phi_ratio,
            5.0,
            f"max_level=2 phi_rms diverged from max_level=0 baseline at R={self.R}: "
            f"L0={r0['phi_rms'][-1]:.3e} L2={r2['phi_rms'][-1]:.3e} "
            f"ratio={phi_ratio:.2f} (n_leaves={r2['n_leaves']}, "
            f"level_max={r2['level_max']})",
        )
        for tag in ("lam_A", "lam_B"):
            lam0, lam2 = r0[tag][-1], r2[tag][-1]
            self.assertLess(
                lam2,
                2.0 * max(lam0, 1.0),
                f"max_level=2 {tag} ran away vs max_level=0 baseline: "
                f"L0={lam0:.4f} L2={lam2:.4f}",
            )


class TestStepCrossUniformVsJunctionMismatch(unittest.TestCase):
    """Isolates whether the residual step_cross runaway that survives the
    cell.L dt-scaling fix (TestSpheromakDecayMultiLevelBisection /
    TestSpheromakDecayMultiLevelRunaway) is an inherent per-cell level-2
    issue, or specifically a coarse-fine NEIGHBOUR-MISMATCH issue:

        uniform  — every leaf split to level 2, no coarse-fine boundary
                   exists anywhere in the domain
        partial  — only the centre refines (level-0 exterior / level-1
                   buffer / level-2 core), genuine L0<->L1<->L2 junctions,
                   same construction as TestByLevelMultiLevelStability

    Both are compared to the same flat max_level=0 baseline. If uniform
    passes and partial fails, the bug is in how a genuine coarse-fine
    junction is handled, not in the per-cell dt/L rate itself.

    RESOLVED (2026-07-09): uniform passed cleanly throughout (no coarse-fine
    junction ever exists there, so the bug below never fires); partial
    failed, confirming the residual was specifically junction-related. Root
    cause: AdaptiveLattice::split() (adaptive_lattice.hpp) copied a parent's
    whole phi vector verbatim into every child instead of dividing it across
    the live fan-out (7 of 8 children for a hex face, 4 of 8 for a sq face —
    see bcc_geometry.hpp wire_split_children Step 5) — invisible to Σphi
    (signed) conservation checks since that sum stays ~0 for an antisymmetric
    field regardless of duplication, but it inflated Σ|phi| by up to 7x every
    time an already-active region (e.g. the ring) got refined. Fixed by
    scaling each child's inherited phi by its actual fan-out at split time.
    """

    R = 8.0
    D1 = 1.0
    W2_SUM = 8 * 9.0 + 6 * 1.0
    N_STEPS = 500
    CROSS_SCALE = 7e-3
    N_THREADS = _N_THREADS

    def _dt(self):
        return 0.4 * self.D1 / self.W2_SUM

    def _build_ring(self, R=None):
        from tst.substrate.adaptive.adaptive_flux_tube import (
            find_toroidal_ring,
            init_toroidal_winding,
        )

        alat, _ = build_adaptive_sphere(R if R is not None else self.R, u_origin=0.0)
        cl = alat[0]
        ring_js, ring_faces = find_toroidal_ring(cl, axis="z")
        self.assertIsNotNone(ring_js, "no toroidal ring found")
        init_toroidal_winding(cl, ring_js, ring_faces, amplitude=1.0, phi_only=True)
        return alat, cl

    def _split_uniform_to_level(self, alat, cl, target_level):
        """Split every leaf, unconditionally, target_level times — every
        leaf ends at the same level, so no coarse-fine boundary exists."""
        from equilibrium import lattice as _lat

        info = _lat.DSTInfo()
        info.split_hi = 0.5
        info.split_lo = 0.0
        info.max_level = target_level
        cl.info = info
        for _ in range(target_level):
            cl.set_criterion_max(np.full(cl.n_leaves, 1.0))
            alat.adapt(0)
            alat.rebuild(0)
        lvl = np.array(cl.leaf_level)
        self.assertTrue(
            np.all(lvl == target_level),
            f"uniform split left mixed levels: {np.unique(lvl)}",
        )

    def _build_partial_junction(self, alat, cl):
        """Shell r<3 -> level 1, then its inner core r<1.5 -> level 2 —
        level-0 exterior stays untouched, producing genuine L0<->L1<->L2
        junctions (same construction as TestByLevelMultiLevelStability)."""
        from equilibrium import lattice as _lat

        info = _lat.DSTInfo()
        info.split_hi = 0.5
        info.split_lo = 0.0
        info.max_level = 2
        cl.info = info

        r = np.sqrt((np.array(cl.leaf_pos) ** 2).sum(axis=1))
        cmax = np.full(cl.n_leaves, np.nan)
        cmax[r < 3.0] = 1.0
        cl.set_criterion_max(cmax)
        alat.adapt(0)
        alat.rebuild(0)

        r = np.sqrt((np.array(cl.leaf_pos) ** 2).sum(axis=1))
        lvl = np.array(cl.leaf_level)
        cmax = np.full(cl.n_leaves, np.nan)
        cmax[(lvl == 1) & (r < 1.5)] = 1.0
        cl.set_criterion_max(cmax)
        alat.adapt(0)
        alat.rebuild(0)

        lvl = np.array(cl.leaf_level)
        self.assertGreater(int(lvl.max()), 1, "split plan did not reach level 2")

    def _run(self, alat, cl, n_steps=None):
        from equilibrium import lattice as _lat

        alat.set_n_threads(self.N_THREADS)
        dt = self._dt()
        ts = _lat.TimeStepper.by_level(dt)
        evolver = _lat.AdaptiveLatticeEvolver(alat)
        for _ in range(n_steps if n_steps is not None else self.N_STEPS):
            evolver.step(ts)
            alat.step_cross(0, dt, self.CROSS_SCALE)
        phi = np.array(cl.leaf_phi)
        return float(np.sqrt(np.mean(phi**2)))

    # Odd27 (27 children/split, current default) gives 27**2=729x growth for
    # a uniform 2-level split, vs the 8**2=64x this test's R=8.0/N_STEPS=500
    # were tuned for under the old Hex8 model (8 children/split) -- an
    # ~11.4x bigger mesh than intended. Use a smaller radius here (scoped to
    # this test only, NOT the shared self.R -- test_partial_refinement_
    # junction_mismatch's absolute radius thresholds r<3.0/r<1.5 depend on
    # self.R staying at 8.0) to restore the original total-cell-count scale.
    R_UNIFORM = 8.0 / (27**2 / 8**2) ** (1 / 3)

    def test_uniform_level2_no_junction_stable(self):
        """Every leaf at level 2, no coarse-fine boundary anywhere:
        step_cross must not diverge from the flat L0 baseline."""
        alat0, cl0 = self._build_ring(R=self.R_UNIFORM)
        phi_rms_0 = self._run(alat0, cl0)

        alat2, cl2 = self._build_ring(R=self.R_UNIFORM)
        self._split_uniform_to_level(alat2, cl2, target_level=2)
        phi_rms_2 = self._run(alat2, cl2)

        ratio = phi_rms_2 / max(phi_rms_0, 1e-300)
        self.assertLess(
            ratio,
            5.0,
            f"uniform level-2 (no junctions) phi_rms diverged from flat L0: "
            f"L0={phi_rms_0:.3e} uniformL2={phi_rms_2:.3e} ratio={ratio:.2f}",
        )

    def test_partial_refinement_junction_mismatch(self):
        """Only the centre refines to level 2 (genuine L0<->L1<->L2
        junctions) — compares against the same flat L0 baseline.

        Uses more steps than the uniform test (1500 vs 500): the R=30
        production-pipeline diagnostic showed this residual divergence is
        slow (ratio ~3.4x by step 500, only crossing the 5x threshold around
        step ~1000), so 500 steps alone would pass vacuously here too. The
        partial-junction mesh stays small (only the centre refines), so the
        extra steps are cheap — unlike the uniform test's ~170k-leaf mesh.
        """
        alat0, cl0 = self._build_ring()
        phi_rms_0 = self._run(alat0, cl0, n_steps=1500)

        alat2, cl2 = self._build_ring()
        self._build_partial_junction(alat2, cl2)
        phi_rms_2 = self._run(alat2, cl2, n_steps=1500)

        ratio = phi_rms_2 / max(phi_rms_0, 1e-300)
        self.assertLess(
            ratio,
            5.0,
            f"partial (junction) level-2 phi_rms diverged from flat L0: "
            f"L0={phi_rms_0:.3e} partialL2={phi_rms_2:.3e} ratio={ratio:.2f}",
        )


class TestGrowingAMRSchedule(unittest.TestCase):
    """GrowingAMRSchedule.due() cadence."""

    def test_doubling_sequence(self):
        s = GrowingAMRSchedule(start_interval=10, max_interval=1000)
        due_steps = [step for step in range(1, 1300) if s.due(step)]
        self.assertEqual(due_steps, [10, 30, 70, 150, 310, 630, 1270])

    def test_caps_at_max_interval(self):
        s = GrowingAMRSchedule(start_interval=10, max_interval=1000)
        due_steps = [step for step in range(1, 5000) if s.due(step)]
        gaps = [b - a for a, b in zip(due_steps, due_steps[1:])]
        self.assertTrue(gaps, "schedule never fired twice in range")
        self.assertLessEqual(max(gaps), 1000, "interval exceeded max_interval")
        self.assertEqual(gaps[-1], 1000, "interval did not converge to max_interval")

    def test_first_adapt_at_start_interval(self):
        s = GrowingAMRSchedule(start_interval=25, max_interval=1000)
        for step in range(1, 25):
            self.assertFalse(s.due(step), f"fired early at step {step}")
        self.assertTrue(s.due(25))

    def test_rejects_invalid_bounds(self):
        with self.assertRaises(ValueError):
            GrowingAMRSchedule(start_interval=0)
        with self.assertRaises(ValueError):
            GrowingAMRSchedule(start_interval=100, max_interval=10)


if __name__ == "__main__":
    unittest.main()
