"""
Compare AdaptiveLattice (C++ extension) with SubstrateLattice (Python reference)
for a uniform BCC sphere with the same initial conditions.

With info.max_level=0 (flat lattice) and the same wave-equation physics
the two implementations must agree to machine precision after N steps.

Also includes TestHarrisPHAREUniformVsAdaptive: builds the same periodic Harris
sheet box in both lattices with the PHARE-matched IC and verifies that phi_E,
u_max, and u_vol agree to machine precision after N steps.  This validates that:
  1. KelvinLattice sq_offsets (-x,+x,-y,+y,-z,+z) match the adaptive convention.
  2. wire_periodic_bcc and the uniform slab builder wire equivalent topologies.
  3. The wave equation step is identical in both paths.
"""

import os
import unittest

import numpy as np

from equilibrium import lattice as _alat_mod
from equilibrium.substrate.adaptive import wire_periodic_bcc
from equilibrium.substrate.lattice import _K_REV, SubstrateLattice, SubstrateTile
from tst.substrate import build_substrate_sphere
from tst.substrate.adaptive import build_adaptive_sphere


class TestAdaptiveVsSubstrate(unittest.TestCase):

    R = 3.0
    N_STEPS = 50
    # stable_dt ≈ 0.4 / (sigma * sum(W^2)) = 0.4/78 ≈ 0.005; use 0.003
    DT = 0.003

    def setUp(self):
        self.sl, self.sl_origin = build_substrate_sphere(self.R)
        self.alat, self.al_origin = build_adaptive_sphere(self.R)

    def test_same_cell_count(self):
        """Both builders must produce the same number of cells for radius R."""
        self.assertEqual(self.sl.N, self.alat[0].n_leaves)

    def test_u_matches_after_steps(self):
        """u arrays must agree to machine precision after N steps."""
        dt = self.DT
        n = self.N_STEPS
        cl = self.alat[0]

        for _ in range(n):
            self.sl.step(dt)
        for _ in range(n):
            self.alat.step(0, dt)

        # Substrate u: already a contiguous array
        u_sub = self.sl.u.copy()

        # Adaptive u: packed leaf array
        u_adp = np.array(cl.leaf_u)

        # Both should have the same shape
        self.assertEqual(u_sub.shape, u_adp.shape)

        # Cell ordering may differ between the two builders, so compare sorted
        # radial profiles rather than element-by-element.
        r_sub = np.sqrt((self.sl.pos**2).sum(axis=1))
        r_adp = np.sqrt((cl.leaf_pos**2).sum(axis=1))

        # Sort by radius then by u (to handle degeneracy at same r)
        idx_sub = np.lexsort((u_sub, r_sub))
        idx_adp = np.lexsort((u_adp, r_adp))

        np.testing.assert_allclose(
            u_sub[idx_sub],
            u_adp[idx_adp],
            atol=1e-12,
            rtol=0,
            err_msg="u profile mismatch between substrate and adaptive lattice",
        )

    def test_origin_u_decreases(self):
        """Origin u should decrease from 1.0 as the wave spreads (wave equation)."""
        dt = self.DT
        cl = self.alat[0]
        for _ in range(self.N_STEPS):
            self.alat.step(0, dt)
        u_origin = float(cl.leaf_u[self.al_origin])
        self.assertLess(u_origin, 1.0, "origin u should decrease from 1.0")

    def test_phi_nonzero_after_steps(self):
        """phi should become non-zero after stepping (wave field builds up)."""
        dt = self.DT
        cl = self.alat[0]
        for _ in range(self.N_STEPS):
            self.alat.step(0, dt)
        phi = np.array(cl.leaf_phi)  # shape (14, N)
        self.assertGreater(np.abs(phi).max(), 0.0, "phi should be non-zero after steps")


class TestAdaptiveWithSplitting(unittest.TestCase):
    """Splitting tests: verify leaf-count bookkeeping, u inheritance,
    criterion NaN lifecycle, and threshold-based loss function behaviour."""

    R = 5.0
    DT = 0.003
    N_WARM = 30  # steps before splitting to populate criterion_max
    N_POST = 30  # steps after splitting

    def setUp(self):
        self.alat, self.origin_leaf = build_adaptive_sphere(self.R, max_level=2)

    # ------------------------------------------------------------------
    # structural / bookkeeping

    def test_split_increases_leaf_count_by_twenty_six(self):
        """One Odd27 split adds 27 children; the parent stops being a leaf → net +26."""
        cl = self.alat[0]
        for _ in range(self.N_WARM):
            self.alat.step(0, self.DT)
        n_before = cl.n_leaves
        self.assertTrue(self.alat.can_split(0, self.origin_leaf))
        self.alat.split(0, self.origin_leaf)
        self.alat.rebuild(0)
        self.assertEqual(cl.n_leaves, n_before + 26)

    def test_split_children_inherit_u(self):
        """Children must receive the parent's u at the moment of splitting (constant prolongation)."""
        cl = self.alat[0]
        for _ in range(self.N_WARM):
            self.alat.step(0, self.DT)
        u_parent = float(np.array(cl.leaf_u)[self.origin_leaf])
        self.alat.split(0, self.origin_leaf)
        self.alat.rebuild(0)
        pos = cl.leaf_pos
        r2 = (pos**2).sum(axis=1)
        # 27 Odd27 children sit at r² in {0, 1/9, 4/27, 8/27} (center/corner/face/edge)
        children_mask = r2 < 0.5
        self.assertEqual(children_mask.sum(), 27)
        u_children = np.array(cl.leaf_u)[children_mask]
        np.testing.assert_allclose(
            u_children,
            u_parent,
            atol=1e-14,
            err_msg="all 27 children must inherit the parent u exactly",
        )

    # ------------------------------------------------------------------
    # criterion NaN lifecycle

    def test_criterion_nan_for_new_children_after_split(self):
        """New children have NaN criterion_max immediately after split+rebuild (not yet stepped)."""
        cl = self.alat[0]
        for _ in range(self.N_WARM):
            self.alat.step(0, self.DT)
        self.alat.split(0, self.origin_leaf)
        self.alat.rebuild(0)
        cmax = np.array(cl.criterion_max)
        # 27 children added, parent stops being a leaf (net +26 in leaf_indices)
        self.assertEqual(
            np.isnan(cmax).sum(),
            26,
            "exactly 26 new-child leaves should have NaN criterion after rebuild",
        )

    def test_criterion_finite_after_post_split_steps(self):
        """After splitting and running more steps all criterion values become finite."""
        cl = self.alat[0]
        for _ in range(self.N_WARM):
            self.alat.step(0, self.DT)
        self.alat.split(0, self.origin_leaf)
        self.alat.rebuild(0)
        for _ in range(self.N_POST):
            self.alat.step(0, self.DT)
        cmax = np.array(cl.criterion_max)
        self.assertTrue(
            np.isfinite(cmax).all(),
            "all criterion_max values must be finite after post-split stepping",
        )

    # ------------------------------------------------------------------
    # loss / threshold function

    def test_criterion_max_highest_near_source(self):
        """criterion_max threshold selects cells near the source, not the quiet outer region."""
        cl = self.alat[0]
        for _ in range(self.N_WARM):
            self.alat.step(0, self.DT)
        cmax = np.array(cl.criterion_max)
        lpos = cl.leaf_pos
        r_all = np.sqrt((lpos**2).sum(axis=1))
        # Use the median as a data-driven threshold — half the cells are above it.
        threshold = float(np.median(cmax))
        above = cmax > threshold
        self.assertGreater(above.sum(), 0, "some cells must be above median threshold")
        # Cells above the threshold must be closer to the source on average.
        self.assertLess(
            r_all[above].mean(),
            r_all[~above].mean(),
            "above-threshold cells must be closer to the source than below-threshold cells",
        )

    def test_threshold_active_cells_remain_near_source_after_split(self):
        """After splitting above-threshold cells and stepping, the criterion still
        localises near the source — above-threshold leaves must have smaller mean
        radius than below-threshold leaves (loss function is spatially coherent)."""
        cl = self.alat[0]
        for _ in range(self.N_WARM):
            self.alat.step(0, self.DT)

        cmax_before = np.array(cl.criterion_max)
        # Use 90th percentile as split threshold — top 10 % are "active".
        threshold = float(np.percentile(cmax_before, 90))
        above_indices = list(np.where(cmax_before > threshold)[0])
        self.assertGreater(len(above_indices), 0, "need active cells to split")

        # Split every above-threshold leaf.
        for leaf_idx in above_indices:
            if self.alat.can_split(0, leaf_idx):
                self.alat.split(0, leaf_idx)
        self.alat.rebuild(0)

        # Step until all new children have finite criterion.
        for _ in range(self.N_POST):
            self.alat.step(0, self.DT)

        cmax_after = np.array(cl.criterion_max)
        lpos = cl.leaf_pos
        r_all = np.sqrt((lpos**2).sum(axis=1))

        # Finer cells near the source will have large absolute flux; cells in the
        # quiet outer region remain below threshold.  The median is a data-driven
        # threshold that adapts to the new resolution level.
        thr_post = float(np.median(cmax_after))
        above_post = cmax_after > thr_post
        self.assertGreater(above_post.sum(), 0)
        self.assertLess(
            r_all[above_post].mean(),
            r_all[~above_post].mean(),
            "above-threshold cells must remain closer to the source than below-threshold cells",
        )


class TestMultipleSplits(unittest.TestCase):
    """Recursive center-only split test: structural integrity, criterion
    lifecycle, and threshold-function coherence after many refinement
    rounds. max_level must reach N_SPLITS since each split descends one
    more level into the same chain (see `_run_bfs_splits`'s docstring).

    Default depth (N_SPLITS=11) is a fast, routine regression check.
    Override via EQ_TEST_DEEP_SPLITS=<n> for an occasional deeper validation
    run (e.g. 20) — a flat but properly CFL-sized dt (see `_cfl_dt`,
    recomputed every round from the current deepest level) keeps this cheap
    at any depth, unlike level-scaled subcycling whose cost grows as
    ~3^level per outer step."""

    R = 5.0
    N_WARM = 20
    N_BETWEEN = 5
    N_SPLITS = int(os.environ.get("EQ_TEST_DEEP_SPLITS", "11"))
    W2_SUM = 8 * 9.0 + 6 * 1.0

    def _cfl_dt(self, cl):
        """Flat dt sized to the CURRENT deepest leaf present — recompute
        after every split, since L_min shrinks by 3x each round. Matches
        TestFluxConservation's own _dt() formula exactly (2026-08-14: a
        fixed dt=0.003 blows up exponentially once L_min gets small enough,
        starting around split 5 — this scheme stays stable at any depth
        tested, including 20, at negligible per-round cost)."""
        lam = np.array(cl.leaf_lam, dtype=np.float64)
        lvl = np.array(cl.leaf_level)
        sigma_max = float(np.max(1.0 / (1.0 + lam)))
        L_min = 1.0 / (3.0 ** int(lvl.max()))
        return 0.4 * L_min / (sigma_max * self.W2_SUM)

    def setUp(self):
        self.alat, self.origin_g = build_adaptive_sphere(
            self.R, max_level=self.N_SPLITS + 2
        )
        self.n_initial = self.alat[0].n_leaves
        dt = self._cfl_dt(self.alat[0])
        for _ in range(self.N_WARM):
            self.alat.step(0, dt)

    def _run_bfs_splits(self, n_splits: int):
        """Recursive center-only split from origin: each round, split the
        current cell and descend into whichever of its 27 Odd27 children sits
        at the parent's own centre (offset (0,0,0) in Odd27Table).

        That center child's 14 faces are ALL internal (route to a sibling of
        the same split, per odd27_count_internal) -- it never gains a face
        neighbour more than one level away, so this maintains strict 2:1
        level balance by construction, unlike a sibling-fan-out BFS (which
        enqueues every splittable child, not just the center one): a
        surface-facing sibling later split produces a genuine >1-level-gap
        neighbour pointer, an out-of-spec topology this AMR's ancestor-
        substitution machinery only handles one level at a time (see `now`,
        2026-08-13 stale-proxy investigation). Uses the global index
        split() returns so leaf-array index drift after rebuild never
        causes a stale-index fault.

        Returns the number of splits actually performed (can stop early if
        max_level is reached).
        """
        g = self.origin_g
        n_done = 0
        for _ in range(n_splits):
            if not self.alat.can_split(0, g):
                break
            cl = self.alat[0]
            gidx = np.where(cl.leaf_global_indices == g)[0][0]
            parent_pos = cl.leaf_pos[gidx].copy()

            first_child_g = self.alat.split(0, g)
            cl = self.alat[0]
            child_gs = np.arange(first_child_g, first_child_g + 27)
            j_of = {int(cg): j for j, cg in enumerate(cl.leaf_global_indices)}
            d2 = np.array(
                [
                    np.sum((cl.leaf_pos[j_of[int(cg)]] - parent_pos) ** 2)
                    for cg in child_gs
                ]
            )
            center_g = int(child_gs[np.argmin(d2)])

            dt = self._cfl_dt(self.alat[0])
            for _ in range(self.N_BETWEEN):
                self.alat.step(0, dt)
            g = center_g
            n_done += 1
        return n_done

    def test_leaf_count_after_twenty_splits(self):
        """N_SPLITS Odd27 splits must each add exactly 26 leaves."""
        self._run_bfs_splits(self.N_SPLITS)
        cl = self.alat[0]
        self.assertEqual(
            cl.n_leaves - self.n_initial,
            26 * self.N_SPLITS,
            "each Odd27 split adds 27 children; the parent stops being a leaf (net +26)",
        )

    def test_no_nan_criterion_after_twenty_splits(self):
        """After N_SPLITS splits each followed by N_BETWEEN steps, all criterion
        values must be finite — no new child is left unstepped at test end."""
        self._run_bfs_splits(self.N_SPLITS)
        cmax = np.array(self.alat[0].criterion_max)
        self.assertTrue(
            np.isfinite(cmax).all(),
            f"criterion_max has {np.isnan(cmax).sum()} NaN entries after {self.N_SPLITS} splits",
        )

    def test_threshold_loss_localised_after_twenty_splits(self):
        """After N_SPLITS BFS splits from origin and interleaved stepping, the
        median criterion correctly partitions leaves: above-median leaves are
        closer to the source than below-median leaves — the loss/threshold
        function is spatially coherent across all refinement levels."""
        self._run_bfs_splits(self.N_SPLITS)
        cl = self.alat[0]
        cmax = np.array(cl.criterion_max)
        lpos = cl.leaf_pos
        r_all = np.sqrt((lpos**2).sum(axis=1))
        threshold = float(np.median(cmax))
        above = cmax > threshold
        self.assertGreater(above.sum(), 0)
        self.assertLess(
            r_all[above].mean(),
            r_all[~above].mean(),
            "above-median-criterion leaves must be closer to source than below-median leaves",
        )

    def test_ten_splits_leaf_count(self):
        """10-split variant: leaf count increases by exactly 260."""
        self._run_bfs_splits(10)
        cl = self.alat[0]
        self.assertEqual(cl.n_leaves - self.n_initial, 260)


_HEX_DIR = np.array(
    [
        [-1, -1, -1],
        [-1, -1, +1],
        [-1, +1, -1],
        [-1, +1, +1],
        [+1, -1, -1],
        [+1, -1, +1],
        [+1, +1, -1],
        [+1, +1, +1],
    ],
    dtype=np.float64,
)
_SQ_DIR = np.array(
    [
        [-1, 0, 0],
        [+1, 0, 0],
        [0, -1, 0],
        [0, +1, 0],
        [0, 0, -1],
        [0, 0, +1],
    ],
    dtype=np.float64,
)


def _uniform_from_adaptive(cl):
    """Build a SubstrateLattice that mirrors an AdaptiveCellCluster's topology.

    Uses the cluster's positions and leaf_nb neighbour array directly, so
    both lattices are guaranteed to have the same N, positions, and wiring.
    The adaptive leaf_nb stores global IDs; this converts them to leaf indices.
    """
    pos = cl.leaf_pos  # (N, 3)
    nb_a = cl.leaf_nb  # (N, 14) — global IDs, -1 = no neighbour
    gid = cl.leaf_global_indices  # (N,)
    g_to_j = {int(gid[j]): j for j in range(len(gid))}
    N = len(pos)

    lat = SubstrateLattice()
    lat.pos = pos.copy()
    lat.ell = np.ones(N, dtype=np.float64)
    lat.lam = np.zeros(N, dtype=np.float64)
    lat.u = np.zeros(N, dtype=np.float64)
    lat.phi = np.zeros((N, 14), dtype=np.float64)
    lat.fixed_u = np.zeros(N, dtype=bool)
    lat.fixed_phi = np.zeros((N, 14), dtype=bool)
    lat.tiles = [SubstrateTile(i, lat) for i in range(N)]

    for j in range(N):
        for k in range(14):
            g_nb = int(nb_a[j, k])
            if g_nb >= 0:
                leaf_nb = g_to_j.get(g_nb, -1)
                if leaf_nb >= 0:
                    lat.tiles[j].neighbours[k] = lat.tiles[leaf_nb]

    lat._dirty = True
    return lat


class TestStepCrossSignConvention(unittest.TestCase):
    """
    Regression test for a real behavioural discrepancy that used to exist between
    the uniform C++ step_cross (eq::substrate_step_cross, inc/eq/substrate.hpp)
    and the adaptive C++ step_cross (AdaptiveLattice::_step_cross_local/_mirror,
    inc/eq/core/data/lattice/adaptive_lattice.hpp).

    Both now restrict to INWARD phi (phi[i,k] < 0) only — each physical face
    crossing is processed once, from the receiving cell's perspective. Before
    the 2026-07-04 fix, the adaptive path processed phi of EITHER sign ("sign
    tracks phi"), which double-processed every antisymmetric face pair (once
    from each adjoining cell) — confirmed via a single-pair test predicting the
    exact 2x factor (phi[i,k]=-A, phi[nb_k,mirror_k]=+A, lam=0): adaptive drained
    the shared face by 2*n_adj*dt*scale*A vs uniform's n_adj*dt*scale*A. See
    doc/mems/project_step_cross.md for the full history.

    This test now asserts the FIXED (matching, single-processing) behaviour on
    both backends.
    """

    R = 3.0
    A = 1.0
    DT = 0.003
    SCALE = 0.01
    K = 0  # hex face (0-7), n_adj = 6
    N_ADJ = 6
    MIRROR = [
        7,
        6,
        5,
        4,
        3,
        2,
        1,
        0,
        9,
        8,
        11,
        10,
        13,
        12,
    ]  # same table, both conventions

    def setUp(self):
        self.alat, self.al_origin = build_adaptive_sphere(self.R)
        self.cl = self.alat[0]

    def _pick_pair(self):
        """Origin leaf + one of its hex neighbours, as (j_i, j_nb, mirror_k)."""
        nb_g = np.array(self.cl.leaf_nb)
        leaf_g = self.cl.leaf_global_indices
        g_to_j = {int(g): j for j, g in enumerate(leaf_g)}
        j_i = self.al_origin
        g_nb = int(nb_g[j_i, self.K])
        self.assertGreaterEqual(
            g_nb, 0, "origin must have a valid hex neighbour at R=3"
        )
        j_nb = g_to_j[g_nb]
        return j_i, j_nb, self.MIRROR[self.K]

    def test_adaptive_single_processes_shared_face(self):
        """Adaptive step_cross drains the shared face by the single-sided amount only."""
        j_i, j_nb, mk = self._pick_pair()
        phi = np.array(self.cl.leaf_phi)
        phi[self.K, j_i] = -self.A
        phi[mk, j_nb] = self.A
        self.cl.set_leaf_phi(phi)

        self.alat.step_cross(0, self.DT, self.SCALE)

        phi_after = float(np.array(self.cl.leaf_phi)[self.K, j_i])
        expected_single = -self.A + self.N_ADJ * self.DT * self.SCALE * self.A

        self.assertAlmostEqual(
            phi_after,
            expected_single,
            places=12,
            msg=f"adaptive step_cross: got {phi_after}, expected single-processed "
            f"{expected_single}",
        )

    def test_uniform_single_processes_shared_face(self):
        """Uniform step_cross drains the shared face by only the single-sided amount."""
        j_i, j_nb, mk = self._pick_pair()
        lat = _uniform_from_adaptive(self.cl)
        lat.phi[j_i, self.K] = -self.A
        lat.phi[j_nb, mk] = self.A

        lat.step_cross(self.DT, scale=self.SCALE)

        phi_after = float(lat.phi[j_i, self.K])
        expected_single = -self.A + self.N_ADJ * self.DT * self.SCALE * self.A

        self.assertAlmostEqual(
            phi_after,
            expected_single,
            places=12,
            msg=f"uniform step_cross: got {phi_after}, expected single-processed "
            f"{expected_single}",
        )


class TestStepCrossBarrierGate(unittest.TestCase):
    """
    AdaptiveLattice.step_cross(gate="barrier") (AdaptiveLattice::_step_cross_local
    /_step_cross_mirror, inc/eq/core/data/lattice/adaptive_lattice.hpp) must match
    SubstrateLattice.step_cross(gate="barrier") (eq::substrate_step_cross_barrier,
    inc/eq/substrate.hpp) bit-for-bit -- see doc/mems/project_step_cross_v2.md.

    Uses an all-periodic BCC box (build_bcc_box + wire_periodic_bcc), NOT a
    finite sphere: on a sphere, boundary cells hit a real, pre-existing,
    documented semantic difference between the two backends' v1 kernel
    (run/substrate/cross_v2_uniform.py's module docstring) -- adaptive's
    _step_cross_local/_step_cross_mirror count n_adj over LIVE adjacent faces
    only, while the uniform kernel's n_adj is the fixed per-face-type count
    (6 hex/4 sq) regardless of neighbour validity. That gap is orthogonal to
    the barrier gate and pre-dates it (confirmed present under gate="v1" too on
    a sphere fixture during this test's development); it's irrelevant here
    because a periodic box has every face live everywhere (has_ghosts=True
    resolves every nb[k]), so the two n_adj conventions coincide exactly --
    this is also the reason "we should continue to support periodic special
    cases" (2026-07-14): it's the only fixture that isolates the gate itself
    from that boundary-convention gap.

    _uniform_from_adaptive builds the uniform lattice directly from the
    adaptive cluster's own positions/nb wiring, so both sides share identical
    indexing -- no sorting/matching needed.
    """

    LX = 8.0
    LY = 8.0
    LZ = 3.5
    DT = 0.003
    SCALE = 0.01
    M0 = 1e-3

    def setUp(self):
        self.alat = _alat_mod.AdaptiveLattice()
        self.alat.add_cluster()
        self.cl = self.alat[0]
        H = 1.0 / np.sqrt(3.0)
        _alat_mod.build_bcc_box(
            self.cl, self.LX, self.LY, self.LZ, 1.0, u0=0.0, lam0=0.0
        )
        self.alat.rebuild(0)
        wire_periodic_bcc(self.cl, H, verbose=False)
        self.lat = _uniform_from_adaptive(self.cl)
        self.N = self.cl.n_leaves
        self.assertTrue(
            (self.lat.nb >= 0).all(), "periodic box must have no boundary faces"
        )

    def _set_random_phi(self, seed, scale=1.0):
        """Antisymmetric random phi -- matches the real invariant a wave-stepped
        field always satisfies (phi[i,k] == -phi[nb_k(i), mirror(k)]), which
        step_cross's single-processing convention (only inward phi fires)
        assumes."""
        nb = self.lat.nb
        valid = nb >= 0
        safe = np.where(valid, nb, 0)
        rng = np.random.default_rng(seed)
        A = scale * rng.normal(size=(self.N, 14))
        phi = np.where(valid, A - A[safe, _K_REV[None, :]], 0.0)
        self.lat.phi = phi.copy()
        self.cl.set_leaf_phi(np.ascontiguousarray(phi.T))  # cl expects (14, N)
        return phi

    def test_barrier_matches_uniform(self):
        """Random phi field: adaptive and uniform barrier-gate steps agree exactly."""
        self._set_random_phi(seed=3, scale=1.0)

        self.alat.step_cross(0, self.DT, self.SCALE, gate="barrier", m0=self.M0)
        self.lat.step_cross(self.DT, scale=self.SCALE, gate="barrier", m0=self.M0)

        phi_adp = np.array(self.cl.leaf_phi).T  # (N, 14)
        phi_sub = self.lat.phi
        np.testing.assert_allclose(
            phi_adp,
            phi_sub,
            atol=1e-12,
            rtol=0,
            err_msg="adaptive vs uniform barrier-gate step_cross mismatch",
        )

    def test_sub_threshold_never_fires(self):
        """Drive well below m0: phi must not move on either backend, any number of ticks."""
        phi0 = self._set_random_phi(seed=11, scale=1e-6)  # drive << M0
        for _ in range(500):
            self.alat.step_cross(0, self.DT, self.SCALE, gate="barrier", m0=self.M0)
            self.lat.step_cross(self.DT, scale=self.SCALE, gate="barrier", m0=self.M0)

        np.testing.assert_allclose(
            np.array(self.cl.leaf_phi).T,
            phi0,
            atol=1e-15,
            rtol=0,
            err_msg="adaptive barrier gate fired sub-threshold",
        )
        np.testing.assert_allclose(
            self.lat.phi,
            phi0,
            atol=1e-15,
            rtol=0,
            err_msg="uniform barrier gate fired sub-threshold",
        )


def _harris_ic(pos, Lx, Ly, B0=0.05, lam_w=3.0, u_bg=0.4, lam_bg=0.4, perturb=0.1):
    N = len(pos)
    y0 = Ly / 4.0

    def bx(y):
        return B0 * (-1.0 + np.tanh((y + y0) / lam_w) - np.tanh((y - y0) / lam_w))

    bx_eq = bx(pos[:, 1])
    u_init = u_bg + (B0**2 - bx_eq**2) / 2.0
    lam_init = np.full(N, lam_bg)
    phi_init = np.zeros((N, 14), dtype=np.float64)
    for k in range(14):
        fd = _HEX_DIR[k] if k < 8 else _SQ_DIR[k - 8]
        phi_init[:, k] += bx_eq * fd[0]
    if perturb > 0:
        kx = 2.0 * np.pi / Lx
        dB = perturb * B0
        for sign, y_null in ((+1, +y0), (-1, -y0)):
            y_r = pos[:, 1] - y_null
            env = np.exp(-(y_r**2) / lam_w**2)
            kx_x = kx * (pos[:, 0] + Lx / 2.0)
            bx_p = sign * (-2.0 * y_r / lam_w**2) * dB * np.cos(kx_x) * env
            by_p = sign * kx * dB * np.sin(kx_x) * env
            for k in range(14):
                fd = _HEX_DIR[k] if k < 8 else _SQ_DIR[k - 8]
                phi_init[:, k] += bx_p * fd[0] + by_p * fd[1]
    return u_init, lam_init, phi_init


class TestHarrisPHAREUniformVsAdaptive(unittest.TestCase):
    """
    Same PHARE Harris IC applied to both the uniform SubstrateLattice (Python)
    and the AdaptiveLattice (C++).  After N steps both must agree to machine
    precision on phi_E, u_max, and sorted u/phi arrays.

    Validates:
    - KelvinLattice.sq_offsets ordering matches adaptive convention
    - Periodic slab wiring matches wire_periodic_bcc
    - Wave equation step is numerically identical
    """

    LX = 12.0
    LY = 12.0
    LZ = 3.5
    DT = 0.003
    N_STEPS = 30
    ATOL = 1e-10

    def setUp(self):
        # Adaptive — build first; uniform mirrors its topology exactly
        self.alat = _alat_mod.AdaptiveLattice()
        self.alat.add_cluster()
        cl = self.alat[0]
        H = 1.0 / np.sqrt(3.0)
        _alat_mod.build_bcc_box(cl, self.LX, self.LY, self.LZ, 1.0, u0=0.0, lam0=0.4)
        self.alat.rebuild(0)
        wire_periodic_bcc(cl, H, verbose=False)

        pos_a = cl.leaf_pos
        u_ic, lam_ic, phi_ic = _harris_ic(pos_a, self.LX, self.LY)

        cl.set_leaf_u(u_ic)
        cl.set_leaf_lam(lam_ic)
        cl.set_leaf_phi(np.ascontiguousarray(phi_ic.T))  # C-contiguous (14,N)

        # Uniform — cloned from adaptive topology; same positions, same wiring
        self.ulat = _uniform_from_adaptive(cl)
        self.ulat.u = u_ic.copy()
        self.ulat.lam = lam_ic.copy()
        self.ulat.phi = phi_ic.copy()

    def test_same_cell_count(self):
        N_u = self.ulat.N
        N_a = self.alat[0].n_leaves
        self.assertEqual(N_u, N_a, f"uniform N={N_u} != adaptive N={N_a}")

    def test_phi_E_matches_after_steps(self):
        dt = self.DT
        ts = _alat_mod.TimeStepper.fixed(dt)
        ev = _alat_mod.AdaptiveLatticeEvolver(self.alat)
        for _ in range(self.N_STEPS):
            self.ulat.step(dt)
            ev.step(ts)
        phi_E_u = float(np.sum(self.ulat.phi**2))
        phi_E_a = float(np.sum(self.alat[0].leaf_phi ** 2))
        np.testing.assert_allclose(
            phi_E_u,
            phi_E_a,
            atol=self.ATOL,
            err_msg=f"phi_E mismatch: uniform={phi_E_u:.8e} adaptive={phi_E_a:.8e}",
        )

    def test_u_matches_after_steps(self):
        """u arrays must agree element-by-element (same positions, same ordering)."""
        dt = self.DT
        ts = _alat_mod.TimeStepper.fixed(dt)
        ev = _alat_mod.AdaptiveLatticeEvolver(self.alat)
        for _ in range(self.N_STEPS):
            self.ulat.step(dt)
            ev.step(ts)
        u_u = self.ulat.u
        u_a = np.array(self.alat[0].leaf_u)
        np.testing.assert_allclose(
            u_u,
            u_a,
            atol=self.ATOL,
            err_msg="u mismatch element-by-element after Harris IC stepping",
        )


if __name__ == "__main__":
    unittest.main()
