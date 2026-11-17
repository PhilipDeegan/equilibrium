"""
Tests for equilibrium.substrate.adaptive.dst_state — mesh-structure
introspection (cells-per-level histogram, hex-face neighbour-level-jump
stats).
"""

import unittest

import numpy as np

from equilibrium.substrate.adaptive import dst_state
from tst.substrate.adaptive import build_adaptive_sphere


class TestDstState(unittest.TestCase):
    R = 5.0

    def test_cells_per_level_matches_leaf_level_histogram_uniform(self):
        alat, _ = build_adaptive_sphere(self.R, max_level=0)
        cl = alat[0]
        state = dst_state(cl)
        self.assertEqual(state["n_leaves"], cl.n_leaves)
        self.assertEqual(state["max_level"], 0)
        self.assertEqual(state["cells_per_level"], {0: cl.n_leaves})
        # A uniform level-0 mesh has same-level hex neighbours everywhere:
        # max jump is 0 (not None — that's reserved for zero valid hex faces
        # at all), and no violation.
        self.assertEqual(state["max_hex_leaf_jump"], 0)
        self.assertEqual(state["n_hex_leaf_jumps_gt1"], 0)

    def test_cells_per_level_after_single_split(self):
        alat, origin_g = build_adaptive_sphere(self.R, max_level=1)
        alat.split(0, origin_g)
        cl = alat[0]
        state = dst_state(cl)
        self.assertEqual(state["n_leaves"], cl.n_leaves)
        self.assertEqual(state["max_level"], 1)
        self.assertEqual(state["cells_per_level"][1], 27)
        self.assertEqual(sum(state["cells_per_level"].values()), cl.n_leaves)

    def _center_only_split(self, alat, g, n_splits):
        """2:1-respecting recursive descent (see TestMultipleSplits's own
        _run_bfs_splits, tst/substrate/test_adaptive_vs_substrate.py, for the
        production-test version this mirrors)."""
        for _ in range(n_splits):
            if not alat.can_split(0, g):
                break
            cl = alat[0]
            gidx = np.where(cl.leaf_global_indices == g)[0][0]
            parent_pos = cl.leaf_pos[gidx].copy()
            first_child_g = alat.split(0, g)
            cl = alat[0]
            child_gs = np.arange(first_child_g, first_child_g + 27)
            j_of = {int(cg): j for j, cg in enumerate(cl.leaf_global_indices)}
            d2 = np.array(
                [
                    np.sum((cl.leaf_pos[j_of[int(cg)]] - parent_pos) ** 2)
                    for cg in child_gs
                ]
            )
            g = int(child_gs[np.argmin(d2)])
        return g

    def test_no_violation_on_2to1_respecting_mesh(self):
        """Recursive center-only splitting never creates a >1-level hex-face
        gap (every center child's faces are sibling-internal by
        construction) — dst_state must report zero violations at any depth."""
        alat, origin_g = build_adaptive_sphere(self.R, max_level=6)
        self._center_only_split(alat, origin_g, 6)
        cl = alat[0]
        state = dst_state(cl)
        self.assertEqual(state["max_level"], 6)
        self.assertEqual(
            state["n_hex_leaf_jumps_gt1"],
            0,
            "2:1-respecting construction must never produce a >1-level hex-face jump",
        )

    def test_violation_detected_on_sibling_fanout_mesh(self):
        """A sibling-fan-out BFS (split every splittable child, not just one)
        manufactures a genuine >1-level hex-face gap — a surface-facing
        child, once split itself, hands its own children a direct pointer to
        the still-unsplit ORIGINAL neighbour two levels down (see `now`,
        2026-08-13 stale inter-parent proxy investigation). dst_state must
        catch this."""
        from collections import deque

        alat, origin_g = build_adaptive_sphere(self.R, max_level=4)
        queue = deque([origin_g])
        for _ in range(6):
            if not queue:
                break
            g = queue.popleft()
            if not alat.can_split(0, g):
                continue
            first_child_g = alat.split(0, g)
            for c in range(27):
                if alat.can_split(0, first_child_g + c):
                    queue.append(first_child_g + c)
        cl = alat[0]
        state = dst_state(cl)
        self.assertGreater(
            state["n_hex_leaf_jumps_gt1"],
            0,
            "sibling-fan-out splitting is expected to violate 2:1 balance",
        )
        self.assertGreater(state["max_hex_leaf_jump"], 1)


if __name__ == "__main__":
    unittest.main()
