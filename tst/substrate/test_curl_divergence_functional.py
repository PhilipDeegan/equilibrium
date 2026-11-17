"""
Minimal, few-cell functional-expectation tests for how curl and divergence
relate on this lattice -- same discipline as test_step_cross_functional.py /
test_phi_diffuse_functional.py: no winding/knot/spheromak aggregate, hand
derivable exact values.

Motivation (2026-08-18 discussion): the classical identity div(curl A)=0 is
NOT a free algebraic fact here the way curl(grad u)=0 is (that one IS
discrete-exact, see equilibrium/substrate/discrete_curl.py). Per-cell
divergence Sum_k W_k*phi[i,k] is, up to sign, exactly the Ampere update's
du_i/dt -- so "divergence=0" isn't an independent structural field property
to check against Helmholtz decomposition, it's the definition of a
STATIONARY (non-instantaneously-relaxing) configuration. A genuinely
curl-full construction is only divergence-free if it's built to conserve
the WEIGHTED quantity W_k*phi[i,k] (the Kirchhoff/current-carrying
quantity) around a loop, not the raw phi[i,k] value -- these differ the
moment a loop crosses a hex/sq face-type boundary (W_HEX=3 vs W_SQ=1).

Every ring construction actually used elsewhere in this repo
(find_toroidal_ring, any mode) happens to be pure-sq-face (uniform
weight=1), so this distinction has never been exercised there. This file
uses the same minimal mixed hex/hex/sq 3-cell loop test_maxwell.py's
TestDiscreteFaraday.test_faraday_plaquette_circulation already verified
closes (faces 0, 3, 9) to exercise it directly.
"""

import unittest

import numpy as np

from equilibrium.substrate.lattice import W_HEX, W_SQ, SubstrateLattice, _K_REV

_W = np.array([W_HEX] * 8 + [W_SQ] * 6, dtype=np.float64)
_FACES = [0, 3, 9]  # hex, hex, sq -- the verified-closing mixed 3-cycle


def _mixed_loop():
    """Same fixture as test_maxwell.py's mixed 3-cycle: hex k=0, hex k=3,
    sq k=9, verified to close back to the start cell."""
    lat = SubstrateLattice(radius_cells=6.0)
    c0 = int(np.argmin(np.linalg.norm(lat.pos, axis=1)))
    cells = [c0, int(lat.nb[c0, 0]), -1]
    cells[2] = int(lat.nb[cells[1], 3])
    assert int(lat.nb[cells[2], 9]) == c0, "loop does not close back to start cell"
    return lat, cells


def _seed_flat_current(lat, cells, amplitude):
    """The OLD (pre-fix) construction: same raw phi magnitude on every face
    regardless of type -- what test_ring_circulation_conservation.py's
    seed_constant_current did before this session's fix."""
    for m in range(3):
        ci, cj = cells[m], cells[(m + 1) % 3]
        k = _FACES[m]
        mk = int(_K_REV[k])
        lat.phi[ci, k] = amplitude
        lat.phi[cj, mk] = -amplitude


def _seed_weighted_current(lat, cells, amplitude):
    """The FIXED construction: phi_k = amplitude / W_k, so the conserved
    quantity is W_k*phi_k (Kirchhoff current), not phi_k itself."""
    for m in range(3):
        ci, cj = cells[m], cells[(m + 1) % 3]
        k = _FACES[m]
        mk = int(_K_REV[k])
        a = amplitude / _W[k]
        lat.phi[ci, k] = a
        lat.phi[cj, mk] = -a


def _per_cell_divergence(lat, i):
    return float((_W * lat.phi[i]).sum())


class TestFlatCurrentLeaksDivergenceAtWeightJunction(unittest.TestCase):
    """Documents the pitfall: an unweighted 'constant current' is only
    divergence-free by coincidence, when every hop happens to share the same
    face weight. The moment a loop crosses a hex/sq boundary, it leaks a
    real, nonzero, hand-predictable divergence."""

    def test_junction_cells_have_nonzero_divergence(self):
        lat, cells = _mixed_loop()
        _seed_flat_current(lat, cells, amplitude=1.0)

        # cells[0]: outgoing hex(W=3, +A), incoming sq(W=1, -A from cells[2])
        # cells[2]: outgoing sq(W=1, +A), incoming hex(W=3, -A from cells[1])
        # cells[1]: outgoing hex(W=3, +A), incoming hex(W=3, -A) -- same weight, balances
        expected = {
            cells[0]: W_HEX * 1.0 - W_SQ * 1.0,  # +2.0
            cells[1]: 0.0,
            cells[2]: W_SQ * 1.0 - W_HEX * 1.0,  # -2.0
        }
        for c, exp in expected.items():
            self.assertAlmostEqual(_per_cell_divergence(lat, c), exp, places=12)

        self.assertNotEqual(_per_cell_divergence(lat, cells[0]), 0.0)
        self.assertNotEqual(_per_cell_divergence(lat, cells[2]), 0.0)

    def test_same_weight_only_loop_never_shows_the_leak(self):
        """Sanity check for WHY this went unnoticed: a loop confined to one
        face weight type (like every find_toroidal_ring construction in this
        repo) is trivially safe -- both terms cancel regardless of weight."""
        lat = SubstrateLattice(radius_cells=6.0)
        c0 = int(np.argmin(np.linalg.norm(lat.pos, axis=1)))
        sq_faces = [8, 10, 9, 11]
        cells = [c0, int(lat.nb[c0, 8]), -1, -1]
        cells[2] = int(lat.nb[cells[1], 10])
        cells[3] = int(lat.nb[cells[2], 9])
        assert int(lat.nb[cells[3], 11]) == c0

        for m in range(4):
            ci, cj = cells[m], cells[(m + 1) % 4]
            k = sq_faces[m]
            mk = int(_K_REV[k])
            lat.phi[ci, k] = 1.0
            lat.phi[cj, mk] = -1.0

        for c in cells:
            self.assertAlmostEqual(_per_cell_divergence(lat, c), 0.0, places=12)


class TestWeightedCurrentIsDivergenceFree(unittest.TestCase):
    """Documents the fix: weighting by 1/W_k makes the SAME mixed loop
    exactly divergence-free at every cell, including the junction -- the
    discrete, lattice-native form of div(curl A)=0: a genuinely stationary
    curl-full construction must conserve W_k*phi[i,k], not phi[i,k]."""

    def test_all_three_cells_are_exactly_divergence_free(self):
        lat, cells = _mixed_loop()
        _seed_weighted_current(lat, cells, amplitude=1.0)
        for c in cells:
            self.assertAlmostEqual(_per_cell_divergence(lat, c), 0.0, places=12)

    def test_weighting_does_not_destroy_genuine_circulation(self):
        """The fix must not accidentally make the field curl-free too --
        confirm real, nonzero circulation survives around the loop."""
        lat, cells = _mixed_loop()
        _seed_weighted_current(lat, cells, amplitude=1.0)

        circ = sum(float(lat.phi[cells[m], _FACES[m]]) for m in range(3))
        self.assertGreater(
            abs(circ), 0.0, "weighted current should still carry real circulation"
        )
        # hand-derived: A/W_HEX + A/W_HEX + A/W_SQ = 1/3 + 1/3 + 1 = 5/3
        self.assertAlmostEqual(circ, 1.0 / W_HEX + 1.0 / W_HEX + 1.0 / W_SQ, places=12)


if __name__ == "__main__":
    unittest.main()
