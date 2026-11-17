"""
Minimal, single/few-cell functional-expectation tests for
equilibrium.substrate.mhd.step_phi_diffuse -- the OTHER reconnection
candidate (project_dst_phi_diffusion), structurally separate from
step_cross (project_diffusion_vs_gate_mismatch, project_step_cross_v2).
Same discipline as test_step_cross_functional.py: no winding/knot/spheromak
aggregate, hand-derived exact expected values on the smallest fixture that
exercises the formula.

Kernel (equilibrium/substrate/mhd.py step_phi_diffuse):

    L[i, m] = sum_d W_d * (phi[nb[i,d], m] - phi[i, m])
    dphi[i, m] = eta * dt * sigma_face * (L[i, m] - L[nb[i,m], rev m]) / 2

For a FIXED face-channel m, L is an ordinary discrete Laplacian of that channel's value across
the cell's 14 REAL SPATIAL neighbours (direction d, weight W_d = W_HEX or W_SQ). Since
2026-09-29 each face gets one update shared by both stored copies (sigma at the face, the two
copies' Laplacians averaged), so phi_ki = -phi_ik survives; on an antisymmetric field in the
interior with uniform sigma this equals the per-copy Laplacian exactly. Fixtures therefore seed
faces as antisymmetric pairs: a single nonzero copy is not an antisymmetric state (its partner is
0), which is what the earlier single-copy fixtures used. This is the key structural difference from
step_cross, whose kernel does the opposite: it never looks across cells at
all, only redistributes among a cell's OWN 14 face-channels via _FACE_ADJ.
"""

import unittest

import numpy as np

from equilibrium.substrate.lattice import SubstrateLattice, _K_REV
from equilibrium.substrate.mhd import W_HEX, W_SQ  # the diffusion's weights: the Kelvin star 3:1
from equilibrium.substrate.mhd import step_phi_diffuse

DT = 1e-3
ETA = 0.1
M = 0  # the face-channel being diffused; arbitrary, 0 is a hex face


def _center_lattice():
    lat = SubstrateLattice(radius_cells=4.0)
    i = int(np.argmin(np.linalg.norm(lat.pos, axis=1)))
    assert lat._valid[i].all(), "centre cell must have all 14 neighbours"
    return lat, i


def _spike(lat, i, m, A):
    """Face (i, m) = A as an antisymmetric pair: its partner copy (nb[i,m], rev m) = -A."""
    lat.phi[i, m] = A
    lat.phi[int(lat.nb[i, m]), int(_K_REV[m])] = -A


class TestPhiDiffuseLaplacianExactValue(unittest.TestCase):
    """
    Single spike: the centre cell's face m = A (its partner copy -A), everything else 0.
    Predicts the exact centre-cell decrease (sum over all 14 real
    neighbours, total weight 8*W_HEX+6*W_SQ=30 -- the same total-weight
    constant Ampere's du/dt uses) and the exact increase at one hex neighbour
    and one sq neighbour (different weights, W_HEX=3 vs W_SQ=1).
    """

    def setUp(self):
        self.lat, self.i0 = _center_lattice()
        self.A = 1.0
        _spike(self.lat, self.i0, M, self.A)

    def test_centre_cell_decreases_by_total_weight(self):
        phi_before = float(self.lat.phi[self.i0, M])
        step_phi_diffuse(self.lat, DT, eta=ETA)
        # dphi = eta*dt*sigma*(sum_d W_d*(0-A)) = -eta*dt*A*30
        expected = phi_before - ETA * DT * self.A * (8 * W_HEX + 6 * W_SQ)
        self.assertAlmostEqual(float(self.lat.phi[self.i0, M]), expected, places=12)

    def test_hex_neighbour_increases_by_W_HEX(self):
        k_hex = 0  # M itself is a hex direction; pick a hex neighbour of i0
        j = int(self.lat.nb[self.i0, k_hex])
        step_phi_diffuse(self.lat, DT, eta=ETA)
        expected = ETA * DT * W_HEX * self.A
        self.assertAlmostEqual(float(self.lat.phi[j, M]), expected, places=12)

    def test_sq_neighbour_increases_by_W_SQ(self):
        k_sq = 8
        j = int(self.lat.nb[self.i0, k_sq])
        step_phi_diffuse(self.lat, DT, eta=ETA)
        expected = ETA * DT * W_SQ * self.A
        self.assertAlmostEqual(float(self.lat.phi[j, M]), expected, places=12)

    def test_other_face_channels_of_centre_cell_are_untouched(self):
        """Only channels M and rev M carry anything, so only they can move; the centre's rev-M
        channel picks up the partner copy's diffusion from its neighbour in direction M:
        -eta dt W_M A (same as the per-copy Laplacian on this antisymmetric fixture)."""
        step_phi_diffuse(self.lat, DT, eta=ETA)
        rev = int(_K_REV[M])
        for k in range(14):
            if k in (M, rev):
                continue
            self.assertEqual(float(self.lat.phi[self.i0, k]), 0.0, f"face {k} of the centre cell moved")
        self.assertAlmostEqual(float(self.lat.phi[self.i0, rev]), -ETA * DT * W_HEX * self.A, places=12)

    def test_neighbours_other_face_channels_are_untouched(self):
        j_hex = int(self.lat.nb[self.i0, 0])
        rev = int(_K_REV[M])
        step_phi_diffuse(self.lat, DT, eta=ETA)
        for k in range(14):
            if k not in (M, rev):
                self.assertEqual(float(self.lat.phi[j_hex, k]), 0.0)


class TestPhiDiffuseNoThreshold(unittest.TestCase):
    """
    Same "photon must not erode" question (R0) as step_cross's v1 gate,
    asked of the OTHER reconnection candidate: does phi_diffuse have any
    amplitude below which it stops dissipating a gradient? Its formula is
    linear in phi with no threshold term anywhere -- confirmed here the same
    way TestV1UnconditionalErosion confirmed it for step_cross gate="v1".
    If phi_diffuse is meant to replace step_cross as the actual reconnection
    mechanism, it currently inherits the exact same R0 problem: nothing
    gates it on amplitude, topology, or motion.
    """

    def test_erodes_at_photon_scale_amplitude_by_the_same_fraction(self):
        A_unit = 1.0
        lat_unit, i_unit = _center_lattice()
        _spike(lat_unit, i_unit, M, A_unit)
        step_phi_diffuse(lat_unit, DT, eta=ETA)

        A_small = 1e-8
        lat_small, i_small = _center_lattice()
        _spike(lat_small, i_small, M, A_small)
        step_phi_diffuse(lat_small, DT, eta=ETA)

        frac_unit = (float(lat_unit.phi[i_unit, M]) - A_unit) / A_unit
        frac_small = (float(lat_small.phi[i_small, M]) - A_small) / A_small
        self.assertNotEqual(
            frac_small, 0.0, "phi_diffuse must have moved even a tiny spike"
        )
        self.assertAlmostEqual(frac_small, frac_unit, places=6)


class TestPhiDiffusePerCellDivergence(unittest.TestCase):
    """
    Two invariants, checked separately: pairwise ANTISYMMETRY (phi[i,k] = -phi[nb[i,k], rev k],
    the discrete div B = 0 of test_maxwell.py) and the PER-CELL divergence sum_k W_k phi[i,k].
    An antisymmetric divergence-free field is built by projecting a random antisymmetric field
    onto the kernel of the per-cell divergence (least squares). In the interior (cells whose
    neighbours all have full neighbour sets) with uniform sigma, the per-channel Laplacian commutes
    with the per-cell weighted sum, so div stays 0 there (project_dst_phi_diffusion).
    """

    def _div_free(self, lat, seed=0):
        import scipy.sparse as sp
        from scipy.sparse.linalg import lsqr

        from equilibrium.substrate.ambient import thermal_state

        _, phi = thermal_state(lat.nb, 1.0, np.random.default_rng(seed))
        i, k = np.nonzero(lat.nb >= 0)
        j = lat.nb[i, k]
        once = i < j
        fi, fk, fj = i[once], k[once], j[once]
        w = np.array([W_HEX**0.5] * 8 + [W_SQ**0.5] * 6)[fk]  # step()'s per-leg W
        F = len(fi)
        G = sp.csr_matrix(
            (np.concatenate([w, -w]), (np.concatenate([np.arange(F)] * 2), np.concatenate([fj, fi]))),
            shape=(F, lat.N),
        )
        f = phi[fi, fk]
        f = f - G @ lsqr(G, f, atol=1e-14, btol=1e-14, iter_lim=50000)[0]
        out = np.zeros_like(phi)
        out[fi, fk] = f
        out[fj, _K_REV[fk]] = -f
        return out, np.array([W_HEX**0.5] * 8 + [W_SQ**0.5] * 6)

    def test_per_cell_divergence_preserved_in_the_interior(self):
        lat, _ = _center_lattice()
        lat.phi[:], W = self._div_free(lat)
        div = lambda: (W * np.where(lat.nb >= 0, lat.phi, 0.0)).sum(axis=1)
        self.assertLess(np.abs(div()).max(), 1e-10)
        full = lat._valid.all(axis=1)
        interior = full & np.array([full[lat.nb[c]].all() if full[c] else False for c in range(lat.N)])
        before = lat.phi.copy()
        step_phi_diffuse(lat, DT, eta=ETA)
        self.assertGreater(np.abs(lat.phi - before).max(), 0.0)
        self.assertLess(np.abs(div()[interior]).max(), 1e-10)

    def test_antisymmetry_preserved(self):
        lat, i0 = _center_lattice()
        _spike(lat, i0, M, 1.0)
        step_phi_diffuse(lat, DT, eta=ETA)
        i, k = np.nonzero(lat.nb >= 0)
        self.assertEqual(float(np.max(np.abs(lat.phi[i, k] + lat.phi[lat.nb[i, k], _K_REV[k]]))), 0.0)


if __name__ == "__main__":
    unittest.main()
