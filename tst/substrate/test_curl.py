"""
Verify equilibrium.substrate.curl's discrete curl(phi)=B against the exact
classical straight-wire magnetostatic profile (zero at the axis, linear
ramp to a peak at the wire's own radius, 1/r falloff outside) -- built
independently of equilibrium.substrate.mhd (see curl.py's module
docstring for why: mhd.py's SQ_PLAQUETTES["y"] loop uses the wrong cyclic
face order, a confirmed sign bug, and equilibrium/substrate/lattice.py and
mhd.py's own comments disagree about which face index is +x vs -x).

Method: derive the exact analytic vector potential A_z(r) for a uniform-
current-density wire along z, sample it onto phi via a discrete line
integral per directed edge (phi[i,k] = A_z(r_mid)*dz_k), then check the
recovered tangential B component against the known target. This is a pure
static geometry test -- no time-stepping.
"""

import unittest

import numpy as np

from equilibrium.substrate.curl import b_vec_vectorized
from equilibrium.substrate.lattice import SubstrateLattice

R = 25.0
A_WIRE = 6.0
J = 1.0
# Sq-face hop distance is 2/sqrt(3) (see lattice.py _D_FACE), not 1 -- the
# discrete line-integral construction below picks up a (hop distance)^2
# calibration factor relative to the continuum target using mu0=J=1 in
# raw cell-position units. Confirmed exactly: (2/sqrt(3))^2 = 4/3.
_HOP_SQ_FACTOR = (2.0 / np.sqrt(3.0)) ** 2


def _A_z(r, a):
    r = np.asarray(r, dtype=np.float64)
    inside = r < a
    return np.where(
        inside,
        -J * r**2 / 4.0,
        -(J * a**2 / 4.0) * (1.0 + 2.0 * np.log(np.maximum(r, 1e-12) / a)),
    )


def _b_phi_target(r, a):
    r = np.asarray(r, dtype=np.float64)
    inside = r < a
    return np.where(inside, J * r / 2.0, J * a**2 / (2.0 * np.maximum(r, 1e-12)))


def _measure(radius_cells, a):
    lat = SubstrateLattice(radius_cells=radius_cells)
    pos, nb = lat.pos, lat.nb
    valid = nb >= 0
    safe_nb = np.where(valid, nb, 0)
    mid = 0.5 * (pos[:, None, :] + pos[safe_nb])
    r_mid = np.linalg.norm(mid[:, :, :2], axis=2)
    dz = pos[safe_nb][:, :, 2] - pos[:, None, 2]
    lat.phi = np.where(valid, _A_z(r_mid, a) * dz, 0.0)

    B = b_vec_vectorized(pos, nb, lat.phi)
    Bx, By, Bz = B[:, 0], B[:, 1], B[:, 2]
    r = np.linalg.norm(pos[:, :2], axis=1)
    theta = np.arctan2(pos[:, 1], pos[:, 0])
    have = np.isfinite(Bx) & np.isfinite(By) & np.isfinite(Bz)
    mask = have & (np.abs(pos[:, 2]) < 2.0) & (r > 0.5)

    b_phi_meas = -Bx * np.sin(theta) + By * np.cos(theta)
    b_r_meas = Bx * np.cos(theta) + By * np.sin(theta)
    return r, mask, b_phi_meas, b_r_meas, Bz


class TestCurlStraightWire(unittest.TestCase):
    def test_bz_and_br_vanish(self):
        """A z-only vector potential A_z(r) has B_z=0 and B_r=0 everywhere."""
        r, mask, b_phi_meas, b_r_meas, Bz = _measure(R, A_WIRE)
        self.assertEqual(float(np.max(np.abs(Bz[mask]))), 0.0)
        # residual is small compared to the real B_phi signal (checked below)
        signal = np.sqrt(np.mean(_b_phi_target(r[mask], A_WIRE) ** 2))
        residual = np.sqrt(np.mean(b_r_meas[mask] ** 2))
        self.assertLess(residual, 0.1 * signal)

    def test_b_phi_matches_classical_wire_profile(self):
        """
        B_phi(r): zero-at-axis, linear ramp inside the wire, 1/r falloff
        outside -- constant ratio to the classical target across the whole
        range (inside AND outside), up to the known hop-distance^2 factor.
        """
        a = A_WIRE
        r, mask, b_phi_meas, _, _ = _measure(R, a)
        edges = np.linspace(0.0, R * 0.5, 14)
        ratios = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            in_bin = mask & (r >= lo) & (r < hi)
            if in_bin.sum() < 3:
                continue
            target = float(np.mean(_b_phi_target(r[in_bin], a)))
            if abs(target) < 1e-9:
                continue
            meas = float(np.mean(b_phi_meas[in_bin]))
            ratios.append(meas / target)
        self.assertGreaterEqual(len(ratios), 6, "not enough populated bins to test")
        ratios = np.array(ratios)
        # every bin (inside AND outside the wire) should land on the same
        # constant ratio -- that's the real claim: the SHAPE matches exactly.
        self.assertAlmostEqual(float(ratios.std()), 0.0, delta=0.05)
        self.assertAlmostEqual(float(ratios.mean()), _HOP_SQ_FACTOR, delta=0.05)


if __name__ == "__main__":
    unittest.main()
