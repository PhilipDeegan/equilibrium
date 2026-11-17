"""Validates the Kelvin-circulation-conservation property of the DST wave
update (`_step_pass1`/`_step_pass2`) and identifies `step_cross` as the
resistive/topology-opening term that erodes it -- the mechanism that would
let two rings' fields actually reconnect/merge (PHARE-style flux-tube
merging, see `project_dst_discrete_curl` memory), unlike the current
curl-free dipole-mode spheromak IC which has zero circulation to begin with
and so nothing to erode.

Two circulation constructions on the SAME ring (`find_toroidal_ring`),
compared:

1. `init_toroidal_winding`'s default `amplitude*cos(K*(idx+0.5))` standing
   wave -- sums to EXACTLY zero over one ring period (confirmed
   analytically and numerically, ~1e-15), so it is expressible as some
   scalar's discrete derivative around the ring -- curl-free in disguise.
   `phi_ring_circulation` reads ~1e-15 for it, not a bug.
2. `seed_constant_current` (this file) -- the SAME phi value all the way
   around the ring, sum = amplitude*N_ring != 0, which CANNOT be any
   scalar's discrete derivative around a closed loop (the necessary
   zero-sum condition fails by construction) -- a genuine persistent
   current / curl-full field, no Cartesian formula involved (unlike
   `init_spheromak_ic`, a DIFFERENT, older function in the same module that
   is the same buggy Cartesian Taylor-state construction
   `dipole_mode_ic.py` replaced -- not used here).
"""

import unittest

import numpy as np

from equilibrium import lattice as _lat
from equilibrium.substrate.lattice import W_HEX, W_SQ
from tst.substrate.adaptive.adaptive_flux_tube import (
    find_toroidal_ring,
    init_toroidal_winding,
    phi_ring_circulation,
    _MIRROR,
)

R_DOMAIN = 18.0
_W = np.array([W_HEX] * 8 + [W_SQ] * 6, dtype=np.float64)


def _build():
    alat = _lat.AdaptiveLattice()
    cl = alat[0]
    _lat.build_bcc_sphere(cl, R_DOMAIN, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)
    return alat, cl


def seed_constant_current(cl, ring_js, ring_faces, amplitude=1.0):
    """
    A genuinely STATIONARY current loop must conserve W_k*phi[i,k] (the
    Kirchhoff/current-carrying quantity), not phi[i,k] itself -- per-cell
    divergence Sum_k W_k*phi[i,k] is, up to sign, exactly the Ampere update's
    du/dt, so an unweighted flat amplitude leaves real nonzero divergence
    (an immediate, spurious du/dt) at any hex/sq face-type junction (measured
    +-2.0 on a minimal 3-cell mixed loop -- see
    tst/substrate/test_curl_divergence_functional.py). Every ring this file
    actually uses is pure-sq-face (find_toroidal_ring, any mode), where
    W_SQ=1 makes this identical to the old flat version -- this fix is a
    no-op here, only load-bearing for a future mixed hex/sq loop.
    """
    N_ring = len(ring_js)
    phi = cl.leaf_phi.copy()
    for idx in range(N_ring):
        j_i = ring_js[idx]
        j_nxt = ring_js[(idx + 1) % N_ring]
        k = ring_faces[idx]
        mk = int(_MIRROR[k])
        a = amplitude / _W[k]
        phi[k, j_i] = a
        phi[mk, j_nxt] = -a
    cl.set_leaf_phi(phi)


def _H(cl):
    u = np.array(cl.leaf_u)
    phi = np.array(cl.leaf_phi)
    return float(np.sum(u**2) + 0.5 * np.sum(phi**2))


class TestRingCirculationConservation(unittest.TestCase):
    def setUp(self):
        self.alat, self.cl = _build()
        ring_js, ring_faces = find_toroidal_ring(
            self.cl, axis="z", mode="square", hops=3
        )
        self.assertIsNotNone(
            ring_js, "ring-finding failed -- check R_DOMAIN/hops fit the domain"
        )
        self.ring_js, self.ring_faces = ring_js, ring_faces
        self.centre = np.array(self.cl.leaf_pos)[ring_js].mean(axis=0)
        self.dt = 0.4 / 6.0  # conservative fixed dt, matches probe script

    def test_standing_wave_winding_has_zero_net_circulation(self):
        # Documents the pitfall from project_spheromak_decay's 2026-07-03
        # correction: init_toroidal_winding's cos-pattern is a standing wave
        # with zero net circulation BY CONSTRUCTION, not a bug in the
        # measurement.
        init_toroidal_winding(
            self.cl,
            self.ring_js,
            self.ring_faces,
            amplitude=1.0,
            frozen=False,
            phi_only=True,
        )
        circ = phi_ring_circulation(self.cl, self.ring_js, self.centre)
        self.assertLess(
            abs(circ),
            1e-9,
            "standing-wave ring winding should have ~zero net circulation",
        )

    def test_constant_current_has_real_nonzero_circulation(self):
        seed_constant_current(self.cl, self.ring_js, self.ring_faces, amplitude=1.0)
        circ = phi_ring_circulation(self.cl, self.ring_js, self.centre)
        self.assertGreater(
            abs(circ),
            1.0,
            "constant-current ring should carry real, O(1)+ net circulation",
        )

    def test_pure_wave_dynamics_conserves_circulation_exactly(self):
        seed_constant_current(self.cl, self.ring_js, self.ring_faces, amplitude=1.0)
        circ0 = phi_ring_circulation(self.cl, self.ring_js, self.centre)
        H0 = _H(self.cl)

        ts = _lat.TimeStepper.by_level(self.dt)
        evolver = _lat.AdaptiveLatticeEvolver(self.alat)
        for _ in range(10):
            evolver.step(ts)

        circ1 = phi_ring_circulation(self.cl, self.ring_js, self.centre)
        H1 = _H(self.cl)
        self.assertAlmostEqual(
            circ1,
            circ0,
            delta=1e-9 * abs(circ0),
            msg="pure wave dynamics (no step_cross) must conserve ring circulation exactly "
            "(Kelvin-circulation analogue: pass2's dphi/dt=-sigma*grad(u) always has zero "
            "circulation, same telescoping fact as equilibrium.substrate.discrete_curl)",
        )
        self.assertAlmostEqual(
            H1, H0, delta=1e-9 * H0, msg="H must also be exactly conserved"
        )

    def test_step_cross_erodes_circulation(self):
        seed_constant_current(self.cl, self.ring_js, self.ring_faces, amplitude=1.0)
        circ0 = phi_ring_circulation(self.cl, self.ring_js, self.centre)
        H0 = _H(self.cl)

        ts = _lat.TimeStepper.by_level(self.dt)
        evolver = _lat.AdaptiveLatticeEvolver(self.alat)
        cross_scale = 1e-2
        for _ in range(10):
            evolver.step(ts)
            self.alat.step_cross_all(self.dt, cross_scale)

        circ1 = phi_ring_circulation(self.cl, self.ring_js, self.centre)
        H1 = _H(self.cl)
        self.assertLess(
            abs(circ1),
            abs(circ0) * 0.999,
            "step_cross (the resistive/topology-opening term) should measurably erode "
            "circulation that pure wave dynamics conserves exactly -- this is the DST "
            "analogue of resistive reconnection eroding frozen-in flux",
        )
        self.assertLess(
            H1, H0, "step_cross should dissipate H alongside eroding circulation"
        )


if __name__ == "__main__":
    unittest.main()
