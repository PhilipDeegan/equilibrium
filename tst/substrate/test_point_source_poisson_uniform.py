"""
Static charge exterior, discrete model (numpy step(), current weights). Continuous-model
counterpart: test_point_source_poisson_continuous.py.

A static state of step() has uniform u and divergence-free phi, so a charge's exterior is radial
phi with uniform u (equilibrium/substrate/charge_exterior.py). The earlier version pinned u = 1 at
a cell (a pump) and asked for a static 1/r profile in u; step() has no static u gradient (phi
grows), and that profile only settled through the old sponge's hidden sink
(exp/2026/09/29/{sponge_assess,bc_design,charge_exterior}/status.md).
"""

import unittest

import numpy as np

from equilibrium.substrate.charge_exterior import capacitor_source, divergence, seed_charge
from equilibrium.substrate.lattice import _W, SubstrateLattice, wave_speed
from tst.substrate._charge_exterior import Q, ChargeExteriorChecks

R = 12.0
V_CELL = 4.0 / 3.0**1.5  # BCC Voronoi cell, d1 = 1: a^3 / 2 with a = 2 / sqrt3


class TestChargeExteriorDiscrete(unittest.TestCase, ChargeExteriorChecks):
    @classmethod
    def setUpClass(cls):
        lat = SubstrateLattice(radius_cells=R)
        cls.pos, cls.nb, cls.W = lat.pos, lat.nb, _W
        s, cls.core, cls.shell = capacitor_source(cls.pos, Q)
        cls.phi, cls.psi = seed_charge(cls.nb, cls.W, s)
        cls.c, cls.v_cell = wave_speed(), V_CELL

    def evolve(self, phi, held, t_end):
        lat = SubstrateLattice(radius_cells=R)
        lat.phi[:] = phi
        lat.u[:] = 0.0
        if held:
            lat.fixed_u[self.core] = True
            lat.fixed_u[self.shell] = True
        dt = lat.stable_dt()
        core_div = []
        for _ in range(int(round(t_end / dt))):
            lat.step(dt)
            core_div.append(divergence(lat.phi, lat.nb, _W)[self.core])
        return lat.phi, lat.u, core_div


if __name__ == "__main__":
    unittest.main()
