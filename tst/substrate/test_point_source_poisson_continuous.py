"""
Static charge exterior, continuous model: C++ lattice.continuous AdaptiveLattice step() at level
0 (iso weights W = sqrt2:1). Discrete counterpart: test_point_source_poisson_uniform.py (same
assertions, tst/substrate/_charge_exterior.py).

Same construction (equilibrium/substrate/charge_exterior.py): radial phi with uniform u is a
static state of the step() rule; the earlier pinned-u pump version only settled through the old
sponge's hidden sink (exp/2026/09/29/sponge_assess/status.md). The E sector (step_ae) has its
own statics (A, u static, E = -grad u - A_t), not covered here.
"""

import os
import unittest

import numpy as np

from equilibrium.substrate import continuous as C
from equilibrium.substrate.charge_exterior import capacitor_source, divergence, seed_charge
from tst.substrate._charge_exterior import Q, ChargeExteriorChecks

_N_THREADS = int(os.environ.get("EQ_TEST_N_THREADS", 1))

R = 12.0
V_CELL = 4.0 / 3.0**1.5


class TestChargeExteriorContinuous(unittest.TestCase, ChargeExteriorChecks):
    @classmethod
    def setUpClass(cls):
        ac = C.anchor_cluster(R, 1.0, n_threads=_N_THREADS)
        cls.pos, cls.nb, cls.W = ac.pos, ac.nb, C.W
        s, cls.core, cls.shell = capacitor_source(cls.pos, Q)
        cls.phi, cls.psi = seed_charge(cls.nb, cls.W, s)
        cls.c, cls.v_cell = C.C0, V_CELL

    def evolve(self, phi, held, t_end):
        ac = C.anchor_cluster(R, 1.0, n_threads=_N_THREADS)
        ac.phi = phi
        ac.u = np.zeros(ac.N)
        if held:
            fu = np.zeros(ac.N, dtype=np.uint8)
            fu[self.core] = 1
            fu[self.shell] = 1
            ac.cl.set_leaf_fixed_u(fu)
        dt = C.stable_dt(ac)
        core_div = []
        for _ in range(int(round(t_end / dt))):
            ac.alat.step(0, dt)
            core_div.append(divergence(ac.phi_rows([self.core])[None, 0], self.nb[[self.core]], self.W)[0])
        return ac.phi, ac.u, core_div


if __name__ == "__main__":
    unittest.main()
