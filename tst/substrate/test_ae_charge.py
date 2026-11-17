"""
Charge in the continuous E/A model (C++ step_ae, iso weights) -- exp/2026/09/29/ae_charge.

1. Charge conservation: with the DEC curl (default), div E = sum_k W_k e_k per cell is exactly
   invariant (curl curl has no divergence), and with a current d(div E)/dt = -div J (continuity).
   The LS curl does not conserve it (kept only as a reference, set_curl('ls')).
2. A hand-seeded div E (u = A = 0, nothing held, no sponge) drives u to its static Coulomb
   potential: time-averaged u -> psi, the discrete Poisson solution of the same source
   (equilibrium/substrate/charge_exterior.py). The seeded state has no carrier: AE defines
   rho = div E and has no Gauss constraint tying it to matter, so this checks conservation and
   statics, not that the vacuum can hold a charge (physically it cannot: J is a prescribed
   stand-in for matter, and with J = 0 div E never appears).
"""

import unittest

import numpy as np

from equilibrium.substrate import continuous as C
from equilibrium.substrate.ambient import thermal_state
from equilibrium.substrate.charge_exterior import capacitor_source, divergence, seed_charge


def _random(curl, seed=0):
    ac = C.anchor_cluster(6.0, 0.5, n_threads=1)
    ac.set_curl(curl)
    rng = np.random.default_rng(seed)
    u, phi = thermal_state(ac.nb, 1.0, rng)
    _, e = thermal_state(ac.nb, 1.0, rng)
    ac.u, ac.phi, ac.e = u, phi, e
    return ac, rng


class TestChargeConservation(unittest.TestCase):
    def test_dec_conserves_div_e(self):
        ac, _ = _random("dec")
        d0 = divergence(ac.e, ac.nb, C.W)
        dt = C.stable_dt(ac)
        for _ in range(200):
            ac.step_ae(dt)
        self.assertLess(np.abs(divergence(ac.e, ac.nb, C.W) - d0).max(), 1e-11 * np.abs(d0).max())

    def test_continuity_with_current(self):
        ac, rng = _random("dec", seed=1)
        _, J = thermal_state(ac.nb, 1.0, rng)
        d0 = divergence(ac.e, ac.nb, C.W)
        dt = C.stable_dt(ac)
        ac.step_ae(dt, J, scale=0.7)
        change = divergence(ac.e, ac.nb, C.W) - d0
        expect = -dt * 0.7 * divergence(J, ac.nb, C.W)
        self.assertLess(np.abs(change - expect).max(), 1e-12 * np.abs(expect).max())

    def test_ls_does_not_conserve_div_e(self):
        """Reference: the least-squares curl leaks charge (O(1) on a random state)."""
        ac, _ = _random("ls")
        d0 = divergence(ac.e, ac.nb, C.W)
        dt = C.stable_dt(ac)
        for _ in range(200):
            ac.step_ae(dt)
        self.assertGreater(np.abs(divergence(ac.e, ac.nb, C.W) - d0).max(), 0.1 * np.abs(d0).max())


class TestBareChargeBuildsPotential(unittest.TestCase):
    def test_time_averaged_u_is_coulomb(self):
        R = 8.0
        ac = C.anchor_cluster(R, 1.0, n_threads=1)
        s, core, shell = capacitor_source(ac.pos)
        e0, psi = seed_charge(ac.nb, C.W, s)
        ac.e = e0
        ac.u = np.zeros(ac.N)
        ac.phi = np.zeros((ac.N, 14))
        dt = C.stable_dt(ac)
        n = int(round(12 * 2 * R / C.C0 / dt))
        acc = np.zeros(ac.N)
        for i in range(n):
            ac.step_ae(dt)
            if i >= n // 2:
                acc += ac.u
        u = acc / (n - n // 2)
        self.assertLess(np.abs(divergence(ac.e, ac.nb, C.W)[core] - 1.0), 1e-12)
        m = ~shell
        x, y = psi[m] - psi[m].mean(), u[m] - u[m].mean()
        a = float(x @ y / (x @ x))
        self.assertAlmostEqual(a, 1.0, delta=0.02)
        self.assertLess(np.sqrt(np.mean((y - a * x) ** 2)) / np.abs(y).max(), 2e-3)


if __name__ == "__main__":
    unittest.main()
