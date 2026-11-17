"""Shared assertions for the static charge exterior (equilibrium/substrate/charge_exterior.py).

Subclasses (unittest.TestCase + ChargeExteriorChecks) provide in setUpClass:
  pos, nb, W, phi, psi, core, shell   the seeded state
  c, v_cell                           wave speed and cell volume (lattice units)
  evolve(phi, held, t_end) -> (phi_end, u_end, core_div(t) samples)
"""

import numpy as np

from equilibrium.substrate.charge_exterior import divergence

Q = 1.0


class ChargeExteriorChecks:
    def _fit(self):
        r = np.linalg.norm(self.pos, axis=1)
        b = r[self.shell].mean()
        m = (r > 1.5) & (r < b - 2.0)
        X = 1.0 / r[m] - 1.0 / b
        y = self.psi[m] - self.psi[self.shell].mean()
        a = float(np.sum(X * y) / np.sum(X * X))
        return r, b, m, X, y, a

    def test_divergence_localised(self):
        d = divergence(self.phi, self.nb, self.W)
        rest = ~self.shell
        rest[self.core] = False
        self.assertAlmostEqual(d[self.core], Q, places=10)
        self.assertAlmostEqual(d[self.shell].sum(), -Q, places=10)
        self.assertLess(np.abs(d[rest]).max(), 1e-10)

    def test_coulomb_profile(self):
        _, _, _, X, y, a = self._fit()
        resid = np.sqrt(np.mean((y - a * X) ** 2)) / np.abs(y).max()
        self.assertLess(resid, 5e-3)

    def test_direction_dependence(self):
        """psi r / fit along [100], [110], [111] at mid radius: within the measured lattice
        anisotropy (2.4% [100], 0.5% diagonals at R = 12, 2026-09-29)."""
        r, b, m, _, _, a = self._fit()
        for n in (np.array([1, 0, 0.0]), np.array([1, 1, 0.0]) / np.sqrt(2), np.ones(3) / np.sqrt(3)):
            sel = m & ((self.pos @ n) / np.maximum(r, 1e-12) > 0.97)
            ratio = np.mean((self.psi[sel] - self.psi[self.shell].mean()) / (a * (1 / r[sel] - 1 / b)))
            self.assertLess(abs(ratio - 1.0), 0.035, f"direction {n}: {ratio:.4f}")

    def test_eps_is_c2_over_cell_volume(self):
        *_, a = self._fit()
        eps = Q / (4 * np.pi * a)
        self.assertAlmostEqual(eps / (self.c**2 / self.v_cell), 1.0, delta=0.01)

    def test_held_core_is_static(self):
        phi_end, u_end, _ = self.evolve(self.phi, held=True, t_end=12.0 / self.c)
        self.assertLess(np.abs(u_end).max(), 1e-12)
        self.assertLess(np.abs(phi_end - self.phi).max(), 1e-12)

    def test_bare_core_does_not_hold(self):
        """Without something holding it, the core divergence leaves as u waves (characterises the
        rule; a mechanism that holds charge would make this fail)."""
        _, _, core_div = self.evolve(self.phi, held=False, t_end=18.0 / self.c)
        late = np.abs(np.asarray(core_div)[len(core_div) // 2 :]).mean()
        self.assertLess(late, 0.2 * Q, f"mean |core div| over the second half: {late:.3f}")
