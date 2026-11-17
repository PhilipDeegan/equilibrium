import os
import unittest

import numpy as np

from equilibrium.substrate.walk import CppWalk, cpp_box, rest_state, to_complex
from equilibrium.substrate.walk_energy import minus_branch, quasi_energy

_N_THREADS = int(os.environ.get("EQ_TEST_N_THREADS", 1))


def _box(P):
    alat = cpp_box(P, n_threads=max(1, _N_THREADS))
    pos = np.array(alat[0].leaf_pos)
    ix = (np.rint(pos * np.sqrt(3)).astype(np.int64) + P // 2) % P
    return alat, pos, ix


def _rough(pos, P):
    side = P / np.sqrt(3)
    rng = np.random.default_rng(0)
    ph = sum(rng.normal() * np.sin(2 * np.pi * (rng.integers(1, 3, 3) @ pos.T) / side + rng.random() * 6) for _ in range(4))
    d = np.linalg.norm(pos - pos[0], axis=1)
    return np.sqrt(1 - 0.6 * np.exp(-d**2 / 20)) * np.exp(1j * ph)


def _psi(w):
    L, R = w.state()
    return np.concatenate([to_complex(L), to_complex(R)], 1)


class TestWalkEnergy(unittest.TestCase):
    P, m = 16, 0.8

    def test_rest_state(self):
        """uniform rest state: E_q = om0 x norm, no branch- content."""
        alat, pos, ix = _box(self.P)
        w = CppWalk(alat, 0, m=self.m, g=0.0, parity=True)
        w.seed(np.ones(len(pos), complex))
        psi = _psi(w)
        _, om0 = rest_state(self.m)
        nrm = np.sum(np.abs(psi)**2)
        self.assertAlmostEqual(quasi_energy(psi, ix, self.P, self.m) / nrm, om0, places=10)
        self.assertLess(np.sum(np.abs(minus_branch(psi, ix, self.P, self.m))**2) / nrm, 1e-20)

    def test_projector(self):
        """P_- is an orthogonal projector."""
        alat, pos, ix = _box(self.P)
        w = CppWalk(alat, 0, m=self.m, g=0.0, parity=True)
        w.seed(_rough(pos, self.P))
        psi = _psi(w)
        pm = minus_branch(psi, ix, self.P, self.m)
        nrm = np.sum(np.abs(psi)**2)
        self.assertGreater(np.sum(np.abs(pm)**2) / nrm, 1e-3)
        self.assertLess(np.max(np.abs(minus_branch(pm, ix, self.P, self.m) - pm)), 1e-10)
        self.assertLess(abs(np.vdot(pm, psi - pm)) / nrm, 1e-12)

    def test_g0_invariants(self):
        """g = 0 parity walk: E_q and the branch- norm are exact invariants of the two-tick map."""
        alat, pos, ix = _box(self.P)
        w = CppWalk(alat, 0, m=self.m, g=0.0, parity=True)
        w.seed(_rough(pos, self.P))
        psi = _psi(w)
        E0, N0 = quasi_energy(psi, ix, self.P, self.m), np.sum(np.abs(minus_branch(psi, ix, self.P, self.m))**2)
        for _ in range(3):
            w.step(40)
            psi = _psi(w)
            self.assertAlmostEqual(quasi_energy(psi, ix, self.P, self.m), E0, delta=1e-9 * abs(E0))
            self.assertAlmostEqual(np.sum(np.abs(minus_branch(psi, ix, self.P, self.m))**2), N0, delta=1e-9 * N0)


if __name__ == "__main__":
    unittest.main()
