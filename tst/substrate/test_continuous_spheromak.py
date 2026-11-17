"""Taylor spheromak (curl A = lam A inside radius a, lam a = 4.4934) under step() + Cartan in the periodic continuous
box, first turnover (a / v): energy held, seed near Beltrami, structure kept, no content in the non-vector face DOF.
The breakdown at ~3.4 turnovers (content poured into those DOF, onset converged a 9 -> 20) is too long for CI; see
exp/2026/10/22/spheromak/status in `now` 10-22."""
import os
import unittest

import numpy as np
from scipy.special import spherical_jn

from equilibrium.substrate import continuous as C
from equilibrium.substrate.walk import cpp_box

_N_THREADS = int(os.environ.get("EQ_TEST_N_THREADS", 1))


def taylor(X, a, off):
    lam = 4.4934094579 / a
    x = X - off
    r = np.linalg.norm(x, axis=1)
    rs = np.maximum(r, 1e-9)
    ct, st = x[:, 2] / rs, np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2) / rs
    rho = np.maximum(np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2), 1e-12)
    cp, sp = x[:, 0] / rho, x[:, 1] / rho
    k = lam * rs
    j1, j1p = spherical_jn(1, k), spherical_jn(1, k, derivative=True)
    Br, Bt, Bp = 2 * j1 / k * ct, -(j1 / k + j1p) * st, j1 * st
    B = np.stack([Br * st * cp + Bt * ct * cp - Bp * sp, Br * st * sp + Bt * ct * sp + Bp * cp, Br * ct - Bt * st], 1)
    return np.where((r < a)[:, None], B, 0.0)


class TestSpheromak(unittest.TestCase):
    def test_taylor_is_beltrami(self):
        a, off = 6.0, np.zeros(3)
        X = (np.random.default_rng(1).random((200, 3)) - 0.5) * 10
        X = X[np.linalg.norm(X, axis=1) < 0.9 * a]
        e = 1e-5

        def d(i, j):
            dx = np.zeros(3)
            dx[j] = e
            return (taylor(X + dx, a, off)[:, i] - taylor(X - dx, a, off)[:, i]) / (2 * e)

        curl = np.stack([d(2, 1) - d(1, 2), d(0, 2) - d(2, 0), d(1, 0) - d(0, 1)], 1)
        B = taylor(X, a, off)
        self.assertLess(np.abs(curl - 4.4934094579 / a * B).max() / np.abs(B).max(), 1e-8)

    def test_first_turnover(self):
        C.spacing_of, keep = (lambda lat: 1.0), C.spacing_of
        try:
            a, P, vmax = 6.0, 64, 0.3
            side = P / np.sqrt(3)
            off = np.array([0.13, 0.07, 0.11])
            alat = cpp_box(P, n_threads=_N_THREADS)
            ac = C.AnchorCluster(alat)
            phi = C.sample_A(ac, lambda X: taylor((X - off + side / 2) % side - side / 2 + off, a, off))
            for _ in range(int(round((a / 9) ** 2 / 0.02))):
                phi = phi - 0.02 * ac.curl_curl_dec(phi)
            phi = C.project_div_free(ac, phi)
            phi *= vmax / np.abs(C.reconstruct_A(ac, phi)).max()
            x = (ac.pos - off + side / 2) % side - side / 2
            m = np.linalg.norm(x, axis=1) < a
            A, B = C.reconstruct_A(ac, phi), C.curl_A(ac, phi)
            self.assertGreater(np.sum(A[m] * B[m]) / np.sqrt(np.sum(A[m] ** 2) * np.sum(B[m] ** 2)), 0.95)
            ac.u, ac.phi = np.zeros(ac.N), phi
            dt = 0.4 / float(np.sum(C.W**2))
            E = lambda prev, ph: np.sum(ac.u**2) + 0.5 * np.sum(prev * ph)
            prev = ac.phi
            for _ in range(int(round(a / vmax / dt)) - 1):
                alat.step_cartan(0, dt, 1.0, 2)
            prev = ac.phi
            alat.step_cartan(0, dt, 1.0, 2)
            ph = ac.phi
            self.assertLess(abs(E(prev, ph) / (0.5 * np.sum(phi**2)) - 1), 1e-6)
            self.assertGreater(np.sum(ph * phi) / np.sqrt(np.sum(ph**2) * np.sum(phi**2)), 0.97)
            odd = 0.5 * (ph / C.W - (ph / C.W)[:, C.K_MIRROR])
            self.assertLess(np.sum(C.anchor_residual(ac, ph) ** 2) / np.sum(odd**2), 0.01)
        finally:
            C.spacing_of = keep


if __name__ == "__main__":
    unittest.main()
