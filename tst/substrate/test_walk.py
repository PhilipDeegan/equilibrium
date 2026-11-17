import os
import unittest

import numpy as np

from equilibrium.substrate.walk import (AS, HS, CppWalk, Walk, cpp_box, cpp_sphere, healing_g, solid_angle, sponge_factor,
                                        to_complex, to_real)

_N_THREADS = int(os.environ.get("EQ_TEST_N_THREADS", 1))


def _ring(pos, r0=4.0, xi=1.5, z0=0.0):
    x, y, z = pos.T
    rho = np.hypot(x, y)
    d1, d2 = np.hypot(rho - r0, z - z0), np.hypot(rho + r0, z - z0)
    ph = np.arctan2(z - z0, rho - r0) - np.arctan2(z - z0, rho + r0)
    return d1 / np.sqrt(d1**2 + xi**2) * d2 / np.sqrt(d2**2 + xi**2) * np.exp(1j * ph)


def _periodic_bcc(nn):
    """BCC sites of an nn^3 integer grid (x, y, z all even or all odd), nb (N, 8) = the site at x + HS[k] (wrapped)."""
    g = np.stack(np.meshgrid(*(np.arange(nn),) * 3, indexing="ij"), -1).reshape(-1, 3)
    pos = g[(g % 2 == g[:, :1] % 2).all(1)]
    idx = {tuple(p): i for i, p in enumerate(pos)}
    nb = np.array([[idx[tuple((p + h) % nn)] for h in HS.astype(int)] for p in pos])
    return (pos - nn // 2).astype(float), nb  # centred: face_of_hops reads the most central cell


def _symbol(k, m, sign_k=1.0):
    n = np.sqrt(1 - m * m)
    Wk = np.einsum("h,hij->ij", np.exp(1j * sign_k * (k @ HS.T)), AS)
    I = np.eye(2)
    return np.block([[n * Wk, 1j * m * I], [1j * m * I, n * Wk.conj().T]])


class TestWalkSymbol(unittest.TestCase):
    def test_plane_wave(self):
        """periodic box, g = 0: one plain tick = S(k); plain + parity tick = B S(-k) B S(k) (B = L / R swap)."""
        nn, m = 8, 0.8
        pos, nb = _periodic_bcc(nn)
        Z, I = np.zeros((2, 2)), np.eye(2)
        B = np.block([[Z, I], [I, Z]])
        rng = np.random.default_rng(0)
        for q in [(1, 0, 0), (1, 2, 0), (1, 1, 3), (2, -1, 1)]:
            k = 2 * np.pi * np.array(q, float) / nn
            u = rng.normal(size=4) + 1j * rng.normal(size=4)
            psi = np.exp(1j * pos @ k)[:, None] * u[None, :]
            for parity, ticks, U in [(False, 1, _symbol(k, m)), (True, 2, B @ _symbol(k, m, -1) @ B @ _symbol(k, m))]:
                w = Walk(pos, nb, m=m, parity=parity)
                w.set_state(to_real(psi[:, :2]), to_real(psi[:, 2:]), t=0)
                w.step(ticks)
                out = np.concatenate([to_complex(w.L), to_complex(w.R)], 1)
                np.testing.assert_allclose(out, np.exp(1j * pos @ k)[:, None] * (U @ u)[None, :], rtol=0, atol=1e-12)


class TestSolidAngle(unittest.TestCase):
    @staticmethod
    def _numpy(P, X, apex):
        A, B = P, np.roll(P, -1, 0)
        a, b, c = apex[None, None] - X[:, None], A[None] - X[:, None], B[None] - X[:, None]
        la, lb, lc = (np.linalg.norm(v, axis=2) for v in (a, b, c))
        num = np.einsum("nmk,nmk->nm", a, np.cross(b, c))
        dot = lambda u, v: np.einsum("nmk,nmk->nm", u, v)
        den = la * lb * lc + dot(a, b) * lc + dot(a, c) * lb + dot(b, c) * la
        return 2 * np.arctan2(num, den).sum(1)

    def test_matches_numpy_with_shifts(self):
        rng = np.random.default_rng(3)
        s = np.linspace(0, 2 * np.pi, 97, endpoint=False)
        P = np.stack([(9 + 3 * np.cos(5 * s)) * np.cos(s), (9 + 3 * np.cos(5 * s)) * np.sin(s), 3 * np.sin(5 * s)], 1)
        X = (rng.random((500, 3)) - 0.5) * 40
        apex = np.array([0.3183, -0.2718, 0.5772])
        shifts = np.array([[0, 0, 0], [40.0, 0, 0], [0, -40.0, 0], [13.0, 7.0, -40.0]])
        ref = sum(self._numpy(P + d, X, apex) for d in shifts)
        np.testing.assert_allclose(solid_angle(P, X, apex, shifts, n_threads=_N_THREADS), ref, rtol=0, atol=1e-11)
        np.testing.assert_allclose(solid_angle(P, X, apex, n_threads=_N_THREADS), self._numpy(P, X, apex), rtol=0, atol=1e-11)

    def test_circle_on_axis(self):
        R, z = 5.0, np.array([-7.0, -1.5, 2.0, 11.0])
        s = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
        P = np.stack([R * np.cos(s), R * np.sin(s), 0 * s], 1)
        X = np.stack([0 * z, 0 * z, z], 1)
        Om = solid_angle(P, X, np.array([0.3183, -0.2718, 50.0]), n_threads=_N_THREADS)
        exact = -2 * np.pi * np.sign(z) * (1 - np.abs(z) / np.hypot(z, R))
        np.testing.assert_allclose(np.exp(0.5j * Om), np.exp(0.5j * exact), atol=1e-5)


class TestWalkBox(unittest.TestCase):
    def test_periodic_box(self):
        """build_bcc_box_periodic: every face wired, C++ plane wave on the box == symbol (plain + parity tick), and the
        uniform background with the density turn stays stationary."""
        P, m = 12, 0.8
        alat = cpp_box(P, n_threads=max(2, _N_THREADS))
        pos, nb = np.array(alat[0].leaf_pos), np.array(alat.leaf_nb_j(0))
        self.assertEqual(len(pos), 2 * (P // 2) ** 3)
        self.assertTrue((nb >= 0).all())
        Z, I = np.zeros((2, 2)), np.eye(2)
        B = np.block([[Z, I], [I, Z]])
        x = pos * np.sqrt(3)  # integer grid coordinates
        k = 2 * np.pi * np.array([1.0, 2.0, -1.0]) / P
        u = np.array([0.3 + 0.1j, -0.2 + 0.5j, 0.7 - 0.4j, 0.1 + 0.2j])
        psi = np.exp(1j * x @ k)[:, None] * u[None, :]
        w = CppWalk(alat, 0, m=m, parity=True)
        w.set_state(to_real(psi[:, :2]), to_real(psi[:, 2:]), t=0)
        w.step(2)
        L, R = w.state()
        out = np.concatenate([to_complex(L), to_complex(R)], 1)
        U = B @ _symbol(k, m, -1) @ B @ _symbol(k, m)
        np.testing.assert_allclose(out, np.exp(1j * x @ k)[:, None] * (U @ u)[None, :], rtol=0, atol=1e-12)
        w2 = CppWalk(cpp_box(P, n_threads=1), 0, m=m, g=0.05, parity=True)
        w2.seed(np.ones(len(pos), complex))
        w2.step(101)
        np.testing.assert_allclose(w2.density(), 1.0, rtol=0, atol=1e-12)


class TestWalkCpp(unittest.TestCase):
    def test_matches_numpy_parity(self):
        """parity-image alternation: C++ == numpy, and the uniform background stays stationary."""
        alat = cpp_sphere(10.0, n_threads=max(2, _N_THREADS))
        pos, nb = np.array(alat[0].leaf_pos), np.array(alat.leaf_nb_j(0))
        f, _ = sponge_factor(pos, sponge=0.25, damp=0.2)
        kw = dict(m=0.8, g=healing_g(0.8, 1.5), f=f, parity=True)
        ref, cpp = Walk(pos, nb, **kw), CppWalk(alat, 0, **kw)
        chi = _ring(pos)
        ref.seed(chi)
        cpp.seed(chi)
        ref.step(61)
        cpp.step(61)
        L, R = cpp.state()
        np.testing.assert_allclose(L, ref.L, rtol=0, atol=1e-12)
        np.testing.assert_allclose(R, ref.R, rtol=0, atol=1e-12)
        cpp.seed(np.ones(len(pos), complex))
        cpp.step(101)
        np.testing.assert_allclose(cpp.density(), 1.0, rtol=0, atol=1e-12)

    def test_matches_numpy(self):
        """C++ step_walk == numpy reference (hop, mass, density turn, background ghosts, sponge), 1 and n threads."""
        for n_threads in sorted({1, max(2, _N_THREADS)}):
            alat = cpp_sphere(10.0, n_threads=n_threads)
            pos, nb = np.array(alat[0].leaf_pos), np.array(alat.leaf_nb_j(0))
            f, _ = sponge_factor(pos, sponge=0.25, damp=0.2)
            kw = dict(m=0.5, g=healing_g(0.5, 1.5), f=f)
            ref, cpp = Walk(pos, nb, **kw), CppWalk(alat, 0, **kw)
            chi = _ring(pos)
            ref.seed(chi)
            cpp.seed(chi)
            ref.step(60)
            cpp.step(60)
            self.assertEqual(cpp.t, ref.t)
            L, R = cpp.state()
            np.testing.assert_allclose(L, ref.L, rtol=0, atol=1e-12)
            np.testing.assert_allclose(R, ref.R, rtol=0, atol=1e-12)
            np.testing.assert_allclose(cpp.density(), ref.density(), rtol=0, atol=1e-12)

    def test_background_stationary(self):
        """uniform background: density stays rho0 everywhere (ghosts and sponge target turn with it)."""
        alat = cpp_sphere(8.0, n_threads=_N_THREADS)
        pos = np.array(alat[0].leaf_pos)
        f, _ = sponge_factor(pos)
        w = CppWalk(alat, 0, m=0.5, g=0.05, rho0=1.0, f=f)
        w.seed(np.ones(len(pos), complex))
        w.step(200)
        np.testing.assert_allclose(w.density(), 1.0, rtol=0, atol=1e-12)
        bgL, bgR = w.bg(w.t)
        L, R = w.state()
        np.testing.assert_allclose(L, np.broadcast_to(bgL, L.shape), rtol=0, atol=1e-11)
        np.testing.assert_allclose(R, np.broadcast_to(bgR, R.shape), rtol=0, atol=1e-11)


# clock unused outside this test since 10-08; goes with set_walk_clock when removed
class TestWalkClock(unittest.TestCase):
    """local clock sigma = 1 + beta (rho0 - rho1) / rho0 (cell.L / L0 vs depletion)."""

    def test_matches_numpy(self):
        """C++ == numpy with the clock on (hop, mass, clock coin, scaled density turn, ghosts, sponge, parity). 12 ticks:
        beta 0.5 makes the background focusing (d omega / d rho = g - beta (om0 + g) < 0), round-off grows ~0.3 / tick."""
        for beta in (0.5, -0.5):
            alat = cpp_sphere(10.0, n_threads=max(2, _N_THREADS))
            pos, nb = np.array(alat[0].leaf_pos), np.array(alat.leaf_nb_j(0))
            f, _ = sponge_factor(pos, sponge=0.25, damp=0.2)
            kw = dict(m=0.8, g=healing_g(0.8, 1.5), f=f, parity=True, beta=beta)
            ref, cpp = Walk(pos, nb, **kw), CppWalk(alat, 0, **kw)
            chi = _ring(pos)
            ref.seed(chi)
            cpp.seed(chi)
            ref.step(12)
            cpp.step(12)
            L, R = cpp.state()
            np.testing.assert_allclose(L, ref.L, rtol=0, atol=1e-12)
            np.testing.assert_allclose(R, ref.R, rtol=0, atol=1e-12)

    def test_norm_and_background(self):
        """periodic box: total norm conserved to round-off with a vortex ring and the clock on; the uniform background
        (sigma = 1) stays stationary."""
        P = 16
        alat = cpp_box(P, n_threads=max(2, _N_THREADS))
        pos = np.array(alat[0].leaf_pos)
        w = CppWalk(alat, 0, m=0.8, g=healing_g(0.8, 1.5), parity=True, beta=0.7)
        w.seed(_ring(pos, r0=3.0, xi=1.2))
        n0 = w.density().sum()
        w.step(200)
        self.assertLess(abs(w.density().sum() - n0) / n0, 1e-12)
        w.seed(np.ones(len(pos), complex))
        w.step(101)
        np.testing.assert_allclose(w.density(), 1.0, rtol=0, atol=1e-12)

    def test_uniform_rate(self):
        """a uniform k = 0 state at density rho turns at sigma (om0 + g rho) per tick (the rest rate scaled by sigma)."""
        P, m, g, beta, rho = 12, 0.8, 0.05, 0.6, 0.49
        alat = cpp_box(P, n_threads=1)
        w = CppWalk(alat, 0, m=m, g=g, parity=True, beta=beta)
        w.seed(np.full(len(alat[0].leaf_pos), np.sqrt(rho), complex))
        a0 = w.amp()[0]
        w.step(3)
        sig = 1 + beta * (1 - rho)
        np.testing.assert_allclose(np.angle(w.amp()[0] / a0), np.angle(np.exp(-3j * sig * (w.om0 + g * rho))), atol=1e-12)
        np.testing.assert_allclose(w.density(), rho, rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
