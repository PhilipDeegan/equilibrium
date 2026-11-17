"""
equilibrium.substrate.continuous: faces-as-anchors mapping of a continuum
(u, A) onto the uniform (u, phi) substrate, and step()'s convergence to the
continuum system u_t = -C0SQ div A, A_t = -grad u under refinement.

Measured 2026-09-24, step() W = 3:1 (R=8 physical, gaussian plane pulse w=1.2, T=0.6):
  h      err_u (staggered)   err_u (unstaggered)
  0.5    7.60e-3             8.74e-3
  0.25   1.91e-3             2.55e-3
  0.125  4.79e-4             8.35e-4
Staggered ratios 3.97, 4.00 (2nd order). phi sampled WITHOUT W_k: 0.369 at
both h=0.5 and 0.25 -- no convergence. step() W = sqrt3:1 (2026-09-28): staggered
6.459e-3 / 1.634e-3 (ratio 3.95); unweighted 0.182 / 0.181. These read the discrete step() and so
run under set_weights("discrete"); the continuous model's default is "iso"
(2026-09-28).
"""

import os
import unittest

import numpy as np

from equilibrium.substrate import continuous as C
from equilibrium.substrate.lattice import SubstrateLattice

# Override via EQ_TEST_N_THREADS=<n> to avoid oversubscribing when other
# n_threads-heavy runs are active on the same machine.
_N_THREADS = int(os.environ.get("EQ_TEST_N_THREADS", 1))


class _Discrete(unittest.TestCase):
    """Runs the class under set_weights(WEIGHTS): "discrete" (step()'s W) or
    "legacy" (W = 3:1, both models before 2026-09-28)."""

    WEIGHTS = "discrete"

    @classmethod
    def setUpClass(cls):
        C.set_weights(cls.WEIGHTS)
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        C.set_weights("iso")
        super().tearDownClass()


G = np.array([[0.3, -1.2, 0.5], [0.7, 0.1, -0.4], [2.0, 0.9, 1.1]])


def _smooth_A(x):
    return np.stack(
        [np.sin(x[:, 1]), np.cos(x[:, 2]) * x[:, 0], np.sin(x[:, 0] + x[:, 1])], axis=1
    )


class TestSampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lat = SubstrateLattice(radius_cells=4.0, spacing=0.5)
        cls.inner = (cls.lat.nb >= 0).all(axis=1)

    def test_constants(self):
        self.assertEqual(C.WEIGHTS_NAME, "iso")
        self.assertAlmostEqual(C.C0SQ, 4.0, places=12)
        try:
            C.set_weights("discrete")
            self.assertAlmostEqual(C.C0SQ, 16.0 / 3.0, places=12)
            C.set_weights("legacy")
            self.assertAlmostEqual(C.C0SQ, 40.0 / 3.0, places=12)
        finally:
            C.set_weights("iso")
        self.assertAlmostEqual(C.spacing_of(self.lat), 0.5, places=12)

    def test_antisymmetric(self):
        lat = self.lat
        phi = C.sample_A(lat, _smooth_A)
        nb = lat.nb
        valid = nb >= 0
        back = phi[np.where(valid, nb, 0), C.K_MIRROR[None, :]]
        self.assertLess(np.abs(np.where(valid, phi + back, 0.0)).max(), 1e-13)

    def test_linear_A_exact(self):
        lat = self.lat
        lat.phi = C.sample_A(lat, lambda x: x @ G.T)
        i = self.inner
        self.assertLess(np.abs(C.reconstruct_A(lat)[i] - lat.pos[i] @ G.T).max(), 1e-12)
        self.assertLess(np.abs(C.div_A(lat)[i] - np.trace(G)).max(), 1e-12)
        self.assertLess(np.abs(C.anchor_residual(lat)[i]).max(), 1e-12)

    def test_residual_second_order(self):
        res = []
        for h in (0.5, 0.25):
            lat = SubstrateLattice(radius_cells=4.0, spacing=h)
            lat.phi = C.sample_A(lat, _smooth_A)
            res.append(np.abs(C.anchor_residual(lat)[(lat.nb >= 0).all(axis=1)]).max())
        self.assertGreater(res[0] / res[1], 3.5)


class TestPlaneWaveConvergence(_Discrete):
    R, W_PULSE, T, X0 = 8.0, 1.2, 0.6, -1.0
    SPACINGS = (0.5, 0.25)

    @classmethod
    def _err(cls, h, weighted=True):
        f = lambda s: np.exp(-0.5 * ((s - cls.X0) / cls.W_PULSE) ** 2)
        u_fn, A_fn = C.plane_wave([1, 0, 0], f)
        lat = SubstrateLattice(radius_cells=cls.R, spacing=h)
        dt = C.stable_dt(lat)
        n = int(np.ceil(cls.T / dt))
        dt = cls.T / n
        C.seed(lat, u_fn, A_fn, dt=dt)
        if not weighted:
            lat.phi = lat.phi / C.W[None, :]
        C.advance(lat, dt, n, n_threads=_N_THREADS)
        r = np.linalg.norm(lat.pos, axis=1)
        m = r < cls.R - C.C0 * cls.T - 1.0
        ue = u_fn(lat.pos, cls.T)
        return float(np.sqrt(np.mean((lat.u[m] - ue[m]) ** 2) / np.mean(ue[m] ** 2)))

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.errs = [cls._err(h) for h in cls.SPACINGS]

    def test_matches_measured(self):
        for e, ref in zip(self.errs, (6.459e-3, 1.634e-3)):
            self.assertAlmostEqual(e / ref, 1.0, delta=0.02)

    def test_second_order(self):
        self.assertGreater(self.errs[0] / self.errs[1], 3.8)

    def test_unweighted_sampling_does_not_converge(self):
        e = [self._err(h, weighted=False) for h in self.SPACINGS]
        self.assertGreater(e[1], 0.1)
        self.assertLess(e[0] / e[1], 1.5)


class TestTransverseWaveNeedsE(_Discrete):
    WEIGHTS = "legacy"  # reproduces the 09-24 / 09-27 AE numbers

    """u=0, A = y f(x-ct)/c, E = y f'(x-ct). step() (E=0) freezes transverse A;
    the EXPERIMENTAL A-E extension (continuous.AE) propagates it at c.
    Measured 2026-09-24 (discrete weights): step() 1.037 / 1.040, AE 7.243e-2
    / 1.819e-2 at h=0.5 / 0.25 (errA, relative L2 of reconstructed A), with
    the symmetric curl_curl_sym (curl="ls"); curl_curl_dec 5.518e-3 /
    1.303e-3. The first, non-symmetric operator gave 2.022e-2 / 5.241e-3
    but had exponentially growing modes (see exp/2026/09/24/e_field).
    Default iso weights (2026-09-28; c = 2, so the pulse travels 1.2 instead
    of 2.19 in T -- not comparable with the discrete numbers): see
    TestTransverseWaveIso."""

    R, W_PULSE, T, X0 = 8.0, 1.2, 0.6, -1.0

    @classmethod
    def _err(cls, h, use_E, curl="ls"):
        c = C.C0
        f = lambda s: np.exp(-0.5 * ((s - cls.X0) / cls.W_PULSE) ** 2)
        fp = lambda s: -(s - cls.X0) / cls.W_PULSE**2 * f(s)
        zero = lambda x: np.zeros(len(x))
        A_fn = lambda x, t: np.stack([zero(x), f(x[:, 0] - c * t) / c, zero(x)], 1)
        E_fn = lambda x: np.stack([zero(x), fp(x[:, 0]), zero(x)], 1)
        lat = SubstrateLattice(radius_cells=cls.R, spacing=h)
        dt = C.stable_dt(lat)
        n = int(np.ceil(cls.T / dt))
        dt = cls.T / n
        C.seed(lat, lambda x, t: zero(x), A_fn, dt=dt)
        if use_E:
            C.AE(lat, C.sample_A(lat, E_fn), curl=curl).advance(dt, n)
        else:
            C.advance(lat, dt, n, n_threads=_N_THREADS)
        m = np.linalg.norm(lat.pos, axis=1) < cls.R - c * cls.T - 1.5
        Ae = A_fn(lat.pos, cls.T + 0.5 * dt)
        return float(np.sqrt(np.mean((C.reconstruct_A(lat)[m] - Ae[m]) ** 2) / np.mean(Ae[m] ** 2)))

    def test_step_freezes_transverse(self):
        """A frozen pulse vs the exact one shifted by d = cT: relative L2
        sqrt(2 (1 - exp(-d^2 / (4 w^2))))."""
        C.set_weights("discrete")  # advance() runs the discrete step()
        try:
            d = C.C0 * self.T
            frozen = np.sqrt(2 * (1 - np.exp(-(d**2) / (4 * self.W_PULSE**2))))
            self.assertAlmostEqual(self._err(0.5, use_E=False) / frozen, 1.0, delta=0.1)
        finally:
            C.set_weights(self.WEIGHTS)

    def test_AE_propagates_second_order(self):
        e = [self._err(h, use_E=True) for h in (0.5, 0.25)]
        self.assertAlmostEqual(e[0] / 7.243e-2, 1.0, delta=0.02)
        self.assertGreater(e[0] / e[1], 3.7)

    def test_AE_dec_propagates_second_order(self):
        """Default curl 'dec' (2026-09-27): 5.518e-3 / 1.303e-3, 13x below LS."""
        e = [self._err(h, use_E=True, curl="dec") for h in (0.5, 0.25)]
        self.assertAlmostEqual(e[0] / 5.518e-3, 1.0, delta=0.02)
        self.assertGreater(e[0] / e[1], 3.7)


class TestTransverseWaveIso(unittest.TestCase):
    """TestTransverseWaveNeedsE's AE runs under the default iso weights.
    Measured 2026-09-28 at h = 0.5 / 0.25: ls 4.181e-2 / 1.065e-2, dec
    2.577e-3 / 6.462e-4 (both 2nd order, ratio 3.92 / 3.99)."""

    ISO_LS, ISO_DEC = (4.1813e-2, 1.0654e-2), (2.5771e-3, 6.4616e-4)

    def _check(self, curl, ref):
        e = [TestTransverseWaveNeedsE._err(h, use_E=True, curl=curl) for h in (0.5, 0.25)]
        self.assertAlmostEqual(e[0] / ref[0], 1.0, delta=0.02)
        self.assertAlmostEqual(e[1] / ref[1], 1.0, delta=0.02)

    def test_ls(self):
        self._check("ls", self.ISO_LS)

    def test_dec(self):
        self._check("dec", self.ISO_DEC)


class TestStaticCurrentBalance(unittest.TestCase):
    """A static A with C0SQ * curl_curl_sym(A) = J exists only for J in
    range(K^T). current_from_magnetisation (J = curl m of a smoothed-disc
    loop magnetisation) is; the anchor-sampled loop current is not -- CG
    on it diverges (its null(K) part is what made the 09-24 induction run's
    self-flux drift). Measured 2026-09-25, R=6, h=0.5, a=2, s=0.6: rel
    residual mag 8.9e-3 / 6.2e-3 (100 / 200 its), sampled 7.0 / 13.1.
    Dynamic check: exp/2026/09/25/induction."""

    A_LOOP, S = 2.0, 0.6

    @classmethod
    def setUpClass(cls):
        from scipy.sparse.linalg import LinearOperator, cg
        from scipy.special import erfc

        lat = SubstrateLattice(radius_cells=6.0, spacing=0.5)
        a, s = cls.A_LOOP, cls.S

        def m_fn(p):
            rh = np.hypot(p[:, 0], p[:, 1])
            gz = np.exp(-0.5 * (p[:, 2] / s) ** 2) / (np.sqrt(2 * np.pi) * s)
            mz = gz * 0.5 * erfc((rh - a) / (np.sqrt(2) * s))
            return np.stack([0 * mz, 0 * mz, mz], 1)

        def J_fn(p):
            rh = np.maximum(np.hypot(p[:, 0], p[:, 1]), 1e-12)
            g = np.exp(-0.5 * ((rh - a) ** 2 + p[:, 2] ** 2) / s**2) / (2 * np.pi * s**2)
            return np.stack([-p[:, 1] / rh * g, p[:, 0] / rh * g, 0 * g], 1)

        shape = lat.phi.shape

        def op(v):
            return (C.C0SQ * C.curl_curl_sym(lat, v.reshape(shape))).ravel()

        L = LinearOperator((lat.phi.size,) * 2, matvec=op, dtype=np.float64)

        def res(J, n):
            b = J.ravel()
            x, _ = cg(L, b, rtol=1e-12, atol=0.0, maxiter=n)
            return float(np.linalg.norm(op(x) - b) / np.linalg.norm(b))

        Jm = C.current_from_magnetisation(lat, m_fn(lat.pos))
        Js = C.sample_A(lat, J_fn)
        cls.mag = [res(Jm, n) for n in (100, 200)]
        cls.sampled = [res(Js, n) for n in (100, 200)]

    def test_magnetisation_current_balances(self):
        self.assertLess(self.mag[1], 1e-2)
        self.assertLess(self.mag[1], self.mag[0])

    def test_sampled_current_has_no_static_balance(self):
        self.assertGreater(self.sampled[0], 1.0)
        self.assertGreater(self.sampled[1], self.sampled[0])


class TestCartanTransport(unittest.TestCase):
    """cartan_transport = -i_v(da) on the Delaunay triangles (exp/2026/10/13/cartan):
    antisymmetric, energy-orthogonal to step()'s sum phi^2 / 2, zero where every
    triangle at a face is curl-free, and exactly -W (Omega x u) . d for uniform
    curl and flow at any spacing."""

    @classmethod
    def setUpClass(cls):
        cls.lats = {h: SubstrateLattice(radius_cells=6.0, spacing=h) for h in (0.5, 0.25)}
        cls.rng = np.random.default_rng(3)

    def _inner(self, lat):
        full = np.all(lat.nb >= 0, 1)
        return full & np.all(full[np.where(lat.nb >= 0, lat.nb, 0)], 1)

    def _random(self, lat):
        x = np.where(lat.nb >= 0, self.rng.standard_normal(lat.phi.shape), 0.0)
        return C._antisym(lat, x)

    def test_antisymmetric_and_energy_orthogonal(self):
        lat = self.lats[0.5]
        phi = self._random(lat)
        out = C.cartan_transport(lat, phi)
        valid = lat.nb >= 0
        back = out[np.where(valid, lat.nb, 0), C.K_MIRROR[None, :]]
        self.assertLess(np.abs(np.where(valid, out + back, 0.0)).max(), 1e-12)
        self.assertLess(abs(np.sum(phi * out)) / np.sum(np.abs(phi * out)), 1e-13)

    def test_zero_on_irrotational(self):
        lat = self.lats[0.5]
        u = self.rng.standard_normal(lat.N)
        grad = np.where(lat.nb >= 0, C.W[None, :] * (u[np.where(lat.nb >= 0, lat.nb, 0)] - u[:, None]), 0.0)
        self.assertLess(np.abs(C.cartan_transport(lat, grad)).max(), 1e-12)
        r = np.linalg.norm(lat.pos, axis=1)
        local = np.where((r < 1.0)[:, None] | (r[np.where(lat.nb >= 0, lat.nb, 0)] < 1.0), self._random(lat), 0.0)
        out = C.cartan_transport(lat, C._antisym(lat, 0.5 * local) + grad)
        far = (r > 1.0 + 4 * C.spacing_of(lat))[:, None] & np.ones((1, 14), bool)
        self.assertGreater(np.abs(out).max(), 1e-3)
        self.assertLess(np.abs(out[far]).max(), 1e-12)

    def test_uniform_curl_and_flow(self):
        Om, u0 = np.array([0.3, -0.5, 0.7]), np.array([0.4, 0.9, -0.2])
        want = -C.W[None, :] * (C.D_HAT @ np.cross(Om, u0))[None, :]
        for h, lat in self.lats.items():
            pO = C.sample_A(lat, lambda p: 0.5 * np.cross(Om[None, :], p))
            pu = C.sample_A(lat, lambda p: np.broadcast_to(u0, p.shape))
            T = C.cartan_transport
            cross = T(lat, pO + pu) - T(lat, pO) - T(lat, pu)
            m = self._inner(lat)
            self.assertLess(np.abs(cross[m] - want).max(), 1e-12, h)


if __name__ == "__main__":
    unittest.main()
