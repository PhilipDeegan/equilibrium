"""
C++ lattice.continuous.AdaptiveLattice (level 0) against its references:
- step          == discrete::step at level 0 under set_weights('discrete'),
                   to round-off (plumbing only: the continuous model is not
                   expected to equal the discrete rule, and its default
                   weights are 'iso', sqrt2:1; not bit for bit: separate
                   compiled copies contract to FMA differently)
- curl_curl_sym == continuous.curl_curl_sym (numpy), to round-off
- step_ae       == continuous.AE (numpy), to round-off, with and without J
                (LS curl here; default is 'dec', tested in TestDecCurl)
- step_ae energy  E-B energy 1/2 e.e + 1/2 phi.K phi (K = C0SQ curl_curl_sym),
                staggered form + dt/2 e.K phi: conserved before the wall,
                bounded after (no secular growth)
- S grad        curl_curl_sym of a discrete gradient is NOT zero; for smooth
                u it vanishes at 2nd order (grid-scale u leaks fully into B)
- dispersion    u follows the exact leapfrog step() dispersion; E/B (LS
                reconstruction) 2nd order, >5x more dispersive than u
- DEC curl      curl_curl_dec == numpy, step_ae('dec') == AE(curl='dec');
                curl(grad) = 0; 'dec' (step()'s star) output has zero
                step()-divergence; >10x less dispersive
- face medium   zero medium == none; uniform sigma medium == the same sigma
                from lam; values must agree across each face anchor
"""

import os
import unittest

import numpy as np

from equilibrium import lattice as _lat
from equilibrium.substrate import continuous as C

# Override via EQ_TEST_N_THREADS=<n> to avoid oversubscribing when other
# n_threads-heavy runs are active on the same machine.
_N_THREADS = int(os.environ.get("EQ_TEST_N_THREADS", 1))



def _random_state(ac, seed):
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(ac.N)
    phi = C._antisym(ac, rng.standard_normal((ac.N, 14)))
    return u, phi


class TestStepMatchesDiscreteLevel0(unittest.TestCase):
    def test_same_rule(self):
        R, n = 6.0, 40
        rng = np.random.default_rng(1)
        lats = []
        C.set_weights("discrete")
        self.addCleanup(C.set_weights, "iso")
        for mod in (_lat.discrete, _lat.continuous):
            alat = mod.AdaptiveLattice()
            if mod is _lat.continuous:
                alat.set_weights("discrete")
            mod.build_bcc_sphere(alat[0], R, d1=1.0)
            alat.rebuild(0)
            alat.set_n_threads(_N_THREADS)
            lats.append(alat)
        cl0 = lats[0][0]
        N = cl0.n_leaves
        u = rng.standard_normal(N)
        lam = rng.uniform(0.0, 2.0, N)
        ac = C.AnchorCluster(lats[1], 0)
        phi = 0.5 * C._antisym(ac, rng.standard_normal((N, 14))).T
        for alat in lats:
            cl = alat[0]
            cl.set_leaf_u(u)
            cl.set_leaf_lam(lam)
            cl.set_leaf_phi(np.ascontiguousarray(phi))
            for _ in range(n):
                alat.step(0, 0.004)
        a, b = lats[0][0], lats[1][0]
        for x, y in ((a.leaf_u, b.leaf_u), (a.leaf_phi, b.leaf_phi)):
            x, y = np.array(x), np.array(y)
            self.assertLess(np.abs(x - y).max() / np.abs(x).max(), 1e-13)


class TestAnchorOperatorsMatchNumpy(unittest.TestCase):
    R, H = 5.0, 0.5

    def test_curl_curl_sym(self):
        ac = C.anchor_cluster(self.R, self.H, n_threads=_N_THREADS)
        _, phi = _random_state(ac, 2)
        a = C.curl_curl_sym(ac, phi)
        b = ac.curl_curl_sym(phi)
        self.assertLess(np.abs(a - b).max() / np.abs(a).max(), 1e-13)

    def _run_pair(self, with_J, n=30):
        ac = C.anchor_cluster(self.R, self.H, n_threads=_N_THREADS)
        u, phi = _random_state(ac, 3)
        e0 = C._antisym(ac, np.random.default_rng(4).standard_normal((ac.N, 14)))
        J = C._antisym(ac, np.random.default_rng(5).standard_normal((ac.N, 14))) if with_J else None
        dt = C.stable_dt(ac)

        class Ref:
            pass

        ref = Ref()
        ref.pos, ref.nb, ref.N = ac.pos, ac.nb, ac.N
        ref.u, ref.phi = u.copy(), phi.copy()
        ref.fixed_u = np.zeros(ac.N, bool)
        ref.fixed_phi = np.zeros((ac.N, 14), bool)
        ae = C.AE(ref, e0.copy(), curl="ls")
        ac.set_curl("ls")
        ac.u, ac.phi, ac.e = u, phi, e0
        for _ in range(n):
            ae.step(dt, J)
            ac.step_ae(dt, J)
        return ref, ae, ac

    def _check(self, with_J):
        ref, ae, ac = self._run_pair(with_J)
        for a, b in ((ref.u, ac.u), (ref.phi, ac.phi), (ae.e, ac.e)):
            self.assertLess(np.abs(a - b).max() / np.abs(a).max(), 1e-11)

    def test_step_ae(self):
        self._check(with_J=False)

    def test_step_ae_with_current(self):
        self._check(with_J=True)




def _rot_gauss(X):
    g = np.exp(-0.5 * np.sum(X**2, -1))
    return np.stack([X[..., 1] * g, -X[..., 0] * g, 0 * g], -1)


def _face_sample(ac, h, A_fn):
    return np.where(ac.nb >= 0, np.einsum("k,nkc,kc->nk", C.W, A_fn(C.face_midpoints(ac)), C.D_HAT * h), 0.0)


class TestStepAEEnergy(unittest.TestCase):
    def test_bounded(self):
        h = 0.5
        ac = C.anchor_cluster(6.0, h, n_threads=_N_THREADS)
        ac.set_curl("ls")
        ac.phi = _face_sample(ac, h, _rot_gauss)
        dt = C.stable_dt(ac)

        def H(ac):
            phi, e = ac.phi, ac.e
            K = C.C0SQ * ac.curl_curl_sym(phi)
            return 0.5 * np.sum(e**2) + 0.5 * np.sum(phi * K) + 0.5 * dt * np.sum(e * K)

        h0, early, late = H(ac), 0.0, 0.0
        for i in range(1600):
            ac.step_ae(dt)
            if i % 20 == 0:
                d = abs(H(ac) - h0) / h0
                if i < 200:
                    early = max(early, d)
                late = max(late, d)
        self.assertLess(early, 2e-3)
        self.assertLess(late, 2e-2)

    def test_bounded_dec(self):
        """Same with the default 'dec' curl; its metric is w_k = *1_k / W_k^2
        (hex : square = 1 : 1 for the 9:1 star, so Euclidean)."""
        h = 0.5
        ac = C.anchor_cluster(6.0, h, n_threads=_N_THREADS)
        self.assertEqual(ac.alat.curl, "dec")
        ac.phi = _face_sample(ac, h, _rot_gauss)
        dt = C.stable_dt(ac)

        def H(ac):
            phi, e = ac.phi, ac.e
            K = C.C0SQ * ac.curl_curl_dec(phi)
            return 0.5 * np.sum(e**2) + 0.5 * np.sum(phi * K) + 0.5 * dt * np.sum(e * K)

        h0, early, late = H(ac), 0.0, 0.0
        for i in range(1600):
            ac.step_ae(dt)
            if i % 20 == 0:
                d = abs(H(ac) - h0) / h0
                if i < 200:
                    early = max(early, d)
                late = max(late, d)
        self.assertLess(early, 2e-3)
        self.assertLess(late, 2e-2)


class TestCurlCurlOfGradient(unittest.TestCase):
    def _leak(self, h):
        ac = C.anchor_cluster(4.0, h, n_threads=_N_THREADS)
        x, nb = ac.pos, ac.nb
        u = np.exp(-0.5 * np.sum(x**2, 1))
        g = np.where(nb >= 0, C.W[None, :] * (u[np.maximum(nb, 0)] - u[:, None]), 0.0)
        inn = np.linalg.norm(x, axis=1) < 2.5
        return np.abs(ac.curl_curl_sym(g)[inn]).max() / np.abs(g[inn]).max()

    def test_smooth_second_order(self):
        a, b = self._leak(0.5), self._leak(0.25)
        self.assertGreater(a / b, 2.5)

    def test_grid_scale_not_null(self):
        ac = C.anchor_cluster(4.0, 0.5, n_threads=1)
        u = np.random.default_rng(0).standard_normal(ac.N)
        nb = ac.nb
        g = np.where(nb >= 0, C.W[None, :] * (u[np.maximum(nb, 0)] - u[:, None]), 0.0)
        self.assertGreater(np.abs(ac.curl_curl_sym(g)).max() / np.abs(g).max(), 0.1)


def _packet_dc_c(h, mode, n=(1.0, 0.0, 0.0), R=10.0, travel=3.0, lam=4.0, s=1.2, rho_m=3.0, curl="ls"):
    """Phase-speed error dc/c of a plane-wave packet on step_ae: transverse
    (T, from E) or longitudinal/step() sector (L, from u). Returns (measured,
    exact leapfrog step() prediction for L)."""
    c, k = C.C0, 2 * np.pi / lam
    n = np.asarray(n, float) / np.linalg.norm(n)
    ref = np.eye(3)[int(np.argmin(np.abs(n)))]
    pol = ref - (ref @ n) * n
    pol /= np.linalg.norm(pol)
    xi0 = -travel / 2

    def f(xi, d=0):
        ch = xi - xi0
        g = np.exp(-0.5 * ch**2 / s**2)
        return np.cos(k * ch) * g if d == 0 else (-k * np.sin(k * ch) - ch / s**2 * np.cos(k * ch)) * g

    ac = C.anchor_cluster(R, h, n_threads=_N_THREADS)
    ac.set_curl(curl)
    dt = C.stable_dt(ac)
    nst = int(np.ceil(travel / c / dt))
    dt = travel / c / nst
    xn = ac.pos @ n
    rho = np.linalg.norm(ac.pos - xn[:, None] * n, axis=1)
    idx = np.nonzero((rho < rho_m) & (np.abs(xn) < travel / 2 + 4 * s))[0]
    ph = np.exp(-1j * k * xn[idx])
    if mode == "T":
        ac.phi = C.sample_A(ac, lambda y: f(y @ n - 0.5 * c * dt)[:, None] * pol)
        ac.e = C.sample_A(ac, lambda y: (c * f(y @ n, 1))[:, None] * pol)
    else:
        ac.u = f(ac.pos @ n)
        ac.phi = C.sample_A(ac, lambda y: (f(y @ n - 0.5 * c * dt) / c)[:, None] * n)
    for _ in range(nst):
        ac.step_ae(dt)
    xi = xn[idx] - c * nst * dt
    if mode == "T":
        q, qe = C.reconstruct_A_at(ac, idx, ac.e_rows(idx)) @ pol, c * f(xi, 1)
    else:
        q, qe = np.asarray(ac.u)[idx], f(xi)
    dc = -np.angle((q * ph).sum() / (qe * ph).sum()) / (c * k * nst * dt)
    S = float(np.sum(C.W**2 * (1 - np.cos(h * k * (C.D_HAT @ n))))) / h**2
    return dc, (2 / dt) * np.arcsin(0.5 * dt * np.sqrt(S)) / (c * k) - 1


class TestPlaneWaveDispersion(unittest.TestCase):
    """exp/2026/09/27/dispersion: u (step() face difference) follows the exact
    leapfrog step() dispersion; E/B (LS reconstruction in curl_curl_sym) is
    2nd order but ~9x more dispersive along a cubic axis (~4x along [111])."""

    def test_u_matches_step_prediction(self):
        dc, pred = _packet_dc_c(0.5, "L")
        self.assertLess(abs(dc / pred - 1), 0.02)

    def test_transverse_second_order(self):
        a, _ = _packet_dc_c(0.5, "T")
        b, _ = _packet_dc_c(0.25, "T")
        self.assertLess(a, 0)
        self.assertGreater(a / b, 3.5)
        self.assertLess(a / b, 4.5)

    def test_transverse_slower_than_u(self):
        t, _ = _packet_dc_c(0.5, "T")
        _, pred = _packet_dc_c(0.5, "L")
        self.assertGreater(t / pred, 5.0)


class TestDecCurl(unittest.TestCase):
    """curl_curl_dec (C++) == continuous.curl_curl_dec (numpy); step_ae with
    set_curl('dec') == continuous.AE(curl='dec'); curl(grad) = 0 for both
    stars; output divergence 0 exactly in step()'s sum_k W_k ('dec') or the
    unweighted sum ('dec_geo'); transverse packet 2nd order and >10x less
    dispersive than curl_curl_sym."""

    def test_matches_numpy(self):
        ac = C.anchor_cluster(5.0, 0.5, n_threads=_N_THREADS)
        _, phi = _random_state(ac, 6)
        for star in ("dec", "dec_geo"):
            a = C.curl_curl_dec(ac, phi, star)
            b = ac.curl_curl_dec(phi, star)
            self.assertLess(np.abs(a - b).max() / np.abs(a).max(), 1e-13)

    def test_step_ae_matches_numpy(self):
        ac = C.anchor_cluster(5.0, 0.5, n_threads=_N_THREADS)
        ac.set_curl("dec")
        u, phi = _random_state(ac, 7)
        e0 = C._antisym(ac, np.random.default_rng(8).standard_normal((ac.N, 14)))
        J = C._antisym(ac, np.random.default_rng(9).standard_normal((ac.N, 14)))
        dt = C.stable_dt(ac)

        class Ref:
            pass

        ref = Ref()
        ref.pos, ref.nb, ref.N = ac.pos, ac.nb, ac.N
        ref.u, ref.phi = u.copy(), phi.copy()
        ref.fixed_u = np.zeros(ac.N, bool)
        ref.fixed_phi = np.zeros((ac.N, 14), bool)
        ae = C.AE(ref, e0.copy(), curl="dec")
        ac.u, ac.phi, ac.e = u, phi, e0
        for _ in range(30):
            ae.step(dt, J)
            ac.step_ae(dt, J)
        for a, b in ((ref.u, ac.u), (ref.phi, ac.phi), (ae.e, ac.e)):
            self.assertLess(np.abs(a - b).max() / np.abs(a).max(), 1e-11)

    def test_exact_identities(self):
        ac = C.anchor_cluster(4.0, 0.5, n_threads=_N_THREADS)
        nb = ac.nb
        u = np.random.default_rng(10).standard_normal(ac.N)
        g = np.where(nb >= 0, C.W[None, :] * (u[np.maximum(nb, 0)] - u[:, None]), 0.0)
        inn = np.linalg.norm(ac.pos, axis=1) < 2.5
        _, phi = _random_state(ac, 11)
        for star in ("dec", "dec_geo"):
            r, _ = C.dec_star(star)
            div_w = np.where(np.arange(14) < 8, r, 1.0) / C.W  # sum_k (*1_k / W_k) out_k
            if star == "dec":
                np.testing.assert_allclose(div_w, C.W, rtol=1e-14)  # = step()'s divergence
            self.assertLess(np.abs(ac.curl_curl_dec(g, star)[inn]).max() / np.abs(g[inn]).max(), 1e-12)
            cc = ac.curl_curl_dec(phi, star)
            self.assertLess(np.abs((cc @ div_w)[inn]).max() / np.abs(cc[inn]).max(), 1e-12)

    def test_transverse_dispersion(self):
        a, _ = _packet_dc_c(0.5, "T", curl="dec")
        b, _ = _packet_dc_c(0.25, "T", curl="dec")
        ls, _ = _packet_dc_c(0.5, "T")
        self.assertLess(a, 0)
        self.assertGreater(a / b, 3.0)
        self.assertLess(a / b, 5.0)
        self.assertGreater(ls / a, 10.0)


class TestFaceMedium(unittest.TestCase):
    R, H, n = 5.0, 0.5, 40

    def _run(self, lam=0.0, medium=None):
        ac = C.anchor_cluster(self.R, self.H, n_threads=_N_THREADS)
        u, phi = _random_state(ac, 3)
        ac.u, ac.phi = u, 0.5 * phi
        ac.cl.set_leaf_lam(np.full(ac.N, lam))
        if medium is not None:
            ac.set_face_medium(*medium)
        dt = C.stable_dt(ac)
        for _ in range(self.n):
            ac.alat.step(0, dt)
        return ac.u, ac.phi

    def _close(self, a, b, tol):
        for x, y in zip(a, b):
            self.assertLess(np.abs(x - y).max() / np.abs(x).max(), tol)

    def test_zero_medium_is_none(self):
        self._close(self._run(), self._run(medium=(0.0, 0.0)), 1e-15)

    def test_uniform_sigma_matches_lam(self):
        d = -0.3
        self._close(self._run(lam=1.0 / (1.0 + d) - 1.0), self._run(medium=(d, 0.0)), 1e-13)

    def test_sigma_over_n(self):
        self._close(self._run(medium=(0.2, 0.0)), self._run(medium=(0.44, 0.2)), 1e-13)

    def test_face_mismatch_rejected(self):
        ac = C.anchor_cluster(self.R, self.H, n_threads=1)
        dsig = np.zeros((ac.N, 14))
        dsig[ac.N // 2, 0] = 0.1
        with self.assertRaises(RuntimeError):
            ac.set_face_medium(dsig, 0.0)

    def test_medium_with_lam_rejected(self):
        ac = C.anchor_cluster(self.R, self.H, n_threads=1)
        ac.set_face_medium(0.1, 0.0)
        ac.cl.set_leaf_lam(np.full(ac.N, 0.5))
        with self.assertRaises(Exception):
            ac.alat.step(0, 0.01)


class TestCartanTransport(unittest.TestCase):
    """cartan_transport (C++) == continuous.cartan_transport (numpy), and on the
    C++ output directly: antisymmetric, energy-orthogonal (sum phi out = 0),
    zero on gradients and away from a localized vortex, exact -W (Omega x u) . d
    for uniform curl Omega and flow u at h 0.5 / 0.25 (exp/2026/10/13/cartan)."""

    def test_matches_numpy(self):
        ac = C.anchor_cluster(5.0, 0.5, n_threads=_N_THREADS)
        _, phi = _random_state(ac, 12)
        a, b = C.cartan_transport(ac, phi), ac.cartan_transport(phi)
        self.assertLess(np.abs(a - b).max() / np.abs(a).max(), 1e-13)

    def test_antisymmetric_and_energy_orthogonal(self):
        ac = C.anchor_cluster(5.0, 0.5, n_threads=_N_THREADS)
        _, phi = _random_state(ac, 13)
        out = ac.cartan_transport(phi)
        nb = ac.nb
        back = out[np.maximum(nb, 0), C.K_MIRROR[None, :]]
        self.assertLess(np.abs(np.where(nb >= 0, out + back, 0.0)).max() / np.abs(out).max(), 1e-14)
        self.assertLess(abs(np.sum(phi * out)) / np.sum(np.abs(phi * out)), 1e-13)

    def test_sigma_weighted(self):
        """set_cartan_sigma: uniform sigma scales the output, a varying sigma keeps
        it antisymmetric and energy-orthogonal, off ignores the medium."""
        ac = C.anchor_cluster(5.0, 0.5, n_threads=_N_THREADS)
        _, phi = _random_state(ac, 16)
        plain = ac.cartan_transport(phi)
        ac.set_face_medium(0.3, 0.0)
        self.assertLess(np.abs(ac.cartan_transport(phi) - plain).max(), 1e-14 * np.abs(plain).max())
        ac.alat.set_cartan_sigma(True)
        self.assertLess(np.abs(ac.cartan_transport(phi) - 1.3 * plain).max(), 1e-13 * np.abs(plain).max())
        fm = C.face_midpoints(ac)
        ac.set_face_medium(0.4 * np.sin(fm[..., 0]) * np.cos(0.7 * fm[..., 1]), 0.0)
        out = ac.cartan_transport(phi)
        nb = ac.nb
        back = out[np.maximum(nb, 0), C.K_MIRROR[None, :]]
        self.assertGreater(np.abs(out - plain).max(), 1e-2 * np.abs(plain).max())
        self.assertLess(np.abs(np.where(nb >= 0, out + back, 0.0)).max() / np.abs(out).max(), 1e-14)
        self.assertLess(abs(np.sum(phi * out)) / np.sum(np.abs(phi * out)), 1e-13)

    def test_vcap(self):
        """set_cartan_vcap: large v leaves the output, a small v changes it and keeps
        it antisymmetric and energy-orthogonal."""
        ac = C.anchor_cluster(5.0, 0.5, n_threads=_N_THREADS)
        _, phi = _random_state(ac, 17)
        plain = ac.cartan_transport(phi)
        ac.alat.set_cartan_vcap(1e8)
        self.assertLess(np.abs(ac.cartan_transport(phi) - plain).max(), 1e-12 * np.abs(plain).max())
        ac.alat.set_cartan_vcap(0.3 * np.abs(phi / C.W).max())
        out = ac.cartan_transport(phi)
        nb = ac.nb
        back = out[np.maximum(nb, 0), C.K_MIRROR[None, :]]
        self.assertGreater(np.abs(out - plain).max(), 1e-1 * np.abs(plain).max())
        self.assertLess(np.abs(np.where(nb >= 0, out + back, 0.0)).max() / np.abs(out).max(), 1e-14)
        self.assertLess(abs(np.sum(phi * out)) / np.sum(np.abs(phi * out)), 1e-13)

    def test_cell_capacity(self):
        """set_cell_capacity: huge S_c leaves step() and the transport as they were;
        a small S_c drives u by dH/dphi = phi (G'_j + G'_n) / 2 and keeps the transport
        antisymmetric and orthogonal to dH/dphi; step_cartan keeps H = sum u^2 / 2 +
        sum G(S_j) to O(dt^2) (exp/2026/10/20/cap)."""
        ac = C.anchor_cluster(4.0, 0.5, n_threads=_N_THREADS)
        u, phi = _random_state(ac, 18)
        nb = ac.nb
        m = nb >= 0
        S = 0.25 * np.sum(np.where(m, phi, 0.0) ** 2, 1)
        plain = ac.cartan_transport(phi)
        ac.alat.set_cell_capacity(1e12 * S.max())
        self.assertLess(np.abs(ac.cartan_transport(phi) - plain).max(), 1e-10 * np.abs(plain).max())
        Sc = 0.3 * S.max()
        ac.alat.set_cell_capacity(Sc, 2)
        g = 1 + (S / Sc) ** 2
        Fp = np.where(m, phi * 0.5 * (g[:, None] + g[np.maximum(nb, 0)]), 0.0)
        out = ac.cartan_transport(phi)
        back = out[np.maximum(nb, 0), C.K_MIRROR[None, :]]
        self.assertGreater(np.abs(out - plain).max(), 1e-1 * np.abs(plain).max())
        self.assertLess(np.abs(np.where(m, out + back, 0.0)).max() / np.abs(out).max(), 1e-14)
        self.assertLess(abs(np.sum(Fp * out)) / np.sum(np.abs(Fp * out)), 1e-13)
        self.assertGreater(abs(np.sum(phi * out)) / np.sum(np.abs(phi * out)), 1e-5)
        dt = 0.1 * C.stable_dt(ac)
        for cap, F in ((0.0, phi), (Sc, Fp)):
            ac.alat.set_cell_capacity(cap, 2)
            ac.u, ac.phi = u, phi
            ac.alat.step(ac.i, dt)
            du_ref = -dt * np.sum(C.W[None, :] / 0.5 * F, 1)
            self.assertLess(np.abs((ac.u - u) - du_ref).max() / np.abs(du_ref).max(), 1e-12)

        def H(ph_prev, ph):
            S = 0.25 * np.sum(np.where(m, ph_prev * ph, 0.0), 1)
            return 0.5 * np.sum(ac.u**2) + np.sum(S + Sc / 3 * (S / Sc) ** 3)

        drift = []
        for f in (1.0, 0.5):
            dt = f * 0.3 * C.stable_dt(ac)
            ac.u, ac.phi = 0.3 * u, 0.3 * phi
            prev, E = ac.phi, []
            for _ in range(int(round(2.0 / dt))):
                ac.alat.step_cartan(ac.i, dt, 1.0, 3)
                E.append(H(prev, ac.phi))
                prev = ac.phi
            E = np.array(E)
            drift.append(np.abs(E - E[0]).max() / E[0])
        self.assertLess(drift[0], 1e-2)
        self.assertLess(drift[1], 0.4 * drift[0])

    def test_hold(self):
        """set_cartan_hold: kappa 1 == off bit-identical; random kappa == numpy, antisymmetric,
        energy-orthogonal, zero on gradients; kappa 0 changes the output (exp/2026/10/22/depression)."""
        ac = C.anchor_cluster(5.0, 0.5, n_threads=_N_THREADS)
        u, phi = _random_state(ac, 19)
        nb = ac.nb
        plain = ac.cartan_transport(phi)
        ac.set_cartan_hold(np.ones(ac.N))
        self.assertTrue(np.array_equal(ac.cartan_transport(phi), plain))
        kappa = np.random.default_rng(20).random(ac.N)
        ac.set_cartan_hold(kappa)
        out = ac.cartan_transport(phi)
        ref = C.cartan_transport(ac, phi, hold=kappa)
        self.assertLess(np.abs(out - ref).max() / np.abs(ref).max(), 1e-13)
        self.assertGreater(np.abs(out - plain).max(), 1e-2 * np.abs(plain).max())
        back = out[np.maximum(nb, 0), C.K_MIRROR[None, :]]
        self.assertLess(np.abs(np.where(nb >= 0, out + back, 0.0)).max() / np.abs(out).max(), 1e-14)
        self.assertLess(abs(np.sum(phi * out)) / np.sum(np.abs(phi * out)), 1e-13)
        g = np.where(nb >= 0, C.W[None, :] * (u[np.maximum(nb, 0)] - u[:, None]), 0.0)
        self.assertLess(np.abs(ac.cartan_transport(g)).max(), 1e-12 * np.abs(g).max() ** 2)
        ac.set_cartan_hold(None)
        self.assertTrue(np.array_equal(ac.cartan_transport(phi), plain))

    def test_zero_on_irrotational_and_local(self):
        h = 0.5
        ac = C.anchor_cluster(5.0, h, n_threads=_N_THREADS)
        nb = ac.nb
        rng = np.random.default_rng(14)
        u = rng.standard_normal(ac.N)
        g = np.where(nb >= 0, C.W[None, :] * (u[np.maximum(nb, 0)] - u[:, None]), 0.0)
        self.assertLess(np.abs(ac.cartan_transport(g)).max(), 1e-12 * np.abs(g).max() ** 2)
        r = np.linalg.norm(ac.pos, axis=1)
        near = (r < 1.0)[:, None] | (r[np.maximum(nb, 0)] < 1.0)
        v = C._antisym(ac, np.where(near, rng.standard_normal((ac.N, 14)), 0.0))
        out = ac.cartan_transport(0.5 * v + g)
        far = np.broadcast_to((r > 1.0 + 4 * h)[:, None], out.shape)
        self.assertGreater(np.abs(out).max(), 1e-3)
        self.assertLess(np.abs(out[far]).max(), 1e-11)

    def test_step_cartan_matches_numpy(self):
        """step_cartan == step() + numpy implicit-midpoint carry (exp/2026/10/13/cartan/vill.py)."""
        kappa, n_iter = 0.7, 3
        acs = [C.anchor_cluster(4.0, 0.5, n_threads=_N_THREADS) for _ in range(2)]
        u, phi = _random_state(acs[0], 15)
        for ac in acs:
            ac.u, ac.phi = u, 0.3 * phi
        dt = C.stable_dt(acs[0])
        for _ in range(10):
            acs[0].alat.step_cartan(acs[0].i, dt, kappa, n_iter)
            ac = acs[1]
            ref = ac.phi
            ac.alat.step(ac.i, dt)
            ph0, a = ac.phi, kappa * dt
            d = C.cartan_transport(ac, 0.5 * (ref + ph0))
            for _ in range(n_iter):
                d = C.cartan_transport(ac, 0.5 * (ref + ph0 + a * d))
            ac.phi = ph0 + a * d
        for x, y in ((acs[0].u, acs[1].u), (acs[0].phi, acs[1].phi)):
            self.assertLess(np.abs(x - y).max() / np.abs(y).max(), 1e-12)

    def test_uniform_curl_and_flow(self):
        Om, u0 = np.array([0.3, -0.5, 0.7]), np.array([0.4, 0.9, -0.2])
        want = -C.W * (C.D_HAT @ np.cross(Om, u0))
        for h in (0.5, 0.25):
            ac = C.anchor_cluster(3.0, h, n_threads=_N_THREADS)
            pO = _face_sample(ac, h, lambda X: 0.5 * np.cross(Om, X)) / h
            pu = _face_sample(ac, h, lambda X: np.broadcast_to(u0, X.shape)) / h
            T = ac.cartan_transport
            cross = T(pO + pu) - T(pO) - T(pu)
            full = np.all(ac.nb >= 0, 1)
            inn = full & np.all(full[np.maximum(ac.nb, 0)], 1)
            self.assertGreater(inn.sum(), 50)
            self.assertLess(np.abs(cross[inn] - want).max(), 1e-12, h)


if __name__ == "__main__":
    unittest.main()
