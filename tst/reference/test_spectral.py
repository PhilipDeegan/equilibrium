"""equilibrium.reference.spectral: exact references reproduce their own invariants."""
import unittest

import numpy as np

from equilibrium.reference import spectral as S


def abc(box, m=1, a=1.0, b=0.7, c=0.4):
    k = 2 * np.pi * m / box.side
    x, y, z = box.x
    return np.stack([a * np.sin(k * z) + c * np.cos(k * y), b * np.sin(k * x) + a * np.cos(k * z),
                     c * np.sin(k * y) + b * np.cos(k * x)]), k


class TestSpectral(unittest.TestCase):
    def test_beltrami_static_ideal_and_exact_resistive(self):
        box = S.Box(16, 2 * np.pi)
        B, k = abc(box)
        self.assertAlmostEqual(S.lam_eff(box, B), k, places=10)
        m = S.MHD(box, np.zeros_like(B), B, nu=0.0, eta=0.0)
        for _ in range(5):
            m.step(0.1)
        self.assertLess(np.abs(m.B - B).max(), 1e-12)
        self.assertLess(np.abs(m.v).max(), 1e-12)
        m = S.MHD(box, np.zeros_like(B), B, nu=0.01, eta=0.05)
        for _ in range(10):
            m.step(0.1)
        self.assertAlmostEqual(S.energy(box, m.B) / S.energy(box, B), np.exp(-2 * 0.05 * k**2 * 1.0), places=12)
        self.assertLess(np.abs(m.B - S.diffuse(box, B, 0.05, 1.0)).max(), 1e-12)

    def test_ideal_mhd_conserves_energy_and_helicity(self):
        box = S.Box(32, 2 * np.pi)
        B, _ = abc(box)
        x, y, z = box.x
        v = 0.5 * np.stack([np.sin(y) * np.cos(2 * z), np.sin(z + 0.3) * np.cos(x), np.sin(2 * x) * np.cos(y)])
        m = S.MHD(box, v, B, nu=0.0, eta=0.0)
        E0, H0 = S.energy(box, m.v) + S.energy(box, m.B), S.helicity(box, m.B)
        for _ in range(20):
            m.step(0.01)
        self.assertLess(abs(S.energy(box, m.v) + S.energy(box, m.B) - E0) / E0, 1e-6)
        self.assertLess(abs(S.helicity(box, m.B) - H0) / abs(H0), 1e-6)
        self.assertGreater(np.abs(m.v - box.ifft(box.project_h(box.fft(v)))).max(), 1e-3)

    def test_maxwell_vacuum_energy_and_held_static(self):
        box = S.Box(32, 40.0)
        B = S.taylor_spheromak(box, 8.0, smooth=1.0)
        A0 = box.ifft(box.uncurl_h(box.fft(B)))
        en = lambda A, E: S.energy(box, E) + S.energy(box, box.curl(A))
        e0 = en(A0, 0 * A0)
        for t in (3.0, 17.0):
            A, E = S.maxwell_vacuum(box, A0, t)
            self.assertAlmostEqual(en(A, E) / e0, 1.0, places=10)
        J = box.curl(box.curl(A0))
        A, E = S.maxwell_held(box, A0, J, 11.0)
        self.assertLess(np.abs(A - A0).max() / np.abs(A0).max(), 1e-10)
        self.assertLess(np.abs(E).max(), 1e-10)

    def test_taylor_seed(self):
        box = S.Box(64, 48.0)
        a = 12.0
        B = S.taylor_spheromak(box, a, smooth=0.1)
        self.assertAlmostEqual(S.lam_eff(box, B) * a / S.LAM_A, 1.0, delta=0.05)
        self.assertLess(np.abs(np.sum(box.k * box.fft(B), 0)).max(), 1e-8)
        self.assertGreater(S.helicity(box, B), 0)
        for ax in (0, 1):
            Bx = S.taylor_spheromak(box, a, smooth=0.1, axis=ax)
            self.assertAlmostEqual(S.energy(box, Bx) / S.energy(box, B), 1.0, places=6)

    def test_linear_landau_damping_rate(self):
        """k lambda_D 0.5: gamma -0.1533, omega 1.4156 (textbook). Fit the decay of the |E| peaks over t 2-20."""
        k = 0.5
        vp = S.VlasovPoisson1D(64, 256, 2 * np.pi / k, 8.0, S.landau_f0(k, 0.01))
        ts, ee = [], []
        W0 = vp.field_energy() + vp.kinetic_energy()
        for _ in range(400):
            vp.step(0.05)
            ts.append(vp.t)
            ee.append(vp.field_energy())
        ts, ee = np.array(ts), np.array(ee)
        pk = [i for i in range(1, len(ee) - 1) if ee[i] > ee[i - 1] and ee[i] > ee[i + 1] and 2 < ts[i] < 20]
        gamma = np.polyfit(ts[pk], 0.5 * np.log(ee[pk]), 1)[0]
        omega = np.pi / np.mean(np.diff(ts[pk]))
        self.assertAlmostEqual(gamma, -0.1533, delta=0.003)
        self.assertAlmostEqual(omega, 1.4156, delta=0.01)
        self.assertLess(abs(vp.field_energy() + vp.kinetic_energy() - W0) / W0, 1e-4)


if __name__ == "__main__":
    unittest.main()
