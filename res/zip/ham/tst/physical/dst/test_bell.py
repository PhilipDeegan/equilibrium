"""
DST Bell inequality tests.

Verifies:
  - Classical LHV model does NOT violate Bell (S ≤ 2)
  - DST canal model DOES violate Bell (S ≈ 2√2)
  - Both correlation curves match their analytic forms
"""

import unittest
import numpy as np
from equilibrium.physical.dst.bell import (
    classical_lhv,
    dst_canal,
    dst_joint_canal,
    chsh,
)


class TestClassicalLHV(unittest.TestCase):
    """Classical pre-set values: no Bell violation."""

    def test_no_bell_violation(self):
        result = chsh(classical_lhv, n=300_000)
        self.assertLessEqual(
            result["S"],
            2.0 + 0.05,
            f"LHV should not violate Bell but S={result['S']:.4f}",
        )

    def test_correlation_form(self):
        """C(Δθ) ≈ −½ cos(2Δθ) for classical LHV."""
        for delta in [0, np.pi / 8, np.pi / 4, np.pi / 2]:
            C = classical_lhv(0, delta, n=200_000)
            C_exp = -0.5 * np.cos(2 * delta)
            self.assertAlmostEqual(
                C,
                C_exp,
                delta=0.03,
                msg=f"Δθ={delta:.3f}: got {C:.4f} expected {C_exp:.4f}",
            )

    def test_same_angle_anticorrelated(self):
        """Same angle → perfectly anticorrelated (singlet: opposite polarisations)."""
        C = classical_lhv(0, 0, n=200_000)
        self.assertAlmostEqual(C, -0.5, delta=0.03)

    def test_orthogonal_uncorrelated(self):
        """90° offset → zero correlation (for LHV: −½ cos(π) = +½, not 0)."""
        C = classical_lhv(0, np.pi / 2, n=200_000)
        self.assertAlmostEqual(C, 0.5, delta=0.03)


class TestDSTCanal(unittest.TestCase):
    """DST canal model: violates Bell at S = 2√2."""

    def test_bell_violated(self):
        result = chsh(dst_canal, n=300_000)
        self.assertGreater(
            result["S"], 2.0, f"DST canal should violate Bell but S={result['S']:.4f}"
        )

    def test_chsh_near_tsirelson(self):
        """S should be close to 2√2 (Tsirelson bound)."""
        result = chsh(dst_canal, n=300_000)
        self.assertAlmostEqual(
            result["S"],
            2 * np.sqrt(2),
            delta=0.10,
            msg=f"Expected S≈{2*np.sqrt(2):.4f}, got {result['S']:.4f}",
        )

    def test_correlation_form(self):
        """C(Δθ) ≈ −cos(2Δθ) for DST canal."""
        for delta in [0, np.pi / 8, np.pi / 4, np.pi / 2]:
            C = dst_canal(0, delta, n=200_000)
            C_exp = -np.cos(2 * delta)
            self.assertAlmostEqual(
                C,
                C_exp,
                delta=0.03,
                msg=f"Δθ={delta:.3f}: got {C:.4f} expected {C_exp:.4f}",
            )

    def test_canal_stronger_than_lhv(self):
        """Canal correlations must have strictly larger magnitude than LHV
        at angles away from the zero-crossing at π/4."""
        for delta in [np.pi / 8, 3 * np.pi / 8]:  # skip π/4 where both ≈ 0
            C_lhv = classical_lhv(0, delta, n=200_000)
            C_canal = dst_canal(0, delta, n=200_000)
            self.assertGreater(
                abs(C_canal),
                abs(C_lhv),
                f"|canal|={abs(C_canal):.4f} should exceed |lhv|={abs(C_lhv):.4f} "
                f"at Δθ={delta:.3f}",
            )


class TestDSTJointCanal(unittest.TestCase):
    """
    Joint canal model — correct for spacelike separation.
    No cascade, no ordering assumption.  Both A and B sample the
    pre-existing non-factorisable canal state simultaneously.
    """

    def test_bell_violated(self):
        result = chsh(dst_joint_canal, n=300_000)
        self.assertGreater(
            result["S"], 2.0, f"Joint canal should violate Bell but S={result['S']:.4f}"
        )

    def test_chsh_near_tsirelson(self):
        result = chsh(dst_joint_canal, n=300_000)
        self.assertAlmostEqual(result["S"], 2 * np.sqrt(2), delta=0.10)

    def test_correlation_form(self):
        """C(Δθ) ≈ −cos(2Δθ) — same as cascade but valid for spacelike."""
        for delta in [0, np.pi / 8, np.pi / 4, np.pi / 2]:
            C = dst_joint_canal(0, delta, n=200_000)
            C_exp = -np.cos(2 * delta)
            self.assertAlmostEqual(
                C,
                C_exp,
                delta=0.03,
                msg=f"Δθ={delta:.3f}: got {C:.4f} expected {C_exp:.4f}",
            )

    def test_agrees_with_cascade(self):
        """Joint canal and cascade give the same C(Δθ) at all angles."""
        for delta in [np.pi / 8, np.pi / 4, 3 * np.pi / 8]:
            C_joint = dst_joint_canal(0, delta, n=300_000)
            C_cascade = dst_canal(0, delta, n=300_000)
            self.assertAlmostEqual(
                C_joint,
                C_cascade,
                delta=0.04,
                msg=f"Models disagree at Δθ={delta:.3f}: "
                f"joint={C_joint:.4f} cascade={C_cascade:.4f}",
            )

    def test_perfect_anticorrelation_same_angle(self):
        """At Δθ=0, singlet gives C=−1: perfectly anticorrelated."""
        C = dst_joint_canal(0, 0, n=200_000)
        self.assertAlmostEqual(C, -1.0, delta=0.03)

    def test_lhv_same_angle_is_not_perfect(self):
        """LHV at Δθ=0 gives only C=−½, not −1: key difference from QM."""
        C_lhv = classical_lhv(0, 0, n=200_000)
        C_joint = dst_joint_canal(0, 0, n=200_000)
        self.assertAlmostEqual(C_lhv, -0.5, delta=0.03)
        self.assertAlmostEqual(C_joint, -1.0, delta=0.03)


class TestCHSHValues(unittest.TestCase):
    """Numerical CHSH S values for both models."""

    @classmethod
    def setUpClass(cls):
        cls.lhv = chsh(classical_lhv, n=400_000)
        cls.canal = chsh(dst_canal, n=400_000)
        cls.joint = chsh(dst_joint_canal, n=400_000)

    def test_lhv_s_near_sqrt2(self):
        self.assertAlmostEqual(self.lhv["S"], np.sqrt(2), delta=0.08)

    def test_canal_s_near_2sqrt2(self):
        self.assertAlmostEqual(self.canal["S"], 2 * np.sqrt(2), delta=0.08)

    def test_lhv_not_violated(self):
        self.assertFalse(self.lhv["violated"])

    def test_canal_violated(self):
        self.assertTrue(self.canal["violated"])

    def test_joint_violated(self):
        self.assertTrue(self.joint["violated"])

    def test_joint_s_near_2sqrt2(self):
        self.assertAlmostEqual(self.joint["S"], 2 * np.sqrt(2), delta=0.08)


if __name__ == "__main__":
    from equilibrium.physical.dst.bell import chsh, classical_lhv, dst_canal

    print("Classical LHV:")
    r = chsh(classical_lhv, n=500_000)
    print(
        f"  S = {r['S']:.4f}  (classical max = {r['S_classical_max']}, "
        f"QM = {r['S_qm']:.4f})  violated={r['violated']}"
    )
    print("DST canal:")
    r = chsh(dst_canal, n=500_000)
    print(
        f"  S = {r['S']:.4f}  (classical max = {r['S_classical_max']}, "
        f"QM = {r['S_qm']:.4f})  violated={r['violated']}"
    )
    print()
    unittest.main(verbosity=2)
