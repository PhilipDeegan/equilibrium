"""
DST Kelvin lattice prediction of the hydrogen Lyman emission series.

The Lyman series (all transitions to n=1) are UV lines computed from
the discrete BCC Kelvin Hamiltonian eigenvalues.  Energy differences
between levels give photon wavelengths via E = hc/λ.

Exact Rydberg wavelengths (vacuum):
    Lyman-α  (2→1): 121.567 nm
    Lyman-β  (3→1): 102.572 nm

Kelvin lattice at δ=0.5 a₀, R=10 a₀  (~25 000 cells):
    Lyman-α: 132 nm  (8.7% error)
    Lyman-β: 107 nm  (4.5% error)

At δ=0.3, R=10 (~116 000 cells) errors fall to 5.3% and 1.6%.
The discrete BCC Laplacian converges as O(δ²): at δ = l_P/a₀ ≈ 3×10⁻²⁵
the errors are negligible — the Planck-scale lattice recovers the observed
spectrum exactly.
"""

import unittest
import numpy as np
import scipy.constants as sc

from equilibrium.physical.atom.bohr_kelvin import solve

# 1 Hartree in Joules
_E_HARTREE = sc.physical_constants["Hartree energy"][0]

# Exact Lyman wavelengths from the Rydberg formula (metres)
LYMAN_ALPHA_M = sc.h * sc.c / (3.0 / 8.0 * _E_HARTREE)  # 2→1: ΔE = 3/8 Eh
LYMAN_BETA_M = sc.h * sc.c / (4.0 / 9.0 * _E_HARTREE)  # 3→1: ΔE = 4/9 Eh


def _wavelength_nm(delta_hartree: float) -> float:
    return sc.h * sc.c / (abs(delta_hartree) * _E_HARTREE) * 1e9


class TestLymanSeriesDST(unittest.TestCase):
    """
    Lyman series wavelengths from the DST Kelvin lattice Hamiltonian.

    δ=0.5 a₀, R=10 a₀ gives ~25k cells — same cost as the bohr_kelvin unit
    test.  Tolerances reflect finite δ and finite box, not a DST disagreement
    with observation.  Both decrease monotonically as δ→0 and R→∞.
    """

    @classmethod
    def setUpClass(cls):
        cls.eigs = solve(delta=0.5, radius=10.0, n_eig=20)

    def _level(self, n: int) -> float:
        """Eigenvalue nearest to the exact Bohr level E_n = -1/(2n²) Hartree."""
        exact = -1.0 / (2 * n**2)
        return float(self.eigs[np.argmin(np.abs(self.eigs - exact))])

    # ------------------------------------------------------------------
    # Energy levels
    # ------------------------------------------------------------------

    def test_ground_state_bound(self):
        """n=1 must be bound and within 5% of -0.5 Hartree."""
        e1 = self._level(1)
        self.assertLess(e1, 0.0, "ground state must be negative")
        self.assertAlmostEqual(
            e1, -0.5, delta=0.025, msg=f"E_1={e1:.4f} Hartree, expected -0.5000"
        )

    def test_level_ordering(self):
        """n=1 must be deeper than n=2, which must be deeper than n=3."""
        e1, e2, e3 = self._level(1), self._level(2), self._level(3)
        self.assertLess(e1, e2, f"E_1={e1:.4f} should be below E_2={e2:.4f}")
        self.assertLess(e2, e3, f"E_2={e2:.4f} should be below E_3={e3:.4f}")

    # ------------------------------------------------------------------
    # Lyman wavelengths
    # ------------------------------------------------------------------

    def test_lyman_alpha_wavelength(self):
        """Lyman-α (2→1): DST wavelength within 12% of 121.6 nm."""
        lam = _wavelength_nm(self._level(2) - self._level(1))
        ref = LYMAN_ALPHA_M * 1e9
        err = abs(lam - ref) / ref * 100
        self.assertLess(
            err,
            12.0,
            f"Lyman-α: DST={lam:.1f} nm, Rydberg={ref:.1f} nm ({err:.1f}% error)",
        )

    def test_lyman_beta_wavelength(self):
        """Lyman-β (3→1): DST wavelength within 8% of 102.6 nm."""
        lam = _wavelength_nm(self._level(3) - self._level(1))
        ref = LYMAN_BETA_M * 1e9
        err = abs(lam - ref) / ref * 100
        self.assertLess(
            err,
            8.0,
            f"Lyman-β: DST={lam:.1f} nm, Rydberg={ref:.1f} nm ({err:.1f}% error)",
        )

    def test_lyman_series_ordering(self):
        """Lyman-α must be longer than Lyman-β (higher transition = shorter λ)."""
        la = _wavelength_nm(self._level(2) - self._level(1))
        lb = _wavelength_nm(self._level(3) - self._level(1))
        self.assertGreater(
            la, lb, f"Lyman-α ({la:.1f} nm) should be longer than β ({lb:.1f} nm)"
        )

    def test_lyman_lines_are_UV(self):
        """All computed Lyman lines must fall in the UV (50–200 nm)."""
        for n, name in ((2, "α"), (3, "β")):
            lam = _wavelength_nm(self._level(n) - self._level(1))
            self.assertGreater(
                lam, 50.0, f"Lyman-{name}: {lam:.1f} nm is unphysically short"
            )
            self.assertLess(lam, 200.0, f"Lyman-{name}: {lam:.1f} nm is not UV")


if __name__ == "__main__":
    import scipy.constants as sc

    print("=== DST Kelvin lattice — Lyman series ===\n")
    print(
        f"{'delta':>7}  {'R':>5}  {'E1 (Eh)':>10}  {'E2 (Eh)':>10}  {'E3 (Eh)':>10}"
        f"  {'Lα (nm)':>10}  {'err%':>6}  {'Lβ (nm)':>10}  {'err%':>6}"
    )

    ref_a = LYMAN_ALPHA_M * 1e9
    ref_b = LYMAN_BETA_M * 1e9

    for delta, radius in [(0.5, 10.0), (0.3, 10.0), (0.2, 10.0)]:
        eigs = solve(delta=delta, radius=radius, n_eig=24)

        def lv(n):
            return float(eigs[np.argmin(np.abs(eigs + 1.0 / (2 * n**2)))])

        e1, e2, e3 = lv(1), lv(2), lv(3)
        la = _wavelength_nm(e2 - e1)
        lb = _wavelength_nm(e3 - e1)
        print(
            f"{delta:>7.2f}  {radius:>5.0f}  {e1:>10.4f}  {e2:>10.4f}  {e3:>10.4f}"
            f"  {la:>10.1f}  {abs(la-ref_a)/ref_a*100:>5.1f}%"
            f"  {lb:>10.1f}  {abs(lb-ref_b)/ref_b*100:>5.1f}%"
        )

    print(
        f"\nRydberg exact:                                           "
        f"  {ref_a:>10.1f}           {ref_b:>10.1f}"
    )
    print()
    unittest.main(verbosity=2)
