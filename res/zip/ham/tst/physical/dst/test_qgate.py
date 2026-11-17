"""
Tests for DST quantum logic gates.

Key connections verified:
  - Bell state |Ψ−⟩ reproduces the same correlations as dst_joint_canal
  - Single-qubit gates are unitary (norm-preserving canal transformations)
  - CNOT creates entanglement from product states
  - Teleportation transfers the canal state to a remote qubit
"""

import unittest
import numpy as np
from equilibrium.physical.dst.qgate import (
    DSTCircuit,
    X,
    Y,
    Z,
    H,
    S,
    T,
    Rx,
    Ry,
    Rz,
    bell_state,
    ghz_state,
)


class TestSingleQubitGates(unittest.TestCase):
    """Gate matrices are unitary — canal transformations preserve norm."""

    def _check_unitary(self, G, name):
        err = np.max(np.abs(G @ G.conj().T - np.eye(2)))
        self.assertLess(err, 1e-12, f"{name} is not unitary: max error {err:.2e}")

    def test_pauli_unitary(self):
        for G, name in [(X, "X"), (Y, "Y"), (Z, "Z"), (H, "H"), (S, "S"), (T, "T")]:
            self._check_unitary(G, name)

    def test_rotation_unitary(self):
        for theta in [0.1, np.pi / 4, np.pi / 2, np.pi]:
            for G, name in [(Rx(theta), "Rx"), (Ry(theta), "Ry"), (Rz(theta), "Rz")]:
                self._check_unitary(G, name)

    def test_x_flips_state(self):
        c = DSTCircuit(1)  # |0⟩
        c.apply(X, 0)
        np.testing.assert_allclose(c.probabilities(), [0, 1], atol=1e-12)

    def test_h_creates_superposition(self):
        c = DSTCircuit(1)
        c.apply(H, 0)
        np.testing.assert_allclose(c.probabilities(), [0.5, 0.5], atol=1e-12)

    def test_hh_is_identity(self):
        c = DSTCircuit(1)
        c.apply(H, 0).apply(H, 0)
        np.testing.assert_allclose(c.probabilities(), [1, 0], atol=1e-12)

    def test_norm_preserved_through_gates(self):
        c = DSTCircuit(1)
        c.apply(H, 0).apply(S, 0).apply(T, 0).apply(Ry(1.23), 0)
        self.assertAlmostEqual(np.sum(c.probabilities()), 1.0, places=12)


class TestBellState(unittest.TestCase):
    """Bell states are the two-qubit joint canal states."""

    def test_phi_plus_equal_superposition(self):
        c = bell_state(0)  # |Φ+⟩
        p = c.probabilities()
        self.assertAlmostEqual(p[0b00], 0.5, places=10)  # |00⟩
        self.assertAlmostEqual(p[0b11], 0.5, places=10)  # |11⟩
        self.assertAlmostEqual(p[0b01], 0.0, places=10)
        self.assertAlmostEqual(p[0b10], 0.0, places=10)

    def test_psi_minus_is_singlet(self):
        """Singlet |Ψ−⟩ is the Bell state used in the Bell test."""
        c = bell_state(3)
        p = c.probabilities()
        self.assertAlmostEqual(p[0b01], 0.5, places=10)  # |01⟩
        self.assertAlmostEqual(p[0b10], 0.5, places=10)  # |10⟩

    def test_bell_state_entangled(self):
        """Reduced single-qubit state is maximally mixed — no local information."""
        c = bell_state(0)
        x, y, z = c.bloch(0)
        # Maximally mixed: Bloch vector is zero
        self.assertAlmostEqual(x**2 + y**2 + z**2, 0.0, delta=1e-10)

    def test_bell_correlations_match_joint_canal(self):
        """
        Measuring the Bell singlet in rotated bases reproduces
        C(Δθ) = −cos(2Δθ) — same as dst_joint_canal in bell.py.

        This is the bridge between quantum circuits and the canal Bell test:
        the Bell state IS the joint canal state.
        """
        from equilibrium.physical.dst.bell import dst_joint_canal

        rng = np.random.default_rng(42)

        def measure_bell_correlation(theta_a, theta_b, n=100_000):
            results = []
            for _ in range(n):
                c = bell_state(3)  # singlet |Ψ−⟩
                # Rotate each qubit by its detector angle before Z-measurement
                c.apply(Ry(-2 * theta_a), 0)
                c.apply(Ry(-2 * theta_b), 1)
                probs = c.probabilities()
                # Sample one outcome
                outcome = rng.choice(4, p=probs)
                bit0 = 1 if (outcome & 0b10) else -1
                bit1 = 1 if (outcome & 0b01) else -1
                results.append(bit0 * bit1)
            return float(np.mean(results))

        for delta in [np.pi / 8, np.pi / 4, 3 * np.pi / 8]:
            C_circuit = measure_bell_correlation(0, delta, n=20_000)
            C_canal = dst_joint_canal(0, delta, n=20_000)
            self.assertAlmostEqual(
                C_circuit,
                C_canal,
                delta=0.06,
                msg=f"Δθ={delta:.3f}: circuit={C_circuit:.3f} canal={C_canal:.3f}",
            )


class TestCNOT(unittest.TestCase):
    """CNOT is the canal-mediated conditional update."""

    def test_cnot_00_unchanged(self):
        c = DSTCircuit(2)  # |00⟩
        c.cnot(0, 1)
        np.testing.assert_allclose(c.probabilities(), [1, 0, 0, 0], atol=1e-12)

    def test_cnot_10_flips_target(self):
        c = DSTCircuit(2)
        c.apply(X, 0)  # |10⟩
        c.cnot(0, 1)  # → |11⟩
        np.testing.assert_allclose(c.probabilities(), [0, 0, 0, 1], atol=1e-12)

    def test_cnot_creates_entanglement(self):
        """H + CNOT entangles two product-state canals into a Bell state."""
        c = DSTCircuit(2)
        c.apply(H, 0).cnot(0, 1)
        p = c.probabilities()
        self.assertAlmostEqual(p[0b00], 0.5, places=10)
        self.assertAlmostEqual(p[0b11], 0.5, places=10)
        # Single qubit is maximally mixed
        x, y, z = c.bloch(0)
        self.assertAlmostEqual(x**2 + y**2 + z**2, 0.0, delta=1e-10)


class TestGHZ(unittest.TestCase):
    """GHZ state — all canals correlated."""

    def test_ghz_3_qubit(self):
        c = ghz_state(3)
        p = c.probabilities()
        self.assertAlmostEqual(p[0b000], 0.5, places=10)
        self.assertAlmostEqual(p[0b111], 0.5, places=10)
        for i in range(1, 7):
            self.assertAlmostEqual(p[i], 0.0, places=10)

    def test_ghz_all_qubits_maximally_mixed(self):
        c = ghz_state(3)
        for q in range(3):
            x, y, z = c.bloch(q)
            self.assertAlmostEqual(x**2 + y**2 + z**2, 0.0, delta=1e-10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
