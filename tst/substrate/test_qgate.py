"""
DST qubit — single-qubit gates from substrate lattice dynamics.

Qubit encoding
--------------
Two substrate cells A and B are coupled by a transverse face flux φ_x.

    u_A  ↔  essence in cell A  (amplitude of basis state |0⟩)
    u_B  ↔  essence in cell B  (amplitude of basis state |1⟩)

The symplectic-Euler wave update (same rule as SubstrateLattice.step):

    Δu_A   = −dt · w · φ_x            (flux outward from A)
    Δu_B   = +dt · w · φ_x            (flux inward to B)
    Δφ_x   = −dt · w · (u_B_new − u_A_new)   (φ driven by essence gradient)

Conserved quantities
--------------------
    u_A + u_B            = const   total essence (exactly conserved in discrete time)
    u_A² + u_B² + φ_x²  = const   energy        (symplectic: bounded drift, no growth)

Analytical solution (continuous time, starting from u_A=1, u_B=0, φ_x=0)
--------------------------------------------------------------------------
    ω  = w√2          Rabi frequency of the coupled pair
    u_A(t) = ½(1 + cos ωt)
    u_B(t) = ½(1 − cos ωt)
    φ_x(t) = +(1/√2) sin ωt       (φ_x = 0 at t = nπ/ω)

Gate times (φ_x = 0 at completion → clean Born-rule read)
----------------------------------------------------------
    T_X = π/ω    →  u_A = 0, u_B = 1   X gate: full population inversion
    T_S = π/2ω   →  u_A = u_B = ½      S₂ gate: equal superposition

Born rule / Malus's law (at gate completion times, when φ_x = 0)
-----------------------------------------------------------------
    P(|0⟩) = u_A / (u_A + u_B)
    P(|1⟩) = u_B / (u_A + u_B)

Connection to the full Kelvin lattice
--------------------------------------
In the full SubstrateLattice the gate would be realised by two canal cells
sitting on adjacent hex-face neighbours.  The transverse sq-face between them
carries φ_x (W_SQ = 1).  Anisotropic face areas A_ij can tune w in situ,
setting the gate time without changing the topology.
"""

import unittest

import numpy as np

_W = 1.0  # sq-face weight (W_SQ = 1 in SubstrateLattice)
_OMEGA = _W * np.sqrt(2)  # Rabi frequency
T_X = np.pi / _OMEGA  # X-gate duration  (full inversion)
T_S = T_X / 2  # S₂-gate duration (equal split)
_DT = T_X / 4000  # timestep — well inside the stability bound


def _step(u_A, u_B, phi_x, dt, w):
    """One symplectic-Euler tick of the two-cell coupled system."""
    u_A_new = u_A - dt * w * phi_x
    u_B_new = u_B + dt * w * phi_x
    phi_x -= dt * w * (u_B_new - u_A_new)
    return u_A_new, u_B_new, phi_x


def _run(u_A, u_B, T, w=_W, dt=_DT):
    """Integrate the two-cell DST system for time T; return (u_A, u_B, φ_x)."""
    phi_x = 0.0
    for _ in range(max(1, round(T / dt))):
        u_A, u_B, phi_x = _step(u_A, u_B, phi_x, dt, w)
    return u_A, u_B, phi_x


class TestDSTGates(unittest.TestCase):
    # ------------------------------------------------------------------
    # Conservation laws
    # ------------------------------------------------------------------

    def test_essence_exactly_conserved(self):
        """u_A + u_B = 1 at every discrete step (not just approximately)."""
        u_A, u_B, phi_x = 1.0, 0.0, 0.0
        for _ in range(round(2 * T_X / _DT)):
            u_A, u_B, phi_x = _step(u_A, u_B, phi_x, _DT, _W)
            self.assertAlmostEqual(u_A + u_B, 1.0, places=14)

    def test_energy_bounded(self):
        """u_A² + u_B² + φ_x² stays within 1e-6 of its initial value."""
        u_A, u_B, phi_x = 1.0, 0.0, 0.0
        E0 = u_A**2 + u_B**2 + phi_x**2
        for _ in range(round(2 * T_X / _DT)):
            u_A, u_B, phi_x = _step(u_A, u_B, phi_x, _DT, _W)
        self.assertAlmostEqual(u_A**2 + u_B**2 + phi_x**2, E0, places=6)

    # ------------------------------------------------------------------
    # No coupling → identity
    # ------------------------------------------------------------------

    def test_identity_w_zero(self):
        """w = 0: no canal flux, state frozen at (1, 0)."""
        u_A, u_B, phi_x = _run(1.0, 0.0, T=T_X, w=0.0)
        self.assertAlmostEqual(u_A, 1.0, places=15)
        self.assertAlmostEqual(u_B, 0.0, places=15)
        self.assertAlmostEqual(phi_x, 0.0, places=15)

    # ------------------------------------------------------------------
    # X gate — full population inversion
    # ------------------------------------------------------------------

    def test_x_gate_from_zero(self):
        """T_X from |0⟩: u_A → 0, u_B → 1, φ_x → 0."""
        u_A, u_B, phi_x = _run(1.0, 0.0, T_X)
        self.assertAlmostEqual(u_A, 0.0, places=5)
        self.assertAlmostEqual(u_B, 1.0, places=5)
        self.assertAlmostEqual(phi_x, 0.0, places=5)

    def test_x_gate_from_one(self):
        """T_X from |1⟩: u_A → 1, u_B → 0 (X is its own inverse)."""
        u_A, u_B, phi_x = _run(0.0, 1.0, T_X)
        self.assertAlmostEqual(u_A, 1.0, places=5)
        self.assertAlmostEqual(u_B, 0.0, places=5)

    def test_double_x_is_identity(self):
        """Two X gates return to the initial state."""
        u_A, u_B, _ = _run(1.0, 0.0, 2 * T_X)
        self.assertAlmostEqual(u_A, 1.0, places=5)
        self.assertAlmostEqual(u_B, 0.0, places=5)

    # ------------------------------------------------------------------
    # S₂ gate — equal superposition
    # ------------------------------------------------------------------

    def test_s2_equal_split(self):
        """T_S from |0⟩: q = u_A − u_B crosses zero between steps n_S and n_S+1.

        The symplectic-Euler half-step offset puts the true q = 0 crossing
        between the two discrete steps that straddle T_S, so the structurally
        correct discrete assertion brackets the crossing rather than testing
        q ≈ 0 with a tolerance.  Both bracket values must also sit within one
        step's maximum swing |Δq| ≤ √2·dt·w of zero (|φ_x| ≤ 1/√2), so the
        crossing is tight, not merely somewhere in the run.
        """
        u_A_at, u_B_at, phi = _run(1.0, 0.0, T_S)
        u_A_next, u_B_next, _ = _step(u_A_at, u_B_at, phi, _DT, _W)
        q_at = u_A_at - u_B_at
        q_next = u_A_next - u_B_next
        self.assertGreater(q_at, 0.0)  # q > 0 just before the crossing
        self.assertLess(q_next, 0.0)  # q < 0 just after the crossing
        max_swing = np.sqrt(2) * _DT * _W
        self.assertLess(abs(q_at), max_swing)
        self.assertLess(abs(q_next), max_swing)

    def test_s2_born_probabilities(self):
        """T_S: Born probability P(|0⟩) = u_A/(u_A+u_B) crosses 0.5 at the gate."""
        u_A_at, u_B_at, phi = _run(1.0, 0.0, T_S)
        u_A_next, u_B_next, _ = _step(u_A_at, u_B_at, phi, _DT, _W)
        self.assertGreater(u_A_at / (u_A_at + u_B_at), 0.5)
        self.assertLess(u_A_next / (u_A_next + u_B_next), 0.5)

    # ------------------------------------------------------------------
    # Gate speed scales with coupling strength
    # ------------------------------------------------------------------

    def test_stronger_coupling_faster_gate(self):
        """Doubling w halves T_X: X gate achieved in half the time."""
        T_X_double = np.pi / (2 * _W * np.sqrt(2))  # T_X for w = 2*_W
        u_A, u_B, phi_x = _run(1.0, 0.0, T=T_X_double, w=2 * _W, dt=_DT / 4)
        self.assertAlmostEqual(u_A, 0.0, places=4)
        self.assertAlmostEqual(u_B, 1.0, places=4)

    # ------------------------------------------------------------------
    # φ_x is zero at gate completion (clean measurement window)
    # ------------------------------------------------------------------

    def test_phi_zero_at_x_gate(self):
        """φ_x = 0 at t = T_X: measurement is unambiguous (no canal in flight)."""
        _, _, phi_x = _run(1.0, 0.0, T_X)
        self.assertAlmostEqual(phi_x, 0.0, places=5)

    def test_phi_zero_at_double_x(self):
        """φ_x = 0 at t = 2·T_X (full period)."""
        _, _, phi_x = _run(1.0, 0.0, 2 * T_X)
        self.assertAlmostEqual(phi_x, 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
